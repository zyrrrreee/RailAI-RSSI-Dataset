"""Stage-2 front/rear OBM observation model for one wayside AP.

The existing diagnostic pipeline intentionally remains a single-OBM pipeline.
This module adds a separate research API so front and rear measurements can be
validated before any upper-layer OBM selection, multiple-AP association, or
handover logic is introduced.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional

import numpy as np

from motion_sampling import generate_trajectory
from receiver_measurement import (
    apply_receiver_reporting,
    integrate_rssi_dbm_causally,
    mw_to_dbm,
)
from signal_generation import (
    DEFAULT_RICIAN_K_LINEAR,
    TIME_DOMAIN_SAMPLING,
    _antenna_gain,
    _correlated_gaussian,
    _path_loss,
    _rician_fading_dB,
    _rotate,
    _unit_vector,
    generate_rssi_simulation,
)


DUAL_OBM_TOPOLOGY_SCHEMA_VERSION = (
    "paper-stage2-single-ap-dual-antenna-front-rear-obm-v1"
)
DEFAULT_TRAIN_LENGTH_M = 200.0


def _resolved_generator_params(
    generator_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the single-OBM defaults without changing its public contract."""
    supplied = dict(generator_kwargs)
    supplied.pop("return_metadata", None)
    requested_mode = supplied.pop("sampling_mode", TIME_DOMAIN_SAMPLING)
    if str(requested_mode) != TIME_DOMAIN_SAMPLING:
        raise ValueError("双OBM第一版只支持 fixed_time_report 时间域采样")

    signature = inspect.signature(generate_rssi_simulation)
    unknown = set(supplied) - set(signature.parameters)
    if unknown:
        raise ValueError(f"双OBM仿真收到未知参数: {sorted(unknown)}")

    resolved: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name in {"return_metadata", "sampling_mode"}:
            continue
        if name in supplied:
            resolved[name] = supplied[name]
        elif parameter.default is not inspect.Parameter.empty:
            resolved[name] = parameter.default
    resolved["sampling_mode"] = TIME_DOMAIN_SAMPLING
    return resolved


def _crossing_time_s(
    time_s: np.ndarray,
    position_m: np.ndarray,
    target_position_m: float,
) -> Optional[float]:
    """Linearly interpolate the first time a monotonic trace crosses a KP."""
    time = np.asarray(time_s, dtype=float)
    position = np.asarray(position_m, dtype=float)
    offset = position - float(target_position_m)
    exact = np.flatnonzero(np.isclose(offset, 0.0, rtol=0.0, atol=1e-12))
    if len(exact):
        return float(time[int(exact[0])])
    crossings = np.flatnonzero(offset[:-1] * offset[1:] < 0.0)
    if not len(crossings):
        return None
    index = int(crossings[0])
    fraction = (
        float(target_position_m) - float(position[index])
    ) / float(position[index + 1] - position[index])
    return float(time[index] + fraction * (time[index + 1] - time[index]))


def _report_slice(values: np.ndarray, report_mask: np.ndarray) -> np.ndarray:
    return np.asarray(values)[np.asarray(report_mask, dtype=bool)].copy()


def generate_dual_obm_observation(
    *,
    train_length_m: float = DEFAULT_TRAIN_LENGTH_M,
    position_alignment_offset_override_m: Optional[float] = None,
    **generator_kwargs: Any,
) -> dict[str, Any]:
    """Generate independent front/rear RSSI reports on one shared trajectory.

    ``x_start`` and ``x_end`` describe the geometric centre of the train.
    Front and rear are defined by the direction of motion, not by which one has
    the numerically larger KP coordinate.  Both OBMs sample one stationary
    large-scale shadow field.  Small-scale fading and receiver measurement
    noise use separate reproducible random substreams for the two OBMs.

    The function deliberately returns two traces.  It does not select a
    "winning" OBM and does not feed either trace into the stage-1 diagnostic
    models.
    """
    params = _resolved_generator_params(generator_kwargs)
    length = float(train_length_m)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("train_length_m must be finite and greater than zero")
    if not bool(params["dual_antenna"]):
        raise ValueError("双OBM论文场景要求每个AP保留两副反向定向天线")

    trajectory = generate_trajectory(
        x_start=float(params["x_start"]),
        x_end=float(params["x_end"]),
        simulation_step_s=float(params["simulation_step_s"]),
        report_interval_s=float(params["report_interval_s"]),
        speed_mps=float(params["v"]),
        speed_profile=params["speed_profile"],
        direction=params["direction"],
    )
    motion_direction = int(trajectory.metadata["direction"])
    half_length = 0.5 * length
    centre_x = trajectory.x_m.copy()
    front_x = centre_x + motion_direction * half_length
    rear_x = centre_x - motion_direction * half_length

    seed_sequence = np.random.SeedSequence(params["seed"])
    (
        common_seed,
        shadow_seed,
        front_small_seed,
        rear_small_seed,
        front_receiver_seed,
        rear_receiver_seed,
    ) = seed_sequence.spawn(6)
    common_rng = np.random.default_rng(common_seed)
    shadow_rng = np.random.default_rng(shadow_seed)

    sampled_position_offset_m = float(
        common_rng.normal(
            0.0, max(float(params["position_alignment_sigma_m"]), 0.0)
        )
    )
    if position_alignment_offset_override_m is None:
        position_offset_m = sampled_position_offset_m
        position_offset_source = "sampled_for_this_dual_obm_scene"
    else:
        position_offset_m = float(position_alignment_offset_override_m)
        if not np.isfinite(position_offset_m):
            raise ValueError(
                "position_alignment_offset_override_m must be finite or None"
            )
        position_offset_source = "provided_by_parent_multi_ap_scene"
    pointing_jitter_deg = float(
        common_rng.normal(
            0.0, max(float(params["pointing_jitter_sigma_deg"]), 0.0)
        )
    )
    trip_power_offset_dB = float(
        common_rng.normal(0.0, max(float(params["trip_power_sigma_dB"]), 0.0))
    )

    nominal_lobe_1 = _unit_vector(
        float(params["main_lobe_dir_x"]),
        float(params["main_lobe_dir_y"]),
    )
    lobe_1 = _rotate(nominal_lobe_1, pointing_jitter_deg)
    lobe_2 = -lobe_1

    # Generate one stationary unit-variance shadow field on the union of both
    # OBM routes.  Each receiver samples this field at its own physical KP.
    physical_front_x = front_x + position_offset_m
    physical_rear_x = rear_x + position_offset_m
    motion_steps = np.abs(np.diff(centre_x))
    positive_steps = motion_steps[motion_steps > 1e-12]
    field_step_m = (
        float(np.median(positive_steps))
        if len(positive_steps)
        else max(float(params["dx"]), 1e-3)
    )
    field_min_m = float(min(np.min(physical_front_x), np.min(physical_rear_x)))
    field_max_m = float(max(np.max(physical_front_x), np.max(physical_rear_x)))
    field_count = max(
        2, int(np.ceil((field_max_m - field_min_m) / field_step_m)) + 1
    )
    shadow_field_x_m = np.linspace(
        field_min_m, field_max_m, field_count, dtype=float
    )
    unit_shadow_field = _correlated_gaussian(
        shadow_rng,
        field_count,
        1.0,
        np.diff(shadow_field_x_m),
        float(params["shadow_corr_distance_m"]),
    )

    measurement_window_m = params["measurement_window_m"]
    if measurement_window_m is None:
        measurement_coordinate = trajectory.t_s
        measurement_window_width = float(params["measurement_window_s"])
        measurement_coordinate_unit = "s"
    else:
        measurement_window_width = float(measurement_window_m)
        if measurement_window_width < 0.0:
            raise ValueError("measurement_window_m must be non-negative or None")
        measurement_coordinate = np.concatenate(
            (
                np.asarray([0.0]),
                np.cumsum(np.abs(np.diff(centre_x)), dtype=float),
            )
        )
        measurement_coordinate_unit = "m"

    def build_obm_trace(
        *,
        obm_id: str,
        role: str,
        nominal_x: np.ndarray,
        small_seed: np.random.SeedSequence,
        receiver_seed: np.random.SeedSequence,
    ) -> dict[str, Any]:
        physical_x = nominal_x + position_offset_m
        size = len(physical_x)
        horizontal_vectors = np.column_stack(
            (
                physical_x - float(params["antenna_x"]),
                np.full(size, -float(params["antenna_y"])),
            )
        )
        vertical_separation_m = float(params["obm_height_m"]) - float(
            params["ap_height_m"]
        )
        distances = np.sqrt(
            np.sum(horizontal_vectors**2, axis=1) + vertical_separation_m**2
        )
        gain_1_dB = _antenna_gain(
            horizontal_vectors,
            lobe_1,
            float(params["G_max_dB"]),
            float(params["theta_half_deg"]),
            float(params["max_antenna_attenuation_dB"]),
        )
        gain_2_dB = _antenna_gain(
            horizontal_vectors,
            lobe_2,
            float(params["G_max_dB"]),
            float(params["theta_half_deg"]),
            float(params["max_antenna_attenuation_dB"]),
        )
        path_loss_dB = _path_loss(
            distances,
            float(params["fc"]),
            float(params["n"]),
            float(params["PL0_dB"]),
            float(params["d0"]),
            bool(params["use_free_space"]),
            float(params["path_loss_breakpoint_m"]),
            float(params["path_loss_exponent_far"]),
        )

        sampled_unit_shadow = np.interp(
            physical_x, shadow_field_x_m, unit_shadow_field
        )
        if float(params["path_loss_breakpoint_m"]) > float(params["d0"]):
            shadow_sigma_profile_dB = np.where(
                distances <= float(params["path_loss_breakpoint_m"]),
                max(float(params["sigma_shadow"]), 0.0),
                max(float(params["shadow_sigma_far_dB"]), 0.0),
            )
        else:
            shadow_sigma_profile_dB = np.full(
                size, max(float(params["sigma_shadow"]), 0.0), dtype=float
            )
        shadow_dB = sampled_unit_shadow * shadow_sigma_profile_dB

        base_k_dB = 10.0 * np.log10(
            max(float(params["K_linear"]), 1e-12)
        )
        k_profile_dB = np.clip(
            base_k_dB
            + float(params["rician_K_slope_dB_per_100m"])
            * distances
            / 100.0,
            min(-20.0, base_k_dB),
            max(40.0, base_k_dB),
        )
        small_fading_dB = _rician_fading_dB(
            np.random.default_rng(small_seed),
            size,
            np.abs(np.diff(nominal_x)),
            float(params["fc"]),
            10.0 ** (k_profile_dB / 10.0),
        )
        receiver_noise_dB = np.random.default_rng(receiver_seed).normal(
            0.0,
            max(float(params["receiver_noise_sigma_dB"]), 0.0),
            size,
        )

        common_link_dBm = (
            float(params["Pt_dBm"])
            + trip_power_offset_dB
            + float(params["Gr_dBi"])
            - path_loss_dB
            + shadow_dB
            + small_fading_dB
        )
        branch_1_instantaneous_dBm = common_link_dBm + gain_1_dB
        branch_2_instantaneous_dBm = common_link_dBm + gain_2_dB
        branch_1_integration = integrate_rssi_dbm_causally(
            branch_1_instantaneous_dBm,
            measurement_coordinate,
            measurement_window_width,
        )
        branch_2_integration = integrate_rssi_dbm_causally(
            branch_2_instantaneous_dBm,
            measurement_coordinate,
            measurement_window_width,
        )
        branch_1_raw_dBm = (
            mw_to_dbm(branch_1_integration.averaged_power_mW)
            + receiver_noise_dB
        )
        branch_2_raw_dBm = (
            mw_to_dbm(branch_2_integration.averaged_power_mW)
            + receiver_noise_dB
        )
        selected_antenna = np.where(
            branch_1_raw_dBm >= branch_2_raw_dBm, 1, 2
        ).astype(np.int8)
        selected_raw_dBm = np.maximum(branch_1_raw_dBm, branch_2_raw_dBm)
        receiver_report = apply_receiver_reporting(
            selected_raw_dBm,
            params["receiver_sensitivity_dBm"],
            params["receiver_saturation_dBm"],
            float(params["rssi_quantization_dB"]),
            str(params["below_sensitivity_policy"]),
        )
        branch_1_report = apply_receiver_reporting(
            branch_1_raw_dBm,
            params["receiver_sensitivity_dBm"],
            params["receiver_saturation_dBm"],
            float(params["rssi_quantization_dB"]),
            str(params["below_sensitivity_policy"]),
        )
        branch_2_report = apply_receiver_reporting(
            branch_2_raw_dBm,
            params["receiver_sensitivity_dBm"],
            params["receiver_saturation_dBm"],
            float(params["rssi_quantization_dB"]),
            str(params["below_sensitivity_policy"]),
        )
        ideal_rssi_dBm = (
            float(params["Pt_dBm"])
            + float(params["Gr_dBi"])
            + np.maximum(gain_1_dB, gain_2_dB)
            - path_loss_dB
        )

        time_domain = {
            "t_s": trajectory.t_s.copy(),
            "x_m": nominal_x.copy(),
            "v_mps": trajectory.v_mps.copy(),
            "ideal_rssi_dBm": ideal_rssi_dBm,
            "reported_rssi_dBm": receiver_report.reported_rssi_dBm,
            "raw_rssi_dBm": selected_raw_dBm,
            "receiver_report_status": receiver_report.status,
            "selected_antenna": selected_antenna,
            "antenna_1_candidate_instantaneous_rssi_dBm": (
                branch_1_instantaneous_dBm
            ),
            "antenna_2_candidate_instantaneous_rssi_dBm": (
                branch_2_instantaneous_dBm
            ),
            "antenna_1_candidate_raw_rssi_dBm": branch_1_raw_dBm,
            "antenna_2_candidate_raw_rssi_dBm": branch_2_raw_dBm,
            "antenna_1_candidate_reported_rssi_dBm": (
                branch_1_report.reported_rssi_dBm
            ),
            "antenna_2_candidate_reported_rssi_dBm": (
                branch_2_report.reported_rssi_dBm
            ),
            "path_loss_dB": path_loss_dB,
            "gain_1_dB": gain_1_dB,
            "gain_2_dB": gain_2_dB,
            "shared_shadow_dB": shadow_dB,
            "independent_small_fading_dB": small_fading_dB,
            "independent_receiver_noise_dB": receiver_noise_dB,
            "receiver_integration_support_width": (
                branch_1_integration.support_width
            ),
            "receiver_integration_contributing_sample_count": (
                branch_1_integration.contributing_sample_count
            ),
            "ap_to_obm_distance_m": distances,
        }
        mask = trajectory.report_mask
        return {
            "obm_id": obm_id,
            "role": role,
            "t_s": trajectory.report_t_s.copy(),
            "x_m": _report_slice(nominal_x, mask),
            "v_mps": trajectory.report_v_mps.copy(),
            "ideal_rssi_dBm": _report_slice(ideal_rssi_dBm, mask),
            "reported_rssi_dBm": _report_slice(
                receiver_report.reported_rssi_dBm, mask
            ),
            "raw_rssi_dBm": _report_slice(selected_raw_dBm, mask),
            "receiver_report_status": _report_slice(
                receiver_report.status, mask
            ),
            "receiver_below_sensitivity_mask": _report_slice(
                receiver_report.below_sensitivity_mask, mask
            ),
            "receiver_saturation_mask": _report_slice(
                receiver_report.saturation_mask, mask
            ),
            "receiver_missing_mask": _report_slice(
                receiver_report.missing_mask, mask
            ),
            "selected_antenna": _report_slice(selected_antenna, mask),
            "selection_margin_dB": _report_slice(
                np.abs(branch_1_raw_dBm - branch_2_raw_dBm), mask
            ),
            "antenna_1_candidate_raw_rssi_dBm": _report_slice(
                branch_1_raw_dBm, mask
            ),
            "antenna_2_candidate_raw_rssi_dBm": _report_slice(
                branch_2_raw_dBm, mask
            ),
            "antenna_1_candidate_reported_rssi_dBm": _report_slice(
                branch_1_report.reported_rssi_dBm, mask
            ),
            "antenna_2_candidate_reported_rssi_dBm": _report_slice(
                branch_2_report.reported_rssi_dBm, mask
            ),
            "shared_shadow_dB": _report_slice(shadow_dB, mask),
            "independent_small_fading_dB": _report_slice(
                small_fading_dB, mask
            ),
            "independent_receiver_noise_dB": _report_slice(
                receiver_noise_dB, mask
            ),
            "ap_to_obm_distance_m": _report_slice(distances, mask),
            "time_domain": time_domain,
        }

    front = build_obm_trace(
        obm_id="OBM-front",
        role="front_in_direction_of_motion",
        nominal_x=front_x,
        small_seed=front_small_seed,
        receiver_seed=front_receiver_seed,
    )
    rear = build_obm_trace(
        obm_id="OBM-rear",
        role="rear_opposite_direction_of_motion",
        nominal_x=rear_x,
        small_seed=rear_small_seed,
        receiver_seed=rear_receiver_seed,
    )

    front_crossing_s = _crossing_time_s(
        trajectory.t_s, front_x, float(params["antenna_x"])
    )
    rear_crossing_s = _crossing_time_s(
        trajectory.t_s, rear_x, float(params["antenna_x"])
    )
    actual_crossing_delay_s = (
        None
        if front_crossing_s is None or rear_crossing_s is None
        else float(rear_crossing_s - front_crossing_s)
    )
    expected_constant_speed_delay_s = (
        length / abs(float(params["v"]))
        if params["speed_profile"] is None and float(params["v"]) != 0.0
        else None
    )

    metadata = {
        "topology_schema_version": DUAL_OBM_TOPOLOGY_SCHEMA_VERSION,
        "modeling_stage": 2,
        "scene": "single_wayside_ap_dual_directional_antennas_front_rear_obms",
        "link_direction": "AP_downlink_to_each_onboard_OBM",
        "train_reference_position": "geometric_centre",
        "train_length_m": length,
        "motion_direction": motion_direction,
        "front_position_formula": "x_center + direction * train_length / 2",
        "rear_position_formula": "x_center - direction * train_length / 2",
        "obm_combining_policy": "none_keep_independent_traces",
        "large_scale_shadow_policy": (
            "one_stationary_spatial_field_shared_by_front_and_rear"
        ),
        "small_scale_fading_policy": "independent_reproducible_stream_per_obm",
        "receiver_noise_policy": "independent_reproducible_stream_per_obm",
        "common_trip_power_offset_dB": trip_power_offset_dB,
        "common_position_alignment_offset_m": position_offset_m,
        "common_position_alignment_offset_source": position_offset_source,
        "common_ap_pointing_jitter_deg": pointing_jitter_deg,
        "receiver_integration_alignment": "causal_trailing_window",
        "receiver_integration_coordinate_unit": measurement_coordinate_unit,
        "receiver_integration_window_width": measurement_window_width,
        "front_ap_crossing_time_s": front_crossing_s,
        "rear_ap_crossing_time_s": rear_crossing_s,
        "actual_front_to_rear_crossing_delay_s": actual_crossing_delay_s,
        "expected_constant_speed_crossing_delay_s": (
            expected_constant_speed_delay_s
        ),
        "report_interval_s": float(params["report_interval_s"]),
        "simulation_step_s": float(params["simulation_step_s"]),
        "report_sample_count": int(len(trajectory.report_t_s)),
        "internal_sample_count": int(len(trajectory.t_s)),
        "seed": params["seed"],
        "evidence_status": (
            "paper_aligned_geometry_with_engineering_randomness_assumptions"
        ),
        "deferred_scope": (
            "multiple_APs_serving_state_EWMA_hysteresis_TTT_and_upper_layer_OBM_policy"
        ),
        "generator_parameters": {
            key: value
            for key, value in params.items()
            if key
            not in {
                "speed_profile",
            }
        },
    }
    return {
        "topology_schema_version": DUAL_OBM_TOPOLOGY_SCHEMA_VERSION,
        "time_s": trajectory.report_t_s.copy(),
        "train_center_position_m": trajectory.report_x_m.copy(),
        "train_center_speed_mps": trajectory.report_v_mps.copy(),
        "front_obm": front,
        "rear_obm": rear,
        "shared_environment": {
            "shadow_field_position_m": shadow_field_x_m,
            "unit_shadow_field": unit_shadow_field,
        },
        "metadata": metadata,
    }
