"""Railway RSSI generation with physically interpretable channel components.

The public functions keep the original six-array return value so the GUI remains
compatible.  ``generate_fault_rssi_pair`` is the preferred training-data API:
healthy and faulty curves reuse exactly the same random channel realization and
only the physical fault mechanism changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

import numpy as np

from motion_sampling import SpeedProfile, Trajectory, generate_trajectory
from receiver_measurement import (
    DEFAULT_BELOW_SENSITIVITY_POLICY,
    apply_receiver_reporting,
    integrate_rssi_dbm_causally,
    mw_to_dbm,
)


DEFAULT_RICIAN_K_DB = 4.16
DEFAULT_RICIAN_K_LINEAR = 10.0 ** (DEFAULT_RICIAN_K_DB / 10.0)
LEGACY_SPATIAL_SAMPLING = "fixed_space_legacy"
TIME_DOMAIN_SAMPLING = "fixed_time_report"

# Paper-aligned phase-1 communication scene.  Later phases may extend this to
# multiple APs and front/rear OBMs, but must not silently change this contract.
STAGE1_TOPOLOGY_SCHEMA_VERSION = "paper-stage1-single-ap-dual-antenna-single-obm-v2"
STAGE1_OBSERVATION_DEFINITION = "onboard_obm_reported_ap_downlink_rssi_dBm"


@dataclass
class _ChannelState:
    x: np.ndarray
    distances: np.ndarray
    path_loss_dB: np.ndarray
    gain_1_dB: np.ndarray
    gain_2_dB: np.ndarray
    gain_dB: np.ndarray
    shadow_dB: np.ndarray
    small_fading_dB: np.ndarray
    measurement_noise_dB: np.ndarray
    measurement_window_samples: int
    measurement_coordinate: np.ndarray
    measurement_window_width: float
    measurement_coordinate_unit: str
    trip_power_offset_dB: float
    effective_main_lobe_1: np.ndarray
    effective_main_lobe_2: np.ndarray
    metadata: Dict[str, Any]


def _stage1_topology_metadata(
    *,
    antenna_x: float,
    antenna_y: float,
    ap_height_m: float,
    obm_height_m: float,
    lobe_1: np.ndarray,
    lobe_2: np.ndarray,
) -> Dict[str, Any]:
    """Describe the paper-aligned phase-1 AP/antenna/OBM observation contract."""
    return {
        "topology_schema_version": STAGE1_TOPOLOGY_SCHEMA_VERSION,
        "modeling_stage": 1,
        "scene": "single_wayside_ap_dual_opposite_directional_antennas_single_onboard_obm",
        "observation_definition": STAGE1_OBSERVATION_DEFINITION,
        "link_direction": "AP_downlink_to_onboard_OBM",
        "ap_count": 1,
        "obm_count": 1,
        "ap_id": "AP-001",
        "obm_id": "OBM-front",
        "ap_track_position_m": float(antenna_x),
        "ap_lateral_offset_m": float(antenna_y),
        "ap_antenna_height_m": float(ap_height_m),
        "obm_antenna_height_m": float(obm_height_m),
        "vertical_separation_m": abs(float(ap_height_m) - float(obm_height_m)),
        "antenna_1_main_lobe_unit_vector": np.asarray(lobe_1, dtype=float).copy(),
        "antenna_2_main_lobe_unit_vector": np.asarray(lobe_2, dtype=float).copy(),
        "antenna_selection_rule": "maximum_candidate_power_before_receiver_reporting",
        "antenna_selection_assumption": (
            "explicit_phase1_modeling_assumption; paper does not specify RF branch combining"
        ),
        "receiver_reporting_order": [
            "instantaneous_link_budget_and_spatial_shadowing",
            "instantaneous_Rician_small_scale_fading",
            "complete_candidate_link_linear_power_averaging",
            "maximum_averaged_candidate_branch_selection",
            "receiver_measurement_noise",
            "RSSI_quantization",
            "sensitivity_and_saturation_limits",
        ],
    }


def _unit_vector(x: float, y: float) -> np.ndarray:
    vector = np.asarray([x, y], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Main-lobe direction cannot be the zero vector")
    return vector / norm


def _rotate(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.radians(float(angle_deg))
    matrix = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )
    return matrix @ vector


def _antenna_gain(
    direction_vectors: np.ndarray,
    main_lobe: np.ndarray,
    max_gain_dB: float,
    half_power_angle_deg: float,
    max_attenuation_dB: float,
) -> np.ndarray:
    norms = np.linalg.norm(direction_vectors, axis=1)
    cos_theta = np.sum(direction_vectors * main_lobe[None, :], axis=1) / np.maximum(norms, 1e-12)
    theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    attenuation = 3.0 * (theta_deg / max(float(half_power_angle_deg), 1e-6)) ** 2
    attenuation = np.minimum(attenuation, float(max_attenuation_dB))
    return float(max_gain_dB) - attenuation


def _path_loss(
    distances: np.ndarray,
    fc: float,
    n: float,
    PL0_dB: float,
    d0: float,
    use_free_space: bool,
    breakpoint_m: float = 0.0,
    n_far: Optional[float] = None,
) -> np.ndarray:
    distances = np.maximum(np.asarray(distances, dtype=float), max(float(d0), 1e-3))
    if use_free_space:
        return 20.0 * np.log10(distances) + 20.0 * np.log10(float(fc)) - 147.55
    near_loss = float(PL0_dB) + 10.0 * float(n) * np.log10(
        distances / float(d0)
    )
    breakpoint = float(breakpoint_m)
    if breakpoint <= float(d0) or n_far is None:
        return near_loss
    loss_at_breakpoint = float(PL0_dB) + 10.0 * float(n) * np.log10(
        breakpoint / float(d0)
    )
    far_loss = loss_at_breakpoint + 10.0 * float(n_far) * np.log10(
        distances / breakpoint
    )
    return np.where(distances <= breakpoint, near_loss, far_loss)


def _correlated_gaussian(
    rng: np.random.Generator,
    size: int,
    sigma: float,
    dx: float | np.ndarray,
    correlation_distance_m: float,
) -> np.ndarray:
    """Generate stationary Gaussian shadowing with exponential spatial ACF."""
    sigma = max(float(sigma), 0.0)
    if size <= 0 or sigma == 0.0:
        return np.zeros(max(size, 0), dtype=float)
    if correlation_distance_m <= 0.0:
        return rng.normal(0.0, sigma, size)

    innovations = rng.normal(0.0, sigma, size)
    values = np.empty(size, dtype=float)
    values[0] = innovations[0]
    spacing = np.asarray(dx, dtype=float)
    if spacing.ndim == 0:
        spacing = np.full(max(size - 1, 0), abs(float(spacing)), dtype=float)
    if spacing.shape != (max(size - 1, 0),):
        raise ValueError("dx must be scalar or contain one spacing per sample transition")
    for idx in range(1, size):
        rho = float(
            np.exp(-abs(float(spacing[idx - 1])) / float(correlation_distance_m))
        )
        innovation_scale = np.sqrt(max(1.0 - rho**2, 0.0))
        values[idx] = rho * values[idx - 1] + innovation_scale * innovations[idx]
    return values


def _rician_fading_dB(
    rng: np.random.Generator,
    size: int,
    dx: float | np.ndarray,
    fc: float,
    K_linear: float,
) -> np.ndarray:
    """Generate instantaneous spatial Rician fading in linear complex amplitude.

    At 2.4 GHz and a 0.5 m position interval, adjacent samples are already many
    wavelengths apart.  A weak complex AR relation is retained for smaller dx.
    Receiver integration is applied later to each complete candidate link in
    linear power, rather than to this fading component alone.
    """
    if size <= 0:
        return np.asarray([], dtype=float)
    wavelength = 299_792_458.0 / float(fc)
    coherence_distance = max(wavelength / 2.0, 1e-6)
    spacing = np.asarray(dx, dtype=float)
    if spacing.ndim == 0:
        spacing = np.full(max(size - 1, 0), abs(float(spacing)), dtype=float)
    if spacing.shape != (max(size - 1, 0),):
        raise ValueError("dx must be scalar or contain one spacing per sample transition")

    innovations = (rng.normal(size=size) + 1j * rng.normal(size=size)) / np.sqrt(2.0)
    scatter = np.empty(size, dtype=complex)
    scatter[0] = innovations[0]
    for idx in range(1, size):
        rho = float(np.exp(-abs(float(spacing[idx - 1])) / coherence_distance))
        innovation_scale = np.sqrt(max(1.0 - rho**2, 0.0))
        scatter[idx] = rho * scatter[idx - 1] + innovation_scale * innovations[idx]

    K_linear = np.maximum(np.asarray(K_linear, dtype=float), 0.0)
    if K_linear.ndim == 0:
        K_linear = np.full(size, float(K_linear), dtype=float)
    if K_linear.shape != (size,):
        raise ValueError("K_linear must be scalar or match the position grid")
    los_amplitude = np.sqrt(K_linear / (K_linear + 1.0))
    scatter_amplitude = np.sqrt(1.0 / (K_linear + 1.0))
    cumulative_distance = np.concatenate(
        (np.asarray([0.0]), np.cumsum(np.abs(spacing), dtype=float))
    )
    los_phase = (
        rng.uniform(0.0, 2.0 * np.pi)
        + 2.0 * np.pi * cumulative_distance / wavelength
    )
    channel = los_amplitude * np.exp(1j * los_phase) + scatter_amplitude * scatter
    power = np.maximum(np.abs(channel) ** 2, 1e-12)

    return 10.0 * np.log10(power)


def _build_channel_state(
    *,
    x_start: float,
    x_end: float,
    dx: float,
    v: float,
    antenna_x: float,
    antenna_y: float,
    ap_height_m: float,
    obm_height_m: float,
    main_lobe_dir_x: float,
    main_lobe_dir_y: float,
    G_max_dB: float,
    theta_half_deg: float,
    n: float,
    path_loss_breakpoint_m: float,
    path_loss_exponent_far: float,
    PL0_dB: float,
    d0: float,
    sigma_shadow: float,
    shadow_sigma_far_dB: float,
    fc: float,
    K_linear: float,
    rician_K_slope_dB_per_100m: float,
    use_free_space: bool,
    seed: Optional[int],
    shadow_corr_distance_m: float,
    measurement_window_m: Optional[float],
    measurement_window_s: float,
    receiver_noise_sigma_dB: float,
    max_antenna_attenuation_dB: float,
    dual_antenna: bool,
    trip_power_sigma_dB: float,
    position_alignment_sigma_m: float,
    pointing_jitter_sigma_deg: float,
    sample_positions_m: Optional[np.ndarray] = None,
    sample_times_s: Optional[np.ndarray] = None,
    sampling_metadata: Optional[Mapping[str, Any]] = None,
) -> _ChannelState:
    if dx <= 0.0:
        raise ValueError("dx must be greater than zero")
    if x_end == x_start or (sample_positions_m is None and x_end < x_start):
        raise ValueError("legacy sampling requires x_end greater than x_start")
    if fc <= 0.0:
        raise ValueError("fc must be greater than zero")
    if d0 <= 0.0:
        raise ValueError("d0 must be greater than zero")
    if theta_half_deg <= 0.0:
        raise ValueError("theta_half_deg must be greater than zero")
    if v < 0.0:
        raise ValueError("v must be non-negative")
    if measurement_window_s < 0.0:
        raise ValueError("measurement_window_s must be non-negative")
    if ap_height_m < 0.0 or obm_height_m < 0.0:
        raise ValueError("AP and OBM antenna heights must be non-negative")

    seed_sequence = np.random.SeedSequence(seed)
    channel_seed_sequence, receiver_seed_sequence = seed_sequence.spawn(2)
    rng = np.random.default_rng(channel_seed_sequence)
    receiver_rng = np.random.default_rng(receiver_seed_sequence)
    if sample_positions_m is None:
        x = np.arange(float(x_start), float(x_end) + float(dx) * 0.5, float(dx))
        sample_times = None
        spatial_steps = np.full(max(len(x) - 1, 0), abs(float(dx)), dtype=float)
        representative_dx = abs(float(dx))
        position_grid_semantics = "OBM_position_aligned_and_resampled_to_fixed_spatial_grid"
    else:
        x = np.asarray(sample_positions_m, dtype=float).copy()
        if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
            raise ValueError("sample_positions_m must be a finite one-dimensional trajectory")
        signed_steps = np.diff(x)
        if not (np.all(signed_steps > 0.0) or np.all(signed_steps < 0.0)):
            raise ValueError("sample_positions_m must be strictly monotonic")
        spatial_steps = np.abs(signed_steps)
        representative_dx = float(np.median(spatial_steps))
        position_grid_semantics = "time_driven_internal_OBM_trajectory"
        if sample_times_s is None:
            raise ValueError("sample_times_s is required with sample_positions_m")
        sample_times = np.asarray(sample_times_s, dtype=float).copy()
        if (
            sample_times.shape != x.shape
            or not np.all(np.isfinite(sample_times))
            or np.any(np.diff(sample_times) <= 0.0)
        ):
            raise ValueError("sample_times_s must be finite, increasing, and match positions")
    size = len(x)

    position_offset = rng.normal(0.0, max(float(position_alignment_sigma_m), 0.0))
    physical_x = x + position_offset
    direction_vectors = np.column_stack(
        (physical_x - float(antenna_x), np.full(size, -float(antenna_y)))
    )
    vertical_separation_m = float(obm_height_m) - float(ap_height_m)
    distances = np.sqrt(
        np.sum(direction_vectors**2, axis=1) + vertical_separation_m**2
    )

    nominal_lobe_1 = _unit_vector(main_lobe_dir_x, main_lobe_dir_y)
    pointing_jitter = rng.normal(0.0, max(float(pointing_jitter_sigma_deg), 0.0))
    lobe_1 = _rotate(nominal_lobe_1, pointing_jitter)
    lobe_2 = -lobe_1

    gain_1 = _antenna_gain(
        direction_vectors,
        lobe_1,
        G_max_dB,
        theta_half_deg,
        max_antenna_attenuation_dB,
    )
    if dual_antenna:
        gain_2 = _antenna_gain(
            direction_vectors,
            lobe_2,
            G_max_dB,
            theta_half_deg,
            max_antenna_attenuation_dB,
        )
        gain = np.maximum(gain_1, gain_2)
    else:
        gain_2 = np.full(size, -np.inf, dtype=float)
        gain = gain_1.copy()

    if path_loss_breakpoint_m < 0.0:
        raise ValueError("path_loss_breakpoint_m must be non-negative")
    path_loss_dB = _path_loss(
        distances,
        fc,
        n,
        PL0_dB,
        d0,
        use_free_space,
        path_loss_breakpoint_m,
        path_loss_exponent_far,
    )
    unit_shadow = _correlated_gaussian(
        rng, size, 1.0, spatial_steps, shadow_corr_distance_m
    )
    if float(path_loss_breakpoint_m) > float(d0):
        shadow_sigma_profile_dB = np.where(
            distances <= float(path_loss_breakpoint_m),
            max(float(sigma_shadow), 0.0),
            max(float(shadow_sigma_far_dB), 0.0),
        )
    else:
        shadow_sigma_profile_dB = np.full(
            size, max(float(sigma_shadow), 0.0), dtype=float
        )
    shadow_dB = unit_shadow * shadow_sigma_profile_dB
    if measurement_window_m is None:
        if sample_times is None:
            effective_measurement_window_m = abs(float(v)) * float(measurement_window_s)
            measurement_window_samples = max(
                1,
                int(round(effective_measurement_window_m / max(representative_dx, 1e-9))),
            )
            if float(v) > 0.0:
                measurement_coordinate = np.concatenate(
                    (
                        np.asarray([0.0]),
                        np.cumsum(spatial_steps / abs(float(v)), dtype=float),
                    )
                )
                measurement_window_width = float(measurement_window_s)
                measurement_coordinate_unit = "s"
                measurement_window_mode = "fixed_time_on_legacy_spatial_grid"
            else:
                measurement_coordinate = np.concatenate(
                    (np.asarray([0.0]), np.cumsum(spatial_steps, dtype=float))
                )
                measurement_window_width = 0.0
                measurement_coordinate_unit = "m"
                measurement_window_mode = "disabled_for_static_legacy_grid"
        else:
            representative_dt = float(np.median(np.diff(sample_times)))
            measurement_window_samples = max(
                1,
                int(round(float(measurement_window_s) / max(representative_dt, 1e-12))),
            )
            effective_measurement_window_m = abs(float(v)) * float(measurement_window_s)
            measurement_coordinate = sample_times.copy()
            measurement_window_width = float(measurement_window_s)
            measurement_coordinate_unit = "s"
            measurement_window_mode = "fixed_time_on_internal_time_grid"
    else:
        if float(measurement_window_m) < 0.0:
            raise ValueError("measurement_window_m must be non-negative or None")
        effective_measurement_window_m = float(measurement_window_m)
        measurement_window_samples = max(
            1,
            int(round(effective_measurement_window_m / max(representative_dx, 1e-9))),
        )
        measurement_coordinate = np.concatenate(
            (np.asarray([0.0]), np.cumsum(spatial_steps, dtype=float))
        )
        measurement_window_width = float(measurement_window_m)
        measurement_coordinate_unit = "m"
        measurement_window_mode = "fixed_distance_override"
    base_rician_K_dB = float(10.0 * np.log10(max(float(K_linear), 1e-12)))
    rician_K_profile_dB = np.clip(
        base_rician_K_dB
        + float(rician_K_slope_dB_per_100m) * distances / 100.0,
        min(-20.0, base_rician_K_dB),
        max(40.0, base_rician_K_dB),
    )
    rician_K_profile_linear = 10.0 ** (rician_K_profile_dB / 10.0)
    small_fading_dB = _rician_fading_dB(
        rng, size, spatial_steps, fc, rician_K_profile_linear
    )
    measurement_noise_dB = receiver_rng.normal(
        0.0, max(float(receiver_noise_sigma_dB), 0.0), size
    )
    trip_power_offset_dB = float(
        rng.normal(0.0, max(float(trip_power_sigma_dB), 0.0))
    )

    metadata: Dict[str, Any] = {
        "seed": seed,
        "speed_mps": float(v),
        "frequency_hz": float(fc),
        "path_loss_exponent": float(n),
        "path_loss_breakpoint_m": float(path_loss_breakpoint_m),
        "path_loss_exponent_far": float(path_loss_exponent_far),
        "path_loss_model": (
            "free_space"
            if bool(use_free_space)
            else (
                "continuous_two_slope_log_distance"
                if float(path_loss_breakpoint_m) > float(d0)
                else "single_slope_log_distance"
            )
        ),
        "shadow_sigma_dB": float(sigma_shadow),
        "shadow_sigma_far_dB": float(shadow_sigma_far_dB),
        "shadow_sigma_profile_dB": shadow_sigma_profile_dB.copy(),
        "shadow_corr_distance_m": float(shadow_corr_distance_m),
        "rician_K_linear": float(K_linear),
        "rician_K_dB": float(10.0 * np.log10(max(float(K_linear), 1e-12))),
        "rician_K_slope_dB_per_100m": float(rician_K_slope_dB_per_100m),
        "rician_K_profile_dB": rician_K_profile_dB.copy(),
        "measurement_window_s": float(measurement_window_s),
        "measurement_window_m": float(effective_measurement_window_m),
        "measurement_window_mode": measurement_window_mode,
        "measurement_window_samples": int(measurement_window_samples),
        "receiver_averaging_domain": "linear_mW_complete_candidate_link_before_measurement_noise",
        "receiver_integration_alignment": "causal_trailing_window",
        "receiver_integration_interpolation": "piecewise_linear_trapezoidal",
        "receiver_integration_startup_policy": "partial_available_history",
        "receiver_integration_coordinate_unit": measurement_coordinate_unit,
        "receiver_integration_window_width": float(measurement_window_width),
        "random_stream_policy": "seed_sequence_independent_channel_and_receiver_substreams",
        "trip_power_offset_dB": float(trip_power_offset_dB),
        "position_offset_m": float(position_offset),
        "pointing_jitter_deg": float(pointing_jitter),
        "dual_antenna": bool(dual_antenna),
        "position_grid_semantics": position_grid_semantics,
        "spatial_sample_interval_m": float(representative_dx),
        "nominal_report_interval_s": (
            float(dx) / float(v) if float(v) > 0.0 else None
        ),
        "nominal_report_rate_hz": (
            float(v) / float(dx) if float(v) > 0.0 else None
        ),
        "independent_simulation_unit": "one_complete_trip_identified_by_seed",
        "paired_healthy_usage": "simulation_mechanism_validation_only_not_available_in_deployment",
        "geometry_model": "3D_distance_with_horizontal_azimuth_antenna_pattern",
        "evidence_level": "mechanism_check_and_literature_prior_range_check",
    }
    if sample_times is not None:
        metadata.update(
            {
                "internal_time_s": sample_times.copy(),
                "internal_position_m": x.copy(),
                "internal_time_step_s_median": float(np.median(np.diff(sample_times))),
                "internal_spatial_step_m_median": float(representative_dx),
            }
        )
    if sampling_metadata:
        metadata.update(dict(sampling_metadata))
    if dual_antenna:
        metadata.update(
            _stage1_topology_metadata(
                antenna_x=antenna_x,
                antenna_y=antenna_y,
                ap_height_m=ap_height_m,
                obm_height_m=obm_height_m,
                lobe_1=lobe_1,
                lobe_2=lobe_2,
            )
        )
    return _ChannelState(
        x=x,
        distances=distances,
        path_loss_dB=path_loss_dB,
        gain_1_dB=gain_1,
        gain_2_dB=gain_2,
        gain_dB=gain,
        shadow_dB=shadow_dB,
        small_fading_dB=small_fading_dB,
        measurement_noise_dB=measurement_noise_dB,
        measurement_window_samples=measurement_window_samples,
        measurement_coordinate=measurement_coordinate,
        measurement_window_width=measurement_window_width,
        measurement_coordinate_unit=measurement_coordinate_unit,
        trip_power_offset_dB=trip_power_offset_dB,
        effective_main_lobe_1=lobe_1,
        effective_main_lobe_2=lobe_2,
        metadata=metadata,
    )


def _reported_rssi(
    raw_rssi_dBm: np.ndarray,
    receiver_sensitivity_dBm: Optional[float],
    receiver_saturation_dBm: Optional[float],
    rssi_quantization_dB: float,
    below_sensitivity_policy: str = DEFAULT_BELOW_SENSITIVITY_POLICY,
) -> np.ndarray:
    return apply_receiver_reporting(
        raw_rssi_dBm,
        receiver_sensitivity_dBm,
        receiver_saturation_dBm,
        rssi_quantization_dB,
        below_sensitivity_policy,
    ).reported_rssi_dBm


def _instantaneous_incoming_rssi(
    state: _ChannelState,
    gain_dB: np.ndarray,
    Pt_dBm: float,
    Gr_dBi: float,
) -> np.ndarray:
    """Return instantaneous received power before receiver integration/noise."""
    return (
        float(Pt_dBm)
        + state.trip_power_offset_dB
        + np.asarray(gain_dB, dtype=float)
        + float(Gr_dBi)
        - state.path_loss_dB
        + state.shadow_dB
        + state.small_fading_dB
    )


def _receiver_average_raw_rssi(
    state: _ChannelState,
    instantaneous_rssi_dBm: np.ndarray,
) -> np.ndarray:
    """Causally integrate the complete incoming link, then add meter noise."""
    integration = integrate_rssi_dbm_causally(
        np.asarray(instantaneous_rssi_dBm, dtype=float),
        state.measurement_coordinate,
        state.measurement_window_width,
    )
    state.metadata.setdefault(
        "receiver_integration_support_width",
        integration.support_width.copy(),
    )
    state.metadata.setdefault(
        "receiver_integration_contributing_sample_count",
        integration.contributing_sample_count.copy(),
    )
    steady_support = integration.support_width[1:]
    steady_counts = integration.contributing_sample_count[1:]
    if len(steady_support):
        state.metadata.setdefault(
            "receiver_integration_support_width_median",
            float(np.median(steady_support)),
        )
        state.metadata.setdefault(
            "receiver_integration_sample_count_median",
            float(np.median(steady_counts)),
        )
    return mw_to_dbm(integration.averaged_power_mW) + state.measurement_noise_dB


def _raw_rssi_for_gain(
    state: _ChannelState,
    gain_dB: np.ndarray,
    Pt_dBm: float,
    Gr_dBi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    instantaneous = _instantaneous_incoming_rssi(state, gain_dB, Pt_dBm, Gr_dBi)
    return _receiver_average_raw_rssi(state, instantaneous), instantaneous


def _compose_rssi(
    state: _ChannelState,
    gain_dB: np.ndarray,
    Pt_dBm: float,
    Gr_dBi: float,
    receiver_sensitivity_dBm: Optional[float],
    receiver_saturation_dBm: Optional[float],
    rssi_quantization_dB: float,
    below_sensitivity_policy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    raw, _ = _raw_rssi_for_gain(state, gain_dB, Pt_dBm, Gr_dBi)
    return (
        _reported_rssi(
            raw,
            receiver_sensitivity_dBm,
            receiver_saturation_dBm,
            rssi_quantization_dB,
            below_sensitivity_policy,
        ),
        raw,
    )


def _candidate_branch_raw_rssi(
    state: _ChannelState,
    Pt_dBm: float,
    Gr_dBi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the two AP antenna candidate powers observed at the same OBM.

    The propagation realization and receiver noise are shared because both
    co-located AP antenna branches are evaluated for the same time/position
    sample.  Selection therefore occurs before sensitivity clipping and RSSI
    quantization, matching the explicit phase-1 observation contract.
    """
    branch_1_raw, _ = _raw_rssi_for_gain(
        state, state.gain_1_dB, Pt_dBm, Gr_dBi
    )
    branch_2_raw, _ = _raw_rssi_for_gain(
        state, state.gain_2_dB, Pt_dBm, Gr_dBi
    )
    return branch_1_raw, branch_2_raw


def _selection_metadata(
    state: _ChannelState,
    Pt_dBm: float,
    Gr_dBi: float,
    receiver_sensitivity_dBm: Optional[float],
    receiver_saturation_dBm: Optional[float],
    rssi_quantization_dB: float,
    below_sensitivity_policy: str,
) -> Dict[str, Any]:
    branch_1_raw, branch_2_raw = _candidate_branch_raw_rssi(state, Pt_dBm, Gr_dBi)
    branch_1_instantaneous = _instantaneous_incoming_rssi(
        state, state.gain_1_dB, Pt_dBm, Gr_dBi
    )
    branch_2_instantaneous = _instantaneous_incoming_rssi(
        state, state.gain_2_dB, Pt_dBm, Gr_dBi
    )
    selected = np.where(branch_1_raw >= branch_2_raw, 1, 2).astype(np.int8)
    selected_raw = np.maximum(branch_1_raw, branch_2_raw)
    branch_1_report = apply_receiver_reporting(
        branch_1_raw,
        receiver_sensitivity_dBm,
        receiver_saturation_dBm,
        rssi_quantization_dB,
        below_sensitivity_policy,
    )
    branch_2_report = apply_receiver_reporting(
        branch_2_raw,
        receiver_sensitivity_dBm,
        receiver_saturation_dBm,
        rssi_quantization_dB,
        below_sensitivity_policy,
    )
    return {
        "antenna_1_candidate_raw_rssi_dBm": branch_1_raw.copy(),
        "antenna_2_candidate_raw_rssi_dBm": branch_2_raw.copy(),
        "antenna_1_candidate_instantaneous_rssi_dBm": branch_1_instantaneous.copy(),
        "antenna_2_candidate_instantaneous_rssi_dBm": branch_2_instantaneous.copy(),
        "antenna_1_candidate_reported_rssi_dBm": branch_1_report.reported_rssi_dBm,
        "antenna_2_candidate_reported_rssi_dBm": branch_2_report.reported_rssi_dBm,
        "antenna_1_receiver_report_status": branch_1_report.status,
        "antenna_2_receiver_report_status": branch_2_report.status,
        "selected_antenna": selected,
        "selected_candidate_raw_rssi_dBm": selected_raw.copy(),
        "selection_margin_dB": np.abs(branch_1_raw - branch_2_raw),
    }


def _trajectory_for_sampling(
    *,
    sampling_mode: str,
    x_start: float,
    x_end: float,
    v: float,
    simulation_step_s: float,
    report_interval_s: float,
    speed_profile: Optional[SpeedProfile],
    direction: Optional[int],
) -> Optional[Trajectory]:
    mode = str(sampling_mode)
    if mode == LEGACY_SPATIAL_SAMPLING:
        if speed_profile is not None or direction is not None:
            raise ValueError(
                "speed_profile and direction require sampling_mode='fixed_time_report'"
            )
        return None
    if mode != TIME_DOMAIN_SAMPLING:
        raise ValueError(
            f"sampling_mode must be {LEGACY_SPATIAL_SAMPLING!r} or {TIME_DOMAIN_SAMPLING!r}"
        )
    return generate_trajectory(
        x_start=x_start,
        x_end=x_end,
        simulation_step_s=simulation_step_s,
        report_interval_s=report_interval_s,
        speed_mps=v,
        speed_profile=speed_profile,
        direction=direction,
    )


def _sampling_build_arguments(
    trajectory: Optional[Trajectory],
    sampling_mode: str,
) -> dict[str, Any]:
    if trajectory is None:
        return {}
    metadata = dict(trajectory.metadata)
    metadata.update(
        {
            "sampling_mode": str(sampling_mode),
            "position_grid_semantics": "time_driven_internal_grid_with_fixed_time_OBM_reports",
            "nominal_report_interval_s": float(metadata["report_interval_s"]),
            "nominal_report_rate_hz": 1.0 / float(metadata["report_interval_s"]),
        }
    )
    return {
        "sample_positions_m": trajectory.x_m,
        "sample_times_s": trajectory.t_s,
        "sampling_metadata": metadata,
    }


def _report_sampled_metadata(
    metadata: Mapping[str, Any],
    trajectory: Optional[Trajectory],
    **internal_series: np.ndarray,
) -> Dict[str, Any]:
    """Expose report-aligned arrays while retaining the raw time-domain series."""
    result = dict(metadata)
    if trajectory is None:
        result.setdefault("sampling_mode", LEGACY_SPATIAL_SAMPLING)
        return result

    mask = np.asarray(trajectory.report_mask, dtype=bool)
    internal_count = len(mask)
    time_domain = {
        "t_s": trajectory.t_s.copy(),
        "x_m": trajectory.x_m.copy(),
        "v_mps": trajectory.v_mps.copy(),
    }
    for name, values in internal_series.items():
        array = np.asarray(values)
        if array.shape[:1] != (internal_count,):
            raise ValueError(f"internal series {name!r} does not match the time grid")
        time_domain[name] = array.copy()

    # Existing consumers expect top-level per-sample metadata to have the same
    # length as the returned x/RSSI arrays.  Slice those arrays to report times,
    # but preserve the unsliced originals under ``time_domain`` above.
    for key, value in list(result.items()):
        if key.startswith("internal_"):
            continue
        if isinstance(value, np.ndarray) and value.shape[:1] == (internal_count,):
            # Keep a complete, auditable copy of every channel/receiver series
            # on the internal time grid.  The top-level arrays below remain
            # report-aligned for backward compatibility with existing
            # consumers (feature extraction, HI and the GUI).
            time_domain.setdefault(key, value.copy())
            result[key] = value[mask].copy()

    report_x = trajectory.report_x_m
    report_steps = np.abs(np.diff(report_x))
    result.update(
        {
            "sampling_mode": TIME_DOMAIN_SAMPLING,
            "report_time_s": trajectory.report_t_s.copy(),
            "report_position_m": report_x.copy(),
            "report_speed_mps": trajectory.report_v_mps.copy(),
            "report_mask_on_internal_grid": mask.copy(),
            "reported_spatial_interval_m_median": (
                float(np.median(report_steps)) if len(report_steps) else 0.0
            ),
            "time_domain": time_domain,
        }
    )
    return result


def generate_ideal_rssi(
    x_start: float = -400.0,
    x_end: float = 150.0,
    dx: float = 0.5,
    v: float = 20.0,
    antenna_x: float = 0.0,
    antenna_y: float = 5.0,
    main_lobe_dir_x: float = -1.0,
    main_lobe_dir_y: float = 0.0,
    G_max_dB: float = 12.0,
    theta_half_deg: float = 30.0,
    Pt_dBm: float = 20.0,
    Gr_dBi: float = 0.0,
    n: float = 2.8,
    PL0_dB: float = 40.0,
    d0: float = 1.0,
    sigma_shadow: float = 2.5,
    fc: float = 2.4e9,
    K_linear: float = DEFAULT_RICIAN_K_LINEAR,
    N_paths: int = 20,
    use_free_space: bool = False,
    seed: Optional[int] = None,
    max_antenna_attenuation_dB: float = 30.0,
    dual_antenna: bool = True,
    ap_height_m: float = 5.0,
    obm_height_m: float = 4.1,
    path_loss_breakpoint_m: float = 0.0,
    path_loss_exponent_far: float = 3.88,
    sampling_mode: str = LEGACY_SPATIAL_SAMPLING,
    simulation_step_s: float = 0.01,
    report_interval_s: float = 0.05,
    speed_profile: Optional[SpeedProfile] = None,
    direction: Optional[int] = None,
):
    """Generate deterministic link-budget RSSI without stochastic fading."""
    del sigma_shadow, K_linear, N_paths, seed
    if dx <= 0.0:
        raise ValueError("Require dx > 0")
    trajectory = _trajectory_for_sampling(
        sampling_mode=sampling_mode,
        x_start=x_start,
        x_end=x_end,
        v=v,
        simulation_step_s=simulation_step_s,
        report_interval_s=report_interval_s,
        speed_profile=speed_profile,
        direction=direction,
    )
    if trajectory is None:
        if x_end <= x_start:
            raise ValueError("Legacy spatial sampling requires x_end > x_start")
        x = np.arange(float(x_start), float(x_end) + float(dx) * 0.5, float(dx))
    else:
        x = trajectory.report_x_m.copy()
    vectors = np.column_stack((x - float(antenna_x), np.full(len(x), -float(antenna_y))))
    if ap_height_m < 0.0 or obm_height_m < 0.0:
        raise ValueError("AP and OBM antenna heights must be non-negative")
    height_difference = float(obm_height_m) - float(ap_height_m)
    distances = np.sqrt(np.sum(vectors**2, axis=1) + height_difference**2)
    lobe_1 = _unit_vector(main_lobe_dir_x, main_lobe_dir_y)
    gain_1 = _antenna_gain(
        vectors, lobe_1, G_max_dB, theta_half_deg, max_antenna_attenuation_dB
    )
    if dual_antenna:
        gain_2 = _antenna_gain(
            vectors, -lobe_1, G_max_dB, theta_half_deg, max_antenna_attenuation_dB
        )
        gains = np.maximum(gain_1, gain_2)
    else:
        gains = gain_1
    losses = _path_loss(
        distances,
        fc,
        n,
        PL0_dB,
        d0,
        use_free_space,
        path_loss_breakpoint_m,
        path_loss_exponent_far,
    )
    rssi = float(Pt_dBm) + gains + float(Gr_dBi) - losses
    return x, rssi, gains, losses, np.zeros(len(x), dtype=float), distances


def generate_rssi_simulation(
    x_start: float = -400.0,
    x_end: float = 150.0,
    dx: float = 0.5,
    v: float = 20.0,
    antenna_x: float = 0.0,
    antenna_y: float = 5.0,
    main_lobe_dir_x: float = -1.0,
    main_lobe_dir_y: float = 0.0,
    G_max_dB: float = 12.0,
    theta_half_deg: float = 30.0,
    Pt_dBm: float = 20.0,
    Gr_dBi: float = 0.0,
    n: float = 2.8,
    PL0_dB: float = 40.0,
    d0: float = 1.0,
    sigma_shadow: float = 2.5,
    fc: float = 2.4e9,
    K_linear: float = DEFAULT_RICIAN_K_LINEAR,
    N_paths: int = 20,
    use_free_space: bool = False,
    seed: Optional[int] = None,
    shadow_corr_distance_m: float = 15.0,
    measurement_window_m: Optional[float] = None,
    measurement_window_s: float = 0.05,
    receiver_noise_sigma_dB: float = 0.6,
    rssi_quantization_dB: float = 0.5,
    receiver_sensitivity_dBm: Optional[float] = -100.0,
    receiver_saturation_dBm: Optional[float] = -20.0,
    below_sensitivity_policy: str = DEFAULT_BELOW_SENSITIVITY_POLICY,
    max_antenna_attenuation_dB: float = 30.0,
    dual_antenna: bool = True,
    trip_power_sigma_dB: float = 0.8,
    position_alignment_sigma_m: float = 0.75,
    pointing_jitter_sigma_deg: float = 1.5,
    return_metadata: bool = False,
    ap_height_m: float = 5.0,
    obm_height_m: float = 4.1,
    path_loss_breakpoint_m: float = 0.0,
    path_loss_exponent_far: float = 3.88,
    shadow_sigma_far_dB: float = 4.2,
    rician_K_slope_dB_per_100m: float = 0.0,
    sampling_mode: str = LEGACY_SPATIAL_SAMPLING,
    simulation_step_s: float = 0.01,
    report_interval_s: float = 0.05,
    speed_profile: Optional[SpeedProfile] = None,
    direction: Optional[int] = None,
):
    """Generate an RSSI trace using a literature-constrained railway profile.

    ``N_paths`` is retained for API compatibility; the implementation now uses a
    spatial complex Gaussian process instead of a fragile aliased Jakes loop.
    """
    trajectory = _trajectory_for_sampling(
        sampling_mode=sampling_mode,
        x_start=x_start,
        x_end=x_end,
        v=v,
        simulation_step_s=simulation_step_s,
        report_interval_s=report_interval_s,
        speed_profile=speed_profile,
        direction=direction,
    )
    state = _build_channel_state(
        x_start=x_start,
        x_end=x_end,
        dx=dx,
        v=v,
        antenna_x=antenna_x,
        antenna_y=antenna_y,
        ap_height_m=ap_height_m,
        obm_height_m=obm_height_m,
        main_lobe_dir_x=main_lobe_dir_x,
        main_lobe_dir_y=main_lobe_dir_y,
        G_max_dB=G_max_dB,
        theta_half_deg=theta_half_deg,
        n=n,
        path_loss_breakpoint_m=path_loss_breakpoint_m,
        path_loss_exponent_far=path_loss_exponent_far,
        PL0_dB=PL0_dB,
        d0=d0,
        sigma_shadow=sigma_shadow,
        shadow_sigma_far_dB=shadow_sigma_far_dB,
        fc=fc,
        K_linear=K_linear,
        rician_K_slope_dB_per_100m=rician_K_slope_dB_per_100m,
        use_free_space=use_free_space,
        seed=seed,
        shadow_corr_distance_m=shadow_corr_distance_m,
        measurement_window_m=measurement_window_m,
        measurement_window_s=measurement_window_s,
        receiver_noise_sigma_dB=receiver_noise_sigma_dB,
        max_antenna_attenuation_dB=max_antenna_attenuation_dB,
        dual_antenna=dual_antenna,
        trip_power_sigma_dB=trip_power_sigma_dB,
        position_alignment_sigma_m=position_alignment_sigma_m,
        pointing_jitter_sigma_deg=pointing_jitter_sigma_deg,
        **_sampling_build_arguments(trajectory, sampling_mode),
    )
    selection_info = None
    if dual_antenna:
        selection_info = _selection_metadata(
            state,
            Pt_dBm,
            Gr_dBi,
            receiver_sensitivity_dBm,
            receiver_saturation_dBm,
            rssi_quantization_dB,
            below_sensitivity_policy,
        )
        raw = np.asarray(selection_info["selected_candidate_raw_rssi_dBm"], dtype=float)
        rssi = _reported_rssi(
            raw,
            receiver_sensitivity_dBm,
            receiver_saturation_dBm,
            rssi_quantization_dB,
            below_sensitivity_policy,
        )
        selected = np.asarray(selection_info["selected_antenna"], dtype=int)
        selected_gain = np.where(selected == 1, state.gain_1_dB, state.gain_2_dB)
    else:
        rssi, raw = _compose_rssi(
            state,
            state.gain_1_dB,
            Pt_dBm,
            Gr_dBi,
            receiver_sensitivity_dBm,
            receiver_saturation_dBm,
            rssi_quantization_dB,
            below_sensitivity_policy,
        )
        selected_gain = state.gain_1_dB
    receiver_report = apply_receiver_reporting(
        raw,
        receiver_sensitivity_dBm,
        receiver_saturation_dBm,
        rssi_quantization_dB,
        below_sensitivity_policy,
    )
    rssi = receiver_report.reported_rssi_dBm
    sample_mask = (
        np.asarray(trajectory.report_mask, dtype=bool)
        if trajectory is not None
        else slice(None)
    )
    result = (
        state.x[sample_mask],
        rssi[sample_mask],
        selected_gain[sample_mask],
        state.path_loss_dB[sample_mask],
        state.small_fading_dB[sample_mask],
        state.distances[sample_mask],
    )
    if not return_metadata:
        return result
    metadata = dict(state.metadata)
    metadata.update(
        {
            "gain_1_dB": state.gain_1_dB.copy(),
            "gain_2_dB": state.gain_2_dB.copy(),
            "shadow_fading_dB": state.shadow_dB.copy(),
            "small_fading_dB": state.small_fading_dB.copy(),
            "instantaneous_small_fading_dB": state.small_fading_dB.copy(),
            "measurement_noise_dB": state.measurement_noise_dB.copy(),
            "raw_rssi_dBm": raw.copy(),
            "reported_rssi_dBm": rssi.copy(),
            "receiver_sensitivity_dBm": receiver_sensitivity_dBm,
            "receiver_saturation_dBm": receiver_saturation_dBm,
            "rssi_quantization_dB": float(rssi_quantization_dB),
            "below_sensitivity_policy": str(below_sensitivity_policy),
            "quantized_rssi_before_limits_dBm": receiver_report.quantized_rssi_dBm.copy(),
            "receiver_report_status": receiver_report.status.copy(),
            "receiver_below_sensitivity_mask": receiver_report.below_sensitivity_mask.copy(),
            "receiver_saturation_mask": receiver_report.saturation_mask.copy(),
            "receiver_missing_mask": receiver_report.missing_mask.copy(),
            "N_paths_compatibility_value_ignored": int(N_paths),
        }
    )
    if selection_info is not None:
        metadata.update(selection_info)
    metadata = _report_sampled_metadata(
        metadata,
        trajectory,
        reported_rssi_dBm=rssi,
        raw_rssi_dBm=raw,
        receiver_report_status=receiver_report.status,
        selected_antenna_gain_dB=selected_gain,
        path_loss_dB=state.path_loss_dB,
        small_fading_dB=state.small_fading_dB,
        shadow_fading_dB=state.shadow_dB,
        ap_to_obm_distance_m=state.distances,
    )
    return result + (metadata,)


def generate_time_domain_rssi_simulation(
    *,
    simulation_step_s: float = 0.01,
    report_interval_s: float = 0.05,
    speed_profile: Optional[SpeedProfile] = None,
    direction: Optional[int] = None,
    **generator_kwargs,
):
    """Named time-driven API that keeps the legacy six-array tuple contract.

    Returned arrays are aligned to OBM report times.  With
    ``return_metadata=True``, the unsampled internal ``RSSI(t)`` and trajectory
    remain available under ``metadata['time_domain']``.
    """
    params = dict(generator_kwargs)
    params.pop("sampling_mode", None)
    params.update(
        {
            "sampling_mode": TIME_DOMAIN_SAMPLING,
            "simulation_step_s": float(simulation_step_s),
            "report_interval_s": float(report_interval_s),
            "speed_profile": speed_profile,
            "direction": direction,
        }
    )
    return generate_rssi_simulation(**params)


def generate_stage1_obm_observation(**generator_kwargs) -> Dict[str, Any]:
    """Generate the explicit paper phase-1 AP-to-OBM RSSI observation.

    This is a named research API over the backward-compatible tuple generator.
    It fixes the topology to one AP with two opposite directional antennas and
    one onboard OBM.  The returned ``reported_rssi_dBm`` is the value available
    to the health indicator and diagnostic models; candidate branch powers are
    retained for mechanism validation and fault attribution.
    """
    params = dict(generator_kwargs)
    if not bool(params.get("dual_antenna", True)):
        raise ValueError("第一阶段论文场景必须启用 AP 的两副反向定向天线")
    params["dual_antenna"] = True
    params["return_metadata"] = True
    result = generate_rssi_simulation(**params)
    x, reported, gain, path_loss, small_fading, distances, metadata = result
    return {
        "topology_schema_version": metadata["topology_schema_version"],
        "observation_definition": metadata["observation_definition"],
        "x_m": x,
        "t_s": metadata.get("report_time_s"),
        "v_mps": metadata.get("report_speed_mps"),
        "reported_rssi_dBm": reported,
        "receiver_report_status": metadata["receiver_report_status"],
        "selected_antenna_gain_dB": gain,
        "path_loss_dB": path_loss,
        "small_fading_dB": small_fading,
        "ap_to_obm_distance_m": distances,
        "antenna_1_candidate_raw_rssi_dBm": metadata[
            "antenna_1_candidate_raw_rssi_dBm"
        ],
        "antenna_2_candidate_raw_rssi_dBm": metadata[
            "antenna_2_candidate_raw_rssi_dBm"
        ],
        "antenna_1_candidate_instantaneous_rssi_dBm": metadata[
            "antenna_1_candidate_instantaneous_rssi_dBm"
        ],
        "antenna_2_candidate_instantaneous_rssi_dBm": metadata[
            "antenna_2_candidate_instantaneous_rssi_dBm"
        ],
        "selected_antenna": metadata["selected_antenna"],
        "selection_margin_dB": metadata["selection_margin_dB"],
        "geometry_model": metadata["geometry_model"],
        "measurement_window_samples": metadata["measurement_window_samples"],
        "receiver_integration_alignment": metadata["receiver_integration_alignment"],
        "receiver_integration_coordinate_unit": metadata[
            "receiver_integration_coordinate_unit"
        ],
        "receiver_averaging_domain": metadata["receiver_averaging_domain"],
        "metadata": metadata,
    }


def generate_fault_rssi_pair(
    fault_type: str | Sequence[tuple[str, Mapping[str, float]]],
    fault_kwargs: Optional[Dict[str, float]] = None,
    *,
    x_start: float = -400.0,
    x_end: float = 150.0,
    dx: float = 0.5,
    v: float = 20.0,
    antenna_x: float = 0.0,
    antenna_y: float = 5.0,
    main_lobe_dir_x: float = -1.0,
    main_lobe_dir_y: float = 0.0,
    G_max_dB: float = 12.0,
    theta_half_deg: float = 30.0,
    Pt_dBm: float = 20.0,
    Gr_dBi: float = 0.0,
    n: float = 2.8,
    PL0_dB: float = 40.0,
    d0: float = 1.0,
    sigma_shadow: float = 2.5,
    fc: float = 2.4e9,
    K_linear: float = DEFAULT_RICIAN_K_LINEAR,
    N_paths: int = 20,
    use_free_space: bool = False,
    seed: Optional[int] = None,
    shadow_corr_distance_m: float = 15.0,
    measurement_window_m: Optional[float] = None,
    measurement_window_s: float = 0.05,
    receiver_noise_sigma_dB: float = 0.6,
    rssi_quantization_dB: float = 0.5,
    receiver_sensitivity_dBm: Optional[float] = -100.0,
    receiver_saturation_dBm: Optional[float] = -20.0,
    below_sensitivity_policy: str = DEFAULT_BELOW_SENSITIVITY_POLICY,
    max_antenna_attenuation_dB: float = 30.0,
    dual_antenna: bool = True,
    trip_power_sigma_dB: float = 0.8,
    position_alignment_sigma_m: float = 0.75,
    pointing_jitter_sigma_deg: float = 1.5,
    return_metadata: bool = False,
    ap_height_m: float = 5.0,
    obm_height_m: float = 4.1,
    path_loss_breakpoint_m: float = 0.0,
    path_loss_exponent_far: float = 3.88,
    shadow_sigma_far_dB: float = 4.2,
    rician_K_slope_dB_per_100m: float = 0.0,
    sampling_mode: str = LEGACY_SPATIAL_SAMPLING,
    simulation_step_s: float = 0.01,
    report_interval_s: float = 0.05,
    speed_profile: Optional[SpeedProfile] = None,
    direction: Optional[int] = None,
):
    """Generate a healthy/faulty pair under one shared stochastic channel.

    Supported physical mechanisms are the five failure modes used by the local
    predictive-maintenance paper: total link attenuation, per-antenna power
    reduction, and per-antenna pointing tilt.  A sequence of ``(type, params)``
    pairs produces a physically composed multi-label fault without collapsing
    it into one of the five single-fault labels.
    """
    if isinstance(fault_type, str):
        fault_specs = [(fault_type, dict(fault_kwargs or {}))]
    else:
        if fault_kwargs:
            raise ValueError("Composite faults must carry parameters per component")
        fault_specs = [
            (str(component_name), dict(component_params))
            for component_name, component_params in fault_type
        ]
        if len(fault_specs) < 2:
            raise ValueError("Composite fault requires at least two components")
    trajectory = _trajectory_for_sampling(
        sampling_mode=sampling_mode,
        x_start=x_start,
        x_end=x_end,
        v=v,
        simulation_step_s=simulation_step_s,
        report_interval_s=report_interval_s,
        speed_profile=speed_profile,
        direction=direction,
    )
    state = _build_channel_state(
        x_start=x_start,
        x_end=x_end,
        dx=dx,
        v=v,
        antenna_x=antenna_x,
        antenna_y=antenna_y,
        ap_height_m=ap_height_m,
        obm_height_m=obm_height_m,
        main_lobe_dir_x=main_lobe_dir_x,
        main_lobe_dir_y=main_lobe_dir_y,
        G_max_dB=G_max_dB,
        theta_half_deg=theta_half_deg,
        n=n,
        path_loss_breakpoint_m=path_loss_breakpoint_m,
        path_loss_exponent_far=path_loss_exponent_far,
        PL0_dB=PL0_dB,
        d0=d0,
        sigma_shadow=sigma_shadow,
        shadow_sigma_far_dB=shadow_sigma_far_dB,
        fc=fc,
        K_linear=K_linear,
        rician_K_slope_dB_per_100m=rician_K_slope_dB_per_100m,
        use_free_space=use_free_space,
        seed=seed,
        shadow_corr_distance_m=shadow_corr_distance_m,
        measurement_window_m=measurement_window_m,
        measurement_window_s=measurement_window_s,
        receiver_noise_sigma_dB=receiver_noise_sigma_dB,
        max_antenna_attenuation_dB=max_antenna_attenuation_dB,
        dual_antenna=dual_antenna,
        trip_power_sigma_dB=trip_power_sigma_dB,
        position_alignment_sigma_m=position_alignment_sigma_m,
        pointing_jitter_sigma_deg=pointing_jitter_sigma_deg,
        **_sampling_build_arguments(trajectory, sampling_mode),
    )
    if dual_antenna:
        healthy_branch_1_raw, healthy_branch_2_raw = _candidate_branch_raw_rssi(
            state, Pt_dBm, Gr_dBi
        )
        healthy_selected_antenna = np.where(
            healthy_branch_1_raw >= healthy_branch_2_raw, 1, 2
        ).astype(np.int8)
        healthy_raw = np.maximum(healthy_branch_1_raw, healthy_branch_2_raw)
        healthy_gain = np.where(
            healthy_selected_antenna == 1, state.gain_1_dB, state.gain_2_dB
        )
        healthy = _reported_rssi(
            healthy_raw,
            receiver_sensitivity_dBm,
            receiver_saturation_dBm,
            rssi_quantization_dB,
            below_sensitivity_policy,
        )
    else:
        healthy, healthy_raw = _compose_rssi(
            state,
            state.gain_1_dB,
            Pt_dBm,
            Gr_dBi,
            receiver_sensitivity_dBm,
            receiver_saturation_dBm,
            rssi_quantization_dB,
            below_sensitivity_policy,
        )
        healthy_gain = state.gain_1_dB

    supported_types = {
        "全链路功率衰减",
        "天线1功率下降",
        "天线2功率下降",
        "天线1方向偏移",
        "天线2方向偏移",
    }
    global_attenuation_dB = 0.0
    branch_1_drop_dB = 0.0
    branch_2_drop_dB = 0.0
    antenna_1_tilt_deg = 0.0
    antenna_2_tilt_deg = 0.0
    mechanisms = []
    for component_name, params in fault_specs:
        if component_name not in supported_types:
            raise ValueError(f"Unsupported physical fault type: {component_name}")
        if component_name == "全链路功率衰减":
            global_attenuation_dB += max(float(params.get("atten_dB", 6.0)), 0.0)
            mechanisms.append("transmit/feed-chain power attenuation")
        elif component_name == "天线1功率下降":
            branch_1_drop_dB += max(float(params.get("drop_dB", 7.0)), 0.0)
            mechanisms.append("antenna-1 branch gain reduction")
        elif component_name == "天线2功率下降":
            branch_2_drop_dB += max(float(params.get("drop_dB", 7.0)), 0.0)
            mechanisms.append("antenna-2 branch gain reduction")
        elif component_name == "天线1方向偏移":
            antenna_1_tilt_deg += abs(float(params.get("tilt_deg", 10.0)))
            mechanisms.append("antenna-1 pattern rotation")
        elif component_name == "天线2方向偏移":
            antenna_2_tilt_deg += abs(float(params.get("tilt_deg", 10.0)))
            mechanisms.append("antenna-2 pattern rotation")

    vectors = np.column_stack(
        (
            state.x + float(state.metadata["position_offset_m"]) - float(antenna_x),
            np.full(len(state.x), -float(antenna_y)),
        )
    )
    gain_1 = state.gain_1_dB.copy()
    gain_2 = state.gain_2_dB.copy()
    if antenna_1_tilt_deg > 0.0:
        gain_1 = _antenna_gain(
            vectors,
            _rotate(state.effective_main_lobe_1, -antenna_1_tilt_deg),
            G_max_dB,
            theta_half_deg,
            max_antenna_attenuation_dB,
        )
    if antenna_2_tilt_deg > 0.0 and dual_antenna:
        gain_2 = _antenna_gain(
            vectors,
            _rotate(state.effective_main_lobe_2, antenna_2_tilt_deg),
            G_max_dB,
            theta_half_deg,
            max_antenna_attenuation_dB,
        )
    gain_1 = gain_1 - branch_1_drop_dB
    gain_2 = gain_2 - branch_2_drop_dB
    faulty_power = float(Pt_dBm) - global_attenuation_dB
    mechanism = " + ".join(mechanisms)

    if dual_antenna:
        faulty_branch_1_raw, _ = _raw_rssi_for_gain(
            state, gain_1, faulty_power, Gr_dBi
        )
        faulty_branch_2_raw, _ = _raw_rssi_for_gain(
            state, gain_2, faulty_power, Gr_dBi
        )
        faulty_selected_antenna = np.where(
            faulty_branch_1_raw >= faulty_branch_2_raw, 1, 2
        ).astype(np.int8)
        faulty_raw = np.maximum(faulty_branch_1_raw, faulty_branch_2_raw)
        faulty_gain = np.where(faulty_selected_antenna == 1, gain_1, gain_2)
        faulty = _reported_rssi(
            faulty_raw,
            receiver_sensitivity_dBm,
            receiver_saturation_dBm,
            rssi_quantization_dB,
            below_sensitivity_policy,
        )
    else:
        faulty_gain = gain_1
        faulty, faulty_raw = _compose_rssi(
            state,
            faulty_gain,
            faulty_power,
            Gr_dBi,
            receiver_sensitivity_dBm,
            receiver_saturation_dBm,
            rssi_quantization_dB,
            below_sensitivity_policy,
        )
    healthy_receiver_report = apply_receiver_reporting(
        healthy_raw,
        receiver_sensitivity_dBm,
        receiver_saturation_dBm,
        rssi_quantization_dB,
        below_sensitivity_policy,
    )
    faulty_receiver_report = apply_receiver_reporting(
        faulty_raw,
        receiver_sensitivity_dBm,
        receiver_saturation_dBm,
        rssi_quantization_dB,
        below_sensitivity_policy,
    )
    healthy = healthy_receiver_report.reported_rssi_dBm
    faulty = faulty_receiver_report.reported_rssi_dBm
    sample_mask = (
        np.asarray(trajectory.report_mask, dtype=bool)
        if trajectory is not None
        else slice(None)
    )
    if not return_metadata:
        return state.x[sample_mask], healthy[sample_mask], faulty[sample_mask]

    metadata = dict(state.metadata)
    component_parameters = {
        name: dict(params) for name, params in fault_specs
    }
    metadata_fault_type = fault_specs[0][0] if len(fault_specs) == 1 else "复合故障"
    metadata.update(
        {
            "fault_type": metadata_fault_type,
            "fault_components": [name for name, _ in fault_specs],
            "fault_parameters": (
                dict(fault_specs[0][1])
                if len(fault_specs) == 1
                else component_parameters
            ),
            "fault_mechanism": mechanism,
            "healthy_gain_dB": healthy_gain.copy(),
            "faulty_gain_dB": faulty_gain.copy(),
            "faulty_gain_1_dB": gain_1.copy(),
            "faulty_gain_2_dB": gain_2.copy(),
            "global_attenuation_dB": float(global_attenuation_dB),
            "antenna_1_power_drop_dB": float(branch_1_drop_dB),
            "antenna_2_power_drop_dB": float(branch_2_drop_dB),
            "antenna_1_tilt_deg": float(antenna_1_tilt_deg),
            "antenna_2_tilt_deg": float(antenna_2_tilt_deg),
            "shadow_fading_dB": state.shadow_dB.copy(),
            "small_fading_dB": state.small_fading_dB.copy(),
            "healthy_raw_rssi_dBm": healthy_raw.copy(),
            "faulty_raw_rssi_dBm": faulty_raw.copy(),
            "fault_delta_dB": faulty - healthy,
            "receiver_sensitivity_dBm": receiver_sensitivity_dBm,
            "receiver_saturation_dBm": receiver_saturation_dBm,
            "rssi_quantization_dB": float(rssi_quantization_dB),
            "below_sensitivity_policy": str(below_sensitivity_policy),
            "healthy_quantized_rssi_before_limits_dBm": (
                healthy_receiver_report.quantized_rssi_dBm.copy()
            ),
            "faulty_quantized_rssi_before_limits_dBm": (
                faulty_receiver_report.quantized_rssi_dBm.copy()
            ),
            "healthy_receiver_report_status": healthy_receiver_report.status.copy(),
            "faulty_receiver_report_status": faulty_receiver_report.status.copy(),
            "healthy_receiver_missing_mask": healthy_receiver_report.missing_mask.copy(),
            "faulty_receiver_missing_mask": faulty_receiver_report.missing_mask.copy(),
            "N_paths_compatibility_value_ignored": int(N_paths),
        }
    )
    if dual_antenna:
        metadata.update(
            {
                "healthy_antenna_1_candidate_raw_rssi_dBm": healthy_branch_1_raw.copy(),
                "healthy_antenna_2_candidate_raw_rssi_dBm": healthy_branch_2_raw.copy(),
                "healthy_selected_antenna": healthy_selected_antenna.copy(),
                "faulty_antenna_1_candidate_raw_rssi_dBm": faulty_branch_1_raw.copy(),
                "faulty_antenna_2_candidate_raw_rssi_dBm": faulty_branch_2_raw.copy(),
                "faulty_selected_antenna": faulty_selected_antenna.copy(),
            }
        )
    metadata = _report_sampled_metadata(
        metadata,
        trajectory,
        healthy_reported_rssi_dBm=healthy,
        faulty_reported_rssi_dBm=faulty,
        healthy_raw_rssi_dBm=healthy_raw,
        faulty_raw_rssi_dBm=faulty_raw,
        healthy_receiver_report_status=healthy_receiver_report.status,
        faulty_receiver_report_status=faulty_receiver_report.status,
        healthy_selected_gain_dB=healthy_gain,
        faulty_selected_gain_dB=faulty_gain,
        path_loss_dB=state.path_loss_dB,
        small_fading_dB=state.small_fading_dB,
        shadow_fading_dB=state.shadow_dB,
        ap_to_obm_distance_m=state.distances,
    )
    return state.x[sample_mask], healthy[sample_mask], faulty[sample_mask], metadata


def generate_composite_fault_rssi_pair(
    fault_components: Sequence[tuple[str, Mapping[str, float]]],
    **generator_kwargs,
):
    """Generate a multi-label fault pair while preserving component metadata."""
    return generate_fault_rssi_pair(
        list(fault_components),
        None,
        **generator_kwargs,
    )
