"""Literature-constrained plausibility checks for simulated railway RSSI.

This module deliberately uses the term *plausibility*, not ground-truth
validation.  Public measurement ranges can reject obviously unrealistic
parameters, while final validation still requires aligned healthy traces from
the target railway system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from scipy import stats

from receiver_measurement import (
    causal_trailing_linear_average,
    integrate_rssi_dbm_causally,
    mw_to_dbm,
)
from signal_generation import (
    LEGACY_SPATIAL_SAMPLING,
    STAGE1_OBSERVATION_DEFINITION,
    STAGE1_TOPOLOGY_SCHEMA_VERSION,
    TIME_DOMAIN_SAMPLING,
    generate_fault_rssi_pair,
    generate_ideal_rssi,
    generate_rssi_simulation,
    generate_stage1_obm_observation,
)


REFERENCE_RANGES = {
    "path_loss_exponent": {
        "min": 2.0,
        "max": 3.5,
        "default": 2.8,
        "basis": "HSR measurement at 465 MHz reported n=2.8; interval is a conservative LOS railway prior.",
        "source": "https://new.eurasip.org/Proceedings/Eusipco/eusipco2019/Proceedings/papers/1570531144.pdf",
    },
    "shadow_sigma_dB": {
        "min": 2.0,
        "max": 6.0,
        "default": 2.5,
        "basis": "Railway measurements report about 2.5 dB and 2.47-4.93 dB; 3GPP RMa LOS uses 4/6 dB.",
        "source": "https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.00.00_60/tr_138901v180000p.pdf",
    },
    "shadow_corr_distance_m": {
        "min": 5.0,
        "max": 120.0,
        "default": 15.0,
        "basis": "2.35 GHz HSR measurements reported 8.3, 11.9 and 17.7 m; other open HSR scenarios can exceed 100 m.",
        "source": "https://doi.org/10.1155/2016/8782671",
    },
    "rician_K_dB": {
        "min": 0.0,
        "max": 10.0,
        "default": 4.16,
        "basis": "930 MHz HSR viaduct measurements reported overall mean K=4.16 dB and standard deviation 3.94 dB.",
        "source": "https://wides.usc.edu/Updated_pdf/he2013.pdf",
    },
}


FAULT_PROBES = {
    "全链路功率衰减": {"atten_dB": 8.0},
    "天线1功率下降": {"drop_dB": 8.0},
    "天线2功率下降": {"drop_dB": 8.0},
    "天线1方向偏移": {"tilt_deg": 12.0},
    "天线2方向偏移": {"tilt_deg": 12.0},
}


PAPER_STAGE1_REFERENCE = {
    "title": "An AI-Based Method for Predictive Maintenance of Railway Radio Communication Systems",
    "venue": "2024 IEEE Intelligent Transportation Systems Conference",
    "doi": "10.1109/ITSC58415.2024.10919734",
    "implemented_scope": (
        "one wayside AP, two directional antennas pointing in opposite track "
        "directions, one onboard OBM, receiver-reported AP downlink RSSI"
    ),
    "deferred_scope": "multiple APs, front/rear OBMs, association and handover",
    "explicit_assumption": (
        "the paper does not specify RF branch combining; phase 1 uses maximum "
        "candidate antenna power before receiver reporting"
    ),
}


NONSTATIONARY_RAILWAY_EVIDENCE = {
    "hsr_hilly_2p4GHz": {
        "source": "https://doi.org/10.1155/2013/378407",
        "evidence": (
            "2.4 GHz measurement reported a two-slope path-loss model with "
            "breakpoint about 788.6 m, exponents 2.40/3.88 and shadow standard "
            "deviations 3.3/4.2 dB."
        ),
        "applicability": (
            "mechanism and optional stress-profile prior only; its macro-link "
            "distance range is not the current short-range trackside AP geometry"
        ),
    },
    "hsr_viaduct_930MHz": {
        "source": "https://wides.usc.edu/Updated_pdf/he2013.pdf",
        "evidence": (
            "measurements reported a 400 m path-loss breakpoint, distance-varying "
            "Rician K and different shadow variance before/after the breakpoint"
        ),
        "applicability": (
            "mechanism support only; 930 MHz, 43 dBm macro BS and about 20 m "
            "relative antenna height must not be used as trackside-WLAN defaults"
        ),
    },
    "hsr_shadow_correlation": {
        "source": "https://doi.org/10.1109/ChinaCom.2011.6158343",
        "evidence": (
            "lognormal shadowing and spatial autocorrelation were measured; a "
            "double-exponential ACF fitted the cited viaduct data better"
        ),
        "applicability": "retain current exponential model until coefficients are available or calibrated",
    },
}


def _in_range(value: float, range_name: str) -> bool:
    target = REFERENCE_RANGES[range_name]
    return float(target["min"]) <= float(value) <= float(target["max"])


def _spatial_correlation_distance(curves: Iterable[np.ndarray], dx: float) -> float:
    accumulated = None
    count = 0
    for curve in curves:
        values = np.asarray(curve, dtype=float)
        values = values - np.mean(values)
        if len(values) < 3 or np.allclose(values, 0.0):
            continue
        fft_size = 1 << int(np.ceil(np.log2(2 * len(values) - 1)))
        spectrum = np.fft.rfft(values, n=fft_size)
        acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[: len(values)]
        overlap = np.arange(len(values), 0, -1, dtype=float)
        acf = acf / overlap
        acf = acf / max(float(acf[0]), 1e-12)
        accumulated = acf if accumulated is None else accumulated + acf
        count += 1
    if accumulated is None or count == 0:
        return float("nan")
    mean_acf = accumulated / float(count)
    below = np.flatnonzero(mean_acf <= np.exp(-1.0))
    if len(below) == 0:
        return float((len(mean_acf) - 1) * dx)
    return float(below[0] * dx)


def _fit_path_loss_exponent(distances: np.ndarray, path_loss_dB: np.ndarray) -> float:
    distances = np.asarray(distances, dtype=float)
    path_loss_dB = np.asarray(path_loss_dB, dtype=float)
    valid = np.isfinite(distances) & np.isfinite(path_loss_dB) & (distances > 1.0)
    slope, _ = np.polyfit(np.log10(distances[valid]), path_loss_dB[valid], 1)
    return float(slope / 10.0)


def _fault_mechanism_checks(generator_kwargs: Dict, seed: int) -> Dict:
    profiles = {}
    details = {}
    antenna_x = float(generator_kwargs.get("antenna_x", 0.0))
    for index, (fault_name, fault_params) in enumerate(FAULT_PROBES.items()):
        x, healthy, faulty = generate_fault_rssi_pair(
            fault_name,
            fault_params,
            seed=seed + index,
            **generator_kwargs,
        )
        delta = np.asarray(faulty) - np.asarray(healthy)
        profiles[fault_name] = delta
        left = x <= antenna_x
        right = x > antenna_x
        left_effect = float(np.mean(np.abs(delta[left]))) if np.any(left) else 0.0
        right_effect = float(np.mean(np.abs(delta[right]))) if np.any(right) else 0.0
        details[fault_name] = {
            "mean_delta_dB": float(np.mean(delta)),
            "profile_std_dB": float(np.std(delta)),
            "left_mean_abs_delta_dB": left_effect,
            "right_mean_abs_delta_dB": right_effect,
        }

    global_ok = details["全链路功率衰减"]["mean_delta_dB"] < -6.0
    global_ok = global_ok and details["全链路功率衰减"]["profile_std_dB"] < 0.75
    antenna_1_power_ok = (
        details["天线1功率下降"]["left_mean_abs_delta_dB"]
        > details["天线1功率下降"]["right_mean_abs_delta_dB"] + 1.0
    )
    antenna_2_power_ok = (
        details["天线2功率下降"]["right_mean_abs_delta_dB"]
        > details["天线2功率下降"]["left_mean_abs_delta_dB"] + 1.0
    )
    antenna_1_tilt_ok = (
        details["天线1方向偏移"]["profile_std_dB"] > 0.5
        and details["天线1方向偏移"]["left_mean_abs_delta_dB"]
        > details["天线1方向偏移"]["right_mean_abs_delta_dB"] + 0.5
    )
    antenna_2_tilt_ok = (
        details["天线2方向偏移"]["profile_std_dB"] > 0.5
        and details["天线2方向偏移"]["right_mean_abs_delta_dB"]
        > details["天线2方向偏移"]["left_mean_abs_delta_dB"] + 0.5
    )

    checks = {
        "global_attenuation_is_link_wide": bool(global_ok),
        "antenna_1_power_is_left_selective": bool(antenna_1_power_ok),
        "antenna_2_power_is_right_selective": bool(antenna_2_power_ok),
        "antenna_1_tilt_is_nonconstant_and_left_selective": bool(antenna_1_tilt_ok),
        "antenna_2_tilt_is_nonconstant_and_right_selective": bool(antenna_2_tilt_ok),
    }
    return {"checks": checks, "details": details}


def _parameter_sensitivity_checks(generator_kwargs: Dict, seed: int) -> Dict:
    """Verify that key GUI parameters change the generated signal, not metadata only."""
    base = dict(generator_kwargs)
    base.pop("seed", None)
    base.pop("return_metadata", None)

    slow_kwargs = dict(base)
    slow_kwargs.update(
        {
            "v": 10.0,
            "measurement_window_m": None,
            "measurement_window_s": 0.05,
            "sampling_mode": "fixed_time_report",
            "simulation_step_s": 0.01,
            "report_interval_s": 0.05,
            "speed_profile": None,
            "direction": None,
        }
    )
    fast_kwargs = dict(slow_kwargs)
    fast_kwargs["v"] = 30.0
    slow = generate_rssi_simulation(seed=seed, return_metadata=True, **slow_kwargs)
    fast = generate_rssi_simulation(seed=seed, return_metadata=True, **fast_kwargs)
    fast_raw_on_slow_grid = np.interp(
        slow[0], fast[0], fast[6]["raw_rssi_dBm"]
    )
    speed_curve_difference = float(
        np.mean(np.abs(slow[6]["raw_rssi_dBm"] - fast_raw_on_slow_grid))
    )
    slow_lookup = {round(float(x), 8): index for index, x in enumerate(slow[0])}
    fast_lookup = {round(float(x), 8): index for index, x in enumerate(fast[0])}
    common_positions = sorted(set(slow_lookup) & set(fast_lookup))
    # Compare both physical antenna patterns, not the post-integration selected
    # branch.  Causal averaging may legitimately move the branch crossover, but
    # speed must not alter either antenna's geometric gain at the same position.
    same_position_gain_error = max(
        max(
            abs(
                float(slow[6]["gain_1_dB"][slow_lookup[x]])
                - float(fast[6]["gain_1_dB"][fast_lookup[x]])
            ),
            abs(
                float(slow[6]["gain_2_dB"][slow_lookup[x]])
                - float(fast[6]["gain_2_dB"][fast_lookup[x]])
            ),
        )
        for x in common_positions
    )
    same_position_path_loss_error = max(
        abs(float(slow[3][slow_lookup[x]]) - float(fast[3][fast_lookup[x]]))
        for x in common_positions
    )

    fixed_distance_slow_kwargs = dict(slow_kwargs)
    fixed_distance_slow_kwargs["measurement_window_m"] = 1.0
    fixed_distance_slow_kwargs["sampling_mode"] = "fixed_space_legacy"
    fixed_distance_slow_kwargs["speed_profile"] = None
    fixed_distance_slow_kwargs["direction"] = None
    fixed_distance_fast_kwargs = dict(fixed_distance_slow_kwargs)
    fixed_distance_fast_kwargs["v"] = 30.0
    fixed_distance_slow = generate_rssi_simulation(
        seed=seed, return_metadata=True, **fixed_distance_slow_kwargs
    )
    fixed_distance_fast = generate_rssi_simulation(
        seed=seed, return_metadata=True, **fixed_distance_fast_kwargs
    )
    fixed_distance_curve_difference = float(
        np.max(np.abs(fixed_distance_slow[1] - fixed_distance_fast[1]))
    )

    low_n_kwargs = dict(base)
    low_n_kwargs["n"] = 2.2
    high_n_kwargs = dict(base)
    high_n_kwargs["n"] = 3.4
    low_n = generate_rssi_simulation(seed=seed, return_metadata=True, **low_n_kwargs)
    high_n = generate_rssi_simulation(seed=seed, return_metadata=True, **high_n_kwargs)
    path_loss_difference = float(np.mean(high_n[3] - low_n[3]))
    rssi_difference = float(np.mean(low_n[1]) - np.mean(high_n[1]))

    details = {
        "speed_10mps_effective_window_m": float(slow[6]["measurement_window_m"]),
        "speed_30mps_effective_window_m": float(fast[6]["measurement_window_m"]),
        "speed_receiver_averaged_rssi_mean_abs_difference_dB": speed_curve_difference,
        "speed_10mps_report_spacing_m": float(
            slow[6]["reported_spatial_interval_m_median"]
        ),
        "speed_30mps_report_spacing_m": float(
            fast[6]["reported_spatial_interval_m_median"]
        ),
        "speed_10mps_trip_duration_s": float(slow[6]["trip_duration_s"]),
        "speed_30mps_trip_duration_s": float(fast[6]["trip_duration_s"]),
        "same_position_gain_max_error_dB": float(same_position_gain_error),
        "same_position_path_loss_max_error_dB": float(
            same_position_path_loss_error
        ),
        "fixed_distance_window_speed_max_difference_dB": fixed_distance_curve_difference,
        "path_loss_exponent_mean_path_loss_difference_dB": path_loss_difference,
        "path_loss_exponent_mean_rssi_difference_dB": rssi_difference,
    }
    checks = {
        "speed_changes_fixed_time_receiver_averaging": bool(
            slow[6]["measurement_window_m"] == 0.5
            and fast[6]["measurement_window_m"] == 1.5
            and np.isclose(slow[6]["reported_spatial_interval_m_median"], 0.5)
            and np.isclose(fast[6]["reported_spatial_interval_m_median"], 1.5)
            and slow[6]["trip_duration_s"] > fast[6]["trip_duration_s"]
            and same_position_gain_error < 1e-12
            and same_position_path_loss_error < 1e-12
            and speed_curve_difference > 0.01
        ),
        "path_loss_exponent_changes_path_loss_and_rssi": bool(
            path_loss_difference > 1.0 and rssi_difference > 1.0
        ),
        "fixed_distance_override_removes_speed_from_receiver_window": bool(
            fixed_distance_slow[6]["measurement_window_mode"]
            == "fixed_distance_override"
            and fixed_distance_fast[6]["measurement_window_mode"]
            == "fixed_distance_override"
            and fixed_distance_curve_difference < 1e-12
        ),
    }
    return {"checks": checks, "details": details}


def _stage1_topology_checks(generator_kwargs: Dict, seed: int) -> Dict:
    """Validate the phase-1 paper topology and the AP-to-OBM observation chain."""
    params = dict(generator_kwargs)
    params.pop("seed", None)
    params.pop("return_metadata", None)
    antenna_x = float(params.get("antenna_x", 0.0))
    probe_dx = 0.5
    time_domain_sampling = (
        str(params.get("sampling_mode", LEGACY_SPATIAL_SAMPLING))
        == TIME_DOMAIN_SAMPLING
    )
    params.update(
        {
            "x_start": antenna_x - 100.0,
            "x_end": antenna_x + 100.0,
            "dx": probe_dx,
            "dual_antenna": True,
            "sigma_shadow": 0.0,
            "K_linear": 1.0e24,
            "receiver_noise_sigma_dB": 0.0,
            "rssi_quantization_dB": 0.0,
            "receiver_sensitivity_dBm": None,
            "trip_power_sigma_dB": 0.0,
            "position_alignment_sigma_m": 0.0,
            "pointing_jitter_sigma_deg": 0.0,
            # Geometry symmetry is an instantaneous-link property.  Disable
            # causal receiver integration here so its physically expected lag
            # is not mistaken for an asymmetric antenna pattern.
            "measurement_window_m": None if time_domain_sampling else 0.0,
            "measurement_window_s": 0.0,
            "seed": int(seed),
        }
    )
    observation = generate_stage1_obm_observation(**params)
    metadata = observation["metadata"]
    x = np.asarray(observation["x_m"], dtype=float)
    branch_1 = np.asarray(
        observation["antenna_1_candidate_raw_rssi_dBm"], dtype=float
    )
    branch_2 = np.asarray(
        observation["antenna_2_candidate_raw_rssi_dBm"], dtype=float
    )
    selected = np.asarray(observation["selected_antenna"], dtype=int)
    selected_raw = np.maximum(branch_1, branch_2)
    reported = np.asarray(observation["reported_rssi_dBm"], dtype=float)
    lobe_1 = np.asarray(metadata["antenna_1_main_lobe_unit_vector"], dtype=float)
    lobe_2 = np.asarray(metadata["antenna_2_main_lobe_unit_vector"], dtype=float)

    left = x < antenna_x
    right = x > antenna_x
    left_branch_1_rate = float(np.mean(selected[left] == 1))
    right_branch_2_rate = float(np.mean(selected[right] == 2))
    selection_switches = int(np.count_nonzero(np.diff(selected)))
    branch_symmetry_error = float(np.max(np.abs(branch_1 - branch_2[::-1])))
    combined_symmetry_error = float(np.max(np.abs(reported - reported[::-1])))
    selected_raw_error = float(
        np.max(
            np.abs(
                np.asarray(metadata["selected_candidate_raw_rssi_dBm"], dtype=float)
                - selected_raw
            )
        )
    )
    reporting_error = float(np.max(np.abs(reported - selected_raw)))

    checks = {
        "stage1_topology_contract_is_explicit": bool(
            observation["topology_schema_version"] == STAGE1_TOPOLOGY_SCHEMA_VERSION
            and metadata["ap_count"] == 1
            and metadata["obm_count"] == 1
            and metadata["dual_antenna"]
        ),
        "stage1_observation_is_onboard_downlink_rssi": bool(
            observation["observation_definition"] == STAGE1_OBSERVATION_DEFINITION
            and metadata["link_direction"] == "AP_downlink_to_onboard_OBM"
        ),
        "two_ap_antennas_point_in_opposite_directions": bool(
            np.isclose(np.linalg.norm(lobe_1), 1.0, atol=1e-12)
            and np.isclose(np.linalg.norm(lobe_2), 1.0, atol=1e-12)
            and np.isclose(float(np.dot(lobe_1, lobe_2)), -1.0, atol=1e-12)
        ),
        "antenna_1_serves_negative_track_side": left_branch_1_rate >= 0.99,
        "antenna_2_serves_positive_track_side": right_branch_2_rate >= 0.99,
        "antenna_selection_has_one_center_transition": selection_switches == 1,
        "selected_power_is_maximum_candidate_power": selected_raw_error < 1e-10,
        "receiver_reporting_follows_branch_selection": reporting_error < 1e-10,
        "opposite_branches_are_geometrically_symmetric": branch_symmetry_error < 1e-8,
        "healthy_combined_curve_is_geometrically_symmetric": combined_symmetry_error < 1e-8,
    }
    details = {
        "topology_schema_version": observation["topology_schema_version"],
        "observation_definition": observation["observation_definition"],
        "left_side_antenna_1_selection_rate": left_branch_1_rate,
        "right_side_antenna_2_selection_rate": right_branch_2_rate,
        "selection_switch_count": selection_switches,
        "branch_symmetry_max_error_dB": branch_symmetry_error,
        "combined_symmetry_max_error_dB": combined_symmetry_error,
        "selected_power_max_error_dB": selected_raw_error,
        "reporting_order_max_error_dB": reporting_error,
    }
    return {"checks": checks, "details": details}


def _deterministic_probe_params(generator_kwargs: Dict, seed: int) -> Dict:
    params = dict(generator_kwargs)
    params.pop("seed", None)
    params.pop("return_metadata", None)
    antenna_x = float(params.get("antenna_x", 0.0))
    time_domain_sampling = (
        str(params.get("sampling_mode", LEGACY_SPATIAL_SAMPLING))
        == TIME_DOMAIN_SAMPLING
    )
    params.update(
        {
            "x_start": antenna_x - 100.0,
            "x_end": antenna_x + 100.0,
            "dx": 0.5,
            "sigma_shadow": 0.0,
            "K_linear": 1.0e24,
            "receiver_noise_sigma_dB": 0.0,
            "receiver_sensitivity_dBm": None,
            "receiver_saturation_dBm": None,
            "rssi_quantization_dB": 0.0,
            "trip_power_sigma_dB": 0.0,
            "position_alignment_sigma_m": 0.0,
            "pointing_jitter_sigma_deg": 0.0,
            # Link-budget reconciliation is evaluated before receiver
            # integration; integration itself has a dedicated reconstruction
            # check in ``_receiver_measurement_checks``.
            "measurement_window_m": None if time_domain_sampling else 0.0,
            "measurement_window_s": 0.0,
            "seed": int(seed),
        }
    )
    return params


def _link_budget_and_antenna_checks(generator_kwargs: Dict, seed: int) -> Dict:
    params = _deterministic_probe_params(generator_kwargs, seed)
    result = generate_rssi_simulation(return_metadata=True, **params)
    _, _, selected_gain, path_loss, _, distances, metadata = result
    selected_raw = np.asarray(metadata["selected_candidate_raw_rssi_dBm"], dtype=float)
    expected_raw = (
        float(params.get("Pt_dBm", 20.0))
        + float(metadata["trip_power_offset_dB"])
        + np.asarray(selected_gain, dtype=float)
        + float(params.get("Gr_dBi", 0.0))
        - np.asarray(path_loss, dtype=float)
        + np.asarray(metadata["shadow_fading_dB"], dtype=float)
        + np.asarray(metadata["small_fading_dB"], dtype=float)
        + np.asarray(metadata["measurement_noise_dB"], dtype=float)
    )
    reconciliation_error = float(np.max(np.abs(selected_raw - expected_raw)))

    ideal_params = {
        "x_start": 1.0,
        "x_end": 1.001,
        "dx": 0.001,
        "antenna_x": 0.0,
        "antenna_y": 0.0,
        "ap_height_m": 0.0,
        "obm_height_m": 0.0,
        "main_lobe_dir_x": 1.0,
        "main_lobe_dir_y": 0.0,
        "dual_antenna": False,
        "PL0_dB": float(params.get("PL0_dB", 40.0)),
        "d0": 1.0,
        "n": float(params.get("n", 2.8)),
        "use_free_space": False,
    }
    reference_result = generate_ideal_rssi(**ideal_params)
    reference_losses = np.asarray(reference_result[3], dtype=float)
    reference_loss_error = abs(reference_losses[0] - ideal_params["PL0_dB"])
    reference_continuity_step = abs(reference_losses[1] - reference_losses[0])

    gain_1 = np.asarray(metadata["gain_1_dB"], dtype=float)
    gain_2 = np.asarray(metadata["gain_2_dB"], dtype=float)
    configured_gain = float(params.get("G_max_dB", 12.0))
    configured_front_back = float(params.get("max_antenna_attenuation_dB", 30.0))
    antenna_1_front_back = float(gain_1[0] - gain_1[-1])
    antenna_2_front_back = float(gain_2[-1] - gain_2[0])
    boresight_error = float(
        max(abs(gain_1[0] - configured_gain), abs(gain_2[-1] - configured_gain))
    )

    center = int(np.argmin(distances))
    left_outward = np.asarray(path_loss[: center + 1], dtype=float)[::-1]
    right_outward = np.asarray(path_loss[center:], dtype=float)
    path_loss_monotonic = bool(
        np.all(np.diff(left_outward) >= -1e-12)
        and np.all(np.diff(right_outward) >= -1e-12)
    )
    expected_nearest_distance = float(
        np.hypot(
            float(params.get("antenna_y", 5.0)),
            float(params.get("ap_height_m", 5.0))
            - float(params.get("obm_height_m", 4.1)),
        )
    )
    nearest_distance_error = abs(float(distances[center]) - expected_nearest_distance)
    increased_height_params = dict(params)
    increased_height_params["ap_height_m"] = float(
        increased_height_params.get("obm_height_m", 4.1)
    ) + 10.0
    increased_height = generate_rssi_simulation(
        return_metadata=True, **increased_height_params
    )
    height_path_loss_increase = float(
        increased_height[3][center] - np.asarray(path_loss, dtype=float)[center]
    )
    checks = {
        "link_budget_terms_reconcile": reconciliation_error < 1e-10,
        "reference_distance_path_loss_is_continuous": bool(
            reference_loss_error < 1e-12 and reference_continuity_step < 0.05
        ),
        "path_loss_increases_outward_from_ap": path_loss_monotonic,
        "antenna_boresight_approaches_configured_max_gain": boresight_error < 0.1,
        "antenna_front_to_back_attenuation_is_enforced": bool(
            antenna_1_front_back > configured_front_back - 0.1
            and antenna_2_front_back > configured_front_back - 0.1
        ),
        "three_dimensional_ap_obm_distance_is_correct": bool(
            metadata["geometry_model"]
            == "3D_distance_with_horizontal_azimuth_antenna_pattern"
            and nearest_distance_error < 1e-10
        ),
        "vertical_separation_increases_path_loss": height_path_loss_increase > 1.0,
    }
    details = {
        "link_budget_reconciliation_max_error_dB": reconciliation_error,
        "reference_distance_loss_error_dB": float(reference_loss_error),
        "reference_distance_continuity_step_dB": float(reference_continuity_step),
        "antenna_1_front_to_back_dB": antenna_1_front_back,
        "antenna_2_front_to_back_dB": antenna_2_front_back,
        "boresight_gain_max_error_dB": boresight_error,
        "nearest_3d_distance_error_m": float(nearest_distance_error),
        "ten_meter_vertical_separation_path_loss_increase_dB": height_path_loss_increase,
    }
    return {"checks": checks, "details": details}


def _receiver_measurement_checks(generator_kwargs: Dict, seed: int) -> Dict:
    base = _deterministic_probe_params(generator_kwargs, seed)

    quantized_params = dict(base)
    quantized_params["rssi_quantization_dB"] = 0.5
    quantized = generate_rssi_simulation(return_metadata=True, **quantized_params)
    quantized_values = np.asarray(quantized[1], dtype=float)
    quantization_grid_error = float(
        np.max(np.abs(quantized_values / 0.5 - np.round(quantized_values / 0.5)))
    )
    quantization_raw_error = float(
        np.max(np.abs(quantized_values - quantized[6]["raw_rssi_dBm"]))
    )

    floor_params = dict(base)
    floor_params.update(
        {
            "Pt_dBm": 0.0,
            "receiver_sensitivity_dBm": -70.0,
            "receiver_saturation_dBm": -20.0,
        }
    )
    floored = np.asarray(generate_rssi_simulation(**floor_params)[1], dtype=float)
    floor_fraction = float(np.mean(np.isclose(floored, -70.0)))

    saturation_params = dict(base)
    saturation_params.update(
        {
            "Pt_dBm": 60.0,
            "receiver_sensitivity_dBm": -100.0,
            "receiver_saturation_dBm": -30.0,
        }
    )
    saturated = np.asarray(generate_rssi_simulation(**saturation_params)[1], dtype=float)
    saturation_fraction = float(np.mean(np.isclose(saturated, -30.0)))

    unaligned_params = dict(saturation_params)
    unaligned_params.update(
        {
            "receiver_sensitivity_dBm": -99.3,
            "receiver_saturation_dBm": -20.2,
            "rssi_quantization_dB": 0.5,
        }
    )
    unaligned = np.asarray(generate_rssi_simulation(**unaligned_params)[1], dtype=float)

    integration_params = dict(base)
    integration_params.update(
        {
            "K_linear": 0.0,
            "measurement_window_m": 1.5,
            "receiver_noise_sigma_dB": 0.0,
        }
    )
    integrated = generate_rssi_simulation(
        return_metadata=True, **integration_params
    )
    integration_metadata = integrated[6]
    # The receiver integrates on the fine internal time grid and only then
    # emits report samples.  Reconstruct the average at the same internal
    # layer; using the report-aligned series here would incorrectly treat the
    # reporting interval as the channel simulation interval.
    integration_time_domain = integration_metadata.get("time_domain", {})
    instantaneous_branch_1 = np.asarray(
        integration_time_domain.get(
            "antenna_1_candidate_instantaneous_rssi_dBm",
            integration_metadata["antenna_1_candidate_instantaneous_rssi_dBm"],
        ),
        dtype=float,
    )
    window_samples = int(integration_metadata["measurement_window_samples"])
    coordinate_unit = str(integration_metadata["receiver_integration_coordinate_unit"])
    if coordinate_unit == "s":
        integration_coordinate = np.asarray(
            integration_time_domain.get(
                "t_s",
                integration_metadata["internal_time_s"],
            ),
            dtype=float,
        )
    else:
        internal_x = np.asarray(
            integration_time_domain.get(
                "x_m",
                integration_metadata.get("internal_position_m", integrated[0]),
            ),
            dtype=float,
        )
        integration_coordinate = np.concatenate(
            (np.asarray([0.0]), np.cumsum(np.abs(np.diff(internal_x))))
        )
    window_width = float(integration_metadata["receiver_integration_window_width"])
    expected_integration = integrate_rssi_dbm_causally(
        instantaneous_branch_1,
        integration_coordinate,
        window_width,
    )
    expected_branch_1_raw = mw_to_dbm(expected_integration.averaged_power_mW)
    actual_branch_1_raw = np.asarray(
        integration_time_domain.get(
            "antenna_1_candidate_raw_rssi_dBm",
            integration_metadata["antenna_1_candidate_raw_rssi_dBm"],
        ),
        dtype=float,
    )
    complete_link_average_error = float(
        np.max(np.abs(expected_branch_1_raw - actual_branch_1_raw))
    )
    incorrect_dB_average = (
        causal_trailing_linear_average(
            instantaneous_branch_1 + 200.0,
            integration_coordinate,
            window_width,
        ).averaged_power_mW
        - 200.0
    )
    linear_vs_dB_average_difference = float(
        np.mean(np.abs(expected_branch_1_raw - incorrect_dB_average))
    )

    invalid_limits_rejected = False
    try:
        invalid = dict(base)
        invalid.update(
            {
                "receiver_sensitivity_dBm": -20.0,
                "receiver_saturation_dBm": -30.0,
            }
        )
        generate_rssi_simulation(**invalid)
    except ValueError:
        invalid_limits_rejected = True

    checks = {
        "receiver_quantization_uses_configured_step": quantization_grid_error < 1e-10,
        "receiver_quantization_error_is_bounded": quantization_raw_error <= 0.2500000001,
        "receiver_sensitivity_floor_is_active": bool(
            np.min(floored) >= -70.0 and floor_fraction > 0.0
        ),
        "receiver_saturation_limit_is_active": bool(
            np.max(saturated) <= -30.0 and saturation_fraction > 0.0
        ),
        "receiver_limits_hold_after_unaligned_quantization": bool(
            np.min(unaligned) >= -99.3 and np.max(unaligned) <= -20.2
        ),
        "invalid_receiver_limits_are_rejected": invalid_limits_rejected,
        "complete_candidate_link_is_averaged_in_linear_power": bool(
            window_samples > 1 and complete_link_average_error < 1e-10
        ),
        "receiver_does_not_average_rssi_in_dB": linear_vs_dB_average_difference > 0.01,
        "receiver_integration_is_causal": (
            integration_metadata["receiver_integration_alignment"]
            == "causal_trailing_window"
        ),
        "receiver_preserves_raw_value_and_report_status": bool(
            "receiver_report_status" in integration_metadata
            and "raw_rssi_dBm" in integration_metadata
        ),
    }
    details = {
        "quantization_grid_max_error": quantization_grid_error,
        "quantization_raw_max_error_dB": quantization_raw_error,
        "sensitivity_floor_fraction": floor_fraction,
        "saturation_fraction": saturation_fraction,
        "unaligned_reported_min_dBm": float(np.min(unaligned)),
        "unaligned_reported_max_dBm": float(np.max(unaligned)),
        "complete_link_linear_average_max_error_dB": complete_link_average_error,
        "linear_vs_dB_average_mean_difference_dB": linear_vs_dB_average_difference,
        "receiver_integration_coordinate_unit": coordinate_unit,
        "receiver_integration_window_width": window_width,
    }
    return {"checks": checks, "details": details}


def _small_scale_fading_checks(generator_kwargs: Dict, seed: int) -> Dict:
    base = _deterministic_probe_params(generator_kwargs, seed)
    base.update({"x_start": -50.0, "x_end": 50.0})
    low_k = []
    high_k = []
    for index in range(12):
        low_params = dict(base)
        low_params.update({"K_linear": 0.0, "seed": seed + index})
        high_params = dict(base)
        high_params.update({"K_linear": 1.0e6, "seed": seed + index})
        low_k.append(
            generate_rssi_simulation(return_metadata=True, **low_params)[6][
                "small_fading_dB"
            ]
        )
        high_k.append(
            generate_rssi_simulation(return_metadata=True, **high_params)[6][
                "small_fading_dB"
            ]
        )
    low_values = np.concatenate(low_k)
    high_values = np.concatenate(high_k)
    low_std = float(np.std(low_values))
    high_std = float(np.std(high_values))
    low_linear_mean = float(np.mean(10.0 ** (low_values / 10.0)))
    checks = {
        "K_zero_has_rayleigh_like_power_variability": low_std > 4.0,
        "large_K_approaches_deterministic_los": high_std < 0.05,
        "rayleigh_linear_power_is_normalized": 0.85 <= low_linear_mean <= 1.15,
        "Rician_variability_decreases_with_K": high_std < low_std * 0.05,
    }
    details = {
        "K_zero_fading_std_dB": low_std,
        "K_large_fading_std_dB": high_std,
        "K_zero_linear_power_mean": low_linear_mean,
    }
    return {"checks": checks, "details": details}


def _optional_nonstationary_channel_checks(generator_kwargs: Dict, seed: int) -> Dict:
    """Verify paper-motivated optional mechanisms without enabling them by default."""
    params = _deterministic_probe_params(generator_kwargs, seed)
    params.update(
        {
            "n": 2.40,
            "path_loss_breakpoint_m": 50.0,
            "path_loss_exponent_far": 3.88,
            "sigma_shadow": 2.0,
            "shadow_sigma_far_dB": 5.0,
            "K_linear": 10.0 ** (8.0 / 10.0),
            "rician_K_slope_dB_per_100m": -1.0,
        }
    )
    result = generate_rssi_simulation(return_metadata=True, **params)
    distances = np.asarray(result[5], dtype=float)
    path_loss = np.asarray(result[3], dtype=float)
    metadata = result[6]
    near = (distances >= 10.0) & (distances <= 45.0)
    far = (distances >= 60.0) & (distances <= 100.0)
    near_exponent = _fit_path_loss_exponent(distances[near], path_loss[near])
    far_exponent = _fit_path_loss_exponent(distances[far], path_loss[far])

    ideal = generate_ideal_rssi(
        x_start=49.99,
        x_end=50.01,
        dx=0.01,
        antenna_x=0.0,
        antenna_y=0.0,
        ap_height_m=0.0,
        obm_height_m=0.0,
        main_lobe_dir_x=1.0,
        main_lobe_dir_y=0.0,
        dual_antenna=False,
        n=2.40,
        path_loss_breakpoint_m=50.0,
        path_loss_exponent_far=3.88,
    )
    breakpoint_step_dB = float(np.max(np.abs(np.diff(ideal[3]))))

    shadow_curves = []
    for index in range(40):
        shadow_params = dict(params)
        shadow_params["seed"] = seed + index
        shadow_curves.append(
            generate_rssi_simulation(return_metadata=True, **shadow_params)[6][
                "shadow_fading_dB"
            ]
        )
    shadow_matrix = np.vstack(shadow_curves)
    near_shadow_std = float(np.mean(np.std(shadow_matrix[:, near], axis=0, ddof=1)))
    far_shadow_std = float(np.mean(np.std(shadow_matrix[:, far], axis=0, ddof=1)))

    k_profile = np.asarray(metadata["rician_K_profile_dB"], dtype=float)
    fitted_k_slope_per_100m = float(np.polyfit(distances, k_profile, 1)[0] * 100.0)

    default_params = _deterministic_probe_params(generator_kwargs, seed + 1_000)
    # This check verifies the factory/default stage-1 profile, independent of any
    # optional stress profile currently selected in the GUI.
    default_params.update(
        {
            "path_loss_breakpoint_m": 0.0,
            "rician_K_slope_dB_per_100m": 0.0,
        }
    )
    default_result = generate_rssi_simulation(return_metadata=True, **default_params)
    default_metadata = default_result[6]
    checks = {
        "optional_two_slope_path_loss_recovers_near_exponent": abs(near_exponent - 2.40) < 0.02,
        "optional_two_slope_path_loss_recovers_far_exponent": abs(far_exponent - 3.88) < 0.02,
        "optional_two_slope_path_loss_is_continuous": breakpoint_step_dB < 0.01,
        "optional_far_shadow_variability_uses_configured_sigma": bool(
            1.7 <= near_shadow_std <= 2.3 and 4.3 <= far_shadow_std <= 5.7
        ),
        "optional_distance_varying_K_recovers_configured_slope": abs(
            fitted_k_slope_per_100m + 1.0
        ) < 0.01,
        "nonstationary_models_are_disabled_by_default": bool(
            default_metadata["path_loss_model"] == "single_slope_log_distance"
            and np.allclose(
                default_metadata["shadow_sigma_profile_dB"],
                default_metadata["shadow_sigma_dB"],
            )
            and np.allclose(
                default_metadata["rician_K_profile_dB"],
                default_metadata["rician_K_dB"],
            )
        ),
    }
    details = {
        "fitted_near_path_loss_exponent": near_exponent,
        "fitted_far_path_loss_exponent": far_exponent,
        "breakpoint_adjacent_max_step_dB": breakpoint_step_dB,
        "near_shadow_empirical_std_dB": near_shadow_std,
        "far_shadow_empirical_std_dB": far_shadow_std,
        "fitted_K_slope_dB_per_100m": fitted_k_slope_per_100m,
        "default_path_loss_model": default_metadata["path_loss_model"],
    }
    return {"checks": checks, "details": details}


def _reproducibility_and_nuisance_checks(generator_kwargs: Dict, seed: int) -> Dict:
    base = dict(generator_kwargs)
    base.pop("seed", None)
    base.pop("return_metadata", None)
    first = generate_rssi_simulation(seed=seed, return_metadata=True, **base)
    repeated = generate_rssi_simulation(seed=seed, return_metadata=True, **base)
    changed = generate_rssi_simulation(seed=seed + 1, return_metadata=True, **base)

    deterministic = _deterministic_probe_params(generator_kwargs, seed)
    position_reference = dict(deterministic)
    position_shifted = dict(deterministic)
    position_shifted["position_alignment_sigma_m"] = 2.0
    position_a = generate_rssi_simulation(return_metadata=True, **position_reference)
    position_b = generate_rssi_simulation(return_metadata=True, **position_shifted)

    pointing_reference = dict(deterministic)
    pointing_shifted = dict(deterministic)
    pointing_shifted["pointing_jitter_sigma_deg"] = 5.0
    pointing_a = generate_rssi_simulation(return_metadata=True, **pointing_reference)
    pointing_b = generate_rssi_simulation(return_metadata=True, **pointing_shifted)

    trip_reference = dict(deterministic)
    trip_shifted = dict(deterministic)
    trip_shifted["trip_power_sigma_dB"] = 2.0
    trip_a = generate_rssi_simulation(return_metadata=True, **trip_reference)
    trip_b = generate_rssi_simulation(return_metadata=True, **trip_shifted)
    trip_delta = np.asarray(trip_b[6]["raw_rssi_dBm"]) - np.asarray(
        trip_a[6]["raw_rssi_dBm"]
    )

    checks = {
        "same_seed_reproduces_identical_observation": bool(
            np.array_equal(first[1], repeated[1])
            and np.array_equal(first[6]["shadow_fading_dB"], repeated[6]["shadow_fading_dB"])
        ),
        "different_seed_changes_stochastic_observation": bool(
            not np.array_equal(first[1], changed[1])
        ),
        "position_alignment_error_changes_physical_geometry": bool(
            abs(float(position_b[6]["position_offset_m"])) > 1e-12
            and not np.array_equal(position_a[5], position_b[5])
        ),
        "pointing_jitter_changes_directional_gain": bool(
            abs(float(pointing_b[6]["pointing_jitter_deg"])) > 1e-12
            and not np.array_equal(pointing_a[2], pointing_b[2])
        ),
        "trip_offset_is_curve_wide_not_point_noise": bool(
            abs(float(np.mean(trip_delta))) > 1e-12
            and float(np.std(trip_delta)) < 1e-10
        ),
    }
    details = {
        "different_seed_mean_abs_difference_dB": float(
            np.mean(np.abs(first[1] - changed[1]))
        ),
        "position_offset_probe_m": float(position_b[6]["position_offset_m"]),
        "pointing_jitter_probe_deg": float(pointing_b[6]["pointing_jitter_deg"]),
        "trip_offset_probe_dB": float(np.mean(trip_delta)),
        "trip_offset_spatial_std_dB": float(np.std(trip_delta)),
    }
    return {"checks": checks, "details": details}


def _fault_severity_checks(generator_kwargs: Dict, seed: int) -> Dict:
    base = dict(generator_kwargs)
    base.pop("seed", None)
    base.pop("return_metadata", None)
    base.update(
        {
            "receiver_sensitivity_dBm": None,
            "receiver_saturation_dBm": None,
            "rssi_quantization_dB": 0.0,
        }
    )
    metrics = {}
    checks = {}
    definitions = (
        ("全链路功率衰减", "atten_dB", "all", (3.0, 6.0, 9.0, 12.0)),
        ("天线1功率下降", "drop_dB", "left", (3.0, 6.0, 9.0, 12.0)),
        ("天线2功率下降", "drop_dB", "right", (3.0, 6.0, 9.0, 12.0)),
        ("天线1方向偏移", "tilt_deg", "left", (3.0, 6.0, 12.0, 18.0)),
        ("天线2方向偏移", "tilt_deg", "right", (3.0, 6.0, 12.0, 18.0)),
    )
    for fault_index, (fault_name, parameter_name, support, severity_levels) in enumerate(
        definitions
    ):
        level_metrics = []
        for level in severity_levels:
            seed_metrics = []
            for repeat in range(5):
                x, healthy, faulty = generate_fault_rssi_pair(
                    fault_name,
                    {parameter_name: level},
                    seed=seed + fault_index * 100 + repeat,
                    **base,
                )
                if support == "left":
                    mask = x <= float(base.get("antenna_x", 0.0))
                elif support == "right":
                    mask = x > float(base.get("antenna_x", 0.0))
                else:
                    mask = np.ones(len(x), dtype=bool)
                seed_metrics.append(float(np.mean(np.abs(faulty[mask] - healthy[mask]))))
            level_metrics.append(float(np.mean(seed_metrics)))
        differences = np.diff(level_metrics)
        metrics[fault_name] = {
            "severity_levels": list(severity_levels),
            "mean_abs_effect_dB": level_metrics,
            "successive_increase_dB": differences.tolist(),
            "seeds_per_level": 5,
        }
        checks[f"{fault_name}_severity_is_monotonic"] = bool(np.all(differences > 0.05))
    return {"checks": checks, "details": metrics}


def run_stage1_validation(
    seed: int = 2026,
    generator_kwargs: Optional[Dict] = None,
) -> Dict:
    """Run the paper phase-1 mechanism and software validation gates."""
    generator_kwargs = dict(generator_kwargs or {})
    if "seed" in generator_kwargs:
        seed = int(generator_kwargs.pop("seed"))
    component_reports = {
        "topology_and_observation": _stage1_topology_checks(generator_kwargs, seed),
        "link_budget_and_antennas": _link_budget_and_antenna_checks(
            generator_kwargs, seed + 1_000
        ),
        "receiver_measurement": _receiver_measurement_checks(
            generator_kwargs, seed + 2_000
        ),
        "small_scale_fading": _small_scale_fading_checks(
            generator_kwargs, seed + 3_000
        ),
        "optional_nonstationary_channel": _optional_nonstationary_channel_checks(
            generator_kwargs, seed + 3_500
        ),
        "reproducibility_and_nuisance": _reproducibility_and_nuisance_checks(
            generator_kwargs, seed + 4_000
        ),
        "parameter_sensitivity": _parameter_sensitivity_checks(
            generator_kwargs, seed + 5_000
        ),
        "fault_mechanisms": _fault_mechanism_checks(
            generator_kwargs, seed + 6_000
        ),
        "fault_severity": _fault_severity_checks(generator_kwargs, seed + 7_000),
    }
    checks = {
        name: passed
        for report in component_reports.values()
        for name, passed in report["checks"].items()
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "pass" if not failed else "warning",
        "evidence_level": "mechanism_check",
        "claim_limit": (
            "Software and physical-direction checks only; no target-line healthy "
            "calibration or external real-fault validation has been performed."
        ),
        "seed": int(seed),
        "reference": PAPER_STAGE1_REFERENCE,
        "literature_model_evidence": NONSTATIONARY_RAILWAY_EVIDENCE,
        "checks": checks,
        "failed_checks": failed,
        "components": component_reports,
        "known_limitations": [
            "AP and OBM heights are configurable engineering priors, not target-line measurements.",
            "No multi-AP association, handover, or front/rear dual-OBM behavior.",
            "No missing-report process because downstream feature pipelines do not yet impute missing RSSI.",
            "Maximum-power AP antenna branch selection is an explicit assumption pending equipment evidence.",
            "Parameters are literature/equipment priors, not target-line calibration results.",
        ],
    }


def run_realism_validation(
    n_curves: int = 30,
    seed: int = 2026,
    generator_kwargs: Optional[Dict] = None,
) -> Dict:
    """Run statistical, spatial and fault-mechanism plausibility checks."""
    if n_curves < 5:
        raise ValueError("n_curves must be at least 5")
    generator_kwargs = dict(generator_kwargs or {})
    if "seed" in generator_kwargs:
        seed = int(generator_kwargs.pop("seed"))

    rssi_curves = []
    shadow_curves = []
    first_result = None
    for index in range(n_curves):
        result = generate_rssi_simulation(
            seed=seed + index,
            return_metadata=True,
            **generator_kwargs,
        )
        if first_result is None:
            first_result = result
        rssi_curves.append(np.asarray(result[1], dtype=float))
        shadow_curves.append(np.asarray(result[6]["shadow_fading_dB"], dtype=float))

    assert first_result is not None
    x, _, _, path_loss_dB, _, distances, metadata = first_result
    dx = float(np.mean(np.diff(x)))
    rssi_matrix = np.vstack(rssi_curves)
    shadow_matrix = np.vstack(shadow_curves)
    shadow_flat = shadow_matrix.ravel()

    mean_curve = np.mean(rssi_matrix, axis=0)
    correlations = []
    for curve in rssi_matrix:
        if np.std(curve) > 0.0 and np.std(mean_curve) > 0.0:
            correlations.append(float(np.corrcoef(curve, mean_curve)[0, 1]))

    metrics = {
        "path_loss_exponent_fitted": _fit_path_loss_exponent(distances, path_loss_dB),
        "shadow_sigma_dB": float(np.mean(np.std(shadow_matrix, axis=1, ddof=1))),
        "shadow_skewness": float(stats.skew(shadow_flat, bias=False)),
        "shadow_excess_kurtosis": float(stats.kurtosis(shadow_flat, fisher=True, bias=False)),
        "shadow_corr_distance_m": _spatial_correlation_distance(shadow_curves, dx),
        "rician_K_dB": float(metadata["rician_K_dB"]),
        "cross_trip_curve_correlation_mean": float(np.mean(correlations)),
        "cross_trip_curve_correlation_min": float(np.min(correlations)),
        "positionwise_between_trip_std_dB": float(np.mean(np.std(rssi_matrix, axis=0, ddof=1))),
        "reported_rssi_min_dBm": float(np.min(rssi_matrix)),
        "reported_rssi_max_dBm": float(np.max(rssi_matrix)),
        "reported_rssi_mean_dBm": float(np.mean(rssi_matrix)),
    }

    checks = {
        "path_loss_exponent_in_measurement_prior": _in_range(
            metrics["path_loss_exponent_fitted"], "path_loss_exponent"
        ),
        "shadow_sigma_in_measurement_prior": _in_range(
            metrics["shadow_sigma_dB"], "shadow_sigma_dB"
        ),
        "shadow_marginal_is_approximately_gaussian": (
            abs(metrics["shadow_skewness"]) < 0.5
            and abs(metrics["shadow_excess_kurtosis"]) < 1.0
        ),
        "shadow_spatial_correlation_in_measurement_prior": _in_range(
            metrics["shadow_corr_distance_m"], "shadow_corr_distance_m"
        ),
        "rician_K_in_measurement_prior": _in_range(metrics["rician_K_dB"], "rician_K_dB"),
        "aligned_trip_shape_is_repeatable": metrics["cross_trip_curve_correlation_mean"] >= 0.75,
        "rssi_values_are_finite": bool(np.isfinite(rssi_matrix).all()),
    }

    stage1_report = run_stage1_validation(
        seed=seed + 10_000, generator_kwargs=generator_kwargs
    )
    all_checks = {
        **checks,
        **stage1_report["checks"],
    }
    failed = [name for name, passed in all_checks.items() if not passed]
    status = "pass" if not failed else "warning"
    return {
        "status": status,
        "interpretation": (
            "Literature-constrained plausibility passed; this is not a substitute for target-system field validation."
            if status == "pass"
            else "One or more literature/statistical checks need calibration or model review."
        ),
        "n_curves": int(n_curves),
        "seed": int(seed),
        "metrics": metrics,
        "checks": all_checks,
        "failed_checks": failed,
        "evidence_level": "mechanism_check_and_literature_prior_range_check",
        "fault_mechanisms": stage1_report["components"]["fault_mechanisms"]["details"],
        "parameter_sensitivity": stage1_report["components"]["parameter_sensitivity"]["details"],
        "paper_stage1": stage1_report,
        "reference_ranges": REFERENCE_RANGES,
    }


def _print_summary(report: Dict) -> None:
    print(f"仿真可信度检查: {report['status'].upper()}")
    for name, passed in report["checks"].items():
        print(f"  [{'通过' if passed else '警告'}] {name}")
    if report.get("metrics"):
        print("关键统计量:")
        for name, value in report["metrics"].items():
            print(f"  {name}: {value:.4f}")
    print("说明:", report.get("interpretation", report.get("claim_limit", "")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate simulated railway RSSI plausibility")
    parser.add_argument("--curves", type=int, default=30, help="number of healthy traces")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="run only phase-1 topology, mechanism and software checks",
    )
    parser.add_argument(
        "--gui-defaults",
        action="store_true",
        help="validate the current GUI time-domain sampling contract",
    )
    parser.add_argument("--output", type=Path, default=Path("models/simulation_realism_report.json"))
    args = parser.parse_args()

    generator_kwargs = None
    if args.gui_defaults:
        from gui_parameter_contract import default_gui_generator_params

        generator_kwargs = default_gui_generator_params()

    report = (
        run_stage1_validation(seed=args.seed, generator_kwargs=generator_kwargs)
        if args.stage1_only
        else run_realism_validation(
            n_curves=args.curves,
            seed=args.seed,
            generator_kwargs=generator_kwargs,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(report)
    print("报告已保存:", args.output.resolve())


if __name__ == "__main__":
    main()
