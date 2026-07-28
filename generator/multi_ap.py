"""Stage-2 multi-AP candidate measurements for front and rear OBMs.

This module deliberately stops before association and handover.  It produces
all AP candidate reports plus two diagnostic references:

* ideal geometric strongest AP, used to inspect coverage design;
* reported instantaneous strongest AP, used only as a noisy reference.

Neither reference is a serving-AP state, and neither performs handover.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from dual_obm import (
    DEFAULT_TRAIN_LENGTH_M,
    _resolved_generator_params,
    generate_dual_obm_observation,
)


MULTI_AP_CANDIDATE_SCHEMA_VERSION = (
    "paper-stage2-multi-ap-dual-obm-candidates-v1"
)


@dataclass(frozen=True)
class WaysideAP:
    """Physical parameters that may differ between wayside APs."""

    ap_id: str
    track_position_m: float
    lateral_offset_m: float = 5.0
    antenna_height_m: float = 5.0
    transmit_power_dBm: float = 20.0
    max_antenna_gain_dBi: float = 12.0
    half_power_beamwidth_deg: float = 30.0


DEFAULT_WAYSIDE_APS = (
    WaysideAP("AP-001", -300.0),
    WaysideAP("AP-002", 0.0),
    WaysideAP("AP-003", 300.0),
)


_AP_SPECIFIC_GENERATOR_KEYS = {
    "antenna_x",
    "antenna_y",
    "ap_height_m",
    "Pt_dBm",
    "G_max_dB",
    "theta_half_deg",
}


def _normalise_ap(ap: WaysideAP | Mapping[str, Any]) -> WaysideAP:
    if isinstance(ap, WaysideAP):
        result = ap
    elif isinstance(ap, Mapping):
        result = WaysideAP(**dict(ap))
    else:
        raise TypeError("each AP must be a WaysideAP or mapping")

    ap_id = str(result.ap_id).strip()
    if not ap_id:
        raise ValueError("AP identifier cannot be empty")
    numeric = {
        "track_position_m": result.track_position_m,
        "lateral_offset_m": result.lateral_offset_m,
        "antenna_height_m": result.antenna_height_m,
        "transmit_power_dBm": result.transmit_power_dBm,
        "max_antenna_gain_dBi": result.max_antenna_gain_dBi,
        "half_power_beamwidth_deg": result.half_power_beamwidth_deg,
    }
    if not all(np.isfinite(float(value)) for value in numeric.values()):
        raise ValueError(f"AP {ap_id} contains a non-finite parameter")
    if float(result.lateral_offset_m) < 0.0:
        raise ValueError(f"AP {ap_id} lateral offset must be non-negative")
    if float(result.antenna_height_m) < 0.0:
        raise ValueError(f"AP {ap_id} antenna height must be non-negative")
    if float(result.max_antenna_gain_dBi) < 0.0:
        raise ValueError(f"AP {ap_id} maximum gain must be non-negative")
    if not 0.0 < float(result.half_power_beamwidth_deg) <= 180.0:
        raise ValueError(
            f"AP {ap_id} half-power beamwidth must be in (0, 180] degrees"
        )
    return WaysideAP(
        ap_id=ap_id,
        track_position_m=float(result.track_position_m),
        lateral_offset_m=float(result.lateral_offset_m),
        antenna_height_m=float(result.antenna_height_m),
        transmit_power_dBm=float(result.transmit_power_dBm),
        max_antenna_gain_dBi=float(result.max_antenna_gain_dBi),
        half_power_beamwidth_deg=float(result.half_power_beamwidth_deg),
    )


def normalise_wayside_aps(
    aps: Iterable[WaysideAP | Mapping[str, Any]],
) -> tuple[WaysideAP, ...]:
    """Validate AP identities and return them in caller-specified order."""
    values = tuple(_normalise_ap(ap) for ap in aps)
    if len(values) < 2:
        raise ValueError("multi-AP candidate generation requires at least two APs")
    ids = [ap.ap_id for ap in values]
    if len(set(ids)) != len(ids):
        raise ValueError("AP identifiers must be unique")
    return values


def _reference_strongest(
    matrix_dBm: np.ndarray,
    available: np.ndarray,
    ap_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a reference argmax while preserving 'no available AP' samples."""
    values = np.asarray(matrix_dBm, dtype=float)
    mask = np.asarray(available, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("candidate matrix and availability mask must match")
    safe = np.where(mask & np.isfinite(values), values, -np.inf)
    any_available = np.any(np.isfinite(safe) & (safe > -np.inf), axis=0)
    indices = np.argmax(safe, axis=0).astype(np.int16)
    indices[~any_available] = -1
    strongest = np.full(values.shape[1], np.nan, dtype=float)
    columns = np.flatnonzero(any_available)
    strongest[columns] = values[indices[columns], columns]
    width = max(1, max(len(str(item)) for item in ap_ids))
    ids = np.full(values.shape[1], "", dtype=f"<U{width}")
    for index, ap_id in enumerate(ap_ids):
        ids[indices == index] = str(ap_id)
    return indices, ids, strongest


def _obm_candidate_view(
    role: str,
    ap_ids: Sequence[str],
    ap_links: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates = {
        ap_id: ap_links[ap_id][role]
        for ap_id in ap_ids
    }
    first = candidates[ap_ids[0]]
    report_matrix = np.vstack(
        [candidates[ap_id]["reported_rssi_dBm"] for ap_id in ap_ids]
    )
    raw_matrix = np.vstack(
        [candidates[ap_id]["raw_rssi_dBm"] for ap_id in ap_ids]
    )
    ideal_matrix = np.vstack(
        [candidates[ap_id]["ideal_rssi_dBm"] for ap_id in ap_ids]
    )
    available_matrix = np.vstack(
        [
            ~(
                candidates[ap_id]["receiver_below_sensitivity_mask"]
                | candidates[ap_id]["receiver_missing_mask"]
            )
            for ap_id in ap_ids
        ]
    )
    all_ideal_available = np.isfinite(ideal_matrix)
    report_index, report_id, report_value = _reference_strongest(
        report_matrix, available_matrix, ap_ids
    )
    ideal_index, ideal_id, ideal_value = _reference_strongest(
        ideal_matrix, all_ideal_available, ap_ids
    )
    return {
        "obm_id": first["obm_id"],
        "role": first["role"],
        "t_s": first["t_s"].copy(),
        "x_m": first["x_m"].copy(),
        "v_mps": first["v_mps"].copy(),
        "candidates_by_ap": candidates,
        "candidate_reported_rssi_matrix_dBm": report_matrix,
        "candidate_raw_rssi_matrix_dBm": raw_matrix,
        "candidate_ideal_rssi_matrix_dBm": ideal_matrix,
        "candidate_available_matrix": available_matrix,
        "reference_report_strongest_ap_index": report_index,
        "reference_report_strongest_ap_id": report_id,
        "reference_report_strongest_rssi_dBm": report_value,
        "reference_ideal_strongest_ap_index": ideal_index,
        "reference_ideal_strongest_ap_id": ideal_id,
        "reference_ideal_strongest_rssi_dBm": ideal_value,
    }


def generate_multi_ap_dual_obm_candidates(
    *,
    aps: Iterable[WaysideAP | Mapping[str, Any]] = DEFAULT_WAYSIDE_APS,
    train_length_m: float = DEFAULT_TRAIN_LENGTH_M,
    seed: int | None = 123,
    **generator_kwargs: Any,
) -> dict[str, Any]:
    """Generate every AP candidate report for both onboard OBMs.

    AP-specific geometry and transmitter parameters come from ``aps``.  The
    remaining generator parameters apply to every AP.  The list order is kept
    as the stable matrix-row order and is part of the experiment definition.
    """
    ambiguous = sorted(set(generator_kwargs) & _AP_SPECIFIC_GENERATOR_KEYS)
    if ambiguous:
        raise ValueError(
            "AP-specific parameters must be supplied inside aps, not as "
            f"global generator arguments: {ambiguous}"
        )
    ap_values = normalise_wayside_aps(aps)
    resolved = _resolved_generator_params(
        {**generator_kwargs, "seed": seed}
    )
    position_sigma_m = max(
        float(resolved["position_alignment_sigma_m"]), 0.0
    )
    root_sequence = np.random.SeedSequence(seed)
    children = root_sequence.spawn(len(ap_values) + 1)
    global_rng = np.random.default_rng(children[0])
    global_position_offset_m = float(
        global_rng.normal(0.0, position_sigma_m)
    )

    ap_links: dict[str, dict[str, Any]] = {}
    ap_seed_values: dict[str, int] = {}
    for ap, child_sequence in zip(ap_values, children[1:]):
        ap_seed = int(child_sequence.generate_state(1, dtype=np.uint32)[0])
        ap_seed_values[ap.ap_id] = ap_seed
        per_ap_kwargs = dict(generator_kwargs)
        per_ap_kwargs.update(
            {
                "antenna_x": ap.track_position_m,
                "antenna_y": ap.lateral_offset_m,
                "ap_height_m": ap.antenna_height_m,
                "Pt_dBm": ap.transmit_power_dBm,
                "G_max_dB": ap.max_antenna_gain_dBi,
                "theta_half_deg": ap.half_power_beamwidth_deg,
                "seed": ap_seed,
            }
        )
        ap_links[ap.ap_id] = generate_dual_obm_observation(
            train_length_m=train_length_m,
            position_alignment_offset_override_m=global_position_offset_m,
            **per_ap_kwargs,
        )

    first_link = ap_links[ap_values[0].ap_id]
    reference_time = first_link["time_s"]
    reference_centre = first_link["train_center_position_m"]
    for ap in ap_values[1:]:
        link = ap_links[ap.ap_id]
        if not np.array_equal(link["time_s"], reference_time):
            raise RuntimeError("AP links did not share the same reporting clock")
        if not np.array_equal(
            link["train_center_position_m"], reference_centre
        ):
            raise RuntimeError("AP links did not share the same train trajectory")

    ap_ids = tuple(ap.ap_id for ap in ap_values)
    front = _obm_candidate_view("front_obm", ap_ids, ap_links)
    rear = _obm_candidate_view("rear_obm", ap_ids, ap_links)
    return {
        "candidate_schema_version": MULTI_AP_CANDIDATE_SCHEMA_VERSION,
        "ap_ids": ap_ids,
        "ap_definitions": [asdict(ap) for ap in ap_values],
        "time_s": reference_time.copy(),
        "train_center_position_m": reference_centre.copy(),
        "train_center_speed_mps": (
            first_link["train_center_speed_mps"].copy()
        ),
        "front_obm": front,
        "rear_obm": rear,
        "ap_links": ap_links,
        "metadata": {
            "candidate_schema_version": MULTI_AP_CANDIDATE_SCHEMA_VERSION,
            "modeling_stage": "2_multi_ap_candidate_measurements",
            "ap_count": len(ap_values),
            "ap_matrix_row_order": ap_ids,
            "train_length_m": float(train_length_m),
            "global_position_alignment_offset_m": (
                global_position_offset_m
            ),
            "position_alignment_policy": (
                "one_train_level_offset_shared_across_all_AP_links"
            ),
            "ap_channel_policy": (
                "independent_reproducible_shadow_fading_noise_streams_per_AP"
            ),
            "candidate_reporting_assumption": (
                "synchronous_reports_before_scan_scheduling_is_modeled"
            ),
            "reference_strongest_policy": (
                "diagnostic_argmax_only_not_serving_state_not_handover"
            ),
            "serving_ap_policy": "not_implemented",
            "obm_combining_policy": "none_keep_front_and_rear_independent",
            "seed": seed,
            "ap_seed_values": ap_seed_values,
            "deferred_scope": (
                "initial_association_scan_schedule_EWMA_hysteresis_TTT_"
                "handover_execution_and_upper_layer_OBM_policy"
            ),
        },
    }
