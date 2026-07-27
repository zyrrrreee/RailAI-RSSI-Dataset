"""Stateful AP association and handover for the stage-2 dual-OBM scene.

The input to this module is the receiver-reported RSSI candidate matrix from
``multi_ap.py``.  It deliberately does not compare ideal path loss directly.

The default decision chain is:

    receiver report -> time-aware EWMA -> hysteresis H -> TTT
    -> execution delay -> serving AP update

Front and rear OBMs run independent state machines.  The engineering defaults
in :class:`HandoverConfig` are experiment starting points, not target-line
equipment values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from multi_ap import (
    MULTI_AP_CANDIDATE_SCHEMA_VERSION,
    generate_multi_ap_dual_obm_candidates,
)


HANDOVER_SCHEMA_VERSION = "paper-stage2-stateful-handover-v2"
_DECISION_FILTERS = {"ewma", "moving_average", "none"}


@dataclass(frozen=True)
class HandoverConfig:
    """Decision-layer parameters, all expressed in physical time or dB."""

    decision_filter: str = "ewma"
    filter_tau_s: float = 0.20
    moving_average_window_s: float = 0.40
    filter_reset_after_s: float = 1.00
    hysteresis_db: float = 3.0
    time_to_trigger_s: float = 0.30
    handover_execution_s: float = 0.10
    emergency_on_serving_unavailable: bool = True
    ping_pong_window_s: float = 2.0


def normalise_handover_config(
    config: HandoverConfig | Mapping[str, Any] | None,
) -> HandoverConfig:
    """Validate configuration and return an immutable normalized copy."""
    if config is None:
        result = HandoverConfig()
    elif isinstance(config, HandoverConfig):
        result = config
    elif isinstance(config, Mapping):
        result = HandoverConfig(**dict(config))
    else:
        raise TypeError("config must be HandoverConfig, mapping, or None")

    decision_filter = str(result.decision_filter).strip().lower()
    if decision_filter not in _DECISION_FILTERS:
        raise ValueError(
            f"decision_filter must be one of {sorted(_DECISION_FILTERS)}"
        )
    numeric = {
        "filter_tau_s": result.filter_tau_s,
        "moving_average_window_s": result.moving_average_window_s,
        "filter_reset_after_s": result.filter_reset_after_s,
        "hysteresis_db": result.hysteresis_db,
        "time_to_trigger_s": result.time_to_trigger_s,
        "handover_execution_s": result.handover_execution_s,
        "ping_pong_window_s": result.ping_pong_window_s,
    }
    if not all(np.isfinite(float(value)) for value in numeric.values()):
        raise ValueError("handover parameters must be finite")
    if float(result.filter_tau_s) <= 0.0:
        raise ValueError("filter_tau_s must be positive")
    if float(result.moving_average_window_s) <= 0.0:
        raise ValueError("moving_average_window_s must be positive")
    if float(result.filter_reset_after_s) < 0.0:
        raise ValueError("filter_reset_after_s cannot be negative")
    if float(result.hysteresis_db) < 0.0:
        raise ValueError("hysteresis_db cannot be negative")
    if float(result.time_to_trigger_s) < 0.0:
        raise ValueError("time_to_trigger_s cannot be negative")
    if float(result.handover_execution_s) < 0.0:
        raise ValueError("handover_execution_s cannot be negative")
    if float(result.ping_pong_window_s) < 0.0:
        raise ValueError("ping_pong_window_s cannot be negative")
    return HandoverConfig(
        decision_filter=decision_filter,
        filter_tau_s=float(result.filter_tau_s),
        moving_average_window_s=float(result.moving_average_window_s),
        filter_reset_after_s=float(result.filter_reset_after_s),
        hysteresis_db=float(result.hysteresis_db),
        time_to_trigger_s=float(result.time_to_trigger_s),
        handover_execution_s=float(result.handover_execution_s),
        emergency_on_serving_unavailable=bool(
            result.emergency_on_serving_unavailable
        ),
        ping_pong_window_s=float(result.ping_pong_window_s),
    )


def _validate_candidate_arrays(
    time_s: np.ndarray,
    position_m: np.ndarray,
    report_matrix_dBm: np.ndarray,
    available_matrix: np.ndarray,
    ap_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    time = np.asarray(time_s, dtype=float)
    position = np.asarray(position_m, dtype=float)
    reports = np.asarray(report_matrix_dBm, dtype=float)
    available = np.asarray(available_matrix, dtype=bool)
    identifiers = tuple(str(value) for value in ap_ids)
    if time.ndim != 1 or len(time) == 0:
        raise ValueError("time_s must be a non-empty one-dimensional array")
    if position.shape != time.shape:
        raise ValueError("position_m must match time_s")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(position)):
        raise ValueError("time and position must be finite")
    if len(time) > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    if reports.ndim != 2 or reports.shape[1] != len(time):
        raise ValueError("candidate report matrix must have shape (AP, time)")
    if available.shape != reports.shape:
        raise ValueError("candidate availability matrix must match reports")
    if reports.shape[0] != len(identifiers) or len(identifiers) < 2:
        raise ValueError("AP identifiers must match at least two matrix rows")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("AP identifiers must be unique")
    return time, position, reports, available, identifiers


def time_aware_ewma(
    time_s: np.ndarray,
    values_dBm: np.ndarray,
    available: np.ndarray | None = None,
    *,
    tau_s: float,
    reset_after_s: float,
) -> np.ndarray:
    """Filter each AP report stream using actual elapsed time.

    Missing or unavailable reports produce NaN decision values and are never
    treated as 0 dBm.  A short gap preserves the internal state.  If the gap
    exceeds ``reset_after_s``, the next valid report reinitializes that AP's
    filter state instead of blending with stale history.
    """
    time = np.asarray(time_s, dtype=float)
    values = np.asarray(values_dBm, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
        squeeze = True
    elif values.ndim == 2:
        squeeze = False
    else:
        raise ValueError("values_dBm must be one- or two-dimensional")
    if time.ndim != 1 or values.shape[1] != len(time):
        raise ValueError("values_dBm time dimension must match time_s")
    if len(time) > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    if not np.isfinite(float(tau_s)) or float(tau_s) <= 0.0:
        raise ValueError("tau_s must be positive and finite")
    if (
        not np.isfinite(float(reset_after_s))
        or float(reset_after_s) < 0.0
    ):
        raise ValueError("reset_after_s must be non-negative and finite")
    if available is None:
        mask = np.isfinite(values)
    else:
        mask = np.asarray(available, dtype=bool)
        if mask.ndim == 1 and values.shape[0] == 1:
            mask = mask[np.newaxis, :]
        if mask.shape != values.shape:
            raise ValueError("available mask must match values_dBm")
        mask = mask & np.isfinite(values)

    output = np.full(values.shape, np.nan, dtype=float)
    for row in range(values.shape[0]):
        state = float("nan")
        last_valid_time = float("nan")
        for column, timestamp in enumerate(time):
            if not mask[row, column]:
                continue
            value = float(values[row, column])
            elapsed = float(timestamp - last_valid_time)
            if (
                not np.isfinite(state)
                or not np.isfinite(last_valid_time)
                or elapsed > float(reset_after_s)
            ):
                state = value
            else:
                alpha = 1.0 - np.exp(-elapsed / float(tau_s))
                state = (1.0 - alpha) * state + alpha * value
            output[row, column] = state
            last_valid_time = float(timestamp)
    return output[0] if squeeze else output


def causal_moving_average(
    time_s: np.ndarray,
    values_dBm: np.ndarray,
    available: np.ndarray | None = None,
    *,
    window_s: float,
    reset_after_s: float,
) -> np.ndarray:
    """Return a causal trailing moving average in the decision dBm domain.

    At time ``t[k]`` the result only uses valid reports in
    ``[t[k] - window_s, t[k]]``.  A missing current report remains missing and
    is never replaced by a stale average.  After a gap longer than
    ``reset_after_s`` the next valid report starts a fresh window.

    This is a decision-layer smoother.  It must not be confused with the
    receiver's physical linear-power integration in ``receiver_measurement``.
    """
    time = np.asarray(time_s, dtype=float)
    values = np.asarray(values_dBm, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False
    if values.ndim != 2 or values.shape[1] != len(time):
        raise ValueError("values_dBm must have shape (stream, time)")
    if len(time) == 0 or not np.all(np.isfinite(time)):
        raise ValueError("time_s must be non-empty and finite")
    if len(time) > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    if not np.isfinite(float(window_s)) or float(window_s) <= 0.0:
        raise ValueError("window_s must be positive and finite")
    if not np.isfinite(float(reset_after_s)) or float(reset_after_s) < 0.0:
        raise ValueError("reset_after_s must be non-negative and finite")
    if available is None:
        mask = np.isfinite(values)
    else:
        mask = np.asarray(available, dtype=bool)
        if mask.ndim == 1 and values.shape[0] == 1:
            mask = mask[np.newaxis, :]
        if mask.shape != values.shape:
            raise ValueError("available mask must match values_dBm")
        mask = mask & np.isfinite(values)

    output = np.full(values.shape, np.nan, dtype=float)
    for row in range(values.shape[0]):
        valid_indices: list[int] = []
        last_valid_time = float("nan")
        for column, timestamp in enumerate(time):
            if not mask[row, column]:
                continue
            if (
                np.isfinite(last_valid_time)
                and float(timestamp - last_valid_time) > float(reset_after_s)
            ):
                valid_indices.clear()
            lower_bound = float(timestamp) - float(window_s)
            valid_indices = [
                index
                for index in valid_indices
                if float(time[index]) >= lower_bound - 1e-12
            ]
            valid_indices.append(column)
            output[row, column] = float(
                np.mean(values[row, valid_indices], dtype=float)
            )
            last_valid_time = float(timestamp)
    return output[0] if squeeze else output


def _strongest_available_index(
    values_dBm: np.ndarray,
    available: np.ndarray,
    *,
    exclude_index: int = -1,
) -> int:
    valid = np.asarray(available, dtype=bool) & np.isfinite(values_dBm)
    if 0 <= int(exclude_index) < len(valid):
        valid[int(exclude_index)] = False
    if not np.any(valid):
        return -1
    safe = np.where(valid, values_dBm, -np.inf)
    return int(np.argmax(safe))


def _ping_pong_count(
    completion_events: list[dict[str, Any]],
    *,
    window_s: float,
) -> int:
    count = 0
    for previous, current in zip(
        completion_events[:-1], completion_events[1:]
    ):
        reversed_pair = (
            previous["from_ap_id"] == current["to_ap_id"]
            and previous["to_ap_id"] == current["from_ap_id"]
        )
        separation_s = (
            float(current["completion_time_s"])
            - float(previous["completion_time_s"])
        )
        if reversed_pair and separation_s <= float(window_s) + 1e-12:
            count += 1
    return count


def run_obm_handover(
    *,
    time_s: np.ndarray,
    position_m: np.ndarray,
    report_matrix_dBm: np.ndarray,
    available_matrix: np.ndarray,
    ap_ids: Sequence[str],
    obm_id: str,
    config: HandoverConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one OBM's state machine over a candidate report matrix."""
    cfg = normalise_handover_config(config)
    time, position, reports, available, identifiers = (
        _validate_candidate_arrays(
            time_s,
            position_m,
            report_matrix_dBm,
            available_matrix,
            ap_ids,
        )
    )
    usable = available & np.isfinite(reports)
    if cfg.decision_filter == "ewma":
        decisions = time_aware_ewma(
            time,
            reports,
            usable,
            tau_s=cfg.filter_tau_s,
            reset_after_s=cfg.filter_reset_after_s,
        )
    elif cfg.decision_filter == "moving_average":
        decisions = causal_moving_average(
            time,
            reports,
            usable,
            window_s=cfg.moving_average_window_s,
            reset_after_s=cfg.filter_reset_after_s,
        )
    else:
        decisions = np.where(usable, reports, np.nan)

    sample_count = len(time)
    id_width = max(1, max(len(value) for value in identifiers))
    state_width = max(
        len(value)
        for value in (
            "unassociated",
            "connected",
            "ttt_pending",
            "executing",
            "serving_outage",
        )
    )
    serving_indices = np.full(sample_count, -1, dtype=np.int16)
    serving_ids = np.full(sample_count, "", dtype=f"<U{id_width}")
    serving_report = np.full(sample_count, np.nan, dtype=float)
    serving_decision = np.full(sample_count, np.nan, dtype=float)
    states = np.full(sample_count, "unassociated", dtype=f"<U{state_width}")
    pending_indices = np.full(sample_count, -1, dtype=np.int16)
    pending_elapsed_s = np.zeros(sample_count, dtype=float)
    execution_target_indices = np.full(sample_count, -1, dtype=np.int16)
    trigger_mask = np.zeros(sample_count, dtype=bool)
    completion_mask = np.zeros(sample_count, dtype=bool)

    events: list[dict[str, Any]] = []
    serving_index = -1
    pending_index = -1
    pending_start_time_s = float("nan")
    execution_target_index = -1
    execution_start_time_s = float("nan")
    execution_due_time_s = float("nan")
    execution_condition_start_s = float("nan")
    execution_reason = ""

    def complete_handover(
        sample_index: int,
        target_index: int,
        reason: str,
        condition_start_s: float,
        trigger_time_s: float,
    ) -> bool:
        nonlocal serving_index
        old_index = serving_index
        if not usable[target_index, sample_index]:
            events.append(
                {
                    "event_type": "handover_failed",
                    "sample_index": int(sample_index),
                    "time_s": float(time[sample_index]),
                    "obm_position_m": float(position[sample_index]),
                    "from_ap_id": (
                        identifiers[old_index] if old_index >= 0 else ""
                    ),
                    "to_ap_id": identifiers[target_index],
                    "reason": "target_unavailable_at_execution_completion",
                    "original_trigger_reason": reason,
                }
            )
            return False
        serving_index = int(target_index)
        events.append(
            {
                "event_type": "handover_complete",
                "sample_index": int(sample_index),
                "condition_start_time_s": float(condition_start_s),
                "trigger_time_s": float(trigger_time_s),
                "completion_time_s": float(time[sample_index]),
                "obm_position_m": float(position[sample_index]),
                "from_ap_id": (
                    identifiers[old_index] if old_index >= 0 else ""
                ),
                "to_ap_id": identifiers[target_index],
                "reason": reason,
                "execution_elapsed_s": float(
                    time[sample_index] - trigger_time_s
                ),
            }
        )
        completion_mask[sample_index] = True
        return True

    def trigger_handover(
        sample_index: int,
        target_index: int,
        reason: str,
        condition_start_s: float,
    ) -> bool:
        nonlocal execution_target_index
        nonlocal execution_start_time_s
        nonlocal execution_due_time_s
        nonlocal execution_condition_start_s
        nonlocal execution_reason
        trigger_time = float(time[sample_index])
        trigger_mask[sample_index] = True
        events.append(
            {
                "event_type": "handover_trigger",
                "sample_index": int(sample_index),
                "condition_start_time_s": float(condition_start_s),
                "trigger_time_s": trigger_time,
                "expected_completion_time_s": (
                    trigger_time + cfg.handover_execution_s
                ),
                "obm_position_m": float(position[sample_index]),
                "from_ap_id": (
                    identifiers[serving_index] if serving_index >= 0 else ""
                ),
                "to_ap_id": identifiers[target_index],
                "reason": reason,
                "serving_decision_rssi_dBm": (
                    float(decisions[serving_index, sample_index])
                    if serving_index >= 0
                    and np.isfinite(decisions[serving_index, sample_index])
                    else None
                ),
                "target_decision_rssi_dBm": float(
                    decisions[target_index, sample_index]
                ),
                "decision_margin_dB": (
                    float(
                        decisions[target_index, sample_index]
                        - decisions[serving_index, sample_index]
                    )
                    if serving_index >= 0
                    and np.isfinite(decisions[serving_index, sample_index])
                    else None
                ),
            }
        )
        if cfg.handover_execution_s <= 1e-12:
            return complete_handover(
                sample_index,
                target_index,
                reason,
                condition_start_s,
                trigger_time,
            )
        execution_target_index = int(target_index)
        execution_start_time_s = trigger_time
        execution_due_time_s = trigger_time + cfg.handover_execution_s
        execution_condition_start_s = float(condition_start_s)
        execution_reason = str(reason)
        return False

    for column, timestamp in enumerate(time):
        completed_this_sample = False

        if execution_target_index >= 0:
            if float(timestamp) + 1e-12 >= execution_due_time_s:
                completed_this_sample = complete_handover(
                    column,
                    execution_target_index,
                    execution_reason,
                    execution_condition_start_s,
                    execution_start_time_s,
                )
                execution_target_index = -1
                execution_start_time_s = float("nan")
                execution_due_time_s = float("nan")
                execution_condition_start_s = float("nan")
                execution_reason = ""

        if serving_index < 0 and execution_target_index < 0:
            initial_index = _strongest_available_index(
                decisions[:, column], usable[:, column]
            )
            if initial_index >= 0:
                serving_index = initial_index
                events.append(
                    {
                        "event_type": "initial_association",
                        "sample_index": int(column),
                        "time_s": float(timestamp),
                        "obm_position_m": float(position[column]),
                        "from_ap_id": "",
                        "to_ap_id": identifiers[initial_index],
                        "reason": "strongest_available_candidate",
                        "decision_rssi_dBm": float(
                            decisions[initial_index, column]
                        ),
                    }
                )
                completed_this_sample = True

        if (
            serving_index >= 0
            and execution_target_index < 0
            and not completed_this_sample
        ):
            serving_usable = bool(usable[serving_index, column]) and bool(
                np.isfinite(decisions[serving_index, column])
            )
            if not serving_usable:
                pending_index = -1
                pending_start_time_s = float("nan")
                if cfg.emergency_on_serving_unavailable:
                    emergency_index = _strongest_available_index(
                        decisions[:, column],
                        usable[:, column],
                        exclude_index=serving_index,
                    )
                    if emergency_index >= 0:
                        trigger_handover(
                            column,
                            emergency_index,
                            "serving_unavailable_emergency",
                            float(timestamp),
                        )
            else:
                best_index = _strongest_available_index(
                    decisions[:, column],
                    usable[:, column],
                    exclude_index=serving_index,
                )
                qualifies = (
                    best_index >= 0
                    and float(decisions[best_index, column])
                    > float(decisions[serving_index, column])
                    + cfg.hysteresis_db
                )
                if not qualifies:
                    pending_index = -1
                    pending_start_time_s = float("nan")
                else:
                    if pending_index != best_index:
                        pending_index = int(best_index)
                        pending_start_time_s = float(timestamp)
                    elapsed_s = float(timestamp - pending_start_time_s)
                    if elapsed_s + 1e-12 >= cfg.time_to_trigger_s:
                        trigger_handover(
                            column,
                            pending_index,
                            "candidate_exceeded_serving_by_hysteresis_for_ttt",
                            pending_start_time_s,
                        )
                        pending_index = -1
                        pending_start_time_s = float("nan")

        if execution_target_index >= 0:
            state = "executing"
        elif serving_index < 0:
            state = "unassociated"
        elif not usable[serving_index, column]:
            state = "serving_outage"
        elif pending_index >= 0:
            state = "ttt_pending"
        else:
            state = "connected"

        serving_indices[column] = serving_index
        if serving_index >= 0:
            serving_ids[column] = identifiers[serving_index]
            serving_report[column] = reports[serving_index, column]
            serving_decision[column] = decisions[serving_index, column]
        states[column] = state
        pending_indices[column] = pending_index
        if pending_index >= 0:
            pending_elapsed_s[column] = float(
                timestamp - pending_start_time_s
            )
        execution_target_indices[column] = execution_target_index

    completion_events = [
        event for event in events if event["event_type"] == "handover_complete"
    ]
    initial_events = [
        event
        for event in events
        if event["event_type"] == "initial_association"
    ]
    failed_events = [
        event for event in events if event["event_type"] == "handover_failed"
    ]
    return {
        "handover_schema_version": HANDOVER_SCHEMA_VERSION,
        "obm_id": str(obm_id),
        "ap_ids": identifiers,
        "time_s": time.copy(),
        "position_m": position.copy(),
        "candidate_reported_rssi_matrix_dBm": reports.copy(),
        "candidate_available_matrix": available.copy(),
        "decision_rssi_matrix_dBm": decisions,
        "serving_ap_index": serving_indices,
        "serving_ap_id": serving_ids,
        "serving_reported_rssi_dBm": serving_report,
        "serving_decision_rssi_dBm": serving_decision,
        "state": states,
        "pending_target_ap_index": pending_indices,
        "pending_elapsed_s": pending_elapsed_s,
        "execution_target_ap_index": execution_target_indices,
        "handover_trigger_mask": trigger_mask,
        "handover_completion_mask": completion_mask,
        "events": events,
        "summary": {
            "initial_association_count": len(initial_events),
            "handover_trigger_count": int(np.count_nonzero(trigger_mask)),
            "handover_completion_count": len(completion_events),
            "handover_failure_count": len(failed_events),
            "ping_pong_count": _ping_pong_count(
                completion_events,
                window_s=cfg.ping_pong_window_s,
            ),
            "unassociated_sample_count": int(
                np.count_nonzero(serving_indices < 0)
            ),
            "serving_outage_sample_count": int(
                np.count_nonzero(states == "serving_outage")
            ),
        },
        "metadata": {
            "handover_schema_version": HANDOVER_SCHEMA_VERSION,
            "decision_source": (
                "receiver_reported_RSSI_from_multi_ap_candidate_layer"
            ),
            "decision_filter": cfg.decision_filter,
            "filter_domain": (
                "reported_dBm_decision_layer_not_physical_power_integration"
            ),
            "ewma_time_rule": (
                "alpha_k=1-exp(-(t_k-t_last_valid)/filter_tau_s)"
            ),
            "moving_average_rule": (
                "causal_mean_of_valid_reports_in_[t-window_s,t]"
            ),
            "missing_report_rule": (
                "NaN_or_unavailable_is_not_zero_and_is_not_a_candidate"
            ),
            "filter_reappearance_rule": (
                "reset_after_configured_gap_otherwise_continue_with_elapsed_time"
            ),
            "candidate_tie_rule": "stable_AP_matrix_row_order",
            "initial_association_rule": (
                "strongest_available_decision_value_not_a_handover"
            ),
            "normal_trigger_rule": (
                "candidate_gt_serving_plus_H_continuously_for_TTT"
            ),
            "emergency_rule": (
                "immediate_trigger_to_strongest_available_candidate_when_"
                "serving_unavailable"
                if cfg.emergency_on_serving_unavailable
                else "disabled"
            ),
            "execution_rule": (
                "serving_updates_after_execution_delay_if_target_available"
            ),
            "parameter_status": (
                "engineering_experiment_defaults_not_target_line_values"
            ),
            "config": asdict(cfg),
        },
    }


def run_dual_obm_handover(
    candidate_result: Mapping[str, Any],
    *,
    config: HandoverConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run independent serving-AP state machines for front and rear OBMs."""
    if (
        candidate_result.get("candidate_schema_version")
        != MULTI_AP_CANDIDATE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported or missing multi-AP candidate schema")
    ap_ids = tuple(candidate_result["ap_ids"])
    outputs: dict[str, dict[str, Any]] = {}
    for role in ("front_obm", "rear_obm"):
        view = candidate_result[role]
        outputs[role] = run_obm_handover(
            time_s=view["t_s"],
            position_m=view["x_m"],
            report_matrix_dBm=view[
                "candidate_reported_rssi_matrix_dBm"
            ],
            available_matrix=view["candidate_available_matrix"],
            ap_ids=ap_ids,
            obm_id=view["obm_id"],
            config=config,
        )
    cfg = normalise_handover_config(config)
    return {
        "handover_schema_version": HANDOVER_SCHEMA_VERSION,
        "candidate_schema_version": candidate_result[
            "candidate_schema_version"
        ],
        "ap_ids": ap_ids,
        "ap_definitions": list(candidate_result["ap_definitions"]),
        "time_s": np.asarray(candidate_result["time_s"], dtype=float).copy(),
        "train_center_position_m": np.asarray(
            candidate_result["train_center_position_m"], dtype=float
        ).copy(),
        "train_center_speed_mps": np.asarray(
            candidate_result["train_center_speed_mps"], dtype=float
        ).copy(),
        "front_obm": outputs["front_obm"],
        "rear_obm": outputs["rear_obm"],
        "candidate_measurements": candidate_result,
        "metadata": {
            "handover_schema_version": HANDOVER_SCHEMA_VERSION,
            "modeling_stage": "2_stateful_dual_obm_multi_ap_handover",
            "obm_state_policy": (
                "front_and_rear_run_independent_state_machines"
            ),
            "obm_combining_policy": "none",
            "serving_ap_policy": (
                "stateful_initial_association_EWMA_H_TTT_execution_delay"
            ),
            "config": asdict(cfg),
            "deferred_scope": (
                "scan_scheduling_A_to_D_full_comparison_upper_layer_OBM_"
                "policy_GUI_and_diagnostic_model_integration"
            ),
        },
    }


def generate_dual_obm_handover(
    *,
    config: HandoverConfig | Mapping[str, Any] | None = None,
    **candidate_kwargs: Any,
) -> dict[str, Any]:
    """Generate multi-AP measurements, then run both OBM state machines."""
    candidates = generate_multi_ap_dual_obm_candidates(**candidate_kwargs)
    return run_dual_obm_handover(candidates, config=config)
