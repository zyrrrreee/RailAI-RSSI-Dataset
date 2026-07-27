"""Receiver-side RSSI integration and reporting primitives.

This module deliberately stops at the physical receiver-report boundary.
Decision-layer smoothing (for example EWMA used before handover) belongs to the
later AP-selection state machine and must not be mixed with this integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


DEFAULT_BELOW_SENSITIVITY_POLICY = "clamp_report_preserve_raw_status"
SUPPORTED_BELOW_SENSITIVITY_POLICIES = {
    DEFAULT_BELOW_SENSITIVITY_POLICY,
    "missing_report_preserve_raw_status",
    "raw_report_with_status",
}


@dataclass(frozen=True)
class ReceiverIntegrationResult:
    """Causal trailing-window integration result on a monotonic coordinate."""

    averaged_power_mW: np.ndarray
    support_width: np.ndarray
    contributing_sample_count: np.ndarray


@dataclass(frozen=True)
class ReceiverReportingResult:
    """Receiver-reported RSSI plus auditable pre-limit values and status."""

    reported_rssi_dBm: np.ndarray
    quantized_rssi_dBm: np.ndarray
    status: np.ndarray
    below_sensitivity_mask: np.ndarray
    saturation_mask: np.ndarray
    missing_mask: np.ndarray


def dbm_to_mw(rssi_dBm: np.ndarray) -> np.ndarray:
    """Convert dBm to mW without averaging in the logarithmic domain."""
    values = np.asarray(rssi_dBm, dtype=float)
    return 10.0 ** (values / 10.0)


def mw_to_dbm(power_mW: np.ndarray) -> np.ndarray:
    """Convert mW to dBm with a numerical floor only for log safety."""
    values = np.asarray(power_mW, dtype=float)
    return 10.0 * np.log10(np.maximum(values, 1e-30))


def causal_trailing_linear_average(
    power_mW: np.ndarray,
    coordinate: np.ndarray,
    window_width: float,
) -> ReceiverIntegrationResult:
    """Average linear power over ``[coordinate[i] - window_width, coordinate[i]]``.

    The signal between known points is represented by linear interpolation and
    integrated with the trapezoidal rule.  This makes the result time-weighted
    on irregular time grids and, critically, uses no future samples.  Before a
    full window exists, all available history is used.  At the very first point
    the instantaneous value is returned.
    """
    values = np.asarray(power_mW, dtype=float)
    axis = np.asarray(coordinate, dtype=float)
    if values.ndim != 1 or axis.ndim != 1 or values.shape != axis.shape:
        raise ValueError("power_mW and coordinate must be matching one-dimensional arrays")
    if len(values) == 0:
        return ReceiverIntegrationResult(
            averaged_power_mW=values.copy(),
            support_width=np.asarray([], dtype=float),
            contributing_sample_count=np.asarray([], dtype=int),
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("power_mW must contain finite non-negative values")
    if not np.all(np.isfinite(axis)) or (len(axis) > 1 and np.any(np.diff(axis) <= 0.0)):
        raise ValueError("coordinate must be finite and strictly increasing")
    width = float(window_width)
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("window_width must be finite and non-negative")
    if width == 0.0 or len(values) == 1:
        return ReceiverIntegrationResult(
            averaged_power_mW=values.copy(),
            support_width=np.zeros(len(values), dtype=float),
            contributing_sample_count=np.ones(len(values), dtype=int),
        )

    interval_integrals = 0.5 * (values[:-1] + values[1:]) * np.diff(axis)
    cumulative_integral = np.concatenate(
        (np.asarray([0.0]), np.cumsum(interval_integrals, dtype=float))
    )
    averaged = np.empty_like(values)
    support = np.empty_like(values)
    sample_count = np.empty(len(values), dtype=int)

    averaged[0] = values[0]
    support[0] = 0.0
    sample_count[0] = 1
    for idx in range(1, len(values)):
        start = max(float(axis[0]), float(axis[idx]) - width)
        duration = float(axis[idx]) - start
        if duration <= 0.0:
            averaged[idx] = values[idx]
            support[idx] = 0.0
            sample_count[idx] = 1
            continue

        left_idx = int(np.searchsorted(axis, start, side="right") - 1)
        left_idx = min(max(left_idx, 0), idx)
        if left_idx == idx or start == float(axis[left_idx]):
            integral_at_start = float(cumulative_integral[left_idx])
        else:
            span = float(axis[left_idx + 1] - axis[left_idx])
            fraction = (start - float(axis[left_idx])) / span
            value_at_start = float(
                values[left_idx]
                + fraction * (values[left_idx + 1] - values[left_idx])
            )
            integral_at_start = float(cumulative_integral[left_idx]) + 0.5 * (
                float(values[left_idx]) + value_at_start
            ) * (start - float(axis[left_idx]))

        integrated_power = float(cumulative_integral[idx]) - integral_at_start
        averaged[idx] = integrated_power / duration
        support[idx] = duration
        sample_count[idx] = idx - left_idx + 1

    return ReceiverIntegrationResult(
        averaged_power_mW=averaged,
        support_width=support,
        contributing_sample_count=sample_count,
    )


def integrate_rssi_dbm_causally(
    instantaneous_rssi_dBm: np.ndarray,
    coordinate: np.ndarray,
    window_width: float,
) -> ReceiverIntegrationResult:
    """Causally integrate a full candidate link in linear power."""
    result = causal_trailing_linear_average(
        dbm_to_mw(instantaneous_rssi_dBm),
        coordinate,
        window_width,
    )
    return ReceiverIntegrationResult(
        averaged_power_mW=result.averaged_power_mW,
        support_width=result.support_width,
        contributing_sample_count=result.contributing_sample_count,
    )


def apply_receiver_reporting(
    raw_rssi_dBm: np.ndarray,
    receiver_sensitivity_dBm: Optional[float],
    receiver_saturation_dBm: Optional[float],
    rssi_quantization_dB: float,
    below_sensitivity_policy: str = DEFAULT_BELOW_SENSITIVITY_POLICY,
) -> ReceiverReportingResult:
    """Apply quantization, sensitivity policy, and saturation in that order."""
    raw = np.asarray(raw_rssi_dBm, dtype=float)
    if raw.ndim != 1:
        raise ValueError("raw_rssi_dBm must be one-dimensional")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw_rssi_dBm must be finite before receiver reporting")
    if (
        receiver_sensitivity_dBm is not None
        and receiver_saturation_dBm is not None
        and float(receiver_saturation_dBm) <= float(receiver_sensitivity_dBm)
    ):
        raise ValueError("receiver_saturation_dBm must exceed receiver_sensitivity_dBm")
    policy = str(below_sensitivity_policy)
    if policy not in SUPPORTED_BELOW_SENSITIVITY_POLICIES:
        raise ValueError(
            "below_sensitivity_policy must be one of "
            f"{sorted(SUPPORTED_BELOW_SENSITIVITY_POLICIES)}"
        )
    quantization = float(rssi_quantization_dB)
    if not np.isfinite(quantization) or quantization < 0.0:
        raise ValueError("rssi_quantization_dB must be finite and non-negative")

    quantized = raw.copy()
    if quantization > 0.0:
        quantized = np.round(quantized / quantization) * quantization
    reported = quantized.copy()
    below = np.zeros(len(raw), dtype=bool)
    saturated = np.zeros(len(raw), dtype=bool)
    status = np.full(len(raw), "valid", dtype="<U48")

    if receiver_sensitivity_dBm is not None:
        sensitivity = float(receiver_sensitivity_dBm)
        below = quantized < sensitivity
        if policy == DEFAULT_BELOW_SENSITIVITY_POLICY:
            reported[below] = sensitivity
            status[below] = "below_sensitivity_clamped_raw_preserved"
        elif policy == "missing_report_preserve_raw_status":
            reported[below] = np.nan
            status[below] = "below_sensitivity_missing_raw_preserved"
        else:
            status[below] = "below_sensitivity_raw_reported"

    if receiver_saturation_dBm is not None:
        saturation = float(receiver_saturation_dBm)
        saturated = quantized > saturation
        reported[saturated] = saturation
        status[saturated] = "above_saturation_clamped_raw_preserved"

    missing = ~np.isfinite(reported)
    return ReceiverReportingResult(
        reported_rssi_dBm=reported,
        quantized_rssi_dBm=quantized,
        status=status,
        below_sensitivity_mask=below,
        saturation_mask=saturated,
        missing_mask=missing,
    )
