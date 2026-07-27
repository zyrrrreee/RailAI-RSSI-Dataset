"""Deterministic train motion and OBM reporting grids.

The channel simulator needs two time scales: a relatively fine internal update
grid and the (usually slower) RSSI reporting period exposed by the onboard
measurement unit.  This module deliberately contains no radio-channel or
random-number logic, so the kinematics are reproducible and independently
testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


SpeedProfile = Sequence[tuple[float, float]] | Mapping[float, float]


@dataclass(frozen=True)
class Trajectory:
    """Internal motion grid plus the subset of times reported by the OBM."""

    t_s: np.ndarray
    x_m: np.ndarray
    v_mps: np.ndarray
    report_mask: np.ndarray
    metadata: dict[str, Any]

    @property
    def report_t_s(self) -> np.ndarray:
        return self.t_s[self.report_mask]

    @property
    def report_x_m(self) -> np.ndarray:
        return self.x_m[self.report_mask]

    @property
    def report_v_mps(self) -> np.ndarray:
        return self.v_mps[self.report_mask]


def _normalise_speed_profile(
    speed_mps: Optional[float],
    speed_profile: Optional[SpeedProfile],
) -> tuple[np.ndarray, np.ndarray, str]:
    if speed_profile is None:
        if speed_mps is None or float(speed_mps) <= 0.0:
            raise ValueError("speed_mps must be greater than zero for a moving trajectory")
        return (
            np.asarray([0.0], dtype=float),
            np.asarray([float(speed_mps)], dtype=float),
            "constant_speed",
        )

    items = list(speed_profile.items()) if isinstance(speed_profile, Mapping) else list(speed_profile)
    if not items:
        raise ValueError("speed_profile cannot be empty")
    times = np.asarray([float(item[0]) for item in items], dtype=float)
    speeds = np.asarray([float(item[1]) for item in items], dtype=float)
    order = np.argsort(times)
    times = times[order]
    speeds = speeds[order]
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(speeds)):
        raise ValueError("speed_profile values must be finite")
    if abs(float(times[0])) > 1e-12:
        raise ValueError("speed_profile must start at t=0 s")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("speed_profile times must be strictly increasing")
    if np.any(speeds < 0.0):
        raise ValueError("speed_profile contains a negative speed magnitude")
    if float(speeds[-1]) <= 0.0:
        raise ValueError("the last speed_profile value must be positive to reach the endpoint")
    return times, speeds, "piecewise_linear_speed_profile"


def _profile_speed_and_distance(
    query_t_s: np.ndarray,
    knot_t_s: np.ndarray,
    knot_speed_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a piecewise-linear speed profile and its analytic integral."""
    query = np.asarray(query_t_s, dtype=float)
    speed = np.empty_like(query)
    distance = np.empty_like(query)

    cumulative = np.zeros(len(knot_t_s), dtype=float)
    if len(knot_t_s) > 1:
        cumulative[1:] = np.cumsum(
            0.5
            * (knot_speed_mps[:-1] + knot_speed_mps[1:])
            * np.diff(knot_t_s)
        )

    for index, current_t in np.ndenumerate(query):
        t_value = max(float(current_t), 0.0)
        interval = int(np.searchsorted(knot_t_s, t_value, side="right") - 1)
        interval = max(interval, 0)
        if interval >= len(knot_t_s) - 1:
            speed[index] = knot_speed_mps[-1]
            distance[index] = cumulative[-1] + knot_speed_mps[-1] * (
                t_value - knot_t_s[-1]
            )
            continue
        elapsed = t_value - knot_t_s[interval]
        interval_duration = knot_t_s[interval + 1] - knot_t_s[interval]
        slope = (
            knot_speed_mps[interval + 1] - knot_speed_mps[interval]
        ) / interval_duration
        speed[index] = knot_speed_mps[interval] + slope * elapsed
        distance[index] = (
            cumulative[interval]
            + knot_speed_mps[interval] * elapsed
            + 0.5 * slope * elapsed**2
        )
    return speed, distance


def _travel_time_s(
    distance_m: float,
    knot_t_s: np.ndarray,
    knot_speed_mps: np.ndarray,
) -> float:
    """Find the first time at which the integrated profile reaches distance_m."""
    low = 0.0
    high = max(float(knot_t_s[-1]), 1.0)
    while True:
        _, travelled = _profile_speed_and_distance(
            np.asarray([high]), knot_t_s, knot_speed_mps
        )
        if float(travelled[0]) >= float(distance_m):
            break
        high *= 2.0
        if high > 1.0e7:
            raise ValueError("speed profile cannot reach the requested endpoint")
    for _ in range(80):
        middle = 0.5 * (low + high)
        _, travelled = _profile_speed_and_distance(
            np.asarray([middle]), knot_t_s, knot_speed_mps
        )
        if float(travelled[0]) < float(distance_m):
            low = middle
        else:
            high = middle
    return high


def _regular_times(duration_s: float, interval_s: float) -> np.ndarray:
    values = np.arange(0.0, float(duration_s), float(interval_s), dtype=float)
    if len(values) == 0 or abs(float(values[-1]) - float(duration_s)) > 1e-10:
        values = np.append(values, float(duration_s))
    else:
        values[-1] = float(duration_s)
    return values


def generate_trajectory(
    x_start: float,
    x_end: float,
    simulation_step_s: float = 0.01,
    report_interval_s: float = 0.05,
    speed_mps: Optional[float] = 20.0,
    speed_profile: Optional[SpeedProfile] = None,
    direction: Optional[int] = None,
) -> Trajectory:
    """Generate a deterministic time-position-speed trajectory.

    ``speed_profile`` consists of ``(time_s, speed_mps)`` points.  Speed is
    linearly interpolated between points and remains at the final positive
    value afterwards.  The end position is always included as a final report,
    even when the trip duration is not an exact multiple of the report period.
    """
    if not np.isfinite(x_start) or not np.isfinite(x_end) or x_start == x_end:
        raise ValueError("x_start and x_end must be finite and different")
    if float(simulation_step_s) <= 0.0:
        raise ValueError("simulation_step_s must be greater than zero")
    if float(report_interval_s) <= 0.0:
        raise ValueError("report_interval_s must be greater than zero")
    if float(simulation_step_s) > float(report_interval_s):
        raise ValueError("simulation_step_s must not exceed report_interval_s")

    inferred_direction = 1 if float(x_end) > float(x_start) else -1
    if direction is None:
        motion_direction = inferred_direction
    else:
        motion_direction = int(direction)
        if motion_direction not in {-1, 1}:
            raise ValueError("direction must be +1 or -1")
        if motion_direction != inferred_direction:
            raise ValueError("direction must point from x_start towards x_end")

    knot_t_s, knot_speed_mps, profile_mode = _normalise_speed_profile(
        speed_mps, speed_profile
    )
    route_length_m = abs(float(x_end) - float(x_start))
    duration_s = _travel_time_s(route_length_m, knot_t_s, knot_speed_mps)
    simulation_times = _regular_times(duration_s, float(simulation_step_s))
    report_times = _regular_times(duration_s, float(report_interval_s))
    simulation_times = np.round(simulation_times, decimals=12)
    report_times = np.round(report_times, decimals=12)
    simulation_times[-1] = duration_s
    report_times[-1] = duration_s

    # Include every report instant in the internal grid.  With the default
    # 0.01/0.05 s settings these already coincide; the union also supports
    # other valid pairs without losing an exact reporting timestamp.
    combined_times = np.unique(
        np.round(np.concatenate((simulation_times, report_times)), decimals=12)
    )
    combined_times[0] = 0.0
    combined_times[-1] = duration_s
    speed_magnitude, travelled_m = _profile_speed_and_distance(
        combined_times, knot_t_s, knot_speed_mps
    )
    travelled_m = np.minimum(travelled_m, route_length_m)
    positions = float(x_start) + motion_direction * travelled_m
    positions[-1] = float(x_end)
    signed_velocity = motion_direction * speed_magnitude

    report_mask = np.zeros(len(combined_times), dtype=bool)
    report_indices = np.searchsorted(combined_times, report_times)
    if np.any(report_indices >= len(combined_times)) or not np.allclose(
        combined_times[report_indices], report_times, rtol=0.0, atol=1e-9
    ):
        raise RuntimeError("failed to align report times with the internal time grid")
    report_mask[report_indices] = True

    metadata = {
        "trajectory_schema_version": "time-position-sampling-v1",
        "profile_mode": profile_mode,
        "direction": int(motion_direction),
        "x_start_m": float(x_start),
        "x_end_m": float(x_end),
        "route_length_m": float(route_length_m),
        "trip_duration_s": float(duration_s),
        "simulation_step_s": float(simulation_step_s),
        "report_interval_s": float(report_interval_s),
        "internal_sample_count": int(len(combined_times)),
        "report_sample_count": int(np.count_nonzero(report_mask)),
        "terminal_report_forced": bool(
            abs(duration_s / float(report_interval_s) - round(duration_s / float(report_interval_s)))
            > 1e-9
        ),
        "speed_profile_time_s": knot_t_s.copy(),
        "speed_profile_mps": knot_speed_mps.copy(),
    }
    return Trajectory(
        t_s=combined_times,
        x_m=positions,
        v_mps=signed_velocity,
        report_mask=report_mask,
        metadata=metadata,
    )
