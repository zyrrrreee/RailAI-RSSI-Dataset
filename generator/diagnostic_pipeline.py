"""One physical observation and residual path shared by GUI and training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np

from baseline_artifact import validate_healthy_baseline
from feature_extraction import extract_features, features_to_array
from pipeline_contract import TRADITIONAL_FEATURE_NAMES, effective_generator_config
from signal_generation import generate_fault_rssi_pair, generate_rssi_simulation


def generate_diagnostic_observation(
    *,
    generator_kwargs: Mapping[str, Any] | None = None,
    seed: int,
    fault_type: str | Sequence[tuple[str, Mapping[str, float]]] | None = None,
    fault_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = effective_generator_config(generator_kwargs)
    simulation = generate_rssi_simulation(seed=int(seed), **config, return_metadata=True)
    x, healthy, gains, path_losses, small_fading, distances, metadata = simulation
    faulty = np.asarray(healthy, dtype=float).copy()
    fault_metadata = None
    if fault_type is not None:
        pair_x, pair_healthy, faulty, fault_metadata = generate_fault_rssi_pair(
            fault_type,
            dict(fault_kwargs or {}),
            seed=int(seed),
            return_metadata=True,
            **config,
        )
        if not np.array_equal(np.asarray(pair_x), np.asarray(x)):
            raise RuntimeError("物理故障生成器与健康仿真器的位置网格不一致")
        if not np.array_equal(np.asarray(pair_healthy), np.asarray(healthy)):
            raise RuntimeError("物理故障生成器与健康仿真器未共享同一随机信道")
    return {
        "x": np.asarray(x, dtype=float),
        "healthy": np.asarray(healthy, dtype=float),
        "faulty": np.asarray(faulty, dtype=float),
        "fault_effect": np.asarray(faulty, dtype=float) - np.asarray(healthy, dtype=float),
        "gains": np.asarray(gains, dtype=float),
        "path_losses": np.asarray(path_losses, dtype=float),
        "small_fading": np.asarray(small_fading, dtype=float),
        "distances": np.asarray(distances, dtype=float),
        "simulation_metadata": metadata,
        "fault_metadata": fault_metadata,
        "generator_config": config,
        "seed": int(seed),
        "fault_type": (
            fault_metadata.get("fault_type") if fault_metadata is not None else None
        ),
        "fault_components": (
            list(fault_metadata.get("fault_components", []))
            if fault_metadata is not None
            else []
        ),
        "fault_kwargs": dict(fault_kwargs or {}),
    }


def baseline_reference_for_x(
    baseline_artifact: Mapping[str, Any],
    x: np.ndarray,
    generator_kwargs: Mapping[str, Any] | None = None,
) -> np.ndarray:
    errors = validate_healthy_baseline(baseline_artifact, generator_kwargs)
    if errors:
        raise ValueError("健康基线不兼容: " + "; ".join(errors))
    baseline_x = np.asarray(baseline_artifact["x_reference"], dtype=float)
    reference = np.asarray(baseline_artifact["reference_curve"], dtype=float)
    x = np.asarray(x, dtype=float)
    if np.array_equal(x, baseline_x):
        return reference.copy()
    if x[0] < baseline_x[0] or x[-1] > baseline_x[-1]:
        raise ValueError("观测位置范围超出健康基线范围")
    return np.interp(x, baseline_x, reference)


def diagnostic_residual(
    baseline_artifact: Mapping[str, Any],
    x: np.ndarray,
    signal_curve: np.ndarray,
    generator_kwargs: Mapping[str, Any] | None = None,
) -> np.ndarray:
    reference = baseline_reference_for_x(baseline_artifact, x, generator_kwargs)
    return np.asarray(signal_curve, dtype=float) - reference


def traditional_feature_vector(
    baseline_artifact: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> tuple[dict[str, float], np.ndarray]:
    x = np.asarray(observation["x"], dtype=float)
    reference = baseline_reference_for_x(
        baseline_artifact, x, observation.get("generator_config")
    )
    split_position = float(
        observation.get("generator_config", {}).get("antenna_x", 0.0)
    )
    features = extract_features(
        reference,
        observation["faulty"],
        x,
        split_position=split_position,
    )
    vector = features_to_array(features)
    if vector.shape != (len(TRADITIONAL_FEATURE_NAMES),):
        raise RuntimeError(f"传统特征形状错误: {vector.shape}")
    return features, vector
