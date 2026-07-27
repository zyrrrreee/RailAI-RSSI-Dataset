"""Shared 12-trip healthy reference used by every diagnostic path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from pipeline_contract import (
    BASELINE_SCHEMA_VERSION,
    PIPELINE_SCHEMA_VERSION,
    contract_source_hashes,
    curve_hash,
    effective_generator_config,
    generator_config_hash,
    utc_now_iso,
)
from signal_generation import generate_rssi_simulation


DEFAULT_BASELINE_PATH = "models/healthy_baseline.pkl"
DEFAULT_TRACE_COUNT = 12
DEFAULT_BASELINE_SEED = 20260715


def build_healthy_baseline(
    generator_kwargs: Mapping[str, Any] | None = None,
    *,
    trace_count: int = DEFAULT_TRACE_COUNT,
    seed: int = DEFAULT_BASELINE_SEED,
) -> dict[str, Any]:
    if trace_count < 3:
        raise ValueError("健康基线至少需要3条轨迹")
    config = effective_generator_config(generator_kwargs)
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(1, 2**31 - 1)) for _ in range(int(trace_count))]
    x_reference = None
    curves = []
    for curve_seed in seeds:
        x, curve = generate_rssi_simulation(seed=curve_seed, **config)[:2]
        if x_reference is None:
            x_reference = np.asarray(x, dtype=float)
            curves.append(np.asarray(curve, dtype=float))
        else:
            curves.append(np.interp(x_reference, x, curve))
    reference_curve = np.mean(np.vstack(curves), axis=0)
    dx = float(np.mean(np.diff(x_reference))) if len(x_reference) > 1 else 1.0
    reference_hash = curve_hash(x_reference, reference_curve)
    return {
        "artifact_type": "healthy_baseline",
        "schema_version": BASELINE_SCHEMA_VERSION,
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "trace_count": int(trace_count),
        "seed": int(seed),
        "trace_seeds": seeds,
        "generator_config": config,
        "generator_config_hash": generator_config_hash(config),
        "source_hashes": contract_source_hashes(("simulator", "diagnostic")),
        "x_reference": x_reference,
        "reference_curve": reference_curve,
        "dx": dx,
        "reference_hash": reference_hash,
    }


def validate_healthy_baseline(
    artifact: Mapping[str, Any] | None,
    generator_kwargs: Mapping[str, Any] | None = None,
    *,
    trace_count: int = DEFAULT_TRACE_COUNT,
) -> list[str]:
    if not artifact:
        return ["健康基线不存在"]
    errors = []
    if artifact.get("artifact_type") != "healthy_baseline":
        errors.append("工件类型不是 healthy_baseline")
    if artifact.get("schema_version") != BASELINE_SCHEMA_VERSION:
        errors.append("健康基线模式版本不一致")
    if artifact.get("pipeline_schema_version") != PIPELINE_SCHEMA_VERSION:
        errors.append("诊断流水线版本不一致")
    if int(artifact.get("trace_count", -1)) != int(trace_count):
        errors.append(f"健康轨迹数量不是 {trace_count}")
    expected_config_hash = generator_config_hash(generator_kwargs)
    if artifact.get("generator_config_hash") != expected_config_hash:
        errors.append("仿真参数签名不一致")
    expected_sources = contract_source_hashes(("simulator", "diagnostic"))
    if artifact.get("source_hashes") != expected_sources:
        errors.append("仿真器或诊断流水线代码版本不一致")
    try:
        x = np.asarray(artifact["x_reference"], dtype=float)
        reference = np.asarray(artifact["reference_curve"], dtype=float)
        if x.shape != reference.shape or x.ndim != 1:
            errors.append("健康基线位置网格与参考曲线形状不一致")
        elif artifact.get("reference_hash") != curve_hash(x, reference):
            errors.append("健康基线哈希校验失败")
    except Exception as exc:
        errors.append(f"健康基线内容无效: {exc}")
    return errors


def save_healthy_baseline(
    artifact: Mapping[str, Any], model_path: str | Path = DEFAULT_BASELINE_PATH
) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(artifact), path)


def load_healthy_baseline(
    generator_kwargs: Mapping[str, Any] | None = None,
    model_path: str | Path = DEFAULT_BASELINE_PATH,
    *,
    trace_count: int = DEFAULT_TRACE_COUNT,
    return_errors: bool = False,
):
    path = Path(model_path)
    if not path.exists():
        return (None, ["健康基线文件不存在"]) if return_errors else None
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        errors = [f"健康基线读取失败: {exc}"]
        return (None, errors) if return_errors else None
    errors = validate_healthy_baseline(
        artifact, generator_kwargs, trace_count=trace_count
    )
    valid = None if errors else artifact
    return (valid, errors) if return_errors else valid


def ensure_healthy_baseline(
    generator_kwargs: Mapping[str, Any] | None = None,
    model_path: str | Path = DEFAULT_BASELINE_PATH,
    *,
    trace_count: int = DEFAULT_TRACE_COUNT,
    seed: int = DEFAULT_BASELINE_SEED,
):
    artifact, errors = load_healthy_baseline(
        generator_kwargs,
        model_path,
        trace_count=trace_count,
        return_errors=True,
    )
    if artifact is not None:
        return artifact, False, []
    artifact = build_healthy_baseline(
        generator_kwargs, trace_count=trace_count, seed=seed
    )
    save_healthy_baseline(artifact, model_path)
    return artifact, True, errors

