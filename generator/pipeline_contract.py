"""Versioned contracts shared by simulation, datasets, GUI and models."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PIPELINE_SCHEMA_VERSION = "rssi-diagnostic-pipeline-v5"
BASELINE_SCHEMA_VERSION = "healthy-reference-mean12-v3"
TRADITIONAL_FEATURE_SCHEMA_VERSION = "traditional-features-17-v2"
FAULT_SCHEMA_VERSION = "rail-five-physical-v3"
LABEL_SCHEMA_VERSION = "rail-five-single-label-v1"
MULTILABEL_SCHEMA_VERSION = "rail-five-multilabel-pairs-v1"
HEALTH_SCHEMA_VERSION = "health-rmse-shared-reference-v3"
SEQUENCE_SCHEMA_VERSION = "sequence-residual-binary-v2"
WINDOW_SCHEMA_VERSION = "window-residual-14-v2"
MODEL_ARTIFACT_SCHEMA_VERSION = "model-artifact-v2"
MULTILABEL_MODEL_ARTIFACT_SCHEMA_VERSION = "multilabel-model-artifact-v1"
RESEARCH_DATASET_SCHEMA_VERSION = "rail-rssi-research-dataset-v1"

TRADITIONAL_FEATURE_NAMES = (
    "mb",
    "rmse",
    "max_ae",
    "local_std_mean",
    "slope_change_int",
    "asymmetry",
    "spr",
    "hfer",
    "missing_ratio",
    "asc",
    "mean_left_dB",
    "mean_right_dB",
    "rmse_left_dB",
    "rmse_right_dB",
    "max_abs_left_dB",
    "max_abs_right_dB",
    "effect_centroid_offset_m",
)

FAULT_LABELS = (
    "全链路功率衰减",
    "天线1功率下降",
    "天线2功率下降",
    "天线1方向偏移",
    "天线2方向偏移",
)

FAULT_COMPONENT_COLUMNS = (
    "has_global_attenuation",
    "has_antenna_1_power_loss",
    "has_antenna_2_power_loss",
    "has_antenna_1_tilt",
    "has_antenna_2_tilt",
)

PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE_GROUPS = {
    "simulator": (
        "motion_sampling.py",
        "receiver_measurement.py",
        "signal_generation.py",
    ),
    "faults": (
        "motion_sampling.py",
        "receiver_measurement.py",
        "signal_generation.py",
        "fault_injection.py",
    ),
    "features": ("feature_extraction.py",),
    "diagnostic": (
        "pipeline_contract.py",
        "baseline_artifact.py",
        "diagnostic_pipeline.py",
    ),
    "health": ("health_detection.py",),
    "sequence": ("sequence_data.py", "sequence_model.py", "gui.py"),
    "window": ("window_detection.py",),
    "training": ("model_training.py",),
    "multilabel_training": ("data_generation.py", "model_training.py"),
    "research_dataset": (
        "fault_scenarios.py",
        "research_dataset.py",
        "research_dataset_validation.py",
        "simulation_validation.py",
    ),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def stable_json(data: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_value(data), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def stable_hash(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_group_hash(group: str) -> str:
    filenames = SOURCE_GROUPS[group]
    digest = hashlib.sha256()
    for filename in filenames:
        path = PROJECT_ROOT / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def contract_source_hashes(groups: Iterable[str]) -> dict[str, str]:
    return {group: source_group_hash(group) for group in groups}


def effective_generator_config(generator_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve all public generator defaults and exclude per-observation controls."""
    from signal_generation import generate_rssi_simulation

    supplied = dict(generator_kwargs or {})
    supplied.pop("seed", None)
    supplied.pop("return_metadata", None)
    # Kept only in public function signatures for legacy callers.  The current
    # spatial complex-Gaussian Rician model has no discrete path-count control.
    supplied.pop("N_paths", None)
    signature = inspect.signature(generate_rssi_simulation)
    unknown = set(supplied) - set(signature.parameters)
    if unknown:
        raise ValueError(f"未知仿真参数: {sorted(unknown)}")

    config: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name in {"seed", "return_metadata", "N_paths"}:
            continue
        if name in supplied:
            value = supplied[name]
        elif parameter.default is not inspect.Parameter.empty:
            value = parameter.default
        else:
            continue
        config[name] = _json_value(value)
    return config


def generator_config_hash(generator_kwargs: Mapping[str, Any] | None) -> str:
    return stable_hash(effective_generator_config(generator_kwargs))


def curve_hash(x: np.ndarray, curve: np.ndarray) -> str:
    x_arr = np.ascontiguousarray(np.asarray(x, dtype="<f8"))
    curve_arr = np.ascontiguousarray(np.asarray(curve, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(x_arr.shape).encode("ascii"))
    digest.update(x_arr.tobytes())
    digest.update(str(curve_arr.shape).encode("ascii"))
    digest.update(curve_arr.tobytes())
    return digest.hexdigest()


def dataset_fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    X_arr = np.ascontiguousarray(np.asarray(X, dtype="<f8"))
    labels = [str(item) for item in np.asarray(y).tolist()]
    digest = hashlib.sha256()
    digest.update(str(X_arr.shape).encode("ascii"))
    digest.update(X_arr.tobytes())
    digest.update("\n".join(labels).encode("utf-8"))
    return digest.hexdigest()


def runtime_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(data), handle, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_manifest_fields(
    actual: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
    fields: Iterable[str],
) -> list[str]:
    if not actual:
        return ["缺少工件清单"]
    errors = []
    for field in fields:
        if actual.get(field) != expected.get(field):
            errors.append(
                f"{field} 不一致: 已保存={actual.get(field)!r}, 当前={expected.get(field)!r}"
            )
    return errors
