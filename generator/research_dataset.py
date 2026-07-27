"""Build a grouped RSSI research dataset with raw curves and audit metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from baseline_artifact import build_healthy_baseline
from diagnostic_pipeline import (
    baseline_reference_for_x,
    generate_diagnostic_observation,
    traditional_feature_vector,
)
from fault_scenarios import (
    CHANNEL_PROFILE_SPECS,
    COMPOSITE_LABEL,
    FAULT_COMPONENT_COLUMNS,
    FAULT_SCENARIO_SPECS,
    HEALTHY_LABEL,
    SEVERITY_LEVELS,
    component_indicator_values,
    sample_channel_profile,
    sample_composite_scenario,
    sample_fault_scenario,
    severity_for_index,
)
from pipeline_contract import (
    PIPELINE_SCHEMA_VERSION,
    RESEARCH_DATASET_SCHEMA_VERSION,
    TRADITIONAL_FEATURE_NAMES,
    contract_source_hashes,
    curve_hash,
    effective_generator_config,
    file_sha256,
    stable_hash,
    utc_now_iso,
    write_json,
)
from signal_generation import generate_composite_fault_rssi_pair


DEFAULT_OUTPUT_DIR = Path("datasets/rssi_research_v1")


def _write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    rows = [dict(row) for row in rows]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _site_split_map(site_ids: list[str], seed: int) -> dict[str, str]:
    if len(site_ids) < 9:
        raise ValueError("至少需要9个站点/AP组，才能形成独立训练、验证和测试组")
    site_array = np.asarray(site_ids)
    dummy = np.zeros((len(site_array), 1), dtype=float)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val_idx, test_idx = next(
        outer.split(dummy, groups=site_array)
    )
    train_val_sites = site_array[train_val_idx]
    inner_dummy = np.zeros((len(train_val_sites), 1), dtype=float)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_idx, val_idx = next(
        inner.split(inner_dummy, groups=train_val_sites)
    )
    result = {site_id: "train" for site_id in train_val_sites[train_idx]}
    result.update({site_id: "validation" for site_id in train_val_sites[val_idx]})
    result.update({site_id: "test" for site_id in site_array[test_idx]})
    return result


def _array_digest(arrays: Iterable[np.ndarray], sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    digest.update("\n".join(sample_ids).encode("utf-8"))
    return digest.hexdigest()


def _observation_from_composite(
    component_specs,
    generator_config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    x, healthy, faulty, metadata = generate_composite_fault_rssi_pair(
        component_specs,
        seed=int(seed),
        return_metadata=True,
        **generator_config,
    )
    return {
        "x": np.asarray(x, dtype=float),
        "healthy": np.asarray(healthy, dtype=float),
        "faulty": np.asarray(faulty, dtype=float),
        "fault_effect": np.asarray(faulty, dtype=float) - np.asarray(healthy, dtype=float),
        "simulation_metadata": metadata,
        "fault_metadata": metadata,
        "generator_config": dict(generator_config),
        "seed": int(seed),
        "fault_type": COMPOSITE_LABEL,
    }


def _write_long_curve_csv(
    path: Path,
    sample_ids: list[str],
    x: np.ndarray,
    paired_healthy: np.ndarray,
    reported_rssi: np.ndarray,
    baseline_reference: np.ndarray,
    residual: np.ndarray,
    fault_effect: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "sample_id",
        "position_m",
        "paired_healthy_rssi_dBm",
        "reported_rssi_dBm",
        "baseline_reference_rssi_dBm",
        "diagnostic_residual_dB",
        "physical_fault_effect_dB",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, sample_id in enumerate(sample_ids):
            for point_index, position in enumerate(x):
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "position_m": float(position),
                        "paired_healthy_rssi_dBm": float(paired_healthy[sample_index, point_index]),
                        "reported_rssi_dBm": float(reported_rssi[sample_index, point_index]),
                        "baseline_reference_rssi_dBm": float(baseline_reference[sample_index, point_index]),
                        "diagnostic_residual_dB": float(residual[sample_index, point_index]),
                        "physical_fault_effect_dB": float(fault_effect[sample_index, point_index]),
                    }
                )


def generate_research_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    n_sites: int = 15,
    trips_per_single_class: int = 3,
    healthy_trips_per_site: int = 3,
    composites_per_site: int = 3,
    baseline_trace_count: int = 12,
    seed: int = 20260716,
    base_generator_kwargs: Mapping[str, Any] | None = None,
    write_long_csv: bool = True,
) -> dict[str, Any]:
    """Generate raw curves, features, site baselines and group-safe splits.

    Every site contains all healthy and single-fault classes.  Sites, rather
    than individual curves or windows, are assigned to train/validation/test,
    so no baseline or channel profile is shared across evaluation splits.
    Composite faults are marked as multi-label evaluation samples and are not
    silently inserted into the five-class single-fault task.
    """
    if trips_per_single_class < len(SEVERITY_LEVELS):
        raise ValueError("每个单故障至少需要3条行程，以覆盖轻微、中等、严重三级")
    if healthy_trips_per_site < 1:
        raise ValueError("每个站点至少需要1条独立健康对照行程")
    if composites_per_site < len(SEVERITY_LEVELS):
        raise ValueError("每个站点至少需要3条复合故障行程，以覆盖三级严重度")
    if baseline_trace_count < 3:
        raise ValueError("健康参考基线至少需要3条独立行程")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(seed)
    base_config = effective_generator_config(base_generator_kwargs)
    site_ids = [f"site_{index:03d}" for index in range(n_sites)]
    split_by_site = _site_split_map(site_ids, seed)
    profile_names = tuple(CHANNEL_PROFILE_SPECS)

    site_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    site_baselines: dict[str, dict[str, Any]] = {}
    sample_ids: list[str] = []
    paired_healthy_curves: list[np.ndarray] = []
    reported_curves: list[np.ndarray] = []
    reference_curves: list[np.ndarray] = []
    residual_curves: list[np.ndarray] = []
    fault_effect_curves: list[np.ndarray] = []
    common_x = None

    for site_index, site_id in enumerate(site_ids):
        site_seed = int(master_rng.integers(1, 2**31 - 1))
        site_rng = np.random.default_rng(site_seed)
        profile_name = profile_names[site_index % len(profile_names)]
        sampled_profile = sample_channel_profile(profile_name, site_rng)
        site_config = dict(base_config)
        site_config.update(sampled_profile["generator_overrides"])
        site_config = effective_generator_config(site_config)
        baseline_seed = int(site_rng.integers(1, 2**31 - 1))
        baseline = build_healthy_baseline(
            site_config,
            trace_count=baseline_trace_count,
            seed=baseline_seed,
        )
        site_baselines[site_id] = baseline
        forbidden_seeds = set(int(item) for item in baseline["trace_seeds"])
        site_rows.append(
            {
                "site_id": site_id,
                "group_id": site_id,
                "split": split_by_site[site_id],
                "profile_name": profile_name,
                "profile_description": sampled_profile["profile_description"],
                "site_seed": site_seed,
                "baseline_seed": baseline_seed,
                "baseline_trace_count": baseline_trace_count,
                "baseline_hash": baseline["reference_hash"],
                "baseline_trace_seeds_json": json.dumps(baseline["trace_seeds"]),
                "generator_config_hash": baseline["generator_config_hash"],
                "generator_config_json": json.dumps(
                    site_config, ensure_ascii=False, sort_keys=True
                ),
                "speed_mps": float(site_config["v"]),
                "path_loss_exponent": float(site_config["n"]),
                "shadow_sigma_dB": float(site_config["sigma_shadow"]),
                "shadow_corr_distance_m": float(site_config["shadow_corr_distance_m"]),
                "rician_K_dB": float(sampled_profile["rician_K_dB"]),
                "receiver_noise_sigma_dB": float(site_config["receiver_noise_sigma_dB"]),
                "trip_power_sigma_dB": float(site_config["trip_power_sigma_dB"]),
                "position_alignment_sigma_m": float(site_config["position_alignment_sigma_m"]),
                "pointing_jitter_sigma_deg": float(site_config["pointing_jitter_sigma_deg"]),
                "measurement_window_s": float(site_config["measurement_window_s"]),
                "measurement_window_m_effective": float(
                    site_config["v"] * site_config["measurement_window_s"]
                ),
            }
        )

        scenarios: list[dict[str, Any]] = []
        for healthy_index in range(healthy_trips_per_site):
            scenarios.append(
                {
                    "fault_type": HEALTHY_LABEL,
                    "fault_components": [],
                    "severity_level": "无",
                    "severity_rank": 0,
                    "severity_score": 0.0,
                    "component": "无设备故障",
                    "mechanism": "独立健康行程，仅包含传播与测量波动",
                    "temporal_behavior": "healthy_control",
                    "scenario_index": healthy_index,
                }
            )
        for fault_type_name in FAULT_SCENARIO_SPECS:
            for trip_index in range(trips_per_single_class):
                scenario = sample_fault_scenario(
                    fault_type_name,
                    severity_for_index(trip_index),
                    site_rng,
                )
                scenario["scenario_index"] = trip_index
                scenarios.append(scenario)
        for composite_index in range(composites_per_site):
            scenario = sample_composite_scenario(
                site_index + composite_index,
                severity_for_index(composite_index),
                site_rng,
            )
            scenario["scenario_index"] = composite_index
            scenarios.append(scenario)

        for local_index, scenario in enumerate(scenarios):
            curve_seed = int(site_rng.integers(1, 2**31 - 1))
            while curve_seed in forbidden_seeds:
                curve_seed = int(site_rng.integers(1, 2**31 - 1))
            forbidden_seeds.add(curve_seed)
            fault_type_name = scenario["fault_type"]
            if fault_type_name == HEALTHY_LABEL:
                observation = generate_diagnostic_observation(
                    generator_kwargs=site_config,
                    seed=curve_seed,
                )
                fault_metadata = {
                    "fault_mechanism": scenario["mechanism"],
                    "fault_components": [],
                    "fault_parameters": {},
                }
            elif fault_type_name == COMPOSITE_LABEL:
                observation = _observation_from_composite(
                    scenario["component_specs"], site_config, curve_seed
                )
                fault_metadata = observation["fault_metadata"]
            else:
                observation = generate_diagnostic_observation(
                    generator_kwargs=site_config,
                    seed=curve_seed,
                    fault_type=fault_type_name,
                    fault_kwargs=scenario["fault_kwargs"],
                )
                fault_metadata = observation["fault_metadata"]

            x = np.asarray(observation["x"], dtype=float)
            if common_x is None:
                common_x = x.copy()
            elif not np.array_equal(common_x, x):
                raise RuntimeError("研究数据集内的位置网格必须一致")
            paired_healthy = np.asarray(observation["healthy"], dtype=float)
            reported = np.asarray(observation["faulty"], dtype=float)
            fault_effect = np.asarray(observation["fault_effect"], dtype=float)
            reference = baseline_reference_for_x(baseline, x, site_config)
            residual = reported - reference
            features, feature_vector = traditional_feature_vector(baseline, observation)
            sample_id = f"{site_id}_sample_{local_index:03d}"
            sample_ids.append(sample_id)
            paired_healthy_curves.append(paired_healthy)
            reported_curves.append(reported)
            reference_curves.append(reference)
            residual_curves.append(residual)
            fault_effect_curves.append(fault_effect)

            left = x <= float(site_config["antenna_x"])
            right = ~left
            components = list(scenario["fault_components"])
            row = {
                "sample_id": sample_id,
                "site_id": site_id,
                "group_id": site_id,
                "split": split_by_site[site_id],
                "profile_name": profile_name,
                "sample_role": (
                    "healthy_control"
                    if fault_type_name == HEALTHY_LABEL
                    else "composite_multilabel_evaluation"
                    if fault_type_name == COMPOSITE_LABEL
                    else "single_fault"
                ),
                "fault_type": fault_type_name,
                "fault_present": int(fault_type_name != HEALTHY_LABEL),
                "single_fault_eligible": int(
                    fault_type_name not in {HEALTHY_LABEL, COMPOSITE_LABEL}
                ),
                "fault_components_json": json.dumps(components, ensure_ascii=False),
                "fault_component_count": len(components),
                "severity_level": scenario["severity_level"],
                "severity_rank": scenario["severity_rank"],
                "severity_score": scenario["severity_score"],
                "degradation_stage": scenario["severity_rank"],
                "temporal_behavior": scenario["temporal_behavior"],
                "affected_component": scenario["component"],
                "fault_mechanism": fault_metadata["fault_mechanism"],
                "fault_parameters_json": json.dumps(
                    fault_metadata.get("fault_parameters", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "seed": curve_seed,
                "baseline_seed": baseline_seed,
                "baseline_hash": baseline["reference_hash"],
                "generator_config_hash": baseline["generator_config_hash"],
                "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
                "dataset_schema_version": RESEARCH_DATASET_SCHEMA_VERSION,
                "x_start": float(x[0]),
                "x_end": float(x[-1]),
                "dx": float(np.mean(np.diff(x))),
                "point_count": len(x),
                "speed_mps": float(site_config["v"]),
                "path_loss_exponent": float(site_config["n"]),
                "shadow_sigma_dB": float(site_config["sigma_shadow"]),
                "shadow_corr_distance_m": float(site_config["shadow_corr_distance_m"]),
                "rician_K_dB": float(sampled_profile["rician_K_dB"]),
                "measurement_window_s": float(site_config["measurement_window_s"]),
                "measurement_window_m_effective": float(
                    observation["simulation_metadata"]["measurement_window_m"]
                ),
                "trip_power_offset_dB": float(
                    observation["simulation_metadata"].get("trip_power_offset_dB", 0.0)
                ),
                "position_offset_m": float(
                    observation["simulation_metadata"]["position_offset_m"]
                ),
                "pointing_jitter_deg": float(
                    observation["simulation_metadata"]["pointing_jitter_deg"]
                ),
                "fault_effect_mean_dB": float(np.mean(fault_effect)),
                "fault_effect_abs_mean_dB": float(np.mean(np.abs(fault_effect))),
                "fault_effect_std_dB": float(np.std(fault_effect)),
                "fault_effect_max_abs_dB": float(np.max(np.abs(fault_effect))),
                "left_fault_effect_abs_mean_dB": float(np.mean(np.abs(fault_effect[left]))),
                "right_fault_effect_abs_mean_dB": float(np.mean(np.abs(fault_effect[right]))),
                "paired_healthy_curve_hash": curve_hash(x, paired_healthy),
                "reported_curve_hash": curve_hash(x, reported),
                "reference_curve_hash": baseline["reference_hash"],
                "feature_vector_hash": stable_hash(
                    {
                        name: float(value)
                        for name, value in zip(TRADITIONAL_FEATURE_NAMES, feature_vector)
                    }
                ),
            }
            row.update(component_indicator_values(components))
            row.update(
                {
                    "global_attenuation_dB": float(
                        fault_metadata.get("global_attenuation_dB", 0.0)
                    ),
                    "antenna_1_power_drop_dB": float(
                        fault_metadata.get("antenna_1_power_drop_dB", 0.0)
                    ),
                    "antenna_2_power_drop_dB": float(
                        fault_metadata.get("antenna_2_power_drop_dB", 0.0)
                    ),
                    "antenna_1_tilt_deg": float(
                        fault_metadata.get("antenna_1_tilt_deg", 0.0)
                    ),
                    "antenna_2_tilt_deg": float(
                        fault_metadata.get("antenna_2_tilt_deg", 0.0)
                    ),
                }
            )
            row.update({name: float(features[name]) for name in TRADITIONAL_FEATURE_NAMES})
            sample_rows.append(row)

    if common_x is None:
        raise RuntimeError("未生成任何数据")
    paired_healthy_array = np.stack(paired_healthy_curves)
    reported_array = np.stack(reported_curves)
    reference_array = np.stack(reference_curves)
    residual_array = np.stack(residual_curves)
    fault_effect_array = np.stack(fault_effect_curves)

    features_path = output_dir / "trace_features.csv"
    sites_path = output_dir / "site_profiles.csv"
    curves_path = output_dir / "curves.npz"
    baselines_path = output_dir / "site_baselines.pkl"
    long_curve_path = output_dir / "curve_points.csv"
    _write_csv(sample_rows, features_path)
    _write_csv(site_rows, sites_path)
    np.savez_compressed(
        curves_path,
        sample_ids=np.asarray(sample_ids),
        x=common_x,
        paired_healthy_rssi=paired_healthy_array,
        reported_rssi=reported_array,
        baseline_reference=reference_array,
        diagnostic_residual=residual_array,
        physical_fault_effect=fault_effect_array,
    )
    joblib.dump(site_baselines, baselines_path)
    if write_long_csv:
        _write_long_curve_csv(
            long_curve_path,
            sample_ids,
            common_x,
            paired_healthy_array,
            reported_array,
            reference_array,
            residual_array,
            fault_effect_array,
        )

    split_sample_counts = Counter(row["split"] for row in sample_rows)
    split_site_counts = Counter(row["split"] for row in site_rows)
    label_counts = Counter(row["fault_type"] for row in sample_rows)
    severity_counts = Counter(
        row["severity_level"]
        for row in sample_rows
        if row["fault_type"] not in {HEALTHY_LABEL}
    )
    files = {
        "trace_features.csv": file_sha256(features_path),
        "site_profiles.csv": file_sha256(sites_path),
        "curves.npz": file_sha256(curves_path),
        "site_baselines.pkl": file_sha256(baselines_path),
    }
    if write_long_csv:
        files["curve_points.csv"] = file_sha256(long_curve_path)
    manifest = {
        "artifact_type": "rssi_research_dataset",
        "dataset_schema_version": RESEARCH_DATASET_SCHEMA_VERSION,
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "seed": int(seed),
        "site_count": int(n_sites),
        "sample_count": len(sample_rows),
        "point_count_per_trace": int(len(common_x)),
        "baseline_trace_count_per_site": int(baseline_trace_count),
        "feature_names": list(TRADITIONAL_FEATURE_NAMES),
        "single_fault_labels": list(FAULT_SCENARIO_SPECS),
        "labels": [HEALTHY_LABEL, *FAULT_SCENARIO_SPECS, COMPOSITE_LABEL],
        "severity_levels": list(SEVERITY_LEVELS),
        "component_indicator_columns": list(FAULT_COMPONENT_COLUMNS.values()),
        "channel_profiles": list(CHANNEL_PROFILE_SPECS),
        "split_strategy": "GroupShuffleSplit by site_id; 60/20/20 groups",
        "split_site_counts": dict(split_site_counts),
        "split_sample_counts": dict(split_sample_counts),
        "label_counts": dict(label_counts),
        "severity_counts": dict(severity_counts),
        "write_long_csv": bool(write_long_csv),
        "dataset_fingerprint": _array_digest(
            (
                paired_healthy_array,
                reported_array,
                reference_array,
                residual_array,
                fault_effect_array,
            ),
            sample_ids,
        ),
        "source_hashes": contract_source_hashes(
            (
                "simulator",
                "faults",
                "features",
                "diagnostic",
                "research_dataset",
            )
        ),
        "files": files,
        "evidence_scope": (
            "仿真与公开测量先验约束的数据集；不能替代目标线路真实故障验证"
        ),
        "primary_references": [
            "ETSI/3GPP TR 38.901 spatial consistency and mobility modelling",
            "Railway measurements of path loss, log-normal shadowing, correlation and Rician fading",
            "Local five-fault FMEA/predictive-maintenance study",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the grouped RSSI research dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sites", type=int, default=15)
    parser.add_argument("--trips-per-class", type=int, default=3)
    parser.add_argument("--healthy-trips", type=int, default=3)
    parser.add_argument("--composites-per-site", type=int, default=3)
    parser.add_argument("--baseline-traces", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--skip-long-csv", action="store_true")
    args = parser.parse_args()
    manifest = generate_research_dataset(
        args.output,
        n_sites=args.sites,
        trips_per_single_class=args.trips_per_class,
        healthy_trips_per_site=args.healthy_trips,
        composites_per_site=args.composites_per_site,
        baseline_trace_count=args.baseline_traces,
        seed=args.seed,
        write_long_csv=not args.skip_long_csv,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
