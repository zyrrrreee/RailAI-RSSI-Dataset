"""Validate grouped RSSI research data without treating model accuracy as truth."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy import stats

from baseline_artifact import validate_healthy_baseline
from fault_scenarios import (
    COMPOSITE_LABEL,
    FAULT_COMPONENT_COLUMNS,
    FAULT_SCENARIO_SPECS,
    HEALTHY_LABEL,
    SEVERITY_LEVELS,
)
from feature_extraction import extract_features, features_to_array
from pipeline_contract import (
    PIPELINE_SCHEMA_VERSION,
    RESEARCH_DATASET_SCHEMA_VERSION,
    TRADITIONAL_FEATURE_NAMES,
    contract_source_hashes,
    curve_hash,
    file_sha256,
    read_json,
    write_json,
)
from research_dataset import DEFAULT_OUTPUT_DIR, _array_digest
from signal_generation import generate_ideal_rssi, generate_rssi_simulation
from simulation_validation import REFERENCE_RANGES


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row, key) -> float:
    return float(row[key])


def _recompute_feature_vector(reference, reported, x, baseline) -> np.ndarray:
    """Rebuild one feature row using the same physical AP split as generation."""
    split_position = float(
        baseline["generator_config"].get("antenna_x", 0.0)
    )
    return features_to_array(
        extract_features(
            reference,
            reported,
            x,
            split_position=split_position,
        )
    )


def _speed_window_effect(generator_config) -> tuple[bool, dict]:
    """Check fixed-time reporting and speed-invariant same-position geometry."""
    speed_config = dict(generator_config)
    speed_config["measurement_window_m"] = None
    speed_config["measurement_window_s"] = 0.05
    speed_config["v"] = 10.0
    slow = generate_rssi_simulation(seed=8128, return_metadata=True, **speed_config)
    speed_config["v"] = 30.0
    fast = generate_rssi_simulation(seed=8128, return_metadata=True, **speed_config)
    ideal_parameters = set(inspect.signature(generate_ideal_rssi).parameters)
    ideal_config = {
        key: value for key, value in speed_config.items() if key in ideal_parameters
    }
    ideal_config["v"] = 10.0
    slow_ideal = generate_ideal_rssi(**ideal_config)
    ideal_config["v"] = 30.0
    fast_ideal = generate_ideal_rssi(**ideal_config)
    slow_lookup = {round(float(x), 8): index for index, x in enumerate(slow_ideal[0])}
    fast_lookup = {round(float(x), 8): index for index, x in enumerate(fast_ideal[0])}
    common_positions = sorted(set(slow_lookup) & set(fast_lookup))
    gain_error = max(
        abs(
            float(slow_ideal[2][slow_lookup[position]])
            - float(fast_ideal[2][fast_lookup[position]])
        )
        for position in common_positions
    )
    path_loss_error = max(
        abs(
            float(slow_ideal[3][slow_lookup[position]])
            - float(fast_ideal[3][fast_lookup[position]])
        )
        for position in common_positions
    )
    fast_on_slow_grid = np.interp(slow[0], fast[0], fast[1])
    details = {
        "10mps_window_m": slow[6]["measurement_window_m"],
        "30mps_window_m": fast[6]["measurement_window_m"],
        "10mps_report_spacing_m": slow[6]["reported_spatial_interval_m_median"],
        "30mps_report_spacing_m": fast[6]["reported_spatial_interval_m_median"],
        "10mps_trip_duration_s": slow[6]["trip_duration_s"],
        "30mps_trip_duration_s": fast[6]["trip_duration_s"],
        "same_position_ideal_gain_max_error_dB": float(gain_error),
        "same_position_path_loss_max_error_dB": float(path_loss_error),
        "reported_rssi_mean_abs_difference_dB": float(
            np.mean(np.abs(np.asarray(slow[1]) - fast_on_slow_grid))
        ),
    }
    passed = (
        np.isclose(details["10mps_window_m"], 0.5)
        and np.isclose(details["30mps_window_m"], 1.5)
        and np.isclose(details["10mps_report_spacing_m"], 0.5)
        and np.isclose(details["30mps_report_spacing_m"], 1.5)
        and details["10mps_trip_duration_s"] > details["30mps_trip_duration_s"]
        and gain_error < 1e-12
        and path_loss_error < 1e-12
    )
    return bool(passed), details


def validate_research_dataset(dataset_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    dataset_dir = Path(dataset_dir)
    checks = []
    details = {}

    def check(name, condition, detail=""):
        passed = bool(condition)
        checks.append({"name": name, "passed": passed, "detail": str(detail)})
        return passed

    manifest = read_json(dataset_dir / "manifest.json")
    check("数据集清单存在", manifest is not None)
    if manifest is None:
        report = {"status": "fail", "checks": checks, "details": details}
        write_json(dataset_dir / "validation_report.json", report)
        return report
    check(
        "研究数据集版本正确",
        manifest.get("dataset_schema_version") == RESEARCH_DATASET_SCHEMA_VERSION
        and manifest.get("pipeline_schema_version") == PIPELINE_SCHEMA_VERSION,
    )
    expected_sources = contract_source_hashes(
        ("simulator", "faults", "features", "diagnostic", "research_dataset")
    )
    check(
        "生成器、故障和数据集代码哈希一致",
        manifest.get("source_hashes") == expected_sources,
    )

    for filename, expected_hash in manifest.get("files", {}).items():
        path = dataset_dir / filename
        check(f"文件存在: {filename}", path.exists())
        if path.exists():
            check(
                f"文件哈希正确: {filename}",
                file_sha256(path) == expected_hash,
            )

    rows = _read_csv(dataset_dir / "trace_features.csv")
    site_rows = _read_csv(dataset_dir / "site_profiles.csv")
    curves = np.load(dataset_dir / "curves.npz")
    baselines = joblib.load(dataset_dir / "site_baselines.pkl")
    sample_ids = curves["sample_ids"].astype(str).tolist()
    x = np.asarray(curves["x"], dtype=float)
    paired = np.asarray(curves["paired_healthy_rssi"], dtype=float)
    reported = np.asarray(curves["reported_rssi"], dtype=float)
    references = np.asarray(curves["baseline_reference"], dtype=float)
    residuals = np.asarray(curves["diagnostic_residual"], dtype=float)
    fault_effects = np.asarray(curves["physical_fault_effect"], dtype=float)

    check("样本数量与清单一致", len(rows) == manifest["sample_count"], len(rows))
    check("站点数量与清单一致", len(site_rows) == manifest["site_count"], len(site_rows))
    check("曲线样本ID与特征表顺序一致", sample_ids == [row["sample_id"] for row in rows])
    expected_shape = (len(rows), len(x))
    for name, values in {
        "配对健康曲线": paired,
        "观测曲线": reported,
        "站点基线": references,
        "诊断残差": residuals,
        "物理故障效应": fault_effects,
    }.items():
        check(f"{name}形状正确", values.shape == expected_shape, values.shape)
        check(f"{name}全部为有限值", np.isfinite(values).all())
    check("诊断残差恒等式成立", np.array_equal(residuals, reported - references))
    check("物理故障效应恒等式成立", np.array_equal(fault_effects, reported - paired))
    check(
        "数据指纹与清单一致",
        _array_digest(
            (paired, reported, references, residuals, fault_effects), sample_ids
        )
        == manifest["dataset_fingerprint"],
    )

    groups_by_split = defaultdict(set)
    for row in rows:
        groups_by_split[row["split"]].add(row["group_id"])
    split_names = {"train", "validation", "test"}
    check("训练/验证/测试三种划分均存在", set(groups_by_split) == split_names)
    overlap = (
        (groups_by_split["train"] & groups_by_split["validation"])
        | (groups_by_split["train"] & groups_by_split["test"])
        | (groups_by_split["validation"] & groups_by_split["test"])
    )
    check("站点/AP组在三个划分之间完全隔离", not overlap, sorted(overlap))
    site_split = {row["site_id"]: row["split"] for row in site_rows}
    check(
        "每条曲线的split与站点表一致",
        all(row["split"] == site_split[row["site_id"]] for row in rows),
    )

    expected_labels = {HEALTHY_LABEL, *FAULT_SCENARIO_SPECS, COMPOSITE_LABEL}
    label_counts_by_split = {}
    for split in sorted(split_names):
        split_counts = Counter(
            row["fault_type"] for row in rows if row["split"] == split
        )
        label_counts_by_split[split] = dict(split_counts)
        check(f"{split}划分包含全部健康/单故障/复合标签", set(split_counts) == expected_labels)
        for fault_name in FAULT_SCENARIO_SPECS:
            severities = {
                row["severity_level"]
                for row in rows
                if row["split"] == split and row["fault_type"] == fault_name
            }
            check(
                f"{split}-{fault_name}覆盖三级严重度",
                severities == set(SEVERITY_LEVELS),
                sorted(severities),
            )
    details["label_counts_by_split"] = label_counts_by_split

    sample_seed_values = [int(row["seed"]) for row in rows]
    check("所有观测行程随机种子唯一", len(sample_seed_values) == len(set(sample_seed_values)))
    baseline_seed_leaks = []
    baseline_errors = {}
    site_row_by_id = {row["site_id"]: row for row in site_rows}
    for site_id, baseline in baselines.items():
        config = baseline["generator_config"]
        errors = validate_healthy_baseline(
            baseline,
            config,
            trace_count=manifest["baseline_trace_count_per_site"],
        )
        if errors:
            baseline_errors[site_id] = errors
        baseline_seeds = set(int(seed) for seed in baseline["trace_seeds"])
        observation_seeds = {
            int(row["seed"]) for row in rows if row["site_id"] == site_id
        }
        if baseline_seeds & observation_seeds:
            baseline_seed_leaks.append(site_id)
        check(
            f"{site_id}基线哈希与站点表一致",
            baseline["reference_hash"] == site_row_by_id[site_id]["baseline_hash"],
        )
    check("全部站点基线版本和参数有效", not baseline_errors, baseline_errors)
    check("基线行程与数据样本不存在种子复用", not baseline_seed_leaks, baseline_seed_leaks)

    healthy_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["fault_type"] == HEALTHY_LABEL]
    )
    single_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["fault_type"] in FAULT_SCENARIO_SPECS]
    )
    composite_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["fault_type"] == COMPOSITE_LABEL]
    )
    check("健康对照的物理故障效应严格为0", np.array_equal(fault_effects[healthy_indices], np.zeros_like(fault_effects[healthy_indices])))
    check("健康对照的观测曲线等于配对健康曲线", np.array_equal(reported[healthy_indices], paired[healthy_indices]))
    check(
        "全部单故障至少产生可观测物理效应",
        np.all(np.mean(np.abs(fault_effects[single_indices]), axis=1) > 0.02),
    )
    pure_attenuation_indices = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if int(row["fault_present"]) == 1
            and int(row["has_antenna_1_tilt"]) == 0
            and int(row["has_antenna_2_tilt"]) == 0
        ]
    )
    tilt_indices = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if int(row["has_antenna_1_tilt"]) == 1
            or int(row["has_antenna_2_tilt"]) == 1
        ]
    )
    check(
        "纯衰减和功率下降不会产生非物理正增益",
        np.max(fault_effects[pure_attenuation_indices]) <= 1e-12,
        float(np.max(fault_effects[pure_attenuation_indices])),
    )
    check(
        "方向偏移仅允许极少局部增益且整体仍为净损失",
        float(np.mean(fault_effects[tilt_indices])) < 0.0
        and float(np.mean(fault_effects[tilt_indices] > 0.0)) < 0.01,
        {
            "mean_delta_dB": float(np.mean(fault_effects[tilt_indices])),
            "positive_point_fraction": float(np.mean(fault_effects[tilt_indices] > 0.0)),
        },
    )

    component_columns = list(FAULT_COMPONENT_COLUMNS.values())
    healthy_component_counts = [sum(int(row[name]) for name in component_columns) for row in rows if row["fault_type"] == HEALTHY_LABEL]
    single_component_counts = [sum(int(row[name]) for name in component_columns) for row in rows if row["fault_type"] in FAULT_SCENARIO_SPECS]
    composite_component_counts = [sum(int(row[name]) for name in component_columns) for row in rows if row["fault_type"] == COMPOSITE_LABEL]
    check("健康样本多标签向量全为0", set(healthy_component_counts) == {0})
    check("单故障样本多标签向量恰有1个分量", set(single_component_counts) == {1})
    check("复合故障样本保留2个故障分量", set(composite_component_counts) == {2})

    curve_hashes = []
    feature_mismatches = []
    baseline_curve_mismatches = []
    for index, row in enumerate(rows):
        curve_hashes.append(row["reported_curve_hash"])
        if row["reported_curve_hash"] != curve_hash(x, reported[index]):
            feature_mismatches.append(f"{row['sample_id']}:curve_hash")
        baseline = baselines[row["site_id"]]
        if not np.array_equal(references[index], baseline["reference_curve"]):
            baseline_curve_mismatches.append(row["sample_id"])
        computed = _recompute_feature_vector(
            references[index], reported[index], x, baseline
        )
        stored = np.asarray([_float(row, name) for name in TRADITIONAL_FEATURE_NAMES])
        if not np.allclose(computed, stored, rtol=0.0, atol=1e-10):
            feature_mismatches.append(f"{row['sample_id']}:features")
    check("观测曲线不存在完全重复样本", len(curve_hashes) == len(set(curve_hashes)))
    check(
        f"每条曲线和{len(TRADITIONAL_FEATURE_NAMES)}维特征均可从原始数组复算",
        not feature_mismatches,
        feature_mismatches[:10],
    )
    check("每条样本引用其所属站点基线", not baseline_curve_mismatches, baseline_curve_mismatches[:10])

    global_rows = [row for row in rows if row["fault_type"] == "全链路功率衰减"]
    antenna_1_power_rows = [row for row in rows if row["fault_type"] == "天线1功率下降"]
    antenna_2_power_rows = [row for row in rows if row["fault_type"] == "天线2功率下降"]
    antenna_1_tilt_rows = [row for row in rows if row["fault_type"] == "天线1方向偏移"]
    antenna_2_tilt_rows = [row for row in rows if row["fault_type"] == "天线2方向偏移"]
    check(
        "全链路衰减呈近似空间恒定效应",
        np.median([_float(row, "fault_effect_std_dB") for row in global_rows]) < 0.35,
    )
    check(
        "天线1功率下降主要影响天线1覆盖侧",
        np.mean([_float(row, "left_fault_effect_abs_mean_dB") for row in antenna_1_power_rows])
        > np.mean([_float(row, "right_fault_effect_abs_mean_dB") for row in antenna_1_power_rows]) + 0.25,
    )
    check(
        "天线2功率下降主要影响天线2覆盖侧",
        np.mean([_float(row, "right_fault_effect_abs_mean_dB") for row in antenna_2_power_rows])
        > np.mean([_float(row, "left_fault_effect_abs_mean_dB") for row in antenna_2_power_rows]) + 0.25,
    )
    check(
        "天线1方向偏移具有非恒定且侧向选择性效应",
        np.mean([_float(row, "fault_effect_std_dB") for row in antenna_1_tilt_rows]) > 0.30
        and np.mean([_float(row, "left_fault_effect_abs_mean_dB") for row in antenna_1_tilt_rows])
        > np.mean([_float(row, "right_fault_effect_abs_mean_dB") for row in antenna_1_tilt_rows]) + 0.20,
    )
    check(
        "天线2方向偏移具有非恒定且侧向选择性效应",
        np.mean([_float(row, "fault_effect_std_dB") for row in antenna_2_tilt_rows]) > 0.30
        and np.mean([_float(row, "right_fault_effect_abs_mean_dB") for row in antenna_2_tilt_rows])
        > np.mean([_float(row, "left_fault_effect_abs_mean_dB") for row in antenna_2_tilt_rows]) + 0.20,
    )

    severity_correlations = {}
    for fault_name in FAULT_SCENARIO_SPECS:
        fault_rows = [row for row in rows if row["fault_type"] == fault_name]
        correlation = stats.spearmanr(
            [_float(row, "severity_score") for row in fault_rows],
            [_float(row, "fault_effect_abs_mean_dB") for row in fault_rows],
        ).statistic
        severity_correlations[fault_name] = float(correlation)
        check(
            f"{fault_name}严重度与物理效应总体单调",
            np.isfinite(correlation) and correlation >= 0.55,
            correlation,
        )
    details["severity_spearman"] = severity_correlations

    prior_checks = {
        "path_loss_exponent": all(
            REFERENCE_RANGES["path_loss_exponent"]["min"] <= _float(row, "path_loss_exponent") <= REFERENCE_RANGES["path_loss_exponent"]["max"]
            for row in site_rows
        ),
        "shadow_sigma_dB": all(
            REFERENCE_RANGES["shadow_sigma_dB"]["min"] <= _float(row, "shadow_sigma_dB") <= REFERENCE_RANGES["shadow_sigma_dB"]["max"]
            for row in site_rows
        ),
        "shadow_corr_distance_m": all(
            REFERENCE_RANGES["shadow_corr_distance_m"]["min"] <= _float(row, "shadow_corr_distance_m") <= REFERENCE_RANGES["shadow_corr_distance_m"]["max"]
            for row in site_rows
        ),
        "rician_K_dB": all(
            REFERENCE_RANGES["rician_K_dB"]["min"] <= _float(row, "rician_K_dB") <= REFERENCE_RANGES["rician_K_dB"]["max"]
            for row in site_rows
        ),
    }
    for name, passed in prior_checks.items():
        check(f"全部站点{name}位于公开测量先验内", passed)

    speed_config = dict(next(iter(baselines.values()))["generator_config"])
    speed_passed, speed_details = _speed_window_effect(speed_config)
    check(
        "速度通过固定时间接收窗口真实影响快衰落平均",
        speed_passed,
        speed_details,
    )
    path_config = dict(speed_config)
    path_config["v"] = 20.0
    path_config["n"] = 2.2
    low_n = generate_rssi_simulation(seed=8128, return_metadata=True, **path_config)
    path_config["n"] = 3.4
    high_n = generate_rssi_simulation(seed=8128, return_metadata=True, **path_config)
    check(
        "路径损耗指数真实改变路径损耗和RSSI",
        not np.array_equal(low_n[3], high_n[3])
        and float(np.mean(high_n[1])) < float(np.mean(low_n[1])),
    )

    details["rssi_summary"] = {
        "mean_dBm": float(np.mean(reported)),
        "std_dB": float(np.std(reported)),
        "min_dBm": float(np.min(reported)),
        "max_dBm": float(np.max(reported)),
    }
    details["label_counts"] = dict(Counter(row["fault_type"] for row in rows))
    details["split_site_counts"] = {
        split: len(groups) for split, groups in groups_by_split.items()
    }
    failed = [item for item in checks if not item["passed"]]
    report = {
        "status": "pass" if not failed else "fail",
        "check_count": len(checks),
        "passed_count": len(checks) - len(failed),
        "failed_count": len(failed),
        "checks": checks,
        "details": details,
        "interpretation": (
            "数据集结构、物理机制和防泄漏检查通过；仍需真实健康数据校准和真实故障外部验证。"
            if not failed
            else "至少一项数据集构建检查失败，应修复后再用于实验。"
        ),
    }
    write_json(dataset_dir / "validation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the grouped RSSI research dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = validate_research_dataset(args.dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
