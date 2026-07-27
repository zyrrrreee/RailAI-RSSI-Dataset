"""Evaluate representative classifiers on held-out AP/site groups.

This script does not create GUI model artifacts.  Its purpose is to report an
honest research metric using the split already stored in the research dataset.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fault_scenarios import FAULT_SCENARIO_SPECS
from pipeline_contract import TRADITIONAL_FEATURE_NAMES, write_json


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _models() -> dict[str, object]:
    return {
        "随机森林": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        ),
        "SVM": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf", C=1.0, gamma="scale")),
            ]
        ),
        "逻辑回归": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _metrics(y_true, y_pred, labels) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_grouped_models(dataset_dir: str | Path) -> dict:
    dataset_dir = Path(dataset_dir)
    rows = [
        row
        for row in _read_rows(dataset_dir / "trace_features.csv")
        if int(row["single_fault_eligible"]) == 1
    ]
    if not rows:
        raise ValueError("研究数据集中没有 single_fault_eligible=1 的样本")

    labels = list(FAULT_SCENARIO_SPECS)
    X = np.asarray(
        [
            [float(row[name]) for name in TRADITIONAL_FEATURE_NAMES]
            for row in rows
        ],
        dtype=float,
    )
    y = np.asarray([row["fault_type"] for row in rows])
    split = np.asarray([row["split"] for row in rows])
    sites = np.asarray([row["site_id"] for row in rows])

    train_mask = split == "train"
    validation_mask = split == "validation"
    test_mask = split == "test"
    if not (train_mask.any() and validation_mask.any() and test_mask.any()):
        raise ValueError("研究数据集必须同时包含 train/validation/test")

    train_sites = set(sites[train_mask])
    validation_sites = set(sites[validation_mask])
    test_sites = set(sites[test_mask])
    overlap = (
        (train_sites & validation_sites)
        | (train_sites & test_sites)
        | (validation_sites & test_sites)
    )
    if overlap:
        raise ValueError(f"站点分组发生泄漏: {sorted(overlap)}")

    model_results = {}
    for model_name, model in _models().items():
        model.fit(X[train_mask], y[train_mask])
        validation_pred = model.predict(X[validation_mask])
        test_pred = model.predict(X[test_mask])
        model_results[model_name] = {
            "validation": _metrics(y[validation_mask], validation_pred, labels),
            "test": _metrics(y[test_mask], test_pred, labels),
        }

    return {
        "evaluation_type": "held-out-site single-fault classification",
        "feature_names": list(TRADITIONAL_FEATURE_NAMES),
        "labels": labels,
        "sample_counts": {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "site_counts": {
            "train": len(train_sites),
            "validation": len(validation_sites),
            "test": len(test_sites),
        },
        "site_overlap": sorted(overlap),
        "models": model_results,
        "interpretation": (
            "测试集站点未参与训练；这些数值衡量仿真域跨站点泛化，"
            "不等于真实线路故障识别准确率。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="validation_outputs/rssi_research_v1",
    )
    parser.add_argument(
        "--output",
        default="validation_outputs/research_model_evaluation.json",
    )
    args = parser.parse_args()
    report = evaluate_grouped_models(args.dataset)
    write_json(args.output, report)
    print(f"grouped model evaluation written to {args.output}")
    for name, result in report["models"].items():
        print(
            f"- {name}: validation={result['validation']['accuracy']:.3f}, "
            f"test={result['test']['accuracy']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

