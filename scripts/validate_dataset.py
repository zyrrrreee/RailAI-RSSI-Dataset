"""Validate the public repository's version-1 dataset contract.

The canonical representation is a long table.  Rows from different AP--OBM
links may share the same report time, so temporal monotonicity is checked per
physical link instead of across the whole CSV file.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


CANONICAL_LABELS = {
    "healthy",
    "global_power_attenuation",
    "antenna_1_power_loss",
    "antenna_2_power_loss",
    "antenna_1_direction_offset",
    "antenna_2_direction_offset",
}
OFFICIAL_SPLITS = ("train", "validation", "test_id", "test_ood")
AP_ID_PATTERN = re.compile(r"^AP-\d{3}$")
VERSION_1_OBM_IDS = {"OBM-front", "OBM-rear"}

REQUIRED_SCENARIO_COLUMNS = {
    "scenario_id",
    "route_start_m",
    "route_end_m",
    "ap_count",
    "obm_count",
}
REQUIRED_RUN_COLUMNS = {
    "scenario_id",
    "run_id",
    "target_ap_id",
    "label",
    "fault_target",
    "fault_parameter_name",
    "fault_parameter_value",
    "fault_parameter_unit",
    "random_seed",
    "generator_version",
    "config_hash",
    "normal_pair_run_id",
    "split",
}
REQUIRED_SAMPLE_COLUMNS = {
    "scenario_id",
    "run_id",
    "sample_id",
    "target_ap_id",
    "window_reference_obm_id",
    "window_start_m",
    "window_end_m",
    "label",
    "fault_target",
    "fault_parameter_name",
    "fault_parameter_value",
    "fault_parameter_unit",
    "quality_flag",
    "observation_file",
}
REQUIRED_OBSERVATION_COLUMNS = {
    "scenario_id",
    "run_id",
    "sample_id",
    "target_ap_id",
    "time_s",
    "position_m",
    "speed_mps",
    "ap_id",
    "obm_id",
    "rssi_dbm",
    "serving_ap_id",
    "is_serving",
    "is_valid",
    "receiver_status",
    "label",
}


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def require_columns(
    path: Path,
    columns: list[str],
    required: set[str],
    errors: list[str],
) -> bool:
    missing = required - set(columns)
    if missing:
        errors.append(f"{path} misses columns: {sorted(missing)}")
        return False
    return True


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def validate_splits(
    root: Path,
    sample_keys: set[tuple[str, str, str]],
    runs: list[dict[str, str]],
    errors: list[str],
) -> None:
    # Expected repository layout: <repo>/data/sample.
    repo_root = root.resolve().parents[1]
    split_root = repo_root / "splits"
    if not split_root.exists():
        return

    assignments: dict[tuple[str, str, str], str] = {}
    scenario_splits: dict[str, set[str]] = defaultdict(set)
    run_splits: dict[tuple[str, str], set[str]] = defaultdict(set)

    for split in OFFICIAL_SPLITS:
        path = split_root / f"{split}.csv"
        try:
            columns, rows = read_table(path)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if not require_columns(
            path,
            columns,
            {"scenario_id", "run_id", "sample_id"},
            errors,
        ):
            continue
        for row in rows:
            key = (row["scenario_id"], row["run_id"], row["sample_id"])
            if key not in sample_keys:
                errors.append(f"{path} references unknown sample: {key}")
                continue
            if key in assignments:
                errors.append(
                    f"sample appears in multiple splits: {key} "
                    f"({assignments[key]}, {split})"
                )
                continue
            assignments[key] = split
            scenario_splits[row["scenario_id"]].add(split)
            run_splits[(row["scenario_id"], row["run_id"])].add(split)

    unassigned = sample_keys - set(assignments)
    for key in sorted(unassigned):
        errors.append(f"sample is missing from official splits: {key}")

    for run_key, split_names in run_splits.items():
        if len(split_names) > 1:
            errors.append(f"run crosses official splits: {run_key}")

    for scenario_id, split_names in scenario_splits.items():
        if "test_ood" in split_names and split_names != {"test_ood"}:
            errors.append(
                f"test_ood scenario also appears in another split: {scenario_id}"
            )

    run_by_key = {
        (row["scenario_id"], row["run_id"]): row
        for row in runs
    }
    for run_key, row in run_by_key.items():
        assigned = run_splits.get(run_key, set())
        if assigned and row["split"] not in assigned:
            errors.append(
                f"run split metadata disagrees with split files: {run_key}"
            )
        pair_id = row["normal_pair_run_id"].strip()
        if not pair_id:
            continue
        pair_key = (row["scenario_id"], pair_id)
        if pair_key not in run_by_key:
            errors.append(f"run references unknown healthy pair: {run_key}")
            continue
        pair_splits = run_splits.get(pair_key, set())
        if assigned and pair_splits and assigned != pair_splits:
            errors.append(
                f"healthy/fault paired runs cross splits: {run_key}, {pair_key}"
            )


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    metadata = root / "metadata"

    scenario_path = metadata / "scenarios.csv"
    run_path = metadata / "runs.csv"
    sample_path = metadata / "samples.csv"
    scenario_columns, scenarios = read_table(scenario_path)
    run_columns, runs = read_table(run_path)
    sample_columns, samples = read_table(sample_path)

    if not require_columns(
        scenario_path,
        scenario_columns,
        REQUIRED_SCENARIO_COLUMNS,
        errors,
    ):
        return errors
    if not require_columns(run_path, run_columns, REQUIRED_RUN_COLUMNS, errors):
        return errors
    if not require_columns(
        sample_path,
        sample_columns,
        REQUIRED_SAMPLE_COLUMNS,
        errors,
    ):
        return errors

    scenario_ids = {row["scenario_id"] for row in scenarios}
    run_keys = {(row["scenario_id"], row["run_id"]) for row in runs}
    sample_keys = {
        (row["scenario_id"], row["run_id"], row["sample_id"])
        for row in samples
    }
    run_by_key = {
        (row["scenario_id"], row["run_id"]): row
        for row in runs
    }

    if len(scenario_ids) != len(scenarios):
        errors.append("scenario_id is not unique")
    if len(run_keys) != len(runs):
        errors.append("(scenario_id, run_id) is not unique")
    if len(sample_keys) != len(samples):
        errors.append("(scenario_id, run_id, sample_id) is not unique")

    for row in runs:
        if row["scenario_id"] not in scenario_ids:
            errors.append(f"run references unknown scenario: {row['run_id']}")
        if not AP_ID_PATTERN.fullmatch(row["target_ap_id"]):
            errors.append(f"invalid target AP ID in run: {row['target_ap_id']}")
        if row["label"] not in CANONICAL_LABELS:
            errors.append(f"unknown run label: {row['label']}")
        if row["split"] not in OFFICIAL_SPLITS:
            errors.append(f"unknown run split: {row['split']}")

    sample_count_by_run = Counter(
        (row["scenario_id"], row["run_id"]) for row in samples
    )
    for run_key in run_keys:
        if sample_count_by_run[run_key] != 1:
            errors.append(
                "version-1 contract requires exactly one sample per run: "
                f"{run_key}"
            )

    for row in samples:
        run_key = (row["scenario_id"], row["run_id"])
        sample_key = (*run_key, row["sample_id"])
        if run_key not in run_keys:
            errors.append(f"sample references unknown run: {row['sample_id']}")
            continue
        run = run_by_key[run_key]
        for field in (
            "target_ap_id",
            "label",
            "fault_target",
            "fault_parameter_name",
            "fault_parameter_value",
            "fault_parameter_unit",
        ):
            if row[field] != run[field]:
                errors.append(
                    f"sample {field} disagrees with run: {row['sample_id']}"
                )
        if row["label"] == "healthy":
            if row["fault_parameter_name"] != "none":
                errors.append(
                    f"healthy sample has a fault parameter: {row['sample_id']}"
                )
            if (
                row["fault_parameter_value"].strip()
                or row["fault_parameter_unit"].strip()
            ):
                errors.append(
                    f"healthy sample has a fault value/unit: {row['sample_id']}"
                )
        else:
            if row["fault_parameter_name"] == "none":
                errors.append(
                    f"fault sample lacks fault_parameter_name: {row['sample_id']}"
                )
            try:
                if float(row["fault_parameter_value"]) < 0:
                    raise ValueError
            except ValueError:
                errors.append(
                    f"fault sample has invalid fault_parameter_value: "
                    f"{row['sample_id']}"
                )
            if not row["fault_parameter_unit"].strip():
                errors.append(
                    f"fault sample lacks fault_parameter_unit: {row['sample_id']}"
                )
        if row["window_reference_obm_id"] not in VERSION_1_OBM_IDS:
            errors.append(
                "invalid window_reference_obm_id: "
                f"{row['window_reference_obm_id']}"
            )

        observation_path = (root / row["observation_file"]).resolve()
        try:
            observation_path.relative_to(root.resolve())
        except ValueError:
            errors.append(
                f"observation file escapes dataset root: {row['observation_file']}"
            )
            continue

        try:
            observation_columns, observations = read_table(observation_path)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if not observations:
            errors.append(f"sample has no observations: {row['sample_id']}")
            continue
        if not require_columns(
            observation_path,
            observation_columns,
            REQUIRED_OBSERVATION_COLUMNS,
            errors,
        ):
            continue

        link_times: dict[tuple[str, str], list[float]] = defaultdict(list)
        serving_groups: dict[tuple[float, str], list[dict[str, str]]] = (
            defaultdict(list)
        )
        seen_observation_keys: set[tuple[float, str, str]] = set()
        observed_ap_ids: set[str] = set()
        observed_obm_ids: set[str] = set()

        for observation in observations:
            observed_key = (
                observation["scenario_id"],
                observation["run_id"],
                observation["sample_id"],
            )
            if observed_key != sample_key:
                errors.append(
                    f"observation hierarchy mismatch in {observation_path}"
                )
                break
            if observation["target_ap_id"] != row["target_ap_id"]:
                errors.append(
                    f"observation target_ap_id mismatch in {observation_path}"
                )
                break
            if observation["label"] != row["label"]:
                errors.append(f"observation label mismatch in {observation_path}")
                break

            ap_id = observation["ap_id"]
            obm_id = observation["obm_id"]
            serving_ap_id = observation["serving_ap_id"]
            observed_ap_ids.add(ap_id)
            observed_obm_ids.add(obm_id)
            if not AP_ID_PATTERN.fullmatch(ap_id):
                errors.append(f"invalid candidate AP ID: {ap_id}")
            if not AP_ID_PATTERN.fullmatch(serving_ap_id):
                errors.append(f"invalid serving AP ID: {serving_ap_id}")
            if obm_id not in VERSION_1_OBM_IDS:
                errors.append(f"invalid version-1 OBM ID: {obm_id}")

            try:
                time_s = float(observation["time_s"])
                is_serving = parse_bool(observation["is_serving"])
                parse_bool(observation["is_valid"])
                float(observation["position_m"])
                float(observation["speed_mps"])
            except ValueError as exc:
                errors.append(f"{observation_path}: {exc}")
                continue

            unique_key = (time_s, ap_id, obm_id)
            if unique_key in seen_observation_keys:
                errors.append(
                    "duplicate (time_s, ap_id, obm_id) observation: "
                    f"{unique_key}"
                )
            seen_observation_keys.add(unique_key)
            link_times[(ap_id, obm_id)].append(time_s)
            serving_groups[(time_s, obm_id)].append(observation)

            if is_serving != (ap_id == serving_ap_id):
                errors.append(
                    "is_serving disagrees with ap_id/serving_ap_id: "
                    f"{unique_key}"
                )

        if row["target_ap_id"] not in observed_ap_ids:
            errors.append(
                f"target AP has no observations: {row['target_ap_id']}"
            )
        if len(observed_ap_ids) < 2:
            errors.append(
                f"sample lacks neighboring AP context: {row['sample_id']}"
            )
        missing_obms = VERSION_1_OBM_IDS - observed_obm_ids
        if missing_obms:
            errors.append(
                f"sample misses version-1 OBMs: {sorted(missing_obms)}"
            )

        for link, times in link_times.items():
            if any(
                right <= left
                for left, right in zip(times, times[1:])
            ):
                errors.append(
                    "time_s is not strictly increasing within link "
                    f"{link}: {row['sample_id']}"
                )

        for group_key, group_rows in serving_groups.items():
            serving_ids = {item["serving_ap_id"] for item in group_rows}
            try:
                serving_count = sum(
                    parse_bool(item["is_serving"]) for item in group_rows
                )
            except ValueError:
                continue
            if len(serving_ids) != 1 or serving_count != 1:
                errors.append(
                    "each (time_s, obm_id) group must identify exactly one "
                    f"serving AP: {group_key}"
                )

    validate_splits(root, sample_keys, runs, errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/sample"))
    args = parser.parse_args()
    errors = validate(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"PASS: version-1 dataset contract is valid under {args.root}")


if __name__ == "__main__":
    main()
