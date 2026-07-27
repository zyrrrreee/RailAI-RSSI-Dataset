"""Validate the public repository's Scenario/Run/Sample dataset contract."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REQUIRED_OBSERVATION_COLUMNS = {
    "scenario_id",
    "run_id",
    "sample_id",
    "time_s",
    "position_m",
    "speed_mps",
    "ap_id",
    "obm_id",
    "rssi_dbm",
    "serving_ap",
    "is_valid",
    "receiver_status",
    "label",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    metadata = root / "metadata"
    scenarios = read_rows(metadata / "scenarios.csv")
    runs = read_rows(metadata / "runs.csv")
    samples = read_rows(metadata / "samples.csv")

    scenario_ids = {row["scenario_id"] for row in scenarios}
    run_keys = {(row["scenario_id"], row["run_id"]) for row in runs}
    sample_keys = {
        (row["scenario_id"], row["run_id"], row["sample_id"]) for row in samples
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

    for row in samples:
        run_key = (row["scenario_id"], row["run_id"])
        sample_key = (*run_key, row["sample_id"])
        if run_key not in run_keys:
            errors.append(f"sample references unknown run: {row['sample_id']}")
        observation_path = root / row["observation_file"]
        observations = read_rows(observation_path)
        if not observations:
            errors.append(f"sample has no observations: {row['sample_id']}")
            continue
        missing_columns = REQUIRED_OBSERVATION_COLUMNS - set(observations[0])
        if missing_columns:
            errors.append(
                f"{observation_path} misses columns: {sorted(missing_columns)}"
            )
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
        times = [float(row_["time_s"]) for row_ in observations]
        if any(right <= left for left, right in zip(times, times[1:])):
            errors.append(f"time_s is not strictly increasing: {row['sample_id']}")

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
    print(f"PASS: dataset contract is valid under {args.root}")


if __name__ == "__main__":
    main()

