"""Convert long-format observations into a fixed-length RSSI matrix and mask."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_groups(path: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["sample_id"]].append(row)
    if not groups:
        raise ValueError("No observation rows were found")
    return dict(groups)


def resample(rows: list[dict[str, str]], length: int) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda row: float(row["time_s"]))
    time = np.asarray([float(row["time_s"]) for row in ordered], dtype=float)
    rssi = np.asarray([float(row["rssi_dbm"]) for row in ordered], dtype=float)
    valid = np.asarray(
        [row["is_valid"].strip().lower() in {"1", "true", "yes"} for row in ordered],
        dtype=bool,
    )
    if len(time) == 1:
        return np.repeat(rssi, length), np.repeat(valid, length)
    target = np.linspace(float(time[0]), float(time[-1]), int(length))
    values = np.interp(target, time, rssi)
    nearest = np.searchsorted(time, target, side="left")
    nearest = np.clip(nearest, 0, len(time) - 1)
    mask = valid[nearest]
    return values, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=256)
    args = parser.parse_args()
    if args.length < 2:
        raise ValueError("--length must be at least 2")

    groups = read_groups(args.observations)
    sample_ids = sorted(groups)
    arrays = []
    masks = []
    labels = []
    for sample_id in sample_ids:
        values, mask = resample(groups[sample_id], args.length)
        arrays.append(values)
        masks.append(mask)
        labels.append(groups[sample_id][0]["label"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=np.asarray(arrays, dtype=np.float32),
        mask=np.asarray(masks, dtype=bool),
        sample_ids=np.asarray(sample_ids),
        labels=np.asarray(labels),
    )
    print(
        f"Wrote {len(sample_ids)} samples with length {args.length} to {args.output}"
    )


if __name__ == "__main__":
    main()

