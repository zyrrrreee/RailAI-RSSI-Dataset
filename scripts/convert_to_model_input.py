"""Convert one explicitly selected AP--OBM link to fixed-length model input.

The public long table can contain several candidate APs and two OBMs at the
same time.  Mixing those rows into one curve destroys physical link identity,
so this version requires an explicit link whenever more than one is present.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


HierarchyKey = tuple[str, str, str]
LinkKey = tuple[str, str]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def read_groups(path: Path) -> dict[HierarchyKey, list[dict[str, str]]]:
    groups: dict[HierarchyKey, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "scenario_id",
            "run_id",
            "sample_id",
            "target_ap_id",
            "time_s",
            "ap_id",
            "obm_id",
            "rssi_dbm",
            "is_valid",
            "label",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Observation table misses columns: {sorted(missing)}")
        for row in reader:
            key = (row["scenario_id"], row["run_id"], row["sample_id"])
            groups[key].append(row)
    if not groups:
        raise ValueError("No observation rows were found")
    return dict(groups)


def select_link(
    rows: list[dict[str, str]],
    ap_id: str | None,
    obm_id: str | None,
) -> tuple[LinkKey, list[dict[str, str]]]:
    links = sorted({(row["ap_id"], row["obm_id"]) for row in rows})
    candidates = [
        link
        for link in links
        if (ap_id is None or link[0] == ap_id)
        and (obm_id is None or link[1] == obm_id)
    ]
    if len(candidates) != 1:
        available = ", ".join(f"{ap}/{obm}" for ap, obm in links)
        raise ValueError(
            "Expected exactly one AP--OBM link after filtering; "
            f"available links: {available}. Specify both --ap-id and --obm-id."
        )
    selected = candidates[0]
    return selected, [
        row
        for row in rows
        if (row["ap_id"], row["obm_id"]) == selected
    ]


def resample(
    rows: list[dict[str, str]],
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda row: float(row["time_s"]))
    time = np.asarray([float(row["time_s"]) for row in ordered], dtype=float)
    if len(np.unique(time)) != len(time):
        raise ValueError("The selected AP--OBM link contains duplicate time_s")

    valid = np.asarray(
        [parse_bool(row["is_valid"]) for row in ordered],
        dtype=bool,
    )
    rssi = np.full(len(ordered), np.nan, dtype=float)
    for index, row in enumerate(ordered):
        text = row["rssi_dbm"].strip()
        if not text:
            if valid[index]:
                raise ValueError("A valid observation has an empty rssi_dbm")
            continue
        try:
            rssi[index] = float(text)
        except ValueError as exc:
            if valid[index]:
                raise ValueError(
                    f"A valid observation has non-numeric rssi_dbm: {text!r}"
                ) from exc

    usable = valid & np.isfinite(rssi)
    if not np.any(usable):
        raise ValueError("The selected AP--OBM link has no valid RSSI values")

    if len(time) == 1:
        values = np.repeat(rssi[usable][0], length)
        return values, np.repeat(valid[0], length)

    target = np.linspace(float(time[0]), float(time[-1]), int(length))
    # Invalid receiver reports never enter the interpolation values.
    values = np.interp(target, time[usable], rssi[usable])
    nearest = np.searchsorted(time, target, side="left")
    nearest = np.clip(nearest, 0, len(time) - 1)
    left = np.maximum(nearest - 1, 0)
    choose_left = np.abs(target - time[left]) < np.abs(time[nearest] - target)
    nearest[choose_left] = left[choose_left]
    mask = valid[nearest]
    return values, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--ap-id")
    parser.add_argument("--obm-id")
    args = parser.parse_args()
    if args.length < 2:
        raise ValueError("--length must be at least 2")

    groups = read_groups(args.observations)
    hierarchy_keys = sorted(groups)
    arrays = []
    masks = []
    labels = []
    target_ap_ids = []
    selected_links = []

    for hierarchy_key in hierarchy_keys:
        selected_link, selected_rows = select_link(
            groups[hierarchy_key],
            args.ap_id,
            args.obm_id,
        )
        values, mask = resample(selected_rows, args.length)
        arrays.append(values)
        masks.append(mask)
        labels.append(selected_rows[0]["label"])
        target_ap_ids.append(selected_rows[0]["target_ap_id"])
        selected_links.append(selected_link)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=np.asarray(arrays, dtype=np.float32),
        mask=np.asarray(masks, dtype=bool),
        scenario_ids=np.asarray([key[0] for key in hierarchy_keys]),
        run_ids=np.asarray([key[1] for key in hierarchy_keys]),
        sample_ids=np.asarray([key[2] for key in hierarchy_keys]),
        target_ap_ids=np.asarray(target_ap_ids),
        selected_ap_ids=np.asarray([link[0] for link in selected_links]),
        selected_obm_ids=np.asarray([link[1] for link in selected_links]),
        labels=np.asarray(labels),
    )
    print(
        f"Wrote {len(hierarchy_keys)} samples with length {args.length} "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
