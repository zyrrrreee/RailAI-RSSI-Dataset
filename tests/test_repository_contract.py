from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_required_paths_exist(self):
        required = [
            "README.md",
            "DATASET_CARD.md",
            "CITATION.cff",
            "LICENSE",
            "configs",
            "data/sample",
            "metadata",
            "metadata/label_dictionary.csv",
            "splits",
            "splits/train.csv",
            "splits/validation.csv",
            "splits/test_id.csv",
            "splits/test_ood.csv",
            "generator",
            "scripts",
            "baselines",
            "docs",
            "checksums",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_sample_dataset_contract(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "validate_dataset.py"),
                "--root",
                str(ROOT / "data" / "sample"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_sample_is_multi_ap_and_dual_obm(self):
        observation_path = (
            ROOT / "data" / "sample" / "observations" / "sample_demo.csv"
        )
        import csv

        with observation_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["target_ap_id"] for row in rows}, {"AP-002"})
        self.assertEqual(
            {row["ap_id"] for row in rows},
            {"AP-001", "AP-002", "AP-003"},
        )
        self.assertEqual(
            {row["obm_id"] for row in rows},
            {"OBM-front", "OBM-rear"},
        )
        # Repeated report times are legal because rows retain AP--OBM identity.
        self.assertLess(len({row["time_s"] for row in rows}), len(rows))

    def test_converter_rejects_implicit_multi_link_mixing(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "convert_to_model_input.py"),
                "--observations",
                str(
                    ROOT
                    / "data"
                    / "sample"
                    / "observations"
                    / "sample_demo.csv"
                ),
                "--output",
                str(ROOT / "artifacts" / "should_not_exist.npz"),
                "--length",
                "8",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Specify both --ap-id and --obm-id", result.stderr)

    def test_converter_preserves_selected_link_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "selected_link.npz"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "convert_to_model_input.py"),
                    "--observations",
                    str(
                        ROOT
                        / "data"
                        / "sample"
                        / "observations"
                        / "sample_demo.csv"
                    ),
                    "--ap-id",
                    "AP-002",
                    "--obm-id",
                    "OBM-front",
                    "--output",
                    str(output),
                    "--length",
                    "8",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with np.load(output) as artifact:
                self.assertEqual(artifact["X"].shape, (1, 8))
                self.assertEqual(artifact["mask"].shape, (1, 8))
                self.assertEqual(artifact["selected_ap_ids"].tolist(), ["AP-002"])
                self.assertEqual(
                    artifact["selected_obm_ids"].tolist(),
                    ["OBM-front"],
                )


if __name__ == "__main__":
    unittest.main()
