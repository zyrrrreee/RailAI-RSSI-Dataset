from pathlib import Path
import subprocess
import sys
import unittest


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
            "splits",
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


if __name__ == "__main__":
    unittest.main()

