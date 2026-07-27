"""Run the current reproducible RSSI research-data generator.

This is a compatibility entry point for the verified stage-one generator.
The public Scenario/Run/Sample export layer is intentionally tracked as a
separate next step instead of silently changing the existing simulator.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "generator"


def main() -> None:
    sys.path.insert(0, str(GENERATOR))
    sys.argv[0] = str(GENERATOR / "research_dataset.py")
    runpy.run_path(str(GENERATOR / "research_dataset.py"), run_name="__main__")


if __name__ == "__main__":
    main()

