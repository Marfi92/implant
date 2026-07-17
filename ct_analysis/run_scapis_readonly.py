"""Read SCAPIS DICOM headers and save all results in Marfi's folder."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

SOURCE = Path(r"Q:\users\leejo\data\scapis\datahub")
OUTPUT = Path(r"Q:\users\marfi\CT_analysis")
WORKERS = 8


def normalized(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def main() -> int:
    source = normalized(SOURCE)
    output = normalized(OUTPUT)

    if not SOURCE.is_dir():
        print(f"ERROR: Source folder was not found: {SOURCE}")
        print("Check that the Q: drive is connected.")
        return 2

    if os.path.commonpath([source, output]) == source:
        print("ERROR: Output must not be inside the read-only source folder.")
        return 2

    analyzer = Path(__file__).with_name("04_build_site_dicom_inventories.py")
    if not analyzer.is_file():
        print(f"ERROR: Required analyzer is missing: {analyzer}")
        return 2

    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("READ-ONLY SOURCE (nothing is changed or deleted):")
    print(f"  {SOURCE}")
    print("ALL OUTPUTS ARE SAVED HERE:")
    print(f"  {OUTPUT}\n")

    sys.argv = [
        str(analyzer),
        "--root",
        str(SOURCE),
        "--output",
        str(OUTPUT),
        "--workers",
        str(WORKERS),
    ]
    try:
        runpy.run_path(str(analyzer), run_name="__main__")
    except SystemExit as error:
        return int(error.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
