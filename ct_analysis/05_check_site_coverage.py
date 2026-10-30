"""Explain why a site has no inventory workbook.

The check walks each top-level folder under the data root, counts archives,
subfolders and DICOM headers, and compares that with the SQLite database and the
workbooks already present in the output folder.

Example (Windows):
    python 05_check_site_coverage.py \
      --root "Q:\\users\\leejo\\data\\scapis\\datahub" \
      --output "Q:\\users\\marfi\\CT_analysis"
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd
import pydicom

ARCHIVE_SUFFIXES = {".7z", ".zip", ".rar", ".gz", ".tar", ".tgz", ".bz2", ".xz"}
SITE_HINT_RE = re.compile(r"site[_-]?(?P<site>\d+)", re.IGNORECASE)
SAMPLE_FILES = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report per-site DICOM coverage and missing workbooks."
    )
    parser.add_argument("--root", default=r"Q:\users\leejo\data\scapis\datahub")
    parser.add_argument("--output", default=r"Q:\users\marfi\CT_analysis")
    parser.add_argument(
        "--sample",
        type=int,
        default=SAMPLE_FILES,
        help="Files per folder to test with pydicom (0 tests every file).",
    )
    return parser.parse_args()


def site_number(name: str) -> str:
    match = SITE_HINT_RE.search(name)
    return match.group("site") if match else "unknown"


def is_dicom(path: Path) -> bool:
    """Match script 04: accept a preamble or the identifying UIDs."""
    try:
        with path.open("rb") as stream:
            prefix = stream.read(132)
        if len(prefix) == 132 and prefix[128:132] == b"DICM":
            return True
        dataset = pydicom.dcmread(
            str(path), stop_before_pixels=True, force=True, defer_size=256
        )
    except Exception:
        return False
    return bool(
        dataset.get("SOPClassUID")
        or (dataset.get("StudyInstanceUID") and dataset.get("SeriesInstanceUID"))
        or (dataset.get("SeriesInstanceUID") and dataset.get("Modality"))
    )


def scan_folder(folder: Path, sample: int) -> dict[str, object]:
    files = 0
    archives = 0
    subfolders = 0
    tested = 0
    dicom_hits = 0
    first_file = ""

    for path in folder.rglob("*"):
        if path.is_dir():
            subfolders += 1
            continue
        files += 1
        if not first_file:
            first_file = str(path)
        if path.suffix.lower() in ARCHIVE_SUFFIXES:
            archives += 1
            continue
        if sample and tested >= sample:
            continue
        tested += 1
        if is_dicom(path):
            dicom_hits += 1

    if dicom_hits:
        status = "dicom_found"
    elif archives:
        status = "only_archives_extract_first"
    elif files:
        status = "no_dicom_headers"
    else:
        status = "empty_folder"

    return {
        "folder": folder.name,
        "site_number": site_number(folder.name),
        "files": files,
        "subfolders": subfolders,
        "archive_files": archives,
        "files_tested": tested,
        "dicom_headers_found": dicom_hits,
        "status": status,
        "example_file": first_file,
    }


def database_sites(database_path: Path) -> dict[str, int]:
    if not database_path.is_file():
        return {}
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT site_number, COUNT(*) FROM dicom_files GROUP BY site_number"
        ).fetchall()
    except sqlite3.DatabaseError:
        return {}
    finally:
        connection.close()
    return {str(site): int(count) for site, count in rows}


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)

    if not root.is_dir():
        print(f"ERROR: data root does not exist: {root}")
        return 2

    folders = sorted(item for item in root.iterdir() if item.is_dir())
    if not folders:
        print(f"ERROR: no folders below {root}")
        return 2

    print(f"Data root: {root}")
    print(f"Output:    {output}")
    print(f"Folders:   {len(folders)}\n")

    rows = []
    for folder in folders:
        row = scan_folder(folder, max(0, args.sample))
        rows.append(row)
        print(
            f"{row['folder']}: {row['files']:,} files, "
            f"{row['archive_files']:,} archives, "
            f"{row['dicom_headers_found']:,}/{row['files_tested']:,} sampled files are DICOM "
            f"-> {row['status']}"
        )

    coverage = pd.DataFrame(rows)
    counts = database_sites(output / "scapis_dicom_inventory.sqlite")
    coverage["dicom_files_in_database"] = coverage["site_number"].map(counts).fillna(0)
    coverage["workbook_exists"] = coverage["site_number"].map(
        lambda site: (output / f"site_{site}_dicom_inventory.xlsx").is_file()
    )

    report = output / "site_coverage_check.xlsx"
    output.mkdir(parents=True, exist_ok=True)
    coverage.to_excel(report, index=False)

    missing = coverage[~coverage["workbook_exists"]]
    if missing.empty:
        print("\nEvery folder has a matching site workbook.")
    else:
        print("\nFolders without a site workbook:")
        for _, row in missing.iterrows():
            print(f"  {row['folder']} (status={row['status']})")
        if (missing["status"] == "only_archives_extract_first").any():
            print(
                "  Extract the .7z/.zip archives in those folders "
                "(or point --root at the extracted copy) and re-run script 04."
            )

    print(f"\nSaved {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
