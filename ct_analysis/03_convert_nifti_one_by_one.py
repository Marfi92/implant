"""
Script 3: Convert DICOM series to NIfTI one-by-one (space-efficient).

Strategy:
  1. Read the series summary Excel
  2. For each series (CASC or CCTA):
     a. Load DICOM slices -> 3D volume
     b. Save as NIfTI (.nii.gz) to output folder
     c. Optionally DELETE the original DICOM folder to free space
  3. Track progress in a log file so you can resume if interrupted

Input:  Q:\users\marfi\CT_analysis\dicom_series_summary.xlsx
        V:\datahub\CT-site-*  (source DICOM files)
Output: Q:\users\marfi\CT_analysis\nifti\<patient_id>\<series_type>_<series_number>.nii.gz

Usage:
    python 03_convert_nifti_one_by_one.py [--delete-after] [--type casc|ccta|all]

Options:
    --delete-after    Delete source DICOM files after successful conversion
    --type            Which series types to convert (default: all usable)
    --min-slices      Minimum slices to consider series valid (default: 20)
    --resume          Resume from last completed conversion
"""

import pydicom
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import json
import time
import sys
import shutil

# ============================================================
# CONFIGURATION
# ============================================================
WORK_DIR = Path(r"Q:\users\marfi\CT_analysis")
SUMMARY_FILE = WORK_DIR / "dicom_series_summary.xlsx"
OUT_NIFTI = WORK_DIR / "nifti"
PROGRESS_LOG = WORK_DIR / "conversion_progress.json"

OUT_NIFTI.mkdir(parents=True, exist_ok=True)


# ============================================================
# Progress tracking
# ============================================================
def load_progress():
    if PROGRESS_LOG.exists():
        with open(PROGRESS_LOG, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "skipped": []}


def save_progress(progress):
    with open(PROGRESS_LOG, "w") as f:
        json.dump(progress, f, indent=2)


# ============================================================
# DICOM to NIfTI conversion
# ============================================================
def find_dicom_files_for_series(example_file: str, series_uid: str) -> list:
    """Find all DICOM files belonging to a series from the example file path."""
    example_path = Path(example_file)

    if not example_path.exists():
        return []

    # The series folder is typically the parent of the example file
    series_folder = example_path.parent

    # Collect all DICOM files in this folder
    dicom_files = []
    for f in series_folder.iterdir():
        if f.is_file():
            try:
                with open(f, "rb") as fh:
                    pre = fh.read(132)
                if len(pre) == 132 and pre[128:132] == b"DICM":
                    # Verify it belongs to the same series
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                    if getattr(ds, "SeriesInstanceUID", "") == series_uid:
                        dicom_files.append(f)
            except Exception:
                pass

    return dicom_files


def sort_dicom_files(dicom_files: list) -> list:
    """Sort DICOM files by slice position or instance number."""
    def sort_key(path):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            # Prefer ImagePositionPatient[2] (z-position)
            if hasattr(ds, "ImagePositionPatient"):
                return float(ds.ImagePositionPatient[2])
            # Fallback to InstanceNumber
            if hasattr(ds, "InstanceNumber"):
                return int(ds.InstanceNumber)
        except Exception:
            pass
        return 0

    return sorted(dicom_files, key=sort_key)


def convert_series_to_nifti(series_row, output_path: Path) -> bool:
    """Convert one DICOM series to NIfTI using SimpleITK."""
    example_file = series_row["example_file"]
    series_uid = series_row["series_uid"]

    # Find all files for this series
    dicom_files = find_dicom_files_for_series(example_file, series_uid)

    if not dicom_files:
        print(f"    WARNING: No DICOM files found for series {series_uid}")
        return False

    # Sort by position
    dicom_files = sort_dicom_files(dicom_files)

    # Use SimpleITK to read the series
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(f) for f in dicom_files])
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        image = reader.Execute()

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write NIfTI
        sitk.WriteImage(image, str(output_path))
        return True

    except Exception as e:
        print(f"    ERROR converting: {e}")
        return False


def get_series_folder(example_file: str) -> Path:
    """Get the folder containing the series DICOM files."""
    return Path(example_file).parent


# ============================================================
# Main pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Convert DICOM series to NIfTI one-by-one")
    parser.add_argument("--delete-after", action="store_true",
                        help="Delete source DICOM files after successful conversion")
    parser.add_argument("--type", choices=["casc", "ccta", "all"], default="all",
                        help="Which series types to convert")
    parser.add_argument("--min-slices", type=int, default=20,
                        help="Minimum slices to consider series valid")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed conversion")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be converted without doing it")
    args = parser.parse_args()

    print("=" * 70)
    print("DICOM to NIfTI Converter (One-by-One)")
    print("=" * 70)
    print(f"Source summary: {SUMMARY_FILE}")
    print(f"Output folder:  {OUT_NIFTI}")
    print(f"Delete after:   {args.delete_after}")
    print(f"Series type:    {args.type}")
    print(f"Min slices:     {args.min_slices}")
    print()

    # Load summary
    df = pd.read_excel(SUMMARY_FILE, engine="openpyxl")
    print(f"Loaded {len(df)} series from summary")

    # Filter by type
    if args.type == "casc":
        df = df[df["series_type"] == "calcium_scoring"]
    elif args.type == "ccta":
        df = df[df["series_type"] == "ccta"]
    else:
        # All usable types (exclude localizers)
        df = df[df["series_type"].isin(["calcium_scoring", "ccta", "contrast", "other"])]

    # Filter by minimum slices
    df = df[df["n_slices"] >= args.min_slices]
    print(f"Series to convert: {len(df)} (after filtering)")

    if len(df) == 0:
        print("No series to convert.")
        return

    # Load progress
    progress = load_progress() if args.resume else {"completed": [], "failed": [], "skipped": []}

    # Dry run
    if args.dry_run:
        print("\n--- DRY RUN ---")
        for idx, row in df.iterrows():
            status = "SKIP" if row["series_uid"] in progress["completed"] else "CONVERT"
            print(f"  [{status}] {row['patient_id']} / {row['series_type']} "
                  f"({row['n_slices']} slices) - {row['series_description']}")
        return

    # Convert
    print(f"\nStarting conversion...")
    total = len(df)
    converted = 0
    failed = 0

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        series_uid = row["series_uid"]
        patient_id = str(row["patient_id"])
        series_type = row["series_type"]
        series_num = str(row.get("series_number", "000"))

        # Skip if already done
        if args.resume and series_uid in progress["completed"]:
            print(f"  [{idx}/{total}] SKIP (already done): {patient_id}/{series_type}")
            continue

        # Output filename
        safe_desc = "".join(c if c.isalnum() or c in "-_" else "_"
                           for c in str(row.get("series_description", "")))[:50]
        out_name = f"{series_type}_{series_num}_{safe_desc}.nii.gz"
        out_path = OUT_NIFTI / patient_id / out_name

        print(f"\n  [{idx}/{total}] Converting: {patient_id} / {series_type}")
        print(f"    Description: {row.get('series_description', '')}")
        print(f"    Slices: {row['n_slices']}")
        print(f"    Output: {out_path}")

        start = time.time()
        success = convert_series_to_nifti(row, out_path)
        elapsed = time.time() - start

        if success:
            converted += 1
            progress["completed"].append(series_uid)
            print(f"    OK ({elapsed:.1f}s)")

            # Free space estimate
            if out_path.exists():
                nifti_size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"    NIfTI size: {nifti_size_mb:.1f} MB")

            # Delete source if requested
            if args.delete_after:
                series_folder = get_series_folder(row["example_file"])
                if series_folder.exists():
                    folder_size = sum(
                        f.stat().st_size for f in series_folder.rglob("*") if f.is_file()
                    ) / (1024 * 1024)
                    shutil.rmtree(series_folder)
                    print(f"    DELETED source folder ({folder_size:.1f} MB freed)")

        else:
            failed += 1
            progress["failed"].append(series_uid)
            print(f"    FAILED ({elapsed:.1f}s)")

        # Save progress after each conversion
        save_progress(progress)

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"CONVERSION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Converted: {converted}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {total}")
    print(f"\nProgress saved to: {PROGRESS_LOG}")
    print(f"NIfTI files in:    {OUT_NIFTI}")


if __name__ == "__main__":
    main()
