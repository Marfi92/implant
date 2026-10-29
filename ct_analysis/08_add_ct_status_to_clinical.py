"""Copy the SCAPIS clinical CSV to Excel and add the CT status of every patient.

Every original CSV column is kept untouched; these columns are appended on the right:

    ct_available            YES / NO   (any CT series for this Subject in the inventory)
    segmentation_ready      ok / not ok (a series usable for LA + atrial-fat work)
    why                     one sentence: why this series is good, or which rule removed it
    series_description      the chosen (or closest-to-passing) series
    slice_thickness_mm, n_slice_positions, kvp, convolution_kernel, phase_percent,
    heart_rate_bpm, duplicate_sop_count
    series_folder           the DICOM folder on Q: to convert
    example_file            one DICOM file from that series

The NIfTI file name per patient is assigned by script 07 and listed in nifti_index.xlsx.

Usage (Windows, from this folder):

    python 08_add_ct_status_to_clinical.py
    python 08_add_ct_status_to_clinical.py --allow-duplicates

In a notebook cell use `!python 08_add_ct_status_to_clinical.py`.
The selection rules are imported from 06_build_af_cohort.py, so both scripts always
agree on what "usable" means.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd

cohort = importlib.import_module("06_build_af_cohort")

CLINICAL_CSV = Path(
    r"Q:\users\marfi\CT_analysis\SCAPIS1-DATA-PETITION-1110-20260407(in).csv"
)
SERIES_CSV = Path(r"Q:\users\marfi\CT_analysis\all_series_inventory.csv")
OUTPUT = Path(r"Q:\users\marfi\CT_analysis\SCAPIS_clinical_with_CT_status.xlsx")

ADDED_COLUMNS = [
    "ct_available",
    "segmentation_ready",
    "why",
    "series_description",
    "slice_thickness_mm",
    "n_slice_positions",
    "kvp",
    "convolution_kernel",
    "phase_percent",
    "heart_rate_bpm",
    "duplicate_sop_count",
    "series_folder",
    "example_file",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-csv", type=Path, default=CLINICAL_CSV)
    parser.add_argument("--series-csv", type=Path, default=SERIES_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--min-slices", type=int, default=150)
    parser.add_argument("--max-thickness", type=float, default=1.0)
    parser.add_argument("--target-phase", type=float, default=70.0)
    parser.add_argument("--allow-duplicates", action="store_true")
    # parse_known_args so the notebook kernel's own -f argument is ignored
    return parser.parse_known_args(argv)[0]


def why_good(row: pd.Series) -> str:
    """Plain-language reason this series suits left-atrium and atrial-fat work."""
    parts = ["contrast CCTA, ORIGINAL reconstruction"]
    thickness = pd.to_numeric(row.get("slice_thickness_mm"), errors="coerce")
    if pd.notna(thickness):
        parts.append(f"{thickness:g} mm slices")
    kernel = str(row.get("convolution_kernel", "")).strip()
    if kernel:
        parts.append(f"{kernel} kernel")
    slices = pd.to_numeric(row.get("slices"), errors="coerce")
    if pd.notna(slices):
        parts.append(f"{int(slices)} slice positions covering the whole heart")
    phase = pd.to_numeric(row.get("phase_percent"), errors="coerce")
    if pd.notna(phase):
        parts.append(f"cardiac phase {phase:g}%")
    elif "bestdiast" in str(row.get("series_description", "")).lower():
        parts.append("best-diastolic reconstruction")
    return ", ".join(parts)


def status_table(
    series_csv: Path, args: argparse.Namespace
) -> tuple[pd.DataFrame, set[str]]:
    """Per-patient chosen series (or closest-to-passing one) plus the reason."""
    series = pd.read_csv(series_csv, dtype=str, low_memory=False)
    table = cohort.prepare_series(series, args.target_phase)
    kept, dropped = cohort.filter_candidates(
        table, args.min_slices, args.max_thickness, args.allow_duplicates
    )
    selected = cohort.pick_one_per_patient(kept)

    good = selected.copy()
    good["segmentation_ready"] = "ok"
    good["why"] = good.apply(why_good, axis=1)

    closest = dropped.copy()
    closest["is_ccta"] = (
        closest["series_role"].astype(str).isin(["ccta", "probable_ccta"]).astype(int)
    )
    closest = closest.sort_values(
        ["patient_id", "is_ccta", "slices"], ascending=[True, False, False]
    ).drop_duplicates(subset=["patient_id"], keep="first")
    bad = closest[~closest["patient_id"].isin(set(good["patient_id"]))].copy()
    bad["segmentation_ready"] = "not ok"
    bad["why"] = "removed because: " + bad["rejected_because"].astype(str)

    both = pd.concat([good, bad], ignore_index=True)
    both["n_slice_positions"] = both["slices"]
    both["ct_available"] = "YES"
    columns = ["patient_id"] + [
        column for column in ADDED_COLUMNS if column in both.columns
    ]
    return both[columns], set(table["patient_id"])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"Reading {args.series_csv} ...")
    status, patients_with_ct = status_table(args.series_csv, args)

    print(f"Reading {args.clinical_csv} ...")
    clinical = pd.read_csv(
        args.clinical_csv, dtype=str, low_memory=False, encoding="utf-8-sig"
    )
    subject = cohort.first_column(clinical, ["Subject", "subject", "SubjectID"])
    clinical["patient_id"] = clinical[subject].astype(str).str.strip()

    report = clinical.merge(status, on="patient_id", how="left").drop(
        columns=["patient_id"]
    )
    has_ct = clinical["patient_id"].isin(patients_with_ct)
    report["ct_available"] = report["ct_available"].fillna(
        has_ct.map({True: "YES", False: "NO"})
    )
    report["segmentation_ready"] = report["segmentation_ready"].fillna("not ok")
    report["why"] = report["why"].fillna(
        has_ct.map(
            {
                True: "CT exists but no series reached the quality checks",
                False: "no CT series for this Subject in the archive on Q:",
            }
        )
    )
    ready = report["segmentation_ready"].eq("ok")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        cohort.sanitize(report).to_excel(writer, sheet_name="Patients", index=False)

    af = report.get("AF_CT_Baseline", pd.Series(dtype=str)).astype(str).str.upper()
    print()
    print(f"patients in the clinical CSV        {len(report):>8,}")
    print(f"with CT in the archive              {int(has_ct.sum()):>8,}")
    print(f"ready for LA segmentation           {int(ready.sum()):>8,}")
    if not af.empty:
        print(f"AF patients                         {int((af == 'YES').sum()):>8,}")
        print(
            f"AF ready for LA segmentation        {int((ready & (af == 'YES')).sum()):>8,}"
        )
    print()
    print("Why patients are not ready (top reasons):")
    for reason, count in report.loc[~ready, "why"].value_counts().head(12).items():
        print(f"  {reason:<55} {count:>7,}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
