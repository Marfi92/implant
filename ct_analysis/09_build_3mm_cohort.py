"""Select the thick (3 mm) contrast CT series, one per patient, for a second NIfTI set.

The 0.5 mm cohort (06_build_af_cohort.py) is the thin reconstruction used for the
left atrium and the atrial fat. This script does the same job for the low
resolution reconstruction: contrast series with a slice thickness around 3 mm,
which is what most patients have when no thin reconstruction was stored.

Automatic check: a 3 mm series is only kept when it has enough slice positions to
cover the heart (--min-slices, default 40 = about 120 mm at 3 mm), has an ORIGINAL
reconstruction, contrast in the header, and no duplicated or missing slices.
Series with too few slices are never converted; they are listed in the Excel file
under Too_Few_Slices so nothing is silently lost.

Input : all_series_inventory_6sites.csv  (04_build_site_dicom_inventories.py)
        SCAPIS clinical CSV
Output: ct_3mm_cohort.xlsx
        Summary          - how many series/patients survive each filter
        Per_Site         - series and selected patients per site, so a missing site
                           (sites 5 and 6 were once absent from a stale inventory)
                           is visible before anything is converted
        All_Patients     - one row per clinical patient, segmentation_ready ok / not ok
        Selected         - the chosen 3 mm series per patient (input for script 07)
        Too_Few_Slices   - series dropped only because of poor coverage
        Rejected         - every dropped series with its reason

Then convert the selected series to one flat folder of NIfTI files:

    python 07_convert_cohort_to_nifti.py ^
        --cohort "Q:\\users\\marfi\\CT_analysis\\ct_3mm_cohort.xlsx" ^
        --sheet Selected --layout flat --output "W:\\SCAPIS_seg_lowres_3mm"

Usage
    python 09_build_3mm_cohort.py
    python 09_build_3mm_cohort.py --min-thickness 2.0 --max-thickness 3.5
    python 09_build_3mm_cohort.py --min-slices 50

In a Jupyter cell, pass the options as a list instead of relying on sys.argv:
    from importlib import import_module
    build = import_module("09_build_3mm_cohort")
    build.main(["--min-slices", "50"])
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

build_af = __import__("importlib").import_module("06_build_af_cohort")

SERIES_CSV = Path(r"Q:\users\marfi\CT_analysis\all_series_inventory_6sites.csv")
CLINICAL_CSV = Path(
    r"Q:\users\marfi\CT_analysis\SCAPIS1-DATA-PETITION-1110-20260407(in).csv"
)
OUTPUT = Path(r"Q:\users\marfi\CT_analysis\ct_3mm_cohort.xlsx")

SELECTED_COLUMNS = [
    "patient_id",
    "segmentation_ready",
    "group",
    "site_number",
    "site_name",
    "sex",
    "age",
    "bmi",
    "series_uid",
    "series_description",
    "series_role",
    "volume_class",
    "slices",
    "slice_thickness_mm",
    "spacing_between_slices_mm",
    "pixel_spacing",
    "rows",
    "columns",
    "kvp",
    "convolution_kernel",
    "phase_percent",
    "has_contrast",
    "image_type",
    "duplicate_sop_count",
    "n_dicom_files",
    "study_uid",
    "series_folder",
    "example_file",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-csv", type=Path, default=SERIES_CSV)
    parser.add_argument("--clinical-csv", type=Path, default=CLINICAL_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--site", default="", help="restrict to one site number")
    parser.add_argument(
        "--min-thickness",
        type=float,
        default=2.0,
        help="lowest slice thickness counted as a 3 mm series (default 2.0 mm)",
    )
    parser.add_argument(
        "--max-thickness",
        type=float,
        default=3.5,
        help="highest slice thickness counted as a 3 mm series (default 3.5 mm)",
    )
    parser.add_argument(
        "--min-slices",
        type=int,
        default=40,
        help="minimum unique slice positions; fewer means the heart is not covered "
        "and the series is not converted (default 40)",
    )
    parser.add_argument(
        "--require-contrast",
        action="store_true",
        default=True,
        help="keep only series with contrast in the header (default)",
    )
    parser.add_argument(
        "--allow-non-contrast",
        dest="require_contrast",
        action="store_false",
        help="also keep non-contrast 3 mm series",
    )
    parser.add_argument(
        "--target-phase",
        type=float,
        default=70.0,
        help="preferred cardiac phase percent (default 70)",
    )
    return parser.parse_known_args(argv)[0]


def filter_3mm(
    table: pd.DataFrame,
    min_thickness: float,
    max_thickness: float,
    min_slices: int,
    require_contrast: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep usable thick-slice volumes; return (kept, dropped-with-reason)."""
    reasons = pd.Series("", index=table.index)
    volume = table["volume_class"].astype(str)

    checks = [
        (~volume.isin(["3D", "4D", "4D (multi-series)"]), "not an image volume"),
        (table["is_original"] == 0, "not an ORIGINAL reconstruction"),
        (table["thickness"].isna(), "missing SliceThickness"),
        (table["thickness"] < min_thickness, f"thinner than {min_thickness} mm"),
        (table["thickness"] > max_thickness, f"thicker than {max_thickness} mm"),
        (table["duplicates"] > 0, "duplicate SOPInstanceUID"),
        (
            (table["sop_instances"] > 0)
            & ((table["slices"] - table["sop_instances"]).abs() > 2),
            "slice positions do not match image count",
        ),
        (table["slices"] < min_slices, f"fewer than {min_slices} slice positions"),
    ]
    if require_contrast:
        checks.insert(2, (~table["contrast"], "no contrast agent in the header"))

    for condition, reason in checks:
        reasons = reasons.mask((reasons == "") & condition.fillna(False), reason)

    kept = table[reasons == ""].copy()
    dropped = table[reasons != ""].copy()
    dropped["rejected_because"] = reasons[reasons != ""]
    return kept, dropped


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"Reading {args.series_csv} ...")
    series = pd.read_csv(args.series_csv, dtype=str, low_memory=False)
    if args.site:
        series = series[series["site_number"].astype(str) == str(args.site)]
    table = build_af.prepare_series(series, args.target_phase)

    kept, dropped = filter_3mm(
        table,
        args.min_thickness,
        args.max_thickness,
        args.min_slices,
        args.require_contrast,
    )
    selected = build_af.pick_one_per_patient(kept)

    print(f"Reading {args.clinical_csv} ...")
    clinical = build_af.load_clinical(args.clinical_csv)
    merged = selected.merge(clinical, on="patient_id", how="inner")
    merged["group"] = merged["af"].map({"YES": "AF", "NO": "control"}).fillna("unknown")
    merged["segmentation_ready"] = "ok"

    status = build_af.patient_status(clinical, table, dropped, selected)
    status = status.rename(columns={"usable": "segmentation_ready"})

    per_site = (
        table.groupby(["site_number", "site_name"])
        .size()
        .rename("series_in_inventory")
        .reset_index()
        .merge(
            kept.groupby(["site_number", "site_name"])
            .size()
            .rename("candidate_series")
            .reset_index(),
            on=["site_number", "site_name"],
            how="left",
        )
        .merge(
            merged.groupby(["site_number", "site_name"])
            .size()
            .rename("patients_selected")
            .reset_index(),
            on=["site_number", "site_name"],
            how="left",
        )
        .fillna({"candidate_series": 0, "patients_selected": 0})
        .sort_values("site_number")
    )
    print()
    print(per_site.to_string(index=False))
    empty = per_site.loc[per_site["patients_selected"] == 0, "site_name"].tolist()
    if empty:
        print(f"WARNING: no patient selected for site(s) {', '.join(empty)}")

    thin = dropped[dropped["rejected_because"].str.startswith("fewer than")]
    few_patients = thin.loc[~thin["patient_id"].isin(set(selected["patient_id"]))]

    summary = pd.DataFrame(
        [
            ("series in the inventory", len(table)),
            ("rejected series", len(dropped)),
            (
                f"candidate {args.min_thickness}-{args.max_thickness} mm series",
                len(kept),
            ),
            ("patients with a usable 3 mm series", len(selected)),
            ("of those, present in the clinical CSV", len(merged)),
            (
                f"series dropped for fewer than {args.min_slices} slices",
                len(thin),
            ),
            (
                "patients left without a volume because of too few slices",
                int(few_patients["patient_id"].nunique()),
            ),
            ("AF patients with a usable 3 mm series", int((merged["af"] == "YES").sum())),
            ("non-AF patients with a usable 3 mm series", int((merged["af"] == "NO").sum())),
        ],
        columns=["item", "count"],
    )
    print()
    for item, count in summary.itertuples(index=False):
        print(f"{item:<56} {count:>8,}")

    columns = [column for column in SELECTED_COLUMNS if column in merged.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        build_af.sanitize(summary).to_excel(writer, sheet_name="Summary", index=False)
        build_af.sanitize(per_site).to_excel(
            writer, sheet_name="Per_Site", index=False
        )
        build_af.sanitize(status).to_excel(
            writer, sheet_name="All_Patients", index=False
        )
        build_af.sanitize(merged[columns]).to_excel(
            writer, sheet_name="Selected", index=False
        )
        build_af.sanitize(
            thin[
                [
                    c
                    for c in [
                        "patient_id",
                        "series_uid",
                        "series_description",
                        "slices",
                        "slice_thickness_mm",
                        "series_folder",
                    ]
                    if c in thin.columns
                ]
            ]
        ).to_excel(writer, sheet_name="Too_Few_Slices", index=False)
        build_af.sanitize(
            dropped["rejected_because"]
            .value_counts()
            .rename("series")
            .reset_index()
            .rename(columns={"index": "rejected_because"})
        ).to_excel(writer, sheet_name="Rejected", index=False)

    print(f"\nExcel : {args.output}")
    print("Next  : python 07_convert_cohort_to_nifti.py "
          f'--cohort "{args.output}" --sheet Selected '
          '--layout flat --output "W:\\SCAPIS_seg_lowres_3mm"')


if __name__ == "__main__":
    main()
