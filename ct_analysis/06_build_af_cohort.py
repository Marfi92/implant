"""Build an AF vs non-AF CCTA cohort for left-atrium segmentation.

Inputs
    all_series_inventory.csv  - written by 04_build_site_dicom_inventories.py (all six sites)
    SCAPIS clinical CSV       - Subject, AF_CT_Baseline, Sex, AgeAtVisitOne, Height, Weight, Site

Output: af_cohort.xlsx
    Summary          - how many series/patients survive each filter
    AF_Selected      - one clean CCTA series per AF patient
    Control_Selected - one clean CCTA series per non-AF patient (matching pool)
    Matched_Pairs    - 1:1 AF/control pairs matched on site, sex, kVp, thickness, age, BMI
    Pilot_nnUNet     - the small balanced starter set to annotate first
    Rejected         - every series that was dropped, with the reason

Why these filters: the left atrium and the atrial fat are only visible on the
contrast CCTA thin reconstruction, they need full heart coverage, no duplicated
or missing slices, and the same cardiac phase in both groups.

Usage
    python 06_build_af_cohort.py
    python 06_build_af_cohort.py --pilot 10 --site 1
    python 06_build_af_cohort.py --min-slices 200 --target-phase 40

In a Jupyter cell, pass the options as a list instead of relying on sys.argv:
    from importlib import import_module
    build = import_module("06_build_af_cohort")
    build.main(["--pilot", "10"])
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

EXCEL_ILLEGAL_RE = re.compile(r"[\000-\010\013\014\016-\037]")

SERIES_CSV = Path(r"Q:\users\marfi\CT_analysis\all_series_inventory.csv")
CLINICAL_CSV = Path(
    r"Q:\users\marfi\CT_analysis\SCAPIS1-DATA-PETITION-1110-20260407(in).csv"
)
OUTPUT = Path(r"Q:\users\marfi\CT_analysis\af_cohort.xlsx")

SITE_BY_NUMBER = {
    "1": "Goteborg",
    "2": "Malmo",
    "3": "Stockholm",
    "4": "Linkoping",
    "5": "Uppsala",
    "6": "Umea",
}

IMAGE_COLUMNS = [
    "site_number",
    "site_name",
    "patient_id",
    "study_uid",
    "series_uid",
    "series_number",
    "series_description",
    "series_role",
    "volume_class",
    "phase_percent",
    "n_dicom_files",
    "n_unique_sop_instances",
    "n_unique_slice_positions",
    "duplicate_sop_count",
    "slice_thickness_mm",
    "spacing_between_slices_mm",
    "pixel_spacing",
    "rows",
    "columns",
    "reconstruction_diameter_mm",
    "kvp",
    "convolution_kernel",
    "has_contrast",
    "contrast_agent",
    "heart_rate_bpm",
    "image_type",
    "data_quality_flags",
    "series_folder",
    "example_file",
]

CLINICAL_COLUMNS = [
    "af",
    "sex",
    "age",
    "height_cm",
    "weight_kg",
    "bmi",
    "clinical_site",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-csv", type=Path, default=SERIES_CSV)
    parser.add_argument("--clinical-csv", type=Path, default=CLINICAL_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--site", default="", help="restrict to one site number")
    parser.add_argument(
        "--min-slices",
        type=int,
        default=150,
        help="minimum unique slice positions for full heart coverage (default 150)",
    )
    parser.add_argument(
        "--max-thickness",
        type=float,
        default=1.0,
        help="maximum slice thickness in mm (default 1.0)",
    )
    parser.add_argument(
        "--target-phase",
        type=float,
        default=70.0,
        help="preferred cardiac phase percent (default 70 = BestDiast)",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=10,
        help="matched pairs to put in the pilot annotation set (default 10)",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help=(
            "keep series whose images are repeated by overlapping archive batches "
            "(script 07 then converts the de-duplicated slices)"
        ),
    )
    parser.add_argument(
        "--age-tolerance",
        type=float,
        default=5.0,
        help="maximum age difference within a matched pair (default 5 years)",
    )
    # parse_known_args so the notebook kernel's own -f argument is ignored
    return parser.parse_known_args(argv)[0]


def sanitize(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip control characters that openpyxl refuses to write."""
    cleaned = frame.copy()
    for column in cleaned.columns[cleaned.dtypes == object]:
        cleaned[column] = cleaned[column].map(
            lambda value: (
                EXCEL_ILLEGAL_RE.sub("", value) if isinstance(value, str) else value
            )
        )
    return cleaned


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce")


def first_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise SystemExit(
        f"None of the expected columns {candidates} exist in the clinical CSV. "
        f"Columns found: {list(frame.columns)[:20]}"
    )


def load_clinical(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8-sig")
    subject = first_column(raw, ["Subject", "subject", "SubjectID"])
    af = first_column(raw, ["AF_CT_Baseline", "AF_CT_baseline", "AF"])
    table = pd.DataFrame({"patient_id": raw[subject].astype(str).str.strip()})
    table["af"] = raw[af].astype(str).str.strip().str.upper()
    for name, candidates in [
        ("sex", ["Sex", "sex", "Gender"]),
        ("age", ["AgeAtVisitOne", "Age", "age"]),
        ("height_cm", ["Height", "height"]),
        ("weight_kg", ["Weight", "weight"]),
        ("clinical_site", ["Site", "site", "City"]),
    ]:
        for candidate in candidates:
            if candidate in raw.columns:
                table[name] = raw[candidate]
                break
        else:
            table[name] = pd.NA
    table["age"] = pd.to_numeric(table["age"], errors="coerce")
    height_m = pd.to_numeric(table["height_cm"], errors="coerce") / 100.0
    table["bmi"] = (
        pd.to_numeric(table["weight_kg"], errors="coerce") / height_m.pow(2)
    ).round(1)
    return table


def prepare_series(frame: pd.DataFrame, target_phase: float) -> pd.DataFrame:
    table = frame.copy()
    table["site_number"] = table["site_number"].astype(str)
    table["site_name"] = table["site_number"].map(SITE_BY_NUMBER).fillna("unknown")
    table["slices"] = numeric(table, "n_unique_slice_positions").fillna(0)
    table["sop_instances"] = numeric(table, "n_unique_sop_instances").fillna(0)
    table["duplicates"] = numeric(table, "duplicate_sop_count").fillna(0)
    table["thickness"] = numeric(table, "slice_thickness_mm")
    table["phase_percent"] = numeric(table, "description_phase_percent").fillna(
        numeric(table, "scan_options_phase_percent")
    )
    table["phase_distance"] = (table["phase_percent"] - target_phase).abs().fillna(999)
    table["is_best_diast"] = (
        table["series_description"]
        .astype(str)
        .str.contains("bestdiast", case=False, na=False)
        .astype(int)
    )
    table["is_thin_kernel"] = (
        table["convolution_kernel"]
        .astype(str)
        .str.contains("i26f", case=False, na=False)
        .astype(int)
    )
    table["is_original"] = (
        table["image_type"]
        .astype(str)
        .str.upper()
        .str.contains("ORIGINAL", na=False)
        .astype(int)
    )
    table["contrast"] = table["has_contrast"].astype(str).str.upper().isin(
        ["TRUE", "1", "YES"]
    )
    # matching strata must compare values, not their text formatting (0.5 vs 0.500)
    table["kvp_key"] = numeric(table, "kvp").round(0)
    table["thickness_key"] = table["thickness"].round(2)
    return table


def filter_candidates(
    table: pd.DataFrame,
    min_slices: int,
    max_thickness: float,
    allow_duplicates: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only series usable for left-atrium segmentation."""
    reasons = pd.Series("", index=table.index)
    role = table["series_role"].astype(str)
    volume = table["volume_class"].astype(str)

    checks = [
        (~role.isin(["ccta", "probable_ccta"]), "not a CCTA series"),
        (~volume.isin(["3D", "4D", "4D (multi-series)"]), "not an image volume"),
        (~table["contrast"], "no contrast agent in the header"),
        (table["is_original"] == 0, "not an ORIGINAL reconstruction"),
        (table["thickness"].isna(), "missing SliceThickness"),
        (table["thickness"] > max_thickness, f"thicker than {max_thickness} mm"),
        (table["slices"] < min_slices, f"fewer than {min_slices} slice positions"),
    ]
    if not allow_duplicates:
        checks += [
            (table["duplicates"] > 0, "duplicate SOPInstanceUID"),
            (
                (table["sop_instances"] > 0)
                & ((table["slices"] - table["sop_instances"]).abs() > 2),
                "slice positions do not match image count",
            ),
        ]
    for condition, reason in checks:
        reasons = reasons.mask((reasons == "") & condition.fillna(False), reason)

    kept = table[reasons == ""].copy()
    dropped = table[reasons != ""].copy()
    dropped["rejected_because"] = reasons[reasons != ""]
    return kept, dropped


def pick_one_per_patient(kept: pd.DataFrame) -> pd.DataFrame:
    """Prefer BestDiast / target phase, then the I26f thin kernel, then most slices."""
    if kept.empty:
        return kept
    ordered = kept.sort_values(
        by=[
            "patient_id",
            "is_best_diast",
            "phase_distance",
            "is_thin_kernel",
            "thickness",
            "slices",
        ],
        ascending=[True, False, True, False, True, False],
    )
    return ordered.drop_duplicates(subset=["patient_id"], keep="first")


def match_pairs(
    af_table: pd.DataFrame, control_table: pd.DataFrame, age_tolerance: float
) -> pd.DataFrame:
    """Greedy 1:1 matching inside exact strata, closest age then closest BMI."""
    strata = ["site_number", "sex", "kvp_key", "thickness_key"]
    available = control_table.copy()
    available["used"] = False
    pairs: list[dict[str, object]] = []

    for _, case in af_table.iterrows():
        mask = ~available["used"]
        for column in strata:
            mask &= available[column].astype(str) == str(case[column])
        pool = available[mask]
        if pool.empty:
            pairs.append({"af_patient_id": case["patient_id"], "matched": False})
            continue
        age_difference = (pool["age"] - case["age"]).abs()
        bmi_difference = (pool["bmi"] - case["bmi"]).abs()
        eligible = pool[age_difference <= age_tolerance]
        if eligible.empty:
            eligible = pool
            age_difference = age_difference.loc[eligible.index]
            bmi_difference = bmi_difference.loc[eligible.index]
        else:
            age_difference = age_difference.loc[eligible.index]
            bmi_difference = bmi_difference.loc[eligible.index]
        score = age_difference.fillna(99) + bmi_difference.fillna(99) / 5.0
        best = score.idxmin()
        available.loc[best, "used"] = True
        control = control_table.loc[best]
        pairs.append(
            {
                "af_patient_id": case["patient_id"],
                "af_site": case["site_name"],
                "af_sex": case["sex"],
                "af_age": case["age"],
                "af_bmi": case["bmi"],
                "af_kvp": case["kvp"],
                "af_thickness_mm": case["slice_thickness_mm"],
                "af_slices": case["slices"],
                "af_series_uid": case["series_uid"],
                "af_series_folder": case["series_folder"],
                "af_kernel": case["convolution_kernel"],
                "af_phase_percent": case["phase_percent"],
                "af_heart_rate": case.get("heart_rate_bpm"),
                "control_patient_id": control["patient_id"],
                "control_age": control["age"],
                "control_bmi": control["bmi"],
                "control_slices": control["slices"],
                "control_series_uid": control["series_uid"],
                "control_series_folder": control["series_folder"],
                "control_kernel": control["convolution_kernel"],
                "control_phase_percent": control["phase_percent"],
                "control_heart_rate": control.get("heart_rate_bpm"),
                "control_sex": control["sex"],
                "control_kvp": control["kvp"],
                "control_thickness_mm": control["slice_thickness_mm"],
                "age_difference": abs(float(case["age"]) - float(control["age"]))
                if pd.notna(case["age"]) and pd.notna(control["age"])
                else None,
                "matched": True,
            }
        )
    table = pd.DataFrame(pairs)
    for column in ["af_patient_id", "matched", "control_patient_id"]:
        if column not in table.columns:
            table[column] = False if column == "matched" else pd.NA
    return table


PILOT_DETAIL_COLUMNS = [
    "site_name",
    "sex",
    "age",
    "bmi",
    "kvp",
    "slice_thickness_mm",
    "spacing_between_slices_mm",
    "pixel_spacing",
    "rows",
    "columns",
    "reconstruction_diameter_mm",
    "convolution_kernel",
    "phase_percent",
    "heart_rate_bpm",
    "slices",
    "n_dicom_files",
    "duplicate_sop_count",
    "series_description",
    "image_type",
    "contrast_agent",
    "study_uid",
    "series_folder",
    "example_file",
]


def pilot_set(pairs: pd.DataFrame, merged: pd.DataFrame, count: int) -> pd.DataFrame:
    """Balanced starter set for annotation: one nnU-Net case per row, full detail."""
    matched = pairs[pairs["matched"] == True].head(count)
    by_patient = merged.set_index("patient_id")
    detail = [column for column in PILOT_DETAIL_COLUMNS if column in merged.columns]
    rows: list[dict[str, object]] = []
    for index, pair in enumerate(matched.itertuples(index=False), start=1):
        for offset, (group, patient, series_uid) in enumerate(
            [
                ("AF", pair.af_patient_id, pair.af_series_uid),
                ("control", pair.control_patient_id, pair.control_series_uid),
            ]
        ):
            source = by_patient.loc[patient]
            row: dict[str, object] = {
                "case_id": f"SCAPIS_{2 * index - 1 + offset:03d}",
                "group": group,
                "pair_id": index,
                "patient_id": patient,
                "series_uid": series_uid,
            }
            for column in detail:
                row[column] = source[column]
            rows.append(row)
    return pd.DataFrame(rows)


def af_not_usable(
    table: pd.DataFrame, dropped: pd.DataFrame, clinical: pd.DataFrame, kept_ids: set[str]
) -> pd.DataFrame:
    """One row per AF patient without a usable series, and the rule that removed it.

    The reason shown is the one from that patient's closest-to-passing series (the
    contrast CCTA row with the most slice positions), so relaxing that single rule
    tells you how many AF patients you would recover.
    """
    af_ids = set(clinical.loc[clinical["af"] == "YES", "patient_id"])
    missing = af_ids - kept_ids
    with_series = set(table.loc[table["patient_id"].isin(missing), "patient_id"])
    rows: list[dict[str, object]] = []
    for patient in sorted(missing - with_series):
        rows.append(
            {
                "patient_id": patient,
                "reason": "no CT series in the inventory",
                "series_description": "",
                "slices": 0,
                "duplicate_sop_count": "",
                "slice_thickness_mm": "",
            }
        )
    candidates = dropped[dropped["patient_id"].isin(with_series)]
    preferred = candidates[
        candidates["series_role"].astype(str).isin(["ccta", "probable_ccta"])
    ]
    for patient, group in (preferred if not preferred.empty else candidates).groupby(
        "patient_id"
    ):
        best = group.sort_values("slices", ascending=False).iloc[0]
        rows.append(
            {
                "patient_id": str(patient),
                "reason": best["rejected_because"],
                "series_description": best.get("series_description", ""),
                "slices": best["slices"],
                "duplicate_sop_count": best.get("duplicate_sop_count", ""),
                "slice_thickness_mm": best.get("slice_thickness_mm", ""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "patient_id",
            "reason",
            "series_description",
            "slices",
            "duplicate_sop_count",
            "slice_thickness_mm",
        ],
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"Reading {args.series_csv} ...")
    series = pd.read_csv(args.series_csv, dtype=str, low_memory=False)
    if args.site:
        series = series[series["site_number"].astype(str) == str(args.site)]
    table = prepare_series(series, args.target_phase)

    kept, dropped = filter_candidates(
        table, args.min_slices, args.max_thickness, args.allow_duplicates
    )
    selected = pick_one_per_patient(kept)

    print(f"Reading {args.clinical_csv} ...")
    clinical = load_clinical(args.clinical_csv)

    merged = selected.merge(clinical, on="patient_id", how="inner")
    af_table = merged[merged["af"] == "YES"].copy()
    control_table = merged[merged["af"] == "NO"].copy()

    pairs = match_pairs(af_table, control_table, args.age_tolerance)
    matched_controls = set(pairs.loc[pairs["matched"], "control_patient_id"])
    pilot = pilot_set(pairs, merged, args.pilot)
    lost = af_not_usable(table, dropped, clinical, set(selected["patient_id"]))
    reasons = (
        dropped["rejected_because"].value_counts().rename("series").reset_index()
    )
    reasons.columns = ["rejected_because", "series"]
    lost_reasons = lost["reason"].value_counts().rename("af_patients").reset_index()
    lost_reasons.columns = ["reason", "af_patients"]

    summary = pd.DataFrame(
        [
            ("series in the inventory", len(table)),
            ("rejected series", len(dropped)),
            ("candidate CCTA series", len(kept)),
            ("patients with a usable CCTA series", len(selected)),
            ("of those, present in the clinical CSV", len(merged)),
            ("AF patients in the clinical CSV", int((clinical["af"] == "YES").sum())),
            ("AF patients with a usable series", len(af_table)),
            ("AF patients without a usable series", len(lost)),
            ("non-AF patients with a usable series", len(control_table)),
            ("AF patients successfully matched 1:1", int(pairs["matched"].sum())),
            ("AF patients without a match", int((~pairs["matched"]).sum())),
            ("pilot cases to annotate", len(pilot)),
        ],
        columns=["item", "count"],
    )
    print()
    for item, count in summary.itertuples(index=False):
        print(f"{item:<40} {count:>8,}")

    image_columns = [column for column in IMAGE_COLUMNS if column in merged.columns]
    columns = image_columns + [
        column for column in CLINICAL_COLUMNS if column in merged.columns
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        sanitize(summary).to_excel(writer, sheet_name="Summary", index=False)
        sanitize(af_table[columns]).to_excel(
            writer, sheet_name="AF_Selected", index=False
        )
        controls = control_table[
            control_table["patient_id"].isin(matched_controls)
        ]
        sanitize(controls[columns]).to_excel(
            writer, sheet_name="Control_Selected", index=False
        )
        sanitize(pairs).to_excel(writer, sheet_name="Matched_Pairs", index=False)
        sanitize(pilot).to_excel(writer, sheet_name="Pilot_nnUNet", index=False)
        sanitize(lost).to_excel(writer, sheet_name="AF_Not_Usable", index=False)
        sanitize(lost_reasons).to_excel(
            writer, sheet_name="AF_Loss_Reasons", index=False
        )
        sanitize(reasons).to_excel(
            writer, sheet_name="Rejection_Reasons", index=False
        )
        sanitize(
            dropped[
                [column for column in image_columns if column in dropped.columns]
                + ["rejected_because"]
            ].head(200_000)
        ).to_excel(writer, sheet_name="Rejected", index=False)
    print(f"\nWrote {args.output}")
    print()
    print("Why AF patients have no usable series:")
    for reason, count in lost_reasons.itertuples(index=False):
        print(f"  {reason!s:<45} {count:>5}")
    print("\nNext: python 07_convert_cohort_to_nifti.py --sheet Pilot_nnUNet")


if __name__ == "__main__":
    main()
