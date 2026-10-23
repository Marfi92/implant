"""Every AF, HF and stroke patient of the non-contrast cohort plus matched controls.

The pilot used 20 patients per group. This script takes all of them: every
patient of the 3 mm non-contrast cohort whose clinical record says AF, heart
failure or ischaemic stroke, and the same number of controls, matched on sex,
age and BMI.

A patient can carry more than one diagnosis, so two columns are written:

    group       one label for the group comparisons, by the priority AF, HF,
                stroke (--priority changes the order), control for the rest
    diagnoses   every label the patient carries, e.g. 'AF+HF', so the overlap
                is never hidden by the single group label

The Overlaps sheet counts each combination of the three diagnoses, which is the
table to report in the manuscript before any comparison is made.

Only patients that actually have a converted volume are used; patients with a
diagnosis but no volume are listed in Missing_Volume so nothing is silently lost.

Input : the clinical CSV
        the NIfTI folder of the non-contrast cohort (<patient_id>.nii.gz or
        <patient_id>_0000.nii.gz)
Output: <output>\\case_control_set.xlsx
            Summary          counts per step and per group
            Groups           patient_id, group, diagnoses, sex, age, bmi, source file
            Overlaps         how many patients carry each combination of diagnoses
            Matching         which control was matched to which case
            Missing_Volume   diagnosed patients without a converted volume
        <test folder>\\<patient_id>.nii.gz           copies of the selected scans
        <test folder>\\test80_groups.xlsx            patient_id and group, read by
                                                     scripts 11, 14 and 15

Usage
    python 18_build_case_control_set.py
    python 18_build_case_control_set.py --controls-per-case 2
    python 18_build_case_control_set.py --no-copy        only the workbook
    python 18_build_case_control_set.py --age-tolerance 3 --bmi-tolerance 2

In a Jupyter cell:
    from importlib import import_module
    m = import_module("18_build_case_control_set")
    m.main(["--images", r"W:\\SCAPIS_seg_lowres_3mm_noncontrast"])
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

BASE = Path(r"Q:\users\marfi\Automatic segmentations\finalllll\final")
IMAGES = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast")
CLINICAL = Path(r"Q:\users\marfi\CT_analysis\SCAPIS1-DATA-PETITION-1110-20260407(in).csv")
GROUP_COLUMNS = {
    "AF": "AF_CT_Baseline",
    "HF": "Prev_HF",
    "stroke": "Ischemic_Stroke_IC_Baseline",
}
EXTRA_COLUMNS = {"sex": "sex", "age": "age", "bmi": "bmi"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--clinical", type=Path, default=CLINICAL)
    parser.add_argument("--test-folder", type=Path, default=BASE / "test_all")
    parser.add_argument("--output", type=Path, default=BASE / "test_all_analysis")
    parser.add_argument(
        "--priority",
        default="AF,HF,stroke",
        help="which label wins when a patient carries several (default AF,HF,stroke)",
    )
    parser.add_argument("--controls-per-case", type=float, default=1.0)
    parser.add_argument("--age-tolerance", type=float, default=5.0)
    parser.add_argument("--bmi-tolerance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="write the workbook only, do not copy any volume (saves disk space)",
    )
    return parser.parse_args(argv)


def patient_of(name: str) -> str:
    """VALLA_1234_0000.nii.gz and VALLA_1234.nii.gz both mean VALLA_1234."""
    stem = name
    for suffix in (".nii.gz", ".nii", ".nrrd", ".mha"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.removesuffix("_0000")


def index_images(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise SystemExit(f"image folder not found: {folder}")
    found: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.name.lower().endswith((".nii", ".nii.gz", ".nrrd", ".mha")):
            found.setdefault(patient_of(path.name), path)
    return found


def read_clinical(path: Path) -> pd.DataFrame:
    """One row per patient with a yes/no column per diagnosis and sex, age, bmi."""
    if not path.exists():
        raise SystemExit(f"clinical file not found: {path}")
    raw = pd.read_csv(path, sep=None, engine="python", dtype=str)
    subject = "Subject" if "Subject" in raw.columns else next(iter(raw.columns))
    table = pd.DataFrame({"patient_id": raw[subject].astype(str).str.strip()})
    for label, column in GROUP_COLUMNS.items():
        if column not in raw.columns:
            raise SystemExit(f"column {column} missing from {path.name}")
        table[label] = raw[column].astype(str).str.strip().str.upper().eq("YES")
        table[label + "_missing"] = raw[column].isna() | raw[column].astype(
            str
        ).str.strip().eq("")
    for name, column in EXTRA_COLUMNS.items():
        match = [c for c in raw.columns if c.lower() == column]
        table[name] = (
            pd.to_numeric(raw[match[0]], errors="coerce")
            if match and name != "sex"
            else (raw[match[0]].astype(str).str.strip() if match else pd.NA)
        )
    return table.drop_duplicates("patient_id")


def label_rows(table: pd.DataFrame, priority: list[str]) -> pd.DataFrame:
    """Add the single group label and the full diagnoses string."""
    labels = [[name for name in priority if bool(row[name])] for _, row in table.iterrows()]
    table = table.copy()
    table["diagnoses"] = ["+".join(names) if names else "control" for names in labels]
    table["group"] = [names[0] if names else "control" for names in labels]
    table["n_diagnoses"] = [len(names) for names in labels]
    return table


def overlap_table(cases: pd.DataFrame, priority: list[str]) -> pd.DataFrame:
    """How many patients carry each combination of the three diagnoses."""
    rows = []
    for combination, part in cases.groupby("diagnoses"):
        rows.append({"diagnoses": combination, "patients": len(part)})
    for name in priority:
        rows.append({"diagnoses": f"any {name}", "patients": int(cases[name].sum())})
    for first in range(len(priority)):
        for second in range(first + 1, len(priority)):
            a, b = priority[first], priority[second]
            rows.append(
                {"diagnoses": f"{a} and {b}", "patients": int((cases[a] & cases[b]).sum())}
            )
    rows.append(
        {
            "diagnoses": " and ".join(priority),
            "patients": int(cases[priority].all(axis=1).sum()),
        }
    )
    rows.append({"diagnoses": "any diagnosis (total cases)", "patients": len(cases)})
    return pd.DataFrame(rows)


def match_controls(
    cases: pd.DataFrame, pool: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy nearest control per case: same sex, then closest age and BMI."""
    wanted = round(len(cases) * args.controls_per_case)
    available = pool.sample(frac=1.0, random_state=args.seed).to_dict("records")
    used: set[str] = set()
    taken: list[dict] = []
    rows = []
    for _, case in cases.iterrows():
        if len(taken) >= wanted:
            break
        best, best_cost = None, None
        for control in available:
            if control["patient_id"] in used:
                continue
            if (
                pd.notna(case["sex"])
                and pd.notna(control["sex"])
                and str(case["sex"]) != str(control["sex"])
            ):
                continue
            age_gap = abs(float(case["age"]) - float(control["age"])) if _both(case, control, "age") else 0.0
            bmi_gap = abs(float(case["bmi"]) - float(control["bmi"])) if _both(case, control, "bmi") else 0.0
            if age_gap > args.age_tolerance or bmi_gap > args.bmi_tolerance:
                continue
            cost = age_gap + bmi_gap
            if best_cost is None or cost < best_cost:
                best, best_cost = control, cost
        if best is None:  # no match inside the tolerance, take any unused control
            rest = [c for c in available if c["patient_id"] not in used]
            if not rest:
                break
            best, best_cost = rest[0], None
        taken.append(best)
        used.add(best["patient_id"])
        rows.append(
            {
                "case": case["patient_id"],
                "case_group": case["group"],
                "control": best["patient_id"],
                "sex": best["sex"],
                "age": best["age"],
                "bmi": best["bmi"],
                "distance": None if best_cost is None else round(best_cost, 2),
                "matched_within_tolerance": best_cost is not None,
            }
        )
    for control in available:  # controls-per-case above 1
        if len(taken) >= wanted:
            break
        if control["patient_id"] in used:
            continue
        taken.append(control)
        used.add(control["patient_id"])
        rows.append({"case": "", "case_group": "", "control": control["patient_id"]})
    controls = pd.DataFrame(taken)
    return controls, pd.DataFrame(rows)


def _both(case, control, field) -> bool:
    return pd.notna(case[field]) and pd.notna(control[field])


def copy_volumes(selected: pd.DataFrame, folder: Path) -> list[str]:
    """Copy one volume per selected patient into the test folder."""
    folder.mkdir(parents=True, exist_ok=True)
    failed = []
    for _, row in selected.iterrows():
        target = folder / (row["patient_id"] + ".nii.gz")
        if target.exists():
            continue
        try:
            shutil.copy2(row["source_file"], target)
        except OSError as error:  # out of space, locked file
            failed.append(f"{row['patient_id']}: {error}")
    return failed


def main(argv=None) -> None:
    args = parse_args(argv)
    priority = [name.strip() for name in args.priority.split(",") if name.strip()]
    images = index_images(args.images)
    clinical = label_rows(read_clinical(args.clinical), priority)
    print(f"{len(images)} volumes, {len(clinical)} clinical records")

    clinical["has_volume"] = clinical["patient_id"].isin(images)
    clinical["source_file"] = clinical["patient_id"].map(
        lambda pid: str(images.get(pid, ""))
    )
    diagnosed = clinical[clinical["group"] != "control"]
    cases = diagnosed[diagnosed["has_volume"]].sort_values("patient_id")
    missing = diagnosed[~diagnosed["has_volume"]][["patient_id", "group", "diagnoses"]]
    pool = clinical[
        (clinical["group"] == "control")
        & clinical["has_volume"]
        & ~clinical[[f"{name}_missing" for name in priority]].any(axis=1)
    ]
    print(f"{len(cases)} cases with a volume, {len(missing)} without, {len(pool)} possible controls")

    controls, matching = match_controls(cases, pool, args)
    selected = pd.concat([cases, controls], ignore_index=True)
    columns = [
        "patient_id",
        "group",
        "diagnoses",
        "n_diagnoses",
        "sex",
        "age",
        "bmi",
        "source_file",
    ]
    selected = selected[columns]

    failed = [] if args.no_copy else copy_volumes(selected, args.test_folder)
    if not args.no_copy:
        args.test_folder.mkdir(parents=True, exist_ok=True)
        selected[["patient_id", "group", "diagnoses"]].to_excel(
            args.test_folder / "test80_groups.xlsx", index=False
        )

    summary = [{"step": "volumes in the image folder", "n": len(images)}]
    summary += [{"step": "clinical records", "n": len(clinical)}]
    for name in priority:
        summary.append({"step": f"{name} yes in the clinical file", "n": int(clinical[name].sum())})
        summary.append(
            {"step": f"{name} yes with a volume", "n": int(clinical[clinical.has_volume][name].sum())}
        )
    summary += [
        {"step": "cases selected (any diagnosis, with a volume)", "n": len(cases)},
        {"step": "diagnosed patients without a volume", "n": len(missing)},
        {"step": "controls matched", "n": len(controls)},
        {
            "step": "controls matched inside the tolerance",
            "n": int(matching.get("matched_within_tolerance", pd.Series(dtype=bool)).sum()),
        },
        {"step": "patients in the test folder", "n": len(selected)},
        {"step": "copies that failed", "n": len(failed)},
    ]
    for group, part in selected.groupby("group"):
        summary.append({"step": f"group {group}", "n": len(part)})

    args.output.mkdir(parents=True, exist_ok=True)
    workbook = args.output / "case_control_set.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)
        selected.to_excel(writer, sheet_name="Groups", index=False)
        overlap_table(cases, priority).to_excel(writer, sheet_name="Overlaps", index=False)
        matching.to_excel(writer, sheet_name="Matching", index=False)
        missing.to_excel(writer, sheet_name="Missing_Volume", index=False)
        if failed:
            pd.DataFrame({"error": failed}).to_excel(
                writer, sheet_name="Copy_Errors", index=False
            )
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nOverlaps\n", overlap_table(cases, priority).to_string(index=False))
    print("\nwritten", workbook)
    if not args.no_copy:
        print("volumes in", args.test_folder)


if __name__ == "__main__":
    main()
