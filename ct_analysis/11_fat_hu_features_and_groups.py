"""Clean the predicted fat masks with a HU window, extract features, compare the groups.

The nnU-Net fat model marks the whole pericardial fat compartment, so the mask
still holds voxels that are not fat. Only voxels of the CT image between
--fat-low and --fat-high HU (-190 to -30 by default) are kept, the cleaned mask
is written to its own folder, and the measurements are taken from that mask.

After the HU window, voxels that lie inside the left atrium are dropped, because
non-contrast blood and the HU window overlap and the atrial cavity otherwise
leaks into the fat mask. The LA mask is hole filled (in 3D and slice by slice)
and grown by --la-dilate voxels first, so the atrial wall goes with it. One
voxel is the default: epicardial fat sits directly on the atrial wall, so a
larger dilation starts eating real fat (--la-dilate 0 subtracts the cavity
only). Use --keep-la-inside to switch the subtraction off.

Per patient 10 features are measured:

    fat_volume_ml          cleaned fat volume
    fat_volume_raw_ml      volume of the model mask before the HU window
    fat_kept_fraction      cleaned / raw, how much of the mask was real fat
    fat_mean_hu            mean HU inside the cleaned fat
    fat_sd_hu              HU spread, a fibrotic/inflamed fat marker
    fat_p25_hu             lower HU quartile
    fat_p75_hu             upper HU quartile
    la_volume_ml           left atrium volume
    fat_in_la_removed_ml   fat volume that fell inside the atrium and was dropped
    la_mean_hu             mean HU inside the atrium (contrast/blood signal)
    fat_to_la_ratio        fat volume / atrium volume
    fat_index_ml_per_m     fat volume per cm of covered heart length (extra)

Then every group (AF, HF, stroke) is compared with the controls of the same
sheet: median per group, Mann-Whitney p value, and a box plot per feature.

Input : the image folder with <patient_id>_0000.nii.gz (or <patient_id>.nii.gz)
        the fat masks   from nnUNetv2_predict
        the LA masks    from nnUNetv2_predict   (optional, --la-masks)
        test80_groups.xlsx with patient_id and group, or the clinical CSV
Output: <output>\\fat_hu_masks\\<patient_id>.nii.gz   cleaned masks
        <output>\\fat_la_features.xlsx
            Features     one row per patient, the 10 features plus the group
            Group_Stats  median and IQR per group and feature
            Tests        Mann-Whitney of each group against the controls
        <output>\\plots\\<feature>.png

Usage
    python 11_fat_hu_features_and_groups.py
    python 11_fat_hu_features_and_groups.py --fat-low -190 --fat-high -30
    python 11_fat_hu_features_and_groups.py --la-dilate 3
    python 11_fat_hu_features_and_groups.py --keep-la-inside
    python 11_fat_hu_features_and_groups.py --no-plots

In a Jupyter cell:
    from importlib import import_module
    m = import_module("11_fat_hu_features_and_groups")
    m.main(["--images", r"Q:\\...\\test80", "--fat-masks", r"Q:\\...\\test80_seg_fat"])
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

BASE = Path(r"Q:\users\marfi\Automatic segmentations\finalllll\final")
IMAGES = BASE / "test80"
FAT_MASKS = BASE / "test80_seg_fat"
LA_MASKS = BASE / "test80_seg_la"
GROUPS = BASE / "test80" / "test80_groups.xlsx"
OUTPUT = BASE / "test80_analysis"
CLINICAL = Path(r"Q:\users\marfi\CT_analysis\SCAPIS1-DATA-PETITION-1110-20260407(in).csv")

GROUP_COLUMNS = {
    "AF": "AF_CT_Baseline",
    "stroke": "Ischemic_Stroke_IC_Baseline",
    "HF": "Prev_HF",
}
FEATURES = [
    "fat_volume_ml",
    "fat_volume_raw_ml",
    "fat_kept_fraction",
    "fat_mean_hu",
    "fat_sd_hu",
    "fat_p25_hu",
    "fat_p75_hu",
    "la_volume_ml",
    "la_mean_hu",
    "fat_to_la_ratio",
    "fat_index_ml_per_cm",
    "fat_in_la_removed_ml",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--fat-masks", type=Path, default=FAT_MASKS)
    parser.add_argument("--la-masks", type=Path, default=LA_MASKS)
    parser.add_argument("--groups", type=Path, default=GROUPS)
    parser.add_argument("--clinical", type=Path, default=CLINICAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fat-low", type=float, default=-190.0)
    parser.add_argument("--fat-high", type=float, default=-30.0)
    parser.add_argument(
        "--keep-la-inside",
        action="store_true",
        help="keep voxels inside the left atrium instead of dropping them",
    )
    parser.add_argument(
        "--la-dilate",
        type=int,
        default=1,
        help="grow the atrium mask by this many voxels before subtracting it",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_known_args(argv)[0]


def patient_of(name: str) -> str:
    """VALLA_1234_0000.nii.gz and VALLA_1234.nii.gz both mean VALLA_1234."""
    stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    return stem[:-5] if stem.endswith("_0000") else stem


def index_folder(folder: Path) -> dict[str, Path]:
    if not folder or not folder.exists():
        return {}
    return {patient_of(path.name): path for path in sorted(folder.glob("*.nii.gz"))}


def read_array(path: Path) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image), image


def voxel_ml(image: sitk.Image) -> float:
    spacing = image.GetSpacing()
    return float(spacing[0] * spacing[1] * spacing[2]) / 1000.0


def align(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Masks come back in the image grid; refuse a mismatch instead of guessing."""
    if mask.shape != reference.shape:
        raise ValueError(f"mask shape {mask.shape} does not match image {reference.shape}")
    return mask


def hu_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def blood_pool(atrium: np.ndarray, dilate: int) -> np.ndarray:
    """The atrial cavity: holes filled in 3D and per slice, then grown to take the wall."""
    from scipy.ndimage import binary_dilation, binary_fill_holes

    filled = binary_fill_holes(atrium)
    for index in range(filled.shape[0]):
        filled[index] = binary_fill_holes(filled[index])
    if dilate > 0:
        filled = binary_dilation(filled, iterations=dilate)
    return filled


def measure_one(
    image_path: Path,
    fat_path: Path,
    la_path: Path | None,
    low: float,
    high: float,
    cleaned_target: Path,
    exclude_la: bool = True,
    la_dilate: int = 1,
) -> dict[str, object]:
    """HU window on the fat mask, write the cleaned mask, return the features."""
    volume, image = read_array(image_path)
    fat_raw, fat_image = read_array(fat_path)
    fat_raw = align(fat_raw, volume) > 0

    inside_window = (volume >= low) & (volume <= high)
    fat = fat_raw & inside_window

    atrium = None
    removed_voxels = 0
    if la_path and la_path.exists():
        atrium, _ = read_array(la_path)
        atrium = align(atrium, volume) > 0
        if exclude_la and atrium.any():
            cavity = blood_pool(atrium, la_dilate)
            removed_voxels = int((fat & cavity).sum())
            fat = fat & ~cavity

    cleaned = sitk.GetImageFromArray(fat.astype(np.uint8))
    cleaned.CopyInformation(fat_image)
    cleaned_target.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(cleaned, str(cleaned_target), True)

    millilitre = voxel_ml(image)
    fat_ml = float(fat.sum()) * millilitre
    raw_ml = float(fat_raw.sum()) * millilitre
    fat_hu = hu_stats(volume[fat])

    row: dict[str, object] = {
        "patient_id": patient_of(image_path.name),
        "fat_volume_ml": round(fat_ml, 2),
        "fat_volume_raw_ml": round(raw_ml, 2),
        "fat_kept_fraction": round(fat_ml / raw_ml, 3) if raw_ml else float("nan"),
        "fat_mean_hu": round(fat_hu["mean"], 1),
        "fat_sd_hu": round(fat_hu["sd"], 1),
        "fat_p25_hu": round(fat_hu["p25"], 1),
        "fat_p75_hu": round(fat_hu["p75"], 1),
        "la_volume_ml": float("nan"),
        "la_mean_hu": float("nan"),
        "fat_to_la_ratio": float("nan"),
        "fat_in_la_removed_ml": round(removed_voxels * millilitre, 2),
        "voxel_ml": round(millilitre, 5),
        "slice_mm": round(float(image.GetSpacing()[2]), 2),
        "cleaned_mask": str(cleaned_target),
    }

    if fat.any():
        covered = np.where(fat.any(axis=(1, 2)))[0]
        length_cm = (covered[-1] - covered[0] + 1) * float(image.GetSpacing()[2]) / 10.0
        row["fat_index_ml_per_cm"] = round(fat_ml / length_cm, 2) if length_cm else float("nan")
    else:
        row["fat_index_ml_per_cm"] = float("nan")

    if atrium is not None:
        la_ml = float(atrium.sum()) * millilitre
        row["la_volume_ml"] = round(la_ml, 2)
        row["la_mean_hu"] = round(hu_stats(volume[atrium])["mean"], 1)
        row["fat_to_la_ratio"] = round(fat_ml / la_ml, 3) if la_ml else float("nan")
    return row


def read_groups(args: argparse.Namespace, patients: list[str]) -> pd.DataFrame:
    """The group of each patient, from the test sheet or from the clinical CSV."""
    if args.groups and args.groups.exists():
        sheet = pd.read_excel(args.groups, dtype=str)
        if {"patient_id", "group"} <= set(sheet.columns):
            return sheet[["patient_id", "group"]].drop_duplicates("patient_id")
    if not args.clinical.exists():
        return pd.DataFrame({"patient_id": patients, "group": "unknown"})
    clinical = pd.read_csv(args.clinical, sep=None, engine="python", dtype=str)
    clinical = clinical[clinical["Subject"].astype(str).isin(patients)]
    rows = []
    for patient, record in zip(clinical["Subject"].astype(str), clinical.to_dict("records")):
        labels = [
            name
            for name, column in GROUP_COLUMNS.items()
            if str(record.get(column, "")).strip().upper() == "YES"
        ]
        rows.append({"patient_id": patient, "group": labels[0] if labels else "control"})
    return pd.DataFrame(rows)


def group_stats(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        if feature not in table:
            continue
        for group, part in table.groupby("group"):
            values = pd.to_numeric(part[feature], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "feature": feature,
                    "group": group,
                    "n": len(values),
                    "median": round(values.median(), 3),
                    "q1": round(values.quantile(0.25), 3),
                    "q3": round(values.quantile(0.75), 3),
                    "mean": round(values.mean(), 3),
                    "sd": round(values.std(), 3),
                }
            )
    return pd.DataFrame(rows)


def tests_against_control(table: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney of every group against the controls, per feature."""
    from scipy.stats import mannwhitneyu

    controls = table[table["group"] == "control"]
    rows = []
    for feature in FEATURES:
        if feature not in table:
            continue
        reference = pd.to_numeric(controls[feature], errors="coerce").dropna()
        for group, part in table.groupby("group"):
            if group == "control":
                continue
            values = pd.to_numeric(part[feature], errors="coerce").dropna()
            if len(values) < 3 or len(reference) < 3:
                continue
            statistic, p = mannwhitneyu(values, reference, alternative="two-sided")
            rows.append(
                {
                    "feature": feature,
                    "group": group,
                    "n_group": len(values),
                    "n_control": len(reference),
                    "median_group": round(values.median(), 3),
                    "median_control": round(reference.median(), 3),
                    "u": float(statistic),
                    "p_value": round(float(p), 5),
                    "significant_0.05": "yes" if p < 0.05 else "no",
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sort_values("p_value") if not frame.empty else frame


def draw_plots(table: pd.DataFrame, folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    order = [name for name in ["control", "AF", "HF", "stroke"] if name in set(table["group"])]
    order += [name for name in sorted(set(table["group"])) if name not in order]
    for feature in FEATURES:
        if feature not in table:
            continue
        data = [
            pd.to_numeric(table.loc[table["group"] == group, feature], errors="coerce")
            .dropna()
            .to_numpy()
            for group in order
        ]
        if not any(len(values) for values in data):
            continue
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.boxplot(data, showmeans=True)
        axis.set_xticks(range(1, len(order) + 1), order)
        axis.set_title(feature.replace("_", " "))
        axis.set_ylabel(feature)
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(folder / f"{feature}.png", dpi=150)
        plt.close(figure)


def main(argv=None) -> None:
    args = parse_args(argv)
    images = index_folder(args.images)
    fat = index_folder(args.fat_masks)
    atria = index_folder(args.la_masks)
    shared = [patient for patient in images if patient in fat]
    if args.limit:
        shared = shared[: args.limit]
    print(f"images {len(images)}, fat masks {len(fat)}, LA masks {len(atria)}")
    print(f"{len(shared)} patients with image and fat mask")
    if not shared:
        raise SystemExit("no patient has both an image and a fat mask, check the folders")

    cleaned_folder = args.output / "fat_hu_masks"
    rows = []
    for number, patient in enumerate(shared, start=1):
        target = cleaned_folder / f"{patient}.nii.gz"
        try:
            row = measure_one(
                images[patient],
                fat[patient],
                atria.get(patient),
                args.fat_low,
                args.fat_high,
                target,
                exclude_la=not args.keep_la_inside,
                la_dilate=args.la_dilate,
            )
            row["status"] = "ok"
            row["error"] = ""
        except Exception as problem:
            row = {
                "patient_id": patient,
                "status": "failed",
                "error": f"{type(problem).__name__}: {problem}",
            }
        rows.append(row)
        volume = row.get("fat_volume_ml", "")
        print(f"[{number}/{len(shared)}] {patient}: {row['status']} fat {volume} ml")

    table = pd.DataFrame(rows)
    groups = read_groups(args, table["patient_id"].tolist())
    table = table.merge(groups, on="patient_id", how="left")
    table["group"] = table["group"].fillna("unknown")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "fat_la_features.xlsx"
    stats = group_stats(table[table["status"] == "ok"])
    tests = tests_against_control(table[table["status"] == "ok"])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Features", index=False)
        if not stats.empty:
            stats.to_excel(writer, sheet_name="Group_Stats", index=False)
        if not tests.empty:
            tests.to_excel(writer, sheet_name="Tests", index=False)
    if not args.no_plots:
        draw_plots(table[table["status"] == "ok"], args.output / "plots")

    print()
    print(table["status"].value_counts().to_string())
    print(table["group"].value_counts().to_string())
    if not tests.empty:
        print()
        print(tests.head(10).to_string(index=False))
    print(f"\nCleaned masks : {cleaned_folder}")
    print(f"Excel         : {path}")
    if not args.no_plots:
        print(f"Plots         : {args.output / 'plots'}")


if __name__ == "__main__":
    main()
