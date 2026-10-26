"""Radiomic and volumetric features per compartment: LA, LA-EAT, whole-heart EAT.

The masks are taken as they are; no HU window is applied here, script 11 already
cleaned the fat masks and the atrial blood pool was removed. If no LA-EAT mask
folder is given, LA-EAT is derived as the fat sitting within --la-eat-mm of the
atrial surface.

Features come from PyRadiomics, so the definitions are the standard ones (IBSI).
Per compartment:

    first order   Mean, StandardDeviation, Skewness, Kurtosis, Entropy
    GLSZM         SmallAreaEmphasis, LargeAreaEmphasis, GreyLevelNonUniformity,
                  SizeZoneNonUniformity, ZonePercentage, ZoneEntropy
    volume        millilitres from the voxel count

Composite volumetric indices of whole-heart EAT (W), LA volume (L) and LA-EAT (E):

    cmi1 = W * L
    cmi2 = W**2 * L**0.1 * E**3

Script 15 re-optimises those exponents on cross-validated AUC; the two written
here are the published starting point.

Output: <output>\\radiomic_features.xlsx
            Features   one row per patient
            Group_Stats / Tests as in script 11

Usage
    python 14_radiomic_features.py
    python 14_radiomic_features.py --la-eat-masks "Q:\\...\\test80_seg_laeat"
    python 14_radiomic_features.py --bin-width 25 --la-eat-mm 10
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

BASE = Path(r"Q:\users\marfi\Automatic segmentations\finalllll\final")
ANALYSIS = BASE / "test80_analysis"
COMPARTMENTS = ("la", "laeat", "eat")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=BASE / "test80")
    parser.add_argument("--eat-masks", type=Path, default=ANALYSIS / "fat_hu_masks")
    parser.add_argument("--la-masks", type=Path, default=BASE / "test80_seg_la")
    parser.add_argument("--la-eat-masks", type=Path, default=None)
    parser.add_argument("--groups", type=Path, default=BASE / "test80" / "test80_groups.xlsx")
    parser.add_argument(
        "--clinical",
        type=Path,
        default=Path(r"Q:\users\marfi\CT_analysis\SCAPIS1-DATA-PETITION-1110-20260407(in).csv"),
    )
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument(
        "--bin-width",
        type=float,
        default=25.0,
        help="HU width of a grey level bin for the texture matrices",
    )
    parser.add_argument(
        "--la-eat-mm",
        type=float,
        default=10.0,
        help="distance from the atrial surface counted as LA-EAT when no mask is given",
    )
    parser.add_argument(
        "--exclude-la",
        action="store_true",
        help="subtract the atrial cavity from the EAT mask, for masks not cleaned yet",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="every PyRadiomics class including shape, GLCM, GLRLM, GLDM and NGTDM",
    )
    parser.add_argument(
        "--name",
        default="",
        help="workbook name, default radiomic_features.xlsx or _full with --full",
    )
    parser.add_argument("--la-dilate", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_known_args(argv)[0]


features_module = None


def shared_helpers():
    """Script 11 already has the folder index, the group reading and the tests."""
    global features_module
    if features_module is None:
        import importlib.util
        import sys

        name = "11_fat_hu_features_and_groups"
        if name in sys.modules:
            features_module = sys.modules[name]
            return features_module
        for folder in (Path(__file__).resolve().parent, Path.cwd()):
            path = folder / f"{name}.py"
            if path.exists():
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                features_module = module
                return module
        raise SystemExit(f"{name}.py not found next to this script or in {Path.cwd()}")
    return features_module


WANTED = {
    "firstorder_Mean": "mean",
    "firstorder_StandardDeviation": "sd",
    "firstorder_Skewness": "skewness",
    "firstorder_Kurtosis": "kurtosis",
    "firstorder_Entropy": "entropy",
    "firstorder_Variance": "variance",
    "firstorder_MeanAbsoluteDeviation": "mad",
    "firstorder_RobustMeanAbsoluteDeviation": "robust_mad",
    "firstorder_InterquartileRange": "iqr",
    "glszm_SmallAreaEmphasis": "small_area_emphasis",
    "glszm_LargeAreaEmphasis": "large_area_emphasis",
    "glszm_GrayLevelNonUniformity": "grey_level_non_uniformity",
    "glszm_SizeZoneNonUniformity": "zone_size_non_uniformity",
    "glszm_ZonePercentage": "zone_percentage",
    "glszm_ZoneEntropy": "zone_entropy",
}

extractor = None


def short_name(key: str) -> str:
    """original_shape_SurfaceVolumeRatio -> shape_surface_volume_ratio."""
    key = key.replace("original_", "")
    family, _, feature = key.partition("_")
    feature = feature.replace("2D", "2d").replace("3D", "3d")
    feature = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", feature)
    feature = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", feature)
    return f"{family}_{feature.lower()}"


def radiomics_extractor(bin_width: float, full: bool = False):
    """PyRadiomics, the small hand-picked set or every class with --full."""
    global extractor
    if extractor is None:
        import logging

        from radiomics.featureextractor import RadiomicsFeatureExtractor

        logging.getLogger("radiomics").setLevel(logging.ERROR)
        extractor = RadiomicsFeatureExtractor(binWidth=bin_width, label=1)
        if full:
            extractor.enableAllFeatures()  # shape, firstorder, glcm, glrlm, glszm, gldm, ngtdm
            return extractor
        extractor.disableAllFeatures()
        extractor.enableFeaturesByName(
            firstorder=[
                "Mean",
                "StandardDeviation",
                "Skewness",
                "Kurtosis",
                "Entropy",
                "Variance",
                "MeanAbsoluteDeviation",
                "RobustMeanAbsoluteDeviation",
                "InterquartileRange",
            ],
            glszm=[
                "SmallAreaEmphasis",
                "LargeAreaEmphasis",
                "GrayLevelNonUniformity",
                "SizeZoneNonUniformity",
                "ZonePercentage",
                "ZoneEntropy",
            ],
        )
    return extractor


def compartment_features(
    image: sitk.Image, mask: np.ndarray, bin_width: float, full: bool = False
) -> dict[str, float]:
    """Run PyRadiomics on one binary mask and keep the wanted features."""
    if int(mask.sum()) < 10:
        return {} if full else {name: float("nan") for name in WANTED.values()}
    label_image = sitk.GetImageFromArray(mask.astype(np.uint8))
    label_image.CopyInformation(image)
    result = radiomics_extractor(bin_width, full).execute(image, label_image)
    features: dict[str, float] = {}
    if full:
        for key, value in result.items():
            if not str(key).startswith("original_"):
                continue  # the diagnostics entries are not numbers
            try:
                features[short_name(str(key))] = round(float(value), 4)
            except (TypeError, ValueError):
                continue
        return features
    for key, name in WANTED.items():
        value = result.get(f"original_{key}")
        features[name] = round(float(value), 4) if value is not None else float("nan")
    return features


def derive_la_eat(fat: np.ndarray, atrium: np.ndarray, spacing: tuple, millimetres: float):
    """Fat within <millimetres> of the atrial surface, when no LA-EAT mask exists."""
    from scipy.ndimage import distance_transform_edt

    if not atrium.any():
        return np.zeros_like(fat)
    sampling = (spacing[2], spacing[1], spacing[0])  # array order is z, y, x
    distance = distance_transform_edt(~atrium, sampling=sampling)
    return fat & (distance <= millimetres)


def measure_one(
    image_path: Path,
    masks: dict[str, Path | None],
    args: argparse.Namespace,
) -> dict[str, object]:
    helpers = shared_helpers()
    volume, image = helpers.read_array(image_path)
    millilitre = helpers.voxel_ml(image)
    spacing = image.GetSpacing()
    image = sitk.Cast(image, sitk.sitkFloat32)

    arrays: dict[str, np.ndarray] = {}
    for name in ("la", "eat"):
        path = masks.get(name)
        if path and path.exists():
            array, _ = helpers.read_array(path)
            arrays[name] = helpers.align(array, volume) > 0
    if args.exclude_la and "eat" in arrays and "la" in arrays and arrays["la"].any():
        cavity = helpers.blood_pool(arrays["la"], args.la_dilate)
        arrays["eat"] = arrays["eat"] & ~cavity

    if masks.get("laeat") and masks["laeat"].exists():
        array, _ = helpers.read_array(masks["laeat"])
        arrays["laeat"] = helpers.align(array, volume) > 0
    elif "eat" in arrays and "la" in arrays:
        arrays["laeat"] = derive_la_eat(arrays["eat"], arrays["la"], spacing, args.la_eat_mm)

    row: dict[str, object] = {
        "patient_id": helpers.patient_of(image_path.name),
        "voxel_ml": round(millilitre, 5),
    }
    for name in COMPARTMENTS:
        mask = arrays.get(name)
        if mask is None or not mask.any():
            continue
        row[f"{name}_volume_ml"] = round(float(mask.sum()) * millilitre, 2)
        for key, value in compartment_features(image, mask, args.bin_width, args.full).items():
            row[f"{name}_{key}"] = value

    whole = row.get("eat_volume_ml")
    atrium = row.get("la_volume_ml")
    near = row.get("laeat_volume_ml")
    if whole and atrium:
        row["cmi1"] = round(float(whole) * float(atrium), 1)
    if whole and atrium and near:
        row["cmi2"] = round(float(whole) ** 2 * float(atrium) ** 0.1 * float(near) ** 3, 1)
    if whole and near:
        row["laeat_fraction_of_eat"] = round(float(near) / float(whole), 3)
    return row


def main(argv=None) -> None:
    args = parse_args(argv)
    helpers = shared_helpers()
    images = helpers.index_folder(args.images)
    eat = helpers.index_folder(args.eat_masks)
    atria = helpers.index_folder(args.la_masks)
    near = helpers.index_folder(args.la_eat_masks) if args.la_eat_masks else {}
    shared = [patient for patient in images if patient in eat]
    if args.limit:
        shared = shared[: args.limit]
    print(
        f"images {len(images)}, EAT {len(eat)}, LA {len(atria)}, "
        f"LA-EAT {len(near) if near else 'derived from LA'}"
    )
    print(f"{len(shared)} patients with image and EAT mask")
    if not shared:
        raise SystemExit("no patient has both an image and an EAT mask, check the folders")

    rows = []
    for number, patient in enumerate(shared, start=1):
        try:
            row = measure_one(
                images[patient],
                {"eat": eat.get(patient), "la": atria.get(patient), "laeat": near.get(patient)},
                args,
            )
            row["status"] = "ok"
            row["error"] = ""
        except Exception as problem:  # noqa: BLE001 - one bad scan must not stop the cohort
            row = {
                "patient_id": patient,
                "status": "failed",
                "error": f"{type(problem).__name__}: {problem}",
            }
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"[{number}/{len(shared)}] {patient}: ok "
                f"EAT {row.get('eat_volume_ml', '')} ml, LA {row.get('la_volume_ml', '')} ml, "
                f"LA-EAT {row.get('laeat_volume_ml', '')} ml"
            )
        else:
            print(f"[{number}/{len(shared)}] {patient}: failed, {row['error']}")
            if number >= 3 and all(item["status"] == "failed" for item in rows):
                raise SystemExit(
                    "the first cases all failed with the same error, fix that before "
                    "running the whole cohort"
                )

    table = pd.DataFrame(rows)
    groups = helpers.read_groups(args, table["patient_id"].tolist())
    table = table.merge(groups, on="patient_id", how="left")
    table["group"] = table["group"].fillna("unknown")

    numeric = [
        name
        for name in table.columns
        if name not in ("patient_id", "status", "error", "group", "voxel_ml")
    ]
    helpers.FEATURES = numeric
    good = table[table["status"] == "ok"]
    stats = helpers.group_stats(good)
    tests = helpers.tests_against_control(good)
    if not args.no_plots:
        helpers.draw_plots(good, args.output / "plots_radiomic")

    args.output.mkdir(parents=True, exist_ok=True)
    name = args.name or (
        "radiomic_features_full.xlsx" if args.full else "radiomic_features.xlsx"
    )
    path = args.output / name
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Features", index=False)
        if not stats.empty:
            stats.to_excel(writer, sheet_name="Group_Stats", index=False)
        if not tests.empty:
            tests.to_excel(writer, sheet_name="Tests", index=False)

    print()
    print(table["status"].value_counts().to_string())
    print(table["group"].value_counts().to_string())
    print(f"{len(numeric)} features per patient")
    if not tests.empty:
        print()
        print(tests.head(15).to_string(index=False))
    print(f"\nExcel : {path}")
    if not args.no_plots:
        print(f"Plots : {args.output / 'plots_radiomic'}")


if __name__ == "__main__":
    main()
