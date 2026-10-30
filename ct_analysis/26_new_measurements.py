"""New image measurements that are not another version of LA volume.

Everything here is computed from the image, the LA mask and the cleaned fat mask,
and is chosen to carry information the volumes and the PyRadiomics features do not:

    shells          fat volume, mean HU and fraction in 0-3, 3-5, 5-10 and 10-20 mm
                    bands around the atrial surface, so where the fat sits, not only
                    how much there is
    thickness       distance transform inside the fat touching the atrium: mean, p90
                    and maximum local fat thickness in millimetres
    histogram       fraction of fat voxels in -190..-150, -150..-110, -110..-70 and
                    -70..-30 HU, the shape of the attenuation distribution
    body size       thorax width, depth and cross-section area from a body mask at
                    -300 HU, and every volume divided by it, so a big atrium in a big
                    person is separated from a big atrium in a small one
    form            LA bounding box lengths, their ratios, surface per volume and the
                    fat area touching the atrium per LA surface area

Output: <output>\\new_measurements.xlsx, sheet Features, one row per patient.
It merges on patient_id, so script 25 takes it as a second measurement workbook.

Usage
    python 26_new_measurements.py --images ... --eat-masks ... --la-masks ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

BASE = Path(r"Q:\users\marfi\Automatic segmentations\finalllll\final")
ANALYSIS = BASE / "test80_analysis"
SHELLS = ((0, 3), (3, 5), (5, 10), (10, 20))
HU_BINS = ((-190, -150), (-150, -110), (-110, -70), (-70, -30))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=BASE / "test80")
    parser.add_argument("--eat-masks", type=Path, default=ANALYSIS / "fat_hu_masks")
    parser.add_argument("--la-masks", type=Path, default=BASE / "test80_seg_la")
    parser.add_argument("--groups", type=Path, default=BASE / "test80" / "test80_groups.xlsx")
    parser.add_argument(
        "--clinical",
        type=Path,
        default=Path(r"Q:\users\marfi\CT_analysis\SCAPIS1-DATA-PETITION-1110-20260407(in).csv"),
    )
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument("--body-hu", type=float, default=-300.0, help="body mask threshold")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="start from scratch, by default the partial csv is read and extended",
    )
    return parser.parse_known_args(argv)[0]


features_module = None


def shared_helpers():
    """Script 11 has the folder index, the group reading and the tests."""
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


def sampling_of(image: sitk.Image) -> tuple[float, float, float]:
    """Array order is z, y, x while SimpleITK spacing is x, y, z."""
    spacing = image.GetSpacing()
    return (spacing[2], spacing[1], spacing[0])


def shell_features(
    volume: np.ndarray,
    fat: np.ndarray,
    atrium: np.ndarray,
    sampling: tuple[float, float, float],
    millilitre: float,
) -> dict[str, float]:
    """How much fat sits in each distance band around the atrium, and how dense."""
    from scipy.ndimage import distance_transform_edt

    if not atrium.any() or not fat.any():
        return {}
    distance = distance_transform_edt(~atrium, sampling=sampling)
    total = float(fat.sum())
    row: dict[str, float] = {}
    for near, far in SHELLS:
        band = fat & (distance > near) & (distance <= far)
        count = float(band.sum())
        tag = f"shell_{near}_{far}mm"
        row[f"{tag}_volume_ml"] = round(count * millilitre, 3)
        row[f"{tag}_fraction"] = round(count / total, 4) if total else float("nan")
        row[f"{tag}_mean_hu"] = round(float(volume[band].mean()), 2) if count else float("nan")
    inner = fat & (distance <= 10)
    if inner.any():
        row["shell_0_10mm_hu_sd"] = round(float(volume[inner].std()), 2)
    return row


def thickness_features(
    fat: np.ndarray,
    atrium: np.ndarray,
    sampling: tuple[float, float, float],
) -> dict[str, float]:
    """Local fat thickness in millimetres, from the distance transform inside the fat."""
    from scipy.ndimage import distance_transform_edt, label

    if not fat.any() or not atrium.any():
        return {}
    touching = distance_transform_edt(~atrium, sampling=sampling) <= 10
    pieces, count = label(fat & touching)
    if count == 0:
        return {}
    inside = distance_transform_edt(fat, sampling=sampling)[pieces > 0]
    if inside.size == 0:
        return {}
    radius = inside * 2  # a distance to the edge is half a local thickness
    return {
        "fat_thickness_mean_mm": round(float(radius.mean()), 3),
        "fat_thickness_p90_mm": round(float(np.percentile(radius, 90)), 3),
        "fat_thickness_max_mm": round(float(radius.max()), 3),
        "fat_pieces_near_la": int(count),
    }


def histogram_features(volume: np.ndarray, fat: np.ndarray) -> dict[str, float]:
    """Shape of the fat attenuation distribution, not only its mean."""
    values = volume[fat]
    if values.size < 50:
        return {}
    row = {}
    for low, high in HU_BINS:
        share = float(((values >= low) & (values < high)).mean())
        row[f"fat_hu_{abs(low)}_{abs(high)}_fraction"] = round(share, 4)
    row["fat_hu_p10"] = round(float(np.percentile(values, 10)), 2)
    row["fat_hu_p90"] = round(float(np.percentile(values, 90)), 2)
    row["fat_hu_skew"] = round(
        float(((values - values.mean()) ** 3).mean() / (values.std() ** 3 + 1e-9)), 4
    )
    return row


def body_features(
    volume: np.ndarray, image: sitk.Image, threshold: float, millilitre: float
) -> dict[str, float]:
    """Thorax width, depth and area, so the volumes can be made size independent."""
    from scipy.ndimage import binary_fill_holes

    body = volume > threshold
    if not body.any():
        return {}
    middle = body[body.shape[0] // 2]
    middle = binary_fill_holes(middle)
    if not middle.any():
        return {}
    spacing = image.GetSpacing()
    rows, columns = np.where(middle)
    width = float(np.ptp(columns) + 1) * spacing[0]
    depth = float(np.ptp(rows) + 1) * spacing[1]
    area = float(middle.sum()) * spacing[0] * spacing[1] / 100.0  # cm2
    return {
        "body_width_mm": round(width, 1),
        "body_depth_mm": round(depth, 1),
        "body_area_cm2": round(area, 1),
        "body_volume_ml": round(float(body.sum()) * millilitre, 1),
    }


def form_features(
    atrium: np.ndarray, fat: np.ndarray, image: sitk.Image, millilitre: float
) -> dict[str, float]:
    """Bounding box form of the atrium and how much fat touches its surface."""
    from scipy.ndimage import binary_dilation

    if not atrium.any():
        return {}
    spacing = image.GetSpacing()
    where = np.where(atrium)
    lengths = sorted(
        [
            float(np.ptp(where[0]) + 1) * spacing[2],
            float(np.ptp(where[1]) + 1) * spacing[1],
            float(np.ptp(where[2]) + 1) * spacing[0],
        ]
    )
    small, middle, large = lengths
    box = large * middle * small
    atrial_ml = float(atrium.sum()) * millilitre
    row = {
        "la_box_short_mm": round(small, 1),
        "la_box_middle_mm": round(middle, 1),
        "la_box_long_mm": round(large, 1),
        "la_box_elongation": round(small / large, 4) if large else float("nan"),
        "la_box_flatness": round(middle / large, 4) if large else float("nan"),
        "la_box_fill": round(atrial_ml * 1000.0 / box, 4) if box else float("nan"),
    }
    surface = binary_dilation(atrium) & ~atrium
    area = float(surface.sum()) * spacing[0] * spacing[1]
    touching = float((surface & fat).sum())
    if area:
        row["la_surface_mm2"] = round(area, 1)
        row["fat_touching_la_fraction"] = round(touching / float(surface.sum()), 4)
        row["laeat_per_la_surface"] = round(atrial_ml * 1000.0 / area, 4)
    return row


def normalised(row: dict[str, float]) -> dict[str, float]:
    """Volumes per body size and the atrium against the fat around it."""
    out: dict[str, float] = {}
    area = row.get("body_area_cm2")
    width = row.get("body_width_mm")
    for name in ("la_volume_ml", "fat_volume_ml", "shell_0_3mm_volume_ml"):
        value = row.get(name)
        if value and area:
            out[f"{name}_per_body_area"] = round(float(value) / float(area), 5)
        if value and width:
            out[f"{name}_per_body_width"] = round(float(value) / float(width), 5)
    near = row.get("shell_0_5mm_volume_ml") or row.get("shell_0_3mm_volume_ml")
    atrial = row.get("la_volume_ml")
    if near and atrial:
        out["near_fat_per_la_volume"] = round(float(near) / float(atrial), 5)
    thickness = row.get("fat_thickness_mean_mm")
    if thickness and atrial:
        out["thickness_times_la"] = round(float(thickness) * float(atrial), 3)
    return out


def measure_one(image_path: Path, fat_path: Path, la_path: Path | None, args) -> dict:
    helpers = shared_helpers()
    volume, image = helpers.read_array(image_path)
    millilitre = helpers.voxel_ml(image)
    sampling = sampling_of(image)

    fat_array, _ = helpers.read_array(fat_path)
    fat = helpers.align(fat_array, volume) > 0
    atrium = np.zeros_like(fat)
    if la_path and la_path.exists():
        la_array, _ = helpers.read_array(la_path)
        atrium = helpers.align(la_array, volume) > 0

    row: dict[str, object] = {
        "patient_id": helpers.patient_of(image_path.name),
        "la_volume_ml": round(float(atrium.sum()) * millilitre, 2),
        "fat_volume_ml": round(float(fat.sum()) * millilitre, 2),
    }
    row.update(shell_features(volume, fat, atrium, sampling, millilitre))
    row.update(thickness_features(fat, atrium, sampling))
    row.update(histogram_features(volume, fat))
    row.update(body_features(volume, image, args.body_hu, millilitre))
    row.update(form_features(atrium, fat, image, millilitre))
    row.update(normalised(row))
    return row


def main(argv=None) -> None:
    args = parse_args(argv)
    helpers = shared_helpers()
    images = helpers.index_folder(args.images)
    fat = helpers.index_folder(args.eat_masks)
    atria = helpers.index_folder(args.la_masks)
    shared = [patient for patient in images if patient in fat]
    if args.limit:
        shared = shared[: args.limit]
    print(f"images {len(images)}, fat masks {len(fat)}, LA masks {len(atria)}")
    print(f"{len(shared)} patients with image and fat mask")

    args.output.mkdir(parents=True, exist_ok=True)
    partial = args.output / "new_measurements_partial.csv"
    rows: list[dict] = []
    if partial.exists() and not args.no_resume:
        rows = pd.read_csv(partial).to_dict("records")
        print(f"resuming, {len(rows)} already done")
    done = {str(row["patient_id"]) for row in rows}
    shared = [patient for patient in shared if str(patient) not in done]

    for number, patient in enumerate(shared, start=1):
        try:
            row = measure_one(images[patient], fat[patient], atria.get(patient), args)
            row["status"] = "ok"
            row["error"] = ""
            print(
                f"[{number}/{len(shared)}] {patient}: ok "
                f"LA {row.get('la_volume_ml')} ml, "
                f"thickness {row.get('fat_thickness_mean_mm')} mm, "
                f"body area {row.get('body_area_cm2')} cm2"
            )
        except Exception as problem:  # noqa: BLE001 - one bad scan must not stop the cohort
            row = {
                "patient_id": patient,
                "status": "failed",
                "error": f"{type(problem).__name__}: {problem}",
            }
            print(f"[{number}/{len(shared)}] {patient}: failed, {row['error']}")
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial, index=False)

    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("nothing was measured")
    groups = helpers.read_groups(args, table["patient_id"].tolist())
    table = table.merge(groups, on="patient_id", how="left")
    table["group"] = table["group"].fillna("unknown")

    numeric = [
        name
        for name in table.columns
        if name not in ("patient_id", "status", "error", "group")
        and pd.api.types.is_numeric_dtype(table[name])
    ]
    helpers.FEATURES = numeric
    good = table[table["status"] == "ok"]
    stats = helpers.group_stats(good)
    tests = helpers.tests_against_control(good)

    path = args.output / "new_measurements.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Features", index=False)
        if not stats.empty:
            stats.to_excel(writer, sheet_name="Group_Stats", index=False)
        if not tests.empty:
            tests.to_excel(writer, sheet_name="Tests", index=False)

    print()
    print(table["status"].value_counts().to_string())
    print(f"{len(numeric)} new measurements per patient")
    if not tests.empty:
        print()
        print(tests.head(15).to_string(index=False))
    print(f"\nExcel : {path}")


if __name__ == "__main__":
    main()
