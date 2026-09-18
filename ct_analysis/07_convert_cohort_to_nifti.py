"""Convert the selected DICOM series to NIfTI in nnU-Net layout, one series at a time.

Input : af_cohort.xlsx (sheet Pilot_nnUNet) written by 06_build_af_cohort.py
Output: <output>/Dataset001_SCAPIS_LA/imagesTr/SCAPIS_001_0000.nii.gz ...
        <output>/Dataset001_SCAPIS_LA/labelsTr/          (empty - your annotations go here)
        <output>/nifti_index.xlsx                        one row per patient, sheet Images:
                AF / control, patient_id, site, sex, age, BMI, kVp, kernel, phase,
                heart rate, the DICOM folder on Q:, the NIfTI path, and the geometry
                read back out of the written NIfTI (dimensions, voxel spacing,
                origin, direction, HU range) so you can verify every case in Excel.

Nothing is copied or extracted: each series is read straight from its source folder,
written as one .nii.gz, and the reader is released before the next series starts.

Usage
    python 07_convert_cohort_to_nifti.py --output "W:\\SCAPIS_nnUNet"
    python 07_convert_cohort_to_nifti.py --output "W:\\SCAPIS_nnUNet" --limit 4     REM try 4 cases first
    python 07_convert_cohort_to_nifti.py --cohort af_cohort.xlsx --sheet Pilot_nnUNet

In a Jupyter cell, pass the options as a list instead of relying on sys.argv:
    from importlib import import_module
    convert = import_module("07_convert_cohort_to_nifti")
    convert.main(["--output", r"W:\SCAPIS_nnUNet", "--limit", "4"])
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

COHORT = Path(r"Q:\users\marfi\CT_analysis\af_cohort.xlsx")
OUTPUT = Path(r"W:\SCAPIS_nnUNet")
DATASET = "Dataset001_SCAPIS_LA"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--sheet", default="Pilot_nnUNet")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument(
        "--limit", type=int, default=0, help="convert only the first N cases"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite files that already exist (default: skip, so the run resumes)",
    )
    # parse_known_args so the notebook kernel's own -f argument is ignored
    return parser.parse_known_args(argv)[0]


def series_files(folder: Path, series_uid: str) -> list[str]:
    """File names of exactly this SeriesInstanceUID, in slice order."""
    reader = sitk.ImageSeriesReader()
    identifiers = reader.GetGDCMSeriesIDs(str(folder))
    if series_uid and series_uid in identifiers:
        return list(reader.GetGDCMSeriesFileNames(str(folder), series_uid))
    if len(identifiers) == 1:
        return list(reader.GetGDCMSeriesFileNames(str(folder), identifiers[0]))
    return []


def geometry(image: sitk.Image) -> dict[str, object]:
    spacing = image.GetSpacing()
    size = image.GetSize()
    origin = image.GetOrigin()
    statistics = sitk.MinimumMaximumImageFilter()
    statistics.Execute(image)
    return {
        "dim_x": size[0],
        "dim_y": size[1],
        "dim_z": size[2],
        "spacing_x_mm": round(spacing[0], 4),
        "spacing_y_mm": round(spacing[1], 4),
        "spacing_z_mm": round(spacing[2], 4),
        "fov_x_mm": round(spacing[0] * size[0], 1),
        "fov_y_mm": round(spacing[1] * size[1], 1),
        "coverage_z_mm": round(spacing[2] * size[2], 1),
        "origin": " ".join(f"{value:.2f}" for value in origin),
        "direction": " ".join(f"{value:.0f}" for value in image.GetDirection()),
        "hu_min": int(statistics.GetMinimum()),
        "hu_max": int(statistics.GetMaximum()),
        "pixel_type": image.GetPixelIDTypeAsString(),
    }


def convert_one(
    folder: Path, series_uid: str, target: Path
) -> tuple[int, dict[str, object], str]:
    """Return (slice count, geometry, error). One series in, one .nii.gz out."""
    if not folder.exists():
        return 0, {}, f"folder not found: {folder}"
    files = series_files(folder, series_uid)
    if not files:
        return 0, {}, "SeriesInstanceUID not found in the folder"
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    target.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(target), useCompression=True)
    shape = geometry(image)
    del image, reader
    return len(files), shape, ""


def write_dataset_json(dataset_dir: Path, cases: int) -> None:
    content = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "left_atrium": 1},
        "numTraining": cases,
        "file_ending": ".nii.gz",
        "description": "SCAPIS CCTA, best-diastolic phase, left-atrium segmentation",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(content, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cohort = pd.read_excel(args.cohort, sheet_name=args.sheet, dtype=str)
    if args.limit:
        cohort = cohort.head(args.limit)

    dataset_dir = args.output / args.dataset
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for position, record in enumerate(cohort.to_dict("records"), start=1):
        case_id = str(record["case_id"])
        folder = Path(str(record["series_folder"]))
        target = images_dir / f"{case_id}_0000.nii.gz"
        print(
            f"[{position}/{len(cohort)}] {case_id} "
            f"{record['group']} {record['patient_id']}"
        )

        started = time.time()
        try:
            if target.exists() and not args.overwrite:
                slices, shape, error = (
                    0,
                    geometry(sitk.ReadImage(str(target))),
                    "",
                )
                status = "already converted"
            else:
                slices, shape, error = convert_one(
                    folder, str(record["series_uid"]), target
                )
                status = "ok" if not error else "failed"
        except Exception as problem:  # a single bad series must not stop the run
            slices, shape, error = 0, {}, f"{type(problem).__name__}: {problem}"
            status = "failed"
        print(f"    {status} ({slices} slices, {time.time() - started:.0f} s)")
        if error:
            print(f"    {error}")

        row: dict[str, object] = dict(record)
        row["dicom_folder"] = str(folder)
        row["nifti_file"] = str(target)
        row["label_file_to_create"] = str(labels_dir / f"{case_id}.nii.gz")
        row["slices_converted"] = slices
        row["nifti_size_mb"] = (
            round(target.stat().st_size / 1024 / 1024, 1) if target.exists() else 0
        )
        row["status"] = status
        row["error"] = error
        row.update(shape)
        rows.append(row)

    log = pd.DataFrame(rows)
    log_path = args.output / "nifti_index.xlsx"
    with pd.ExcelWriter(log_path, engine="openpyxl") as writer:
        log.to_excel(writer, sheet_name="Images", index=False)
        log.groupby("group")["status"].value_counts().rename("cases").reset_index(
        ).to_excel(writer, sheet_name="Summary", index=False)
    write_dataset_json(dataset_dir, int((log["status"] != "failed").sum()))

    overview = [
        column
        for column in [
            "case_id",
            "group",
            "patient_id",
            "dim_x",
            "dim_y",
            "dim_z",
            "spacing_x_mm",
            "spacing_z_mm",
            "nifti_size_mb",
            "status",
        ]
        if column in log.columns
    ]
    print()
    print(log[overview].to_string(index=False))
    print(f"\nImages : {images_dir}")
    print(f"Labels : {labels_dir}  (put your segmentations here, same case_id)")
    print(f"Excel  : {log_path}")


if __name__ == "__main__":
    main()
