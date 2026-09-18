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

With --layout per-patient the same volumes are written as one folder per patient,
named after the patient, which is what a manual segmentation session wants:
        <output>/VALLA_1234/VALLA_1234.nii.gz          the image to segment
        <output>/VALLA_1234/VALLA_1234_info.json       voxel spacing, dimensions, FOV,
                                                      kVp, kernel, phase, source folder
        <output>/VALLA_1234/                           save VALLA_1234_seg.nii.gz here
        <output>/nifti_index.xlsx                      one row per patient, with paths

Any sheet works as input: af_cohort.xlsx (Pilot_nnUNet, AF_Selected, All_Patients) or
SCAPIS_clinical_with_CT_status.xlsx (Patients). Only rows marked ok for segmentation
are converted, so `--sheet All_Patients` gives every usable patient in the cohort;
--include-not-ok also tries the rejected ones.

Usage
    python 07_convert_cohort_to_nifti.py --output "W:\\SCAPIS_nnUNet"
    python 07_convert_cohort_to_nifti.py --output "W:\\SCAPIS_nnUNet" --limit 4     REM try 4 cases first
    python 07_convert_cohort_to_nifti.py --cohort af_cohort.xlsx --sheet Pilot_nnUNet
    python 07_convert_cohort_to_nifti.py --layout per-patient --output "W:\\SCAPIS_seg"

In a Jupyter cell, pass the options as a list instead of relying on sys.argv:
    from importlib import import_module
    convert = import_module("07_convert_cohort_to_nifti")
    convert.main(["--output", r"W:\\SCAPIS_nnUNet", "--limit", "4"])
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import pydicom
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
        "--layout",
        choices=["nnunet", "per-patient"],
        default="nnunet",
        help="nnunet: imagesTr/SCAPIS_001_0000.nii.gz; "
        "per-patient: <patient_id>/<patient_id>.nii.gz",
    )
    parser.add_argument(
        "--copy-from",
        type=Path,
        default=None,
        help="folder holding already converted volumes "
        "(e.g. W:\\SCAPIS_nnUNet\\Dataset001_SCAPIS_LA\\imagesTr); "
        "a matching file is copied instead of read from DICOM again",
    )
    parser.add_argument(
        "--include-not-ok",
        action="store_true",
        help="also try rows the quality rules rejected "
        "(default: only rows marked ok for segmentation)",
    )
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


def deduplicate(files: list[str]) -> tuple[list[str], int]:
    """Drop repeated images, keep one file per slice position, sort along z.

    Overlapping archive batches export the same image twice; a series that still
    holds those repeats cannot be read as a volume until they are removed.
    """
    positions: dict[float, tuple[float, str]] = {}
    seen_instances: set[str] = set()
    unreadable = 0
    for name in files:
        try:
            header = pydicom.dcmread(name, stop_before_pixels=True, force=True)
            instance = str(header.get("SOPInstanceUID", ""))
            position = header.get("ImagePositionPatient")
            z = round(float(position[2]), 3) if position else float(len(positions))
        except (OSError, ValueError, TypeError, IndexError, KeyError):
            unreadable += 1
            continue
        if instance and instance in seen_instances:
            continue
        if instance:
            seen_instances.add(instance)
        positions.setdefault(z, (z, name))
    ordered = [name for _, name in sorted(positions.values())]
    return ordered, len(files) - len(ordered) - unreadable


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


def existing_volume(source: Path | None, case_id: str, patient: str) -> Path | None:
    """An already converted .nii.gz for this case, under any of the usual names.

    case_id is only trusted when the sheet supplied one: a number invented here
    would happily match another patient's file.
    """
    if source is None or not source.exists():
        return None
    names = [f"{patient}_0000.nii.gz", f"{patient}.nii.gz"]
    if case_id:
        names += [f"{case_id}_0000.nii.gz", f"{case_id}.nii.gz"]
    for name in names:
        for candidate in (source / name, source / patient / name):
            if candidate.exists():
                return candidate
    return None


def convert_one(
    folder: Path, series_uid: str, target: Path
) -> tuple[int, dict[str, object], str]:
    """Return (slice count, geometry, error). One series in, one .nii.gz out."""
    if not folder.exists():
        return 0, {}, f"folder not found: {folder}"
    files = series_files(folder, series_uid)
    if not files:
        return 0, {}, "SeriesInstanceUID not found in the folder"
    unique, removed = deduplicate(files)
    if len(unique) < 3:
        return 0, {}, f"only {len(unique)} usable slices after de-duplication"
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(unique)
    image = reader.Execute()
    target.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(target), useCompression=True)
    shape = geometry(image)
    shape["duplicate_slices_removed"] = removed
    del image, reader
    return len(unique), shape, ""


def usable_rows(cohort: pd.DataFrame, include_not_ok: bool) -> pd.DataFrame:
    """Keep the rows that carry a series and, unless asked, only the ok ones."""
    if "patient_id" not in cohort.columns:
        for name in ["Subject", "subject", "SubjectID"]:
            if name in cohort.columns:
                cohort = cohort.rename(columns={name: "patient_id"})
                break
    if not include_not_ok:
        for column in ["segmentation_ready", "usable"]:
            if column in cohort.columns:
                cohort = cohort[
                    cohort[column].astype(str).str.strip().str.lower() == "ok"
                ]
                break
    if "series_folder" in cohort.columns:
        cohort = cohort[cohort["series_folder"].notna()]
    return cohort.drop_duplicates(subset=["patient_id"], keep="first").reset_index(
        drop=True
    )


def write_patient_info(folder: Path, patient: str, row: dict[str, object]) -> Path:
    """Voxel spacing, dimensions and acquisition parameters next to the volume."""
    keys = [
        "group",
        "series_description",
        "slice_thickness_mm",
        "kvp",
        "convolution_kernel",
        "phase_percent",
        "heart_rate_bpm",
        "dim_x",
        "dim_y",
        "dim_z",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "fov_x_mm",
        "fov_y_mm",
        "coverage_z_mm",
        "origin",
        "direction",
        "hu_min",
        "hu_max",
        "slices_converted",
        "duplicate_slices_removed",
        "dicom_folder",
        "nifti_file",
    ]
    content = {"patient_id": patient}
    content.update(
        {key: row[key] for key in keys if key in row and pd.notna(row.get(key))}
    )
    path = folder / f"{patient}_info.json"
    path.write_text(json.dumps(content, indent=2, default=str))
    return path


def write_dataset_json(dataset_dir: Path, cases: int) -> None:
    content = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "left_atrium": 1},
        "numTraining": cases,
        "file_ending": ".nii.gz",
        "description": "SCAPIS CCTA, best-diastolic phase, left-atrium segmentation",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(content, indent=2))


def write_index(log: pd.DataFrame, output: Path) -> Path:
    """One row per patient, rewritten as the run proceeds so nothing is lost."""
    path = output / "nifti_index.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        log.to_excel(writer, sheet_name="Images", index=False)
        counts = (
            log.groupby("group")["status"].value_counts()
            if "group" in log.columns
            else log["status"].value_counts()
        )
        counts.rename("cases").reset_index().to_excel(
            writer, sheet_name="Summary", index=False
        )
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cohort = pd.read_excel(args.cohort, sheet_name=args.sheet, dtype=str)
    before = len(cohort)
    cohort = usable_rows(cohort, args.include_not_ok)
    print(f"{before} rows in sheet {args.sheet}, {len(cohort)} to convert")
    if args.limit:
        cohort = cohort.head(args.limit)

    per_patient = args.layout == "per-patient"
    dataset_dir = args.output / args.dataset
    images_dir = args.output if per_patient else dataset_dir / "imagesTr"
    labels_dir = args.output if per_patient else dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for position, record in enumerate(cohort.to_dict("records"), start=1):
        sheet_case_id = str(record.get("case_id") or "")
        case_id = sheet_case_id or f"SCAPIS_{position:03d}"
        patient = str(record["patient_id"])
        folder = Path(str(record["series_folder"]))
        if per_patient:
            target = images_dir / patient / f"{patient}.nii.gz"
            label = target.parent / f"{patient}_seg.nii.gz"
        else:
            target = images_dir / f"{case_id}_0000.nii.gz"
            label = labels_dir / f"{case_id}.nii.gz"
        print(
            f"[{position}/{len(cohort)}] {case_id} {record.get('group', '')} {patient}"
        )

        started = time.time()
        source_volume = existing_volume(args.copy_from, sheet_case_id, patient)
        try:
            if (
                source_volume is not None
                and source_volume != target
                and (args.overwrite or not target.exists())
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_volume, target)
                slices, shape, error = 0, geometry(sitk.ReadImage(str(target))), ""
                status = f"copied from {source_volume.name}"
            elif target.exists() and not args.overwrite:
                slices, shape, error = (
                    0,
                    geometry(sitk.ReadImage(str(target))),
                    "",
                )
                status = "already converted"
            else:
                slices, shape, error = convert_one(
                    folder, str(record.get("series_uid", "")), target
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
        row["patient_folder"] = str(target.parent)
        row["label_file_to_create"] = str(label)
        row["slices_converted"] = slices
        row["nifti_size_mb"] = (
            round(target.stat().st_size / 1024 / 1024, 1) if target.exists() else 0
        )
        row["status"] = status
        row["error"] = error
        row.update(shape)
        if per_patient and target.exists():
            row["info_file"] = str(write_patient_info(target.parent, patient, row))
        rows.append(row)
        if position % 25 == 0:
            write_index(pd.DataFrame(rows), args.output)

    log = pd.DataFrame(rows)
    log_path = write_index(log, args.output)
    if not per_patient:
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
    if per_patient:
        print("Labels : next to each image, as <patient_id>_seg.nii.gz")
    else:
        print(f"Labels : {labels_dir}  (put your segmentations here, same case_id)")
    print(f"Excel  : {log_path}")


if __name__ == "__main__":
    main()
