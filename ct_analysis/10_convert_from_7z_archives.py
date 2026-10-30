"""Convert the series whose DICOM folders only exist inside the .7z archives.

Lee's unpacked copy on Q: holds the contrast (ccta) batches; the calcium scoring
(casc) batches were never unpacked, so 07_convert_cohort_to_nifti.py reports
"folder not found" for those patients. The packed batches are still on the
scapis share, one archive per batch:

    V:\\datahub\\CT-site-1-casc\\ct_site_1_casc_11.7z

This script works batch by batch: it extracts only the patient folders it needs
from one archive into a scratch folder, converts them with the same code as
script 07, writes the volume into the same output folder, and deletes the
extracted DICOM again before moving to the next archive. Peak extra disk use is
therefore one batch (a few GB), not the whole archive set.

Input : the cohort workbook (sheet Selected) and, if it exists, the
        nifti_index.xlsx of the earlier run, so only the rows that actually
        failed with "folder not found" are retried
Output: <output>/<patient_id>.nii.gz        same naming as before
        <output>/nifti_index_from_7z.xlsx   one row per retried patient

Usage
    python 10_convert_from_7z_archives.py ^
        --cohort "Q:\\users\\marfi\\CT_analysis\\ct_3mm_noncontrast.xlsx" ^
        --output "W:\\SCAPIS_seg_lowres_3mm_noncontrast"

    python 10_convert_from_7z_archives.py --limit 2      REM try two batches first
    python 10_convert_from_7z_archives.py --keep         REM keep the extracted DICOM

In a Jupyter cell:
    from importlib import import_module
    run = import_module("10_convert_from_7z_archives")
    run.main(["--limit", "2"])
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

convert_cohort = __import__("importlib").import_module("07_convert_cohort_to_nifti")

COHORT = Path(r"Q:\users\marfi\CT_analysis\ct_3mm_noncontrast.xlsx")
OUTPUT = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast")
ARCHIVE_ROOT = Path(r"V:\datahub")
SCRATCH = Path(r"W:\dicom_scratch")

SEVEN_ZIP_CANDIDATES = [
    Path(r"C:\Program Files\7-Zip\7z.exe"),
    Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    Path(r"C:\ProgramData\chocolatey\bin\7z.exe"),
]
BATCH = re.compile(r"ct_site_(\d+)_(casc|ccta)_(\d+)", re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--sheet", default="Selected")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=SCRATCH,
        help="where a batch is unpacked; emptied after every batch (default W:\\dicom_scratch)",
    )
    parser.add_argument(
        "--seven-zip",
        type=Path,
        default=None,
        help="path to 7z.exe when it is not in one of the usual places",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="process only the first N archives"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="do not delete the extracted DICOM after each batch",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite volumes that already exist (default: skip, so the run resumes)",
    )
    return parser.parse_known_args(argv)[0]


def find_seven_zip(explicit: Path | None) -> Path | None:
    """Path of 7z.exe, or None when the py7zr library is used instead."""
    for candidate in ([explicit] if explicit else []) + SEVEN_ZIP_CANDIDATES:
        if candidate and candidate.exists():
            return candidate
    found = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
    if found:
        return Path(found)
    try:
        import py7zr  # noqa: F401
    except ImportError:
        raise SystemExit(
            "no reader for the archives. Either pip install py7zr, or install "
            "7-Zip (https://www.7-zip.org) and pass --seven-zip C:\\path\\to\\7z.exe"
        )
    return None


def batch_of(series_folder: str) -> str:
    """ct_site_1_casc_11 out of the inventory path of a series."""
    match = BATCH.search(series_folder.replace("/", "\\"))
    return match.group(0).lower() if match else ""


def archive_for(batch: str, archive_root: Path) -> Path | None:
    """V:\\datahub\\CT-site-1-casc\\ct_site_1_casc_11.7z for that batch name."""
    match = BATCH.search(batch)
    if not match:
        return None
    site, kind, _ = match.groups()
    for folder in [
        archive_root / f"CT-site-{site}-{kind}",
        archive_root / f"ct-site-{site}-{kind}",
        archive_root / f"CT_site_{site}_{kind}",
    ]:
        for suffix in [".7z", ".zip"]:
            candidate = folder / f"{batch}{suffix}"
            if candidate.exists():
                return candidate
    for suffix in [".7z", ".zip"]:
        matches = sorted(archive_root.glob(f"*/{batch}{suffix}"))
        if matches:
            return matches[0]
    return None


def rows_to_retry(args: argparse.Namespace) -> pd.DataFrame:
    """Cohort rows whose DICOM folder is missing on disk and not yet converted."""
    cohort = pd.read_excel(args.cohort, sheet_name=args.sheet, dtype=str)
    cohort = convert_cohort.usable_rows(cohort, include_not_ok=False)
    cohort = cohort[cohort["series_folder"].notna()]

    missing = ~cohort["series_folder"].map(lambda path: Path(str(path)).exists())
    retry = cohort[missing].copy()
    if not args.overwrite:
        done = retry["patient_id"].map(
            lambda patient: (args.output / f"{patient}.nii.gz").exists()
        )
        retry = retry[~done]
    retry["batch"] = retry["series_folder"].map(batch_of)
    return retry


def extract_with_py7zr(
    archive: Path, patients: list[str], scratch: Path, chunk: int = 20
) -> None:
    """Unpack the folders of these patients with the library instead of 7z.exe.

    The archive is opened here and the open file is handed to py7zr: that keeps
    the library on its single threaded path, which otherwise opens one handle
    per worker and dies with "Too many open files" on a batch of 200 patients.
    The patients are also done in small groups, so one call never carries
    thousands of targets.
    """
    import py7zr

    wanted = {patient.lower() for patient in patients}
    with archive.open("rb") as stream:
        with py7zr.SevenZipFile(stream, mode="r") as handle:
            names = handle.getnames()
        groups: dict[str, list[str]] = {}
        for name in names:
            parts = name.replace("\\", "/").split("/")
            match = wanted & {part.lower() for part in parts}
            if match:
                groups.setdefault(next(iter(match)), []).append(name)
        if not groups:
            raise FileNotFoundError(f"no requested patient inside {archive.name}")
        found = list(groups)
        for start in range(0, len(found), chunk):
            targets = [
                name for patient in found[start : start + chunk] for name in groups[patient]
            ]
            stream.seek(0)
            with py7zr.SevenZipFile(stream, mode="r") as handle:
                handle.extract(path=scratch, targets=targets)


def extract_patients(
    seven_zip: Path | None, archive: Path, patients: list[str], scratch: Path
) -> None:
    """Unpack only the folders of these patients out of one archive."""
    scratch.mkdir(parents=True, exist_ok=True)
    if seven_zip is None:
        extract_with_py7zr(archive, patients, scratch)
        return
    command = [
        str(seven_zip),
        "x",
        str(archive),
        f"-o{scratch}",
        "-y",
        "-bso0",
        "-bsp0",
    ]
    for patient in patients:
        command.append(f"-ir!*\\{patient}\\*")
        command.append(f"-ir!*/{patient}/*")
    subprocess.run(command, check=True)


def locate_series(scratch: Path, series_folder: str, patient: str) -> Path | None:
    """The extracted copy of the series folder recorded in the inventory."""
    tail = Path(series_folder.replace("/", "\\"))
    series_uid_folder = tail.name
    for candidate in scratch.rglob(series_uid_folder):
        if candidate.is_dir():
            return candidate
    for candidate in scratch.rglob(patient):
        if candidate.is_dir():
            deepest = [folder for folder in candidate.rglob("*") if folder.is_dir()]
            if deepest:
                return max(deepest, key=lambda folder: len(list(folder.glob("*"))))
    return None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seven_zip = find_seven_zip(args.seven_zip)
    retry = rows_to_retry(args)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Reader: {seven_zip if seven_zip else 'py7zr library'}")
    print(f"{len(retry)} patients to recover from the archives")
    if retry.empty:
        return
    counts = retry["batch"].value_counts()
    print(f"{len(counts)} archives involved, largest {counts.max()} patients")

    batches = list(counts.index)
    if args.limit:
        batches = batches[: args.limit]

    rows: list[dict[str, object]] = []
    for number, batch in enumerate(batches, start=1):
        group = retry[retry["batch"] == batch]
        archive = archive_for(batch, args.archive_root)
        print(f"\n[{number}/{len(batches)}] {batch}: {len(group)} patients")
        if archive is None:
            print(f"    archive not found under {args.archive_root}")
            for record in group.to_dict("records"):
                rows.append(
                    {**record, "status": "failed", "error": f"no archive for {batch}"}
                )
            continue

        started = time.time()
        scratch = args.scratch / batch
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        try:
            extract_patients(
                seven_zip, archive, group["patient_id"].tolist(), scratch
            )
        except (subprocess.CalledProcessError, OSError) as problem:
            print(f"    extraction failed: {problem}")
            for record in group.to_dict("records"):
                rows.append({**record, "status": "failed", "error": str(problem)})
            shutil.rmtree(scratch, ignore_errors=True)
            continue
        print(f"    extracted in {time.time() - started:.0f} s")

        for record in group.to_dict("records"):
            patient = str(record["patient_id"])
            target = args.output / f"{patient}.nii.gz"
            folder = locate_series(scratch, str(record["series_folder"]), patient)
            if folder is None:
                slices, shape, error = 0, {}, "patient folder not in the archive"
            else:
                try:
                    slices, shape, error = convert_cohort.convert_one(
                        folder, str(record.get("series_uid", "")), target
                    )
                except Exception as problem:  # one bad series must not stop the run
                    slices, shape, error = 0, {}, f"{type(problem).__name__}: {problem}"
            status = "ok" if not error else "failed"
            print(f"    {patient}: {status} ({slices} slices)")
            if error:
                print(f"        {error}")
            row: dict[str, object] = dict(record)
            row["archive"] = str(archive)
            row["dicom_folder"] = str(folder) if folder else ""
            row["nifti_file"] = str(target)
            row["slices_converted"] = slices
            row["nifti_size_mb"] = (
                round(target.stat().st_size / 1024 / 1024, 1) if target.exists() else 0
            )
            row["status"] = status
            row["error"] = error
            row.update(shape)
            rows.append(row)

        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)
        log = pd.DataFrame(rows)
        path = args.output / "nifti_index_from_7z.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            log.to_excel(writer, sheet_name="Images", index=False)
            log["status"].value_counts().rename("cases").reset_index().to_excel(
                writer, sheet_name="Summary", index=False
            )

    log = pd.DataFrame(rows)
    print()
    print(log["status"].value_counts().to_string())
    print(f"\nImages : {args.output}")
    print(f"Excel  : {args.output / 'nifti_index_from_7z.xlsx'}")


if __name__ == "__main__":
    main()
