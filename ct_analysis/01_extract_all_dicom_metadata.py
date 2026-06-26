"""
Script 1: Extract ALL DICOM metadata from SCAPIS CT data to Excel.

This script scans all DICOM files in the datahub folder structure
(CT-site-X-casc, CT-site-X-ccta) and extracts EVERY available DICOM tag
into a comprehensive Excel file.

Input:  V:\datahub  (network drive with CT-site-* folders)
Output: Q:\users\marfi\CT_analysis\dicom_all_metadata.xlsx

Usage:
    python 01_extract_all_dicom_metadata.py
"""

import pydicom
from pathlib import Path
import pandas as pd
from collections import defaultdict
import time
import sys

# ============================================================
# CONFIGURATION - adjust these paths to your setup
# ============================================================
DATA_ROOT = Path(r"V:\datahub")               # SCAPIS datahub on network drive
OUT_DIR = Path(r"Q:\users\marfi\CT_analysis")  # Output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EXCEL = OUT_DIR / "dicom_all_metadata.xlsx"
OUT_SUMMARY = OUT_DIR / "dicom_series_summary.xlsx"

# ============================================================
# DICOM detection
# ============================================================
def is_dicom(path: Path) -> bool:
    """Check if file is DICOM by reading the 128-byte preamble + 'DICM' magic."""
    try:
        with open(path, "rb") as f:
            pre = f.read(132)
        return len(pre) == 132 and pre[128:132] == b"DICM"
    except Exception:
        return False


# ============================================================
# Extract ALL DICOM tags from one file
# ============================================================
def extract_all_tags(dcm_path: Path, site_folder: str) -> dict:
    """Read a DICOM file and extract every tag as a flat dictionary."""
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    except Exception as e:
        return {"file_path": str(dcm_path), "site_folder": site_folder, "error": str(e)}

    record = {
        "file_path": str(dcm_path),
        "site_folder": site_folder,
    }

    # Standard important tags (always include even if empty)
    important_tags = [
        "PatientID", "PatientName", "PatientAge", "PatientSex",
        "StudyInstanceUID", "StudyDate", "StudyTime", "StudyDescription",
        "SeriesInstanceUID", "SeriesNumber", "SeriesDescription",
        "Modality", "Manufacturer", "ManufacturerModelName",
        "InstitutionName", "StationName",
        "Rows", "Columns", "BitsAllocated", "BitsStored",
        "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
        "ImagePositionPatient", "ImageOrientationPatient",
        "KVP", "XRayTubeCurrent", "Exposure", "ExposureTime",
        "ConvolutionKernel", "FilterType",
        "ContrastBolusAgent", "ContrastBolusRoute",
        "AcquisitionDate", "AcquisitionTime", "AcquisitionNumber",
        "InstanceNumber", "ImageType",
        "WindowCenter", "WindowWidth",
        "RescaleIntercept", "RescaleSlope",
        "GantryDetectorTilt", "TableHeight",
        "ReconstructionDiameter", "DataCollectionDiameter",
        "ProtocolName", "BodyPartExamined",
        "ScanOptions", "CTDIvol",
        "SliceLocation",
        "PhotometricInterpretation", "SamplesPerPixel",
        "DistanceSourceToDetector", "DistanceSourceToPatient",
        "FocalSpots", "RevolutionTime",
        "SingleCollimationWidth", "TotalCollimationWidth",
        "TableSpeed", "TableFeedPerRotation",
        "SpiralPitchFactor",
    ]

    for tag_name in important_tags:
        val = getattr(ds, tag_name, None)
        if val is not None:
            # Convert pydicom types to plain Python
            if isinstance(val, pydicom.multival.MultiValue):
                record[tag_name] = str(list(val))
            elif isinstance(val, pydicom.uid.UID):
                record[tag_name] = str(val)
            elif hasattr(val, "original_string"):
                record[tag_name] = str(val)
            else:
                record[tag_name] = str(val)
        else:
            record[tag_name] = ""

    # Also extract any OTHER tags not in the list above
    for elem in ds:
        if elem.VR == "SQ":  # Skip sequences (nested)
            continue
        if elem.keyword and elem.keyword not in record:
            try:
                val = elem.value
                if isinstance(val, (bytes, bytearray)):
                    record[elem.keyword] = f"<binary {len(val)} bytes>"
                elif isinstance(val, pydicom.multival.MultiValue):
                    record[elem.keyword] = str(list(val))
                else:
                    record[elem.keyword] = str(val)
            except Exception:
                record[elem.keyword] = "<unreadable>"

    return record


# ============================================================
# Classify series type from description and parameters
# ============================================================
def classify_series(desc, slice_thickness, n_slices):
    """Classify a DICOM series based on description and parameters."""
    d = (desc or "").lower()

    if "topo" in d or "localizer" in d or "scout" in d or n_slices <= 2:
        return "localizer"

    if any(k in d for k in ["casc", "ca sc", "calcium", "b35f", "score"]):
        return "calcium_scoring"

    if any(k in d for k in ["ccta", "cta", "coronary", "angio"]):
        return "ccta"

    if any(k in d for k in ["contrast", "arterial", "venous", "portal"]):
        return "contrast"

    # Thin slices without contrast keywords -> likely CCTA
    try:
        if slice_thickness and float(slice_thickness) < 1.0 and n_slices > 100:
            return "ccta"
    except (ValueError, TypeError):
        pass

    # Thick slices -> likely calcium scoring
    try:
        if slice_thickness and float(slice_thickness) >= 2.5:
            return "calcium_scoring"
    except (ValueError, TypeError):
        pass

    return "other"


# ============================================================
# Main scanning pipeline
# ============================================================
def scan_all_data():
    """Scan all CT-site-* folders and extract metadata."""

    print("=" * 70)
    print("SCAPIS CT DICOM Metadata Extraction")
    print("=" * 70)
    print(f"Data root:  {DATA_ROOT}")
    print(f"Output dir: {OUT_DIR}")
    print()

    # Find all CT-site-* folders
    site_folders = sorted([
        d for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("CT-site-")
    ])

    if not site_folders:
        print("ERROR: No CT-site-* folders found in", DATA_ROOT)
        sys.exit(1)

    print(f"Found {len(site_folders)} site folders:")
    for sf in site_folders:
        print(f"  - {sf.name}")
    print()

    # Scan all files
    all_records = []
    total_files = 0
    total_dicoms = 0

    for site_folder in site_folders:
        print(f"\nScanning: {site_folder.name} ...")
        folder_dicoms = 0

        for f in site_folder.rglob("*"):
            if not f.is_file():
                continue
            total_files += 1

            if is_dicom(f):
                total_dicoms += 1
                folder_dicoms += 1

                # For efficiency: only read first file per directory fully
                # (read every file for complete metadata)
                record = extract_all_tags(f, site_folder.name)
                all_records.append(record)

                if total_dicoms % 500 == 0:
                    print(f"  ... processed {total_dicoms} DICOM files so far")

        print(f"  -> {folder_dicoms} DICOM files in {site_folder.name}")

    print(f"\n{'=' * 70}")
    print(f"Total files scanned: {total_files}")
    print(f"Total DICOM files:   {total_dicoms}")
    print(f"{'=' * 70}")

    return all_records


def create_series_summary(df):
    """Create a per-series summary from the full metadata."""
    if "SeriesInstanceUID" not in df.columns:
        print("WARNING: No SeriesInstanceUID found, skipping series summary.")
        return None

    series_groups = df.groupby("SeriesInstanceUID")

    summaries = []
    for uid, group in series_groups:
        first = group.iloc[0]
        n_slices = len(group)

        desc = first.get("SeriesDescription", "")
        thickness = first.get("SliceThickness", "")

        series_type = classify_series(desc, thickness, n_slices)

        # Pixel spacing
        px_str = first.get("PixelSpacing", "")
        try:
            # Parse "[0.5, 0.5]" format
            if px_str and px_str.startswith("["):
                px_vals = [float(x.strip().strip("'\"")) for x in px_str.strip("[]").split(",")]
                px_row = px_vals[0] if len(px_vals) > 0 else ""
                px_col = px_vals[1] if len(px_vals) > 1 else ""
            else:
                px_row = px_col = px_str
        except Exception:
            px_row = px_col = ""

        summaries.append({
            "series_uid": uid,
            "series_type": series_type,
            "n_slices": n_slices,
            "site_folder": first.get("site_folder", ""),
            "patient_id": first.get("PatientID", ""),
            "study_date": first.get("StudyDate", ""),
            "series_number": first.get("SeriesNumber", ""),
            "series_description": desc,
            "modality": first.get("Modality", ""),
            "manufacturer": first.get("Manufacturer", ""),
            "model": first.get("ManufacturerModelName", ""),
            "rows": first.get("Rows", ""),
            "columns": first.get("Columns", ""),
            "pixel_spacing_row_mm": px_row,
            "pixel_spacing_col_mm": px_col,
            "slice_thickness_mm": thickness,
            "spacing_between_slices_mm": first.get("SpacingBetweenSlices", ""),
            "kvp": first.get("KVP", ""),
            "tube_current_mA": first.get("XRayTubeCurrent", ""),
            "exposure": first.get("Exposure", ""),
            "convolution_kernel": first.get("ConvolutionKernel", ""),
            "contrast_agent": first.get("ContrastBolusAgent", ""),
            "protocol_name": first.get("ProtocolName", ""),
            "body_part": first.get("BodyPartExamined", ""),
            "ctdi_vol": first.get("CTDIvol", ""),
            "reconstruction_diameter": first.get("ReconstructionDiameter", ""),
            "window_center": first.get("WindowCenter", ""),
            "window_width": first.get("WindowWidth", ""),
            "example_file": first.get("file_path", ""),
        })

    return pd.DataFrame(summaries)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    start_time = time.time()

    all_records = scan_all_data()

    if not all_records:
        print("No DICOM files found. Check DATA_ROOT path.")
        sys.exit(1)

    # Create full metadata DataFrame
    print("\nCreating full metadata DataFrame...")
    df_full = pd.DataFrame(all_records)

    # Save full metadata to Excel
    print(f"Saving full metadata to: {OUT_EXCEL}")
    df_full.to_excel(OUT_EXCEL, index=False, engine="openpyxl")

    # Create series summary
    print("\nCreating series summary...")
    df_summary = create_series_summary(df_full)

    if df_summary is not None:
        print(f"Saving series summary to: {OUT_SUMMARY}")
        df_summary.to_excel(OUT_SUMMARY, index=False, engine="openpyxl")

        # Print quick stats
        print("\n" + "=" * 70)
        print("QUICK SUMMARY")
        print("=" * 70)
        print(f"\nTotal unique series: {len(df_summary)}")
        print(f"\nSeries types:")
        print(df_summary["series_type"].value_counts().to_string())
        print(f"\nPer-site counts:")
        print(df_summary.groupby("site_folder")["series_type"].value_counts()
              .unstack(fill_value=0).to_string())
        print(f"\nUnique patients: {df_summary['patient_id'].nunique()}")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f} seconds.")
    print(f"Output files:")
    print(f"  1. {OUT_EXCEL}")
    print(f"  2. {OUT_SUMMARY}")
