# CT DICOM Analysis Pipeline

Scripts for analyzing SCAPIS CT DICOM data (CASC and CCTA). The scalable inventory script is configured for the extracted folders under `Q:\users\leejo\data\scapis\datahub`.

## Requirements

```bash
pip install pydicom pandas openpyxl matplotlib seaborn numpy SimpleITK
```

## Scripts

### 1. `01_extract_all_dicom_metadata.py` — Extract ALL DICOM metadata

Scans every DICOM file in `V:\datahub\CT-site-*` and extracts **all** DICOM tags
to a comprehensive Excel file.

**Output:**
- `Q:\users\marfi\CT_analysis\dicom_all_metadata.xlsx` — every tag from every file
- `Q:\users\marfi\CT_analysis\dicom_series_summary.xlsx` — one row per series with classification

**Run:**
```
python 01_extract_all_dicom_metadata.py
```

> Note: This may take 30-60 minutes depending on the number of files and network speed.

---

### 2. `02_analyze_and_plot.py` — Analyze and visualize

Reads the series summary and generates:
- 10 analysis plots (bar charts, distributions, pie charts)
- Multi-sheet Excel report with per-site, per-patient, per-kernel summaries

**Output:**
- `Q:\users\marfi\CT_analysis\plots\*.png`
- `Q:\users\marfi\CT_analysis\analysis_report.xlsx`

**Run:**
```
python 02_analyze_and_plot.py
```

---

### 3. `03_convert_nifti_one_by_one.py` — NIfTI conversion (space-efficient)

Converts DICOM series to NIfTI one at a time. Optionally deletes source files
after conversion to save disk space.

**Output:**
- `Q:\users\marfi\CT_analysis\nifti\<patient_id>\<series_type>_<name>.nii.gz`

**Run:**
```
# Preview what will be converted (no actual conversion)
python 03_convert_nifti_one_by_one.py --dry-run

# Convert all CASC series
python 03_convert_nifti_one_by_one.py --type casc

# Convert CCTA and delete source after each successful conversion
python 03_convert_nifti_one_by_one.py --type ccta --delete-after

# Resume a previous interrupted run
python 03_convert_nifti_one_by_one.py --resume

# Convert everything with auto-cleanup
python 03_convert_nifti_one_by_one.py --type all --delete-after --resume
```

---

### 4. `04_build_site_dicom_inventories.py` — All-site scalable DICOM inventory

Recursively scans every file below the extracted datahub, reads DICOM headers without loading pixel data, and safely resumes by skipping unchanged files. Folder names such as `ct_site_4_ccta_24` are parsed into site, protocol, and archive batch.

The raw inventory is stored in SQLite because a complete per-file/all-tag inventory can exceed Excel's 1,048,576-row limit. Excel workbooks contain the analysis-ready patient, study, series, acquisition, reconstruction, protocol, and 3D/4D summaries with clickable source-folder links.

**Default input:**
- `Q:\users\leejo\data\scapis\datahub`

**Default output:**
- `Q:\users\leejo\data\scapis\dicom_inventory\scapis_dicom_inventory.sqlite` — every DICOM file plus all non-pixel header tags as JSON
- `site_<N>_dicom_inventory.xlsx` — one workbook per site
- `SCAPIS_all_sites_3D_4D_inventory.xlsx` — master list of all confirmed 3D/4D and conservative possible-4D series
- `all_series_inventory.csv` — portable one-row-per-series inventory

Each site workbook contains:
- `Overview`, `Source_Folders`, `Patients`, and `Studies`
- `Series`, `CASC_Series`, and `CCTA_Series` with important DICOM acquisition/reconstruction metadata
- `3D_4D_Images` with folder and example-file hyperlinks
- `Protocols` for protocol comparison
- `Quality_Summary` and `Read_Errors`

**Recommended Windows run:**
```bat
run_scapis_inventory_windows.bat
```

Or run directly:
```bat
python 04_build_site_dicom_inventories.py ^
  --root "Q:\users\leejo\data\scapis\datahub" ^
  --output "Q:\users\leejo\data\scapis\dicom_inventory" ^
  --workers 8
```

Useful options:
```bat
REM Rebuild Excel without scanning DICOM again
python 04_build_site_dicom_inventories.py --report-only

REM Include one row per DICOM file in Excel (can create very large workbooks)
python 04_build_site_dicom_inventories.py --include-file-sheets

REM Reduce database size by retaining curated fields only
python 04_build_site_dicom_inventories.py --no-all-tags
```

The 4D classification is intentionally conservative:
- `4D`: multiple temporal positions are directly observed inside one series.
- `4D (multi-series)`: multiple cardiac-phase percentages form a matching family of 3D reconstructions.
- `possible_4D`: a multi-phase count is reported, but per-image phase evidence needs manual review.
- Multiple reconstructions at the same phase remain separate 3D series, not 4D.

> DICOM headers may contain personal data. Keep the SQLite and Excel outputs on approved SCAPIS storage and do not upload them to public services.

---

## Workflow

1. For the complete extracted datahub, run Script 4 to create the resumable raw database, per-site workbooks, and master 3D/4D index.
2. Use Script 2 for plots when needed (it consumes the legacy Script 1 series workbook).
3. Use Script 3 only when selected DICOM series need NIfTI conversion.
4. Scripts 1–3 remain available for the earlier `V:\datahub` workflow.

## Folder Structure

```
V:\datahub\
  CT-site-1-casc\
  CT-site-1-ccta\
  CT-site-2-casc\
  ...

Q:\users\leejo\data\scapis\dicom_inventory\
  scapis_dicom_inventory.sqlite       <- Script 4, all files and all non-pixel tags
  site_1_dicom_inventory.xlsx         <- Script 4, one workbook per site
  SCAPIS_all_sites_3D_4D_inventory.xlsx <- Script 4, master volume index
  all_series_inventory.csv            <- Script 4

Q:\users\marfi\CT_analysis\
  dicom_all_metadata.xlsx     <- Script 1 (legacy workflow)
  dicom_series_summary.xlsx   <- Script 1
  analysis_report.xlsx        <- Script 2
  plots\                      <- Script 2
    01_series_per_site.png
    02_slices_per_site.png
    ...
  nifti\                      <- Script 3
    <patient_id>\
      calcium_scoring_*.nii.gz
      ccta_*.nii.gz
  conversion_progress.json    <- Script 3 (resume tracking)
```

## Configuration

Edit the paths at the top of each script if your drives are mounted differently:

```python
DATA_ROOT = Path(r"V:\datahub")               # SCAPIS network drive
OUT_DIR   = Path(r"Q:\users\marfi\CT_analysis")  # Output folder
```
