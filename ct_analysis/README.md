# CT DICOM Analysis Pipeline

Scripts for analyzing SCAPIS CT DICOM data (CASC and CCTA) from `V:\datahub`.

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

## Workflow

1. Run Script 1 first (metadata extraction) — no disk space needed, just Excel output
2. Run Script 2 (analysis/plots) — understand your data
3. Run Script 3 (NIfTI conversion) — convert one-by-one with `--delete-after` if space is tight

## Folder Structure

```
V:\datahub\
  CT-site-1-casc\
  CT-site-1-ccta\
  CT-site-2-casc\
  ...

Q:\users\marfi\CT_analysis\
  dicom_all_metadata.xlsx     <- Script 1
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
