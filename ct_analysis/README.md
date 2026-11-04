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

Recursively scans every file below the extracted datahub, reads DICOM headers without loading pixel data, and safely resumes by skipping unchanged files. Folder names such as `ct_site_4_ccta_24` are parsed into site, protocol, and archive batch; names that deviate (`site-5-casc`, `ct_site_5_ccta_24b`) fall back to a `site<number>` search so no site is silently dropped.

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

A site only gets a workbook when DICOM headers were actually read for it. `SCAPIS_all_sites_3D_4D_inventory.xlsx` therefore has a `Folder_Coverage` sheet listing every top-level source folder with file counts and a status:
- `dicom_found` — headers were read, so the site has a workbook.
- `only_archives_extract_first` — the folder still holds `.7z`/`.zip` archives, so extract them (or point `--root` at the extracted copy) and re-run.
- `no_dicom_headers` / `empty_folder` — files exist but none are DICOM, or the folder is empty.

The same list is printed at the end of the run, one workbook failure no longer stops the remaining sites, and control characters in DICOM text are stripped before writing Excel.

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

### 5. `05_check_site_coverage.py` — Why is a site missing?

Fast per-folder check that answers "why is there no Excel for site 5 or site 6?" without re-scanning every DICOM. It counts files, archives, and sampled DICOM headers per top-level folder, compares them with the SQLite database and the workbooks already in the output folder, and writes `site_coverage_check.xlsx`.

```bat
python 05_check_site_coverage.py ^
  --root "Q:\users\leejo\data\scapis\datahub" ^
  --output "Q:\users\marfi\CT_analysis"

REM Test every file instead of 25 per folder
python 05_check_site_coverage.py --sample 0
```

---

### 6. `06_build_af_cohort.py` — AF vs non-AF CCTA cohort for LA segmentation

Joins the clinical CSV (`Subject`, `AF_CT_Baseline`, `Sex`, `AgeAtVisitOne`, `Height`, `Weight`) to `all_series_inventory.csv`, keeps only series usable for left-atrium and atrial-fat segmentation, picks one series per patient, and matches AF patients 1:1 to non-AF controls.

Kept series must be contrast CCTA, an `ORIGINAL` reconstruction, a 3D volume, `<= 1.0 mm`, `duplicate_sop_count = 0`, with at least `--min-slices` unique slice positions matching the image count. Ties are broken towards `BestDiast` / the phase closest to `--target-phase` (default 70), the `I26f` kernel, then the thinnest recon with the most slices. Controls are matched inside exact site/sex/kVp/thickness strata, then on closest age and BMI.

**Output:** `af_cohort.xlsx` with `Summary`, `AF_Selected`, `Control_Selected`, `Matched_Pairs`, `Pilot_nnUNet` (one nnU-Net case per row, with the geometry columns and the `series_folder` path), `All_Patients` (every patient in the clinical CSV with `group` AF/control, `usable` ok/not ok and the `reason`), `AF_Not_Usable`, `AF_Loss_Reasons`, and `Rejected` / `Rejection_Reasons` (every dropped series with `rejected_because`, plus the counts).

```bat
python 06_build_af_cohort.py --pilot 10
python 06_build_af_cohort.py --site 1 --min-slices 200 --target-phase 40
python 06_build_af_cohort.py --allow-duplicates
```

Use `AF_Loss_Reasons` before relaxing anything. `--allow-duplicates` keeps series whose images are repeated by overlapping archive batches — script 07 then drops the repeated `SOPInstanceUID`s and keeps one file per slice position, so the written volume is still correct.

`--target-phase 70` is the motion-reduced best-diastolic reconstruction available for nearly every patient; the left atrium is largest at end-systole (~30-40%), so choose the phase from the study endpoint and apply the same value to both groups.

---

### 7. `07_convert_cohort_to_nifti.py` — cohort DICOM to nnU-Net NIfTI

Converts the cohort sheet to `<output>/Dataset001_SCAPIS_LA/imagesTr/<case_id>_0000.nii.gz`, one series at a time, reading each series straight from its source folder so nothing is staged or extracted. Existing files are skipped, so an interrupted run resumes.

`<output>/nifti_index.xlsx` lists one row per patient: AF/control group, clinical values, acquisition parameters, the DICOM folder, the NIfTI path, the label file to create, and the geometry read back out of the written volume (dimensions, voxel spacing, FOV, z coverage, origin, direction, HU range).

```bat
python 07_convert_cohort_to_nifti.py --output "W:\SCAPIS_nnUNet" --limit 4
python 07_convert_cohort_to_nifti.py --output "W:\SCAPIS_nnUNet"
```

`--layout flat` writes every volume named after the patient in the output folder itself — `<output>\<patient_id>.nii.gz`, with the segmentation saved beside it as `<patient_id>_seg.nii.gz`; `--layout per-patient` is the same with one subfolder per patient. Voxel spacing and dimensions are in the NIfTI header and in `nifti_index.xlsx`; `--info` additionally writes `<patient_id>_info.json` beside each volume. `--copy-from` points at a folder of volumes that were already converted, and a volume named after that patient is copied instead of being read from DICOM again.

Any sheet can drive it — `Pilot_nnUNet`, `AF_Selected`, `All_Patients`, or the `Patients` sheet of `SCAPIS_clinical_with_CT_status.xlsx`. Only rows marked ok for segmentation are converted (`--include-not-ok` overrides), so `--sheet All_Patients` converts every usable patient. `nifti_index.xlsx` is rewritten every 25 cases, so a long run can be interrupted and resumed.

```bat
python 07_convert_cohort_to_nifti.py --layout flat --output "W:\SCAPIS_seg" --sheet All_Patients
python 07_convert_cohort_to_nifti.py --layout flat --output "W:\SCAPIS_seg" --sheet All_Patients --copy-from "W:\SCAPIS_nnUNet\Dataset001_SCAPIS_LA\imagesTr"
```

---

### 8. `08_add_ct_status_to_clinical.py` — clinical CSV plus CT status per patient

Writes `SCAPIS_clinical_with_CT_status.xlsx`: your clinical CSV with every original column untouched, plus `ct_available` (YES/NO), `segmentation_ready` (ok / not ok), `why` (one sentence — either why that series suits LA and atrial-fat work, or the single rule that removed it), the chosen series and its parameters, and `series_folder` / `example_file`. It imports the selection rules from script 6, so both always agree on "usable".

```bat
python 08_add_ct_status_to_clinical.py
python 08_add_ct_status_to_clinical.py --allow-duplicates
```

---

### 9. `09_build_3mm_cohort.py` — the thick-slice (3 mm) cohort

Same selection as script 6 but for the low resolution reconstruction: contrast series with a slice thickness between `--min-thickness` and `--max-thickness` (default 2.0–3.5 mm), ORIGINAL reconstruction, no duplicated or missing slices, and at least `--min-slices` slice positions (default 40 ≈ 120 mm of coverage at 3 mm). Series with too few slices are never converted and are listed in the `Too_Few_Slices` sheet, so poor coverage never reaches the segmentation folder.

`ct_3mm_cohort.xlsx` has `Summary`, `All_Patients` (one row per clinical patient with `segmentation_ready`), `Selected` (the chosen series, the input for script 7), `Too_Few_Slices`, and `Rejected`.

```bat
python 09_build_3mm_cohort.py
python 07_convert_cohort_to_nifti.py --cohort "Q:\users\marfi\CT_analysis\ct_3mm_cohort.xlsx" --sheet Selected --layout flat --output "W:\SCAPIS_seg_lowres_3mm"
```

`--allow-non-contrast` keeps the non-contrast 3 mm series as well, `--non-contrast-only` keeps only the calcium scoring reconstruction, and `--min-slices 50` is stricter about coverage.

---

### 10. `10_convert_from_7z_archives.py` — recover the series that were never unpacked

The unpacked datahub on `Q:` holds the contrast (`ccta`) batches only, so script 7 reports `folder not found` for every calcium scoring (`casc`) patient. Those batches are still on the share as one archive per batch, for example `V:\datahub\CT-site-1-casc\ct_site_1_casc_11.7z`.

This script takes the cohort rows whose `series_folder` does not exist on disk, groups them by batch, and for each batch extracts only the patient folders it needs into a scratch folder, converts them with the same code as script 7, writes `<patient_id>.nii.gz` into the same output folder, then deletes the extracted DICOM before the next batch. Peak extra disk use is one batch. The archives themselves are never modified. It resumes: volumes that already exist are skipped, and `nifti_index_from_7z.xlsx` is rewritten after every batch.

```bat
python 10_convert_from_7z_archives.py --cohort "Q:\users\marfi\CT_analysis\ct_3mm_noncontrast.xlsx" --output "W:\SCAPIS_seg_lowres_3mm_noncontrast"
python 10_convert_from_7z_archives.py --limit 2
```

`--limit 2` tries two batches first, `--keep` leaves the extracted DICOM in place, and `--seven-zip` points at `7z.exe` when it is not in the usual place. Without 7-Zip the archives are read with the `py7zr` package (`pip install py7zr`), which needs no admin rights.

---

### 11. `11_fat_hu_features_and_groups.py` — clean the fat masks by HU, measure, compare groups

The nnU-Net fat model outputs the whole pericardial fat compartment, so the mask still contains non-fat voxels. This script keeps only the mask voxels whose CT value lies between `--fat-low` and `--fat-high` (-190 to -30 HU by default), writes the cleaned mask to `<output>\fat_hu_masks\<patient_id>.nii.gz`, and measures 11 features per patient: cleaned and raw fat volume, kept fraction, fat mean/SD/P25/P75 HU, left-atrium volume and mean HU, fat-to-LA ratio, and fat volume per cm of covered heart length.

Groups come from `test80_groups.xlsx` when present, otherwise from the clinical CSV (`AF_CT_Baseline`, `Prev_HF`, `Ischemic_Stroke_IC_Baseline`; `YES`/`NO` coded). `fat_la_features.xlsx` holds `Features` (one row per patient), `Group_Stats` (median, IQR, mean, SD per group), and `Tests` (Mann-Whitney of each group against the controls), plus one box plot per feature under `plots\`.

Input names may be either `<patient_id>_0000.nii.gz` (the nnU-Net input copies) or `<patient_id>.nii.gz`; nothing is renamed and the masks are read where `nnUNetv2_predict` wrote them.

```bat
python 11_fat_hu_features_and_groups.py
python 11_fat_hu_features_and_groups.py --fat-low -190 --fat-high -30 --limit 5
python 11_fat_hu_features_and_groups.py --no-plots
```

---

### 12. `12_random_forest_groups.py` — random forest per group on the fat/LA features

Takes the `Features` sheet of script 11, adds age, sex and BMI from the clinical CSV (epicardial fat grows with body size, so the group effect has to survive that), and trains one random forest per group against the controls with repeated stratified cross validation. Reported AUC, sensitivity and specificity are out of fold, so they are not the training fit.

`random_forest.xlsx` has `Performance`, `Importance` (mean importance per feature and group) and `Predictions` (out-of-fold probability per patient); `plots\rf_<group>_roc.png` and `plots\rf_<group>_importance.png` show the same; `models\rf_<group>.joblib` is the forest refit on all patients.

```bat
python 12_random_forest_groups.py
python 12_random_forest_groups.py --groups AF --trees 1000 --repeats 20

REM score new patients whose features came out of script 11
python 12_random_forest_groups.py --apply AF ^
  --features "Q:\...\new_cases\fat_la_features.xlsx" ^
  --models "Q:\...\test80_analysis\models"
```

---

### 13. `13_compare_models.py` — several classifiers side by side, plus combined features

Runs logistic regression (L2 and elastic net), random forest, gradient boosting, an RBF support vector machine and k nearest neighbours over the *same* repeated 5-fold splits, so the AUCs are comparable rather than six separate experiments. Continuous features are standardised inside each fold, so the held-out patients never contribute to the mean and SD.

On top of the measured features it derives combinations, because a raw fat volume largely reports body size: `fat_per_bmi`, `fat_per_la`, `fat_per_year`, `la_per_bmi`, `fat_hu_iqr`, `fat_hu_skew`, `fat_dense_fraction`, `la_to_fat_hu`, `fat_density_ratio`, `fat_volume_z`. Use `--no-derived` to compare against the plain feature set.

`model_comparison.xlsx` has `Performance` (AUC mean and SD per fold, out-of-fold AUC, accuracy, sensitivity, specificity per model and group), `Best`, `Importance` (importances or coefficients of the best model) and `Predictions`; `plots\compare_<group>_auc.png` and `plots\compare_<group>_roc.png` show the same.

With roughly 20 patients per group the per-fold SD is around 0.15, so differences between models of that size mean nothing; the script prints the SD next to each AUC for that reason.

```bat
python 13_compare_models.py
python 13_compare_models.py --folds 5 --repeats 10
python 13_compare_models.py --no-derived --groups AF
```

---

### 14. `14_radiomic_features.py` — PyRadiomics features per compartment

Features for the left atrium, the LA-EAT and the whole-heart EAT, extracted with PyRadiomics so the definitions are the standard (IBSI) ones: first order `Mean`, `StandardDeviation`, `Skewness`, `Kurtosis`, `Entropy`, and GLSZM `SmallAreaEmphasis`, `LargeAreaEmphasis`, `GrayLevelNonUniformity`, `SizeZoneNonUniformity`, `ZonePercentage`, `ZoneEntropy`. Volumes in millilitres come from the voxel count. No HU window is applied here; the EAT masks are expected to be the cleaned ones with the atrial blood pool already removed (`fat_hu_masks`).

If no LA-EAT mask folder is passed, LA-EAT is derived as the EAT sitting within `--la-eat-mm` (10 mm by default) of the atrial surface, measured by a distance transform in real millimetres. Grey levels are binned by `--bin-width` HU (25 by default), the PyRadiomics default style of fixed bin width rather than a fixed count.

Composite indices `cmi1 = W * L` and `cmi2 = W**2 * L**0.1 * E**3` of whole-heart EAT (W), LA volume (L) and LA-EAT (E) are written alongside; script 15 re-optimises those exponents.

Output: `radiomic_features.xlsx` with `Features`, `Group_Stats` and `Tests`, and box plots in `plots_radiomic`.

```bat
pip install pyradiomics
python 14_radiomic_features.py
python 14_radiomic_features.py --la-eat-masks "Q:\...\test80_seg_laeat"
python 14_radiomic_features.py --bin-width 25 --la-eat-mm 10
```

---

### 15. `15_feature_search.py` — composite exponents and the 1-4 feature search

Reads the sheet from script 14 (merged with script 11 and with age, sex and BMI) and, per group against the controls:

1. searches the exponents of `W**a * L**b * E**c` on a grid, scoring each candidate by the cross-validated AUC of that single index, and keeps the best as `cmi_opt` next to the published `cmi1` and `cmi2`;
2. ranks every feature by univariate cross-validated AUC and carries the top `--pool` forward, since all 1-4 combinations of 50 features is far beyond what 40 patients support;
3. trains every combination of 1, 2, 3 and 4 of that pool with logistic regression, random forest, SVM-RBF and linear discriminant analysis, each a pipeline of median imputation and z-scoring, under the same stratified 5-fold cross validation repeated `--repeats` times.

Two things to keep in mind when reading the table: the best of a few thousand combinations is optimistic by construction, and the exponent grid is scored on the same splits as everything else. `--holdout 0.25` keeps a quarter of the patients out of the entire search and reports `auc_holdout` for the top rows, which is the honest number.

Output: `feature_search.xlsx` with `Best_Models`, `Top_Combinations`, `Composite` and `Univariate`, and `plots\search_<group>.png`.

```bat
python 15_feature_search.py
python 15_feature_search.py --pool 10 --max-size 4 --repeats 3
python 15_feature_search.py --holdout 0.25 --groups AF
```

---

### 18. `18_build_case_control_set.py` — every AF, HF and stroke patient plus matched controls

Takes the whole 3 mm non-contrast cohort instead of the 20-per-group pilot: every patient whose clinical record says AF, heart failure or ischaemic stroke and who has a converted volume, plus the same number of controls matched on sex, age and BMI (`--controls-per-case 2` for two per case).

Because a patient can carry several diagnoses, two columns are written: `group` (one label, by the priority AF, HF, stroke) for the group comparisons, and `diagnoses` (`AF+HF`, `AF+HF+stroke`, ...) so the overlap is never hidden. The `Overlaps` sheet counts each combination and each pair, which is the table to report before any comparison.

The selected volumes are copied into `W:\SCAPIS_seg_lowres_3mm_noncontrast\analysis\test_AF_HF_stroke_control` as `<patient_id>_0000.nii.gz`, the name nnU-Net/PoinUNet expects (`--plain-names` drops the `_0000`), together with `test80_groups.xlsx`, the file scripts 11, 14 and 15 read, so the rest of the chain runs unchanged. `--no-copy` writes the workbook only when disk space is short.

Output: `case_control_set.xlsx` with `Summary`, `Groups`, `Overlaps`, `Matching` and `Missing_Volume` (diagnosed patients without a volume).

```bat
python 18_build_case_control_set.py
python 18_build_case_control_set.py --controls-per-case 2 --age-tolerance 3
python 18_build_case_control_set.py --no-copy
```

---

## Workflow

1. For the complete extracted datahub, run Script 4 to create the resumable raw database, per-site workbooks, and master 3D/4D index. If a site has no workbook afterwards, run Script 5 to see whether that folder still holds archives or contains no DICOM headers.
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
