"""
Script 2: Analyze DICOM metadata and generate plots.

Reads the series summary Excel from Script 1 and creates:
- Per-site bar charts (number of series, slices)
- Series type distribution
- Parameter distributions (slice thickness, pixel spacing, kVp, mA)
- Contrast vs non-contrast breakdown
- Patient-level summary

Input:  Q:\users\marfi\CT_analysis\dicom_series_summary.xlsx
Output: Q:\users\marfi\CT_analysis\plots\*.png
        Q:\users\marfi\CT_analysis\analysis_report.xlsx

Usage:
    python 02_analyze_and_plot.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import seaborn as sns
from pathlib import Path
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
WORK_DIR = Path(r"Q:\users\marfi\CT_analysis")
SUMMARY_FILE = WORK_DIR / "dicom_series_summary.xlsx"
PLOT_DIR = WORK_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT = WORK_DIR / "analysis_report.xlsx"

# Style
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


# ============================================================
# Load data
# ============================================================
def load_data():
    print(f"Loading: {SUMMARY_FILE}")
    df = pd.read_excel(SUMMARY_FILE, engine="openpyxl")
    print(f"  -> {len(df)} series loaded")
    return df


# ============================================================
# Plot 1: Series count per site
# ============================================================
def plot_series_per_site(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ct = df.groupby(["site_folder", "series_type"]).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Number of Series per Site (by Type)", fontsize=14)
    ax.set_xlabel("Site Folder")
    ax.set_ylabel("Number of Series")
    ax.legend(title="Series Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = PLOT_DIR / "01_series_per_site.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Plot 2: Total slices per site
# ============================================================
def plot_slices_per_site(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ct = df.groupby(["site_folder", "series_type"])["n_slices"].sum().unstack(fill_value=0)
    ct.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Total Slices per Site (by Type)", fontsize=14)
    ax.set_xlabel("Site Folder")
    ax.set_ylabel("Total Slices")
    ax.legend(title="Series Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = PLOT_DIR / "02_slices_per_site.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Plot 3: Series type distribution (pie chart)
# ============================================================
def plot_series_type_pie(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # By count
    counts = df["series_type"].value_counts()
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Series Type Distribution (by count)")

    # By total slices
    slice_counts = df.groupby("series_type")["n_slices"].sum()
    axes[1].pie(slice_counts, labels=slice_counts.index, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Series Type Distribution (by total slices)")

    plt.tight_layout()
    out = PLOT_DIR / "03_series_type_pie.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Plot 4: Slice thickness distribution
# ============================================================
def plot_slice_thickness(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    thickness = pd.to_numeric(df["slice_thickness_mm"], errors="coerce").dropna()

    if len(thickness) > 0:
        sns.histplot(thickness, bins=30, kde=True, ax=ax)
        ax.set_title("Slice Thickness Distribution", fontsize=14)
        ax.set_xlabel("Slice Thickness (mm)")
        ax.set_ylabel("Count")

        # Add vertical lines for common thresholds
        ax.axvline(x=0.6, color="r", linestyle="--", alpha=0.7, label="0.6mm (thin CCTA)")
        ax.axvline(x=3.0, color="g", linestyle="--", alpha=0.7, label="3.0mm (CASC)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid slice thickness data", ha="center", va="center")

    plt.tight_layout()
    out = PLOT_DIR / "04_slice_thickness.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Plot 5: Slice count distribution per series type
# ============================================================
def plot_slice_count_boxplot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="series_type", y="n_slices", ax=ax)
    ax.set_title("Number of Slices per Series Type", fontsize=14)
    ax.set_xlabel("Series Type")
    ax.set_ylabel("Number of Slices")
    plt.xticks(rotation=30)
    plt.tight_layout()
    out = PLOT_DIR / "05_slice_count_boxplot.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Plot 6: kVp and tube current
# ============================================================
def plot_acquisition_params(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # kVp
    kvp = pd.to_numeric(df["kvp"], errors="coerce").dropna()
    if len(kvp) > 0:
        sns.histplot(kvp, bins=20, ax=axes[0], color="steelblue")
        axes[0].set_title("Tube Voltage (kVp)")
        axes[0].set_xlabel("kVp")

    # Tube current
    ma = pd.to_numeric(df["tube_current_mA"], errors="coerce").dropna()
    if len(ma) > 0:
        sns.histplot(ma, bins=30, ax=axes[1], color="coral")
        axes[1].set_title("Tube Current (mA)")
        axes[1].set_xlabel("mA")

    plt.tight_layout()
    out = PLOT_DIR / "06_acquisition_params.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Plot 7: Pixel spacing distribution
# ============================================================
def plot_pixel_spacing(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    px = pd.to_numeric(df["pixel_spacing_row_mm"], errors="coerce").dropna()

    if len(px) > 0:
        sns.histplot(px, bins=30, kde=True, ax=ax)
        ax.set_title("Pixel Spacing Distribution (Row)", fontsize=14)
        ax.set_xlabel("Pixel Spacing (mm)")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "No valid pixel spacing data", ha="center", va="center")

    plt.tight_layout()
    out = PLOT_DIR / "07_pixel_spacing.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Plot 8: Contrast vs non-contrast
# ============================================================
def plot_contrast_breakdown(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = df.copy()
    df_plot["has_contrast"] = df_plot["contrast_agent"].apply(
        lambda x: "With Contrast" if pd.notna(x) and str(x).strip() != "" else "No Contrast"
    )

    ct = df_plot.groupby(["site_folder", "has_contrast"]).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=ax, color=["#2196F3", "#FF5722"])
    ax.set_title("Contrast vs Non-Contrast Series per Site", fontsize=14)
    ax.set_xlabel("Site Folder")
    ax.set_ylabel("Number of Series")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = PLOT_DIR / "08_contrast_breakdown.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Plot 9: Manufacturer distribution
# ============================================================
def plot_manufacturers(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    mfr = df["manufacturer"].value_counts()

    if len(mfr) > 0:
        mfr.plot(kind="barh", ax=ax, color="teal")
        ax.set_title("CT Scanner Manufacturers", fontsize=14)
        ax.set_xlabel("Number of Series")
    else:
        ax.text(0.5, 0.5, "No manufacturer data", ha="center", va="center")

    plt.tight_layout()
    out = PLOT_DIR / "09_manufacturers.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Plot 10: Patient count per site
# ============================================================
def plot_patients_per_site(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    patients = df.groupby("site_folder")["patient_id"].nunique()
    patients.plot(kind="bar", ax=ax, color="mediumpurple")
    ax.set_title("Unique Patients per Site", fontsize=14)
    ax.set_xlabel("Site Folder")
    ax.set_ylabel("Number of Patients")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = PLOT_DIR / "10_patients_per_site.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Create analysis report Excel
# ============================================================
def create_report(df):
    """Create a multi-sheet Excel report with various summaries."""
    print(f"\nCreating analysis report: {OUT_REPORT}")

    with pd.ExcelWriter(OUT_REPORT, engine="openpyxl") as writer:
        # Sheet 1: Full series summary
        df.to_excel(writer, sheet_name="All_Series", index=False)

        # Sheet 2: Per-site summary
        site_summary = df.groupby("site_folder").agg(
            n_series=("series_uid", "count"),
            n_patients=("patient_id", "nunique"),
            total_slices=("n_slices", "sum"),
            n_casc=("series_type", lambda x: (x == "calcium_scoring").sum()),
            n_ccta=("series_type", lambda x: (x == "ccta").sum()),
            n_localizer=("series_type", lambda x: (x == "localizer").sum()),
            n_other=("series_type", lambda x: (x == "other").sum()),
        ).reset_index()
        site_summary.to_excel(writer, sheet_name="Site_Summary", index=False)

        # Sheet 3: Per-patient summary
        patient_summary = df.groupby("patient_id").agg(
            site_folder=("site_folder", "first"),
            n_series=("series_uid", "count"),
            total_slices=("n_slices", "sum"),
            series_types=("series_type", lambda x: ", ".join(sorted(set(x)))),
            has_casc=("series_type", lambda x: "calcium_scoring" in x.values),
            has_ccta=("series_type", lambda x: "ccta" in x.values),
            has_contrast=("contrast_agent", lambda x: any(
                pd.notna(v) and str(v).strip() != "" for v in x
            )),
        ).reset_index()
        patient_summary.to_excel(writer, sheet_name="Patient_Summary", index=False)

        # Sheet 4: Series type x site cross-tab
        crosstab = pd.crosstab(df["site_folder"], df["series_type"], margins=True)
        crosstab.to_excel(writer, sheet_name="Type_x_Site")

        # Sheet 5: Acquisition parameters summary
        param_cols = ["slice_thickness_mm", "pixel_spacing_row_mm", "kvp",
                      "tube_current_mA", "n_slices"]
        df_params = df[param_cols].apply(pd.to_numeric, errors="coerce")
        param_stats = df_params.describe()
        param_stats.to_excel(writer, sheet_name="Parameter_Stats")

        # Sheet 6: Kernel/reconstruction summary
        if "convolution_kernel" in df.columns:
            kernel_summary = df.groupby(["series_type", "convolution_kernel"]).size()
            kernel_summary = kernel_summary.reset_index(name="count")
            kernel_summary.to_excel(writer, sheet_name="Kernels", index=False)

        # Sheet 7: CASC-specific details
        df_casc = df[df["series_type"] == "calcium_scoring"].copy()
        if len(df_casc) > 0:
            df_casc.to_excel(writer, sheet_name="CASC_Details", index=False)

        # Sheet 8: CCTA-specific details
        df_ccta = df[df["series_type"] == "ccta"].copy()
        if len(df_ccta) > 0:
            df_ccta.to_excel(writer, sheet_name="CCTA_Details", index=False)

    print(f"  Saved: {OUT_REPORT}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    df = load_data()

    print("\n--- Generating Plots ---")
    plot_series_per_site(df)
    plot_slices_per_site(df)
    plot_series_type_pie(df)
    plot_slice_thickness(df)
    plot_slice_count_boxplot(df)
    plot_acquisition_params(df)
    plot_pixel_spacing(df)
    plot_contrast_breakdown(df)
    plot_manufacturers(df)
    plot_patients_per_site(df)

    print("\n--- Creating Report ---")
    create_report(df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll plots saved in: {PLOT_DIR}")
    print(f"Report saved: {OUT_REPORT}")

    # Print quick stats
    print(f"\n--- Quick Stats ---")
    print(f"Total series: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"\nSeries types:")
    print(df["series_type"].value_counts().to_string())
