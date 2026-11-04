"""Manuscript table of every measurement, radiomic feature and index.

For AF, HF and stroke against the matched controls, each feature gets the
median and interquartile range of both arms, the relative difference, the
rank AUC (the Mann-Whitney U divided by n_case x n_control, so no model and no
cross-validation is involved), the Mann-Whitney p and the Benjamini-Hochberg q
within the group. Features are labelled by block, measurement, radiomics of LA,
of LA-EAT or of the whole fat ROI, and index, so the table can be split in the
paper.

Input : <results>\\fat_la_features.xlsx and <results>\\radiomic_features.xlsx
Output: <output>\\significance_table.xlsx
            All           every feature, every group, sorted by q
            Significant   q below --alpha only
            Manuscript    one row per feature, the three groups side by side,
                          medians formatted as "median (p25-p75)"

Usage
    python 21_significance_table.py
    python 21_significance_table.py --alpha 0.01 --output "W:\\...\\results"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ANALYSIS = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast\analysis\results")
COMPARTMENTS = {"la": "LA", "laeat": "LA-EAT", "eat": "whole fat ROI"}
VOLUMES = {"W": "eat_volume_ml", "L": "la_volume_ml", "E": "laeat_volume_ml"}
NOT_A_FEATURE = {
    "patient_id",
    "group",
    "diagnoses",
    "status",
    "error",
    "voxel_ml",
    "slice_mm",
    "cleaned_mask",
    "sex",
    "age",
    "bmi",
}
CLINICAL_WORDS = ("hba1c", "cholesterol", "glucose", "bmi", "age", "sex", "pdff", "blood")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiomics", type=Path, default=ANALYSIS / "radiomic_features.xlsx")
    parser.add_argument("--measurements", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument("--alpha", type=float, default=0.05, help="q threshold")
    parser.add_argument("--decimals", type=int, default=2)
    return parser.parse_known_args(argv)[0]


def read_tables(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for path in (args.measurements, args.radiomics):
        if not path or not path.exists():
            print(f"missing {path}, skipped")
            continue
        table = pd.read_excel(path, sheet_name="Features")
        table.columns = [str(name).strip() for name in table.columns]
        frames.append(table)
    if not frames:
        raise SystemExit("neither workbook was found")
    table = frames[0]
    for other in frames[1:]:
        shared = [name for name in other.columns if name in table.columns and name != "patient_id"]
        table = table.merge(other.drop(columns=shared), on="patient_id", how="outer")
    if "group" not in table.columns:
        raise SystemExit("no group column, run scripts 11 and 14 with --groups")
    return table[table["group"].notna()].replace([np.inf, -np.inf], np.nan)


def add_indices(table: pd.DataFrame) -> list[str]:
    """The same combinations of the three component volumes as script 19."""
    have = {key: table[column] for key, column in VOLUMES.items() if column in table.columns}
    if len(have) < 2:
        return []
    W, L, E = have.get("W"), have.get("L"), have.get("E")

    def safe(series):
        return series.replace(0, np.nan)

    made: dict[str, pd.Series] = {}
    if W is not None and L is not None:
        made["cmi1_W_L"] = W * L
        made["ratio_W_over_L"] = W / safe(L)
        made["log_W_L"] = np.log1p(W * L)
    if W is not None and E is not None:
        made["prod_W_E"] = W * E
        made["ratio_E_over_W"] = E / safe(W)
    if L is not None and E is not None:
        made["prod_L_E"] = L * E
        made["ratio_E_over_L"] = E / safe(L)
    if W is not None and L is not None and E is not None:
        made["cmi2_W2_L01_E3"] = (W**2) * (L**0.1) * (E**3)
        made["prod_W_L_E"] = W * L * E
        made["sum_W_E_over_L"] = (W + E) / safe(L)
        made["W_over_L_plus_E"] = W / safe(L + E)
        made["sqrt_W_L_E"] = np.cbrt(W * L * E)
    for name, series in made.items():
        table[name] = series.replace([np.inf, -np.inf], np.nan)
    return list(made)


def block_of(name: str, indices: list[str]) -> str:
    if name in indices:
        return "index"
    for prefix, label in COMPARTMENTS.items():
        if name.startswith(prefix + "_"):
            return "measurement" if name.endswith("_volume_ml") else f"radiomics {label}"
    return "measurement"


def numeric_features(table: pd.DataFrame) -> list[str]:
    keep = []
    for name in table.columns:
        low = name.lower()
        if name in NOT_A_FEATURE or any(word in low for word in CLINICAL_WORDS):
            continue
        if not pd.api.types.is_numeric_dtype(table[name]):
            continue
        if table[name].notna().sum() < 10 or not table[name].std(skipna=True) > 0:
            continue
        keep.append(name)
    return keep


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    total = len(ranked)
    adjusted = ranked * total / np.arange(1, total + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1.0)
    corrected = np.empty(total)
    corrected[order] = adjusted
    return corrected


def describe(values: pd.Series, decimals: int) -> dict:
    p25, median, p75 = values.quantile([0.25, 0.5, 0.75])
    return {
        "n": len(values),
        "median": round(float(median), decimals),
        "p25": round(float(p25), decimals),
        "p75": round(float(p75), decimals),
        "text": f"{median:.{decimals}f} ({p25:.{decimals}f}-{p75:.{decimals}f})",
    }


def compare(table: pd.DataFrame, features: list[str], indices: list[str], args) -> pd.DataFrame:
    rows = []
    for group in ("AF", "HF", "stroke"):
        pair = table[table["group"].isin([group, "control"])]
        if pair["group"].nunique() < 2:
            print(f"{group}: no patients, skipped")
            continue
        for name in features:
            cases = pair.loc[pair["group"] == group, name].dropna()
            controls = pair.loc[pair["group"] == "control", name].dropna()
            if len(cases) < 3 or len(controls) < 3:
                continue
            test = mannwhitneyu(cases, controls, alternative="two-sided")
            auc = test.statistic / (len(cases) * len(controls))
            case = describe(cases, args.decimals)
            control = describe(controls, args.decimals)
            change = (
                100 * (case["median"] - control["median"]) / abs(control["median"])
                if control["median"]
                else np.nan
            )
            rows.append(
                {
                    "group": group,
                    "feature": name,
                    "block": block_of(name, indices),
                    "n_case": case["n"],
                    "n_control": control["n"],
                    "case_median_iqr": case["text"],
                    "control_median_iqr": control["text"],
                    "median_case": case["median"],
                    "median_control": control["median"],
                    "change_percent": round(float(change), 1),
                    "direction": "higher" if auc > 0.5 else "lower",
                    "auc": round(float(max(auc, 1 - auc)), 3),
                    "p_value": float(test.pvalue),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise SystemExit("no feature could be compared")
    result["q_fdr"] = np.nan
    for _, part in result.groupby("group"):
        result.loc[part.index, "q_fdr"] = benjamini_hochberg(
            part["p_value"].to_numpy(dtype=float)
        )
    result["significant"] = np.where(result["q_fdr"] < args.alpha, "yes", "no")
    return result.sort_values(["group", "q_fdr", "auc"], ascending=[True, True, False])


def manuscript_sheet(result: pd.DataFrame) -> pd.DataFrame:
    """One row per feature, the three comparisons side by side."""
    base = result.drop_duplicates("feature")[["feature", "block"]].set_index("feature")
    controls = result.drop_duplicates("feature").set_index("feature")["control_median_iqr"]
    wide = base.join(controls.rename("control median (IQR)"))
    for group in ("AF", "HF", "stroke"):
        part = result[result["group"] == group].set_index("feature")
        wide[f"{group} median (IQR)"] = part["case_median_iqr"]
        wide[f"{group} AUC"] = part["auc"]
        wide[f"{group} q"] = part["q_fdr"].map(
            lambda value: "<0.001" if value < 0.001 else f"{value:.3f}"
        )
    order = {"measurement": 0, "index": 1}
    wide["_sort"] = wide["block"].map(lambda name: order.get(name, 2))
    best = result.groupby("feature")["auc"].max()
    wide["_auc"] = best
    return wide.sort_values(["_sort", "_auc"], ascending=[True, False]).drop(
        columns=["_sort", "_auc"]
    ).reset_index()


def main(argv=None) -> None:
    args = parse_args(argv)
    table = read_tables(args)
    indices = add_indices(table)
    features = numeric_features(table)
    print(f"{len(table)} patients, {len(features)} features including {len(indices)} indices")
    print(table["group"].value_counts().to_string())

    result = compare(table, features, indices, args)
    significant = result[result["significant"] == "yes"]
    wide = manuscript_sheet(result)

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "significance_table.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="All", index=False)
        significant.to_excel(writer, sheet_name="Significant", index=False)
        wide.to_excel(writer, sheet_name="Manuscript", index=False)

    for group in ("AF", "HF", "stroke"):
        part = significant[significant["group"] == group]
        print(f"\n{group}: {len(part)} of {result['group'].eq(group).sum()} features q<{args.alpha}")
        columns = ["feature", "block", "case_median_iqr", "control_median_iqr", "auc", "q_fdr"]
        print(part.nlargest(15, "auc")[columns].to_string(index=False))
    print(f"\nExcel : {path}")


if __name__ == "__main__":
    main()
