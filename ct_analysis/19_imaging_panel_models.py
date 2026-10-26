"""Image-only panel: top radiomics per compartment, measurements, indices, 4-feature models.

No sex, no age, no BMI, no blood biomarkers. Everything comes from the two
workbooks written earlier:

    fat_la_features.xlsx    script 11, volumes and attenuation after the HU window
    radiomic_features.xlsx  script 14, radiomics of LA, LA-EAT and whole EAT

Three feature blocks are built per group (AF, HF, stroke, each against the
controls of the same sheet):

    radiomics      the --top-radiomics best features of each compartment by
                   cross-validated single-feature AUC, 5 + 5 + 5 by default
    measurements   the three component volumes plus attenuation, spread and
                   the ratios between the components
    indices        mathematical combinations of the three volumes,
                   W whole EAT, L left atrium, E LA-EAT:
                       cmi1 = W * L
                       cmi2 = W**2 * L**0.1 * E**3
                       W*E, L*E, W*L*E, E/W, L/W, E/L, (W+E)/L, W/(L+E),
                       log and square root forms of the products

Then every combination of exactly --max-size features out of the best --pool
features is scored with logistic regression, and the best --refit of them are
scored again with random forest and SVM, so the winner is not a model artefact.
AUC is the out-of-fold AUC of a stratified --folds split, repeated --repeats
times, which is the number to report.

Output: <output>\\panel_models.xlsx
            Univariate      every feature, block, AUC, Mann-Whitney p
            Top_Radiomics   the selected features per compartment
            Best_<group>    the ranked 4-feature combinations
            Best_Models     one row per group, the winning combination and model
        <output>\\plots_panel\\*.png   ROC of the winner, AUC bars, block comparison

Usage
    python 19_imaging_panel_models.py --radiomics "W:\\...\\results\\radiomic_features.xlsx"
    python 19_imaging_panel_models.py --max-size 4 --pool 20 --models logistic,forest,svm
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ANALYSIS = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast\analysis\results")
COMPARTMENTS = {"la": "LA", "laeat": "LA-EAT", "eat": "whole EAT"}
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
    parser.add_argument("--top-radiomics", type=int, default=5, help="per compartment")
    parser.add_argument("--pool", type=int, default=20, help="features entering the search")
    parser.add_argument("--max-size", type=int, default=4, help="features per model")
    parser.add_argument("--models", default="logistic,forest,svm")
    parser.add_argument("--refit", type=int, default=30, help="top combinations re-scored")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_known_args(argv)[0]


def build_model(name: str, seed: int):
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        )
    if name == "forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1),
        )
    if name == "svm":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(probability=True, random_state=seed),
        )
    raise SystemExit(f"unknown model {name}, use logistic, forest or svm")


def read_tables(args: argparse.Namespace) -> pd.DataFrame:
    """Join the two workbooks on patient_id, keeping the group column."""
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
    table = table[table["group"].notna()]
    return table.replace([np.inf, -np.inf], np.nan)


def add_indices(table: pd.DataFrame) -> list[str]:
    """Mathematical combinations of the three component volumes."""
    names = {
        "W": "eat_volume_ml",
        "L": "la_volume_ml",
        "E": "laeat_volume_ml",
    }
    have = {key: table[column] for key, column in names.items() if column in table.columns}
    if len(have) < 2:
        return []
    W = have.get("W")
    L = have.get("L")
    E = have.get("E")

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
    for prefix in COMPARTMENTS:
        if name.startswith(prefix + "_"):
            return "measurement" if name.endswith("_volume_ml") else f"radiomics {prefix}"
    return "measurement"


def numeric_features(table: pd.DataFrame) -> list[str]:
    keep = []
    for name in table.columns:
        low = name.lower()
        if name in NOT_A_FEATURE or any(word in low for word in CLINICAL_WORDS):
            continue
        if not pd.api.types.is_numeric_dtype(table[name]):
            continue
        if table[name].notna().sum() < 10 or table[name].std(skipna=True) in (0, np.nan):
            continue
        keep.append(name)
    return keep


def score(
    data: pd.DataFrame, target: np.ndarray, model_name: str, args: argparse.Namespace
) -> tuple[float, np.ndarray]:
    """Out-of-fold AUC, averaged over the repeats."""
    values = data.to_numpy(dtype=float)
    aucs, last = [], np.zeros(len(target))
    for repeat in range(args.repeats):
        folds = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed + repeat
        )
        model = build_model(model_name, args.seed + repeat)
        predicted = cross_val_predict(
            model, values, target, cv=folds, method="predict_proba"
        )[:, 1]
        aucs.append(roc_auc_score(target, predicted))
        last = predicted
    return float(np.mean(aucs)), last


def univariate(
    table: pd.DataFrame, features: list[str], indices: list[str], args: argparse.Namespace
) -> pd.DataFrame:
    rows = []
    for group in ("AF", "HF", "stroke"):
        pair = table[table["group"].isin([group, "control"])]
        if pair["group"].nunique() < 2 or len(pair) < 4 * args.folds:
            print(f"{group}: too few patients, skipped")
            continue
        target = (pair["group"] == group).to_numpy(dtype=int)
        for name in features:
            column = pair[[name]]
            if column[name].notna().sum() < len(pair) * 0.5:
                continue
            auc, _ = score(column, target, "logistic", args)
            cases = pair.loc[target == 1, name].dropna()
            controls = pair.loc[target == 0, name].dropna()
            p = (
                mannwhitneyu(cases, controls).pvalue
                if len(cases) > 2 and len(controls) > 2
                else np.nan
            )
            rows.append(
                {
                    "group": group,
                    "feature": name,
                    "block": block_of(name, indices),
                    "auc": round(max(auc, 1 - auc), 4),
                    "median_case": round(float(cases.median()), 4) if len(cases) else np.nan,
                    "median_control": round(float(controls.median()), 4)
                    if len(controls)
                    else np.nan,
                    "p_value": p,
                }
            )
    return pd.DataFrame(rows)


def pick_pool(single: pd.DataFrame, group: str, args: argparse.Namespace) -> tuple[list, pd.DataFrame]:
    """Top radiomics per compartment plus every measurement and index, then the best --pool."""
    table = single[single["group"] == group]
    chosen: list[str] = []
    picked_rows = []
    for prefix in COMPARTMENTS:
        block = table[table["block"] == f"radiomics {prefix}"].nlargest(args.top_radiomics, "auc")
        chosen += block["feature"].tolist()
        picked_rows.append(block)
    rest = table[table["block"].isin(["measurement", "index"])]
    chosen += rest["feature"].tolist()
    ranked = table[table["feature"].isin(chosen)].nlargest(args.pool, "auc")
    return ranked["feature"].tolist(), pd.concat(picked_rows) if picked_rows else pd.DataFrame()


def search(
    table: pd.DataFrame, group: str, pool: list[str], args: argparse.Namespace
) -> tuple[pd.DataFrame, dict]:
    pair = table[table["group"].isin([group, "control"])]
    target = (pair["group"] == group).to_numpy(dtype=int)
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    combinations = list(itertools.combinations(pool, args.max_size))
    print(f"{group}: {len(pair)} patients, {len(pool)} features, {len(combinations)} combinations")

    rows = []
    for names in combinations:
        auc, _ = score(pair[list(names)], target, "logistic", args)
        rows.append({"features": " + ".join(names), "logistic": round(auc, 4), "_names": names})
    ranked = pd.DataFrame(rows).sort_values("logistic", ascending=False).reset_index(drop=True)

    best = {"group": group, "auc": 0.0}
    for position, row in ranked.head(args.refit).iterrows():
        for model_name in models:
            if model_name == "logistic":
                auc, predicted = row["logistic"], None
            else:
                auc, predicted = score(pair[list(row["_names"])], target, model_name, args)
                ranked.loc[position, model_name] = round(auc, 4)
            if auc > best["auc"]:
                if predicted is None:
                    auc, predicted = score(pair[list(row["_names"])], target, model_name, args)
                best = {
                    "group": group,
                    "model": model_name,
                    "features": row["features"],
                    "auc": round(auc, 4),
                    "n_cases": int(target.sum()),
                    "n_controls": int(len(target) - target.sum()),
                    "_target": target,
                    "_predicted": predicted,
                }
    return ranked.drop(columns="_names"), best


def plot_results(single: pd.DataFrame, winners: list[dict], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    for best in winners:
        if "_predicted" not in best:
            continue
        fpr, tpr, _ = roc_curve(best["_target"], best["_predicted"])
        plt.plot(fpr, tpr, label=f"{best['group']} {best['model']} AUC {best['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("1 - specificity")
    plt.ylabel("sensitivity")
    plt.title("Best imaging-only models, out-of-fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "roc_best_models.png", dpi=200)
    plt.close()

    for group, table in single.groupby("group"):
        top = table.nlargest(15, "auc").sort_values("auc")
        plt.figure(figsize=(7, 6))
        colours = {"index": "tab:orange", "measurement": "tab:green"}
        plt.barh(
            top["feature"],
            top["auc"],
            color=[colours.get(block, "tab:blue") for block in top["block"]],
        )
        plt.xlabel("single feature AUC")
        plt.title(f"{group} vs control, best single features")
        plt.tight_layout()
        plt.savefig(folder / f"single_feature_auc_{group}.png", dpi=200)
        plt.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    table = read_tables(args)
    indices = add_indices(table)
    features = numeric_features(table)
    print(f"{len(table)} patients, {len(features)} image-only features, {len(indices)} indices")
    print(table["group"].value_counts().to_string())

    single = univariate(table, features, indices, args)
    if single.empty:
        raise SystemExit("no group could be compared")

    sheets: dict[str, pd.DataFrame] = {"Univariate": single.sort_values(["group", "auc"], ascending=[True, False])}
    picked, winners = [], []
    for group in single["group"].unique():
        pool, chosen = pick_pool(single, group, args)
        picked.append(chosen)
        ranked, best = search(table, group, pool, args)
        sheets[f"Best_{group}"] = ranked.head(200)
        if "model" in best:
            winners.append(best)
            print(f"{group}: best {best['model']} AUC {best['auc']} with {best['features']}")

    sheets["Top_Radiomics"] = pd.concat(picked) if picked else pd.DataFrame()
    sheets["Best_Models"] = pd.DataFrame(
        [{key: value for key, value in best.items() if not key.startswith("_")} for best in winners]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    workbook = args.output / "panel_models.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"Excel : {workbook}")

    if not args.no_plots:
        plot_results(single, winners, args.output / "plots_panel")
        print(f"Plots : {args.output / 'plots_panel'}")


if __name__ == "__main__":
    main()
