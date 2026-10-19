"""Compare classifiers on the fat/LA features: logistic regression, forest, boosting, SVM, kNN.

Same 5 fold stratified cross validation, repeated --repeats times, same splits
for every model and every group, so the AUCs can be read against each other.
Continuous features are standardised inside each fold (inside the pipeline, so
the test fold never contributes to the mean and SD).

Beside the measured features, combinations are derived, because a raw fat volume
mostly tells you how big the patient is:

    fat_per_bmi          fat_volume_ml / bmi
    fat_per_la           fat_volume_ml / la_volume_ml
    fat_per_year         fat_volume_ml / age
    la_per_bmi           la_volume_ml / bmi
    fat_hu_iqr           fat_p75_hu - fat_p25_hu
    fat_hu_skew          (fat_mean_hu - median proxy) / fat_sd_hu
    fat_dense_fraction   how much of the fat sits in the dense (-190..-100) half
    la_to_fat_hu         la_mean_hu / fat_mean_hu
    fat_volume_z         fat volume as a z score inside the sheet

Output: <output>\\model_comparison.xlsx
            Performance    AUC mean/SD, accuracy, sensitivity, specificity per model and group
            Best           the best model per group
            Importance     coefficients / importances of the best model per group
            Predictions    out of fold probability per patient for the best model
        <output>\\plots\\compare_<group>_auc.png   model AUCs with their spread
        <output>\\plots\\compare_<group>_roc.png   ROC per model

Usage
    python 13_compare_models.py
    python 13_compare_models.py --folds 5 --repeats 10
    python 13_compare_models.py --no-derived --groups AF
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"Q:\users\marfi\Automatic segmentations\finalllll\final\test80_analysis")
FEATURES_FILE = BASE / "fat_la_features.xlsx"
CLINICAL = Path(r"Q:\users\marfi\CT_analysis\SCAPIS1-DATA-PETITION-1110-20260407(in).csv")


def load_random_forest_module():
    """Script 12 holds the clinical join and the feature matrix helpers."""
    import importlib.util
    import sys

    name = "12_random_forest_groups"
    if name in sys.modules:
        return sys.modules[name]
    for folder in (Path(__file__).resolve().parent, Path.cwd()):
        path = folder / f"{name}.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    raise SystemExit(f"{name}.py not found next to this script or in {Path.cwd()}")


random_forest_module = load_random_forest_module()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_FILE)
    parser.add_argument("--sheet", default="Features")
    parser.add_argument("--clinical", type=Path, default=CLINICAL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--groups", nargs="*", default=["AF", "HF", "stroke"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--no-clinical", action="store_true")
    parser.add_argument("--no-derived", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_known_args(argv)[0]


def add_derived(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ratios and spreads built from the measured columns."""
    table = table.copy()
    number = lambda name: pd.to_numeric(table[name], errors="coerce") if name in table else None
    fat, raw = number("fat_volume_ml"), number("fat_volume_raw_ml")
    atrium, bmi, age = number("la_volume_ml"), number("bmi"), number("age")
    mean_hu, sd_hu = number("fat_mean_hu"), number("fat_sd_hu")
    low_hu, high_hu = number("fat_p25_hu"), number("fat_p75_hu")
    la_hu = number("la_mean_hu")

    safe = lambda series: series.replace(0, np.nan) if series is not None else None
    derived: dict[str, pd.Series] = {}
    if fat is not None and bmi is not None:
        derived["fat_per_bmi"] = fat / safe(bmi)
    if fat is not None and atrium is not None:
        derived["fat_per_la"] = fat / safe(atrium)
    if fat is not None and age is not None:
        derived["fat_per_year"] = fat / safe(age)
    if atrium is not None and bmi is not None:
        derived["la_per_bmi"] = atrium / safe(bmi)
    if low_hu is not None and high_hu is not None:
        derived["fat_hu_iqr"] = high_hu - low_hu
    if mean_hu is not None and sd_hu is not None and low_hu is not None and high_hu is not None:
        derived["fat_hu_skew"] = (mean_hu - (low_hu + high_hu) / 2) / safe(sd_hu)
    if mean_hu is not None and low_hu is not None and high_hu is not None:
        span = safe(high_hu - low_hu)
        derived["fat_dense_fraction"] = (mean_hu - low_hu) / span
    if la_hu is not None and mean_hu is not None:
        derived["la_to_fat_hu"] = la_hu / safe(mean_hu)
    if fat is not None and raw is not None:
        derived["fat_density_ratio"] = fat / safe(raw)
    if fat is not None and fat.std(skipna=True):
        derived["fat_volume_z"] = (fat - fat.mean()) / fat.std()
    for name, values in derived.items():
        table[name] = values
    return table, list(derived)


def models(args) -> dict[str, object]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    scaled = lambda estimator: Pipeline(
        [("scale", StandardScaler()), ("model", estimator)]
    )
    return {
        "logistic_l2": scaled(
            LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
        ),
        "logistic_elasticnet": scaled(
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                penalty="elasticnet",
                l1_ratio=0.5,
                C=0.5,
                solver="saga",
            )
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=args.trees,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=0,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0
        ),
        "svm_rbf": scaled(SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0)),
        "knn": scaled(KNeighborsClassifier(n_neighbors=7)),
    }


def importance_of(name: str, model, columns: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        kind = "importance"
    elif hasattr(estimator, "coef_"):
        values = estimator.coef_.ravel()
        kind = "coefficient"
    else:
        return pd.DataFrame()
    frame = pd.DataFrame({"feature": columns, kind: np.round(values, 4)})
    frame["abs"] = frame[kind].abs()
    return frame.sort_values("abs", ascending=False).drop(columns="abs").assign(model=name)


def evaluate_group(group: str, table: pd.DataFrame, columns: list[str], args) -> dict:
    """Every model on the same repeated 5 fold splits of this group vs control."""
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import RepeatedStratifiedKFold

    part = table[table["group"].isin([group, "control"])].copy()
    labels = (part["group"] == group).astype(int).to_numpy()
    matrix = random_forest_module.design(part, columns)
    if labels.sum() < 5 or (labels == 0).sum() < 5:
        return {"rows": [{"group": group, "note": "too few patients"}]}

    folds = min(args.folds, int(labels.sum()), int((labels == 0).sum()))
    splits = list(
        RepeatedStratifiedKFold(
            n_splits=folds, n_repeats=args.repeats, random_state=0
        ).split(matrix, labels)
    )
    rows, curves, probabilities, fitted = [], {}, {}, {}
    for name, estimator in models(args).items():
        probability = np.zeros(len(labels))
        seen = np.zeros(len(labels))
        aucs = []
        for train, test in splits:
            model = clone(estimator).fit(matrix[train], labels[train])
            scores = model.predict_proba(matrix[test])[:, 1]
            probability[test] += scores
            seen[test] += 1
            if len(set(labels[test])) == 2:
                aucs.append(roc_auc_score(labels[test], scores))
        probability = np.divide(probability, np.maximum(seen, 1))
        predicted = (probability >= 0.5).astype(int)
        positives = labels == 1
        rows.append(
            {
                "group": group,
                "model": name,
                "n_group": int(labels.sum()),
                "n_control": int((labels == 0).sum()),
                "features": len(columns),
                "auc_out_of_fold": round(float(roc_auc_score(labels, probability)), 3),
                "auc_mean_per_fold": round(float(np.mean(aucs)), 3),
                "auc_sd_per_fold": round(float(np.std(aucs)), 3),
                "accuracy": round(float((predicted == labels).mean()), 3),
                "sensitivity": round(float(predicted[positives].mean()), 3),
                "specificity": round(float(1 - predicted[~positives].mean()), 3),
            }
        )
        curves[name] = roc_curve(labels, probability)
        probabilities[name] = probability
        fitted[name] = clone(estimator).fit(matrix, labels)
    best = max(rows, key=lambda row: row["auc_mean_per_fold"])["model"]
    predictions = pd.DataFrame(
        {
            "patient_id": part["patient_id"].to_numpy(),
            "group": part["group"].to_numpy(),
            "is_case": labels,
            "model": best,
            "probability": probabilities[best].round(3),
        }
    ).sort_values("probability", ascending=False)
    return {
        "rows": rows,
        "curves": curves,
        "best": best,
        "importance": importance_of(best, fitted[best], columns),
        "predictions": predictions,
        "labels": labels,
    }


def draw(group: str, result: dict, rows: pd.DataFrame, folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    part = rows[rows["group"] == group].sort_values("auc_mean_per_fold")
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.barh(part["model"], part["auc_mean_per_fold"], xerr=part["auc_sd_per_fold"], capsize=3)
    axis.axvline(0.5, color="grey", linestyle="--")
    axis.set_xlim(0.3, 1.0)
    axis.set_xlabel("AUC, mean over folds")
    axis.set_title(f"{group} vs control")
    figure.tight_layout()
    figure.savefig(folder / f"compare_{group}_auc.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 5))
    for name, (false_positive, true_positive, _) in result["curves"].items():
        auc = part.loc[part["model"] == name, "auc_out_of_fold"].iloc[0]
        axis.plot(false_positive, true_positive, label=f"{name} {auc}")
    axis.plot([0, 1], [0, 1], "--", color="grey")
    axis.set_xlabel("1 - specificity")
    axis.set_ylabel("sensitivity")
    axis.set_title(f"{group} vs control, out of fold")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(folder / f"compare_{group}_roc.png", dpi=150)
    plt.close(figure)


def main(argv=None) -> None:
    args = parse_args(argv)
    output = args.output or args.features.parent
    table = pd.read_excel(args.features, sheet_name=args.sheet)
    table["patient_id"] = table["patient_id"].astype(str)
    if "status" in table:
        table = table[table["status"] == "ok"]
    extra: list[str] = []
    if not args.no_clinical:
        table, extra = random_forest_module.add_clinical(table, args.clinical)
    derived: list[str] = []
    if not args.no_derived:
        table, derived = add_derived(table)

    columns = random_forest_module.feature_columns(table, extra + derived)
    print(f"{len(table)} patients, {len(columns)} features")
    print(f"derived: {', '.join(name for name in derived if name in columns) or 'none kept'}")
    print(table["group"].value_counts().to_string())

    performance, importances, predictions, results = [], [], [], {}
    for group in args.groups:
        if group not in set(table["group"]):
            print(f"\n{group}: not in the sheet, skipped")
            continue
        result = evaluate_group(group, table, columns, args)
        performance.extend(result["rows"])
        if "curves" not in result:
            print(f"\n{group}: {result['rows'][0].get('note')}")
            continue
        results[group] = result
        importances.append(result["importance"].assign(group=group))
        predictions.append(result["predictions"])
        frame = pd.DataFrame(result["rows"]).sort_values("auc_mean_per_fold", ascending=False)
        print(f"\n{group} vs control ({args.folds} fold x {args.repeats})")
        print(
            frame[
                ["model", "auc_mean_per_fold", "auc_sd_per_fold", "auc_out_of_fold",
                 "sensitivity", "specificity"]
            ].to_string(index=False)
        )
        print(f"best: {result['best']}")

    rows = pd.DataFrame(performance)
    best = (
        rows.sort_values("auc_mean_per_fold", ascending=False)
        .drop_duplicates("group")
        .sort_values("group")
        if "auc_mean_per_fold" in rows
        else pd.DataFrame()
    )
    if not args.no_plots:
        for group, result in results.items():
            draw(group, result, rows, output / "plots")

    output.mkdir(parents=True, exist_ok=True)
    path = output / "model_comparison.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        rows.to_excel(writer, sheet_name="Performance", index=False)
        if not best.empty:
            best.to_excel(writer, sheet_name="Best", index=False)
        if importances:
            pd.concat(importances).to_excel(writer, sheet_name="Importance", index=False)
        if predictions:
            pd.concat(predictions).to_excel(writer, sheet_name="Predictions", index=False)
    print(f"\nExcel : {path}")
    if not args.no_plots:
        print(f"Plots : {output / 'plots'}")


if __name__ == "__main__":
    main()
