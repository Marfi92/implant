"""Random forest on the fat/LA features: AF, HF and stroke against the controls.

Input is the Features sheet of 11_fat_hu_features_and_groups.py. For every group
one random forest is trained against the controls of the same sheet with repeated
stratified cross validation, so the reported AUC comes from patients the forest
never saw. Age, sex and BMI are added from the clinical CSV when available,
because epicardial fat grows with body size and the group effect has to survive
that.

Output: <output>\\random_forest.xlsx
            Performance   AUC, accuracy, sensitivity, specificity per group
            Importance    mean feature importance per group
            Predictions   out-of-fold probability per patient
        <output>\\plots\\rf_<group>_roc.png
        <output>\\plots\\rf_<group>_importance.png
        <output>\\models\\rf_<group>.joblib   forest trained on all patients

Apply a saved forest to new patients (their features come from script 11):

    python 12_random_forest_groups.py --apply AF ^
        --features "Q:\\...\\new_cases\\fat_la_features.xlsx" ^
        --models "Q:\\...\\test80_analysis\\models"

which writes <features folder>\\rf_predictions_AF.xlsx with one probability per
patient and nothing else touched.

Usage
    python 12_random_forest_groups.py
    python 12_random_forest_groups.py --groups AF HF --trees 1000
    python 12_random_forest_groups.py --apply AF --features ... --models ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"Q:\users\marfi\Automatic segmentations\finalllll\final\test80_analysis")
FEATURES_FILE = BASE / "fat_la_features.xlsx"
CLINICAL = Path(r"Q:\users\marfi\CT_analysis\SCAPIS1-DATA-PETITION-1110-20260407(in).csv")

IMAGE_FEATURES = [
    "fat_volume_ml",
    "fat_volume_raw_ml",
    "fat_kept_fraction",
    "fat_mean_hu",
    "fat_sd_hu",
    "fat_p25_hu",
    "fat_p75_hu",
    "la_volume_ml",
    "la_mean_hu",
    "fat_to_la_ratio",
    "fat_index_ml_per_cm",
]
CLINICAL_COLUMNS = {"age": "AgeAtVisitOne", "sex": "Sex", "height": "Height", "weight": "Weight"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_FILE)
    parser.add_argument("--sheet", default="Features")
    parser.add_argument("--clinical", type=Path, default=CLINICAL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", type=Path, default=None)
    parser.add_argument("--groups", nargs="*", default=["AF", "HF", "stroke"])
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--no-clinical", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--apply", default="")
    return parser.parse_known_args(argv)[0]


def add_clinical(table: pd.DataFrame, clinical_file: Path) -> tuple[pd.DataFrame, list[str]]:
    """Age, sex and BMI beside the image features, when the CSV is there."""
    if not clinical_file.exists():
        return table, []
    clinical = pd.read_csv(clinical_file, sep=None, engine="python", dtype=str)
    if "Subject" not in clinical:
        return table, []
    keep = {"Subject": "patient_id"}
    for name, column in CLINICAL_COLUMNS.items():
        if column in clinical:
            keep[column] = name
    clinical = clinical[list(keep)].rename(columns=keep)
    clinical["patient_id"] = clinical["patient_id"].astype(str)
    for name in ("age", "height", "weight"):
        if name in clinical:
            clinical[name] = pd.to_numeric(clinical[name], errors="coerce")
    if "sex" in clinical:
        clinical["sex_male"] = (
            clinical["sex"].astype(str).str.strip().str.upper().isin(["MALE", "M", "1"]).astype(int)
        )
    if {"height", "weight"} <= set(clinical):
        metres = clinical["height"] / 100.0
        clinical["bmi"] = (clinical["weight"] / metres.pow(2)).round(2)
    extra = [name for name in ("age", "sex_male", "bmi") if name in clinical]
    merged = table.merge(clinical[["patient_id"] + extra], on="patient_id", how="left")
    return merged, extra


def feature_columns(table: pd.DataFrame, extra: list[str]) -> list[str]:
    """Features present, numeric, and not constant or mostly empty."""
    columns = []
    for name in IMAGE_FEATURES + extra:
        if name not in table:
            continue
        values = pd.to_numeric(table[name], errors="coerce")
        if values.notna().sum() < max(5, 0.5 * len(values)) or values.nunique() < 2:
            continue
        columns.append(name)
    return columns


def design(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Numeric matrix with the median filled in for the few empty cells."""
    frame = table[columns].apply(pd.to_numeric, errors="coerce")
    return frame.fillna(frame.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)


def forest(trees: int, seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=trees,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def evaluate(group: str, table: pd.DataFrame, columns: list[str], args) -> dict[str, object]:
    """Repeated cross validated forest, group against the controls."""
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import RepeatedStratifiedKFold

    part = table[table["group"].isin([group, "control"])].copy()
    labels = (part["group"] == group).astype(int).to_numpy()
    matrix = design(part, columns)
    if labels.sum() < 5 or (labels == 0).sum() < 5:
        return {"group": group, "note": "too few patients", "n_group": int(labels.sum())}

    folds = min(args.folds, int(labels.sum()), int((labels == 0).sum()))
    splitter = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=args.repeats, random_state=0
    )
    probability = np.zeros(len(labels))
    seen = np.zeros(len(labels))
    importance = np.zeros(len(columns))
    aucs = []
    for seed, (train, test) in enumerate(splitter.split(matrix, labels)):
        model = forest(args.trees, seed).fit(matrix[train], labels[train])
        scores = model.predict_proba(matrix[test])[:, 1]
        probability[test] += scores
        seen[test] += 1
        importance += model.feature_importances_
        if len(set(labels[test])) == 2:
            aucs.append(roc_auc_score(labels[test], scores))
    probability = np.divide(probability, np.maximum(seen, 1))
    splits = seed + 1

    predicted = (probability >= 0.5).astype(int)
    positives = labels == 1
    result = {
        "group": group,
        "n_group": int(labels.sum()),
        "n_control": int((labels == 0).sum()),
        "features": len(columns),
        "auc_out_of_fold": round(float(roc_auc_score(labels, probability)), 3),
        "auc_mean_per_fold": round(float(np.mean(aucs)), 3) if aucs else float("nan"),
        "auc_sd_per_fold": round(float(np.std(aucs)), 3) if aucs else float("nan"),
        "accuracy": round(float((predicted == labels).mean()), 3),
        "sensitivity": round(float(predicted[positives].mean()), 3),
        "specificity": round(float(1 - predicted[~positives].mean()), 3),
    }
    curve = roc_curve(labels, probability)
    ranking = pd.DataFrame(
        {"feature": columns, "importance": (importance / splits).round(4)}
    ).sort_values("importance", ascending=False)
    predictions = pd.DataFrame(
        {
            "patient_id": part["patient_id"].to_numpy(),
            "group": part["group"].to_numpy(),
            "is_case": labels,
            "probability": probability.round(3),
        }
    ).sort_values("probability", ascending=False)
    return {**result, "_curve": curve, "_importance": ranking, "_predictions": predictions}


def draw(group: str, result: dict, folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    false_positive, true_positive, _ = result["_curve"]
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(false_positive, true_positive, label=f"AUC {result['auc_out_of_fold']}")
    axis.plot([0, 1], [0, 1], "--", color="grey")
    axis.set_xlabel("1 - specificity")
    axis.set_ylabel("sensitivity")
    axis.set_title(f"{group} vs control, out of fold")
    axis.legend()
    figure.tight_layout()
    figure.savefig(folder / f"rf_{group}_roc.png", dpi=150)
    plt.close(figure)

    ranking = result["_importance"].head(12).iloc[::-1]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.barh(ranking["feature"], ranking["importance"])
    axis.set_title(f"{group}: feature importance")
    figure.tight_layout()
    figure.savefig(folder / f"rf_{group}_importance.png", dpi=150)
    plt.close(figure)


def load_features(args) -> tuple[pd.DataFrame, list[str]]:
    table = pd.read_excel(args.features, sheet_name=args.sheet)
    table["patient_id"] = table["patient_id"].astype(str)
    if "status" in table:
        table = table[table["status"] == "ok"]
    extra: list[str] = []
    if not args.no_clinical:
        table, extra = add_clinical(table, args.clinical)
    return table, extra


def apply_saved_model(args) -> None:
    """Score new patients with a forest trained earlier."""
    from joblib import load

    models = args.models or (args.features.parent / "models")
    bundle = load(models / f"rf_{args.apply}.joblib")
    table, _ = load_features(args)
    columns = bundle["features"]
    missing = [name for name in columns if name not in table]
    if missing:
        raise SystemExit(f"these features are missing from {args.features}: {missing}")
    probability = bundle["model"].predict_proba(design(table, columns))[:, 1]
    out = pd.DataFrame(
        {"patient_id": table["patient_id"].to_numpy(), f"probability_{args.apply}": probability.round(3)}
    ).sort_values(f"probability_{args.apply}", ascending=False)
    if "group" in table:
        out = out.merge(table[["patient_id", "group"]], on="patient_id", how="left")
    target = (args.output or args.features.parent) / f"rf_predictions_{args.apply}.xlsx"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(target, index=False)
    print(out.head(10).to_string(index=False))
    print(f"\n{len(out)} patients scored with rf_{args.apply}.joblib")
    print(f"Excel : {target}")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.apply:
        apply_saved_model(args)
        return

    output = args.output or args.features.parent
    table, extra = load_features(args)
    columns = feature_columns(table, extra)
    print(f"{len(table)} patients, {len(columns)} features: {', '.join(columns)}")
    print(table["group"].value_counts().to_string())

    from joblib import dump

    performance, importances, predictions = [], [], []
    for group in args.groups:
        if group not in set(table["group"]):
            print(f"\n{group}: not in the sheet, skipped")
            continue
        result = evaluate(group, table, columns, args)
        if "_curve" not in result:
            print(f"\n{group}: {result.get('note')}")
            performance.append({key: value for key, value in result.items() if not key.startswith("_")})
            continue
        performance.append({key: value for key, value in result.items() if not key.startswith("_")})
        importances.append(result["_importance"].assign(group=group))
        predictions.append(result["_predictions"])
        print(
            f"\n{group} vs control: AUC {result['auc_out_of_fold']} "
            f"(per fold {result['auc_mean_per_fold']} +- {result['auc_sd_per_fold']}), "
            f"sens {result['sensitivity']}, spec {result['specificity']}"
        )
        print(result["_importance"].head(5).to_string(index=False))
        if not args.no_plots:
            draw(group, result, output / "plots")
        part = table[table["group"].isin([group, "control"])]
        model = forest(args.trees).fit(
            design(part, columns), (part["group"] == group).astype(int).to_numpy()
        )
        (output / "models").mkdir(parents=True, exist_ok=True)
        dump({"model": model, "features": columns, "group": group}, output / "models" / f"rf_{group}.joblib")

    path = output / "random_forest.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(performance).to_excel(writer, sheet_name="Performance", index=False)
        if importances:
            pd.concat(importances).to_excel(writer, sheet_name="Importance", index=False)
        if predictions:
            pd.concat(predictions).to_excel(writer, sheet_name="Predictions", index=False)
    print(f"\nExcel  : {path}")
    print(f"Models : {output / 'models'}")
    if not args.no_plots:
        print(f"Plots  : {output / 'plots'}")


if __name__ == "__main__":
    main()
