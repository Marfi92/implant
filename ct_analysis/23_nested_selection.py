"""Honest AUC: the feature selection happens inside the cross-validation.

Scripts 19 and 22 search thousands of combinations and report the best, which is
optimistic by roughly 0.01 to 0.03 because the winner was chosen on the same
patients it is scored on. Here the whole pipeline, rank the features, drop the
redundant ones, keep the best --sizes of them, fit the model, is repeated inside
every training fold, and the test fold never takes part in the choice. That is
the number for the manuscript.

Selection inside a fold:
    1. rank every feature by its AUC on the training patients only
    2. walk down the ranking and drop a feature correlated above --max-r with
       one already kept, so the panel is complementary rather than the same
       measurement three times
    3. keep the first k, fit the model, predict the held-out fold

Input : <results>\\fat_la_features.xlsx and a radiomics workbook, the --full one
        from script 14 if it exists
Output: <output>\\nested_models.xlsx
            Curve      AUC and SD per group, model and panel size
            Chosen     how often each feature was selected, over all folds
            Best       the best size per group and model
        <output>\\plots_index\\nested_curve.png

Usage
    python 23_nested_selection.py --group AF --sizes 1,2,3,4,5,6,8,10
    python 23_nested_selection.py --models logistic,gboost --repeats 3
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ANALYSIS = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast\analysis\results")
COMPARTMENTS = ("la", "laeat", "eat")
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
    parser.add_argument("--radiomics", type=Path, default=ANALYSIS / "radiomic_features_full.xlsx")
    parser.add_argument("--measurements", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument("--group", default="AF", help="AF, HF, stroke or all")
    parser.add_argument("--sizes", default="1,2,3,4,5,6,8,10")
    parser.add_argument("--models", default="logistic,gboost,forest")
    parser.add_argument("--max-r", type=float, default=0.9, help="drop a feature above this |r|")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
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
    have = {key: table[column] for key, column in VOLUMES.items() if column in table.columns}
    if len(have) < 3:
        return []
    W, L, E = have["W"], have["L"], have["E"]
    safe = E.replace(0, np.nan)
    made = {
        "cmi1_W_L": W * L,
        "cmi2_W2_L01_E3": (W**2) * (L**0.1) * (E**3),
        "prod_W_L_E": W * L * E,
        "index_W_L_over_E": (W * L) / safe,
        "ratio_E_over_L": E / L.replace(0, np.nan),
        "ratio_W_over_L": W / L.replace(0, np.nan),
    }
    for name, series in made.items():
        table[name] = series.replace([np.inf, -np.inf], np.nan)
    return list(made)


def numeric_features(table: pd.DataFrame) -> list[str]:
    keep = []
    for name in table.columns:
        low = name.lower()
        if name in NOT_A_FEATURE or any(word in low for word in CLINICAL_WORDS):
            continue
        if not pd.api.types.is_numeric_dtype(table[name]):
            continue
        if table[name].notna().sum() < len(table) * 0.5:
            continue
        if not table[name].std(skipna=True) > 0:
            continue
        keep.append(name)
    return keep


def build_model(name: str, seed: int, args: argparse.Namespace):
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)
        )
    if name == "forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=args.trees, random_state=seed, n_jobs=args.jobs),
        )
    if name == "gboost":
        return make_pipeline(
            SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=seed)
        )
    if name == "svm":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(probability=True, random_state=seed),
        )
    raise SystemExit(f"unknown model {name}")


def rank_features(data: pd.DataFrame, target: np.ndarray, features: list[str]) -> list[str]:
    """Training-fold AUC of every single feature, best first."""
    scores = []
    for name in features:
        column = data[name]
        good = column.notna().to_numpy()
        if good.sum() < 20 or len(np.unique(target[good])) < 2:
            continue
        auc = roc_auc_score(target[good], column.to_numpy(dtype=float)[good])
        scores.append((max(auc, 1 - auc), name))
    scores.sort(reverse=True)
    return [name for _, name in scores]


def complementary(data: pd.DataFrame, ordered: list[str], wanted: int, max_r: float) -> list[str]:
    """Walk down the ranking, skip anything that repeats a feature already kept."""
    kept: list[str] = []
    for name in ordered:
        if not kept:
            kept.append(name)
        else:
            correlation = data[kept].corrwith(data[name]).abs()
            if correlation.max(skipna=True) < max_r:
                kept.append(name)
        if len(kept) == wanted:
            break
    return kept


def evaluate(
    pair: pd.DataFrame, target: np.ndarray, features: list[str], args: argparse.Namespace
) -> tuple[pd.DataFrame, Counter]:
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    chosen: Counter = Counter()
    rows = []
    for size in sizes:
        per_model: dict[str, list[float]] = {name: [] for name in models}
        for repeat in range(args.repeats):
            splitter = StratifiedKFold(
                n_splits=args.folds, shuffle=True, random_state=args.seed + repeat
            )
            predicted = {name: np.zeros(len(target)) for name in models}
            for train, test in splitter.split(pair, target):
                train_data = pair.iloc[train]
                ordered = rank_features(train_data, target[train], features)
                panel = complementary(train_data, ordered, size, args.max_r)
                chosen.update(panel)
                for name in models:
                    model = build_model(name, args.seed + repeat, args)
                    model.fit(train_data[panel].to_numpy(dtype=float), target[train])
                    predicted[name][test] = model.predict_proba(
                        pair.iloc[test][panel].to_numpy(dtype=float)
                    )[:, 1]
            for name in models:
                per_model[name].append(roc_auc_score(target, predicted[name]))
        for name in models:
            values = per_model[name]
            rows.append(
                {
                    "model": name,
                    "size": size,
                    "auc": round(float(np.mean(values)), 4),
                    "auc_sd": round(float(np.std(values)), 4) if len(values) > 1 else 0.0,
                    "folds": args.folds,
                    "repeats": args.repeats,
                }
            )
            print(f"{name:9s} {size:2d} feature(s): AUC {rows[-1]['auc']:.4f} "
                  f"(SD {rows[-1]['auc_sd']:.4f})")
    return pd.DataFrame(rows), chosen


def main(argv=None) -> None:
    args = parse_args(argv)
    table = read_tables(args)
    indices = add_indices(table)
    features = numeric_features(table)
    print(f"{len(table)} patients, {len(features)} features including {len(indices)} indices")

    groups = ("AF", "HF", "stroke") if args.group.lower() == "all" else (args.group,)
    curves, picks = [], []
    for group in groups:
        pair = table[table["group"].isin([group, "control"])].reset_index(drop=True)
        if pair["group"].nunique() < 2:
            print(f"{group}: no patients, skipped")
            continue
        target = (pair["group"] == group).to_numpy(dtype=int)
        print(
            f"\n{group}: {int(target.sum())} cases, {int(len(target) - target.sum())} controls, "
            f"selection inside every fold"
        )
        curve, chosen = evaluate(pair, target, features, args)
        curve.insert(0, "group", group)
        curves.append(curve)
        total = sum(chosen.values())
        picks.append(
            pd.DataFrame(
                [
                    {
                        "group": group,
                        "feature": name,
                        "times_selected": count,
                        "share": round(count / total, 4) if total else np.nan,
                    }
                    for name, count in chosen.most_common(40)
                ]
            )
        )

    if not curves:
        raise SystemExit("nothing to report")
    curve_table = pd.concat(curves, ignore_index=True)
    best = curve_table.sort_values("auc", ascending=False).groupby(["group", "model"]).head(1)

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "nested_models.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        curve_table.to_excel(writer, sheet_name="Curve", index=False)
        pd.concat(picks, ignore_index=True).to_excel(writer, sheet_name="Chosen", index=False)
        best.to_excel(writer, sheet_name="Best", index=False)

    print("\nhonest AUC, selection inside the cross-validation")
    print(curve_table.pivot_table(index="size", columns=["group", "model"], values="auc").to_string())
    print("\nbest per model")
    print(best.to_string(index=False))

    if not args.no_plots:
        folder = args.output / "plots_index"
        folder.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 5))
        for (group, model), part in curve_table.groupby(["group", "model"]):
            plt.errorbar(part["size"], part["auc"], yerr=part["auc_sd"], marker="o",
                         capsize=3, label=f"{group} {model}")
        plt.xlabel("features in the panel")
        plt.ylabel("AUC, selection inside the folds")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(folder / "nested_curve.png", dpi=200)
        plt.close()

    print(f"\nExcel : {path}")


if __name__ == "__main__":
    main()
