"""Optimise the composite indices and search every 1-4 feature combination.

Input is the `Features` sheet of script 14 (radiomic and volumetric features),
optionally merged with the sheet of script 11 and with age, sex and BMI.

Three steps per group (AF, HF, stroke) against the controls:

1. Composite index. The exponents of W**a * L**b * E**c (whole-heart EAT, LA
   volume, LA-EAT volume) are searched on a grid and scored by cross-validated
   AUC of that single index, so the index is chosen the same way as the rest.
   The published pair (1, 1, 0) and (2, 0.1, 3) is always scored too.
2. Screening. Features are ranked by univariate cross-validated AUC and the top
   --pool are carried forward, because 1-4 combinations of everything is far too
   many fits for 40 patients.
3. Combinations. Every combination of 1, 2, 3 and 4 features of that pool is
   trained with logistic regression, random forest, SVM-RBF and LDA, in a
   pipeline of median imputation and z-scoring, under the same 5-fold stratified
   cross validation, repeated --repeats times. The AUC reported is the mean over
   folds with its SD.

A warning that matters more than the ranking: with roughly 20 patients per group
the best of a few thousand combinations is optimistic by construction. The
per-fold SD and the --holdout option (a split never seen by the search) are there
to show how much of the top AUC survives.

Output: <output>\\feature_search.xlsx
            Best_Models      the best combination per group and model
            Top_Combinations the top --top rows per group
            Composite        the exponent grid per group
            Univariate       single feature AUC per group
        <output>\\plots\\search_<group>.png

Usage
    python 15_feature_search.py
    python 15_feature_search.py --pool 14 --max-size 4 --repeats 5
    python 15_feature_search.py --holdout 0.25 --groups AF
"""

from __future__ import annotations

import argparse
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"Q:\users\marfi\Automatic segmentations\finalllll\final\test80_analysis")
CLINICAL = Path(r"Q:\users\marfi\CT_analysis\SCAPIS1-DATA-PETITION-1110-20260407(in).csv")
SKIP = {"patient_id", "status", "error", "group", "voxel_ml", "slice_mm", "cleaned_mask"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=BASE / "radiomic_features.xlsx")
    parser.add_argument(
        "--extra-features",
        type=Path,
        default=BASE / "fat_la_features.xlsx",
        help="second workbook merged on patient_id, script 11 by default",
    )
    parser.add_argument("--clinical", type=Path, default=CLINICAL)
    parser.add_argument("--output", type=Path, default=BASE)
    parser.add_argument("--groups", nargs="*", default=["AF", "HF", "stroke"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=-1, help="parallel workers, -1 for all cores")
    parser.add_argument("--pool", type=int, default=10, help="features carried into the search")
    parser.add_argument("--max-size", type=int, default=4, help="largest combination")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--holdout", type=float, default=0.0, help="share kept out of the search")
    parser.add_argument(
        "--biomarkers",
        nargs="*",
        default=[],
        help="extra clinical CSV columns to merge, e.g. HbA1c and cholesterol",
    )
    parser.add_argument("--no-clinical", action="store_true")
    parser.add_argument("--no-composite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_known_args(argv)[0]


def pipelines(seed: int = 0) -> dict[str, object]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def wrap(estimator):
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )

    return {
        "logistic": wrap(LogisticRegression(max_iter=5000, class_weight="balanced")),
        "random_forest": wrap(
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )
        ),
        "svm_rbf": wrap(SVC(kernel="rbf", probability=False, class_weight="balanced")),
        "lda": wrap(LinearDiscriminantAnalysis()),
    }


def scores_of(model, matrix: np.ndarray) -> np.ndarray:
    """Probability when the model has one, otherwise the decision value."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, 1]
    return model.decision_function(matrix)


def cross_validated_auc(matrix, labels, estimator, splits) -> tuple[float, float]:
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    aucs = []
    for train, test in splits:
        if len(set(labels[train])) < 2 or len(set(labels[test])) < 2:
            continue
        model = clone(estimator).fit(matrix[train], labels[train])
        aucs.append(roc_auc_score(labels[test], scores_of(model, matrix[test])))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.std(aucs))


def load_features(args) -> pd.DataFrame:
    table = pd.read_excel(args.features, sheet_name="Features")
    table["patient_id"] = table["patient_id"].astype(str)
    if "status" in table:
        table = table[table["status"] == "ok"]
    if args.extra_features and Path(args.extra_features).exists():
        extra = pd.read_excel(args.extra_features, sheet_name="Features")
        extra["patient_id"] = extra["patient_id"].astype(str)
        if "status" in extra:
            extra = extra[extra["status"] == "ok"]
        columns = [
            name
            for name in extra.columns
            if name not in SKIP and name not in table.columns
        ]
        table = table.merge(extra[["patient_id"] + columns], on="patient_id", how="left")
    if not args.no_clinical and Path(args.clinical).exists():
        import importlib.util
        import sys

        name = "12_random_forest_groups"
        module = sys.modules.get(name)
        if module is None:
            for folder in (Path(__file__).resolve().parent, Path.cwd()):
                path = folder / f"{name}.py"
                if path.exists():
                    spec = importlib.util.spec_from_file_location(name, path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    break
        if module is not None:
            table, _ = module.add_clinical(table, Path(args.clinical))
        table = add_biomarkers(table, Path(args.clinical), args.biomarkers)
    return table.drop(columns=["sex", "sex_male"], errors="ignore")


def add_biomarkers(table: pd.DataFrame, clinical_file: Path, columns: list[str]) -> pd.DataFrame:
    """Merge named clinical columns, HbA1c and the lipids, as numeric features."""
    if not columns or not clinical_file.exists():
        return table
    clinical = pd.read_csv(clinical_file, sep=None, engine="python", dtype=str)
    if "Subject" not in clinical:
        return table
    present = [name for name in columns if name in clinical.columns]
    missing = [name for name in columns if name not in clinical.columns]
    if missing:
        print(f"biomarkers not in the CSV: {', '.join(missing)}")
    if not present:
        return table
    block = clinical[["Subject"] + present].rename(columns={"Subject": "patient_id"})
    block["patient_id"] = block["patient_id"].astype(str)
    for name in present:
        block[name] = pd.to_numeric(
            block[name].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    return table.merge(block, on="patient_id", how="left")


def usable_columns(table: pd.DataFrame) -> list[str]:
    columns = []
    for name in table.columns:
        if name in SKIP:
            continue
        values = pd.to_numeric(table[name], errors="coerce")
        if values.notna().sum() < max(10, 0.6 * len(values)) or values.nunique() < 3:
            continue
        columns.append(name)
    return columns


def composite_search(part: pd.DataFrame, labels, splits, estimator) -> pd.DataFrame:
    """Exponents of W**a * L**b * E**c scored as a single predictor."""
    needed = ("eat_volume_ml", "la_volume_ml", "laeat_volume_ml")
    if not all(name in part for name in needed):
        return pd.DataFrame()
    whole, atrium, near = (
        pd.to_numeric(part[name], errors="coerce").to_numpy(dtype=float) for name in needed
    )
    if np.nanmin([whole.min(), atrium.min(), near.min()]) <= 0:
        whole, atrium, near = (np.clip(values, 1e-6, None) for values in (whole, atrium, near))
    grid = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
    rows = []
    for a, b, c in list(product(grid, repeat=3)) + [(1.0, 1.0, 0.0), (2.0, 0.1, 3.0)]:
        if a == b == c == 0:
            continue
        index = np.log(whole) * a + np.log(atrium) * b + np.log(near) * c
        matrix = index.reshape(-1, 1)
        mean, sd = cross_validated_auc(matrix, labels, estimator, splits)
        rows.append({"a_eat": a, "b_la": b, "c_laeat": c, "auc": round(mean, 3), "auc_sd": round(sd, 3)})
    frame = pd.DataFrame(rows).drop_duplicates(["a_eat", "b_la", "c_laeat"])
    return frame.sort_values("auc", ascending=False)


def add_composites(table: pd.DataFrame, exponents: tuple[float, float, float]) -> list[str]:
    needed = ("eat_volume_ml", "la_volume_ml", "laeat_volume_ml")
    if not all(name in table for name in needed):
        return []
    whole, atrium, near = (
        np.clip(pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=float), 1e-6, None)
        for name in needed
    )
    added = []
    if "cmi1" not in table:
        table["cmi1"] = whole * atrium
        added.append("cmi1")
    if "cmi2" not in table:
        table["cmi2"] = whole**2 * atrium**0.1 * near**3
        added.append("cmi2")
    a, b, c = exponents
    table["cmi_opt"] = np.exp(np.log(whole) * a + np.log(atrium) * b + np.log(near) * c)
    added.append("cmi_opt")
    return added


def search_group(group: str, table: pd.DataFrame, columns: list[str], args) -> dict:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

    part = table[table["group"].isin([group, "control"])].reset_index(drop=True)
    labels = (part["group"] == group).astype(int).to_numpy()
    if labels.sum() < 8 or (labels == 0).sum() < 8:
        return {"note": f"{group}: too few patients"}

    search_index = np.arange(len(labels))
    holdout_index = np.array([], dtype=int)
    if args.holdout > 0:
        search_index, holdout_index = train_test_split(
            search_index, test_size=args.holdout, stratify=labels, random_state=0
        )

    folds = min(args.folds, int(labels[search_index].sum()), int((labels[search_index] == 0).sum()))
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=args.repeats, random_state=0)
    models = pipelines()

    def splits_for(index):
        return [
            (index[train], index[test])
            for train, test in splitter.split(np.zeros(len(index)), labels[index])
        ]

    splits = splits_for(search_index)
    matrix_of = lambda names: part[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    composite = pd.DataFrame()
    exponents = (1.0, 1.0, 0.0)
    if not args.no_composite:
        composite = composite_search(part, labels, splits, models["logistic"])
        if not composite.empty:
            best = composite.iloc[0]
            exponents = (best["a_eat"], best["b_la"], best["c_laeat"])
    columns = columns + [name for name in add_composites(part, exponents) if name not in columns]

    univariate = []
    for name in columns:
        mean, sd = cross_validated_auc(matrix_of([name]), labels, models["logistic"], splits)
        univariate.append({"feature": name, "auc": round(mean, 3), "auc_sd": round(sd, 3)})
    ranked = (
        pd.DataFrame(univariate)
        .assign(distance=lambda frame: (frame["auc"] - 0.5).abs())
        .sort_values("distance", ascending=False)
        .drop(columns="distance")
    )
    pool = ranked["feature"].head(args.pool).tolist()

    from joblib import Parallel, delayed

    wanted = [
        names
        for size in range(1, min(args.max_size, len(pool)) + 1)
        for names in combinations(pool, size)
    ]
    print(f"{len(wanted)} combinations x {len(models)} models")

    def score_one(names):
        matrix = matrix_of(list(names))
        out = []
        for model_name, estimator in models.items():
            mean, sd = cross_validated_auc(matrix, labels, estimator, splits)
            out.append(
                {
                    "group": group,
                    "model": model_name,
                    "n_features": len(names),
                    "features": " + ".join(names),
                    "auc": round(mean, 3),
                    "auc_sd": round(sd, 3),
                }
            )
        return out

    try:
        batches = Parallel(n_jobs=args.jobs, prefer="processes")(
            delayed(score_one)(names) for names in wanted
        )
    except Exception as error:  # noqa: BLE001 - any worker death falls back to one process
        print(f"parallel search failed ({type(error).__name__}), running in one process")
        batches = [score_one(names) for names in wanted]
    results = pd.DataFrame([row for batch in batches for row in batch]).sort_values(
        "auc", ascending=False
    )

    if len(holdout_index):
        from sklearn.base import clone

        checked = []
        for _, row in results.head(args.top).iterrows():
            names = row["features"].split(" + ")
            matrix = matrix_of(names)
            model = clone(models[row["model"]]).fit(matrix[search_index], labels[search_index])
            scores = scores_of(model, matrix[holdout_index])
            auc = (
                roc_auc_score(labels[holdout_index], scores)
                if len(set(labels[holdout_index])) == 2
                else float("nan")
            )
            checked.append(round(float(auc), 3))
        top = results.head(args.top).copy()
        top["auc_holdout"] = checked
        results = pd.concat([top, results.iloc[args.top :]], ignore_index=True)

    return {
        "results": results,
        "univariate": ranked.assign(group=group),
        "composite": composite.assign(group=group) if not composite.empty else composite,
        "pool": pool,
        "n_group": int(labels.sum()),
        "n_control": int((labels == 0).sum()),
        "n_holdout": len(holdout_index),
    }


def draw(group: str, results: pd.DataFrame, folder: Path, top: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    best = results.head(top).iloc[::-1]
    labels = [
        f"{row.model}: {row.features if len(row.features) < 60 else row.features[:57] + '...'}"
        for row in best.itertuples()
    ]
    figure, axis = plt.subplots(figsize=(9, 0.32 * len(best) + 1.5))
    axis.barh(range(len(best)), best["auc"], xerr=best["auc_sd"], capsize=2)
    axis.set_yticks(range(len(best)), labels, fontsize=7)
    axis.axvline(0.5, color="grey", linestyle="--")
    axis.set_xlim(0.4, 1.0)
    axis.set_xlabel("AUC, mean over folds")
    axis.set_title(f"{group} vs control, best {len(best)} combinations")
    figure.tight_layout()
    figure.savefig(folder / f"search_{group}.png", dpi=150)
    plt.close(figure)


def main(argv=None) -> None:
    args = parse_args(argv)
    table = load_features(args)
    if "group" not in table:
        raise SystemExit("the features sheet has no group column, run script 14 first")
    columns = usable_columns(table)
    print(f"{len(table)} patients, {len(columns)} usable features")
    print(table["group"].value_counts().to_string())

    best_rows, tops, univariates, composites = [], [], [], []
    for group in args.groups:
        if group not in set(table["group"]):
            print(f"\n{group}: not in the sheet, skipped")
            continue
        result = search_group(group, table, columns, args)
        if "results" not in result:
            print(f"\n{result['note']}")
            continue
        results = result["results"]
        tops.append(results.head(args.top))
        univariates.append(result["univariate"])
        if not result["composite"].empty:
            composites.append(result["composite"].head(20))
        for model_name, part in results.groupby("model"):
            row = part.iloc[0].to_dict()
            row.update(
                {
                    "n_group": result["n_group"],
                    "n_control": result["n_control"],
                    "n_holdout": result["n_holdout"],
                }
            )
            best_rows.append(row)
        print(f"\n{group} vs control, {result['n_group']} vs {result['n_control']}")
        if not result["composite"].empty:
            best = result["composite"].iloc[0]
            print(
                f"best index W^{best['a_eat']} * L^{best['b_la']} * E^{best['c_laeat']}"
                f"  AUC {best['auc']} +- {best['auc_sd']}"
            )
        print(f"pool: {', '.join(result['pool'])}")
        print(results.head(10).drop(columns="group").to_string(index=False))
        if not args.no_plots:
            draw(group, results, args.output / "plots", args.top)

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "feature_search.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if best_rows:
            pd.DataFrame(best_rows).sort_values(["group", "auc"], ascending=[True, False]).to_excel(
                writer, sheet_name="Best_Models", index=False
            )
        if tops:
            pd.concat(tops).to_excel(writer, sheet_name="Top_Combinations", index=False)
        if composites:
            pd.concat(composites).to_excel(writer, sheet_name="Composite", index=False)
        if univariates:
            pd.concat(univariates).to_excel(writer, sheet_name="Univariate", index=False)
    print(f"\nExcel : {path}")
    if not args.no_plots:
        print(f"Plots : {args.output / 'plots'}")


if __name__ == "__main__":
    main()
