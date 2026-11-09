"""One table for the paper: every model against every patient group, same folds.

This is the cheap check to run before the paper is written. No GPU and no
network training: it reads what is already on disk and scores it.

Models, each cross-validated in the same folds for every group:

    LA volume alone             the simplest clinical surrogate
    best four features          the published model, AUC 0.788 in AF
    all attenuation radiomics   fold-safe screen, the best --screen by training
                                fold AUC, then logistic
    latent (script 29)          the out of fold rim encoder score, read from
                                rim\\scores_<group>.csv if that run has finished
    latent radiomics (30)       the B block score, from latent\\scores_<group>.csv
    four features + latent      rank average of the two

Groups, each against the same controls:

    AF, HF, stroke, and AF+HF+stroke pooled as "any disease", so the question
    "is this a fat and atrium signal for disease in general or for AF in
    particular" has an answer in one table.

A model is only listed for a group if its input exists, so this runs before
scripts 29 to 31 have finished and fills in as they do.

Input : <output>\\fat_la_features.xlsx          patient_id, group
        <output>\\radiomic_features*.xlsx       any workbook from script 14
        <output>\\rim\\scores_*.csv             optional, script 29
        <output>\\latent\\scores_*.csv          optional, script 30
Output: <output>\\summary\\group_summary.xlsx   AUC per model per group
        <output>\\summary\\plots\\*.png         ROC per group, and a bar chart

Usage
    python 32_group_summary.py
    python 32_group_summary.py --groups-to-test AF,HF,stroke,AF+HF+stroke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEG = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast")
ANALYSIS = SEG / "analysis" / "results"
COMPARTMENTS = ("la", "laeat", "eat")
BEST_FOUR = (
    "la_firstorder_interquartile_range",
    "la_shape_surface_area",
    "laeat_shape_flatness",
    "eat_firstorder_median",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument("--groups", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--tables", default="", help="comma separated radiomics workbooks")
    parser.add_argument("--groups-to-test", default="AF,HF,stroke,AF+HF+stroke",
                        help="+ pools labels into one case group")
    parser.add_argument("--screen", type=int, default=20,
                        help="features the all-radiomics model keeps, chosen in the training fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_known_args(argv)[0]


# ------------------------------------------------------------------ the tables


def compartment_of(name: str) -> str:
    for part in COMPARTMENTS:
        if name.startswith(part + "_"):
            return part
    return "other"


def read_features(args: argparse.Namespace) -> pd.DataFrame:
    """Every radiomics workbook plus the measurement workbook, joined on patient_id."""
    paths = ([Path(item.strip()) for item in args.tables.split(",") if item.strip()]
             or sorted(args.output.glob("radiomic_features*.xlsx")))
    paths = [args.groups] + [path for path in paths if path != args.groups]
    merged = None
    for path in paths:
        if not path.exists():
            continue
        table = pd.read_excel(path, sheet_name="Features")
        table.columns = [str(name).strip() for name in table.columns]
        table["patient_id"] = table["patient_id"].astype(str)
        if "status" in table.columns:
            table = table[table["status"].astype(str) == "ok"]
        keep = ["patient_id"] + [name for name in table.columns
                                 if name != "patient_id" and pd.api.types.is_numeric_dtype(table[name])]
        print(f"  {path.name}: {len(table)} patients, {len(keep) - 1} numbers")
        table = table[keep]
        merged = table if merged is None else merged.merge(
            table, on="patient_id", how="inner", suffixes=("", "_dup"))
    if merged is None:
        raise SystemExit(f"no workbook found in {args.output}")
    return merged.loc[:, ~merged.columns.str.endswith("_dup")]


def read_scores(folder: Path, cases: list) -> dict:
    """Out of fold scores already written by scripts 29 and 30, if those runs exist."""
    found = {}
    for name in ("_".join(cases), "+".join(cases), cases[0] if len(cases) == 1 else ""):
        path = folder / f"scores_{name}.csv"
        if name and path.exists():
            table = pd.read_csv(path)
            table["patient_id"] = table["patient_id"].astype(str)
            found = {column: dict(zip(table["patient_id"], table[column]))
                     for column in table.columns
                     if column not in ("patient_id", "group", "case")}
            break
    return found


# ----------------------------------------------------------------- the scoring


def ranks(values: np.ndarray) -> np.ndarray:
    return stats.rankdata(np.where(np.isfinite(values), values, np.nanmin(values))) / len(values)


def logistic(train_x, train_y, test_x, seed: int = 0):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=4000, class_weight="balanced",
                                             random_state=seed))
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def screened(train_x, train_y, test_x, keep: int):
    """The best --screen features by training fold AUC, so the choice sees no test patient."""
    clean = np.nan_to_num(train_x, nan=0.0, posinf=0.0, neginf=0.0)
    order = stats.rankdata(clean, axis=0)
    cases, controls = train_y.sum(), (1 - train_y).sum()
    positive = order[train_y == 1].sum(axis=0)
    auc = (positive - cases * (cases + 1) / 2.0) / max(cases * controls, 1)
    best = np.argsort(np.abs(auc - 0.5))[::-1][:keep]
    return logistic(train_x[:, best], train_y, test_x[:, best])


def delong(target: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """Bootstrap p for the AUC difference; plain and enough for a table of six models."""
    if np.allclose(first, second):
        return 1.0
    rng = np.random.default_rng(0)
    seen = []
    for _ in range(600):
        pick = rng.integers(0, len(target), len(target))
        if len(np.unique(target[pick])) < 2:
            continue
        seen.append(roc_auc_score(target[pick], first[pick]) - roc_auc_score(target[pick], second[pick]))
    if not seen:
        return 1.0
    spread = np.std(seen) or 1e-9
    return float(2 * stats.norm.sf(abs(np.mean(seen)) / spread))


def cross_validate(values: np.ndarray, target: np.ndarray, args, how) -> np.ndarray:
    scores = np.full(len(target), np.nan)
    for repeat in range(1, args.repeats + 1):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + repeat)
        for train, test in splitter.split(values, target):
            scores[test] = how(values[train], target[train], values[test])
    return scores


# -------------------------------------------------------------------- one group


def one_group(table: pd.DataFrame, label_of: dict, cases: list, args) -> tuple:
    names = table["patient_id"].tolist()
    label = np.array([label_of.get(name, "") for name in names])
    keep = np.isin(label, cases) | (label == "control")
    part, target = table[keep].reset_index(drop=True), np.isin(label[keep], cases).astype(int)
    rows, curves = [], []
    if target.sum() < 10 or (1 - target).sum() < 10:
        print(f"   too few patients, {int(target.sum())} cases")
        return rows, curves

    panels = {}
    if "la_volume_ml" in part.columns:
        panels["LA volume alone"] = part[["la_volume_ml"]].to_numpy(dtype=float)
    have = [name for name in BEST_FOUR if name in part.columns]
    if len(have) == len(BEST_FOUR):
        panels["best four features"] = part[have].to_numpy(dtype=float)
    radiomics = [name for name in part.columns if compartment_of(name) != "other"]
    if len(radiomics) > len(BEST_FOUR):
        panels[f"all radiomics ({len(radiomics)}), screened in fold"] = part[radiomics].to_numpy(dtype=float)

    four = None
    for name, values in panels.items():
        how = ((lambda a, b, c: screened(a, b, c, args.screen)) if name.startswith("all radiomics")
               else logistic)
        scores = cross_validate(values, target, args, how)
        auc = float(roc_auc_score(target, scores))
        rows.append({"model": name, "n_case": int(target.sum()), "n_control": int((1 - target).sum()),
                     "auc": round(auc, 4)})
        curves.append((name, target, scores, auc))
        if name == "best four features":
            four = scores
        print(f"   {name:46s} AUC {auc:.4f}")

    for source, folder in (("rim encoder (29)", args.output / "rim"),
                           ("latent radiomics (30)", args.output / "latent")):
        stored = read_scores(folder, cases)
        for column, lookup in stored.items():
            values = np.array([lookup.get(name, np.nan) for name in part["patient_id"]], dtype=float)
            if np.isfinite(values).sum() < 0.8 * len(values):
                continue
            auc = float(roc_auc_score(target[np.isfinite(values)], values[np.isfinite(values)]))
            title = f"{source} {column}"
            rows.append({"model": title, "n_case": int(target.sum()),
                         "n_control": int((1 - target).sum()), "auc": round(auc, 4)})
            curves.append((title, target, values, auc))
            print(f"   {title:46s} AUC {auc:.4f}")
            if four is not None and column in ("rim_score", "A_plus_B"):
                blend = 0.5 * (ranks(four) + ranks(np.nan_to_num(values, nan=np.nanmedian(values))))
                mixed = float(roc_auc_score(target, blend))
                rows.append({"model": f"four features + {title}", "n_case": int(target.sum()),
                             "n_control": int((1 - target).sum()), "auc": round(mixed, 4)})
                curves.append((f"four + {column}", target, blend, mixed))
                print(f"   {'four features + ' + title:46s} AUC {mixed:.4f}")

    if four is not None:
        for row, curve in zip(rows, curves):
            row["p_vs_four_features"] = (
                "" if row["model"] == "best four features" else round(delong(target, curve[2], four), 4)
            )
    return rows, curves


# ------------------------------------------------------------------- the plots


def plot_roc(curves: list, folder: Path, title: str) -> None:
    plt.figure(figsize=(6.6, 6.0))
    for name, target, scores, auc in curves:
        known = np.isfinite(scores)
        fpr, tpr, _ = roc_curve(target[known], scores[known])
        plt.plot(fpr, tpr, lw=1.4, label=f"{name}  AUC {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="0.6", lw=0.8)
    plt.xlabel("1 - specificity")
    plt.ylabel("sensitivity")
    plt.title(f"{title} against control")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(folder / f"roc_{title.replace('+', '_')}.png", dpi=200)
    plt.close()


def plot_bars(table: pd.DataFrame, folder: Path) -> None:
    pivot = table.pivot_table(index="model", columns="group", values="auc")
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values().index)
    pivot.plot.barh(figsize=(9, 0.5 * len(pivot) + 2.2))
    plt.axvline(0.5, color="0.6", lw=0.8)
    plt.xlim(0.4, 1.0)
    plt.xlabel("AUC, cross-validated")
    plt.ylabel("")
    plt.title("Every model against every group, the same folds")
    plt.tight_layout()
    plt.savefig(folder / "all_models.png", dpi=200)
    plt.close()


# ------------------------------------------------------------------------- main


def main(argv=None) -> None:
    args = parse_args(argv)
    folder = args.output / "summary"
    plots = folder / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    print("1. reading the workbooks")
    table = read_features(args)
    groups = pd.read_excel(args.groups, sheet_name="Features")
    groups.columns = [str(name).strip() for name in groups.columns]
    label_of = dict(zip(groups["patient_id"].astype(str), groups["group"].astype(str)))
    counts = pd.Series(list(label_of.values())).value_counts().to_dict()
    print(f"   groups in the workbook: {counts}")

    collected = []
    for wanted in [item.strip() for item in args.groups_to_test.split(",") if item.strip()]:
        cases = [part.strip() for part in wanted.split("+") if part.strip()]
        missing = [name for name in cases if name not in counts]
        if missing:
            print(f"\n2. {wanted}: skipped, {missing} not in the workbook")
            continue
        print(f"\n2. {wanted} against control")
        rows, curves = one_group(table, label_of, cases, args)
        for row in rows:
            collected.append({"group": wanted} | row)
        if curves and not args.no_plots:
            plot_roc(curves, plots, wanted)

    if not collected:
        raise SystemExit("no group could be scored, check the group names in the workbook")
    summary = pd.DataFrame(collected)
    wide = summary.pivot_table(index="model", columns="group", values="auc")
    print("\n3. AUC, model by group")
    print(wide.round(3).to_string())

    book = folder / "group_summary.xlsx"
    with pd.ExcelWriter(book) as writer:
        summary.to_excel(writer, sheet_name="AUC", index=False)
        wide.round(4).to_excel(writer, sheet_name="Wide")
    if not args.no_plots:
        plot_bars(summary, plots)
        print(f"Plots : {plots}")
    print(f"Excel : {book}")


if __name__ == "__main__":
    main()
