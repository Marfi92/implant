"""Screen thousands of radiomics down to a short list, one by one then two by two.

Script 14 with --full --filters wavelet,log writes about 1000 features per
compartment. This script searches them in the order you asked for:

    1. every feature alone, cross-validated AUC
    2. the best --keep per compartment, after dropping a feature that repeats one
       already kept (correlation above --max-correlation)
    3. every pair of the kept features
    4. the best pairs grown by one more feature

Everything that chooses a feature happens inside the training fold, so the AUC
of the winner is not the AUC of a feature picked on the test patients. The
difference is not small: on 329 cases, picking the best of 3000 features on all
the data flatters it by roughly 0.03 to 0.05.

A pair or a triple is scored with the closed form linear discriminant, which is
one matrix solve, so 45,000 pairs take seconds rather than hours; the few
winners are then refitted with logistic regression and compared with the
published four feature model.

Input: the workbooks written by script 14, e.g.

    python 14_radiomic_features.py --full --filters wavelet,log --compartments la
    python 14_radiomic_features.py --full --filters wavelet,log --compartments eat
    python 14_radiomic_features.py --full --filters wavelet,log --compartments laeat

Output: <output>\\ladder\\ladder.xlsx
            Single     every feature, train and test AUC, FDR q value
            Kept       the short list per compartment
            Pairs      the best pairs
            Triples    the best triples
            Final      logistic AUC of the winners against the four feature model
        plus two plots, and progress.json so a crash costs one fold.

Usage
    python 31_feature_screen_ladder.py --group AF
    python 31_feature_screen_ladder.py --group AF --keep 100 --pairs 300 --folds 5
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
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
    parser.add_argument(
        "--tables",
        default="",
        help="comma separated workbooks from script 14; default every "
        "radiomic_features*.xlsx in --output",
    )
    parser.add_argument("--groups", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument("--group", default="AF",
                        help="AF, HF or stroke, or several pooled into one case group, "
                             "e.g. --group AF,HF,stroke for diseased against control")
    parser.add_argument("--keep", type=int, default=100, help="features kept per compartment")
    parser.add_argument(
        "--max-correlation",
        type=float,
        default=0.9,
        help="a feature correlating above this with a better one is dropped",
    )
    parser.add_argument("--pairs", type=int, default=300, help="best pairs carried to the triples")
    parser.add_argument("--report", type=int, default=50, help="rows written per sheet")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regularisation", type=float, default=1.0)
    parser.add_argument("--fresh", action="store_true", help="ignore progress.json")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_known_args(argv)[0]


# ------------------------------------------------------------------ the tables


def compartment_of(name: str) -> str:
    for part in COMPARTMENTS:
        if name.startswith(part + "_"):
            return part
    return "other"


def read_tables(args: argparse.Namespace) -> pd.DataFrame:
    """One row per patient, the workbooks joined on patient_id."""
    if args.tables:
        paths = [Path(item.strip()) for item in args.tables.split(",") if item.strip()]
    else:
        paths = sorted(args.output.glob("radiomic_features*.xlsx"))
    if not paths:
        raise SystemExit(
            "no radiomics workbook found, run 14_radiomic_features.py --full "
            "--filters wavelet,log --compartments la (then eat, then laeat) first"
        )
    merged = None
    for path in paths:
        table = pd.read_excel(path, sheet_name="Features")
        table["patient_id"] = table["patient_id"].astype(str)
        if "status" in table.columns:
            table = table[table["status"].astype(str) == "ok"]
        keep = ["patient_id"] + [
            name
            for name in table.columns
            if compartment_of(name) != "other" and pd.api.types.is_numeric_dtype(table[name])
        ]
        table = table[keep]
        print(f"  {path.name}: {len(table)} patients, {len(keep) - 1} features")
        merged = table if merged is None else merged.merge(table, on="patient_id", how="inner",
                                                           suffixes=("", "_dup"))
    merged = merged.loc[:, ~merged.columns.str.endswith("_dup")]
    return merged


def labels_of(args: argparse.Namespace, table: pd.DataFrame) -> tuple:
    groups = pd.read_excel(args.groups, sheet_name="Features")
    groups["patient_id"] = groups["patient_id"].astype(str)
    label_of = dict(zip(groups["patient_id"], groups["group"]))
    wanted = [name in label_of for name in table["patient_id"]]
    table = table[wanted].reset_index(drop=True)
    cases = [name.strip() for name in str(args.group).split(",") if name.strip()]
    target = np.array([1 if label_of[name] in cases else 0 for name in table["patient_id"]])
    keep = (target == 1) | np.array(
        [label_of[name] in ("control", "Control") for name in table["patient_id"]]
    )
    return table[keep].reset_index(drop=True), target[keep]


# -------------------------------------------------------------- the fast parts


def auc_of_columns(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """AUC of every column at once, from the ranks, so 3000 features cost one sort."""
    try:
        ranks = rankdata(values, axis=0)
    except TypeError:  # scipy before 1.10 has no axis here
        ranks = np.apply_along_axis(rankdata, 0, values)
    cases, controls = target.sum(), (1 - target).sum()
    if cases == 0 or controls == 0:
        return np.full(values.shape[1], 0.5)
    positive = ranks[target == 1].sum(axis=0)
    return (positive - cases * (cases + 1) / 2.0) / (cases * controls)


def discriminant(values: np.ndarray, target: np.ndarray, sets: np.ndarray) -> np.ndarray:
    """Linear discriminant weights for many feature sets at once.

    For one set, w = S^-1 (mean of the cases - mean of the controls), the direction
    that separates the two groups best when their spread is the same. Stacking the
    sets lets numpy solve 45,000 of these in one call.
    """
    cases = values[target == 1]
    controls = values[target == 0]
    gap = cases.mean(axis=0) - controls.mean(axis=0)
    spread = np.cov(values, rowvar=False)
    blocks = spread[sets[:, :, None], sets[:, None, :]]
    blocks = blocks + np.eye(sets.shape[1]) * 1e-6 * np.trace(spread) / values.shape[1]
    return np.linalg.solve(blocks, gap[sets][:, :, None])[:, :, 0]


def fdr(auc: np.ndarray, cases: int, controls: int) -> tuple:
    """Mann-Whitney p from the AUC, then Benjamini-Hochberg over all the features.

    Needed because the best of 3000 features is expected to look good by chance:
    at p 0.05 about 150 of them would pass with no signal at all.
    """
    from scipy.stats import norm

    spread = np.sqrt((cases + controls + 1) / (12.0 * cases * controls))
    p = 2 * norm.sf(np.abs(auc - 0.5) / spread)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return p, out


def prune(order: np.ndarray, values: np.ndarray, limit: int, ceiling: float) -> list:
    """Walk down the ranking and skip a feature that repeats one already kept."""
    kept: list[int] = []
    for index in order:
        column = values[:, index]
        if np.std(column) == 0:
            continue
        if kept:
            taken = values[:, kept]
            with np.errstate(invalid="ignore", divide="ignore"):
                same = np.abs(np.corrcoef(column, taken, rowvar=False)[0, 1:])
            if np.nanmax(same) > ceiling:
                continue
        kept.append(int(index))
        if len(kept) >= limit:
            break
    return kept


def logistic_auc(values: np.ndarray, target: np.ndarray, train, test, strength: float) -> float:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=strength, class_weight="balanced"),
    ).fit(values[train], target[train])
    score = model.predict_proba(values[test])[:, 1]
    return float(auc_of_columns(score[:, None], target[test])[0])


# ---------------------------------------------------------------------- folds


class Journal:
    """progress.json, so a crash costs one fold and not the whole search."""

    def __init__(self, path: Path, fresh: bool) -> None:
        self.path = path
        self.state = {} if fresh or not path.exists() else json.loads(path.read_text())

    def get(self, key: str):
        return self.state.get(key)

    def put(self, key: str, value) -> None:
        self.state[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state))


def one_fold(values, target, names, train, test, args) -> dict:
    """The whole ladder, fitted on the training patients and read on the test ones."""
    middle = np.nanmedian(values[train], axis=0)
    filled = np.where(np.isnan(values), middle, values)
    filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
    centre = filled[train].mean(axis=0)
    width = filled[train].std(axis=0)
    width[width == 0] = 1.0
    scaled = (filled - centre) / width

    alone_train = auc_of_columns(scaled[train], target[train])
    alone_test = auc_of_columns(scaled[test], target[test])

    kept: list[int] = []
    for part in COMPARTMENTS:
        inside = np.array([number for number, name in enumerate(names)
                           if compartment_of(name) == part])
        if not len(inside):
            continue
        strength = np.abs(alone_train[inside] - 0.5)
        order = inside[np.argsort(-strength)]
        kept.extend(prune(order, scaled[train], args.keep, args.max_correlation))
    kept = np.array(sorted(set(kept)))

    pairs = np.array(list(itertools.combinations(range(len(kept)), 2)))
    pair_index = kept[pairs]
    weights = discriminant(scaled[train], target[train], pair_index)
    pair_train = auc_of_columns((scaled[train][:, pair_index] * weights).sum(axis=2), target[train])
    pair_train = np.where(pair_train < 0.5, 1 - pair_train, pair_train)
    best = np.argsort(-pair_train)[: args.pairs]
    pair_test = auc_of_columns((scaled[test][:, pair_index[best]] * weights[best]).sum(axis=2),
                               target[test])
    pair_test = np.where(pair_test < 0.5, 1 - pair_test, pair_test)

    triples = []
    for row in pair_index[best[: max(args.pairs // 10, 10)]]:
        for extra in kept:
            if extra not in row:
                triples.append([row[0], row[1], extra])
    triple_index = np.unique(np.sort(np.array(triples), axis=1), axis=0)
    weights3 = discriminant(scaled[train], target[train], triple_index)
    triple_train = auc_of_columns((scaled[train][:, triple_index] * weights3).sum(axis=2),
                                  target[train])
    triple_train = np.where(triple_train < 0.5, 1 - triple_train, triple_train)
    pick = np.argsort(-triple_train)[: args.report]
    triple_test = auc_of_columns((scaled[test][:, triple_index[pick]] * weights3[pick]).sum(axis=2),
                                 target[test])
    triple_test = np.where(triple_test < 0.5, 1 - triple_test, triple_test)

    winner_pair = [names[number] for number in pair_index[best[0]]]
    winner_triple = [names[number] for number in triple_index[pick[0]]]
    honest = {
        "pair": logistic_auc(filled[:, pair_index[best[0]]], target, train, test, args.regularisation),
        "triple": logistic_auc(filled[:, triple_index[pick[0]]], target, train, test, args.regularisation),
    }
    four = [number for number, name in enumerate(names) if name in BEST_FOUR]
    if len(four) == len(BEST_FOUR):
        honest["four"] = logistic_auc(filled[:, four], target, train, test, args.regularisation)

    return {
        "single": {names[number]: [float(alone_train[number]), float(alone_test[number])]
                   for number in np.argsort(-np.abs(alone_train - 0.5))[: args.report * 4]},
        "kept": [names[number] for number in kept],
        "pairs": [[names[pair_index[number][0]], names[pair_index[number][1]],
                   float(pair_train[number]), float(pair_test[position])]
                  for position, number in enumerate(best[: args.report])],
        "triples": [[names[triple_index[number][0]], names[triple_index[number][1]],
                     names[triple_index[number][2]], float(triple_train[number]),
                     float(triple_test[position])]
                    for position, number in enumerate(pick[: args.report])],
        "winner_pair": winner_pair,
        "winner_triple": winner_triple,
        "honest": honest,
    }


# ---------------------------------------------------------------------- output


def gather(folds: list, key: str, columns: list) -> pd.DataFrame:
    """Average what every fold reported about the same feature or pair."""
    rows: dict[tuple, list] = {}
    for fold in folds:
        for item in fold[key]:
            name = tuple(item[: len(columns) - 2])
            rows.setdefault(name, []).append(item[len(columns) - 2:])
    table = [
        [*name, float(np.mean([value[0] for value in values])),
         float(np.mean([value[1] for value in values])), len(values)]
        for name, values in rows.items()
    ]
    frame = pd.DataFrame(table, columns=[*columns, "folds"])
    return frame.sort_values("test_auc", ascending=False).reset_index(drop=True)


def plots_of(single: pd.DataFrame, pairs: pd.DataFrame, folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(single["train_auc"], single["test_auc"], s=8, alpha=0.5)
    axis.plot([0.3, 0.9], [0.3, 0.9], color="grey", linewidth=0.8)
    axis.set_xlabel("AUC on the training fold, where it was chosen")
    axis.set_ylabel("AUC on the test fold")
    axis.set_title("How much of a single feature survives to unseen patients")
    figure.tight_layout()
    figure.savefig(folder / "01_single_train_test.png", dpi=150)
    plt.close(figure)

    top = pairs.head(20).iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh([f"{row.feature_a}\n+ {row.feature_b}" for row in top.itertuples()], top["test_auc"])
    axis.set_xlabel("AUC on the test folds")
    axis.set_title("Best pairs")
    axis.tick_params(axis="y", labelsize=6)
    figure.tight_layout()
    figure.savefig(folder / "02_best_pairs.png", dpi=150)
    plt.close(figure)


def main(argv=None) -> None:
    args = parse_args(argv)
    folder = args.output / "ladder"
    folder.mkdir(parents=True, exist_ok=True)
    journal = Journal(folder / "progress.json", args.fresh)

    print("1. reading the radiomics workbooks")
    table = read_tables(args)
    table, target = labels_of(args, table)
    names = [name for name in table.columns if name != "patient_id"]
    values = table[names].to_numpy(dtype=float)
    print(f"2. {args.group} {int(target.sum())} against control {int((1 - target).sum())}, "
          f"{len(names)} features")
    if target.sum() < 10:
        raise SystemExit("too few cases, check --group and the groups workbook")

    folds = []
    print(f"3. {args.repeats} x {args.folds} folds, every choice made inside the training fold")
    for repeat in range(1, args.repeats + 1):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + repeat)
        for number, (train, test) in enumerate(splitter.split(values, target), 1):
            key = (f"{args.group} keep{args.keep} pairs{args.pairs} n{len(target)}"
                   f" f{len(names)} r{repeat} fold{number}")
            saved = journal.get(key)
            if saved is not None:
                folds.append(saved)
                print(f"  repeat {repeat} fold {number} reused")
                continue
            result = one_fold(values, target, names, train, test, args)
            journal.put(key, result)
            folds.append(result)
            honest = result["honest"]
            print(f"  repeat {repeat} fold {number}: best pair {honest['pair']:.3f}, "
                  f"best triple {honest['triple']:.3f}"
                  + (f", four features {honest['four']:.3f}" if "four" in honest else ""))

    single = gather([{"single": [[name, *value] for name, value in fold["single"].items()]}
                     for fold in folds], "single", ["feature", "train_auc", "test_auc"])
    whole = auc_of_columns(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), target)
    p, q = fdr(whole, int(target.sum()), int((1 - target).sum()))
    book_of = {name: (float(whole[number]), float(p[number]), float(q[number]))
               for number, name in enumerate(names)}
    single["auc_all_patients"] = [book_of[name][0] for name in single["feature"]]
    single["p"] = [book_of[name][1] for name in single["feature"]]
    single["q_fdr"] = [book_of[name][2] for name in single["feature"]]
    pairs = gather(folds, "pairs", ["feature_a", "feature_b", "train_auc", "test_auc"])
    triples = gather(folds, "triples", ["feature_a", "feature_b", "feature_c", "train_auc", "test_auc"])
    counted: dict[str, int] = {}
    for fold in folds:
        for name in fold["kept"]:
            counted[name] = counted.get(name, 0) + 1
    kept = pd.DataFrame(
        [[name, compartment_of(name), count] for name, count in counted.items()],
        columns=["feature", "compartment", "folds_kept"],
    ).sort_values(["folds_kept", "feature"], ascending=[False, True])

    final = pd.DataFrame(
        [[name, float(np.mean([fold["honest"][key] for fold in folds if key in fold["honest"]])),
          float(np.std([fold["honest"][key] for fold in folds if key in fold["honest"]]))]
         for key, name in (("four", "published four features"),
                           ("pair", "best pair, chosen inside each fold"),
                           ("triple", "best triple, chosen inside each fold"))
         if any(key in fold["honest"] for fold in folds)],
        columns=["model", "auc", "sd"],
    )

    print("\n4. logistic AUC, the selection repeated inside every training fold")
    for row in final.itertuples():
        print(f"   {row.model:38s} AUC {row.auc:.4f} (SD {row.sd:.4f})")
    print("\n5. the best pairs over the folds")
    for row in pairs.head(10).itertuples():
        print(f"   {row.test_auc:.4f}  {row.feature_a} + {row.feature_b}")

    tag = args.group.replace(",", "_")
    book = folder / (f"ladder_{tag}.xlsx" if tag != "AF" else "ladder.xlsx")
    with pd.ExcelWriter(book) as writer:
        single.head(args.report * 20).to_excel(writer, sheet_name="Single", index=False)
        kept.to_excel(writer, sheet_name="Kept", index=False)
        pairs.head(args.report * 4).to_excel(writer, sheet_name="Pairs", index=False)
        triples.head(args.report * 4).to_excel(writer, sheet_name="Triples", index=False)
        final.to_excel(writer, sheet_name="Final", index=False)
    if not args.no_plots:
        plots_of(single, pairs, folder / "plots")
        print(f"Plots : {folder / 'plots'}")
    print(f"Excel : {book}")


if __name__ == "__main__":
    main()
