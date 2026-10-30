"""AF versus controls: the best model of 1, 2, 3, 4 and 5 features.

Exhaustive search, no greedy shortcut. For each block, measurements only,
radiomics only, indices only and all blocks together, every combination of the
pool is scored with logistic regression and stratified cross-validation, and the
best out-of-fold AUC of each size is printed. That is the curve that shows where
adding features stops paying.

The pool of each block is its best --pool features by single-feature AUC, so the
search stays finite: C(--pool, 5) models per block.

Input : <results>\\fat_la_features.xlsx and <results>\\radiomic_features.xlsx
Output: <output>\\af_size_curve.xlsx   Curve (best of each size), All (every
        combination scored), and <output>\\plots_index\\af_size_curve.png

Usage
    python 22_af_size_curve.py
    python 22_af_size_curve.py --group HF --max-size 4 --pool 12
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    parser.add_argument("--group", default="AF", help="AF, HF or stroke")
    parser.add_argument("--max-size", type=int, default=5)
    parser.add_argument("--pool", type=int, default=14, help="features per block")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-plot", type=int, default=20, help="combinations in the bar chart")
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
        made["fit_index_W_L_over_E"] = (W * L) / safe(E)
    for name, series in made.items():
        table[name] = series.replace([np.inf, -np.inf], np.nan)
    return list(made)


def block_of(name: str, indices: list[str]) -> str:
    if name in indices:
        return "indices"
    for prefix in COMPARTMENTS:
        if name.startswith(prefix + "_"):
            return "measurements" if name.endswith("_volume_ml") else "radiomics"
    return "measurements"


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


def model():
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)
    )


def score(data: pd.DataFrame, target: np.ndarray, args: argparse.Namespace) -> float:
    """Mean out-of-fold AUC over the repeats."""
    values = data.to_numpy(dtype=float)
    aucs = []
    for repeat in range(args.repeats):
        folds = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed + repeat
        )
        predicted = cross_val_predict(
            model(), values, target, cv=folds, method="predict_proba"
        )[:, 1]
        aucs.append(roc_auc_score(target, predicted))
    return float(np.mean(aucs))


def combination_plots(
    rows: pd.DataFrame, curve: pd.DataFrame, folder: Path, group: str, top: int
) -> None:
    """Which combinations win and which features keep appearing in the winners."""
    best = rows.head(top)
    labels = [name.replace(" + ", "\n") for name in best["features"]]
    plt.figure(figsize=(8, 0.5 * len(best) + 1.5))
    plt.barh(range(len(best)), best["auc"], color="#3b6ea5")
    plt.yticks(range(len(best)), labels, fontsize=6)
    plt.gca().invert_yaxis()
    plt.xlim(max(0.5, best["auc"].min() - 0.05), best["auc"].max() + 0.01)
    plt.xlabel("out-of-fold AUC")
    plt.title(f"{group} vs control, best {len(best)} combinations")
    plt.tight_layout()
    plt.savefig(folder / f"{group.lower()}_best_combinations.png", dpi=200)
    plt.close()

    winners = rows.head(max(top * 5, 50))
    counts: dict[str, int] = {}
    for names in winners["features"]:
        for name in names.split(" + "):
            counts[name] = counts.get(name, 0) + 1
    order = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]
    if order:
        plt.figure(figsize=(7, 0.35 * len(order) + 1.5))
        plt.barh([name for name, _ in order], [count for _, count in order], color="#9a5b3b")
        plt.gca().invert_yaxis()
        plt.yticks(fontsize=7)
        plt.xlabel(f"appearances among the best {len(winners)} models")
        plt.tight_layout()
        plt.savefig(folder / f"{group.lower()}_feature_frequency.png", dpi=200)
        plt.close()

    heat = curve.pivot(index="size", columns="block", values="auc")
    plt.figure(figsize=(5, 4))
    plt.imshow(heat.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    plt.colorbar(label="AUC")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=30, ha="right", fontsize=8)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.ylabel("features in the model")
    for row, size in enumerate(heat.index):
        for column, block in enumerate(heat.columns):
            value = heat.loc[size, block]
            if pd.notna(value):
                plt.text(
                    column, row, f"{value:.3f}", ha="center", va="center",
                    color="white", fontsize=7,
                )
    plt.title(f"{group} vs control")
    plt.tight_layout()
    plt.savefig(folder / f"{group.lower()}_block_size_heatmap.png", dpi=200)
    plt.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    table = read_tables(args)
    indices = add_indices(table)
    features = numeric_features(table)
    pair = table[table["group"].isin([args.group, "control"])]
    if pair["group"].nunique() < 2:
        raise SystemExit(f"no {args.group} patients")
    target = (pair["group"] == args.group).to_numpy(dtype=int)
    print(
        f"{args.group}: {int(target.sum())} cases, {int(len(target) - target.sum())} controls, "
        f"{len(features)} features"
    )

    single = []
    for name in features:
        if pair[name].notna().sum() < len(pair) * 0.5:
            continue
        single.append(
            {
                "feature": name,
                "block": block_of(name, indices),
                "auc": score(pair[[name]], target, args),
            }
        )
    ranked = pd.DataFrame(single).sort_values("auc", ascending=False)

    blocks = {
        "measurements": ranked[ranked["block"] == "measurements"],
        "radiomics": ranked[ranked["block"] == "radiomics"],
        "indices": ranked[ranked["block"] == "indices"],
        "all blocks": ranked,
    }

    rows, curve = [], []
    for label, part in blocks.items():
        pool = part.head(args.pool)["feature"].tolist()
        if not pool:
            continue
        for size in range(1, min(args.max_size, len(pool)) + 1):
            best = {"auc": 0.0, "features": ""}
            total = 0
            for names in itertools.combinations(pool, size):
                auc = score(pair[list(names)], target, args)
                total += 1
                rows.append(
                    {
                        "block": label,
                        "size": size,
                        "features": " + ".join(names),
                        "auc": round(auc, 4),
                    }
                )
                if auc > best["auc"]:
                    best = {"auc": auc, "features": " + ".join(names)}
            curve.append(
                {
                    "group": args.group,
                    "block": label,
                    "size": size,
                    "combinations": total,
                    "auc": round(best["auc"], 4),
                    "features": best["features"],
                }
            )
            print(f"{label:13s} {size} feature(s): AUC {best['auc']:.4f}  {best['features']}")

    curve_table = pd.DataFrame(curve)
    all_rows = pd.DataFrame(rows).sort_values("auc", ascending=False)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.group.lower()}_size_curve.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        curve_table.to_excel(writer, sheet_name="Curve", index=False)
        all_rows.to_excel(writer, sheet_name="All", index=False)
        ranked.to_excel(writer, sheet_name="Univariate", index=False)

    print("\nAUC by number of features")
    print(curve_table.pivot(index="size", columns="block", values="auc").to_string())

    if not args.no_plots and not curve_table.empty:
        folder = args.output / "plots_index"
        folder.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 5))
        for label, part in curve_table.groupby("block"):
            plt.plot(part["size"], part["auc"], marker="o", label=label)
        plt.xlabel("features in the model")
        plt.ylabel("out-of-fold AUC")
        plt.title(f"{args.group} vs control, logistic regression")
        plt.legend()
        plt.tight_layout()
        plt.savefig(folder / f"{args.group.lower()}_size_curve.png", dpi=200)
        plt.close()
        combination_plots(all_rows, curve_table, folder, args.group, args.top_plot)

    print(f"\nExcel : {path}")


if __name__ == "__main__":
    main()
