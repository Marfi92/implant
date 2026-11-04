"""The ladder: LA volume alone, then the index, then one added feature at a time.

Answers two questions in one run.

1. What does each step buy? Starting from a named feature, usually LA volume,
   the script adds the single feature that raises the out-of-fold AUC most, and
   prints a row per step with the AUC and the gain. A flat ladder means the new
   feature repeats what is already in the model.
2. Why is it flat? The Redundancy sheet gives the correlation of the top
   features with each other and with the first one, and the principal component
   analysis of that block, so you can quote how few independent directions 300
   radiomic features really contain.

Named starts are scored on their own first: LA volume, the whole fat ROI volume,
the LA-EAT volume, cmi1, cmi2 and the fitted W*L/E index.

Input : <results>\\fat_la_features.xlsx and a radiomics workbook
Output: <output>\\feature_ladder.xlsx   Ladder, Starts, Redundancy, Components
        <output>\\plots_index\\<group>_ladder.png and _redundancy.png

Usage
    python 24_feature_ladder.py --group AF --start la_volume_ml --max-size 10
    python 24_feature_ladder.py --radiomics ...\\radiomic_features_full.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    parser.add_argument("--group", default="AF", help="AF, HF or stroke")
    parser.add_argument("--start", default="la_volume_ml", help="first rung of the ladder")
    parser.add_argument("--max-size", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=60, help="features tried at each step")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
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
        "ratio_W_over_L": W / L.replace(0, np.nan),
        "ratio_E_over_L": E / L.replace(0, np.nan),
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


def model():
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)
    )


def score(data: pd.DataFrame, target: np.ndarray, args: argparse.Namespace) -> float:
    values = data.to_numpy(dtype=float)
    aucs = []
    for repeat in range(args.repeats):
        folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + repeat)
        predicted = cross_val_predict(model(), values, target, cv=folds, method="predict_proba")
        aucs.append(roc_auc_score(target, predicted[:, 1]))
    return float(np.mean(aucs))


def named_starts(
    pair: pd.DataFrame, target: np.ndarray, features: list[str], args: argparse.Namespace
) -> pd.DataFrame:
    """The single features you keep asking about, each on its own."""
    wanted = [
        "la_volume_ml",
        "eat_volume_ml",
        "laeat_volume_ml",
        "fat_volume_raw_ml",
        "cmi1_W_L",
        "cmi2_W2_L01_E3",
        "index_W_L_over_E",
        "la_shape_mesh_volume",
        "la_shape_surface_area",
        "la_shape_sphericity",
        "la_glszm_grey_level_non_uniformity",
        "la_firstorder_mean_absolute_deviation",
    ]
    rows = []
    for name in wanted:
        if name not in features:
            continue
        rows.append({"feature": name, "auc": round(score(pair[[name]], target, args), 4)})
    table = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print("\nsingle features")
    for row in table.itertuples():
        print(f"  {row.feature:42s} AUC {row.auc:.4f}")
    return table


def ladder(
    pair: pd.DataFrame, target: np.ndarray, features: list[str], args: argparse.Namespace
) -> pd.DataFrame:
    """Forward addition: at every step keep the feature that gains the most."""
    start = args.start if args.start in features else None
    if start is None:
        print(f"{args.start} not in the data, starting from the best single feature")
    ranked = sorted(
        ((score(pair[[name]], target, args), name) for name in features), reverse=True
    )
    candidates = [name for _, name in ranked[: args.candidates]]
    if start is None:
        start = candidates[0]
    chosen = [start]
    current = score(pair[chosen], target, args)
    rows = [
        {
            "step": 1,
            "added": start,
            "model": start,
            "auc": round(current, 4),
            "gain": round(current - 0.5, 4),
        }
    ]
    print(f"\nladder, starting from {start}")
    print(f"  1 {start:42s} AUC {current:.4f}")
    while len(chosen) < args.max_size:
        best_name, best_auc = None, current
        for name in candidates:
            if name in chosen:
                continue
            auc = score(pair[[*chosen, name]], target, args)
            if auc > best_auc:
                best_name, best_auc = name, auc
        if best_name is None:
            print("  no further feature raises the AUC, the ladder stops here")
            break
        chosen.append(best_name)
        rows.append(
            {
                "step": len(chosen),
                "added": best_name,
                "model": " + ".join(chosen),
                "auc": round(best_auc, 4),
                "gain": round(best_auc - current, 4),
            }
        )
        print(
            f"  {len(chosen)} + {best_name:40s} AUC {best_auc:.4f} "
            f"(gain {best_auc - current:+.4f})"
        )
        current = best_auc
    return pd.DataFrame(rows)


def redundancy(
    pair: pd.DataFrame, features: list[str], start: str, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How much of the 300 features is one and the same measurement."""
    block = pair[features].apply(pd.to_numeric, errors="coerce")
    filled = block.fillna(block.median())
    values = StandardScaler().fit_transform(filled.to_numpy(dtype=float))
    parts = PCA(n_components=min(10, values.shape[1])).fit(values)
    share = np.cumsum(parts.explained_variance_ratio_)
    components = pd.DataFrame(
        {
            "component": range(1, len(share) + 1),
            "variance_explained": np.round(parts.explained_variance_ratio_, 4),
            "cumulative": np.round(share, 4),
        }
    )
    print(
        f"\n{len(features)} features, the first 3 components already carry "
        f"{share[min(2, len(share) - 1)]:.1%} of their variance"
    )

    reference = start if start in features else features[0]
    correlation = filled.corrwith(filled[reference]).abs().sort_values(ascending=False)
    pairs = pd.DataFrame(
        {
            "feature": correlation.index,
            f"abs_r_with_{reference}": np.round(correlation.to_numpy(dtype=float), 3),
        }
    )
    high = int((correlation.drop(labels=[reference]) > 0.8).sum())
    print(f"{high} of them correlate above 0.8 with {reference} alone")
    return pairs, components


def main(argv=None) -> None:
    args = parse_args(argv)
    table = read_tables(args)
    indices = add_indices(table)
    features = numeric_features(table)
    pair = table[table["group"].isin([args.group, "control"])].reset_index(drop=True)
    if pair["group"].nunique() < 2:
        raise SystemExit(f"no {args.group} patients")
    target = (pair["group"] == args.group).to_numpy(dtype=int)
    print(
        f"{args.group}: {int(target.sum())} cases, {int(len(target) - target.sum())} controls, "
        f"{len(features)} features including {len(indices)} indices"
    )

    starts = named_starts(pair, target, features, args)
    steps = ladder(pair, target, features, args)
    pairs, components = redundancy(pair, features, args.start, args)

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "feature_ladder.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        steps.to_excel(writer, sheet_name="Ladder", index=False)
        starts.to_excel(writer, sheet_name="Starts", index=False)
        pairs.to_excel(writer, sheet_name="Redundancy", index=False)
        components.to_excel(writer, sheet_name="Components", index=False)

    if not args.no_plots:
        folder = args.output / "plots_index"
        folder.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 4.5))
        plt.plot(steps["step"], steps["auc"], marker="o", color="#3b6ea5")
        for row in steps.itertuples():
            plt.annotate(row.added, (row.step, row.auc), fontsize=6, rotation=30,
                         textcoords="offset points", xytext=(0, 6))
        plt.xlabel("features in the model")
        plt.ylabel("out-of-fold AUC")
        plt.title(f"{args.group} vs control, one feature added at a time")
        plt.tight_layout()
        plt.savefig(folder / f"{args.group.lower()}_ladder.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.bar(components["component"], components["variance_explained"], color="#9a5b3b")
        plt.plot(components["component"], components["cumulative"], marker="o", color="#333333")
        plt.xlabel("principal component")
        plt.ylabel("share of variance")
        plt.title("how few independent directions the features contain")
        plt.tight_layout()
        plt.savefig(folder / f"{args.group.lower()}_redundancy.png", dpi=200)
        plt.close()

    print(f"\nExcel : {path}")


if __name__ == "__main__":
    main()
