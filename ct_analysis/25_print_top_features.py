"""Print the best features for one group. Nothing is written to disk.

Three printed blocks, nothing else.

1. The top --top radiomic features by AUC, with median of the group and of the
   controls so you can see the direction.
2. The measurements of LA, LA-EAT and the fat ROI side by side: volume, mean HU,
   spread, the ratios between the three compartments, the powers and the
   composite indices, each with its AUC.
3. The AUC of a logistic regression and a random forest on the features you
   name with --features, so you can test your own panel in one line.

The AUC of a single feature is the rank AUC, the same number Mann-Whitney gives.
A panel is scored out of fold with stratified cross-validation.

Usage
    python 25_print_top_features.py --group AF --top 20
    python 25_print_top_features.py --features la_volume_ml,eat_volume_ml,la_mad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast\analysis\results")
COMPARTMENTS = {"la": "LA", "laeat": "LA-EAT", "eat": "fat ROI"}
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
    parser.add_argument("--group", default="AF", help="AF, HF or stroke")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--features", default="", help="comma separated panel to score")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
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
    """Ratios, powers and the composite indices of the three component volumes."""
    have = {key: table[column] for key, column in VOLUMES.items() if column in table.columns}
    if len(have) < 3:
        return []
    W, L, E = have["W"], have["L"], have["E"]

    def safe(series):
        return series.replace(0, np.nan)

    made = {
        "ratio_W_over_L": W / safe(L),
        "ratio_E_over_L": E / safe(L),
        "ratio_E_over_W": E / safe(W),
        "sum_W_E_over_L": (W + E) / safe(L),
        "cmi1_W_L": W * L,
        "cmi2_W2_L01_E3": (W**2) * (L**0.1) * (E**3),
        "prod_W_L_E": W * L * E,
        "index_W_L_over_E": (W * L) / safe(E),
        "sqrt_W_L_E": np.cbrt(W * L * E),
        "log_W_L": np.log1p(W * L),
    }
    for name, series in made.items():
        table[name] = series.replace([np.inf, -np.inf], np.nan)
    return list(made)


def rank_auc(values: pd.Series, target: np.ndarray) -> float:
    good = values.notna().to_numpy()
    if good.sum() < 20 or len(np.unique(target[good])) < 2:
        return float("nan")
    return float(roc_auc_score(target[good], values.to_numpy(dtype=float)[good]))


def line(name: str, values: pd.Series, target: np.ndarray) -> str:
    auc = rank_auc(values, target)
    case = values[target == 1].median()
    control = values[target == 0].median()
    direction = "higher" if case > control else "lower"
    return (
        f"  {name:52s} AUC {max(auc, 1 - auc):.4f}  "
        f"{case:12.3f} vs {control:12.3f}  {direction}"
    )


def panel_auc(data: pd.DataFrame, target: np.ndarray, args: argparse.Namespace) -> None:
    values = data.to_numpy(dtype=float)
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    models = {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=args.trees, random_state=args.seed, n_jobs=1),
        ),
    }
    for name, model in models.items():
        predicted = cross_val_predict(model, values, target, cv=folds, method="predict_proba")
        print(f"  {name:10s} AUC {roc_auc_score(target, predicted[:, 1]):.4f}")


def main(argv=None) -> None:
    args = parse_args(argv)
    table = read_tables(args)
    indices = add_indices(table)
    pair = table[table["group"].isin([args.group, "control"])].reset_index(drop=True)
    if pair["group"].nunique() < 2:
        raise SystemExit(f"no {args.group} patients")
    target = (pair["group"] == args.group).to_numpy(dtype=int)
    print(
        f"\n{args.group}: {int(target.sum())} cases against "
        f"{int(len(target) - target.sum())} controls"
    )

    usable = [
        name
        for name in table.columns
        if name not in NOT_A_FEATURE
        and not any(word in name.lower() for word in CLINICAL_WORDS)
        and pd.api.types.is_numeric_dtype(table[name])
        and pair[name].notna().sum() > len(pair) * 0.5
        and pair[name].std(skipna=True) > 0
    ]

    measurement_words = ("volume", "_hu", "_mad", "_iqr", "_sd", "variance", "fraction", "index_ml")
    radiomic = [
        name
        for name in usable
        if name not in indices
        and name.split("_")[0] in COMPARTMENTS
        and not name.endswith("_volume_ml")
    ]
    measurements = [
        name
        for name in usable
        if name not in indices
        and name not in radiomic
        and any(word in name.lower() for word in measurement_words)
    ]

    print(f"\ntop {args.top} radiomics, {len(radiomic)} tested, median {args.group} vs control")
    ranked = sorted(
        radiomic, key=lambda name: -abs(rank_auc(pair[name], target) - 0.5)
    )[: args.top]
    for name in ranked:
        print(line(name, pair[name], target))

    print(f"\nmeasurements, {len(measurements)} of them")
    for prefix, label in COMPARTMENTS.items():
        part = [name for name in measurements if name.startswith(prefix + "_")]
        if not part:
            continue
        print(f" {label}")
        for name in sorted(part, key=lambda name: -abs(rank_auc(pair[name], target) - 0.5)):
            print(line(name, pair[name], target))
    rest = [name for name in measurements if name.split("_")[0] not in COMPARTMENTS]
    if rest:
        print(" whole fat mask and shared")
        for name in sorted(rest, key=lambda name: -abs(rank_auc(pair[name], target) - 0.5)):
            print(line(name, pair[name], target))

    if indices:
        print("\nratios, powers and indices")
        for name in sorted(indices, key=lambda name: -abs(rank_auc(pair[name], target) - 0.5)):
            print(line(name, pair[name], target))

    chosen = [name.strip() for name in args.features.split(",") if name.strip()]
    missing = [name for name in chosen if name not in pair.columns]
    if missing:
        print(f"\nnot in the data: {', '.join(missing)}")
    chosen = [name for name in chosen if name in pair.columns]
    if chosen:
        print(f"\npanel of {len(chosen)}: {' + '.join(chosen)}")
        panel_auc(pair[chosen], target, args)
    else:
        print(f"\nbest {min(4, len(ranked))} radiomics together, out of fold")
        if ranked:
            panel_auc(pair[ranked[:4]], target, args)


if __name__ == "__main__":
    main()
