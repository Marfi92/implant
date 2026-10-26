"""One index, learnt exponents: index = W**alpha * L**beta * E**gamma.

W is the whole EAT / fat ROI volume after the HU window, L the left atrium
volume, E the LA-EAT volume, all in ml, all from script 11 and script 14.

Because log(index) = alpha*log W + beta*log L + gamma*log E, a logistic
regression on the three log volumes gives the exponents directly: the fitted
coefficients are alpha, beta and gamma, and the model probability is a monotone
function of the index, so the index and the model rank patients identically and
share the same AUC.

Per group (AF, HF, stroke against the controls of the same sheet) the script
reports

    alpha, beta, gamma  from the fit on all patients, plus the spread over the
                        cross-validation folds, which shows how stable they are
    AUC                 out-of-fold, stratified --folds cross-validation
                        repeated --repeats times, mean and SD
    rounded index       the exponents rounded to --round, re-scored out of fold
                        as well, so the paper can quote a simple formula
    cut-off             the Youden point of the index with sensitivity and
                        specificity
    reference           AUC of cmi1 = W*L and cmi2 = W**2 * L**0.1 * E**3 and
                        of each volume alone, same cross-validation

Output: <output>\\component_index.xlsx
            Exponents   one row per group, alpha, beta, gamma, AUC, cut-off
            Folds       the exponents of every fold
            Reference   the indices and single volumes, same AUC definition
        <output>\\plots_index\\*.png   ROC per group, index distribution by group

Usage
    python 20_component_index_fit.py --measurements "W:\\...\\fat_la_features.xlsx"
    python 20_component_index_fit.py --round 1 --repeats 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

ANALYSIS = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast\analysis\results")
COMPONENTS = {"W": "eat_volume_ml", "L": "la_volume_ml", "E": "laeat_volume_ml"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--radiomics", type=Path, default=ANALYSIS / "radiomic_features.xlsx")
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--round", type=int, default=1, help="decimals of the simple formula")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_known_args(argv)[0]


def read_volumes(args: argparse.Namespace) -> pd.DataFrame:
    """The three component volumes and the group, from either workbook."""
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

    missing = [column for column in COMPONENTS.values() if column not in table.columns]
    if missing:
        raise SystemExit(f"missing volume columns: {', '.join(missing)}")
    if "group" not in table.columns:
        raise SystemExit("no group column, run scripts 11 and 14 with --groups")

    keep = ["patient_id", "group", *COMPONENTS.values()]
    table = table[keep].replace([np.inf, -np.inf], np.nan).dropna()
    for column in COMPONENTS.values():
        table = table[table[column] > 0]  # a log needs a positive volume
    return table


def log_matrix(table: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.log(table[column].to_numpy(dtype=float)) for column in COMPONENTS.values()])


def index_values(table: pd.DataFrame, exponents: np.ndarray) -> np.ndarray:
    """W**alpha * L**beta * E**gamma, computed in logs to stay in range."""
    return np.exp(log_matrix(table) @ exponents)


def fit_exponents(values: np.ndarray, target: np.ndarray, seed: int) -> np.ndarray:
    """The logistic coefficients on the log volumes are the exponents."""
    model = LogisticRegression(max_iter=5000, C=1e6, random_state=seed)
    model.fit(values, target)
    return model.coef_[0]


def cross_validate(
    values: np.ndarray, target: np.ndarray, args: argparse.Namespace
) -> tuple[float, float, float, np.ndarray, list[dict]]:
    """Out-of-fold AUC of the fitted index, and the exponents of every fold."""
    aucs, rounded_aucs, folds, scores = [], [], [], np.zeros(len(target))
    for repeat in range(args.repeats):
        splitter = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed + repeat
        )
        predicted = np.zeros(len(target))
        rounded = np.zeros(len(target))
        for number, (train, test) in enumerate(splitter.split(values, target), start=1):
            exponents = fit_exponents(values[train], target[train], args.seed + repeat)
            predicted[test] = values[test] @ exponents  # log index, rank is what counts
            rounded[test] = values[test] @ np.round(exponents, args.round)
            folds.append(
                {
                    "repeat": repeat + 1,
                    "fold": number,
                    "alpha_W": round(float(exponents[0]), 4),
                    "beta_L": round(float(exponents[1]), 4),
                    "gamma_E": round(float(exponents[2]), 4),
                }
            )
        aucs.append(roc_auc_score(target, predicted))
        rounded_aucs.append(single_auc(rounded, target))
        scores = predicted
    spread = float(np.std(aucs)) if len(aucs) > 1 else 0.0
    return float(np.mean(aucs)), spread, float(np.mean(rounded_aucs)), scores, folds


def single_auc(score: np.ndarray, target: np.ndarray) -> float:
    """AUC of a ready-made score, flipped if the index runs the other way."""
    auc = roc_auc_score(target, score)
    return float(max(auc, 1 - auc))


def youden(score: np.ndarray, target: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(target, score)
    best = int(np.argmax(tpr - fpr))
    return {
        "cut_off": round(float(thresholds[best]), 4),
        "sensitivity": round(float(tpr[best]), 4),
        "specificity": round(float(1 - fpr[best]), 4),
    }


def reference_rows(
    table: pd.DataFrame, target: np.ndarray, group: str, args: argparse.Namespace
) -> list[dict]:
    """The fixed indices and the single volumes, scored the same way."""
    W = table[COMPONENTS["W"]].to_numpy(dtype=float)
    L = table[COMPONENTS["L"]].to_numpy(dtype=float)
    E = table[COMPONENTS["E"]].to_numpy(dtype=float)
    fixed = {
        "W whole EAT volume": W,
        "L left atrium volume": L,
        "E LA-EAT volume": E,
        "cmi1 = W * L": W * L,
        "cmi2 = W**2 * L**0.1 * E**3": (W**2) * (L**0.1) * (E**3),
        "W * L * E": W * L * E,
    }
    return [
        {"group": group, "score": name, "auc": round(single_auc(np.log(values), target), 4)}
        for name, values in fixed.items()
    ]


def plot_group(best: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    for row in best["roc"]:
        fpr, tpr, _ = roc_curve(row["target"], row["score"])
        plt.plot(fpr, tpr, label=f"{row['group']} AUC {row['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("1 - specificity")
    plt.ylabel("sensitivity")
    plt.title("Fitted component index, out-of-fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "roc_component_index.png", dpi=200)
    plt.close()


def plot_distribution(table: pd.DataFrame, group: str, values: np.ndarray, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"group": table["group"].to_numpy(), "index": values})
    plt.figure(figsize=(5, 5))
    order = ["control", group]
    plt.boxplot([frame.loc[frame["group"] == name, "index"] for name in order])
    plt.xticks([1, 2], order)
    plt.ylabel("log index")
    plt.title(f"{group} against control")
    plt.tight_layout()
    plt.savefig(folder / f"index_{group}.png", dpi=200)
    plt.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    table = read_volumes(args)
    print(f"{len(table)} patients with all three volumes")
    print(table["group"].value_counts().to_string())

    rows, fold_rows, reference, roc_rows = [], [], [], []
    for group in ("AF", "HF", "stroke"):
        pair = table[table["group"].isin([group, "control"])].reset_index(drop=True)
        if pair["group"].nunique() < 2 or len(pair) < 4 * args.folds:
            print(f"{group}: too few patients, skipped")
            continue
        target = (pair["group"] == group).to_numpy(dtype=int)
        values = log_matrix(pair)

        auc, spread, simple_auc, scores, folds = cross_validate(values, target, args)
        exponents = fit_exponents(values, target, args.seed)
        simple = np.round(exponents, args.round)

        cut = youden(scores, target)
        rows.append(
            {
                "group": group,
                "n_cases": int(target.sum()),
                "n_controls": int(len(target) - target.sum()),
                "alpha_W": round(float(exponents[0]), 4),
                "beta_L": round(float(exponents[1]), 4),
                "gamma_E": round(float(exponents[2]), 4),
                "auc_out_of_fold": round(auc, 4),
                "auc_sd": round(spread, 4),
                "alpha_rounded": float(simple[0]),
                "beta_rounded": float(simple[1]),
                "gamma_rounded": float(simple[2]),
                "auc_rounded_index": round(float(simple_auc), 4),
                "formula": (
                    f"W**{simple[0]:g} * L**{simple[1]:g} * E**{simple[2]:g}"
                ),
                **cut,
            }
        )
        fold_rows += [dict(row, group=group) for row in folds]
        reference += reference_rows(pair, target, group, args)
        roc_rows.append({"group": group, "target": target, "score": scores, "auc": auc})
        print(
            f"{group}: AUC {auc:.4f} (SD {spread:.4f}) with "
            f"W**{exponents[0]:.3f} * L**{exponents[1]:.3f} * E**{exponents[2]:.3f}, "
            f"rounded {rows[-1]['formula']} AUC {simple_auc:.4f}"
        )

        if not args.no_plots:
            plot_distribution(pair, group, values @ exponents, args.output / "plots_index")

    if not rows:
        raise SystemExit("no group could be compared")

    args.output.mkdir(parents=True, exist_ok=True)
    workbook = args.output / "component_index.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Exponents", index=False)
        pd.DataFrame(fold_rows).to_excel(writer, sheet_name="Folds", index=False)
        pd.DataFrame(reference).to_excel(writer, sheet_name="Reference", index=False)
    print(f"Excel : {workbook}")

    if not args.no_plots:
        plot_group({"roc": roc_rows}, args.output / "plots_index")
        print(f"Plots : {args.output / 'plots_index'}")


if __name__ == "__main__":
    main()
