"""Radiomics of the learnt feature maps, plus the part of the code the radiomics cannot express.

The question this answers: is there AF information in the CT images beyond the
hand made radiomics, and what is it? Three blocks are built from the same rim
crops and scored in the same folds, so they can be compared honestly.

    A  attenuation radiomics. The established features, computed on the HU
       values (la_firstorder_interquartile_range, la_shape_surface_area,
       laeat_shape_flatness, eat_firstorder_median). This is the 0.788 model.
    B  latent radiomics. The encoder of script 29 turns the box into feature
       maps; every channel of every level is treated as a small 3D image and
       first order and texture statistics are taken inside the atrium, the fat
       and the LA-EAT rim. So this is radiomics of the learnt filters rather
       than of the attenuation, which is the new part.
    C  residual latent. Each latent dimension is regressed on block A inside
       the training fold and only the residual is kept, so C is linearly
       independent of A by construction and can only carry image information
       the radiomics do not have.

Then A, B, C, A+B, A+C and A+B+C are all cross-validated, and a DeLong test
says whether adding B or C to A is significant rather than merely larger.

Fold safety: the encoder is pretrained inside the training fold (no label, and
no test patient), the PCA and the orthogonalisation of C are fitted on the
training fold only, and every classifier is fitted on the training fold only.

Crash recovery: the crops come from the script 29 cache and are not cut again
unless they are missing, each fold keeps its pretrained checkpoint, and each
finished fold is written to progress.json with the settings in the key. Re-run
the identical command to carry on; --fresh throws the progress away.

Input : the same folders as script 29, plus its rim_crops.npz if it exists
Output: <output>\\latent\\latent.xlsx            AUC and DeLong per block
        <output>\\latent\\scores_<group>.csv     out of fold score per block
        <output>\\latent\\plots\\*.png           ROC, where the signal sits

Usage
    python 30_latent_radiomics.py --group AF --limit 60 --pretrain-epochs 5 --folds 3
    python 30_latent_radiomics.py --group AF
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent


def load_rim_module():
    """Script 29 holds the crops, the encoder and the pretraining; reuse them."""
    path = HERE / "29_rim_representation.py"
    if not path.exists():
        raise SystemExit(f"29_rim_representation.py must sit next to this script ({HERE})")
    spec = importlib.util.spec_from_file_location("rim29", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RIM = load_rim_module()
BEST_FOUR = RIM.BEST_FOUR
STATS = ("mean", "sd", "energy", "gradient")
REGIONS = ("atrium", "fat", "rim")


def parse_args(argv=None) -> argparse.Namespace:
    """The script 29 options, plus the few this script adds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, default=48,
                        help="PCA size of the latent radiomics inside each training fold")
    parser.add_argument("--regularisation", type=float, default=0.1, help="C of the logistic models")
    extra = parser.parse_known_args(argv)[0]
    args = RIM.parse_args(argv)
    args.components = extra.components
    args.regularisation = extra.regularisation
    return args


# ---------------------------------------------------------- the three regions


def region_masks(labels, args, level: int):
    """Atrium, fat and LA-EAT rim at the resolution of one encoder level."""
    import torch.nn.functional as F

    rim_voxels = max(int(round(args.rim_mm / args.spacing)), 1)
    atrium = (labels > 0.75).float()
    fat = ((labels > 0.25) & (labels <= 0.75)).float()
    near = F.max_pool3d(atrium, 2 * rim_voxels + 1, stride=1, padding=rim_voxels)
    rim = (fat * near).clamp(0.0, 1.0)
    step = 2 ** level
    return {
        "atrium": F.avg_pool3d(atrium, step),
        "fat": F.avg_pool3d(fat, step),
        "rim": F.avg_pool3d(rim, step),
    }


def channel_statistics(maps, mask):
    """First order and a texture proxy per channel, inside one region.

    mean and sd are the histogram of the channel in that region, energy the mean
    square (how strongly the filter answers there), and gradient the mean
    absolute difference between neighbouring voxels, which is what a texture
    feature measures: how fast the values change.
    """
    import torch

    weight = mask.clamp(min=0.0)
    total = weight.sum(dim=(2, 3, 4)).clamp(min=1e-3)
    mean = (maps * weight).sum(dim=(2, 3, 4)) / total
    centred = maps - mean[:, :, None, None, None]
    var = (weight * centred ** 2).sum(dim=(2, 3, 4)) / total
    energy = (weight * maps ** 2).sum(dim=(2, 3, 4)) / total
    gradient = torch.zeros_like(mean)
    for axis in (2, 3, 4):
        difference = (maps.diff(dim=axis)).abs()
        pair = torch.minimum(weight.narrow(axis, 0, weight.shape[axis] - 1),
                             weight.narrow(axis, 1, weight.shape[axis] - 1))
        gradient = gradient + (difference * pair).sum(dim=(2, 3, 4)) / pair.sum(dim=(2, 3, 4)).clamp(min=1e-3)
    return {"mean": mean, "sd": var.clamp(min=0).sqrt(), "energy": energy, "gradient": gradient / 3.0}


def latent_radiomics(model, device, crops, args) -> tuple:
    """Block B: radiomics of every channel of every level, in the three regions."""
    import torch

    model.eval()
    collected, names = [], None
    with torch.no_grad():
        for start in range(0, len(crops), args.batch):
            batch = RIM.as_float(crops[start : start + args.batch]).to(device)
            _, maps, _ = model.encoder(batch)
            labels = batch[:, 1:2]
            piece, piece_names = [], []
            for level, feature_maps in enumerate(maps, 1):
                masks = region_masks(labels, args, level)
                for region in REGIONS:
                    mask = masks[region]
                    if mask.shape[2:] != feature_maps.shape[2:]:
                        mask = torch.nn.functional.interpolate(mask, size=feature_maps.shape[2:], mode="trilinear",
                                                               align_corners=False)
                    values = channel_statistics(feature_maps, mask)
                    for name in STATS:
                        piece.append(values[name].cpu().numpy())
                        piece_names += [f"L{level}_{region}_{name}_c{channel}"
                                        for channel in range(feature_maps.shape[1])]
            collected.append(np.concatenate(piece, axis=1))
            names = names or piece_names
    return np.concatenate(collected).astype(np.float64), names


# -------------------------------------------------------------- the three blocks


def fit_predict(train_features, train_target, test_features, args) -> np.ndarray:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=6000, C=args.regularisation)),
        ]
    )
    pipeline.fit(train_features, train_target)
    return pipeline.predict_proba(test_features)[:, 1]


def reduce_block(values: np.ndarray, train, args) -> np.ndarray:
    """PCA fitted on the training fold only, so thousands of numbers fit 991 patients."""
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(clean[train])
    size = min(args.components, len(train) - 1, clean.shape[1])
    pca = PCA(n_components=size, random_state=args.seed).fit(scaler.transform(clean[train]))
    return pca.transform(scaler.transform(clean))


def orthogonalise(latent: np.ndarray, concepts: np.ndarray, train) -> np.ndarray:
    """Block C: what is left of the code once block A has been regressed out.

    Each latent dimension is explained as well as possible by the four features
    using the training patients, and the prediction is subtracted. What remains
    is uncorrelated with the four features on the training fold, so an AUC gain
    from it cannot be the radiomics in disguise.
    """
    filled = SimpleImputer(strategy="median").fit(concepts[train]).transform(concepts)
    model = LinearRegression().fit(filled[train], latent[train])
    return latent - model.predict(filled)


def delong(target: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """Two sided p value for two AUCs on the same patients, DeLong's method."""
    target = np.asarray(target).astype(int)
    positive, negative = target == 1, target == 0
    if positive.sum() < 2 or negative.sum() < 2:
        return float("nan")
    aucs, structural = [], []
    for scores in (first, second):
        cases, controls = scores[positive], scores[negative]
        compare = np.sign(cases[:, None] - controls[None, :]) * 0.5 + 0.5
        aucs.append(compare.mean())
        structural.append((compare.mean(axis=1), compare.mean(axis=0)))
    case_matrix = np.stack([piece[0] for piece in structural])
    control_matrix = np.stack([piece[1] for piece in structural])
    case_cov = np.cov(case_matrix) / len(case_matrix[0])
    control_cov = np.cov(control_matrix) / len(control_matrix[0])
    cov = np.atleast_2d(case_cov) + np.atleast_2d(control_cov)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ cov @ contrast)
    if variance <= 0:
        return float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(variance)
    return float(2 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------- plotting


def plot_roc(curves: list, folder: Path, group: str) -> None:
    plt.figure(figsize=(6.8, 6.2))
    for name, target, scores, auc in curves:
        known = np.isfinite(scores)
        fpr, tpr, _ = roc_curve(target[known], scores[known])
        plt.plot(fpr, tpr, lw=1.4, label=f"{name}  AUC {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="0.6", lw=0.8)
    plt.xlabel("1 - specificity")
    plt.ylabel("sensitivity")
    plt.title(f"Attenuation radiomics, latent radiomics and the residual code, {group}")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(folder / "01_roc.png", dpi=200)
    plt.close()


def plot_where(importance: dict, folder: Path, group: str) -> None:
    """Which level, region and statistic of the latent radiomics carries the AF signal."""
    if not importance:
        return
    labels = list(importance)
    values = [importance[name] for name in labels]
    order = np.argsort(values)[::-1][:18]
    plt.figure(figsize=(7.5, 0.34 * len(order) + 1.6))
    plt.barh([labels[number] for number in order][::-1], [values[number] for number in order][::-1],
             color="#4a7ebb")
    plt.xlabel("mean |t| of the group difference, standardised")
    plt.title(f"Where the latent radiomics separate {group} from control")
    plt.tight_layout()
    plt.savefig(folder / "02_where.png", dpi=200)
    plt.close()


def importance_of(values: np.ndarray, names: list, target: np.ndarray) -> dict:
    """Group the thousands of latent radiomics into level x region x statistic."""
    if not names or values.shape[1] != len(names):
        return {}
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = StandardScaler().fit_transform(clean)
    cases, controls = scaled[target == 1], scaled[target == 0]
    spread = np.sqrt(cases.var(axis=0) / max(len(cases), 1) + controls.var(axis=0) / max(len(controls), 1))
    score = np.abs(cases.mean(axis=0) - controls.mean(axis=0)) / np.where(spread > 1e-9, spread, 1.0)
    grouped: dict = {}
    for name, value in zip(names, score):
        key = "_".join(name.split("_")[:3])
        grouped.setdefault(key, []).append(value)
    return {key: float(np.mean(part)) for key, part in grouped.items()}


# ------------------------------------------------------------------------- main


def main(argv=None) -> None:
    args = parse_args(argv)
    folder = args.output / "latent"
    plots = folder / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    journal = RIM.Journal(folder / "progress.json", args.fresh)

    groups = pd.read_excel(args.groups, sheet_name="Features")
    groups.columns = [str(name).strip() for name in groups.columns]
    groups = groups[groups["group"].isin([args.group, "control"])]
    wanted = groups["patient_id"].astype(str).tolist()
    print(f"1. {args.group} {int((groups['group'] == args.group).sum())} "
          f"against control {int((groups['group'] == 'control').sum())}")

    if args.crops is None:
        shared = args.output / "rim" / "rim_crops.npz"
        if shared.exists():
            args.crops = shared
            print(f"   reusing the script 29 crops, {shared}")
    crops, ids = RIM.build_crops(args, wanted, folder)
    label_of = dict(zip(groups["patient_id"].astype(str), groups["group"]))
    target = np.array([1 if label_of[name] == args.group else 0 for name in ids])
    print(f"2. {len(ids)} patients, {int(target.sum())} {args.group}, box {args.box} at {args.spacing} mm")
    if target.sum() < 10 or (len(target) - target.sum()) < 10:
        raise SystemExit("too few patients of one class to cross-validate")

    merged = RIM.measurement_table(ids, args)
    if merged is None:
        raise SystemExit("the workbook with the hand made features is needed for blocks A and C")
    have = [name for name in BEST_FOUR if name in merged.columns]
    if len(have) != len(BEST_FOUR):
        print(f"   WARNING: missing {set(BEST_FOUR) - set(have)}, block A is incomplete")
    concepts = merged[have].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    blocks = ["A attenuation radiomics", "B latent radiomics", "C residual latent",
              "A + B", "A + C", "A + B + C"]
    sums = {name: np.zeros(len(target)) for name in blocks}
    seen = {name: np.zeros(len(target)) for name in blocks}
    per_fold: list = []
    keep_values, keep_names = None, None

    print(f"3. fold-safe, {args.repeats} x {args.folds} folds, encoder pretrained inside each training fold")
    for repeat, number, train, test in RIM.splits_of(crops, target, args):
        key = (f"{RIM.signature(args)} k{args.folds} r{repeat}/{args.repeats} n{len(target)}"
               f" pca{args.components} C{args.regularisation} fold{number}")
        stored = journal.get(key)
        if stored:
            for name in blocks:
                sums[name][test] += np.array(stored["scores"][name], dtype=float)
                seen[name][test] += 1.0
            per_fold.append(stored["aucs"] | {"repeat": repeat, "fold": number})
            print(f"  repeat {repeat} fold {number} reused")
            continue

        print(f"  repeat {repeat} fold {number}, {len(train)} train, {len(test)} test")
        aim = RIM.standardise(concepts, train)
        model, device = RIM.pretrain(
            RIM.Subset(crops, train), aim[train] if aim is not None else None, args,
            folder / f"pretrain_r{repeat}_f{number}.pt", quiet=False,
        )
        values, names = latent_radiomics(model, device, crops, args)
        features = RIM.pooled_features(model, device, crops, np.arange(len(target)), args)
        latent = features["latent"].astype(np.float64)
        if keep_values is None:
            keep_values, keep_names = values, names

        deep = reduce_block(values, train, args)
        residual = orthogonalise(latent, concepts, train)
        sources = {
            "A attenuation radiomics": concepts,
            "B latent radiomics": deep,
            "C residual latent": residual,
            "A + B": np.hstack([concepts, deep]),
            "A + C": np.hstack([concepts, residual]),
            "A + B + C": np.hstack([concepts, deep, residual]),
        }
        fold_scores, aucs = {}, {}
        for name, source in sources.items():
            scores = fit_predict(source[train], target[train], source[test], args)
            fold_scores[name] = scores.tolist()
            aucs[name] = round(float(roc_auc_score(target[test], scores)), 4)
            sums[name][test] += scores
            seen[name][test] += 1.0
        per_fold.append(aucs | {"repeat": repeat, "fold": number})
        short = {"A attenuation radiomics": "A", "B latent radiomics": "B", "C residual latent": "C",
                 "A + B": "A+B", "A + C": "A+C", "A + B + C": "A+B+C"}
        print("    " + "  ".join(f"{short[name]} {value:.3f}" for name, value in aucs.items()))
        journal.put(key, {"scores": fold_scores, "aucs": aucs, "test": test.tolist()})

        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    pooled = {name: np.where(seen[name] > 0, sums[name] / np.maximum(seen[name], 1.0), np.nan) for name in blocks}
    rows, curves = [], []
    print("\n4. AUC per block, the same patients and the same folds")
    for name in blocks:
        folds = [row[name] for row in per_fold if name in row]
        scores = pooled[name]
        known = np.isfinite(scores)
        auc = float(roc_auc_score(target[known], scores[known]))
        p_value = delong(target[known], scores[known], pooled["A attenuation radiomics"][known])
        rows.append({"block": name, "auc": round(auc, 4), "mean_fold_auc": round(float(np.mean(folds)), 4),
                     "sd": round(float(np.std(folds)), 4),
                     "delong_p_vs_A": "" if name.startswith("A ") else round(p_value, 4)})
        curves.append((name, target, scores, auc))
        extra = "" if name.startswith("A ") else f"   DeLong p against A {p_value:.4f}"
        print(f"   {name:26s} AUC {auc:.4f} (fold mean {np.mean(folds):.4f}, SD {np.std(folds):.4f}){extra}")

    best = max(("A + B", "A + C", "A + B + C"), key=lambda name: rows[blocks.index(name)]["auc"])
    gain = rows[blocks.index(best)]["auc"] - rows[blocks.index("A attenuation radiomics")]["auc"]
    verdict = ("the images carry AF information the radiomics do not"
               if gain > 0.01 and (rows[blocks.index(best)]["delong_p_vs_A"] or 1) < 0.05
               else "no information beyond the radiomics could be shown at this sample size")
    print(f"\n5. {best} is the strongest combination, {gain:+.4f} against block A, so {verdict}")

    table = pd.DataFrame({"patient_id": ids, "group": [label_of[name] for name in ids], "case": target})
    for name in blocks:
        table[name.replace(" ", "_").replace("+", "plus")] = pooled[name]
    table.to_csv(folder / f"scores_{args.group}.csv", index=False)

    importance = importance_of(keep_values, keep_names, target) if keep_values is not None else {}
    if not args.no_plots:
        plot_roc(curves, plots, args.group)
        plot_where(importance, plots, args.group)
        print(f"Plots : {plots}")

    workbook = folder / "latent.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="AUC", index=False)
        pd.DataFrame(per_fold).to_excel(writer, sheet_name="Folds", index=False)
        if importance:
            (pd.DataFrame({"block": list(importance), "mean_abs_t": list(importance.values())})
             .sort_values("mean_abs_t", ascending=False)
             .to_excel(writer, sheet_name="Latent_radiomics", index=False))
        table.head(2000).to_excel(writer, sheet_name="Scores", index=False)
    print(f"Excel : {workbook}")
    journal.put("finished", True)


if __name__ == "__main__":
    main()
