"""Learn the AF pattern from the images themselves: no radiomics, encoder-decoder only.

The pipeline of the diagram:

    1-3  crop a fixed size box around the left atrium and its fat, the nnU-Net
         masks only place the box, they are not features
    4-5  train a 3D encoder-decoder with no labels (reconstruction loss), then
         read features out of every encoder level, level 1 being local fat
         texture and the bottleneck global atrial shape
    6    one logistic head per level, the same stratified cross-validation as the
         feature models, so the AUCs are directly comparable
    7    the AF direction in the latent space, the line from the control mean to
         the case mean, which gives one learnt score per patient
    8    decode along that direction: a control is slid towards AF and rebuilt at
         every step, and the difference image shows where fat thickens or the
         atrium bulges
    9    cluster the latents with no labels at all, then count AF per cluster
    10   compare with the interpretable model, read from --measurements, which is
         the only place a hand made feature appears

Leakage: with --fold-safe the autoencoder is retrained inside every training
fold, so no test patient ever shapes the representation. Without it one encoder
is trained on all patients (faster, fine for looking at the latent space, but
the AUC is then optimistic and must not be quoted).

Input : --images     <patient_id>_0000.nii.gz or <patient_id>.nii.gz
        --la-masks   left atrium masks from nnUNetv2_predict
        --fat-masks  cleaned fat masks from script 11 (fat_hu_masks)
        --groups     any workbook with patient_id and group (fat_la_features.xlsx)
Output: <output>\\representation\\crops.npz           the cropped boxes, reused
        <output>\\representation\\latents_<group>.csv one row per patient
        <output>\\representation\\representation.xlsx AUC per level, clusters
        <output>\\representation\\plots\\*.png        latent scatter, ROC,
                                                      decoded series, clusters

Usage
    python 27_representation_learning.py --group AF
    python 27_representation_learning.py --group AF --fold-safe --epochs 60
    python 27_representation_learning.py --group HF --encoder dinov2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast\analysis\results")
SEG = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast")
HU_WINDOW = (-200.0, 200.0)  # fat at -100, non contrast blood and wall near +40
BOX = (32, 96, 96)  # z, y, x of the resampled box, 3 mm data


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=SEG / "images")
    parser.add_argument("--la-masks", type=Path, default=SEG / "la_masks")
    parser.add_argument("--fat-masks", type=Path, default=SEG / "analysis" / "fat_hu_masks")
    parser.add_argument("--groups", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--measurements", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--radiomics", type=Path, default=ANALYSIS / "radiomic_features_full.xlsx")
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument("--group", default="AF", help="AF, HF or stroke")
    parser.add_argument("--encoder", choices=["denovo", "dinov2"], default="denovo")
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad-mm", type=float, default=10.0, help="margin around the masks")
    parser.add_argument("--limit", type=int, default=0, help="first N patients, for a trial run")
    parser.add_argument("--fold-safe", action="store_true", help="retrain the encoder per fold")
    parser.add_argument("--recrop", action="store_true", help="ignore the cached crops.npz")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_known_args(argv)[0]


# ---------------------------------------------------------------- 1 to 3, crops


def patient_of(name: str) -> str:
    stem = name.replace(".nii.gz", "").replace(".nii", "")
    return stem[:-5] if stem.endswith("_0000") else stem


def find_images(folder: Path) -> dict:
    found = {}
    for path in sorted(folder.glob("*.nii*")):
        found.setdefault(patient_of(path.name), path)
    return found


def crop_one(image_path: Path, la_path: Path, fat_path: Path, args) -> np.ndarray | None:
    """One patient as a 2 x BOX array: the HU image and the atrium/fat outline."""
    volume = nib.load(str(image_path))
    image = np.asarray(volume.dataobj, dtype=np.float32)
    spacing = np.abs(np.diag(volume.affine))[:3]

    masks = []
    for path in (la_path, fat_path):
        if path is None or not path.exists():
            masks.append(np.zeros_like(image, dtype=np.uint8))
            continue
        mask = np.asarray(nib.load(str(path)).dataobj)
        if mask.shape != image.shape:
            return None
        masks.append((mask > 0).astype(np.uint8))
    la, fat = masks
    both = la | fat
    if both.sum() < 100:
        return None

    pad = np.maximum((args.pad_mm / np.maximum(spacing, 1e-3)).astype(int), 1)
    box = ndimage.find_objects(both)[0]
    slices = []
    for axis, piece in enumerate(box):
        start = max(int(piece.start) - int(pad[axis]), 0)
        stop = min(int(piece.stop) + int(pad[axis]), image.shape[axis])
        slices.append(slice(start, stop))
    slices = tuple(slices)

    low, high = HU_WINDOW
    patch = np.clip(image[slices], low, high)
    patch = (patch - low) / (high - low)
    outline = (la[slices].astype(np.float32) + 0.5 * fat[slices].astype(np.float32)) / 1.5

    # the box is in voxels of the original scan, so resample both channels to BOX
    channels = []
    for part in (patch, outline):
        zoom = [BOX[axis] / part.shape[axis] for axis in range(3)]
        channels.append(ndimage.zoom(part, zoom, order=1).astype(np.float32))
    stacked = np.stack(channels)
    return np.clip(stacked, 0.0, 1.0)


def build_crops(args: argparse.Namespace, patients: list) -> tuple:
    """Cropped boxes for the patients that have an image and a mask."""
    cache = args.output / "representation" / "crops.npz"
    if cache.exists() and not args.recrop:
        stored = np.load(cache, allow_pickle=True)
        ids = [str(name) for name in stored["patient_id"]]
        keep = [i for i, name in enumerate(ids) if name in set(patients)]
        print(f"crops: {len(keep)} of {len(ids)} reused from {cache.name}")
        return stored["crops"][keep], [ids[i] for i in keep]

    images = find_images(args.images)
    la_masks = find_images(args.la_masks) if args.la_masks.exists() else {}
    fat_masks = find_images(args.fat_masks) if args.fat_masks.exists() else {}
    crops, ids = [], []
    wanted = [name for name in patients if name in images]
    if args.limit:
        wanted = wanted[: args.limit]
    for number, name in enumerate(wanted, 1):
        patch = crop_one(images[name], la_masks.get(name), fat_masks.get(name), args)
        if patch is None:
            continue
        crops.append(patch)
        ids.append(name)
        if number % 25 == 0:
            print(f"  cropped {number} / {len(wanted)}")
    if not crops:
        raise SystemExit("no patient could be cropped, check --images and the mask folders")
    crops = np.stack(crops)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, crops=crops, patient_id=np.array(ids))
    print(f"crops: {crops.shape} saved to {cache}")
    return crops, ids


# ------------------------------------------------------------- 4 to 5, encoder


def build_autoencoder(args: argparse.Namespace):
    """A small 3D encoder-decoder; the encoder keeps one feature map per level."""
    import torch
    from torch import nn

    class Encoder(nn.Module):
        def __init__(self, latent: int):
            super().__init__()
            widths = (16, 32, 64, 64)
            blocks, channels = [], 2
            for width in widths:
                blocks.append(
                    nn.Sequential(
                        nn.Conv3d(channels, width, 3, padding=1),
                        nn.InstanceNorm3d(width),
                        nn.LeakyReLU(0.01, inplace=True),
                        nn.Conv3d(width, width, 3, padding=1),
                        nn.InstanceNorm3d(width),
                        nn.LeakyReLU(0.01, inplace=True),
                        nn.AvgPool3d(2),
                    )
                )
                channels = width
            self.blocks = nn.ModuleList(blocks)
            self.to_latent = nn.Linear(widths[-1], latent)

        def forward(self, x):
            levels = []
            for block in self.blocks:
                x = block(x)
                levels.append(x)
            pooled = x.mean(dim=(2, 3, 4))
            return self.to_latent(pooled), levels

    class Decoder(nn.Module):
        def __init__(self, latent: int, shape):
            super().__init__()
            self.shape = shape
            self.from_latent = nn.Linear(latent, 64 * int(np.prod(shape)))
            widths = (64, 32, 16, 16)
            blocks, channels = [], 64
            for width in widths:
                blocks.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                        nn.Conv3d(channels, width, 3, padding=1),
                        nn.InstanceNorm3d(width),
                        nn.LeakyReLU(0.01, inplace=True),
                    )
                )
                channels = width
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Conv3d(channels, 2, 1)

        def forward(self, z):
            x = self.from_latent(z).view(-1, 64, *self.shape)
            return torch.sigmoid(self.head(self.blocks(x)))

    class AutoEncoder(nn.Module):
        def __init__(self, latent: int):
            super().__init__()
            self.encoder = Encoder(latent)
            self.decoder = Decoder(latent, tuple(size // 16 for size in BOX))

        def forward(self, x):
            z, levels = self.encoder(x)
            return self.decoder(z), z, levels

    return AutoEncoder(args.latent)


def train_autoencoder(crops: np.ndarray, args: argparse.Namespace, quiet: bool = False):
    """Reconstruction only, the group label is never seen here."""
    import torch

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_autoencoder(args).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.from_numpy(crops)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(data))
        total = 0.0
        for start in range(0, len(order), args.batch):
            batch = data[order[start : start + args.batch]].to(device)
            rebuilt, _, _ = model(batch)
            # the fat and atrium region counts double, the air around it once
            weight = 1.0 + 2.0 * batch[:, 1:2]
            loss = (weight * (rebuilt - batch) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(batch)
        if not quiet and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  reconstruction loss {total / len(data):.5f}")
    return model, device


def level_features(model, crops: np.ndarray, device, batch: int = 8) -> dict:
    """Mean and SD of every encoder level plus the latent code."""
    import torch

    model.eval()
    collected = {}
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            piece = torch.from_numpy(crops[start : start + batch]).to(device)
            latent, levels = model.encoder(piece)
            for number, maps in enumerate(levels, 1):
                pooled = torch.cat([maps.mean(dim=(2, 3, 4)), maps.std(dim=(2, 3, 4))], dim=1)
                collected.setdefault(f"level{number}", []).append(pooled.cpu().numpy())
            collected.setdefault("latent", []).append(latent.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def dinov2_features(crops: np.ndarray, args: argparse.Namespace) -> dict:
    """The frozen pretrained comparison: DINOv2 on the slices, mean pooled."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    size = 14 * 16
    out = []
    with torch.no_grad():
        for patch in crops:
            image = torch.from_numpy(patch[0])[:, None].repeat(1, 3, 1, 1)
            image = torch.nn.functional.interpolate(image, (size, size), mode="bilinear")
            tokens = backbone(image.to(device))
            out.append(torch.cat([tokens.mean(0), tokens.std(0)]).cpu().numpy())
    return {"dinov2": np.stack(out)}


# ------------------------------------------------------------------- 6, models


def cross_validated_auc(features: np.ndarray, target: np.ndarray, args) -> tuple:
    """Out-of-fold AUC of a logistic head on one feature block."""
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    scores = np.zeros(len(target), dtype=float)
    folds = []
    for train, test in splitter.split(features, target):
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, C=0.1)),
            ]
        )
        pipeline.fit(features[train], target[train])
        scores[test] = pipeline.predict_proba(features[test])[:, 1]
        folds.append(roc_auc_score(target[test], scores[test]))
    return float(np.mean(folds)), float(np.std(folds)), scores


def fold_safe_auc(crops: np.ndarray, target: np.ndarray, args) -> tuple:
    """The representation is retrained inside every training fold."""
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    scores = np.zeros(len(target), dtype=float)
    folds = []
    for number, (train, test) in enumerate(splitter.split(crops, target), 1):
        print(f"  fold {number} of {args.folds}, training the encoder on {len(train)} patients")
        model, device = train_autoencoder(crops[train], args, quiet=True)
        blocks = level_features(model, crops, device, args.batch)
        features = np.concatenate([blocks["latent"], blocks["level4"]], axis=1)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, C=0.1)),
            ]
        )
        pipeline.fit(features[train], target[train])
        scores[test] = pipeline.predict_proba(features[test])[:, 1]
        folds.append(roc_auc_score(target[test], scores[test]))
        print(f"    fold AUC {folds[-1]:.4f}")
    return float(np.mean(folds)), float(np.std(folds)), scores


def reference_auc(ids: list, target: np.ndarray, args) -> list:
    """The interpretable model, read from the workbooks, as a comparison line."""
    frames = []
    for path in (args.measurements, args.radiomics):
        if path and path.exists():
            table = pd.read_excel(path, sheet_name="Features")
            table.columns = [str(name).strip() for name in table.columns]
            frames.append(table.set_index("patient_id"))
    if not frames:
        return []
    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.join(extra[[c for c in extra.columns if c not in merged.columns]])
    merged = merged.reindex(ids)

    panels = {
        "LA volume alone": ["la_volume_ml"],
        "best four features": [
            "la_firstorder_interquartile_range",
            "la_shape_surface_area",
            "laeat_shape_flatness",
            "eat_firstorder_median",
        ],
    }
    rows = []
    for name, columns in panels.items():
        have = [c for c in columns if c in merged.columns]
        if len(have) != len(columns):
            print(f"reference '{name}' skipped, missing {set(columns) - set(have)}")
            continue
        values = merged[have].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        auc, spread, _ = cross_validated_auc(values, target, args)
        rows.append({"block": name, "features": len(have), "auc": round(auc, 4), "sd": round(spread, 4)})
    return rows


# --------------------------------------------------------- 7 to 9, the pattern


def af_direction(latent: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The line from the control mean to the case mean, unit length."""
    direction = latent[target == 1].mean(axis=0) - latent[target == 0].mean(axis=0)
    return direction / (np.linalg.norm(direction) + 1e-9)


def decode_towards(model, device, latent: np.ndarray, direction: np.ndarray, steps: int = 5):
    """Slide one control along the AF direction and rebuild it at every step."""
    import torch

    spread = float(np.std(latent @ direction))
    walk = np.stack([latent + factor * spread * direction for factor in np.linspace(0, 3, steps)])
    with torch.no_grad():
        rebuilt = model.decoder(torch.from_numpy(walk.astype(np.float32)).to(device))
    return rebuilt[:, 0].cpu().numpy()


def plot_latent(latent, target, args, folder: Path) -> None:
    from sklearn.decomposition import PCA

    points = PCA(n_components=2, random_state=args.seed).fit_transform(StandardScaler().fit_transform(latent))
    plt.figure(figsize=(6.5, 6))
    for value, colour, label in ((0, "#5dade2", "control"), (1, "#e74c3c", args.group)):
        plt.scatter(
            points[target == value, 0],
            points[target == value, 1],
            s=12,
            alpha=0.6,
            c=colour,
            label=label,
        )
    plt.legend()
    plt.xlabel("latent component 1")
    plt.ylabel("latent component 2")
    plt.title(f"Latent space of the images, {args.group} against control")
    plt.tight_layout()
    plt.savefig(folder / "01_latent_space.png", dpi=200)
    plt.close()


def plot_decoded(series: np.ndarray, folder: Path, group: str) -> None:
    middle = series.shape[1] // 2
    steps = len(series)
    fig, axes = plt.subplots(2, steps, figsize=(2.6 * steps, 5.6))
    for number in range(steps):
        axes[0, number].imshow(series[number, middle], cmap="gray", vmin=0, vmax=1)
        axes[0, number].set_title("start" if number == 0 else f"+{number} step", fontsize=10)
        axes[0, number].axis("off")
        difference = series[number, middle] - series[0, middle]
        axes[1, number].imshow(difference, cmap="bwr", vmin=-0.3, vmax=0.3)
        axes[1, number].set_title("difference", fontsize=9)
        axes[1, number].axis("off")
    fig.suptitle(f"A control decoded towards {group}: red gains tissue, blue loses it")
    plt.tight_layout()
    plt.savefig(folder / "03_decoded_towards_group.png", dpi=200)
    plt.close()


def plot_roc(curves: list, folder: Path, group: str) -> None:
    plt.figure(figsize=(6.5, 6))
    for name, target, scores, auc in curves:
        fpr, tpr, _ = roc_curve(target, scores)
        plt.plot(fpr, tpr, lw=1.4, label=f"{name}  AUC {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="0.6", lw=0.8)
    plt.xlabel("1 - specificity")
    plt.ylabel("sensitivity")
    plt.title(f"Learnt representation against the feature model, {group}")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(folder / "02_roc_levels.png", dpi=200)
    plt.close()


def plot_clusters(table: pd.DataFrame, folder: Path, group: str) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.bar(table["cluster"].astype(str), 100 * table["prevalence"], color="#884ea0")
    for number, row in table.iterrows():
        plt.text(number, 100 * row["prevalence"] + 0.6, f"n={int(row['patients'])}", ha="center", fontsize=9)
    plt.ylabel(f"{group} prevalence (%)")
    plt.xlabel("cluster of the latent space, found without labels")
    plt.title(f"{group} prevalence per image phenotype")
    plt.tight_layout()
    plt.savefig(folder / "04_clusters.png", dpi=200)
    plt.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    folder = args.output / "representation"
    plots = folder / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    groups = pd.read_excel(args.groups, sheet_name="Features")
    groups.columns = [str(name).strip() for name in groups.columns]
    groups = groups[groups["group"].isin([args.group, "control"])]
    wanted = groups["patient_id"].astype(str).tolist()
    print(f"1. {args.group} {int((groups['group'] == args.group).sum())} "
          f"against control {int((groups['group'] == 'control').sum())}")

    crops, ids = build_crops(args, wanted)
    label_of = dict(zip(groups["patient_id"].astype(str), groups["group"]))
    target = np.array([1 if label_of[name] == args.group else 0 for name in ids])
    print(f"2. cropped {len(ids)} patients, {int(target.sum())} {args.group}, box {BOX}")
    if target.sum() < 10 or (len(target) - target.sum()) < 10:
        raise SystemExit("too few patients of one class to cross-validate")

    rows, curves = [], []
    if args.encoder == "dinov2":
        blocks = dinov2_features(crops, args)
        model = device = None
    else:
        print("3. training the encoder-decoder, no labels used")
        model, device = train_autoencoder(crops, args)
        blocks = level_features(model, crops, device, args.batch)

    print("\n4. AUC per encoder level, same cross-validation as the feature models")
    for name, features in blocks.items():
        auc, spread, scores = cross_validated_auc(features, target, args)
        rows.append({"block": name, "features": features.shape[1], "auc": round(auc, 4), "sd": round(spread, 4)})
        curves.append((name, target, scores, auc))
        print(f"   {name:8s} {features.shape[1]:4d} numbers   AUC {auc:.4f} (SD {spread:.4f})")

    if args.fold_safe and args.encoder == "denovo":
        print("\n5. fold-safe check, the encoder is retrained inside every training fold")
        auc, spread, scores = fold_safe_auc(crops, target, args)
        rows.append({"block": "fold-safe latent+level4", "features": -1, "auc": round(auc, 4), "sd": round(spread, 4)})
        curves.append(("fold-safe", target, scores, auc))
        print(f"   fold-safe AUC {auc:.4f} (SD {spread:.4f})  <- the number to quote")

    reference = reference_auc(ids, target, args)
    for row in reference:
        curves.append((row["block"], target, np.zeros(len(target)), row["auc"]))
    curves = [curve for curve in curves if curve[2].any()]
    if reference:
        print("\n6. the interpretable reference on the same patients")
        for row in reference:
            print(f"   {row['block']:20s} AUC {row['auc']:.4f} (SD {row['sd']:.4f})")

    latent = blocks.get("latent", list(blocks.values())[0])
    direction = af_direction(latent, target)
    score = latent @ direction
    learnt_auc = roc_auc_score(target, score)
    print(f"\n7. the {args.group} direction in the latent space scores AUC {learnt_auc:.4f} "
          "in sample (not cross-validated, for the picture only)")

    clusters = KMeans(n_clusters=args.clusters, n_init=10, random_state=args.seed).fit_predict(
        StandardScaler().fit_transform(latent)
    )
    cluster_table = (
        pd.DataFrame({"cluster": clusters, "case": target})
        .groupby("cluster")
        .agg(patients=("case", "size"), cases=("case", "sum"))
        .reset_index()
    )
    cluster_table["prevalence"] = cluster_table["cases"] / cluster_table["patients"]
    print(f"\n8. image phenotypes found without labels, {args.group} prevalence per cluster")
    print(cluster_table.to_string(index=False))

    latent_table = pd.DataFrame(latent, columns=[f"z{number + 1}" for number in range(latent.shape[1])])
    latent_table.insert(0, "patient_id", ids)
    latent_table.insert(1, "group", [label_of[name] for name in ids])
    latent_table["direction_score"] = score
    latent_table["cluster"] = clusters
    latent_table.to_csv(folder / f"latents_{args.group}.csv", index=False)

    if not args.no_plots:
        plot_latent(latent, target, args, plots)
        if curves:
            plot_roc(curves, plots, args.group)
        plot_clusters(cluster_table, plots, args.group)
        if model is not None:
            control = latent[target == 0][int(np.argmin(score[target == 0]))]
            plot_decoded(decode_towards(model, device, control, direction), plots, args.group)
        print(f"\nPlots : {plots}")

    workbook = folder / "representation.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows + reference).to_excel(writer, sheet_name="AUC", index=False)
        cluster_table.to_excel(writer, sheet_name="Clusters", index=False)
        latent_table.head(1000).to_excel(writer, sheet_name="Latents", index=False)
    print(f"Excel : {workbook}")


if __name__ == "__main__":
    main()
