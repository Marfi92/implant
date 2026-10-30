"""Learn the AF pattern from the images themselves: no radiomics, encoder-decoder only.

The pipeline of the diagram:

    1-3  crop a fixed size box around the left atrium and its fat, the nnU-Net
         masks only place the box, they are not features
    4-5  train a 3D encoder-decoder with no labels (reconstruction loss), then
         read features out of every encoder level, level 1 being local fat
         texture and the bottleneck global atrial shape
    6    one logistic head per level, plus all levels together, the same
         stratified cross-validation as the feature models, so the AUCs are
         directly comparable
    7    fine-tune the encoder with the label and augmentation, inside the
         training folds only, which is the strongest honest number
    8    the AF direction in the latent space, the line from the control mean to
         the case mean, which gives one learnt score per patient
    9    decode along that direction: a control is slid towards AF and rebuilt at
         every step, and the difference image shows where fat thickens or the
         atrium bulges
    10   cluster the latents with no labels at all, then count AF per cluster
    11   compare with the interpretable model, read from --measurements, which is
         the only place a hand made feature appears, and with the hybrid of both

Leakage: the fine-tuned head (step 7) always trains on the training fold only.
With --fold-safe the autoencoder is retrained inside every fold as well, so no
test patient shapes the representation at all. Without it one autoencoder is
trained on every patient, which is fine for the pictures but makes the frozen
per-level AUCs slightly optimistic.

Crash recovery: the crops are written to crops.npz as they are cut and reused,
the autoencoder saves a checkpoint every few epochs, and each finished stage is
recorded in progress.json. Re-run the same command and it carries on from the
last finished piece; add --fresh to start over.

Input : --images     <patient_id>_0000.nii.gz or <patient_id>.nii.gz
        --la-masks   left atrium masks from nnUNetv2_predict (seg_la)
        --fat-masks  cleaned fat masks from script 11 (fat_hu_masks)
        --groups     any workbook with patient_id and group (fat_la_features.xlsx)
Output: <output>\\representation\\crops.npz           the cropped boxes, reused
        <output>\\representation\\checkpoint*.pt      encoder weights per epoch
        <output>\\representation\\progress.json       what has already finished
        <output>\\representation\\latents_<group>.csv one row per patient
        <output>\\representation\\representation.xlsx AUC per block, clusters
        <output>\\representation\\plots\\*.png        latent map, ROC, decoded
                                                      series, difference map,
                                                      phenotype images

Usage
    python 27_representation_learning.py --group AF
    python 27_representation_learning.py --group AF --fold-safe --epochs 120
    python 27_representation_learning.py --group HF --encoder dinov2
"""

from __future__ import annotations

import argparse
import json
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
BEST_FOUR = [
    "la_firstorder_interquartile_range",
    "la_shape_surface_area",
    "laeat_shape_flatness",
    "eat_firstorder_median",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=SEG / "analysis" / "test_AF_HF_stroke_control")
    parser.add_argument("--la-masks", type=Path, default=SEG / "analysis" / "seg_la")
    parser.add_argument("--fat-masks", type=Path, default=ANALYSIS / "fat_hu_masks")
    parser.add_argument("--groups", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--measurements", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--radiomics", type=Path, default=ANALYSIS / "radiomic_features_full.xlsx")
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument("--group", default="AF", help="AF, HF or stroke")
    parser.add_argument("--encoder", choices=["denovo", "dinov2"], default="denovo")
    parser.add_argument("--box", default="48,128,128", help="z,y,x of the resampled box")
    parser.add_argument("--levels", type=int, default=4, help="pooling stages, box must divide by 2**levels")
    parser.add_argument("--width", type=int, default=24, help="channels of encoder level 1")
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--no-finetune", action="store_true", help="skip the supervised stage")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad-mm", type=float, default=12.0, help="margin around the masks")
    parser.add_argument("--limit", type=int, default=0, help="first N patients, for a trial run")
    parser.add_argument("--fold-safe", action="store_true", help="retrain the autoencoder per fold")
    parser.add_argument("--recrop", action="store_true", help="ignore the cached crops.npz")
    parser.add_argument("--fresh", action="store_true", help="ignore checkpoints and progress.json")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_known_args(argv)[0]
    args.box = tuple(int(part) for part in str(args.box).replace(" ", "").split(","))
    step = 2 ** args.levels
    if len(args.box) != 3 or any(size % step for size in args.box):
        raise SystemExit(f"--box {args.box} must be three sizes, each divisible by {step}")
    return args


# ------------------------------------------------------------ crash recovery


class Journal:
    """What has already finished, so a re-run carries on instead of starting over."""

    def __init__(self, path: Path, fresh: bool):
        self.path = path
        self.data = {}
        if path.exists() and not fresh:
            try:
                self.data = json.loads(path.read_text())
                print(f"resuming, progress.json holds {sorted(self.data)}")
            except json.JSONDecodeError:
                pass

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value) -> None:
        self.data[key] = value
        self.path.write_text(json.dumps(self.data, indent=1))


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
    """One patient as a 2 x box array of bytes: the HU image and the atrium/fat outline."""
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

    # the box is in voxels of the original scan, so resample both channels
    channels = []
    for part in (patch, outline):
        zoom = [args.box[axis] / part.shape[axis] for axis in range(3)]
        channels.append(ndimage.zoom(part, zoom, order=1).astype(np.float32))
    stacked = np.clip(np.stack(channels), 0.0, 1.0)
    return np.round(255 * stacked).astype(np.uint8)  # bytes, so 991 boxes fit in memory


def build_crops(args: argparse.Namespace, patients: list) -> tuple:
    """Cropped boxes, written as they are cut so a crash loses at most 25 patients."""
    cache = args.output / "representation" / "crops.npz"
    done, ids = {}, []
    if cache.exists() and not args.recrop:
        stored = np.load(cache, allow_pickle=True)
        if tuple(stored["box"]) == tuple(args.box):
            for number, name in enumerate(stored["patient_id"]):
                done[str(name)] = stored["crops"][number]
            print(f"crops: {len(done)} already cut, reused from {cache.name}")
        else:
            print(f"crops: cache is box {tuple(stored['box'])}, this run is {args.box}, cutting again")

    images = find_images(args.images)
    la_masks = find_images(args.la_masks) if args.la_masks.exists() else {}
    fat_masks = find_images(args.fat_masks) if args.fat_masks.exists() else {}
    wanted = [name for name in patients if name in images]
    if args.limit:
        wanted = wanted[: args.limit]
    missing = [name for name in wanted if name not in done]
    if missing:
        print(f"  {len(missing)} of {len(wanted)} still to cut")

    def save() -> None:
        order = [name for name in wanted if name in done]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            crops=np.stack([done[name] for name in order]),
            patient_id=np.array(order),
            box=np.array(args.box),
        )

    for number, name in enumerate(missing, 1):
        patch = crop_one(images[name], la_masks.get(name), fat_masks.get(name), args)
        if patch is not None:
            done[name] = patch
        if number % 25 == 0 or number == len(missing):
            print(f"  cropped {number} / {len(missing)}")
            save()
    if not done:
        raise SystemExit("no patient could be cropped, check --images and the mask folders")
    if missing:
        save()
    ids = [name for name in wanted if name in done]
    crops = np.stack([done[name] for name in ids])
    print(f"crops: {crops.shape} in {cache}")
    return crops, ids


def as_float(batch: np.ndarray):
    """Bytes to the [0, 1] tensor the network wants."""
    import torch

    return torch.from_numpy(batch.astype(np.float32) / 255.0)


def augment(batch):
    """Shifts, noise and gain, so the encoder cannot memorise 991 boxes."""
    import torch

    shifted = batch
    for axis in (2, 3, 4):
        move = int(torch.randint(-3, 4, (1,)).item())
        if move:
            shifted = torch.roll(shifted, move, dims=axis)
    gain = 1.0 + 0.1 * (torch.rand(len(batch), 1, 1, 1, 1, device=batch.device) - 0.5)
    noise = 0.02 * torch.randn_like(shifted)
    out = shifted.clone()
    out[:, 0:1] = torch.clamp(shifted[:, 0:1] * gain + noise[:, 0:1], 0.0, 1.0)
    return out


# ------------------------------------------------------------- 4 to 5, encoder


def build_autoencoder(args: argparse.Namespace):
    """A 3D encoder-decoder; the encoder keeps one feature map per level."""
    import torch
    from torch import nn

    widths = tuple(min(args.width * 2 ** step, 192) for step in range(args.levels))

    def block(channels: int, width: int, up: bool):
        layers = [nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)] if up else []
        layers += [
            nn.Conv3d(channels, width, 3, padding=1),
            nn.InstanceNorm3d(width),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(width, width, 3, padding=1),
            nn.InstanceNorm3d(width),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if not up:
            layers.append(nn.AvgPool3d(2))
        return nn.Sequential(*layers)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            channels = 2
            blocks = []
            for width in widths:
                blocks.append(block(channels, width, up=False))
                channels = width
            self.blocks = nn.ModuleList(blocks)
            self.to_latent = nn.Linear(widths[-1], args.latent)

        def forward(self, x):
            levels = []
            for piece in self.blocks:
                x = piece(x)
                levels.append(x)
            return self.to_latent(x.mean(dim=(2, 3, 4))), levels

    class Decoder(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape
            self.channels = widths[-1]
            self.from_latent = nn.Linear(args.latent, self.channels * int(np.prod(shape)))
            channels = self.channels
            blocks = []
            for width in reversed(widths):
                blocks.append(block(channels, width, up=True))
                channels = width
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Conv3d(channels, 2, 1)

        def forward(self, z):
            x = self.from_latent(z).view(-1, self.channels, *self.shape)
            return torch.sigmoid(self.head(self.blocks(x)))

    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder(tuple(size // 2 ** args.levels for size in args.box))

        def forward(self, x):
            z, levels = self.encoder(x)
            return self.decoder(z), z, levels

    return AutoEncoder()


def train_autoencoder(crops: np.ndarray, args, quiet: bool = False, checkpoint: Path | None = None):
    """Reconstruction only, the group label is never seen here."""
    import torch

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_autoencoder(args).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = 1
    if checkpoint is not None and checkpoint.exists() and not args.fresh:
        state = torch.load(checkpoint, map_location=device)
        if state.get("signature") == signature(args) and state["patients"] == len(crops):
            model.load_state_dict(state["model"])
            optimiser.load_state_dict(state["optimiser"])
            first = state["epoch"] + 1
            print(f"  encoder resumed from epoch {state['epoch']} of {args.epochs}")
    if first > args.epochs:
        return model, device

    for epoch in range(first, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(crops))
        total = 0.0
        for start in range(0, len(order), args.batch):
            batch = as_float(crops[order[start : start + args.batch]]).to(device)
            batch = augment(batch)
            rebuilt, _, _ = model(batch)
            # the fat and atrium region counts triple, the air around it once
            weight = 1.0 + 2.0 * batch[:, 1:2]
            loss = (weight * (rebuilt - batch) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(batch)
        if not quiet and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  reconstruction loss {total / len(crops):.5f}")
        if checkpoint is not None and (epoch % 5 == 0 or epoch == args.epochs):
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "epoch": epoch,
                    "patients": len(crops),
                    "signature": signature(args),
                },
                checkpoint,
            )
    return model, device


def signature(args) -> str:
    """Checkpoints only match a run of the same shape."""
    return f"{args.box}-{args.levels}-{args.width}-{args.latent}-{args.epochs}"


def level_features(model, crops: np.ndarray, device, batch: int = 8) -> dict:
    """Mean and SD of every encoder level plus the latent code."""
    import torch

    model.eval()
    collected = {}
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            piece = as_float(crops[start : start + batch]).to(device)
            latent, levels = model.encoder(piece)
            for number, maps in enumerate(levels, 1):
                pooled = torch.cat([maps.mean(dim=(2, 3, 4)), maps.std(dim=(2, 3, 4))], dim=1)
                collected.setdefault(f"level{number}", []).append(pooled.cpu().numpy())
            collected.setdefault("latent", []).append(latent.cpu().numpy())
    blocks = {name: np.concatenate(parts) for name, parts in collected.items()}
    blocks["all levels"] = np.concatenate([blocks[name] for name in sorted(blocks)], axis=1)
    return blocks


def dinov2_features(crops: np.ndarray, args: argparse.Namespace) -> dict:
    """The frozen pretrained comparison: DINOv2 on the slices, mean pooled."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    size = 14 * 16
    out = []
    with torch.no_grad():
        for patch in crops:
            image = as_float(patch[None])[0, 0][:, None].repeat(1, 3, 1, 1)
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


# ------------------------------------------------------- 7, supervised fine-tune


def finetune_cv(crops: np.ndarray, target: np.ndarray, args, folder: Path, journal: Journal) -> tuple:
    """The encoder itself is trained on the label, inside the training fold only."""
    import torch
    from torch import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    # the key carries the settings, so changing them does not reuse the old folds
    tag = f"folds {signature(args)} safe={args.fold_safe} k={args.folds} ft={args.finetune_epochs}"
    done = journal.get(tag) or {}
    scores = np.zeros(len(target), dtype=float)
    folds = []

    for number, (train, test) in enumerate(splitter.split(crops, target), 1):
        key = str(number)
        if key in done:
            scores[test] = np.array(done[key]["scores"])
            folds.append(done[key]["auc"])
            print(f"  fold {number} reused, AUC {folds[-1]:.4f}")
            continue
        print(f"  fold {number} of {args.folds}, {len(train)} train, {len(test)} test")
        if args.fold_safe:
            model, device = train_autoencoder(
                crops[train], args, quiet=True, checkpoint=folder / f"checkpoint_fold{number}.pt"
            )
        else:
            model, device = train_autoencoder(crops, args, quiet=True, checkpoint=folder / "checkpoint.pt")

        head = nn.Sequential(nn.Linear(args.latent, 64), nn.LeakyReLU(0.01), nn.Dropout(0.3), nn.Linear(64, 1))
        head = head.to(device)
        parameters = list(model.encoder.parameters()) + list(head.parameters())
        optimiser = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)
        positives = max(int(target[train].sum()), 1)
        weight = torch.tensor([(len(train) - positives) / positives], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

        for epoch in range(1, args.finetune_epochs + 1):
            model.encoder.train()
            head.train()
            order = np.random.permutation(train)
            total = 0.0
            for start in range(0, len(order), args.batch):
                index = order[start : start + args.batch]
                batch = augment(as_float(crops[index]).to(device))
                labels = torch.from_numpy(target[index].astype(np.float32))[:, None].to(device)
                latent, _ = model.encoder(batch)
                loss = criterion(head(latent), labels)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += float(loss.detach()) * len(index)
            if epoch % 10 == 0:
                print(f"    epoch {epoch:3d}  label loss {total / len(order):.4f}")

        model.encoder.eval()
        head.eval()
        with torch.no_grad():
            piece = []
            for start in range(0, len(test), args.batch):
                batch = as_float(crops[test[start : start + args.batch]]).to(device)
                latent, _ = model.encoder(batch)
                piece.append(torch.sigmoid(head(latent))[:, 0].cpu().numpy())
        scores[test] = np.concatenate(piece)
        folds.append(float(roc_auc_score(target[test], scores[test])))
        print(f"    fold AUC {folds[-1]:.4f}")
        done[key] = {"auc": folds[-1], "scores": scores[test].tolist(), "test": test.tolist()}
        journal.put(tag, done)

    for key, value in done.items():
        scores[np.array(value["test"])] = np.array(value["scores"])
    return float(np.mean(folds)), float(np.std(folds)), scores


# --------------------------------------------------- 11, the hand made reference


def measurement_table(ids: list, args) -> pd.DataFrame | None:
    frames = []
    for path in (args.measurements, args.radiomics):
        if path and path.exists():
            table = pd.read_excel(path, sheet_name="Features")
            table.columns = [str(name).strip() for name in table.columns]
            frames.append(table.set_index("patient_id"))
    if not frames:
        return None
    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.join(extra[[c for c in extra.columns if c not in merged.columns]])
    return merged.reindex(ids)


def reference_auc(merged, target: np.ndarray, args, latent: np.ndarray) -> list:
    """The interpretable model and the hybrid of features plus latent."""
    rows, curves = [], []
    if merged is None:
        return rows, curves
    panels = {"LA volume alone": ["la_volume_ml"], "best four features": BEST_FOUR}
    for name, columns in panels.items():
        have = [c for c in columns if c in merged.columns]
        if len(have) != len(columns):
            print(f"reference '{name}' skipped, missing {set(columns) - set(have)}")
            continue
        values = merged[have].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        auc, spread, scores = cross_validated_auc(values, target, args)
        rows.append({"block": name, "features": len(have), "auc": round(auc, 4), "sd": round(spread, 4)})
        curves.append((name, target, scores, auc))
        if name == "best four features":
            both = np.concatenate([values, latent], axis=1)
            auc, spread, scores = cross_validated_auc(both, target, args)
            rows.append(
                {"block": "hybrid, four features + latent", "features": both.shape[1],
                 "auc": round(auc, 4), "sd": round(spread, 4)}
            )
            curves.append(("hybrid", target, scores, auc))
    return rows, curves


# ---------------------------------------------------------- 8 to 10, the pattern


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
    return rebuilt.cpu().numpy()


def plot_latent(latent, target, args, folder: Path) -> None:
    scaled = StandardScaler().fit_transform(latent)
    try:  # UMAP separates far better than PCA when it is installed
        import umap

        points = umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=args.seed).fit_transform(scaled)
        name = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA

        points = PCA(n_components=2, random_state=args.seed).fit_transform(scaled)
        name = "principal component"
    plt.figure(figsize=(6.5, 6))
    for value, colour, label in ((0, "#5dade2", "control"), (1, "#e74c3c", args.group)):
        plt.scatter(points[target == value, 0], points[target == value, 1], s=12, alpha=0.6, c=colour, label=label)
    plt.legend()
    plt.xlabel(f"{name} 1")
    plt.ylabel(f"{name} 2")
    plt.title(f"Latent space of the images, {args.group} against control")
    plt.tight_layout()
    plt.savefig(folder / "01_latent_space.png", dpi=200)
    plt.close()


def plot_decoded(series: np.ndarray, folder: Path, group: str) -> None:
    """The decoded walk in three planes, image on top, what changed below."""
    image = series[:, 0]
    steps = len(image)
    centre = [size // 2 for size in image.shape[1:]]
    planes = [
        ("axial", lambda volume: volume[centre[0]]),
        ("coronal", lambda volume: volume[:, centre[1]]),
        ("sagittal", lambda volume: volume[:, :, centre[2]]),
    ]
    fig, axes = plt.subplots(2 * len(planes), steps, figsize=(2.3 * steps, 3.4 * len(planes)))
    for row, (name, cut) in enumerate(planes):
        for number in range(steps):
            top = axes[2 * row, number]
            top.imshow(cut(image[number]), cmap="gray", vmin=0, vmax=1)
            top.set_title("control" if number == 0 else f"+{number} step", fontsize=9)
            top.axis("off")
            if number == 0:
                top.set_ylabel(name)
            low = axes[2 * row + 1, number]
            low.imshow(cut(image[number]) - cut(image[0]), cmap="bwr", vmin=-0.25, vmax=0.25)
            low.set_title(f"{name} difference", fontsize=8)
            low.axis("off")
    fig.suptitle(f"A control decoded towards {group}: red gains tissue, blue loses it")
    plt.tight_layout()
    plt.savefig(folder / "03_decoded_towards_group.png", dpi=200)
    plt.close()


def plot_difference_map(series: np.ndarray, folder: Path, group: str) -> None:
    """Where the learnt pattern lives, summed over the whole box."""
    change = series[-1, 0] - series[0, 0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for axis, (name, projection) in zip(
        axes,
        (
            ("axial", change.mean(axis=0)),
            ("coronal", change.mean(axis=1)),
            ("sagittal", change.mean(axis=2)),
        ),
    ):
        limit = float(np.abs(projection).max()) or 1e-3
        picture = axis.imshow(projection, cmap="bwr", vmin=-limit, vmax=limit)
        axis.set_title(f"{name}, mean change")
        axis.axis("off")
        fig.colorbar(picture, ax=axis, fraction=0.046)
    fig.suptitle(f"Where the {group} pattern sits, averaged through the box")
    plt.tight_layout()
    plt.savefig(folder / "05_difference_map.png", dpi=200)
    plt.close()


def plot_phenotypes(crops: np.ndarray, clusters: np.ndarray, table: pd.DataFrame, folder: Path, group: str) -> None:
    """The average heart of each phenotype, so the clusters can be looked at."""
    labels = sorted(set(clusters.tolist()))
    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 4.4))
    axes = np.atleast_1d(axes)
    for axis, label in zip(axes, labels):
        picture = crops[clusters == label, 0].astype(np.float32).mean(axis=0) / 255.0
        axis.imshow(picture[picture.shape[0] // 2], cmap="gray", vmin=0, vmax=1)
        row = table[table["cluster"] == label].iloc[0]
        axis.set_title(f"cluster {label}, n={int(row['patients'])}, {group} {100 * row['prevalence']:.0f}%")
        axis.axis("off")
    fig.suptitle("Mean image of each phenotype, found without labels")
    plt.tight_layout()
    plt.savefig(folder / "06_phenotype_images.png", dpi=200)
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
    journal = Journal(folder / "progress.json", args.fresh)

    groups = pd.read_excel(args.groups, sheet_name="Features")
    groups.columns = [str(name).strip() for name in groups.columns]
    groups = groups[groups["group"].isin([args.group, "control"])]
    wanted = groups["patient_id"].astype(str).tolist()
    print(f"1. {args.group} {int((groups['group'] == args.group).sum())} "
          f"against control {int((groups['group'] == 'control').sum())}")

    crops, ids = build_crops(args, wanted)
    label_of = dict(zip(groups["patient_id"].astype(str), groups["group"]))
    target = np.array([1 if label_of[name] == args.group else 0 for name in ids])
    print(f"2. cropped {len(ids)} patients, {int(target.sum())} {args.group}, box {args.box}")
    if target.sum() < 10 or (len(target) - target.sum()) < 10:
        raise SystemExit("too few patients of one class to cross-validate")

    rows, curves = [], []
    if args.encoder == "dinov2":
        blocks = dinov2_features(crops, args)
        model = device = None
    else:
        print("3. training the encoder-decoder, no labels used")
        model, device = train_autoencoder(crops, args, checkpoint=folder / "checkpoint.pt")
        blocks = level_features(model, crops, device, args.batch)

    print("\n4. AUC per encoder level, same cross-validation as the feature models")
    for name, features in blocks.items():
        auc, spread, scores = cross_validated_auc(features, target, args)
        rows.append({"block": name, "features": features.shape[1], "auc": round(auc, 4), "sd": round(spread, 4)})
        curves.append((name, target, scores, auc))
        print(f"   {name:12s} {features.shape[1]:4d} numbers   AUC {auc:.4f} (SD {spread:.4f})")

    latent = blocks.get("latent", list(blocks.values())[0])

    if not args.no_finetune and args.encoder == "denovo":
        safe = "fold-safe" if args.fold_safe else "shared encoder"
        print(f"\n5. fine-tuning the encoder on the label inside each training fold ({safe})")
        auc, spread, scores = finetune_cv(crops, target, args, folder, journal)
        rows.append({"block": "fine-tuned encoder", "features": -1, "auc": round(auc, 4), "sd": round(spread, 4)})
        curves.append(("fine-tuned encoder", target, scores, auc))
        print(f"   fine-tuned AUC {auc:.4f} (SD {spread:.4f})  <- the number to quote")

    merged = measurement_table(ids, args)
    reference, reference_curves = reference_auc(merged, target, args, latent)
    curves += reference_curves
    if reference:
        print("\n6. the interpretable reference on the same patients")
        for row in reference:
            print(f"   {row['block']:30s} AUC {row['auc']:.4f} (SD {row['sd']:.4f})")

    direction = af_direction(latent, target)
    score = latent @ direction
    print(f"\n7. the {args.group} direction in the latent space scores AUC {roc_auc_score(target, score):.4f} "
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
        plot_phenotypes(crops, clusters, cluster_table, plots, args.group)
        if model is not None:
            control = latent[target == 0][int(np.argmin(score[target == 0]))]
            series = decode_towards(model, device, control, direction)
            plot_decoded(series, plots, args.group)
            plot_difference_map(series, plots, args.group)
        print(f"\nPlots : {plots}")

    workbook = folder / "representation.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows + reference).to_excel(writer, sheet_name="AUC", index=False)
        cluster_table.to_excel(writer, sheet_name="Clusters", index=False)
        latent_table.head(1000).to_excel(writer, sheet_name="Latents", index=False)
    print(f"Excel : {workbook}")
    journal.put("finished", True)


if __name__ == "__main__":
    main()
