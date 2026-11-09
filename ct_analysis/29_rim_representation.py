"""The learnt representation, rebuilt around the fat rim: every improvement in one script.

Why this exists: script 27 learnt 0.74 and 0.753 fine-tuned against 0.788 for the
four hand made features, because a plain autoencoder on a whole atrium box spends
its capacity on big bright structures and almost none on the thin fat rim where
the AF signal sits. Five changes here, all of them in the same run:

    1  rim crops at a fixed fine spacing. The box is resampled to --spacing mm
       (default 1.5) and centred on the left atrium, so the atrial wall and the
       LA-EAT rim keep their texture instead of being squashed to 3 mm.
    2  the loss is masked to where the signal is. A shell is grown from the
       atrial surface (--rim-mm) and the fat touching it; voxels outside the
       shell count only --background times as much, so the code must encode fat
       texture to lower the loss.
    3  attention pooling per encoder level instead of mean and SD, so each level
       is read where it is informative, not averaged over air as well.
    4  distillation: the pretrained code also has to predict the four hand made
       features, standardised inside the training fold, so it is forced to carry
       at least what the 0.788 model carries. No AF label is used for this.
    5  SimSiam contrastive pretraining of two augmented views, added to a masked
       reconstruction loss, which beats reconstruction alone at this sample size.

    6  on top: several seeds per fold averaged, test time augmentation, and
       repeated k-fold, which is worth about 0.01 and tightens the interval.

Honesty: everything that touches a test patient is trained inside the training
fold only, pretraining included, so the AUC printed here needs no caveat. That
costs folds x repeats pretrainings, which is why it is slow.

Crash recovery: the crops are written to rim_crops.npz as they are cut, every
fold keeps its own pretrained checkpoint, and each finished fold is stored in
progress.json with the settings in the key. Re-run the identical command and it
carries on; --fresh throws the progress away, --recrop cuts the boxes again.

Input : --images     <patient_id>_0000.nii.gz or <patient_id>.nii.gz
        --la-masks   atrium masks from nnUNetv2_predict (seg_la)
        --fat-masks  cleaned fat masks from script 11 (fat_hu_masks)
        --groups     workbook with patient_id and group (fat_la_features.xlsx)
Output: <output>\\rim\\rim_crops.npz        the cropped boxes, reused
        <output>\\rim\\progress.json        finished folds
        <output>\\rim\\rim.xlsx             AUC per block and per fold
        <output>\\rim\\scores_<group>.csv   one out of fold score per patient
        <output>\\rim\\plots\\*.png         ROC, latent map, where it looks

Usage
    python 29_rim_representation.py --group AF --limit 40 --pretrain-epochs 5 --finetune-epochs 5
    python 29_rim_representation.py --group AF
    python 29_rim_representation.py --group AF --folds 10 --repeats 2 --seeds 3
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast\analysis\results")
SEG = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast")
BEST_FOUR = [
    "la_firstorder_interquartile_range",
    "la_shape_surface_area",
    "laeat_shape_flatness",
    "eat_firstorder_median",
]
LA_VALUE, FAT_VALUE = 255, 128  # how the two masks are stored in channel 2


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=SEG / "analysis" / "test_AF_HF_stroke_control")
    parser.add_argument("--la-masks", type=Path, default=SEG / "analysis" / "seg_la")
    parser.add_argument("--fat-masks", type=Path, default=ANALYSIS / "fat_hu_masks")
    parser.add_argument("--groups", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--measurements", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--radiomics", type=Path, default=ANALYSIS / "radiomic_features_full.xlsx")
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument("--group", default="AF",
                        help="AF, HF or stroke, or several pooled into one case group, "
                             "e.g. --group AF,HF,stroke for diseased against control")

    parser.add_argument("--box", default="48,128,128", help="z,y,x voxels of the resampled box")
    parser.add_argument("--spacing", type=float, default=1.5, help="mm per voxel of the box, the fine resolution")
    parser.add_argument("--rim-mm", type=float, default=10.0, help="thickness of the shell around the atrium")
    parser.add_argument("--background", type=float, default=0.05, help="weight of voxels outside the shell")
    parser.add_argument("--hu", default="-200,200", help="HU window of channel 1")

    parser.add_argument("--levels", type=int, default=4, help="pooling stages, box must divide by 2**levels")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--mask-fraction", type=float, default=0.4, help="share of blocks hidden from the decoder")
    parser.add_argument("--simsiam", type=float, default=1.0, help="weight of the contrastive term")
    parser.add_argument("--distil", type=float, default=1.0, help="weight of the four feature distillation")

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1, help="repeated k-fold, each with a new split")
    parser.add_argument("--seeds", type=int, default=2, help="networks per fold, averaged")
    parser.add_argument("--tta", type=int, default=4, help="test time augmentations, 1 turns it off")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--limit", type=int, default=0, help="first N patients, for a trial run")
    parser.add_argument("--crops", type=Path, default=None, help="a crop cache to reuse")
    parser.add_argument("--recrop", action="store_true", help="cut the boxes again")
    parser.add_argument("--fresh", action="store_true", help="ignore progress.json and the checkpoints")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_known_args(argv)[0]
    args.box = tuple(int(part) for part in str(args.box).replace(" ", "").split(","))
    args.hu = tuple(float(part) for part in str(args.hu).replace(" ", "").split(","))
    step = 2 ** args.levels
    if len(args.box) != 3 or any(size % step for size in args.box):
        raise SystemExit(f"--box {args.box} must be three sizes, each divisible by {step}")
    args.cases = case_labels(args.group)
    args.tag = "_".join(args.cases)  # for file names and titles when groups are pooled
    return args


def case_labels(group: str) -> list:
    """One disease, or several pooled: --group AF,HF,stroke is diseased against control."""
    return [name.strip() for name in str(group).split(",") if name.strip()]


def signature(args) -> str:
    """Two runs share a checkpoint only when every number that shapes it matches."""
    return (f"{args.box}@{args.spacing}mm rim{args.rim_mm} bg{args.background} L{args.levels}"
            f" w{args.width} z{args.latent} pre{args.pretrain_epochs} ft{args.finetune_epochs}"
            f" mask{args.mask_fraction} sim{args.simsiam} dis{args.distil}")


class Journal:
    """What has already finished, so a re-run carries on instead of starting over."""

    def __init__(self, path: Path, fresh: bool):
        self.path = path
        self.data = {}
        if path.exists() and not fresh:
            try:
                self.data = json.loads(path.read_text())
                print(f"resuming, progress.json holds {len(self.data)} entries")
            except json.JSONDecodeError:
                pass

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value) -> None:
        self.data[key] = value
        self.path.write_text(json.dumps(self.data, indent=1))


# -------------------------------------------------------------- 1, the rim crop


def patient_of(name: str) -> str:
    stem = name.replace(".nii.gz", "").replace(".nii", "")
    return stem[:-5] if stem.endswith("_0000") else stem


def find_images(folder: Path) -> dict:
    found = {}
    for path in sorted(folder.glob("*.nii*")):
        found.setdefault(patient_of(path.name), path)
    return found


def crop_one(image_path: Path, la_path: Path, fat_path: Path, args) -> np.ndarray | None:
    """One patient as 2 x box bytes, resampled to --spacing mm around the atrium.

    Channel 1 is the HU window, channel 2 holds the atrium (255) and the fat (128),
    which is all the shell in step 2 needs.
    """
    volume = nib.load(str(image_path))
    image = np.asarray(volume.dataobj, dtype=np.float32)
    spacing = np.abs(np.diag(volume.affine))[:3]
    spacing = np.where(spacing > 1e-3, spacing, 1.0)

    masks = []
    for path in (la_path, fat_path):
        if path is None or not path.exists():
            masks.append(np.zeros(image.shape, dtype=np.uint8))
            continue
        mask = np.asarray(nib.load(str(path)).dataobj)
        if mask.shape != image.shape:
            return None
        masks.append((mask > 0).astype(np.uint8))
    la, fat = masks
    anchor = la if la.sum() >= 100 else (la | fat)
    if anchor.sum() < 100:
        return None

    # the box is centred on the atrium and sampled at a fixed mm step, so the rim
    # keeps its resolution whatever the scan spacing is
    centre = np.array(ndimage.center_of_mass(anchor), dtype=float)
    grids = []
    for axis, size in enumerate(args.box):
        offset = (np.arange(size, dtype=float) - size / 2 + 0.5) * args.spacing / spacing[axis]
        grids.append(centre[axis] + offset)
    coords = np.stack(np.meshgrid(*grids, indexing="ij"))

    low, high = args.hu
    patch = ndimage.map_coordinates(image, coords, order=1, mode="nearest")
    patch = (np.clip(patch, low, high) - low) / (high - low)
    la_box = ndimage.map_coordinates(la, coords, order=0, mode="constant", cval=0)
    fat_box = ndimage.map_coordinates(fat, coords, order=0, mode="constant", cval=0)
    labels = np.where(la_box > 0, LA_VALUE, np.where(fat_box > 0, FAT_VALUE, 0))
    stacked = np.stack([np.round(255 * np.clip(patch, 0, 1)), labels])
    return stacked.astype(np.uint8)


class Subset:
    """A view of the crops, so a training fold is never copied into memory.

    A 991 x 2 x 64 x 160 x 160 cache is 3 GB, and crops[train] would ask the
    machine for another 2.4 GB. This hands out the batches only, which is all
    the training loop ever reads.
    """

    def __init__(self, data, index):
        self.data = data
        self.index = np.asarray(index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return np.asarray(self.data[self.index[key[0]]])[(slice(None), *key[1:])]
        return np.asarray(self.data[self.index[key]])

    @property
    def shape(self):
        return (len(self.index), *self.data.shape[1:])


def flat_cache(cache: Path, tag: np.ndarray) -> tuple:
    """The cache memory mapped, read from disk in batches instead of held in RAM.

    A compressed npz has to be unpacked whole, which needs twice its size in
    memory, so it is unpacked once into a plain .npy beside it and only that is
    opened from then on.
    """
    flat = cache.with_name(cache.stem + "_flat.npy")
    note = cache.with_name(cache.stem + "_flat.json")
    if flat.exists() and note.exists():
        meta = json.loads(note.read_text())
        if np.allclose(np.asarray(meta["settings"], dtype=float), tag):
            print(f"crops: {len(meta['patient_id'])} reused, memory mapped from {flat.name}")
            return [str(name) for name in meta["patient_id"]], np.load(flat, mmap_mode="r")
    if not cache.exists():
        return [], None
    stored = np.load(cache, allow_pickle=True)
    if not np.allclose(np.asarray(stored["settings"], dtype=float), tag):
        print(f"crops: the cache was cut with other settings ({stored['settings']}), cutting again")
        return [], None
    names = [str(name) for name in stored["patient_id"]]
    print(f"crops: unpacking {len(names)} to {flat.name} once, this takes a few minutes")
    np.save(flat, stored["crops"])
    note.write_text(json.dumps({"patient_id": names, "settings": [float(value) for value in tag]}))
    del stored
    return names, np.load(flat, mmap_mode="r")


def build_crops(args: argparse.Namespace, patients: list, folder: Path) -> tuple:
    """Cropped boxes, written as they are cut so a crash loses at most 25 patients."""
    cache = args.crops or (folder / "rim_crops.npz")
    tag = np.array([*args.box, args.spacing, *args.hu], dtype=float)
    cached, mapped = ([], None) if args.recrop else flat_cache(cache, tag)
    position = {name: number for number, name in enumerate(cached)}
    done = {}

    images = find_images(args.images)
    la_masks = find_images(args.la_masks) if args.la_masks.exists() else {}
    fat_masks = find_images(args.fat_masks) if args.fat_masks.exists() else {}
    wanted = [name for name in patients if name in images]
    if args.limit:
        wanted = wanted[: args.limit]
    missing = [name for name in wanted if name not in position]
    if not missing and mapped is not None:
        # nothing to cut, so the cache is used where it lies, nothing is copied
        order = [position[name] for name in wanted]
        crops = mapped if order == list(range(len(mapped))) else Subset(mapped, order)
        print(f"crops: {tuple(crops.shape)} memory mapped, no patient cut")
        return crops, wanted
    if mapped is not None:
        for name in wanted:
            if name in position:
                done[name] = np.asarray(mapped[position[name]])
    print(f"  {len(missing)} of {len(wanted)} still to cut at {args.spacing} mm")

    def save() -> None:
        order = [name for name in wanted if name in done]
        cache.parent.mkdir(parents=True, exist_ok=True)
        for stale in (cache.with_name(cache.stem + "_flat.npy"), cache.with_name(cache.stem + "_flat.json")):
            stale.unlink(missing_ok=True)  # the memory mapped copy is out of date now
        np.savez_compressed(
            cache,
            crops=np.stack([done[name] for name in order]),
            patient_id=np.array(order),
            settings=tag,
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
    ids = [name for name in wanted if name in done]
    crops = np.stack([done[name] for name in ids])
    print(f"crops: {crops.shape} in {cache}")
    return crops, ids


# ------------------------------------------------- 2, the shell and the batches


def as_float(batch: np.ndarray):
    import torch

    return torch.from_numpy(batch.astype(np.float32) / 255.0)


def shell_weight(labels, args, voxels: int):
    """Grow a shell from the atrial surface and keep the fat that touches it.

    labels is channel 2 in [0, 1]; the atrium is 1.0 and the fat about 0.5. The
    band is the atrial wall, half of --rim-mm outwards and half inwards, so it
    stays thin instead of swallowing the whole atrium, and the fat is kept where
    it lies within --rim-mm of the surface, which is the LA-EAT definition.
    Everything else counts only --background, which is what forces the code to
    spend itself on the rim.
    """
    import torch.nn.functional as F

    atrium = (labels > 0.75).float()
    fat = ((labels > 0.25) & (labels <= 0.75)).float()
    half = max(voxels // 2, 1)
    grown = F.max_pool3d(atrium, 2 * half + 1, stride=1, padding=half)
    shrunk = -F.max_pool3d(-atrium, 2 * half + 1, stride=1, padding=half)
    band = (grown - shrunk).clamp(0.0, 1.0)
    near = F.max_pool3d(atrium, 2 * voxels + 1, stride=1, padding=voxels)
    rim_fat = (fat * near).clamp(0.0, 1.0)
    shell = (band + rim_fat).clamp(0.0, 1.0)
    return args.background + shell


def augment(batch, strength: float = 1.0):
    """Two views must differ, so shifts, flips, gain and noise, masks shifted alike."""
    import torch

    out = batch
    for axis in (2, 3, 4):
        move = int(torch.randint(-3, 4, (1,)).item())
        if move:
            out = torch.roll(out, move, dims=axis)
    if bool(torch.rand(1) < 0.5):
        out = torch.flip(out, dims=(4,))
    out = out.clone()
    gain = 1.0 + 0.15 * strength * (torch.rand(len(batch), 1, 1, 1, 1, device=batch.device) - 0.5)
    shift = 0.05 * strength * (torch.rand(len(batch), 1, 1, 1, 1, device=batch.device) - 0.5)
    noise = 0.03 * strength * torch.randn_like(out[:, 0:1])
    out[:, 0:1] = torch.clamp(out[:, 0:1] * gain + shift + noise, 0.0, 1.0)
    return out


def block_mask(batch, args):
    """Hide cubes from the decoder, the masked autoencoder part of the loss."""
    import torch
    import torch.nn.functional as F

    coarse = [max(size // 8, 1) for size in batch.shape[2:]]
    keep = (torch.rand(len(batch), 1, *coarse, device=batch.device) > args.mask_fraction).float()
    grid = F.interpolate(keep, size=batch.shape[2:], mode="nearest")
    hidden = batch.clone()
    hidden[:, 0:1] = hidden[:, 0:1] * grid
    return hidden, 1.0 - grid


# ------------------------------------------- 3 and 4, the network with attention


def build_network(args: argparse.Namespace):
    """Encoder with attention pooling per level, a decoder, and the three heads."""
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

    class AttentionPool(nn.Module):
        """One weight per voxel, learnt, so a level is read where it matters."""

        def __init__(self, width: int):
            super().__init__()
            self.score = nn.Conv3d(width, 1, 1)

        def forward(self, maps):
            flat = maps.flatten(2)
            weight = torch.softmax(self.score(maps).flatten(2), dim=2)
            return (flat * weight).sum(dim=2)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            channels, blocks, pools = 2, [], []
            for width in widths:
                blocks.append(block(channels, width, up=False))
                pools.append(AttentionPool(width))
                channels = width
            self.blocks = nn.ModuleList(blocks)
            self.pools = nn.ModuleList(pools)
            self.to_latent = nn.Sequential(nn.Linear(sum(widths), 128), nn.LeakyReLU(0.01), nn.Linear(128, args.latent))

        def forward(self, x):
            maps, pooled = [], []
            for piece, pool in zip(self.blocks, self.pools):
                x = piece(x)
                maps.append(x)
                pooled.append(pool(x))
            return self.to_latent(torch.cat(pooled, dim=1)), maps, pooled

    class Decoder(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape
            self.channels = widths[-1]
            self.from_latent = nn.Linear(args.latent, self.channels * int(np.prod(shape)))
            channels, blocks = self.channels, []
            for width in reversed(widths):
                blocks.append(block(channels, width, up=True))
                channels = width
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Conv3d(channels, 2, 1)

        def forward(self, z):
            x = self.from_latent(z).view(-1, self.channels, *self.shape)
            return torch.sigmoid(self.head(self.blocks(x)))

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder(tuple(size // 2 ** args.levels for size in args.box))
            # SimSiam: a projector on both views and a predictor on one of them
            self.projector = nn.Sequential(
                nn.Linear(args.latent, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.01), nn.Linear(128, 128)
            )
            self.predictor = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(0.01), nn.Linear(64, 128))
            self.distil = nn.Linear(args.latent, len(BEST_FOUR))  # step 4, no label involved
            self.classifier = nn.Sequential(
                nn.Linear(args.latent, 64), nn.LeakyReLU(0.01), nn.Dropout(0.3), nn.Linear(64, 1)
            )

        def forward(self, x):
            latent, maps, pooled = self.encoder(x)
            return latent, maps, pooled

    return Network()


# ------------------------------------------------------- 5, fold-safe pretraining


def pretrain(crops: np.ndarray, targets: np.ndarray | None, args, checkpoint: Path, quiet: bool = True):
    """Masked reconstruction on the shell + SimSiam + distillation. No group label."""
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = build_network(args).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    first = 1
    if checkpoint.exists() and not args.fresh:
        state = torch.load(checkpoint, map_location=device)
        if state.get("signature") == signature(args) and state.get("patients") == len(crops):
            model.load_state_dict(state["model"])
            optimiser.load_state_dict(state["optimiser"])
            first = state["epoch"] + 1
            if first <= args.pretrain_epochs:
                print(f"    pretraining resumed from epoch {state['epoch']}")
    if first > args.pretrain_epochs:
        return model, device

    rim_voxels = max(int(round(args.rim_mm / args.spacing)), 1)
    aim = None if targets is None else torch.from_numpy(targets.astype(np.float32)).to(device)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(args.pretrain_epochs, 1))
    for _ in range(1, first):
        schedule.step()

    for epoch in range(first, args.pretrain_epochs + 1):
        model.train()
        order = np.random.permutation(len(crops))
        total = 0.0
        for start in range(0, len(order), args.batch):
            index = order[start : start + args.batch]
            if len(index) < 2:  # BatchNorm in the projector needs two
                continue
            clean = as_float(crops[index]).to(device)
            view_one, view_two = augment(clean), augment(clean)
            weight = shell_weight(view_one[:, 1:2], args, rim_voxels)

            hidden, hole = block_mask(view_one, args)
            latent_one, _, _ = model.encoder(hidden)
            rebuilt = model.decoder(latent_one)
            # only the hidden part of the shell is scored, so the rim must be learnt
            focus = weight * (args.background + hole)
            recon = (focus * (rebuilt - view_one) ** 2).sum() / focus.sum().clamp(min=1.0)

            loss = recon
            if args.simsiam > 0:
                latent_two, _, _ = model.encoder(view_two)
                first_projection = model.projector(latent_one)
                second_projection = model.projector(latent_two)
                cosine = 0.5 * (
                    -F.cosine_similarity(model.predictor(first_projection), second_projection.detach(), dim=1).mean()
                    - F.cosine_similarity(model.predictor(second_projection), first_projection.detach(), dim=1).mean()
                )
                loss = loss + args.simsiam * (1.0 + cosine)
            if aim is not None and args.distil > 0:
                wanted = aim[index]
                known = torch.isfinite(wanted)
                if bool(known.any()):
                    guess = model.distil(latent_one)
                    loss = loss + args.distil * F.smooth_l1_loss(guess[known], wanted[known])

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(index)
        schedule.step()
        if not quiet and (epoch % 5 == 0 or epoch == 1):
            print(f"    pretrain epoch {epoch:3d}  loss {total / max(len(crops), 1):.4f}")
        if epoch % 5 == 0 or epoch == args.pretrain_epochs:
            torch.save(
                {"model": model.state_dict(), "optimiser": optimiser.state_dict(), "epoch": epoch,
                 "patients": len(crops), "signature": signature(args)},
                checkpoint,
            )
    return model, device


def finetune(model, device, crops, target, train, args, seed: int, aim=None):
    """Train the whole encoder on the label inside the training fold."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    torch.manual_seed(args.seed + 1000 * seed)
    parameters = list(model.encoder.parameters()) + list(model.classifier.parameters()) + list(model.distil.parameters())
    optimiser = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(args.finetune_epochs, 1))
    positives = max(int(target[train].sum()), 1)
    weight = torch.tensor([(len(train) - positives) / positives], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    wanted = None if aim is None else torch.from_numpy(aim.astype(np.float32)).to(device)

    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        order = np.random.permutation(train)
        total = 0.0
        for start in range(0, len(order), args.batch):
            index = order[start : start + args.batch]
            batch = augment(as_float(crops[index]).to(device))
            labels = torch.from_numpy(target[index].astype(np.float32))[:, None].to(device)
            latent, _, _ = model.encoder(batch)
            loss = criterion(model.classifier(latent), labels)
            if wanted is not None and args.distil > 0:
                aux = wanted[index]
                known = torch.isfinite(aux)
                if bool(known.any()):
                    loss = loss + 0.3 * args.distil * F.smooth_l1_loss(model.distil(latent)[known], aux[known])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(index)
        schedule.step()
        if epoch % 10 == 0:
            print(f"      seed {seed + 1} epoch {epoch:3d}  label loss {total / max(len(order), 1):.4f}")
    return model


def predict(model, device, crops, index, args):
    """Probabilities, averaged over --tta augmented copies."""
    import torch

    model.eval()
    rounds = max(args.tta, 1)
    out = np.zeros(len(index), dtype=float)
    with torch.no_grad():
        for turn in range(rounds):
            piece = []
            for start in range(0, len(index), args.batch):
                batch = as_float(crops[index[start : start + args.batch]]).to(device)
                if turn:
                    batch = augment(batch, strength=0.6)
                latent, _, _ = model.encoder(batch)
                piece.append(torch.sigmoid(model.classifier(latent))[:, 0].cpu().numpy())
            out += np.concatenate(piece) / rounds
    return out


def pooled_features(model, device, crops, index, args) -> dict:
    """Attention pooled vector of every level plus the latent, for the probes."""
    import torch

    model.eval()
    collected = {}
    with torch.no_grad():
        for start in range(0, len(index), args.batch):
            batch = as_float(crops[index[start : start + args.batch]]).to(device)
            latent, _, pooled = model.encoder(batch)
            for number, vector in enumerate(pooled, 1):
                collected.setdefault(f"level{number}", []).append(vector.cpu().numpy())
            collected.setdefault("latent", []).append(latent.cpu().numpy())
    blocks = {name: np.concatenate(parts) for name, parts in collected.items()}
    blocks["all levels"] = np.concatenate([blocks[name] for name in sorted(blocks)], axis=1)
    return blocks


# ---------------------------------------------------------------- the evaluation


def probe(train_features, train_target, test_features) -> np.ndarray:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, C=0.1)),
        ]
    )
    pipeline.fit(train_features, train_target)
    return pipeline.predict_proba(test_features)[:, 1]


def splits_of(crops, target, args):
    for repeat in range(args.repeats):
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + repeat)
        for number, (train, test) in enumerate(splitter.split(crops, target), 1):
            yield repeat + 1, number, train, test


def ranks(values: np.ndarray) -> np.ndarray:
    """Rank scaled to [0, 1], so two models can be averaged; unknown becomes 0.5."""
    filled = np.where(np.isfinite(values), values, np.nanmedian(values))
    order = np.argsort(np.argsort(filled))
    out = order / max(len(filled) - 1, 1)
    return np.where(np.isfinite(values), out, 0.5)


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


def distillation_targets(merged, args) -> np.ndarray | None:
    """The four hand made features as regression targets, NaN where unknown."""
    if merged is None:
        return None
    have = [name for name in BEST_FOUR if name in merged.columns]
    if not have:
        print("   distillation skipped, the four features are not in the workbook")
        return None
    values = merged[have].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    out = np.full((len(values), len(BEST_FOUR)), np.nan)
    out[:, : len(have)] = values
    return out


def standardise(values: np.ndarray | None, train) -> np.ndarray | None:
    """Scaled on the training fold only, so a test patient never sets the scale."""
    if values is None:
        return None
    middle = np.nanmedian(values[train], axis=0)
    spread = np.nanstd(values[train], axis=0)
    spread = np.where(spread > 1e-9, spread, 1.0)
    return (values - middle) / spread


def plot_roc(curves: list, folder: Path, group: str) -> None:
    plt.figure(figsize=(6.5, 6))
    for name, target, scores, auc in curves:
        fpr, tpr, _ = roc_curve(target, scores)
        plt.plot(fpr, tpr, lw=1.4, label=f"{name}  AUC {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="0.6", lw=0.8)
    plt.xlabel("1 - specificity")
    plt.ylabel("sensitivity")
    plt.title(f"Rim representation against the feature model, {group}")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(folder / "01_roc.png", dpi=200)
    plt.close()


def plot_latent(latent, target, args, folder: Path) -> None:
    scaled = StandardScaler().fit_transform(latent)
    try:
        import umap

        points = umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=args.seed).fit_transform(scaled)
        name = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA

        points = PCA(n_components=2, random_state=args.seed).fit_transform(scaled)
        name = "principal component"
    plt.figure(figsize=(6.5, 6))
    for value, colour, label in ((0, "#5dade2", "control"), (1, "#e74c3c", args.tag)):
        plt.scatter(points[target == value, 0], points[target == value, 1], s=12, alpha=0.6, c=colour, label=label)
    plt.legend()
    plt.xlabel(f"{name} 1")
    plt.ylabel(f"{name} 2")
    plt.title(f"Held out rim representation, {args.tag} against control")
    plt.tight_layout()
    plt.savefig(folder / "02_latent_space.png", dpi=200)
    plt.close()


def plot_attention(model, device, crops, args, folder: Path, group: str) -> None:
    """Where the first level attends, averaged over a few patients."""
    import torch

    model.eval()
    index = np.arange(min(len(crops), 16))
    boxes = crops[index]
    with torch.no_grad():
        batch = as_float(boxes).to(device)
        _, maps, _ = model.encoder(batch)
        scores = model.encoder.pools[0].score(maps[0])
        weight = torch.softmax(scores.flatten(2), dim=2).view_as(scores)
    picture = weight[:, 0].mean(dim=0).cpu().numpy()
    image = np.asarray(boxes)[:, 0].astype(np.float32).mean(axis=0) / 255.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    cuts = (
        ("axial", picture.mean(axis=0), image[image.shape[0] // 2]),
        ("coronal", picture.mean(axis=1), image[:, image.shape[1] // 2]),
        ("sagittal", picture.mean(axis=2), image[:, :, image.shape[2] // 2]),
    )
    for axis, (name, heat, grey) in zip(axes, cuts):
        axis.imshow(grey, cmap="gray", vmin=0, vmax=1)
        zoom = [grey.shape[number] / heat.shape[number] for number in range(2)]
        axis.imshow(ndimage.zoom(heat, zoom, order=1), cmap="inferno", alpha=0.45)
        axis.set_title(f"{name}, level 1 attention")
        axis.axis("off")
    fig.suptitle(f"Where the rim encoder looks, mean of {len(index)} patients, {group}")
    plt.tight_layout()
    plt.savefig(folder / "03_attention.png", dpi=200)
    plt.close()


def plot_shell(crops, args, folder: Path) -> None:
    """The shell the loss is restricted to, so it can be checked by eye."""
    import torch

    rim_voxels = max(int(round(args.rim_mm / args.spacing)), 1)
    batch = as_float(crops[: min(len(crops), 4)])
    with torch.no_grad():
        weight = shell_weight(batch[:, 1:2], args, rim_voxels)[0, 0].numpy()
    image = batch[0, 0].numpy()
    middle = image.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    axes[0].imshow(image[middle], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"image at {args.spacing} mm")
    axes[1].imshow(image[middle], cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(weight[middle], cmap="viridis", alpha=0.5)
    axes[1].set_title(f"loss weight, shell {args.rim_mm} mm")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(folder / "04_shell.png", dpi=200)
    plt.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    folder = args.output / "rim"
    plots = folder / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    journal = Journal(folder / "progress.json", args.fresh)

    groups = pd.read_excel(args.groups, sheet_name="Features")
    groups.columns = [str(name).strip() for name in groups.columns]
    cases = args.cases
    groups = groups[groups["group"].isin([*cases, "control"])]
    wanted = groups["patient_id"].astype(str).tolist()
    print(f"1. {'+'.join(cases)} {int(groups['group'].isin(cases).sum())} "
          f"against control {int((groups['group'] == 'control').sum())}")

    crops, ids = build_crops(args, wanted, folder)
    label_of = dict(zip(groups["patient_id"].astype(str), groups["group"]))
    target = np.array([1 if label_of[name] in cases else 0 for name in ids])
    print(f"2. {len(ids)} patients, {int(target.sum())} {args.tag}, box {args.box} at {args.spacing} mm")
    if not args.limit and len(ids) < 0.9 * len(wanted):
        print(f"   WARNING: {len(wanted)} patients are in the workbook but only {len(ids)} are cropped,"
              " let the cropping finish before reading any AUC below")
    if target.sum() < 10 or (len(target) - target.sum()) < 10:
        raise SystemExit("too few patients of one class to cross-validate")

    merged = measurement_table(ids, args)
    aim = distillation_targets(merged, args)

    print(f"3. fold-safe training, {args.repeats} x {args.folds} folds, {args.seeds} seeds, TTA {args.tta}")
    # a patient is a test patient once per repeat, so the repeats are averaged
    image_sum = np.zeros(len(target), dtype=float)
    image_seen = np.zeros(len(target), dtype=float)
    probe_sum: dict = {}
    latent_out = np.zeros((len(target), args.latent), dtype=float)
    fold_aucs, model, device = [], None, None

    def keep(fold_scores: dict, scores: np.ndarray, test) -> None:
        image_sum[test] += scores
        image_seen[test] += 1.0
        for name, values in fold_scores.items():
            holder = probe_sum.setdefault(name, [np.zeros(len(target)), np.zeros(len(target))])
            holder[0][test] += np.asarray(values, dtype=float)
            holder[1][test] += 1.0

    for repeat, number, train, test in splits_of(crops, target, args):
        key = (f"{signature(args)} {args.tag} k{args.folds} r{repeat}/{args.repeats}"
               f" s{args.seeds} tta{args.tta} n{len(target)} fold{number}")
        stored = journal.get(key)
        if stored:
            latent_out[test] = np.array(stored["latent"])
            keep(stored["probes"], np.array(stored["scores"], dtype=float), test)
            fold_aucs.append({"repeat": repeat, "fold": number, "auc": stored["auc"]})
            print(f"  repeat {repeat} fold {number} reused, AUC {stored['auc']:.4f}")
            continue

        print(f"  repeat {repeat} fold {number}, {len(train)} train, {len(test)} test")
        scaled = standardise(aim, train)
        base, device = pretrain(
            Subset(crops, train),
            None if scaled is None else scaled[train],
            args,
            folder / f"pretrain_r{repeat}_f{number}.pt",
            quiet=False,
        )

        # the frozen probes read the pretrained encoder, no label has touched it
        features = pooled_features(base, device, crops, np.arange(len(target)), args)
        fold_probes = {name: probe(values[train], target[train], values[test]) for name, values in features.items()}
        latent_out[test] = features["latent"][test]

        import copy

        import torch

        seed_scores = []
        for seed in range(args.seeds):
            tuned = copy.deepcopy(base)
            tuned = finetune(tuned, device, crops, target, train, args, seed, None if scaled is None else scaled)
            seed_scores.append(ranks(predict(tuned, device, crops, test, args)))
            if model is None or (repeat == 1 and number == 1 and seed == 0):
                model = tuned
            del tuned
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        fold_scores = np.mean(seed_scores, axis=0)
        auc = float(roc_auc_score(target[test], fold_scores))
        keep(fold_probes, fold_scores, test)
        fold_aucs.append({"repeat": repeat, "fold": number, "auc": auc})
        print(f"    fold AUC {auc:.4f}")
        journal.put(
            key,
            {
                "auc": auc,
                "scores": fold_scores.tolist(),
                "latent": latent_out[test].tolist(),
                "probes": {name: values.tolist() for name, values in fold_probes.items()},
                "test": test.tolist(),
            },
        )

    image_scores = np.where(image_seen > 0, image_sum / np.maximum(image_seen, 1.0), np.nan)
    probe_scores = {
        name: np.where(seen > 0, total / np.maximum(seen, 1.0), np.nan)
        for name, (total, seen) in probe_sum.items()
    }

    rows, curves = [], []
    print("\n4. frozen attention pooled probes, the pretrained code only")
    for name in sorted(probe_scores):
        scores = probe_scores[name]
        known = np.isfinite(scores)
        auc = float(roc_auc_score(target[known], scores[known]))
        rows.append({"block": name, "auc": round(auc, 4), "sd": ""})
        curves.append((name, target[known], scores[known], auc))
        print(f"   {name:12s} AUC {auc:.4f}")

    per_fold = np.array([row["auc"] for row in fold_aucs], dtype=float)
    pooled_auc = float(roc_auc_score(target[np.isfinite(image_scores)], image_scores[np.isfinite(image_scores)]))
    rows.append({"block": "rim encoder, fine-tuned", "auc": round(float(per_fold.mean()), 4),
                 "sd": round(float(per_fold.std()), 4)})
    curves.append(("rim encoder, fine-tuned", target, image_scores, pooled_auc))
    print(f"\n5. the rim encoder, fold-safe, {args.seeds} seeds and TTA averaged")
    print(f"   AUC {per_fold.mean():.4f} (SD {per_fold.std():.4f}) over {len(per_fold)} folds, pooled {pooled_auc:.4f}")

    reference_rows = []
    if merged is not None:
        print("\n6. the interpretable reference and the hybrid, same patients and folds")
        panels = {"LA volume alone": ["la_volume_ml"], "best four features": BEST_FOUR}
        for name, columns in panels.items():
            have = [c for c in columns if c in merged.columns]
            if len(have) != len(columns):
                print(f"   '{name}' skipped, missing {set(columns) - set(have)}")
                continue
            values = merged[have].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            scores = np.full(len(target), np.nan)
            folds = []
            for repeat, number, train, test in splits_of(crops, target, args):
                scores[test] = probe(values[train], target[train], values[test])
                folds.append(roc_auc_score(target[test], scores[test]))
            auc, spread = float(np.mean(folds)), float(np.std(folds))
            reference_rows.append({"block": name, "auc": round(auc, 4), "sd": round(spread, 4)})
            curves.append((name, target, scores, auc))
            print(f"   {name:30s} AUC {auc:.4f} (SD {spread:.4f})")
            if name == "best four features":
                blended = 0.5 * (ranks(scores) + ranks(image_scores))
                hybrid = float(roc_auc_score(target, blended))
                reference_rows.append({"block": "hybrid, four features + rim encoder", "auc": round(hybrid, 4), "sd": ""})
                curves.append(("hybrid", target, blended, hybrid))
                print(f"   {'hybrid, four features + rim encoder':30s} AUC {hybrid:.4f}")

    table = pd.DataFrame({"patient_id": ids, "group": [label_of[name] for name in ids],
                          "case": target, "rim_score": image_scores})
    for name in sorted(probe_scores):
        table[f"probe_{name.replace(' ', '_')}"] = probe_scores[name]
    table.to_csv(folder / f"scores_{args.tag}.csv", index=False)

    if not args.no_plots:
        plot_roc(curves, plots, args.tag)
        plot_latent(latent_out, target, args, plots)
        plot_shell(crops, args, plots)
        if model is not None and device is not None:
            plot_attention(model, device, crops, args, plots, args.tag)
        print(f"\nPlots : {plots}")

    workbook = folder / (f"rim_{args.tag}.xlsx" if args.tag != "AF" else "rim.xlsx")
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows + reference_rows).to_excel(writer, sheet_name="AUC", index=False)
        pd.DataFrame(fold_aucs).to_excel(writer, sheet_name="Folds", index=False)
        table.head(2000).to_excel(writer, sheet_name="Scores", index=False)
    print(f"Excel : {workbook}")
    journal.put("finished", True)


if __name__ == "__main__":
    main()
