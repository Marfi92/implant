# DINOv2 + LoRA for cardiac CT: left atrium or epicardial fat, nnU-Net folder layout.
# Same design as the MRI script (LoRA in every DINOv2 block, 3D decoder over a stack
# of slices, CNN skips, edge weighted loss), changed for CT:
#   1. HU windowing instead of percentile scaling, so the intensities mean the same
#      thing in every scan: fat near -100 HU, blood near 100 HU after contrast.
#   2. Trains on imagesTr/labelsTr and reports Dice on imagesTs/labelsTs, the same
#      test cases as nnU-Net, so the two Dice tables are directly comparable.
#   3. No largest component cleanup on fat, which is diffuse and never one blob.
# CLASSES must match the label values in labelsTr: ['LA'] or ['EAT'] for a one class
# dataset, ['LA', 'EAT'] if one label file holds LA = 1 and fat = 2.
# Jupyter: %run 16_dinov2_lora_ct_la_fat.py      Only predict: MODE = 'predict'
import os
import random
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn

# ---------------- settings ----------------
DATASET = r"Q:\users\marfi\Automatic segmentations\finalllll\final\LA82"
TRAIN_IMG_DIR = DATASET + r"\imagesTr"
TRAIN_LAB_DIR = DATASET + r"\labelsTr"
TEST_IMG_DIR = DATASET + r"\imagesTs"
TEST_LAB_DIR = DATASET + r"\labelsTs"
OUT = DATASET + r"\dinov2_lora"
BACKBONE = "dinov2_vits14"
SIZE = 448
CLASSES = ["LA"]  # label value 1 is LA; use ['EAT'] for the fat dataset
NCLS = len(CLASSES) + 1
HU_WINDOW = (-200.0, 400.0)  # fat, muscle and contrast filled atrium all inside
DEPTH = 8  # slices per 3D stack (GPU out of memory: 6)
LORA_R, LORA_A = 8, 16
EPOCHS = 30
LR, LR_LORA = 1e-3, 3e-4
CLS_W = [1.0] + [2.0 if name == "EAT" else 1.0 for name in CLASSES]  # loss weight per class
EDGE_W = 2.0
PAD_MM = 15
ZPAD = 3
MODE = "all"  # 'train', 'predict' or 'all'
QUICK = False  # True = 5 cases, 2 epochs, to test the script
PRED_TRAIN = False  # also predict the training cases
REFINE = 1
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def use_dataset(folder: str, classes: list) -> None:
    """Point the script at an nnU-Net dataset folder, e.g. use_dataset(BASE + r'\\FAT82', ['EAT'])."""
    global TRAIN_IMG_DIR, TRAIN_LAB_DIR, TEST_IMG_DIR, TEST_LAB_DIR, OUT, CLASSES
    TRAIN_IMG_DIR = os.path.join(folder, "imagesTr")
    TRAIN_LAB_DIR = os.path.join(folder, "labelsTr")
    TEST_IMG_DIR = os.path.join(folder, "imagesTs")
    TEST_LAB_DIR = os.path.join(folder, "labelsTs")
    OUT = os.path.join(folder, "dinov2_lora")
    CLASSES = classes


def set_up() -> None:
    """Make the output folder, fix the seeds and follow CLASSES if it was changed."""
    global NCLS, CLS_W
    NCLS = len(CLASSES) + 1
    CLS_W = [1.0] + [2.0 if name == "EAT" else 1.0 for name in CLASSES]
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


def case_id(name: str) -> str:
    """VALLA_1234_0000.nii.gz, VALLA_1234.nii.nii.gz and VALLA_1234.nii.gz all mean VALLA_1234."""
    stem, changed = name, True
    while changed:
        changed = False
        for suffix in (".gz", ".nii", ".nrrd", ".mha"):
            if stem.lower().endswith(suffix):
                stem, changed = stem[: -len(suffix)], True
                break
        if not changed and stem.endswith("_0000"):
            stem, changed = stem[:-5], True
    return stem


def index_folder(folder: str) -> dict:
    """case id -> file, for a folder of volumes."""
    if not folder or not os.path.isdir(folder):
        return {}
    found = {}
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".nii", ".nii.gz", ".nrrd", ".mha")):
            found.setdefault(case_id(name), os.path.join(folder, name))
    return found


def case_lists() -> tuple[list, list, dict, dict]:
    """Training cases from imagesTr/labelsTr and test cases from imagesTs/labelsTs."""
    images = index_folder(TRAIN_IMG_DIR) | index_folder(TEST_IMG_DIR)
    labels = index_folder(TRAIN_LAB_DIR) | index_folder(TEST_LAB_DIR)
    train_cases = sorted(set(index_folder(TRAIN_IMG_DIR)) & set(index_folder(TRAIN_LAB_DIR)))
    test_cases = sorted(set(index_folder(TEST_IMG_DIR)) & set(index_folder(TEST_LAB_DIR)))
    if not train_cases and MODE != "predict":
        raise SystemExit("no training case has both an image and a label, check the folders")
    if not test_cases:
        print("no test case found, Dice will only be reported on the training cases")
    return [train_cases, test_cases], images, labels


def load_image(path: str):
    """CT scaled from the HU window to 0..1."""
    nii = nib.load(path)
    volume = np.asarray(nii.dataobj, dtype=np.float32)  # nibabel applies the HU scaling
    low, high = HU_WINDOW
    volume = np.clip((volume - low) / (high - low), 0, 1).astype(np.float32)
    return volume, nii


def load_label(cid: str, shape, labels: dict) -> np.ndarray:
    """Label volume, class values 1..len(CLASSES), anything above that treated as background."""
    path = labels.get(cid)
    if path is None:
        return np.zeros(shape, np.int64)
    label = np.asarray(nib.load(path).dataobj).astype(np.int64)
    label[label > len(CLASSES)] = 0
    return label


def largest_cc(mask: np.ndarray) -> np.ndarray:
    labelled, count = ndimage.label(mask)
    if count <= 1:
        return mask
    return labelled == (np.argmax(np.bincount(labelled.ravel())[1:]) + 1)


def roi_box(mask: np.ndarray, spacing):
    if not mask.any():
        return None
    xs, ys, zs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max((xs.max() - xs.min()) * spacing[0], (ys.max() - ys.min()) * spacing[1]) / 2 + PAD_MM
    return (
        cx,
        cy,
        half / np.mean(spacing[:2]),
        max(0, zs.min() - ZPAD),
        min(mask.shape[2] - 1, zs.max() + ZPAD),
    )


def crop_coords(cx, cy, half):
    return round(cx - half), round(cy - half), max(8, round(2 * half))


def crop2d(array, x0, y0, size):
    out = np.zeros(array.shape[:-2] + (size, size), array.dtype)
    width, height = array.shape[-2:]
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(width, x0 + size), min(height, y0 + size)
    if ax1 > ax0 and ay1 > ay0:
        out[..., ax0 - x0 : ax1 - x0, ay0 - y0 : ay1 - y0] = array[..., ax0:ax1, ay0:ay1]
    return out


def stack3(volume, z):
    zz = np.clip([z - 1, z, z + 1], 0, volume.shape[2] - 1)
    return volume[:, :, zz].transpose(2, 0, 1)


def make_stack(volume, zs, x0, y0, size):
    """-> (D, 3, SIZE, SIZE)"""
    x = torch.from_numpy(np.stack([crop2d(stack3(volume, z), x0, y0, size) for z in zs]))
    return F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------- model ----------------
class LoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)
        self.s = alpha / rank

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.s


def blk2(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def blk3(i, o):
    return nn.Sequential(
        nn.Conv3d(i, o, 3, padding=1), nn.InstanceNorm3d(o, affine=True), nn.ReLU(inplace=True)
    )


class DinoSeg3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.hub.load("facebookresearch/dinov2", BACKBONE)
        for p in self.enc.parameters():
            p.requires_grad = False
        for block in self.enc.blocks:
            block.attn.qkv = LoRA(block.attn.qkv, LORA_R, LORA_A)
            block.attn.proj = LoRA(block.attn.proj, LORA_R, LORA_A)
        for p in self.enc.norm.parameters():
            p.requires_grad = True
        channels = self.enc.embed_dim * 4
        self.neck = nn.Sequential(
            nn.Conv2d(channels, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.c1 = nn.Sequential(blk2(3, 16), blk2(16, 16))
        self.c2 = blk2(16, 32)
        self.c4 = blk2(32, 64)
        self.d8 = nn.Sequential(blk3(256, 128), blk3(128, 128))
        self.d4 = nn.Sequential(blk3(128 + 64, 64), blk3(64, 64))
        self.u2 = blk2(64 + 32, 32)
        self.u1 = blk2(32 + 16, 32)
        self.head = nn.Conv2d(32, NCLS, 1)

    def forward(self, x):
        """x (B, D, 3, H, W) -> (B, D, NCLS, H, W)"""
        batch, depth, _, height, width = x.shape
        x = x.flatten(0, 1)

        def to3(t):
            return t.view(batch, depth, *t.shape[1:]).transpose(1, 2)

        def to2(t):
            return t.transpose(1, 2).flatten(0, 1)

        f = self.enc.get_intermediate_layers(
            (x - MEAN.to(x.device)) / STD.to(x.device), n=4, reshape=True
        )
        f = self.d8(to3(self.neck(torch.cat(f, 1))))
        s1 = self.c1(x)
        s2 = self.c2(F.max_pool2d(s1, 2))
        s4 = self.c4(F.max_pool2d(s2, 2))
        f = F.interpolate(f, size=(depth,) + s4.shape[-2:], mode="trilinear", align_corners=False)
        f = to2(self.d4(torch.cat([f, to3(s4)], 1)))
        f = self.u2(
            torch.cat(
                [F.interpolate(f, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1
            )
        )
        f = self.u1(
            torch.cat(
                [
                    F.interpolate(f, size=(height, width), mode="bilinear", align_corners=False),
                    s1,
                ],
                1,
            )
        )
        return self.head(f).view(batch, depth, NCLS, height, width)


# ---------------- data ----------------
class Stacks(torch.utils.data.Dataset):
    """Stacks of DEPTH slices, cropped around the heart with random shift and zoom."""

    def __init__(self, cases, images, labels):
        self.vols, self.items = [], []
        for cid in cases:
            volume, nii = load_image(images[cid])
            label = load_label(cid, volume.shape, labels)
            box = roi_box(label >= 1, nii.header.get_zooms()[:3])
            if box is None:
                continue
            index = len(self.vols)
            self.vols.append((volume, label, box))
            for z in range(box[3], box[4] + 1, DEPTH):
                self.items.append((index, z))
        print(f"{len(self.vols)} volumes, {len(self.items)} stacks of {DEPTH} slices")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        index, z = self.items[i]
        volume, label, (cx, cy, half, z0, z1) = self.vols[index]
        z = int(np.clip(z + random.randint(-DEPTH // 2, DEPTH // 2), z0, max(z0, z1 - DEPTH + 1)))
        zs = np.clip(np.arange(z, z + DEPTH), 0, volume.shape[2] - 1)
        zoomed = half * random.uniform(0.85, 1.25)
        x0, y0, size = crop_coords(
            cx + random.uniform(-0.15, 0.15) * half,
            cy + random.uniform(-0.15, 0.15) * half,
            zoomed,
        )
        x = make_stack(volume, zs, x0, y0, size)
        y = torch.from_numpy(np.stack([crop2d(label[:, :, zz], x0, y0, size) for zz in zs]))
        y = F.interpolate(y[:, None].float(), size=(SIZE, SIZE), mode="nearest")[:, 0].long()
        y[y >= NCLS] = 255
        if random.random() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        if random.random() < 0.5:
            x, y = x.flip(-2), y.flip(-2)
        x = (x * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)).clamp(0, 1)
        return x, y


def dice_loss(logit, y):
    probability = logit.softmax(1)
    known = y != 255
    one_hot = F.one_hot(y.clamp(0, NCLS - 1), NCLS).permute(0, 3, 1, 2).float() * known[:, None]
    probability = probability * known[:, None]
    inter = (probability * one_hot).sum((0, 2, 3))
    den = probability.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1))[1:].mean()


def edge_ce(logit, y, weight):
    one_hot = F.one_hot(y.clamp(0, NCLS - 1), NCLS).permute(0, 3, 1, 2).float()
    edge = (F.max_pool2d(one_hot, 5, 1, 2) + F.max_pool2d(-one_hot, 5, 1, 2)).sum(1) > 0
    pixel_weight = (1 + EDGE_W * edge.float()) * (y != 255)
    ce = F.cross_entropy(logit, y, weight=weight, ignore_index=255, reduction="none")
    return (ce * pixel_weight).sum() / pixel_weight.sum().clamp(min=1)


# ---------------- train ----------------
def train(cases, images, labels):
    dataset = Stacks(cases[:5] if QUICK else cases, images, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True
    )
    model = DinoSeg3D().to(DEV)
    encoder_parameters = [p for p in model.enc.parameters() if p.requires_grad]
    decoder_parameters = [
        p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("enc.")
    ]
    groups = [
        {"params": decoder_parameters, "lr": LR},
        {"params": encoder_parameters, "lr": LR_LORA},
    ]
    optimiser = torch.optim.AdamW(groups, weight_decay=1e-4)
    epochs = 2 if QUICK else EPOCHS
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=[g["lr"] for g in groups], total_steps=epochs * len(loader)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    weight = torch.tensor(CLS_W, device=DEV)
    print(
        f"trainable: LoRA + norm {sum(p.numel() for p in encoder_parameters) / 1e6:.2f} M, "
        f"decoder {sum(p.numel() for p in decoder_parameters) / 1e6:.2f} M"
    )
    for epoch in range(epochs):
        model.train()
        started, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV).flatten(0, 1)
            with torch.autocast(device_type=DEV, enabled=DEV == "cuda"):
                logit = model(x)
            logit = logit.float().flatten(0, 1)
            loss = edge_ce(logit, y, weight) + dice_loss(logit, y)
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total += loss.item()
        print(
            f"epoch {epoch + 1}/{epochs}  loss {total / len(loader):.4f} "
            f" {time.time() - started:.0f} s"
        )
        torch.save(model.state_dict(), os.path.join(OUT, "dinov2_ct.pt"))
    return model


# ---------------- predict ----------------
@torch.no_grad()
def predict_case(model, volume, box):
    width, height, slices = volume.shape
    if box is None:
        box = ((width - 1) / 2, (height - 1) / 2, max(width, height) / 2, 0, slices - 1)
    cx, cy, half, z0, z1 = box
    x0, y0, size = crop_coords(cx, cy, half)
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(width, x0 + size), min(height, y0 + size)
    summed = np.zeros((NCLS, ax1 - ax0, ay1 - ay0, slices), np.float32)
    counted = np.zeros(slices, np.float32)
    starts = list(range(z0, max(z0, z1 - DEPTH + 1) + 1, max(1, DEPTH // 2)))
    if starts[-1] + DEPTH - 1 < z1:
        starts.append(z1 - DEPTH + 1)
    for start in starts:
        zs = np.clip(np.arange(start, start + DEPTH), 0, slices - 1)
        x = make_stack(volume, zs, x0, y0, size)[None].to(DEV)
        with torch.autocast(device_type=DEV, enabled=DEV == "cuda"):
            logit = model(x) + model(x.flip(-1)).flip(-1)
        probability = (
            F.interpolate(logit[0].float(), size=(size, size), mode="bilinear", align_corners=False)
            .softmax(1)
            .cpu()
            .numpy()
        )
        for d, z in enumerate(zs):
            summed[:, :, :, z] += probability[d][
                :, ax0 - x0 : ax1 - x0, ay0 - y0 : ay1 - y0
            ]
            counted[z] += 1
    full = np.zeros((NCLS, width, height, slices), np.float32)
    full[0] = 1
    for z in np.nonzero(counted)[0]:
        full[:, ax0:ax1, ay0:ay1, z] = summed[:, :, :, z] / counted[z]
    return full


def dice(a, b):
    total = a.sum() + b.sum()
    return np.nan if total == 0 else 2 * np.logical_and(a, b).sum() / total


def predict(model, splits, images, labels):
    model.eval()
    rows = []
    groups = [("test", splits[1])] + ([("train", splits[0])] if PRED_TRAIN else [])
    for name, cases in groups:
        folder = os.path.join(OUT, name)
        os.makedirs(folder, exist_ok=True)
        for cid in cases[:2] if QUICK else cases:
            volume, nii = load_image(images[cid])
            spacing = nii.header.get_zooms()[:3]
            label = load_label(cid, volume.shape, labels)
            box = roi_box(label >= 1, spacing)
            if box is None:
                first = predict_case(model, volume, None).argmax(0)
                box = roi_box(first >= 1, spacing)
                print(cid, "no crop box found, box from a first pass on the whole slice")
            predicted = predict_case(model, volume, box).argmax(0).astype(np.uint8)
            for _ in range(REFINE):
                refined = roi_box(predicted >= 1, spacing)
                if refined is not None:
                    predicted = predict_case(model, volume, refined).argmax(0).astype(np.uint8)
            if "LA" in CLASSES:  # the atrium is one blob, the fat is not
                atrium = predicted == CLASSES.index("LA") + 1
                predicted[atrium & ~largest_cc(atrium)] = 0
            nib.save(
                nib.Nifti1Image(predicted, nii.affine, nii.header),
                os.path.join(folder, cid + ".nii.gz"),
            )
            row = {"group": name, "case": cid}
            for value, class_name in enumerate(CLASSES, 1):
                row[class_name] = dice(predicted == value, label == value)
            rows.append(row)
            print(name, cid, " ".join(f"{k} {row[k]:.3f}" for k in CLASSES))
    if rows:
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(OUT, "dice_dinov2_ct.csv"), index=False)
        print("\nMEAN Dice per group\n", table.groupby("group")[CLASSES].mean().round(3))
    print("saved in", OUT)


def main():
    set_up()
    print("device", DEV, "classes", CLASSES)
    splits, images, labels = case_lists()
    print(f"{len(splits[0])} training cases, {len(splits[1])} test cases")
    if MODE in ("train", "all"):
        model = train(splits[0], images, labels)
    else:
        model = DinoSeg3D().to(DEV)
        model.load_state_dict(torch.load(os.path.join(OUT, "dinov2_ct.pt"), map_location=DEV))
    if MODE in ("predict", "all"):
        predict(model, splits, images, labels)


if __name__ == "__main__":
    main()
