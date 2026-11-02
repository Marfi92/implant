"""Three supervised image models for AF, trained on the crops of script 27.

Script 27 learns a representation by reconstruction, which spends its capacity
on whatever is large and bright rather than on what separates AF. This one
drops reconstruction and trains for the label directly, with three different
networks, and averages them:

    resnet3d   3D residual network, group norm, squeeze and excitation,
               attention pooling, heavy augmentation and mixup
    slice2d    ImageNet pretrained 2D ResNet18 on the axial slices through the
               atrium, with attention over slices, so millions of pretrained
               filters stand in for the data this cohort does not have
    multitask  the 3D network with two extra heads that regress the atrial and
               the fat volume of the same crop, which steadies a small dataset

Every model is trained inside the training fold only, so every AUC printed here
is fold-safe and quotable. Each one also gives a representation: the pooled
embedding before its head, written to the workbook and plotted.

    ensemble   the three out-of-fold probabilities averaged, after ranking
    hybrid     that ensemble rank-averaged with the four feature model, which
               is the only place a hand made feature appears

Crash recovery: every finished fold of every model is written to progress.json
with its out-of-fold scores, so re-running the same command carries on at the
fold it died in. --fresh throws that away.

Input : --crops   crops.npz written by script 27 (48,128,128 recommended)
        --groups  workbook with patient_id and group
Output: <output>\\supervised\\progress.json      finished folds
        <output>\\supervised\\scores_<group>.csv one row per patient
        <output>\\supervised\\supervised.xlsx    AUC of every model
        <output>\\supervised\\plots\\*.png       ROC, embeddings, saliency

Usage
    python 28_supervised_image_models.py --group AF
    python 28_supervised_image_models.py --group AF --models resnet3d,slice2d
    python 28_supervised_image_models.py --group HF --epochs 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

ANALYSIS = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast\analysis\results")
BEST_FOUR = [
    "la_firstorder_interquartile_range",
    "la_shape_surface_area",
    "laeat_shape_flatness",
    "eat_firstorder_median",
]
MODELS = ("resnet3d", "slice2d", "multitask")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, default=ANALYSIS / "representation" / "crops.npz")
    parser.add_argument("--groups", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--measurements", type=Path, default=ANALYSIS / "fat_la_features.xlsx")
    parser.add_argument("--radiomics", type=Path, default=ANALYSIS / "radiomic_features_full.xlsx")
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument("--group", default="AF", help="AF, HF or stroke")
    parser.add_argument("--models", default=",".join(MODELS), help="comma separated subset of " + ",".join(MODELS))
    parser.add_argument("--width", type=int, default=32, help="channels of the first 3D stage")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1, help="restarts per fold, averaged, 2 or 3 is steadier")
    parser.add_argument("--mixup", type=float, default=0.3, help="0 switches mixup off")
    parser.add_argument("--slices", type=int, default=24, help="axial slices kept by slice2d")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N patients, for a trial run")
    parser.add_argument("--fresh", action="store_true", help="ignore progress.json")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_known_args(argv)[0]
    args.models = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = set(args.models) - set(MODELS)
    if unknown:
        raise SystemExit(f"--models {sorted(unknown)} unknown, pick from {MODELS}")
    return args


class Journal:
    """Finished folds, so a crash costs one fold and not the whole run."""

    def __init__(self, path: Path, fresh: bool):
        self.path = path
        self.data = {}
        if path.exists() and not fresh:
            try:
                self.data = json.loads(path.read_text())
                print(f"resuming, progress.json holds {len(self.data)} finished pieces")
            except json.JSONDecodeError:
                pass

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value) -> None:
        self.data[key] = value
        self.path.write_text(json.dumps(self.data, indent=1))


# ------------------------------------------------------------------- the data


def load_crops(args) -> tuple:
    if not args.crops.exists():
        raise SystemExit(f"{args.crops} not found, run 27_representation_learning.py first to cut the boxes")
    stored = np.load(args.crops, allow_pickle=True)
    crops = stored["crops"]
    ids = [str(name) for name in stored["patient_id"]]
    if args.limit:
        crops, ids = crops[: args.limit], ids[: args.limit]
    return crops, ids


def as_float(batch: np.ndarray):
    import torch

    return torch.from_numpy(batch.astype(np.float32) / 255.0)


def augment(batch):
    """Flips, shifts, gain and noise. Left-right flip only, the heart has a side."""
    import torch

    out = batch
    if bool(torch.rand(1) < 0.5):
        out = torch.flip(out, dims=(4,))
    for axis in (2, 3, 4):
        move = int(torch.randint(-4, 5, (1,)).item())
        if move:
            out = torch.roll(out, move, dims=axis)
    gain = 1.0 + 0.12 * (torch.rand(len(out), 1, 1, 1, 1, device=out.device) - 0.5)
    out = out.clone()
    out[:, 0:1] = torch.clamp(out[:, 0:1] * gain + 0.02 * torch.randn_like(out[:, 0:1]), 0.0, 1.0)
    return out


def volumes_of(crops: np.ndarray) -> np.ndarray:
    """Atrial and fat share of the box, read from the mask channel, as aux targets."""
    outline = crops[:, 1].astype(np.float32) / 255.0
    atrium = (outline > 0.8).mean(axis=(1, 2, 3))
    fat = ((outline > 0.2) & (outline <= 0.8)).mean(axis=(1, 2, 3))
    both = np.stack([atrium, fat], axis=1)
    return (both - both.mean(axis=0)) / (both.std(axis=0) + 1e-6)


# ----------------------------------------------------------------- the networks


def build_resnet3d(args, aux: bool):
    """Residual 3D network with group norm, squeeze and excitation, attention pooling."""
    import torch
    from torch import nn

    widths = [args.width, args.width * 2, args.width * 4, args.width * 8]

    def norm(channels: int):
        return nn.GroupNorm(min(8, channels), channels)

    class Squeeze(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.gate = nn.Sequential(
                nn.Linear(channels, max(channels // 8, 4)),
                nn.ReLU(inplace=True),
                nn.Linear(max(channels // 8, 4), channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            weights = self.gate(x.mean(dim=(2, 3, 4)))
            return x * weights[:, :, None, None, None]

    class Residual(nn.Module):
        def __init__(self, inside: int, outside: int, stride: int):
            super().__init__()
            self.path = nn.Sequential(
                nn.Conv3d(inside, outside, 3, stride=stride, padding=1, bias=False),
                norm(outside),
                nn.ReLU(inplace=True),
                nn.Conv3d(outside, outside, 3, padding=1, bias=False),
                norm(outside),
                Squeeze(outside),
            )
            self.skip = (
                nn.Sequential(nn.Conv3d(inside, outside, 1, stride=stride, bias=False), norm(outside))
                if stride != 1 or inside != outside
                else nn.Identity()
            )
            self.out = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.out(self.path(x) + self.skip(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv3d(2, widths[0], 3, stride=1, padding=1, bias=False),
                                      norm(widths[0]), nn.ReLU(inplace=True))
            stages, channels = [], widths[0]
            for width in widths:
                stages += [Residual(channels, width, stride=2), Residual(width, width, stride=1)]
                channels = width
            self.stages = nn.Sequential(*stages)
            self.attention = nn.Conv3d(channels, 1, 1)
            self.size = channels
            self.drop = nn.Dropout(0.3)
            self.head = nn.Linear(channels, 1)
            self.aux = nn.Linear(channels, 2) if aux else None

        def embed(self, x):
            maps = self.stages(self.stem(x))
            weights = torch.softmax(self.attention(maps).flatten(2), dim=2)
            return (maps.flatten(2) * weights).sum(dim=2)

        def forward(self, x):
            pooled = self.embed(x)
            logit = self.head(self.drop(pooled))
            return (logit, self.aux(pooled) if self.aux is not None else None, pooled)

    return Net()


def build_slice2d(args):
    """ImageNet ResNet18 on the axial slices, attention over the slices."""
    import torch
    from torch import nn

    try:
        from torchvision.models import resnet18

        try:
            backbone = resnet18(weights="IMAGENET1K_V1")
            pretrained = True
        except Exception:
            backbone = resnet18(weights=None)
            pretrained = False
    except Exception as problem:  # torchvision missing
        raise SystemExit(f"slice2d needs torchvision: {problem}")
    if not pretrained:
        print("  warning: ImageNet weights could not be downloaded, slice2d starts from random")
    backbone.fc = nn.Identity()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.size = 512
            self.attention = nn.Linear(512, 1)
            self.drop = nn.Dropout(0.3)
            self.head = nn.Linear(512, 1)
            self.aux = None

        def embed(self, x):
            # keep the middle slices, the atrium sits in the centre of the box
            depth = x.shape[2]
            keep = min(args.slices, depth)
            first = (depth - keep) // 2
            picked = x[:, :, first : first + keep]
            patients, _, kept, height, width = picked.shape
            flat = picked.permute(0, 2, 1, 3, 4).reshape(patients * kept, 2, height, width)
            three = torch.cat([flat, flat[:, 0:1]], dim=1)  # image, mask, image
            tokens = self.backbone(three).view(patients, kept, 512)
            weights = torch.softmax(self.attention(tokens), dim=1)
            return (tokens * weights).sum(dim=1)

        def forward(self, x):
            pooled = self.embed(x)
            return self.head(self.drop(pooled)), None, pooled

    return Net()


def build_model(name: str, args):
    if name == "slice2d":
        return build_slice2d(args)
    return build_resnet3d(args, aux=name == "multitask")


# -------------------------------------------------------------------- training


def train_one(name, crops, target, aux_target, train, test, args, seed: int):
    """One model on one training fold; returns test probabilities and all embeddings."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(name, args).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(args.epochs, 1))
    positives = max(int(target[train].sum()), 1)
    weight = torch.tensor([(len(train) - positives) / positives], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    regression = nn.SmoothL1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(train)
        total = 0.0
        for start in range(0, len(order), args.batch):
            index = order[start : start + args.batch]
            batch = augment(as_float(crops[index]).to(device))
            labels = torch.from_numpy(target[index].astype(np.float32))[:, None].to(device)
            extra = torch.from_numpy(aux_target[index].astype(np.float32)).to(device)
            if args.mixup > 0 and len(index) > 1:
                share = float(np.random.beta(args.mixup, args.mixup))
                shuffle = torch.randperm(len(index), device=device)
                batch = share * batch + (1 - share) * batch[shuffle]
                labels = share * labels + (1 - share) * labels[shuffle]
                extra = share * extra + (1 - share) * extra[shuffle]
            logit, predicted, _ = model(batch)
            loss = criterion(logit, labels)
            if predicted is not None:
                loss = loss + 0.2 * regression(predicted, extra)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(index)
        schedule.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"      epoch {epoch:3d}  loss {total / len(order):.4f}")

    model.eval()
    probabilities, embeddings = [], []
    with torch.no_grad():
        for start in range(0, len(crops), args.batch):
            batch = as_float(crops[start : start + args.batch]).to(device)
            logit, _, pooled = model(batch)
            probabilities.append(torch.sigmoid(logit)[:, 0].cpu().numpy())
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(embeddings), model, device


def cross_validate(name, crops, target, aux_target, args, journal) -> tuple:
    """Fold-safe out-of-fold probabilities and embeddings for one model."""
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    tag = f"{name} k={args.folds} e={args.epochs} w={args.width} r={args.repeats} n={len(target)}"
    done = journal.get(tag) or {}
    scores = np.zeros(len(target), dtype=float)
    embedding = None
    folds = []
    last = None

    for number, (train, test) in enumerate(splitter.split(crops, target), 1):
        key = str(number)
        if key in done:
            scores[test] = np.array(done[key]["scores"])
            folds.append(done[key]["auc"])
            print(f"    fold {number} reused, AUC {folds[-1]:.4f}")
            continue
        print(f"    fold {number} of {args.folds}, {len(train)} train, {len(test)} test")
        collected, pooled_runs = [], []
        for repeat in range(args.repeats):
            probabilities, pooled, model, device = train_one(
                name, crops, target, aux_target, train, test, args, args.seed + 100 * repeat + number
            )
            collected.append(probabilities)
            pooled_runs.append(pooled)
            last = (model, device)
        probabilities = np.mean(collected, axis=0)
        pooled = np.mean(pooled_runs, axis=0)
        if embedding is None:
            embedding = np.zeros((len(target), pooled.shape[1]), dtype=np.float32)
        embedding[test] = pooled[test]  # only ever from a model that did not see these patients
        scores[test] = probabilities[test]
        folds.append(float(roc_auc_score(target[test], scores[test])))
        print(f"      fold AUC {folds[-1]:.4f}")
        done[key] = {"auc": folds[-1], "scores": scores[test].tolist(), "test": test.tolist()}
        journal.put(tag, done)

    for value in done.values():
        scores[np.array(value["test"])] = np.array(value["scores"])
    return float(np.mean(folds)), float(np.std(folds)), scores, embedding, last


# ------------------------------------------------------------------- reference


def measurement_table(ids: list, args):
    frames = []
    for path in (args.measurements, args.radiomics):
        if path and Path(path).exists():
            table = pd.read_excel(path, sheet_name="Features")
            table.columns = [str(name).strip() for name in table.columns]
            frames.append(table.set_index("patient_id"))
    if not frames:
        return None
    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.join(extra[[c for c in extra.columns if c not in merged.columns]])
    merged.index = merged.index.astype(str)
    return merged.reindex(ids)


def logistic_auc(values: np.ndarray, target: np.ndarray, args) -> tuple:
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    scores = np.zeros(len(target), dtype=float)
    folds = []
    for train, test in splitter.split(values, target):
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, C=0.1)),
            ]
        )
        pipeline.fit(values[train], target[train])
        scores[test] = pipeline.predict_proba(values[test])[:, 1]
        folds.append(roc_auc_score(target[test], scores[test]))
    return float(np.mean(folds)), float(np.std(folds)), scores


def fold_auc(scores: np.ndarray, target: np.ndarray, args) -> tuple:
    """Mean and SD over the same folds, for a score that already exists."""
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = [roc_auc_score(target[test], scores[test]) for _, test in splitter.split(scores[:, None], target)]
    return float(np.mean(folds)), float(np.std(folds))


def ranked(scores: np.ndarray) -> np.ndarray:
    return rankdata(scores) / len(scores)


# ----------------------------------------------------------------------- plots


def plot_roc(curves: list, folder: Path, group: str) -> None:
    plt.figure(figsize=(6.5, 6))
    for name, target, scores, auc in curves:
        fpr, tpr, _ = roc_curve(target, scores)
        plt.plot(fpr, tpr, lw=1.4, label=f"{name}  AUC {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="0.6", lw=0.8)
    plt.xlabel("1 - specificity")
    plt.ylabel("sensitivity")
    plt.title(f"Supervised image models, {group}, all fold-safe")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(folder / "01_roc_supervised.png", dpi=200)
    plt.close()


def plot_embedding(embedding: np.ndarray, target: np.ndarray, name: str, args, folder: Path) -> None:
    scaled = StandardScaler().fit_transform(np.nan_to_num(embedding))
    try:
        import umap

        points = umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=args.seed).fit_transform(scaled)
        axis_name = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA

        points = PCA(n_components=2, random_state=args.seed).fit_transform(scaled)
        axis_name = "principal component"
    plt.figure(figsize=(6.5, 6))
    for value, colour, label in ((0, "#5dade2", "control"), (1, "#e74c3c", args.group)):
        plt.scatter(points[target == value, 0], points[target == value, 1], s=12, alpha=0.6, c=colour, label=label)
    plt.legend()
    plt.xlabel(f"{axis_name} 1")
    plt.ylabel(f"{axis_name} 2")
    plt.title(f"Representation of {name}, held out patients only")
    plt.tight_layout()
    plt.savefig(folder / f"02_embedding_{name}.png", dpi=200)
    plt.close()


def saliency_map(model, device, crops: np.ndarray, target: np.ndarray, args, folder: Path, name: str) -> None:
    """Where the network looks: gradient of the AF logit, averaged over cases."""
    import torch

    cases = np.where(target == 1)[0][:40]
    if len(cases) == 0:
        return
    model.eval()
    total = None
    for start in range(0, len(cases), args.batch):
        batch = as_float(crops[cases[start : start + args.batch]]).to(device)
        batch.requires_grad_(True)
        logit, _, _ = model(batch)
        logit.sum().backward()
        piece = batch.grad.detach().abs()[:, 0].cpu().numpy().sum(axis=0)
        total = piece if total is None else total + piece
    total = total / total.max()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for axis, (plane, projection) in zip(
        axes,
        (("axial", total.mean(axis=0)), ("coronal", total.mean(axis=1)), ("sagittal", total.mean(axis=2))),
    ):
        picture = axis.imshow(projection, cmap="inferno")
        axis.set_title(f"{plane}")
        axis.axis("off")
        fig.colorbar(picture, ax=axis, fraction=0.046)
    fig.suptitle(f"Where {name} looks for {args.group}, mean gradient over cases")
    plt.tight_layout()
    plt.savefig(folder / f"03_saliency_{name}.png", dpi=200)
    plt.close()


# ------------------------------------------------------------------------ main


def main(argv=None) -> None:
    args = parse_args(argv)
    folder = args.output / "supervised"
    plots = folder / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    journal = Journal(folder / "progress.json", args.fresh)

    crops, ids = load_crops(args)
    groups = pd.read_excel(args.groups, sheet_name="Features")
    groups.columns = [str(name).strip() for name in groups.columns]
    label_of = dict(zip(groups["patient_id"].astype(str), groups["group"]))
    keep = [number for number, name in enumerate(ids) if label_of.get(name) in (args.group, "control")]
    crops, ids = crops[keep], [ids[number] for number in keep]
    target = np.array([1 if label_of[name] == args.group else 0 for name in ids])
    aux_target = volumes_of(crops)
    print(f"1. {len(ids)} patients, {int(target.sum())} {args.group}, box {crops.shape[2:]}")
    if target.sum() < 10 or (len(target) - target.sum()) < 10:
        raise SystemExit("too few patients of one class to cross-validate")

    rows, curves, model_scores = [], [], {}
    for name in args.models:
        print(f"\n2. {name}, fold-safe {args.folds}-fold, {args.repeats} restart(s) per fold")
        auc, spread, scores, embedding, last = cross_validate(name, crops, target, aux_target, args, journal)
        rows.append({"model": name, "auc": round(auc, 4), "sd": round(spread, 4)})
        curves.append((name, target, scores, auc))
        model_scores[name] = scores
        print(f"   {name:10s} AUC {auc:.4f} (SD {spread:.4f})")
        if not args.no_plots:
            if embedding is not None and embedding.any():
                plot_embedding(embedding, target, name, args, plots)
            if last is not None and name != "slice2d":
                saliency_map(last[0], last[1], crops, target, args, plots, name)

    ensemble = None
    if len(model_scores) > 1:
        ensemble = np.mean([ranked(score) for score in model_scores.values()], axis=0)
        auc, spread = fold_auc(ensemble, target, args)
        rows.append({"model": "ensemble of the three", "auc": round(auc, 4), "sd": round(spread, 4)})
        curves.append(("ensemble", target, ensemble, auc))
        print(f"\n3. ensemble AUC {auc:.4f} (SD {spread:.4f})")

    merged = measurement_table(ids, args)
    best_scores = None
    if merged is not None:
        for name, columns in (("LA volume alone", ["la_volume_ml"]), ("best four features", BEST_FOUR)):
            have = [c for c in columns if c in merged.columns]
            if len(have) != len(columns):
                print(f"reference '{name}' skipped, missing {set(columns) - set(have)}")
                continue
            values = merged[have].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            auc, spread, scores = logistic_auc(values, target, args)
            rows.append({"model": name, "auc": round(auc, 4), "sd": round(spread, 4)})
            curves.append((name, target, scores, auc))
            print(f"   {name:22s} AUC {auc:.4f} (SD {spread:.4f})")
            if name == "best four features":
                best_scores = scores

    if best_scores is not None:
        combined = ensemble if ensemble is not None else ranked(list(model_scores.values())[0])
        hybrid = 0.5 * ranked(best_scores) + 0.5 * ranked(combined)
        auc, spread = fold_auc(hybrid, target, args)
        rows.append({"model": "hybrid, four features + images", "auc": round(auc, 4), "sd": round(spread, 4)})
        curves.append(("hybrid", target, hybrid, auc))
        print(f"\n4. hybrid of the four features and the images AUC {auc:.4f} (SD {spread:.4f})")

    table = pd.DataFrame({"patient_id": ids, "group": [label_of[name] for name in ids], "case": target})
    for name, scores in model_scores.items():
        table[f"score_{name}"] = scores
    if ensemble is not None:
        table["score_ensemble"] = ensemble
    table.to_csv(folder / f"scores_{args.group}.csv", index=False)

    if not args.no_plots and curves:
        plot_roc(curves, plots, args.group)
        print(f"\nPlots : {plots}")

    workbook = folder / "supervised.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(rows).sort_values("auc", ascending=False).to_excel(writer, sheet_name="AUC", index=False)
        table.to_excel(writer, sheet_name="Scores", index=False)
    print(f"Excel : {workbook}")
    journal.put("finished", True)


if __name__ == "__main__":
    main()
