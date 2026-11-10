"""The manuscript figures: the method diagram, the flow chart and the panels.

Nothing is modelled here. The diagram is drawn from the settings, the other
figures are assembled from what scripts 29 to 32 already wrote, so this takes
seconds and can be re-run after every new result.

    figure1_method.png   the whole model on one page: rim crop, encoder,
                         attention pooling, the latent code, and the three
                         blocks A attenuation radiomics, B radiomics of the
                         learnt feature maps, C the residual code
    figure2_flow.png     patients in, patients excluded, patients analysed
    figure3_results.png  the AUC of every model and every group, from
                         summary\\group_summary.xlsx
    figure4_latent.png   the learnt side as a strip: the shell the loss uses,
                         where the encoder attends, the latent space, and
                         which latent radiomics separate the cases, taken from
                         the plots scripts 29 and 30 wrote
    figure5_roc.png      the ROC curves side by side

Usage
    python 33_paper_figures.py
    python 33_paper_figures.py --output W:\\...\\results --case 329 --control 664
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

SEG = Path(r"W:\SCAPIS_seg_lowres_3mm_noncontrast")
ANALYSIS = SEG / "analysis" / "results"

BLUE, GREEN, RED, SAND, GREY = "#cfe0f3", "#d6ecd2", "#f6d3cd", "#fdf0cd", "#e8e8e8"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ANALYSIS)
    parser.add_argument("--case", type=int, default=329)
    parser.add_argument("--control", type=int, default=664)
    parser.add_argument("--analysed", type=int, default=991)
    parser.add_argument("--group", default="AF")
    parser.add_argument("--box", default="64,160,160")
    parser.add_argument("--spacing", type=float, default=1.25)
    return parser.parse_known_args(argv)[0]


# ------------------------------------------------------------------- drawing


def box(axis, x, y, width, height, text, colour, size=8.5, bold=False):
    axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012",
                                  linewidth=0.9, edgecolor="#5a5a5a", facecolor=colour))
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=size,
              fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(axis, start, end, style="-|>", colour="#444444", dashed=False):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11,
                                   linewidth=0.9, color=colour,
                                   linestyle="--" if dashed else "-",
                                   connectionstyle="arc3,rad=0.0"))


def slab(axis, x, y, width, height, colour, depth=0.012, label="", under="", size=7.4):
    """A box drawn as a small 3D slab, so a stack of them reads as a feature map."""
    axis.add_patch(Polygon([(x, y + height), (x + depth, y + height + depth * 1.6),
                            (x + width + depth, y + height + depth * 1.6), (x + width, y + height)],
                           closed=True, facecolor=colour, edgecolor="#5a5a5a", linewidth=0.7))
    axis.add_patch(Polygon([(x + width, y), (x + width + depth, y + depth * 1.6),
                            (x + width + depth, y + height + depth * 1.6), (x + width, y + height)],
                           closed=True, facecolor=colour, edgecolor="#5a5a5a", linewidth=0.7,
                           alpha=0.75))
    axis.add_patch(Rectangle((x, y), width, height, facecolor=colour, edgecolor="#5a5a5a",
                             linewidth=0.8))
    if label:
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=size)
    if under:
        axis.text(x + width / 2, y - 0.022, under, ha="center", va="top", fontsize=6.8,
                  color="#444444")


def tokens(axis, x, y, count, size, colour, gap=0.004):
    """A row of small squares, used for the latent code and the attention tokens."""
    for index in range(count):
        axis.add_patch(Rectangle((x + index * (size + gap), y), size, size, facecolor=colour,
                                 edgecolor="#5a5a5a", linewidth=0.6))
    return x + count * (size + gap) - gap


def figure_method(args, folder: Path) -> None:
    """The network on one page: tensors, encoder levels, attention, latent, heads, blocks."""
    figure, axis = plt.subplots(figsize=(15.0, 8.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("Self-supervised encoder on peri-atrial CT, and radiomics of its latent "
                   "feature maps, for atrial fibrillation",
                   fontsize=12.5, fontweight="bold", y=1.0)

    depth, rows, columns = [int(v) for v in str(args.box).split(",")]
    shapes, width_now = [], 24
    for level in range(4):
        shapes.append((width_now, depth // 2 ** (level + 1), rows // 2 ** (level + 1),
                       columns // 2 ** (level + 1)))
        width_now *= 2

    # input ------------------------------------------------------------------
    axis.text(0.055, 0.955, "input", fontsize=9, fontweight="bold", ha="center")
    slab(axis, 0.015, 0.74, 0.085, 0.145, BLUE, depth=0.016,
         label=f"rim crop\n2 x {depth} x {rows} x {columns}\nat {args.spacing} mm",
         under="channel 1 HU window\nchannel 2 LA and fat labels")
    slab(axis, 0.015, 0.545, 0.085, 0.085, SAND, depth=0.016, label="shell weight\nw = 1 / 0.05",
         under="atrial wall and the fat\nwithin 10 mm")
    arrow(axis, (0.058, 0.74), (0.058, 0.66))

    # encoder ----------------------------------------------------------------
    axis.text(0.255, 0.955, "encoder, 3D convolutional, stride 2 per level", fontsize=9,
              fontweight="bold", ha="center")
    left, top, height = 0.135, 0.885, 0.145
    for level, (channels, d, r, c) in enumerate(shapes):
        scale = 1.0 - 0.17 * level
        slab(axis, left, top - height * scale, 0.052, height * scale, GREEN, depth=0.013,
             label=f"L{level + 1}\n{channels}ch", under=f"{d}x{r}x{c}", size=7.2)
        if level:
            arrow(axis, (left - 0.016, 0.80), (left - 0.002, 0.80))
        left += 0.068
    arrow(axis, (0.10, 0.81), (0.133, 0.81))
    arrow(axis, (0.10, 0.588), (0.133, 0.76), dashed=True)
    axis.text(0.255, 0.615, "two 3x3x3 convolutions, instance norm, GELU, residual",
              fontsize=7.6, ha="center", color="#444444", style="italic")

    # attention pooling ------------------------------------------------------
    box(axis, 0.135, 0.455, 0.25, 0.12,
        "attention pooling at every level\nq . k over the voxel tokens of the level, softmax,\n"
        "weighted sum, so the readout is where the model looks,\nnot the mean of the volume",
        "#e5ddf2", size=8.2)
    for index in range(4):
        arrow(axis, (0.161 + 0.068 * index, 0.70 - 0.025 * index), (0.20 + 0.04 * index, 0.575))

    # latent code ------------------------------------------------------------
    axis.text(0.465, 0.955, "latent", fontsize=9, fontweight="bold", ha="center")
    right = tokens(axis, 0.415, 0.80, 12, 0.0145, "#b9c7e8")
    axis.text(0.465, 0.775, f"latent code, {96} values", fontsize=7.6, ha="center")
    arrow(axis, (0.385, 0.515), (0.415, 0.72))
    for index in range(4):
        slab(axis, 0.418 + 0.006 * index, 0.60 + 0.006 * index, 0.075, 0.085, "#cdd9ef",
             depth=0.010, label="" if index < 3 else "latent\nfeature maps", size=7.0)
    axis.text(0.465, 0.575, "every channel of every level\nis itself a small 3D image",
              fontsize=7.4, ha="center", va="top", color="#444444")

    # pretraining heads ------------------------------------------------------
    axis.text(0.635, 0.955, "pretraining, no labels, inside the training fold", fontsize=9,
              fontweight="bold", ha="center")
    box(axis, 0.545, 0.845, 0.185, 0.085,
        "masked reconstruction\n40 % of blocks hidden, decoder rebuilds them,\n"
        "L1 weighted by the shell", GREY, size=7.8)
    box(axis, 0.545, 0.735, 0.185, 0.095,
        "self-distillation, SimSiam / DINO style\ntwo augmented views, student predicts the\n"
        "stop-gradient teacher of the other view", GREY, size=7.8)
    box(axis, 0.545, 0.635, 0.185, 0.085,
        "feature distillation\npredict the four handcrafted features,\nso the code must contain them",
        GREY, size=7.8)
    for height_at in (0.887, 0.782, 0.677):
        arrow(axis, (0.50, 0.81), (0.545, height_at))

    # the three blocks -------------------------------------------------------
    axis.text(0.845, 0.955, "feature blocks and evaluation", fontsize=9, fontweight="bold",
              ha="center")
    box(axis, 0.762, 0.835, 0.17, 0.10,
        "A  attenuation radiomics\nLA HU IQR, LA surface area,\nLA-EAT flatness, fat median HU",
        RED, size=8.0, bold=True)
    box(axis, 0.762, 0.705, 0.17, 0.115,
        "B  latent radiomics\nmean, SD, energy and gradient of\nevery latent channel inside the\n"
        "atrium, the fat and the LA-EAT rim", RED, size=8.0, bold=True)
    box(axis, 0.762, 0.575, 0.17, 0.115,
        "C  residual latent\nblock A regressed out of the code\ninside the training fold, so C is\n"
        "what the radiomics cannot express", RED, size=8.0, bold=True)
    axis.annotate("", xy=(0.80, 0.835), xytext=(0.10, 0.90),
                  arrowprops=dict(arrowstyle="-|>", color="#888888", linewidth=0.9,
                                  connectionstyle="arc3,rad=-0.14"))
    axis.text(0.44, 0.995, "the same masks, handcrafted", fontsize=7.6, ha="center",
              color="#666666")
    arrow(axis, (0.50, 0.645), (0.762, 0.762))
    arrow(axis, (0.50, 0.615), (0.762, 0.632))

    box(axis, 0.762, 0.40, 0.17, 0.13,
        "logistic model per block\nA, B, C, A+B, A+C, A+B+C\n\nDeLong against A",
        GREEN, size=8.4, bold=True)
    for height_at in (0.835, 0.705, 0.575):
        axis.annotate("", xy=(0.847, 0.53), xytext=(0.847, height_at),
                      arrowprops=dict(arrowstyle="-|>", color="#444444", linewidth=0.9,
                                      connectionstyle="arc3,rad=0.0"))
    box(axis, 0.762, 0.255, 0.17, 0.10,
        f"{args.group} against control\nAUC, repeated 5-fold,\nseed averaged, test time augmented",
        SAND, size=8.2)
    arrow(axis, (0.847, 0.40), (0.847, 0.355))

    box(axis, 0.10, 0.07, 0.62, 0.10,
        "Fold safety: the split is made first, then cropping, pretraining, PCA, the "
        "orthogonalisation of C,\nthe feature screen and every classifier are fitted on the "
        "training patients only.", GREY, size=8.4)
    plt.tight_layout()
    plt.savefig(folder / "figure1_method.png", dpi=240)
    plt.close()


def figure_flow(args, folder: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 7.0))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    total = args.case + args.control
    steps = [
        (f"SCAPIS non-contrast cardiac CT\nall available scans", BLUE),
        (f"{args.group} cases identified  n = {args.case}", BLUE),
        (f"Controls matched on age, sex and site, 2:1\nn = {args.control}", BLUE),
        (f"Cohort  n = {total}", GREEN),
        (f"Left atrium and pericardial fat segmented\n(nnU-Net v2), fat kept at -190 to -30 HU", SAND),
        (f"Analysed  n = {args.analysed}\n({total - args.analysed} excluded, no usable mask or crop)", GREEN),
    ]
    height, gap = 0.115, 0.038
    y = 0.88
    for text, colour in steps:
        box(axis, 0.06, y, 0.88, height, text, colour, size=9)
        if y - gap - height > 0:
            arrow(axis, (0.5, y), (0.5, y - gap))
        y -= height + gap
    axis.set_title("Study flow", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(folder / "figure2_flow.png", dpi=240)
    plt.close()


def figure_results(args, folder: Path) -> bool:
    book = args.output / "summary" / "group_summary.xlsx"
    if not book.exists():
        print(f"   figure3 skipped, run 32_group_summary.py first ({book} missing)")
        return False
    table = pd.read_excel(book, sheet_name="AUC")
    wide = table.pivot_table(index="model", columns="group", values="auc")
    wide = wide.reindex(wide.mean(axis=1).sort_values().index)
    wide.plot.barh(figsize=(9.5, 0.52 * len(wide) + 2.4), width=0.78)
    plt.axvline(0.5, color="0.6", lw=0.8)
    plt.xlim(0.4, 1.0)
    plt.xlabel("AUC, cross-validated")
    plt.ylabel("")
    plt.title("Discrimination of every model, by patient group")
    plt.legend(title="", fontsize=8)
    plt.tight_layout()
    plt.savefig(folder / "figure3_results.png", dpi=240)
    plt.close()
    return True


def figure_blocks(args, folder: Path) -> bool:
    """Radiomics against the latent blocks, drawn from latent.xlsx alone.

    Three panels: the AUC of every block with its DeLong p against the radiomics,
    which latent radiomics carry the separation, and the two scores against each
    other, which shows whether the latent block is complementary or a copy.
    """
    book = args.output / "latent" / (
        "latent.xlsx" if args.group == "AF" else f"latent_{args.group.replace(',', '_')}.xlsx")
    if not book.exists():
        print(f"   figure6 skipped, run 30_latent_radiomics.py first ({book} missing)")
        return False
    sheets = pd.read_excel(book, sheet_name=None)
    auc = sheets["AUC"]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    order = auc.sort_values("auc")
    bars = axes[0].barh(order["block"], order["auc"],
                        xerr=pd.to_numeric(order.get("sd"), errors="coerce"),
                        color=["#c44e52" if str(name).startswith("A ") else "#4c78a8"
                               for name in order["block"]], height=0.66, capsize=3)
    for bar, (_, row) in zip(bars, order.iterrows()):
        note = f"{row['auc']:.3f}"
        if str(row.get("delong_p_vs_A", "")) not in {"", "nan"}:
            note += f"  p={float(row['delong_p_vs_A']):.3f}"
        shift = float(pd.to_numeric(row.get("sd"), errors="coerce") or 0) + 0.014
        axes[0].text(row["auc"] + shift, bar.get_y() + bar.get_height() / 2, note,
                     va="center", fontsize=8)
    axes[0].axvline(0.5, color="0.6", lw=0.8)
    axes[0].set_xlim(0.45, min(1.02, order["auc"].max() + 0.16))
    axes[0].set_xlabel("AUC, cross-validated")
    axes[0].set_title("(a) attenuation radiomics against the latent blocks", fontsize=10, loc="left")

    where = sheets.get("Latent_radiomics")
    if where is not None and len(where):
        top = where.head(15).iloc[::-1]
        axes[1].barh(top["block"].astype(str), top["mean_abs_t"], color="#55a868", height=0.68)
        axes[1].set_xlabel("mean |t| between cases and controls")
        axes[1].tick_params(axis="y", labelsize=7)
    axes[1].set_title("(b) which latent radiomics separate the cases", fontsize=10, loc="left")

    scores = sheets.get("Scores")
    first = "A_attenuation_radiomics"
    second = next((name for name in ("B_latent_radiomics", "C_residual_latent")
                   if scores is not None and name in scores.columns), None)
    if scores is not None and first in scores.columns and second:
        for value, colour, name in ((0, "#9aa7b8", "control"), (1, "#c44e52", "case")):
            part = scores[scores["case"] == value]
            axes[2].scatter(part[first], part[second], s=9, alpha=0.65, c=colour, label=name)
        axes[2].set_xlabel("block A score, attenuation radiomics")
        axes[2].set_ylabel(f"{second.split('_')[0]} score, latent")
        axes[2].legend(fontsize=8, frameon=False)
    axes[2].set_title("(c) the two scores against each other", fontsize=10, loc="left")

    figure.suptitle("The learnt latent representation against conventional radiomics",
                    fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(folder / "figure6_blocks.png", dpi=240)
    plt.close()
    return True


def tile(paths: list, titles: list, out: Path, title: str, columns: int = 2) -> bool:
    have = [(path, name) for path, name in zip(paths, titles) if path.exists()]
    if not have:
        print(f"   {out.name} skipped, none of its panels exist yet")
        return False
    rows = (len(have) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(6.4 * columns, 5.2 * rows))
    for axis, (path, name), letter in zip(list(axes.ravel()) if rows * columns > 1 else [axes],
                                          have, "abcdefgh"):
        axis.imshow(mpimg.imread(str(path)))
        axis.set_title(f"({letter}) {name}", fontsize=10, loc="left")
        axis.axis("off")
    for axis in list(axes.ravel())[len(have):] if rows * columns > 1 else []:
        axis.axis("off")
    figure.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return True


def figure_rim(args, folder: Path) -> None:
    """The rim encoder-decoder on one page, with its own AUC bars under it.

    Top: the crop, the four encoder levels with their tensor sizes, the latent
    code, the decoder that rebuilds the shell, and the two auxiliary heads.
    Bottom: what every readout of that encoder scored, from rim.xlsx, against
    LA volume and the four handcrafted features.
    """
    figure = plt.figure(figsize=(14.0, 9.6))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.14)
    axis = figure.add_subplot(grid[0])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("Encoder-decoder on the peri-atrial rim, and the latent representation "
                   "it learns", fontsize=12.5, fontweight="bold", y=1.02)

    depth, rows_, columns_ = [int(v) for v in str(args.box).split(",")]
    shapes, channels = [], 24
    for level in range(4):
        shapes.append((channels, depth // 2 ** (level + 1), rows_ // 2 ** (level + 1),
                       columns_ // 2 ** (level + 1)))
        channels *= 2

    slab(axis, 0.012, 0.42, 0.10, 0.34, BLUE, depth=0.018,
         label=f"rim crop\n2 x {depth} x {rows_} x {columns_}\nat {args.spacing} mm",
         under="HU and the LA / fat labels,\nthe atrium at the centre", size=8.0)
    arrow(axis, (0.125, 0.60), (0.155, 0.60))

    left = 0.158
    for level, (width_now, d, r, c) in enumerate(shapes):      # encoder, narrowing
        height = 0.34 - 0.062 * level
        slab(axis, left, 0.59 - height / 2, 0.056, height, GREEN, depth=0.014,
             label=f"L{level + 1}\n{width_now}ch", under=f"{d}x{r}x{c}", size=7.4)
        arrow(axis, (left + 0.058, 0.60), (left + 0.072, 0.60))
        left += 0.074
    axis.text(0.31, 0.90, "encoder, 3x3x3 convolutions, stride 2, attention pooling per level",
              fontsize=8.6, ha="center", color="#333333")

    tokens(axis, 0.455, 0.575, 10, 0.0135, "#b9c7e8")
    axis.text(0.52, 0.665, f"latent code, {96} values", fontsize=7.8, ha="center")
    for index in range(4):                                     # the feature-map stack
        slab(axis, 0.458 + 0.006 * index, 0.37 + 0.006 * index, 0.088, 0.095, "#cdd9ef",
             depth=0.011, label="" if index < 3 else "latent\nfeature maps", size=7.2)
    axis.text(0.505, 0.345, "each channel is a small 3D image,\nblock B measures it",
              fontsize=7.4, ha="center", va="top", color="#444444")
    arrow(axis, (0.573, 0.60), (0.60, 0.60))

    left = 0.603                                               # decoder, widening back
    for level in range(4):
        height = 0.15 + 0.062 * level
        slab(axis, left, 0.59 - height / 2, 0.052, height, SAND, depth=0.014,
             label=f"D{level + 1}", size=7.4)
        arrow(axis, (left + 0.054, 0.60), (left + 0.067, 0.60))
        left += 0.069
    axis.text(0.72, 0.90, "decoder, transposed convolutions", fontsize=8.6, ha="center",
              color="#333333")

    slab(axis, 0.885, 0.44, 0.095, 0.30, RED, depth=0.016,
         label="rebuilt crop", under="L1 loss weighted 20x on the\natrial wall and the fat "
         "within 10 mm", size=8.0)
    box(axis, 0.60, 0.17, 0.175, 0.12,
        "self-distillation\ntwo views, the student predicts the\nstop-gradient teacher",
        GREY, size=7.8)
    box(axis, 0.80, 0.17, 0.175, 0.12,
        "feature distillation\nthe code must predict the four\nhandcrafted features",
        GREY, size=7.8)
    arrow(axis, (0.55, 0.37), (0.655, 0.29), dashed=True)
    arrow(axis, (0.56, 0.38), (0.855, 0.29), dashed=True)
    axis.text(0.50, 0.075, "40 % of the crop is hidden before the encoder, and every step here "
              "is fitted inside the training fold only",
              fontsize=8.2, ha="center", style="italic", color="#444444")

    lower = figure.add_subplot(grid[1])
    book = args.output / "rim" / ("rim.xlsx" if args.group == "AF" else f"rim_{args.group}.xlsx")
    if not book.exists():
        lower.axis("off")
        lower.text(0.5, 0.5, f"run 29_rim_representation.py first, {book.name} not found",
                   ha="center", va="center", fontsize=10, color="#777777")
    else:
        table = pd.read_excel(book, sheet_name="AUC")
        table["auc"] = pd.to_numeric(table["auc"], errors="coerce")
        table = table.dropna(subset=["auc"]).sort_values("auc")
        reference = ("LA volume alone", "best four features")   # the models to beat
        colours = ["#c0504d" if str(name).strip() in reference else "#4f81bd"
                   for name in table["block"]]
        bars = lower.barh(table["block"].astype(str), table["auc"], color=colours)
        for bar, value in zip(bars, table["auc"]):
            lower.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                       va="center", fontsize=8)
        lower.axvline(0.5, color="0.6", lw=0.8)
        lower.set_xlim(0.45, min(1.0, float(table["auc"].max()) + 0.06))
        lower.set_xlabel("AUC, out of fold")
        lower.set_title("What each readout of the encoder scores, red = reference model",
                        fontsize=10, loc="left")
        lower.grid(axis="x", alpha=0.3)
    plt.savefig(folder / "figure7_rim.png", dpi=220, bbox_inches="tight")
    plt.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    folder = args.output / "paper"
    folder.mkdir(parents=True, exist_ok=True)
    rim = args.output / "rim" / "plots"
    latent = args.output / "latent" / "plots"
    summary = args.output / "summary" / "plots"

    print("1. drawing the method diagram and the flow chart")
    figure_method(args, folder)
    figure_flow(args, folder)
    print("2. the result figures, from what is already on disk")
    figure_results(args, folder)
    figure_blocks(args, folder)
    figure_rim(args, folder)
    tile([rim / "04_shell.png", rim / "03_attention.png",
          rim / "02_latent_space.png", latent / "02_where.png"],
         ["the shell the loss is weighted by", "where the encoder attends",
          "the learnt representation, held out patients",
          "which latent radiomics separate the cases"],
         folder / "figure4_latent.png", "The learnt representation")
    tile([summary / f"roc_{args.group}.png", rim / "01_roc.png", latent / "01_roc.png"],
         ["every model", "the rim encoder", "attenuation, latent and residual"],
         folder / "figure5_roc.png", "Discrimination", columns=3)
    print(f"Figures: {folder}")


if __name__ == "__main__":
    main()
