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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def figure_method(args, folder: Path) -> None:
    """One page that shows every part of the model and how the blocks are formed."""
    figure, axis = plt.subplots(figsize=(13.2, 7.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("Attenuation radiomics, learnt representation and radiomics of the learnt "
                   "feature maps for atrial fibrillation on non-contrast CT",
                   fontsize=11.5, fontweight="bold", y=1.02)

    box(axis, 0.01, 0.74, 0.15, 0.16,
        f"Non-contrast CT\n3 mm, {args.analysed} patients\n+ nnU-Net masks\nleft atrium and\n"
        "pericardial fat", BLUE, bold=True)
    box(axis, 0.01, 0.50, 0.15, 0.16,
        f"Rim crop\ncentred on the atrium\n{args.box} voxels\nat {args.spacing} mm\n"
        "channel 1 HU window\nchannel 2 labels", BLUE)
    box(axis, 0.01, 0.24, 0.15, 0.18,
        "Shell weight\nw = 1 inside the atrial\nwall and the fat within\n10 mm, 0.05 elsewhere\n"
        "so the loss is paid\nwhere LA-EAT sits", SAND)
    arrow(axis, (0.085, 0.74), (0.085, 0.665))
    arrow(axis, (0.085, 0.50), (0.085, 0.425))

    box(axis, 0.20, 0.52, 0.16, 0.26,
        "Encoder\n4 levels, 3D conv\nwidth 24, stride 2\n\nattention pooling\nper level, not mean\n"
        "latent code, 96", GREEN, bold=True)
    arrow(axis, (0.16, 0.58), (0.20, 0.62))
    arrow(axis, (0.16, 0.33), (0.20, 0.56), dashed=True)

    box(axis, 0.40, 0.80, 0.17, 0.13,
        "Masked reconstruction\n40 % of blocks hidden\nshell-weighted L1", GREY)
    box(axis, 0.40, 0.63, 0.17, 0.13,
        "SimSiam\ntwo augmented views\nof the same patient", GREY)
    box(axis, 0.40, 0.46, 0.17, 0.13,
        "Distillation\npredict the four\nhandcrafted features", GREY)
    for height in (0.865, 0.695, 0.525):
        arrow(axis, (0.36, 0.65), (0.40, height))
    axis.text(0.485, 0.41, "pretraining, no labels, inside the training fold only",
              ha="center", fontsize=8.2, style="italic")

    box(axis, 0.63, 0.78, 0.17, 0.15,
        "A  attenuation radiomics\nLA IQR, LA surface area,\nLA-EAT flatness,\nfat median HU", RED,
        bold=True)
    box(axis, 0.63, 0.58, 0.17, 0.17,
        "B  latent radiomics\nevery channel of every\nlevel as a small 3D image;\nmean, SD, energy,\n"
        "gradient inside atrium,\nfat and LA-EAT rim", RED, bold=True)
    box(axis, 0.63, 0.38, 0.17, 0.17,
        "C  residual latent\nblock A regressed out\nof the code inside the\ntraining fold, so C is\n"
        "what radiomics cannot\nexpress", RED, bold=True)
    arrow(axis, (0.57, 0.60), (0.63, 0.66))
    arrow(axis, (0.57, 0.55), (0.63, 0.46))
    axis.annotate("", xy=(0.715, 0.93), xytext=(0.085, 0.90),
                  arrowprops=dict(arrowstyle="-|>", color="#777777", linewidth=0.9,
                                  connectionstyle="arc3,rad=-0.18"))
    axis.text(0.40, 0.985, "handcrafted features from the same masks", fontsize=8, ha="center",
              color="#555555")

    box(axis, 0.84, 0.55, 0.15, 0.26,
        "Logistic model\nper block and per\ncombination\nA, B, C, A+B,\nA+C, A+B+C\n\n"
        "DeLong against A", GREEN, bold=True)
    for height in (0.855, 0.665, 0.465):
        axis.annotate("", xy=(0.84, 0.68), xytext=(0.80, height),
                      arrowprops=dict(arrowstyle="-|>", color="#444444", linewidth=0.9,
                                      connectionstyle="arc3,rad=0.12"))

    box(axis, 0.84, 0.28, 0.15, 0.20,
        f"{args.group} against control\nAUC, 5-fold\nrepeated, seed averaged\ntest time augmentation",
        SAND)
    arrow(axis, (0.915, 0.55), (0.915, 0.48))

    box(axis, 0.20, 0.06, 0.60, 0.13,
        "Fold safety: the split is made first; cropping, pretraining, PCA, the orthogonalisation of C, "
        "the feature screen\nand every classifier are fitted on the training patients only, so no test "
        "patient is seen before it is scored.", GREY, size=8.5)
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
