#!/usr/bin/env python3
"""Overview figures for the manuscript: federated training, experiment design,
and the evaluation pipeline. Publication style: white background, no clutter."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# site colours, kept identical to every other figure in the paper
C_OS = "#1f77b4"   # Ostergotland (Linkoping)
C_ST = "#ff7f0e"   # Stockholm
C_TRO = "#2ca02c"  # Tromso
C_TUR = "#d62728"  # Turku
C_SERVER = "#4a4a4a"
C_BOX = "#f2f6fa"
C_EDGE = "#2c3e50"


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_EDGE, fs=9, weight="normal",
        tc="black", r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.005,rounding_size={r}",
                                linewidth=1.3, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=C_EDGE, style="-|>", lw=1.4, ls="-",
          rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, linewidth=lw, color=color,
                                 linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def blank_ax(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- Figure 1
def fig_training(path="fig1_federated_training.png"):
    """Principle of one federated round + what is trained."""
    fig, ax = blank_ax((11, 6.4))
    ax.set_title("Federated training of the clinical language model\n"
                 "one communication round; raw clinical text never leaves a site",
                 fontsize=12.5, fontweight="bold", y=1.0)

    # numbered legend of the round
    box(ax, 0.05, 0.855, 0.90, 0.145, "", fc="#fafafa", ec="#cccccc")

    # server
    box(ax, 0.27, 0.700, 0.46, 0.115,
        "AGGREGATION SERVER   —   global model $w^{(t)}$",
        fc="#e8eef5", ec=C_SERVER, fs=10.5, weight="bold")

    sites = [("Östergötland\n(Linköping, SE)", C_OS),
             ("Stockholm\n(SE)", C_ST),
             ("Tromsø\n(NO)", C_TRO),
             ("Turku\n(FI)", C_TUR)]
    xs = [0.045, 0.283, 0.521, 0.759]
    for (label, colour), x in zip(sites, xs):
        cx = x + 0.098
        # (1) global model down, on the left of the column
        arrow(ax, cx - 0.030, 0.700, cx - 0.030, 0.545, color=C_SERVER)
        # (3) update back up, on the right of the column
        arrow(ax, cx + 0.030, 0.545, cx + 0.030, 0.700, color=colour)
        box(ax, x, 0.385, 0.196, 0.16, label, fc="white", ec=colour, fs=9.5,
            weight="bold", tc=colour)
        box(ax, x, 0.215, 0.196, 0.125,
            "local clinical text\n(never transferred)", fc="#fbfbfb",
            ec="#aaaaaa", fs=8.5)
        arrow(ax, cx, 0.340, cx, 0.385, color="#888888")

    labels = [("1", 0.961, C_SERVER,
               "global model sent to every participating site"),
              ("2", 0.927, "#333333",
               "one local epoch of masked-language-model training "
               "(only LoRA adapter weights are updated)"),
              ("3", 0.893, "#333333",
               "only the parameter update is returned — no text, no patient data")]
    for tag, yy, col, txt in labels:
        ax.text(0.068, yy, tag, fontsize=10.5, fontweight="bold", color=col,
                va="center")
        ax.text(0.088, yy, txt, fontsize=9, color=col, style="italic",
                va="center")

    # aggregation formula
    box(ax, 0.05, 0.035, 0.90, 0.125,
        "4   weighted aggregation into the next global model:      "
        r"$w^{(t+1)} = \sum_k \dfrac{n_k}{\sum_j n_j}\, w_k^{(t)}$"
        "      ($n_k$ = local iterations at site $k$)\n"
        "the round then repeats; the global model is kept at the round with the "
        "lowest validation loss",
        fc="#f7f4ec", ec="#b7a57a", fs=9.5)

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", path)
    plt.close(fig)





# ---------------------------------------------------------------- Figure 2
def fig_experiments(path="fig2_experiments.png"):
    """Which sites took part in which experiment."""
    fig, ax = blank_ax((13, 4.9))
    ax.set_title("Experimental design: four federations and one single-site reference model",
                 fontsize=12.5, fontweight="bold", y=1.0)

    box(ax, 0.015, 0.68, 0.775, 0.245, "", fc="#f4f4f4", ec="#bbbbbb")
    ax.text(0.40, 0.955, "FEDERATED EXPERIMENTS", fontsize=10,
            fontweight="bold", ha="center", color="#444444")
    box(ax, 0.810, 0.68, 0.175, 0.245, "", fc="#f4f4f4", ec="#bbbbbb")
    ax.text(0.8975, 0.955, "LOCAL EXPERIMENT", fontsize=10,
            fontweight="bold", ha="center", color="#444444")

    runs = [
        (0.030, "Östergötland\n+ Stockholm", "StÖs", "model_8",
         "2 sites, 50 rounds"),
        (0.215, "Östergötland\n+ Tromsø", "ÖsTro", "model_10",
         "2 sites, 50 rounds"),
        (0.400, "Östergötland\n+ Stockholm + Tromsø", "StÖsTro", "model_19",
         "3 sites, 50 rounds"),
        (0.585, "Östergötland + Stockholm\n+ Tromsø + Turku", "StÖsTroTur",
         "model_99", "4 sites, 100 rounds"),
        (0.825, "Östergötland\nonly", "Ös", "local, epoch 100",
         "1 site, no federation"),
    ]
    for x, members, name, ckpt, note in runs:
        fed = name != "Ös"
        w = 0.145
        fc = "#e9f5e9" if fed else "#e4f2fb"
        ec = "#2ca02c" if fed else C_OS
        box(ax, x, 0.700, w, 0.175, members, fc=fc, ec=ec, fs=7.8)
        arrow(ax, x + w / 2, 0.700, x + w / 2, 0.555, color="#e0952a", lw=2)
        box(ax, x, 0.390, w, 0.16, f"{name}\n({ckpt})\n{note}",
            fc=fc, ec=ec, fs=9, weight="bold")
        arrow(ax, x + w / 2, 0.390, x + w / 2, 0.265, color="#e0952a", lw=2)
        box(ax, x, 0.095, w, 0.165,
            "word-vector\ndatabase\n$\\downarrow$\nranking performance",
            fc="white", ec=ec, fs=8)

    ax.text(0.5, 0.015,
            "All five models are evaluated with the identical glossary, stop list, "
            "candidate vocabulary and the same 100 fixed splits, so that only the "
            "training data differ.",
            fontsize=8.8, ha="center", style="italic", color="#333333")

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", path)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig_evaluation(path="fig3_evaluation_pipeline.png"):
    """From a trained checkpoint to the reported metrics."""
    fig, ax = blank_ax((11.5, 6.6))
    ax.set_title("Evaluation pipeline: from a trained checkpoint to ranking performance",
                 fontsize=12.5, fontweight="bold", y=1.0)

    steps = [
        ("STEP 1  Word representations",
         "the frozen checkpoint is applied to the clinical text; every token\n"
         "occurrence yields a 768-dimensional contextual vector",
         "#eaf3fb", C_OS),
        ("STEP 2  Aggregation to word types",
         "sub-word vectors are pooled per occurrence and averaged over all\n"
         "occurrences of a word:   " r"$e(w)=\frac{1}{n_w}\sum_i h(w_i)$" "   (L2-normalised)",
         "#eaf3fb", C_OS),
        ("STEP 3  Candidate scoring",
         "cosine similarity of every candidate word to the reference implants;\n"
         "eleven scoring methods (neighbourhood, similarity, lexical, PU learning)",
         "#f0f7ea", "#2ca02c"),
        ("STEP 4  Evaluation on fixed splits",
         "100 pre-computed splits of the glossary (reference / development / held-out test);\n"
         "ROC AUC, average precision, and precision, recall and F1 at depths K = 50/100/200",
         "#fdf1e8", "#e0952a"),
    ]
    y = 0.79
    for title, body, fc, ec in steps:
        box(ax, 0.06, y, 0.72, 0.155, "", fc=fc, ec=ec)
        ax.text(0.075, y + 0.115, title, fontsize=10, fontweight="bold",
                color=ec, va="center")
        ax.text(0.075, y + 0.048, body, fontsize=8.8, va="center",
                linespacing=1.4)
        if y > 0.2:
            arrow(ax, 0.42, y, 0.42, y - 0.045, lw=1.8)
        y -= 0.20

    # right-hand annotations
    notes = [
        (0.865, "one vector per\ntoken occurrence"),
        (0.660, "one vector per\nvocabulary word\n(268,242 terms)"),
        (0.455, "one score per\ncandidate word"),
        (0.250, "mean ± SD over\nthe 100 splits"),
    ]
    for yy, txt in notes:
        ax.text(0.80, yy, txt, fontsize=8.3, va="center", color="#555555",
                style="italic")

    box(ax, 0.06, 0.03, 0.72, 0.135, "", fc="#f7f4ec", ec="#b7a57a")
    ax.text(0.42, 0.128,
            "Ranked candidate list  →  expert review  →  glossary expansion",
            fontsize=10, fontweight="bold", ha="center", va="center")
    ax.text(0.42, 0.072,
            "The pipeline prioritises terms for human review; it does not itself "
            "determine MR safety.",
            fontsize=8.8, ha="center", va="center", style="italic")
    arrow(ax, 0.42, 0.19, 0.42, 0.165, lw=1.8)

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    fig_training()
    fig_experiments()
    fig_evaluation()
