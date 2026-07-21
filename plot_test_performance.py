#!/usr/bin/env python3
"""Plot MLM test/dev perplexity vs federated round from test_performance.csv files.

Usage:
    python3 plot_test_performance.py path1.csv=model_8 path2.csv=model_10 ... --out fl_perplexity
"""
import argparse
import csv
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rounds, dev, test = [], {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            name = r["model_checkpoint"]
            m = re.search(r"FL_global_model_(\d+)\.pt", name)
            if not m:
                continue  # skip best_FL_global_model.pt
            rnd = int(m.group(1))
            dev[rnd] = float(r["dev_perplexity"])
            test[rnd] = float(r["test_perplexity"])
    xs = sorted(dev)
    return xs, [dev[x] for x in xs], [test[x] for x in xs]


# raw label -> model name by participating sites (A=Abragam, B=DSV, C=nse, D=utu)
MODEL_NAME = {
    "model_8":  "Model AB",
    "model_10": "Model AC",
    "model_19": "Model ABC",
    "model_99": "Model ABCD",
}
# fixed colour per model so it is consistent across all figures
MODEL_COLOR = {
    "Model AB":   "#1f77b4",  # blue
    "Model AC":   "#ff7f0e",  # orange
    "Model ABC":  "#2ca02c",  # green
    "Model ABCD": "#d62728",  # red
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="path.csv=label")
    ap.add_argument("--out", default="fl_perplexity")
    args = ap.parse_args()

    runs = []
    for item in args.inputs:
        path, _, label = item.partition("=")
        label = label or path
        label = MODEL_NAME.get(label, label)
        xs, dev, test = load(path)
        runs.append((label, xs, dev, test))
        print(f"{label}: {len(xs)} rounds (0..{max(xs)}), "
              f"final test PPL={test[-1]:.3f}, best test PPL={min(test):.3f}")

    # combined test-perplexity convergence, with each model's BEST (saved) round marked
    plt.figure(figsize=(11, 6.5))
    for label, xs, dev, test in runs:
        colour = MODEL_COLOR.get(label)
        # best = lowest test perplexity = the checkpoint FLARE actually saves
        bi = min(range(len(test)), key=lambda i: test[i])
        best_r, best_p = xs[bi], test[bi]
        plt.plot(xs, test, marker="o", ms=3, color=colour,
                 label=f"{label}  (best PPL={best_p:.2f} @ round {best_r}, "
                       f"trained {max(xs)} rounds)")
        # star + annotation on the best point = the saved checkpoint
        plt.plot(best_r, best_p, marker="*", ms=18, color=colour,
                 markeredgecolor="black", zorder=5)
        plt.annotate(f"{label.split()[-1]} best\nround {best_r}",
                     (best_r, best_p), textcoords="offset points",
                     xytext=(0, -32), ha="center", fontsize=8, color=colour)
    plt.xlabel("Federated round")
    plt.ylabel("Test perplexity (lower = better)")
    plt.title("Federated MLM convergence: test perplexity per round\n"
              "★ = best round = the checkpoint that is saved and used "
              "(lines end at each model's trained rounds: 50 or 100)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{args.out}_test.png", dpi=150)
    print(f"wrote {args.out}_test.png")

    # per-model dev vs test panels
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (label, xs, dev, test) in zip(axes, runs):
        ax.plot(xs, dev, marker="o", ms=3, label="dev")
        ax.plot(xs, test, marker="o", ms=3, label="test")
        ax.set_title(label)
        ax.set_xlabel("Federated round")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Perplexity (lower = better)")
    fig.suptitle("Dev vs test perplexity per round (all models)")
    fig.tight_layout()
    fig.savefig(f"{args.out}_dev_test.png", dpi=150)
    print(f"wrote {args.out}_dev_test.png")


if __name__ == "__main__":
    main()
