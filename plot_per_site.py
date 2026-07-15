#!/usr/bin/env python3
"""Multi-panel per-site validation metric vs round from parse_fl_log CSV.

Usage:
    python3 plot_per_site.py fl_all_full_metrics.csv --out fl_per_site_panels
"""
import argparse
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", default="fl_per_site_panels")
    ap.add_argument("--order", default="model_8,model_10,model_19,model_99")
    args = ap.parse_args()

    # data[model][series] = list of (round, value)
    data = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(args.csv_path)):
        try:
            rnd = int(float(r["round"]))
            val = float(r["validation_metric"])
        except (ValueError, KeyError):
            continue
        data[r["log_label"]][r["series"]].append((rnd, val))

    models = [m for m in args.order.split(",") if m in data]
    models += [m for m in data if m not in models]

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        for series in sorted(data[model]):
            pts = sorted(data[model][series])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if series == "global_best":
                ax.plot(xs, ys, "k-", lw=2.5, label="global (best)")
            else:
                ax.plot(xs, ys, marker="o", ms=3, label=series.replace("site:", ""))
        ax.set_title(model)
        ax.set_xlabel("Federated round")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Validation metric (FLARE key metric; higher = better)")
    fig.suptitle("Per-site validation metric per round (all models)")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(f"wrote {args.out}.png")


if __name__ == "__main__":
    main()
