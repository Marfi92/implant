#!/usr/bin/env python3
"""Multi-panel per-site validation LOSS vs round from parse_fl_log CSV.

Boss-requested conventions:
  * fixed colour per site across every panel (nse always green, etc.)
  * sites renamed A/B/C/D; models named by their participating sites (AB, AC, ...)
  * plotted as loss (= -validation_metric) so lower = better (not upside-down),
    consistent with the perplexity figures
  * black line = average across that model's sites per round (spans all rounds)

Usage:
    python3 plot_per_site.py fl_all_full_metrics.csv --out fl_per_site_panels
"""
import argparse
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# site -> (letter, fixed colour)
SITE = {
    "CCO-Abragam": ("A", "#1f77b4"),  # blue
    "DSV":         ("B", "#ff7f0e"),  # orange
    "nse":         ("C", "#2ca02c"),  # green
    "utu":         ("D", "#d62728"),  # red
}

# raw model label -> pretty name by participating sites
MODEL_NAME = {
    "model_8":  "Model AB",
    "model_10": "Model AC",
    "model_19": "Model ABC",
    "model_99": "Model ABCD",
}


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
        # collect per-site sequences (as loss = -metric)
        site_round_val = defaultdict(dict)  # site -> {round: loss}
        for series, pts in data[model].items():
            if series == "global_best":
                continue
            site = series.replace("site:", "")
            for rnd, v in pts:
                site_round_val[site][rnd] = -v  # flip sign -> loss (lower = better)

        # plot each site with its fixed colour/letter
        for site in sorted(site_round_val,
                           key=lambda s: SITE.get(s, ("Z", "#000"))[0]):
            letter, colour = SITE.get(site, (site, None))
            pts = sorted(site_round_val[site].items())
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", ms=3, lw=1, color=colour,
                    label=f"{letter} ({site})")

        # black line = average across sites per round (all rounds)
        all_rounds = sorted({r for d in site_round_val.values() for r in d})
        avg_xs, avg_ys = [], []
        for r in all_rounds:
            vals = [site_round_val[s][r] for s in site_round_val if r in site_round_val[s]]
            if vals:
                avg_xs.append(r)
                avg_ys.append(sum(vals) / len(vals))
        if avg_xs:
            ax.plot(avg_xs, avg_ys, "k-", lw=2.5, label="average (global)")

        ax.set_title(MODEL_NAME.get(model, model))
        ax.set_xlabel("Federated round")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Validation loss (= -key metric; lower = better)")
    fig.suptitle("Per-site validation loss per round (all models)")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(f"wrote {args.out}.png")


if __name__ == "__main__":
    main()
