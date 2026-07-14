"""Extract per-round, per-site FL validation metrics for ALL models at once,
write one combined CSV, and produce combined plots.

Point it at the results_nov12_2025 folder; it maps the known job UUIDs to model
names automatically. Edit MODELS below if the UUIDs/paths differ.

Usage:
  python3 extract_all_models.py /home/abragam23/federatedhealth_20250617/results_nov12_2025
  # or override the output prefix:
  python3 extract_all_models.py <results_root> --out fl_all
"""
import argparse
import os
import re
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# UUID -> model label (the checkpoint each run corresponds to)
MODELS = {
    "1ddd748b-3d5d-4a66-80e3-685f0f5d04f2": "model_8",
    "2f3ecdb8-c55a-46e1-9853-d043448e8d25": "model_10",
    "17dc75eb-6f4c-466b-92bc-60882b73c01c": "model_19",
    "ca859b72-ee44-4eca-a823-5a82191fd7dc": "model_99",
}

RE_METRIC = re.compile(r"validation metric\s+(-?\d+\.?\d*(?:e-?\d+)?)\s+from client\s+(\S+)")
RE_ACCEPT = re.compile(r"Contribution from\s+(\S+)\s+ACCEPTED by the aggregator at round\s+(\d+)")
RE_BEST = re.compile(r"new best validation metric at round\s+(\d+):\s+(-?\d+\.?\d*(?:e-?\d+)?)")
RE_ROUND_START = re.compile(r"Round\s+(\d+)\s+started")


def parse_log(path):
    per_site = defaultdict(list)
    global_best = []
    cur_round = 0
    pending = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_ROUND_START.search(line)
            if m:
                cur_round = int(m.group(1))
            m = RE_METRIC.search(line)
            if m:
                pending.append((m.group(2), float(m.group(1))))
                continue
            m = RE_ACCEPT.search(line)
            if m:
                site, rnd = m.group(1), int(m.group(2))
                for i in range(len(pending) - 1, -1, -1):
                    if pending[i][0] == site:
                        s, v = pending.pop(i)
                        per_site[s].append((rnd, v))
                        break
                continue
            m = RE_BEST.search(line)
            if m:
                global_best.append((int(m.group(1)), float(m.group(2))))
    for s in per_site:
        per_site[s].sort()
    global_best.sort()
    return dict(per_site), global_best


def find_log(root, uuid):
    for cand in (os.path.join(root, uuid, "workspace", "log.txt"),
                 os.path.join(root, uuid, "log.txt")):
        if os.path.isfile(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_root", help="path to results_nov12_2025")
    ap.add_argument("--out", default="fl_all")
    args = ap.parse_args()

    data = {}  # label -> (per_site, global_best)
    for uuid, label in MODELS.items():
        path = find_log(args.results_root, uuid)
        if not path:
            print(f"  WARNING: no log.txt for {label} ({uuid})")
            continue
        data[label] = parse_log(path)
        ps, gb = data[label]
        print(f"{label}: sites={list(ps)} rounds<= {max((r for s in ps.values() for r,_ in s), default=0)} "
              f"global_pts={len(gb)}")

    # ---- combined CSV
    csv_path = f"{args.out}_metrics.csv"
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["model", "series", "round", "validation_metric"])
        for label, (ps, gb) in data.items():
            for site, seq in ps.items():
                for r, v in seq:
                    w.writerow([label, f"site:{site}", r, v])
            for r, v in gb:
                w.writerow([label, "global_best", r, v])
    print("wrote", csv_path)

    # ---- one per-site figure per model (subplots)
    n = len(data)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True, squeeze=False)
        for ax, (label, (ps, gb)) in zip(axes[0], data.items()):
            for site, seq in sorted(ps.items()):
                ax.plot([r for r, _ in seq], [v for _, v in seq], marker="o", ms=3, lw=1, label=site)
            if gb:
                ax.plot([r for r, _ in gb], [v for _, v in gb], color="black", lw=2, label="global")
            ax.set_title(label)
            ax.set_xlabel("Federated round")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("Validation metric (higher = better)")
        axes[0][-1].legend(fontsize=8)
        fig.suptitle("Per-site validation metric per round (all models)")
        fig.tight_layout()
        p1 = f"{args.out}_per_site.png"
        fig.savefig(p1, dpi=150)
        print("wrote", p1)

    # ---- combined global convergence (all models, as loss)
    plt.figure(figsize=(10, 6))
    for label, (ps, gb) in data.items():
        if gb:
            plt.plot([r for r, _ in gb], [-v for _, v in gb], marker="o", ms=3, lw=1.4, label=label)
    plt.xlabel("Federated round")
    plt.ylabel("Loss (= -validation metric; lower = better)")
    plt.title("Global model convergence across rounds (all models)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p2 = f"{args.out}_global_loss.png"
    plt.savefig(p2, dpi=150)
    print("wrote", p2)


if __name__ == "__main__":
    main()
