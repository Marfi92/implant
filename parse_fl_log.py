"""Parse a NVIDIA FLARE server log.txt and plot per-site + global metrics per round.

Extracts, from the FLARE ScatterAndGather server log:
  - per-site validation metric   ("validation metric <x> from client <name>")
  - the federated round it belongs to ("... ACCEPTED by the aggregator at round <N>")
  - the aggregated/global best metric ("new best validation metric at round <N>: <x>")

FLARE logs the *key metric* negated (negate_key_metric=true), so the raw value is
negative; we plot it as-is (higher = better) and also plot the sign-flipped loss.

Usage (one model):
  python3 parse_fl_log.py /path/to/workspace/log.txt --label model_19 --out model_19

Usage (all 4 models on one figure):
  python3 parse_fl_log.py \
    /path/.../17dc75eb-.../workspace/log.txt=model_19 \
    /path/.../1ddd748b-.../workspace/log.txt=model_8 \
    /path/.../2f3ecdb8-.../workspace/log.txt=model_10 \
    /path/.../ca859b72-.../workspace/log.txt=model_99 \
    --out all_models
"""
import argparse
import re
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RE_METRIC = re.compile(r"validation metric\s+(-?\d+\.?\d*(?:e-?\d+)?)\s+from client\s+(\S+)")
RE_ACCEPT = re.compile(r"Contribution from\s+(\S+)\s+ACCEPTED by the aggregator at round\s+(\d+)")
RE_BEST = re.compile(r"new best validation metric at round\s+(\d+):\s+(-?\d+\.?\d*(?:e-?\d+)?)")
RE_ROUND_START = re.compile(r"Round\s+(\d+)\s+started")


def parse_log(path):
    """Return (per_site, global_best).

    per_site[site] = list of (round, metric)
    global_best     = list of (round, metric)
    """
    per_site = {}
    global_best = []
    cur_round = 0
    pending = []  # metrics seen since last accept, waiting for a round number

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_ROUND_START.search(line)
            if m:
                cur_round = int(m.group(1))

            m = RE_METRIC.search(line)
            if m:
                val = float(m.group(1))
                site = m.group(2)
                pending.append((site, val))
                continue

            m = RE_ACCEPT.search(line)
            if m:
                site = m.group(1)
                rnd = int(m.group(2))
                cur_round = rnd
                # attach the most recent pending metric for this site
                for i in range(len(pending) - 1, -1, -1):
                    if pending[i][0] == site:
                        s, v = pending.pop(i)
                        per_site.setdefault(s, []).append((rnd, v))
                        break
                continue

            m = RE_BEST.search(line)
            if m:
                global_best.append((int(m.group(1)), float(m.group(2))))

    for s in per_site:
        per_site[s].sort()
    global_best.sort()
    return per_site, global_best


def parse_arg(a):
    if "=" in a:
        path, label = a.rsplit("=", 1)
        return path, label
    return a, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", help="path[=label] to log.txt (one or more)")
    ap.add_argument("--label", default=None, help="label when a single log is given")
    ap.add_argument("--out", default="fl_log", help="output prefix for png/csv")
    args = ap.parse_args()

    entries = [parse_arg(a) for a in args.logs]
    if len(entries) == 1 and args.label:
        entries[0] = (entries[0][0], args.label)

    # ---- CSV dump of everything
    csv_path = f"{args.out}_metrics.csv"
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["log_label", "series", "round", "validation_metric"])
        for path, label in entries:
            label = label or path
            per_site, gbest = parse_log(path)
            for site, seq in per_site.items():
                for r, v in seq:
                    w.writerow([label, f"site:{site}", r, v])
            for r, v in gbest:
                w.writerow([label, "global_best", r, v])
    print("wrote", csv_path)

    # ---- Plot 1: per-site metric vs round (uses the log with the most rounds)
    best_entry = max(entries, key=lambda e: len(parse_log(e[0])[1]) or 0)
    per_site, gbest = parse_log(best_entry[0])
    lbl = best_entry[1] or best_entry[0]

    plt.figure(figsize=(10, 6))
    for site, seq in sorted(per_site.items()):
        xs = [r for r, _ in seq]
        ys = [v for _, v in seq]
        plt.plot(xs, ys, marker="o", ms=3, lw=1, label=f"site: {site}")
    if gbest:
        plt.plot([r for r, _ in gbest], [v for _, v in gbest],
                 color="black", lw=2.2, label="global (aggregated best)")
    plt.xlabel("Federated round")
    plt.ylabel("Validation metric (FLARE key metric; higher = better)")
    plt.title(f"Per-site and global validation metric per round\n{lbl}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p1 = f"{args.out}_per_site.png"
    plt.savefig(p1, dpi=150)
    print("wrote", p1)

    # ---- Plot 2: global metric as loss (sign-flipped) for all logs
    plt.figure(figsize=(10, 6))
    for path, label in entries:
        label = label or path
        _, gb = parse_log(path)
        if not gb:
            continue
        xs = [r for r, _ in gb]
        ys = [-v for _, v in gb]  # flip: -metric = loss-like (lower = better)
        plt.plot(xs, ys, marker="o", ms=3, lw=1.4, label=label)
    plt.xlabel("Federated round")
    plt.ylabel("Loss (= -validation metric; lower = better)")
    plt.title("Global model convergence across rounds")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p2 = f"{args.out}_global_loss.png"
    plt.savefig(p2, dpi=150)
    print("wrote", p2)


if __name__ == "__main__":
    main()
