#!/usr/bin/env python3
"""Report the best (saved) checkpoint of one or more federated runs.

Point it at a run directory (the UUID folder) or at the parent results
directory and it prints, per run: the participating sites, the configured
rounds, the round FLARE recorded as its last new best validation metric,
the round with the lowest dev/test perplexity, and the checkpoint file to use.

Usage:
    python3 find_best_checkpoint.py /path/to/results_dir
    python3 find_best_checkpoint.py /path/to/results_dir/<uuid> [...]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

RE_BEST = re.compile(
    r"new best validation metric at round\s+(\d+):\s+(-?\d+\.?\d*(?:e-?\d+)?)"
)
RE_METRIC_CLIENT = re.compile(r"validation metric\s+-?\d+\.?\d*\S*\s+from client\s+(\S+)")
RE_CKPT_ROUND = re.compile(r"FL_global_model_(\d+)\.pt")


def find_runs(paths):
    """A run directory is one that contains a workspace/ subdirectory."""
    runs = []
    for p in paths:
        p = Path(p)
        if (p / "workspace").is_dir():
            runs.append(p)
            continue
        runs.extend(sorted(c for c in p.iterdir()
                           if c.is_dir() and (c / "workspace").is_dir()))
    return runs


def read_log(run):
    """Return (last_best_round, last_best_metric, sites) from workspace/log.txt."""
    log = run / "workspace" / "log.txt"
    if not log.is_file():
        return None, None, []
    best_round, best_metric, sites = None, None, []
    with open(log, errors="ignore") as f:
        for line in f:
            m = RE_BEST.search(line)
            if m:
                best_round, best_metric = int(m.group(1)), float(m.group(2))
            c = RE_METRIC_CLIENT.search(line)
            if c and c.group(1) not in sites:
                sites.append(c.group(1))
    return best_round, best_metric, sites


def read_config(run):
    """Pull num_rounds / min_clients out of any config JSON under the workspace."""
    out = {}
    ws = run / "workspace"
    for jf in sorted(ws.rglob("*config*.json")) + sorted(ws.rglob("meta.json")):
        try:
            data = json.load(open(jf))
        except Exception:
            continue
        stack = [data]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for k, v in cur.items():
                    if k in ("num_rounds", "n_rounds") and not isinstance(v, (dict, list)):
                        out.setdefault("num_rounds", v)
                    if k in ("min_clients", "min_num_clients") and not isinstance(v, (dict, list)):
                        out.setdefault("min_clients", v)
                    stack.append(v)
            elif isinstance(cur, list):
                stack.extend(cur)
    return out


def read_perplexity(run):
    """Return [(round, dev_ppl, test_ppl)] from any test_performance.csv in the run."""
    rows = []
    for csv_path in sorted(run.rglob("test_performance.csv")):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                m = RE_CKPT_ROUND.search(r.get("model_checkpoint", ""))
                if not m:
                    continue  # skip best_FL_global_model.pt
                try:
                    rows.append((int(m.group(1)),
                                 float(r["dev_perplexity"]),
                                 float(r["test_perplexity"])))
                except (KeyError, ValueError):
                    continue
        if rows:
            break
    return sorted(rows)


def report(run):
    print(f"\n=== {run.name} ===")
    print(f"  path                : {run}")

    cfg = read_config(run)
    best_round, best_metric, sites = read_log(run)

    if sites:
        print(f"  sites               : {', '.join(sites)} ({len(sites)} sites)")
    if "num_rounds" in cfg:
        print(f"  configured rounds   : {cfg['num_rounds']}")
    if "min_clients" in cfg:
        print(f"  min clients / round : {cfg['min_clients']}")

    if best_round is not None:
        # FLARE negates the key metric, so validation loss = -metric
        print(f"  FLARE last new best : round {best_round} "
              f"(key metric {best_metric:.4f}, validation loss {-best_metric:.4f})")
    else:
        print("  FLARE last new best : not found in log.txt")

    ppl = read_perplexity(run)
    if ppl:
        rounds = [r for r, _, _ in ppl]
        b_test = min(ppl, key=lambda t: t[2])
        b_dev = min(ppl, key=lambda t: t[1])
        print(f"  evaluated rounds    : {min(rounds)}..{max(rounds)} ({len(rounds)} checkpoints)")
        print(f"  BEST test perplexity: {b_test[2]:.4f} at round {b_test[0]}")
        print(f"  best dev perplexity : {b_dev[1]:.4f} at round {b_dev[0]}")
        print(f"  final test perplexity: {ppl[-1][2]:.4f} at round {ppl[-1][0]}")
        if b_test[0] < max(rounds) * 0.5:
            print("  note                : best round is early and loss rises "
                  "afterwards -> client drift / heterogeneous sites")
        print(f"  >>> USE CHECKPOINT  : FL_global_model_{b_test[0]}.pt")
        target = f"FL_global_model_{b_test[0]}.pt"
    else:
        print("  perplexity          : no test_performance.csv found")
        target = (f"FL_global_model_{best_round}.pt"
                  if best_round is not None else None)
        if target:
            print(f"  >>> USE CHECKPOINT  : {target} (from FLARE best, no perplexity file)")

    found = sorted(run.rglob("FL_global_model_*.pt"))
    if found:
        print(f"  checkpoint files    : {len(found)} found, e.g. {found[0].parent}")
        if target and not any(p.name == target for p in found):
            print(f"  WARNING             : {target} is not on disk "
                  "(only the best/last checkpoints may have been kept)")
    best_pt = sorted(run.rglob("best_FL_global_model.pt"))
    if best_pt:
        print(f"  FLARE best_*.pt     : {best_pt[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="run directory (UUID folder) or a results directory")
    args = ap.parse_args()

    runs = find_runs(args.paths)
    if not runs:
        sys.exit("no run directories found (expected a folder containing workspace/)")
    for run in runs:
        report(run)
    print(f"\n{len(runs)} run(s) reported. "
          "Rebuild the vector database from the chosen checkpoint before "
          "running the ranking evaluation.")


if __name__ == "__main__":
    main()
