"""Paired comparison of two models over the same splits (reviewer comment 2).

Reads two --per-split-out CSVs from compare_all.py and reports, per method and
metric, the mean difference (A - B), its SD, a 95% CI and the Wilcoxon
signed-rank p-value over the splits both runs share.

  python paired_diff.py per_split_StLiTro_nolex.csv per_split_Li_nolex.csv \
      --names StLiTro Li --methods pu_lr mindist_all centroid
"""
import argparse
import csv
import math
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon


def load(path):
    d = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r['method']][r['split']] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--names', nargs=2, default=['A', 'B'])
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--metrics', nargs='*',
                    default=['roc_auc', 'average_precision', 'precision_at_50', 'precision_at_100'])
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    methods = args.methods or sorted(set(A) & set(B))
    na, nb = args.names
    print(f"{'method':<13}{'metric':<19}{na:>9}{nb:>9}{'diff':>9}{'sd':>8}{'95% CI':>19}{'p':>10}{'n':>5}")
    for m in methods:
        common = sorted(set(A[m]) & set(B[m]), key=str)
        for mt in args.metrics:
            a = np.array([float(A[m][s][mt]) for s in common])
            b = np.array([float(B[m][s][mt]) for s in common])
            d = a - b
            n = len(d)
            se = d.std(ddof=1) / math.sqrt(n)
            lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            print(f"{m:<13}{mt:<19}{a.mean():>9.4f}{b.mean():>9.4f}{d.mean():>+9.4f}{d.std(ddof=1):>8.4f}"
                  f"   [{lo:+.4f}, {hi:+.4f}]{p:>10.3g}{n:>5}")


if __name__ == '__main__':
    main()
