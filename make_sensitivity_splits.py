"""Build split files for the glossary sensitivity analyses (reviewer comment 3).

Uses the same 80/10/10 proportions, 100 splits and seed as the official splits.

  verified : only expert-verified glossary terms are positives. The unverified
             glossary terms are written to an exclusion list; append it to the
             stop list so they are neither positives nor negatives.
  grouped  : spelling variants of the same implant (e.g. stent, stents, stentet)
             are kept together, so a variant never sits in ref while another is
             in test. Terms are grouped when they match after lower-casing and
             removing - and space, or differ only by an inflection ending.

  python make_sensitivity_splits.py --official implant_split_official.json \
      --verified verified_terms.txt --stop_list stop_list_freq_1.txt --outdir sens
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SUFFIXES = ['s', 'n', 't', 'a', 'e', 'r', 'ar', 'er', 'or', 'en', 'et', 'na', 'ne', 'es', 'ns',
            'ts', 'arna', 'erna', 'orna', 'ena', 'ene', 'ane', 'ens', 'ets', 'ers', 'ars']


def load_list(path):
    with open(path, encoding='utf-8') as f:
        return [ln.strip().split()[0] for ln in f if ln.strip()]


def norm(w):
    return w.lower().replace('-', '').replace(' ', '')


def variant_groups(terms):
    """Union terms that are equal after normalisation, or where one is the
    other plus an inflection ending (stent/stents/stentet, protes/proteser)."""
    parent = {t: t for t in terms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_norm = defaultdict(list)
    for t in terms:
        by_norm[norm(t)].append(t)
    for n, ts in by_norm.items():
        for t in ts[1:]:
            parent[find(t)] = find(ts[0])
        if len(n) < 4:
            continue
        for suf in SUFFIXES:
            for t2 in by_norm.get(n + suf, []):
                parent[find(t2)] = find(ts[0])
    groups = defaultdict(list)
    for t in terms:
        groups[find(t)].append(t)
    return groups


def term_splits(terms, n_splits, seed):
    random.seed(seed)
    out = {}
    for i in range(n_splits):
        items = sorted(terms)
        random.shuffle(items)
        n = len(items)
        n_ref, n_dev = int(n * 0.8), int(n * 0.1)
        out[str(i)] = {'ref': items[:n_ref], 'dev': items[n_ref:n_ref + n_dev],
                       'test': items[n_ref + n_dev:]}
    return out


def group_splits(terms, n_splits, seed):
    groups = variant_groups(terms)
    keys = sorted(groups)
    n = len(terms)
    random.seed(seed)
    out = {}
    for i in range(n_splits):
        random.shuffle(keys)
        test, dev, ref = [], [], []
        for k in keys:
            if len(test) < n * 0.1:
                test += groups[k]
            elif len(dev) < n * 0.1:
                dev += groups[k]
            else:
                ref += groups[k]
        out[str(i)] = {'ref': ref, 'dev': dev, 'test': test}
    return out, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--official', required=True)
    ap.add_argument('--verified', required=True, help='expert-verified implant terms, one per line')
    ap.add_argument('--stop_list', required=True)
    ap.add_argument('--outdir', default='sens')
    ap.add_argument('--n-splits', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    off = json.load(open(args.official))
    first = next(iter(off.values()))
    universe = set(first['ref']) | set(first['dev']) | set(first['test'])
    verified = set(load_list(args.verified))
    ver = sorted(universe & verified)
    unver = sorted(universe - verified)
    print(f"glossary terms in official splits: {len(universe)}")
    print(f"  expert-verified: {len(ver)}   not verified: {len(unver)}")
    print(f"  verified-list terms not in the official splits: {len(verified - universe)}")

    json.dump(term_splits(ver, args.n_splits, args.seed),
              open(out / 'splits_verified.json', 'w'), ensure_ascii=False)
    stop = load_list(args.stop_list)
    with open(out / 'stop_plus_unverified.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(stop + unver) + '\n')

    gs, groups = group_splits(sorted(universe), args.n_splits, args.seed)
    json.dump(gs, open(out / 'splits_grouped.json', 'w'), ensure_ascii=False)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"grouped: {len(universe)} terms in {len(groups)} groups; "
          f"{len(multi)} groups with >1 variant ({sum(map(len, multi.values()))} terms)")
    with open(out / 'variant_groups.txt', 'w', encoding='utf-8') as f:
        for k in sorted(multi):
            f.write(f"{k}\t{', '.join(sorted(multi[k]))}\n")
    s0 = gs['0']
    print(f"  split 0 sizes ref/dev/test: {len(s0['ref'])}/{len(s0['dev'])}/{len(s0['test'])}")
    print(f"files written to {out}/ (check variant_groups.txt by eye)")


if __name__ == '__main__':
    main()
