"""Centroid (prototype) similarity evaluation for implant-word detection.

Idea (unsupervised / PU-style, needs only the implant TERM LIST + word vectors):
  1. For each split, average the vectors of the `ref` implant words -> one
     "implant prototype" (centroid) vector.
  2. Score every word by cosine similarity to that centroid.
  3. Evaluate ranking on the held-out `test` implants vs the negatives
     (every word not in the implant list and not a stop word), using the SAME
     100 splits as the official method so the numbers are comparable.

No annotated text and no clean negatives are required: the implant term list is
the positive label set; the centroid is built only from the `ref` portion, so
the held-out `test` implants are never seen when forming the prototype.
"""
import argparse
import json
import numpy as np
import lancedb
from sklearn.metrics import roc_auc_score, average_precision_score


def load_vectors(lancedb_path, table_name):
    db = lancedb.connect(lancedb_path)
    try:
        t = db.open_table(table_name)
    except Exception as exc:
        raise SystemExit(f"could not open table '{table_name}': {exc}")
    tbl = t.to_arrow()
    words = [str(w) for w in tbl['word'].to_pylist()]
    vec_chunked = tbl['vector'].combine_chunks()
    # FixedSizeListArray -> flat child values, then reshape to (n, dim)
    flat = np.asarray(vec_chunked.values, dtype=np.float32)
    n = len(words)
    dim = flat.size // n
    vecs = flat.reshape(n, dim)
    return words, vecs


def l2_normalize(mat):
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return mat / norm


def precision_at_k(y_sorted, k):
    k = min(k, len(y_sorted))
    if k == 0:
        return 0.0
    return float(y_sorted[:k].sum()) / k


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('lancedb', help='path to the lancedb_direct folder (holds the word vectors)')
    ap.add_argument('--table', default='words_aggregated')
    ap.add_argument('--splits-file', required=True)
    ap.add_argument('--stop_list', dest='stop_list', default=None)
    ap.add_argument('--restrict-words', dest='restrict_words', default=None,
                    help='optional file (one word per line) limiting the scored '
                         'vocabulary to these words -- use the query-words list '
                         'from the vote method to make precision@k comparable.')
    ap.add_argument('--ks', default='50,100,200', help='comma-separated k for precision@k')
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(',')]

    print("Loading word vectors from lancedb ...")
    words, vecs = load_vectors(args.lancedb, args.table)
    vecs = l2_normalize(vecs)
    print(f"  {len(words)} words, dim={vecs.shape[1]}")

    word_to_idx = {}
    for i, w in enumerate(words):
        if w not in word_to_idx:        # keep first occurrence if duplicates
            word_to_idx[w] = i

    stop = set()
    if args.stop_list:
        with open(args.stop_list, encoding='utf-8') as f:
            stop = {ln.strip() for ln in f if ln.strip()}
    print(f"  {len(stop)} stop words")

    with open(args.splits_file, encoding='utf-8') as f:
        splits = json.load(f)
    if isinstance(splits, dict):
        split_items = list(splits.items())
    else:
        split_items = [(str(i), s) for i, s in enumerate(splits)]
    print(f"  {len(split_items)} splits")

    # master implant set = union of ref/dev/test of the first split (same list every split)
    first = split_items[0][1]
    implant_all = set(first['ref']) | set(first['dev']) | set(first['test'])

    restrict = None
    if args.restrict_words:
        with open(args.restrict_words, encoding='utf-8') as f:
            restrict = {ln.split()[0] for ln in f if ln.strip()}
        print(f"  restricting scored vocabulary to {len(restrict)} words from {args.restrict_words}")

    is_implant = np.array([w in implant_all for w in words], dtype=bool)
    is_stop = np.array([w in stop for w in words], dtype=bool)
    neg_mask = (~is_implant) & (~is_stop)     # constant across splits
    if restrict is not None:
        in_restrict = np.array([w in restrict for w in words], dtype=bool)
        neg_mask = neg_mask & in_restrict     # negatives limited to the restrict vocab
    print(f"  negatives (non-implant, non-stop): {int(neg_mask.sum())}")

    metric_names = ['roc_auc', 'average_precision'] + [f'precision_at_{k}' for k in ks]
    rows = []

    for sid, sp in split_items:
        ref, dev, test = sp['ref'], sp['dev'], sp['test']
        ref_idx = [word_to_idx[w] for w in ref if w in word_to_idx]
        test_idx = [word_to_idx[w] for w in test if w in word_to_idx]
        if not ref_idx or not test_idx:
            continue

        centroid = vecs[ref_idx].mean(axis=0)
        cn = np.linalg.norm(centroid)
        if cn == 0:
            continue
        centroid /= cn
        scores = vecs @ centroid          # cosine similarity, all words

        test_pos = np.zeros(len(words), dtype=bool)
        test_pos[test_idx] = True
        mask = test_pos | neg_mask
        y = test_pos[mask].astype(int)
        s = scores[mask]

        order = np.argsort(-s, kind='stable')
        y_sorted = y[order]

        row = {
            'split': sid,
            'roc_auc': roc_auc_score(y, s),
            'average_precision': average_precision_score(y, s),
            'n_test_pos': int(y.sum()),
            'n_neg': int((y == 0).sum()),
        }
        for k in ks:
            row[f'precision_at_{k}'] = precision_at_k(y_sorted, k)
        rows.append(row)

    if not rows:
        raise SystemExit("no usable splits")

    print("\n" + "=" * 64)
    print("AGGREGATED RESULTS ACROSS ALL SPLITS (centroid similarity)")
    print("=" * 64)
    for m in metric_names:
        vals = np.array([r[m] for r in rows], dtype=float)
        mean = vals.mean()
        std = vals.std()
        ci = 1.96 * std / np.sqrt(len(vals))
        print(f"{m:>20}: mean={mean:.4f} \u00b1 {std:.4f}  "
              f"median={np.median(vals):.4f}  min={vals.min():.4f}  max={vals.max():.4f}  "
              f"95% CI=[{mean-ci:.4f}, {mean+ci:.4f}]")
    print("=" * 64)
    print(f"splits used: {len(rows)}   "
          f"test positives/ split ~{int(np.mean([r['n_test_pos'] for r in rows]))}   "
          f"negatives ~{int(np.mean([r['n_neg'] for r in rows]))}")


if __name__ == '__main__':
    main()
