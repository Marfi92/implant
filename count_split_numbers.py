"""Count the numbers needed for reviewer comment 1. No model is trained.

Uses exactly the same inputs and the same label/negative logic as compare_all.py:
  positives  = held-out `test` glossary terms that are query words AND have a vector
  ranked     = positives + negative pool minus the classifier-training negatives
               (random pu_lr negatives, seed 0, and pu_rn reliable negatives)

Run with the SAME arguments you used for compare_all.py, e.g.
  python count_split_numbers.py /path/lancedb_direct \
      --store /path/neighbour_raw_X.h5 --splits-file official_splits.json \
      --stop_list stop.txt

Outputs: a summary on screen and split_counts.csv (one row per split).
"""
import argparse
import csv
import json
import numpy as np
import lancedb
import h5py


def load_lancedb_vectors(path, table):
    t = lancedb.connect(path).open_table(table).to_arrow()
    words = [str(w) for w in t['word'].to_pylist()]
    vec = t['vector'].combine_chunks()
    mat = np.asarray(vec.values, dtype=np.float32).reshape(len(words), -1)
    return words, mat


def l2norm(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def load_set(path):
    if not path:
        return set()
    with open(path, encoding='utf-8') as f:
        return {ln.split()[0] for ln in f if ln.strip()}


def mean_range(a):
    a = np.asarray(a, dtype=float)
    return f"mean {a.mean():.2f}  sd {a.std():.2f}  min {a.min():.0f}  max {a.max():.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lancedb')
    ap.add_argument('--table', default='words_aggregated')
    ap.add_argument('--store', required=True)
    ap.add_argument('--splits-file', required=True)
    ap.add_argument('--stop_list', default=None)
    ap.add_argument('--neg-ratio-train', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ks', default='50,100,200')
    ap.add_argument('--precision', default='0.1732,0.1469,0.1105',
                    help='saved pu_lr mean Precision@K for this model (default: StLiTro)')
    ap.add_argument('--out', default='split_counts.csv')
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(',')]
    prec = [float(x) for x in args.precision.split(',')]
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.store, 'r') as st:
        vocab = np.array([w.decode('utf-8') if isinstance(w, bytes) else str(w)
                          for w in st['vocab'][:]], dtype=object)
        qwords = vocab[st['query_word_ids'][:]]
    Nq = len(qwords)

    lwords, lmat = load_lancedb_vectors(args.lancedb, args.table)
    lmat = l2norm(lmat)
    word2vec = {w: i for i, w in enumerate(lwords)}
    qvec_idx = np.array([word2vec.get(w, -1) for w in qwords])
    have_vec = qvec_idx >= 0
    qvecs = np.zeros((Nq, lmat.shape[1]), dtype=np.float32)
    qvecs[have_vec] = lmat[qvec_idx[have_vec]]

    with open(args.splits_file) as f:
        splits = json.load(f)
    split_items = list(splits.items())
    stop = load_set(args.stop_list)
    first = split_items[0][1]
    implant_all = set(first['ref']) | set(first['dev']) | set(first['test'])
    qword_to_qidx = {w: i for i, w in enumerate(qwords)}
    qset = set(qwords)

    is_implant = np.array([w in implant_all for w in qwords], dtype=bool)
    is_stop = np.array([w in stop for w in qwords], dtype=bool)
    neg_pool = (~is_implant) & (~is_stop) & have_vec
    neg_pool_idx = np.flatnonzero(neg_pool)

    g_vec = {w for w in implant_all if w in word2vec}
    g_q = {w for w in implant_all if w in qset}
    g_usable = {w for w in g_q if have_vec[qword_to_qidx[w]]}

    print("=" * 70)
    print("GLOBAL COUNTS")
    print(f"  vocabulary words (lancedb, table {args.table}): {len(lwords)}")
    print(f"  query words (neighbour store):                  {Nq}")
    print(f"  query words with a vector:                      {int(have_vec.sum())}")
    print(f"  stop-list words among query words:              {int(is_stop.sum())}")
    print(f"  negative pool:                                  {len(neg_pool_idx)}")
    print(f"  glossary terms in the splits file:              {len(implant_all)}")
    print(f"  ... with a vector in lancedb:                   {len(g_vec)}")
    print(f"  ... that are query words:                       {len(g_q)}")
    print(f"  ... query word AND vector (usable positives):   {len(g_usable)}")
    print(f"  ... with a vector but NOT a query word:         {len(g_vec - qset)}")
    print(f"  usable positives / (usable + negative pool):    "
          f"{len(g_usable) / (len(g_usable) + len(neg_pool_idx)):.4%}")

    rows = []
    for sid, sp in split_items:
        ref, dev, test = sp['ref'], sp.get('dev', []), sp['test']
        ref_idx = [qword_to_qidx[x] for x in ref
                   if x in qword_to_qidx and have_vec[qword_to_qidx[x]]]
        if not ref_idx:
            continue
        centroid = qvecs[ref_idx].mean(axis=0)
        cn = np.linalg.norm(centroid)
        if cn == 0:
            continue
        csim = qvecs @ (centroid / cn)
        test_set = set(test)
        test_pos = np.array([w in test_set for w in qwords], dtype=bool) & have_vec
        if not test_pos.any():
            continue

        n_train_neg = min(len(neg_pool_idx), args.neg_ratio_train * len(ref_idx))
        train_neg = rng.choice(neg_pool_idx, size=n_train_neg, replace=False)
        rn_neg = neg_pool_idx[np.argsort(csim[neg_pool_idx])[:n_train_neg]]
        used = np.union1d(train_neg, rn_neg)
        n_eval_neg = len(neg_pool_idx) - len(used)

        P = int(test_pos.sum())
        N = n_eval_neg
        row = {'split': sid, 'ref': len(ref), 'dev': len(dev), 'test': len(test),
               'ref_usable': len(ref_idx), 'test_pos_P': P,
               'train_neg_used': len(used), 'ranked_neg_N': N, 'ranked_total': P + N,
               'prevalence': P / (P + N)}
        for k in ks:
            row[f'chance_at_{k}'] = k * P / (P + N)
        rows.append(row)

    with open(args.out, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("=" * 70)
    print(f"PER SPLIT ({len(rows)} splits; full table in {args.out})")
    for key in ['ref', 'dev', 'test', 'ref_usable', 'test_pos_P',
                'train_neg_used', 'ranked_neg_N', 'ranked_total']:
        print(f"  {key:<15} {mean_range([r[key] for r in rows])}")
    prev = np.mean([r['prevalence'] for r in rows])
    print(f"  prevalence      mean {prev:.5f} ({prev:.3%})")

    print("=" * 70)
    print("CHANCE AND ENRICHMENT (observed = saved pu_lr Precision@K x K)")
    for k, p in zip(ks, prec):
        ch = np.mean([r[f'chance_at_{k}'] for r in rows])
        obs = p * k
        print(f"  K={k:<4} chance {ch:.3f}   observed {obs:.2f}   enrichment {obs / ch:.0f}-fold")
    print("=" * 70)


if __name__ == '__main__':
    main()
