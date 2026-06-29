"""Compare all implant-word scoring methods on ONE shared vocabulary / splits.

Runs every method on the SAME query-word universe (recovered from the neighbour
store) and the SAME 100 splits, so the numbers are directly comparable:

  vote      - distance-weighted count of `ref` implant neighbours (original)
  mindist   - similarity to the single nearest `ref` implant neighbour
  avgdist   - average similarity to the `ref` implant neighbours
  centroid  - cosine similarity to the mean vector of the `ref` implants
  hybrid    - z-normalised vote + z-normalised centroid (simple combination)
  pu_lr     - logistic-regression PU classifier that LEARNS to combine the
              features above (trained on glossary positives vs sampled unlabeled
              "negatives"; no annotated text required)

Inputs needed (all things you already have):
  - lancedb_direct (word vectors, table words_aggregated)
  - a neighbour_raw_*.h5 store (gives the vote vocabulary + neighbour lists)
  - the 100-split json and the stop list

Labels follow the official method: a query word is a positive iff it is a
held-out `test` implant; negatives are query words that are not in the implant
glossary and not stop words. The classifier is trained only on `ref` implants
and a disjoint sample of negatives, so held-out `test` implants are never seen.
"""
import argparse
import json
import numpy as np
import lancedb
import h5py
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def load_lancedb_vectors(path, table):
    db = lancedb.connect(path)
    t = db.open_table(table)
    tbl = t.to_arrow()
    words = [str(w) for w in tbl['word'].to_pylist()]
    vec = tbl['vector'].combine_chunks()
    flat = np.asarray(vec.values, dtype=np.float32)
    mat = flat.reshape(len(words), -1)
    return words, mat


def l2norm(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


def precision_at_k(y_sorted, k):
    k = min(k, len(y_sorted))
    if k == 0:
        return 0.0
    return float(y_sorted[:k].sum()) / k


def load_set(path):
    if not path:
        return set()
    with open(path, encoding='utf-8') as f:
        return {ln.split()[0] for ln in f if ln.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('lancedb', help='path to lancedb_direct')
    ap.add_argument('--table', default='words_aggregated')
    ap.add_argument('--store', required=True, help='neighbour_raw_*.h5')
    ap.add_argument('--splits-file', required=True)
    ap.add_argument('--stop_list', default=None)
    ap.add_argument('--ks', default='50,100,200')
    ap.add_argument('--neg-ratio-train', type=int, default=10,
                    help='negatives per positive used to TRAIN the classifier')
    ap.add_argument('--use-all-neg', action='store_true',
                    help='train on ALL pool negatives (no subsampling); makes '
                         'pu_rn identical to pu_lr')
    ap.add_argument('--no-class-weight', action='store_true',
                    help='do NOT use class_weight=balanced (use natural imbalance)')
    ap.add_argument('--eval-full-pool', action='store_true',
                    help='evaluate on the FULL negative pool (include words used '
                         'as training negatives) to match the standalone runs')
    ap.add_argument('--no-lexical', action='store_true',
                    help='skip the lexical (string-similarity) feature/method')
    ap.add_argument('--lex-chunk', type=int, default=20000,
                    help='row chunk size for the lexical similarity computation')
    ap.add_argument('--no-full-sim', action='store_true',
                    help='skip the full-embedding "distance to ALL implants" methods')
    ap.add_argument('--topk-all', type=int, default=10,
                    help='k for votetopk_all (sum of top-k implant similarities)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    use_lexical = not args.no_lexical
    use_full = not args.no_full_sim
    class_weight = None if args.no_class_weight else 'balanced'

    ks = [int(x) for x in args.ks.split(',')]
    rng = np.random.default_rng(args.seed)

    # ---- load neighbour store (defines the vote vocabulary + neighbour lists)
    print("Loading neighbour store ...")
    with h5py.File(args.store, 'r') as st:
        vocab = np.array([w.decode('utf-8') if isinstance(w, bytes) else str(w)
                          for w in st['vocab'][:]], dtype=object)
        query_ids = st['query_word_ids'][:]
        neigh_ids = st['neighbour_ids'][:]          # (Nq, L) ids into vocab
        neigh_sim = st['neighbour_distances'][:].astype(np.float32)  # cosine sim
    qwords = vocab[query_ids]
    Nq = len(qwords)
    print(f"  query words: {Nq}   neighbours/word: {neigh_ids.shape[1]}")

    # ---- load vectors and align to query words
    print("Loading word vectors from lancedb ...")
    lwords, lmat = load_lancedb_vectors(args.lancedb, args.table)
    lmat = l2norm(lmat)
    word2vec = {w: i for i, w in enumerate(lwords)}
    qvec_idx = np.array([word2vec.get(w, -1) for w in qwords])
    have_vec = qvec_idx >= 0
    qvecs = np.zeros((Nq, lmat.shape[1]), dtype=np.float32)
    qvecs[have_vec] = lmat[qvec_idx[have_vec]]
    print(f"  {int(have_vec.sum())}/{Nq} query words have a vector")

    # ---- splits / labels
    with open(args.splits_file) as f:
        splits = json.load(f)
    split_items = list(splits.items())
    stop = load_set(args.stop_list)
    first = split_items[0][1]
    implant_all = set(first['ref']) | set(first['dev']) | set(first['test'])

    qword_to_qidx = {w: i for i, w in enumerate(qwords)}
    is_implant = np.array([w in implant_all for w in qwords], dtype=bool)
    is_stop = np.array([w in stop for w in qwords], dtype=bool)
    neg_pool = (~is_implant) & (~is_stop) & have_vec   # candidate negatives
    print(f"  negative pool: {int(neg_pool.sum())}")

    # ---- full-embedding similarity to ALL implants (no top-128 cutoff)
    full_sim = None
    gloss_col = {}
    if use_full:
        print("Building full query x all-implants similarity (no 128 cutoff) ...")
        gloss_words = [w for w in implant_all if w in word2vec]
        gloss_mat = l2norm(lmat[[word2vec[w] for w in gloss_words]])
        gloss_col = {w: c for c, w in enumerate(gloss_words)}
        full_sim = qvecs @ gloss_mat.T                 # (Nq, n_gloss) cosine sim
        print(f"  implants with vectors: {len(gloss_words)}   matrix: {full_sim.shape}")

    # ---- lexical (character n-gram) matrix over all query words, once
    q_tfidf = None
    if use_lexical:
        print("Building lexical character n-gram matrix ...")
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), norm='l2')
        q_tfidf = vec.fit_transform(qwords)            # (Nq, V) sparse, L2-normed
        print(f"  lexical features: {q_tfidf.shape[1]}")

    def lexical_max_to_ref(ref_words):
        """max char-ngram cosine similarity of every query word to any ref word."""
        ref_tfidf = vec.transform(ref_words)           # (n_ref, V), L2-normed
        out = np.zeros(Nq, dtype=np.float32)
        ch = args.lex_chunk
        for s in range(0, Nq, ch):
            block = q_tfidf[s:s + ch] @ ref_tfidf.T     # sparse (chunk, n_ref)
            out[s:s + ch] = np.asarray(block.max(axis=1).todense()).ravel()
        return out

    methods = ['vote', 'mindist', 'avgdist', 'centroid']
    if use_full:
        methods += ['mindist_all', 'avgdist_all', 'votetopk_all']
    if use_lexical:
        methods.append('lexical')
    methods += ['hybrid', 'pu_lr', 'pu_rn']
    metric_names = ['roc_auc', 'average_precision'] + [f'precision_at_{k}' for k in ks]
    results = {m: {mn: [] for mn in metric_names} for m in methods}

    for sid, sp in split_items:
        ref, test = sp['ref'], sp['test']
        ref_set = set(ref)
        ref_mask_vocab = np.array([w in ref_set for w in vocab], dtype=bool)

        # neighbour-based features (vectorised)
        ref_nb = ref_mask_vocab[neigh_ids]            # (Nq, L) bool
        w = neigh_sim * ref_nb                          # similarity weight where ref
        vote = w.sum(axis=1)                            # distance-weighted count
        sim_masked = np.where(ref_nb, neigh_sim, -np.inf)
        mindist = np.where(ref_nb.any(axis=1), sim_masked.max(axis=1), 0.0)
        cnt = ref_nb.sum(axis=1)
        avgdist = np.where(cnt > 0, w.sum(axis=1) / np.maximum(cnt, 1), 0.0)

        # centroid feature
        ref_idx = [qword_to_qidx[x] for x in ref if x in qword_to_qidx and have_vec[qword_to_qidx[x]]]
        if not ref_idx:
            continue
        centroid = qvecs[ref_idx].mean(axis=0)
        cn = np.linalg.norm(centroid)
        if cn == 0:
            continue
        centroid /= cn
        csim = qvecs @ centroid

        # full-embedding "distance to ALL implants" (no 128 cutoff)
        mindist_all = avgdist_all = votetopk_all = None
        if use_full:
            ref_cols = [gloss_col[w] for w in ref if w in gloss_col]
            sub = full_sim[:, ref_cols]                  # (Nq, n_ref) cosine sims
            mindist_all = sub.max(axis=1)
            avgdist_all = sub.mean(axis=1)
            kk = min(args.topk_all, sub.shape[1])
            topk = np.partition(sub, -kk, axis=1)[:, -kk:]
            votetopk_all = topk.sum(axis=1)
            del sub, topk

        # lexical feature (string similarity to ref implant names)
        lex = lexical_max_to_ref(ref) if use_lexical else None

        # labels
        test_set = set(test)
        test_pos = np.array([w in test_set for w in qwords], dtype=bool) & have_vec
        if not test_pos.any():
            continue

        # features for the classifiers
        feat_cols = [vote, mindist, avgdist, csim, cnt.astype(np.float32)]
        if use_full:
            feat_cols += [mindist_all, avgdist_all, votetopk_all]
        if use_lexical:
            feat_cols.append(lex)
        feats = np.nan_to_num(np.column_stack(feat_cols), nan=0.0)

        pos_train = np.array([i for i in ref_idx], dtype=int)
        neg_pool_idx = np.flatnonzero(neg_pool)

        # --- pu_lr: unlabeled words as "negatives"
        if args.use_all_neg:
            n_train_neg = len(neg_pool_idx)
            train_neg = neg_pool_idx
        else:
            n_train_neg = min(len(neg_pool_idx), args.neg_ratio_train * len(pos_train))
            train_neg = rng.choice(neg_pool_idx, size=n_train_neg, replace=False)

        # --- pu_rn: RELIABLE negatives = unlabeled words LEAST similar to implants
        #     (lowest centroid similarity) -> cleaner negative set ("two-step" PU)
        csim_neg = csim[neg_pool_idx]
        order_neg = np.argsort(csim_neg)            # ascending: least similar first
        rn_neg = neg_pool_idx[order_neg[:n_train_neg]]

        def fit_clf(neg_idx):
            Xtr = np.vstack([feats[pos_train], feats[neg_idx]])
            ytr = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_idx))])
            sc = StandardScaler().fit(Xtr)
            Xtr_s = np.nan_to_num(sc.transform(Xtr), nan=0.0)
            c = LogisticRegression(max_iter=1000, class_weight=class_weight)
            c.fit(Xtr_s, ytr)
            return sc, c

        sc_lr, clf_lr = fit_clf(train_neg)
        sc_rn, clf_rn = fit_clf(rn_neg)

        # evaluation universe: test positives + negatives.
        # By default exclude any word used as a training negative by EITHER
        # classifier; --eval-full-pool keeps the full pool (matches standalone runs).
        if args.eval_full_pool:
            eval_neg = neg_pool_idx
        else:
            used_neg = set(train_neg.tolist()) | set(rn_neg.tolist())
            eval_neg = np.array([i for i in neg_pool_idx if i not in used_neg], dtype=int)
        eval_idx = np.concatenate([np.flatnonzero(test_pos), eval_neg])
        y = np.concatenate([np.ones(int(test_pos.sum())), np.zeros(len(eval_neg))]).astype(int)

        def z(a):
            m, s = a.mean(), a.std()
            return (a - m) / (s if s > 0 else 1.0)
        hybrid_all = z(vote) + z(csim) + (z(lex) if use_lexical else 0.0)
        pu_lr_all = clf_lr.predict_proba(np.nan_to_num(sc_lr.transform(feats), nan=0.0))[:, 1]
        pu_rn_all = clf_rn.predict_proba(np.nan_to_num(sc_rn.transform(feats), nan=0.0))[:, 1]

        score_map = {
            'vote': vote, 'mindist': mindist, 'avgdist': avgdist,
            'centroid': csim, 'hybrid': hybrid_all,
            'pu_lr': pu_lr_all, 'pu_rn': pu_rn_all,
        }
        if use_full:
            score_map['mindist_all'] = mindist_all
            score_map['avgdist_all'] = avgdist_all
            score_map['votetopk_all'] = votetopk_all
        if use_lexical:
            score_map['lexical'] = lex
        for m in methods:
            s = score_map[m][eval_idx]
            order = np.argsort(-s, kind='stable')
            y_sorted = y[order]
            results[m]['roc_auc'].append(roc_auc_score(y, s))
            results[m]['average_precision'].append(average_precision_score(y, s))
            for k in ks:
                results[m][f'precision_at_{k}'].append(precision_at_k(y_sorted, k))

    # ---- aggregate + print
    print("\n" + "=" * 78)
    print("COMPARISON ACROSS ALL SPLITS (same vocabulary, same splits)")
    print("=" * 78)
    header = f"{'method':<10}" + "".join(f"{mn:>20}" for mn in metric_names)
    print(header)
    print("-" * len(header))
    for m in methods:
        row = f"{m:<10}"
        for mn in metric_names:
            vals = results[m][mn]
            row += f"{np.mean(vals):>13.4f}±{np.std(vals):<6.4f}"
        print(row)
    print("=" * 78)
    n_used = len(results['vote']['roc_auc'])
    print(f"splits used: {n_used}")


if __name__ == '__main__':
    main()
