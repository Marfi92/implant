#!/usr/bin/env python3
"""The whole implant-ranking pipeline in three readable steps.

This file is written to be READ, not to replace the production scripts.
It shows, in the simplest possible form, what happens between a trained
federated checkpoint and the final evaluation numbers in the paper.

    STEP 1  build_embeddings()  text + model  ->  one vector per word
    STEP 2  score_words()       vectors       ->  one score per candidate word
    STEP 3  evaluate()          scores        ->  AUC / AP / P@K / R@K / F1@K

Run one step at a time:

    python3 pipeline_simple.py embed  --checkpoint FL_global_model_19.pt \
                                      --texts corpus.txt --out vectors.npz
    python3 pipeline_simple.py score  --vectors vectors.npz \
                                      --glossary implants.txt --out scores.csv
    python3 pipeline_simple.py eval   --scores scores.csv --positives heldout.txt

Or `python3 pipeline_simple.py demo` to run all three on synthetic data.
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------------
# STEP 1: EMBEDDING + AGGREGATION
# ----------------------------------------------------------------------------
def build_embeddings(checkpoint, text_lines, layer=-1, max_len=512):
    """One 768-d vector per *word type*, averaged over all its occurrences.

    The model is FROZEN here - this is inference only, no training. The
    aggregation (averaging over occurrences) is part of this step; there is no
    separate "aggregate" stage. This is why the stored table is called
    `words_aggregated`.

    Returns (words, vectors, counts).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base")
    state = torch.load(checkpoint, map_location="cpu")
    # FLARE stores the weights under a 'model' key in some exports
    state = state.get("model", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {checkpoint}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()

    # running sum + counter per word, so raw occurrence vectors are never kept
    vec_sum = defaultdict(lambda: np.zeros(model.config.hidden_size, dtype=np.float64))
    counts = defaultdict(int)

    with torch.no_grad():
        for i, line in enumerate(text_lines):
            words = line.split()
            if not words:
                continue
            # is_split_into_words lets us map every sub-token back to its word
            enc = tok(words, is_split_into_words=True, return_tensors="pt",
                      truncation=True, max_length=max_len)
            hidden = model(**enc).last_hidden_state[0].numpy()  # (n_tokens, 768)
            word_ids = enc.word_ids(0)

            # (a) sub-token vectors -> one vector per word occurrence
            per_occurrence = defaultdict(list)
            for tok_pos, w_idx in enumerate(word_ids):
                if w_idx is not None:            # skip <s> / </s> / padding
                    per_occurrence[w_idx].append(hidden[tok_pos])

            # (b) accumulate that occurrence into the word type's running sum
            for w_idx, sub_vecs in per_occurrence.items():
                word = words[w_idx].lower()
                vec_sum[word] += np.mean(sub_vecs, axis=0)
                counts[word] += 1

            if (i + 1) % 1000 == 0:
                print(f"  {i + 1} lines, {len(counts)} distinct words")

    # (c) divide by the count -> the AGGREGATED type-level vector
    words = sorted(counts)
    vectors = np.stack([vec_sum[w] / counts[w] for w in words]).astype(np.float32)
    # (d) L2-normalise so that cosine similarity is just a dot product
    vectors = l2_normalise(vectors)
    return words, vectors, np.array([counts[w] for w in words], dtype=np.int64)


def l2_normalise(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ----------------------------------------------------------------------------
# STEP 2: SCORING
# ----------------------------------------------------------------------------
def score_words(words, vectors, counts, glossary):
    """Score every word by how implant-like it looks. Returns {method: scores}.

    Vectors are already L2-normalised, so `vectors @ vectors.T` is cosine
    similarity. `sim` below is (n_words x n_glossary): each column is one known
    implant, each row one candidate word.
    """
    index = {w: i for i, w in enumerate(words)}
    ref_idx = [index[g] for g in glossary if g in index]
    if not ref_idx:
        raise SystemExit("none of the glossary words are in the vocabulary")
    print(f"{len(ref_idx)}/{len(glossary)} glossary words found in the vocabulary")

    sim = vectors @ vectors[ref_idx].T          # cosine similarity to each implant

    # A word is trivially most similar to itself; blank that out so a glossary
    # word is not ranked highly just because it is its own neighbour.
    for col, row in enumerate(ref_idx):
        sim[row, col] = -1.0

    scores = {
        # closest single implant  -> good when a term matches ONE implant type
        "mindist_all": sim.max(axis=1),
        # average over all implants -> rewards words near the whole glossary
        "avgdist_all": sim.mean(axis=1),
        # sum of the top-10 -> a middle ground: several implants, not just one
        "votetopk_all": np.sort(sim, axis=1)[:, -10:].sum(axis=1),
        # similarity to the mean glossary vector ("what an implant looks like")
        "centroid": vectors @ l2_normalise(vectors[ref_idx].mean(axis=0, keepdims=True)).ravel(),
        # how often the word occurs: frequent words have more reliable vectors
        "cnt": np.log1p(counts).astype(np.float32),
    }
    return scores, ref_idx


def combine_pu(scores, ref_idx, n_words, seed=0):
    """PU learning: LEARN how to weight the features instead of guessing.

    We have positives (the glossary) but no labelled negatives - only unlabelled
    words. `pu_lr` treats all unlabelled words as negatives. `pu_rn` first picks
    RELIABLE negatives (the words LEAST similar to the glossary centroid), which
    gives a cleaner negative set and therefore better-behaved weights.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    names = sorted(scores)
    feats = np.column_stack([scores[n] for n in names])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler().fit(feats)         # standardise -> comparable weights
    X = scaler.transform(feats)

    y = np.zeros(n_words, dtype=int)
    y[ref_idx] = 1

    out = {}
    for variant in ("pu_lr", "pu_rn"):
        if variant == "pu_lr":
            neg = np.setdiff1d(np.arange(n_words), ref_idx)
        else:
            # reliable negatives: the 10x least centroid-similar unlabelled words
            unlab = np.setdiff1d(np.arange(n_words), ref_idx)
            order = np.argsort(scores["centroid"][unlab])
            neg = unlab[order[:max(len(ref_idx) * 10, 1)]]
        rows = np.concatenate([ref_idx, neg])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=seed).fit(X[rows], y[rows])
        out[variant] = clf.predict_proba(X)[:, 1]
        print(f"  {variant} weights: " +
              ", ".join(f"{n}={w:+.3f}" for n, w in zip(names, clf.coef_.ravel())))
    return out


# ----------------------------------------------------------------------------
# STEP 3: EVALUATION
# ----------------------------------------------------------------------------
def evaluate(score, labels, ks=(50, 100, 200)):
    """Rank by score, then measure how many held-out implants are near the top.

    Under extreme class imbalance (~200 positives in ~190,000 words):
      - Precision@K  = of the K terms we asked an expert to review, how many
                       were real implants -> the practical "review cost" metric.
      - Recall@K     = of all held-out implants, how many we surfaced in the top
                       K. Capped at K/n_pos, so it is LOW by construction.
      - Sensitivity  = identical to Recall in this binary setting.
      - Specificity  = ~0.999 always, because the negative pool is enormous and
                       K is tiny -> it carries almost no information here.
      - Average Precision is the most informative single number, because it
                       accounts for the whole ranking and for the imbalance.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    order = np.argsort(-score)              # descending: best candidates first
    y = labels[order]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    res = {
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "roc_auc": roc_auc_score(labels, score),
        "average_precision": average_precision_score(labels, score),
    }
    for k in ks:
        tp = int(y[:k].sum())
        fp = k - tp
        prec = tp / k
        rec = tp / n_pos if n_pos else 0.0
        res[f"precision@{k}"] = prec
        res[f"recall@{k}"] = rec                       # == sensitivity@k
        res[f"specificity@{k}"] = (n_neg - fp) / n_neg
        res[f"f1@{k}"] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return res


# ----------------------------------------------------------------------------
# glue
# ----------------------------------------------------------------------------
def read_words(path):
    with open(path) as f:
        return [ln.strip().lower() for ln in f if ln.strip()]


def cmd_embed(a):
    words, vectors, counts = build_embeddings(a.checkpoint, open(a.texts, errors="ignore"))
    np.savez_compressed(a.out, words=np.array(words), vectors=vectors, counts=counts)
    print(f"wrote {a.out}: {len(words)} words x {vectors.shape[1]} dims")


def cmd_score(a):
    d = np.load(a.vectors, allow_pickle=True)
    words = [str(w) for w in d["words"]]
    scores, ref_idx = score_words(words, d["vectors"], d["counts"], read_words(a.glossary))
    scores.update(combine_pu(scores, ref_idx, len(words)))
    with open(a.out, "w", newline="") as f:
        cols = sorted(scores)
        wr = csv.writer(f)
        wr.writerow(["word"] + cols)
        for i, w in enumerate(words):
            wr.writerow([w] + [f"{scores[c][i]:.6f}" for c in cols])
    print(f"wrote {a.out}: {len(words)} rows, methods = {', '.join(sorted(scores))}")


def cmd_eval(a):
    positives = set(read_words(a.positives))
    words, cols, data = [], None, []
    with open(a.scores) as f:
        rd = csv.reader(f)
        cols = next(rd)[1:]
        for row in rd:
            words.append(row[0])
            data.append([float(x) for x in row[1:]])
    mat = np.array(data)
    labels = np.array([1 if w in positives else 0 for w in words])
    print(f"{labels.sum()} positives / {len(labels)} words "
          f"({100 * labels.mean():.3f}% -> extreme imbalance)\n")
    print(f"{'method':<16}{'ROC AUC':>9}{'AP':>9}{'P@50':>8}{'R@50':>8}{'F1@50':>8}")
    for j, name in enumerate(cols):
        r = evaluate(mat[:, j], labels)
        print(f"{name:<16}{r['roc_auc']:>9.4f}{r['average_precision']:>9.4f}"
              f"{r['precision@50']:>8.3f}{r['recall@50']:>8.3f}{r['f1@50']:>8.3f}")


def cmd_demo(a):
    """Steps 2 and 3 on synthetic data, so you can see the shapes and numbers."""
    rng = np.random.default_rng(0)
    n, dim, n_impl = 2000, 32, 40
    # implants live in one cluster; everything else is spread out
    implant_dir = rng.normal(size=dim)
    vectors = rng.normal(size=(n, dim))
    vectors[:n_impl] += 3 * implant_dir            # 40 true implants
    vectors[n_impl:n_impl + 60] += 1.5 * implant_dir  # 60 "nearby" implants
    vectors = l2_normalise(vectors)
    words = [f"w{i}" for i in range(n)]
    counts = rng.integers(1, 500, size=n)

    glossary = words[:n_impl]                     # known implants (the queries)
    scores, ref_idx = score_words(words, vectors, counts, glossary)
    scores.update(combine_pu(scores, ref_idx, n))

    # held-out positives = the "nearby" ones the glossary does NOT contain
    labels = np.zeros(n, dtype=int)
    labels[n_impl:n_impl + 60] = 1
    print(f"\n{labels.sum()} held-out positives / {n} words\n")
    print(f"{'method':<16}{'ROC AUC':>9}{'AP':>9}{'P@50':>8}{'R@50':>8}{'F1@50':>8}")
    for name in sorted(scores):
        r = evaluate(scores[name], labels)
        print(f"{name:<16}{r['roc_auc']:>9.4f}{r['average_precision']:>9.4f}"
              f"{r['precision@50']:>8.3f}{r['recall@50']:>8.3f}{r['f1@50']:>8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("embed", help="STEP 1: text + checkpoint -> one vector per word")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--texts", required=True, help="one document/sentence per line")
    p.add_argument("--out", default="vectors.npz")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("score", help="STEP 2: vectors -> one score per word")
    p.add_argument("--vectors", required=True)
    p.add_argument("--glossary", required=True, help="known implant terms, one per line")
    p.add_argument("--out", default="scores.csv")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("eval", help="STEP 3: scores -> AUC / AP / P@K / R@K / F1@K")
    p.add_argument("--scores", required=True)
    p.add_argument("--positives", required=True, help="held-out implant terms")
    p.set_defaults(func=cmd_eval)

    sub.add_parser("demo", help="run steps 2-3 on synthetic data").set_defaults(func=cmd_demo)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
