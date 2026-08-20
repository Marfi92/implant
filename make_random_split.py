"""Label-randomisation control.

Writes a copy of the official split file in which every implant term is replaced
by a random word drawn from the same query vocabulary (same word for the same
term in every split, so the ref/dev/test structure and all set sizes are kept).
Running the evaluation with this file must give ROC AUC ~0.5; anything higher
indicates leakage in the pipeline.
"""
import argparse
import json

import h5py
import numpy as np


def load_set(path):
    if not path:
        return set()
    with open(path) as f:
        return {ln.split()[0] for ln in f if ln.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True, help='neighbour_raw_*.h5')
    ap.add_argument('--splits-file', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stop_list', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with h5py.File(args.store, 'r') as st:
        vocab = np.array([w.decode('utf-8') if isinstance(w, bytes) else str(w)
                          for w in st['vocab'][:]], dtype=object)
        query_ids = st['query_word_ids'][:]
    qwords = vocab[query_ids]

    with open(args.splits_file) as f:
        splits = json.load(f)

    first = next(iter(splits.values()))
    implants = set(first['ref']) | set(first['dev']) | set(first['test'])
    stop = load_set(args.stop_list)

    pool = [w for w in qwords if w not in implants and w not in stop]
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(len(pool), size=len(implants), replace=False)
    mapping = dict(zip(sorted(implants), [pool[i] for i in picked]))

    def remap(value):
        if isinstance(value, list):
            return [mapping[w] for w in value if w in mapping]
        return value

    out = {sid: {key: remap(value) for key, value in sp.items()}
           for sid, sp in splits.items()}
    with open(args.out, 'w') as f:
        json.dump(out, f)

    print(f'implant terms replaced: {len(mapping)}')
    print(f'candidate pool size:    {len(pool)}')
    print(f'splits written:         {len(out)}')
    print(f'example: {list(mapping.items())[:3]}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
