#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pickle
from collections import defaultdict
from hashlib import md5
import multiprocessing
import random

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# -------------------------
# Split generation helpers
# -------------------------
def make_random_split(master_list, n_repeats=10, n_folds=10, seed=42):
    random.seed(seed)
    all_splits = {}
    split_id = 0

    for r in range(n_repeats):
        for f in range(n_folds):
            items = master_list[:]
            random.shuffle(items)

            n = len(items)
            n_ref = int(n * 0.80)
            n_dev = int(n * 0.10)
            n_test = n - n_ref - n_dev

            all_splits[split_id] = {
                "ref": items[:n_ref],
                "dev": items[n_ref:n_ref+n_dev],
                "test": items[n_ref+n_dev:]
            }
            split_id += 1

    return all_splits


def generate_and_save_split(master_list_path, output_json_path):
    with open(master_list_path, "r", encoding="utf-8") as fp:
        master_list = [line.strip() for line in fp if line.strip()]

    splits = make_random_split(master_list)

    with open(output_json_path, "w", encoding="utf-8") as fp:
        json.dump(splits, fp, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(splits)} random splits to: {output_json_path}\n")


# -------------------------
# Dataset wrapper
# -------------------------
class NeighbourHoodDataset(Dataset):
    def __init__(self, neighbourhoods_dir: Path):
        self.neighbourhood_files = sorted(neighbourhoods_dir.glob("*.pkl"))
        neighbourhood_hash = md5()
        for neighbourhood_file in tqdm(self.neighbourhood_files, desc="Hashing neighbourhood files"):
            neighbourhood_hash.update(str(neighbourhood_file).encode('utf-8'))

        self.neighbourhood_hash = neighbourhood_hash.hexdigest()
        super().__init__()

    def __len__(self):
        return len(self.neighbourhood_files)

    def __getitem__(self, index):
        neighbourhood_file = self.neighbourhood_files[index]
        neighbourhood = get_arrays_for_file(neighbourhood_file)
        return neighbourhood


def no_tensor_collator(batch):
    return batch


EVAL_METRICS = ('roc_auc', 'precision', 'recall', 'f1')


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze the neighbourhoods extracted from a vector database for a given test dataset.")
    parser.add_argument('neighbourhoods', help='Directory with neighbourhood .pkl files (positional).', type=Path)
    parser.add_argument('--output-dir', help='Directory to write analysis results to.', type=Path)
    parser.add_argument('--num-workers', help='Number of workers for processing.', type=int, default=0)
    parser.add_argument('--recalculate', help='If set, recalculate the votes HDF5 store', action='store_true')
    parser.add_argument('--threshold-metric', help="Metric to use for setting threshold", choices=('f1', 'ba'), default='f1')
    parser.add_argument('--chunk-size', help="Chunk size for HDF5 writes", type=int, default=2**16)

    # split generation
    parser.add_argument('--make-splits', help='Generate random 80/10/10 splits and exit.', action='store_true')
    parser.add_argument('--master_list', help='Path to master implant list for split generation.', type=Path)
    parser.add_argument('--split_output', help='Where to save generated split JSON.', type=Path, default="implant_splits.json")

    # split-based evaluation
    parser.add_argument('--splits-file', help='Path to the splits JSON file for split-based evaluation.', type=Path)

    args = parser.parse_args()

    # --- Split generation mode ---
    if args.make_splits:
        if not args.master_list:
            raise ValueError("Provide --master_list when using --make-splits")
        generate_and_save_split(args.master_list, args.split_output)
        return

    # --- Analysis mode ---
    output_dir = args.output_dir if args.output_dir else args.neighbourhoods / "analysis"
    neighbours_dataset = NeighbourHoodDataset(args.neighbourhoods)
    output_dir.mkdir(exist_ok=True, parents=True)
    votes_file = output_dir / f"neighbourhood_analysis_{neighbours_dataset.neighbourhood_hash}.h5"

    if args.recalculate or not votes_file.exists():
        partial_file: Path = votes_file.with_suffix('.tmp')
        dataloader = DataLoader(
            neighbours_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=no_tensor_collator
        )
        with h5py.File(partial_file, 'w') as store:
            n_words = 0
            query_word_classes = []
            query_word_lists = []   # NEW
            votes = []
            for batch in tqdm(dataloader, desc="Reading neighbourhoods"):
                for example in batch:
                    batch_query_words = example.get('query_words', [])
                    batch_query_word_classes = example['query_word_classes']
                    n_batch_words = len(batch_query_word_classes)
                    n_words += n_batch_words

                    query_word_lists.append(batch_query_words)
                    query_word_classes.append(batch_query_word_classes)
                    votes.append(example['votes'])

                    if n_words >= args.chunk_size:
                        n_words, query_word_classes, votes, query_word_lists = record_results(
                            store, query_word_classes, votes, args.chunk_size, query_word_lists
                        )

            # flush remaining
            record_results(store, query_word_classes, votes, args.chunk_size, query_word_lists)
        partial_file.rename(votes_file)

    # If no splits-file provided -> standard analysis
    if not args.splits_file:
        metrics_file = output_dir / "analyzed_neighbourhoods.csv"
        if not metrics_file.exists() or args.recalculate:
            metrics_df = compute_statistics(votes_file, num_workers=args.num_workers)
            metrics_df.to_csv(metrics_file, index=False)
        else:
            metrics_df = pd.read_csv(metrics_file)

        for eval_metric in EVAL_METRICS:
            for threshold_on, threshold_df in metrics_df.groupby('threshold_on'):
                plt.figure()
                for weight_type, scores_df in threshold_df.groupby('weight_type'):
                    scores_df = scores_df.sort_values('n_neighbours')
                    n_neighbours = scores_df['n_neighbours']
                    score = scores_df[eval_metric]
                    plt.plot(n_neighbours, score, label=weight_type)
                plt.xlabel("Number of Neighbours")
                plt.ylabel(f"{eval_metric} Score")
                plt.title(f"{eval_metric} vs Number of Neighbours (thresholded on {threshold_on})")
                plt.legend()
                plt.savefig(output_dir / f"{threshold_on}_{eval_metric}_vs_neighbours.png")
                plt.close()
    else:
        # Split-based evaluation
        print(f"\nRunning split-based evaluation using: {args.splits_file}")
        with open(args.splits_file, "r", encoding="utf-8") as fp:
            splits = json.load(fp)
        print(f"Loaded {len(splits)} splits")

        all_split_results = []
        for split_id, split_data in tqdm(splits.items(), desc="Evaluating splits"):
            ref_words = set(split_data["ref"])
            dev_words = set(split_data["dev"])
            test_words = set(split_data["test"])

            split_metrics = compute_statistics_with_split(
                votes_file,
                ref_words=ref_words,
                dev_words=dev_words,
                test_words=test_words,
                num_workers=args.num_workers
            )

            if split_metrics is not None:
                split_metrics['split_id'] = int(split_id)
                all_split_results.append(split_metrics)

        if all_split_results:
            results_df = pd.DataFrame(all_split_results)
            per_split_file = output_dir / "per_split_results.csv"
            results_df.to_csv(per_split_file, index=False)
            print(f"\nPer-split results saved to: {per_split_file}")

            numeric_cols = ['roc_auc', 'precision', 'recall', 'f1', 'average_precision', 'threshold']
            available_cols = [c for c in numeric_cols if c in results_df.columns]

            agg_results = {}
            for col in available_cols:
                agg_results[f"{col}_mean"] = results_df[col].mean()
                agg_results[f"{col}_std"] = results_df[col].std()

            agg_df = pd.DataFrame([agg_results])
            agg_file = output_dir / "aggregated_split_results.csv"
            agg_df.to_csv(agg_file, index=False)

            print("\n" + "="*60)
            print("AGGREGATED RESULTS ACROSS ALL SPLITS")
            print("="*60)
            for col in available_cols:
                mean_val = results_df[col].mean()
                std_val = results_df[col].std()
                print(f"  {col:>20s}: {mean_val:.4f} ± {std_val:.4f}")
            print("="*60)

            for metric in available_cols:
                if metric == 'threshold':
                    continue
                plt.figure(figsize=(8, 5))
                plt.hist(results_df[metric], bins=20, edgecolor='black', alpha=0.7)
                mean_val = results_df[metric].mean()
                plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
                plt.xlabel(metric)
                plt.ylabel("Count")
                plt.title(f"Distribution of {metric} across {len(results_df)} splits\nMean: {mean_val:.4f}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / f"split_distribution_{metric}.png")
                plt.close()
            print(f"\nDistribution plots saved to: {output_dir}")
        else:
            print("WARNING: No split results were computed. Check your splits file and data.")

# -------------------------
# HDF5 record / compute helpers
# -------------------------
def record_results(store: h5py.File, query_word_classes, votes, chunk_size, query_word_lists):
    if len(query_word_classes) == 0:
        return 0, [], [], []

    # classes
    concatenated_query_word_classes = np.concatenate(query_word_classes)
    store_query_word_classes = concatenated_query_word_classes[:chunk_size]
    remaining_query_word_classes_arr = concatenated_query_word_classes[chunk_size:]
    n_remaining = len(remaining_query_word_classes_arr)
    remaining_word_classes = []
    if n_remaining > 0:
        remaining_word_classes.append(remaining_query_word_classes_arr)

    # query words
    concatenated_query_words = list(np.concatenate(query_word_lists)) if query_word_lists else []
    store_query_words = concatenated_query_words[:chunk_size]
    remaining_query_words_arr = concatenated_query_words[chunk_size:]
    remaining_query_words = []
    if len(remaining_query_words_arr) > 0:
        remaining_query_words.append(np.array(remaining_query_words_arr, dtype='U'))

    # write / append classes
    if 'query_word_classes' not in store:
        store.create_dataset('query_word_classes', data=store_query_word_classes, maxshape=(None,), chunks=(chunk_size,))
    else:
        ds = store['query_word_classes']
        cur = ds.shape[0]
        new = cur + len(store_query_word_classes)
        ds.resize((new,), axis=0)
        ds[cur:new] = store_query_word_classes

    # write / append query words
    dt = h5py.string_dtype(encoding='utf-8')
    store_query_words_encoded = np.array(store_query_words, dtype=object)
    if 'query_words' not in store:
        store.create_dataset('query_words', data=store_query_words_encoded, maxshape=(None,), dtype=dt, chunks=(chunk_size,))
    else:
        dsw = store['query_words']
        cur = dsw.shape[0]
        new = cur + len(store_query_words_encoded)
        dsw.resize((new,), axis=0)
        dsw[cur:new] = store_query_words_encoded

    # flatten votes
    flattened_votes = defaultdict(lambda: defaultdict(list))
    for votes_batch in votes:
        for weight_type, neighbour_votes in votes_batch.items():
            for n, votes_array in neighbour_votes.items():
                flattened_votes[weight_type][n].append(votes_array)

    remaining_votes_batch = {}
    any_remaining = False

    for weight_type, neighbour_votes in flattened_votes.items():
        g = store.require_group(weight_type)
        all_n_neighbours = set(int(n) for n in neighbour_votes.keys())
        if 'n_neighbours' in g.attrs:
            all_n_neighbours.update(g.attrs['n_neighbours'])
        g.attrs['n_neighbours'] = sorted(all_n_neighbours)

        remaining_neighbour_votes = {}
        for n_neighbours, votes_value in neighbour_votes.items():
            concatenated_vote_values = np.concatenate(votes_value)
            store_vote_values = concatenated_vote_values[:chunk_size]
            remaining_vote_values_arr = concatenated_vote_values[chunk_size:]

            if str(n_neighbours) not in g:
                g.create_dataset(str(n_neighbours), data=store_vote_values, maxshape=(None,), chunks=(chunk_size,))
            else:
                ds = g[str(n_neighbours)]
                cur = ds.shape[0]
                new = cur + store_vote_values.shape[0]
                ds.resize((new,), axis=0)
                ds[cur:new] = store_vote_values

            if len(remaining_vote_values_arr) != n_remaining:
                raise RuntimeError("The remaining number of votes and number of remaining word classes differ")

            if len(remaining_vote_values_arr) > 0:
                remaining_neighbour_votes[n_neighbours] = remaining_vote_values_arr
                any_remaining = True

        remaining_votes_batch[weight_type] = remaining_neighbour_votes

    # FIX: Store the weighting function names as a root-level attribute
    # so that compute_statistics / compute_statistics_with_split can find them.
    all_weight_types = sorted(flattened_votes.keys())
    if 'weighting_function' in store.attrs:
        existing = set(store.attrs['weighting_function'])
        existing.update(all_weight_types)
        all_weight_types = sorted(existing)
    store.attrs['weighting_function'] = all_weight_types

    remaining_votes = [remaining_votes_batch] if any_remaining else []

    return n_remaining, remaining_word_classes, remaining_votes, remaining_query_words


def compute_statistics(votes_file: Path, num_workers=0):
    work_packages = []
    with h5py.File(votes_file, 'r') as store:
        weighting_functions = store.attrs['weighting_function']
        for weight_type in weighting_functions:
            g = store[weight_type]
            all_n_neighbours = g.attrs['n_neighbours']
            for n_neighbours in all_n_neighbours:
                work_package = (votes_file, weight_type, n_neighbours)
                work_packages.append(work_package)

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            records = list(tqdm(pool.imap_unordered(statistics_worker, work_packages), total=len(work_packages)))
    else:
        records = [statistics_worker(work_package) for work_package in tqdm(work_packages)]

    df = pd.DataFrame.from_records([record for record_pair in records for record in record_pair])
    return df


def statistics_worker(work_package):
    store_file, weight_type, n_neighbours = work_package
    records = []
    with h5py.File(store_file) as store:
        query_word_classes = store['query_word_classes'][:]
        votes = store[weight_type][str(n_neighbours)][:]
        fpr, tpr, roc_thresholds = roc_curve(query_word_classes, votes)
        roc_auc = np.trapz(tpr, fpr)
        precision_sweep, recall_sweep, prc_thresholds = precision_recall_curve(query_word_classes, votes)
        ap = -np.sum(np.diff(recall_sweep) * precision_sweep[:-1])

        youdens_j = tpr - fpr
        best_threshold_index = np.argmax(youdens_j)
        best_threshold = roc_thresholds[best_threshold_index]
        discretized_votes = votes > best_threshold
        precision = precision_score(query_word_classes, discretized_votes)
        f1 = f1_score(query_word_classes, discretized_votes)
        recall = recall_score(query_word_classes, discretized_votes)
        performance_record_ba = {'weight_type': weight_type,
                                 'n_neighbours': n_neighbours,
                                 'roc_auc': roc_auc,
                                 'average_precision': ap,
                                 'threshold': best_threshold,
                                 'precision': precision,
                                 'recall': recall,
                                 'f1': f1,
                                 'threshold_on': 'ba'}
        records.append(performance_record_ba)

        f1_sweep = 2 * precision_sweep[:-1] * recall_sweep[:-1] / (precision_sweep[:-1] + recall_sweep[:-1] + 1e-12)
        best_threshold_index = np.argmax(f1_sweep)
        best_threshold = prc_thresholds[best_threshold_index]
        discretized_votes = votes > best_threshold
        precision = precision_sweep[best_threshold_index]
        recall = recall_sweep[best_threshold_index]
        f1 = f1_sweep[best_threshold_index]
        performance_record_f1 = {'weight_type': weight_type,
                                 'n_neighbours': n_neighbours,
                                 'roc_auc': roc_auc,
                                 'average_precision': ap,
                                 'threshold': best_threshold,
                                 'precision': precision,
                                 'recall': recall,
                                 'f1': f1,
                                 'threshold_on': 'f1'}
        records.append(performance_record_f1)
    return records


# -------------------------
# Split-based evaluation helper
# -------------------------
def compute_statistics_with_split(votes_file, ref_words, dev_words, test_words, num_workers=0):
    best_result = None
    best_roc_auc = -1

    with h5py.File(votes_file, 'r') as store:
        query_word_classes = store['query_word_classes'][:]
        raw_qwords = store['query_words'][:]
        query_words = [w.decode('utf-8') if isinstance(w, bytes) else w for w in raw_qwords]

        test_mask = np.array([w in test_words for w in query_words], dtype=bool)
        if not test_mask.any():
            return None

        weighting_functions = store.attrs['weighting_function']

        for weight_type in weighting_functions:
            g = store[weight_type]
            all_n_neighbours = g.attrs['n_neighbours']
            for n_neighbours in all_n_neighbours:
                votes = g[str(n_neighbours)][:]

                votes_test = votes[test_mask]
                classes_test = query_word_classes[test_mask]

                try:
                    fpr, tpr, roc_thresholds = roc_curve(classes_test, votes_test)
                    roc_auc = np.trapz(tpr, fpr)
                    precision_sweep, recall_sweep, prc_thresholds = precision_recall_curve(classes_test, votes_test)
                    ap = -np.sum(np.diff(recall_sweep) * precision_sweep[:-1])

                    f1_sweep = 2 * precision_sweep[:-1] * recall_sweep[:-1] / (precision_sweep[:-1] + recall_sweep[:-1] + 1e-12)
                    best_threshold_index = np.argmax(f1_sweep)
                    best_threshold = prc_thresholds[best_threshold_index]
                    precision = precision_sweep[best_threshold_index]
                    recall = recall_sweep[best_threshold_index]
                    f1 = f1_sweep[best_threshold_index]

                    if roc_auc > best_roc_auc:
                        best_roc_auc = roc_auc
                        best_result = {
                            'weight_type': weight_type,
                            'n_neighbours': n_neighbours,
                            'roc_auc': roc_auc,
                            'average_precision': ap,
                            'threshold': best_threshold,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                        }
                except Exception:
                    continue

    return best_result


# -------------------------
# File parsing and vote logic
# -------------------------
def get_arrays_for_file(neighbourhood_file):
    neighbourhood_classes = []
    neighbourhood_distances = []
    query_word_classes = []
    max_neighbours = 0
    with open(neighbourhood_file, 'rb') as fp:
        neighbourhood_data = pickle.load(fp)
        neighbourhoods = neighbourhood_data["neighbourhoods"]
        class_mappings = neighbourhood_data["class_mapping"]

    skip_labels = (class_mappings.get('stop_word'), class_mappings.get('known_positive'))
    query_words = []

    for (query_word, query_label), neighbours in neighbourhoods:
        if query_label in skip_labels:
            continue

        max_neighbours = max(max_neighbours, len(neighbours))
        query_neighbour_classes = []
        query_distances = []
        for cossim, neighbour_word, neighbour_label in neighbours:
            query_neighbour_classes.append(neighbour_label)
            query_distances.append(cossim)
        if query_neighbour_classes:
            query_word_classes.append(query_label)
            neighbourhood_classes.append(query_neighbour_classes)
            neighbourhood_distances.append(query_distances)
            query_words.append(query_word)

    neighbourhood_classes = np.array(neighbourhood_classes, dtype=np.int8)
    neighbourhood_distances = np.array(neighbourhood_distances, dtype=np.float32)
    query_word_classes = np.array(query_word_classes, dtype=np.int8)
    if (len(query_word_classes) != len(neighbourhood_classes)
        or len(neighbourhood_distances) != len(query_word_classes)
        or len(neighbourhood_distances) != len(neighbourhood_classes)):
        raise RuntimeError("Array lengths differs")

    n_neighbours = list(range(1, max_neighbours+1))
    votes = get_votes(neighbourhood_classes, neighbourhood_distances, n_neighbours)

    return {
        "query_words": query_words,
        "query_word_classes": query_word_classes,
        "votes": votes
    }


def get_votes(neighbourhood_classes, neighbourhood_distances, n_neighbours):
    votes = {}
    for weight_type in ['constant', 'inverse', 'exponential']:
        weight_type_votes = {}
        neighbour_weights = weighting_function(neighbourhood_distances, weight_type=weight_type)
        neighbour_weighted_classes = neighbour_weights * neighbourhood_classes
        neighbour_cumsums = np.cumsum(neighbour_weighted_classes, axis=1)
        for i in n_neighbours:
            weight_type_votes[i] = neighbour_cumsums[:, i-1]
        votes[weight_type] = weight_type_votes
    return votes


def weighting_function(distances, weight_type='inverse', eps=1e-5):
    if weight_type == 'inverse':
        return 1 / (distances + eps)
    elif weight_type == 'exponential':
        return np.exp(-distances)
    elif weight_type == 'constant':
        return np.ones_like(distances)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")


if __name__ == "__main__":
    main()
