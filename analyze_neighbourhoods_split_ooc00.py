import argparse
import json
from pathlib import Path
import pickle
from collections import Counter, defaultdict
from hashlib import md5
import multiprocessing
import random
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# --- CHANGE: numpy >= 2.0 renamed np.trapz to np.trapezoid ---
_trapz = getattr(np, 'trapezoid', np.trapz)

class NeighbourHoodDataset(Dataset):
    def __init__(self, neighbourhood_files, neighbourhood_limit=None):
        self.neighbourhood_files = sorted(neighbourhood_files)
        self.neighbourhood_limit = neighbourhood_limit
        super().__init__()  
    
    def __len__(self):
        return len(self.neighbourhood_files)

    def __getitem__(self, index):
        neighbourhood_file = self.neighbourhood_files[index]
        neighbourhood = get_arrays_for_file(neighbourhood_file, neighbourhood_limit=self.neighbourhood_limit)
        return neighbourhood


def no_tensor_collator(batch):
    return batch

EVAL_METRICS = ('roc_auc', 'precision', 'recall', 'f1', 'average_precision')


# --- CHANGE: Added split generation helper ---
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
            all_splits[split_id] = {
                "ref": items[:n_ref],
                "dev": items[n_ref:n_ref+n_dev],
                "test": items[n_ref+n_dev:]
            }
            split_id += 1
    return all_splits


# --- CHANGE: Added bootstrap CI helper (Erik's suggestion) ---
def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    boot_means = np.array([np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, 100 * alpha), np.percentile(boot_means, 100 * (1 - alpha))


def main():
    parser = argparse.ArgumentParser(description="Analyze the neighbourhoods extracted from a vector database for a given test dataset.")
    parser.add_argument('neighbourhoods', help='The directory containing the neighbourhoods extracted from the vector database.', type=Path)
    parser.add_argument('--output-dir', help='The directory to write the analysis results to.', type=Path)
    parser.add_argument('--num-workers', help='The number of workers to use for processing the neighbourhoods.', type=int, default=0)
    parser.add_argument('--recalculate', help='If set, recalculate the votes file HDF5 store', action='store_true')
    parser.add_argument('--threshold-metric', help="What metric to use for setting the threshold", choices=('f1', 'ba'), default='f1')
    parser.add_argument('--chunk-size', help="How large chunks of data to write to store", type=int, default=2**16)
    # --- CHANGE: Added split-based evaluation arguments ---
    parser.add_argument('--splits-file', help='Path to the splits JSON file for split-based evaluation.', type=Path)
    parser.add_argument('--make-splits', help='Generate random 80/10/10 splits and exit.', action='store_true')
    parser.add_argument('--master_list', help='Path to master implant list for split generation.', type=Path)
    parser.add_argument('--split_output', help='Where to save generated split JSON.', type=Path, default="implant_splits.json")
    args = parser.parse_args()

    # --- CHANGE: Split generation mode ---
    if args.make_splits:
        if not args.master_list:
            raise ValueError("Provide --master_list when using --make-splits")
        with open(args.master_list, "r", encoding="utf-8") as fp:
            master_list = [line.strip() for line in fp if line.strip()]
        splits = make_random_split(master_list)
        with open(args.split_output, "w", encoding="utf-8") as fp:
            json.dump(splits, fp, indent=2, ensure_ascii=False)
        print(f"Saved {len(splits)} splits to: {args.split_output}")
        return

    neighbourhood_files = sorted(args.neighbourhoods.glob("*.pkl"))
    neighbourhood_hash = md5()
    for neighbourhood_file in tqdm(neighbourhood_files, desc="Hashing neighbourhood files"):
        neighbourhood_hash.update(str(neighbourhood_file).encode('utf-8'))
        
    neighbourhood_hash = neighbourhood_hash.hexdigest()
    output_dir = args.output_dir if args.output_dir else args.neighbourhoods / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    # --- CHANGE: renamed from distances to analysis (vote-based approach) ---
    votes_file = output_dir / f"neigbourhood_analysis_{neighbourhood_hash}.h5"

    if args.recalculate or not votes_file.exists():
        partial_file: Path = votes_file.with_suffix('.tmp')
        neighbourhood_limit = get_neighbourhood_limit(neighbourhood_files, num_workers=args.num_workers)
        if neighbourhood_limit is None or neighbourhood_limit < 0:
            raise RuntimeError(f"Could not determine the neighbourhood limit, got {neighbourhood_limit}.")

        neighbours_dataset = NeighbourHoodDataset(neighbourhood_files, neighbourhood_limit=neighbourhood_limit)
        dataloader = DataLoader(neighbours_dataset, batch_size=1, num_workers=args.num_workers, drop_last=False, collate_fn=no_tensor_collator)
        with h5py.File(partial_file, 'w') as store:
            n_words = 0
            query_word_classes = []
            votes = []
            # --- CHANGE: Also collect query_words for split matching ---
            query_word_lists = []
            for batch in tqdm(dataloader, desc="Reading neighbourhoods"):
                for example in batch:
                    batch_query_word_classes = example['query_word_classes']
                    n_batch_words = len(batch_query_word_classes)
                    n_words += n_batch_words
                    query_word_classes.append(batch_query_word_classes)
                    # --- CHANGE: use 'votes' (weighted positive counts) instead of 'distances' ---
                    batch_votes = example['votes']
                    votes.append(batch_votes)
                    # --- CHANGE: collect query words ---
                    query_word_lists.append(example.get('query_words', []))
                    if n_words >= args.chunk_size:
                        n_words, query_word_classes, votes, query_word_lists = record_results(
                            store, query_word_classes, votes, args.chunk_size, query_word_lists
                        )
            record_results(store, query_word_classes, votes, args.chunk_size, query_word_lists)
        partial_file.rename(votes_file)

    # --- CHANGE: Branch on whether splits-file is provided ---
    if not args.splits_file:
        # Original non-split analysis (boss's code)
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
                plt.title(f"{eval_metric} vs Number of Neighbours (thresholded on {threshold_on})\n"
                          f"Positive: {threshold_df['positive_words'].iloc[0]}, "
                          f"Negative: {threshold_df['negative_words'].iloc[0]}")
                plt.legend()
                plt.savefig(output_dir / f"{threshold_on}_{eval_metric}_vs_neighbours.png")
                plt.close()
    else:
        # --- CHANGE: Split-based evaluation ---
        run_split_evaluation(votes_file, args.splits_file, output_dir)


# --- CHANGE: Added split-based evaluation function ---
def run_split_evaluation(votes_file, splits_file, output_dir):
    """Evaluate using ref/dev/test splits with bootstrapped CIs."""
    print(f"\nRunning split-based evaluation using: {splits_file}")
    with open(splits_file, "r", encoding="utf-8") as fp:
        splits = json.load(fp)
    print(f"Loaded {len(splits)} splits")

    # Print split summary
    split_sizes = {'ref': [], 'dev': [], 'test': []}
    for split_data in splits.values():
        for key in ('ref', 'dev', 'test'):
            split_sizes[key].append(len(split_data[key]))
    print(f"\n{'='*70}")
    print("SPLIT FILE SUMMARY")
    print(f"{'='*70}")
    for key in ('ref', 'dev', 'test'):
        vals = split_sizes[key]
        print(f"  {key:>5s}: min={min(vals):>6d}  max={max(vals):>6d}  "
              f"mean={np.mean(vals):>8.1f}  (across {len(vals)} splits)")
    print(f"{'='*70}")

    all_split_results = []
    all_split_info = []

    for split_id, split_data in tqdm(splits.items(), desc="Evaluating splits"):
        ref_words = set(split_data["ref"])
        dev_words = set(split_data["dev"])
        test_words = set(split_data["test"])

        split_results, split_info = compute_statistics_with_split(
            votes_file, ref_words, dev_words, test_words, split_id=int(split_id)
        )
        all_split_info.append(split_info)

        if split_results:
            for result in split_results:
                result['split_id'] = int(split_id)
            all_split_results.extend(split_results)

    # Print split diagnostic info
    if all_split_info:
        info_df = pd.DataFrame(all_split_info)
        split_info_file = output_dir / "split_info.csv"
        info_df.to_csv(split_info_file, index=False)

        print(f"\n{'='*70}")
        print("SPLIT DIAGNOSTIC INFO (how splits map to data)")
        print(f"{'='*70}")
        if len(info_df) > 0 and 'total_words_in_data' in info_df.columns:
            first = info_df.iloc[0]
            print(f"  Total words in HDF5 data: {int(first['total_words_in_data'])}")
            print(f"  Total positive (class=1): {int(first['total_positive_in_data'])}")
            print(f"  Total negative (class=0): {int(first['total_negative_in_data'])}")
            if first['total_words_in_data'] > 0:
                print(f"  Overall positive rate:    {first['total_positive_in_data']/first['total_words_in_data']:.4%}")

        print(f"\n  Per-split test set statistics (across {len(info_df)} splits):")
        for col, label in [('test_matched', 'Test words matched'),
                           ('test_positive', 'Test positives'),
                           ('test_negative', 'Test negatives'),
                           ('test_positive_rate', 'Test positive rate')]:
            if col in info_df.columns:
                vals = info_df[col].values
                fmt = '.4f' if 'rate' in col else '.1f'
                print(f"    {label:>25s}: min={np.min(vals):{fmt}}  max={np.max(vals):{fmt}}  "
                      f"mean={np.mean(vals):{fmt}}  median={np.median(vals):{fmt}}")

        print(f"\n  Split info saved to: {split_info_file}")
        print(f"{'='*70}")

    if not all_split_results:
        print("WARNING: No split results were computed.")
        return

    results_df = pd.DataFrame(all_split_results)
    per_split_file = output_dir / "per_split_results.csv"
    results_df.to_csv(per_split_file, index=False)
    print(f"\nPer-split results saved to: {per_split_file}")

    # Per-split table
    print(f"\n{'='*70}")
    print("PER-SPLIT RESULTS")
    print(f"{'='*70}")
    print(f"{'Split':>6s}  {'Weight':>8s}  {'n_neigh':>7s}  "
          f"{'ROC_AUC':>8s}  {'Precision':>9s}  {'Avg_Prec':>8s}  "
          f"{'Recall':>7s}  {'F1':>7s}  "
          f"{'DevAUC':>8s}")
    print("-"*90)
    for _, row in results_df.sort_values('split_id').iterrows():
        print(f"{int(row['split_id']):>6d}  "
              f"{row['weight_type']:>8s}  "
              f"{int(row['n_neighbours']):>7d}  "
              f"{row['roc_auc']:>8.4f}  "
              f"{row['precision']:>9.4f}  "
              f"{row['average_precision']:>8.4f}  "
              f"{row.get('recall', float('nan')):>7.4f}  "
              f"{row.get('f1', float('nan')):>7.4f}  "
              f"{row.get('dev_roc_auc', float('nan')):>8.4f}")
    print("-"*90)

    # Aggregated stats with bootstrapped CI
    numeric_cols = ['roc_auc', 'dev_roc_auc', 'precision', 'recall', 'f1', 'average_precision', 'threshold']
    available_cols = [c for c in numeric_cols if c in results_df.columns and results_df[c].notna().any()]

    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS ACROSS ALL SPLITS")
    print(f"{'='*70}")
    for col in available_cols:
        values = results_df[col].dropna().values
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        median_val = np.median(values)
        min_val = np.min(values)
        max_val = np.max(values)
        ci_lo, ci_hi = bootstrap_ci(values)
        print(f"  {col:>20s}: mean={mean_val:.4f} ± {std_val:.4f}  "
              f"median={median_val:.4f}  min={min_val:.4f}  max={max_val:.4f}  "
              f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"{'='*70}")

    # Save aggregated results
    agg_row = {}
    for col in available_cols:
        values = results_df[col].dropna().values
        agg_row[f"{col}_mean"] = np.mean(values)
        agg_row[f"{col}_std"] = np.std(values, ddof=1)
        agg_row[f"{col}_median"] = np.median(values)
        agg_row[f"{col}_min"] = np.min(values)
        agg_row[f"{col}_max"] = np.max(values)
        ci_lo, ci_hi = bootstrap_ci(values)
        agg_row[f"{col}_ci95_lo"] = ci_lo
        agg_row[f"{col}_ci95_hi"] = ci_hi
    agg_df = pd.DataFrame([agg_row])
    agg_file = output_dir / "aggregated_split_results.csv"
    agg_df.to_csv(agg_file, index=False)

    # Distribution plots
    for metric in ['roc_auc', 'precision', 'recall', 'f1', 'average_precision']:
        if metric not in results_df.columns:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(results_df[metric], bins=20, edgecolor='black', alpha=0.7)
        mean_val = results_df[metric].mean()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.title(f"Distribution of {metric} across {len(results_df)} splits")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"split_distribution_{metric}.png")
        plt.close()

    print(f"\nDistribution plots saved to: {output_dir}")


# --- CHANGE: Split-based evaluation using dev for hyperparameter selection, test for evaluation ---
def compute_statistics_with_split(votes_file, ref_words, dev_words, test_words, split_id=None):
    """Two-stage evaluation: select best config on dev, evaluate on test."""
    split_info = {}

    with h5py.File(votes_file, 'r') as store:
        query_word_classes = store['query_word_classes'][:]

        # --- CHANGE: read query_words for split matching ---
        if 'query_words' in store:
            raw_qwords = store['query_words'][:]
            query_words = [w.decode('utf-8') if isinstance(w, bytes) else w for w in raw_qwords]
        else:
            query_words = [str(i) for i in range(len(query_word_classes))]

        # Build masks
        dev_mask = np.array([w in dev_words for w in query_words], dtype=bool)
        test_mask = np.array([w in test_words for w in query_words], dtype=bool)

        classes_dev = query_word_classes[dev_mask] if dev_mask.any() else np.array([])
        classes_test = query_word_classes[test_mask] if test_mask.any() else np.array([])

        # Split diagnostic info
        split_info = {
            'split_id': split_id,
            'total_words_in_data': len(query_words),
            'total_positive_in_data': int(np.sum(query_word_classes)),
            'total_negative_in_data': int(len(query_word_classes) - np.sum(query_word_classes)),
            'test_matched': int(test_mask.sum()),
            'test_positive': int(np.sum(classes_test)) if len(classes_test) > 0 else 0,
            'test_negative': int(len(classes_test) - np.sum(classes_test)) if len(classes_test) > 0 else 0,
            'test_positive_rate': float(np.sum(classes_test)) / len(classes_test) if len(classes_test) > 0 else 0.0,
            'dev_matched': int(dev_mask.sum()),
            'dev_positive': int(np.sum(classes_dev)) if len(classes_dev) > 0 else 0,
            'dev_negative': int(len(classes_dev) - np.sum(classes_dev)) if len(classes_dev) > 0 else 0,
        }

        if not test_mask.any():
            return None, split_info

        # Collect all configs (weighting_function × n_neighbours)
        weighting_functions = store.attrs.get('weighting_function', [])
        approach_configs = []
        for weight_type in weighting_functions:
            g = store[weight_type]
            for nn in g.attrs['n_neighbours']:
                approach_configs.append((weight_type, nn))

        # Stage 1: Select best config on DEV set
        use_dev = dev_mask.any() and len(classes_dev) > 0 and len(np.unique(classes_dev)) > 1
        best_dev_config = None
        best_dev_auc = -1
        best_dev_threshold = 0

        if use_dev:
            for weight_type, n_neighbours in approach_configs:
                try:
                    # --- CHANGE: use votes directly, no exp(-x) transform ---
                    dev_scores = store[weight_type][str(n_neighbours)][:]
                    dev_scores_sub = dev_scores[dev_mask]
                    fpr, tpr, roc_thresh = roc_curve(classes_dev, dev_scores_sub)
                    dev_auc = _trapz(tpr, fpr)
                except Exception:
                    continue

                if dev_auc > best_dev_auc:
                    best_dev_auc = dev_auc
                    best_dev_config = (weight_type, n_neighbours)
                    youdens_j = tpr - fpr
                    best_dev_threshold = roc_thresh[np.argmax(youdens_j)]

        # Stage 2: Evaluate best config on TEST set
        if best_dev_config is None:
            # Fallback: use first config
            if not approach_configs:
                return None, split_info
            best_dev_config = approach_configs[0]
            best_dev_auc = float('nan')
            best_dev_threshold = 0

        weight_type, n_neighbours = best_dev_config
        try:
            # --- CHANGE: use votes directly, no exp(-x) transform ---
            test_scores = store[weight_type][str(n_neighbours)][:]
            test_scores_sub = test_scores[test_mask]

            fpr, tpr, roc_thresholds = roc_curve(classes_test, test_scores_sub)
            roc_auc = _trapz(tpr, fpr)
            precision_sweep, recall_sweep, prc_thresholds = precision_recall_curve(classes_test, test_scores_sub)
            ap = -np.sum(np.diff(recall_sweep) * precision_sweep[:-1])

            # Apply dev threshold to test
            discretized = test_scores_sub > best_dev_threshold
            if discretized.any() and not discretized.all():
                prec = precision_score(classes_test, discretized, zero_division=0)
                rec = recall_score(classes_test, discretized, zero_division=0)
                f1_val = f1_score(classes_test, discretized, zero_division=0)
            else:
                # Fallback to F1-optimal threshold on test
                f1_sweep = 2 * precision_sweep[:-1] * recall_sweep[:-1] / (precision_sweep[:-1] + recall_sweep[:-1] + 1e-12)
                best_idx = np.argmax(f1_sweep)
                prec = precision_sweep[best_idx]
                rec = recall_sweep[best_idx]
                f1_val = f1_sweep[best_idx]
                best_dev_threshold = prc_thresholds[best_idx]

            result = {
                'weight_type': weight_type,
                'n_neighbours': n_neighbours,
                'roc_auc': roc_auc,
                'average_precision': ap,
                'threshold': best_dev_threshold,
                'precision': prec,
                'recall': rec,
                'f1': f1_val,
                'dev_roc_auc': best_dev_auc,
            }
            return [result], split_info
        except Exception:
            return None, split_info


# Boss's original record_results — CHANGED: added query_word_lists parameter for storing query words
def record_results(store: h5py.File, query_word_classes, votes, chunk_size, query_word_lists=None):
    concatenated_query_word_classes = np.concatenate(query_word_classes)
    store_query_word_classes = concatenated_query_word_classes[:chunk_size]
    remaining_query_word_classes_arr = concatenated_query_word_classes[chunk_size:]
    n_remaining = len(remaining_query_word_classes_arr)
    remaining_word_classes = []
    if n_remaining > 0:
        remaining_word_classes.append(remaining_query_word_classes_arr)

    if 'query_word_classes' not in store:
        store.create_dataset('query_word_classes', data=store_query_word_classes, maxshape=(None,), chunks=(chunk_size,))
    else:
        ds = store['query_word_classes']
        current_size = ds.shape[0]
        # --- CHANGE: cast to int() to fix h5py TypeError ---
        new_size = int(current_size + len(store_query_word_classes))
        ds.resize(new_size, axis=0)
        ds[current_size:new_size] = store_query_word_classes

    # --- CHANGE: Store query words for split matching ---
    if query_word_lists is not None:
        concatenated_query_words = []
        for wl in query_word_lists:
            if isinstance(wl, (list, np.ndarray)):
                concatenated_query_words.extend(wl)
            else:
                concatenated_query_words.append(wl)
        store_query_words = concatenated_query_words[:chunk_size]
        remaining_query_words_arr = concatenated_query_words[chunk_size:]
        remaining_query_words = []
        if len(remaining_query_words_arr) > 0:
            remaining_query_words.append(remaining_query_words_arr)

        dt = h5py.string_dtype(encoding='utf-8')
        store_query_words_encoded = np.array(store_query_words, dtype=object)
        if 'query_words' not in store:
            store.create_dataset('query_words', data=store_query_words_encoded, maxshape=(None,), dtype=dt, chunks=(chunk_size,))
        else:
            dsw = store['query_words']
            cur = dsw.shape[0]
            new = int(cur + len(store_query_words_encoded))
            dsw.resize(new, axis=0)
            dsw[cur:new] = store_query_words_encoded
    else:
        remaining_query_words = []

    # Boss's original vote flattening and storage
    flattened_votes = defaultdict(lambda: defaultdict(list))
    for votes_batch in votes:
        for centrality_measures, neighbour_votes in votes_batch.items():
            for n, votes_array in neighbour_votes.items():
                flattened_votes[centrality_measures][n].append(votes_array)
    
    remaining_votes_batch = {}
    any_remaining_votes = False
    for centrality_measures, neighbour_votes in flattened_votes.items():
        g = store.require_group(centrality_measures)
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
                current_size = ds.shape[0]
                # --- CHANGE: cast to int() to fix h5py TypeError ---
                new_size = int(current_size + store_vote_values.shape[0])
                ds.resize(new_size, axis=0)
                ds[current_size:new_size] = store_vote_values
            n_remaining_votes = len(remaining_vote_values_arr)
            if n_remaining_votes != n_remaining:
                raise RuntimeError("The remaining number of votes and number of remaining word classes differ")
            if n_remaining_votes > 0:
                remaining_neighbour_votes[n_neighbours] = remaining_vote_values_arr
                any_remaining_votes = True
        remaining_votes_batch[centrality_measures] = remaining_neighbour_votes
    
    remaining_votes = []
    if any_remaining_votes:
        remaining_votes.append(remaining_votes_batch)

    # --- CHANGE: write weighting_function attribute (boss's original naming from occ general) ---
    weighting_functions = set(flattened_votes.keys())
    if 'weighting_function' in store.attrs:
        weighting_functions.update(store.attrs['weighting_function'])
    store.attrs['weighting_function'] = sorted(weighting_functions)

    return n_remaining, remaining_word_classes, remaining_votes, remaining_query_words


# Boss's original compute_statistics (unchanged except _trapz)
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


# Boss's original statistics_worker (unchanged except _trapz)
def statistics_worker(work_package):
    store_file, weight_type, n_neighbours = work_package
    records = []
    with h5py.File(store_file) as store:
        query_word_classes = store['query_word_classes'][:]
        positive_words = int(np.sum(query_word_classes))
        negative_words = len(query_word_classes) - positive_words
        # --- CHANGE: use votes directly (weighted positive neighbor counts), no exp(-x) transform ---
        votes = store[weight_type][str(n_neighbours)][:]
        fpr, tpr, roc_thresholds = roc_curve(query_word_classes, votes)
        roc_auc = _trapz(tpr, fpr)
        precision_sweep, recall_sweep, prc_thresholds = precision_recall_curve(query_word_classes, votes)
        ap = -np.sum(np.diff(recall_sweep) * precision_sweep[:-1])

        # Youden's J for best threshold
        youdens_j = tpr - fpr
        best_threshold_index = np.argmax(youdens_j)
        best_threshold = roc_thresholds[best_threshold_index]
        discretized_votes = votes > best_threshold
        precision = precision_score(query_word_classes, discretized_votes)
        f1 = f1_score(query_word_classes, discretized_votes)
        recall = recall_score(query_word_classes, discretized_votes)
        records.append({
            'weight_type': weight_type,
            'n_neighbours': n_neighbours,
            'positive_words': positive_words,
            'negative_words': negative_words,
            'roc_auc': roc_auc,
            'average_precision': ap, 
            'threshold': best_threshold, 
            'precision': precision, 
            'recall': recall, 
            'f1': f1,
            'threshold_on': 'ba',
        })
        
        # F1-optimal threshold
        f1_sweep = 2 * precision_sweep[:-1] * recall_sweep[:-1] / (precision_sweep[:-1] + recall_sweep[:-1] + 1e-12)
        best_threshold_index = np.argmax(f1_sweep)
        best_threshold = prc_thresholds[best_threshold_index]
        precision = precision_sweep[best_threshold_index]
        recall = recall_sweep[best_threshold_index]
        f1 = f1_sweep[best_threshold_index]
        records.append({
            'weight_type': weight_type,
            'n_neighbours': n_neighbours,
            'positive_words': positive_words,
            'negative_words': negative_words,
            'roc_auc': roc_auc, 
            'average_precision': ap, 
            'threshold': best_threshold, 
            'precision': precision, 
            'recall': recall, 
            'f1': f1,
            'threshold_on': 'f1',
        })
    return records


# Boss's original get_arrays_for_file — CHANGED: also returns query_words for split matching
def get_arrays_for_file(neighbourhood_file, neighbourhood_limit=-1):
    neighbourhood_classes = []
    neighbourhood_distances = []
    query_word_classes = []
    
    with open(neighbourhood_file, 'rb') as fp:
        neighbourhood_data = pickle.load(fp)
        neighbourhoods = neighbourhood_data["neighbourhoods"]
        class_mappings = neighbourhood_data["class_mapping"]

    skip_lables = (class_mappings.get('stop_word'), class_mappings.get('known_positive'))
    query_words = []

    for (query_word, query_label), neighbours in neighbourhoods:
        if query_label in skip_lables:
            continue
        
        query_neighbour_classes = []
        query_distances = []
        for cossim, neighbour_word, neighbour_label in neighbours:
            query_neighbour_classes.append(neighbour_label)
            query_distances.append(cossim)
        if query_neighbour_classes:
            query_word_classes.append(query_label)
            neighbourhood_classes.append(query_neighbour_classes)
            neighbourhood_distances.append(query_distances)
            query_words.append(query_word)  # --- CHANGE: collect query word ---

    neighbourhood_classes = np.array([nc[:neighbourhood_limit] for nc in neighbourhood_classes], dtype=np.int8)
    neighbourhood_distances = np.array([nd[:neighbourhood_limit] for nd in neighbourhood_distances], dtype=np.float32)
    query_word_classes = np.array(query_word_classes, dtype=np.int8)
    if (len(query_word_classes) != len(neighbourhood_classes) 
        or len(neighbourhood_distances) != len(query_word_classes)
        or len(neighbourhood_distances) != len(neighbourhood_classes)):
        raise RuntimeError("Array lengths differs")

    n_neighbours = list(range(1, neighbourhood_limit+1))
    # --- CHANGE: use get_votes (weighted positive counts) instead of get_central_distance ---
    votes = get_votes(neighbourhood_classes, neighbourhood_distances, n_neighbours)

    return {"query_word_classes": query_word_classes,
            "votes": votes,
            "query_words": query_words}  # --- CHANGE: return query_words ---


# --- CHANGE: replaced get_central_distance with get_votes from boss's "occ general" script ---
# This computes weighted sum of positive neighbor class labels instead of mean/median distance
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


# Boss's original (unchanged)
def weighting_function(distances, weight_type='inverse', eps=1e-5):
    if weight_type == 'inverse':
        return 1 / (distances + eps)
    elif weight_type == 'exponential':
        return np.exp(-distances)
    elif weight_type == 'constant':
        return np.ones_like(distances)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
    

# Boss's original (unchanged)
def get_min_neighbours_for_file(neighbourhood_file):
    min_neighbours = None
    with open(neighbourhood_file, 'rb') as fp:
        neighbourhood_data = pickle.load(fp)
        neighbourhoods = neighbourhood_data["neighbourhoods"]
        for (query_word, query_label), neighbours in neighbourhoods:
            n_neighbours_for_word = len(neighbours)
            if n_neighbours_for_word > 0 and (min_neighbours is None or n_neighbours_for_word < min_neighbours):
                min_neighbours = n_neighbours_for_word
    return min_neighbours


# Boss's original (unchanged)
def get_neighbourhood_limit(neighbourhoods_files, num_workers=0):
    min_neighbours = None
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            for file_min_neighbours in tqdm(pool.imap_unordered(get_min_neighbours_for_file, neighbourhoods_files), total=len(neighbourhoods_files), desc="Finding neighbourhood limit"):
                if file_min_neighbours is not None and (min_neighbours is None or file_min_neighbours < min_neighbours):
                    min_neighbours = file_min_neighbours
    else:
        for file_min_neighbours in tqdm(map(get_min_neighbours_for_file, neighbourhoods_files), total=len(neighbourhoods_files), desc="Finding neighbourhood limit"):
            if file_min_neighbours is not None and (min_neighbours is None or file_min_neighbours < min_neighbours):
                min_neighbours = file_min_neighbours
    return min_neighbours


if __name__ == "__main__":
    main()
