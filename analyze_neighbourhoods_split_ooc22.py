cd /home/abragam23/src/aimplant/aimplant_demonstrator
cp implant/analyze_neighbourhoods_split_ooc22.py analyze_neighbourhoods_split_ooc22.py
python analyze_neighbourhoods_split_ooc22.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --splits-file /home/abragam23/fedhealth_data/implant_split.json --recalculate
#!/usr/bin/env python3
import argparse
import json
import warnings
from pathlib import Path
import pickle
from collections import defaultdict
from hashlib import md5
import multiprocessing
import random
 
warnings.filterwarnings("ignore", category=DeprecationWarning)
 
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader
import pandas as pd
 
# Use np.trapezoid if available (numpy >= 2.0), else fall back to np.trapz
_trapz = getattr(np, 'trapezoid', np.trapz)
 
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
    def __init__(self, neighbourhood_files, neighbourhood_limit=None):
        self.neighbourhood_files = sorted(neighbourhood_files)
        self.neighbourhood_limit = neighbourhood_limit
        neighbourhood_hash = md5()
        for neighbourhood_file in tqdm(self.neighbourhood_files, desc="Hashing neighbourhood files"):
            neighbourhood_hash.update(str(neighbourhood_file).encode('utf-8'))
        self.neighbourhood_hash = neighbourhood_hash.hexdigest()
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
    neighbourhood_files = sorted(args.neighbourhoods.glob("*.pkl"))
    output_dir = args.output_dir if args.output_dir else args.neighbourhoods / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
 
    neighbourhood_limit = get_neighbourhood_limit(neighbourhood_files, num_workers=args.num_workers)
    if neighbourhood_limit is None or neighbourhood_limit < 0:
        raise RuntimeError(f"Could not determine the neighbourhood limit, got {neighbourhood_limit}. "
                           "This means that all neighbourhoods are empty, which is unexpected.")
 
    neighbours_dataset = NeighbourHoodDataset(neighbourhood_files, neighbourhood_limit=neighbourhood_limit)
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
            query_word_lists = []
            distances = []
            for batch in tqdm(dataloader, desc="Reading neighbourhoods"):
                for example in batch:
                    batch_query_words = example.get('query_words', [])
                    batch_query_word_classes = example['query_word_classes']
                    n_batch_words = len(batch_query_word_classes)
                    n_words += n_batch_words
 
                    query_word_lists.append(batch_query_words)
                    query_word_classes.append(batch_query_word_classes)
                    distances.append(example['distances'])
 
                    if n_words >= args.chunk_size:
                        n_words, query_word_classes, distances, query_word_lists = record_results(
                            store, query_word_classes, distances, args.chunk_size, query_word_lists
                        )
 
            # flush remaining
            record_results(store, query_word_classes, distances, args.chunk_size, query_word_lists)
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
                plt.title(f"{eval_metric} vs Number of Neighbours (thresholded on {threshold_on})\n"
                          f"Positive: {threshold_df['positive_words'].iloc[0]}, "
                          f"Negative: {threshold_df['negative_words'].iloc[0]}")
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
                values = results_df[col].dropna().values
                agg_results[f"{col}_mean"] = np.mean(values)
                agg_results[f"{col}_std"] = np.std(values, ddof=1)
                agg_results[f"{col}_median"] = np.median(values)
                # Bootstrapped 95% CI of the mean
                ci_lo, ci_hi = bootstrap_ci(values, n_bootstrap=10000)
                agg_results[f"{col}_ci95_lo"] = ci_lo
                agg_results[f"{col}_ci95_hi"] = ci_hi
 
            agg_df = pd.DataFrame([agg_results])
            agg_file = output_dir / "aggregated_split_results.csv"
            agg_df.to_csv(agg_file, index=False)
 
            print("\n" + "="*60)
            print("AGGREGATED RESULTS ACROSS ALL SPLITS")
            print("="*60)
            for col in available_cols:
                values = results_df[col].dropna().values
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                median_val = np.median(values)
                ci_lo = agg_results[f"{col}_ci95_lo"]
                ci_hi = agg_results[f"{col}_ci95_hi"]
                print(f"  {col:>20s}: mean={mean_val:.4f} ± {std_val:.4f}  "
                      f"median={median_val:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
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
 
def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    boot_means = np.array([np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, 100 * alpha), np.percentile(boot_means, 100 * (1 - alpha))
 
 
# -------------------------
# HDF5 record / compute helpers
# -------------------------
def record_results(store: h5py.File, query_word_classes, distances, chunk_size, query_word_lists):
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
        new = int(cur + len(store_query_word_classes))
        ds.resize(new, axis=0)
        ds[cur:new] = store_query_word_classes
 
    # write / append query words
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
 
    # flatten distances (structured like votes: centrality_measure -> n_neighbours -> array)
    flattened_distances = defaultdict(lambda: defaultdict(list))
    for dist_batch in distances:
        for centrality_measure, neighbour_distances in dist_batch.items():
            for n, dist_array in neighbour_distances.items():
                flattened_distances[centrality_measure][n].append(dist_array)
 
    remaining_distances_batch = {}
    any_remaining = False
 
    for centrality_measure, neighbour_distances in flattened_distances.items():
        g = store.require_group(centrality_measure)
        all_n_neighbours = set(int(n) for n in neighbour_distances.keys())
        if 'n_neighbours' in g.attrs:
            all_n_neighbours.update(g.attrs['n_neighbours'])
        g.attrs['n_neighbours'] = sorted(all_n_neighbours)
 
        remaining_neighbour_distances = {}
        for n_neighbours, dist_value in neighbour_distances.items():
            concatenated_dist_values = np.concatenate(dist_value)
            store_dist_values = concatenated_dist_values[:chunk_size]
            remaining_dist_values_arr = concatenated_dist_values[chunk_size:]
 
            if str(n_neighbours) not in g:
                g.create_dataset(str(n_neighbours), data=store_dist_values, maxshape=(None,), chunks=(chunk_size,))
            else:
                ds = g[str(n_neighbours)]
                cur = ds.shape[0]
                new = int(cur + store_dist_values.shape[0])
                ds.resize(new, axis=0)
                ds[cur:new] = store_dist_values
 
            if len(remaining_dist_values_arr) != n_remaining:
                raise RuntimeError("The remaining number of distances and number of remaining word classes differ")
 
            if len(remaining_dist_values_arr) > 0:
                remaining_neighbour_distances[n_neighbours] = remaining_dist_values_arr
                any_remaining = True
 
        remaining_distances_batch[centrality_measure] = remaining_neighbour_distances
 
    # Store centrality measure names as root-level attribute
    all_measures = sorted(flattened_distances.keys())
    if 'centrality_measures' in store.attrs:
        existing = set(store.attrs['centrality_measures'])
        existing.update(all_measures)
        all_measures = sorted(existing)
    store.attrs['centrality_measures'] = all_measures
 
    remaining_distances = [remaining_distances_batch] if any_remaining else []
 
    return n_remaining, remaining_word_classes, remaining_distances, remaining_query_words
 
 
def compute_statistics(votes_file: Path, num_workers=0):
    work_packages = []
    with h5py.File(votes_file, 'r') as store:
        centrality_measures = store.attrs['centrality_measures']
        for weight_type in centrality_measures:
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
        positive_words = int(np.sum(query_word_classes))
        negative_words = len(query_word_classes) - positive_words
        # Transform distances to similarity scores: smaller distance -> higher score
        votes = np.exp(-store[weight_type][str(n_neighbours)][:])
        fpr, tpr, roc_thresholds = roc_curve(query_word_classes, votes)
        roc_auc = _trapz(tpr, fpr)
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
                                 'positive_words': positive_words,
                                 'negative_words': negative_words,
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
                                 'positive_words': positive_words,
                                 'negative_words': negative_words,
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
 
        centrality_measures = store.attrs['centrality_measures']
 
        for weight_type in centrality_measures:
            g = store[weight_type]
            all_n_neighbours = g.attrs['n_neighbours']
            for n_neighbours in all_n_neighbours:
                raw_distances = g[str(n_neighbours)][:]
                # Transform distances to similarity scores
                votes = np.exp(-raw_distances)
 
                votes_test = votes[test_mask]
                classes_test = query_word_classes[test_mask]
 
                try:
                    fpr, tpr, roc_thresholds = roc_curve(classes_test, votes_test)
                    roc_auc = _trapz(tpr, fpr)
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
# File parsing and distance logic
# -------------------------
def get_arrays_for_file(neighbourhood_file, neighbourhood_limit=-1):
    neighbourhood_classes = []
    neighbourhood_distances = []
    query_word_classes = []
 
    with open(neighbourhood_file, 'rb') as fp:
        neighbourhood_data = pickle.load(fp)
        neighbourhoods = neighbourhood_data["neighbourhoods"]
        class_mappings = neighbourhood_data["class_mapping"]
 
    skip_labels = (class_mappings.get('stop_word'), class_mappings.get('known_positive'))
    query_words = []
 
    for (query_word, query_label), neighbours in neighbourhoods:
        if query_label in skip_labels:
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
            query_words.append(query_word)
 
    neighbourhood_classes = np.array([nc[:neighbourhood_limit] for nc in neighbourhood_classes], dtype=np.int8)
    neighbourhood_distances = np.array([nd[:neighbourhood_limit] for nd in neighbourhood_distances], dtype=np.float32)
    query_word_classes = np.array(query_word_classes, dtype=np.int8)
    if (len(query_word_classes) != len(neighbourhood_classes)
        or len(neighbourhood_distances) != len(query_word_classes)
        or len(neighbourhood_distances) != len(neighbourhood_classes)):
        raise RuntimeError("Array lengths differs")
 
    n_neighbours = list(range(1, neighbourhood_limit + 1))
    distances = get_central_distance(neighbourhood_classes, neighbourhood_distances, n_neighbours)
 
    return {
        "query_words": query_words,
        "query_word_classes": query_word_classes,
        "distances": distances
    }
 
 
def get_central_distance(neighbourhood_classes, neighbourhood_distances, n_neighbours):
    distances = {}
    for measure, measure_fun in [('mean', np.mean), ('median', np.median)]:
        distance_per_neighbourhood = {}
        for i in n_neighbours:
            distance_per_neighbourhood[i] = measure_fun(neighbourhood_distances[:, :i], axis=1)
        distances[measure] = distance_per_neighbourhood
    return distances
 
 
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
 
 
def get_neighbourhood_limit(neighbourhood_files, num_workers=0):
    min_neighbours = None
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            for file_min_neighbours in tqdm(pool.imap_unordered(get_min_neighbours_for_file, neighbourhood_files),
                                            total=len(neighbourhood_files), desc="Finding neighbourhood limit"):
                if file_min_neighbours is not None and (min_neighbours is None or file_min_neighbours < min_neighbours):
                    min_neighbours = file_min_neighbours
    else:
        for file_min_neighbours in tqdm(map(get_min_neighbours_for_file, neighbourhood_files),
                                        total=len(neighbourhood_files), desc="Finding neighbourhood limit"):
            if file_min_neighbours is not None and (min_neighbours is None or file_min_neighbours < min_neighbours):
                min_neighbours = file_min_neighbours
    return min_neighbours
 
 
if __name__ == "__main__":
    main()
