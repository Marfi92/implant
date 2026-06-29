python3 analyze_neighbourhoods_split_ooc_vot.py \
  /home/abragam23/federatedhealth_20250617/results_nov12_2025/2f3ecdb8-c55a-46e1-9853-d043448e8d25/local_test_results/vector_database_FL_global_model_10/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods \
  --official --splits-file /home/abragam23/fedhealth_data/implant_split_official.json --stop_list /home/abragam23/fedhealth_data/stop_list_freq_1.txt

python3 compare_all.py \
  /home/abragam23/federatedhealth_20250617/results_nov12_2025/2f3ecdb8-c55a-46e1-9853-d043448e8d25/local_test_results/vector_database_FL_global_model_10/lancedb_direct \
  --store /home/abragam23/federatedhealth_20250617/results_nov12_2025/2f3ecdb8-c55a-46e1-9853-d043448e8d25/local_test_results/vector_database_FL_global_model_10/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/analysis/neighbour_raw_*.h5 \
  --splits-file /home/abragam23/fedhealth_data/implant_split_official.json --stop_list /home/abragam23/fedhealth_data/stop_list_freq_1.txt






mkdir -p /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal15
cp /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official2/neighbour_raw_*.h5 /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal15/
python analyze_neighbourhoods_split_ooc_vot_OFFICIAL_5.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --splits-file /home/abragam23/fedhealth_data/implant_split_official.json --official --stop_list /home/abragam23/fedhealth_data/stop_list_freq_1.txt --neg_ratio 15 --output-dir /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal15/


python analyze_neighbourhoods_split_ooc_vot_OFFICIAL_5.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --splits-file /home/abragam23/fedhealth_data/implant_split_official.json --official --stop_list /home/abragam23/fedhealth_data/stop_list_freq_1.txt --neg_ratio 5 --output-dir /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal05/


head -2 /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal05/split_info.csv
python analyze_neighbourhoods_split_ooc_vot_OFFICIAL_5.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --splits-file /home/abragam23/fedhealth_data/implant_split_official.json --official --stop_list /home/abragam23/fedhealth_data/stop_list_freq_1.txt --neg_ratio 5 --output-dir /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal05/

mkdir -p /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal05
cp /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official2/neighbour_raw_*.h5 /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal05/
python analyze_neighbourhoods_split_ooc_vot_OFFICIAL_5.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --splits-file /home/abragam23/fedhealth_data/implant_split_official.json --official --stop_list /home/abragam23/fedhealth_data/stop_list_freq_1.txt --neg_ratio 5 --output-dir /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal05/


mkdir -p /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal10
cp /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official2/neighbour_raw_*.h5 /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal10/
python analyze_neighbourhoods_split_ooc_vot_OFFICIAL_1.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --splits-file /home/abragam23/fedhealth_data/implant_split_official.json --official --stop_list /home/abragam23/fedhealth_data/stop_list_freq_1.txt --neg_ratio 10 --output-dir /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_official_bal10/

#!/usr/bin/env python3
# Step 1: Generate stratified splits (takes ~5 seconds, reads existing HDF5)
#python analyze_neighbourhoods_split_ooc_vot.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --make-stratified-splits --split_output /home/abragam23/fedhealth_data/implant_split_stratified.json
#python analyze_neighbourhoods_split_ooc_vot_3.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --splits-file /home/abragam23/fedhealth_data/implant_split_stratified.json
python analyze_neighbourhoods_split_ooc_vot_OFFICIAL.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --output-dir /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_balanced2/ --recalculate

python analyze_neighbourhoods_split_ooc_vot_OFFICIAL.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --make-stratified-splits --master_list /home/abragam23/fedhealth_data/Glossary_updated_master.txt --stop_list /home/abragam23/fedhealth_data/stop_list_freq_1.txt --neg_ratio 10 --output-dir /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_balanced2/ --split_output /home/abragam23/fedhealth_data/implant_split_balanced10.json

python analyze_neighbourhoods_split_ooc_vot_OFFICIAL.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --splits-file /home/abragam23/fedhealth_data/implant_split_balanced10.json --output-dir /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/analysis_balanced2/

python analyze_neighbourhoods_split_ooc_vot_3.py \
  /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ \
  --make-stratified-splits \
  --master_list /home/abragam23/fedhealth_data/Glossary_updated_master.txt \
  --stop_list /home/abragam23/fedhealth_data/stop_list_freq_1.txt \
  --neg_ratio 10 \
  --split_output /home/abragam23/fedhealth_data/implant_split_balanced.json

python analyze_neighbourhoods_split_ooc_vot_4.py \
  /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ \
  --splits-file /home/abragam23/fedhealth_data/implant_split_balanced.json

#python analyze_neighbourhoods_split_ooc_vot.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --make-stratified-splits --master_list /home/abragam23/fedhealth_data/Glossary_updated_master.txt --stop_list /home/abragam23/fedhealth_data/stop_list_freq_1.txt --neg_ratio 10 --split_output /home/abragam23/fedhealth_data/implant_split_balanced.json
python analyze_neighbourhoods_split_ooc_vot.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --make-stratified-splits --master_list /home/abragam23/fedhealth_data/Glossary_updated_master.txt --stop_list /home/abragam23/fedhealth_data/stop_list_freq_1.txt --neg_ratio 10 --split_output /home/abragam23/fedhealth_data/implant_split_balanced.json


# Step 2: Run evaluation with the new stratified splits
#python analyze_neighbourhoods_split_ooc_vot.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/ --splits-file /home/abragam23/fedhealth_data/implant_split_stratified.json
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
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
 
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
            all_data = []  # holds both votes and distances per batch
            for batch in tqdm(dataloader, desc="Reading neighbourhoods"):
                for example in batch:
                    batch_query_words = example.get('query_words', [])
                    batch_query_word_classes = example['query_word_classes']
                    n_batch_words = len(batch_query_word_classes)
                    n_words += n_batch_words
 
                    query_word_lists.append(batch_query_words)
                    query_word_classes.append(batch_query_word_classes)
                    all_data.append({
                        'votes': example['votes'],
                        'distances': example['distances'],
                    })
 
                    if n_words >= args.chunk_size:
                        n_words, query_word_classes, all_data, query_word_lists = record_results(
                            store, query_word_classes, all_data, args.chunk_size, query_word_lists
                        )
 
            # flush remaining
            record_results(store, query_word_classes, all_data, args.chunk_size, query_word_lists)
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
 
        # ---- Print split summary info ----
        split_sizes = {'ref': [], 'dev': [], 'test': []}
        for split_id, split_data in splits.items():
            for key in ('ref', 'dev', 'test'):
                split_sizes[key].append(len(split_data[key]))
        print(f"\n{'='*70}")
        print(f"SPLIT FILE SUMMARY")
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
                votes_file,
                ref_words=ref_words,
                dev_words=dev_words,
                test_words=test_words,
                split_id=int(split_id),
                num_workers=args.num_workers
            )
 
            all_split_info.append(split_info)
 
            if split_results:
                for result in split_results:
                    result['split_id'] = int(split_id)
                all_split_results.extend(split_results)
 
        # ---- Print split diagnostic info ----
        if all_split_info:
            info_df = pd.DataFrame(all_split_info)
            split_info_file = output_dir / "split_info.csv"
            info_df.to_csv(split_info_file, index=False)
 
            print(f"\n{'='*70}")
            print(f"SPLIT DIAGNOSTIC INFO (how splits map to data)")
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
 
            print(f"\n  Per-split ref/dev set statistics:")
            for col, label in [('ref_matched', 'Ref words matched'),
                               ('ref_positive', 'Ref positives'),
                               ('ref_negative', 'Ref negatives'),
                               ('dev_matched', 'Dev words matched'),
                               ('dev_positive', 'Dev positives'),
                               ('dev_negative', 'Dev negatives')]:
                if col in info_df.columns:
                    vals = info_df[col].values
                    print(f"    {label:>25s}: min={np.min(vals):>6.0f}  max={np.max(vals):>6.0f}  "
                          f"mean={np.mean(vals):>8.1f}  median={np.median(vals):>8.1f}")
 
            print(f"\n  Split info saved to: {split_info_file}")
            print(f"{'='*70}")
 
        if all_split_results:
            results_df = pd.DataFrame(all_split_results)
            per_split_file = output_dir / "per_split_results.csv"
            results_df.to_csv(per_split_file, index=False)
            print(f"\nPer-split results saved to: {per_split_file}")
 
            # Print results grouped by approach
            for approach in sorted(results_df['approach'].unique()):
                approach_df = results_df[results_df['approach'] == approach]
                print(f"\n{'='*70}")
                print(f"APPROACH: {approach}")
                print(f"{'='*70}")
 
                # Per-split table
                has_dev_auc = 'dev_roc_auc' in approach_df.columns and approach_df['dev_roc_auc'].notna().any()
                header = (f"{'Split':>6s}  {'Weight':>12s}  {'n_neigh':>7s}  "
                          f"{'ROC_AUC':>8s}  {'Precision':>9s}  {'Avg_Prec':>8s}  "
                          f"{'Recall':>7s}  {'F1':>7s}  "
                          f"{'TestPos':>7s}  {'TestNeg':>7s}  "
                          f"{'ScPos_mu':>8s}  {'ScNeg_mu':>8s}")
                if has_dev_auc:
                    header += f"  {'DevAUC':>8s}"
                print(header)
                print("-"*130)
                for _, row in approach_df.sort_values('split_id').iterrows():
                    line = (f"{int(row['split_id']):>6d}  "
                            f"{row['weight_type']:>12s}  "
                            f"{int(row['n_neighbours']):>7d}  "
                            f"{row['roc_auc']:>8.4f}  "
                            f"{row['precision']:>9.4f}  "
                            f"{row['average_precision']:>8.4f}  "
                            f"{row.get('recall', float('nan')):>7.4f}  "
                            f"{row.get('f1', float('nan')):>7.4f}  "
                            f"{int(row.get('test_n_positive', 0)):>7d}  "
                            f"{int(row.get('test_n_negative', 0)):>7d}  "
                            f"{row.get('score_pos_mean', float('nan')):>8.4f}  "
                            f"{row.get('score_neg_mean', float('nan')):>8.4f}")
                    if has_dev_auc:
                        line += f"  {row.get('dev_roc_auc', float('nan')):>8.4f}"
                    print(line)
                print("-"*130)
 
                # Aggregated stats with min/max
                numeric_cols = ['roc_auc', 'dev_roc_auc', 'precision', 'recall', 'f1', 'average_precision', 'threshold']
                available_cols = [c for c in numeric_cols if c in approach_df.columns and approach_df[c].notna().any()]
 
                print(f"\n  AGGREGATED ({approach}):")
                for col in available_cols:
                    values = approach_df[col].dropna().values
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)
                    median_val = np.median(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    ci_lo, ci_hi = bootstrap_ci(values)
                    print(f"    {col:>20s}: mean={mean_val:.4f} ± {std_val:.4f}  "
                          f"median={median_val:.4f}  min={min_val:.4f}  max={max_val:.4f}  "
                          f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
 
                # Score separation diagnostics
                if 'score_pos_mean' in approach_df.columns:
                    pos_means = approach_df['score_pos_mean'].dropna().values
                    neg_means = approach_df['score_neg_mean'].dropna().values
                    if len(pos_means) > 0 and len(neg_means) > 0:
                        sep = pos_means - neg_means
                        print(f"\n  SCORE SEPARATION ({approach}):")
                        print(f"    Positive class score mean: {np.mean(pos_means):.6f} ± {np.std(pos_means):.6f}")
                        print(f"    Negative class score mean: {np.mean(neg_means):.6f} ± {np.std(neg_means):.6f}")
                        print(f"    Mean separation (pos-neg): {np.mean(sep):.6f} ± {np.std(sep):.6f}")
                        print(f"    Separation range:          [{np.min(sep):.6f}, {np.max(sep):.6f}]")
 
            # Save aggregated results for all approaches
            agg_rows = []
            for approach in sorted(results_df['approach'].unique()):
                approach_df = results_df[results_df['approach'] == approach]
                numeric_cols = ['roc_auc', 'precision', 'recall', 'f1', 'average_precision', 'threshold']
                available_cols = [c for c in numeric_cols if c in approach_df.columns]
                agg_row = {'approach': approach}
                for col in available_cols:
                    values = approach_df[col].dropna().values
                    agg_row[f"{col}_mean"] = np.mean(values)
                    agg_row[f"{col}_std"] = np.std(values, ddof=1)
                    agg_row[f"{col}_median"] = np.median(values)
                    agg_row[f"{col}_min"] = np.min(values)
                    agg_row[f"{col}_max"] = np.max(values)
                    ci_lo, ci_hi = bootstrap_ci(values)
                    agg_row[f"{col}_ci95_lo"] = ci_lo
                    agg_row[f"{col}_ci95_hi"] = ci_hi
                agg_rows.append(agg_row)
            agg_df = pd.DataFrame(agg_rows)
            agg_file = output_dir / "aggregated_split_results.csv"
            agg_df.to_csv(agg_file, index=False)
 
            # Distribution plots per approach
            for approach in sorted(results_df['approach'].unique()):
                approach_df = results_df[results_df['approach'] == approach]
                for metric in ['roc_auc', 'precision', 'recall', 'f1', 'average_precision']:
                    if metric not in approach_df.columns:
                        continue
                    plt.figure(figsize=(8, 5))
                    plt.hist(approach_df[metric], bins=20, edgecolor='black', alpha=0.7)
                    mean_val = approach_df[metric].mean()
                    plt.xlabel(metric)
                    plt.ylabel("Count")
                    plt.title(f"Distribution of {metric} ({approach})\n"
                              f"across {len(approach_df)} splits, Mean: {mean_val:.4f}")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(output_dir / f"split_distribution_{metric}_{approach}.png")
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
def record_results(store: h5py.File, query_word_classes, all_data, chunk_size, query_word_lists):
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
 
    # ---- Write VOTES (vote-based approach) ----
    flattened_votes = defaultdict(lambda: defaultdict(list))
    for data_batch in all_data:
        for weight_type, neighbour_votes in data_batch['votes'].items():
            for n, votes_array in neighbour_votes.items():
                flattened_votes[weight_type][n].append(votes_array)
 
    remaining_votes_batch = {}
    any_remaining_votes = False
    votes_group = store.require_group('votes')
 
    for weight_type, neighbour_votes in flattened_votes.items():
        g = votes_group.require_group(weight_type)
        all_n_neighbours = set(int(n) for n in neighbour_votes.keys())
        if 'n_neighbours' in g.attrs:
            all_n_neighbours.update(g.attrs['n_neighbours'])
        g.attrs['n_neighbours'] = sorted(all_n_neighbours)
 
        remaining_neighbour_votes = {}
        for n_neighbours, vote_value in neighbour_votes.items():
            concatenated_vote_values = np.concatenate(vote_value)
            store_vote_values = concatenated_vote_values[:chunk_size]
            remaining_vote_values_arr = concatenated_vote_values[chunk_size:]
 
            if str(n_neighbours) not in g:
                g.create_dataset(str(n_neighbours), data=store_vote_values, maxshape=(None,), chunks=(chunk_size,))
            else:
                ds = g[str(n_neighbours)]
                cur = ds.shape[0]
                new = int(cur + store_vote_values.shape[0])
                ds.resize(new, axis=0)
                ds[cur:new] = store_vote_values
 
            if len(remaining_vote_values_arr) != n_remaining:
                raise RuntimeError("The remaining number of votes and number of remaining word classes differ")
            if len(remaining_vote_values_arr) > 0:
                remaining_neighbour_votes[n_neighbours] = remaining_vote_values_arr
                any_remaining_votes = True
        remaining_votes_batch[weight_type] = remaining_neighbour_votes
 
    all_vote_types = sorted(flattened_votes.keys())
    if 'weighting_function' in votes_group.attrs:
        existing = set(votes_group.attrs['weighting_function'])
        existing.update(all_vote_types)
        all_vote_types = sorted(existing)
    votes_group.attrs['weighting_function'] = all_vote_types
 
    # ---- Write DISTANCES (distance-based approach) ----
    flattened_distances = defaultdict(lambda: defaultdict(list))
    for data_batch in all_data:
        for centrality_measure, neighbour_distances in data_batch['distances'].items():
            for n, dist_array in neighbour_distances.items():
                flattened_distances[centrality_measure][n].append(dist_array)
 
    remaining_distances_batch = {}
    any_remaining_distances = False
    dist_group = store.require_group('distances')
 
    for centrality_measure, neighbour_distances in flattened_distances.items():
        g = dist_group.require_group(centrality_measure)
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
                any_remaining_distances = True
        remaining_distances_batch[centrality_measure] = remaining_neighbour_distances
 
    all_measures = sorted(flattened_distances.keys())
    if 'centrality_measures' in dist_group.attrs:
        existing = set(dist_group.attrs['centrality_measures'])
        existing.update(all_measures)
        all_measures = sorted(existing)
    dist_group.attrs['centrality_measures'] = all_measures
 
    # Build remaining data
    remaining_all_data = []
    if any_remaining_votes or any_remaining_distances:
        remaining_all_data.append({
            'votes': remaining_votes_batch,
            'distances': remaining_distances_batch,
        })
 
    return n_remaining, remaining_word_classes, remaining_all_data, remaining_query_words
 
 
def compute_statistics(votes_file: Path, num_workers=0):
    work_packages = []
    with h5py.File(votes_file, 'r') as store:
        # Vote-based approach
        if 'votes' in store:
            vg = store['votes']
            for weight_type in vg.attrs.get('weighting_function', []):
                g = vg[weight_type]
                for n_neighbours in g.attrs['n_neighbours']:
                    work_packages.append((votes_file, 'votes', weight_type, n_neighbours))
        # Distance-based approach
        if 'distances' in store:
            dg = store['distances']
            for cm in dg.attrs.get('centrality_measures', []):
                g = dg[cm]
                for n_neighbours in g.attrs['n_neighbours']:
                    work_packages.append((votes_file, 'distances', cm, n_neighbours))
 
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            records = list(tqdm(pool.imap_unordered(statistics_worker, work_packages), total=len(work_packages)))
    else:
        records = [statistics_worker(work_package) for work_package in tqdm(work_packages)]
 
    df = pd.DataFrame.from_records([record for record_pair in records for record in record_pair])
    return df
 
 
def statistics_worker(work_package):
    store_file, approach, weight_type, n_neighbours = work_package
    records = []
    with h5py.File(store_file) as store:
        query_word_classes = store['query_word_classes'][:]
        positive_words = int(np.sum(query_word_classes))
        negative_words = len(query_word_classes) - positive_words
 
        raw_scores = store[approach][weight_type][str(n_neighbours)][:]
        if approach == 'distances':
            scores = np.exp(-raw_scores)
        else:
            scores = raw_scores
 
        fpr, tpr, roc_thresholds = roc_curve(query_word_classes, scores)
        roc_auc = _trapz(tpr, fpr)
        precision_sweep, recall_sweep, prc_thresholds = precision_recall_curve(query_word_classes, scores)
        ap = -np.sum(np.diff(recall_sweep) * precision_sweep[:-1])
 
        youdens_j = tpr - fpr
        best_threshold_index = np.argmax(youdens_j)
        best_threshold = roc_thresholds[best_threshold_index]
        discretized = scores > best_threshold
        prec = precision_score(query_word_classes, discretized)
        f1_val = f1_score(query_word_classes, discretized)
        rec = recall_score(query_word_classes, discretized)
        records.append({
            'approach': approach,
            'weight_type': weight_type,
            'n_neighbours': n_neighbours,
            'positive_words': positive_words,
            'negative_words': negative_words,
            'roc_auc': roc_auc,
            'average_precision': ap,
            'threshold': best_threshold,
            'precision': prec,
            'recall': rec,
            'f1': f1_val,
            'threshold_on': 'ba',
        })
 
        f1_sweep = 2 * precision_sweep[:-1] * recall_sweep[:-1] / (precision_sweep[:-1] + recall_sweep[:-1] + 1e-12)
        best_threshold_index = np.argmax(f1_sweep)
        best_threshold = prc_thresholds[best_threshold_index]
        prec = precision_sweep[best_threshold_index]
        rec = recall_sweep[best_threshold_index]
        f1_val = f1_sweep[best_threshold_index]
        records.append({
            'approach': approach,
            'weight_type': weight_type,
            'n_neighbours': n_neighbours,
            'positive_words': positive_words,
            'negative_words': negative_words,
            'roc_auc': roc_auc,
            'average_precision': ap,
            'threshold': best_threshold,
            'precision': prec,
            'recall': rec,
            'f1': f1_val,
            'threshold_on': 'f1',
        })
    return records
 
 
# -------------------------
# Split-based evaluation helper
# -------------------------
def _evaluate_config_on_subset(store, approach, weight_type, n_neighbours, mask, classes):
    """Evaluate a single (approach, weight_type, n_neighbours) config on a data subset."""
    raw_scores = store[approach][weight_type][str(n_neighbours)][:]
    if approach == 'distances':
        scores = np.exp(-raw_scores)
    else:
        scores = raw_scores
 
    scores_sub = scores[mask]
    pos_scores = scores_sub[classes == 1]
    neg_scores = scores_sub[classes == 0]
 
    fpr, tpr, roc_thresholds = roc_curve(classes, scores_sub)
    roc_auc = _trapz(tpr, fpr)
    precision_sweep, recall_sweep, prc_thresholds = precision_recall_curve(classes, scores_sub)
    ap = -np.sum(np.diff(recall_sweep) * precision_sweep[:-1])
 
    f1_sweep = 2 * precision_sweep[:-1] * recall_sweep[:-1] / (precision_sweep[:-1] + recall_sweep[:-1] + 1e-12)
    best_f1_idx = np.argmax(f1_sweep)
    best_threshold = prc_thresholds[best_f1_idx]
    prec = precision_sweep[best_f1_idx]
    rec = recall_sweep[best_f1_idx]
    f1_val = f1_sweep[best_f1_idx]
 
    return {
        'roc_auc': roc_auc,
        'average_precision': ap,
        'threshold': best_threshold,
        'precision': prec,
        'recall': rec,
        'f1': f1_val,
        'scores_sub': scores_sub,
        'pos_scores': pos_scores,
        'neg_scores': neg_scores,
    }
 
 
def compute_statistics_with_split(votes_file, ref_words, dev_words, test_words, split_id=None, num_workers=0):
    results = []
    split_info = {}
 
    with h5py.File(votes_file, 'r') as store:
        query_word_classes = store['query_word_classes'][:]
        raw_qwords = store['query_words'][:]
        query_words = [w.decode('utf-8') if isinstance(w, bytes) else w for w in raw_qwords]
 
        # Build masks for ref/dev/test
        ref_mask = np.array([w in ref_words for w in query_words], dtype=bool)
        dev_mask = np.array([w in dev_words for w in query_words], dtype=bool)
        test_mask = np.array([w in test_words for w in query_words], dtype=bool)
 
        # Collect split info
        classes_ref = query_word_classes[ref_mask] if ref_mask.any() else np.array([])
        classes_dev = query_word_classes[dev_mask] if dev_mask.any() else np.array([])
        classes_test = query_word_classes[test_mask] if test_mask.any() else np.array([])
 
        split_info = {
            'split_id': split_id,
            'total_words_in_data': len(query_words),
            'total_positive_in_data': int(np.sum(query_word_classes)),
            'total_negative_in_data': int(len(query_word_classes) - np.sum(query_word_classes)),
            'ref_n_words': int(len(ref_words)),
            'ref_matched': int(ref_mask.sum()),
            'ref_positive': int(np.sum(classes_ref)) if len(classes_ref) > 0 else 0,
            'ref_negative': int(len(classes_ref) - np.sum(classes_ref)) if len(classes_ref) > 0 else 0,
            'dev_n_words': int(len(dev_words)),
            'dev_matched': int(dev_mask.sum()),
            'dev_positive': int(np.sum(classes_dev)) if len(classes_dev) > 0 else 0,
            'dev_negative': int(len(classes_dev) - np.sum(classes_dev)) if len(classes_dev) > 0 else 0,
            'test_n_words': int(len(test_words)),
            'test_matched': int(test_mask.sum()),
            'test_positive': int(np.sum(classes_test)) if len(classes_test) > 0 else 0,
            'test_negative': int(len(classes_test) - np.sum(classes_test)) if len(classes_test) > 0 else 0,
        }
        if len(classes_test) > 0:
            split_info['test_positive_rate'] = float(np.sum(classes_test)) / len(classes_test)
        else:
            split_info['test_positive_rate'] = 0.0
 
        if not test_mask.any():
            return None, split_info
 
        # Collect all approach configs
        approach_configs = []
        if 'votes' in store:
            vg = store['votes']
            for wt in vg.attrs.get('weighting_function', []):
                g = vg[wt]
                for nn in g.attrs['n_neighbours']:
                    approach_configs.append(('votes', wt, nn))
        if 'distances' in store:
            dg = store['distances']
            for cm in dg.attrs.get('centrality_measures', []):
                g = dg[cm]
                for nn in g.attrs['n_neighbours']:
                    approach_configs.append(('distances', cm, nn))
 
        # ---- STAGE 1: Select best config per approach using DEV set ----
        best_dev_config = {}  # approach -> (weight_type, n_neighbours, dev_auc, dev_threshold)
        use_dev = dev_mask.any() and len(classes_dev) > 0 and len(np.unique(classes_dev)) > 1
 
        if use_dev:
            for approach, weight_type, n_neighbours in approach_configs:
                try:
                    dev_result = _evaluate_config_on_subset(
                        store, approach, weight_type, n_neighbours, dev_mask, classes_dev
                    )
                    key = approach
                    if key not in best_dev_config or dev_result['roc_auc'] > best_dev_config[key][2]:
                        best_dev_config[key] = (weight_type, n_neighbours, dev_result['roc_auc'], dev_result['threshold'])
                except Exception:
                    continue
 
        # ---- STAGE 2: Evaluate best config on TEST set ----
        best_per_approach = {}
 
        if use_dev and best_dev_config:
            # Proper two-stage: use dev-selected config, apply dev threshold to test
            for approach, (best_wt, best_nn, dev_auc, dev_threshold) in best_dev_config.items():
                try:
                    test_result = _evaluate_config_on_subset(
                        store, approach, best_wt, best_nn, test_mask, classes_test
                    )
                    # Also compute precision/recall/f1 using the dev-optimized threshold
                    scores_test = test_result['scores_sub']
                    dev_discretized = scores_test > dev_threshold
                    if dev_discretized.any() and not dev_discretized.all():
                        dev_prec = precision_score(classes_test, dev_discretized, zero_division=0)
                        dev_rec = recall_score(classes_test, dev_discretized, zero_division=0)
                        dev_f1 = f1_score(classes_test, dev_discretized, zero_division=0)
                    else:
                        dev_prec = test_result['precision']
                        dev_rec = test_result['recall']
                        dev_f1 = test_result['f1']
 
                    best_per_approach[approach] = {
                        'approach': approach,
                        'weight_type': best_wt,
                        'n_neighbours': best_nn,
                        'roc_auc': test_result['roc_auc'],
                        'average_precision': test_result['average_precision'],
                        'threshold': dev_threshold,
                        'precision': dev_prec,
                        'recall': dev_rec,
                        'f1': dev_f1,
                        'dev_roc_auc': dev_auc,
                        'test_n_positive': int(len(test_result['pos_scores'])),
                        'test_n_negative': int(len(test_result['neg_scores'])),
                        'score_pos_mean': float(np.mean(test_result['pos_scores'])) if len(test_result['pos_scores']) > 0 else float('nan'),
                        'score_neg_mean': float(np.mean(test_result['neg_scores'])) if len(test_result['neg_scores']) > 0 else float('nan'),
                        'score_pos_std': float(np.std(test_result['pos_scores'])) if len(test_result['pos_scores']) > 0 else float('nan'),
                        'score_neg_std': float(np.std(test_result['neg_scores'])) if len(test_result['neg_scores']) > 0 else float('nan'),
                    }
                except Exception:
                    continue
        else:
            # Fallback: no usable dev set, evaluate all configs on test (old behavior)
            for approach, weight_type, n_neighbours in approach_configs:
                try:
                    test_result = _evaluate_config_on_subset(
                        store, approach, weight_type, n_neighbours, test_mask, classes_test
                    )
                    key = approach
                    if key not in best_per_approach or test_result['roc_auc'] > best_per_approach[key]['roc_auc']:
                        best_per_approach[key] = {
                            'approach': approach,
                            'weight_type': weight_type,
                            'n_neighbours': n_neighbours,
                            'roc_auc': test_result['roc_auc'],
                            'average_precision': test_result['average_precision'],
                            'threshold': test_result['threshold'],
                            'precision': test_result['precision'],
                            'recall': test_result['recall'],
                            'f1': test_result['f1'],
                            'dev_roc_auc': float('nan'),
                            'test_n_positive': int(len(test_result['pos_scores'])),
                            'test_n_negative': int(len(test_result['neg_scores'])),
                            'score_pos_mean': float(np.mean(test_result['pos_scores'])) if len(test_result['pos_scores']) > 0 else float('nan'),
                            'score_neg_mean': float(np.mean(test_result['neg_scores'])) if len(test_result['neg_scores']) > 0 else float('nan'),
                            'score_pos_std': float(np.std(test_result['pos_scores'])) if len(test_result['pos_scores']) > 0 else float('nan'),
                            'score_neg_std': float(np.std(test_result['neg_scores'])) if len(test_result['neg_scores']) > 0 else float('nan'),
                        }
                except Exception:
                    continue
 
        results = list(best_per_approach.values())
 
    return (results if results else None), split_info
 
 
# -------------------------
# File parsing and feature extraction
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
 
    # Compute ALL approaches: votes (raw + normalized) and distances
    votes = get_votes(neighbourhood_classes, neighbourhood_distances, n_neighbours)
    distances = get_central_distance(neighbourhood_classes, neighbourhood_distances, n_neighbours)
 
    return {
        "query_words": query_words,
        "query_word_classes": query_word_classes,
        "votes": votes,
        "distances": distances,
    }
 
 
def weighting_function(distances, weight_type='inverse', eps=1e-5):
    if weight_type == 'inverse':
        return 1 / (distances + eps)
    elif weight_type == 'exponential':
        return np.exp(-distances)
    elif weight_type == 'constant':
        return np.ones_like(distances)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
 
 
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
 
        # Normalized variant: divide weighted positive votes by total weight sum -> [0,1] probability
        norm_key = f"{weight_type}_norm"
        norm_votes = {}
        weight_cumsums = np.cumsum(neighbour_weights, axis=1)
        for i in n_neighbours:
            norm_votes[i] = neighbour_cumsums[:, i-1] / (weight_cumsums[:, i-1] + 1e-12)
        votes[norm_key] = norm_votes
    return votes
 
 
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
 
