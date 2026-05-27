#!/usr/bin/env python3
"""Diagnostic script to investigate low ROC AUC values.

Run from abragam's machine:
    python diagnose_auc.py <path-to-votes-h5-file> --splits-file <path-to-splits-json>

Example:
    python diagnose_auc.py /home/abragam23/federatedhealth_20250617/results_nov12_2025/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19/lancedb_direct-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/analysis/neighbourhood_analysis_*.h5 --splits-file /home/abragam23/fedhealth_data/implant_split.json
"""
import argparse
import json
import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import h5py


def main():
    parser = argparse.ArgumentParser(description="Diagnose low AUC values")
    parser.add_argument('votes_file', help='Path to the HDF5 votes file (supports glob)', type=str)
    parser.add_argument('--splits-file', help='Path to splits JSON', type=Path)
    args = parser.parse_args()

    # Resolve glob
    files = glob.glob(args.votes_file)
    h5_files = [f for f in files if f.endswith('.h5')]
    if not h5_files:
        print(f"ERROR: No .h5 files found matching: {args.votes_file}")
        print("Try providing the full path to the .h5 file in the analysis/ folder")
        return
    votes_file = h5_files[0]
    print(f"Using votes file: {votes_file}\n")

    print("=" * 60)
    print("DIAGNOSTIC REPORT")
    print("=" * 60)

    with h5py.File(votes_file, 'r') as store:
        query_word_classes = store['query_word_classes'][:]
        raw_qwords = store['query_words'][:]
        query_words = [w.decode('utf-8') if isinstance(w, bytes) else w for w in raw_qwords]
        weighting_functions = list(store.attrs['weighting_function'])

        # 1. Check class labels
        print("\n--- 1. CLASS LABEL ANALYSIS ---")
        unique_labels, label_counts = np.unique(query_word_classes, return_counts=True)
        print(f"  Unique class labels: {unique_labels}")
        print(f"  Label counts: {dict(zip(unique_labels.tolist(), label_counts.tolist()))}")
        print(f"  Total samples: {len(query_word_classes)}")
        if len(unique_labels) == 2:
            pos_rate = label_counts[1] / len(query_word_classes)
            print(f"  Positive class rate: {pos_rate:.4f} ({label_counts[1]}/{len(query_word_classes)})")
            if pos_rate < 0.01 or pos_rate > 0.99:
                print("  WARNING: Highly imbalanced classes! This can affect AUC interpretation.")
        elif len(unique_labels) < 2:
            print("  ERROR: Only one class present! AUC is undefined.")
        else:
            print("  WARNING: More than 2 classes found. roc_curve expects binary labels.")

        # 2. Check vote score distributions
        print("\n--- 2. VOTE SCORE DISTRIBUTIONS ---")
        for weight_type in weighting_functions[:1]:  # Just check first weight type
            g = store[weight_type]
            n_neighbours_list = sorted(g.attrs['n_neighbours'])
            # Check a few representative n_neighbours values
            for n in [n_neighbours_list[0], n_neighbours_list[len(n_neighbours_list)//2], n_neighbours_list[-1]]:
                votes = g[str(n)][:]
                pos_mask = query_word_classes == 1
                neg_mask = query_word_classes == 0
                if pos_mask.any() and neg_mask.any():
                    print(f"\n  Weight={weight_type}, n_neighbours={n}:")
                    print(f"    Positive class votes: mean={votes[pos_mask].mean():.4f}, std={votes[pos_mask].std():.4f}, "
                          f"min={votes[pos_mask].min():.4f}, max={votes[pos_mask].max():.4f}")
                    print(f"    Negative class votes: mean={votes[neg_mask].mean():.4f}, std={votes[neg_mask].std():.4f}, "
                          f"min={votes[neg_mask].min():.4f}, max={votes[neg_mask].max():.4f}")
                    overlap = votes[pos_mask].mean() - votes[neg_mask].mean()
                    print(f"    Mean difference (pos - neg): {overlap:.4f}")
                    if abs(overlap) < 0.01:
                        print("    WARNING: Positive and negative vote distributions are nearly identical!")

        # 3. Verify AUC with sklearn directly
        print("\n--- 3. AUC VERIFICATION (full dataset, no splits) ---")
        for weight_type in weighting_functions:
            g = store[weight_type]
            n_neighbours_list = sorted(g.attrs['n_neighbours'])
            best_auc = -1
            best_n = -1
            for n in n_neighbours_list:
                votes = g[str(n)][:]
                try:
                    auc_trapz = np.trapz(*roc_curve(query_word_classes, votes)[:2][::-1])
                    auc_sklearn = roc_auc_score(query_word_classes, votes)
                    if auc_sklearn > best_auc:
                        best_auc = auc_sklearn
                        best_n = n
                except Exception:
                    pass
            print(f"  {weight_type}: best AUC = {best_auc:.4f} (at n_neighbours={best_n})")

        # 4. Check if splits reduce data too much
        if args.splits_file:
            print("\n--- 4. SPLIT-BASED ANALYSIS ---")
            with open(args.splits_file, "r") as fp:
                splits = json.load(fp)

            # Check first 3 splits
            for split_id in list(splits.keys())[:3]:
                split_data = splits[split_id]
                test_words = set(split_data["test"])
                test_mask = np.array([w in test_words for w in query_words], dtype=bool)
                n_test = test_mask.sum()
                if n_test == 0:
                    print(f"  Split {split_id}: NO test words found in data!")
                    continue

                classes_test = query_word_classes[test_mask]
                unique_test, counts_test = np.unique(classes_test, return_counts=True)
                print(f"  Split {split_id}: {n_test} test samples, labels={dict(zip(unique_test.tolist(), counts_test.tolist()))}")

                if len(unique_test) < 2:
                    print(f"    WARNING: Only one class in test split! AUC undefined.")
                    continue

                # Check AUC for this split
                for weight_type in weighting_functions[:1]:
                    g = store[weight_type]
                    n_neighbours_list = sorted(g.attrs['n_neighbours'])
                    mid_n = n_neighbours_list[len(n_neighbours_list)//2]
                    votes = g[str(mid_n)][:]
                    votes_test = votes[test_mask]
                    try:
                        auc = roc_auc_score(classes_test, votes_test)
                        print(f"    {weight_type}, n={mid_n}: AUC = {auc:.4f}")
                    except Exception as e:
                        print(f"    {weight_type}, n={mid_n}: ERROR computing AUC: {e}")

        # 5. Check for potential issues
        print("\n--- 5. POTENTIAL ISSUES ---")
        issues = []

        # Check if query_word_classes are binary
        if not set(unique_labels.tolist()).issubset({0, 1}):
            issues.append(f"Class labels are {unique_labels.tolist()}, not [0, 1]. "
                          "roc_curve may not interpret them correctly.")

        # Check for NaN/Inf in votes
        for weight_type in weighting_functions[:1]:
            g = store[weight_type]
            n_neighbours_list = sorted(g.attrs['n_neighbours'])
            votes = g[str(n_neighbours_list[0])][:]
            if np.any(np.isnan(votes)):
                issues.append(f"NaN values found in votes for {weight_type}")
            if np.any(np.isinf(votes)):
                issues.append(f"Inf values found in votes for {weight_type}")

        # Check class balance
        if len(unique_labels) == 2:
            minority_rate = min(label_counts) / sum(label_counts)
            if minority_rate < 0.05:
                issues.append(f"Severe class imbalance: minority class is only {minority_rate:.1%}")

        if issues:
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("  No obvious issues detected in the data.")
            print("  The low AUC may indicate that the voting scheme genuinely has")
            print("  limited discriminative power for this dataset/split configuration.")

    print("\n" + "=" * 60)
    print("END OF DIAGNOSTIC REPORT")
    print("=" * 60)


if __name__ == "__main__":
    main()
