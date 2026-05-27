import argparse
import json
import glob
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import h5py

# Use np.trapezoid if available (numpy >= 2.0), else fall back to np.trapz
_trapz = getattr(np, 'trapezoid', np.trapz)


def main():
    parser = argparse.ArgumentParser(description="Diagnose low AUC values")

    parser.add_argument(
        'votes_file',
        help='Path to the HDF5 votes file (supports glob)',
        type=str
    )

    parser.add_argument(
        '--splits-file',
        help='Optional splits JSON file',
        type=str,
        default=None
    )

    args = parser.parse_args()

    # Support glob paths
    votes_files = glob.glob(args.votes_file)

    if not votes_files:
        print("No HDF5 files found.")
        return

    votes_file = votes_files[0]

    print("=" * 60)
    print("DIAGNOSTIC REPORT")
    print("=" * 60)

    with h5py.File(votes_file, 'r') as store:

        query_word_classes = store['query_word_classes'][:]

        raw_qwords = store['query_words'][:]
        query_words = [
            w.decode('utf-8') if isinstance(w, bytes) else w
            for w in raw_qwords
        ]

        unique_labels, label_counts = np.unique(
            query_word_classes,
            return_counts=True
        )

        print("\n--- 1. CLASS DISTRIBUTION ---")
        print(dict(zip(unique_labels.tolist(), label_counts.tolist())))

        weighting_functions = store.attrs['weighting_function']

        print("\n--- 2. AUC ANALYSIS ---")

        for weight_type in weighting_functions:

            g = store[weight_type]
            n_neighbours_list = sorted(g.attrs['n_neighbours'])

            best_auc = -1
            best_n = None

            for n in n_neighbours_list:

                votes = g[str(n)][:]

                try:
                    fpr, tpr, _ = roc_curve(query_word_classes, votes)

                    auc_trapz = _trapz(tpr, fpr)
                    auc_sklearn = roc_auc_score(query_word_classes, votes)

                    if auc_sklearn > best_auc:
                        best_auc = auc_sklearn
                        best_n = n

                except Exception:
                    pass

            print(
                f"  {weight_type}: "
                f"best AUC = {best_auc:.4f} "
                f"(at n_neighbours={best_n})"
            )

        # Split analysis
        if args.splits_file:

            print("\n--- 3. SPLIT-BASED ANALYSIS ---")

            with open(args.splits_file, "r") as fp:
                splits = json.load(fp)

            for split_id in list(splits.keys())[:3]:

                split_data = splits[split_id]
                test_words = set(split_data["test"])

                test_mask = np.array(
                    [w in test_words for w in query_words],
                    dtype=bool
                )

                n_test = test_mask.sum()

                if n_test == 0:
                    print(f"  Split {split_id}: NO test words found")
                    continue

                classes_test = query_word_classes[test_mask]

                unique_test, counts_test = np.unique(
                    classes_test,
                    return_counts=True
                )

                print(
                    f"  Split {split_id}: "
                    f"{n_test} test samples, "
                    f"labels={dict(zip(unique_test.tolist(), counts_test.tolist()))}"
                )

                if len(unique_test) < 2:
                    print("    WARNING: Only one class in test split")
                    continue

                for weight_type in weighting_functions[:1]:

                    g = store[weight_type]

                    n_neighbours_list = sorted(
                        g.attrs['n_neighbours']
                    )

                    mid_n = n_neighbours_list[
                        len(n_neighbours_list) // 2
                    ]

                    votes = g[str(mid_n)][:]
                    votes_test = votes[test_mask]

                    try:
                        auc = roc_auc_score(classes_test, votes_test)

                        print(
                            f"    {weight_type}, "
                            f"n={mid_n}: AUC = {auc:.4f}"
                        )

                    except Exception as e:

                        print(
                            f"    {weight_type}, "
                            f"n={mid_n}: ERROR computing AUC: {e}"
                        )

        print("\n--- 4. POTENTIAL ISSUES ---")

        issues = []

        if not set(unique_labels.tolist()).issubset({0, 1}):

            issues.append(
                f"Class labels are {unique_labels.tolist()}, "
                f"not [0, 1]"
            )

        for weight_type in weighting_functions[:1]:

            g = store[weight_type]
            n_neighbours_list = sorted(g.attrs['n_neighbours'])

            votes = g[str(n_neighbours_list[0])][:]

            if np.any(np.isnan(votes)):
                issues.append(f"NaN values found in votes for {weight_type}")

            if np.any(np.isinf(votes)):
                issues.append(f"Inf values found in votes for {weight_type}")

        if len(unique_labels) == 2:

            minority_rate = min(label_counts) / sum(label_counts)

            if minority_rate < 0.05:

                issues.append(
                    f"Severe class imbalance: "
                    f"minority class is only {minority_rate:.1%}"
                )

        if issues:

            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

        else:

            print("  No obvious issues detected in the data.")

    print("\n" + "=" * 60)
    print("END OF DIAGNOSTIC REPORT")
    print("=" * 60)


if __name__ == "__main__":
    main()
