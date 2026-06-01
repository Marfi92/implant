import argparse
import json
from pathlib import Path


def check_splits(splits_file):
    with open(splits_file, "r", encoding="utf-8") as fp:
        splits = json.load(fp)

    print(f"Loaded {len(splits)} splits from {splits_file}\n")

    all_ok = True
    for split_id, parts in splits.items():
        ref = set(parts["ref"])
        dev = set(parts["dev"])
        test = set(parts["test"])

        ref_dev = ref & dev
        ref_test = ref & test
        dev_test = dev & test

        # check for duplicates within each subset (list longer than set)
        dup_ref = len(parts["ref"]) != len(ref)
        dup_dev = len(parts["dev"]) != len(dev)
        dup_test = len(parts["test"]) != len(test)

        problems = []
        if ref_dev:
            problems.append(f"ref∩dev={len(ref_dev)}")
        if ref_test:
            problems.append(f"ref∩test={len(ref_test)}")
        if dev_test:
            problems.append(f"dev∩test={len(dev_test)}")
        if dup_ref:
            problems.append("duplicates in ref")
        if dup_dev:
            problems.append("duplicates in dev")
        if dup_test:
            problems.append("duplicates in test")

        if problems:
            all_ok = False
            print(f"  Split {split_id}: NOT DISTINCT -> {', '.join(problems)}")

    if all_ok:
        print("OK: every split has distinct (disjoint) ref/dev/test subsets with no duplicates.")
    else:
        print("\nFAILED: some splits have overlapping or duplicate subsets (see above).")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Verify that each split in a splits JSON has distinct ref/dev/test subsets.")
    parser.add_argument("splits_file", help="Path to the splits JSON file.", type=Path)
    args = parser.parse_args()
    ok = check_splits(args.splits_file)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
