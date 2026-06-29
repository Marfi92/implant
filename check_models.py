#!/usr/bin/env python3
"""Check which federated-learning models are ready for compare_all.py.

Run on your machine:
    python check_models.py

It scans the results folder and reports, for each model:
  1. Does it have word vectors (lancedb_direct)?
  2. Does it have neighbourhood .pkl files?
  3. Does it have the neighbour_raw_*.h5 store?
  4. Is it ready to run compare_all.py?
"""
import os
import glob
from pathlib import Path

# ---- CONFIGURATION (edit these if your paths differ) ----
RESULTS_DIR = Path("/home/abragam23/federatedhealth_20250617/results_nov12_2025")
SPLITS_FILE = Path("/home/abragam23/fedhealth_data/implant_split_official.json")
STOP_LIST   = Path("/home/abragam23/fedhealth_data/stop_list_freq_1.txt")
# ---------------------------------------------------------

print("=" * 70)
print("MODEL READINESS CHECK")
print("=" * 70)

if not RESULTS_DIR.exists():
    print(f"\nERROR: results directory not found: {RESULTS_DIR}")
    print("Edit RESULTS_DIR at the top of this script to match your path.")
    exit(1)

print(f"\nResults dir: {RESULTS_DIR}")
print(f"Splits file: {SPLITS_FILE}  {'OK' if SPLITS_FILE.exists() else 'MISSING!'}")
print(f"Stop list:   {STOP_LIST}  {'OK' if STOP_LIST.exists() else 'MISSING!'}")

model_uuids = sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()])
print(f"\nFound {len(model_uuids)} model folder(s).\n")

ready_models = []

for uuid in model_uuids:
    model_dir = RESULTS_DIR / uuid / "local_test_results"
    print("-" * 70)
    print(f"Model UUID: {uuid}")

    if not model_dir.exists():
        print(f"  local_test_results/ NOT FOUND — skipping")
        continue

    # Find vector databases (lancedb_direct)
    lancedb_dirs = sorted(model_dir.glob("vector_database_*/lancedb_direct"))
    if not lancedb_dirs:
        lancedb_dirs = sorted(model_dir.glob("*/lancedb_direct"))

    if not lancedb_dirs:
        print(f"  lancedb_direct: NOT FOUND")
        continue

    for ldb in lancedb_dirs:
        # Extract model number from parent folder name
        parent_name = ldb.parent.name
        model_num = parent_name.split("model_")[-1] if "model_" in parent_name else "?"
        print(f"\n  Model number: {model_num}")
        print(f"  Vectors (lancedb_direct): {ldb}")

        # Check for table files inside lancedb
        lance_files = list(ldb.glob("*.lance")) + list(ldb.glob("*/*.lance"))
        table_dirs = [d for d in ldb.iterdir() if d.is_dir()] if ldb.exists() else []
        print(f"    Tables/dirs inside: {[d.name for d in table_dirs]}")

        # Check for neighbourhood .pkl files
        nbr_dirs = sorted(ldb.parent.glob("*-neighbourhoods"))
        if not nbr_dirs:
            nbr_dirs = sorted(ldb.parent.glob("*neighbourhoods*"))

        if nbr_dirs:
            for nd in nbr_dirs:
                pkls = list(nd.glob("*.pkl"))
                print(f"  Neighbourhoods dir: {nd.name}")
                print(f"    .pkl files: {len(pkls)}")

                # Check for existing neighbour_raw_*.h5 store
                analysis_dir = nd / "analysis_official2"
                if not analysis_dir.exists():
                    analysis_dir = nd / "analysis"
                h5_stores = []
                for ad in [nd / "analysis_official2", nd / "analysis", nd]:
                    h5_stores.extend(ad.glob("neighbour_raw_*.h5") if ad.exists() else [])

                if h5_stores:
                    for h5 in h5_stores:
                        sz = h5.stat().st_size / (1024*1024)
                        print(f"  neighbour_raw .h5: {h5} ({sz:.1f} MB) -- READY")
                    ready_models.append({
                        'uuid': uuid, 'model_num': model_num,
                        'lancedb': str(ldb), 'store': str(h5_stores[0]),
                        'has_pkls': len(pkls) > 0, 'has_h5': True,
                    })
                else:
                    print(f"  neighbour_raw .h5: NOT FOUND")
                    if len(pkls) > 0:
                        print(f"    -> .pkl files exist, so you can BUILD the .h5 (Step 2)")
                        ready_models.append({
                            'uuid': uuid, 'model_num': model_num,
                            'lancedb': str(ldb), 'nbr_dir': str(nd),
                            'has_pkls': True, 'has_h5': False,
                        })
                    else:
                        print(f"    -> NO .pkl files — need to run neighbourhood extraction first (Step 1)")
        else:
            print(f"  Neighbourhoods dir: NOT FOUND")
            print(f"    -> Need to run neighbourhood extraction first (Step 1)")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if not ready_models:
    print("\nNo models are ready. You need neighbourhood .pkl files first.")
else:
    for m in ready_models:
        status = "READY for compare_all.py" if m.get('has_h5') else "needs Step 2 (.h5 build) first"
        print(f"\n  model_{m['model_num']} ({m['uuid'][:12]}...): {status}")
        if m.get('has_h5'):
            print(f"    RUN: python compare_all.py \\")
            print(f"           {m['lancedb']} \\")
            print(f"           --store {m['store']} \\")
            print(f"           --splits-file {SPLITS_FILE} \\")
            print(f"           --stop_list {STOP_LIST}")
        elif m.get('has_pkls'):
            print(f"    STEP 2: python analyze_neighbourhoods_split_ooc_vot.py \\")
            print(f"              {m.get('nbr_dir','')} \\")
            print(f"              --official \\")
            print(f"              --splits-file {SPLITS_FILE} \\")
            print(f"              --stop_list {STOP_LIST}")
            print(f"    Then run compare_all.py with the generated .h5 file.")

print()
