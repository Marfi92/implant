#!/usr/bin/env python3
"""Run the FULL comparison pipeline on ALL available models.

This script does everything automatically:
  Step 1: (if needed) Build the neighbour_raw_*.h5 store from .pkl files
  Step 2: Run compare_all.py on each model
  Step 3: Print a side-by-side summary of all models

Usage (on your machine):
    python run_all_models.py

    # To also generate .h5 stores where missing:
    python run_all_models.py --build-stores

    # To only run on specific model numbers:
    python run_all_models.py --models 8 10 19 99
"""
import argparse
import subprocess
import sys
import os
import json
import csv as _csv
import re
from pathlib import Path
from collections import OrderedDict

# site letters (A=CCO-Abragam, B=DSV, C=nse, D=utu)
SITE_LETTER = {"CCO-Abragam": "A", "DSV": "B", "nse": "C", "utu": "D"}

# config keys we surface, in print order
CONFIG_KEYS = [
    "num_rounds", "min_clients", "aggregation_epochs", "weigh_by_local_iter",
    "negate_key_metric", "lr", "batch_size", "weight_decay", "mlm_probability",
    "num_train", "num_eval", "r", "lora_alpha", "lora_dropout", "bias", "task_type",
]

# ---- CONFIGURATION (edit these if your paths differ) ----
RESULTS_DIR = Path("/home/abragam23/federatedhealth_20250617/results_nov12_2025")
SPLITS_FILE = Path("/home/abragam23/fedhealth_data/implant_split_official.json")
STOP_LIST   = Path("/home/abragam23/fedhealth_data/stop_list_freq_1.txt")
SCRIPT_DIR  = Path(__file__).resolve().parent   # where compare_all.py lives
# ---------------------------------------------------------


def find_models():
    """Scan RESULTS_DIR and return a list of model dicts.

    Layout-agnostic: for each vector_database_FL_global_model_<N> folder we
    locate (a) the LanceDB database directory (the parent of a *.lance table)
    and (b) a neighbour_raw_*.h5 store, preferring an 'analysis_official'
    variant so results match the official evaluation.
    """
    models = []
    for uuid_dir in sorted(RESULTS_DIR.iterdir()):
        if not uuid_dir.is_dir():
            continue
        test_dir = uuid_dir / "local_test_results"
        if not test_dir.exists():
            continue
        for vdb_dir in sorted(test_dir.glob("vector_database_*model_*")):
            model_num = vdb_dir.name.split("model_")[-1]

            # LanceDB path = parent dir of a *.lance table (prefer words_aggregated.lance)
            lance_tables = sorted(vdb_dir.rglob("*.lance"))
            lance_tables.sort(key=lambda p: 0 if "words_aggregated" in p.name else 1)
            lancedb = lance_tables[0].parent if lance_tables else None

            # neighbour_raw store: prefer the official analysis variant
            all_stores = sorted(vdb_dir.rglob("neighbour_raw_*.h5"))
            def _rank(p):
                s = str(p)
                if "/analysis_official/" in s:
                    return 0
                if "/analysis/" in s:
                    return 1
                if "/analysis_official" in s:  # analysis_official2 etc.
                    return 2
                return 3
            all_stores.sort(key=_rank)
            h5_store = all_stores[0] if all_stores else None

            nbr_dir = h5_store.parent.parent if h5_store else vdb_dir
            has_pkls = bool(list(vdb_dir.rglob("*.pkl")))
            models.append({
                'uuid': uuid_dir.name,
                'model_num': model_num,
                'lancedb': lancedb,
                'nbr_dir': nbr_dir,
                'h5_store': h5_store,
                'has_pkls': has_pkls,
            })
    return models


def _walk_json(obj, found):
    """Recursively collect any CONFIG_KEYS found anywhere in a parsed JSON tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in CONFIG_KEYS and not isinstance(v, (dict, list)):
                found.setdefault(k, v)
            _walk_json(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk_json(v, found)


def read_config(model):
    """Locate the FLARE config files for a run and extract key parameters + sites.

    Returns a dict of {param: value} plus a 'sites'/'model_name' entry. Reads every
    JSON under workspace/*/config and merges the values we care about; sites are
    taken from the client sub-directories under workspace/.
    """
    run_dir = RESULTS_DIR / model['uuid']
    workspace = run_dir / "workspace"
    cfg = {}
    if workspace.exists():
        for jf in sorted(workspace.rglob("config_fed_*.json")) + \
                  sorted(workspace.rglob("*config*.json")):
            try:
                _walk_json(json.load(open(jf)), cfg)
            except (json.JSONDecodeError, OSError):
                continue
    # sites: client folders alongside app_server (exclude server/admin dirs)
    sites = []
    if workspace.exists():
        for d in sorted(workspace.iterdir()):
            if d.is_dir() and d.name in SITE_LETTER:
                sites.append(d.name)
    if not sites:  # fall back to parsing the log
        log = workspace / "log.txt"
        if log.exists():
            seen = set()
            for line in open(log, encoding="utf-8", errors="replace"):
                m = re.search(r"from client\s+(\S+)", line)
                if m and m.group(1) in SITE_LETTER:
                    seen.add(m.group(1))
            sites = sorted(seen)
    cfg['sites'] = ", ".join(sites) if sites else "?"
    cfg['model_name'] = "Model " + "".join(SITE_LETTER[s] for s in sites) if sites else \
                        f"model_{model['model_num']}"
    return cfg


def print_all_configs(models):
    """Print each model's configuration separately and write config_all_models.csv."""
    print("\n" + "=" * 70)
    print("PER-MODEL FEDERATED-TRAINING CONFIGURATION")
    print("(A=CCO-Abragam, B=DSV, C=nse, D=utu)")
    print("=" * 70)
    rows = []
    for m in models:
        cfg = read_config(m)
        print(f"\n--- {cfg['model_name']}  (model_{m['model_num']}, {m['uuid'][:12]}...) ---")
        print(f"  {'sites':<22}: {cfg['sites']}")
        for k in CONFIG_KEYS:
            if k in cfg:
                print(f"  {k:<22}: {cfg[k]}")
        row = {'model_name': cfg['model_name'], 'model_num': m['model_num'],
               'sites': cfg['sites']}
        row.update({k: cfg.get(k, '') for k in CONFIG_KEYS})
        rows.append(row)
    out = SCRIPT_DIR / "config_all_models.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=['model_name', 'model_num', 'sites'] + CONFIG_KEYS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


def build_h5_store(model):
    """Run analyze_neighbourhoods_split_ooc_vot.py --official to create the .h5 store."""
    if not model['has_pkls']:
        print(f"  ERROR: no .pkl files for model_{model['model_num']} — cannot build .h5")
        return None
    script = SCRIPT_DIR / "analyze_neighbourhoods_split_ooc_vot.py"
    cmd = [
        sys.executable, str(script),
        str(model['nbr_dir']),
        "--official",
        "--splits-file", str(SPLITS_FILE),
        "--stop_list", str(STOP_LIST),
    ]
    print(f"\n  Building .h5 store for model_{model['model_num']}...")
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: .h5 build failed (exit code {result.returncode})")
        return None
    # Find the newly created .h5
    for search_dir in [model['nbr_dir'] / "analysis_official2", model['nbr_dir'] / "analysis", model['nbr_dir']]:
        if search_dir.exists():
            stores = list(search_dir.glob("neighbour_raw_*.h5"))
            if stores:
                model['h5_store'] = stores[0]
                return stores[0]
    return None


def run_compare(model):
    """Run compare_all.py on one model and capture the output."""
    script = SCRIPT_DIR / "compare_all.py"
    csv_out = SCRIPT_DIR / f"metrics_model_{model['model_num']}.csv"
    cmd = [
        sys.executable, str(script),
        str(model['lancedb']),
        "--store", str(model['h5_store']),
        "--splits-file", str(SPLITS_FILE),
        "--stop_list", str(STOP_LIST),
        "--csv-out", str(csv_out),
    ]
    print(f"\n{'='*70}")
    print(f"Running compare_all.py on model_{model['model_num']}")
    print(f"  lancedb: {model['lancedb']}")
    print(f"  store:   {model['h5_store']}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.stdout


def parse_results(output):
    """Parse the comparison table from compare_all.py stdout."""
    results = OrderedDict()
    in_table = False
    for line in output.split('\n'):
        if 'COMPARISON ACROSS ALL SPLITS' in line:
            in_table = True
            continue
        if in_table and line.startswith('='):
            if results:
                break
            continue
        if in_table and line.startswith('-'):
            continue
        if in_table and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                method = parts[0]
                # first metric value is roc_auc (e.g. "0.9640±0.0111")
                try:
                    auc = float(parts[1].split('±')[0])
                    ap = float(parts[2].split('±')[0]) if len(parts) > 2 else None
                    results[method] = {'roc_auc': auc, 'avg_prec': ap}
                except (ValueError, IndexError):
                    pass
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build-stores', action='store_true',
                    help='Build .h5 neighbour stores where missing (Step 1)')
    ap.add_argument('--models', nargs='+', default=None,
                    help='Only run these model numbers (e.g. --models 8 10 19 99)')
    ap.add_argument('--config-only', action='store_true',
                    help='Only print each model configuration, then exit')
    args = ap.parse_args()

    print("Scanning for models...")
    models = find_models()

    if args.models:
        models = [m for m in models if m['model_num'] in args.models]

    if not models:
        print("No models found! Check RESULTS_DIR at the top of the script.")
        return

    print(f"\nFound {len(models)} model(s):")
    for m in models:
        status = "READY" if m['h5_store'] else ("can build .h5" if m['has_pkls'] else "needs .pkl files")
        print(f"  model_{m['model_num']} ({m['uuid'][:12]}...): {status}")

    # Print each model's configuration separately (+ config_all_models.csv)
    print_all_configs(models)
    if args.config_only:
        return

    # Build missing .h5 stores if requested
    if args.build_stores:
        for m in models:
            if not m['h5_store'] and m['has_pkls']:
                build_h5_store(m)

    # Run compare_all on all ready models
    all_results = {}
    for m in models:
        if not m['h5_store']:
            print(f"\nSkipping model_{m['model_num']} — no .h5 store (run with --build-stores)")
            continue
        output = run_compare(m)
        parsed = parse_results(output)
        if parsed:
            all_results[f"model_{m['model_num']}"] = parsed

    # Print side-by-side summary
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SIDE-BY-SIDE COMPARISON ACROSS MODELS")
        print("=" * 70)
        model_names = list(all_results.keys())
        methods = list(list(all_results.values())[0].keys())

        header = f"{'method':<15}" + "".join(f"{mn:>18}" for mn in model_names)
        print(header)
        print("-" * len(header))
        for method in methods:
            row = f"{method:<15}"
            for mn in model_names:
                val = all_results[mn].get(method, {}).get('roc_auc')
                row += f"{val:>18.4f}" if val else f"{'—':>18}"
            print(row)
        print("=" * 70)


if __name__ == '__main__':
    main()
