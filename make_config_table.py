#!/usr/bin/env python3
"""Build a per-model federated-training configuration table.

Values are taken from the FLARE run configs we inspected
(workspace/app_server/config/config_fed_server.json + the LoRA/training config).
Shared hyper-parameters are identical across runs; per-model rows differ only in
the participating sites and the number of rounds reached.

Outputs: fl_config_table.md (markdown) and fl_config_table.png (image for slides).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# site letters: A=CCO-Abragam, B=DSV, C=nse, D=utu
# Per-model values verified from run configs via run_all_models.py --config-only.
MODELS = [
    # name,        old id,     sites,        num_rounds, min_clients, note
    ("Model AB",   "model_8",  "A, B",       "50",  "2", "2 sites; drift after ~round 8 (heterogeneous/cross-lingual)"),
    ("Model AC",   "model_10", "A, C",       "50",  "2", "2 sites; flat after ~round 8"),
    ("Model ABC",  "model_19", "A, B, C",    "50",  "3", "3 sites; flat after ~round 10"),
    ("Model ABCD", "model_99", "A, B, C, D", "100", "4", "4 sites; keeps improving, full run"),
]

# configuration shared by all four runs
SHARED = [
    ("Base model",              "XLM-RoBERTa"),
    ("Objective",               "Masked-language modelling (MLM)"),
    ("MLM mask probability",    "0.10"),
    ("Fine-tuning",             "LoRA (low-rank adaptation)"),
    ("LoRA rank r",             "8"),
    ("LoRA alpha",              "8"),
    ("LoRA dropout",            "0.1"),
    ("LoRA bias",               "all"),
    ("LoRA task type",          "TOKEN_CLS"),
    ("Learning rate (lr)",       "1e-4 (verified)"),
    ("Weight decay",            "1e-3 (verified)"),
    ("FL workflow",             "Scatter-and-gather"),
    ("Aggregator",              "InTimeAccumulateWeightedAggregator"),
    ("Local epochs per round",  "1 (aggregation_epochs = 1, verified)"),
    ("Weighted by local iters", "true (verified)"),
    ("Key metric",              "negated validation loss (negate_key_metric = true, verified)"),
    ("Checkpoint persistor",    "EveryEpochPersistor (best kept)"),
]

# NOTE: num_rounds and min_clients are per-model (see table above), NOT shared:
#   AB/AC/ABC = 50 rounds; ABCD = 100 rounds. min_clients = number of sites (2/2/3/4).
# Model AB (model_8) has no train_config.json, so its lr/LoRA/mlm were not saved with
# the run; they are assumed identical to the other runs (same training script).

# ---- markdown
lines = ["# Federated training configuration\n",
         "## Per-model summary\n",
         "| Model | Sites | num_rounds | min_clients | Behaviour |",
         "|---|---|---|---|---|"]
for name, old, sites, rounds, mc, note in MODELS:
    lines.append(f"| **{name}** ({old}) | {sites} | {rounds} | {mc} | {note} |")
lines += ["", "Site key: **A** = CCO-Abragam, **B** = DSV, **C** = nse, **D** = utu.",
          "", "## Shared configuration (identical for all four runs)\n",
          "| Parameter | Value |", "|---|---|"]
for k, v in SHARED:
    lines.append(f"| {k} | {v} |")
open("fl_config_table.md", "w").write("\n".join(lines) + "\n")
print("wrote fl_config_table.md")

# ---- image: two stacked tables
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12),
                               gridspec_kw={"height_ratios": [1, 3]})
for ax in (ax1, ax2):
    ax.axis("off")

ax1.set_title("Per-model federated-training summary\n"
              "(A=CCO-Abragam, B=DSV, C=nse, D=utu)", fontsize=12, weight="bold")
t1 = ax1.table(cellText=[[n, s, r, mc, note] for n, o, s, r, mc, note in MODELS],
               colLabels=["Model", "Sites", "num_rounds", "min_clients", "Behaviour"],
               loc="center", cellLoc="center")
t1.auto_set_font_size(False); t1.set_fontsize(9); t1.scale(1, 1.6)

ax2.set_title("Shared training configuration (all runs)", fontsize=12, weight="bold")
t2 = ax2.table(cellText=[[k, v] for k, v in SHARED],
               colLabels=["Parameter", "Value"],
               loc="center", cellLoc="left")
t2.auto_set_font_size(False); t2.set_fontsize(9); t2.scale(1, 1.5)

fig.tight_layout()
fig.savefig("fl_config_table.png", dpi=150, bbox_inches="tight")
print("wrote fl_config_table.png")
