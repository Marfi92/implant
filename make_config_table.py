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
MODELS = [
    # name,        old id,     sites,            rounds_logged, best_round, note
    ("Model AB",   "model_8",  "A, B",           "0-49",  "~8",  "overfits after round 8"),
    ("Model AC",   "model_10", "A, C",           "0-49",  "~8",  "flat after round 8"),
    ("Model ABC",  "model_19", "A, B, C",        "0-49",  "~10", "flat after round 10"),
    ("Model ABCD", "model_99", "A, B, C, D",     "0-99",  "99",  "still improving, full run"),
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
    ("Learning rate",           "1e-4"),
    ("Batch size",              "32"),
    ("Weight decay",            "1e-3"),
    ("LR scheduler",            "linear"),
    ("Warmup steps",            "0"),
    ("Training samples",        "65,536"),
    ("Evaluation samples",      "4,096"),
    ("FL workflow",             "Scatter-and-gather"),
    ("Aggregator",              "InTimeAccumulateWeightedAggregator"),
    ("Local epochs per round",  "1 (aggregation_epochs = 1)"),
    ("Weighted by local iters", "true"),
    ("Min clients per round",   "4"),
    ("Max rounds",              "100"),
    ("Key metric",              "negated validation loss (negate_key_metric = true)"),
    ("Checkpoint persistor",    "EveryEpochPersistor (best kept)"),
]

# ---- markdown
lines = ["# Federated training configuration\n",
         "## Per-model summary\n",
         "| Model | Sites | Rounds logged | Best round | Behaviour |",
         "|---|---|---|---|---|"]
for name, old, sites, rounds, best, note in MODELS:
    lines.append(f"| **{name}** ({old}) | {sites} | {rounds} | {best} | {note} |")
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
t1 = ax1.table(cellText=[[n, s, r, b, note] for n, o, s, r, b, note in MODELS],
               colLabels=["Model", "Sites", "Rounds", "Best round", "Behaviour"],
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
