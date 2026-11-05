#!/bin/bash
# Reviewer reruns 2-4 (no language-model training; ranking only).
# Usage:  bash run_reruns.sh      (run inside the implant folder, venv active)
set -e
R=/home/abragam23/federatedhealth_20250617/results_nov12_2025
D=/home/abragam23/fedhealth_data
SPL=$D/implant_split_official.json
STOP=$D/stop_list_freq_1.txt
NB=lancedb_direct-words_aggregated-dev_dataset_split_seed_3312143636-cosine-128-neighbourhoods/analysis

# ---- FILL IN: local model Li (lancedb_direct folder and neighbour_raw_*.h5)
LI_DB=FILL_ME/lancedb_direct
LI_STORE=FILL_ME/neighbour_raw_XXXX.h5
# ---- FILL IN: file with the 1,253 expert-verified implant terms (one per line)
VERIFIED=FILL_ME/verified_terms.txt

M19=$R/17dc75eb-6f4c-466b-92bc-60882b73c01c/local_test_results/vector_database_FL_global_model_19
M8=$R/1ddd748b-3d5d-4a66-80e3-685f0f5d04f2/local_test_results/vector_database_FL_global_model_8
M10=$R/2f3ecdb8-c55a-46e1-9853-d043448e8d25/local_test_results/vector_database_FL_global_model_10
M99=$R/ca859b72-ee44-4eca-a823-5a82191fd7dc/local_test_results/vector_database_FL_global_model_99
declare -A DB=( [StLiTro]=$M19/lancedb_direct [StLi]=$M8/lancedb_direct [LiTro]=$M10/lancedb_direct [StLiTroTur]=$M99/lancedb_direct [Li]=$LI_DB )
declare -A ST=( [StLiTro]=$M19/analysis_official/neighbour_raw_4c6048cf267c18221a9ea5d066105c4a.h5
                [StLi]=$M8/$NB/neighbour_raw_06aee68cb4b9b645c6ff9ad1192f4e7a.h5
                [LiTro]=$M10/$NB/neighbour_raw_098739aa2327b3e282a421055086f8db.h5
                [StLiTroTur]=$M99/$NB/neighbour_raw_93f0770866845d27cb0bd776547dd829.h5
                [Li]=$LI_STORE )
mkdir -p reruns

# ---- 2. embedding-only (no spelling feature), all models, per-split results
for m in Li StLiTro StLi LiTro StLiTroTur; do
  python compare_all.py ${DB[$m]} --store ${ST[$m]} --splits-file $SPL --stop_list $STOP \
    --no-lexical --per-split-out reruns/per_split_${m}_nolex.csv | tee reruns/log_${m}_nolex.txt
done
# full pu_lr (with spelling) for Li and StLiTro, also per split
for m in Li StLiTro; do
  python compare_all.py ${DB[$m]} --store ${ST[$m]} --splits-file $SPL --stop_list $STOP \
    --per-split-out reruns/per_split_${m}_full.csv | tee reruns/log_${m}_full.txt
done
# paired differences federated - local over the same 100 splits
for m in StLiTro StLi LiTro StLiTroTur; do
  python paired_diff.py reruns/per_split_${m}_nolex.csv reruns/per_split_Li_nolex.csv \
    --names $m Li --methods pu_lr mindist_all centroid | tee reruns/paired_${m}_vs_Li_nolex.txt
done
python paired_diff.py reruns/per_split_StLiTro_full.csv reruns/per_split_Li_full.csv \
  --names StLiTro Li --methods pu_lr lexical | tee reruns/paired_StLiTro_vs_Li_full.txt

# ---- 3 + 4. glossary sensitivity (StLiTro and Li)
python make_sensitivity_splits.py --official $SPL --verified $VERIFIED --stop_list $STOP \
  --outdir reruns/sens | tee reruns/log_sens_splits.txt
for m in StLiTro Li; do
  python compare_all.py ${DB[$m]} --store ${ST[$m]} --splits-file reruns/sens/splits_verified.json \
    --stop_list reruns/sens/stop_plus_unverified.txt --per-split-out reruns/per_split_${m}_verified.csv \
    | tee reruns/log_${m}_verified.txt
  python compare_all.py ${DB[$m]} --store ${ST[$m]} --splits-file reruns/sens/splits_grouped.json \
    --stop_list $STOP --per-split-out reruns/per_split_${m}_grouped.csv | tee reruns/log_${m}_grouped.txt
done
echo "DONE. Send the reruns/ folder (zip -r reruns.zip reruns)."
