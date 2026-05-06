#!/bin/bash
# sbatch_p69_mixed_seeds_array.sh.
#
# Array launcher for the additional Pythia-6.9B mixed-data SAE seeds (45, 46,
# 47), where array index -> SEED = 45 + SLURM_ARRAY_TASK_ID.
#
# Used in Appendix (P6.9B six-seed cohort) of the paper.
# Reproduce: sbatch scripts/dark_subspace/shell/sbatch_p69_mixed_seeds_array.sh
#
#SBATCH --job-name=p69_mixed_seeds_arr
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/p69_mixed_seeds_arr_%A_%a.out
#SBATCH --error=logs/p69_mixed_seeds_arr_%A_%a.err
#SBATCH --array=0-2
#
# Purpose: expand the matched-HP multi-seed sweep for the central claim. Seeds
# 43 and 44 are covered elsewhere, and seed 42 has its own postfix job.
#
# Race-condition note: because all three array tasks train against the same
# model+HP combo, the default `ls -td | head -1` selection of the freshly
# trained SAE directory is unsafe if two tasks land in the same minute. We
# mitigate by:
#   1. Each task records its own train-start wall-clock time (TRAIN_START).
#   2. We pick the newest directory that contains sae_final.pt AND whose
#      mtime >= TRAIN_START.
#
# Hyperparameters (matched to the existing multi-seed sweep):
#   mult=4, l1=5e-4, 200M tokens, aux=0.1, resample_every=500.

set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p logs

P69_MODEL="runs/controlled_ft/run_20260306_055225/ft_epoch5/model"
P69_BCD="runs/dark_subspace/behavioral_channels/p69_epoch5"
CORPUS="data/memcirc_ctrl_ft/mixed.jsonl"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16

SEED=$((45 + SLURM_ARRAY_TASK_ID))

echo "================================================================"
echo "=== P69 Mixed SAE Array Task ${SLURM_ARRAY_TASK_ID}, Seed ${SEED} ==="
echo "================================================================"
echo "Started: $(date)"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-N/A} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: $(hostname)"

# Record wall-clock start time so we can distinguish OUR freshly-created SAE
# directory from one created by a sibling array task that started earlier.
TRAIN_START=$(date +%s)
echo "TRAIN_START epoch = ${TRAIN_START}"

# Step 1: Train mixed-data SAE
echo ">>> Step 1: Training SAE (seed $SEED)"
env/bin/python3 scripts/shared/train_sae.py \
  --model "$P69_MODEL" \
  --layers $LAYER \
  --d-model-mult 4 \
  --l1-coeff 0.0005 \
  --train-tokens 200000000 \
  --tokens-per-step 4096 \
  --seq-len 256 \
  --batch-size 4 \
  --lr 3e-4 \
  --seed $SEED \
  --corpus "$CORPUS" \
  --corpus-text-field text \
  --aux-coeff 0.1 \
  --resample-dead-features \
  --resample-every 500 \
  --resample-dead-threshold 1e-6 \
  --mode paper \
  --final-eval \
  --runs-dir runs/sae

SAE_PREFIX="train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005"

# Race-safe selection: newest dir, with sae_final.pt, and mtime >= TRAIN_START.
SAE_RUN=""
for cand in $(ls -td runs/sae/${SAE_PREFIX}* 2>/dev/null); do
  if [ ! -f "$cand/sae_final.pt" ]; then
    continue
  fi
  DIR_MTIME=$(stat -c %Y "$cand")
  if [ "$DIR_MTIME" -lt "$TRAIN_START" ]; then
    # This dir predates OUR training run, skip it (belongs to a sibling task).
    continue
  fi
  SAE_RUN="$cand"
  break
done

if [ -z "$SAE_RUN" ]; then
  echo "ERROR: Could not find a freshly-trained SAE (mtime >= ${TRAIN_START}) matching ${SAE_PREFIX}*"
  echo "Diagnostic listing:"
  ls -ltd runs/sae/${SAE_PREFIX}* 2>/dev/null | head -10 || true
  exit 1
fi
SAE_PATH="$SAE_RUN/sae_final.pt"
echo "SAE trained at: $SAE_PATH (mtime $(stat -c %Y $SAE_RUN), TRAIN_START ${TRAIN_START})"

# Step 2: Dark subspace eval
OUTPUT_DIR="runs/dark_subspace/sae_dark_subspace/p69_mixed_sae_seed${SEED}"
echo ">>> Step 2: Dark subspace eval -> $OUTPUT_DIR"
env/bin/python3 scripts/dark_subspace/sae_dark_subspace.py \
  --model-path "$P69_MODEL" \
  --bcd-dir "$P69_BCD" \
  --sae-path "$SAE_PATH" \
  --member-texts "$MEMBER" \
  --nonmember-texts "$NONMEMBER" \
  --layer $LAYER \
  --output-dir "$OUTPUT_DIR" \
  --model-id "p69_mixed_seed${SEED}"

echo "================================================================"
echo "=== Seed ${SEED} COMPLETE: $(date) ==="
echo "================================================================"
echo "SAE checkpoint: $SAE_PATH"
echo "Dark subspace results: $OUTPUT_DIR/results.json"
