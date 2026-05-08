#!/usr/bin/env bash
# sbatch_multiseed_mixed_sae_p69.sh.
#
# Per-seed launcher (alternative to the array form) for one Pythia-6.9B
# mixed-data SAE training plus dark-subspace evaluation. Iterates over five
# seeds in a single SLURM job.
#
# Used in Appendix (P6.9B six-seed cohort) of the paper.
# Reproduce: sbatch scripts/dark_subspace/shell/sbatch_multiseed_mixed_sae_p69.sh
#
#SBATCH --job-name=p69_mixed_seeds
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=logs/p69_mixed_seeds_%j.out
#SBATCH --error=logs/p69_mixed_seeds_%j.err
#SBATCH --nodes=1
#
# Purpose: characterise SAE training variance for the central claim. We need
# at least three additional seeds to report mean and std and confirm the
# effect is robust to SAE initialisation.
#
# Hyperparameters (matched to the P69 member-only SAE for fair comparison):
#   model: runs/controlled_ft/run_20260306_055225/ft_epoch5/model
#   layer: 16
#   d_model_mult: 4  (d_sae = 4 * 4096 = 16384, matches member-only SAE)
#   l1_coeff: 0.0005  (matches member-only SAE)
#   train_tokens: 200M
#   corpus: data/memcirc_ctrl_ft/mixed.jsonl
#   mode: paper (accumulate, hook, nonpad, dedup, shuffle)
#   aux_coeff: 0.1, resample_dead_features, resample_every=500
#
# Note: a previously-existing P69 mixed SAE (seed 43) used 8x/l1=1e-4
# (different from the member-only 4x/l1=5e-4). The capacity mismatch was
# flagged as a confound. These multi-seed runs use MATCHED HP and the prior
# 8x result becomes a sensitivity analysis.
#
# Seeds: 43, 44, 45, 46, 47.
#
# For each seed: (1) train SAE, (2) run dark subspace eval.
# Total runtime: about 5 x (30 min train + 5 min eval) = about 3 hours on
# an H100 node. Conservative SLURM wall time: 36h.

set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p logs

P69_MODEL="runs/controlled_ft/run_20260306_055225/ft_epoch5/model"
P69_BCD="runs/dark_subspace/behavioral_channels/p69_epoch5"
CORPUS="data/memcirc_ctrl_ft/mixed.jsonl"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16

# Include seed 43 to get a matched-HP baseline (existing seed 43 used 8x/l1=1e-4)
SEEDS=(43 44 45 46 47)

for SEED in "${SEEDS[@]}"; do
  echo ""
  echo "----------------------------------------------------------------"
  echo "Pythia-6.9B mixed-data SAE seed $SEED"
  echo "----------------------------------------------------------------"
  echo "Started: $(date)"

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

  # Find the newly created SAE directory
  SAE_PREFIX="train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005"
  SAE_RUN=$(ls -td runs/sae/${SAE_PREFIX}* 2>/dev/null | head -1)
  if [ -z "$SAE_RUN" ]; then
    echo "ERROR: Could not find SAE output directory matching ${SAE_PREFIX}*"
    exit 1
  fi
  SAE_PATH="$SAE_RUN/sae_final.pt"
  if [ ! -f "$SAE_PATH" ]; then
    echo "ERROR: sae_final.pt not found at $SAE_PATH"
    exit 1
  fi
  echo "SAE trained at: $SAE_PATH"

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

  echo "Seed $SEED done: $(date)"
  echo "Results: $OUTPUT_DIR/results.json"
done

echo ""
echo "----------------------------------------------------------------"
echo "Pythia-6.9B multi-seed training complete"
echo "----------------------------------------------------------------"
echo "Matched-hyperparameter seeds (mult=4, L1=5e-4):"
for SEED in "${SEEDS[@]}"; do
  echo "  Seed $SEED: runs/dark_subspace/sae_dark_subspace/p69_mixed_sae_seed${SEED}/results.json"
done
