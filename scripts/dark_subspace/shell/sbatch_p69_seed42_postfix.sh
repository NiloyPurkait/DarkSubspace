#!/usr/bin/env bash
# sbatch_p69_seed42_postfix.sh.
#
# Per-seed launcher for the Pythia-6.9B seed-42 mixed-data SAE training run.
#
# Used by the Pythia-6.9B mixed-data SAE cohort feeding `tab:dark_subspace`.
# Reproduce: sbatch scripts/dark_subspace/shell/sbatch_p69_seed42_postfix.sh
#
#SBATCH --job-name=p69_seed42_postfix
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/p69_seed42_postfix_%j.out
#SBATCH --error=logs/p69_seed42_postfix_%j.err
#
# Trains a mixed-data SAE on Pythia-6.9B layer 16 with the canonical
# multi-seed hyperparameters and runs the SAE reconstruction/residual
# evaluation against the channel-decomposition directions.
#
# Hyperparameters (matched to the Pythia-6.9B multi-seed sweep):
#   model: runs/controlled_ft/run_20260306_055225/ft_epoch5/model
#   layer: 16
#   d_model_mult: 4  (d_sae = 16384)
#   l1_coeff: 5e-4
#   train_tokens: 200M
#   seed: 42
#   corpus: data/memcirc_ctrl_ft/mixed.jsonl
#   aux_coeff: 0.1, resample_dead_features, resample_every=500
#
# Runtime: about 14h train + 5 min eval on a 40GB-VRAM GPU node.

set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p logs

P69_MODEL="runs/controlled_ft/run_20260306_055225/ft_epoch5/model"
P69_BCD="runs/dark_subspace/behavioral_channels/p69_epoch5"
CORPUS="data/memcirc_ctrl_ft/mixed.jsonl"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16
SEED=42
SEED_TAG="seed${SEED}_postfix"

echo "================================================================"
echo "=== Pythia-6.9B Mixed-Data SAE Seed 42 ==="
echo "================================================================"
echo "Started: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"

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

# Find the newly created SAE dir. To avoid race conditions with concurrent
# jobs training the same model+HP combo, we pick the newest directory that
# exists AND contains sae_final.pt.
SAE_PREFIX="train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005"
SAE_RUN=""
for cand in $(ls -td runs/sae/${SAE_PREFIX}* 2>/dev/null); do
  if [ -f "$cand/sae_final.pt" ]; then
    SAE_RUN="$cand"
    break
  fi
done
if [ -z "$SAE_RUN" ]; then
  echo "ERROR: Could not find SAE output directory with sae_final.pt matching ${SAE_PREFIX}*"
  exit 1
fi
SAE_PATH="$SAE_RUN/sae_final.pt"
echo "SAE trained at: $SAE_PATH"

# Step 2: Dark subspace eval
OUTPUT_DIR="runs/dark_subspace/sae_dark_subspace/p69_mixed_sae_${SEED_TAG}"
echo ">>> Step 2: Dark subspace eval -> $OUTPUT_DIR"
env/bin/python3 scripts/dark_subspace/sae_dark_subspace.py \
  --model-path "$P69_MODEL" \
  --bcd-dir "$P69_BCD" \
  --sae-path "$SAE_PATH" \
  --member-texts "$MEMBER" \
  --nonmember-texts "$NONMEMBER" \
  --layer $LAYER \
  --output-dir "$OUTPUT_DIR" \
  --model-id "p69_mixed_${SEED_TAG}"

echo "================================================================"
echo "=== Pythia-6.9B Seed 42 Run COMPLETE ==="
echo "================================================================"
echo "Finished: $(date)"
echo "SAE checkpoint: $SAE_PATH"
echo "Dark subspace results: $OUTPUT_DIR/results.json"
