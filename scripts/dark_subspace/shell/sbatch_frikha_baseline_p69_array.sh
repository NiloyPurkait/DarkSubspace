#!/usr/bin/env bash
# sbatch_frikha_baseline_p69_array.sh.
#
# Frikha-style feature-selection baselines on Pythia-6.9B,
# N=5 SLURM-array extension over seeds 43–46.
#
# This is the 4-task array companion to the original seed-42 single-shot
# script (sbatch_frikha_baseline_p69.sh). Runs the same
# PrivacyScalpel (Frikha 2025) feature-selection ablation against the
# remaining 4 SAE checkpoints that make up the paper's tab:dark_subspace
# P69 (mixed-data) N=5 row. Each task uses --seed matched to the SAE
# seed so the 5-fold CV / sampling RNG also varies across runs (mirrors
# how the paper N=5 cluster reports per-seed variability).
#
# Each task ~5 min on a single L40S-class GPU, based on the seed-42 run.
# ~20 min total wall-clock if 4 GPUs free up; longer if serialized.
#
#SBATCH --job-name=frikha_baseline_p69_array
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/frikha_baseline_p69_array_%A_%a.out
#SBATCH --error=logs/frikha_baseline_p69_array_%A_%a.err

set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p logs

# Parallel arrays: SLURM_ARRAY_TASK_ID indexes into both.
SEEDS=(43 44 45 46)
SAE_PATHS=(
  "runs/sae/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005__20260413_184801/sae_final.pt"
  "runs/sae/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005__20260414_083722/sae_final.pt"
  "runs/sae/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005__20260415_184845/sae_final.pt"
  "runs/sae/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005__20260416_154920/sae_final.pt"
)

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SAE_PATH="${SAE_PATHS[$SLURM_ARRAY_TASK_ID]}"

P69_MODEL="runs/controlled_ft/run_20260306_055225/ft_epoch5/model"
P69_BCD="runs/dark_subspace/behavioral_channels/p69_epoch5"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16

OUTPUT_DIR="results/dark_subspace/generated/frikha_features/frikha_baseline_p69_seed${SEED}"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "=== Frikha baseline on P69 mixed SAE seed${SEED} ==="
echo "================================================================"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Array job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "SEED: $SEED"
echo "SAE checkpoint: $SAE_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"

.venv/bin/python scripts/dark_subspace/frikha_baseline_ablation.py \
  --model-path "$P69_MODEL" \
  --bcd-dir "$P69_BCD" \
  --sae-path "$SAE_PATH" \
  --member-texts "$MEMBER" \
  --nonmember-texts "$NONMEMBER" \
  --layer $LAYER \
  --output-dir "$OUTPUT_DIR" \
  --model-id "p69_mixed_seed${SEED}" \
  --seed $SEED \
  --depths 1 5 50 200

echo "================================================================"
echo "=== Frikha baseline seed${SEED} COMPLETE: $(date) ==="
echo "================================================================"
