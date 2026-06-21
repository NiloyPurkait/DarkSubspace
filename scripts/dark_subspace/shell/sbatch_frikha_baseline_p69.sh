#!/usr/bin/env bash
# sbatch_frikha_baseline_p69.sh.
#
# Frikha-style feature-selection baselines on Pythia-6.9B.
#
# Implements the three PrivacyScalpel (Frikha 2025) feature-selection criteria
# (top-k by activation magnitude on members, mean-difference, steering
# probe), ablates the selected features at depths {1, 5, 50, 200}, and
# measures whether the ablation drops the recall-channel AUROC while
# preserving the residual-probe AUROC.
#
# Single seed (42) on the same SAE checkpoint the paper's
# tab:dark_subspace P69 mixed-data row uses. Analysis only — no training.
#
#SBATCH --job-name=frikha_baseline_p69
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/frikha_baseline_p69_%j.out
#SBATCH --error=logs/frikha_baseline_p69_%j.err

set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p logs

P69_MODEL="runs/controlled_ft/run_20260306_055225/ft_epoch5/model"
P69_BCD="runs/dark_subspace/behavioral_channels/p69_epoch5"
SAE_PATH="runs/sae/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005__20260416_204852/sae_final.pt"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16
SEED=42

OUTPUT_DIR="results/dark_subspace/generated/frikha_features/frikha_baseline_p69"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "=== Frikha baseline on P69 mixed SAE seed42 ==="
echo "================================================================"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "SAE checkpoint: $SAE_PATH"

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
echo "=== Frikha baseline COMPLETE: $(date) ==="
echo "================================================================"
