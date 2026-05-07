#!/usr/bin/env bash
# sbatch_pre_ft_baseline.sh.
#
# SLURM wrapper for scripts/dark_subspace/behavioral_channels.py on the
# un-fine-tuned base Pythia-6.9B (the pre-FT negative control across layers
# 12, 14, 16, 18, 20). Runs the BCD probe on the base model using the same
# member and nonmember corpus.
#
# Used as the pre-FT baseline negative control row in the paper.
# Reproduce: sbatch scripts/dark_subspace/shell/sbatch_pre_ft_baseline.sh
#
# Expected outcome.
#   Membership AUROC near chance (about 0.5), demonstrating the dark
#   subspace effect is fine-tuning-induced and not a base-model property.

#SBATCH --job-name=pre_ft_p69
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/pre_ft_p69_%j.out
#SBATCH --error=logs/pre_ft_p69_%j.err
#SBATCH --nodes=1

set -eu
cd "$(dirname "$0")/../../.."
mkdir -p logs

env/bin/python3 scripts/dark_subspace/behavioral_channels.py \
  --model-path EleutherAI/pythia-6.9b \
  --member-texts data/memcirc_ctrl_ft/member.jsonl \
  --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \
  --layers 12 14 16 18 20 \
  --output-dir runs/dark_subspace/behavioral_channels/p69_BASE_pre_ft

echo "=== pre-FT baseline done: $(date) ==="
echo "Inspect:  runs/dark_subspace/behavioral_channels/p69_BASE_pre_ft/"
