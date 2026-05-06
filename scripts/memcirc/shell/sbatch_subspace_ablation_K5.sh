#!/bin/bash
#SBATCH --job-name=errPC_K5
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/errPC_K5_%j.out
#SBATCH --error=logs/errPC_K5_%j.err
#SBATCH --nodes=1
#
# sbatch_subspace_ablation_K5.sh.
#
# Companion wrapper for the K=5 sparser-support cell in the K-PC residual
# subspace ablation, on the four gate-passing models.
#
# Used in the K-PC causal ablation appendix (sparser-support cell).
# Reproduce. sbatch scripts/memcirc/shell/sbatch_subspace_ablation_K5.sh
#
# Reuses scripts/memcirc/subspace_ablation_eval.py with K=5 only.
# Output. runs/memcirc/causal_ablation_K5/{model}_errPC_K5/results.json.

set -eu
cd "$(dirname "$0")/../../.."
mkdir -p logs runs/memcirc/causal_ablation_K5

env/bin/python3 scripts/memcirc/subspace_ablation_eval.py \
  --roster scripts/memcirc/configs/subspace_ablation_roster.json \
  --member-texts data/memcirc_ctrl_ft/member.jsonl \
  --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \
  --output-dir runs/memcirc/causal_ablation_K5 \
  --batch-size 8 \
  --seq-len 256 \
  --k-values 5 \
  --probe-seeds 0 1 2 \
  --n-folds 5 \
  --n-c1-seeds 100 \
  --n-c2-seeds 100 \
  --n-c4-seeds 20 \
  --c4-mask-fraction 0.05 \
  --bootstrap-n 10000 \
  --bootstrap-seed 12345 \
  --continue-on-fail \
  --bypass-err-ratio-gate \
  --device cuda \
  --seed 42

echo ""
echo "=== K=5 ablation done. $(date) ==="
echo "Inspect.  runs/memcirc/causal_ablation_K5/aggregate.json"
