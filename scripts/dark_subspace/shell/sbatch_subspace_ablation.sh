#!/usr/bin/env bash
# sbatch_subspace_ablation.sh.
#
# SLURM wrapper for the K=10/50/200 residual-PC ablation on the four
# gate-passing models.
#
# Used in the K-PC causal ablation appendix.
# Reproduce. sbatch scripts/dark_subspace/shell/sbatch_subspace_ablation.sh
#
# Original internal design note. causal_intervention.md.
# Script. scripts/dark_subspace/subspace_ablation_eval.py (error-PC target).
#
# Math. E = H - H_hat. U_K = top-K right singular vectors of (E - E.mean(0)).
#       H_ablated = H - (E @ U_K) @ U_K^T. Share E and SVD across K.
#
# Sequential single-job run (20 GPU-h budget, L40S/A40).
#   - Per model. Collect pooled H (batch 8) and 200k-token freq for alive-mask.
#   - Model-level gates (recon_cos >= 0.85, err_ratio in [0.01, 0.30],
#     alive_count >= 100).
#   - One SAE pass and one SVD shared across K.
#   - Fit frozen LogReg probe (3 probe-init seeds x 5 folds = 15 AUROCs).
#   - Per K. Primary ablation, C1 (100 random orthonormal-K rotations of E),
#            C2 (100 matched Gaussian noise into H), C4 (20 random 5 percent
#            W_dec column-mask SAEs, rebuild E, top-K PC ablate).
#   - Paired bootstrap 10k on dAUROC per (model, K).
#   - Holm-Bonferroni across 4 models x 3 K = up to 12 cells.
#
# Outputs. runs/dark_subspace/causal_ablation/{model}_errPC_K{k}/results.json,
#          runs/dark_subspace/causal_ablation/aggregate.json,
#          runs/dark_subspace/causal_ablation/run_config.json.

set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON="env/bin/python3"
SCRIPT="scripts/dark_subspace/subspace_ablation_eval.py"
ROSTER="scripts/dark_subspace/configs/subspace_ablation_roster.json"
DATA_MEM="data/memcirc_ctrl_ft/member.jsonl"
DATA_NONMEM="data/memcirc_ctrl_ft/nonmember.jsonl"
OUT_DIR="runs/dark_subspace/causal_ablation"

mkdir -p logs "${OUT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Error-PC causal ablation (4 models x K in {10, 50, 200}) ==="
echo "Roster.  ${ROSTER}"
echo "Output.  ${OUT_DIR}"
echo ""

JOB_ID=$(sbatch --parsable \
  --job-name="errPC_ablation" \
  --partition=GPU --gres=gpu:1 \
  --nodes=1 \
  --mem=96G --cpus-per-task=8 --time=20:00:00 \
  --output="logs/errPC_ablation_%j.out" --error="logs/errPC_ablation_%j.err" \
  --wrap="${PYTHON} ${SCRIPT} \
    --roster ${ROSTER} \
    --member-texts ${DATA_MEM} \
    --nonmember-texts ${DATA_NONMEM} \
    --output-dir ${OUT_DIR} \
    --batch-size 8 \
    --seq-len 256 \
    --k-values 10 50 200 \
    --probe-seeds 0 1 2 \
    --n-folds 5 \
    --n-c1-seeds 100 \
    --n-c2-seeds 100 \
    --n-c4-seeds 20 \
    --c4-mask-fraction 0.05 \
    --bootstrap-n 10000 \
    --bootstrap-seed 12345 \
    --continue-on-fail \
    --device cuda \
    --seed 42")

echo "Submitted job ${JOB_ID}"
echo ""
echo "Monitor."
echo "  squeue -j ${JOB_ID}"
echo "  tail -f logs/errPC_ablation_${JOB_ID}.out"
echo ""
echo "On completion."
echo "  ${OUT_DIR}/aggregate.json             (Holm across up to 12 cells)"
echo "  ${OUT_DIR}/{model}_errPC_K{k}/results.json (one per cell)"
