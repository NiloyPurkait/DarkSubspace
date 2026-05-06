#!/usr/bin/env bash
# sbatch_baseline_attacks.sh.
#
# Per-model SLURM launcher for the standard MIA baseline attack stack.
# Wraps scripts/memcirc/baseline_attacks_suite.py with the four
# orthogonal-complement gate-passing models (p69, p12b, neo, qwen2) on
# shared tokenization (256 tokens per text, 1000 member + 1000 nonmember
# each).
#
# Used in Appendix app:tpr_paraphrase and the per-method results paragraph
# of the paper.
# Reproduce: bash scripts/memcirc/shell/sbatch_baseline_attacks.sh
#
# Attacks per model.
#   loss, zlib_ratio, minkprob_20, minkprob_10, minkpp_20, original_d_K,
#   residual_d_K. AUROC, TPR@1%FPR, TPR@5%FPR with bootstrap CIs and paired
#   bootstrap deltas (residual_d_K minus each baseline).
#
# Wall estimate (one SLURM job, sequential loop).
#   p69  (6.9B)  about 20 min
#   p12b (12B)   about 35 min
#   neo  (2.7B)  about 12 min
#   qwen2 (7B)   about 20 min
#   total about 90 min, request 3 h for margin.

set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON="env/bin/python3"
SCRIPT="scripts/memcirc/baseline_attacks_suite.py"
ROSTER="scripts/memcirc/configs/oc_roster.json"
DATA_MEM="data/memcirc_ctrl_ft/member.jsonl"
DATA_NONMEM="data/memcirc_ctrl_ft/nonmember.jsonl"
OUT_DIR="runs/memcirc/baseline_attacks"

mkdir -p logs "${OUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== baseline_attacks suite, four gate-passing models ==="

# Pythia-12B (fp32) needs about 50 GB and does NOT fit on 44 GB GPUs.
# Use H100 or L40S class GPUs only.
JOB_ID=$(sbatch --parsable \
  --job-name="baselines" \
  --partition=GPU --gres=gpu:1 \
  --nodes=1 \
  --mem=96G --cpus-per-task=8 --time=03:00:00 \
  --output="logs/baselines_%j.out" --error="logs/baselines_%j.err" \
  --wrap="${PYTHON} ${SCRIPT} \
    --roster ${ROSTER} \
    --gate-passing-only \
    --member-texts ${DATA_MEM} \
    --nonmember-texts ${DATA_NONMEM} \
    --output-dir ${OUT_DIR} \
    --batch-size 8 \
    --seq-len 256 \
    --device cuda \
    --seed 42 \
    --continue-on-fail")

echo "Submitted: job ${JOB_ID}"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Log:      logs/baselines_${JOB_ID}.out"
