#!/usr/bin/env bash
# sbatch_baseline_attacks_p12b.sh.
#
# Pythia-12B variant of the baseline-attacks SLURM launcher. Wraps
# scripts/memcirc/baseline_attacks_suite.py in single-model mode, writing to
# runs/memcirc/baseline_attacks/p12b/{results.json,per_text_scores.json}
# without touching the shared aggregate.json.
#
# Used in Appendix app:tpr_paraphrase and the per-method results paragraph
# of the paper.
# Reproduce: bash scripts/memcirc/shell/sbatch_baseline_attacks_p12b.sh
#
# Notes.
# Pythia-12B in fp32 needs about 50 GB so it must run on a GPU with at least
# 80 GB VRAM. Wall estimate about 35 min, request 2 h for margin.

set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON="env/bin/python3"
SCRIPT="scripts/memcirc/baseline_attacks_suite.py"
DATA_MEM="data/memcirc_ctrl_ft/member.jsonl"
DATA_NONMEM="data/memcirc_ctrl_ft/nonmember.jsonl"
OUT_DIR="runs/memcirc/baseline_attacks"

# p12b roster entry (mirrors scripts/memcirc/configs/oc_roster.json)
MODEL_TAG="p12b"
MODEL_PATH="runs/controlled_ft/run_20260308_001316/ft_epoch5/model"
BCD_DIR="runs/memcirc/behavioral_channels/p12b_epoch5"
SAE_PATH="runs/sae/memcirc_p12b_epoch5_layer18_4x_l1_5e4_member/sae_final.pt"
LAYER=18

mkdir -p logs "${OUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== baseline_attacks suite (p12b only, 80 GB GPU class) ==="

JOB_ID=$(sbatch --parsable \
  --job-name="baselines_p12b" \
  --partition=GPU --gres=gpu:1 \
  --nodes=1 \
  --mem=96G --cpus-per-task=8 --time=02:00:00 \
  --output="logs/baselines_p12b_%j.out" --error="logs/baselines_p12b_%j.err" \
  --wrap="${PYTHON} ${SCRIPT} \
    --model-tag ${MODEL_TAG} \
    --model-path ${MODEL_PATH} \
    --bcd-dir ${BCD_DIR} \
    --sae-path ${SAE_PATH} \
    --layer ${LAYER} \
    --member-texts ${DATA_MEM} \
    --nonmember-texts ${DATA_NONMEM} \
    --output-dir ${OUT_DIR} \
    --batch-size 4 \
    --seq-len 256 \
    --device cuda \
    --seed 42")

echo "Submitted p12b: job ${JOB_ID}"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Log:      logs/baselines_p12b_${JOB_ID}.out"
