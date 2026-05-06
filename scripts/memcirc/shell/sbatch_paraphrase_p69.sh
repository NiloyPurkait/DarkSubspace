#!/usr/bin/env bash
# sbatch_paraphrase_p69.sh.
#
# Pythia-6.9B word-order-shuffled paraphrase audit launcher.
#
# Used in the paraphrase audit appendix (Pythia-6.9B cell).
# Reproduce. sbatch scripts/memcirc/shell/sbatch_paraphrase_p69.sh
#
# Mode. word_shuffle (CPU-deterministic conservative syntactic perturbation).
# The GPU is used only for the FORWARD passes (activation collection on
# original members, paraphrased members, nonmembers). No paraphrase model is
# loaded, so memory footprint and wall time are dominated by the
# Pythia-6.9B forward passes.
#
# Wall estimate.
#   3 sets x 1000 texts x 6.9B forward, around 15 min per set, around 45 min
#   total, plus around 5 min SAE eval. 2 h wall is set for margin.

set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON="env/bin/python3"
SCRIPT="scripts/memcirc/paraphrase_sensitivity.py"
DATA_MEM="data/memcirc_ctrl_ft/member.jsonl"
DATA_NONMEM="data/memcirc_ctrl_ft/nonmember.jsonl"
OUT_DIR="runs/memcirc/paraphrase_sensitivity"

mkdir -p logs "${OUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== paraphrase_sensitivity P69 word_shuffle mode ==="

JOB_ID=$(sbatch --parsable \
  --job-name="paraphrase_p69" \
  --partition=GPU --gres=gpu:1 \
  --nodes=1 \
  --mem=80G --cpus-per-task=8 --time=02:00:00 \
  --output="logs/paraphrase_p69_%j.out" --error="logs/paraphrase_p69_%j.err" \
  --wrap="${PYTHON} ${SCRIPT} \
    --model-tag p69 \
    --model-path runs/controlled_ft/run_20260306_055225/ft_epoch5/model \
    --bcd-dir runs/memcirc/behavioral_channels/p69_epoch5 \
    --sae-path runs/sae/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005__20260413_184801/sae_final.pt \
    --layer 16 \
    --member-texts ${DATA_MEM} \
    --nonmember-texts ${DATA_NONMEM} \
    --output-dir ${OUT_DIR} \
    --mode word_shuffle \
    --batch-size 8 \
    --seq-len 256 \
    --device cuda \
    --seed 42")

echo "Submitted job ${JOB_ID}"
echo "Monitor.  squeue -j ${JOB_ID}"
echo "Log.      logs/paraphrase_p69_${JOB_ID}.out"
