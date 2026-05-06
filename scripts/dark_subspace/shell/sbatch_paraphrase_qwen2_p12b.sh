#!/usr/bin/env bash
# sbatch_paraphrase_qwen2_p12b.sh.
#
# Qwen2-7B and Pythia-12B paraphrase audit cross-model launcher.
#
# Used in the paraphrase audit appendix (cross-model cells).
# Reproduce. sbatch scripts/dark_subspace/shell/sbatch_paraphrase_qwen2_p12b.sh
#
# One sbatch per model (separate jobs for scheduler feasibility and output
# isolation). Same config as the P69 paraphrase run. mode=word_shuffle,
# seed=42, seq_len=256, batch_size=8. Output directories.
#   runs/dark_subspace/paraphrase_sensitivity/qwen2/
#   runs/dark_subspace/paraphrase_sensitivity/p12b/
#
# MODEL/SAE/LAYER values taken from scripts/dark_subspace/configs/oc_roster.json.
#   qwen2. model=run_20260313_192753 layer=16 sae=l10.0005__20260417_092500.
#   p12b.  model=run_20260308_001316 layer=18 sae=memcirc_p12b_epoch5_layer18_8x_l1_2e4_member.
#
# Memory note.
#   qwen2 (around 14 GB fp32). Any H100/L40S/A40 partition is fine.
#   p12b (around 50 GB fp32). MUST NOT land on a 44 GB A40.
#
# Wall estimate per model.
#   3 sets x 1000 texts x {7B, 12B} forward + SAE eval. 2 h wall each.

set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON="env/bin/python3"
SCRIPT="scripts/dark_subspace/paraphrase_sensitivity.py"
DATA_MEM="data/memcirc_ctrl_ft/member.jsonl"
DATA_NONMEM="data/memcirc_ctrl_ft/nonmember.jsonl"
OUT_ROOT="runs/dark_subspace/paraphrase_sensitivity"

mkdir -p logs "${OUT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- qwen2 ---
QWEN2_MODEL="runs/controlled_ft/run_20260313_192753/ft_epoch5/model"
QWEN2_BCD="runs/dark_subspace/behavioral_channels/qwen2_epoch5"
QWEN2_SAE="runs/sae/train_sae__runs_controlled_ft_run_20260313_192753_ft_epoch5_model__layer16__mult4__l10.0005__20260417_092500/sae_final.pt"
QWEN2_LAYER=16

echo "=== paraphrase replication. qwen2 (word_shuffle, seed 42) ==="
JOB_QWEN2=$(sbatch --parsable \
  --job-name="paraphrase_qwen2" \
  --partition=GPU --gres=gpu:1 \
  --nodes=1 \
  --mem=80G --cpus-per-task=8 --time=02:00:00 \
  --output="logs/paraphrase_qwen2_%j.out" \
  --error="logs/paraphrase_qwen2_%j.err" \
  --wrap="${PYTHON} ${SCRIPT} \
    --model-tag qwen2 \
    --model-path ${QWEN2_MODEL} \
    --bcd-dir ${QWEN2_BCD} \
    --sae-path ${QWEN2_SAE} \
    --layer ${QWEN2_LAYER} \
    --member-texts ${DATA_MEM} \
    --nonmember-texts ${DATA_NONMEM} \
    --output-dir ${OUT_ROOT} \
    --mode word_shuffle \
    --batch-size 8 \
    --seq-len 256 \
    --device cuda \
    --seed 42")
echo "Submitted qwen2 job ${JOB_QWEN2}"
echo "Log. logs/paraphrase_qwen2_${JOB_QWEN2}.out"

# --- p12b ---
P12B_MODEL="runs/controlled_ft/run_20260308_001316/ft_epoch5/model"
P12B_BCD="runs/dark_subspace/behavioral_channels/p12b_epoch5"
P12B_SAE="runs/sae/memcirc_p12b_epoch5_layer18_4x_l1_5e4_member/sae_final.pt"
P12B_LAYER=18

echo ""
echo "=== paraphrase replication. p12b (word_shuffle, seed 42) ==="
JOB_P12B=$(sbatch --parsable \
  --job-name="paraphrase_p12b" \
  --partition=GPU --gres=gpu:1 \
  --nodes=1 \
  --mem=120G --cpus-per-task=8 --time=02:30:00 \
  --output="logs/paraphrase_p12b_%j.out" \
  --error="logs/paraphrase_p12b_%j.err" \
  --wrap="${PYTHON} ${SCRIPT} \
    --model-tag p12b \
    --model-path ${P12B_MODEL} \
    --bcd-dir ${P12B_BCD} \
    --sae-path ${P12B_SAE} \
    --layer ${P12B_LAYER} \
    --member-texts ${DATA_MEM} \
    --nonmember-texts ${DATA_NONMEM} \
    --output-dir ${OUT_ROOT} \
    --mode word_shuffle \
    --batch-size 4 \
    --seq-len 256 \
    --device cuda \
    --seed 42")
echo "Submitted p12b job ${JOB_P12B}"
echo "Log. logs/paraphrase_p12b_${JOB_P12B}.out"

echo ""
echo "JOB_QWEN2=${JOB_QWEN2}"
echo "JOB_P12B=${JOB_P12B}"
