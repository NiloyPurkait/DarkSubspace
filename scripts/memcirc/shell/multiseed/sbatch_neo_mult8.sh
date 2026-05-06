#!/bin/bash
# sbatch_neo_mult8.sh.
#
# GPT-Neo-2.7B mult=8 cross-architecture multiseed launcher. Trains one SAE
# per seed and runs sae_dark_subspace.py.
#
# Used in Appendix (cross-architecture multiseed cluster, A:1162-1164) of the
# paper.
# Reproduce: sbatch scripts/memcirc/shell/multiseed/sbatch_neo_mult8.sh
#
#SBATCH --job-name=neo_mult8_multiseed
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/neo_mult8_multiseed_seed%a_%A.out
#SBATCH --error=logs/neo_mult8_multiseed_seed%a_%A.err
#SBATCH --nodes=1
#SBATCH --array=0-4
#
# Goal: lift the GPT-Neo-2.7B row from N=1 (mult=8 single seed 42) to N=5.
# A prior mult=4 multi-seed cohort failed the validity gate (3/4 broken
# reconstruction cosine). The mult=8 single-seed at recon_cos=0.998 is the
# gate-passing config, and tab:dark_subspace currently cites that row.
#
# Hyperparameters (verbatim from the existing single-seed config):
#   model         runs/controlled_ft/run_20260221_115025/ft_epoch5/model
#   layer         16
#   d_model_mult  8
#   l1_coeff      5e-4
#   train_tokens  200,000,000
#   tokens/step   4096
#   seq_len       256
#   batch_size    4
#   lr            3e-4
#   aux_coeff     0.1
#   resample_dead True, every 500, threshold 1e-6
#   mode          paper
#   final_eval    True
#   corpus        data/memcirc_ctrl_ft/mixed.jsonl
#
# Seeds: 42-46 (5 fresh inits, including seed 42 to allow direct comparison).

set -euo pipefail
cd "$(dirname "$0")/../../../.." || exit 1
mkdir -p logs

declare -a SEED_FOR_TASK=(42 43 44 45 46)
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=${SEED_FOR_TASK[${TASK_ID}]}

NEO_MODEL="runs/controlled_ft/run_20260221_115025/ft_epoch5/model"
NEO_BCD="runs/memcirc/behavioral_channels/neo_epoch5"
CORPUS="data/memcirc_ctrl_ft/mixed.jsonl"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16

TASK_RUNS_DIR="runs/sae_array/neo_mult8_multiseed/task${TASK_ID}_seed${SEED}"
mkdir -p "${TASK_RUNS_DIR}"

DSS_OUTPUT_DIR="runs/memcirc/sae_dark_subspace/neo_mixed_mult8_seed${SEED}"

echo "================================================================"
echo "=== GPT-Neo-2.7B Mult8 Mixed SAE Seed ${SEED} (task ${TASK_ID}) ==="
echo "================================================================"
echo "Started:        $(date)"
echo "Hostname:       $(hostname)"
echo "SLURM_JOB_ID:   ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Model:          ${NEO_MODEL}"
echo "Layer:          ${LAYER}"
echo "Seed:           ${SEED}"
echo "Task runs-dir:  ${TASK_RUNS_DIR}"
echo "DSS output:     ${DSS_OUTPUT_DIR}"

# Skip seed 42 if results.json already exists (existing pilot)
if [ "${SEED}" -eq 42 ] && [ -f "${DSS_OUTPUT_DIR}/results.json" ]; then
  echo "Seed 42 already has results.json; SKIPPING this task."
  exit 0
fi

echo ">>> Step 1: Train SAE (isolated --runs-dir)"
env/bin/python3 scripts/shared/train_sae.py \
  --model "${NEO_MODEL}" \
  --layers "${LAYER}" \
  --d-model-mult 8 \
  --l1-coeff 0.0005 \
  --train-tokens 200000000 \
  --tokens-per-step 4096 \
  --seq-len 256 \
  --batch-size 4 \
  --lr 3e-4 \
  --seed ${SEED} \
  --corpus "${CORPUS}" \
  --corpus-text-field text \
  --aux-coeff 0.1 \
  --resample-dead-features \
  --resample-every 500 \
  --resample-dead-threshold 1e-6 \
  --mode paper \
  --final-eval \
  --runs-dir "${TASK_RUNS_DIR}"

SAE_PREFIX="train_sae__runs_controlled_ft_run_20260221_115025_ft_epoch5_model__layer${LAYER}__mult8__l10.0005"
MATCHES=( "${TASK_RUNS_DIR}/${SAE_PREFIX}"* )
if [ "${#MATCHES[@]}" -ne 1 ] || [ ! -d "${MATCHES[0]}" ]; then
  echo "ERROR: expected exactly one SAE training dir under ${TASK_RUNS_DIR}, got ${#MATCHES[@]}:"
  printf '  %s\n' "${MATCHES[@]}"
  exit 2
fi
SAE_RUN="${MATCHES[0]}"
SAE_PATH="${SAE_RUN}/sae_final.pt"
if [ ! -f "${SAE_PATH}" ]; then
  echo "ERROR: sae_final.pt not found at ${SAE_PATH}"
  exit 3
fi
echo "SAE trained at: ${SAE_PATH}"

echo ""
echo ">>> Step 2: Dark subspace eval"
env/bin/python3 scripts/memcirc/sae_dark_subspace.py \
  --model-path "${NEO_MODEL}" \
  --bcd-dir "${NEO_BCD}" \
  --sae-path "${SAE_PATH}" \
  --member-texts "${MEMBER}" \
  --nonmember-texts "${NONMEMBER}" \
  --layer ${LAYER} \
  --output-dir "${DSS_OUTPUT_DIR}" \
  --model-id "neo_mult8_seed${SEED}"

echo ""
echo "=== neo mult8 seed ${SEED} done: $(date) ==="
env/bin/python3 -c "
import json
d = json.load(open('${DSS_OUTPUT_DIR}/results.json'))
rc = d.get('sae_quality', {}).get('reconstruction_cosine')
drop = d.get('dark_subspace_effect', {}).get('auroc_drop_from_recon')
print(f'  seed=${SEED}  task=${TASK_ID}  recon_cos={rc:.4f}  drop={drop:.4f}')
print(f'  SAE: ${SAE_PATH}')
"
