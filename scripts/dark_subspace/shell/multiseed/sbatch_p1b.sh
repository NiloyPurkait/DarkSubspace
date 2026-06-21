#!/usr/bin/env bash
# sbatch_p1b.sh.
#
# Pythia-1B mult=4 cross-architecture multiseed launcher. Trains one SAE per
# seed and runs sae_dark_subspace.py.
#
# Used in Appendix (cross-architecture multiseed cluster, A:1162-1164) of the
# paper.
# Reproduce: sbatch scripts/dark_subspace/shell/multiseed/sbatch_p1b.sh
#
#SBATCH --job-name=p1b_multiseed
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/p1b_multiseed_seed%a_%A.out
#SBATCH --error=logs/p1b_multiseed_seed%a_%A.err
#SBATCH --nodes=1
#SBATCH --array=0-4
#
# Goal: lift the Pythia-1B row in tab:dark_subspace from N=1 to N=5.
# Existing single-seed result: drop=0.0713, recon_cos=0.867 (below the strict
# 0.90 reconstruction-cosine gate).
#
# Hyperparameters (verbatim from the existing single-seed config):
#   model         runs/controlled_ft/run_20260216_150714/ft_epoch5/model
#   layer         14
#   d_model_mult  4
#   l1_coeff      5e-4
#   train_tokens  200,000,000
#   tokens/step   4096
#   seq_len       256
#   batch_size    8
#   lr            3e-4
#   aux_coeff     0.1
#   resample_dead True, every 500, threshold 1e-6
#   mode          paper
#   final_eval    True
#   corpus        data/memcirc_ctrl_ft/mixed.jsonl
#
# Per-task isolation: each task gets its own --runs-dir at
# runs/sae_array/p1b_multiseed/task<i>_seed<s>/.
# Seeds: 42-46 (5 fresh inits).

set -euo pipefail
cd "$(dirname "$0")/../../../.." || exit 1
mkdir -p logs

declare -a SEED_FOR_TASK=(42 43 44 45 46)
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=${SEED_FOR_TASK[${TASK_ID}]}

P1B_MODEL="runs/controlled_ft/run_20260216_150714/ft_epoch5/model"
P1B_BCD="runs/dark_subspace/behavioral_channels/p1b_epoch5"
CORPUS="data/memcirc_ctrl_ft/mixed.jsonl"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=14

TASK_RUNS_DIR="runs/sae_array/p1b_multiseed/task${TASK_ID}_seed${SEED}"
mkdir -p "${TASK_RUNS_DIR}"

DSS_OUTPUT_DIR="runs/dark_subspace/sae_dark_subspace/p1b_mixed_sae_seed${SEED}"

echo "================================================================"
echo "=== Pythia-1B Mixed SAE Seed ${SEED} (task ${TASK_ID}) ==="
echo "================================================================"
echo "Started:        $(date)"
echo "Hostname:       $(hostname)"
echo "SLURM_JOB_ID:   ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Model:          ${P1B_MODEL}"
echo "Layer:          ${LAYER}"
echo "Seed:           ${SEED}"
echo "Task runs-dir:  ${TASK_RUNS_DIR}"
echo "DSS output:     ${DSS_OUTPUT_DIR}"

echo ">>> Step 1: Train SAE (isolated --runs-dir)"
.venv/bin/python scripts/shared/train_sae.py \
  --model "${P1B_MODEL}" \
  --layers "${LAYER}" \
  --d-model-mult 4 \
  --l1-coeff 0.0005 \
  --train-tokens 200000000 \
  --tokens-per-step 4096 \
  --seq-len 256 \
  --batch-size 8 \
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

SAE_PREFIX="train_sae__runs_controlled_ft_run_20260216_150714_ft_epoch5_model__layer${LAYER}__mult4__l10.0005"
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
.venv/bin/python scripts/dark_subspace/sae_dark_subspace.py \
  --model-path "${P1B_MODEL}" \
  --bcd-dir "${P1B_BCD}" \
  --sae-path "${SAE_PATH}" \
  --member-texts "${MEMBER}" \
  --nonmember-texts "${NONMEMBER}" \
  --layer ${LAYER} \
  --output-dir "${DSS_OUTPUT_DIR}" \
  --model-id "p1b_mixed_seed${SEED}"

echo ""
echo "=== p1b seed ${SEED} done: $(date) ==="
.venv/bin/python -c "
import json
d = json.load(open('${DSS_OUTPUT_DIR}/results.json'))
rc = d.get('sae_quality', {}).get('reconstruction_cosine')
drop = d.get('dark_subspace_effect', {}).get('auroc_drop_from_recon')
print(f'  seed=${SEED}  task=${TASK_ID}  recon_cos={rc:.4f}  drop={drop:.4f}')
print(f'  SAE: ${SAE_PATH}')
"
