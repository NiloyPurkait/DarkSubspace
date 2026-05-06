#!/bin/bash
# sbatch_p12b_multiseed_array.sh.
#
# SLURM array launcher for the Pythia-12B mixed-data SAE training and
# dark-subspace evaluation across seeds 47-51, using collision-safe per-task
# isolated runs/sae_array/p12b_freshinit/task<N>_seed<S>/ directories.
#
# Used in Appendix (P12B replication, A:1146) of the paper.
# Reproduce: sbatch scripts/memcirc/shell/sbatch_p12b_multiseed_array.sh
#
#SBATCH --job-name=p12b_freshinit
#SBATCH --partition=GPU
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --output=logs/p12b_freshinit_seed%a_%A.out
#SBATCH --error=logs/p12b_freshinit_seed%a_%A.err
#SBATCH --nodes=1
#SBATCH --array=0-4
#
# Per-task isolation: the SAE trainer builds run_dir as
# <runs_dir>/<run_name>__<UTC_timestamp> where run_name does not include the
# seed and the timestamp resolves to one second. When several tasks launch
# concurrently, multiple tasks can land in the same UTC second and a glob over
# runs/sae/<prefix>* would race.
#
# Each array task therefore uses an ISOLATED --runs-dir under
# runs/sae_array/p12b_freshinit/<task>_seed<seed>/, so concurrent tasks cannot
# collide. Both SLURM_ARRAY_TASK_ID and SEED are baked into all paths
# (logs, dark-subspace output dir name) for traceability. Globs are scoped
# inside the per-task runs-dir, never repo-wide.
#
# Seeds: 47-51 (5 fresh inits).
# Hardware: an 80GB H100 GPU node (P12B fp32 ~50GB).
# Wall time: 14h (about 7.5h observed for one seed, with queue buffer).
#
# Hyperparameters (matched to the seed-42 P12B SAE config):
#   model         = runs/controlled_ft/run_20260308_001316/ft_epoch5/model
#   layer         = 18
#   d_model_mult  = 4
#   l1_coeff      = 5e-4
#   train_tokens  = 200,000,000
#   tokens/step   = 4096       (about 48,829 steps total)
#   seq_len       = 256
#   batch_size    = 4
#   lr            = 3e-4
#   aux_coeff     = 0.1
#   resample_dead = True, every 500, threshold 1e-6
#   mode          = paper
#   final_eval    = True
#   corpus        = data/memcirc_ctrl_ft/mixed.jsonl

set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p logs

# Map task id 0..4 to seeds 47..51.
declare -a SEED_FOR_TASK=(47 48 49 50 51)
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=${SEED_FOR_TASK[${TASK_ID}]}

P12B_MODEL="runs/controlled_ft/run_20260308_001316/ft_epoch5/model"
P12B_BCD="runs/memcirc/behavioral_channels/p12b_epoch5"
CORPUS="data/memcirc_ctrl_ft/mixed.jsonl"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=18

# Each array task has its own private runs-dir so concurrent UTC-second
# collisions cannot put two tasks in the same SAE training dir.
TASK_RUNS_DIR="runs/sae_array/p12b_freshinit/task${TASK_ID}_seed${SEED}"
mkdir -p "${TASK_RUNS_DIR}"

DSS_OUTPUT_DIR="runs/memcirc/sae_dark_subspace/p12b_mixed_sae_seed${SEED}"

echo "================================================================"
echo "=== P12B Mixed SAE Seed ${SEED} (task ${TASK_ID}) ==="
echo "================================================================"
echo "Started:        $(date)"
echo "Hostname:       $(hostname)"
echo "SLURM_JOB_ID:   ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Model:          ${P12B_MODEL}"
echo "Layer:          ${LAYER}"
echo "Seed:           ${SEED}"
echo "Task runs-dir:  ${TASK_RUNS_DIR}"
echo "DSS output:     ${DSS_OUTPUT_DIR}"

echo ">>> Step 1: Train SAE (isolated --runs-dir per task)"
env/bin/python3 scripts/shared/train_sae.py \
  --model "${P12B_MODEL}" \
  --layers "${LAYER}" \
  --d-model-mult 4 \
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

# Locate the SAE WITHIN the per-task runs-dir. The timestamp suffix is decided
# by train_sae.py at runtime, and per-task isolation guarantees one match.
SAE_PREFIX="train_sae__runs_controlled_ft_run_20260308_001316_ft_epoch5_model__layer${LAYER}__mult4__l10.0005"
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
  --model-path "${P12B_MODEL}" \
  --bcd-dir "${P12B_BCD}" \
  --sae-path "${SAE_PATH}" \
  --member-texts "${MEMBER}" \
  --nonmember-texts "${NONMEMBER}" \
  --layer ${LAYER} \
  --output-dir "${DSS_OUTPUT_DIR}" \
  --model-id "p12b_mixed_seed${SEED}"

echo ""
echo "=== Seed ${SEED} done: $(date) ==="
env/bin/python3 -c "
import json
d = json.load(open('${DSS_OUTPUT_DIR}/results.json'))
rc = d.get('sae_quality', {}).get('reconstruction_cosine')
drop = d.get('dark_subspace_effect', {}).get('auroc_drop_from_recon')
print(f'  seed=${SEED}  task=${TASK_ID}  recon_cos={rc:.4f}  drop={drop:.4f}')
print(f'  SAE: ${SAE_PATH}')
"
