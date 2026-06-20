#!/usr/bin/env bash
#SBATCH --job-name=gemma2_2b_sae
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/gemma2_2b_sae_seed%a_%A.out
#SBATCH --error=logs/gemma2_2b_sae_seed%a_%A.err
#SBATCH --nodes=1
#SBATCH --array=0-3
#
# sbatch_gemma2_2b.sh.
#
# Gemma-2-2B layer 16 mixed-data SAE training plus dark-subspace evaluation,
# array of 4 additional seeds (43-46) complementing single-seed 42 result.
# Used in Appendix A:1162-1164 (cross-architecture multiseed cohort).
#
# Reproduce: sbatch scripts/dark_subspace/shell/multiseed/sbatch_gemma2_2b.sh
#
#
# Gemma-2-2B layer 16 mixed-data SAE, 4 additional seeds.
# Step 4 already exists from the seed-42 run.
# This array adds seeds 43, 44, 45, 46 -> N=5 total when combined with existing seed 42.
#
# Per-task isolated run-dir to avoid collision-on-write. Reuses Step-1 FT model + Step-2 BCD
# from 84533 single-job pipeline at runs/controlled_ft/run_20260502_075352/ +
# runs/dark_subspace/behavioral_channels/gemma2_2b_epoch5/.
# This array runs only Step 3 (SAE training) + Step 4 (DSS eval) per seed.
#
# Hyperparameters (verbatim from existing seed 42 config at
# runs/sae_gemma2_2b/single_run/train_sae__runs_controlled_ft_run_20260502_075352_ft_epoch5_model__layer16__mult4__l10.0005__20260502_120029/config.json):
#   model         runs/controlled_ft/run_20260502_075352/ft_epoch5/model
#   layer         16
#   d_model_mult  4
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
# Seeds: 43-46 (4 fresh inits, complementing existing seed 42).

set -euo pipefail
cd "$(dirname "$0")/../../../.." || exit 1
mkdir -p logs

# Read HF token (Gemma-2-2B is a gated model)
ENV="$(pwd)/env"
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_HTTP_TIMEOUT=3600
export HF_HUB_DOWNLOAD_TIMEOUT=3600
export HF_HUB_READ_TIMEOUT=3600
export HF_HUB_MAX_RETRIES=5
if [ -f "$HF_HOME/token" ]; then
  export HF_TOKEN="$(cat "$HF_HOME/token")"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN required for gated google/gemma-2-2b."
  exit 1
fi
export HF_HUB_TOKEN="$HF_TOKEN"

declare -a SEED_FOR_TASK=(43 44 45 46)
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=${SEED_FOR_TASK[${TASK_ID}]}

GEMMA_MODEL="runs/controlled_ft/run_20260502_075352/ft_epoch5/model"
GEMMA_BCD="runs/dark_subspace/behavioral_channels/gemma2_2b_epoch5"
CORPUS="data/memcirc_ctrl_ft/mixed.jsonl"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16

TASK_RUNS_DIR="runs/sae_array/gemma2_2b/task${TASK_ID}_seed${SEED}"
mkdir -p "${TASK_RUNS_DIR}"

DSS_OUTPUT_DIR="runs/dark_subspace/sae_dark_subspace/gemma2_2b_mixed_sae_seed${SEED}"

echo "================================================================"
echo "=== Gemma-2-2B Mixed SAE Seed ${SEED} (task ${TASK_ID}) ==="
echo "================================================================"
echo "Started:        $(date)"
echo "Hostname:       $(hostname)"
echo "SLURM_JOB_ID:   ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Model:          ${GEMMA_MODEL}"
echo "Layer:          ${LAYER}"
echo "Seed:           ${SEED}"
echo "Task runs-dir:  ${TASK_RUNS_DIR}  (isolated per task)"
echo "DSS output:     ${DSS_OUTPUT_DIR}"

echo ">>> Step 1: Train SAE (isolated --runs-dir)"
.venv/bin/python scripts/shared/train_sae.py \
  --model "${GEMMA_MODEL}" \
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

SAE_PREFIX="train_sae__runs_controlled_ft_run_20260502_075352_ft_epoch5_model__layer${LAYER}__mult4__l10.0005"
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
  --model-path "${GEMMA_MODEL}" \
  --bcd-dir "${GEMMA_BCD}" \
  --sae-path "${SAE_PATH}" \
  --member-texts "${MEMBER}" \
  --nonmember-texts "${NONMEMBER}" \
  --layer ${LAYER} \
  --output-dir "${DSS_OUTPUT_DIR}" \
  --model-id "gemma2_2b_mixed_seed${SEED}"

echo ""
echo "=== gemma2-2b mixed seed ${SEED} done: $(date) ==="
.venv/bin/python -c "
import json
d = json.load(open('${DSS_OUTPUT_DIR}/results.json'))
rc = d.get('sae_quality', {}).get('reconstruction_cosine')
drop = d.get('dark_subspace_effect', {}).get('auroc_drop_from_recon')
print(f'  seed=${SEED}  task=${TASK_ID}  recon_cos={rc:.4f}  drop={drop:.4f}')
print(f'  SAE: ${SAE_PATH}')
"
