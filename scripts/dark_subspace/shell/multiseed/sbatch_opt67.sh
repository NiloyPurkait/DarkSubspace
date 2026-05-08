#!/usr/bin/env bash
#SBATCH --job-name=opt67_mixed
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=logs/opt67_mixed_seed%a_%A.out
#SBATCH --error=logs/opt67_mixed_seed%a_%A.err
#SBATCH --nodes=1
#SBATCH --array=0-4
#
# sbatch_opt67.sh.
#
# OPT-6.7B layer 24 mixed-data SAE training plus dark-subspace evaluation,
# 5 fresh seeds. Used in Appendix A:1162-1164 (cross-architecture multiseed cohort).
#
# Reproduce: sbatch scripts/dark_subspace/shell/multiseed/sbatch_opt67.sh
#
#
# OPT-6.7B layer 24 mixed-data SAE, 5 fresh seeds.
# Per-task isolated run-dir to avoid collision-on-write.
#
# Goal: lift OPT-6.7B row from N=1 to N=5 AND switch from member-only to mixed-data SAE.
# Existing single-seed: member-only SAE at L24, drop=0.217, recon_cos=0.533 (well below
#   the strict reconstruction-cosine threshold, ddagger-marked). Even with N=5, recon_cos
#   may not clear the validity gate; this dispatches the test rather than presuming
#   the gate will be met.
#
# Hyperparameters (matched to other mixed-data SAE setups; mult=8 used because OPT
# member-only SAE used mult=8 + l1=2e-4; we widen to mult=4 + l1=5e-4 to align with
# the canonical mixed-data HP across the cross-architecture sweep, since the prior
# member-only HP did not gate-pass anyway):
#   model         runs/controlled_ft/run_20260222_130923/ft_epoch5/model
#   layer         24
#   d_model_mult  4   (was 8 in member-only baseline; aligning with mixed-data HP across sweep)
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
# Seeds: 42-46 (5 fresh inits).
# RISK: This row may STILL not gate-pass. Fallback: report ddagger row with N=5
#   recon_cos statistics, document the consistent gate-fail, demote to Appendix.

set -euo pipefail
cd "$(dirname "$0")/../../../.." || exit 1
mkdir -p logs

declare -a SEED_FOR_TASK=(42 43 44 45 46)
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=${SEED_FOR_TASK[${TASK_ID}]}

OPT_MODEL="runs/controlled_ft/run_20260222_130923/ft_epoch5/model"
OPT_BCD="runs/dark_subspace/behavioral_channels/opt67_epoch5"
CORPUS="data/memcirc_ctrl_ft/mixed.jsonl"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=24

TASK_RUNS_DIR="runs/sae_array/opt67_mixed/task${TASK_ID}_seed${SEED}"
mkdir -p "${TASK_RUNS_DIR}"

DSS_OUTPUT_DIR="runs/dark_subspace/sae_dark_subspace/opt67_mixed_sae_seed${SEED}"

echo "================================================================"
echo "=== OPT-6.7B Mixed SAE Seed ${SEED} (task ${TASK_ID}) ==="
echo "================================================================"
echo "Started:        $(date)"
echo "Hostname:       $(hostname)"
echo "SLURM_JOB_ID:   ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Model:          ${OPT_MODEL}"
echo "Layer:          ${LAYER}"
echo "Seed:           ${SEED}"
echo "Task runs-dir:  ${TASK_RUNS_DIR}  (isolated per task)"
echo "DSS output:     ${DSS_OUTPUT_DIR}"

echo ">>> Step 1: Train SAE (isolated --runs-dir)"
env/bin/python3 scripts/shared/train_sae.py \
  --model "${OPT_MODEL}" \
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

SAE_PREFIX="train_sae__runs_controlled_ft_run_20260222_130923_ft_epoch5_model__layer${LAYER}__mult4__l10.0005"
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
env/bin/python3 scripts/dark_subspace/sae_dark_subspace.py \
  --model-path "${OPT_MODEL}" \
  --bcd-dir "${OPT_BCD}" \
  --sae-path "${SAE_PATH}" \
  --member-texts "${MEMBER}" \
  --nonmember-texts "${NONMEMBER}" \
  --layer ${LAYER} \
  --output-dir "${DSS_OUTPUT_DIR}" \
  --model-id "opt67_mixed_seed${SEED}"

echo ""
echo "=== opt67 mixed seed ${SEED} done: $(date) ==="
env/bin/python3 -c "
import json
d = json.load(open('${DSS_OUTPUT_DIR}/results.json'))
rc = d.get('sae_quality', {}).get('reconstruction_cosine')
drop = d.get('dark_subspace_effect', {}).get('auroc_drop_from_recon')
print(f'  seed=${SEED}  task=${TASK_ID}  recon_cos={rc:.4f}  drop={drop:.4f}')
print(f'  SAE: ${SAE_PATH}')
"
