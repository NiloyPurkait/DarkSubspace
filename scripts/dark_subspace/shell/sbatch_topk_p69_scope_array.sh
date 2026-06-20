#!/usr/bin/env bash
# sbatch_topk_p69_scope_array.sh.
#
# TopK SAE scope test on Pythia-6.9B (Gao et al. 2024 k-Sparse SAE).
# Tests whether the residual-above-reconstruction phenomenon survives in
# modern dictionary architectures (PrivacyScalpel uses TopK SAEs).
#
# Design:
#   - 5 seeds (42-46) x 3 TopK values (32, 64, 128) = 15 array tasks.
#   - All other HPs match the paper's headline P69 mixed-data L1-ReLU SAE row
#     (mult=4, layer 16, mixed.jsonl corpus, 200M tokens, lr=3e-4, paper mode).
#   - Per-task isolated runs-dir to avoid directory collision.
#   - Output dir prefix `topk_` keeps the scope-test JSONs separate from the
#     paper-claim JSONs.
#
# Reproduce:
#   sbatch scripts/dark_subspace/shell/sbatch_topk_p69_scope_array.sh
#
#SBATCH --job-name=topk_p69_scope
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/topk_p69_scope_%A_%a.out
#SBATCH --error=logs/topk_p69_scope_%A_%a.err
#SBATCH --array=0-14
#
# Array index -> (TopK, SEED):
#   0..4   -> K=32,  seeds 42..46
#   5..9   -> K=64,  seeds 42..46
#   10..14 -> K=128, seeds 42..46

set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p logs

P69_MODEL="runs/controlled_ft/run_20260306_055225/ft_epoch5/model"
P69_BCD="runs/dark_subspace/behavioral_channels/p69_epoch5"
CORPUS="data/memcirc_ctrl_ft/mixed.jsonl"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16

declare -a TOPK_LIST=(32 32 32 32 32 64 64 64 64 64 128 128 128 128 128)
declare -a SEED_LIST=(42 43 44 45 46 42 43 44 45 46 42 43 44 45 46)
TASK_ID=${SLURM_ARRAY_TASK_ID}
TOPK=${TOPK_LIST[${TASK_ID}]}
SEED=${SEED_LIST[${TASK_ID}]}

echo "================================================================"
echo "=== TopK SAE scope test Task ${TASK_ID}: K=${TOPK}, Seed=${SEED} ==="
echo "================================================================"
echo "Started: $(date)"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-N/A} SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "Node: $(hostname)"

# Per-task isolated dir (collision-safe)
TASK_RUNS_DIR="runs/sae_scope/topk_sae_p69/topk${TOPK}_seed${SEED}"
mkdir -p "${TASK_RUNS_DIR}"

# Step 1: Train TopK SAE. l1-coeff=0 (TopK does the sparsifying).
echo ">>> Step 1: Training TopK-${TOPK} SAE (seed ${SEED})"
.venv/bin/python scripts/shared/train_sae.py \
  --model "$P69_MODEL" \
  --layers $LAYER \
  --d-model-mult 4 \
  --l1-coeff 0.0 \
  --topk ${TOPK} \
  --train-tokens 200000000 \
  --tokens-per-step 4096 \
  --seq-len 256 \
  --batch-size 4 \
  --lr 3e-4 \
  --seed ${SEED} \
  --corpus "$CORPUS" \
  --corpus-text-field text \
  --resample-dead-features \
  --resample-every 500 \
  --resample-dead-threshold 1e-6 \
  --mode paper \
  --final-eval \
  --runs-dir "${TASK_RUNS_DIR}"

SAE_PREFIX="train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10__topk${TOPK}"
MATCHES=( "${TASK_RUNS_DIR}/${SAE_PREFIX}"* )
if [ "${#MATCHES[@]}" -ne 1 ]; then
  echo "ERROR: expected exactly one SAE dir under ${TASK_RUNS_DIR} matching ${SAE_PREFIX}*, got ${#MATCHES[@]}"
  ls -ltd "${TASK_RUNS_DIR}/"* 2>/dev/null
  exit 2
fi
SAE_RUN="${MATCHES[0]}"
SAE_PATH="${SAE_RUN}/sae_final.pt"
if [ ! -f "${SAE_PATH}" ]; then
  echo "ERROR: SAE final checkpoint not found at ${SAE_PATH}"
  ls -la "${SAE_RUN}/"
  exit 3
fi
echo "SAE trained at: ${SAE_PATH}"

# Step 2: Dark subspace eval — `topk_` prefix to keep paper-claim JSONs safe.
OUTPUT_DIR="runs/dark_subspace/sae_dark_subspace/topk_p69_topk${TOPK}_seed${SEED}"
echo ">>> Step 2: Dark subspace eval -> ${OUTPUT_DIR}"
.venv/bin/python scripts/dark_subspace/sae_dark_subspace.py \
  --model-path "$P69_MODEL" \
  --bcd-dir "$P69_BCD" \
  --sae-path "$SAE_PATH" \
  --member-texts "$MEMBER" \
  --nonmember-texts "$NONMEMBER" \
  --layer $LAYER \
  --output-dir "$OUTPUT_DIR" \
  --model-id "p69_topk${TOPK}_seed${SEED}"

# Stamp config with experiment marker
.venv/bin/python - <<PYEOF
import json
from pathlib import Path
cfg_path = Path("${OUTPUT_DIR}/config.json")
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text())
    cfg["experiment"] = "topk_scope"
    cfg["topk"] = ${TOPK}
    cfg_path.write_text(json.dumps(cfg, indent=2))
PYEOF

echo "================================================================"
echo "=== Task ${TASK_ID} (K=${TOPK}, seed=${SEED}) COMPLETE: $(date) ==="
echo "================================================================"
echo "SAE checkpoint: ${SAE_PATH}"
echo "Dark subspace results: ${OUTPUT_DIR}/results.json"
