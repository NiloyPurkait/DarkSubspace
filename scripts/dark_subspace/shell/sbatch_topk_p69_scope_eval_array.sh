#!/usr/bin/env bash
# sbatch_topk_p69_scope_eval_array.sh.
#
# TopK SAE scope test on Pythia-6.9B — EVAL ONLY rerun.
#
# Context (do not re-debug):
#   The original `sbatch_topk_p69_scope_array.sh` trained 15 TopK SAEs
#   (5 seeds 42-46 x K in {32,64,128}) successfully but its post-training
#   eval step silently exited on every task because the SAE-prefix glob
#   used a double-underscore (`l10__topk${K}`) while train_sae.py emits a
#   single underscore (`l10_topk${K}`). Bash glob never matched, "ERROR:
#   SAE final checkpoint not found" fired, and Step 2 (eval) never ran.
#   Training was unaffected; checkpoints exist on disk. This script reuses
#   those 15 SAEs and runs eval only — no training, no globbing — by
#   hardcoding the 15 verified SAE paths in two parallel bash arrays.
#
# Design:
#   - 5 seeds (42-46) x 3 TopK values (32, 64, 128) = 15 array tasks.
#   - Eval-only: model load + 2000-sample forward + AUROC math.
#   - Output dir prefix `topk_` keeps the scope-test JSONs separate from the
#     paper-claim JSONs.
#
# Reproduce:
#   sbatch scripts/dark_subspace/shell/sbatch_topk_p69_scope_eval_array.sh
#
#SBATCH --job-name=topk_p69_eval
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/topk_p69_eval_%A_%a.out
#SBATCH --error=logs/topk_p69_eval_%A_%a.err
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
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16

# Parallel arrays — TASK_ID index resolves (K, SEED, SAE_PATH).
declare -a TOPK_LIST=(32 32 32 32 32 64 64 64 64 64 128 128 128 128 128)
declare -a SEED_LIST=(42 43 44 45 46 42 43 44 45 46 42 43 44 45 46)
declare -a SAE_PATH_LIST=(
  "runs/sae_scope/topk_sae_p69/topk32_seed42/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk32__20260510_114647/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk32_seed43/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk32__20260510_115857/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk32_seed44/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk32__20260510_115857/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk32_seed45/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk32__20260510_115857/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk32_seed46/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk32__20260510_115857/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk64_seed42/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk64__20260510_191038/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk64_seed43/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk64__20260510_192034/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk64_seed44/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk64__20260510_192041/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk64_seed45/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk64__20260510_192246/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk64_seed46/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk64__20260510_192246/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk128_seed42/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk128__20260511_023612/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk128_seed43/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk128__20260511_024202/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk128_seed44/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk128__20260511_024240/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk128_seed45/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk128__20260511_024639/sae_final.pt"
  "runs/sae_scope/topk_sae_p69/topk128_seed46/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10_topk128__20260511_024739/sae_final.pt"
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
TOPK=${TOPK_LIST[${TASK_ID}]}
SEED=${SEED_LIST[${TASK_ID}]}
SAE_PATH=${SAE_PATH_LIST[${TASK_ID}]}

echo "================================================================"
echo "=== TopK SAE scope test EVAL Task ${TASK_ID}: K=${TOPK}, Seed=${SEED} ==="
echo "================================================================"
echo "Started: $(date)"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-N/A} SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "Node: $(hostname)"
echo "SAE checkpoint (verified pre-submission): ${SAE_PATH}"

# Defensive existence check (no glob; literal path).
if [ ! -f "${SAE_PATH}" ]; then
  echo "ERROR: SAE final checkpoint not found at ${SAE_PATH}"
  exit 2
fi

# Dark subspace eval — `topk_` prefix to keep paper-claim JSONs safe.
OUTPUT_DIR="runs/dark_subspace/sae_dark_subspace/topk_p69_topk${TOPK}_seed${SEED}"
echo ">>> Dark subspace eval -> ${OUTPUT_DIR}"
.venv/bin/python scripts/dark_subspace/sae_dark_subspace.py \
  --model-path "$P69_MODEL" \
  --bcd-dir "$P69_BCD" \
  --sae-path "$SAE_PATH" \
  --member-texts "$MEMBER" \
  --nonmember-texts "$NONMEMBER" \
  --layer $LAYER \
  --output-dir "$OUTPUT_DIR" \
  --model-id "p69_topk${TOPK}_seed${SEED}" \
  --seed ${SEED}

# Stamp config with experiment marker (same block as
# sbatch_topk_p69_scope_array.sh).
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
echo "=== EVAL Task ${TASK_ID} (K=${TOPK}, seed=${SEED}) COMPLETE: $(date) ==="
echo "================================================================"
echo "SAE checkpoint: ${SAE_PATH}"
echo "Dark subspace results: ${OUTPUT_DIR}/results.json"
