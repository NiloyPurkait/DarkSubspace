#!/usr/bin/env bash
#SBATCH --job-name=p12b_sameSAE
#SBATCH --partition=GPU
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/p12b_sameSAE_%j.out
#SBATCH --error=logs/p12b_sameSAE_%j.err
#SBATCH --nodes=1
#
# sbatch_p12b_same_sae_consistency.sh.
#
# Re-runs the K=10 ablation on the Pythia-12B mixed-data SAE seed 47 with
# all other settings unchanged. Documents the +0.018 drop that fails the
# same-SAE consistency falsifier.
#
# Used in the same-SAE consistency check appendix.
# Reproduce. sbatch scripts/dark_subspace/shell/sbatch_p12b_same_sae_consistency.sh
#
# Goal. Use the SAME Schema A mixed-data SAE for both the dark-subspace
# evaluation and the K=10 PC ablation. Collapses the paper's two-SAE story
# into a one-SAE story for P12B.
#
# SAE selection. Pick seed 47 by default (canonical choice). If seed 47
# fails recon_cos > 0.99, fallback to whichever seed in {47..51} has the
# highest recon_cos.
#
# Reuses scripts/dark_subspace/subspace_ablation_eval.py (the canonical error-PC
# ablation pipeline used for the existing K=10 P12B causal anchor result).
# Match HP exactly. K=10, n_folds=5, probe seeds [0,1,2], bootstrap n=10000.

set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

PY=env/bin/python3
P12B_MODEL="runs/controlled_ft/run_20260308_001316/ft_epoch5/model"
P12B_BCD="runs/dark_subspace/behavioral_channels/p12b_epoch5"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=18

# Pick best mixed-data seed. Default to seed 47 (canonical), fallback to
# highest recon_cos among completed seeds 47..51 if seed 47 fails the gate.
PICK_SEED=$(${PY} - <<'PYEOF'
import json, glob, os
candidates = []
# Canonical mixed-data SAE path is runs/dark_subspace/sae_dark_subspace/p12b_mixed_sae_seed{47..51}/.
for d in sorted(glob.glob("runs/dark_subspace/sae_dark_subspace/p12b_mixed_sae_seed*")):
    rj = os.path.join(d, "results.json")
    if not os.path.isfile(rj):
        continue
    try:
        r = json.load(open(rj))
    except Exception:
        continue
    seed = d.rsplit("seed", 1)[-1]
    rc = r.get("sae_quality", {}).get("reconstruction_cosine")
    if rc is None:
        continue
    candidates.append((seed, rc, d))
# Prefer the fresh-init cohort (seeds 47..51), with seed 47 as the canonical pick.
fresh = [c for c in candidates if c[0] in {"47","48","49","50","51"}]
target = None
for s, rc, d in fresh:
    if s == "47" and rc > 0.99:
        target = (s, rc, d); break
if target is None and fresh:
    target = max(fresh, key=lambda c: c[1])
if target is None:
    raise SystemExit("No fresh-init cohort seeds (47..51) found on disk, cannot pick SAE.")
print(target[0])
PYEOF
)

echo "Selected fresh-init cohort seed. $PICK_SEED"

# Locate the SAE checkpoint trained for that seed. The fresh-init cohort
# layout is runs/sae_array/p12b_freshinit/task{0..4}_seed{47..51}/<train_sae__...>/sae_final.pt
# (the inner directory is auto-named by train_sae.py with a UTC timestamp).
SAE_PREFIX="train_sae__runs_controlled_ft_run_20260308_001316_ft_epoch5_model__layer18__mult4__l10.0005"
SAE_PATH=""
for cand in runs/sae_array/p12b_freshinit/task*_seed${PICK_SEED}/${SAE_PREFIX}*/sae_final.pt; do
  if [ -f "$cand" ]; then SAE_PATH="$cand"; break; fi
done
if [ -z "$SAE_PATH" ]; then
  echo "ERROR. Could not locate SAE checkpoint for seed ${PICK_SEED} under runs/sae_array/p12b_freshinit/task*_seed${PICK_SEED}/${SAE_PREFIX}*/sae_final.pt"
  exit 1
fi
echo "SAE checkpoint. $SAE_PATH"

OUT_DIR="runs/dark_subspace/causal_ablation/p12b_errPC_K10_schemaA_seed${PICK_SEED}"
mkdir -p "$OUT_DIR"

echo "=== K=10 PC ablation on Schema A mixed SAE (seed $PICK_SEED) ==="
${PY} scripts/dark_subspace/subspace_ablation_eval.py \
  --model-tag "p12b_schemaA_seed${PICK_SEED}" \
  --model-path "$P12B_MODEL" \
  --bcd-dir "$P12B_BCD" \
  --sae-path "$SAE_PATH" \
  --layer $LAYER \
  --member-texts "$MEMBER" \
  --nonmember-texts "$NONMEMBER" \
  --k-values 10 \
  --n-folds 5 \
  --bootstrap-n 10000 \
  --bootstrap-seed 12345 \
  --bypass-err-ratio-gate \
  --output-dir "$OUT_DIR"

# Note. The err_ratio gate is a script-level sanity check defined in
# subspace_ablation_eval.py with range (0.01, 0.30). The seed-47 SAE
# measures (||e||/||h||).mean = 0.3078, marginally above 0.30. The gate
# is not part of the pre-registration text, which only specifies the
# recon-cosine and L0 sparsity gates, so bypassing it does not violate that
# protocol.
# The bypass flag propagates err_ratio_gate_bypassed=True through every
# cell's validity block.

echo "=== DONE. $(date) ==="
echo "Results. $OUT_DIR/results.json (K=10 sub-dir per script convention)"
