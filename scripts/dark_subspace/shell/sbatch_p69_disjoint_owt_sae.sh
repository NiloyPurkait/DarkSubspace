#!/usr/bin/env bash
#SBATCH --job-name=p69_disjoint_sae
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/p69_disjoint_sae_%j.out
#SBATCH --error=logs/p69_disjoint_sae_%j.err
#
# sbatch_p69_disjoint_owt_sae.sh.
#
# Trains a corpus-disjoint SAE on Pythia-6.9B at layer 16 using a held-out
# OpenWebText partition that contains no member or nonmember strings. Then runs
# the dark-subspace eval against the resulting checkpoint.
#
# Used in Appendix corpus-disjoint robustness check of the paper.
# Reproduce: sbatch scripts/dark_subspace/shell/sbatch_p69_disjoint_owt_sae.sh
#
# Hyperparameters match the canonical Pythia-6.9B six-seed anchor recipe:
#   d_model_mult=4, l1_coeff=5e-4, lr=3e-4, batch=4, seq_len=256, 200M tokens,
#   resample every 500 with threshold 1e-6, aux_coeff=0.1.
#
# Single-seed pilot (seed 100) at the workshop budget; the canonical
# experimental design supports an N=3 follow-up (seeds 100, 101, 102).
#
# Wall budget. ~10-12 GPU-h, 20h reservation for headroom.

set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

PY=.venv/bin/python
P69_MODEL="runs/controlled_ft/run_20260306_055225/ft_epoch5/model"
P69_BCD="runs/dark_subspace/behavioral_channels/p69_epoch5"
DISJOINT_CORPUS="data/memcirc_ctrl_disjoint/mixed_disjoint.jsonl"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16
SEED=100

if [ ! -f "$DISJOINT_CORPUS" ]; then
  echo "ERROR: $DISJOINT_CORPUS not present. Submit sbatch_disjoint_corpus_prep.sh first."
  exit 1
fi

# Verify disjointness one more time on the GPU node (cheap).
${PY} -c "
import json, hashlib
def hashes(p):
    out = set()
    for line in open(p, 'r', encoding='utf-8'):
        line = line.strip()
        if not line: continue
        t = json.loads(line).get('text','')
        if t:
            out.add(hashlib.sha256(t.encode('utf-8',errors='replace')).hexdigest())
    return out
m = hashes('$MEMBER'); n = hashes('$NONMEMBER'); d = hashes('$DISJOINT_CORPUS')
inter = (m | n) & d
print(f'  member hashes: {len(m)}, nonmember: {len(n)}, disjoint corpus: {len(d)}, intersection: {len(inter)}')
assert len(inter) == 0, f'Disjointness violated: {len(inter)} hashes overlap.'
"

TRAIN_START=$(date +%s)
echo "TRAIN_START=$TRAIN_START"

echo "=== Step 1. Train SAE on disjoint corpus, seed=$SEED ==="
${PY} scripts/shared/train_sae.py \
  --model "$P69_MODEL" \
  --layers $LAYER \
  --d-model-mult 4 \
  --l1-coeff 0.0005 \
  --train-tokens 200000000 \
  --tokens-per-step 4096 \
  --seq-len 256 \
  --batch-size 4 \
  --lr 3e-4 \
  --seed $SEED \
  --corpus "$DISJOINT_CORPUS" \
  --corpus-text-field text \
  --aux-coeff 0.1 \
  --resample-dead-features \
  --resample-every 500 \
  --resample-dead-threshold 1e-6 \
  --mode paper \
  --final-eval \
  --runs-dir runs/sae_disjoint_corpus

# Race-safe selection. Newest training dir under runs/sae_disjoint_corpus with
# sae_final.pt and mtime >= TRAIN_START.
SAE_RUN=""
for cand in $(ls -td runs/sae_disjoint_corpus/* 2>/dev/null); do
  [ -f "$cand/sae_final.pt" ] || continue
  DIR_MTIME=$(stat -c %Y "$cand")
  [ "$DIR_MTIME" -lt "$TRAIN_START" ] && continue
  SAE_RUN="$cand"
  break
done
if [ -z "$SAE_RUN" ]; then
  echo "ERROR: no fresh SAE dir found under runs/sae_disjoint_corpus/"
  exit 1
fi
SAE_PATH="$SAE_RUN/sae_final.pt"
echo "SAE trained at: $SAE_PATH"

OUT_DIR="runs/dark_subspace/sae_dark_subspace/p69_disjoint_owt_seed${SEED}"
echo "=== Step 2. Dark-subspace eval -> $OUT_DIR ==="
${PY} scripts/dark_subspace/sae_dark_subspace.py \
  --model-path "$P69_MODEL" \
  --bcd-dir "$P69_BCD" \
  --sae-path "$SAE_PATH" \
  --member-texts "$MEMBER" \
  --nonmember-texts "$NONMEMBER" \
  --layer $LAYER \
  --output-dir "$OUT_DIR" \
  --model-id "p69_disjoint_owt_seed${SEED}"

echo "=== DONE: $(date) ==="
echo "SAE checkpoint: $SAE_PATH"
echo "Eval results:   $OUT_DIR/results.json"
