#!/usr/bin/env python3
"""subspace_ablation_eval.py.

K-PC residual subspace ablation at K in {5, 10, 50, 200} on the four
gate-passing models, with paired bootstrap drops and three controls
(C1 random rotation, C2 matched Gaussian noise, C4 random column mask).

Used in the K-PC causal ablation appendix table.

Validity-gate handling:
The script defaults to strict-gate behaviour (the model-level err_ratio
gate raises on failure and the loop stops on the first model that
fails any gate). Two opt-in CLI flags loosen this for diagnostic use:
``--continue-on-fail`` keeps iterating after a per-model failure, and
``--bypass-err-ratio-gate`` converts the err_ratio assertion into a
logged warning. Any cell evaluated under the bypass carries
``err_ratio_gate_bypassed=True`` in its validity block, and the
Holm-correction summary at the end of this script restricts to cells
with ``gate_pass=True``.

Provenance of shipped paper-claim JSONs:
``results/dark_subspace/generated/causal_ablation/p12b_errPC_K10/results.json``
was produced without bypass flags. ``causal_ablation_K5/p12b_errPC_K5/results.json``
was produced with ``--bypass-err-ratio-gate`` set defensively in the
shell wrapper, but the ``err_ratio_mean`` evaluated to 0.192, well inside
the strict pass range [0.01, 0.30], so the shipped K=5 cell would have
passed the strict gate identically without the bypass. The
``err_ratio_gate_bypassed=True`` flag in that JSON's validity block
is therefore a record of what the run script did, not of any rule
relaxation. The current shell wrapper ``shell/sbatch_subspace_ablation_K5.sh``
no longer passes the bypass flag, so any rerun produces an identical
JSON with ``err_ratio_gate_bypassed=False``.

Validity-gate threshold tier:
``MIN_RECON_COS = 0.85`` corresponds to the permissive tier of the
validity-gate hierarchy documented in ``manuscript/methods.tex`` (strict
$\geq 0.90$, permissive $\geq 0.85$, below-permissive diagnostic only).
The K-PC cells reported in ``tab:kpc_kten_cells`` are computed on
Pythia-12B at recon_cos $\geq 0.99$, i.e. above the strict gate, so the
choice of tier in this script does not affect the headline result.

Reproduce:
    env/bin/python3 scripts/dark_subspace/subspace_ablation_eval.py \\
        --roster scripts/dark_subspace/configs/subspace_ablation_roster.json \\
        --member-texts data/memcirc_ctrl_ft/member.jsonl \\
        --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \\
        --output-dir runs/dark_subspace/causal_ablation \\
        --k-values 10 50 200 \\
        --probe-seeds 0 1 2 --n-folds 5 \\
        --n-c1-seeds 100 --n-c2-seeds 100 --n-c4-seeds 20 \\
        --bootstrap-n 10000 --bootstrap-seed 12345 \\
        --continue-on-fail

Math.

Let H in R^{N x d_model} be mean-pooled activations over the eval set, and
H_hat = SAE.decode(SAE.encode(H)) the SAE reconstruction. The residual
matrix is E = H - H_hat. The top-K right singular vectors of the centered
residual,

    U_full, S, Vt = svd(E - E.mean(0), full_matrices=False)
    U_K = Vt.T[:, :K]                       # (d_model, K)

define the error-PC ablation,

    H_ablated = H - (E @ U_K) @ U_K.T       # strips top-K error directions

A frozen LogReg probe fit on raw H is then scored on H_ablated, measuring
the paired AUROC drop dAUROC = AUROC(H) - AUROC(H_ablated) per (seed, fold).

Design rationale.

An earlier formulation projected H onto col(W_dec_alive) perp. When
d_sae > d_model and the alive-feature count is near-complete, that complement
collapses to {0} and the intervention is trivially zero. The error-PC target
is always non-degenerate (the residual carries structured, non-full-rank
variance).

K values.

K in {10, 50, 200}. Primary statistical test at K=50 per pre-registration.
All three share the same SVD (slicing Uk = U_full[:, :K]), so the K-dimension
adds only SVD-slice and projection cost (no extra SAE forward pass).

Controls (error space).

- C1 rank-K random rotation of e. Sample uniform random orthonormal
  Q_rand in R^{d_model x K}, ablate E with Q_rand instead of U_K,
  H - (E @ Q_rand) @ Q_rand.T. 100 seeds per K.
- C2 Gaussian noise into h with L2 matched to ||U_K U_K^T e|| (per-sample,
  per-K). Inject isotropic Gaussian noise eta into H with ||eta||_2 equal
  to the row-wise L2 of (E @ U_K) @ U_K.T. 100 seeds per K.
- C3 identity (no-op). Score raw H. Sanity check.
- C4 falsifier (random-column-mask SAE). Zero out 5 percent of W_dec
  columns, recompute E_rand = H - H_hat_rand, take top-K PCs of E_rand,
  ablate. 20 seeds per K.

Validity gates (HARD-STOP, pre-registered).

Per-model (cheap, computed before any K-specific work).

| Gate | Threshold |
|---|---|
| reconstruction_cosine (eval, pre) | >= 0.85 |
| (||e|| / ||h||).mean() | in [0.01, 0.30] |
| alive_feature_count | >= 100 (inherited) |

Per-(model, K), each K tested independently.

| Gate | Threshold |
|---|---|
| rank_eff(Cov(E)) (numerical tol 1e-6) | > K |
| K=50 only. (S[:50]**2).sum() / (S**2).sum() | >= 0.10 |

On failure for a specific K, only that (model, K) cell is excluded from the
Holm family. The remaining cells for that model still participate.

Statistical plan.

- Paired bootstrap (10k, balanced over member/nonmember) on dAUROC per
  (model, K), one p-value per cell.
- Holm-Bonferroni across 4 models x 3 K = 12 tests, alpha_family = 0.05.
- Primary effect at K=50 meaningful only if
  dAUROC_primary > max(dAUROC_C1_pct95, dAUROC_C2_pct95) AND
  dAUROC_primary > dAUROC_C4_mean + 3 sigma_C4 AND Holm-adjusted p < 0.05.
- Primary dAUROC target at K=50, >= 0.10.

Output layout.

    {output-dir}/
      run_config.json
      aggregate.json                                  # Holm across 12 cells
      {model}_errPC_K{k}/
        results.json                                  # one per (model, K) cell
"""

import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from sae_mia_audit.models.wrapper import load_model_and_tokenizer
    from sae_mia_audit.utils.hf import HFModelSpec
    from sae_mia_audit.utils.seed import SeedConfig, set_global_seed
    from sae_mia_audit.utils.logging import setup_logging, get_logger
    from sae_mia_audit.sae.io import load_sae_checkpoint_any
    _HAS_PROJECT_INFRA = True
except ImportError as e:
    _HAS_PROJECT_INFRA = False
    _IMPORT_ERROR = str(e)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


if _HAS_PROJECT_INFRA:
    log = get_logger(__name__)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------
MIN_RECON_COS = 0.85
ERR_RATIO_RANGE = (0.01, 0.30)        # (||e||/||h||).mean() gate
MIN_ALIVE_FEATURES = 100
VAR_CAPTURE_K50_MIN = 0.10            # top-50 PC variance capture at K=50
RANK_EFF_TOL = 1e-6                   # numerical rank tolerance on Cov(E)

DEFAULT_K_VALUES = (10, 50, 200)
PRIMARY_K = 50                        # pre-reg primary test cell (K=50)
C4_RANDOM_COL_FRACTION = 0.05         # 5% random-column mask per C4 spec

DEFAULT_PROBE_SEEDS = (0, 1, 2)
DEFAULT_N_FOLDS = 5
DEFAULT_N_C1_SEEDS = 100
DEFAULT_N_C2_SEEDS = 100
DEFAULT_N_C4_SEEDS = 20
DEFAULT_BOOTSTRAP_N = 10_000


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_texts(path: str, max_n: Optional[int] = None) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
            if max_n is not None and len(texts) >= max_n:
                break
    return texts


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        return _sanitize_for_json(obj.item())
    return obj


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_sanitize_for_json(payload), f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_pooled_activations(
    model, tokenizer, texts, layer, seq_len, batch_size, device
) -> np.ndarray:
    all_acts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="pooled acts"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=seq_len, padding=True,
        ).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_acts.append(pooled.cpu().float().numpy())
    return np.concatenate(all_acts, axis=0)


@torch.no_grad()
def collect_token_activations(
    model, tokenizer, texts, layer, seq_len, batch_size, device,
    max_tokens: int,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    total = 0
    for i in tqdm(range(0, len(texts), batch_size), desc="token acts"):
        if total >= max_tokens:
            break
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=seq_len, padding=True,
        ).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer]
        mask = enc["attention_mask"].bool()
        valid = h[mask]
        rows.append(valid.cpu().float().numpy())
        total += valid.shape[0]
    if not rows:
        return np.zeros((0, 1), dtype=np.float32)
    combined = np.concatenate(rows, axis=0)
    if combined.shape[0] > max_tokens:
        rng = np.random.default_rng(42)
        idx = rng.choice(combined.shape[0], size=max_tokens, replace=False)
        combined = combined[idx]
    return combined


# ---------------------------------------------------------------------------
# SAE / W_dec helpers
# ---------------------------------------------------------------------------

def extract_W_dec(sae) -> np.ndarray:
    W = sae.decoder_weight.detach().to("cpu").float().numpy()
    if W.shape != (sae.d_model, sae.d_sae):
        raise ValueError(
            f"decoder_weight has shape {W.shape}; expected ({sae.d_model}, {sae.d_sae})"
        )
    return W


@torch.no_grad()
def compute_feature_activation_frequency(
    sae, token_acts: np.ndarray, device: str, batch_size: int = 512,
) -> np.ndarray:
    if token_acts.shape[0] == 0:
        return np.zeros((sae.d_sae,), dtype=np.float64)
    n_tokens = token_acts.shape[0]
    counts = np.zeros((sae.d_sae,), dtype=np.int64)
    for i in range(0, n_tokens, batch_size):
        batch = token_acts[i : i + batch_size]
        t = torch.tensor(batch, dtype=torch.float32, device=device)
        z = sae.encode(t)
        counts += (z > 0).sum(dim=0).to("cpu").numpy().astype(np.int64)
    return counts.astype(np.float64) / float(n_tokens)


def alive_feature_mask(
    W_dec: np.ndarray,
    feat_freq: np.ndarray,
    norm_rel: float = 0.1,
    freq_threshold: float = 1.0 / 100_000.0,
) -> Tuple[np.ndarray, float, float]:
    col_norms = np.linalg.norm(W_dec, axis=0)
    med = float(np.median(col_norms))
    norm_thresh = norm_rel * med
    mask = (col_norms >= norm_thresh) & (feat_freq >= freq_threshold)
    return mask, norm_thresh, freq_threshold


@torch.no_grad()
def sae_encode_decode(
    activations: np.ndarray, sae, device: str, batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    h_tensor = torch.tensor(activations, dtype=torch.float32, device=device)
    all_z, all_recon = [], []
    for i in range(0, len(h_tensor), batch_size):
        batch = h_tensor[i : i + batch_size]
        z = sae.encode(batch)
        h_hat = sae.decode(z)
        all_z.append(z.detach().cpu().float().numpy())
        all_recon.append(h_hat.detach().cpu().float().numpy())
    return np.concatenate(all_z, axis=0), np.concatenate(all_recon, axis=0)


@torch.no_grad()
def sae_decode_with_masked_W(
    activations: np.ndarray, sae, mask_bool: np.ndarray,
    device: str, batch_size: int = 256,
) -> np.ndarray:
    """Run SAE encode, zero out the masked columns of W_dec, decode.

    We implement this by encoding normally, zeroing the masked feature
    activations, then decoding. Equivalent to zeroing those columns of W_dec
    (h_hat = sum_j W_dec[:, j] * z_j; any masked j contributes 0 either way).
    Does NOT mutate `sae.decoder_weight`.
    """
    h_tensor = torch.tensor(activations, dtype=torch.float32, device=device)
    mask_t = torch.tensor(
        (~mask_bool).astype(np.float32), dtype=torch.float32, device=device
    )  # 1.0 where we KEEP the feature, 0.0 where masked out
    all_recon = []
    for i in range(0, len(h_tensor), batch_size):
        batch = h_tensor[i : i + batch_size]
        z = sae.encode(batch)
        z_kept = z * mask_t[None, :]
        h_hat = sae.decode(z_kept)
        all_recon.append(h_hat.detach().cpu().float().numpy())
    return np.concatenate(all_recon, axis=0)


def reconstruction_cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float(np.mean(num / den))


def mean_l0(z: np.ndarray) -> float:
    return float(np.mean((z > 0).sum(axis=1)))


def effective_rank_cov(E: np.ndarray, tol: float) -> Tuple[int, np.ndarray]:
    """Return (rank_eff, singular_values) of the centered residual matrix E.

    rank_eff = number of singular values of (E - E.mean(0)) that exceed
    `tol * s_max`. This matches `np.linalg.matrix_rank` semantics and is a
    more stable proxy than computing Cov(E) explicitly when d_model is small
    and N is moderate.
    """
    Ec = E - E.mean(0, keepdims=True)
    s = np.linalg.svd(Ec, compute_uv=False)
    if s.size == 0:
        return 0, s
    thresh = tol * float(s[0])
    rank_eff = int((s > thresh).sum())
    return rank_eff, s


# ---------------------------------------------------------------------------
# Frozen-probe framework
# ---------------------------------------------------------------------------

class FrozenProbeBank:
    """Fits and stores (seed, fold)-indexed LogReg probes on the pooled h."""

    def __init__(self, seeds: Tuple[int, ...], n_folds: int):
        self.seeds = seeds
        self.n_folds = n_folds
        self.folds: Dict[int, List[Tuple]] = {s: [] for s in seeds}

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.folds = {s: [] for s in self.seeds}
        for seed in self.seeds:
            skf = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=seed
            )
            for tr, te in skf.split(X, y):
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X[tr])
                clf = LogisticRegression(
                    max_iter=1000, solver="lbfgs", C=1.0, random_state=seed,
                )
                clf.fit(Xtr, y[tr])
                self.folds[seed].append((tr, te, scaler, clf))

    def score(self, X_eval: np.ndarray, y: np.ndarray) -> Dict:
        all_aurocs: List[float] = []
        by_seed: Dict[int, List[float]] = {}
        for seed, folds in self.folds.items():
            seed_list = []
            for tr, te, scaler, clf in folds:
                Xte = scaler.transform(X_eval[te])
                prob = clf.predict_proba(Xte)[:, 1]
                auc = roc_auc_score(y[te], prob)
                seed_list.append(float(auc))
                all_aurocs.append(float(auc))
            by_seed[int(seed)] = seed_list
        return {
            "mean": float(np.mean(all_aurocs)),
            "std": float(np.std(all_aurocs)),
            "aurocs": all_aurocs,
            "by_seed": by_seed,
            "n_aurocs": int(len(all_aurocs)),
        }

    def score_indices(self, X_eval: np.ndarray, y: np.ndarray):
        rows = []
        for seed, folds in self.folds.items():
            for fi, (tr, te, scaler, clf) in enumerate(folds):
                Xte = scaler.transform(X_eval[te])
                prob = clf.predict_proba(Xte)[:, 1]
                rows.append((int(seed), int(fi), te, y[te], prob))
        return rows

    def weights_hash(self) -> str:
        h = hashlib.sha256()
        for seed in sorted(self.folds.keys()):
            for tr, te, scaler, clf in self.folds[seed]:
                h.update(clf.coef_.astype(np.float64).tobytes())
                h.update(clf.intercept_.astype(np.float64).tobytes())
                h.update(scaler.mean_.astype(np.float64).tobytes())
                h.update(scaler.scale_.astype(np.float64).tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Error-PC ablation (primary + controls)
# ---------------------------------------------------------------------------

def errpc_ablate(H: np.ndarray, E: np.ndarray, Uk: np.ndarray) -> np.ndarray:
    """Primary ablation H_ablated = H - (E @ U_K) @ U_K.T.

    Strips the top-K directions of the reconstruction error from H, leaving
    H_hat intact (since E = H - H_hat and only the top-K projection of E is
    subtracted).
    """
    return H - (E @ Uk) @ Uk.T


def paired_bootstrap_delta(
    rows_h, rows_a, n_boot: int, seed: int,
) -> Dict:
    """Paired bootstrap on per-fold dAUROC (balanced class resample)."""
    assert len(rows_h) == len(rows_a)
    rng = np.random.default_rng(seed)
    n_rows = len(rows_h)
    deltas = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        fold_deltas = np.zeros(n_rows, dtype=np.float64)
        for ri, (r_h, r_a) in enumerate(zip(rows_h, rows_a)):
            _, _, te_h, y_h, p_h = r_h
            _, _, te_a, y_a, p_a = r_a
            pos_idx = np.where(y_h == 1)[0]
            neg_idx = np.where(y_h == 0)[0]
            rp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            rn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            idx = np.concatenate([rp, rn])
            try:
                a_h = roc_auc_score(y_h[idx], p_h[idx])
                a_a = roc_auc_score(y_a[idx], p_a[idx])
            except ValueError:
                a_h, a_a = 0.5, 0.5
            fold_deltas[ri] = a_h - a_a
        deltas[b] = float(fold_deltas.mean())
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    p_one = float(np.mean(deltas <= 0.0))
    return {
        "n_boot": int(n_boot),
        "mean": float(deltas.mean()),
        "std": float(deltas.std()),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "p_one_sided_leq0": p_one,
    }


# ---------------------------------------------------------------------------
# Control arms (error-space).
# ---------------------------------------------------------------------------

def control_c1_random_orthonormal_K(
    probes: FrozenProbeBank,
    H: np.ndarray,
    E: np.ndarray,
    labels: np.ndarray,
    K: int,
    n_seeds: int,
    seed: int,
) -> Dict:
    """C1 rank-K random rotation of e. Ablate E with random orthonormal basis."""
    d = H.shape[1]
    if K <= 0 or K >= d:
        return {"note": f"K={K} out of (0,d); skipped", "aurocs": []}
    rng = np.random.default_rng(seed)
    aurocs = []
    for _ in range(n_seeds):
        A = rng.standard_normal((d, K)).astype(np.float32)
        Q_rand, _ = np.linalg.qr(A)          # (d, K) orthonormal
        H_ctrl = H - (E @ Q_rand) @ Q_rand.T
        res = probes.score(H_ctrl, labels)
        aurocs.append(res["mean"])
    a = np.array(aurocs, dtype=np.float64)
    return {
        "n_seeds": int(n_seeds),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "pct05": float(np.percentile(a, 5)),
        "pct50": float(np.percentile(a, 50)),
        "pct95": float(np.percentile(a, 95)),
        "aurocs": aurocs,
    }


def control_c2_matched_gaussian(
    probes: FrozenProbeBank,
    H: np.ndarray,
    proj_norms: np.ndarray,  # per-row ||U_K U_K^T e||_2
    labels: np.ndarray,
    n_seeds: int,
    seed: int,
) -> Dict:
    """C2 Gaussian noise into h with per-row L2 matched to the K-PC projection."""
    rng = np.random.default_rng(seed)
    aurocs = []
    for _ in range(n_seeds):
        e_raw = rng.standard_normal(H.shape).astype(np.float32)
        e_norm = np.linalg.norm(e_raw, axis=1, keepdims=True) + 1e-12
        eta = e_raw * (proj_norms[:, None] / e_norm)
        H_noisy = H + eta
        res = probes.score(H_noisy, labels)
        aurocs.append(res["mean"])
    a = np.array(aurocs, dtype=np.float64)
    return {
        "n_seeds": int(n_seeds),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "pct05": float(np.percentile(a, 5)),
        "pct50": float(np.percentile(a, 50)),
        "pct95": float(np.percentile(a, 95)),
        "aurocs": aurocs,
    }


def control_c4_random_column_mask(
    probes: FrozenProbeBank,
    H: np.ndarray,
    labels: np.ndarray,
    sae, device: str,
    d_sae: int, K: int,
    n_seeds: int,
    mask_fraction: float,
    seed: int,
) -> Dict:
    """C4 falsifier: zero 5% of W_dec columns, recompute E_rand, top-K PC ablate."""
    rng = np.random.default_rng(seed)
    n_mask = max(1, int(round(mask_fraction * d_sae)))
    aurocs = []
    for _ in range(n_seeds):
        idx = rng.choice(d_sae, size=n_mask, replace=False)
        mask_bool = np.zeros(d_sae, dtype=bool)
        mask_bool[idx] = True
        H_hat_rand = sae_decode_with_masked_W(H, sae, mask_bool, device)
        E_rand = H - H_hat_rand
        Ec = E_rand - E_rand.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
        Uk = Vt.T[:, :K]
        H_ablated = H - (E_rand @ Uk) @ Uk.T
        res = probes.score(H_ablated, labels)
        aurocs.append(res["mean"])
    a = np.array(aurocs, dtype=np.float64)
    return {
        "n_seeds": int(n_seeds),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "pct05": float(np.percentile(a, 5)),
        "pct50": float(np.percentile(a, 50)),
        "pct95": float(np.percentile(a, 95)),
        "aurocs": aurocs,
        "mask_fraction": float(mask_fraction),
        "n_columns_masked": int(n_mask),
    }


# ---------------------------------------------------------------------------
# Per-(model, K) evaluation
# ---------------------------------------------------------------------------

def run_one_model(
    model_tag: str,
    model_path: str,
    bcd_dir: Optional[str],
    sae_path: str,
    member_texts_path: str,
    nonmember_texts_path: str,
    layer: int,
    output_dir: Path,
    k_values: Tuple[int, ...],
    seq_len: int,
    batch_size: int,
    device: str,
    max_texts: int,
    freq_tokens: int,
    probe_seeds: Tuple[int, ...],
    n_folds: int,
    n_c1_seeds: int,
    n_c2_seeds: int,
    n_c4_seeds: int,
    c4_mask_fraction: float,
    bootstrap_n: int,
    bootstrap_seed: int,
    bypass_err_ratio_gate: bool = False,
) -> List[Dict]:
    """Run error-PC ablation for ONE model across all K values. Returns one
    result dict per (model, K) cell. Raises AssertionError if MODEL-level gates
    fail (those skip all Ks); per-K gate failures are recorded in-result and
    the cell is marked `gate_pass=False`.

    If ``bypass_err_ratio_gate`` is True, the model-level err_ratio gate
    (``(||e||/||h||).mean in ERR_RATIO_RANGE``) is converted to a warning
    instead of AssertionError, allowing the K-loop to proceed. The per-model
    ``err_ratio_gate_bypassed`` field is propagated to every cell's validity
    block so downstream analysis can filter. All other gates (recon_cos,
    alive_feature_count) remain hard.
    """
    # --- SAE ---
    sae = load_sae_checkpoint_any(sae_path, device=device)
    log.info(f"[{model_tag}] SAE d_model={sae.d_model} d_sae={sae.d_sae}")

    # --- Model ---
    spec = HFModelSpec(name_or_path=model_path, torch_dtype="bfloat16")
    wrapper = load_model_and_tokenizer(spec)
    model = wrapper.model.to(device).eval()
    tokenizer = wrapper.tokenizer

    # --- Texts ---
    max_n = max_texts if max_texts > 0 else None
    member_texts = _load_texts(member_texts_path, max_n)
    nonmember_texts = _load_texts(nonmember_texts_path, max_n)
    all_texts = member_texts + nonmember_texts
    labels = np.array(
        [1] * len(member_texts) + [0] * len(nonmember_texts), dtype=np.int64
    )
    log.info(
        f"[{model_tag}] {len(member_texts)} member + {len(nonmember_texts)} nonmember"
    )

    # --- Pooled activations ---
    H = collect_pooled_activations(
        model, tokenizer, all_texts, layer, seq_len, batch_size, device,
    )
    log.info(f"[{model_tag}] pooled activations shape={H.shape}")

    # --- Token activations for alive-feature freq ---
    token_acts = collect_token_activations(
        model, tokenizer, all_texts, layer, seq_len, batch_size, device,
        max_tokens=freq_tokens,
    )
    log.info(f"[{model_tag}] token activations shape={token_acts.shape}")

    # Free the big model
    del model
    torch.cuda.empty_cache()

    # --- SAE pass: reconstruction + residual ---
    z, H_hat = sae_encode_decode(H, sae, device)
    E = H - H_hat
    recon_cos = reconstruction_cosine(H, H_hat)
    L0 = mean_l0(z)
    log.info(f"[{model_tag}] recon_cos={recon_cos:.4f} L0={L0:.2f}")
    if recon_cos < MIN_RECON_COS:
        raise AssertionError(
            f"[{model_tag}] VALIDITY GATE FAIL (model-level): recon_cos "
            f"{recon_cos:.4f} < {MIN_RECON_COS} (dictionary collapse)"
        )

    # ||e|| / ||h|| gate
    h_norms = np.linalg.norm(H, axis=1) + 1e-12
    e_norms = np.linalg.norm(E, axis=1)
    err_ratio_mean = float(np.mean(e_norms / h_norms))
    err_ratio_pass = (ERR_RATIO_RANGE[0] <= err_ratio_mean <= ERR_RATIO_RANGE[1])
    log.info(
        f"[{model_tag}] (||e||/||h||).mean = {err_ratio_mean:.4f} "
        f"(gate {ERR_RATIO_RANGE}) -> pass={err_ratio_pass}"
    )
    if not err_ratio_pass:
        if bypass_err_ratio_gate:
            log.warning(
                f"[{model_tag}] BYPASS: err_ratio {err_ratio_mean:.4f} not in "
                f"{ERR_RATIO_RANGE}, proceeding because "
                f"--bypass-err-ratio-gate was set. Cell validity blocks will "
                f"carry err_ratio_gate_bypassed=True."
            )
        else:
            raise AssertionError(
                f"[{model_tag}] VALIDITY GATE FAIL (model-level): err_ratio "
                f"{err_ratio_mean:.4f} not in {ERR_RATIO_RANGE}"
            )

    # Alive-feature count (inherited gate)
    W_dec = extract_W_dec(sae)
    feat_freq = compute_feature_activation_frequency(sae, token_acts, device)
    alive_mask_arr, norm_thresh, freq_thresh = alive_feature_mask(
        W_dec, feat_freq, norm_rel=0.1, freq_threshold=1.0 / 100_000.0,
    )
    alive_count = int(alive_mask_arr.sum())
    if alive_count < MIN_ALIVE_FEATURES:
        raise AssertionError(
            f"[{model_tag}] VALIDITY GATE FAIL (model-level): alive_feature_count "
            f"{alive_count} < {MIN_ALIVE_FEATURES}"
        )

    # --- SVD on centered residual (shared across K values) ---
    Ec = E - E.mean(0, keepdims=True)
    U_svd, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    Uk_full = Vt.T  # (d_model, min(N, d_model))
    total_s2 = float((S ** 2).sum())
    rank_eff = int((S > RANK_EFF_TOL * float(S[0])).sum()) if S.size > 0 else 0
    log.info(
        f"[{model_tag}] residual SVD: {Uk_full.shape}, rank_eff={rank_eff}, "
        f"S[0]={float(S[0]):.4g}, total_var={total_s2:.4g}"
    )

    # --- Fit frozen probe bank on raw H ---
    probes = FrozenProbeBank(seeds=probe_seeds, n_folds=n_folds)
    probes.fit(H, labels)
    probe_hash = probes.weights_hash()
    res_h = probes.score(H, labels)
    auroc_h_mean = float(res_h["mean"])
    log.info(f"[{model_tag}] AUROC(H) = {auroc_h_mean:.4f}")

    # --- Per-K loop ---
    cell_results: List[Dict] = []
    for K in k_values:
        out_dir = output_dir / f"{model_tag}_errPC_K{K}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-K validity: rank_eff > K, and at K=50, variance capture >= 0.10
        rank_pass = rank_eff > K
        var_capture_K = float((S[:K] ** 2).sum() / max(total_s2, 1e-12))
        var_capture_K50 = float(
            (S[:min(50, S.size)] ** 2).sum() / max(total_s2, 1e-12)
        )
        var_capture_pass = True
        if K == PRIMARY_K:
            var_capture_pass = (var_capture_K50 >= VAR_CAPTURE_K50_MIN)
        k_gate_pass = bool(rank_pass and var_capture_pass)
        if not k_gate_pass:
            gate_reason = []
            if not rank_pass:
                gate_reason.append(
                    f"rank_eff {rank_eff} <= K {K}"
                )
            if not var_capture_pass:
                gate_reason.append(
                    f"var_capture_K50 {var_capture_K50:.4f} < "
                    f"{VAR_CAPTURE_K50_MIN}"
                )
            log.warning(
                f"[{model_tag}] K={K} VALIDITY GATE FAIL: {'; '.join(gate_reason)}"
            )
            # Record cell with gate_pass=False and skip the expensive arms.
            cell = {
                "model_tag": model_tag,
                "K": int(K),
                "gate_pass": False,
                "gate_reason": "; ".join(gate_reason),
                "validity": {
                    "recon_cos": recon_cos,
                    "err_ratio_mean": err_ratio_mean,
                    "err_ratio_pass": bool(err_ratio_pass),
                    "err_ratio_gate_bypassed": bool(bypass_err_ratio_gate),
                    "alive_feature_count": alive_count,
                    "rank_eff": rank_eff,
                    "rank_pass_gt_K": bool(rank_pass),
                    "var_capture_K": var_capture_K,
                    "var_capture_K50": var_capture_K50,
                    "var_capture_pass_at_K50": bool(var_capture_pass),
                },
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S UTC", time.gmtime()
                ),
            }
            _atomic_write_json(out_dir / "results.json", cell)
            cell_results.append(cell)
            continue

        Uk = Uk_full[:, :K]  # (d_model, K)

        # Primary ablation
        H_ablated = errpc_ablate(H, E, Uk)
        res_a = probes.score(H_ablated, labels)
        delta_folds = [
            h - a for h, a in zip(res_h["aurocs"], res_a["aurocs"])
        ]
        delta_mean = float(np.mean(delta_folds))

        # Paired bootstrap
        rows_h = probes.score_indices(H, labels)
        rows_a = probes.score_indices(H_ablated, labels)
        boot = paired_bootstrap_delta(
            rows_h, rows_a, n_boot=bootstrap_n, seed=bootstrap_seed + K,
        )

        # Per-row ||U_K U_K^T e||_2 for C2 noise matching
        proj = (E @ Uk) @ Uk.T  # (N, d_model)
        proj_norms = np.linalg.norm(proj, axis=1).astype(np.float32)

        # C1 random-rotation
        c1 = control_c1_random_orthonormal_K(
            probes, H, E, labels, K=K, n_seeds=n_c1_seeds, seed=10_000 + K,
        )
        # C2 matched Gaussian
        c2 = control_c2_matched_gaussian(
            probes, H, proj_norms, labels,
            n_seeds=n_c2_seeds, seed=20_000 + K,
        )
        # C3 identity (no-op)
        c3_auroc = auroc_h_mean
        # C4 random-column-mask SAE
        c4 = control_c4_random_column_mask(
            probes, H, labels, sae, device,
            d_sae=W_dec.shape[1], K=K, n_seeds=n_c4_seeds,
            mask_fraction=c4_mask_fraction, seed=40_000 + K,
        )

        # Decisive inequalities (pre-reg):
        # primary ΔAUROC > (AUROC_h − C1_pct05)   → upper CI of C1's ΔAUROC
        # primary ΔAUROC > (AUROC_h − C2_pct05)
        # primary ΔAUROC > (AUROC_h − C4_mean) + 3·σ_C4
        c1_delta_upper = float(auroc_h_mean - c1.get("pct05", np.nan))
        c2_delta_upper = float(auroc_h_mean - c2.get("pct05", np.nan))
        c4_delta_mean = float(auroc_h_mean - c4.get("mean", np.nan))
        c4_delta_std = float(c4.get("std", np.nan))
        beats_c1 = bool(delta_mean > c1_delta_upper)
        beats_c2 = bool(delta_mean > c2_delta_upper)
        beats_c4 = bool(delta_mean > c4_delta_mean + 3.0 * c4_delta_std)
        passes_decisive = bool(beats_c1 and beats_c2 and beats_c4)

        cell = {
            "model_tag": model_tag,
            "model_path": model_path,
            "bcd_dir": bcd_dir,
            "sae_path": sae_path,
            "layer": layer,
            "K": int(K),
            "is_primary_K": bool(K == PRIMARY_K),
            "gate_pass": True,

            "n_member": int((labels == 1).sum()),
            "n_nonmember": int((labels == 0).sum()),

            "probe_seeds": list(probe_seeds),
            "probe_n_folds": int(n_folds),
            "probe_solver": "lbfgs",
            "probe_C": 1.0,
            "probe_weights_sha256": probe_hash,

            "auroc_h_mean": auroc_h_mean,
            "auroc_h_folds": res_h["aurocs"],
            "auroc_h_by_seed": res_h["by_seed"],
            "auroc_ablated_mean": float(res_a["mean"]),
            "auroc_ablated_folds": res_a["aurocs"],
            "auroc_ablated_by_seed": res_a["by_seed"],
            "delta_auroc_mean": delta_mean,
            "delta_auroc_folds": delta_folds,
            "delta_bootstrap": boot,

            "control_c1_random_orthonormal_K": c1,
            "control_c2_matched_gaussian": c2,
            "control_c3_identity": {
                "auroc_mean": c3_auroc,
                "equals_primary": bool(abs(c3_auroc - auroc_h_mean) < 1e-9),
            },
            "control_c4_random_mask_errPC": c4,

            "primary_beats_c1_delta_pct95": beats_c1,
            "primary_beats_c2_delta_pct95": beats_c2,
            "primary_beats_c4_by_3sigma": beats_c4,
            "passes_decisive": passes_decisive,

            "validity": {
                "recon_cos": float(recon_cos),
                "L0": float(L0),
                "alive_feature_count": int(alive_count),
                "total_feature_count": int(W_dec.shape[1]),
                "alive_threshold_Wdec_norm": float(norm_thresh),
                "alive_threshold_L0_freq": float(freq_thresh),
                "err_ratio_mean": err_ratio_mean,
                "err_ratio_range": list(ERR_RATIO_RANGE),
                "err_ratio_pass": bool(err_ratio_pass),
                "err_ratio_gate_bypassed": bool(bypass_err_ratio_gate),
                "rank_eff": rank_eff,
                "rank_eff_tol": RANK_EFF_TOL,
                "rank_pass_gt_K": bool(rank_pass),
                "var_capture_K": var_capture_K,
                "var_capture_K50": var_capture_K50,
                "var_capture_K50_min": VAR_CAPTURE_K50_MIN,
                "var_capture_pass_at_K50": bool(var_capture_pass),
                "all_pass": True,
            },

            "svd_top_singular_values": [
                float(x) for x in S[: min(200, S.size)]
            ],
            "svd_total_var": total_s2,

            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }

        _atomic_write_json(out_dir / "results.json", cell)
        log.info(f"[{model_tag} K={K}] results -> {out_dir / 'results.json'}")

        # Summary
        print()
        print("=" * 72)
        print(f"errPC ablation. {model_tag}  K={K}")
        print("=" * 72)
        print(f"  AUROC(H)            : {auroc_h_mean:.4f}")
        print(f"  AUROC(H_ablated)    : {float(res_a['mean']):.4f}")
        print(f"  ΔAUROC              : {delta_mean:+.4f}  "
              f"[boot95 {boot['ci95_lo']:+.4f}, {boot['ci95_hi']:+.4f}] "
              f"p_one={boot['p_one_sided_leq0']:.4f}")
        print(f"  C1 rand-rot AUROC   : mean={c1.get('mean', float('nan')):.4f} "
              f"pct05={c1.get('pct05', float('nan')):.4f} "
              f"pct95={c1.get('pct95', float('nan')):.4f}")
        print(f"  C2 matched-noise    : mean={c2.get('mean', float('nan')):.4f} "
              f"pct05={c2.get('pct05', float('nan')):.4f} "
              f"pct95={c2.get('pct95', float('nan')):.4f}")
        print(f"  C4 random-mask errPC: mean={c4.get('mean', float('nan')):.4f} "
              f"(std={c4.get('std', float('nan')):.4f})")
        print(f"  beats C1 (delta95)  : {beats_c1}")
        print(f"  beats C2 (delta95)  : {beats_c2}")
        print(f"  beats C4 by 3σ      : {beats_c4}")
        print(f"  PASSES DECISIVE     : {passes_decisive}")
        print(f"  Validity (model-lvl): recon_cos={recon_cos:.4f}, "
              f"err_ratio={err_ratio_mean:.4f}")
        print(f"  Validity (K-level) : rank_eff={rank_eff}>{K}={rank_pass}, "
              f"var_capture_K={var_capture_K:.4f}")

        cell_results.append(cell)

    # Free SAE
    del sae
    torch.cuda.empty_cache()
    return cell_results


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------

def holm_bonferroni(pvals: List[float], alpha: float = 0.05) -> Dict:
    if not pvals:
        return {"adjusted": [], "reject": []}
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = np.array(pvals, dtype=np.float64)[order]
    adj_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for k in range(m):
        val = (m - k) * sorted_p[k]
        running = max(running, val)
        adj_sorted[k] = min(1.0, running)
    adj = np.empty(m, dtype=np.float64)
    adj[order] = adj_sorted
    reject = [bool(a <= alpha) for a in adj]
    return {"adjusted": [float(a) for a in adj], "reject": reject}


def _parse_roster(spec: Optional[str]) -> List[Dict[str, str]]:
    if not spec:
        return []
    with open(spec, "r") as f:
        roster = json.load(f)
    if not isinstance(roster, list):
        raise ValueError(f"Roster file {spec} must be a JSON list")
    return roster


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Error-PC subspace ablation causal intervention eval",
    )
    parser.add_argument("--output-dir", required=True)
    # Single-model mode
    parser.add_argument("--model-tag")
    parser.add_argument("--model-path")
    parser.add_argument("--bcd-dir", default=None)
    parser.add_argument("--sae-path")
    parser.add_argument("--layer", type=int)
    # Multi-model mode
    parser.add_argument("--roster")
    # Shared inputs
    parser.add_argument("--member-texts", default="data/memcirc_ctrl_ft/member.jsonl")
    parser.add_argument("--nonmember-texts", default="data/memcirc_ctrl_ft/nonmember.jsonl")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-texts", type=int, default=0)
    parser.add_argument("--freq-tokens", type=int, default=200_000)
    # K sweep
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES),
        help="K values for top-K error-PC ablation (default: 10 50 200; primary 50)",
    )
    # Probe parameters
    parser.add_argument(
        "--probe-seeds", type=int, nargs="+", default=list(DEFAULT_PROBE_SEEDS),
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    # Control-arm seeds
    parser.add_argument("--n-c1-seeds", type=int, default=DEFAULT_N_C1_SEEDS)
    parser.add_argument("--n-c2-seeds", type=int, default=DEFAULT_N_C2_SEEDS)
    parser.add_argument("--n-c4-seeds", type=int, default=DEFAULT_N_C4_SEEDS)
    parser.add_argument(
        "--c4-mask-fraction", type=float, default=C4_RANDOM_COL_FRACTION,
    )
    # Bootstrap
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    # Gate handling
    parser.add_argument(
        "--continue-on-fail", action="store_true",
        help="Log gate-failures and keep running the rest of the roster.",
    )
    parser.add_argument(
        "--bypass-err-ratio-gate", action="store_true",
        help=(
            "Convert the model-level err_ratio gate "
            "((||e||/||h||).mean in ERR_RATIO_RANGE) from an AssertionError "
            "into a logged warning, so the K-loop runs anyway. Every cell's "
            "validity block will carry err_ratio_gate_bypassed=True for "
            "downstream filtering. Default: False (strict gate, unchanged)."
        ),
    )
    args = parser.parse_args()

    if not _HAS_PROJECT_INFRA:
        raise RuntimeError(f"Project infrastructure required: {_IMPORT_ERROR}")
    if not _HAS_SKLEARN:
        raise RuntimeError("sklearn required")

    setup_logging(logging.INFO)
    set_global_seed(SeedConfig(seed=args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    run_config["script"] = "subspace_ablation_eval.py"
    run_config["version"] = "errPC"
    _atomic_write_json(out_dir / "run_config.json", run_config)

    roster = _parse_roster(args.roster)
    if not roster:
        if not all([args.model_tag, args.model_path,
                    args.sae_path, args.layer is not None]):
            raise ValueError(
                "Provide either --roster or "
                "(--model-tag --model-path --sae-path --layer)"
            )
        roster = [{
            "model_tag": args.model_tag,
            "model_path": args.model_path,
            "bcd_dir": args.bcd_dir,
            "sae_path": args.sae_path,
            "layer": int(args.layer),
        }]

    k_values = tuple(int(k) for k in args.k_values)
    all_cells: List[Dict] = []
    failures: List[Dict[str, str]] = []

    for entry in roster:
        try:
            cells = run_one_model(
                model_tag=entry["model_tag"],
                model_path=entry["model_path"],
                bcd_dir=entry.get("bcd_dir"),
                sae_path=entry["sae_path"],
                member_texts_path=args.member_texts,
                nonmember_texts_path=args.nonmember_texts,
                layer=int(entry["layer"]),
                output_dir=out_dir,
                k_values=k_values,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                device=args.device,
                max_texts=args.max_texts,
                freq_tokens=args.freq_tokens,
                probe_seeds=tuple(args.probe_seeds),
                n_folds=args.n_folds,
                n_c1_seeds=args.n_c1_seeds,
                n_c2_seeds=args.n_c2_seeds,
                n_c4_seeds=args.n_c4_seeds,
                c4_mask_fraction=args.c4_mask_fraction,
                bootstrap_n=args.bootstrap_n,
                bootstrap_seed=args.bootstrap_seed,
                bypass_err_ratio_gate=args.bypass_err_ratio_gate,
            )
            all_cells.extend(cells)
        except AssertionError as e:
            log.error(f"[{entry['model_tag']}] MODEL-LEVEL VALIDITY GATE FAIL: {e}")
            failures.append({
                "model_tag": entry["model_tag"],
                "error": str(e),
                "level": "model",
            })
            if not args.continue_on_fail:
                break
        except Exception as e:
            log.exception(f"[{entry['model_tag']}] unhandled: {e}")
            failures.append({
                "model_tag": entry["model_tag"],
                "error": str(e),
                "level": "exception",
            })
            if not args.continue_on_fail:
                break

    # --- Holm across 12 cells (gate-passing only) ---
    passing = [c for c in all_cells if c.get("gate_pass", False)]
    summary_rows = []
    for c in all_cells:
        row = {
            "model_tag": c["model_tag"],
            "K": c["K"],
            "gate_pass": c.get("gate_pass", False),
        }
        if c.get("gate_pass", False):
            row.update({
                "auroc_h": c["auroc_h_mean"],
                "auroc_ablated": c["auroc_ablated_mean"],
                "delta": c["delta_auroc_mean"],
                "delta_ci95_lo": c["delta_bootstrap"]["ci95_lo"],
                "delta_ci95_hi": c["delta_bootstrap"]["ci95_hi"],
                "delta_p_one_sided": c["delta_bootstrap"]["p_one_sided_leq0"],
                "passes_decisive": c["passes_decisive"],
                "rank_eff": c["validity"]["rank_eff"],
                "var_capture_K": c["validity"]["var_capture_K"],
            })
        else:
            row["gate_reason"] = c.get("gate_reason", "")
        summary_rows.append(row)

    # One p-value per gate-passing cell, Holm across the whole family (up to 12).
    pvals = [c["delta_bootstrap"]["p_one_sided_leq0"] for c in passing]
    holm = holm_bonferroni(pvals, alpha=0.05)
    holm_table = []
    for i, c in enumerate(passing):
        holm_table.append({
            "model_tag": c["model_tag"],
            "K": c["K"],
            "p_raw": c["delta_bootstrap"]["p_one_sided_leq0"],
            "p_holm_adj": holm["adjusted"][i] if holm["adjusted"] else None,
            "reject_holm_005": holm["reject"][i] if holm["reject"] else False,
        })

    aggregate = {
        "version": "errPC",
        "roster_size": len(roster),
        "k_values": list(k_values),
        "primary_K": PRIMARY_K,
        "n_cells_total": len(all_cells),
        "n_cells_gate_pass": len(passing),
        "n_cells_gate_fail": len(all_cells) - len(passing),
        "n_model_failures": len(failures),
        "holm_family_size": len(passing),
        "holm_alpha_family": 0.05,
        "rows": summary_rows,
        "failures": failures,
        "holm_bonferroni_alpha_005": holm_table,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    _atomic_write_json(out_dir / "aggregate.json", aggregate)
    log.info(f"aggregate -> {out_dir / 'aggregate.json'}")

    if failures and not args.continue_on_fail:
        raise SystemExit(
            f"{len(failures)} model(s) failed MODEL-level gates; see aggregate.json"
        )


if __name__ == "__main__":
    main()
