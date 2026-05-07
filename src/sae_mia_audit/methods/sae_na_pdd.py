from __future__ import annotations

"""SAE-NA-PDD: NA-PDD-style pretraining-data detection in SAE feature space.

This module implements the integration described in the repository docs:

  - Replace NA-PDD's neuron basis with *sparse autoencoder (SAE) features*.
  - Define "feature activity" from SAE codes (top-k or per-feature thresholds).
  - Estimate member/non-member feature frequencies on reference sets.
  - Compute log-odds weights w_i = log((f_mem+eps)/(f_non+eps)).
  - Score candidate texts by a magnitude-aware member-vs-non advantage.
  - Select layers by validation separation (AUC, TPR@FPR, or tail_sep) rather
    than simple count heuristics.

The goal is to make NA-PDD less dominated by polysemantic lexical neurons by
using SAE features, which are often more monosemantic in practice.

Extended components:
  - **Frequency calibration**: DC-PDD-style corpus-level feature frequency
    calibration (TF-IDF in SAE feature space). Weights rare-but-discriminative
    features higher. Adapts Zhang et al. (EMNLP 2024) to SAE space.
  - **Per-sample feature selection**: Min-K%-style selection in feature space.
    Only the top-K% most extreme features (by |z*w|) contribute per sample,
    preventing noise dilution. Adapts Shi et al. (2023) to SAE space.
  - **Feature z-scoring**: Per-feature normalization using non-member statistics
    (mu, sigma). Makes cross-feature aggregation meaningful and ensures
    score_norm_zscore actually changes scores.
  - **Multi-scale temporal evidence**: Computes evidence at multiple temporal
    resolutions (token, window, sequence) and fuses them. Captures both
    local memorisation patterns and global distribution shifts.

Safety/ethics:
  This repo is intended for *authorized auditing* of open-weight models and
  datasets. Do not use these methods against systems you do not own or do not
  have explicit permission to test.

Key design decisions for robustness:
  - Stability gating is applied BEFORE aggregation and layer selection.
  - Layer selection defaults to tpr@fpr rather than AUROC.
  - Score normalisation is fitted on non-members only.
  - Frequency calibration uses separate reference corpus statistics.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as torchF
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from sae_mia_audit.data.pdd import PDDExample
from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
from sae_mia_audit.eval.calibration import NormMode, NormScope, create_normalizer
from sae_mia_audit.methods.agg import AggMode, aggregate
from sae_mia_audit.models.wrapper import CausalLMWrapper
from sae_mia_audit.sae.io import load_sae_checkpoint_any
from sae_mia_audit.sae.adapters import SAEProtocol


# Note: AggMode is imported from agg.py, but we keep local aliases for backward compat
ActivationMode = Literal["topk", "threshold"]
LayerSelectMetric = Literal["auc", "tpr@fpr", "tail_sep"]
ScoreTransform = Literal["log_ratio", "ratio"]
FeatureSelectMetric = Literal["tpr@fpr", "tail_sep", "auc"]


@dataclass(frozen=True)
class SAENAPDDConfig:
    """Configuration for SAE-NA-PDD.

    Notes:
        - `sae_paths` and `layer_indices` can repeat a layer index to indicate
          multiple SAEs for the same layer (e.g., an L1 sweep). The implementation
          will group them as {layer -> [sae_0, sae_1, ...]}.
        - Activation-mode defaults to `topk`, which avoids brittle global thresholds.
        - Stability gating is applied BEFORE aggregation and layer selection.
        - Layer selection defaults to tpr@fpr rather than AUROC.
    """

    # SAE checkpoints and their transformer layer indices
    sae_paths: List[str]
    layer_indices: List[int]

    # How to aggregate SAE codes across tokens (and across features/layers)
    agg: AggMode = "max"
    agg_k: Optional[int] = 8  # for topk_mean
    agg_trim_frac: Optional[float] = 0.1  # for trimmed_mean

    # Activity definition (for frequency stats)
    activation_mode: ActivationMode = "topk"
    topk_active: int = 64

    # Per-feature thresholding (optional alternative to top-k)
    threshold_quantile: float = 0.95
    threshold_sample_size: int = 2048

    # Quality filters
    min_activation_rate: float = 1e-4  # drop dead features
    max_activation_rate: float = 0.25  # drop overly generic features

    # Stability filter (optional, uses multiple SAEs per layer)
    # NOTE: Stability mask is applied BEFORE aggregation and layer selection
    stability_enabled: bool = True
    stability_top_n: int = 1024  # only check stability for top-|w| features
    stability_cosine: float = 0.9
    stability_require_all: bool = True  # require match in all other SAEs

    # Feature selection (default to tpr@fpr to avoid silent AUROC optimization)
    feature_select_metric: FeatureSelectMetric = "tpr@fpr"
    feature_select_fpr: float = 1e-3
    # NOTE: n_features_per_layer is reserved for future use (feature selection by count)
    # Currently all features passing quality/stability filters are used
    n_features_per_layer: Optional[int] = None  # None = use all (after quality/stability filtering)

    # Layer selection (default to tpr@fpr rather than AUROC)
    top_k_layers: int = 5
    layer_select_metric: LayerSelectMetric = "tpr@fpr"  # Changed from "auc"
    layer_select_fpr: float = 1e-3  # for metric 'tpr@fpr'

    # Score normalization (fit on non-members only from reference set)
    # "none": no normalization (raw scores)
    # "zscore": z-score normalization per-layer or global
    # "quantile": quantile normalization (maps to CDF then optionally to Gaussian)
    score_norm: NormMode = "none"
    # "feature": normalize each layer independently before aggregation
    # "sae" or "global": normalize aggregated scores
    score_norm_scope: NormScope = "feature"

    # Calibration split inside the provided calibration set
    ref_frac: float = 0.5
    seed: int = 0

    # Scoring
    # eps: Laplace smoothing for frequency log-ratios (log((f+eps)/(g+eps))).
    # Separate from hard-coded 1e-8 used as numerical stability floors for
    # std and norm denominators throughout the scoring pipeline.
    eps: float = 1e-6
    delta: float = 1e-6
    score_transform: ScoreTransform = "log_ratio"

    # Tokenization/batching
    seq_len: int = 256
    batch_size: int = 4

    # Ensemble across SAEs within a layer
    ensemble_across_saes: bool = False  # default: use the first SAE per layer

    # Frequency smoothing / feature selection (Bayesian)
    freq_prior: float = 0.5          # Jeffreys prior; use 1.0 for Laplace
    min_total_count: int = 2         # drop features seen < this many times total
    dominance_alpha: float = 1.0     # if >1, keep only |log_ratio| >= log(alpha)
    weight_mode: Literal["log_ratio", "log_odds"] = "log_ratio"

    # Token-level pooling (inspired by Gap-K% sequential smoothing)
    pooling: Literal["sequence", "token_topk"] = "sequence"
    token_smooth_window: int = 16      # window length for smoothing
    token_topk_percent: float = 0.2    # fraction of tokens/windows to keep (top evidence)

    # Scoring mode: "magnitude" uses z*w (original), "binary" uses activity mask * w
    # Weight estimation is now aligned with scoring mode:
    # - "binary": weights from binary activation frequencies (p_mem/p_non)
    # - "magnitude": weights from mean activation magnitudes (mu_mem/mu_non)
    # This alignment ensures the weighting scheme matches how scores are computed.
    scoring_mode: Literal["magnitude", "binary"] = "magnitude"

    # =====================================================================
    # Extended Components: Frequency Calibration, Feature Selection, Z-Scoring
    # These adapt ideas from DC-PDD, Min-K%, and ReCaLL to SAE feature space.
    # =====================================================================

    # --- Frequency Calibration (DC-PDD in SAE space) ---
    # Inspired by Zhang et al. (EMNLP 2024) "DC-PDD", which calibrates token
    # probabilities using corpus-level frequency distributions.
    # We adapt this to SAE features: weight_i *= log(1/corpus_freq_i + eps)
    # This applies TF-IDF-style calibration: rare features get higher weight,
    # frequent (generic) features are downweighted.
    # Novel: DC-PDD operates on token probabilities; we apply it to SAE codes.
    freq_calibration: bool = False  # enable via --sae-na-freq-calibration
    freq_calibration_temperature: float = 1.0  # temperature scaling for IDF weights

    # --- Per-Sample Feature Selection (Min-K% in Feature Space) ---
    # Inspired by Min-K% (Shi et al. 2023), which selects the K% most
    # surprising tokens for scoring. We adapt this to SAE features:
    # for each sample, only score using the top-K% features by |z*w|.
    # Novel: Min-K% operates on token log-probs; we apply it to SAE features.
    sample_topk_features: float = 0.0  # 0 = disabled; 0.3 = use top 30% per sample

    # --- Feature Z-Scoring ---
    # Per-feature normalization using reference non-member statistics.
    # For each feature i, compute (z_i - mu_non_i) / sigma_non_i before scoring.
    # This makes cross-feature aggregation meaningful and ensures features with
    # different activation scales contribute proportionally to the score.
    # This ALSO fixes the ablation bug where score_norm_zscore had no effect:
    # without per-feature z-scoring, z-scores of the aggregate are dominated
    # by scale differences between features.
    feature_zscore: bool = False  # enable via --sae-na-feature-zscore

    # --- Multi-Scale Temporal Evidence ---
    # Compute evidence at multiple temporal resolutions and fuse them.
    # Different membership signals manifest at different scales:
    # - Token-level: individual rare token memorisation
    # - Window-level: phrase/sentence memorisation
    # - Sequence-level: global distributional shift
    # The final score is a weighted combination of all scales.
    multi_scale_windows: Tuple[int, ...] = (1, 8, 32)  # window sizes for multi-scale
    multi_scale_enabled: bool = False  # disabled by default for backward compat

    # =====================================================================
    # Auxiliary Components: Reconstruction Error, Context Z-Scoring, SAE-ReCaLL
    # Three structurally different ideas exploiting orthogonal signals:
    # 1. Reconstruction error: SAE decode fidelity as membership proxy
    # 2. Context-dependent z-scoring: per-position feature normalization
    # 3. SAE-ReCaLL: prefix-conditioned score ratio (robustness to context)
    # =====================================================================

    # --- SAE Reconstruction Error (Anomaly Detection Transfer) ---
    # Use ||h - SAE.decode(SAE.encode(h))||^2 as an auxiliary membership signal.
    # Members' hidden states should lie closer to the SAE's learned submanifold
    # (lower reconstruction error) because the SAE was trained on in-distribution
    # activations (same distribution as the LLM's training data).
    # Novel: No MIA work has used SAE reconstruction error as membership evidence.
    # Inspired by sparse-representation-based anomaly detection literature.
    recon_error_weight: float = 0.0  # 0 = disabled; 0.3 = blend 30% recon, 70% napdd
    # "layer_mean": average recon error across selected layers
    # "layer_min": min recon error (best-reconstructed layer)
    recon_error_pool: Literal["layer_mean", "layer_min"] = "layer_mean"

    # --- Context-Dependent Feature Z-Scoring ("Feature-K%++") ---
    # Adapted from Min-K%++ (Zhang et al., ICLR 2025): instead of normalizing
    # features by global non-member stats (feature_zscore), normalize each
    # feature at each token position by the LOCAL feature distribution at that
    # position: z_norm_{t,i} = (z_{t,i} - mu_t) / sigma_t, where mu_t and
    # sigma_t are computed across all F features at position t (dimension 2
    # of the [B, T, F] tensor).
    #
    # Contrast with Min-K%++ which normalizes token log-probs by vocabulary-wide
    # statistics (mu, sigma over the V-dimensional next-token distribution).
    # Here we analogously normalize SAE codes by the F-dimensional feature
    # activation distribution at each position.
    #
    # This asks: "Is feature i unusually active at THIS position?" rather than
    # "Is feature i unusually active compared to non-members on average?"
    context_zscore: bool = False  # enable via --sae-na-context-zscore

    # --- SAE-ReCaLL: Prefix-Conditioned Score Perturbation ---
    # Adapted from ReCaLL (Xie et al., EMNLP 2024): compare SAE-NA-PDD score
    # of text x alone vs. score of x preceded by irrelevant prefix P.
    # Members are "anchored" — their SAE score is robust to prefix perturbation.
    # Non-members' scores shift more because the model hasn't memorised them.
    # score_recall = SAE_score(prefix + x) / SAE_score(x)   (ratio mode)
    # score_recall = SAE_score(prefix + x) - SAE_score(x)   (diff mode)
    # Novel: ReCaLL has only been applied to token log-probs. Applying it to
    # SAE feature-space scores is unexplored and leverages the abstraction
    # advantage of SAE features over surface-level token probabilities.
    recall_enabled: bool = False  # enable via --sae-na-recall
    recall_n_prefixes: int = 3  # number of non-member prefixes to ensemble
    recall_prefix_tokens: int = 64  # number of tokens from each prefix text
    recall_mode: Literal["ratio", "diff"] = "ratio"  # how to combine conditioned/unconditioned


def _split_ref_val(examples: Sequence[PDDExample], ref_frac: float, seed: int) -> Tuple[List[PDDExample], List[PDDExample]]:
    """Stratified split of examples into (ref, val) by label."""
    if not 0.0 < ref_frac < 1.0:
        raise ValueError(f"ref_frac must be in (0,1), got {ref_frac}")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(examples))
    labels = np.asarray([e.label for e in examples], dtype=int)
    ref_idx: List[int] = []
    val_idx: List[int] = []
    for y in [0, 1]:
        y_idx = idx[labels == y]
        rng.shuffle(y_idx)
        m = len(y_idx)
        if m < 2:
            raise ValueError("Need at least 2 examples per class for a ref/val split.")
        k = int(round(ref_frac * m))
        k = min(max(1, k), m - 1)  # ensure at least 1 in both ref and val
        ref_idx.extend(y_idx[:k].tolist())
        val_idx.extend(y_idx[k:].tolist())
    rng.shuffle(ref_idx)
    rng.shuffle(val_idx)
    ref = [examples[i] for i in ref_idx]
    val = [examples[i] for i in val_idx]
    return ref, val


def _tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _thr = roc_curve(y_true, scores)
    # roc_curve returns increasing fpr; find best tpr with fpr <= target
    mask = fpr <= fpr_target
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def _tail_separation_metric(
    mem_scores: np.ndarray,
    non_scores: np.ndarray,
    qs: Tuple[float, ...] = (0.95, 0.96, 0.97, 0.98, 0.99),
) -> float:
    """Compute mean tail separation across upper quantiles.
    
    Returns the mean of (quantile(mem, q) - quantile(non, q)) for q in qs.
    Higher => better separation at the tails.
    """
    if len(mem_scores) == 0 or len(non_scores) == 0:
        return 0.0
    diffs = []
    for q in qs:
        d = float(np.quantile(mem_scores, q) - np.quantile(non_scores, q))
        diffs.append(d)
    return float(np.mean(diffs)) if diffs else 0.0


@torch.no_grad()
def _aggregate_codes(
    z: torch.Tensor,
    agg: AggMode,
    agg_k: Optional[int] = None,
    agg_trim_frac: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Aggregate per-token SAE codes to per-sequence codes.

    Args:
        z: [B, T, F]
        agg: Aggregation mode (mean, max, topk_mean, trimmed_mean).
        agg_k: k for topk_mean.
        agg_trim_frac: Trim fraction for trimmed_mean.
        attention_mask: [B, T] with 1 for real tokens, 0 for padding. If None,
            all tokens are included (backward compatible).
    Returns:
        z_agg: [B, F]
    """
    B, T, F = z.shape
    
    # If no attention mask, fall back to including all tokens
    if attention_mask is None:
        if agg == "mean":
            return z.mean(dim=1)
        if agg == "max":
            return z.max(dim=1).values
        # For complex modes, use the shared aggregate function
        z_np = z.detach().float().cpu().numpy()
        result = np.zeros((B, F), dtype=np.float32)
        for b in range(B):
            for f in range(F):
                result[b, f] = aggregate(z_np[b, :, f], mode=agg, k=agg_k, trim_frac=agg_trim_frac)
        return torch.from_numpy(result).to(z.device, dtype=z.dtype)
    
    # Attention-mask-aware aggregation (excludes padding tokens)
    m = attention_mask.to(device=z.device, dtype=torch.bool)  # [B, T]
    
    if agg == "mean":
        mf = m.to(dtype=z.dtype).unsqueeze(-1)  # [B, T, 1]
        denom = mf.sum(dim=1).clamp_min(1.0)    # [B, 1]
        return (z * mf).sum(dim=1) / denom
    
    if agg == "max":
        # For nonnegative SAE codes, setting pads to -inf prevents pads winning the max
        z_masked = z.masked_fill(~m.unsqueeze(-1), float("-inf"))
        out = z_masked.max(dim=1).values
        # If a sequence is fully masked (shouldn't happen), replace -inf with 0
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    
    # For complex modes (topk_mean, trimmed_mean), filter out padding per-sample
    z_np = z.detach().float().cpu().numpy()
    m_np = m.detach().cpu().numpy()
    result = np.zeros((B, F), dtype=np.float32)
    for b in range(B):
        valid_len = int(m_np[b].sum())
        if valid_len == 0:
            continue
        for f in range(F):
            result[b, f] = aggregate(z_np[b, :valid_len, f], mode=agg, k=agg_k, trim_frac=agg_trim_frac)
    return torch.from_numpy(result).to(z.device, dtype=z.dtype)


class SAENAPDD:
    """NA-PDD-style pretraining data detection in SAE feature space.

    Key design decisions for robustness:
      - Stability gating is applied BEFORE aggregation and layer selection.
      - Layer selection defaults to tpr@fpr rather than AUROC.
      - Score normalisation is fitted on non-members only.
    """

    def __init__(self, model: CausalLMWrapper, cfg: SAENAPDDConfig, device: Optional[str] = None):
        self.model = model
        self.cfg = cfg
        self.device = next(model.model.parameters()).device

        if len(cfg.sae_paths) != len(cfg.layer_indices):
            raise ValueError("sae_paths and layer_indices must match length")

        # Load SAEs and group by layer.
        groups: Dict[int, List[SAEProtocol]] = {}
        for p, li in zip(cfg.sae_paths, cfg.layer_indices):
            sae = load_sae_checkpoint_any(p, device=self.device)
            groups.setdefault(int(li), []).append(sae)

        self.layer_to_saes: Dict[int, List[SAEProtocol]] = dict(sorted(groups.items(), key=lambda x: x[0]))
        self.layers: List[int] = list(self.layer_to_saes.keys())

        # Learned state after fit()
        # Per-layer, per-feature thresholds (optional) for the *primary* SAE.
        self.tau: Dict[int, torch.Tensor] = {}
        # Per-layer, per-feature weights w_i for the *primary* SAE.
        # These are used for interpretability outputs and mechanistic attribution.
        self.w: Dict[int, torch.Tensor] = {}
        # Per-layer feature masks for the *primary* SAE.
        self.feature_mask: Dict[int, torch.Tensor] = {}

        # Full sweep state (for ensembling across SAE hyperparameters):
        #   layer -> list aligned with self.layer_to_saes[layer]
        self.tau_sweep: Dict[int, List[torch.Tensor]] = {}
        self.w_sweep: Dict[int, List[torch.Tensor]] = {}
        self.feature_mask_sweep: Dict[int, List[torch.Tensor]] = {}
        # Selected discriminative layers (subset of self.layers)
        self.selected_layers: List[int] = []
        # Per-layer validation metrics
        self.layer_val: Dict[int, Dict[str, float]] = {}
        
        # Score normalization (fitted on reference non-members)
        # This is the normalizer object from calibration.py
        self._score_normalizer: Optional[Any] = None
        # Per-layer score statistics for normalization (fitted on ref nonmembers)
        self._layer_score_stats: Dict[int, Dict[str, float]] = {}

        # Per-layer per-feature statistics for frequency calibration and z-scoring
        # corpus_freq[layer] = [F] float: corpus-level activation frequency per feature
        self._corpus_freq: Dict[int, torch.Tensor] = {}
        # feature_stats[layer] = {"mu": [F], "sigma": [F]}: non-member activation stats
        self._feature_stats: Dict[int, Dict[str, torch.Tensor]] = {}
        # idf_weights[layer] = [F] float: IDF weights log(1/corpus_freq + eps)
        self._idf_weights: Dict[int, torch.Tensor] = {}
        # multi_scale_weights[layer] = [n_scales] float: learned scale weights
        self._multi_scale_weights: Dict[int, np.ndarray] = {}

        # SAE-ReCaLL prefix texts (sampled from non-members during fit)
        self._recall_prefix_ids: List[torch.Tensor] = []  # list of [prefix_tokens] tensors

    # ------------------------- Extraction -------------------------

    @torch.no_grad()
    def _extract_layer_codes(
        self,
        texts: Sequence[str],
        layer_idx: int,
        sae: SAEProtocol,
        return_token_codes: bool = False,
    ) -> torch.Tensor:
        """Extract SAE codes for a single layer and SAE.

        Args:
            texts: list of input texts.
            layer_idx: transformer layer index (0-indexed).
            sae: SAE trained on residual-stream activations at this layer.
            return_token_codes: if True, returns [N, T, F]; else returns [N, F].
        """
        # Explicit random_crop=False for deterministic evaluation
        tok_cfg = TokenizeConfig(seq_len=self.cfg.seq_len, random_crop=False)
        out_codes: List[torch.Tensor] = []
        out_tok_codes: List[torch.Tensor] = []

        for i in tqdm(
            range(0, len(texts), self.cfg.batch_size),
            desc=f"sae_codes_l{layer_idx}",
            dynamic_ncols=True,
        ):
            chunk = list(texts[i : i + self.cfg.batch_size])
            batch = tokenize_batch(self.model.tokenizer, chunk, tok_cfg)
            input_ids = batch["input_ids"].to(self.model.model.device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(self.model.model.device)

            # NOTE: output_hidden_states=True returns all layers; for large runs
            # you may want a hook-based recorder to reduce memory.
            outputs = self.model.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            hs = outputs.hidden_states
            h = hs[layer_idx + 1]  # [B, T, D]
            B, T, D = h.shape

            x = h.reshape(B * T, D).to(self.device)

            # Ensure SAE is on correct device
            if next(sae.parameters()).device != x.device:
                sae.to(x.device)
                sae.eval()

            # FIX: ensure dtype match
            x = x.to(dtype=next(sae.parameters()).dtype)

            z = sae.encode(x).reshape(B, T, -1)  # [B, T, F]
            z_agg = _aggregate_codes(z, self.cfg.agg, self.cfg.agg_k, self.cfg.agg_trim_frac, attention_mask=attn)  # [B, F]

            out_codes.append(z_agg.detach().float().cpu())
            if return_token_codes:
                out_tok_codes.append(z.detach().float().cpu())

        codes = torch.cat(out_codes, dim=0)
        if return_token_codes:
            return torch.cat(out_tok_codes, dim=0)
        return codes

    # ------------------------- Activity + frequencies -------------------------

    def _active_mask(self, z_agg: torch.Tensor, tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return boolean activity mask [B, F] from aggregated codes.

        Args:
            z_agg: [B, F] nonnegative aggregated SAE codes.
            tau: Optional per-feature thresholds [F] used when activation_mode='threshold'.
        """
        if self.cfg.activation_mode == "topk":
            B, F = z_agg.shape
            k = min(int(self.cfg.topk_active), F)
            # Take top-k by activation magnitude
            idx = torch.topk(z_agg, k=k, dim=1, largest=True).indices  # [B, k]
            mask = torch.zeros((B, F), dtype=torch.bool)
            # Only count strictly positive activations as "active".
            # (ReLU implies nonneg, but top-k may include zeros.)
            row = torch.arange(B)[:, None]
            mask[row, idx] = z_agg[row, idx] > 0
            return mask

        if tau is None:
            raise RuntimeError("activation_mode='threshold' requires per-feature tau")
        return z_agg > tau[None, :]

    @torch.no_grad()
    def _estimate_thresholds(self, layer_idx: int, sae: SAEProtocol, ref_texts: Sequence[str]) -> torch.Tensor:
        """Compute per-feature thresholds tau_i using a sample of reference texts."""
        n = min(len(ref_texts), int(self.cfg.threshold_sample_size))
        if n <= 0:
            raise ValueError("No reference texts provided for threshold estimation")
        texts = list(ref_texts)[:n]
        z = self._extract_layer_codes(texts, layer_idx=layer_idx, sae=sae, return_token_codes=False)  # [n, F]
        q = float(self.cfg.threshold_quantile)
        # torch.quantile works on CPU tensors
        tau = torch.quantile(z, q=q, dim=0)
        return tau

    @torch.no_grad()
    def _estimate_freqs(
        self,
        layer_idx: int,
        sae: SAEProtocol,
        member_texts: Sequence[str],
        nonmember_texts: Sequence[str],
        tau: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Estimate activation counts c_mem, c_non for one layer/SAE.
        
        When scoring_mode="binary", we count binary activations (feature active or not).
        When scoring_mode="magnitude", we accumulate activation magnitudes (sum of z).
        This ensures weights are estimated consistently with how they will be used.

        Returns:
            c_mem: [F] float - counts or sums for members
            c_non: [F] float - counts or sums for non-members
            n_mem: int - number of member samples
            n_non: int - number of non-member samples
        """
        use_magnitude = (self.cfg.scoring_mode == "magnitude")
        
        # We compute activity masks in batches and accumulate counts.
        def _freq(texts: Sequence[str], label: str) -> Tuple[torch.Tensor, int]:
            counts: Optional[torch.Tensor] = None
            n = 0
            # Explicit random_crop=False for deterministic evaluation
            tok_cfg = TokenizeConfig(seq_len=self.cfg.seq_len, random_crop=False)

            for i in tqdm(
                range(0, len(texts), self.cfg.batch_size),
                desc=f"freq_{label}_l{layer_idx}",
                dynamic_ncols=True,
            ):
                chunk = list(texts[i : i + self.cfg.batch_size])
                batch = tokenize_batch(self.model.tokenizer, chunk, tok_cfg)
                input_ids = batch["input_ids"].to(self.model.model.device)
                attn = batch.get("attention_mask", None)
                if attn is not None:
                    attn = attn.to(self.model.model.device)

                outputs = self.model.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
                hs = outputs.hidden_states
                h = hs[layer_idx + 1]  # [B, T, D]
                B, T, D = h.shape
                x = h.reshape(B * T, D).to(self.device)

                # Ensure SAE is on correct device
                if next(sae.parameters()).device != x.device:
                    sae.to(x.device)
                    sae.eval()

                # FIX: ensure dtype match (BF16 → FP32)
                x = x.to(dtype=next(sae.parameters()).dtype)

                z = sae.encode(x).reshape(B, T, -1)

                z_agg = _aggregate_codes(z, self.cfg.agg, self.cfg.agg_k, self.cfg.agg_trim_frac, attention_mask=attn).detach().float().cpu()  # [B, F]

                if use_magnitude:
                    # Accumulate magnitude (mean over batch)
                    contrib = z_agg.sum(dim=0)  # [F]
                else:
                    # Binary counting (original behaviour)
                    active = self._active_mask(z_agg, tau=tau).float().sum(dim=0)  # [F]
                    contrib = active
                
                if counts is None:
                    counts = torch.zeros_like(contrib)
                counts += contrib
                n += B
            if counts is None:
                # empty dataset
                counts = torch.zeros(sae.d_sae, dtype=torch.float32)
            return counts, n  # return raw counts/sums, not normalized

        c_mem, n_mem = _freq(member_texts, "mem")
        c_non, n_non = _freq(nonmember_texts, "non")
        return c_mem, c_non, n_mem, n_non

    # ------------------------- Fit -------------------------

    @torch.no_grad()
    def fit(
        self,
        calib_examples: Sequence[PDDExample],
        val_examples: Optional[Sequence[PDDExample]] = None,
    ) -> None:
        """Fit the SAE-NA-PDD scorer using calibration and validation sets.

        When val_examples is provided (RECOMMENDED to avoid leakage):
          - calib_examples is used as the reference set (estimate f_mem/f_non, weights w)
          - val_examples is used for layer selection (no data leakage)
          - cfg.ref_frac is IGNORED

        When val_examples is None (backward compatible, but data-starved):
          - calib_examples is split internally using cfg.ref_frac
          - WARNING: This starves both ref and val sets; not recommended.
        """
        ex = list(calib_examples)
        if len(ex) < 4:
            raise ValueError("Need at least a few calibration examples")

        # ------------------------------------------------------------------
        # FIX: ensure all SAEs are on the same device as the model
        # This is the ONLY correct place to do this.
        # ------------------------------------------------------------------
        model_device = next(self.model.model.parameters()).device
        for layer_saes in self.layer_to_saes.values():
            for sae in layer_saes:
                sae.to(model_device)
                sae.eval()
        # ------------------------------------------------------------------

        # Determine ref/val split strategy
        if val_examples is not None:
            # External validation set provided (preferred to avoid leakage)
            ref = list(calib_examples)
            val = list(val_examples)
        else:
            # Internal split (backward compatible but not recommended)
            ref, val = _split_ref_val(ex, ref_frac=self.cfg.ref_frac, seed=self.cfg.seed)

        ref_mem = [e.text for e in ref if e.label == 1]
        ref_non = [e.text for e in ref if e.label == 0]
        val_mem = [e.text for e in val if e.label == 1]
        val_non = [e.text for e in val if e.label == 0]
        if not ref_mem or not ref_non:
            raise ValueError("Calibration split did not produce both member and non-member reference sets")
        if not val_mem or not val_non:
            raise ValueError(
                "Validation split missing a class. Provide more calibration examples per class "
                "or adjust ref_frac so both classes appear in val without reusing ref examples."
            )

        # (everything below remains exactly as in your original file)


        # Fit thresholds (if needed), then frequencies, then weights.
        self.tau = {}
        self.w = {}
        self.feature_mask = {}
        self.tau_sweep = {}
        self.w_sweep = {}
        self.feature_mask_sweep = {}
        self.layer_val = {}

        for layer_idx in self.layers:
            saes = self.layer_to_saes[layer_idx]
            self.tau_sweep[layer_idx] = []
            self.w_sweep[layer_idx] = []
            self.feature_mask_sweep[layer_idx] = []

            # For each SAE in the per-layer sweep, compute its own w/mask.
            # This enables ensembling across SAE hyperparameters (each SAE defines its own feature basis).
            tau_primary: Optional[torch.Tensor] = None

            for s_idx, sae in enumerate(saes):
                tau = None
                if self.cfg.activation_mode == "threshold":
                    mix = list(ref_mem) + list(ref_non)
                    tau = self._estimate_thresholds(layer_idx, sae, mix)
                    self.tau_sweep[layer_idx].append(tau)
                    if s_idx == 0:
                        tau_primary = tau
                else:
                    # Placeholder so lists align with saes
                    self.tau_sweep[layer_idx].append(torch.zeros(sae.d_sae))

                c_mem, c_non, n_mem, n_non = self._estimate_freqs(layer_idx, sae, ref_mem, ref_non, tau=tau)

                # Weight estimation depends on scoring_mode
                if self.cfg.scoring_mode == "magnitude":
                    # For magnitude mode: c_mem/c_non are sums of activations
                    # Compute mean activations and take log ratio
                    mu_mem = c_mem / max(1, n_mem)  # mean activation per sample for members
                    mu_non = c_non / max(1, n_non)  # mean activation per sample for non-members
                    
                    # Log ratio of mean activations (with smoothing)
                    w = torch.log((mu_mem + self.cfg.eps) / (mu_non + self.cfg.eps))
                    
                    # For quality filtering, use total activation rate
                    n_tot = max(1, n_mem + n_non)
                    total_act = (c_mem + c_non) / n_tot
                    # Filter features with too low or too high activation
                    mask = (total_act >= self.cfg.min_activation_rate) & (total_act <= self.cfg.max_activation_rate * 100)
                    # Require minimum total activation mass (scaled by sample count)
                    mask = mask & ((c_mem + c_non) >= float(self.cfg.min_total_count) * 0.1)
                else:
                    # Binary mode (original behaviour): c_mem/c_non are activation counts
                    # Bayesian smoothing: compute smoothed Bernoulli probabilities
                    a = float(self.cfg.freq_prior)
                    p_mem = (c_mem + a) / (max(1, n_mem) + 2.0 * a)
                    p_non = (c_non + a) / (max(1, n_non) + 2.0 * a)

                    if self.cfg.weight_mode == "log_odds":
                        # Log-odds difference: logit(p_mem) - logit(p_non)
                        # where logit(p) = log(p/(1-p))
                        # This is the proper Bayesian log-odds ratio, which behaves
                        # very differently from log_ratio when p_mem, p_non are NOT
                        # close to each other. The key mathematical difference:
                        # - log_ratio: log(p_mem/p_non) ≈ (p_mem - p_non)/p_non for small diffs
                        # - log_odds: logit(p_mem) - logit(p_non) amplifies differences
                        #   near p=0 and p=1 (sigmoid boundaries)
                        #
                        # FIX: Previous implementation clamped BEFORE computing log-odds,
                        # which made it nearly identical to log_ratio for small p values.
                        # Now we compute true log-odds with proper smoothing that preserves
                        # the characteristic S-curve amplification.
                        logit_mem = torch.log(p_mem + self.cfg.eps) - torch.log((1.0 - p_mem) + self.cfg.eps)
                        logit_non = torch.log(p_non + self.cfg.eps) - torch.log((1.0 - p_non) + self.cfg.eps)
                        w = logit_mem - logit_non
                    else:
                        # log ratio (close to NA-PDD's frequency dominance rule)
                        w = torch.log((p_mem + self.cfg.eps) / (p_non + self.cfg.eps))

                    # Quality filtering: rate + minimum evidence
                    n_tot = max(1, n_mem + n_non)
                    p_all = (c_mem + c_non) / n_tot
                    mask = (p_all >= self.cfg.min_activation_rate) & (p_all <= self.cfg.max_activation_rate)
                    mask = mask & ((c_mem + c_non) >= int(self.cfg.min_total_count))

                # Optional dominance filter (NA-PDD-style α threshold expressed as log-ratio)
                if float(self.cfg.dominance_alpha) > 1.0:
                    thr = float(np.log(self.cfg.dominance_alpha))
                    mask = mask & (w.abs() >= thr)

                self.w_sweep[layer_idx].append(w)
                self.feature_mask_sweep[layer_idx].append(mask)

            # Optional stability filter based on a sweep of SAEs for the same layer.
            # We apply this only to the primary SAE, since the goal is to filter the
            # features we later interpret.
            if self.cfg.stability_enabled and len(saes) > 1 and self.cfg.stability_top_n > 0:
                primary = saes[0]
                w0 = self.w_sweep[layer_idx][0]
                stable_mask = self._stability_mask_for_top_features(
                    layer_idx=layer_idx,
                    w=w0,
                    primary=primary,
                    others=saes[1:],
                    top_n=self.cfg.stability_top_n,
                )
                self.feature_mask_sweep[layer_idx][0] = self.feature_mask_sweep[layer_idx][0] & stable_mask

            # Convenience: expose the primary SAE weights/mask under self.w/self.feature_mask.
            self.w[layer_idx] = self.w_sweep[layer_idx][0]
            self.feature_mask[layer_idx] = self.feature_mask_sweep[layer_idx][0]
            if self.cfg.activation_mode == "threshold" and tau_primary is not None:
                self.tau[layer_idx] = tau_primary

        # Select layers by validation discriminativeness.
        self._select_layers(val_mem, val_non)
        
        # Fit frequency calibration and feature z-scoring statistics
        self._fit_feature_statistics(ref_mem, ref_non)
        
        # Prepare ReCaLL prefix token sequences from non-member reference texts
        if self.cfg.recall_enabled:
            self._fit_recall_prefixes(ref_non)
        
        # Fit score normalization on reference non-members if enabled
        self._fit_score_normalization(ref_non)

    @torch.no_grad()
    def _fit_feature_statistics(
        self,
        ref_mem: Sequence[str],
        ref_non: Sequence[str],
    ) -> None:
        """Fit feature-level statistics: frequency calibration, z-scoring, IDF weights.

        These are computed from the reference set and used during scoring.
        
        Novel components:
        1. Corpus frequency: computed from all reference texts (mem + non) as the
           base activation rate per feature. Used for IDF calibration.
        2. Feature z-scoring: mean/std of SAE codes on non-members, per feature.
           Used to normalize features to comparable scales before scoring.
        3. IDF weights: log(1/corpus_freq + eps), amplifying rare features.
           Inspired by DC-PDD (Zhang et al. EMNLP 2024) but applied to SAE features.
        """
        all_ref = list(ref_mem) + list(ref_non)
        
        for layer_idx in (self.selected_layers or self.layers):
            sae = self.layer_to_saes[layer_idx][0]
            
            # Extract aggregated SAE codes for all reference texts
            codes_all = self._extract_layer_codes(all_ref, layer_idx, sae)  # [N_all, F]
            n_all = codes_all.shape[0]
            n_non = len(ref_non)
            
            # Corpus frequency: fraction of texts where feature is active (> 0)
            active_mask = (codes_all > 0).float()  # [N_all, F]
            corpus_freq = active_mask.mean(dim=0)  # [F]
            self._corpus_freq[layer_idx] = corpus_freq
            
            # IDF weights: log(1/freq + eps) — rare features get high weight
            idf = torch.log(1.0 / (corpus_freq + self.cfg.eps) + self.cfg.eps)
            # Temperature scaling (default=1.0 is no-op)
            if self.cfg.freq_calibration_temperature != 1.0:
                idf = idf / max(self.cfg.freq_calibration_temperature, 1e-6)
            self._idf_weights[layer_idx] = idf
            
            # Non-member feature statistics for z-scoring
            # Use only the last n_non codes (ref_non was appended after ref_mem)
            codes_non = codes_all[-n_non:]  # [N_non, F]
            mu_non = codes_non.mean(dim=0)    # [F]
            sigma_non = codes_non.std(dim=0) + 1e-8  # [F]
            self._feature_stats[layer_idx] = {
                "mu": mu_non,
                "sigma": sigma_non,
            }

    @torch.no_grad()
    def _fit_recall_prefixes(self, ref_non: Sequence[str]) -> None:
        """Sample and tokenize non-member prefix texts for SAE-ReCaLL.
        
        We store pre-tokenized prefix sequences so scoring is fast.
        Each prefix is a random non-member text truncated to recall_prefix_tokens.
        """
        rng = np.random.default_rng(self.cfg.seed + 42)
        n = min(self.cfg.recall_n_prefixes, len(ref_non))
        if n <= 0:
            return
        chosen_idx = rng.choice(len(ref_non), size=n, replace=False)
        
        self._recall_prefix_ids = []
        for idx in chosen_idx:
            enc = self.model.tokenizer(
                ref_non[idx],
                truncation=True,
                max_length=self.cfg.recall_prefix_tokens,
                return_tensors="pt",
            )
            prefix_ids = enc["input_ids"][0]  # [prefix_len]
            self._recall_prefix_ids.append(prefix_ids)

    @torch.no_grad()
    def _fit_score_normalization(self, ref_non: Sequence[str]) -> None:
        """Fit score normalization on reference non-member texts.
        
        Per-layer normalization computes z-score parameters (mu, sigma) for each
        layer's raw scores on non-members, then applies normalization before
        final aggregation. This helps stabilize tail metrics across layers with
        different score scales.
        
        Global normalization fits a single normalizer on the aggregated scores.
        """
        if self.cfg.score_norm == "none":
            self._score_normalizer = None
            self._layer_score_stats = {}
            return
        
        if not ref_non:
            return
        
        # Compute raw per-layer scores on non-members
        self._layer_score_stats = {}
        
        if self.cfg.score_norm_scope == "feature":
            # Per-layer normalization
            for layer_idx in self.selected_layers or self.layers:
                raw_scores, _recon = self._score_texts_one_layer_one_sae(
                    ref_non, layer_idx, 
                    self.layer_to_saes[layer_idx][0],
                    w=self.w_sweep[layer_idx][0],
                    mask=self.feature_mask_sweep[layer_idx][0],
                )
                self._layer_score_stats[layer_idx] = {
                    "mu": float(np.mean(raw_scores)),
                    "sigma": float(np.std(raw_scores) + 1e-8),
                }
        else:
            # Global normalization: fit on aggregated scores
            agg_scores = self.score_texts(ref_non)
            from sae_mia_audit.eval.calibration import ZScoreNormalizer, QuantileNormalizer
            
            if self.cfg.score_norm == "zscore":
                self._score_normalizer = ZScoreNormalizer.fit(
                    agg_scores, scope="global"
                )
            elif self.cfg.score_norm == "quantile":
                self._score_normalizer = QuantileNormalizer.fit(
                    agg_scores, scope="global"
                )

    @torch.no_grad()
    def _stability_mask_for_top_features(
        self,
        layer_idx: int,
        w: torch.Tensor,
        primary: SAEProtocol,
        others: Sequence[SAEProtocol],
        top_n: int,
    ) -> torch.Tensor:
        """Compute a boolean mask of stable features (only for top-|w| features).
    
        Motivation:
          SAE dictionaries contain a mixture of high-quality and artifact/dead/polysemantic features.
          A practical quality proxy used in SAE literature is *stability* across runs or hyperparameters.
    
        Implementation:
          For a candidate feature in the primary SAE, we compute its decoded feature vector
          (the column of the decoder) and check whether it has a close cosine match in each
          other SAE's decoded feature set.
    
        Backend compatibility:
          - If the SAE exposes a `decoder.weight` matrix, we use it directly (fast).
          - Otherwise we fall back to decoding one-hot feature vectors and subtracting decode(0),
            which works for SAEs with tied biases.
    
        To keep this scalable, we only evaluate stability for the top-N features by |w|. Features
        outside this candidate set are treated as stable (i.e., not filtered).
        """
        F = int(primary.d_sae)
        top_n = min(int(top_n), F)
        if top_n <= 0:
            return torch.ones(F, dtype=torch.bool)
    
        # Candidate set: top-|w|
        idx = torch.topk(w.abs(), k=top_n, largest=True).indices.detach().cpu()
        cand = idx.tolist()
    
        def _decoder_matrix(sae: SAEProtocol, feature_ids: Sequence[int] | None = None) -> torch.Tensor:
            """Return decoder feature vectors as a [D, F_sel] matrix on CPU."""
            # Fast path: direct decoder weights
            dec = getattr(sae, "decoder", None)
            W = getattr(dec, "weight", None) if dec is not None else None
            if isinstance(W, torch.Tensor):
                Wt = W.detach().float()
                # Optional components dimension
                if Wt.dim() == 3:
                    Wt = Wt[0]
                # Expected shape is [D, F]
                if feature_ids is None:
                    return Wt.cpu()
                return Wt[:, torch.tensor(list(feature_ids), dtype=torch.long)].cpu()
    
            # Fallback: decode one-hot basis (bias-cancelled)
            F_full = int(sae.d_sae)
            ids = list(range(F_full)) if feature_ids is None else list(map(int, feature_ids))
            # Choose a device the SAE is on.
            mod = getattr(sae, "saif", sae)
            try:
                dev = next(mod.parameters()).device  # type: ignore[attr-defined]
            except Exception:
                dev = torch.device("cpu")
            out_cols: list[torch.Tensor] = []
            chunk = 512
            z0 = torch.zeros((1, F_full), device=dev)
            base = sae.decode(z0).detach().float()  # [1, D]
            for i0 in range(0, len(ids), chunk):
                part = ids[i0 : i0 + chunk]
                z = torch.zeros((len(part), F_full), device=dev)
                z[torch.arange(len(part)), torch.tensor(part, device=dev)] = 1.0
                x = sae.decode(z).detach().float() - base
                out_cols.append(x.T.cpu())  # [D, len(part)]
            return torch.cat(out_cols, dim=1)
    
        # Normalize candidate vectors in primary
        W0_c = _decoder_matrix(primary, cand)  # [D, top_n]
        W0_c = W0_c / torch.linalg.norm(W0_c, dim=0, keepdim=True).clamp_min(1e-8)
    
        ok_counts = torch.zeros(top_n, dtype=torch.int32)
        for other in others:
            W1 = _decoder_matrix(other, None)  # [D, F_other]
            W1 = W1 / torch.linalg.norm(W1, dim=0, keepdim=True).clamp_min(1e-8)
            sim = (W0_c.T @ W1).numpy()  # [top_n, F_other]
            max_sim = sim.max(axis=1)
            ok = max_sim >= float(self.cfg.stability_cosine)
            ok_counts += torch.tensor(ok.astype(np.int32))
    
        if self.cfg.stability_require_all:
            ok_final = ok_counts.numpy() == len(others)
        else:
            ok_final = ok_counts.numpy() >= 1
    
        stable = torch.ones(F, dtype=torch.bool)
        for j, fid in enumerate(cand):
            stable[int(fid)] = bool(ok_final[j])  # set False for unstable candidates
        return stable

    @torch.no_grad()
    def _select_layers(self, val_mem: Sequence[str], val_non: Sequence[str]) -> None:
        """Select top-K layers based on validation separation."""
        y = np.asarray([1] * len(val_mem) + [0] * len(val_non), dtype=int)
        texts = list(val_mem) + list(val_non)
        rng = np.random.default_rng(self.cfg.seed)
        # Shuffle validation examples to avoid ordering bias
        perm = rng.permutation(len(texts))
        y = y[perm]
        texts = [texts[i] for i in perm.tolist()]

        layer_scores: List[Tuple[int, float]] = []
        self.layer_val = {}

        for layer_idx in self.layers:
            s = self.score_texts(texts, layer_subset=[layer_idx])
            auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else 0.5
            tpr = _tpr_at_fpr(y, s, fpr_target=float(self.cfg.layer_select_fpr))
            # Tail separation: mean of (q95-q99) quantile differences
            mem_scores = s[y == 1]
            non_scores = s[y == 0]
            tail_sep = _tail_separation_metric(mem_scores, non_scores)
            self.layer_val[layer_idx] = {"auc": auc, "tpr@fpr": tpr, "tail_sep": tail_sep}
            if self.cfg.layer_select_metric == "auc":
                metric = auc
            elif self.cfg.layer_select_metric == "tpr@fpr":
                metric = tpr
            else:  # tail_sep
                metric = tail_sep
            layer_scores.append((layer_idx, float(metric)))

        layer_scores.sort(key=lambda x: -x[1])
        k = min(int(self.cfg.top_k_layers), len(layer_scores))
        self.selected_layers = [li for li, _ in layer_scores[:k]]

    # ------------------------- Scoring -------------------------

    @torch.no_grad()
    def score_texts(self, texts: Sequence[str], layer_subset: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return a continuous membership score per text.

        Higher => more likely member.

        Auxiliary enhancements:
          - Reconstruction error blending (recon_error_weight > 0)
          - SAE-ReCaLL prefix perturbation (recall_enabled)
          - Context-dependent z-scoring (context_zscore)
        """
        if not self.w:
            raise RuntimeError("Call fit() before score_texts().")

        layers = list(layer_subset) if layer_subset is not None else list(self.selected_layers or self.layers)
        if not layers:
            raise ValueError("No layers selected for scoring")

        # Core scoring (unconditional, no prefix)
        napdd_scores, recon_scores = self._score_texts_across_layers(
            texts, layers, prefix_ids=None,
        )

        # SAE-ReCaLL — compare conditioned vs unconditioned scores
        # IMPORTANT: ReCaLL must run BEFORE recon blending so both
        # conditioned and unconditioned scores are on the same raw scale
        # (apples-to-apples comparison).
        if self.cfg.recall_enabled and self._recall_prefix_ids:
            recall_ratios = []
            for pfx_ids in self._recall_prefix_ids:
                cond_scores, _ = self._score_texts_across_layers(
                    texts, layers, prefix_ids=pfx_ids,
                )
                if self.cfg.recall_mode == "ratio":
                    # Members: cond/uncond ≈ 1 (robust to prefix)
                    # Non-members: cond/uncond < 1 (prefix hurts)
                    denom = np.abs(napdd_scores) + 1e-8
                    recall_ratios.append(cond_scores / denom)
                else:  # "diff"
                    recall_ratios.append(cond_scores - napdd_scores)

            # Average across prefixes
            recall_avg = np.mean(recall_ratios, axis=0)
            # Higher recall_avg => score is more robust to prefix => member
            napdd_scores = recall_avg

        # Blend reconstruction error signal (applied AFTER recall
        # so that z-scoring normalises whatever scale napdd_scores is
        # currently in — raw NA-PDD or recall ratios/diffs)
        if recon_scores is not None and self.cfg.recon_error_weight > 0:
            alpha = self.cfg.recon_error_weight
            # Lower recon error => more likely member, so negate
            # Normalize both to z-scores for comparable scale
            napdd_z = (napdd_scores - napdd_scores.mean()) / (napdd_scores.std() + 1e-8)
            recon_z = (recon_scores - recon_scores.mean()) / (recon_scores.std() + 1e-8)
            napdd_scores = (1.0 - alpha) * napdd_z + alpha * (-recon_z)

        # Apply global normalization if enabled (after all fusion)
        if self.cfg.score_norm_scope in ("sae", "global") and self._score_normalizer is not None:
            napdd_scores = self._score_normalizer.transform(napdd_scores)

        return napdd_scores

    @torch.no_grad()
    def _score_texts_across_layers(
        self,
        texts: Sequence[str],
        layers: Sequence[int],
        prefix_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Score texts across selected layers, returning (napdd_scores, recon_errors).
        
        This is the inner loop factored out of score_texts to allow SAE-ReCaLL
        to call it twice (with and without prefix).
        """
        scores_all: List[np.ndarray] = []
        recon_all: List[np.ndarray] = []

        for layer_idx in layers:
            saes = self.layer_to_saes[layer_idx]
            if self.cfg.ensemble_across_saes and len(saes) > 1:
                s_layer_acc = None
                r_layer_acc = None
                for s_idx, sae in enumerate(saes):
                    w = self.w_sweep[layer_idx][s_idx]
                    mask = self.feature_mask_sweep[layer_idx][s_idx]
                    s_i, r_i = self._score_texts_one_layer_one_sae(
                        texts, layer_idx, sae, w=w, mask=mask,
                        prefix_ids=prefix_ids,
                    )
                    s_layer_acc = s_i if s_layer_acc is None else (s_layer_acc + s_i)
                    if r_i is not None:
                        r_layer_acc = r_i if r_layer_acc is None else (r_layer_acc + r_i)
                layer_scores = (s_layer_acc / len(saes)).astype(np.float64)
                layer_recon = (r_layer_acc / len(saes)).astype(np.float64) if r_layer_acc is not None else None
            else:
                w0 = self.w_sweep[layer_idx][0]
                m0 = self.feature_mask_sweep[layer_idx][0]
                layer_scores, layer_recon = self._score_texts_one_layer_one_sae(
                    texts, layer_idx, saes[0], w=w0, mask=m0,
                    prefix_ids=prefix_ids,
                )
                layer_scores = layer_scores.astype(np.float64)
            
            # Apply per-layer normalization if enabled
            if self.cfg.score_norm_scope == "feature" and layer_idx in self._layer_score_stats:
                stats = self._layer_score_stats[layer_idx]
                layer_scores = (layer_scores - stats["mu"]) / stats["sigma"]
            
            scores_all.append(layer_scores)
            if layer_recon is not None:
                recon_all.append(layer_recon)

        # Aggregate across layers by mean
        S = np.stack(scores_all, axis=0)  # [L, N]
        final_scores = S.mean(axis=0)
        
        # Aggregate reconstruction error across layers
        final_recon = None
        if recon_all:
            R = np.stack(recon_all, axis=0)  # [L, N]
            if self.cfg.recon_error_pool == "layer_min":
                final_recon = R.min(axis=0)
            else:  # "layer_mean"
                final_recon = R.mean(axis=0)

        return final_scores, final_recon

    @torch.no_grad()
    def _score_texts_one_layer_one_sae(
        self,
        texts: Sequence[str],
        layer_idx: int,
        sae: SAEProtocol,
        *,
        w: torch.Tensor,
        mask: torch.Tensor,
        prefix_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Score texts for one layer+SAE with all enhancements.
        
        Returns:
            Tuple of (napdd_scores [N], recon_errors [N] or None).
            recon_errors is only computed when cfg.recon_error_weight > 0.
        
        Scoring pipeline:
        1. Extract per-token SAE codes z [B, T, F]
        2. Compute reconstruction error ||h - decode(z)||^2 per sample (if enabled)
        3. Apply feature z-scoring or context-dependent z-scoring (if enabled)
        4. Compute combined weights: w_eff = w * idf (if freq_calibration enabled)
        5. Compute per-token evidence: evidence_t = z_norm * w_eff
        6. (Optional) Multi-scale temporal pooling
        7. (Optional) Per-sample top-K% feature selection
        8. Final aggregation → per-sample score
        
        Args:
            prefix_ids: Optional [P] token IDs to prepend before each text
                        (for SAE-ReCaLL conditional scoring).
        """
        # Prepare weights on CPU for masking logic, then move to device when needed
        w_cpu = w.detach().float().cpu()
        mask_cpu = mask.detach().cpu().bool()

        w2 = w_cpu.clone()
        w2[~mask_cpu] = 0.0

        # Apply frequency calibration (IDF weighting)
        if self.cfg.freq_calibration and layer_idx in self._idf_weights:
            idf = self._idf_weights[layer_idx].cpu().float()
            w2 = w2 * idf  # TF-IDF-style: discriminative weight * rarity weight

        pos_cpu = torch.clamp(w2, min=0.0)
        neg_cpu = torch.clamp(-w2, min=0.0)

        # Feature z-scoring parameters (global, per-feature)
        feat_mu = None
        feat_sigma = None
        use_zscore = self.cfg.feature_zscore and layer_idx in self._feature_stats
        # Context-dependent z-scoring overrides global z-scoring
        use_context_zscore = self.cfg.context_zscore
        if use_context_zscore:
            use_zscore = True  # scoring path should use direct dot product
        if use_zscore and not use_context_zscore:
            feat_mu = self._feature_stats[layer_idx]["mu"].cpu().float()
            feat_sigma = self._feature_stats[layer_idx]["sigma"].cpu().float()
        
        # Whether to compute reconstruction error
        compute_recon = self.cfg.recon_error_weight > 0

        # Explicit random_crop=False for deterministic evaluation
        tok_cfg = TokenizeConfig(seq_len=self.cfg.seq_len, random_crop=False)
        out_scores: List[torch.Tensor] = []
        out_recon: List[torch.Tensor] = []

        for i in tqdm(
            range(0, len(texts), self.cfg.batch_size),
            desc=f"score_l{layer_idx}",
            dynamic_ncols=True,
        ):
            chunk = list(texts[i : i + self.cfg.batch_size])
            batch = tokenize_batch(self.model.tokenizer, chunk, tok_cfg)
            input_ids = batch["input_ids"].to(self.model.model.device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(self.model.model.device)

            # SAE-ReCaLL: prepend prefix tokens if provided
            if prefix_ids is not None:
                pfx = prefix_ids.to(input_ids.device)
                P = pfx.shape[0]
                # Broadcast prefix to batch: [B, P+T]
                pfx_batch = pfx.unsqueeze(0).expand(input_ids.shape[0], -1)
                input_ids = torch.cat([pfx_batch, input_ids], dim=1)
                if attn is not None:
                    pfx_attn = torch.ones(
                        (attn.shape[0], P), device=attn.device, dtype=attn.dtype
                    )
                    attn = torch.cat([pfx_attn, attn], dim=1)

            outputs = self.model.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            h = outputs.hidden_states[layer_idx + 1]  # [B, T_total, D]
            
            # SAE-ReCaLL: strip prefix positions from hidden states
            if prefix_ids is not None:
                h = h[:, P:, :]  # [B, T, D] — only the original text positions
                if attn is not None:
                    attn = attn[:, P:]  # strip prefix from attention mask too

            B, T, D = h.shape
            x = h.reshape(B * T, D)

            # SAE device/dtype
            if next(sae.parameters()).device != x.device:
                sae.to(x.device)
                sae.eval()
            x = x.to(dtype=next(sae.parameters()).dtype)

            z_flat = sae.encode(x)  # [B*T, F]

            # Reconstruction error = ||h - decode(encode(h))||^2 per token
            if compute_recon:
                x_hat = sae.decode(z_flat)  # [B*T, D]
                recon_per_token = ((x.float() - x_hat.float()) ** 2).mean(dim=-1)  # [B*T]
                recon_per_token = recon_per_token.reshape(B, T)
                # Average over valid tokens only
                if attn is not None:
                    attn_f = attn.to(dtype=recon_per_token.dtype)
                    recon_per_sample = (recon_per_token * attn_f).sum(dim=1) / attn_f.sum(dim=1).clamp_min(1.0)
                else:
                    recon_per_sample = recon_per_token.mean(dim=1)  # [B]
                out_recon.append(recon_per_sample.detach().float().cpu())

            z = z_flat.reshape(B, T, -1).float()  # [B, T, F]

            # Context-dependent z-scoring (Feature-K%++)
            # Normalize each feature at each position by the local distribution
            # of ALL features at that position: z_norm = (z - mu_t) / sigma_t
            if use_context_zscore:
                mu_t = z.mean(dim=2, keepdim=True)     # [B, T, 1]
                sigma_t = z.std(dim=2, keepdim=True) + 1e-8  # [B, T, 1]
                z = (z - mu_t) / sigma_t
            # Global per-feature z-scoring (normalize by non-member stats)
            elif feat_mu is not None:
                mu = feat_mu.to(device=z.device, dtype=z.dtype)
                sigma = feat_sigma.to(device=z.device, dtype=z.dtype)
                z = (z - mu[None, None, :]) / sigma[None, None, :]

            # Move weights to device/dtype
            pos = pos_cpu.to(device=z.device, dtype=z.dtype)
            neg = neg_cpu.to(device=z.device, dtype=z.dtype)
            w_masked = w2.to(device=z.device, dtype=z.dtype)

            if self.cfg.pooling == "sequence":
                z_agg = _aggregate_codes(z, self.cfg.agg, self.cfg.agg_k, self.cfg.agg_trim_frac, attention_mask=attn)  # [B, F]
                
                if self.cfg.scoring_mode == "binary":
                    tau = self.tau.get(layer_idx, None)
                    active = self._active_mask(z_agg.cpu(), tau=tau.cpu() if tau is not None else None).to(z.device, dtype=z.dtype)
                    s = (active * w_masked[None, :]).sum(dim=1)
                else:
                    # Per-sample top-K% feature selection
                    if self.cfg.sample_topk_features > 0:
                        s = self._score_with_sample_topk(z_agg, pos, neg, w_masked, use_zscore=use_zscore)
                    elif use_zscore:
                        # Z-scored codes can be negative, so the pos/neg log-ratio
                        # decomposition is invalid.  Use a direct dot product:
                        #   score = sum(z_norm_i * w_i)
                        # Positive z-norm * positive w = evidence for membership.
                        # Negative z-norm * positive w = evidence against.
                        s = (z_agg * w_masked[None, :]).sum(dim=1)
                    else:
                        # Standard magnitude scoring (V1 path, z >= 0)
                        mem = (z_agg * pos[None, :]).sum(dim=1)
                        non = (z_agg * neg[None, :]).sum(dim=1)
                        if self.cfg.score_transform == "ratio":
                            s = (mem + self.cfg.delta) / (non + self.cfg.delta)
                        else:
                            eps_log = 1e-10
                            s = (torch.log(torch.clamp(mem + self.cfg.delta, min=eps_log))
                                 - torch.log(torch.clamp(non + self.cfg.delta, min=eps_log)))
                out_scores.append(s.detach().float().cpu())
                continue

            # ---- Multi-scale or token_topk pooling ----
            if self.cfg.multi_scale_enabled and len(self.cfg.multi_scale_windows) > 1:
                s = self._score_multi_scale(z, pos, neg, w_masked, attn, B, T, use_zscore=use_zscore)
            else:
                s = self._score_token_topk(z, pos, neg, w_masked, attn, B, T, use_zscore=use_zscore)
            
            out_scores.append(s.detach().float().cpu())

        napdd_scores = torch.cat(out_scores, dim=0).numpy().astype(np.float64)
        recon_errors = None
        if out_recon:
            recon_errors = torch.cat(out_recon, dim=0).numpy().astype(np.float64)
        return napdd_scores, recon_errors

    @torch.no_grad()
    def _score_with_sample_topk(
        self,
        z_agg: torch.Tensor,  # [B, F]
        pos: torch.Tensor,    # [F]
        neg: torch.Tensor,    # [F]
        w_masked: torch.Tensor,  # [F]
        use_zscore: bool = False,
    ) -> torch.Tensor:
        """Per-sample top-K% feature selection (Min-K% adapted to feature space).
        
        For each sample, select only the top-K% features by absolute contribution
        |z_i * w_i|, then compute the membership score using only those features.
        This prevents noise from low-evidence features diluting the score.
        
        Novel: Min-K% (Shi et al. 2023) selects surprising tokens; we select
        surprising *features* per sample.
        """
        B, F = z_agg.shape
        # Compute per-feature evidence magnitude
        evidence = z_agg * w_masked[None, :]  # [B, F]
        abs_evidence = evidence.abs()
        
        # Select top-K% features per sample
        k = max(1, int(F * self.cfg.sample_topk_features))
        k = min(k, F)
        
        _, topk_idx = torch.topk(abs_evidence, k=k, dim=1, largest=True)  # [B, k]
        
        # Create a mask for selected features
        topk_mask = torch.zeros((B, F), device=z_agg.device, dtype=torch.bool)
        topk_mask.scatter_(1, topk_idx, True)
        
        # Score using only selected features
        z_selected = z_agg * topk_mask.float()

        if use_zscore:
            # Z-scored: direct dot product (z can be negative)
            return (z_selected * w_masked[None, :]).sum(dim=1)
        else:
            # V1 path: pos/neg log-ratio (z >= 0)
            mem = (z_selected * pos[None, :]).sum(dim=1)
            non = (z_selected * neg[None, :]).sum(dim=1)
            if self.cfg.score_transform == "ratio":
                return (mem + self.cfg.delta) / (non + self.cfg.delta)
            else:
                eps_log = 1e-10
                return (torch.log(torch.clamp(mem + self.cfg.delta, min=eps_log))
                        - torch.log(torch.clamp(non + self.cfg.delta, min=eps_log)))

    @torch.no_grad()
    def _score_multi_scale(
        self,
        z: torch.Tensor,      # [B, T, F]
        pos: torch.Tensor,     # [F]
        neg: torch.Tensor,     # [F]
        w_masked: torch.Tensor,  # [F]
        attn: Optional[torch.Tensor],  # [B, T]
        B: int,
        T: int,
        use_zscore: bool = False,
    ) -> torch.Tensor:
        """Multi-scale temporal evidence fusion.
        
        Computes per-token evidence, then pools at multiple temporal resolutions
        (window sizes), and combines the scores. Different membership signals
        manifest at different scales:
        - Token-level (window=1): individual rare token memorisation
        - Window-level (window=8,16): phrase/sentence memorisation
        - Sequence-level (window=T): global distributional shift
        
        Novel: No prior MIA work combines multi-resolution SAE feature evidence.
        """
        # Per-token evidence scores
        if use_zscore:
            # Z-scored: direct dot product per token
            score_t = (z * w_masked[None, None, :]).sum(dim=2)  # [B, T]
        else:
            mem_t = (z * pos[None, None, :]).sum(dim=2)  # [B, T]
            non_t = (z * neg[None, None, :]).sum(dim=2)  # [B, T]
            eps_log = 1e-10
            score_t = (torch.log(torch.clamp(mem_t + self.cfg.delta, min=eps_log))
                       - torch.log(torch.clamp(non_t + self.cfg.delta, min=eps_log)))
        
        if attn is None:
            attn_b = torch.ones((B, T), device=score_t.device, dtype=torch.bool)
            attn_f = torch.ones((B, T), device=score_t.device, dtype=score_t.dtype)
        else:
            attn_b = attn.to(dtype=torch.bool)
            attn_f = attn.to(dtype=score_t.dtype)
        
        score_t = score_t.masked_fill(~attn_b, float("-inf"))
        lengths = attn_b.sum(dim=1).clamp_min(1)
        
        scale_scores = []
        
        for wlen in self.cfg.multi_scale_windows:
            wlen = max(1, int(wlen))
            
            if wlen == 1:
                # Token-level: top-K% pooling of raw per-token scores
                k_each = torch.ceil(lengths.float() * float(self.cfg.token_topk_percent)).to(torch.int64)
                k_each = torch.clamp(k_each, min=1)
                k_each = torch.minimum(k_each, lengths)
                k_max = int(k_each.max().item())
                vals, _ = torch.topk(score_t, k=k_max, dim=1, largest=True)
                keep = (torch.arange(k_max, device=vals.device)[None, :] < k_each[:, None])
                pooled = (vals.masked_fill(~keep, 0.0).sum(dim=1) / k_each.to(vals.dtype))
                scale_scores.append(pooled)
            elif wlen >= T:
                # Sequence-level: mean of all valid token scores
                valid_scores = torch.where(attn_b, score_t, torch.zeros_like(score_t))
                pooled = valid_scores.sum(dim=1) / lengths.float()
                scale_scores.append(pooled)
            else:
                # Window-level: sliding window mean, then top-K% of windows
                kernel = torch.ones((1, 1, wlen), device=score_t.device, dtype=score_t.dtype)
                pad_left = (wlen - 1) // 2
                pad_right = wlen - 1 - pad_left
                padded_scores = torchF.pad(
                    torch.where(attn_b, score_t, torch.zeros_like(score_t)) * attn_f,
                    (pad_left, pad_right), mode='constant', value=0.0
                )
                padded_mask = torchF.pad(attn_f, (pad_left, pad_right), mode='constant', value=0.0)
                num = torchF.conv1d(padded_scores.unsqueeze(1), kernel, padding=0).squeeze(1)
                den = torchF.conv1d(padded_mask.unsqueeze(1), kernel, padding=0).squeeze(1).clamp_min(1.0)
                smooth = num / den
                smooth = smooth.masked_fill(~attn_b, float("-inf"))
                
                # Top-K% of window scores
                k_each = torch.ceil(lengths.float() * float(self.cfg.token_topk_percent)).to(torch.int64)
                k_each = torch.clamp(k_each, min=1)
                k_each = torch.minimum(k_each, lengths)
                k_max = int(k_each.max().item())
                vals, _ = torch.topk(smooth, k=k_max, dim=1, largest=True)
                keep = (torch.arange(k_max, device=vals.device)[None, :] < k_each[:, None])
                pooled = (vals.masked_fill(~keep, 0.0).sum(dim=1) / k_each.to(vals.dtype))
                scale_scores.append(pooled)
        
        # Combine scales with equal weighting (could be learned from val)
        if len(scale_scores) == 1:
            return scale_scores[0]
        stacked = torch.stack(scale_scores, dim=1)  # [B, n_scales]
        return stacked.mean(dim=1)  # [B]

    @torch.no_grad()
    def _score_token_topk(
        self,
        z: torch.Tensor,      # [B, T, F]
        pos: torch.Tensor,     # [F]
        neg: torch.Tensor,     # [F]
        w_masked: torch.Tensor,  # [F]
        attn: Optional[torch.Tensor],  # [B, T]
        B: int,
        T: int,
        use_zscore: bool = False,
    ) -> torch.Tensor:
        """Original token_topk pooling with sequential smoothing."""
        if use_zscore:
            # Z-scored: direct dot product per token
            score_t = (z * w_masked[None, None, :]).sum(dim=2)  # [B, T]
        else:
            mem_t = (z * pos[None, None, :]).sum(dim=2)  # [B, T]
            non_t = (z * neg[None, None, :]).sum(dim=2)  # [B, T]
            eps_log = 1e-10
            if self.cfg.score_transform == "ratio":
                score_t = (mem_t + self.cfg.delta) / (non_t + self.cfg.delta)
                score_t = torch.log(torch.clamp(score_t, min=eps_log))
            else:
                score_t = (torch.log(torch.clamp(mem_t + self.cfg.delta, min=eps_log))
                           - torch.log(torch.clamp(non_t + self.cfg.delta, min=eps_log)))

        if attn is None:
            attn_f = torch.ones((B, T), device=score_t.device, dtype=score_t.dtype)
            attn_b = torch.ones((B, T), device=score_t.device, dtype=torch.bool)
        else:
            attn_b = attn.to(dtype=torch.bool)
            attn_f = attn.to(dtype=score_t.dtype)

        score_t = score_t.masked_fill(~attn_b, float("-inf"))

        wlen = int(self.cfg.token_smooth_window)
        wlen = max(1, wlen)
        if wlen > 1:
            kernel = torch.ones((1, 1, wlen), device=score_t.device, dtype=score_t.dtype)
            pad_left = (wlen - 1) // 2
            pad_right = wlen - 1 - pad_left
            padded_scores = torchF.pad(
                torch.where(attn_b, score_t, torch.zeros_like(score_t)) * attn_f,
                (pad_left, pad_right), mode='constant', value=0.0
            )
            padded_mask = torchF.pad(attn_f, (pad_left, pad_right), mode='constant', value=0.0)
            num = torchF.conv1d(padded_scores.unsqueeze(1), kernel, padding=0).squeeze(1)
            den = torchF.conv1d(padded_mask.unsqueeze(1), kernel, padding=0).squeeze(1).clamp_min(1.0)
            smooth = num / den
            smooth = smooth.masked_fill(~attn_b, float("-inf"))
        else:
            smooth = score_t

        lengths = attn_b.sum(dim=1).clamp_min(1)
        k_each = torch.ceil(lengths.to(torch.float32) * float(self.cfg.token_topk_percent)).to(torch.int64)
        k_each = torch.clamp(k_each, min=1)
        k_each = torch.minimum(k_each, lengths)

        k_max = int(k_each.max().item())
        vals, _ = torch.topk(smooth, k=k_max, dim=1, largest=True)

        keep = (torch.arange(k_max, device=vals.device)[None, :] < k_each[:, None])
        pooled = (vals.masked_fill(~keep, 0.0).sum(dim=1) / k_each.to(vals.dtype))
        
        return pooled

    # ------------------------- Attribution helpers -------------------------

    @torch.no_grad()
    def top_weighted_features(self, layer_idx: int, top_k: int = 64) -> List[Tuple[int, float]]:
        """Return top-|w| features (after quality masking) for a layer."""
        w = self.w[layer_idx].detach().float().cpu().clone()
        w[~self.feature_mask[layer_idx].cpu()] = 0.0
        idx = torch.topk(w.abs(), k=min(int(top_k), w.numel())).indices
        return [(int(i), float(w[int(i)].item())) for i in idx.tolist()]

    @torch.no_grad()
    def feature_contributions(
        self,
        text: str,
        layer_idx: int,
        sae: Optional[SAEProtocol] = None,
        top_k: int = 32,
    ) -> List[Tuple[int, float, float, float]]:
        """Return top contributing features for a given text.

        Returns list of tuples (feature_id, contribution, activation, weight).
        """
        sae = sae or self.layer_to_saes[layer_idx][0]
        z = self._extract_layer_codes([text], layer_idx=layer_idx, sae=sae, return_token_codes=False)[0]  # [F]
        w = self.w[layer_idx].detach().float().cpu().clone()
        mask = self.feature_mask[layer_idx].cpu()
        w[~mask] = 0.0
        contrib = z * w
        idx = torch.topk(contrib.abs(), k=min(int(top_k), contrib.numel())).indices
        out = []
        for i in idx.tolist():
            i = int(i)
            out.append((i, float(contrib[i].item()), float(z[i].item()), float(w[i].item())))
        # sort by absolute contribution
        out.sort(key=lambda x: -abs(x[1]))
        return out

    # ------------------------- Group C: explain() method -------------------------

    @torch.no_grad()
    def explain(
        self,
        text: str,
        top_k_features: int = 32,
        top_k_positions: int = 5,
        layers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Full attribution explanation for a single text.

        This is the main entry point for Group C attribution. Returns per-layer
        top contributing features along with their token positions and contexts.

        Args:
            text: Input text to explain.
            top_k_features: Number of top features per layer.
            top_k_positions: Number of top token positions per feature.
            layers: Layers to explain (default: selected_layers from fit()).

        Returns:
            Dictionary with:
              - 'text': input text
              - 'score': SFA score
              - 'layers': dict mapping layer_idx to layer explanation
              - 'summary': aggregated statistics

        Example layer explanation:
            {
                'features': [
                    {
                        'feature_id': 1234,
                        'contribution': 0.45,
                        'activation': 2.3,
                        'weight': 0.19,
                        'sign': 'positive',  # member-leaning
                        'positions': [10, 25, 42],
                        'contexts': ['...token context...', ...]
                    },
                    ...
                ],
                'total_positive_contrib': 1.2,
                'total_negative_contrib': -0.3,
                'n_features_for_50pct': 5,  # concentration metric
            }
        """
        if layers is None:
            layers = self.selected_layers

        # Get overall score
        scores = self.score_texts([text])
        score = float(scores[0])

        result = {
            'text': text[:2000],  # Truncate for storage
            'score': score,
            'layers': {},
            'summary': {
                'n_layers': len(layers),
                'total_features_explained': 0,
                'concentration': {},
            },
        }

        # Explicit random_crop=False for deterministic evaluation
        tok_cfg = TokenizeConfig(seq_len=self.cfg.seq_len, random_crop=False)
        batch = tokenize_batch(self.model.tokenizer, [text], tok_cfg)
        input_ids = batch["input_ids"].to(self.model.model.device)
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.model.model.device)

        # Get hidden states once
        outputs = self.model.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        hs = outputs.hidden_states

        all_contributions = []

        for layer_idx in layers:
            if layer_idx not in self.layer_to_saes:
                continue

            sae = self.layer_to_saes[layer_idx][0]

            # Get per-token activations
            h = hs[layer_idx + 1]  # [1, T, D]
            B, T, D = h.shape
            x = h.reshape(B * T, D).to(self.device)

            if next(sae.parameters()).device != x.device:
                sae.to(x.device)
                sae.eval()
            x = x.to(dtype=next(sae.parameters()).dtype)

            z_tok = sae.encode(x).reshape(B, T, -1)  # [1, T, F]
            z_agg = _aggregate_codes(z_tok, self.cfg.agg, self.cfg.agg_k, self.cfg.agg_trim_frac, attention_mask=attn).squeeze(0)  # [F]

            w = self.w[layer_idx].detach().float().cpu().clone()
            mask = self.feature_mask[layer_idx].cpu()
            w[~mask] = 0.0

            contrib = (z_agg.cpu() * w).numpy()

            # Sort by absolute contribution
            sorted_idx = np.argsort(-np.abs(contrib))[:top_k_features]

            layer_features = []
            total_pos = 0.0
            total_neg = 0.0

            for fid in sorted_idx:
                fid = int(fid)
                c = float(contrib[fid])
                z_val = float(z_agg[fid].cpu().item())
                w_val = float(w[fid].item())

                if abs(c) < 1e-9:
                    continue

                all_contributions.append(abs(c))

                if c > 0:
                    total_pos += c
                else:
                    total_neg += c

                # Get top positions for this feature
                z_feat = z_tok[0, :, fid].detach().float().cpu()  # [T]
                k_pos = min(top_k_positions, T)
                top_pos = torch.topk(z_feat, k=k_pos, largest=True).indices.tolist()
                top_pos = [p for p in top_pos if z_feat[p] > 0]  # Only active positions

                # Get context strings
                contexts = []
                for pos in top_pos[:3]:  # Limit contexts
                    lo = max(0, pos - 15)
                    hi = min(int(input_ids.shape[1]), pos + 16)
                    ctx = self.model.tokenizer.decode(input_ids[0, lo:hi].tolist(), skip_special_tokens=True)
                    contexts.append(ctx)

                layer_features.append({
                    'feature_id': fid,
                    'contribution': c,
                    'activation': z_val,
                    'weight': w_val,
                    'sign': 'positive' if c > 0 else 'negative',
                    'positions': top_pos,
                    'contexts': contexts,
                })

            # Compute concentration: how many features explain 50% of total |contrib|?
            abs_contribs = np.array([abs(f['contribution']) for f in layer_features])
            total_abs = abs_contribs.sum() if len(abs_contribs) > 0 else 1.0
            if total_abs > 0:
                cumsum = np.cumsum(abs_contribs) / total_abs
                n_for_50 = int(np.searchsorted(cumsum, 0.5) + 1)
            else:
                n_for_50 = 0

            result['layers'][layer_idx] = {
                'features': layer_features,
                'total_positive_contrib': total_pos,
                'total_negative_contrib': total_neg,
                'n_features_for_50pct': n_for_50,
            }

        # Summary statistics
        result['summary']['total_features_explained'] = len(all_contributions)
        if all_contributions:
            all_contributions = np.array(all_contributions)
            total = all_contributions.sum()
            if total > 0:
                cumsum = np.cumsum(all_contributions) / total
                result['summary']['concentration'] = {
                    'n_for_50pct': int(np.searchsorted(cumsum, 0.5) + 1),
                    'n_for_90pct': int(np.searchsorted(cumsum, 0.9) + 1),
                    'entropy': float(-np.sum((all_contributions / total) * np.log(all_contributions / total + 1e-10))),
                }

        return result
