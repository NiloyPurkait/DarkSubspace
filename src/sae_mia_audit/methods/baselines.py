# ---------------------------------------------------------------------------
# Source repositories for baseline methods:
#   LOSS: Yeom et al. (IEEE CSF 2018). No public repo (method is trivial).
#   Zlib/Lowercase/Reference: Carlini et al. (USENIX Security 2021).
#       https://github.com/ftramer/LM_Memorization
#   Neighbor: Mattern et al. (ACL 2023 Findings).
#       No standalone repo found (verified: justusmattern GitHub has none).
#       Re-implemented by Meeus et al. in mia_llms_benchmark (see below).
#   BoW/TF-IDF: Meeus et al. (SaTML 2025).
#       https://github.com/imperial-aisp/mia_llms_benchmark
#   ReCaLL: Xie et al. (EMNLP 2024).
#       https://github.com/ruoyuxie/recall
#   Con-ReCall: Wang et al. (COLING 2025).
#       No standalone author repo found as of 2026-02-20.
#       Re-implemented by Meeus et al. in mia_llms_benchmark.
#   General reference: MIMIR benchmark (Duan et al., COLM 2024).
#       https://github.com/iamgroot42/mimir
# ---------------------------------------------------------------------------
"""
Baseline MIA methods for reviewer-proof benchmarking.

These implementations follow:
- LOSS: https://ieeexplore.ieee.org/abstract/document/8429311
- Zlib: Carlini et al. "Extracting Training Data from Large Language Models" (Sec 21)
        https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
- Reference-based: https://arxiv.org/abs/2004.15011
- Lowercase: WikiMIA paper baseline
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class LossConfig:
    """Config for loss-based MIA (perplexity baseline)."""
    pass  # No config needed


@dataclass(frozen=True)
class ZlibConfig:
    """Config for zlib-normalized MIA."""
    pass  # No config needed


@dataclass(frozen=True)
class LowercaseConfig:
    """Config for lowercase-normalized MIA."""
    pass  # No config needed


@dataclass(frozen=True)
class BlindConfig:
    """Config for blind baselines (no model access)."""
    seed: int = 42


# ---------------------------------------------------------------------------
# LOSS baseline (perplexity)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cfg: LossConfig,
) -> torch.Tensor:
    """LOSS-based MIA score (negative log-likelihood per token).
    
    Higher score => more likely member (lower loss = better fit = member).
    
    This is the most basic baseline that all MIA papers include.
    If your method doesn't beat this, it's not publishable.
    
    References:
        - Yeom et al., "Privacy Risk in Machine Learning" (IEEE S&P 2018)
        - Carlini et al., "Extracting Training Data" (USENIX Sec 2021)
    """
    from sae_mia_audit.models.logprobs import next_token_logprobs_and_stats
    
    lp = next_token_logprobs_and_stats(logits, input_ids)
    token_logp = lp.token_logp  # [B, T-1]
    
    B, Tm1 = token_logp.shape
    device = token_logp.device
    
    if attention_mask is not None:
        # Align mask to token_logp (predicting tokens 1..T-1)
        valid = attention_mask[:, 1:].to(device=device, dtype=torch.float32)
    else:
        valid = torch.ones((B, Tm1), device=device, dtype=torch.float32)
    
    # Mean log-prob over valid tokens (higher = lower loss = more likely member)
    sum_logp = (token_logp * valid).sum(dim=1)
    n_tokens = valid.sum(dim=1).clamp(min=1)
    mean_logp = sum_logp / n_tokens
    
    return mean_logp  # Higher => more likely member


# ---------------------------------------------------------------------------
# Zlib baseline (compression-normalized perplexity)
# ---------------------------------------------------------------------------

def _zlib_entropy(text: str) -> int:
    """Compute zlib compressed size as entropy proxy."""
    return len(zlib.compress(text.encode('utf-8')))


@torch.no_grad()
def score_zlib(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    texts: List[str],
    cfg: ZlibConfig,
) -> torch.Tensor:
    """Zlib-normalized MIA score.
    
    Score = log_prob / zlib_entropy(text)
    
    The intuition: high zlib entropy means the text is "hard" in general,
    so we should normalize the model's perplexity by this difficulty.
    
    Higher score => more likely member (better fit relative to text complexity).
    
    References:
        - Carlini et al., "Extracting Training Data from Large Language Models" (USENIX Sec 2021)
    """
    from sae_mia_audit.models.logprobs import next_token_logprobs_and_stats
    
    lp = next_token_logprobs_and_stats(logits, input_ids)
    token_logp = lp.token_logp  # [B, T-1]
    
    B, Tm1 = token_logp.shape
    device = token_logp.device
    
    if attention_mask is not None:
        valid = attention_mask[:, 1:].to(device=device, dtype=torch.float32)
    else:
        valid = torch.ones((B, Tm1), device=device, dtype=torch.float32)
    
    sum_logp = (token_logp * valid).sum(dim=1)
    n_tokens = valid.sum(dim=1).clamp(min=1)
    mean_logp = sum_logp / n_tokens
    
    # Normalize by zlib entropy
    zlib_entropies = torch.tensor([_zlib_entropy(t) for t in texts], 
                                   device=device, dtype=torch.float32)
    # Avoid division by zero
    zlib_entropies = zlib_entropies.clamp(min=1.0)
    
    # Score = mean_logp / zlib_entropy (higher = more likely member)
    return mean_logp / zlib_entropies


# ---------------------------------------------------------------------------
# Lowercase baseline (case-normalized perplexity)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_lowercase(
    model,
    texts: List[str],
    seq_len: int,
    cfg: LowercaseConfig,
) -> torch.Tensor:
    """Lowercase-normalized MIA score.
    
    Score = mean_logp(text.lower()) / mean_logp(text)
    
    This matches the official formula from swj0419/detect-pretrain-code
    (run.py line 103) and zjysteven/mink-plus-plus (run_ref.py line 108),
    which both compute ``ll_lower / ll`` (ratio of mean log-likelihoods).
    Since mean log-probs are negative, members (where original text is
    predicted more confidently) produce scores < 1, and non-members produce
    scores ≈ 1.  The score is flipped during orientation calibration so
    that higher => more likely member.
    
    References:
        - Shi et al., "Detecting Pretraining Data from Large Language
          Models" (ICLR 2024). arXiv:2310.16789
        - Official repo: swj0419/detect-pretrain-code run.py line 103
    """
    from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
    from sae_mia_audit.models.logprobs import next_token_logprobs_and_stats
    
    tok_cfg = TokenizeConfig(seq_len=seq_len, random_crop=False)
    device = next(model.model.parameters()).device
    
    # Score original texts
    batch_orig = tokenize_batch(model.tokenizer, texts, tok_cfg)
    input_ids_orig = batch_orig["input_ids"].to(device)
    attn_orig = batch_orig.get("attention_mask", None)
    if attn_orig is not None:
        attn_orig = attn_orig.to(device)
    
    out_orig = model.forward(input_ids=input_ids_orig, attention_mask=attn_orig, output_hidden_states=False)
    lp_orig = next_token_logprobs_and_stats(out_orig.logits, input_ids_orig)
    
    B, Tm1 = lp_orig.token_logp.shape
    if attn_orig is not None:
        valid = attn_orig[:, 1:].float()
    else:
        valid = torch.ones((B, Tm1), device=device, dtype=torch.float32)
    
    sum_logp_orig = (lp_orig.token_logp * valid).sum(dim=1)
    n_tokens = valid.sum(dim=1).clamp(min=1)
    mean_logp_orig = sum_logp_orig / n_tokens
    
    # Score lowercase texts
    texts_lower = [t.lower() for t in texts]
    batch_lower = tokenize_batch(model.tokenizer, texts_lower, tok_cfg)
    input_ids_lower = batch_lower["input_ids"].to(device)
    attn_lower = batch_lower.get("attention_mask", None)
    if attn_lower is not None:
        attn_lower = attn_lower.to(device)
    
    out_lower = model.forward(input_ids=input_ids_lower, attention_mask=attn_lower, output_hidden_states=False)
    lp_lower = next_token_logprobs_and_stats(out_lower.logits, input_ids_lower)
    
    B_l, Tm1_l = lp_lower.token_logp.shape
    if attn_lower is not None:
        valid_l = attn_lower[:, 1:].float()
    else:
        valid_l = torch.ones((B_l, Tm1_l), device=device, dtype=torch.float32)
    
    sum_logp_lower = (lp_lower.token_logp * valid_l).sum(dim=1)
    n_tokens_l = valid_l.sum(dim=1).clamp(min=1)
    mean_logp_lower = sum_logp_lower / n_tokens_l
    
    # Ratio of mean log-probs: mean_logp_lower / mean_logp_orig
    # This matches the official implementation ``ll_lower / ll`` exactly.
    #
    # Both values are negative (log-probs), so the ratio is positive.
    # Members: model predicts original casing well (logp_orig > logp_lower,
    #   both negative), so ratio > 1 (numerator less negative).
    #   Example: logp_lower=-3, logp_orig=-2 => ratio = -3/-2 = 1.5
    # Non-members: both equally uncertain, so ratio ≈ 1.
    #   Example: logp_lower=-5.1, logp_orig=-5 => ratio = -5.1/-5 = 1.02
    #
    # The sign inversion is handled by the score-orientation calibration step
    # in the evaluation pipeline (higher AUROC direction is auto-detected).
    #
    # Guard against division by zero (degenerate texts with logp ≈ 0).
    return mean_logp_lower / mean_logp_orig.clamp(max=-1e-8)


# ---------------------------------------------------------------------------
# Blind baselines (no model access - for sanity checks)
# ---------------------------------------------------------------------------

def score_random(
    texts: List[str],
    cfg: BlindConfig,
) -> np.ndarray:
    """Random score baseline (should give AUC ≈ 0.5).
    
    This is the most important sanity check:
    - If your method's AUC is not significantly above 0.5, there's no signal
    - If your method's AUC is BELOW 0.5, you may have inverted the score direction
    
    Returns:
        Random scores from U(0, 1) with fixed seed for reproducibility.
    """
    rng = np.random.RandomState(cfg.seed)
    return rng.random(len(texts))


def score_length(
    texts: List[str],
    cfg: BlindConfig,
) -> np.ndarray:
    """Length-based score baseline (character count).
    
    If length correlates with membership, your dataset has confounds.
    This is a critical artifact check per reviewer guidelines.
    
    Higher score => longer text. If AUC >> 0.5, length is a confound.
    """
    return np.array([len(t) for t in texts], dtype=np.float32)


def score_word_count(
    texts: List[str],
    cfg: BlindConfig,
) -> np.ndarray:
    """Word count score baseline.
    
    Similar to length but using word count instead of character count.
    """
    return np.array([len(t.split()) for t in texts], dtype=np.float32)


def score_token_count(
    texts: List[str],
    tokenizer,
    cfg: BlindConfig,
) -> np.ndarray:
    """Token count score baseline (using model's tokenizer).
    
    If token count correlates with membership, the tokenizer is leaking information
    about the training data, which could be a legitimate (if surprising) signal.
    """
    from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
    tok_cfg = TokenizeConfig(seq_len=4096, random_crop=False, truncate=False)
    
    counts = []
    for text in texts:
        enc = tokenizer(text, truncation=False, return_tensors="pt")
        counts.append(enc["input_ids"].shape[1])
    
    return np.array(counts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Distribution-shift blind baselines (Meeus et al., SaTML 2025;
# Das et al., 2024 "Blind baselines beat MIAs")
#
# These classifiers detect *distribution shift* between members and
# non-members using only raw text, no model access.  If their AUROC is
# high, the benchmark's membership signal may come from temporal/topical
# artefacts rather than genuine memorization.
#
# Two variants following the literature:
#   1. BoW + Random Forest  (Meeus et al. Table II)
#   2. TF-IDF + Logistic Regression  (standard ML baseline)
#
# Both are trained on the *val* split and produce scores on *test*, so
# the pipeline structure (val → threshold → test) is respected.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoWConfig:
    """Config for bag-of-words blind baseline.

    Follows Meeus et al. (SaTML 2025, §IV-B):
    - Random forest (500 trees, max_depth=2, min_samples_leaf=10)
    - Unigram word counts, min_df=0.05
    """
    min_df: float = 0.05       # Minimum document frequency for vocabulary
    max_features: int = 5000   # Cap vocabulary size
    n_estimators: int = 500    # Random forest trees
    max_depth: int = 2         # Shallow trees (following paper)
    min_samples_leaf: int = 10
    seed: int = 42


@dataclass(frozen=True)
class TfidfConfig:
    """Config for TF-IDF + logistic regression blind baseline."""
    min_df: float = 0.02
    max_features: int = 10000
    ngram_range: tuple = (1, 2)   # Unigrams + bigrams
    C: float = 1.0                # LogReg regularisation
    seed: int = 42


def score_bow(
    train_texts: List[str],
    train_labels: np.ndarray,
    test_texts: List[str],
    cfg: BoWConfig,
) -> np.ndarray:
    """Bag-of-words membership classifier (Meeus et al., SaTML 2025).

    Trains a random forest on word-count features from the *val* split
    and returns predicted P(member) for the *test* split.

    A high AUROC means the benchmark has a distribution shift detectable
    from text surfaces alone — the primary critique of post-hoc MIA
    benchmarks (WikiMIA, ArxivMIA).

    Returns:
        np.ndarray of shape [len(test_texts)] — predicted P(member).
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.ensemble import RandomForestClassifier

    vec = CountVectorizer(
        min_df=cfg.min_df,
        max_features=cfg.max_features,
        stop_words="english",
    )
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, train_labels)

    proba = clf.predict_proba(X_test)
    # Column index for the positive class (label=1)
    pos_idx = list(clf.classes_).index(1)
    return proba[:, pos_idx].astype(np.float32)


def score_tfidf(
    train_texts: List[str],
    train_labels: np.ndarray,
    test_texts: List[str],
    cfg: TfidfConfig,
) -> np.ndarray:
    """TF-IDF + logistic regression membership classifier.

    Stronger than BoW for detecting subtle topical/stylistic shifts.
    Uses unigrams + bigrams with L2-regularised logistic regression.

    Returns:
        np.ndarray of shape [len(test_texts)] — predicted P(member).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression as LR

    vec = TfidfVectorizer(
        min_df=cfg.min_df,
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        sublinear_tf=True,
        stop_words="english",
    )
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    clf = LR(
        C=cfg.C,
        max_iter=1000,
        random_state=cfg.seed,
        solver="saga",
        n_jobs=-1,
    )
    clf.fit(X_train, train_labels)

    proba = clf.predict_proba(X_test)
    pos_idx = list(clf.classes_).index(1)
    return proba[:, pos_idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Reference model baseline (LOSS ratio: target vs reference model)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefConfig:
    """Config for reference-based MIA (loss ratio).
    
    The reference model should be:
    - Similar architecture to the target
    - Trained on different (non-overlapping) data
    - Ideally same tokenizer for fair comparison
    
    Common choices:
    - GPT-2 small (124M) vs GPT-2 large (774M)
    - Pythia-70m vs Pythia-1.4b
    - Any model trained on different data split
    """
    pass  # Reference model passed at call time


@torch.no_grad()
def score_ref(
    target_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cfg: RefConfig,
) -> torch.Tensor:
    """Reference model MIA score (loss ratio).
    
    Score = log_prob_target - log_prob_ref
    
    The intuition: members should have lower loss on target model relative
    to reference model, since target was trained on them.
    
    Higher score => more likely member (target fits better than reference).
    
    References:
        - Carlini et al., "Membership Inference Attacks From First Principles" (IEEE S&P 2022)
        - Mireshghallah et al., "Quantifying Memorization Across Neural Language Models" (ICLR 2022)
    """
    from sae_mia_audit.models.logprobs import next_token_logprobs_and_stats
    
    lp_target = next_token_logprobs_and_stats(target_logits, input_ids)
    lp_ref = next_token_logprobs_and_stats(ref_logits, input_ids)
    
    B, Tm1 = lp_target.token_logp.shape
    device = lp_target.token_logp.device
    
    if attention_mask is not None:
        valid = attention_mask[:, 1:].to(device=device, dtype=torch.float32)
    else:
        valid = torch.ones((B, Tm1), device=device, dtype=torch.float32)
    
    n_tokens = valid.sum(dim=1).clamp(min=1)
    
    mean_logp_target = (lp_target.token_logp * valid).sum(dim=1) / n_tokens
    mean_logp_ref = (lp_ref.token_logp * valid).sum(dim=1) / n_tokens
    
    # Ratio in log space = difference
    return mean_logp_target - mean_logp_ref


# ---------------------------------------------------------------------------
# Neighbor (perturbation-based) baseline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NeighborConfig:
    """Config for neighbor-based MIA (perturbation curvature).
    
    This method perturbs the input text (via mask-and-fill or paraphrasing)
    and measures how much the loss changes. Members should have steeper
    curvature (loss increases more with perturbation).
    
    References:
        - Mattern et al., "Membership Inference Attacks Against Language Models via Neighbourhood Comparison" (ACL 2023)
        - Shi et al., "Detecting Pretraining Data from Large Language Models" (WikiMIA)
    """
    n_neighbors: int = 10  # Number of perturbations to generate
    mask_prob: float = 0.15  # Probability of masking each token
    fill_model: str = "bert-base-uncased"  # Model to use for filling masks
    seed: int = 42


def _generate_neighbors(
    text: str,
    n_neighbors: int,
    mask_prob: float,
    fill_model: str,
    seed: int,
    _cache: dict = {},
) -> List[str]:
    """Generate perturbed neighbors of a text using mask-and-fill."""
    import random
    random.seed(seed)
    
    try:
        from transformers import pipeline
    except ImportError:
        # Fallback: return text with random word deletions
        words = text.split()
        neighbors = []
        for i in range(n_neighbors):
            random.seed(seed + i)
            kept = [w for w in words if random.random() > mask_prob]
            neighbors.append(" ".join(kept) if kept else text)
        return neighbors
    
    # Lazy load fill-mask pipeline
    if "fill_pipe" not in _cache:
        _cache["fill_pipe"] = pipeline("fill-mask", model=fill_model, top_k=1)
    fill_pipe = _cache["fill_pipe"]
    
    neighbors = []
    words = text.split()
    
    for i in range(n_neighbors):
        random.seed(seed + i)
        masked_words = []
        mask_positions = []
        
        for j, word in enumerate(words):
            if random.random() < mask_prob:
                masked_words.append("[MASK]")
                mask_positions.append(j)
            else:
                masked_words.append(word)
        
        if not mask_positions:
            neighbors.append(text)
            continue
        
        # Fill masks one at a time
        masked_text = " ".join(masked_words)
        try:
            for _ in mask_positions:
                if "[MASK]" not in masked_text:
                    break
                result = fill_pipe(masked_text)
                if result and isinstance(result, list):
                    if isinstance(result[0], list):
                        result = result[0]  # Batch format
                    if result and "sequence" in result[0]:
                        masked_text = result[0]["sequence"]
            neighbors.append(masked_text)
        except Exception:
            neighbors.append(text)
    
    return neighbors


@torch.no_grad()
def score_neighbor(
    model,
    texts: List[str],
    seq_len: int,
    cfg: NeighborConfig,
) -> torch.Tensor:
    """Neighbor-based MIA score (curvature estimation).
    
    Score = loss(x) - mean(loss(x_neighbors))
    
    Higher score => original text is much better than neighbors => more likely member.
    
    This is one of the strongest baselines for MIA on language models.
    
    References:
        - Mattern et al., "Membership Inference Attacks Against Language Models via Neighbourhood Comparison" (ACL 2023)
    """
    from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
    from sae_mia_audit.models.logprobs import next_token_logprobs_and_stats
    
    device = next(model.model.parameters()).device
    tok_cfg = TokenizeConfig(seq_len=seq_len, random_crop=False)
    
    # Score original texts
    batch_orig = tokenize_batch(model.tokenizer, texts, tok_cfg)
    input_ids_orig = batch_orig["input_ids"].to(device)
    attn_orig = batch_orig.get("attention_mask", None)
    if attn_orig is not None:
        attn_orig = attn_orig.to(device)
    
    out_orig = model.forward(input_ids=input_ids_orig, attention_mask=attn_orig, output_hidden_states=False)
    lp_orig = next_token_logprobs_and_stats(out_orig.logits, input_ids_orig)
    
    B, Tm1 = lp_orig.token_logp.shape
    if attn_orig is not None:
        valid = attn_orig[:, 1:].float()
    else:
        valid = torch.ones((B, Tm1), device=device, dtype=torch.float32)
    
    mean_logp_orig = (lp_orig.token_logp * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    
    # Generate and score neighbors
    neighbor_logps = []
    for i, text in enumerate(texts):
        neighbors = _generate_neighbors(
            text, cfg.n_neighbors, cfg.mask_prob, cfg.fill_model, cfg.seed + i
        )
        
        batch_nb = tokenize_batch(model.tokenizer, neighbors, tok_cfg)
        input_ids_nb = batch_nb["input_ids"].to(device)
        attn_nb = batch_nb.get("attention_mask", None)
        if attn_nb is not None:
            attn_nb = attn_nb.to(device)
        
        out_nb = model.forward(input_ids=input_ids_nb, attention_mask=attn_nb, output_hidden_states=False)
        lp_nb = next_token_logprobs_and_stats(out_nb.logits, input_ids_nb)
        
        B_nb, Tm1_nb = lp_nb.token_logp.shape
        if attn_nb is not None:
            valid_nb = attn_nb[:, 1:].float()
        else:
            valid_nb = torch.ones((B_nb, Tm1_nb), device=device, dtype=torch.float32)
        
        mean_logp_nb = (lp_nb.token_logp * valid_nb).sum(dim=1) / valid_nb.sum(dim=1).clamp(min=1)
        neighbor_logps.append(mean_logp_nb.mean().item())
    
    neighbor_logps = torch.tensor(neighbor_logps, device=device, dtype=torch.float32)
    
    # Score = original log-prob - mean neighbor log-prob
    # Higher => original much better => more likely member
    return mean_logp_orig - neighbor_logps


# ---------------------------------------------------------------------------
# ReCaLL baseline (nonmember-prefix conditional log-likelihood ratio)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReCaLLConfig:
    """Config for ReCaLL MIA method (Xie et al., 2024).
    
    The original ReCaLL method computes the ratio of the conditional
    log-likelihood of target text given a *nonmember* prefix to its
    unconditional log-likelihood.  For training data, the model has
    memorised the text so the irrelevant prefix barely affects its
    likelihood; for unseen data the prefix degrades predictions.
    
    This implementation matches the official repositories:
        - ruoyuxie/recall  (``pred["recall"] = ll_nonmember / ll``)
        - iamgroot42/mimir (``recall = ll_nonmember / lls``)
    
    The nonmember texts are drawn from the **reference** split (label == 0),
    ensuring no information leakage to val/test.
    
    References:
        - Xie et al., "ReCaLL: Membership Inference via Relative Conditional
          Log-Likelihoods" (2024)
    """
    num_shots: int = 1        # Number of nonmember texts concatenated as prefix
    max_prefix_tokens: int = 256  # Truncate prefix to at most this many tokens
    min_target_tokens: int = 10   # Skip targets shorter than this


@torch.no_grad()
def score_recall_mia(
    model,
    texts: List[str],
    ref_nonmember_texts: List[str],
    seq_len: int,
    cfg: ReCaLLConfig,
) -> torch.Tensor:
    """ReCaLL MIA score (Xie et al., 2024).
    
    Score = mean_logp(text | nonmember_prefix) / mean_logp(text)
    
    For training data the model's prediction is robust to the irrelevant
    prefix, yielding a ratio close to 1.  For unseen data the prefix
    degrades the likelihood, producing a ratio < 1.
    
    Higher score => more likely member.
    
    Implementation follows the official ``ruoyuxie/recall`` repository
    and the ``iamgroot42/mimir`` benchmark suite.
    
    Args:
        model: CausalLMWrapper with ``.model``, ``.tokenizer``, ``.forward()``.
        texts: Target texts to score.
        ref_nonmember_texts: Known nonmember texts for the prefix (from reference split).
        seq_len: Maximum sequence length for the target text.
        cfg: ReCaLLConfig.
    
    Returns:
        Tensor of shape ``[len(texts)]`` with ReCaLL scores.
    """
    device = next(model.model.parameters()).device
    
    # --- Build prefix token IDs from nonmember reference texts ---
    prefix_parts: List[str] = ref_nonmember_texts[: cfg.num_shots]
    if not prefix_parts:
        raise ValueError(
            "ReCaLL requires at least 1 nonmember reference text for the prefix, "
            "but ref_nonmember_texts is empty."
        )
    prefix_str = " ".join(prefix_parts)
    prefix_enc = model.tokenizer(
        prefix_str, truncation=True, max_length=cfg.max_prefix_tokens,
        return_tensors="pt",
    )
    prefix_ids = prefix_enc["input_ids"][0]  # [P]
    
    scores = []
    for text in texts:
        # --- 1. Unconditional log-likelihood: log P(text) ---
        target_enc = model.tokenizer(
            text, truncation=True, max_length=seq_len, return_tensors="pt",
        )
        target_ids = target_enc["input_ids"][0]  # [T]
        T = len(target_ids)
        
        if T < cfg.min_target_tokens:
            scores.append(0.0)
            continue
        
        target_ids_dev = target_ids.unsqueeze(0).to(device)
        out_uncond = model.forward(input_ids=target_ids_dev, output_hidden_states=False)
        
        # Logits at position i predict token i+1
        logits_u = out_uncond.logits[0, :-1, :]      # [T-1, V]
        targets_u = target_ids[1:].to(device)          # [T-1]
        lp_u = torch.log_softmax(logits_u, dim=-1)
        token_lp_u = lp_u[torch.arange(len(targets_u), device=device), targets_u]
        unconditional_ll = token_lp_u.mean()
        
        # --- 2. Conditional log-likelihood: log P(text | nonmember_prefix) ---
        # Concatenate prefix and target token IDs directly (avoids
        # re-tokenisation boundary artefacts from string concatenation).
        cond_ids = torch.cat([prefix_ids, target_ids], dim=0)  # [P+T]
        P = len(prefix_ids)
        
        # Truncate if the combined sequence exceeds model's context window
        max_ctx = getattr(model.model.config, "max_position_embeddings", 2048)
        if len(cond_ids) > max_ctx:
            cond_ids = cond_ids[:max_ctx]
        
        cond_ids_dev = cond_ids.unsqueeze(0).to(device)
        out_cond = model.forward(input_ids=cond_ids_dev, output_hidden_states=False)
        
        # Extract log-probs only for the **target** portion.
        # Position P-1 predicts token P (first target token), etc.
        n_target_in_cond = len(cond_ids) - P
        if n_target_in_cond < 2:
            scores.append(0.0)
            continue
        
        logits_c = out_cond.logits[0, P - 1 : P - 1 + n_target_in_cond, :]  # [n, V]
        targets_c = cond_ids[P : P + n_target_in_cond].to(device)             # [n]
        lp_c = torch.log_softmax(logits_c, dim=-1)
        token_lp_c = lp_c[torch.arange(len(targets_c), device=device), targets_c]
        conditional_ll = token_lp_c.mean()
        
        # --- 3. Score = ratio (conditional / unconditional) ---
        # Both are negative; ratio > 1 means prefix barely hurts → member.
        if unconditional_ll.abs() < 1e-10:
            scores.append(0.0)
        else:
            scores.append((conditional_ll / unconditional_ll).item())
    
    return torch.tensor(scores, device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Con-ReCall baseline (Wang et al., 2024)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConReCallConfig:
    """Config for Con-ReCall MIA method (Wang et al., 2024).

    Con-ReCall computes the membership score as:

        s(x) = [LL(x | P_non) - γ · LL(x | P_mem)] / LL(x)

    where LL(·) denotes the mean log-likelihood, P_mem and P_non are prefixes
    constructed from member and non-member reference texts respectively, and
    γ controls the contrastive strength.

    This implementation follows the original paper (arXiv:2409.03363v2, §3.3).
    Note: No official author repo found as of 2026-02-20.
    Reference impl: imperial-aisp/mia_llms_benchmark (Meeus et al.).

    Key details from the paper:
    - Member and non-member prefixes are constructed by concatenating `num_shots`
      texts from each pool (§4.1, Appendix B).
    - γ is swept from 0.1 to 1.0; best γ is reported per setting (§4.1).
    - 7 shots on WikiMIA; 1-10 shots (best reported) on MIMIR (§4.1).
    - Prefix texts are excluded from evaluation for all methods (Appendix B).

    References:
        Wang et al., "Con-ReCall: Detecting Pre-training Data in LLMs via
        Contrastive Decoding" (2024). arXiv:2409.03363
    """
    num_shots: int = 7            # Number of texts concatenated per prefix
    max_prefix_tokens: int = 256  # Max tokens per prefix (member or non-member)
    min_target_tokens: int = 10   # Skip targets shorter than this
    gamma: float = 0.5            # Contrastive strength (paper sweeps 0.1-1.0)


@torch.no_grad()
def score_con_recall(
    model,
    texts: List[str],
    ref_member_texts: List[str],
    ref_nonmember_texts: List[str],
    seq_len: int,
    cfg: ConReCallConfig,
) -> torch.Tensor:
    """Con-ReCall MIA score (Wang et al., 2024).

    Implements the exact formula from §3.3 of arXiv:2409.03363v2:

        s(x) = [LL(x | P_non) - γ · LL(x | P_mem)] / LL(x)

    where:
    - LL(x) = mean log P(x_t | x_{<t})  (unconditional)
    - LL(x | P) = mean log P(x_t | P, x_{<t})  (conditioned on prefix P)
    - P_mem = concatenation of `num_shots` member reference texts
    - P_non = concatenation of `num_shots` non-member reference texts
    - γ = contrastive strength parameter

    Higher score => more likely member. For members, the non-member prefix
    degrades predictions (low LL(x|P_non)) while the member prefix barely
    helps (LL(x|P_mem) ≈ LL(x)), yielding a lower numerator and thus lower
    score. Wait — per the paper's analysis: for members, LL(x|P_mem) is
    high (similar distribution) and LL(x|P_non) is low, so the numerator
    is negative. For non-members, both conditional LLs are degraded
    similarly, so the contrast is smaller. The sign/direction is handled by
    automatic orientation calibration in the evaluation pipeline.

    Args:
        model: CausalLMWrapper.
        texts: Target texts to score.
        ref_member_texts: Known member texts for the member prefix.
        ref_nonmember_texts: Known non-member texts for the non-member prefix.
        seq_len: Max sequence length for target text.
        cfg: ConReCallConfig.

    Returns:
        Tensor of shape [len(texts)] with Con-ReCall scores.
    """
    device = next(model.model.parameters()).device

    # --- Build member prefix token IDs ---
    mem_prefix_parts = ref_member_texts[: cfg.num_shots]
    if not mem_prefix_parts:
        raise ValueError(
            "Con-ReCall requires at least 1 member reference text, "
            "but ref_member_texts is empty."
        )
    mem_prefix_str = " ".join(mem_prefix_parts)
    mem_prefix_enc = model.tokenizer(
        mem_prefix_str, truncation=True, max_length=cfg.max_prefix_tokens,
        return_tensors="pt",
    )
    mem_prefix_ids = mem_prefix_enc["input_ids"][0]  # [P_mem]

    # --- Build non-member prefix token IDs ---
    non_prefix_parts = ref_nonmember_texts[: cfg.num_shots]
    if not non_prefix_parts:
        raise ValueError(
            "Con-ReCall requires at least 1 non-member reference text, "
            "but ref_nonmember_texts is empty."
        )
    non_prefix_str = " ".join(non_prefix_parts)
    non_prefix_enc = model.tokenizer(
        non_prefix_str, truncation=True, max_length=cfg.max_prefix_tokens,
        return_tensors="pt",
    )
    non_prefix_ids = non_prefix_enc["input_ids"][0]  # [P_non]

    max_ctx = getattr(model.model.config, "max_position_embeddings", 2048)

    def _mean_logp_conditioned(target_ids: torch.Tensor, prefix_ids: torch.Tensor) -> float:
        """Compute mean log P(target | prefix) over target tokens."""
        cond_ids = torch.cat([prefix_ids, target_ids], dim=0)
        P = len(prefix_ids)
        if len(cond_ids) > max_ctx:
            cond_ids = cond_ids[:max_ctx]
        n_target_in_cond = len(cond_ids) - P
        if n_target_in_cond < 2:
            return 0.0
        cond_ids_dev = cond_ids.unsqueeze(0).to(device)
        out = model.forward(input_ids=cond_ids_dev, output_hidden_states=False)
        # Position P-1 predicts token P (first target token)
        logits = out.logits[0, P - 1: P - 1 + n_target_in_cond, :]
        targets = cond_ids[P: P + n_target_in_cond].to(device)
        lp = torch.log_softmax(logits, dim=-1)
        token_lp = lp[torch.arange(len(targets), device=device), targets]
        return token_lp.mean().item()

    def _mean_logp_unconditioned(target_ids: torch.Tensor) -> float:
        """Compute mean log P(target) unconditionally."""
        T = len(target_ids)
        if T < 2:
            return 0.0
        ids_dev = target_ids.unsqueeze(0).to(device)
        out = model.forward(input_ids=ids_dev, output_hidden_states=False)
        logits = out.logits[0, :-1, :]
        targets = target_ids[1:].to(device)
        lp = torch.log_softmax(logits, dim=-1)
        token_lp = lp[torch.arange(len(targets), device=device), targets]
        return token_lp.mean().item()

    scores = []
    for text in texts:
        target_enc = model.tokenizer(
            text, truncation=True, max_length=seq_len, return_tensors="pt",
        )
        target_ids = target_enc["input_ids"][0]

        if len(target_ids) < cfg.min_target_tokens:
            scores.append(0.0)
            continue

        # 1. LL(x) — unconditional
        ll_uncond = _mean_logp_unconditioned(target_ids)

        # 2. LL(x | P_non) — conditioned on non-member prefix
        ll_non = _mean_logp_conditioned(target_ids, non_prefix_ids)

        # 3. LL(x | P_mem) — conditioned on member prefix
        ll_mem = _mean_logp_conditioned(target_ids, mem_prefix_ids)

        # 4. Score = [LL(x|P_non) - γ * LL(x|P_mem)] / LL(x)
        if abs(ll_uncond) < 1e-10:
            scores.append(0.0)
        else:
            s = (ll_non - cfg.gamma * ll_mem) / ll_uncond
            scores.append(s)

    return torch.tensor(scores, device=device, dtype=torch.float32)
