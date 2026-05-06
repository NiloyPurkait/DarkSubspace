from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Literal, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset


def _safe_meta_value(v, max_list_elems: int = 32):
    """Make meta JSON-serializable and bounded.

    Some HF datasets include large lists (neighbors, tokens, etc.). For audit
    reporting and fairness slicing, we keep metadata compact.
    """
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        out = []
        for it in v[:max_list_elems]:
            if isinstance(it, (str, int, float, bool)) or it is None:
                out.append(it)
            else:
                out.append(str(it))
        if len(v) > max_list_elems:
            out.append(f"...(+{len(v)-max_list_elems} more)")
        return out
    # Fallback: stringify
    return str(v)


@dataclass(frozen=True)
class PDDExample:
    text: str
    label: int  # 1=member, 0=non-member
    meta: dict


# NOTE: "ccnews" is renamed to "ccnews_raw" to clarify it's NOT the full CCNewsPDD
# with transformations. Use "ccnews_pdd" for the reviewer-proof version with year
# filtering and text transformations.
PDDName = Literal["wikia", "wikia_para", "mimir", "arxivmia", "bookmia", "ccnews_raw", "ccnews_pdd"]


@dataclass(frozen=True)
class PDDDatasetSpec:
    name: PDDName
    # WikiMIA
    wikia_length: int = 128  # 32/64/128/256
    # WikiMIA paraphrased (zjysteven/WikiMIA_paraphrased_perturbed)
    wikia_para_type: str = "paraphrase"  # "paraphrase" or "perturbed"
    # MIMIR
    mimir_source: str = "pile_cc"
    mimir_split: str = "ngram_13_0.8"  # B8: SoK-recommended (Meeus et al. SaTML 2025)
    # ArxivMIA (HF config name, NOT split - each config only has 'train' split)
    # Valid configs: 'arxiv_mia' (full), 'arxiv_mia_dev', 'arxiv_mia_test'
    arxiv_config: str = "arxiv_mia_test"  # HF config name for test set evaluation
    # CCNewsPDD options (vblagoje/cc_news + monology/pile-uncopyrighted)
    # PAPER EXACT: Only max_chars < 512 filter, NO minimum length!
    ccnews_max_chars: int = 512  # Filter texts by max character length (paper: len < 512)
    # Split sizes matching paper exactly:
    # Train: 200 total (100 member + 100 nonmember) - BOTH from CC News!
    # Dev: 400 total (200 member from Pile-CC + 200 nonmember from CC News)
    # Test: 800 total (400 member from Pile-CC + 400 nonmember from CC News)
    ccnews_split: Literal["train", "dev", "test"] = "test"  # Which split to load
    ccnews_seed: int = 42  # Random seed for reproducibility
    # CCNewsPDD transformation variant (for reviewer-proof non-member generation)
    # "raw" = no transformation (vulnerable to distribution shift critique)
    # "trans" = back-translation (EN→FR→EN via MarianMT) - paper default
    # "mask" = MLM masking + substitution (BERT-style)
    ccnews_variant: Literal["raw", "trans", "mask"] = "trans"
    # B5: Length balancing to prevent length confounds
    # When True, sample member/nonmember pairs with matched length distributions
    # This prevents the "length baseline AUROC >> 0.5" reviewer attack
    ccnews_length_balanced: bool = True  # Default=True for reviewer-proofing
    ccnews_length_bins: int = 10  # Number of length bins for matching
    # general
    limit: Optional[int] = None  # cap examples for quick tests


def _infer_text_column(ds: Dataset) -> str:
    candidates = ["text", "input", "snippet", "sample", "content", "sentence", "document"]
    cols = set(ds.column_names)
    for c in candidates:
        if c in cols:
            return c
    # fallback: first string column
    for c in ds.column_names:
        v = ds[c][0]
        if isinstance(v, str):
            return c
    raise ValueError(f"Could not infer text column from columns={ds.column_names}")


def _infer_label_column(ds: Dataset) -> str:
    candidates = ["label", "labels", "is_member", "member", "gold"]
    cols = set(ds.column_names)
    for c in candidates:
        if c in cols:
            return c
    # fallback: first int-like column
    for c in ds.column_names:
        v = ds[c][0]
        if isinstance(v, (int, bool)):
            return c
    raise ValueError(f"Could not infer label column from columns={ds.column_names}")


def _iter_wikia(length: int, limit: Optional[int]) -> Iterator[PDDExample]:
    split_name = f"WikiMIA_length{length}"
    ds = load_dataset("swj0419/WikiMIA", split=split_name)
    text_col = _infer_text_column(ds)
    label_col = _infer_label_column(ds)

    n = len(ds) if limit is None else min(len(ds), limit)
    for i in range(n):
        row = ds[i]
        meta = {"idx": i, "split": split_name}
        for k, v in row.items():
            if k in (text_col, label_col):
                continue
            meta[k] = _safe_meta_value(v)
        yield PDDExample(text=row[text_col], label=int(row[label_col]), meta=meta)


def _iter_bookmia(limit: Optional[int]) -> Iterator[PDDExample]:
    """Load BookMIA dataset from swj0419/BookMIA.

    BookMIA (Shi et al., 2023) tests membership on book excerpts (length-512 passages).
    Members (label=1) are from books published before 2023 (likely in pretraining corpora);
    non-members (label=0) are from 2023 books. ~9,800 samples total.

    Same authors as WikiMIA; provides substantially more statistical power and tests
    a qualitatively different domain (long-form narrative vs. encyclopedic text).

    Dataset: https://huggingface.co/datasets/swj0419/BookMIA
    """
    ds = load_dataset("swj0419/BookMIA", split="train")
    text_col = _infer_text_column(ds)
    label_col = _infer_label_column(ds)

    n = len(ds) if limit is None else min(len(ds), limit)
    for i in range(n):
        row = ds[i]
        meta = {"idx": i, "split": "train"}
        for k, v in row.items():
            if k in (text_col, label_col):
                continue
            meta[k] = _safe_meta_value(v)
        yield PDDExample(text=row[text_col], label=int(row[label_col]), meta=meta)


def _iter_wikia_paraphrased(length: int, para_type: str, limit: Optional[int]) -> Iterator[PDDExample]:
    """Load WikiMIA paraphrased/perturbed dataset from zjysteven/WikiMIA_paraphrased_perturbed.
    
    This dataset contains paraphrased versions of WikiMIA for robustness testing.
    The paraphrased data is generated by ChatGPT with instruction to replace certain words.
    
    References:
        - Min-K%++ paper: https://arxiv.org/abs/2404.02936
        - Dataset: https://huggingface.co/datasets/zjysteven/WikiMIA_paraphrased_perturbed
    
    Args:
        length: Text length (32, 64, 128, 256)
        para_type: "paraphrase" for ChatGPT-paraphrased, "perturbed" for MLM-perturbed
        limit: Maximum number of examples to load
    """
    # Dataset has splits like "WikiMIA_length64_paraphrase" or "WikiMIA_length64_perturbed"
    split_name = f"WikiMIA_length{length}_{para_type}"
    try:
        ds = load_dataset("zjysteven/WikiMIA_paraphrased_perturbed", split=split_name)
    except Exception as e:
        raise ValueError(f"Failed to load WikiMIA paraphrased dataset split={split_name}: {e}")
    
    text_col = _infer_text_column(ds)
    label_col = _infer_label_column(ds)

    n = len(ds) if limit is None else min(len(ds), limit)
    for i in range(n):
        row = ds[i]
        meta = {"idx": i, "split": split_name, "para_type": para_type}
        for k, v in row.items():
            if k in (text_col, label_col):
                continue
            meta[k] = _safe_meta_value(v)
        yield PDDExample(text=row[text_col], label=int(row[label_col]), meta=meta)


def _iter_arxivmia(config_name: str, limit: Optional[int]) -> Iterator[PDDExample]:
    """Load ArxivMIA benchmark.
    
    IMPORTANT: The HuggingFace dataset zhliu/ArxivMIA uses CONFIGS (not splits) to
    organize data. Each config exposes only a 'train' split:
    
    - 'arxiv_mia': Full dataset (2000 samples)
    - 'arxiv_mia_dev': Development set
    - 'arxiv_mia_test': Test set (for final evaluation)
    
    Usage: load_dataset("zhliu/ArxivMIA", "<config_name>", split="train")
    
    References:
        - Dataset: https://huggingface.co/datasets/zhliu/ArxivMIA
        - Paper: Probing Language Models for Pre-training Data Detection
    
    Args:
        config_name: HF config name: 'arxiv_mia', 'arxiv_mia_dev', or 'arxiv_mia_test'
        limit: Maximum number of examples to load
    """
    # Validate config_name
    valid_configs = ('arxiv_mia', 'arxiv_mia_dev', 'arxiv_mia_test')
    if config_name not in valid_configs:
        raise ValueError(
            f"Invalid ArxivMIA config: '{config_name}'. "
            f"Valid configs: {valid_configs}. "
            f"Note: These are HF config names, not splits. Each config only has 'train' split."
        )
    
    # Load with config name and split='train' per HF structure
    ds = load_dataset("zhliu/ArxivMIA", config_name, split="train")
    text_col = _infer_text_column(ds)
    label_col = _infer_label_column(ds)

    n = len(ds) if limit is None else min(len(ds), limit)
    for i in range(n):
        row = ds[i]
        meta = {"idx": i, "config": config_name, "hf_split": "train"}
        for k, v in row.items():
            if k in (text_col, label_col):
                continue
            meta[k] = _safe_meta_value(v)
        yield PDDExample(text=row[text_col], label=int(row[label_col]), meta=meta)


def _load_mimir_dataset(source: str, split: str) -> Dataset:
    """Load a MIMIR config+split, with fallback to direct JSONL download.

    datasets >= 4.x dropped ``trust_remote_code`` so the custom loading
    script in ``iamgroot42/mimir`` no longer works.  When ``load_dataset``
    fails (cache miss for a config we haven't downloaded yet), we fetch the
    underlying JSONL via ``hf_hub_download`` and build a Dataset ourselves.

    MIMIR repo layout (cache_100_200_1000_512):
      train/<source>_<split>.jsonl   — member texts  (one JSON string per line)
      test/<source>_<split>.jsonl    — nonmember texts (one JSON string per line)
    The loading script pairs them row-by-row into {"member", "nonmember"}.
    """
    import json as _json
    import os as _os

    # --- fast path: try the HF API (works for cached configs like pile_cc) ---
    try:
        return load_dataset("iamgroot42/mimir", source, split=split)
    except (ValueError, FileNotFoundError):
        pass  # config not cached / loading-script unavailable

    # --- slow path: download member + nonmember JSONL, pair them ---------
    _source_file = "wikipedia_(en)" if source == "wikipedia" else source
    parent = "cache_100_200_1000_512"

    member_fname = f"{parent}/train/{_source_file}_{split}.jsonl"
    nonmember_fname = f"{parent}/test/{_source_file}_{split}.jsonl"

    token = _os.environ.get("HF_TOKEN", None)
    from huggingface_hub import hf_hub_download

    def _download(fname):
        return hf_hub_download(
            repo_id="iamgroot42/mimir",
            filename=fname,
            repo_type="dataset",
            token=token,
        )

    member_path = _download(member_fname)
    nonmember_path = _download(nonmember_fname)

    def _read_jsonl_strings(path):
        texts = []
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    texts.append(_json.loads(line))  # each line is a JSON string
        return texts

    members = _read_jsonl_strings(member_path)
    nonmembers = _read_jsonl_strings(nonmember_path)

    if len(members) != len(nonmembers):
        raise ValueError(
            f"MIMIR member/nonmember count mismatch for {source}/{split}: "
            f"{len(members)} vs {len(nonmembers)}"
        )

    rows = [{"member": m, "nonmember": nm} for m, nm in zip(members, nonmembers)]
    return Dataset.from_list(rows)


def _iter_mimir(source: str, split: str, limit: Optional[int]) -> Iterator[PDDExample]:
    # MIMIR is a paired dataset: each row has a member and a nonmember sample.
    ds = _load_mimir_dataset(source, split)

    # Expected features per dataset card:
    # member (str), nonmember (str), member_neighbors (List[str]), nonmember_neighbors (List[str])
    member_col = "member" if "member" in ds.column_names else _infer_text_column(ds)
    nonmember_col = "nonmember" if "nonmember" in ds.column_names else None
    if nonmember_col is None:
        raise ValueError(f"MIMIR dataset missing 'nonmember' column: columns={ds.column_names}")

    n_rows = len(ds) if limit is None else min(len(ds), limit)
    for i in range(n_rows):
        row = ds[i]
        # carry through compact metadata (neighbors, ids, etc.) if present
        base_meta = {"pair_idx": i, "split": split, "source": source}
        for k, v in row.items():
            if k in (member_col, nonmember_col):
                continue
            base_meta[k] = _safe_meta_value(v)
        yield PDDExample(text=row[member_col], label=1, meta={**base_meta, "role": "member"})
        yield PDDExample(text=row[nonmember_col], label=0, meta={**base_meta, "role": "nonmember"})


# ---------------------------------------------------------------------------
# Text transformation utilities for CCNewsPDD (reviewer-proof non-member generation)
# ---------------------------------------------------------------------------

def _backtranslate_en_fr_en(text: str, _cache: dict = {}) -> str:
    """Back-translate text: EN → FR → EN using MarianMT.
    
    This creates a semantically similar but lexically different version,
    ensuring the non-member text was never seen verbatim by the model.
    
    References:
        - NA-PDD paper uses back-translation as one of the transformations
        - MarianMT: https://huggingface.co/Helsinki-NLP/opus-mt-en-fr
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        # Fallback: return original if transformers not available
        return text
    
    # Lazy load models (cached)
    if "en_fr" not in _cache:
        _cache["en_fr_tok"] = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        _cache["en_fr"] = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        _cache["fr_en_tok"] = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        _cache["fr_en"] = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    
    try:
        # EN → FR
        inputs = _cache["en_fr_tok"](text, return_tensors="pt", truncation=True, max_length=512)
        fr_ids = _cache["en_fr"].generate(**inputs, max_length=512)
        fr_text = _cache["en_fr_tok"].decode(fr_ids[0], skip_special_tokens=True)
        
        # FR → EN
        inputs = _cache["fr_en_tok"](fr_text, return_tensors="pt", truncation=True, max_length=512)
        en_ids = _cache["fr_en"].generate(**inputs, max_length=512)
        en_text = _cache["fr_en_tok"].decode(en_ids[0], skip_special_tokens=True)
        
        return en_text
    except Exception:
        return text  # Fallback on error


def _mlm_mask_substitute(text: str, mask_prob: float = 0.15, _cache: dict = {}) -> str:
    """Apply MLM-style masking and substitution using BERT.
    
    Randomly masks ~15% of tokens and replaces with BERT predictions,
    creating lexically different but semantically similar text.
    
    References:
        - NA-PDD paper uses MLM substitution as one of the transformations
    """
    try:
        from transformers import BertTokenizer, BertForMaskedLM
        import torch
    except ImportError:
        return text
    
    # Lazy load model (cached)
    if "bert" not in _cache:
        _cache["bert_tok"] = BertTokenizer.from_pretrained("bert-base-uncased")
        _cache["bert"] = BertForMaskedLM.from_pretrained("bert-base-uncased")
        _cache["bert"].eval()
    
    try:
        tokenizer = _cache["bert_tok"]
        model = _cache["bert"]
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        if len(tokens) < 5:
            return text
        
        # Randomly select tokens to mask (skip special tokens)
        import random
        n_mask = max(1, int(len(tokens) * mask_prob))
        mask_indices = random.sample(range(len(tokens)), min(n_mask, len(tokens)))
        
        # Create masked input
        masked_tokens = tokens.copy()
        for idx in mask_indices:
            masked_tokens[idx] = "[MASK]"
        
        # Encode and predict
        input_ids = tokenizer.encode(masked_tokens, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits
        
        # Replace masks with predictions
        result_tokens = tokens.copy()
        for idx in mask_indices:
            # Find position in input_ids (offset by 1 for [CLS])
            pos = idx + 1
            if pos < predictions.shape[1]:
                pred_id = predictions[0, pos].argmax().item()
                pred_token = tokenizer.convert_ids_to_tokens([pred_id])[0]
                if not pred_token.startswith("["):  # Skip special tokens
                    result_tokens[idx] = pred_token
        
        # Reconstruct text
        return tokenizer.convert_tokens_to_string(result_tokens)
    except Exception:
        return text  # Fallback on error


def _length_balanced_sample(
    texts_a: List[str],
    texts_b: List[str], 
    n_per_class: int,
    n_bins: int,
    rng: "random.Random",
) -> Tuple[List[str], List[str]]:
    """Sample n_per_class texts from each pool with matched length distributions.
    
    B5: This is critical for preventing length confounds in CCNews benchmark.
    The algorithm:
    1. Bin texts_a by length quantiles to create n_bins buckets
    2. For each bucket, sample min(n_per_bucket, available) from both pools
    3. If one pool has insufficient samples in a bucket, skip that bucket
    4. Continue until we have n_per_class from each
    
    This ensures member/nonmember length distributions are nearly identical,
    preventing the "length baseline AUROC >> 0.5" reviewer attack.
    """
    import numpy as np
    
    # Compute length quantile boundaries from texts_a
    lens_a = np.array([len(t) for t in texts_a])
    lens_b = np.array([len(t) for t in texts_b])
    
    # Create bin edges from combined distribution for fairness
    all_lens = np.concatenate([lens_a, lens_b])
    bin_edges = np.percentile(all_lens, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1  # Ensure max length is included
    
    # Assign texts to bins
    bins_a = {i: [] for i in range(n_bins)}
    bins_b = {i: [] for i in range(n_bins)}
    
    for text in texts_a:
        for i in range(n_bins):
            if bin_edges[i] <= len(text) < bin_edges[i + 1]:
                bins_a[i].append(text)
                break
    
    for text in texts_b:
        for i in range(n_bins):
            if bin_edges[i] <= len(text) < bin_edges[i + 1]:
                bins_b[i].append(text)
                break
    
    # Shuffle within each bin
    for i in range(n_bins):
        rng.shuffle(bins_a[i])
        rng.shuffle(bins_b[i])
    
    # Sample equally from each bin (stratified by length)
    samples_a = []
    samples_b = []
    target_per_bin = n_per_class // n_bins + 1
    
    for i in range(n_bins):
        available = min(len(bins_a[i]), len(bins_b[i]), target_per_bin)
        if available > 0:
            samples_a.extend(bins_a[i][:available])
            samples_b.extend(bins_b[i][:available])
    
    # Second pass: if we're short, pull more from bins that had surplus
    if len(samples_a) < n_per_class:
        deficit = n_per_class - len(samples_a)
        for i in range(n_bins):
            already_used = min(len(bins_a[i]), len(bins_b[i]), target_per_bin)
            if already_used <= 0:
                already_used = 0
            extra_avail = min(
                len(bins_a[i]) - already_used,
                len(bins_b[i]) - already_used,
            )
            if extra_avail > 0:
                take = min(extra_avail, deficit)
                samples_a.extend(bins_a[i][already_used:already_used + take])
                samples_b.extend(bins_b[i][already_used:already_used + take])
                deficit -= take
                if deficit <= 0:
                    break
    
    # Trim to exact count and shuffle
    samples_a = samples_a[:n_per_class]
    samples_b = samples_b[:n_per_class]
    
    combined_idx = list(range(len(samples_a)))
    rng.shuffle(combined_idx)
    samples_a = [samples_a[i] for i in combined_idx]
    samples_b = [samples_b[i] for i in combined_idx]
    
    return samples_a, samples_b


def _iter_ccnews_paper_exact(
    split: Literal["train", "dev", "test"],
    max_chars: int,
    seed: int,
    variant: Literal["raw", "trans", "mask"],
    limit: Optional[int],
    length_balanced: bool = True,
    length_bins: int = 10,
) -> Iterator[PDDExample]:
    """Load CCNewsPDD benchmark with reviewer-proof length balancing.
    
    This function replicates the paper's data loading methodology with
    CRITICAL ADDITION: Length balancing to prevent confounds.
    
    PAPER DATA STRUCTURE:
    - Train: 200 samples (100 member + 100 nonmember) - BOTH from CC News!
      → All samples are back-translated (both member and nonmember)
    - Dev: 400 samples (200 member from Pile-CC + 200 nonmember from CC News)
      → Only nonmembers are back-translated, members stay original
    - Test: 800 samples (400 member from Pile-CC + 400 nonmember from CC News)
      → Only nonmembers are back-translated, members stay original
    
    B5: LENGTH BALANCING (reviewer-proofing):
    When length_balanced=True (default), we sample member/nonmember texts with
    matched length distributions to prevent the "length AUROC >> 0.5" attack.
    
    LENGTH FILTER: Paper uses ONLY `len(text) < 512`, NO minimum length!
    
    YEAR FILTER: Paper does NOT filter by year (no temporal restriction)
    
    ⚠️ OPT MEMBERSHIP CAVEAT (B5):
    OPT's training data includes CCNewsV2, so CC News nonmembers may not be true
    nonmembers for OPT. This benchmark should be framed as "source-based audit"
    or restricted to Pythia models for valid membership claims.
    
    References:
        - NA-PDD paper: https://arxiv.org/abs/2310.16789
        - Paper's Colab notebooks for CCNewsPDD
    
    Args:
        split: "train", "dev", or "test" - determines data sources and sizes
        max_chars: Maximum character length (paper uses 512, filter: len < max_chars)
        seed: Random seed for reproducibility (paper uses 42)
        variant: Transformation: "raw", "trans" (back-translation), "mask" (MLM)
        limit: Cap total examples
        length_balanced: If True, balance length distributions between classes
        length_bins: Number of length bins for stratified sampling
    """
    import random
    from tqdm.auto import tqdm
    
    rng = random.Random(seed)
    
    # Split sizes from paper
    SPLIT_SIZES = {
        "train": {"n_member": 100, "n_nonmember": 100},  # Both from CC News!
        "dev": {"n_member": 200, "n_nonmember": 200},
        "test": {"n_member": 400, "n_nonmember": 400},
    }
    
    n_member = SPLIT_SIZES[split]["n_member"]
    n_nonmember = SPLIT_SIZES[split]["n_nonmember"]
    
    # For train split, we need enough CC News samples for BOTH member and nonmember
    # For dev/test splits, nonmembers come from CC News, members from Pile-CC
    
    # B5: For length balancing, we need extra samples to do stratified sampling
    # Test split needs 400 samples — use larger buffer to survive bin attrition
    if length_balanced:
        buffer_mult = 10 if split == "test" else 6
    else:
        buffer_mult = 2
    
    # --------------------------------------------------------------------------
    # Step 1: Load CC News data (used differently depending on split)
    # --------------------------------------------------------------------------
    print(f"Loading CC News data for split={split} (length_balanced={length_balanced})...")
    cc_news = load_dataset("vblagoje/cc_news", split="train", streaming=True)
    
    # Paper filter: len(text) < 512 only, NO minimum length!
    cc_news_texts = []
    for item in tqdm(cc_news, desc="Filtering CC News (len < 512)", dynamic_ncols=True):
        text = item.get("text", "")
        if text and len(text) < max_chars:
            cc_news_texts.append(text)
            # Collect enough for the split
            if split == "train":
                # Train: both classes from CC News (200 total + buffer)
                if len(cc_news_texts) >= (n_member + n_nonmember) * buffer_mult:
                    break
            else:
                # Dev/Test: only nonmembers from CC News
                if len(cc_news_texts) >= n_nonmember * buffer_mult:
                    break
    
    rng.shuffle(cc_news_texts)
    
    # B7: Deduplicate CC News texts (source dataset contains exact duplicates
    # from multi-outlet syndication). Without this, ~12 duplicates leak across
    # ref/val/test splits, causing 100% integrity-check failure on CCNews.
    _seen = set()
    _deduped = []
    for _t in cc_news_texts:
        _h = hashlib.sha256(_t.strip().lower().encode()).hexdigest()[:16]
        if _h not in _seen:
            _seen.add(_h)
            _deduped.append(_t)
    n_cc_dupes = len(cc_news_texts) - len(_deduped)
    if n_cc_dupes:
        print(f"  Removed {n_cc_dupes} intra-source duplicates from CC News pool")
    cc_news_texts = _deduped
    
    # --------------------------------------------------------------------------
    # Step 2: Load Pile-CC data (only needed for dev/test members)
    # --------------------------------------------------------------------------
    pile_cc_texts = []
    if split in ("dev", "test"):
        print(f"Loading Pile-CC data for split={split}...")
        pile_stream = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        
        for item in tqdm(pile_stream, desc="Filtering Pile-CC (len < 512)", dynamic_ncols=True):
            meta = item.get("meta", {})
            pile_set = meta.get("pile_set_name", "")
            text = item.get("text", "")
            
            # Paper filter: len(text) < 512 only, NO minimum length!
            if pile_set == "Pile-CC" and text and len(text) < max_chars:
                pile_cc_texts.append(text)
                if len(pile_cc_texts) >= n_member * buffer_mult:
                    break
        
        rng.shuffle(pile_cc_texts)
        
        # B7: Deduplicate Pile-CC texts
        _seen_pile = set()
        _deduped_pile = []
        for _t in pile_cc_texts:
            _h = hashlib.sha256(_t.strip().lower().encode()).hexdigest()[:16]
            if _h not in _seen_pile:
                _seen_pile.add(_h)
                _deduped_pile.append(_t)
        n_pile_dupes = len(pile_cc_texts) - len(_deduped_pile)
        if n_pile_dupes:
            print(f"  Removed {n_pile_dupes} intra-source duplicates from Pile-CC pool")
        pile_cc_texts = _deduped_pile
        
        # B7: Cross-source deduplication — remove any text appearing in both pools
        _cc_hashes = {hashlib.sha256(t.strip().lower().encode()).hexdigest()[:16]
                      for t in cc_news_texts}
        _pile_before = len(pile_cc_texts)
        pile_cc_texts = [t for t in pile_cc_texts
                         if hashlib.sha256(t.strip().lower().encode()).hexdigest()[:16]
                         not in _cc_hashes]
        n_cross_dupes = _pile_before - len(pile_cc_texts)
        if n_cross_dupes:
            print(f"  Removed {n_cross_dupes} cross-source duplicates (Pile-CC ∩ CC News)")
    
    # --------------------------------------------------------------------------
    # Step 3: Assign texts to member/nonmember based on split
    # B5: With length balancing to prevent confounds
    # --------------------------------------------------------------------------
    if split == "train":
        # PAPER TRAIN: BOTH member and nonmember come from CC News!
        # This is the key insight - training uses the same source for both classes
        # No length balancing needed since same source
        train_cc_news = cc_news_texts[:(n_member + n_nonmember)]
        rng.shuffle(train_cc_news)
        
        member_texts = train_cc_news[:n_member]
        nonmember_texts = train_cc_news[n_member:n_member + n_nonmember]
        
        member_source = "cc_news"  # For training, member is also from CC News!
        
        # PAPER: Back-translate ALL training samples (both member and nonmember)
        if variant == "trans":
            print(f"Back-translating ALL {len(member_texts) + len(nonmember_texts)} training samples...")
            member_texts = [_backtranslate_en_fr_en(t) for t in tqdm(member_texts, desc="Back-translating members")]
            nonmember_texts = [_backtranslate_en_fr_en(t) for t in tqdm(nonmember_texts, desc="Back-translating nonmembers")]
        elif variant == "mask":
            print(f"MLM masking ALL {len(member_texts) + len(nonmember_texts)} training samples...")
            member_texts = [_mlm_mask_substitute(t) for t in tqdm(member_texts, desc="MLM masking members")]
            nonmember_texts = [_mlm_mask_substitute(t) for t in tqdm(nonmember_texts, desc="MLM masking nonmembers")]
        # variant == "raw" → no transformation
        
    else:
        # PAPER DEV/TEST: members from Pile-CC, nonmembers from CC News
        # B5: Apply length balancing to prevent confounds
        if length_balanced:
            print(f"Applying length balancing with {length_bins} bins...")
            member_texts, nonmember_texts = _length_balanced_sample(
                texts_a=pile_cc_texts,
                texts_b=cc_news_texts,
                n_per_class=min(n_member, n_nonmember),
                n_bins=length_bins,
                rng=rng,
            )
            # Ensure we have exactly the right counts
            member_texts = member_texts[:n_member]
            nonmember_texts = nonmember_texts[:n_nonmember]
            print(f"Length-balanced: {len(member_texts)} members, {len(nonmember_texts)} nonmembers")
        else:
            member_texts = pile_cc_texts[:n_member]
            nonmember_texts = cc_news_texts[:n_nonmember]
        
        member_source = "pile_cc"  # For dev/test, member is from Pile-CC
        
        # PAPER: Back-translate ONLY nonmembers in dev/test (members stay original)
        if variant == "trans":
            print(f"Back-translating {len(nonmember_texts)} nonmember samples only...")
            nonmember_texts = [_backtranslate_en_fr_en(t) for t in tqdm(nonmember_texts, desc="Back-translating nonmembers")]
        elif variant == "mask":
            print(f"MLM masking {len(nonmember_texts)} nonmember samples only...")
            nonmember_texts = [_mlm_mask_substitute(t) for t in tqdm(nonmember_texts, desc="MLM masking nonmembers")]
        # variant == "raw" → no transformation
    
    # --------------------------------------------------------------------------
    # Step 4: Validate we have enough samples (graceful degradation)
    # --------------------------------------------------------------------------
    if len(member_texts) < n_member:
        source = "CC News" if split == "train" else "Pile-CC"
        shortfall_pct = (1 - len(member_texts) / n_member) * 100
        msg = (
            f"Length balancing reduced {source} members from {n_member} to "
            f"{len(member_texts)} ({shortfall_pct:.1f}% shortfall). "
            f"Proceeding with available samples."
        )
        if shortfall_pct > 50:
            raise ValueError(
                f"Could not collect enough {source} samples for {split} members: "
                f"got {len(member_texts)}, need {n_member}. Shortfall too large."
            )
        import warnings
        warnings.warn(msg)
        print(f"WARNING: {msg}")
        n_member = len(member_texts)
    if len(nonmember_texts) < n_nonmember:
        shortfall_pct = (1 - len(nonmember_texts) / n_nonmember) * 100
        msg = (
            f"Length balancing reduced CC News nonmembers from {n_nonmember} to "
            f"{len(nonmember_texts)} ({shortfall_pct:.1f}% shortfall). "
            f"Proceeding with available samples."
        )
        if shortfall_pct > 50:
            raise ValueError(
                f"Could not collect enough CC News samples for {split} nonmembers: "
                f"got {len(nonmember_texts)}, need {n_nonmember}. Shortfall too large."
            )
        import warnings
        warnings.warn(msg)
        print(f"WARNING: {msg}")
        n_nonmember = len(nonmember_texts)
    
    # --------------------------------------------------------------------------
    # Step 5: Yield examples
    # --------------------------------------------------------------------------
    # Yield member examples
    for i, text in enumerate(member_texts):
        meta = {
            "idx": i,
            "source": member_source,
            "role": "member",
            "split": split,
            "benchmark": "ccnews_pdd",
            "variant": variant,
            "length_balanced": length_balanced,  # B5: Track for reproducibility
            "char_len": len(text),
        }
        yield PDDExample(text=text, label=1, meta=meta)
    
    # Yield non-member examples (CC News, transformed)
    for i, text in enumerate(nonmember_texts):
        meta = {
            "idx": i,
            "source": "cc_news",
            "role": "nonmember",
            "split": split,
            "benchmark": "ccnews_pdd",
            "variant": variant,
            "length_balanced": length_balanced,  # B5: Track for reproducibility
            "char_len": len(text),
        }
        yield PDDExample(text=text, label=0, meta=meta)


def _iter_enron(limit: Optional[int] = None) -> Iterator[PDDExample]:
    """Load Enron email privacy benchmark from prepared JSONL files."""
    import json as _json
    from pathlib import Path as _Path

    data_dir = _Path(__file__).resolve().parents[3] / "data" / "enron_privacy"
    member_path = data_dir / "member.jsonl"
    nonmember_path = data_dir / "nonmember.jsonl"

    if not member_path.exists():
        raise FileNotFoundError(
            f"Enron data not found at {data_dir}. "
            f"Run: env/bin/python3 scripts/privacy/enron_membership.py --prepare"
        )

    count = 0
    for path, label in [(member_path, 1), (nonmember_path, 0)]:
        with open(path) as f:
            for line in f:
                if limit is not None and count >= limit:
                    return
                if line.strip():
                    text = _json.loads(line)["text"]
                    meta = {
                        "source": "enron_privacy" if label == 1 else "spamassassin_ham",
                        "label_source": "pile_set_name" if label == 1 else "non_pile",
                        "char_len": len(text),
                    }
                    yield PDDExample(text=text, label=label, meta=meta)
                    count += 1


def _iter_enron_within(limit: Optional[int] = None) -> Iterator[PDDExample]:
    """Load within-Enron control split (both halves are real Pile members).

    This is a control experiment: 500 Enron member emails are randomly split
    50/50 into pseudo-members (label=1) and pseudo-nonmembers (label=0).
    Both halves come from the same source (Enron in the Pile), so source-level
    detection methods should yield chance-level AUROC (~0.50).

    Data prepared by: scripts/privacy/within_enron_control.py
    """
    import json as _json
    from pathlib import Path as _Path

    data_dir = _Path(__file__).resolve().parents[3] / "data" / "enron_within"
    member_path = data_dir / "member.jsonl"
    nonmember_path = data_dir / "nonmember.jsonl"

    if not member_path.exists():
        raise FileNotFoundError(
            f"Within-Enron control data not found at {data_dir}. "
            f"Run: env/bin/python3 scripts/privacy/within_enron_control.py"
        )

    count = 0
    for path, label in [(member_path, 1), (nonmember_path, 0)]:
        with open(path) as f:
            for line in f:
                if limit is not None and count >= limit:
                    return
                if line.strip():
                    text = _json.loads(line)["text"]
                    meta = {
                        "source": "enron_within_pseudo_member" if label == 1 else "enron_within_pseudo_nonmember",
                        "label_source": "random_split",
                        "char_len": len(text),
                        "note": "Both halves are real Pile members; label is artificial",
                    }
                    yield PDDExample(text=text, label=label, meta=meta)
                    count += 1


def load_pdd_dataset(spec: PDDDatasetSpec) -> list[PDDExample]:
    if spec.name == "wikia":
        examples = list(_iter_wikia(spec.wikia_length, spec.limit))
    elif spec.name == "wikia_para":
        examples = list(_iter_wikia_paraphrased(spec.wikia_length, spec.wikia_para_type, spec.limit))
    elif spec.name in ("ccnews", "ccnews_pdd", "ccnews_raw"):
        # CCNewsPDD - PAPER EXACT implementation with B5 length balancing
        # Note: "ccnews_raw" now uses variant="raw" but same structure
        variant = "raw" if spec.name == "ccnews_raw" else spec.ccnews_variant
        examples = list(_iter_ccnews_paper_exact(
            split=spec.ccnews_split,
            max_chars=spec.ccnews_max_chars,
            seed=spec.ccnews_seed,
            variant=variant,
            limit=spec.limit,
            length_balanced=spec.ccnews_length_balanced,
            length_bins=spec.ccnews_length_bins,
        ))
    elif spec.name == "mimir":
        examples = list(_iter_mimir(spec.mimir_source, spec.mimir_split, spec.limit))
    elif spec.name == "arxivmia":
        examples = list(_iter_arxivmia(spec.arxiv_config, spec.limit))
    elif spec.name == "bookmia":
        examples = list(_iter_bookmia(spec.limit))
    elif spec.name == "enron":
        examples = list(_iter_enron(spec.limit))
    elif spec.name == "enron_within":
        examples = list(_iter_enron_within(spec.limit))
    else:
        raise ValueError(f"Unknown PDD dataset: {spec.name}")
    
    return examples
