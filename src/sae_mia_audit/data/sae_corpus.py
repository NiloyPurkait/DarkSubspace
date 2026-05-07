"""SAE training corpus iterators with shuffle, dedup, and mixture support.

Provides ``SAECorpusSource`` and ``SAECorpusSpec`` plus streaming text
iterators over Hugging Face datasets and local JSON/JSONL files.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterator, Optional, Tuple

import hashlib
import random
import re

# Optional dependency: keep imports lightweight for environments that only run
# non-HF components (e.g., unit tests).
try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore


_TEXT_FIELD_CANDIDATES = ("text", "content", "document")


@dataclass(frozen=True)
class SAECorpusSource:
    """One text source for SAE training."""

    name: str
    subset: Optional[str] = None
    split: str = "train"
    text_field: Optional[str] = None
    weight: float = 1.0


@dataclass(frozen=True)
class SAECorpusSpec:
    """
    SAE corpus spec.

    Backward-compatible fields (single-source):
      - name/subset/split/text_field

    Hygiene options for credible SAE training corpora:
      - streaming shuffle (buffered)
      - rolling-window dedup (bounded memory)
      - optional multi-source mixture (sources=...)
    """

    # --- Back-compat single-source defaults ---
    name: str = "allenai/c4"
    subset: Optional[str] = "en"
    split: str = "train"
    streaming: bool = True
    text_field: Optional[str] = None
    limit_examples: Optional[int] = None  # number of yielded examples (after filtering)

    # --- New: mixture support (overrides name/subset/split/text_field if provided) ---
    sources: Optional[Tuple[SAECorpusSource, ...]] = None

    # --- New: shuffling ---
    seed: int = 0
    shuffle: bool = True
    # HF streaming shuffle uses a buffer; larger = better shuffle, more RAM.
    shuffle_buffer_size: int = 10_000

    # --- New: filtering ---
    drop_empty: bool = True
    # Helps avoid tons of tiny/boilerplate docs.
    min_chars: int = 200

    # --- New: best-effort dedup ---
    dedupe: bool = True
    # Rolling window size (hashes). 100k ~ a few to tens of MB in Python; tune if needed.
    dedupe_window: int = 100_000
    # Normalize whitespace before hashing to catch near-identical duplicates.
    dedupe_normalize_whitespace: bool = True
    # Usually keep False; set True only if you explicitly want case-insensitive dedup.
    dedupe_lowercase: bool = False
    # Hash only first N chars of normalized text to cap CPU/RAM. Keeps dedupe best-effort.
    dedupe_max_chars: int = 4096

    # Cycle local files (JSONL/JSON) when exhausted instead of stopping.
    # HF streaming datasets (C4, Pile) are large enough to never exhaust in practice,
    # but local JSONL files (e.g., 2000-line member/mixed corpora) exhaust quickly.
    # When True, local file sources are reloaded and re-iterated automatically.
    cycle_local: bool = True


def _infer_text_field(ds, override: Optional[str]) -> str:
    if override is not None:
        return str(override)

    # Prefer schema-based inference (doesn't consume an example).
    features = getattr(ds, "features", None)
    if features is not None:
        for c in _TEXT_FIELD_CANDIDATES:
            if c in features:
                return c
        # fallback: first string feature
        try:
            for k, v in features.items():
                if getattr(v, "dtype", None) == "string":
                    return str(k)
        except Exception:
            pass

    raise ValueError("Could not infer text field for SAE corpus; set text_field explicitly.")


_WS_RE = re.compile(r"\s+")


def _normalize_for_dedupe(text: str, spec: SAECorpusSpec) -> str:
    t = text.strip()
    if spec.dedupe_normalize_whitespace:
        t = _WS_RE.sub(" ", t)
    if spec.dedupe_lowercase:
        t = t.lower()
    if spec.dedupe_max_chars and len(t) > spec.dedupe_max_chars:
        t = t[: spec.dedupe_max_chars]
    return t


def _hash64(s: str) -> int:
    # Stable 64-bit hash (bounded), good enough for rolling-window dedupe.
    h = hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def _load_one_dataset(name: str, subset: Optional[str], split: str, streaming: bool):
    # Support local JSONL/JSON files: if name ends with .json/.jsonl, load as local file
    if name.endswith((".json", ".jsonl")):
        return load_dataset("json", data_files=name, split=split, streaming=streaming)
    if subset is None or str(subset).strip() == "":
        return load_dataset(name, split=split, streaming=streaming)
    return load_dataset(name, subset, split=split, streaming=streaming)


def load_sae_corpus(spec: SAECorpusSpec) -> Iterator[str]:
    """Yield text documents for SAE activation sampling.

    Best-practice behaviour (if enabled in spec):
      - streaming shuffle (buffered) to reduce ordering effects
      - rolling-window dedup (best-effort) to reduce repeated docs
      - optional mixture of multiple sources

    Notes:
      - This yields *documents*. Your training code still needs to tokenize/pack to sequences.
      - Streaming shuffle is approximate; increase shuffle_buffer_size for better mixing.
    """
    if load_dataset is None:  # pragma: no cover
        raise ImportError("datasets is required for load_sae_corpus(). Install with: pip install datasets")

    # Resolve sources (mixture or single-source back-compat)
    if spec.sources is None:
        sources: Tuple[SAECorpusSource, ...] = (
            SAECorpusSource(
                name=spec.name,
                subset=spec.subset,
                split=spec.split,
                text_field=spec.text_field,
                weight=1.0,
            ),
        )
    else:
        if len(spec.sources) == 0:
            raise ValueError("SAECorpusSpec.sources was provided but empty.")
        sources = spec.sources

    rng = random.Random(int(spec.seed))

    # Helper: load a single source and return (iterator, text_field).
    def _make_iter(src: SAECorpusSource, seed_offset: int = 0):
        ds = _load_one_dataset(src.name, src.subset, src.split, streaming=bool(spec.streaming))
        effective_seed = int(spec.seed) + seed_offset
        if spec.shuffle and hasattr(ds, "shuffle"):
            try:
                if spec.streaming:
                    buf = int(spec.shuffle_buffer_size)
                    if buf > 0:
                        ds = ds.shuffle(seed=effective_seed, buffer_size=buf)
                else:
                    ds = ds.shuffle(seed=effective_seed)
            except TypeError:
                ds = ds.shuffle(seed=effective_seed)
        tf = _infer_text_field(ds, src.text_field)
        return iter(ds), tf

    # Prepare iterators (one per source) and their text fields.
    iters = []
    text_fields = []
    weights = []
    source_list = []  # track source objects for reload
    is_local = []     # whether source is a local file (cyclable)
    cycle_count = []  # how many times each source has been reloaded

    for src in sources:
        it, tf = _make_iter(src)
        iters.append(it)
        text_fields.append(tf)
        source_list.append(src)
        is_local.append(src.name.endswith((".json", ".jsonl")))
        cycle_count.append(0)
        w = float(src.weight) if src.weight is not None else 1.0
        weights.append(max(0.0, w))

    if not any(w > 0 for w in weights):
        weights = [1.0 for _ in weights]

    # Rolling-window dedupe state
    seen: set[int] = set()
    q: Deque[int] = deque()

    yielded = 0
    while iters:
        idx = rng.choices(range(len(iters)), weights=weights, k=1)[0]
        try:
            row = next(iters[idx])
        except StopIteration:
            # If this is a local file and cycling is enabled, reload it.
            if spec.cycle_local and is_local[idx]:
                cycle_count[idx] += 1
                new_it, _ = _make_iter(source_list[idx], seed_offset=cycle_count[idx])
                iters[idx] = new_it
                # Clear dedup state so cycled documents are not rejected.
                # Each cycle uses a different shuffle seed, so ordering varies.
                seen.clear()
                q.clear()
            else:
                iters.pop(idx)
                text_fields.pop(idx)
                weights.pop(idx)
                source_list.pop(idx)
                is_local.pop(idx)
                cycle_count.pop(idx)
            continue

        try:
            text = str(row[text_fields[idx]])
        except Exception:
            # If schema doesn't match, force explicit configuration.
            raise ValueError(f"Row missing expected text field '{text_fields[idx]}'. Set text_field explicitly.")

        t = text.strip()

        if spec.drop_empty and not t:
            continue
        if spec.min_chars and len(t) < int(spec.min_chars):
            continue

        if spec.dedupe:
            key = _normalize_for_dedupe(t, spec)
            h = _hash64(key)
            if h in seen:
                continue
            seen.add(h)
            q.append(h)
            if len(q) > int(spec.dedupe_window):
                old = q.popleft()
                seen.discard(old)

        yield text
        yielded += 1
        if spec.limit_examples is not None and yielded >= int(spec.limit_examples):
            break
