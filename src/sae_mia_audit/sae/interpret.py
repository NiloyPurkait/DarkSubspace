"""Heuristic feature interpretation from top-activating contexts.

Generates short n-gram labels for SAE features based on the tokens that
most strongly activate them.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class FeatureLabel:
    feature_id: int
    label: str
    top_ngrams: List[Tuple[str, int]]


def heuristic_label_from_contexts(
    feature_id: int,
    contexts: Sequence[str],
    ngram_max: int = 3,
    top_k: int = 8,
) -> FeatureLabel:
    """Crude 'auto-interpretability' placeholder:
    - build n-gram frequency from top contexts
    - return a short label from the most common n-gram

    For camera-ready work, you can replace this with an LLM-based
    auto-interpretability pipeline (optional).
    """
    tokens = []
    for c in contexts:
        # very rough tokenization
        toks = [t.strip() for t in c.replace("\n", " ").split(" ") if t.strip()]
        tokens.append(toks)

    counts: Counter[str] = Counter()
    for toks in tokens:
        for n in range(1, ngram_max + 1):
            for i in range(0, max(0, len(toks) - n + 1)):
                ng = " ".join(toks[i : i + n]).lower()
                if len(ng) < 3:
                    continue
                counts[ng] += 1

    top = counts.most_common(top_k)
    label = top[0][0] if top else "(unlabeled)"
    return FeatureLabel(feature_id=feature_id, label=label, top_ngrams=[(k, int(v)) for k, v in top])
