# ---------------------------------------------------------------------------
# Original to this project (no external source).
# ---------------------------------------------------------------------------
"""Aggregation utilities for SAE-based membership inference methods.

This module provides consistent aggregation functions used by SAEFeaturePDD
and SAENAPDD for combining per-feature or per-layer scores into a single
membership score.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np


AggMode = Literal["max", "mean", "topk_mean", "trimmed_mean"]


def aggregate(
    v: np.ndarray,
    mode: AggMode,
    k: Optional[int] = None,
    trim_frac: Optional[float] = None,
) -> float:
    """Aggregate a vector of scores into a single scalar.

    Args:
        v: 1D array of per-feature or per-layer scores.
        mode: Aggregation mode.
            - "max": Maximum value.
            - "mean": Arithmetic mean.
            - "topk_mean": Mean of top-k values (requires k).
            - "trimmed_mean": Trimmed mean excluding extreme values (requires trim_frac).
        k: Number of top elements to average (required for topk_mean).
        trim_frac: Fraction of elements to trim from each end (required for trimmed_mean).

    Returns:
        Aggregated scalar score.

    Raises:
        ValueError: If required parameters are missing or mode is unknown.
    """
    if v.size == 0:
        return 0.0

    v = v.ravel()

    if mode == "max":
        return float(v.max())

    if mode == "mean":
        return float(v.mean())

    if mode == "topk_mean":
        if k is None:
            raise ValueError("topk_mean requires k parameter")
        kk = min(k, v.size)
        if kk <= 0:
            return 0.0
        # Partial sort to get top-k elements
        return float(np.partition(v, -kk)[-kk:].mean())

    if mode == "trimmed_mean":
        if trim_frac is None:
            raise ValueError("trimmed_mean requires trim_frac parameter")
        if not 0.0 <= trim_frac < 0.5:
            raise ValueError(f"trim_frac must be in [0, 0.5), got {trim_frac}")
        t = int(np.floor(trim_frac * v.size))
        vv = np.sort(v)
        remaining = v.size - 2 * t
        if remaining > 0:
            vv = vv[t : v.size - t]
        return float(vv.mean())

    raise ValueError(f"Unknown aggregation mode: {mode}")


def aggregate_batch(
    V: np.ndarray,
    mode: AggMode,
    k: Optional[int] = None,
    trim_frac: Optional[float] = None,
    axis: int = 1,
) -> np.ndarray:
    """Aggregate along an axis for a batch of score vectors.

    Args:
        V: 2D array of shape [batch, features] (if axis=1).
        mode: Aggregation mode.
        k: Number of top elements (for topk_mean).
        trim_frac: Trim fraction (for trimmed_mean).
        axis: Axis along which to aggregate (default: 1).

    Returns:
        1D array of aggregated scores, shape [batch] if axis=1.
    """
    if V.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {V.shape}")

    if axis == 0:
        V = V.T  # Transpose so we can iterate over rows

    result = np.array([aggregate(row, mode, k, trim_frac) for row in V])
    return result
