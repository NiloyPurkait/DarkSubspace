"""Global seeding helpers for reproducibility across Python, NumPy, and torch.

Determinism policy.
``SeedConfig.deterministic`` defaults to ``False`` because the paper's GPU
pipeline runs at non-trivial scale (mixed-data SAE training, layer sweeps on
12B-parameter models, decoded-text generation for ROUGE-L) and the
``torch.use_deterministic_algorithms(True)`` requirement materially slows the
pipeline; CUDNN heuristics (``cudnn.benchmark=True``) are needed for tractable
walltime. The bundled paper-claim JSONs and per-seed cohort aggregates record
``seed`` and ``bootstrap_seed`` fields so reviewer-side re-execution can match
the random-number streams used at the Python/NumPy/torch RNG level. Reviewers
who require bit-reproducible CUDA outputs should pass ``deterministic=True``
to ``SeedConfig`` (the paper-cited values were not produced under that
setting; small numerical differences are expected because CUDNN selects
different convolution kernels per-run).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 0
    deterministic: bool = False


def set_global_seed(cfg: SeedConfig) -> None:
    """Best-effort global seeding for reproducibility.

    See module docstring for the determinism policy. ``deterministic=False``
    enables CUDNN heuristics for speed; the bundled paper-claim JSONs were
    produced under this default.
    """
    seed = int(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cfg.deterministic:
        # Reviewer opt-in for bit-reproducible CUDA outputs. Materially slower
        # than the default, and ``torch.use_deterministic_algorithms`` may
        # raise on certain ops; we use ``warn_only=True`` so the run does not
        # fail under unsupported kernels.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
