"""Global seeding helpers for reproducibility across Python, NumPy, and torch."""
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
    """Best-effort global seeding for reproducibility."""
    seed = int(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cfg.deterministic:
        # Note: deterministic can reduce speed and may still not guarantee
        # determinism for all CUDA ops.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
