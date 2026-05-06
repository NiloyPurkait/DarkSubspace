"""Membership auditing / PDD methods.

This package contains multiple methods (likelihood-based, activation-based, and
SAE-based). To keep import time small (and to avoid importing heavy dependencies
like `transformers` unless needed), this `__init__` implements **lazy
re-exports**.

Prefer importing from submodules directly:

```python
from sae_mia_audit.methods.min_k import score_min_kpp
from sae_mia_audit.methods.sae_na_pdd import SAENAPDD
```

But the following names are also available from `sae_mia_audit.methods` and will
be loaded on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

# name -> (module, attr)
_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Likelihood baselines
    "MinKConfig": ("sae_mia_audit.methods.min_k", "MinKConfig"),
    "score_min_k": ("sae_mia_audit.methods.min_k", "score_min_k"),
    "score_min_kpp": ("sae_mia_audit.methods.min_k", "score_min_kpp"),
    "InfillingConfig": ("sae_mia_audit.methods.infilling", "InfillingConfig"),
    "score_infilling": ("sae_mia_audit.methods.infilling", "score_infilling"),
    # Activation baselines
    "NAPDDConfig": ("sae_mia_audit.methods.na_pdd", "NAPDDConfig"),
    "NAPDD": ("sae_mia_audit.methods.na_pdd", "NAPDD"),
    "ProbeConfig": ("sae_mia_audit.methods.probe", "ProbeConfig"),
    "ProbePDD": ("sae_mia_audit.methods.probe", "ProbePDD"),
    # SAE methods
    "SAEFeatureConfig": ("sae_mia_audit.methods.sae_audit", "SAEFeatureConfig"),
    "SAEFeaturePDD": ("sae_mia_audit.methods.sae_audit", "SAEFeaturePDD"),
    "SAENAPDDConfig": ("sae_mia_audit.methods.sae_na_pdd", "SAENAPDDConfig"),
    "SAENAPDD": ("sae_mia_audit.methods.sae_na_pdd", "SAENAPDD"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = _EXPORTS[name]
    mod = import_module(mod_name)
    return getattr(mod, attr)


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = list(_EXPORTS.keys())
