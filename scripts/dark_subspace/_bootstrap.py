"""
_bootstrap.py.

Adds ``scripts/shared/`` and ``scripts/dark_subspace/`` to ``sys.path`` and routes
every paper script through ``repo_bootstrap.ensure_src_on_path()`` so the
``sae_mia_audit`` package and inter-script imports resolve.

Used in infrastructure (imported by every paper script under
``scripts/dark_subspace/``).
Reproduce: imported indirectly. Place ``import _bootstrap  # noqa: F401`` at
the top of any paper script before project imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

_DARK_SUBSPACE_DIR = Path(__file__).resolve().parent
_SHARED_DIR = _DARK_SUBSPACE_DIR.parent / "shared"

# Shared scripts dir (figure_style, repo_bootstrap, etc.).
_shared_str = str(_SHARED_DIR)
if _shared_str not in sys.path:
    sys.path.insert(0, _shared_str)

# dark-subspace scripts dir (for inter-script imports).
_dark_subspace_str = str(_DARK_SUBSPACE_DIR)
if _dark_subspace_str not in sys.path:
    sys.path.insert(0, _dark_subspace_str)

# src/ via repo_bootstrap.
from repo_bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()
