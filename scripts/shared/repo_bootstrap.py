"""repo_bootstrap.py.

Small bootstrap helper that adds the repository ``src/`` directory to
``sys.path`` so any script under ``scripts/`` can ``import sae_mia_audit.*``
without an editable install.

Used as a sys.path shim for every paper script (imported via
``scripts/dark_subspace/_bootstrap.py``).

Why this exists
---------------
Some environments (e.g., multi-node ``accelerate``, managed clusters, notebooks,
or job runners) may not preserve the expected working directory or ``PYTHONPATH``
when launching ``python scripts/<name>.py``. This helper makes the repo runnable
in those settings without requiring an editable install.

An editable install (``pip install -e .``) remains the recommended setup;
this helper removes a common footgun in environments where it is not used.

Reproduce::

    # used implicitly by every paper script via:
    from repo_bootstrap import ensure_src_on_path
    ensure_src_on_path()
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """Walk upward from this file until we find pyproject.toml."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    # Fallback: assume two levels up from scripts/shared/
    return Path(__file__).resolve().parents[2]


def ensure_src_on_path() -> Path:
    """Ensure ``<repo>/src`` is on ``sys.path``.

    Returns
    -------
    Path
        The inferred repository root.
    """

    repo_root = _find_repo_root()
    src_dir = repo_root / "src"
    if src_dir.exists():
        src_str = str(src_dir)
        if src_str not in sys.path:
            # Put it first so source checkout overrides any installed copy.
            sys.path.insert(0, src_str)
    return repo_root


__all__ = ["ensure_src_on_path"]
