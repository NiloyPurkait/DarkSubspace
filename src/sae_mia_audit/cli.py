"""Command-line entry point for the sae_mia_audit package.

Wraps ``scripts/shared/train_sae.py`` so it can be invoked via the
installed ``sae-mia-audit`` console script. Because this app currently
defines a single command, Typer collapses the subcommand and arguments
are passed through directly, e.g.::

    sae-mia-audit -- --model-path <ckpt> --layer 16 ...

Other experiment scripts are run directly (see ``scripts/dark_subspace/``
and ``scripts/shared/``) rather than through this CLI.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import typer

app = typer.Typer(add_completion=False, help="SAE-MIA-Audit command line interface.")


def _run(script: str, args: List[str]) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "scripts" / "shared" / script
    if candidate.exists():
        cmd = [sys.executable, str(candidate)] + args
        raise SystemExit(subprocess.call(cmd))
    # Fallback to flat scripts/ (backward compat)
    script_path = repo_root / "scripts" / script
    cmd = [sys.executable, str(script_path)] + args
    raise SystemExit(subprocess.call(cmd))


@app.command()
def train_sae(args: List[str] = typer.Argument(..., help="Pass-through args to scripts/shared/train_sae.py")):
    """Train sparse autoencoders on residual stream activations."""
    _run("train_sae.py", args)
