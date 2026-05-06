from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(add_completion=False, help="SAE-MIA-Audit command line interface.")


def _run(script: str, args: List[str]) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    # Search shared/ first, then mom/
    for subdir in ("shared", "mom"):
        candidate = repo_root / "scripts" / subdir / script
        if candidate.exists():
            cmd = [sys.executable, str(candidate)] + args
            raise SystemExit(subprocess.call(cmd))
    # Fallback to flat scripts/ (backward compat)
    script_path = repo_root / "scripts" / script
    cmd = [sys.executable, str(script_path)] + args
    raise SystemExit(subprocess.call(cmd))


@app.command()
def download(all: bool = typer.Option(False, "--all"), datasets: bool = False, models: bool = False):
    """Download datasets and/or models into the local HF cache."""
    args: List[str] = []
    if all:
        args.append("--all")
    if datasets:
        args.append("--datasets")
    if models:
        args.append("--models")
    _run("download_assets.py", args)


@app.command()
def train_sae(args: List[str] = typer.Argument(..., help="Pass-through args to scripts/shared/train_sae.py")):
    """Train sparse autoencoders on residual stream activations."""
    _run("train_sae.py", args)


@app.command()
def eval_pdd(args: List[str] = typer.Argument(..., help="Pass-through args to scripts/mom/eval_pdd.py")):
    """Evaluate PDD/MIA methods on benchmarks."""
    _run("eval_pdd.py", args)


@app.command()
def mechanistic(args: List[str] = typer.Argument(..., help="Pass-through args to scripts/mom/mechanistic_attribution.py")):
    """Run mechanistic attribution + causal ablations for top leakage features."""
    _run("mechanistic_attribution.py", args)
