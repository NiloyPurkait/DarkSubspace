"""Integration tests for the public DarkSubspace artefact.

Each test exercises a CPU-only contract that must hold on a fresh clone:

* ``test_verifier_exits_zero`` runs ``scripts/dark_subspace/verify_claims.py``
  and asserts the asserted-check summary reports N/N pass with zero failures.
* ``test_paper_claim_jsons_parse`` walks every ``.json`` under
  ``results/dark_subspace/`` and asserts each parses as valid JSON.
* ``test_no_internal_paths_leak`` scans every text file shipped in the
  artefact (Markdown, Python, JSON, shell, YAML, TOML, CITATION.cff, .bib,
  .txt, config) for cluster-style absolute paths (``/home/u<digits>``),
  legacy run-directory tokens, and unresolved path placeholders.

These tests do not exercise the GPU pipeline scripts. End-to-end re-execution
is documented in ``README.md`` ("Reproducibility caveat") and requires GPU
resources outside CI scope.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_verifier_exits_zero() -> None:
    """``verify_claims.py`` must exit 0 with N/N pass, 0 fail."""
    result = subprocess.run(
        [sys.executable, "scripts/dark_subspace/verify_claims.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"verify_claims.py exited {result.returncode}\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-1000:]}"
    )
    summary_re = re.compile(r"Asserted check summary:\s+(\d+)/(\d+)\s+pass,\s+(\d+)\s+fail",
                            re.IGNORECASE)
    matches = summary_re.findall(result.stdout)
    assert matches, "verifier did not print an Asserted check summary line"
    n_pass, n_total, n_fail = (int(x) for x in matches[-1])
    assert n_fail == 0, f"verifier reported {n_fail} failed checks"
    assert n_pass == n_total, f"verifier pass count {n_pass} != total {n_total}"


def test_paper_claim_jsons_parse() -> None:
    """Every shipped JSON under results/dark_subspace/ must be valid JSON."""
    json_root = REPO_ROOT / "results" / "dark_subspace"
    json_files = list(json_root.rglob("*.json"))
    assert json_files, f"no JSONs found under {json_root}"
    bad: list[tuple[Path, str]] = []
    for p in json_files:
        try:
            json.loads(p.read_text())
        except json.JSONDecodeError as e:
            bad.append((p, str(e)))
    assert not bad, f"unparseable JSONs:\n" + "\n".join(f"  {p}: {e}" for p, e in bad)


def test_no_internal_paths_leak() -> None:
    """Shipped text files must not leak internal absolute paths or sprint codes."""
    forbidden_patterns: list[tuple[str, re.Pattern[str]]] = [
        ("cluster home directory", re.compile(r"/home/u\d{6,}")),
        ("legacy memcirc run path", re.compile(r"runs/memcirc/")),
        ("unresolved <runs> placeholder", re.compile(r"<runs>/sae-mia-audit")),
    ]
    extensions = {
        ".md", ".py", ".json", ".sh", ".yml", ".yaml", ".toml", ".cff",
        ".bib", ".txt", ".cfg", ".ini", ".tex",
    }
    excluded_dirs = {
        ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode",
        "runs", "data", "logs", "outputs", "artifacts", "build", "dist",
    }
    self_path = Path(__file__).resolve()
    leaks: list[tuple[Path, str, str]] = []
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in excluded_dirs for part in p.relative_to(REPO_ROOT).parts):
            continue
        if p.suffix.lower() not in extensions:
            continue
        if p.resolve() == self_path:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in forbidden_patterns:
            match = pattern.search(text)
            if match:
                leaks.append((p, label, match.group(0)))
    assert not leaks, "internal paths leak in shipped files:\n" + "\n".join(
        f"  {p.relative_to(REPO_ROOT)}: {label} ({match!r})" for p, label, match in leaks
    )
