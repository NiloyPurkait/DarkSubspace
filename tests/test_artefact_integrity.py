"""Integration tests for the public DarkSubspace artefact.

Each test exercises a CPU-only contract that must hold on a fresh clone:

* ``test_verifier_exits_zero`` runs ``scripts/dark_subspace/verify_claims.py``
  and asserts the asserted-check summary reports N/N pass with zero failures.
* ``test_paper_claim_jsons_parse`` walks every ``.json`` under
  ``results/dark_subspace/`` and asserts each parses as valid JSON.
* ``test_no_internal_paths_leak`` greps the shipped JSONs for the path tokens
  ``/home/u517685``, ``runs/memcirc/``, and ``<runs>/sae-mia-audit`` and
  asserts none survive.

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
    """Shipped JSONs must not leak internal absolute paths or sprint codes."""
    forbidden = ["/home/u517685", "runs/memcirc/", "<runs>/sae-mia-audit"]
    json_root = REPO_ROOT / "results" / "dark_subspace"
    leaks: list[tuple[Path, str]] = []
    for p in json_root.rglob("*.json"):
        text = p.read_text()
        for token in forbidden:
            if token in text:
                leaks.append((p, token))
    assert not leaks, "internal paths leak in shipped JSONs:\n" + "\n".join(
        f"  {p}: {token}" for p, token in leaks
    )
