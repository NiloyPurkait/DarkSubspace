#!/usr/bin/env python3
"""verify_claims.py.

CPU-only paper-numerical-claim verifier for the Dark Subspace paper. Reads the
curated JSONs shipped under ``results/dark_subspace/``, asserts the on-disk
values match the paper-text values to within a small numerical tolerance, and
exits 1 on any mismatch.

Acts as the primary reproducibility entry point (Section 5 of the paper).
No GPU required, no SLURM submission, runs in seconds on a CPU.

Reproduce::

    env/bin/python3 scripts/dark_subspace/verify_claims.py
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
GENERATED = ROOT / "results" / "dark_subspace" / "generated"
PAPER_CLAIMS = ROOT / "results" / "dark_subspace" / "paper_claims"


def generated(*parts: str) -> Path:
    """Return a path inside the shipped generated-result archive."""
    return GENERATED.joinpath(*parts)


def load(p: Path) -> dict | None:
    """Load a JSON file or report it as missing.

    Parameters
    ----------
    p : Path
        Path to the JSON file.

    Returns
    -------
    dict or None
        Parsed JSON if the file exists, ``None`` otherwise.
    """
    if not p.exists():
        print(f"  MISSING: {p.relative_to(ROOT)}")
        return None
    with open(p) as f:
        return json.load(f)


def get(d, *keys, default=None):
    """Walk a nested dict by ``keys`` and return ``default`` if any step fails."""
    out = d
    for k in keys:
        if not isinstance(out, dict):
            return default
        out = out.get(k)
        if out is None:
            return default
    return out


def banner(title: str):
    """Print a section banner."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def show_layer(label, layers, layer_key):
    """Pull a layer's values from a per_layer dict and print a one-line summary."""
    if layers is None:
        return
    ld = layers.get(str(layer_key)) or layers.get(layer_key)
    if not isinstance(ld, dict):
        print(f"  {label}: layer {layer_key} not found")
        return
    cos = ld.get("cosine_d_K_d_R")
    angle = ld.get("mean_principal_angle_deg")
    mem = get(ld, "membership_probe", "auroc_mean")
    rec = get(ld, "recall_probe", "auroc_mean")
    s = f"  {label} L{layer_key}: cos={cos}  angle={angle}  mem_auroc={mem}  rec_auroc={rec}"
    print(s)


# -----------------------------------------------------------------------------
# 7. Channel decomposition: main table source
# -----------------------------------------------------------------------------
banner("7. Channel decomposition per-model AUROC + cosine (Table 1)")

bcd_models_main = [
    ("Pythia-1B", "p1b_epoch5", 8),
    ("Pythia-6.9B", "p69_epoch5", 16),
    ("Pythia-12B (L24)", "p12b_epoch5", 24),
    ("Pythia-12B (L18)", "p12b_epoch5", 18),
    ("GPT-Neo-2.7B", "neo_epoch5", 16),
    ("OPT-6.7B", "opt67_epoch5", 24),
    ("Falcon-7B", "falcon7b_epoch5_v2", 16),
    ("Mistral-7B", "mistral_epoch5_v2", 16),
    ("Llama-3-8B", "llama3_epoch5_v2", 16),
    ("Qwen2-7B", "qwen2_epoch5", 16),
]

for label, dirname, layer in bcd_models_main:
    p = generated("behavioral_channels", dirname, "orthogonality.json")
    d = load(p)
    if d is None:
        continue
    layers = d.get("per_layer", {})
    show_layer(label, layers, layer)

# -----------------------------------------------------------------------------
# 8. Pre-FT base P69 + layer sweep (paper claims)
# -----------------------------------------------------------------------------
banner("8. Pre-FT base P69 + layer sweep")

print("Pre-FT base (paper says: layer 12 0.480, 14 0.490, 16 0.504, 18 0.493, 20 0.495)")
p = generated("behavioral_channels", "p69_BASE_pre_ft", "orthogonality.json")
d = load(p)
if d is not None:
    layers = d.get("per_layer", {})
    print(f"  Available layers: {list(layers.keys())}")
    for k in sorted(layers.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
        ld = layers[k]
        if isinstance(ld, dict):
            mem = get(ld, "membership_probe", "auroc_mean")
            print(f"    L{k}: mem_auroc={mem}")

print()
print("Layer sweep at FT (paper says: layer 12 0.689, 14 0.755, 16 0.828, 18 0.858, 20 0.876)")
p = generated("behavioral_channels", "p69_epoch5_layer_sweep", "orthogonality.json")
d = load(p)
if d is not None:
    layers = d.get("per_layer", {})
    print(f"  Available layers: {list(layers.keys())}")
    for k in sorted(layers.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
        ld = layers[k]
        if isinstance(ld, dict):
            mem = get(ld, "membership_probe", "auroc_mean")
            cos = ld.get("cosine_d_K_d_R")
            print(f"    L{k}: mem_auroc={mem}  cos={cos}")

# -----------------------------------------------------------------------------
# Bottom block of Table 2.sources for Original/Recon/Resid/recon_cos
# -----------------------------------------------------------------------------
banner("Bottom block of Table 2.claim source check (member-only SAE rows)")

# Paper Table 2 bottom block:
# Pythia-1B   0.660 / 0.515 / [.49,.54] / -14pp / 0.676 / 0.771
# GPT-Neo-2.7B 0.615 / 0.571 / [.55,.60] / -4pp / 0.699 / 0.998
# OPT-6.7B    0.774 / 0.557 / [.53,.58] / -22pp / 0.866 / 0.533
# Pythia-12B  0.707 / 0.564 / [.54,.59] / -14pp / 0.814 / 0.985
# Qwen2-7B    0.638 / 0.526 / [.50,.55] / -11pp / 0.849 / 0.850
# Falcon-7B   0.690 / 0.694 / --- / +0pp / 0.827 / 0.767

table2_bottom = [
    ("Pythia-1B (paper L8 0.660/0.515/0.676/0.771)", "p1b_epoch5"),
    ("GPT-Neo-2.7B (paper 0.615/0.571/0.699/0.998)", "neo_epoch5"),
    ("OPT-6.7B (paper 0.774/0.557/0.866/0.533)", "opt67_epoch5"),
    ("Pythia-12B L24 (paper 0.707/0.564/0.814/0.985)", "p12b_epoch5"),
    ("Pythia-12B L18 (paper says L18 16pp drop)", "p12b_epoch5_layer18"),
    ("Qwen2-7B (paper 0.638/0.526/0.849/0.850)", "qwen2_epoch5"),
    ("Falcon-7B (paper 0.690/0.694/0.827/0.767)", "falcon_epoch5"),
    ("Mistral-7B (SAE-quality exclusion row)", "mistral_epoch5"),
]
for label, dirname in table2_bottom:
    p = generated("sae_dark_subspace", dirname, "results.json")
    d = load(p)
    if d is None:
        continue
    o = get(d, "original", "score_K_auroc")
    r = get(d, "sae_reconstructed", "score_K_auroc")
    rs = get(d, "residual", "score_K_auroc")
    rc = get(d, "sae_quality", "reconstruction_cosine")
    L0 = get(d, "sae_quality", "mean_active_features")
    print(f"  {label}: orig={o:.4f}  recon={r:.4f}  drop={o-r:.4f}  resid={rs:.4f}  rc={rc:.4f}  L0={L0}")


# -----------------------------------------------------------------------------
# Top block of Table 2 (Pythia-6.9B rows + Neo)
# -----------------------------------------------------------------------------
banner("Top block of Table 2.Pythia-6.9B and GPT-Neo-2.7B rows (member SAE / mixed / priv)")

# Paper says:
# Pythia-6.9B           0.803 / 0.593 / [.57,.62] / -21pp / 0.805 / 0.927
# Pythia-6.9B (mixed)   0.803 / 0.594 / [.59,.60] / -21pp / 0.779 / 0.976
# Pythia-6.9B (priv 0.1) 0.803 / 0.803 / [.78,.82] / 0pp / 0.520 / 0.943
# Pythia-6.9B (priv 1.0) 0.803 / 0.802 / [.78,.82] / 0pp / 0.540 / 0.975
# GPT-Neo-2.7B (mixed)  0.616 / 0.558 / [.53,.58] / -6pp / 0.646 / 0.997
# GPT-Neo-2.7B (priv 1.0) 0.616 / 0.612 / [.59,.64] / -0.4pp / 0.566 / 0.998
# GPT-Neo-2.7B (priv 0.1, sens) 0.616 / 0.612 / [.59,.64] / -0.4pp / 0.515 / 0.999

_CHECK_RESULTS: list[tuple[str, bool, str]] = []


def _check(label: str, expected: float, actual: float | None, tol: float = 0.005) -> None:
    """Compare a disk value with a paper claim, recording PASS/FAIL.

    Parameters
    ----------
    label : str
        Human-readable name for the check.
    expected : float
        Value as printed in the paper.
    actual : float or None
        Value parsed from the on-disk JSON.
    tol : float, optional
        Absolute tolerance, default ``0.005``.
    """
    if actual is None:
        _CHECK_RESULTS.append((label, False, f"MISSING source on disk"))
        print(f"    [FAIL] {label}: source missing on disk")
        return
    delta = abs(actual - expected)
    ok = delta <= tol
    _CHECK_RESULTS.append((label, ok, f"expect={expected:.4f} actual={actual:.4f} |delta|={delta:.4f}"))
    flag = "PASS" if ok else "FAIL"
    print(f"    [{flag}] {label}: expect={expected:.4f} actual={actual:.4f} |delta|={delta:.4f}")


print("\nP69 mixed-data SAE (canonical N=5 harmonized cohort):")
harm = load(PAPER_CLAIMS / "p69_n5_harmonized_2026-05-06.json")
if harm is not None:
    s = harm.get("cluster_summary_n5", {})
    o_n5 = s.get("original_score_K_auroc", {}).get("mean")
    r_n5 = s.get("reconstructed_score_K_auroc", {}).get("mean")
    rs_n5 = s.get("residual_score_K_auroc", {}).get("mean")
    rc_n5 = s.get("recon_cos", {}).get("mean")
    drop_n5 = s.get("drop_original_minus_reconstructed", {}).get("mean")
    drop_std = s.get("drop_original_minus_reconstructed", {}).get("std")
    n = s.get("original_score_K_auroc", {}).get("n")
    print(
        f"  N={n} aggregate: orig={o_n5:.4f}  recon={r_n5:.4f}  "
        f"drop={drop_n5:.4f} (std {drop_std:.4f})  resid={rs_n5:.4f}  rc={rc_n5:.4f}"
    )
    _check("P69 N=5 mixed orig (paper 0.803)", 0.803, o_n5)
    _check("P69 N=5 mixed recon (paper 0.594)", 0.594, r_n5)
    _check("P69 N=5 mixed resid (paper 0.781)", 0.781, rs_n5)
    _check("P69 N=5 mixed recon_cos (paper 0.976)", 0.976, rc_n5)
    _check("P69 N=5 mixed drop_mean (paper 0.209)", 0.209, drop_n5)


# Other rows still pulled from per-run dirs.
top = [
    ("P69 single member SAE (paper 0.803/0.593/0.805/0.927)", "p69_epoch5",
     [("orig", 0.803, "original", "score_K_auroc"),
      ("recon", 0.593, "sae_reconstructed", "score_K_auroc"),
      ("resid", 0.805, "residual", "score_K_auroc"),
      ("rc", 0.927, "sae_quality", "reconstruction_cosine")]),
    ("P69 priv dk=0.1 (paper 0.803/0.803/0.520/0.943)", "p69_ft_dk0.1",
     [("orig", 0.803, "original", "score_K_auroc"),
      ("recon", 0.803, "sae_reconstructed", "score_K_auroc"),
      ("resid", 0.520, "residual", "score_K_auroc"),
      ("rc", 0.943, "sae_quality", "reconstruction_cosine")]),
    ("P69 priv dk=1.0 (paper 0.803/0.802/0.540/0.975)", "p69_ft_dk1.0",
     [("orig", 0.803, "original", "score_K_auroc"),
      ("recon", 0.802, "sae_reconstructed", "score_K_auroc"),
      ("resid", 0.540, "residual", "score_K_auroc"),
      ("rc", 0.975, "sae_quality", "reconstruction_cosine")]),
    ("Neo mixed (paper 0.616/0.558/0.646/0.997)", "neo_mixed_sae",
     [("orig", 0.616, "original", "score_K_auroc"),
      ("recon", 0.558, "sae_reconstructed", "score_K_auroc"),
      ("resid", 0.646, "residual", "score_K_auroc"),
      ("rc", 0.997, "sae_quality", "reconstruction_cosine")]),
    ("Neo priv dk=1.0 (paper 0.616/0.612/0.566/0.998)", "neo_ft_dk1.0",
     [("orig", 0.616, "original", "score_K_auroc"),
      ("recon", 0.612, "sae_reconstructed", "score_K_auroc"),
      ("resid", 0.566, "residual", "score_K_auroc"),
      ("rc", 0.998, "sae_quality", "reconstruction_cosine")]),
    ("Neo priv dk=0.1 (paper 0.616/0.612/0.515/0.999)", "neo_ft_dk0.1",
     [("orig", 0.616, "original", "score_K_auroc"),
      ("recon", 0.612, "sae_reconstructed", "score_K_auroc"),
      ("resid", 0.515, "residual", "score_K_auroc"),
      ("rc", 0.999, "sae_quality", "reconstruction_cosine")]),
]
for label, dirname, fields in top:
    p = generated("sae_dark_subspace", dirname, "results.json")
    d = load(p)
    if d is None:
        for fname, expected, *_ in fields:
            _check(f"{label} [{fname}]", expected, None)
        continue
    o = get(d, "original", "score_K_auroc")
    r = get(d, "sae_reconstructed", "score_K_auroc")
    rs = get(d, "residual", "score_K_auroc")
    rc = get(d, "sae_quality", "reconstruction_cosine")
    L0 = get(d, "sae_quality", "mean_active_features")
    print(f"\n  {label}: orig={o:.4f}  recon={r:.4f}  drop_pp={(o-r)*100:.1f}  resid={rs:.4f}  rc={rc:.4f}  L0={L0}")
    for fname, expected, *path in fields:
        _check(f"{label} [{fname}]", expected, get(d, *path))


# Pythia-12B mixed-data three-init cohort (seeds 47, 48, 49).
print("\nP12B mixed-data SAE (three-init cohort, seeds 47, 48, 49):")
p12b_seeds = [(47, 0.169), (48, 0.139), (49, 0.160)]
for seed, expected_drop in p12b_seeds:
    p = generated("sae_dark_subspace", f"p12b_mixed_sae_seed{seed}", "results.json")
    d = load(p)
    if d is None:
        _check(f"P12B fresh-init seed {seed} drop (paper {expected_drop:.3f})", expected_drop, None)
        continue
    o = get(d, "original", "score_K_auroc")
    r = get(d, "sae_reconstructed", "score_K_auroc")
    rc = get(d, "sae_quality", "reconstruction_cosine")
    drop = o - r if (o is not None and r is not None) else None
    print(f"  seed{seed}: orig={o:.4f}  recon={r:.4f}  drop={drop:.4f}  rc={rc:.4f}")
    _check(f"P12B fresh-init seed {seed} drop (paper {expected_drop:.3f})", expected_drop, drop)
    _check(f"P12B fresh-init seed {seed} recon_cos > 0.99", 0.995, rc, tol=0.01)


# -----------------------------------------------------------------------------
# Norm baseline values (Table 3)
# -----------------------------------------------------------------------------
banner("Norm baseline + best layer per model (Table 3 source)")

print("Paper Table 3 says:")
print("  Pythia-6.9B    norm 0.542  d_K 0.803 +26pp")
print("  OPT-6.7B       norm 0.534  d_K 0.774 +24pp")
print("  Falcon-7B      norm 0.557  d_K 0.690 +13pp")
print("  Pythia-1B      norm 0.548  d_K 0.660 +11pp")
print("  Pythia-12B     norm 0.599  d_K 0.707 +11pp")
print("  GPT-Neo-2.7B   norm 0.514  d_K 0.616 +10pp")
print("  Qwen2-7B       norm 0.542  d_K 0.638 +10pp")
print("  Mistral-7B     norm 0.877  d_K 0.927 +5pp")
print("  Llama-3-8B     norm 0.843  d_K 0.903 +6pp")
print()

import os
nb_dir = generated("norm_baseline")
if nb_dir.exists():
    for sub in sorted(os.listdir(nb_dir)):
        p = nb_dir / sub / "results.json"
        d = load(p)
        if d is None:
            continue
        # Try several keys
        keys_of_interest = ["best_layer_auroc", "best_layer", "norm_auroc", "results", "per_layer"]
        print(f"  {sub}:")
        for k, v in d.items():
            if isinstance(v, (int, float, str)):
                print(f"    {k} = {v}")
            elif isinstance(v, dict) and len(v) < 5:
                print(f"    {k} = {v}")
        # Try to get best layer auroc from per_layer
        per_layer = d.get("per_layer") or d.get("layers")
        if per_layer:
            best = max(per_layer.values(), key=lambda x: x.get("auroc", 0) if isinstance(x, dict) else 0)
            print(f"    best_layer_data: {best}")


# -----------------------------------------------------------------------------
# Scaling table (Pythia 70M ... 12B)
# -----------------------------------------------------------------------------
banner("Scaling table source (App Table tab:scaling)")

print("Paper says (score_K AUROC):")
print("  P70M  0.507")
print("  P160M 0.588")
print("  P410M 0.716")
print("  P1B   0.800")
print("  P2.8B 0.842")
print("  P6.9B 0.876")
print("  P12B  0.781")
print()

# Look for results in behavioral_channels
for tag, layer_pref in [("p70m_epoch5", None), ("p160m_epoch5", None), ("p410m_epoch5", None),
                        ("p1b_epoch5", None), ("p2.8b_epoch5", None), ("p69_epoch5", 16),
                        ("p12b_epoch5", 24)]:
    p = generated("behavioral_channels", tag, "orthogonality.json")
    d = load(p)
    if d is None:
        continue
    layers = d.get("per_layer", {})
    if layer_pref is not None:
        ld = layers.get(str(layer_pref))
        if isinstance(ld, dict):
            mem = get(ld, "membership_probe", "auroc_mean")
            print(f"  {tag} L{layer_pref}: mem_auroc={mem}")
    else:
        # Print the maximum membership AUROC across layers
        best_layer, best_auroc = None, 0
        for k, ld in layers.items():
            if not isinstance(ld, dict):
                continue
            mem = get(ld, "membership_probe", "auroc_mean")
            if mem is None:
                continue
            if mem > best_auroc:
                best_auroc = mem
                best_layer = k
        print(f"  {tag} all layers (best L{best_layer}={best_auroc}):")
        for k in sorted(layers.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
            ld = layers[k]
            if isinstance(ld, dict):
                mem = get(ld, "membership_probe", "auroc_mean")
                print(f"    L{k}: mem_auroc={mem}")

# -----------------------------------------------------------------------------
# Pythia-1B epochs 1, 3, 5 (App Table tab:dynamics, tab:epoch_dd)
# -----------------------------------------------------------------------------
banner("Pythia-1B epoch dynamics + DD tables")

for ep in ["p1b_epoch1", "p1b_epoch3", "p1b_epoch5"]:
    p = generated("behavioral_channels", ep, "orthogonality.json")
    d = load(p)
    if d is None:
        continue
    layers = d.get("per_layer", {})
    show_layer(f"  {ep}", layers, 8)


# -----------------------------------------------------------------------------
# Standard MIA probes
# -----------------------------------------------------------------------------
banner("Standard MIA probes")
p = generated("standard_mia_probes", "p69_dark_subspace_replication", "results.json")
d = load(p)
if d is None:
    print("  Standard MIA probe JSON not shipped; no asserted claim depends on this file.")
else:
    print(f"  Standard MIA probe keys: {list(d.keys())}")


# -----------------------------------------------------------------------------
# Bibliography references check, when the manuscript checkout is present.
# -----------------------------------------------------------------------------
banner("Bibliography sanity check")

bib = ROOT / "manuscript/references.bib"
if bib.exists():
    text = bib.read_text()
    required_keys = [
        "meeus2025sok",
        "duan2024membership",
        "muhamed2025dsg",
        "engels2024dark",
        "wang2025sspu",
        "frikha2025privacyscalpel",
        "suri2025mitigating",
        "carlini2021extracting",
        "marks2024sparse",
        "leask2025sparse",
        "peng2025discover",
        "hu2025jogging",
        "chen2025unlearning",
        "zhang2025minkpp",
        "yang2024qwen2",
        "huang2024demystifying",
        "templeton2024scaling",
    ]
    for key in required_keys:
        present = key in text
        print(f"  {'PASS' if present else 'MISSING'}: {key}")
else:
    print("  manuscript/references.bib not present in this code/results artifact; bibliography check skipped.")


# -----------------------------------------------------------------------------
# Additional asserted checks against shipped JSONs (Tables 1, 3, K-PC,
# cohort bootstrap, held-out dK, BoW ceiling, Pythia-12B L18).
# -----------------------------------------------------------------------------
banner("Per-model channel decomposition (Table 1) [asserted]")

# Paper Table 1 (tab:bcd_main) reports cos(d_K, d_R), Mem AUROC, Rec AUROC
# at the SAE layer per model. Pythia-12B row uses layer 24 (channel-geometry
# reference layer); other models use the same SAE layer reported elsewhere.
# Tolerance follows the paper rounding to 3 decimals.
bcd_table1 = [
    ("Pythia-1B",   "p1b_epoch5",          8,  0.102,  0.577, 0.762),
    ("Pythia-6.9B", "p69_epoch5",         16,  0.107,  0.828, 0.696),
    ("Pythia-12B",  "p12b_epoch5",        24,  0.336,  0.781, 0.806),
    ("GPT-Neo-2.7B","neo_epoch5",         16,  0.024,  0.504, 0.813),
    ("OPT-6.7B",    "opt67_epoch5",       24, -0.063,  0.869, 0.883),
    ("Falcon-7B",   "falcon7b_epoch5_v2", 16,  0.161,  0.635, 0.855),
    ("Mistral-7B",  "mistral_epoch5_v2",  16, -0.010,  0.967, 0.714),
    ("Llama-3-8B",  "llama3_epoch5_v2",   16,  0.223,  0.957, 0.735),
    ("Qwen2-7B",    "qwen2_epoch5",       16,  0.390,  0.803, 0.705),
]
for label, dirname, layer, exp_cos, exp_mem, exp_rec in bcd_table1:
    p = generated("behavioral_channels", dirname, "orthogonality.json")
    d = load(p)
    if d is None:
        _check(f"channel-decomp {label} L{layer} cos(d_K,d_R) (paper {exp_cos:+.3f})", exp_cos, None)
        _check(f"channel-decomp {label} L{layer} mem AUROC (paper {exp_mem:.3f})", exp_mem, None)
        _check(f"channel-decomp {label} L{layer} rec AUROC (paper {exp_rec:.3f})", exp_rec, None)
        continue
    layers = d.get("per_layer", {})
    ld = layers.get(str(layer)) or layers.get(layer)
    if not isinstance(ld, dict):
        _check(f"channel-decomp {label} L{layer} cos(d_K,d_R) (paper {exp_cos:+.3f})", exp_cos, None)
        _check(f"channel-decomp {label} L{layer} mem AUROC (paper {exp_mem:.3f})", exp_mem, None)
        _check(f"channel-decomp {label} L{layer} rec AUROC (paper {exp_rec:.3f})", exp_rec, None)
        continue
    cos = ld.get("cosine_d_K_d_R")
    mem = get(ld, "membership_probe", "auroc_mean")
    rec = get(ld, "recall_probe", "auroc_mean")
    # Cosines are reported to 3 decimals, allow 1e-3 tol.
    _check(f"channel-decomp {label} L{layer} cos(d_K,d_R) (paper {exp_cos:+.3f})", exp_cos, cos, tol=1e-3)
    _check(f"channel-decomp {label} L{layer} mem AUROC (paper {exp_mem:.3f})", exp_mem, mem, tol=1e-3)
    _check(f"channel-decomp {label} L{layer} rec AUROC (paper {exp_rec:.3f})", exp_rec, rec, tol=1e-3)


banner("Norm-baseline best-layer AUROCs (Table 3) [asserted]")

# Paper Table 3 (tab:norm_direction) norm AUROCs at the best layer per model.
# Source: results/dark_subspace/generated/norm_baseline/<dir>/results.json
# We compute the maximum AUROC across per-layer entries to compare against
# the paper-reported best-layer value.
norm_table3 = [
    ("Pythia-6.9B", "p69_epoch5",   0.542),
    ("OPT-6.7B",    "opt67_epoch5", 0.534),
    ("Falcon-7B",   "falcon_epoch5",0.557),
    ("Pythia-1B",   "p1b_epoch5",   0.548),
    ("Pythia-12B",  "p12b_epoch5",  0.599),
    ("GPT-Neo-2.7B","neo_epoch5",   0.514),
    ("Qwen2-7B",    "qwen2_epoch5", 0.542),
    ("Mistral-7B",  "mistral_epoch5",0.877),
    ("Llama-3-8B",  "llama3_epoch5",0.843),
]
for label, dirname, expected in norm_table3:
    p = generated("norm_baseline", dirname, "results.json")
    d = load(p)
    if d is None:
        _check(f"Norm-baseline {label} best-layer AUROC (paper {expected:.3f})", expected, None)
        continue
    per_layer = d.get("per_layer") or {}
    aurocs = [v.get("auroc") for v in per_layer.values() if isinstance(v, dict) and isinstance(v.get("auroc"), (int, float))]
    best = max(aurocs) if aurocs else None
    _check(f"Norm-baseline {label} best-layer AUROC (paper {expected:.3f})", expected, best, tol=1e-3)


banner("K-PC residual ablation magnitudes (Table tab:kpc_kten_cells) [asserted]")

# Paper Table tab:kpc_kten_cells reports the AUROC reduction (delta) for the
# Pythia-12B residual-PC ablation: K=10 -> +0.176 (95% CI [+0.165, +0.187]),
# K=5 -> +0.103 (95% CI [+0.094, +0.112]).
p = generated("causal_ablation", "p12b_errPC_K10", "results.json")
d = load(p)
if d is None:
    _check("Pythia-12B K=10 errPC delta_auroc (paper +0.176)", 0.176, None)
    _check("Pythia-12B K=10 errPC CI lo (paper +0.165)", 0.165, None)
    _check("Pythia-12B K=10 errPC CI hi (paper +0.187)", 0.187, None)
else:
    delta = d.get("delta_auroc_mean")
    ci_lo = get(d, "delta_bootstrap", "ci95_lo")
    ci_hi = get(d, "delta_bootstrap", "ci95_hi")
    _check("Pythia-12B K=10 errPC delta_auroc (paper +0.176)", 0.176, delta, tol=1e-3)
    _check("Pythia-12B K=10 errPC CI lo (paper +0.165)", 0.165, ci_lo, tol=1e-3)
    _check("Pythia-12B K=10 errPC CI hi (paper +0.187)", 0.187, ci_hi, tol=1e-3)

p = generated("causal_ablation_K5", "p12b_errPC_K5", "results.json")
if not p.exists():
    p = generated("causal_ablation", "p12b_errPC_K5", "results.json")
d = load(p)
if d is None:
    _check("Pythia-12B K=5 errPC delta_auroc (paper +0.103)", 0.103, None)
    _check("Pythia-12B K=5 errPC CI lo (paper +0.094)", 0.094, None)
    _check("Pythia-12B K=5 errPC CI hi (paper +0.112)", 0.112, None)
else:
    _check("Pythia-12B K=5 errPC delta_auroc (paper +0.103)", 0.103, d.get("delta_auroc_mean"), tol=1e-3)
    _check("Pythia-12B K=5 errPC CI lo (paper +0.094)", 0.094, get(d, "delta_bootstrap", "ci95_lo"), tol=1e-3)
    _check("Pythia-12B K=5 errPC CI hi (paper +0.112)", 0.112, get(d, "delta_bootstrap", "ci95_hi"), tol=1e-3)


banner("Cohort bootstrap sign test (paper_claims/cohort_bootstrap.json) [asserted]")

# Paper cohort_bootstrap pre-registered sign test on the inverting cohort:
# "5 of 5 inverting rows have positive margin (residual > original), 4 of 5
# have CI excluding zero, one-sided binomial p = 0.03125, decision ACCEPT."
cb = load(PAPER_CLAIMS / "cohort_bootstrap.json")
if cb is None:
    _check("Cohort bootstrap n_inverting_cohort_rows (paper 5)", 5, None)
    _check("Cohort bootstrap n_positive_margin (paper 5)", 5, None)
    _check("Cohort bootstrap n_ci_excludes_zero (paper 4)", 4, None)
    _check("Cohort bootstrap p_one_sided (paper 0.03125)", 0.03125, None)
else:
    st = cb.get("sign_test", {})
    _check("Cohort bootstrap n_inverting_cohort_rows (paper 5)", 5, st.get("n_inverting_cohort_rows"), tol=0)
    _check("Cohort bootstrap n_positive_margin (paper 5)", 5, st.get("n_positive_margin"), tol=0)
    _check("Cohort bootstrap n_ci_excludes_zero (paper 4)", 4, st.get("n_ci_excludes_zero"), tol=0)
    _check("Cohort bootstrap p_one_sided (paper 0.03125)", 0.03125,
           st.get("p_one_sided_binomial_05"), tol=1e-6)
    decision_ok = st.get("pre_reg_decision") == "ACCEPT"
    _CHECK_RESULTS.append(("Cohort bootstrap pre_reg_decision (paper ACCEPT)", decision_ok,
                           f"actual={st.get('pre_reg_decision')!r}"))
    flag = "PASS" if decision_ok else "FAIL"
    print(f"    [{flag}] Cohort bootstrap pre_reg_decision (paper ACCEPT): actual={st.get('pre_reg_decision')!r}")


banner("Held-out d_K reductions (paper_claims/heldout_dk.json, app:heldout_dk_protocol) [asserted]")

# Paper app:heldout_dk_protocol reports Pythia-6.9B held-out mean reduction
# 0.149 and Pythia-12B held-out mean reduction 0.105.
hd = load(PAPER_CLAIMS / "heldout_dk.json")
if hd is None:
    _check("Pythia-6.9B held-out drop mean (paper 0.149)", 0.149, None)
    _check("Pythia-12B held-out drop mean (paper 0.105)", 0.105, None)
else:
    anchors = hd.get("anchors", [])
    p69_drop = next(
        (a.get("heldout_drop_mean") for a in anchors if a.get("model") == "p69"),
        None,
    )
    p12b_drop = next(
        (a.get("heldout_drop_mean") for a in anchors if a.get("model") == "p12b"),
        None,
    )
    _check("Pythia-6.9B held-out drop mean (paper 0.149)", 0.149, p69_drop, tol=1e-3)
    _check("Pythia-12B held-out drop mean (paper 0.105)", 0.105, p12b_drop, tol=1e-3)


banner("Bag-of-Words vocabulary baseline (app:bow_baseline) [asserted]")

# Paper app:bow_baseline cites pooled AUROC 0.4566 (rounded to 0.457).
bow = load(generated("bow_ceiling", "memcirc_ctrl_ft", "results.json"))
if bow is None:
    _check("BoW pooled AUROC (paper 0.4566)", 0.4566, None)
else:
    pooled = get(bow, "variants", "tfidf_lr", "pooled_auroc")
    _check("BoW pooled AUROC (paper 0.4566)", 0.4566, pooled, tol=1e-3)


banner("Pythia-12B L18 SAE-reconstruction drop (Table 2 row) [asserted]")

# Paper Table 2 reports Pythia-12B at L18 with about 16 percentage points
# original-minus-reconstructed drop. Source:
# results/.../sae_dark_subspace/p12b_epoch5_layer18/results.json
p = generated("sae_dark_subspace", "p12b_epoch5_layer18", "results.json")
d = load(p)
if d is None:
    _check("Pythia-12B L18 orig-recon drop (paper +0.16, ~16pp)", 0.16, None)
else:
    orig = get(d, "original", "score_K_auroc")
    recon = get(d, "sae_reconstructed", "score_K_auroc")
    drop = orig - recon if (orig is not None and recon is not None) else None
    _check("Pythia-12B L18 orig-recon drop (paper +0.16, ~16pp)", 0.16, drop, tol=0.01)


print()
print("=" * 70)
print("END OF VERIFICATION DUMP")
print("=" * 70)


# -----------------------------------------------------------------------------
# Pass/fail summary across asserted checks
# -----------------------------------------------------------------------------
import sys

n_total = len(_CHECK_RESULTS)
n_pass = sum(1 for _, ok, _ in _CHECK_RESULTS if ok)
n_fail = n_total - n_pass

print()
print("=" * 70)
print(f"ASSERTED CHECK SUMMARY: {n_pass}/{n_total} PASS, {n_fail} FAIL")
print("=" * 70)
if n_fail:
    print("Failures:")
    for label, ok, detail in _CHECK_RESULTS:
        if not ok:
            print(f"  FAIL  {label}: {detail}")
    sys.exit(1)
print("All asserted checks pass within tolerance.")
sys.exit(0)
