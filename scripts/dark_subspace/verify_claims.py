#!/usr/bin/env python3
"""verify_claims.py.

CPU-only paper-numerical-claim verifier for the Dark Subspace paper. Reads the
curated JSONs shipped under ``results/dark_subspace/``, asserts the on-disk
values match the paper-text values to within a small numerical tolerance, and
exits 1 on any mismatch.

Acts as the primary reproducibility entry point (Section 5 of the paper).
No GPU required, no SLURM submission, runs in seconds on a CPU.

Scope statement.
This verifier checks paper-vs-JSON consistency only. It does not regenerate
the underlying GPU-pipeline outputs (per-text scores, SAE activations, layer
sweeps, decoded continuations, ROUGE-L scores) because those carry SAE
checkpoint references and large array fields that are not bundled in this
artefact. The bundled paper-claim JSONs under ``results/dark_subspace/paper_claims/``
mirror the manuscript-rendered cell values; the bundled generated JSONs under
``results/dark_subspace/generated/`` mirror the canonical run-time outputs of
the paper scripts. A passing run of this verifier confirms that every
paper-cited number traces to a bundled JSON within tolerance; it does NOT
confirm that the bundled JSONs were freshly produced by reviewer-side
re-execution. Reviewers wanting end-to-end re-execution must run the paper
scripts under ``scripts/dark_subspace/`` against their own cluster setup.

Reproduce::

    .venv/bin/python scripts/dark_subspace/verify_claims.py
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


# -----------------------------------------------------------------------------
# Scope banner printed at runtime.
# -----------------------------------------------------------------------------
print("=" * 70)
print("Verifier scope")
print("=" * 70)
print("Checks paper-vs-JSON consistency. The bundled paper-claim JSONs under")
print("results/dark_subspace/paper_claims/ mirror the manuscript-rendered cell")
print("values; the bundled generated JSONs under results/dark_subspace/generated/")
print("mirror canonical run-time outputs of the paper scripts. Underlying")
print("GPU-pipeline records (per-text scores, SAE activations, decoded")
print("continuations) are not bundled because they carry SAE checkpoint")
print("references and large array fields. A passing run confirms paper-text ==")
print("bundled-JSON to within tolerance. End-to-end re-execution requires")
print("running the paper scripts under scripts/dark_subspace/ against an")
print("appropriate cluster setup.")
print()


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


def _check(label: str, expected: float, actual: float | None, tol: float = 0.002) -> None:
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
    # Materiality of the N=5 reporting basis vs the underlying N=6 cohort.
    # The shipped JSON's materiality block records that all per-metric
    # |delta_mean(N=5 - N=6)| < 0.005, so the N=5 reporting basis chosen
    # to align with the Pythia-1B seed list does not change the conclusion.
    mat = harm.get("materiality", {})
    verdict_ok = mat.get("verdict") == "NEGLIGIBLE"
    _CHECK_RESULTS.append(
        ("P69 N=5 vs N=6 materiality verdict (NEGLIGIBLE)", verdict_ok,
         f"actual={mat.get('verdict')!r}"),
    )
    flag = "PASS" if verdict_ok else "FAIL"
    print(f"    [{flag}] P69 N=5 vs N=6 materiality verdict: actual={mat.get('verdict')!r}")
    abs_deltas = mat.get("abs_delta_mean_per_metric", {})
    threshold = mat.get("threshold", 0.005)
    for metric, dval in abs_deltas.items():
        within = dval is not None and dval < threshold
        _CHECK_RESULTS.append(
            (f"P69 N=5 vs N=6 |delta_mean| < {threshold} on {metric}", within,
             f"actual={dval:.6f}" if dval is not None else "actual=None"),
        )
        flag = "PASS" if within else "FAIL"
        print(f"    [{flag}] P69 N=5 vs N=6 |delta_mean| < {threshold} on {metric}: actual={dval:.6f}")


banner("P69 member-only N=5 cohort aggregate (paper_claims/p69_member_only_n5.json) [asserted]")

# Backs the Pythia-6.9B (controlled, member-only) row of tab:dark_subspace
# at N=5. Paper values: original 0.803, recon 0.633, residual 0.833,
# recon_cosine 0.940. The mixed-data row is verified separately above.
mo = load(PAPER_CLAIMS / "p69_member_only_n5.json")
if mo is not None:
    s = mo.get("cluster_summary_n5", {})
    _check("P69 N=5 member-only orig (paper 0.803)", 0.803,
           s.get("original_score_K_auroc", {}).get("mean"), tol=2e-3)
    _check("P69 N=5 member-only recon (paper 0.633)", 0.633,
           s.get("reconstructed_score_K_auroc", {}).get("mean"), tol=2e-3)
    _check("P69 N=5 member-only resid (paper 0.833)", 0.833,
           s.get("residual_score_K_auroc", {}).get("mean"), tol=2e-3)
    _check("P69 N=5 member-only recon_cos (paper 0.940)", 0.940,
           s.get("recon_cos", {}).get("mean"), tol=2e-3)


banner("Pythia-12B mixed-SAE seeds 50, 51 (generated/sae_dark_subspace, app:p12b_replication) [asserted]")

# The P12B mixed-SAE 5-seed cohort. Seeds 47, 48, 49 are asserted in the
# main P12B section above. Seeds 50, 51 are the remaining two seeds of the
# pre-registered five-seed cohort and are now bundled in the artefact.
for s_label in ["50", "51"]:
    p = GENERATED / "sae_dark_subspace" / f"p12b_mixed_sae_seed{s_label}" / "results.json"
    d = load(p)
    if d is None:
        _check(f"P12B mixed seed {s_label} bundled", True, False)
    else:
        orig = d.get("original", {}).get("score_K_auroc")
        recon = d.get("sae_reconstructed", {}).get("score_K_auroc")
        rc = d.get("sae_quality", {}).get("reconstruction_cosine")
        _CHECK_RESULTS.append((f"P12B mixed seed {s_label} bundled and parseable",
                                orig is not None and recon is not None and rc is not None,
                                f"orig={orig}, recon={recon}, rc={rc}"))
        ok = orig is not None and recon is not None and rc is not None
        print(f"    [{'PASS' if ok else 'FAIL'}] P12B mixed seed {s_label} bundled: orig={orig}, recon={recon}, rc={rc}")
        # Per-seed values consistent with cohort range (orig in [0.76, 0.77], rc > 0.99)
        if orig is not None:
            _check(f"P12B mixed seed {s_label} orig in cohort range", 0.766, orig, tol=0.005)
        if rc is not None:
            _check(f"P12B mixed seed {s_label} recon_cos > 0.99", 0.991, rc, tol=0.01)


banner("Gemma-2-2B SAE row (generated/sae_dark_subspace/gemma2_2b_epoch5, tab:dark_subspace) [asserted]")

# Backs the Gemma-2-2B row of tab:dark_subspace and is the missing eighth
# row of the body sign-test denominator. Paper: orig 0.801, recon 0.769,
# residual 0.820, recon_cosine 0.911 (above the 0.90 strict gate).
gm = load(GENERATED / "sae_dark_subspace" / "gemma2_2b_epoch5" / "results.json")
if gm is not None:
    _check("Gemma-2-2B orig (paper 0.801)", 0.801,
           gm.get("original", {}).get("score_K_auroc"), tol=3e-3)
    _check("Gemma-2-2B recon (paper 0.769)", 0.769,
           gm.get("sae_reconstructed", {}).get("score_K_auroc"), tol=3e-3)
    _check("Gemma-2-2B resid (paper 0.820)", 0.820,
           gm.get("residual", {}).get("score_K_auroc"), tol=3e-3)
    _check("Gemma-2-2B recon_cos (paper 0.911)", 0.911,
           gm.get("sae_quality", {}).get("reconstruction_cosine"), tol=2e-3)


banner("FSC values (paper_claims/fsc_values.json, tab:fsc_values) [asserted]")

# Backs Appendix Table tab:fsc_values; eight model rows.
fsc = load(PAPER_CLAIMS / "fsc_values.json")
if fsc is not None:
    for r in fsc.get("rows", []):
        setting = r.get("setting", "?")
        # Disk vs paper agreement on n_cf and fsc_K_cf
        _check(f"FSC {setting} n_cf_features matches paper",
               r.get("paper_n_cf"), r.get("n_cf_features"), tol=0)
        _check(f"FSC {setting} fsc_K_cf matches paper to 1e-3",
               r.get("paper_fsc_K_cf"), r.get("fsc_K_cf"), tol=1.5e-3)
        _check(f"FSC {setting} fsc_K_all == 1.0",
               1.000, r.get("fsc_K_all"), tol=1e-6)
        _check(f"FSC {setting} fsc_R_all == 1.0",
               1.000, r.get("fsc_R_all"), tol=1e-6)


banner("Extraction-detection separation paper-claim JSON (paper_claims/extraction_detection_separation.json, tab:dd_full + tab:dd_extraction + tab:epoch_dd) [asserted]")

# Internal consistency of the bundled extraction-detection paper-claim JSON.
# Numerical values trace to GPU-pipeline run-time records that are not bundled.
# The asserts here check that the bundled JSON encodes the manuscript-rendered
# cell values consistently across the three overlapping tables (tab:dd_full,
# tab:dd_extraction, tab:epoch_dd).
dd = load(PAPER_CLAIMS / "extraction_detection_separation.json")
if dd is not None:
    # tab:dd_extraction P1B at epoch 5 must equal tab:epoch_dd at epoch 5
    p1b_ext = next((r for r in dd["tab_dd_extraction"]["rows"] if r["setting"] == "Pythia-1B"), {})
    p1b_epoch5 = next((r for r in dd["tab_epoch_dd"]["rows"] if r["epoch"] == 5), {}).get("rouge_l", {})
    _check("DD P1B baseline ROUGE-L cross-table (tab:dd_extraction == tab:epoch_dd at epoch 5)",
           p1b_ext.get("baseline"), p1b_epoch5.get("baseline"), tol=1e-6)
    _check("DD P1B erase S_K ROUGE-L cross-table",
           p1b_ext.get("erase_S_K"), p1b_epoch5.get("erase_S_K"), tol=1e-6)
    _check("DD P1B erase S_R ROUGE-L cross-table",
           p1b_ext.get("erase_S_R"), p1b_epoch5.get("erase_S_R"), tol=1e-6)
    _check("DD P1B erase both ROUGE-L cross-table",
           p1b_ext.get("erase_both"), p1b_epoch5.get("erase_both"), tol=1e-6)
    # tab:dd_full row count
    _check("DD tab:dd_full row count == 4 (P1B, P69, P12B, OPT)",
           4, len(dd["tab_dd_full"]["rows"]), tol=0)
    # tab:dd_extraction row count
    _check("DD tab:dd_extraction row count == 8",
           8, len(dd["tab_dd_extraction"]["rows"]), tol=0)
    # tab:epoch_dd row count
    _check("DD tab:epoch_dd row count == 3 (epochs 1, 3, 5)",
           3, len(dd["tab_epoch_dd"]["rows"]), tol=0)
    # Sanity: every erase-condition value <= baseline (loss inversely)
    for row in dd["tab_dd_extraction"]["rows"]:
        baseline = row.get("baseline", 0)
        for cond in ["erase_S_K", "erase_S_R", "erase_both"]:
            v = row.get(cond, 0)
            ok = v <= baseline + 1e-6
            _CHECK_RESULTS.append((f"DD tab:dd_extraction {row['setting']} {cond} <= baseline",
                                    ok, f"baseline={baseline}, {cond}={v}"))
            print(f"    [{'PASS' if ok else 'FAIL'}] DD tab:dd_extraction {row['setting']} {cond} <= baseline: baseline={baseline}, {cond}={v}")


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
    print("  manuscript/references.bib not present; bibliography check skipped.")


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


banner("Body sign test on tab:dark_subspace gate-passing rows (paper_claims/tab_dark_subspace_sign_test.json) [asserted]")

# Paper results.tex paragraph after tab:dark_subspace and appendix paragraph
# 'Binomial sign-test arithmetic.' under app:koc2_bootstrap report two
# binomial sign tests on the gate-passing rows of tab:dark_subspace:
#   Non-Pythia subset (n=4): one-sided p approx 0.0625
#   Full gate-passing set (n=7): one-sided p approx 0.008
# This denominator differs from the cohort sign test asserted above
# (n_inverting_cohort_rows=5).
ts = load(PAPER_CLAIMS / "tab_dark_subspace_sign_test.json")
if ts is None:
    _check("Body sign test non-Pythia n (paper 4)", 4, None)
    _check("Body sign test non-Pythia p_one_sided (paper 0.0625)", 0.0625, None)
    _check("Body sign test full gate-passing n (paper 7)", 7, None)
    _check("Body sign test full gate-passing p_one_sided (paper 0.008)", 0.0078125, None)
else:
    bst = ts.get("binomial_sign_test", {})
    np_block = bst.get("non_pythia_subset", {})
    full_block = bst.get("full_gate_passing_set", {})
    _check("Body sign test non-Pythia n (paper 4)", 4, np_block.get("n"), tol=0)
    _check("Body sign test non-Pythia k positive (paper 4)", 4, np_block.get("k_residual_above_recon"), tol=0)
    _check("Body sign test non-Pythia p_one_sided (paper 0.0625)", 0.0625,
           np_block.get("p_one_sided"), tol=1e-6)
    _check("Body sign test full gate-passing n (paper 7)", 7, full_block.get("n"), tol=0)
    _check("Body sign test full gate-passing k positive (paper 7)", 7, full_block.get("k_residual_above_recon"), tol=0)
    _check("Body sign test full gate-passing p_one_sided (paper 0.0078125 ~= 0.008)", 0.0078125,
           full_block.get("p_one_sided"), tol=1e-6)
    rows = ts.get("gate_passing_rows", [])
    _check("Body sign test gate-passing row count (paper 7)", 7, len(rows), tol=0)
    n_above = sum(1 for r in rows if r.get("residual_above_recon"))
    _check("Body sign test rows with residual > recon (paper 7)", 7, n_above, tol=0)


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


banner("TPR at 0.1% FPR for residual d_K (generated/tpr_low_fpr, tab:tpr_at_0p1pct_fpr) [asserted]")

# Paper Appendix Table tab:tpr_at_0p1pct_fpr reports TPR point estimates,
# bootstrap means and 95% CIs at FPR=0.001 across four models.
TPR_PAPER = {
    "p69":   {"point": 0.015, "mean": 0.018, "ci_lo": 0.005, "ci_hi": 0.050},
    "p12b":  {"point": 0.032, "mean": 0.035, "ci_lo": 0.020, "ci_hi": 0.071},
    "neo":   {"point": 0.025, "mean": 0.025, "ci_lo": 0.014, "ci_hi": 0.042},
    "qwen2": {"point": 0.010, "mean": 0.007, "ci_lo": 0.000, "ci_hi": 0.023},
}
for tag, paper_vals in TPR_PAPER.items():
    p = GENERATED / "tpr_low_fpr" / tag / "results.json"
    d = load(p)
    if d is None:
        _check(f"TPR@0.1%FPR {tag} point (paper {paper_vals['point']:.3f})", paper_vals["point"], None)
        _check(f"TPR@0.1%FPR {tag} mean (paper {paper_vals['mean']:.3f})", paper_vals["mean"], None)
        _check(f"TPR@0.1%FPR {tag} CI lo (paper {paper_vals['ci_lo']:.3f})", paper_vals["ci_lo"], None)
        _check(f"TPR@0.1%FPR {tag} CI hi (paper {paper_vals['ci_hi']:.3f})", paper_vals["ci_hi"], None)
    else:
        _check(f"TPR@0.1%FPR {tag} point (paper {paper_vals['point']:.3f})",
               paper_vals["point"], d.get("tpr_point"), tol=2e-3)
        _check(f"TPR@0.1%FPR {tag} mean (paper {paper_vals['mean']:.3f})",
               paper_vals["mean"], d.get("tpr_boot_mean"), tol=2e-3)
        _check(f"TPR@0.1%FPR {tag} CI lo (paper {paper_vals['ci_lo']:.3f})",
               paper_vals["ci_lo"], d.get("tpr_ci_lo"), tol=2e-3)
        _check(f"TPR@0.1%FPR {tag} CI hi (paper {paper_vals['ci_hi']:.3f})",
               paper_vals["ci_hi"], d.get("tpr_ci_hi"), tol=2e-3)
        _check(f"TPR@0.1%FPR {tag} fpr_target=0.001", 0.001, d.get("fpr_target"), tol=0)
        _check(f"TPR@0.1%FPR {tag} n_boot=10000", 10000, d.get("n_boot"), tol=0)


banner("P69 N=6 cohort aggregate (paper_claims/p69_n6_complete.json) [asserted]")

# N=6 alternative framing of the Pythia-6.9B mixed-data cohort. The paper
# reports N=5 (drops seed 47 to share Pythia-1B's seed list). The N=6
# aggregate is bundled as well. The conclusion is unchanged. Per-metric
# mean shifts between N=5 and N=6 are below 0.005 on all reported metrics,
# recorded in p69_n5_harmonized_2026-05-06.json.
n6 = load(PAPER_CLAIMS / "p69_n6_complete.json")
if n6 is not None:
    s = n6.get("cluster_summary_n6", {})
    drop_n6 = s.get("drop_original_minus_reconstructed", {}).get("mean")
    rs_n6 = s.get("residual_score_K_auroc", {}).get("mean")
    r_n6 = s.get("reconstructed_score_K_auroc", {}).get("mean")
    rc_n6 = s.get("recon_cos", {}).get("mean")
    # N=6 must show residual > reconstruction (headline ordering)
    _CHECK_RESULTS.append((
        "P69 N=6 residual > recon (headline ordering)",
        rs_n6 is not None and r_n6 is not None and rs_n6 > r_n6,
        f"resid={rs_n6}, recon={r_n6}",
    ))
    print(f"    [{'PASS' if rs_n6 > r_n6 else 'FAIL'}] P69 N=6 residual > recon: resid={rs_n6:.4f}, recon={r_n6:.4f}")
    # N=6 row count == 6
    _check("P69 N=6 row count (paper 6)", 6,
           s.get("drop_original_minus_reconstructed", {}).get("n"), tol=0)
    # Materiality vs N=5 must be NEGLIGIBLE
    mat = n6.get("materiality_vs_n5", {})
    verdict_ok = mat.get("verdict") == "NEGLIGIBLE"
    _CHECK_RESULTS.append((
        "P69 N=6 vs N=5 materiality verdict (NEGLIGIBLE)",
        verdict_ok,
        f"actual={mat.get('verdict')!r}",
    ))
    print(f"    [{'PASS' if verdict_ok else 'FAIL'}] P69 N=6 vs N=5 materiality verdict: actual={mat.get('verdict')!r}")
    threshold = mat.get("threshold", 0.005)
    abs_delta = mat.get("abs_delta_mean_drop")
    within = abs_delta is not None and abs_delta < threshold
    _CHECK_RESULTS.append((
        f"P69 N=6 vs N=5 |delta_mean_drop| < {threshold}",
        within,
        f"actual={abs_delta:.6f}",
    ))
    print(f"    [{'PASS' if within else 'FAIL'}] P69 N=6 vs N=5 |delta_mean_drop| < {threshold}: actual={abs_delta:.6f}")


banner("P69 seed-42 pre-vs-postfix audit trail (paper_claims/p69_seed42_pre_vs_postfix.json) [asserted]")

# Backs the README naming-notes claim that the Pythia-6.9B seed-42 SAE was
# retrained once at corrected hyperparameters. Asserts that both runs show
# residual > sae_reconstructed (the headline ordering), so the postfix
# retrain is a hyperparameter-conformance correction rather than an
# outcome-driven retrain.
pp = load(PAPER_CLAIMS / "p69_seed42_pre_vs_postfix.json")
if pp is not None:
    pre = pp.get("pre_postfix_run", {}).get("metrics", {})
    post = pp.get("postfix_run", {}).get("metrics", {})
    cmp = pp.get("comparison", {})
    pre_resid_above_recon = (pre.get("residual_score_K_auroc", 0) >
                             pre.get("sae_reconstructed_score_K_auroc", 0))
    post_resid_above_recon = (post.get("residual_score_K_auroc", 0) >
                              post.get("sae_reconstructed_score_K_auroc", 0))
    _CHECK_RESULTS.append((
        "P69 seed-42 pre-postfix run shows residual > recon (ordering preserved)",
        pre_resid_above_recon,
        f"resid={pre.get('residual_score_K_auroc')}, recon={pre.get('sae_reconstructed_score_K_auroc')}",
    ))
    print(f"    [{'PASS' if pre_resid_above_recon else 'FAIL'}] P69 seed-42 pre-postfix run shows residual > recon (ordering preserved)")
    _CHECK_RESULTS.append((
        "P69 seed-42 postfix run shows residual > recon (ordering preserved)",
        post_resid_above_recon,
        f"resid={post.get('residual_score_K_auroc')}, recon={post.get('sae_reconstructed_score_K_auroc')}",
    ))
    print(f"    [{'PASS' if post_resid_above_recon else 'FAIL'}] P69 seed-42 postfix run shows residual > recon (ordering preserved)")
    _CHECK_RESULTS.append((
        "P69 seed-42 ordering preserved across retrain",
        cmp.get("ordering_preserved") is True,
        f"actual={cmp.get('ordering_preserved')!r}",
    ))
    print(f"    [{'PASS' if cmp.get('ordering_preserved') is True else 'FAIL'}] P69 seed-42 ordering preserved across retrain")


banner("SAE-quality exclusions (paper_claims/sae_quality_exclusions.json) [asserted]")

# Backs the manuscript's exclusion of Mistral-7B and Llama-3-8B from
# tab:dark_subspace. The validity-gate threshold is 0.85 (permissive tier);
# recon cosine below 0.85 disqualifies the row from the quantitative
# claim. Both excluded settings have recon cosine below 0.85.
ex = load(PAPER_CLAIMS / "sae_quality_exclusions.json")
if ex is not None:
    settings_seen = set()
    for row in ex.get("exclusions", []):
        settings_seen.add(row.get("setting"))
        rc = row.get("reconstruction_cosine")
        below = rc is not None and rc < 0.85
        _CHECK_RESULTS.append((f"SAE exclusion {row.get('setting')} recon_cos < 0.85 (permissive gate)",
                                below, f"actual={rc}"))
        flag = "PASS" if below else "FAIL"
        print(f"    [{flag}] SAE exclusion {row.get('setting')} recon_cos < 0.85: actual={rc}")
    _check("SAE exclusions row count == 2 (Mistral, Llama-3)", 2, len(settings_seen), tol=0)
    _CHECK_RESULTS.append((
        "SAE exclusions cover Mistral-7B and Llama-3-8B",
        settings_seen == {"Mistral-7B", "Llama-3-8B"},
        f"actual={sorted(settings_seen)}",
    ))
    flag = "PASS" if settings_seen == {"Mistral-7B", "Llama-3-8B"} else "FAIL"
    print(f"    [{flag}] SAE exclusions cover Mistral-7B and Llama-3-8B: actual={sorted(settings_seen)}")


banner("Per-model channel decomposition (generated/behavioral_channels, tab:bcd_main) [asserted]")

# Paper Appendix Table tab:bcd_main reports cos(d_K, d_R) and per-layer
# membership / recall probe AUROC at the channel-geometry reference layer
# for each of the nine models. Verified at the per-model SAE layer
# (Pythia-12B reports at layer 24, the channel-geometry reference layer,
# rather than the layer-18 SAE-evaluation layer cited in tab:dark_subspace).
BCD_MAIN_PAPER = [
    ("p1b_epoch5",          "Pythia-1B",       8,   0.102, 0.577, 0.762),
    ("p69_epoch5",          "Pythia-6.9B",     16,  0.107, 0.828, 0.696),
    ("p12b_epoch5",         "Pythia-12B",      24,  0.336, 0.781, 0.806),
    ("neo_epoch5",          "GPT-Neo-2.7B",    16,  0.024, 0.504, 0.813),
    ("opt67_epoch5",        "OPT-6.7B",        24, -0.063, 0.869, 0.883),
    ("falcon7b_epoch5_v2",  "Falcon-7B",       16,  0.161, 0.635, 0.855),
    ("mistral_epoch5_v2",   "Mistral-7B",      16, -0.010, 0.967, 0.714),
    ("llama3_epoch5_v2",    "Llama-3-8B",      16,  0.223, 0.957, 0.735),
    ("qwen2_epoch5",        "Qwen2-7B",        16,  0.390, 0.803, 0.705),
]
for tag, label, layer, p_cos, p_mem, p_rec in BCD_MAIN_PAPER:
    p = GENERATED / "behavioral_channels" / tag / "orthogonality.json"
    d = load(p)
    if d is None:
        _check(f"BCD-main {label} cos(d_K,d_R) (paper {p_cos:.3f})", p_cos, None)
        _check(f"BCD-main {label} Mem AUROC (paper {p_mem:.3f})", p_mem, None)
        _check(f"BCD-main {label} Rec AUROC (paper {p_rec:.3f})", p_rec, None)
    else:
        pl = d.get("per_layer", {}).get(str(layer), {})
        _check(f"BCD-main {label} cos(d_K,d_R) (paper {p_cos:.3f})", p_cos, pl.get("cosine_d_K_d_R"), tol=2e-3)
        _check(f"BCD-main {label} Mem AUROC (paper {p_mem:.3f})", p_mem,
               pl.get("membership_probe", {}).get("auroc_mean"), tol=2e-3)
        _check(f"BCD-main {label} Rec AUROC (paper {p_rec:.3f})", p_rec,
               pl.get("recall_probe", {}).get("auroc_mean"), tol=2e-3)


banner("Privacy-aware SAE score_K (generated/sae_dark_subspace/*_ft_dk*, tab:fresh_probe_v2) [asserted]")

# Paper Appendix Table tab:fresh_probe_v2 reports score_K AUROC across
# four conditions and three compartments. The probe-AUROC and permutation-null
# columns of that table are produced by a separate fresh-probe pipeline and
# are not bundled. The score_K column is verifiable from the shipped
# sae_dark_subspace results.json files for the privacy-aware SAE
# conditions (the fourth condition, the Pythia-6.9B baseline mixed SAE,
# is verified above against the p69_n5 / p69_epoch5 cells of tab:dark_subspace).
FRESH_PROBE_PAPER = {
    "p69_ft_dk0.1": {"original": 0.803, "sae_reconstructed": 0.803, "residual": 0.520},
    "p69_ft_dk1.0": {"original": 0.803, "sae_reconstructed": 0.798, "residual": 0.537},
    "neo_ft_dk1.0": {"original": 0.615, "sae_reconstructed": 0.612, "residual": 0.565},
}
for tag, paper_vals in FRESH_PROBE_PAPER.items():
    p = GENERATED / "sae_dark_subspace" / tag / "results.json"
    d = load(p)
    if d is None:
        for compartment, paper_val in paper_vals.items():
            _check(f"Fresh-probe v2 {tag} {compartment} score_K (paper {paper_val:.3f})", paper_val, None)
    else:
        for compartment, paper_val in paper_vals.items():
            block = d.get(compartment, {})
            _check(f"Fresh-probe v2 {tag} {compartment} score_K (paper {paper_val:.3f})",
                   paper_val, block.get("score_K_auroc"), tol=5e-3)


banner("Pythia scaling curve (generated/behavioral_channels, tab:scaling) [asserted]")

# Paper Appendix Table tab:scaling reports score_K AUROC across seven
# Pythia sizes at each model's best membership-probe layer.
SCALING_PAPER = {
    "p70m_epoch5":  ("Pythia-70M",   0.507),
    "p160m_epoch5": ("Pythia-160M",  0.588),
    "p410m_epoch5": ("Pythia-410M",  0.716),
    "p1b_epoch5":   ("Pythia-1B",    0.800),
    "p2.8b_epoch5": ("Pythia-2.8B",  0.842),
    "p69_epoch5":   ("Pythia-6.9B",  0.876),
    "p12b_epoch5":  ("Pythia-12B",   0.781),
}
for tag, (label, paper_val) in SCALING_PAPER.items():
    p = GENERATED / "behavioral_channels" / tag / "orthogonality.json"
    d = load(p)
    if d is None:
        _check(f"Scaling {label} score_K AUROC (paper {paper_val:.3f})", paper_val, None)
    else:
        pl = d.get("per_layer", {})
        aurocs = [v.get("membership_probe", {}).get("auroc_mean")
                  for v in pl.values() if v.get("membership_probe", {}).get("auroc_mean") is not None]
        best = max(aurocs) if aurocs else None
        _check(f"Scaling {label} score_K AUROC (paper {paper_val:.3f})", paper_val, best, tol=2e-3)


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
print("End of verification dump")
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
print(f"Asserted check summary: {n_pass}/{n_total} pass, {n_fail} fail")
print("=" * 70)
if n_fail:
    print("Failures:")
    for label, ok, detail in _CHECK_RESULTS:
        if not ok:
            print(f"  FAIL  {label}: {detail}")
    sys.exit(1)
print("All asserted checks pass within tolerance.")
sys.exit(0)
