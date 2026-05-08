#!/usr/bin/env python3
"""figure_data_loader.py.

Loads disk-verified JSONs into typed records (with source-state flags)
and serves as the single source of numerical inputs for the plotting scripts
so figures can be regenerated without invoking the GPU pipeline.

Used in the paper figure-generation pipeline (Methods, Results, Appendix).
Reproduce: ``python3 scripts/dark_subspace/figure_data_loader.py``
(prints a diagnostic summary of every loader; does not write artifacts).

Sources.
* score_K AUROC and recon_cos   results/dark_subspace/generated/sae_dark_subspace/{dir}/results.json
* norm AUROC                    results/dark_subspace/generated/norm_baseline/{dir}/results.json (best_auroc)
* per-layer membership AUROC    results/dark_subspace/generated/behavioral_channels/{dir}/orthogonality.json
* bootstrap CIs (n_boot=10000)  results/dark_subspace/generated/sae_dark_subspace/{all_models_bootstrap_cis,new_bootstrap_cis}.json
* P69 N=6 noise-floor aggregate results/dark_subspace/generated/sae_noise_floor/p69_aggregate.json
* P12B fresh-init aggregate     runs/sae_array/p12b_freshinit/aggregate.json
                                (with single-shot layer18 source when absent)

P12B scaling source. When the fresh-init aggregate is not present, callers use
``load_p12b_scaling_auroc()`` which returns a ``ScalingValue`` whose ``.value``
field contains the on-disk best-layer estimate and whose ``.is_provisional``
flag tells downstream plots to draw an open-face marker.
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Optional

# ── Repo root resolution ────────────────────────────────────────────────────
_THIS = pathlib.Path(__file__).resolve()
REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", _THIS.parents[2]))
RESULTS_ROOT = pathlib.Path(
    os.environ.get("DARK_SUBSPACE_RESULTS_ROOT", REPO_ROOT / "results" / "dark_subspace" / "generated")
)
DARK = RESULTS_ROOT / "sae_dark_subspace"
NORM = RESULTS_ROOT / "norm_baseline"
BC = RESULTS_ROOT / "behavioral_channels"
NOISE_FLOOR = RESULTS_ROOT / "sae_noise_floor"
SAE_ARRAY = pathlib.Path(
    os.environ.get("DARK_SUBSPACE_SAE_ARRAY_ROOT", REPO_ROOT / "runs" / "sae_array")
)

# Canonical model to directory mapping.
# Each tuple: (display_label, dark_subspace_dir, norm_baseline_dir,
#               behavioral_channels_dir, family).
# "v2" suffix prefers the canonical re-run when both exist.
MODEL_REGISTRY = {
    # ── Pythia scaling family
    "Pythia-70M":   {"dark": None,                  "norm": "p70m_epoch5",  "bc": "p70m_epoch5",            "family": "GPT", "params": 70e6},
    "Pythia-160M":  {"dark": None,                  "norm": "p160m_epoch5", "bc": "p160m_epoch5",           "family": "GPT", "params": 160e6},
    "Pythia-410M":  {"dark": None,                  "norm": "p410m_epoch5", "bc": "p410m_epoch5",           "family": "GPT", "params": 410e6},
    "Pythia-1B":    {"dark": "p1b_epoch5",          "norm": "p1b_epoch5",   "bc": "p1b_epoch5",             "family": "GPT", "params": 1e9},
    "Pythia-2.8B":  {"dark": None,                  "norm": None,           "bc": "p2.8b_epoch5",           "family": "GPT", "params": 2.8e9},
    "Pythia-6.9B":  {"dark": "p69_epoch5",          "norm": "p69_epoch5",   "bc": "p69_epoch5",             "family": "GPT", "params": 6.9e9},
    "Pythia-12B":   {"dark": "p12b_epoch5",         "norm": "p12b_epoch5",  "bc": "p12b_epoch5",            "family": "GPT", "params": 12e9},
    # ── Other GPT-family
    "GPT-Neo-2.7B": {"dark": "neo_epoch5",          "norm": "neo_epoch5",   "bc": "neo_epoch5",             "family": "GPT", "params": 2.7e9},
    "OPT-6.7B":     {"dark": "opt67_epoch5",        "norm": "opt67_epoch5", "bc": "opt67_epoch5",           "family": "GPT", "params": 6.7e9},
    "Falcon-7B":    {"dark": "falcon_epoch5",       "norm": "falcon_epoch5","bc": "falcon7b_epoch5_v2",     "family": "GPT", "params": 7e9},
    # ── LLaMA / similar
    "Mistral-7B":   {"dark": "mistral_epoch5",      "norm": "mistral_epoch5","bc": "mistral_epoch5_v2",     "family": "LLaMA","params": 7e9},
    "Llama-3-8B":   {"dark": None,                  "norm": "llama3_epoch5", "bc": "llama3_epoch5_v2",      "family": "LLaMA","params": 8e9},
    "Qwen2-7B":     {"dark": "qwen2_epoch5",        "norm": "qwen2_epoch5", "bc": "qwen2_epoch5",           "family": "LLaMA","params": 7e9},
    # ── Mixed-data control (uses the same P69 source rows but different SAE)
    "Pythia-6.9B (mixed)":
                    {"dark": "p69_mixed_sae",       "norm": "p69_epoch5",   "bc": "p69_epoch5",             "family": "GPT", "params": 6.9e9},
}


@dataclass(frozen=True)
class ScalingValue:
    """Wraps a numeric value and records whether it uses a single-run source.

    Attributes
    ----------
    value      : on-disk estimate
    is_provisional : True when the aggregate source is unavailable
    source     : path or note describing what `value` came from
    """
    value: float
    is_provisional: bool
    source: str


# ── Low-level loaders ──────────────────────────────────────────────────────
def _read_json(path: pathlib.Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def load_dark_subspace(model_label: str) -> dict:
    """Return {original, sae_reconstructed, residual, sae_quality, source} for a model.

    Raises FileNotFoundError when the dark-subspace dir is not registered or
    the results.json does not exist.
    """
    info = MODEL_REGISTRY.get(model_label)
    if info is None or info["dark"] is None:
        raise FileNotFoundError(f"No dark_subspace dir registered for {model_label!r}")
    p = DARK / info["dark"] / "results.json"
    d = _read_json(p)
    return {
        "original":  d["original"],
        "sae_reconstructed": d["sae_reconstructed"],
        "residual":  d["residual"],
        "sae_quality": d["sae_quality"],
        "source": str(p),
    }


def load_norm_auroc(model_label: str) -> dict:
    """Return {auroc, best_layer, source} for the per-token activation-norm baseline."""
    info = MODEL_REGISTRY.get(model_label)
    if info is None or info["norm"] is None:
        raise FileNotFoundError(f"No norm_baseline dir registered for {model_label!r}")
    p = NORM / info["norm"] / "results.json"
    d = _read_json(p)
    return {
        "auroc": float(d["best_auroc"]),
        "best_layer": int(d["best_layer"]),
        "source": str(p),
    }


def load_behavioral_layers(model_label: str) -> dict:
    """Return {layers, aurocs, n_layers, source} from behavioral_channels orthogonality.json."""
    info = MODEL_REGISTRY.get(model_label)
    if info is None or info["bc"] is None:
        raise FileNotFoundError(f"No behavioral_channels dir registered for {model_label!r}")
    p = BC / info["bc"] / "orthogonality.json"
    d = _read_json(p)
    layers = list(d["layers_analyzed"])
    aurocs = [d["per_layer"][str(l)]["membership_probe"]["auroc_mean"] for l in layers]
    return {
        "layers": layers,
        "aurocs": aurocs,
        "n_layers": int(d["n_layers"]),
        "source": str(p),
    }


def load_best_layer_membership_auroc(model_label: str) -> dict:
    """Best-layer membership AUROC. Used as the scaling-curve y-value."""
    bl = load_behavioral_layers(model_label)
    idx = max(range(len(bl["aurocs"])), key=lambda i: bl["aurocs"][i])
    return {
        "auroc": float(bl["aurocs"][idx]),
        "best_layer": int(bl["layers"][idx]),
        "source": bl["source"],
    }


# ── Bootstrap CI loaders (merged across two files) ─────────────────────────
def load_bootstrap_cis() -> dict:
    """Merge `all_models_bootstrap_cis.json` + `new_bootstrap_cis.json` into a
    single dict keyed by directory name.

    Returns
    -------
    dict: dir -> {original, recon, residual, n_boot, source}
    """
    out: dict = {}
    p_all = DARK / "all_models_bootstrap_cis.json"
    if p_all.exists():
        d = _read_json(p_all)
        for entry in d.get("models", []):
            dir_key = entry["dir"]
            out[dir_key] = {
                "label": entry["model"],
                "original": entry["original"],
                "recon":    entry["recon"],
                "residual": entry["residual"],
                "n_boot":   int(entry["original"].get("n_bootstrap", d.get("n_bootstrap", 0))),
                "source":   str(p_all),
            }
    p_new = DARK / "new_bootstrap_cis.json"
    if p_new.exists():
        d = _read_json(p_new)
        for dir_key, entry in d.items():
            # `feature_ablation_*` entries don't have orig/recon/resid sub-objects
            if not isinstance(entry, dict) or "original" not in entry:
                continue
            existing = out.get(dir_key, {})
            out[dir_key] = {
                "label":    existing.get("label", dir_key),
                "original": entry["original"],
                "recon":    entry["recon"],
                "residual": entry["residual"],
                "n_boot":   int(entry["original"].get("n_boot", 0)),
                "source":   str(p_new),
            }
    return out


def load_bootstrap_for_model(model_label: str) -> Optional[dict]:
    """Lookup a model's bootstrap CIs by its dark-subspace directory."""
    info = MODEL_REGISTRY.get(model_label)
    if info is None or info["dark"] is None:
        return None
    cis = load_bootstrap_cis()
    return cis.get(info["dark"])


# ── Aggregates ─────────────────────────────────────────────────────────────
def load_p69_noise_floor() -> dict:
    """Return the P69 N=6 mixed-SAE aggregate (orig/recon/resid means and stds)."""
    p = NOISE_FLOOR / "p69_aggregate.json"
    d = _read_json(p)
    return {
        "summary":     d["summary"],
        "noise_floor": d["noise_floor"],
        "n":           int(d.get("roster_size", len(d.get("per_seed", [])))),
        "source":      str(p),
    }


def load_p12b_scaling_auroc() -> ScalingValue:
    """Return the Pythia-12B AUROC for the scaling curve.

    Tries, in order, the canonical fresh-init aggregate at
    ``runs/sae_array/p12b_freshinit/aggregate.json``, then the
    ``behavioral_channels`` best-layer membership AUROC as a single-shot
    source. The ``is_provisional`` flag is False for the aggregate path
    and True for the single-shot source.
    """
    agg = SAE_ARRAY / "p12b_freshinit" / "aggregate.json"
    if agg.exists():
        d = _read_json(agg)
        mean = (
            d.get("summary", {})
             .get("membership_auroc", {})
             .get("mean")
            or d.get("membership_auroc_mean")
        )
        if mean is not None:
            return ScalingValue(value=float(mean), is_provisional=False, source=str(agg))
    bl = load_best_layer_membership_auroc("Pythia-12B")
    return ScalingValue(
        value=bl["auroc"],
        is_provisional=True,
        source=bl["source"] + " (single-run scaling source)",
    )


# ── Convenience aggregators for figures ────────────────────────────────────
def get_dark_subspace_table(model_labels: list[str]) -> dict:
    """Return per-label dark-subspace records used by the bar and heatmap figures.

    For ``"Pythia-6.9B (mixed)"`` the function uses the N=6 mixed-data SAE
    aggregate (orig 0.803, recon 0.594, resid 0.779) so the displayed bars
    match the paper's results table caption.

    Parameters
    ----------
    model_labels : list of str
        Display labels in ``MODEL_REGISTRY``.

    Returns
    -------
    dict
        Mapping label to ``{orig, recon, resid, recon_cos, drop, source, n}``.
    """
    out = {}
    for lbl in model_labels:
        if lbl == "Pythia-6.9B (mixed)":
            agg = load_p69_noise_floor()
            s = agg["summary"]
            out[lbl] = {
                "orig":      s["original_score_K_auroc"]["mean"],
                "recon":     s["reconstructed_score_K_auroc"]["mean"],
                "resid":     s["residual_score_K_auroc"]["mean"],
                "recon_cos": s["recon_cos"]["mean"],
                "drop":      s["drop_original_minus_reconstructed"]["mean"],
                "source":    agg["source"],
                "n":         agg["n"],
            }
            continue
        ds = load_dark_subspace(lbl)
        out[lbl] = {
            "orig":      ds["original"]["score_K_auroc"],
            "recon":     ds["sae_reconstructed"]["score_K_auroc"],
            "resid":     ds["residual"]["score_K_auroc"],
            "recon_cos": ds["sae_quality"]["reconstruction_cosine"],
            "drop":      ds["original"]["score_K_auroc"] - ds["sae_reconstructed"]["score_K_auroc"],
            "source":    ds["source"],
            "n":         1,
        }
    return out


def get_norm_table(model_labels: list[str]) -> dict:
    """Return {label -> {norm_auroc, bcd_auroc, source_norm, source_bcd}} for the
    norm-vs-direction figures."""
    out = {}
    for lbl in model_labels:
        n = load_norm_auroc(lbl)
        if MODEL_REGISTRY[lbl]["dark"]:
            ds = load_dark_subspace(lbl)
            bcd = ds["original"]["score_K_auroc"]
            bcd_source = ds["source"]
        else:
            ds = load_best_layer_membership_auroc(lbl)
            bcd = ds["auroc"]
            bcd_source = ds["source"]
        out[lbl] = {
            "norm_auroc": n["auroc"],
            "bcd_auroc":  bcd,
            "source_norm": n["source"],
            "source_bcd":  bcd_source,
        }
    return out


def get_scaling_curve_data(model_labels: list[str]) -> dict:
    """Return {labels, params, aurocs, provisional, sources} for the model-size scaling curve."""
    labels, params, aurocs, provisional, sources = [], [], [], [], []
    for lbl in model_labels:
        if lbl == "Pythia-12B":
            value = load_p12b_scaling_auroc()
            aurocs.append(value.value)
            provisional.append(value.is_provisional)
            sources.append(value.source)
        else:
            row = load_best_layer_membership_auroc(lbl)
            aurocs.append(row["auroc"])
            provisional.append(False)
            sources.append(row["source"])
        labels.append(lbl)
        params.append(MODEL_REGISTRY[lbl]["params"])
    return {
        "labels": labels,
        "params": params,
        "aurocs": aurocs,
        "provisional": provisional,
        "sources": sources,
    }


# ── CLI integration check ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("repo_root =", REPO_ROOT)
    print("results_root =", RESULTS_ROOT)
    print("sae_dark_subspace sources present =", DARK.exists())
    print()
    labels_full = [
        "Pythia-6.9B", "Pythia-6.9B (mixed)", "Pythia-1B", "GPT-Neo-2.7B",
        "OPT-6.7B", "Pythia-12B", "Mistral-7B", "Qwen2-7B", "Falcon-7B",
    ]
    print("=== dark_subspace table ===")
    for k, v in get_dark_subspace_table(labels_full).items():
        print(f"  {k:24s}  orig={v['orig']:.3f} recon={v['recon']:.3f} resid={v['resid']:.3f} cos={v['recon_cos']:.3f}")
    print()
    print("=== norm vs channel-decomposition ===")
    for k, v in get_norm_table([
        "Pythia-1B","GPT-Neo-2.7B","Pythia-6.9B","OPT-6.7B","Pythia-12B",
        "Falcon-7B","Mistral-7B","Llama-3-8B","Qwen2-7B"]).items():
        print(f"  {k:24s}  norm={v['norm_auroc']:.3f}  d_K={v['bcd_auroc']:.3f}")
    print()
    print("=== scaling curve ===")
    sc = get_scaling_curve_data(["Pythia-70M","Pythia-160M","Pythia-410M",
                                  "Pythia-1B","Pythia-2.8B","Pythia-6.9B","Pythia-12B"])
    for lbl, p, a, ph in zip(sc["labels"], sc["params"], sc["aurocs"], sc["provisional"]):
        flag = "  (single-run source)" if ph else ""
        print(f"  {lbl:14s}  {p:>10.2e}  AUROC={a:.3f}{flag}")
    print()
    print("=== bootstrap CIs (count) ===")
    cis = load_bootstrap_cis()
    if cis:
        print(f"  {len(cis)} entries with paired bootstrap CIs (n_boot=10000)")
        print(f"  source: results/dark_subspace/generated/sae_dark_subspace/all_models_bootstrap_cis.json")
        sample = next(iter(cis.values()))
        print(f"  sample: original CI=[{sample['original']['ci_lo']:.4f}, {sample['original']['ci_hi']:.4f}]"
              f", recon CI=[{sample['recon']['ci_lo']:.4f}, {sample['recon']['ci_hi']:.4f}]"
              f", residual CI=[{sample['residual']['ci_lo']:.4f}, {sample['residual']['ci_hi']:.4f}]")
        print(f"  callers: load_bootstrap_for_model(model_key) returns the per-model CI dict")
        print(f"  figures: plot_advanced_figures.py:fig2_sae_quality_scatter renders error bars from these")
    else:
        print("  no shipped CI table; figure scripts draw points without CI bands")
    sys.exit(0)
