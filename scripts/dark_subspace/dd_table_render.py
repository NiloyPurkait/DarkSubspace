#!/usr/bin/env python3
"""dd_table_render.py.

Renders the extraction-detection separation tables `tab:dd_full`,
`tab:dd_extraction`, and `tab:epoch_dd` from cached per-condition JSON
records.

The analysis applies subspace erasure (Methods Eq. 3) to the knowledge
basis $\\mathbf{V}_K$ and the recall basis $\\mathbf{V}_R$, then measures
membership detection AUROC on the erased activation and verbatim
extraction (mean member loss, exact-match rate, ROUGE-L) on text
generated from the erased activation. Each per-condition JSON record
holds the AUROC and extraction metrics for one (model, intervention)
cell.

Used in Methods §3.5 (Interventions separate extraction from detection),
Results §4.4 (Feature edits do not close the residual membership gap),
and the corresponding appendix tables.

Reproduce:
    .venv/bin/python scripts/dark_subspace/dd_table_render.py \\
        --records-dir results/dark_subspace/generated/double_dissociation \\
        --table dd_full
    .venv/bin/python scripts/dark_subspace/dd_table_render.py \\
        --records-dir results/dark_subspace/generated/double_dissociation \\
        --table dd_extraction
    .venv/bin/python scripts/dark_subspace/dd_table_render.py \\
        --records-dir results/dark_subspace/generated/double_dissociation_epochs \\
        --table epoch_dd

Inputs.
  ``--records-dir``: directory containing one ``<model_tag>/results.json``
      per (model, intervention) cell. Each record is a flat JSON with
      schema::

          {
            "model_tag": str,
            "intervention": str,        # "none" | "erase_S_K" | "erase_S_R"
            "rank": int,                # n_c used for the erasure basis
            "membership_auroc": float,
            "membership_auroc_ci95_lo": float,
            "membership_auroc_ci95_hi": float,
            "extraction": {
                "mean_member_loss": float,
                "exact_match_rate": float,
                "rouge_l_mean": float,
                "rouge_l_ci95_lo": float,
                "rouge_l_ci95_hi": float
            },
            "epoch": int | null,        # set for epoch_dd; null otherwise
            ...                         # per-run provenance fields
          }

Outputs.
  Markdown table to stdout. The script does not regenerate the cached
  JSONs. Generating those records requires loading the fine-tuned model,
  fitting the channel-decomposition bases, applying the forward-pass
  erasure hook, decoding continuations, and scoring ROUGE-L; per-cell
  records are written under
  ``results/dark_subspace/generated/double_dissociation/``.

Notes.
  This renderer reads the cached records and prints. The forward-pass
  surgery, generation, and scoring that produced those records run on
  GPU outside this artefact.
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Dict, List


_TABLE_FIELDS: Dict[str, List[str]] = {
    "dd_full": [
        "model_tag",
        "intervention",
        "rank",
        "membership_auroc",
        "membership_auroc_ci95_lo",
        "membership_auroc_ci95_hi",
        "extraction.mean_member_loss",
        "extraction.exact_match_rate",
        "extraction.rouge_l_mean",
    ],
    "dd_extraction": [
        "model_tag",
        "intervention",
        "rank",
        "extraction.rouge_l_mean",
        "extraction.rouge_l_ci95_lo",
        "extraction.rouge_l_ci95_hi",
        "extraction.exact_match_rate",
    ],
    "epoch_dd": [
        "model_tag",
        "epoch",
        "intervention",
        "rank",
        "membership_auroc",
        "extraction.rouge_l_mean",
    ],
}


def _walk_records(records_dir: Path) -> List[Path]:
    return sorted(records_dir.glob("**/results.json"))


def _get(record: dict, dotted_key: str):
    cur = record
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _format_value(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_markdown(records_dir: Path, table: str) -> str:
    fields = _TABLE_FIELDS[table]
    paths = _walk_records(records_dir)
    if not paths:
        raise FileNotFoundError(
            f"No results.json files found under {records_dir}. "
            f"The expected path for these extraction-detection separation records is "
            f"results/dark_subspace/generated/double_dissociation/, regenerated from the GPU pipeline. "
            f"See the 'Outputs' section in this script's docstring."
        )

    rows = []
    for p in paths:
        record = json.loads(p.read_text())
        if table == "epoch_dd" and record.get("epoch") is None:
            continue
        if table != "epoch_dd" and record.get("epoch") is not None:
            continue
        rows.append({k: _get(record, k) for k in fields})

    rows.sort(
        key=lambda r: (
            str(r.get("model_tag", "")),
            int(r.get("epoch") or 0),
            str(r.get("intervention", "")),
            int(r.get("rank") or 0),
        )
    )

    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = "\n".join(
        "| " + " | ".join(_format_value(r.get(f)) for f in fields) + " |"
        for r in rows
    )
    return f"{header}\n{sep}\n{body}\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Render the extraction-detection separation tables (tab:dd_full, "
            "tab:dd_extraction, tab:epoch_dd) from cached per-condition JSONs."
        )
    )
    ap.add_argument("--records-dir", type=Path, required=True,
                    help="Directory containing per-cell results.json files.")
    ap.add_argument(
        "--table",
        choices=sorted(_TABLE_FIELDS.keys()),
        default="dd_full",
        help="Which table to render (default dd_full).",
    )
    args = ap.parse_args()

    print(render_markdown(args.records_dir, args.table))


if __name__ == "__main__":
    main()
