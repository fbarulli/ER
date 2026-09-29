#!/usr/bin/env python3
"""Build entity clusters for barcode-less rows (record linkage, no MPN).

MPN parsing is NOT applicable to this dataset: a full scan of the deduped
export found no manufacturer part numbers (only 29 rows mention any
MPN-style key, all false positives). This closes the gap for rows without a
valid GS1 barcode, which are invisible to the GTIN-based identity model
(core.blocking / the canonical build).

The linkage rule lives in core.record_linkage (reusable; reuses
pipeline.jaccard_similarity — no duplication). This script is a thin CLI that
persists the resulting row -> cluster map and census:

  --out  results/barcode_less_linkage.json   the linkage census + cluster stats
  --map  results/barcode_less_entity_clusters.csv  row (product_id) -> cluster

Usage:
  PYTHONPATH=src .venv/bin/python scripts/build_barcode_less_linkage.py \
      --data data/dataset_deduped.csv \
      --out results/barcode_less_linkage.json \
      --map results/barcode_less_entity_clusters.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from core.record_linkage import DEFAULT_JACCARD_THRESHOLD, link_barcode_less


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    # Default outputs resolve through the core.common RESULTS SSOT (honors
    # EUROMONITOR_RESULTS_DIR on Colab workers) — no bare "results/" literal.
    from core.common import RESULTS

    parser.add_argument(
        "--out", type=Path, default=RESULTS / "barcode_less_linkage.json"
    )
    parser.add_argument(
        "--map", type=Path, default=RESULTS / "barcode_less_entity_clusters.csv"
    )
    parser.add_argument("--jaccard", type=float, default=DEFAULT_JACCARD_THRESHOLD)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    if "product_id" not in df.columns:
        parser.error("input data must contain a product_id column")
    cluster_id, census = link_barcode_less(df, jaccard_threshold=args.jaccard)
    rows = [
        {"product_id": df.at[idx, "product_id"], "cluster_id": cid}
        for idx, cid in cluster_id.items()
    ]
    product_ids = [row["product_id"] for row in rows]
    if any(not product_id for product_id in product_ids):
        raise ValueError("linked rows must have non-empty product_id values")
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("linked rows must have unique product_id values")

    # Persist the census JSON.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(census, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Persist the row -> cluster map, keyed by product_id (stable identity).
    args.map.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["product_id", "cluster_id"]).to_csv(
        args.map, index=False
    )

    print(json.dumps(census, indent=2, ensure_ascii=False))
    print(f"[linkage] wrote {args.out} and {args.map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
