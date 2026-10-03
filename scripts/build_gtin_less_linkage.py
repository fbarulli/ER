#!/usr/bin/env python3
"""Build entity clusters for gtin-less rows (record linkage, no MPN).

MPN parsing is NOT applicable to this dataset: a full scan of the deduped
export found no manufacturer part numbers (only 29 rows mention any
MPN-style key, all false positives). This closes the gap for rows without a
valid GS1 gtin, which are invisible to the GTIN-based identity model
(core.blocking / the canonical build).

The linkage rule lives in core.record_linkage (reusable; reuses
pipeline.jaccard_similarity — no duplication). This script is a thin CLI that
persists the resulting row -> cluster map and census:

  --out  results/gtin_less_linkage.json   the linkage census + cluster stats
  --map  results/gtin_less_entity_clusters.csv  row (sku_id) -> cluster

Usage:
  PYTHONPATH=src .venv/bin/python scripts/build_gtin_less_linkage.py \
      --data data/dataset_deduped.csv \
      --out results/gtin_less_linkage.json \
      --map results/gtin_less_entity_clusters.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from core.record_linkage import (
    DEFAULT_JACCARD_THRESHOLD,
    corpus_idf_from_finalized,
    finalized_texts,
    item_uniqueness_from_finalized,
    link_gtin_less,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    # Default outputs resolve through the core.common RESULTS SSOT (honors
    # EUROMONITOR_RESULTS_DIR on Colab workers) — no bare "results/" literal.
    from core.common import RESULTS

    parser.add_argument(
        "--out", type=Path, default=RESULTS / "gtin_less_linkage.json"
    )
    parser.add_argument(
        "--map", type=Path, default=RESULTS / "gtin_less_entity_clusters.csv"
    )
    parser.add_argument("--jaccard", type=float, default=DEFAULT_JACCARD_THRESHOLD)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    if "sku_id" not in df.columns:
        parser.error("input data must contain a sku_id column")
    # ONE finalized-text build shared by linkage + uniqueness (the per-row
    # model composition is the expensive pass; never do it twice).
    finalized = finalized_texts(df)
    cluster_id, census = link_gtin_less(
        df, jaccard_threshold=args.jaccard, finalized=finalized
    )
    idf, unseen = corpus_idf_from_finalized(finalized)
    uniqueness = item_uniqueness_from_finalized(finalized, idf, unseen)
    rows = [
        {
            "sku_id": df.at[idx, "sku_id"],
            "cluster_id": cid,
            "uniqueness": uniqueness.get(idx, 0.0),
        }
        for idx, cid in cluster_id.items()
    ]
    sku_ids = [row["sku_id"] for row in rows]
    if any(not sku_id for sku_id in sku_ids):
        raise ValueError("linked rows must have non-empty sku_id values")
    if len(sku_ids) != len(set(sku_ids)):
        raise ValueError("linked rows must have unique sku_id values")

    # Persist the census JSON.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(census, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Persist the row -> cluster map, keyed by sku_id (stable identity).
    args.map.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["sku_id", "cluster_id"]).to_csv(
        args.map, index=False
    )

    print(json.dumps(census, indent=2, ensure_ascii=False))
    print(f"[linkage] wrote {args.out} and {args.map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
