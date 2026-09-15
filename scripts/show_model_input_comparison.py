#!/usr/bin/env python3
"""Show original review-row fields beside the exact model payload text.

This is an audit view: it does not rerun inference.  It reconstructs the
same source/canonical text path used by ``predict_items.py`` and records the
pipeline additions separately so a reviewer can see what changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core.common import load_config
from core.structured_features import (
    append_text as append_structured_text,
    canonical_info as canonical_structured_info,
    sku_info as sku_structured_info,
)
from pipeline import canonical_model_text, clean_sku_text, strip_schema_words


def text(value: object) -> str:
    return str(value or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--canonicals", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    review = pd.read_csv(args.review, dtype=str, keep_default_na=False)
    data = pd.read_csv(args.dataset, dtype=str, keep_default_na=False).rename(
        columns={
            "sku_id": "product_id",
            "sku_name_eng": "title",
            "description_short_eng": "description",
            "breadcrumbs_eng": "category_path",
            "attribute": "attributes",
        }
    )
    canon = pd.read_csv(args.canonicals, dtype=str, keep_default_na=False)
    for col in ("title", "description", "category_path", "attributes", "brand", "category"):
        if col not in data:
            data[col] = ""

    source = data.assign(__id=data["product_id"].astype(str)).set_index("__id").to_dict("index")
    targets = {}
    gtin_col = "barcode" if "barcode" in data else "gtin"
    for gtin, group in data.groupby(gtin_col, sort=False):
        gtin = text(gtin)
        if gtin:
            targets[gtin] = group.iloc[0].to_dict()
    canonical_map = canon.set_index(canon["gtin"].astype(str)).to_dict("index")
    cfg = load_config()["training"]["structured_features"]
    structured_enabled = bool(cfg["enabled"])
    structured_text = structured_enabled and bool(cfg["append_to_text"])

    rows: list[dict[str, object]] = []
    blocks: list[str] = []
    for _, item in review.head(args.limit).iterrows():
        sku_id = text(item["SKU_ID"])
        target_id = text(item["NEAREST_ITEM_ID"])
        source_row = source.get(sku_id, {})
        canonical_row = canonical_map.get(target_id, {})
        target_row = targets.get(target_id, {})

        raw_source = {
            "product_id": sku_id,
            "title": text(source_row.get("title")),
            "brand": text(source_row.get("brand")),
            "description": text(source_row.get("description")),
            "category": text(source_row.get("category")),
            "breadcrumbs": text(source_row.get("category_path")),
            "attributes": text(source_row.get("attributes")),
        }
        base_source = clean_sku_text(
            raw_source["title"], raw_source["attributes"], raw_source["brand"],
            raw_source["description"], raw_source["category"], raw_source["breadcrumbs"],
        )
        cleaned_source = strip_schema_words(base_source)
        source_info = sku_structured_info(raw_source["title"], raw_source["attributes"])
        source_model = append_structured_text(cleaned_source, source_info, enabled=structured_text)

        canonical_base = " ".join(
            text(canonical_row.get(col))
            for col in ("canonical", "mode_brand", "mode_type", "description_evidence", "breadcrumb_evidence")
        )
        canonical_clean = strip_schema_words(canonical_model_text(canonical_base))
        target_info = canonical_structured_info(canonical_row)
        target_model = append_structured_text(canonical_clean, target_info, enabled=structured_text)

        row = {
            "SKU_ID": sku_id,
            "NEAREST_ITEM_ID": target_id,
            "SCORE": text(item.get("SCORE")),
            "source_original_features": raw_source,
            "source_base_cleaned": base_source,
            "source_after_schema_strip": cleaned_source,
            "source_structured_info": source_info,
            "source_exact_model_text": source_model,
            "target_original_features": {
                "gtin": target_id,
                "title": text(target_row.get("title")),
                "brand": text(target_row.get("brand")),
                "description": text(target_row.get("description")),
                "category": text(target_row.get("category")),
                "breadcrumbs": text(target_row.get("category_path")),
                "attributes": text(target_row.get("attributes")),
            },
            "target_canonical_record": {
                k: text(canonical_row.get(k))
                for k in ("canonical", "mode_brand", "mode_type", "description_evidence", "breadcrumb_evidence", "volume_set", "pack_set", "package_type_set")
            },
            "target_base_canonical": canonical_base,
            "target_after_canonical_cleaning": canonical_clean,
            "target_structured_info": target_info,
            "target_exact_model_text": target_model,
        }
        rows.append(row)
        blocks.append(
            "\n".join([
                "=" * 100,
                f"ROW {len(rows)} | SKU_ID={sku_id} | candidate GTIN={target_id} | score={text(item.get('SCORE'))}",
                "SOURCE — ORIGINAL FEATURES",
                *(f"  {k}: {v}" for k, v in raw_source.items()),
                "SOURCE — PIPELINE CHANGES",
                f"  base cleaned:       {base_source}",
                f"  schema words removed: {cleaned_source}",
                f"  structured info:    {source_info}",
                "SOURCE — EXACT MODEL INPUT",
                f"  {source_model}",
                "TARGET — ORIGINAL FEATURES (representative raw row)",
                *(f"  {k}: {v}" for k, v in row["target_original_features"].items()),
                "TARGET — CANONICAL RECORD AND PIPELINE CHANGES",
                *(f"  {k}: {v}" for k, v in row["target_canonical_record"].items()),
                f"  canonical base:    {canonical_base}",
                f"  canonical cleaned: {canonical_clean}",
                f"  structured info:   {target_info}",
                "TARGET — EXACT MODEL INPUT",
                f"  {target_model}",
            ])
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "model_input_comparison.csv", index=False)
    (args.out_dir / "model_input_comparison.txt").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print("\n\n".join(blocks))
    print(f"\n[saved] {args.out_dir / 'model_input_comparison.txt'}")
    print(f"[saved] {args.out_dir / 'model_input_comparison.csv'}")


if __name__ == "__main__":
    main()
