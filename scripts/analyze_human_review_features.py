#!/usr/bin/env python3
"""Join the human-review queue to original SKU/canonical evidence.

This is an exploratory audit: the queue has score bands, not human labels, so
the output identifies candidate separation rules but does not claim accuracy.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import pandas as pd

from pipeline import extract_all


TOKEN_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)?")


def as_set(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set, frozenset)):
            return {str(x).casefold() for x in parsed}
    except (SyntaxError, ValueError):
        pass
    return {x.casefold() for x in TOKEN_RE.findall(text)}


def words(*values: object) -> set[str]:
    return {
        token
        for value in values
        for token in TOKEN_RE.findall(str(value or "").casefold())
        if len(token) > 1
    }


def overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--canonicals", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    review = pd.read_csv(args.review, dtype=str, keep_default_na=False)
    data = pd.read_csv(args.dataset, dtype=str, keep_default_na=False)
    canon = pd.read_csv(args.canonicals, dtype=str, keep_default_na=False)
    # Accept both the raw export schema and the deduped schema.
    data = data.rename(columns={
        "sku_id": "product_id",
        "sku_name_eng": "title",
        "description_short_eng": "description",
        "breadcrumbs_eng": "category_path",
        "attribute": "attributes",
    })
    for column in ("title", "description", "category_path", "attributes", "brand", "category"):
        if column not in data:
            data[column] = ""
    data["__id"] = data["product_id"].astype(str)
    gtin_column = "barcode" if "barcode" in data else "gtin"
    data["__gtin_key"] = data[gtin_column].astype(str)
    canon["__gtin"] = canon["gtin"].astype(str)
    source = data.set_index("__id").to_dict("index")
    # Preserve all original rows for the target GTIN.  A canonical can have
    # multiple retailer/country offers, so retain deterministic counts and a
    # representative first row rather than silently dropping source fields.
    target_groups = {}
    for gtin, group in data.groupby("__gtin_key", sort=False):
        if not gtin or gtin.lower() == "nan":
            continue
        target_groups[gtin] = {
            "n_original_rows": len(group),
            **group.iloc[0].to_dict(),
        }
    target = canon.set_index("__gtin").to_dict("index")

    rows = []
    for _, item in review.iterrows():
        sku_id = str(item["SKU_ID"])
        target_id = str(item["NEAREST_ITEM_ID"])
        s = source.get(sku_id, {})
        t = target.get(target_id, {})
        t_original = target_groups.get(target_id, {})
        source_words = words(s.get("title"), s.get("description"), s.get("attributes"), s.get("category_path"))
        target_words = words(t.get("canonical"), t.get("mode_flavor"), t.get("mode_type"))
        extracted = extract_all(s.get("title", ""), s.get("attributes", "")) if s else {}
        source_volume = {str(extracted.get("volume_ml"))} if extracted.get("volume_ml", 0) else set()
        target_volume = as_set(t.get("volume_set"))
        source_pack = {str(extracted.get("pack_qty"))} if extracted.get("pack_confidence", 0) else set()
        target_pack = as_set(t.get("pack_set"))
        source_brand = str(s.get("brand", "")).casefold().strip()
        target_brand = str(t.get("mode_brand", "")).casefold().strip()
        enriched_row = {
            **item.to_dict(),
            "source_found": bool(s),
            "canonical_found": bool(t),
            "source_retailer": s.get("retailer", ""),
            "source_country": s.get("country", ""),
            "source_category": s.get("category", ""),
            "source_category_path": s.get("category_path", ""),
            "source_description": s.get("description", ""),
            "source_attributes": s.get("attributes", ""),
            "source_barcode": s.get("barcode", ""),
            "canonical_brand": t.get("mode_brand", ""),
            "canonical_flavor": t.get("mode_flavor", ""),
            "canonical_type": t.get("mode_type", ""),
            "canonical_volume_set": t.get("volume_set", ""),
            "canonical_pack_set": t.get("pack_set", ""),
            "brand_match": bool(source_brand and target_brand and source_brand == target_brand),
            "title_canonical_overlap": overlap(words(s.get("title")), target_words),
            "full_text_canonical_overlap": overlap(source_words, target_words),
            "source_title_words": " ".join(sorted(words(s.get("title")))),
            "canonical_words": " ".join(sorted(target_words)),
            "source_volume_evidence": " ".join(sorted(source_volume)),
            "source_pack_evidence": " ".join(sorted(source_pack)),
            "volume_conflict_known": bool(source_volume and target_volume and source_volume.isdisjoint(target_volume)),
            "pack_conflict_known": bool(source_pack and target_pack and source_pack.isdisjoint(target_pack)),
        }
        # Carry every field from the original source row and every field from
        # a representative original target row, including fields not used by
        # the current model (URLs, image URLs, price, retailer, country, etc.).
        for column, value in s.items():
            if not column.startswith("__"):
                enriched_row[f"source_original_{column}"] = value
        for column, value in t_original.items():
            if column not in {"__id", "__gtin_key"}:
                enriched_row[f"target_original_{column}"] = value
        rows.append(enriched_row)

    enriched = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.out_dir / "human_review_enriched.csv", index=False)

    inventory_rows = []
    for column in data.columns:
        if column.startswith("__"):
            continue
        values = data[column].astype(str)
        inventory_rows.append({
            "original_column": column,
            "rows": len(data),
            "nonempty": int(values.str.strip().ne("").sum()),
            "unique_values": int(values.nunique(dropna=False)),
            "sample": values[values.str.strip().ne("")].head(3).tolist(),
        })
    pd.DataFrame(inventory_rows).to_csv(args.out_dir / "original_feature_inventory.csv", index=False)

    numeric = [
        "SCORE", "title_word_count", "title_canonical_overlap",
        "full_text_canonical_overlap",
    ]
    for col in numeric:
        enriched[col] = pd.to_numeric(enriched[col], errors="coerce")
    summary = enriched.groupby("review_band", dropna=False).agg(
        rows=("SKU_ID", "size"),
        mean_score=("SCORE", "mean"),
        mean_title_overlap=("title_canonical_overlap", "mean"),
        mean_full_overlap=("full_text_canonical_overlap", "mean"),
        brand_match_rate=("brand_match", "mean"),
        source_found_rate=("source_found", "mean"),
        canonical_found_rate=("canonical_found", "mean"),
        known_volume_conflicts=("volume_conflict_known", "sum"),
        known_pack_conflicts=("pack_conflict_known", "sum"),
    ).reset_index()
    summary.to_csv(args.out_dir / "human_review_feature_summary_by_band.csv", index=False)

    missing = enriched[~enriched["source_found"] | ~enriched["canonical_found"]]
    missing[["SKU_ID", "NEAREST_ITEM_ID", "source_found", "canonical_found"]].to_csv(
        args.out_dir / "human_review_join_failures.csv", index=False
    )
    print(f"[done] enriched={len(enriched):,}; join_failures={len(missing):,}; output={args.out_dir}")


if __name__ == "__main__":
    main()
