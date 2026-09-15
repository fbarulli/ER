#!/usr/bin/env python3
"""Show original review-row fields beside the exact model payload text.

This is an audit view: it does not rerun inference.  It reconstructs the
same source/canonical text path used by ``predict_items.py`` and records the
pipeline additions separately so a reviewer can see what changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.model_input import (
    build_canonical_text,
    build_sku_text,
    model_input_composition,
)
from core.structured_features import (
    canonical_info as canonical_structured_info,
    sku_info as sku_structured_info,
)
from pipeline import (
    canonical_evidence_text,
    canonical_model_text,
    clean_sku_text,
    strip_schema_words,
)


def text(value: object) -> str:
    return str(value or "").strip()


def human_value(value: object, *, evidence: bool = False) -> str:
    """Render audit metadata without Python dict/list/set repr syntax."""
    if evidence:
        return canonical_evidence_text(value) or "none"
    if isinstance(value, dict):
        return "; ".join(
            f"{key}={human_value(item)}" for key, item in value.items()
        )
    if isinstance(value, (set, frozenset, list, tuple)):
        rendered = [human_value(item) for item in value]
        return ", ".join(rendered) if rendered else "none"
    return text(value) or "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--canonicals", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    review = pd.read_csv(args.review, dtype=str, keep_default_na=False)
    raw_data = pd.read_csv(args.dataset, dtype=str, keep_default_na=False)
    data = raw_data.rename(
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
    raw_source = raw_data.assign(__id=raw_data["sku_id"].astype(str)).set_index("__id").to_dict("index")
    targets = {}
    gtin_col = "barcode" if "barcode" in data else "gtin"
    raw_gtin_col = "gtin" if "gtin" in raw_data else gtin_col
    raw_targets = {}
    for gtin, group in raw_data.groupby(raw_gtin_col, sort=False):
        gtin = text(gtin)
        if gtin:
            raw_targets[gtin] = group.iloc[0].to_dict()
    for gtin, group in data.groupby(gtin_col, sort=False):
        gtin = text(gtin)
        if gtin:
            targets[gtin] = group.iloc[0].to_dict()
    canonical_map = canon.set_index(canon["gtin"].astype(str)).to_dict("index")
    composition = model_input_composition()

    rows: list[dict[str, object]] = []
    blocks: list[str] = []
    for _, item in review.head(args.limit).iterrows():
        sku_id = text(item["SKU_ID"])
        target_id = text(item["NEAREST_ITEM_ID"])
        source_row = source.get(sku_id, {})
        canonical_row = canonical_map.get(target_id, {})
        target_row = targets.get(target_id, {})

        source_features = {
            "product_id": sku_id,
            **{key: value for key, value in raw_source.get(sku_id, {}).items() if not key.startswith("__")},
        }
        base_source = clean_sku_text(
            text(source_row.get("title")), text(source_row.get("attributes")), text(source_row.get("brand")),
            text(source_row.get("description")), text(source_row.get("category")), text(source_row.get("category_path")),
        )
        cleaned_source = strip_schema_words(base_source)
        source_info = sku_structured_info(text(source_row.get("title")), text(source_row.get("attributes")))
        # The EXACT payload the encoder receives comes from the shared builder,
        # so this audit view can never disagree with what the lanes actually
        # feed the model. The intermediate columns below document the ORIGINAL
        # (legacy) composition steps and are explanatory only.
        source_model = build_sku_text(pd.Series(source_row), source_info)

        canonical_base = " ".join((
            text(canonical_row.get("canonical")),
            text(canonical_row.get("mode_brand")),
            text(canonical_row.get("mode_type")),
            canonical_evidence_text(canonical_row.get("description_evidence", "")),
            canonical_evidence_text(canonical_row.get("breadcrumb_evidence", "")),
        ))
        canonical_clean = strip_schema_words(canonical_model_text(canonical_base))
        target_info = canonical_structured_info(canonical_row)
        target_model = build_canonical_text(canonical_row, target_info)

        row = {
            "SKU_ID": sku_id,
            "NEAREST_ITEM_ID": target_id,
            "model_input_composition": composition.model_dump_json(),
            "SCORE": text(item.get("SCORE")),
            "source_original_features": source_features,
            "source_base_cleaned": base_source,
            "source_after_schema_strip": cleaned_source,
            "source_structured_info": source_info,
            "source_exact_model_text": source_model,
            "target_original_features": {
                **{key: value for key, value in raw_targets.get(target_id, {}).items() if not key.startswith("__")},
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
                f"MODEL INPUT COMPOSITION: {composition.model_dump_json()}",
                "SOURCE — ORIGINAL FEATURES",
                *(f"  {k}: {v}" for k, v in source_features.items()),
                "SOURCE — PIPELINE CHANGES",
                f"  base cleaned:       {base_source}",
                f"  schema words removed: {cleaned_source}",
                f"  structured info:    {human_value(source_info)}",
                "SOURCE — EXACT MODEL INPUT",
                f"  {source_model}",
                "TARGET — ORIGINAL FEATURES (representative raw row)",
                *(f"  {k}: {v}" for k, v in row["target_original_features"].items()),
                "TARGET — CANONICAL RECORD AND PIPELINE CHANGES",
                *(f"  {k}: {v}" for k, v in row["target_canonical_record"].items()),
                f"  canonical base:    {canonical_base}",
                f"  canonical cleaned: {canonical_clean}",
                f"  structured info:   {human_value(target_info)}",
                "TARGET — EXACT MODEL INPUT",
                f"  {target_model}",
            ])
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "model_input_comparison.csv", index=False)
    (args.out_dir / "model_input_comparison.txt").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    markdown: list[str] = [
        "# Model input comparison",
        "",
        "Each review row is shown independently. The model-input blocks are exact strings; no fields are abbreviated.",
        f"Model-input composition in force: `{composition.model_dump_json()}`. The exact-input blocks are produced by `core.model_input`; the intermediate steps describe the original (legacy) composition and are explanatory only.",
        "",
    ]
    for number, row in enumerate(rows, start=1):
        markdown.extend([
            f"## Row {number}: `{row['SKU_ID']}` → `{row['NEAREST_ITEM_ID']}`",
            f"Score: `{row['SCORE']}`",
            "",
            "### Source: original features",
            "",
        ])
        markdown.extend(f"- **{key}:** {value}" for key, value in row["source_original_features"].items())
        markdown.extend([
            "",
            "### Source: pipeline changes",
            "",
            f"- **Base cleaned:** `{row['source_base_cleaned']}`",
            f"- **After schema-word removal:** `{row['source_after_schema_strip']}`",
            f"- **Structured fields extracted:** {human_value(row['source_structured_info'])}",
            "",
            "### Source: exact model input",
            "",
            "```text",
            str(row["source_exact_model_text"]),
            "```",
            "",
            "### Candidate target: original features",
            "",
        ])
        markdown.extend(f"- **{key}:** {value}" for key, value in row["target_original_features"].items())
        markdown.extend([
            "",
            "### Candidate target: canonical record and pipeline changes",
            "",
        ])
        markdown.extend(
            f"- **{key}:** {human_value(value, evidence=key in {'description_evidence', 'breadcrumb_evidence'})}"
            for key, value in row["target_canonical_record"].items()
        )
        markdown.extend([
            "",
            f"- **Canonical base:** `{row['target_base_canonical']}`",
            f"- **After canonical cleaning:** `{row['target_after_canonical_cleaning']}`",
            f"- **Structured fields extracted:** {human_value(row['target_structured_info'])}",
            "",
            "### Candidate target: exact model input",
            "",
            "```text",
            str(row["target_exact_model_text"]),
            "```",
            "",
        ])
    (args.out_dir / "model_input_comparison.md").write_text("\n".join(markdown), encoding="utf-8")
    print("\n\n".join(blocks))
    print(f"\n[saved] {args.out_dir / 'model_input_comparison.txt'}")
    print(f"[saved] {args.out_dir / 'model_input_comparison.csv'}")
    print(f"[saved] {args.out_dir / 'model_input_comparison.md'}")


if __name__ == "__main__":
    main()
