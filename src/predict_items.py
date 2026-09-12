"""Generate SKU_ID -> ITEM_ID predictions from a fine-tuned SentenceTransformer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from core.common import load_dataset_deduped
from pipeline import canonical_model_text, clean_sku_text, load_canonical_map, strip_schema_words


def _text_for_sku(row: pd.Series) -> str:
    return strip_schema_words(
        clean_sku_text(row.get("title", ""), row.get("attr", ""))
    )


def _text_for_item(canonical: object) -> str:
    return strip_schema_words(canonical_model_text(canonical))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Fine-tuned SentenceTransformer directory")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--input", default=None, help="Optional SKU input CSV")
    parser.add_argument("--sample", type=int, default=None, help="Deterministic sample size from the full SKU data")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Minimum cosine score for assigning a canonical ITEM_ID",
    )
    args = parser.parse_args()

    if args.input:
        skus = pd.read_csv(args.input)
    else:
        skus = load_dataset_deduped()
    if args.sample is not None:
        if args.sample <= 0:
            raise ValueError("--sample must be positive")
        skus = skus.sample(n=min(args.sample, len(skus)), random_state=42).reset_index(drop=True)
    if "SKU_ID" not in skus.columns and "product_id" in skus.columns:
        skus = skus.rename(columns={"product_id": "SKU_ID"})
    if "SKU_ID" not in skus.columns:
        raise ValueError("Input data must contain SKU_ID or product_id")

    canonical = load_canonical_map()
    item_ids = list(canonical.keys())
    if not item_ids:
        raise ValueError("Canonical item map is empty")

    model = SentenceTransformer(str(Path(args.model)))
    sku_texts = skus.apply(_text_for_sku, axis=1).tolist()
    item_texts = [_text_for_item(canonical[item_id]) for item_id in item_ids]
    sku_embeddings = model.encode(
        sku_texts,
        batch_size=128,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    item_embeddings = model.encode(
        item_texts,
        batch_size=128,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    nearest = util.semantic_search(sku_embeddings, item_embeddings, top_k=1)

    predictions = []
    for sku_id, hits in zip(skus["SKU_ID"].astype(str), nearest):
        hit = hits[0]
        item_id = item_ids[hit["corpus_id"]]
        if float(hit["score"]) < args.threshold:
            item_id = f"UNMATCHED_{sku_id}"
        predictions.append((sku_id, str(item_id)))

    output = pd.DataFrame(predictions, columns=["SKU_ID", "ITEM_ID"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
