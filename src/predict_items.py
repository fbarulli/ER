"""Generate SKU_ID -> ITEM_ID predictions from a fine-tuned SentenceTransformer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from core.common import (
    F,
    load_config,
    load_dataset_deduped,
    rand_matching_cfg,
)
from core.structured_features import (
    append_text as append_structured_text,
    canonical_info as canonical_structured_info,
    fuse_numpy,
    sku_info as sku_structured_info,
    vector as structured_vector,
)
from pipeline import canonical_model_text, clean_sku_text, load_canonical_map, strip_schema_words


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

    sf_cfg = load_config()["training"]["structured_features"]
    sf_enabled = bool(sf_cfg["enabled"])
    sf_text = sf_enabled and bool(sf_cfg["append_to_text"])
    sf_weight = (
        float(sf_cfg["embedding_weight"])
        if sf_enabled and bool(sf_cfg["feed_to_loss"])
        else 0.0
    )
    canonical_records = pd.read_csv(
        F["canonical_records"], dtype={"gtin": str}, keep_default_na=False
    )
    canonical_record_map = {
        str(row["gtin"]): row.to_dict()
        for _, row in canonical_records.iterrows()
    }
    sku_infos = [
        sku_structured_info(
            row.get("title", ""), row.get("attributes", row.get("attr", ""))
        )
        if sf_enabled
        else {"volume": set(), "pack": set()}
        for _, row in skus.iterrows()
    ]
    item_infos = [
        canonical_structured_info(canonical_record_map.get(item_id, {}))
        if sf_enabled
        else {"volume": set(), "pack": set()}
        for item_id in item_ids
    ]
    sku_texts = [
        append_structured_text(
            strip_schema_words(
                clean_sku_text(
                    row.get("title", ""),
                    row.get("attributes", row.get("attr", "")),
                )
            ),
            info,
            enabled=sf_text,
        )
        for (_, row), info in zip(skus.iterrows(), sku_infos, strict=True)
    ]
    item_texts = [
        append_structured_text(
            strip_schema_words(canonical_model_text(canonical[item_id])),
            info,
            enabled=sf_text,
        )
        for item_id, info in zip(item_ids, item_infos, strict=True)
    ]
    model = SentenceTransformer(str(Path(args.model)))
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
    sku_features = np.asarray(
        [
            structured_vector(
                info,
                volume_scale_ml=float(sf_cfg["volume_scale_ml"]),
                pack_scale=float(sf_cfg["pack_scale"]),
                max_set_size=int(sf_cfg["max_set_size"]),
            )
            for info in sku_infos
        ],
        dtype=np.float32,
    )
    item_features = np.asarray(
        [
            structured_vector(
                info,
                volume_scale_ml=float(sf_cfg["volume_scale_ml"]),
                pack_scale=float(sf_cfg["pack_scale"]),
                max_set_size=int(sf_cfg["max_set_size"]),
            )
            for info in item_infos
        ],
        dtype=np.float32,
    )
    sku_embeddings = fuse_numpy(sku_embeddings.cpu().numpy(), sku_features, sf_weight)
    item_embeddings = fuse_numpy(item_embeddings.cpu().numpy(), item_features, sf_weight)
    sku_embeddings = torch.as_tensor(sku_embeddings)
    item_embeddings = torch.as_tensor(item_embeddings)
    nearest = util.semantic_search(sku_embeddings, item_embeddings, top_k=1)

    unmatched_prefix = rand_matching_cfg()["unmatched_prefix"]
    predictions = []
    for sku_id, hits in zip(skus["SKU_ID"].astype(str), nearest, strict=True):
        hit = hits[0]
        item_id = item_ids[hit["corpus_id"]]
        if float(hit["score"]) < args.threshold:
            item_id = f"{unmatched_prefix}{sku_id}"
        predictions.append((sku_id, str(item_id)))

    output = pd.DataFrame(predictions, columns=["SKU_ID", "ITEM_ID"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
