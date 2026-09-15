#!/usr/bin/env python3
"""Export ANN embeddings and optionally publish an interactive Nomic Atlas map.

This is an optional post-run diagnostic. It does not alter training, inference,
or the required runtime. Set NOMIC_API_KEY and pass --upload to create an Atlas
data map; without --upload it only writes a local .npy matrix and metadata CSV.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.common import load_config, load_local_sentence_transformer
from core.model_input import (
    build_sku_text,
    model_input_info,
    model_input_composition,
)
from core.structured_features import (
    fuse_numpy,
    sku_info as sku_structured_info,
    vector as structured_vector,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="fine-tuned SentenceTransformer directory")
    parser.add_argument("--input", required=True, type=Path, help="CSV containing SKU/product rows")
    parser.add_argument("--output-dir", type=Path, default=Path("results/atlas_embeddings"))
    parser.add_argument("--name", default="euromonitor-ann-embeddings")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--predictions", type=Path, help="optional prediction CSV to join as metadata")
    parser.add_argument("--upload", action="store_true", help="publish the map to Nomic Atlas")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if args.sample is not None:
        if args.sample <= 0:
            raise ValueError("--sample must be positive")
        frame = frame.head(args.sample).copy()
    id_column = "SKU_ID" if "SKU_ID" in frame.columns else "product_id"
    if id_column not in frame.columns:
        raise ValueError("input must contain SKU_ID or product_id")

    cfg = load_config()["training"]["structured_features"]
    enabled = bool(cfg["enabled"])
    rows = frame.to_dict("records")
    # The encoder text comes from the shared composition SSOT.  This script used
    # to build its own copy and called the six-argument `clean_sku_text` with
    # two, so brand/description/category/breadcrumbs silently became "" — a
    # composition that matched neither profile.
    infos = [
        model_input_info(
            sku_structured_info(
                row.get("title", ""), row.get("attributes", row.get("attr", ""))
            )
        )
        if enabled
        else {"volume": set(), "pack": set()}
        for row in rows
    ]
    texts = [
        build_sku_text(pd.Series(row), info)
        for row, info in zip(rows, infos, strict=True)
    ]
    model = load_local_sentence_transformer(args.model, device=args.device)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    if enabled and bool(cfg["feed_to_loss"]):
        features = np.asarray([
            structured_vector(
                info,
                volume_scale_ml=float(cfg["volume_scale_ml"]),
                pack_scale=float(cfg["pack_scale"]),
                max_set_size=int(cfg["max_set_size"]),
            )
            for info in infos
        ], dtype=np.float32)
        embeddings = fuse_numpy(embeddings, features, float(cfg["embedding_weight"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = args.output_dir / "embeddings.npy"
    metadata_path = args.output_dir / "metadata.csv"
    np.save(embedding_path, embeddings)
    metadata = frame.copy()
    metadata.insert(0, "atlas_id", metadata[id_column].astype(str))
    # The matrix is only interpretable together with the composition that
    # produced it, so the artifact names its own input contract.
    provenance = model_input_composition()
    metadata.insert(1, "model_input_profile", provenance.profile)
    metadata.insert(2, "model_input_include_evidence", provenance.include_evidence)
    if args.predictions:
        predictions = pd.read_csv(args.predictions, dtype=str, keep_default_na=False)
        join_key = "SKU_ID" if "SKU_ID" in predictions.columns else "product_id"
        if join_key in predictions.columns:
            metadata = metadata.merge(
                predictions.drop_duplicates(join_key),
                left_on=id_column,
                right_on=join_key,
                how="left",
                suffixes=("", "_prediction"),
            )
    metadata.to_csv(metadata_path, index=False)
    print(f"[atlas] wrote {len(metadata):,} rows, dimension={embeddings.shape[1]}")
    print(f"[atlas] embeddings={embedding_path} metadata={metadata_path}")

    if not args.upload:
        return
    try:
        from nomic import AtlasDataset, login
    except ImportError as exc:
        raise RuntimeError("install the optional Atlas dependency with: uv sync --extra atlas") from exc
    api_key = os.environ.get("NOMIC_API_KEY")
    if api_key:
        login(api_key)
    dataset = AtlasDataset(args.name, description="EuromonitoR ANN model embeddings")
    dataset.add_data(
        embeddings=embeddings,
        data=metadata.to_dict("records"),
    )
    data_map = dataset.create_index()
    print(f"[atlas] map created: {data_map}")


if __name__ == "__main__":
    main()
