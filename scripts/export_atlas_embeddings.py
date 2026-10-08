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

from graph_tracks.data import file_size, load_text_cache
from graph_tracks.text_cache import checkpoint_size
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="optional checkpoint directory to verify saved provenance")
    parser.add_argument("--embeddings", required=True, type=Path, help="verified GPU-produced NPZ with IDs and frozen metadata")
    parser.add_argument("--input", required=True, type=Path, help="CSV containing SKU/product rows")
    parser.add_argument("--output-dir", type=Path, default=Path("results/atlas_embeddings"))
    parser.add_argument("--name", default="euromonitor-ann-embeddings")
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
    id_column = "SKU_ID" if "SKU_ID" in frame.columns else "sku_id"
    if id_column not in frame.columns:
        raise ValueError("input must contain SKU_ID or sku_id")

    ids = frame[id_column].astype(str).tolist()
    if any(not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError('input requires unique nonempty listing IDs')
    embeddings, provenance = load_text_cache(args.embeddings,ids)
    if not np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-4):
        raise ValueError('export vectors must be normalized')
    if args.model and checkpoint_size(Path(args.model)) != provenance['checkpoint_size']:
        raise ValueError('embedding checkpoint differs from requested model')
    # This diagnostic consumes the frozen vector artifact; model operations
    # belong to the prepared Colab GPU stage.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = args.output_dir / "embeddings.npy"
    metadata_path = args.output_dir / "metadata.csv"
    np.save(embedding_path, embeddings)
    metadata = frame.copy()
    metadata.insert(0, "atlas_id", metadata[id_column].astype(str))
    # The matrix is only interpretable together with the composition that
    # produced it, so the artifact names its own input contract.
    metadata.insert(1,"checkpoint_size",provenance['checkpoint_size'])
    (args.output_dir/'embedding_provenance.json').write_text(json.dumps({
        'source_size':file_size(args.embeddings),'metadata':provenance,
        'ids':ids,'embedding_dtype':'float32'},indent=2)+'\n')
    if args.predictions:
        predictions = pd.read_csv(args.predictions, dtype=str, keep_default_na=False)
        join_key = "SKU_ID" if "SKU_ID" in predictions.columns else "sku_id"
        if join_key in predictions.columns:
            if predictions[join_key].duplicated().any():
                raise ValueError('prediction rows require unique listing IDs; pair scores need explicit aggregation')
            metadata = metadata.merge(
                predictions,
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
