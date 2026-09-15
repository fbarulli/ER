"""Persistent HNSW catalog index with an explicit GTIN/SKU identity map."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_FILENAME = "catalog.hnsw"
EMBEDDINGS_FILENAME = "catalog_embeddings.npy"
MAPPING_FILENAME = "catalog_id_mapping.csv"
METADATA_FILENAME = "catalog_index_metadata.json"


def _hnswlib():
    try:
        import hnswlib
    except ImportError as exc:
        raise RuntimeError(
            "HNSW retrieval was requested but hnswlib is unavailable; "
            "install the pinned requirements.txt dependency"
        ) from exc
    return hnswlib


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError(f"HNSW embeddings must be a non-empty 2-D matrix: {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("HNSW embeddings contain a zero-length vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _ids_sha256(ids: Sequence[str]) -> str:
    payload = "\n".join(str(value) for value in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PersistentHnswIndex:
    """Build, persist, validate, load, and query a normalized-IP HNSW index."""

    def __init__(
        self,
        output_dir: Path,
        *,
        ef_construction: int,
        M: int,
        ef_search: int,
        space: str = "ip",
    ) -> None:
        if space != "ip":
            raise ValueError("catalog HNSW requires space='ip' for normalized embeddings")
        self.output_dir = Path(output_dir).resolve()
        self.ef_construction = int(ef_construction)
        self.M = int(M)
        self.ef_search = int(ef_search)
        self.space = space
        self.index = None
        self.ids: list[str] = []
        self.embeddings: np.ndarray | None = None

    @property
    def index_path(self) -> Path:
        return self.output_dir / INDEX_FILENAME

    def build(
        self,
        embeddings: np.ndarray,
        ids: Sequence[str],
        *,
        checkpoint: Path,
        model_name: str,
        preprocessing_fingerprint: str | None = None,
    ) -> dict[str, object]:
        matrix = normalize_embeddings(embeddings)
        normalized_ids = [str(value) for value in ids]
        if len(normalized_ids) != len(matrix):
            raise ValueError(
                f"HNSW ID/embedding count mismatch: {len(normalized_ids)} != {len(matrix)}"
            )
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("HNSW catalog IDs must be unique")

        hnswlib = _hnswlib()
        index = hnswlib.Index(space=self.space, dim=int(matrix.shape[1]))
        index.init_index(
            max_elements=len(normalized_ids),
            ef_construction=self.ef_construction,
            M=self.M,
        )
        labels = np.arange(len(normalized_ids), dtype=np.int64)
        index.add_items(matrix, labels)
        index.set_ef(self.ef_search)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        index.save_index(str(self.index_path))
        np.save(self.output_dir / EMBEDDINGS_FILENAME, matrix, allow_pickle=False)
        pd.DataFrame(
            {
                "ann_label": labels,
                # The indexed catalog identity is the GTIN/item identifier.
                # sku_id is retained explicitly so SKU-catalog indexes can use
                # the same artifact contract later without changing its shape.
                "sku_id": normalized_ids,
                "gtin": normalized_ids,
                "item_id": normalized_ids,
            }
        ).to_csv(self.output_dir / MAPPING_FILENAME, index=False)
        metadata: dict[str, object] = {
            "format_version": 1,
            "backend": "hnswlib",
            "model": str(model_name),
            "checkpoint": str(Path(checkpoint).resolve()),
            "count": len(normalized_ids),
            "dim": int(matrix.shape[1]),
            "metric": "inner_product_on_l2_normalized_vectors",
            "space": self.space,
            "normalized": True,
            "ef_construction": self.ef_construction,
            "M": self.M,
            "ef_search": self.ef_search,
            "id_sha256": _ids_sha256(normalized_ids),
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "files": {
                "index": INDEX_FILENAME,
                "embeddings": EMBEDDINGS_FILENAME,
                "mapping": MAPPING_FILENAME,
            },
        }
        (self.output_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.index = index
        self.ids = normalized_ids
        self.embeddings = matrix
        return metadata

    def load(
        self,
        *,
        ids: Sequence[str],
        dim: int | None = None,
        checkpoint: Path,
        model_name: str,
        preprocessing_fingerprint: str | None = None,
    ) -> dict[str, object]:
        """Load an artifact only when its identity and index contract match."""
        required = [
            self.index_path,
            self.output_dir / EMBEDDINGS_FILENAME,
            self.output_dir / MAPPING_FILENAME,
            self.output_dir / METADATA_FILENAME,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete persisted HNSW artifact: {missing}")
        metadata = json.loads(
            (self.output_dir / METADATA_FILENAME).read_text(encoding="utf-8")
        )
        if dim is None:
            try:
                dim = int(metadata["dim"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("persisted HNSW metadata has no valid dimension") from exc
        normalized_ids = [str(value) for value in ids]
        expected = {
            "backend": "hnswlib",
            "checkpoint": str(Path(checkpoint).resolve()),
            "model": str(model_name),
            "count": len(normalized_ids),
            "dim": int(dim),
            "space": self.space,
            "normalized": True,
            "ef_construction": self.ef_construction,
            "M": self.M,
            "id_sha256": _ids_sha256(normalized_ids),
            "preprocessing_fingerprint": preprocessing_fingerprint,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"persisted HNSW metadata is stale: {mismatches}")
        mapping = pd.read_csv(self.output_dir / MAPPING_FILENAME, dtype=str)
        mapping["ann_label"] = pd.to_numeric(mapping["ann_label"], errors="raise")
        mapped_ids = mapping.sort_values("ann_label")["gtin"].astype(str).tolist()
        if mapped_ids != normalized_ids:
            raise ValueError("persisted HNSW mapping does not match catalog order")
        embeddings = np.load(
            self.output_dir / EMBEDDINGS_FILENAME, allow_pickle=False
        )
        if embeddings.shape != (len(normalized_ids), int(dim)):
            raise ValueError(
                f"persisted embedding shape is stale: {embeddings.shape}"
            )
        hnswlib = _hnswlib()
        index = hnswlib.Index(space=self.space, dim=int(dim))
        index.load_index(str(self.index_path), max_elements=len(normalized_ids))
        index.set_ef(self.ef_search)
        self.index = index
        self.ids = normalized_ids
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        return metadata

    def query(self, query_embeddings: np.ndarray, *, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("HNSW index has not been built or loaded")
        if top_k < 1:
            raise ValueError("HNSW top_k must be positive")
        queries = normalize_embeddings(query_embeddings)
        k = min(int(top_k), len(self.ids))
        labels, distances = self.index.knn_query(queries, k=k)
        return np.asarray(labels, dtype=np.int64), np.asarray(distances, dtype=np.float32)


__all__ = [
    "EMBEDDINGS_FILENAME",
    "INDEX_FILENAME",
    "MAPPING_FILENAME",
    "METADATA_FILENAME",
    "PersistentHnswIndex",
    "normalize_embeddings",
]
