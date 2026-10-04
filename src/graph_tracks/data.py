"""Explicit input contract and train-only graph vocabulary.

Input JSON deliberately excludes gtins, identity links, and raw text.
The relation/numeric schema is derived from the shared extractor contract
(core.sku_identity.graph_schema) rather than a hand-maintained subset,
so it stays aligned with the data the text track trains on. Splits are
supplied by the caller, never randomly generated here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core.sku_identity import ProductIdentity, graph_schema

# DERIVED, never pinned: the graph carries whatever the shared extractor
# (`core.sku_identity.row_identity`) yields — string descriptor fields are
# typed relations, float fields numeric features (see ProductIdentity
# .graph_schema). An update to the extractor's descriptor set moves this
# automatically; staleness of previously prepared listings is caught at load
# time by comparing the prepared manifest against this derivation.
RELATIONS, NUMERIC = graph_schema()
SPLITS = {"train", "dev", "test"}


def file_hash(path: Path | str) -> str:
    path = Path(path)
    if path.is_dir():
        from graph_tracks.text_cache import checkpoint_hash
        return checkpoint_hash(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(path: Path, *, require_training: bool = True) -> list[dict]:
    raw = json.loads(path.read_text())
    if raw.get("schema") != "er-graph-listings-v1":
        raise ValueError("expected schema er-graph-listings-v1")
    records = raw["listings"]
    if not records:
        raise ValueError("empty listings")
    ids = []
    for record in records:
        if set(record) != {"sku_id", "split", "attribute", "numeric"}:
            raise ValueError("listing keys must be sku_id, split, attributes, numeric")
        if not isinstance(record["sku_id"], str) or not record["sku_id"]:
            raise ValueError("sku_id must be a nonempty string")
        ids.append(record["sku_id"])
        allowed_splits = SPLITS if require_training else SPLITS | {"inference"}
        if record["split"] not in allowed_splits:
            raise ValueError("split must be train/dev/test (inference is allowed for encoding only)")
        if set(record["attribute"]) - set(RELATIONS):
            raise ValueError("unsupported attribute relation")
        if set(record["numeric"]) - set(NUMERIC):
            raise ValueError("unsupported numeric feature")
        for values in record["attribute"].values():
            if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError("attributes must contain lists of nonempty strings")
        for values in record["numeric"].values():
            if not isinstance(values, list) or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not np.isfinite(v) or v < 0 for v in values
            ):
                raise ValueError("numeric fields must be lists of finite nonnegative numbers")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sku_id")
    if require_training and not any(r["split"] == "train" for r in records):
        raise ValueError("at least one training listing is required")
    return records


def fit_vocabulary(records: list[dict]) -> dict[str, list[str]]:
    return {relation: sorted({v for r in records if r["split"] == "train"
                             for v in r["attribute"].get(relation, [])})
            for relation in RELATIONS}


@dataclass
class GraphBatch:
    numeric: torch.Tensor
    # Each relation is (listing index, value index); zero is unknown/missing.
    edges: dict[str, tuple[torch.Tensor, torch.Tensor]]


def tensorize(records: list[dict], vocabulary: dict[str, list[str]], device: str) -> GraphBatch:
    numbers = []
    for record in records:
        row = []
        for field in NUMERIC:
            values = sorted(set(record["numeric"].get(field, [])))
            row.extend([float(np.log1p(min(values))) if values else 0.,
                        float(np.log1p(max(values))) if values else 0.,
                        float(bool(values))])
        numbers.append(row)
    edges = {}
    for relation in RELATIONS:
        lookup = {v: i + 1 for i, v in enumerate(vocabulary[relation])}
        source, target = [], []
        for i, record in enumerate(records):
            values = sorted(set(record["attribute"].get(relation, [])))
            # Missing and unseen values use the feature encoder's unknown token,
            # but NEVER share graph context through an unknown-value hub.
            indices = sorted({lookup.get(v, 0) for v in values}) or [0]
            source.extend([i] * len(indices))
            target.extend(indices)
        edges[relation] = (torch.tensor(source, dtype=torch.long, device=device),
                           torch.tensor(target, dtype=torch.long, device=device))
    return GraphBatch(torch.tensor(numbers, dtype=torch.float32, device=device), edges)


def load_text_cache(path: Path, ids: list[str]) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as cache:
        cached_ids = cache["ids"].astype(str).tolist()
        vectors = np.asarray(cache["embeddings"])
        metadata = json.loads(str(cache["metadata"].item()))
    if len(set(cached_ids)) != len(cached_ids):
        raise ValueError("duplicate IDs in text cache")
    if vectors.ndim != 2 or vectors.shape[0] != len(cached_ids) or vectors.shape[1] == 0:
        raise ValueError("invalid text cache dimensions")
    if vectors.dtype != np.float32:
        raise ValueError('text cache embeddings must persist float32; silent dtype coercion is forbidden')
    if metadata.get('embedding_dtype', 'float32') != 'float32':
        raise ValueError('text cache embedding dtype attestation mismatch')
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) <= 1e-12):
        raise ValueError("text cache contains nonfinite or zero vectors")
    if not metadata.get("checkpoint_sha256") or not metadata.get("composition"):
        raise ValueError("text cache must identify checkpoint_sha256 and composition")
    lookup = {v: i for i, v in enumerate(cached_ids)}
    missing = set(ids) - set(lookup)
    if missing:
        raise ValueError(f"text cache missing {len(missing)} listing IDs")
    return vectors[[lookup[v] for v in ids]], metadata


def census(records: list[dict], vocabulary: dict[str, list[str]]) -> dict:
    return {
        "representation_policy": {
            "numeric": "log1p min/max and presence; interior values omitted",
            "categorical": "train vocabulary; missing and all unseen values share token zero; deduplicated",
            "unknown_graph_hub": False,
        },
        "numeric_interior_values_omitted": {
            field: sum(max(0, len(set(r['numeric'].get(field, []))) - 2) for r in records)
            for field in NUMERIC},
        "listings": len(records),
        "splits": {s: sum(r["split"] == s for r in records) for s in sorted(SPLITS)},
        "relations": {rel: {
            "train_values": len(vocabulary[rel]),
            "membership_edges": sum(len(set(r["attribute"].get(rel, []))) for r in records),
            "missing_listings": sum(not r["attribute"].get(rel) for r in records),
            "unseen_values": len({v for r in records if r["split"] != "train"
                                  for v in r["attribute"].get(rel, [])} - set(vocabulary[rel])),
        } for rel in RELATIONS},
    }
