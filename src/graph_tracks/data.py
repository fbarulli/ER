"""Explicit input contract and train-only graph vocabulary.

Input JSON deliberately excludes gtins, identity links, and raw text.
The relation/numeric schema is derived from the shared extractor contract
(core.sku_identity.graph_schema) rather than a hand-maintained subset,
so it stays aligned with the data the text track trains on. Splits are
supplied by the caller, never randomly generated here.

RESPONSIBILITY MAP (single-responsibility decomposition; behaviour pinned)
-------------------------------------------------------------------------
- :func:`file_size` — the one structural size accessor (files and
  directories): bytes on disk, never a content digest.
- :class:`ListingValidator` — the per-listing schema checks
  (:func:`load_records` stays the public face).
- :class:`Vocabulary` — the train-split-only categorical vocabulary
  (:func:`fit_vocabulary`).
- :class:`GraphBatch` — the batch payload (unchanged dataclass).
- :class:`BatchTensorizer` — records -> dense numeric rows + per-relation
  membership edges (:func:`tensorize` stays the public face).
- :class:`TextCache` — the checkpoint-native embedding cache contract
  (:func:`load_text_cache`).
- :func:`census` — the representation-policy census.

TRACE ROWS (core.tracing, the ONE consolidated trace)
-----------------------------------------------------
Stage ``graph_data``. These rows describe the graph inputs' own contract, so
they are OPT-IN: the caller passes the consolidated-trace writer it already
owns (``setup.py`` and ``prepare.py`` do) and no row is written otherwise. That
is deliberate — ``tensorize``/``fit_vocabulary`` sit on the training hot path,
and a library that writes a file per call would be a performance defect, not
traceability. Rows:
  run   listings.validated      raw listings JSON -> contract-checked records
  run   vocabulary.fitted       train-split records -> train-only vocabulary
  run   census.representation   records -> the representation census
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from core.perf_switches import perf_enabled
from core.run_log import RunLogger
from core.sku_identity import ProductIdentity, graph_schema

_LOG = RunLogger(__name__)

#: The pipeline stage these rows belong to (core.tracing ``stage`` column).
STAGE = "graph_data"

# Defaults OFF: the full finite/norm scan stays on so corruption detection is
# unchanged. Opting in skips the O(rows*cols) scan on caches the data gate has
# already attested; the dtype and provenance guards below always run.
_CACHE_SKIP_FULL_SCAN = perf_enabled("graph.cache_skip_full_scan", default=False)

# DERIVED, never pinned: the graph carries whatever the shared extractor
# (`core.sku_identity.row_identity`) yields — string descriptor fields are
# typed relations, float fields numeric features (see ProductIdentity
# .graph_schema). An update to the extractor's descriptor set moves this
# automatically; a changed descriptor set for previously prepared listings is caught at load
# time by comparing the prepared manifest against this derivation.
RELATIONS, NUMERIC = graph_schema()
SPLITS = {"train", "dev", "test"}


def file_size(path: Path | str) -> int:
    """The graph lane's structural size accessor (one home, no digest).

    Forwards to ``core.portable_archive.file_size``: a file's bytes on disk, or
    a directory's summed member bytes. Used as a change detector (frozen
    ablation sources, worker-package verification, git transport checks) — a
    same-size rewrite IS the accepted structural blind spot of the owner
    directive that removed all content hashing.
    """
    from core.portable_archive import file_size as _file_size
    return _file_size(path)


class ListingValidator:
    """The listing-JSON contract, one record at a time."""

    SCHEMA = "er-graph-listings-v1"
    KEYS = {"sku_id", "split", "attribute", "numeric"}

    def __init__(self, *, require_training: bool):
        self._require_training = require_training

    def allowed_splits(self) -> set[str]:
        return SPLITS if self._require_training else SPLITS | {"inference"}

    def check_frame(self, raw: dict) -> list[dict]:
        if raw.get("schema") != self.SCHEMA:
            raise ValueError("expected schema er-graph-listings-v1")
        records = raw["listings"]
        if not records:
            raise ValueError("empty listings")
        ids: list[str] = []
        for record in _LOG.progress(records, desc="listing_contract", unit="listing"):
            self.check_record(record)
            ids.append(record["sku_id"])
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate sku_id")
        if self._require_training and not any(r["split"] == "train" for r in records):
            raise ValueError("at least one training listing is required")
        return records

    def check_record(self, record: dict) -> None:
        if set(record) != self.KEYS:
            raise ValueError("listing keys must be sku_id, split, attributes, numeric")
        if not isinstance(record["sku_id"], str) or not record["sku_id"]:
            raise ValueError("sku_id must be a nonempty string")
        if record["split"] not in self.allowed_splits():
            raise ValueError(
                "split must be train/dev/test (inference is allowed for encoding only)"
            )
        if set(record["attribute"]) - set(RELATIONS):
            raise ValueError("unsupported attribute relation")
        if set(record["numeric"]) - set(NUMERIC):
            raise ValueError("unsupported numeric feature")
        self._check_values(record["attribute"], str_kind=True)
        self._check_values(record["numeric"], str_kind=False)

    @staticmethod
    def _check_values(values_by_field: dict, *, str_kind: bool) -> None:
        for values in values_by_field.values():
            if not isinstance(values, list):
                raise ValueError(
                    "attributes must contain lists of nonempty strings"
                    if str_kind else
                    "numeric fields must be lists of finite nonnegative numbers"
                )
            for value in values:
                if str_kind:
                    if not isinstance(value, str) or not value:
                        raise ValueError("attributes must contain lists of nonempty strings")
                elif (isinstance(value, bool) or not isinstance(value, (int, float))
                      or not np.isfinite(value) or value < 0):
                    raise ValueError("numeric fields must be lists of finite nonnegative numbers")


def load_records(
    path: Path, *, require_training: bool = True, trace=None
) -> list[dict]:
    """Load + validate the listing JSON (contract on :class:`ListingValidator`).

    ``trace`` is the caller's consolidated-trace writer (core.tracing.TraceRun);
    with it the validated population is recorded, with none nothing is written
    (this function is on the training path).
    """
    raw = json.loads(path.read_text())
    records = ListingValidator(require_training=require_training).check_frame(raw)
    if trace is not None:
        raw_listings = raw.get("listings") if isinstance(raw, dict) else None
        trace.add(
            "listings",
            "validated",
            in_count=len(raw_listings) if isinstance(raw_listings, list) else len(records),
            out_count=len(records),
            reason=(
                "every listing must carry exactly sku_id/split/attribute/numeric, "
                "a unique nonempty sku_id, a declared split and values from the "
                "shared extractor's schema"
            ),
            detail={
                "path": str(path),
                "schema": ListingValidator.SCHEMA,
                "require_training": bool(require_training),
                "splits": {
                    split: sum(1 for record in records if record["split"] == split)
                    for split in sorted({record["split"] for record in records})
                },
            },
            source=str(path),
        )
    return records


class Vocabulary:
    """Train-split-only categorical vocabulary (missing/unseen share token 0)."""

    @staticmethod
    def fit(records: list[dict]) -> dict[str, list[str]]:
        return {relation: sorted({v for r in records if r["split"] == "train"
                                 for v in r["attribute"].get(relation, [])})
                for relation in RELATIONS}


def fit_vocabulary(records: list[dict], trace=None) -> dict[str, list[str]]:
    """Train-split-only categorical vocabulary, optionally traced.

    The traced counts are NOT a funnel: one train listing carries SEVERAL distinct
    values, so the vocabulary is legitimately larger than the train population
    (measured live: 3 train listings -> 4 entries). Stating an in/out pair here
    made dropped_count negative, which the row contract forbids, so a unit change
    states only its output and flags the change.

    The rule is the shared one (``pipeline.unit_change_counts``). It is restated
    here as two lines rather than imported so this graph leaf module does not
    take a dependency on ``pipeline``; the arithmetic and the flag are identical.
    """
    vocabulary = Vocabulary.fit(records)
    if trace is not None:
        train_records = sum(1 for record in records if record["split"] == "train")
        entries = sum(len(values) for values in vocabulary.values())
        unit_change = entries > train_records
        trace.add(
            "vocabulary",
            "fitted",
            in_count=None if unit_change else train_records,
            out_count=entries,
            reason=(
                "only TRAIN-split listings feed the vocabulary; a value seen by "
                "several listings is one entry, and missing/unseen values share "
                "the unknown token at use time"
            ),
            detail={
                "listings": len(records),
                "train_listings": train_records,
                "vocabulary_entries": entries,
                "unit_change": unit_change,
                "relations": {
                    relation: len(values)
                    for relation, values in sorted(vocabulary.items())
                },
            },
            source="graph listing records",
        )
    return vocabulary


@dataclass
class GraphBatch:
    numeric: torch.Tensor
    # Each relation is (listing index, value index); zero is unknown/missing.
    edges: dict[str, tuple[torch.Tensor, torch.Tensor]]
    _pool_topology: dict[tuple[str, bool, int, torch.dtype], tuple] = field(
        default_factory=dict, init=False, repr=False)


class BatchTensorizer:
    """Records -> dense rows + per-relation membership edges.

    Dense row: per numeric field, log1p min/max and presence; interior values
    omitted. Edges: deduplicated sorted value indices; missing and unseen
    values use the feature encoder's unknown token, but NEVER share graph
    context through an unknown-value hub.
    """

    def __init__(self, vocabulary: dict[str, list[str]]):
        self._vocabulary = vocabulary

    @staticmethod
    def dense_numbers(records: list[dict]) -> torch.Tensor:
        numbers = []
        for record in _LOG.progress(records, desc="graph_dense_rows", unit="listing"):
            row = []
            for field_name in NUMERIC:
                values = sorted(set(record["numeric"].get(field_name, [])))
                row.extend([float(np.log1p(min(values))) if values else 0.,
                            float(np.log1p(max(values))) if values else 0.,
                            float(bool(values))])
            numbers.append(row)
        return torch.tensor(numbers, dtype=torch.float32)

    def edges(self, records: list[dict]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        edges = {}
        for relation in RELATIONS:
            lookup = {v: i + 1 for i, v in enumerate(self._vocabulary[relation])}
            source, target = [], []
            for i, record in _LOG.progress(
                enumerate(records), desc=f"graph_edges_{relation}", unit="listing",
                total=len(records),
            ):
                values = sorted(set(record["attribute"].get(relation, [])))
                indices = sorted({lookup.get(v, 0) for v in values}) or [0]
                source.extend([i] * len(indices))
                target.extend(indices)
            edges[relation] = (torch.tensor(source, dtype=torch.long),
                               torch.tensor(target, dtype=torch.long))
        return edges

    def batch(self, records: list[dict]) -> GraphBatch:
        return GraphBatch(self.dense_numbers(records), self.edges(records))


def tensorize(records: list[dict], vocabulary: dict[str, list[str]], device: str) -> GraphBatch:
    # Dynamic inference and benchmark batches follow the same CPU-first
    # topology contract as persisted prepared inputs.
    from graph_tracks.pooling import move_batch, prime_batch
    batch = BatchTensorizer(vocabulary).batch(records)
    prime_batch(batch, vocabulary)
    return move_batch(batch, device)


class TextCache:
    """The checkpoint-native embedding cache contract.

    Persisted float32 rows, unique nonempty IDs, finite nonzero vectors, and
    a metadata block that pins ``checkpoint_size`` plus the composition —
    so a cache can never serve embeddings from a different checkpoint
    silently.
    """

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as cache:
            self._ids = cache["ids"].astype(str).tolist()
            self._vectors = np.asarray(cache["embeddings"])
            self._metadata = json.loads(str(cache["metadata"].item()))
        self._check()

    def _check(self) -> None:
        if len(set(self._ids)) != len(self._ids):
            raise ValueError("duplicate IDs in text cache")
        if (self._vectors.ndim != 2 or self._vectors.shape[0] != len(self._ids)
                or self._vectors.shape[1] == 0):
            raise ValueError("invalid text cache dimensions")
        if self._vectors.dtype != np.float32:
            raise ValueError('text cache embeddings must persist float32; silent dtype coercion is forbidden')
        if self._metadata.get('embedding_dtype', 'float32') != 'float32':
            raise ValueError('text cache embedding dtype attestation mismatch')
        if not _CACHE_SKIP_FULL_SCAN and (
                not np.isfinite(self._vectors).all()
                or np.any(np.linalg.norm(self._vectors, axis=1) <= 1e-12)):
            raise ValueError("text cache contains nonfinite or zero vectors")
        if not self._metadata.get("checkpoint_size") or not self._metadata.get("composition"):
            raise ValueError("text cache must identify checkpoint_size and composition")

    def gather(self, ids: list[str]) -> np.ndarray:
        lookup = {v: i for i, v in enumerate(self._ids)}
        missing = set(ids) - set(lookup)
        if missing:
            raise ValueError(f"text cache missing {len(missing)} listing IDs")
        return self._vectors[[lookup[v] for v in ids]]

    @property
    def metadata(self) -> dict:
        return self._metadata


def load_text_cache(path: Path, ids: list[str]) -> tuple[np.ndarray, dict]:
    cache = TextCache(path)
    return cache.gather(ids), cache.metadata


def census(records: list[dict], vocabulary: dict[str, list[str]], trace=None) -> dict:
    """The representation-policy census, optionally traced.

    The funnel is records -> their split assignments (every record has exactly
    one), so the counts close; the per-relation membership detail is what the
    census is for.
    """
    result = {
        "representation_policy": {
            "numeric": "log1p min/max and presence; interior values omitted",
            "categorical": "train vocabulary; missing and all unseen values share token zero; deduplicated",
            "unknown_graph_hub": False,
        },
        "numeric_interior_values_omitted": {
            field_name: sum(max(0, len(set(r['numeric'].get(field_name, []))) - 2) for r in records)
            for field_name in NUMERIC},
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
    if trace is not None:
        trace.add(
            "census",
            "representation",
            in_count=len(records),
            out_count=sum(result["splits"].values()),
            reason=(
                "every listing carries exactly one split, so the census closes; "
                "it records the representation policy and the per-relation "
                "membership/missing/unseen counts"
            ),
            detail=result,
            source="graph listing records + train vocabulary",
        )
    return result
