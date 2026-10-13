"""Resolve MNRL coverage sources from exact negative edges and copy lineage."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from core.schemas import MaskAuditEntry


class MnrlProvenance(BaseModel):
    """Source identities in the frozen plan's payload-index space."""

    sources_by_edge: dict[tuple[int, int], str]
    originals_by_positive_copy: dict[tuple[int, int], int] = Field(default_factory=dict)
    sources_by_twin: dict[tuple[int, int], str] = Field(default_factory=dict)

    def populations(self, triples: Sequence[tuple[int, int, int]]) -> list[str]:
        return [self.source(int(anchor), int(positive), int(negative))
                for anchor, positive, negative in triples]

    def source(self, anchor: int, positive: int, negative: int) -> str:
        edge = (anchor, negative)
        if edge in self.sources_by_edge:
            return self.sources_by_edge[edge]
        original = self.originals_by_positive_copy.get((anchor, positive), anchor)
        edge = (original, negative)
        if edge in self.sources_by_twin:
            return self.sources_by_twin[edge]
        if edge not in self.sources_by_edge:
            raise ValueError(
                f"MNRL triple {(anchor, positive, negative)} has no negative-edge provenance: {edge}")
        return self.sources_by_edge[edge]


class MnrlProvenanceFactory:
    """Build one typed source map without conflating shared edge endpoints."""

    @classmethod
    def build(
        cls, triples: Sequence[tuple[int, int, int]], train_neg: np.ndarray,
        train_neg_sources: np.ndarray | None, *,
        mask_audit: Sequence[dict[str, Any]] | None = None,
        hard_negative_mask_audit: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        from training.training import _base_population_tag

        provenance = MnrlProvenance(sources_by_edge=cls._sources(
            train_neg, train_neg_sources, _base_population_tag))
        # Prepared-bundle schema owns audit validation; these are trusted internal rows.
        for row in mask_audit or ():
            audit = MaskAuditEntry.model_construct(**row)
            positive = (audit.copy_pair_payload_idx if audit.copy_pair_payload_idx is not None
                        else audit.pair_payload_idx)
            key = (audit.copy_payload_idx, positive)
            original = audit.anchor_payload_idx
            if (key in provenance.originals_by_positive_copy
                    and provenance.originals_by_positive_copy[key] != original):
                raise ValueError(f"conflicting MNRL positive-copy provenance: {key}")
            provenance.originals_by_positive_copy[key] = original
        cls._twins(provenance, hard_negative_mask_audit or ())
        return provenance.populations(triples)

    @staticmethod
    def _sources(
        train_neg: np.ndarray, train_neg_sources: np.ndarray | None,
        normalize: Callable[[str], str],
    ) -> dict[tuple[int, int], str]:
        pairs = np.asarray(train_neg, dtype=int).reshape(-1, 2)
        if train_neg_sources is None or len(train_neg_sources) != len(pairs):
            raise ValueError("MNRL negative source rows must match the negative edges")
        sources: dict[tuple[int, int], str] = {}
        for (anchor, negative), raw in zip(pairs, train_neg_sources, strict=True):
            edge = (int(anchor), int(negative))
            source = normalize(str(raw))
            if edge in sources and sources[edge] != source:
                raise ValueError(f"conflicting MNRL negative-edge provenance: {edge}")
            sources[edge] = source
        return sources

    @staticmethod
    def _twins(provenance: MnrlProvenance, rows: Sequence[dict[str, Any]]) -> None:
        for row in rows:
            audit = MaskAuditEntry.model_construct(**row)
            if audit.target_mode != "counterfactual":
                continue
            edge = (audit.copy_payload_idx, audit.pair_payload_idx)
            if edge not in provenance.sources_by_edge:
                continue  # Audits span folds; only surviving train edges license twins.
            anchors = {audit.anchor_payload_idx}
            if audit.copy_source_payload_idx == audit.pair_payload_idx:
                anchors.add(audit.pair_payload_idx)
            source = provenance.sources_by_edge[edge]
            for anchor in anchors:
                twin = (anchor, audit.copy_payload_idx)
                if twin in provenance.sources_by_twin and provenance.sources_by_twin[twin] != source:
                    raise ValueError(f"conflicting MNRL twin provenance: {twin}")
                provenance.sources_by_twin[twin] = source
