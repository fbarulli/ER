"""The cascade's candidate source: a union of WEAK blocking keys.

GTIN is the item key (AGENTS "The task"), and the task sheet is explicit that
brand, category and attributes are extracted/predicted by ML and imperfect, so
no single field is THE blocking key. This owner unions the keys declared in
``config/training.yaml`` ``candidates.keys`` over the REAL listings of the
prepared catalog, and hands the cascade one deterministic candidate list per
query. Building a candidate set asserts nothing: the scorer still decides which
candidate is the query's item.

Two keys are read through their canonical owners, never re-derived here:
``core.record_linkage.BrandBlocks.family_key`` (the alias-folded brand family,
which is what makes "A SHOC" and "Accelerator" one block) and, for the
attribute keys, ``core.text.attribute_field_value`` (the ONE ``Key: value``
cell reader), with the ``macro_category`` rollup from
``core.common.category_macros``.

Measured on ``data/track_setup/eligible_catalog.csv`` (27,709 real listings;
5,545 dev+test positives): the brand block alone puts the pair's item (its GTIN
class) among the query's candidates 0.9987 of the time, at a median 33
candidates per query; the declared union reaches 0.9998 at a median 429.

Virtual supervision endpoints (``canonical:*`` / ``augmentation:*``) are
training endpoints, not products, so they are never candidates: the text ANN
this source replaces holds only the real listings, and both sources must answer
over the same catalog population.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from core.columns import COLUMN_MAPPING
from core.common import category_macros, prepared_setup_layout, training_cfg
from core.record_linkage import BrandBlocks
from core.run_log import RunLogger
from core.schemas import CandidateKey
from core.text import attribute_field_value
from model_tracks.training_data import is_virtual_node

_LOG = RunLogger(__name__)

#: The prepared catalog's column names, from the ONE column SSOT
#: (``config/paths.yaml`` ``column_mapping``) instead of re-spelled literals.
_ID_COLUMN = COLUMN_MAPPING["sku_id"]
_BRAND_COLUMN = COLUMN_MAPPING["brand"]
_CATEGORY_COLUMN = COLUMN_MAPPING["category"]
_ATTRIBUTE_COLUMN = COLUMN_MAPPING["attribute"]

#: Separates a block key's declared name and its components: the unit
#: separator cannot occur in a normalized brand, category or attribute value,
#: and the name is always the first field, so no two distinct declarations can
#: spell one block key.
_COMPONENT_SEPARATOR = "\x1f"


class BlockedCandidates:
    """One built union of declared weak-key blocks over the real listings.

    Build once (:meth:`from_config` / :meth:`from_catalog`), then ask per
    listing: :meth:`candidates` returns the union of that listing's blocks
    without the listing itself, sorted. A listing that carries no component of
    any declared key has NO block; the empty answer is warned, never a shared
    empty block every such row silently joins.
    """

    def __init__(self, blocks: Mapping[str, Sequence[str]],
                 keys_of: Mapping[str, Sequence[str]]) -> None:
        self._blocks = {key: tuple(ids) for key, ids in blocks.items()}
        self._keys_of = {sku_id: tuple(keys) for sku_id, keys in keys_of.items()}

    # ── construction ───────────────────────────────────────────────────────
    @classmethod
    def from_config(cls) -> "BlockedCandidates":
        """Build over the config-declared prepared catalog and candidate keys."""
        from graph_tracks.setup import default_setup_dir

        layout = prepared_setup_layout()
        return cls.from_catalog(Path(default_setup_dir()) / layout.catalog,
                                training_cfg().candidates.keys)

    @classmethod
    def from_catalog(cls, catalog: Path,
                     keys: Sequence[Sequence[CandidateKey]]) -> "BlockedCandidates":
        """Build the union of the declared ``keys`` over one prepared catalog."""
        started = time.monotonic()
        frame = pd.read_csv(catalog, dtype=str, keep_default_na=False, low_memory=False)
        real = frame.loc[~frame[_ID_COLUMN].map(is_virtual_node)]
        sku_ids = real[_ID_COLUMN].tolist()
        values = cls._key_values(real, keys)
        blocks: dict[str, list[str]] = {}
        # Every real listing is a KNOWN query even when it joins no block; an
        # id absent from this mapping is not the catalog's at all.
        keys_of: dict[str, list[str]] = {sku_id: [] for sku_id in sku_ids}
        contributions: dict[str, int] = {}
        for components in keys:
            contributions["+".join(components)] = cls._add_key(
                blocks, keys_of, sku_ids, values, components)
        unblocked = sum(1 for joined in keys_of.values() if not joined)
        detail = " ".join(f"{name}:{count:,}" for name, count in contributions.items())
        report = (f"[candidates] catalog={catalog.name} real={len(sku_ids):,} "
                  f"blocks={len(blocks):,} unblocked={unblocked:,} keys={detail} "
                  f"elapsed={time.monotonic() - started:.1f}s")
        (_LOG.warning if unblocked else _LOG.info)(report)
        return cls(blocks, keys_of)

    @staticmethod
    def _add_key(blocks: dict[str, list[str]], keys_of: dict[str, list[str]],
                 sku_ids: Sequence[str], values: Mapping[CandidateKey, list[str]],
                 components: Sequence[CandidateKey]) -> int:
        """Add ONE declared key's blocks; return the rows that contributed.

        An absent component contributes NOTHING: the row joins no block of this
        key rather than a shared empty block every missing row falls into.
        """
        name = "+".join(components)
        columns = zip(*(values[component] for component in components))
        contributed = 0
        for sku_id, row in zip(sku_ids, columns):
            if not all(row):
                continue
            block = _COMPONENT_SEPARATOR.join((name, *row))
            blocks.setdefault(block, []).append(sku_id)
            keys_of[sku_id].append(block)
            contributed += 1
        return contributed

    @classmethod
    def _key_values(cls, frame: pd.DataFrame,
                    keys: Sequence[Sequence[CandidateKey]]) -> dict[CandidateKey, list[str]]:
        """Each DECLARED key's per-row value (a blank where the row lacks it)."""
        wanted = {component for components in keys for component in components}
        values: dict[CandidateKey, list[str]] = {}
        if "brand_family" in wanted:
            values["brand_family"] = BrandBlocks.family_key(frame[_BRAND_COLUMN]).tolist()
        if "macro_category" in wanted:
            macros = category_macros()
            values["macro_category"] = [macros.get(category, "")
                                        for category in frame[_CATEGORY_COLUMN]]
        for field in ("volume", "flavour"):
            if field in wanted:
                values[field] = [cls._attribute_value(attribute, field)
                                 for attribute in frame[_ATTRIBUTE_COLUMN]]
        return values

    @staticmethod
    def _attribute_value(attribute: str, field: str) -> str:
        """One declared attribute field's values as a sorted key component."""
        return " ".join(sorted(attribute_field_value(attribute, field)))

    # ── query ──────────────────────────────────────────────────────────────
    def candidates(self, sku_id: str) -> tuple[str, ...]:
        """The listing's unioned candidate list, itself excluded and sorted.

        A listing that carries no component of any declared key has no
        candidates: the empty answer is warned here so the gap stays visible.
        """
        keys = self._keys_of.get(sku_id)
        if keys is None:
            reason = ("a virtual supervision endpoint is not a catalog listing"
                      if is_virtual_node(sku_id) else "the prepared catalog does not carry it")
            raise KeyError(f"no candidates for {sku_id!r}: {reason}")
        if not keys:
            _LOG.warning(f"[candidates] {sku_id} carries no declared key value: "
                         "0 candidates (an absent key never forms a shared block)")
            return ()
        others = {other for key in keys for other in self._blocks[key]}
        others.discard(sku_id)
        return tuple(sorted(others))


__all__ = ["BlockedCandidates"]
