"""One negative-supply lane: real partners first, minted only to top-up.

Owner ruling (2026-10-03): the attribute gate leaves the decision path —
it is attribute-driven, so it can never be the label source nor a feature.
It keeps running in SHADOW mode, only to compare "model alone" against
"the gate" on real pairs.

Negative supply, in order:
  1. REAL different-GTIN partners — a blocker (TF-IDF baseline; an
     embeddings stage arrives behind the same interface) ranks real
     other-GTIN rows for every anchor; a candidate supplies a negative
     when EXACTLY ONE whitelisted attribute differs and every other
     populated attribute agrees.
  2. MINTED partners — only for anchors that got NO real partner: one
     partner per anchor, minted with a SINGLE whitelisted token move
     (volume or flavor), the anchor text untouched, every row
     provenance-tagged.
Positives keep their real base x base pairs and receive the SAME
formatting-edit machinery applied to a copied pair, so "edited texture"
cannot predict the label. Evaluation is on real pairs only, grouped by
GTIN.

Entity level is a config decision (``MintSpec.entity_level``): ``gtin``
(default) means a volume move maps to a legitimately DIFFERENT real GTIN,
so a pack flip is a legitimate negative; ``sku`` means product identity
ignores size, so the whitelist drops volume and keeps flavor only.

The module only reads pre-baked CSV columns (canonical_records.csv /
gate_results.csv / labeled_pairs.csv); no live parser runs here, so the
parsing lane's files stay untouched.

Class map (one owner per responsibility):
  - SetColumnReader  — the canonical set-literal cell reader (memoized)
  - TokenMover       — the ONE whitelisted token move both labels share
  - CanonicalIndex   — canonical lookup by GTIN + corpus donor pools
  - AnchorIndex      — anchor-row lookups and the stripped-GTIN array
  - LaneAssembler    — the trainer bridge (negative swap + minted leaves)
  - FoldMap          — GTIN-grouped fold buckets over union components
  - NegativeSupply   — the orchestrator: block -> mine -> mint -> emit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import ClassVar, Literal, Mapping

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from core.disjoint_sets import DisjointSet
from core.run_log import RunLogger
from core.step_trace import timed

_LOG = RunLogger(__name__)

# ── population registry (every emitted row carries exactly one) ─────────────
POPULATION_BASE_POSITIVE = "base_positive"  # real pair, reviewed label 1
POPULATION_BASE_NEGATIVE = "base_negative"  # real pair, reviewed label 0
POPULATION_REAL_PARTNER = "real_partner"  # real different-gtin, one-diff
POPULATION_MINTED_PARTNER = "minted_partner"  # synthetic single whitelisted move
POPULATION_EDITED_POSITIVE = "edited_positive"  # copied positive, symmetric edits

_POPULATION_TYPE = Literal[
    "base_positive", "base_negative", "real_partner",
    "minted_partner", "edited_positive",
]

# whitelisted attribute dimension -> canonical_records.csv set column
_DIMENSION_COLUMNS: Mapping[str, str] = {
    "volume": "volume_set",
    "pack": "pack_set",
    "package_type": "package_type_set",
    "package_material": "package_material_set",
    "flavor": "flavor_set",
    "carbonation": "carbonation_set",
    "sweetener": "sweetener_set",
}
_DIFF_DIMENSIONS = tuple(_DIMENSION_COLUMNS)


class SupplySpecBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BlockerSpec(SupplySpecBase):
    """TF-IDF blocking baseline (the embeddings stage comes behind it).

    top_k        candidates kept per anchor after same-gtin exclusion
    min_score    STRICT floor on the finalized-text cosine
    chunk_rows   anchor rows per matrix chunk (memory bound)
    """

    method: Literal["tfidf"] = "tfidf"
    top_k: int = 20
    min_score: float = 0.55
    chunk_rows: int = 512

    @field_validator("top_k", "chunk_rows")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("blocker sizes must be positive")
        return value


class MintSpec(SupplySpecBase):
    """Mint rules: one partner per uncovered anchor, ONE whitelisted move.

    A volume flip is a legitimate negative only at GTIN entity level (two
    real GTINs that differ in size are two products). The seed drives the
    donor draw so a rerun reproduces a run byte for byte.
    """

    entity_level: Literal["gtin", "sku"] = "gtin"
    moves: tuple[Literal["volume", "flavor"], ...] = ("volume", "flavor")
    prefer: Literal["flavor", "volume", "alternate"] = "alternate"
    max_minted: int = 50_000

    @model_validator(mode="after")
    def _entity_level_gates_the_whitelist(self) -> "MintSpec":
        if self.entity_level == "sku" and "volume" in self.moves:
            self.moves = tuple(move for move in self.moves if move != "volume")
            if not self.moves:
                raise ValueError(
                    "entity_level='sku' allows no move: a pack flip is not a "
                    "legitimate negative below the GTIN entity level"
                )
        return self


class DiscriminatorSpec(SupplySpecBase):
    """Pre-scale real-vs-minted separation thresholds.

    separable_auc     AUC at/above which the minted arm is detectable and
                      the run must STOP minting (the generator has a
                      signature).
    borderline_auc    AUC at/above which the verdict is "borderline"
                      (interpretation, not a stop).
    min_arm_rows      each arm needs at least this many rows before an AUC
                      means anything (below it the verdict is
                      "insufficient", which is NOT a failure).
    cv_folds          grouped-CV folds (GroupKold over the anchor GTIN).
    max_iter          sklearn solver iterations.
    """

    separable_auc: float = Field(default=0.90, ge=0.5, le=1.0)
    borderline_auc: float = Field(default=0.75, ge=0.5, le=1.0)
    min_arm_rows: int = Field(default=10, ge=2)
    cv_folds: int = Field(default=5, ge=2)
    max_iter: int = Field(default=2000, ge=1)

    @model_validator(mode="after")
    def _ordering(self) -> "DiscriminatorSpec":
        if self.borderline_auc > self.separable_auc:
            raise ValueError(
                "discriminator thresholds must be ordered: borderline_auc <= separable_auc"
            )
        return self


DiscriminatorSpec.model_rebuild()


class NegativeSupplySpec(SupplySpecBase):
    """Full lane config; env EUROMONITOR_NEGATIVE_SUPPLY_SPEC (JSON) overrides."""

    blocker: BlockerSpec = Field(default_factory=BlockerSpec)
    mint: MintSpec = Field(default_factory=MintSpec)
    seed: int = 1337
    # Shadow gate comparison columns are attached to every emitted row; they
    # are informational contrast (model-alone vs gate), never labels/features.
    shadow_gate: bool = True
    # Pre-scale discriminator (scripts/negative_supply_discriminator.py) —
    # the run-fail thresholds its verdict reads, so tuning them is a config
    # edit, not a script edit. The script fails with a nonzero exit when the
    # verdict lands on SEPARABLE: fix the mint rules BEFORE minting 50k rows.
    discriminator: DiscriminatorSpec | None = None

    @field_validator("seed")
    @classmethod
    def _seed_range(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value


def load_spec(path: Path | None = None) -> NegativeSupplySpec:
    """Module defaults, or the (env-named) JSON document."""
    override = os.environ.get("EUROMONITOR_NEGATIVE_SUPPLY_SPEC", "")
    document = Path(path) if path is not None else (Path(override) if override else None)
    if document is None:
        return NegativeSupplySpec()
    return NegativeSupplySpec.model_validate_json(document.read_text())


class PairRow(BaseModel):
    """One emitted labelled pair with single-hop provenance."""

    model_config = ConfigDict(extra="forbid", validate_assignment=False)

    anchor_row: int
    partner_row: int  # -1 for minted synthetic partners
    label: int
    population: _POPULATION_TYPE
    is_real: bool
    anchor_gtin: str = ""
    partner_gtin: str = ""
    anchor_text: str = ""
    partner_text: str = ""
    score: float = 0.0  # blocker cosine; minted rows are scored too
    # REAL partner lineage: the one whitelisted attribute that differs.
    diff_dimension: str = ""  # "" | one of _DIFF_DIMENSIONS (validated below)
    # MINT/EDIT lineage: exactly one token move per row.
    edit_field: Literal["", "volume", "flavor"] = ""
    edit_from: str = ""
    edit_to: str = ""
    # Shadow gate: informational comparison columns; never labels/features.
    shadow_gate_decision: str = ""
    shadow_gate_reason: str = ""

    @model_validator(mode="after")
    def _provenance_contract(self) -> "PairRow":
        moved = bool(self.edit_field) or bool(self.edit_from) or bool(self.edit_to)
        edit_tagged = self.population in (
            POPULATION_MINTED_PARTNER, POPULATION_EDITED_POSITIVE,
        )
        if edit_tagged:
            if not moved:
                raise ValueError(f"{self.population!r} rows carry edit provenance")
            if self.edit_field not in ("volume", "flavor"):
                raise ValueError("the edit names a whitelisted move")
            if self.edit_from == self.edit_to:
                raise ValueError("a move that lands nowhere is not an edit")
        elif moved:
            raise ValueError("edit provenance belongs to minted/edited rows only")
        if self.population == POPULATION_REAL_PARTNER:
            if not self.is_real or not self.diff_dimension:
                raise ValueError("real partners are tagged real and name their diff")
            if self.diff_dimension not in _DIFF_DIMENSIONS:
                raise ValueError("the diff names a registered attribute dimension")
        elif self.diff_dimension:
            raise ValueError("diff lineage belongs to real partners only")
        if self.population == POPULATION_MINTED_PARTNER:
            if self.is_real or self.partner_row != -1 or self.partner_gtin:
                raise ValueError("minted partners are synthetic: partner_row=-1")
        if not edit_tagged and not self.is_real:
            raise ValueError(f"population {self.population!r} must be tagged real")
        return self


# ── finalized texts + canonical set-literal reader ──────────────────────────
class SetColumnReader:
    """Reads one canonical_records.csv set-literal cell.

    ``"{'355.0', '500 ml'}"`` -> frozenset({"355.0", "500 ml"}). Blank
    spellings (NaN, empty, ``frozenset()``, ``set()``) all map to the empty
    set. Parse results are memoized per cell text: several pairs share the
    same canonical cell, so the same string is never parsed twice.
    """

    _SET_CHARS = str.maketrans("", "", "'\"{}[]()")
    _EMPTY_CELL_NORMS = frozenset(
        {"", "frozenset()", "frozenset", "set()", "set", "nan", "none"}
    )

    def __init__(self) -> None:
        self._parsed: dict[str, frozenset[str]] = {}

    def read(self, raw: object) -> frozenset[str]:
        """One canonical CSV cell -> the atom set it names (memoized)."""
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return frozenset()
        text = str(raw)
        cached = self._parsed.get(text)
        if cached is None:
            cached = self._parsed[text] = self._parse(text)
        return cached

    def _parse(self, text: str) -> frozenset[str]:
        stripped = text.translate(self._SET_CHARS).strip()
        if stripped.lower() in self._EMPTY_CELL_NORMS:
            return frozenset()
        return frozenset(atom.strip() for atom in stripped.split(",") if atom.strip())


_SET_READER = SetColumnReader()


def read_set_column(raw: object) -> frozenset[str]:
    """A canonical CSV cell like "{'355.0', '500 ml'}" -> frozenset."""
    return _SET_READER.read(raw)


def finalized_texts(df: pd.DataFrame) -> pd.Series:
    """Finalized model-input text per row via the SSOT builder (no re-normalization)."""
    from core.record_linkage import finalized_texts as finalized

    return finalized(df)


# ── the ONE token move both labels share ────────────────────────────────────
class TokenMover:
    """Whitelisted token moves on a text copy (volume / flavor only).

    A move is deterministic: the leftmost whitelisted token that carries a
    replacement target lands ONCE. No move is invented when the text has no
    such surface.
    """

    _VOLUME_TOKEN_RE = re.compile(r"\bvolume_ml_\d+(?:\.\d+)?\b")
    _FLAVOR_TOKEN_RE = re.compile(r"\bflavor_[a-z0-9_]+\b")
    _VOLUME_PREFIX = "volume_ml_"

    _SURFACES: Mapping[str, tuple[re.Pattern, str]] = {
        "volume": (_VOLUME_TOKEN_RE, _VOLUME_PREFIX),
        "flavor": (_FLAVOR_TOKEN_RE, "flavor_"),
    }

    @classmethod
    def move(
        cls, text: str, dimension: str, replacement: Mapping[str, str]
    ) -> tuple[str, str, str] | None:
        """ONE whitelisted token move; returns (new, from, to) or None."""
        pattern, prefix = cls.surface(dimension)
        match = next(
            (
                m for m in pattern.finditer(text)
                if replacement.get(m.group(0)[len(prefix):])
            ),
            None,
        )
        if match is None:
            return None
        moved_from = match.group(0)[len(prefix):]
        moved_to = replacement[moved_from]
        new_text = text[:match.start()] + prefix + moved_to + text[match.end():]
        return new_text, moved_from, moved_to

    @classmethod
    def atoms(cls, text: str, dimension: str) -> tuple[str, ...]:
        """The token surfaces of one dimension present on the text."""
        pattern, prefix = cls.surface(dimension)
        return tuple(m.group(0)[len(prefix):] for m in pattern.finditer(text))

    @classmethod
    def surface(cls, dimension: str) -> tuple[re.Pattern, str]:
        """(token regex, prefix) for a whitelisted move; loud otherwise."""
        try:
            return cls._SURFACES[dimension]
        except KeyError:
            raise ValueError(f"unwhitelisted move: {dimension}") from None


def token_move(
    text: str, dimension: str, replacement: Mapping[str, str]
) -> tuple[str, str, str] | None:
    """ONE whitelisted token move on a text copy; returns (new, from, to)."""
    return TokenMover.move(text, dimension, replacement)


def _text_atoms(text: str, dimension: str) -> tuple[str, ...]:
    """The atom surfaces of one dimension present on the text."""
    return TokenMover.atoms(text, dimension)


# ── canonical lookup by GTIN + corpus donor pools ───────────────────────────
class CanonicalIndex:
    """The canonical_records.csv namespace, keyed by stripped GTIN.

    LAST record wins on a duplicate gtin — the same resolution the former
    dict comprehension produced, kept so attribute evidence can never shift.
    The donor pools for the mint are corpus-owned: every atom observed in
    the corpus for one dimension's set column.
    """

    _ATOM_RES: Mapping[str, re.Pattern] = {
        "volume": re.compile(r"\d+(?:\.\d+)?"),
    }

    def __init__(self, canonical: pd.DataFrame) -> None:
        self._frame = canonical
        self._records: dict[str, Mapping] = {}
        rows = canonical.to_dict("records")
        for row in _LOG.progress(rows, desc="canonical_index", unit="record"):
            self._records[str(row["gtin"]).strip()] = row
        self._pools: dict[str, tuple[str, ...]] = {}

    def record(self, gtin: str) -> Mapping | None:
        """The canonical record of one stripped gtin (None when absent)."""
        return self._records.get(gtin)

    def has_record(self, gtin: str) -> bool:
        """Whether a stripped gtin carries canonical attribute evidence."""
        return gtin in self._records

    def dimension_pool(self, dimension: str) -> frozenset[str]:
        """Every atom observed in the corpus for one dimension's set column."""
        if dimension not in _DIMENSION_COLUMNS:
            raise ValueError(f"unpooled dimension: {dimension}")
        pool: set[str] = set()
        column = self._frame[_DIMENSION_COLUMNS[dimension]]
        for raw in _LOG.progress(column, desc="dimension_pool", unit="record"):
            pool |= read_set_column(raw)
        return frozenset(pool)

    def corpus_donors(self, dimension: str) -> tuple[str, ...]:
        """Sorted unique corpus atoms usable as move targets (token-shaped)."""
        if dimension not in self._pools:
            pattern = self._ATOM_RES.get(
                dimension, re.compile(r"[a-z0-9_]+")
            )
            self._pools[dimension] = tuple(
                sorted(
                    value
                    for value in self.dimension_pool(dimension)
                    if pattern.fullmatch(value)
                )
            )
        return self._pools[dimension]


# ── anchor-row lookups ──────────────────────────────────────────────────────
class AnchorIndex:
    """The anchor frame's GTIN namespace.

    ``first_anchored_row(gtin)`` maps a gtin to its FIRST frame row (base-pair
    lookup); ``gtins`` holds the stripped gtin of every row once, so stages
    stop re-stripping the same column per candidate.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        gtins = df["gtin"].fillna("").astype(str).str.strip()
        positions: dict[str, int] = {}
        for position, gtin in _LOG.progress(
            enumerate(gtins), desc="anchor_row_index", unit="row", total=len(gtins),
        ):
            positions.setdefault(gtin, position)
        self._first_row = positions
        self.gtins = gtins.to_numpy()

    def first_anchored_row(self, gtin: str) -> int | None:
        """First anchor-row position per stripped gtin (None when absent)."""
        return self._first_row.get(gtin)

    def row_gtin(self, position: int) -> str:
        """The stripped gtin of one anchor row."""
        return str(self.gtins[position])


def _ordered_moves(moves: tuple[str, ...], prefer: str, sequence: int) -> list[str]:
    """The move order for one mint attempt (preference first, the rest after)."""
    if prefer == "alternate":
        ordered = (moves[sequence % len(moves)],)
    elif prefer == "flavor":
        ordered = ("flavor",) if "flavor" in moves else moves
    else:
        ordered = ("volume",) if "volume" in moves else moves
    return list(ordered) + [move for move in moves if move not in ordered]


def _max_minted_under_cap(n_minted: int, n_real: int, mint_cap: float) -> int:
    """Minted leaf rows allowed under the negative:minted ratio cap."""
    if mint_cap >= 1.0:
        allowed = n_minted
    else:
        allowed = int((mint_cap * n_real) / (1.0 - mint_cap)) if mint_cap > 0 else 0
    return min(allowed, n_minted)


# ── trainer bridge: base bundle + lane pairs -> trainer contract ────────────
class LaneAssembler:
    """Assembles the trainer contract under the lane's negative source.

    Real negatives = base_negative + real_partner rows mapped by GTIN (the
    gate-derived negative is replaced — same anchors, so the swap is a
    source change, not a semantics change). Minted partners are appended as
    TRAINING-ONLY leaf payload rows replaying the lane's single whitelisted
    move on the TRAINER's own payload text for that anchor; they never enter
    ``neg``, so evaluation stays real-pairs-only and the augmentation stages
    (which read ``neg``) never re-edit an already-minted record.
    """

    def __init__(
        self, base: Mapping, pairs: pd.DataFrame, *, n_sku: int, mint_cap: float = 1.0
    ) -> None:
        self._base = base
        self._pairs = pairs
        self._n_sku = n_sku
        self._mint_cap = mint_cap

    def assemble(self) -> dict:
        """The final trainer contract (see class docstring)."""
        payload = list(self._base["payload"])
        row_bc = [str(value) for value in self._base["row_bc"]]
        structured = [list(row) for row in self._base["structured_features"]]
        gtin_to_row = dict(self._base["gtin_to_row"])
        canon_gtins = row_bc[self._n_sku:]
        gtin_to_canon_idx = {
            g: self._n_sku + i for i, g in enumerate(canon_gtins) if g
        }
        payload_source = ["sku"] * self._n_sku + ["canonical"] * len(canon_gtins)

        neg_rows, neg_source = self._real_negatives(gtin_to_row, gtin_to_canon_idx)
        minted = self._pairs[self._pairs["population"] == POPULATION_MINTED_PARTNER].sort_values(
            "anchor_row", kind="stable"
        )
        n_real = len(neg_rows)
        allowed_minted = _max_minted_under_cap(len(minted), n_real, self._mint_cap)
        dropped_cap = len(minted) - allowed_minted
        neg_minted_rows = self._append_minted_leaves(
            minted, allowed_minted, payload, row_bc, structured, payload_source
        )

        stats = self._stats(neg_source, len(neg_minted_rows), dropped_cap)
        return self._training_data(
            payload, structured, row_bc, neg_rows, neg_source,
            payload_source, neg_minted_rows, stats, gtin_to_row,
        )

    def _real_negatives(
        self, gtin_to_row: Mapping, gtin_to_canon_idx: Mapping
    ) -> tuple[list[tuple[int, int]], list[str]]:
        """Base negative and real-partner rows mapped to trainer row pairs."""
        neg_rows: list[tuple[int, int]] = []
        neg_source: list[str] = []
        real_populations = (POPULATION_BASE_NEGATIVE, POPULATION_REAL_PARTNER)
        rows = list(self._pairs.itertuples(index=False))
        for pair in _LOG.progress(
            rows, desc="lane_real_negatives", unit="pair", total=len(rows),
        ):
            population = str(pair.population)
            if int(pair.label) != 0 or population not in real_populations:
                continue
            anchor = gtin_to_row.get(str(pair.anchor_gtin).strip())
            target = gtin_to_canon_idx.get(str(pair.partner_gtin).strip())
            if anchor is None or target is None:
                continue
            neg_rows.append((int(anchor), int(target)))
            # Literal tags (not the constant) so the coverage-registry scan
            # (tests/test_datapoint_coverage.py) sees the producers.
            if population == POPULATION_BASE_NEGATIVE:
                neg_source.append("base_negative")
            else:
                neg_source.append("real_partner")
        return neg_rows, neg_source

    def _append_minted_leaves(
        self,
        minted: pd.DataFrame,
        allowed_minted: int,
        payload: list[str],
        row_bc: list[str],
        structured: list[list[float]],
        payload_source: list[str],
    ) -> list[tuple[int, int]]:
        """Minted leaves appended (text format matches every other payload row)."""
        neg_minted_rows: list[tuple[int, int]] = []
        width = len(structured[0]) if structured else 0
        leaves = list(minted.itertuples(index=False))[:allowed_minted]
        for pair in _LOG.progress(
            leaves, desc="minted_leaves", unit="leaf", total=len(leaves),
        ):
            anchor = int(pair.anchor_row)
            if not (0 <= anchor < self._n_sku):
                continue
            outcome = token_move(
                payload[anchor], str(pair.edit_field),
                {str(pair.edit_from): str(pair.edit_to)},
            )
            if outcome is None:
                # The trainer's text carries no surface for the lane's move.
                continue
            payload.append(outcome[0])
            row_bc.append(f"minted:{anchor}:{len(payload) - 1}")
            structured.append([0.0] * width)
            payload_source.append("minted")
            neg_minted_rows.append((anchor, len(payload) - 1))
        return neg_minted_rows

    def _stats(
        self, neg_source: list[str], n_minted: int, dropped_cap: int
    ) -> dict:
        """The base run stats extended by the lane's negative-source census."""
        stats = dict(self._base["stats"])
        stats.update({
            "negative_supply_mode": "lane",
            "n_lane_base_negative": neg_source.count("base_negative"),
            "n_lane_real_partner": neg_source.count("real_partner"),
            "n_lane_minted": n_minted,
            "n_lane_minted_dropped_cap": dropped_cap,
        })
        return stats

    def _training_data(
        self,
        payload: list[str],
        structured: list[list[float]],
        row_bc: list[str],
        neg_rows: list[tuple[int, int]],
        neg_source: list[str],
        payload_source: list[str],
        neg_minted_rows: list[tuple[int, int]],
        stats: dict,
        gtin_to_row: Mapping,
    ) -> dict:
        """The final trainer contract under the lane's negative source."""
        from core.schemas import TrainingData

        bundle = TrainingData(
            payload=payload,
            structured_features=structured,
            row_bc=np.array(row_bc),
            pos=np.asarray(self._base["pos"], dtype=int).reshape(-1, 2),
            neg=np.array(neg_rows, dtype=int).reshape(-1, 2),
            targeted_attribute_neg=np.empty((0, 2), dtype=int),
            cross_brand_neg=np.empty((0, 2), dtype=int),
            gtin_to_row=gtin_to_row,
            stats=stats,
            payload_source=payload_source,
            neg_source=neg_source,
            neg_minted=np.array(neg_minted_rows, dtype=int).reshape(-1, 2),
        )
        return bundle.model_dump()


def assemble_lane_bundle(
    base: Mapping, pairs: pd.DataFrame, *, n_sku: int, mint_cap: float = 1.0
) -> dict:
    """Pure assembler: base bundle + lane pairs -> trainer contract.

    Split out from :func:`build_lane_training_data` so the negative swap, the
    minted-leaf append and the cap are unit-testable without the live
    canonical/gate artifacts.
    """
    return LaneAssembler(base, pairs, n_sku=n_sku, mint_cap=mint_cap).assemble()


def build_lane_training_data(
    df: pd.DataFrame, *, payload_variant: str = "full",
    run_tag: str, mint_cap: float = 1.0,
) -> dict:
    """Trainer contract with negatives from the lane instead of the gate.

    Reuses ``pipeline.build_training_data`` for the payload and POSITIVES
    (positives are gate-independent — a row paired with its own canonical),
    then REPLACES the gate-derived negatives (see :class:`LaneAssembler`).

    The gate-derived miners (targeted-attribute, cross-brand) are DROPPED: the
    lane's real partners are their replacement, per the 2026-10-03 ruling.
    """
    from pipeline import build_training_data

    base = build_training_data(df, payload_variant=payload_variant)
    return assemble_lane_bundle(
        base, load_pairs(run_tag), n_sku=len(df), mint_cap=mint_cap
    )


# ── GTIN-grouped split (fold helper) ────────────────────────────────────────
class FoldMap:
    """Item -> fold bucket, assigned per canonical component (order-free).

    Components are ordered canonically (shared core.disjoint_sets), so the
    bucket assignment depends only on the component PARTITION, never on
    union order or root labels.
    """

    def __init__(self, k: int, seed: int) -> None:
        self._k = k
        self._seed = seed

    def buckets(self, components: list) -> dict[str, int]:
        """Item -> bucket for every item of every component."""
        rng = np.random.default_rng(self._seed)
        bucket_of_component = np.asarray(rng.permutation(len(components))) % self._k
        return {
            item: int(bucket_of_component[index])
            for index, component in enumerate(components)
            for item in component
        }


def gtin_group_split(frame: pd.DataFrame, *, k: int = 4, seed: int = 1337) -> pd.Series:
    """Deterministic GTIN-grouped bucket index (0..k-1) per pair row.

    Entities are union components over the gtin namespace: pairs sharing an
    endpoint share a bucket. Minted synthetic rows carry -1 (never scored;
    evaluation is real-pairs-only by owner ruling).
    """
    gtins = frame["anchor_gtin"].astype(str).str.strip()
    partners = frame["partner_gtin"].astype(str).str.strip()
    populations = frame["population"]
    groups = DisjointSet()
    for left, right, minted in _LOG.progress(
        zip(gtins, partners, populations), desc="entity_graph",
        unit="pair", total=len(frame),
    ):
        if minted != POPULATION_MINTED_PARTNER and right and left:
            groups.union(left, right)
    fold_of = FoldMap(k, seed).buckets(groups.components())
    out = [
        -1 if minted == POPULATION_MINTED_PARTNER else fold_of.get(left, -1)
        for left, minted in zip(gtins, populations)
    ]
    return pd.Series(out, index=frame.index, name="group_fold")


# ── trainer bridge lookups ──────────────────────────────────────────────────
def pairs_path(run_tag: str) -> Path:
    from core.common import RESULTS

    return Path(RESULTS) / "negative_supply" / run_tag / "pairs.csv"


def load_pairs(run_tag: str) -> pd.DataFrame:
    """The lane's emitted pairs.csv for one run tag (fail loud when absent)."""
    path = pairs_path(run_tag)
    if not path.is_file():
        raise FileNotFoundError(
            f"negative-supply pairs.csv not found for run_tag={run_tag!r}: {path}"
        )
    return pd.read_csv(
        path,
        dtype={"anchor_gtin": str, "partner_gtin": str},
        keep_default_na=False,
    )


# ── the orchestrator ────────────────────────────────────────────────────────
class _CandidateIndex:
    """The blocker's candidate table: (anchor_row, candidate_row, score)."""

    def __init__(self, rows: list[tuple[int, int, float]]) -> None:
        self.frame = pd.DataFrame(
            rows, columns=["anchor_row", "candidate_row", "score"]
        ).drop_duplicates(subset=["anchor_row", "candidate_row"])


class NegativeSupply(BaseModel):
    """Orchestrator: block -> mine real one-diff partners -> mint the rest.

    The count of anchors WITH at least one real partner decides the mint's
    share: mostly covered means a top-up, thinly covered means it is the
    main source. The gate is consulted only to ATTACH shadow verdicts
    (``shadow_gate_*`` columns), never to choose labels.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    spec: NegativeSupplySpec = Field(default_factory=NegativeSupplySpec)
    df: pd.DataFrame = Field(repr=False, exclude=True)
    canonical: pd.DataFrame = Field(repr=False, exclude=True)
    gates: pd.DataFrame = Field(default=None, repr=False, exclude=True)
    labeled: pd.DataFrame = Field(default=None, repr=False, exclude=True)
    candidates: pd.DataFrame = Field(default=None, repr=False, exclude=True)
    pairs: pd.DataFrame = Field(default=None, repr=False, exclude=True)
    funnel: dict = Field(default_factory=dict, repr=False, exclude=True)

    RESULT_DIR: ClassVar[str] = "results/negative_supply"
    _canonical_index: CanonicalIndex | None = PrivateAttr(default=None)
    _anchors: AnchorIndex | None = PrivateAttr(default=None)
    _texts: pd.Series | None = PrivateAttr(default=None)
    _rng: np.random.Generator | None = PrivateAttr(default=None)
    _diffs_by_pair: dict[tuple[str, str], dict[str, bool]] = PrivateAttr(
        default_factory=dict
    )
    _block_vectorizer: object = PrivateAttr(default=None)
    _block_matrix: object = PrivateAttr(default=None)
    _block_anchor_positions: dict[int, int] = PrivateAttr(default_factory=dict)

    @field_validator("df")
    @classmethod
    def _frame_contract(cls, df: pd.DataFrame) -> pd.DataFrame:
        if "gtin" not in df.columns:
            raise ValueError("anchor frame must carry a gtin column")
        return df

    @field_validator("canonical")
    @classmethod
    def _canonical_contract(cls, canonical: pd.DataFrame) -> pd.DataFrame:
        needs = {"gtin", _DIMENSION_COLUMNS["volume"], _DIMENSION_COLUMNS["flavor"]}
        missing = needs - set(canonical.columns)
        if missing:
            raise ValueError(f"canonical records missing columns: {sorted(missing)}")
        return canonical

    @field_validator("gates", "labeled", "candidates", "pairs", mode="before")
    @classmethod
    def _none_frames(cls, value: object) -> object:
        if value is None:
            return pd.DataFrame()
        return value

    # ── lookups / preparation ────────────────────────────────────────────────
    def _prepare(self) -> None:
        """Build every lazy singleton once (canonical/anchor indexes, texts, RNG)."""
        if self._canonical_index is None:
            self._canonical_index = CanonicalIndex(self.canonical)
        if self._anchors is None:
            self._anchors = AnchorIndex(self.df)
        if self._texts is None:
            self._texts = finalized_texts(self.df)
        if self._rng is None:
            self._rng = np.random.default_rng(self.spec.seed)

    def canonical_by_gtin(self, gtin: str) -> Mapping | None:
        self._prepare()
        return self._canonical_index.record(gtin)

    @property
    def anchor_mask(self) -> pd.Series:
        """Anchorable rows: checksum-valid, non-empty gtin AND a finalized title."""
        from core.gtin import gtin_validity

        self._prepare()
        gtins = self.df["gtin"].fillna("").astype(str).str.strip()
        valid = pd.Series(gtin_validity(gtins).to_numpy(), index=self.df.index)
        titled = self._texts.str.strip().ne("")
        return valid & gtins.ne("") & titled

    def _row_by_gtin(self, gtin: str) -> int | None:
        self._prepare()
        return self._anchors.first_anchored_row(gtin)

    def _donors(self, dimension: str, exclude: frozenset[str]) -> tuple[str, ...]:
        """Corpus atoms for a dimension, minus the anchor's own surfaces."""
        pool = self._canonical_index.corpus_donors(dimension)
        return tuple(value for value in pool if value not in exclude)

    # ── stage 1: blocking ────────────────────────────────────────────────────
    def block(self) -> pd.DataFrame:
        """Real different-GTIN candidates per anchor, TF-IDF cosine top-k."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        with _LOG.section("negative_supply.block"):
            self._prepare()
            anchors = np.flatnonzero(self.anchor_mask.to_numpy())
            if len(anchors) == 0:
                return self._block_empty_pool()
            gtins = self._anchors.gtins[anchors]
            vectorizer = TfidfVectorizer(sublinear_tf=True)
            matrix = vectorizer.fit_transform(self._texts.iloc[anchors].tolist())
            self._freeze_block_blocker(vectorizer, matrix, anchors)
            rows = self._chunked_top_candidates(matrix, gtins, anchors)
            self.candidates = _CandidateIndex(rows).frame
            self.funnel["block"] = self._block_funnel(len(anchors))
            return self.candidates

    def _block_empty_pool(self) -> pd.DataFrame:
        """The empty block result: no candidates and a zeroed funnel entry."""
        self.candidates = pd.DataFrame()
        self.funnel["block"] = {"anchors": 0, "candidates": 0}
        return self.candidates

    def _block_funnel(self, anchors: int) -> dict:
        """The blocking stage's funnel entry over one anchor count."""
        return {
            "anchors": int(anchors),
            "candidates": len(self.candidates),
            "min_score": self.spec.blocker.min_score,
            "top_k": self.spec.blocker.top_k,
        }

    def _chunked_top_candidates(
        self, matrix: "np.ndarray", gtins: "np.ndarray", anchors: "np.ndarray"
    ) -> list[tuple[int, int, float]]:
        """Anchor rows per matrix chunk (memory bound): cosine, exclusion, top-k."""
        rows: list[tuple[int, int, float]] = []
        chunk_rows = self.spec.blocker.chunk_rows
        chunk_starts = range(0, len(anchors), chunk_rows)
        total_chunks = max(1, -(-len(anchors) // chunk_rows))
        for start in _LOG.progress(
            chunk_starts, desc="block_chunks", unit="chunk",
            total=total_chunks,
        ):
            chunk = (matrix[start:start + chunk_rows] @ matrix.T).toarray()
            chunk = chunk.astype(np.float32)
            _suppress_same_gtin(chunk, gtins, start)
            rows.extend(self._top_candidates(chunk, gtins, start, anchors,
                                             len(anchors)))
        return rows

    def _freeze_block_blocker(self, vectorizer, matrix, anchors: "np.ndarray") -> None:
        """Keep the blocker frozen for minted-arm scoring parity."""
        self._block_vectorizer = vectorizer
        self._block_matrix = matrix
        self._block_anchor_positions = {
            int(anchor): index for index, anchor in enumerate(anchors)
        }

    def _top_candidates(
        self,
        chunk: "np.ndarray",
        gtins: "np.ndarray",
        start: int,
        anchors: "np.ndarray",
        n_anchors: int,
    ) -> list[tuple[int, int, float]]:
        """The top-k above-floor cross-gtin candidates of one anchor chunk.

        Rows are suppressed below the STRICT floor with an early break, since
        the local columns descend by score; same-gtin leftovers are skipped.
        """
        limit = min(self.spec.blocker.top_k, n_anchors)
        pairs: list[tuple[int, int, float]] = []
        for position, nearest in enumerate(self._descending_nearest(chunk, limit)):
            for column in nearest:
                score = float(chunk[position, column])
                if score <= self.spec.blocker.min_score:
                    break
                if gtins[start + position] == gtins[column]:
                    continue
                pairs.append((
                    int(anchors[start + position]),
                    int(anchors[column]),
                    round(score, 6),
                ))
        return pairs

    def _descending_nearest(self, chunk: "np.ndarray", limit: int) -> "np.ndarray":
        """Each anchor's candidate columns, best cosine first, floor-blind."""
        local = np.argpartition(-chunk, limit - 1, axis=1)[:, :limit]
        return np.take_along_axis(
            local,
            np.argsort(-np.take_along_axis(chunk, local, axis=1), axis=1),
            axis=1,
        )

    # ── stage 2: real one-diff partners ──────────────────────────────────────
    def _candidate_diff(self, left_gtin: str, right_gtin: str) -> dict[str, bool]:
        """Attribute diff per gtin pair, memoized (arms share partners)."""
        key = tuple(sorted((left_gtin, right_gtin)))
        if key not in self._diffs_by_pair:
            self._diffs_by_pair[key] = self.attribute_diff(left_gtin, right_gtin)
        return self._diffs_by_pair[key]

    def _real_partner_row(self, candidate, diffs: dict[str, bool]) -> PairRow:
        """One REAL partner row: exactly one whitelisted dimension differs."""
        return PairRow(
            anchor_row=candidate.anchor_row,
            partner_row=candidate.candidate_row,
            label=0,
            population=POPULATION_REAL_PARTNER,
            is_real=True,
            anchor_gtin=self._anchors.row_gtin(candidate.anchor_row),
            partner_gtin=self._anchors.row_gtin(candidate.candidate_row),
            anchor_text=str(self._texts.iloc[candidate.anchor_row]),
            partner_text=str(self._texts.iloc[candidate.candidate_row]),
            score=float(candidate.score),
            diff_dimension=next(
                dim for dim in self.spec.mint.moves if diffs[dim]
            ),
        )

    def _single_whitelisted_diff(self, diffs: dict[str, bool]) -> bool:
        """Exactly one attribute differs AND it is a whitelisted move target."""
        return sum(diffs.values()) == 1 and any(
            diffs[dim] for dim in self.spec.mint.moves
        )

    def mine_real_partners(self) -> list[PairRow]:
        """Real different-GTIN pairs with EXACTLY ONE whitelisted diff.

        Every candidate above the floor is evaluated; the funnel reports
        survivors by differing-dimension count, so 'why N?' is answerable
        from the trace without re-running the filters.
        """
        self._prepare()
        if self.candidates is None:
            self.block()
        rows: list[PairRow] = []
        census = {
            "no_canonical_record": 0, "diff_count_1": 0, "diff_count_other": 0,
        }
        with _LOG.section("negative_supply.mine_real_partners"):
            candidates = list(self.candidates.itertuples(index=False))
            for candidate in _LOG.progress(
                candidates, desc="mine_real_partners", unit="candidate",
                total=len(candidates),
            ):
                row = self._mined_candidate(candidate, census)
                if row is not None:
                    rows.append(row)
        rows.sort(key=lambda pair: (pair.anchor_row, pair.partner_row))
        self.funnel["mine_real"] = {**census, "real_partners": len(rows)}
        return rows

    def _mined_candidate(
        self, candidate, census: dict[str, int]
    ) -> PairRow | None:
        """One candidate's outcome: the canonical-gate then one-diff verdict."""
        left_gtin = self._anchors.row_gtin(candidate.anchor_row)
        right_gtin = self._anchors.row_gtin(candidate.candidate_row)
        if not self._canonical_index.has_record(left_gtin) \
                or not self._canonical_index.has_record(right_gtin):
            census["no_canonical_record"] += 1
            return None
        diffs = self._candidate_diff(left_gtin, right_gtin)
        if self._single_whitelisted_diff(diffs):
            census["diff_count_1"] += 1
            return self._real_partner_row(candidate, diffs)
        census["diff_count_other"] += 1
        return None

    # ── stage 3: minted partners for uncovered anchors ───────────────────────
    def mint(self, covered: set[int]) -> list[PairRow]:
        """ONE partner per uncovered anchor with a single whitelisted move.

        The partner is a synthetic token-move copy of its own anchor (the
        anchor text stays untouched); the donor value is drawn deterministically
        from the sorted corpus pool under the spec seed. When the anchor's
        text carries no surface for the preferred move, the other whitelisted
        move is tried before the anchor is skipped (reported, not silent).
        """
        with _LOG.section("negative_supply.mint"):
            self._prepare()
            anchor_gtins = self._anchors.gtins
            uncovered = self._uncovered_anchor_rows(covered, anchor_gtins)
            rows, skipped = self._mint_uncovered(anchor_gtins, uncovered)
            self.funnel["mint"] = self._mint_funnel(anchor_gtins, uncovered, rows, skipped)
            return rows

    def _mint_uncovered(
        self, anchor_gtins: "np.ndarray", uncovered: list[int]
    ) -> tuple[list[PairRow], dict[str, int]]:
        """One pass over the uncovered anchors, then the under-floor drop."""
        rows: list[PairRow] = []
        skipped = {"no_move_surface": 0, "empty_pool": 0, "target_cap": 0,
                   "same_entity": 0, "below_blocker_floor": 0}
        minted_gtins: set[str] = set()
        moves = self.spec.mint.moves or ("flavor",)
        for sequence, anchor in _LOG.progress(
            enumerate(sorted(uncovered)), desc="mint_partners", unit="anchor",
            total=len(uncovered),
        ):
            outcome = self._mint_outcome(
                anchor, sequence, anchor_gtins, moves, skipped, minted_gtins, rows
            )
            if outcome is None:
                continue
            rows.append(outcome)
            minted_gtins.add(anchor_gtins[anchor])
        rows.sort(key=lambda pair: pair.anchor_row)
        rows = self._score_below_floor(rows, skipped)
        return rows, skipped

    def _mint_funnel(
        self,
        anchor_gtins: "np.ndarray",
        uncovered: list[int],
        rows: list[PairRow],
        skipped: dict[str, int],
    ) -> dict:
        """The mint stage's funnel entry: coverage counts, skipped census."""
        return {
            **skipped,
            "anchors_uncovered": (len({anchor_gtins[position] for position in uncovered})
                                  if self.spec.mint.entity_level == "gtin" else len(uncovered)),
            "uncovered_sku_rows": len(uncovered),
            "minted": len(rows),
        }

    def _uncovered_anchor_rows(
        self, covered: set[int], anchor_gtins: "np.ndarray"
    ) -> list[int]:
        """Anchored rows still without a partner, per the configured entity level.

        ``gtin`` entity level: any row of an uncovered GTIN qualifies. ``sku``:
        only the exact uncovered row positions.
        """
        covered_gtins = {anchor_gtins[position] for position in covered}
        uncovered = [
            int(position)
            for position, covered_row in
            enumerate(self.anchor_mask.to_numpy())
            if covered_row and (
                anchor_gtins[position] not in covered_gtins
                if self.spec.mint.entity_level == "gtin" else position not in covered
            )
        ]
        return uncovered

    def _mint_outcome(
        self,
        anchor: int,
        sequence: int,
        anchor_gtins: "np.ndarray",
        moves: tuple[str, ...],
        skipped: dict[str, int],
        minted_gtins: set[str],
        rows: list[PairRow],
    ) -> PairRow | None:
        """One mint attempt: entity dedupe, cap, preferred-then-fallback move.

        Returns None when the anchor was skipped (reason counted in
        ``skipped``) or the move could not land; otherwise the minted
        synthetic row.
        """
        if self.spec.mint.entity_level == "gtin" and anchor_gtins[anchor] in minted_gtins:
            skipped["same_entity"] += 1
            return None
        if len(rows) >= self.spec.mint.max_minted:
            skipped["target_cap"] += 1
            return None
        landed = self._try_moves(
            anchor, _ordered_moves(moves, self.spec.mint.prefer, sequence), skipped
        )
        if landed is None:
            return None
        return self._minted_row(anchor, anchor_gtins, landed)

    def _minted_row(
        self, anchor: int, anchor_gtins: "np.ndarray", landed: tuple
    ) -> PairRow:
        """The minted synthetic row over one landed token move."""
        new_text, dimension, moved_from, moved_to = landed
        return PairRow(
            anchor_row=anchor,
            partner_row=-1,
            label=0,
            population=POPULATION_MINTED_PARTNER,
            is_real=False,
            anchor_gtin=anchor_gtins[anchor],
            partner_gtin="",
            anchor_text=str(self._texts.iloc[anchor]),
            partner_text=new_text,
            edit_field=dimension,
            edit_from=moved_from,
            edit_to=moved_to,
        )

    def _try_moves(
        self, anchor: int, attempted: list[str], skipped: dict[str, int]
    ) -> tuple[str, str, str, str] | None:
        """Try each whitelisted move in order; report exhausted pools.

        Returns (new_text, dimension, moved_from, moved_to), or None when no
        move could land (the surface is missing — reported, not silent).
        """
        anchor_text = str(self._texts.iloc[anchor])
        outcome = None
        for move in attempted:
            atoms = set(_text_atoms(anchor_text, move))
            pool = self._donors(move, exclude=frozenset(atoms))
            if not pool:
                skipped["empty_pool"] += 1
                continue
            moved_to = pool[int(self._rng.integers(len(pool)))]
            replacement = {atom: moved_to for atom in atoms}
            moved = token_move(anchor_text, move, replacement)
            if moved is not None:
                outcome = (moved[0], move, moved[1], moved[2])
                break
        if outcome is None:
            skipped["no_move_surface"] += 1
            return None
        return outcome

    def _score_below_floor(self, rows: list[PairRow], skipped: dict[str, int]) -> list[PairRow]:
        """Score minted rows with the frozen blocker; drop the under-floor ones.

        Minted partners get the real-pair blocker cosine so no texture feature
        separates them from real arms downstream.
        """
        self._score_pairs(rows)
        eligible = [pair for pair in rows if pair.score > self.spec.blocker.min_score]
        skipped["below_blocker_floor"] = len(rows) - len(eligible)
        return eligible

    def _score_pairs(self, rows: list[PairRow]) -> None:
        """Anchor x partner cosine under the frozen blocking vocabulary."""
        if not rows:
            return
        if self._block_vectorizer is None:
            self.block()
        # Match real-pair geometry: frozen blocker IDF and the actual anchor,
        # without the quadratic minted-by-catalog intermediate.
        minted = self._block_vectorizer.transform([pair.partner_text for pair in rows])
        anchors = self._block_vectorizer.transform([pair.anchor_text for pair in rows])
        scores = np.asarray(minted.multiply(anchors).sum(axis=1)).ravel()
        for pair, score in zip(rows, scores):
            pair.score = round(float(score), 6)

    # ── positives receive the same edit machinery ────────────────────────────
    def edited_positives(self) -> list[PairRow]:
        """Copy each real base positive and move one token on BOTH sides the
        same way — the label stays 1 while 'edited texture' spans both labels."""
        self._prepare()
        rows: list[PairRow] = []
        positives = self.labeled[self.labeled["true_label"].astype(int) == 1]
        for positive in _LOG.progress(
            positives.itertuples(index=False), desc="edited_positives",
            unit="positive", total=len(positives),
        ):
            outcome = self._symmetric_edit_outcome(positive)
            if outcome is None:
                continue
            rows.append(self._edited_positive_row(positive, outcome))
        return rows

    def _symmetric_edit_outcome(self, positive) -> tuple | None:
        """One whitelisted move applied to the pair copy, symmetrically.

        Returns (left_text, right_text, origin, move, moved_to), or None when
        no whitelisted move lands on the pair (reported by omission).
        """
        left_row = self._row_by_gtin(str(positive.gtin1).strip())
        right_row = self._row_by_gtin(str(positive.gtin2).strip())
        if left_row is None or right_row is None:
            return None
        for move in self.spec.mint.moves or ("flavor",):
            applied = self._apply_pair_move(left_row, right_row, move)
            if applied is not None:
                left_text, right_text, origin, moved_to = applied
                return left_text, right_text, origin, move, moved_to
        return None

    def _apply_pair_move(self, left_row: int, right_row: int, move: str):
        """One token moved the same way on both sides (whichever carries it)."""
        left_text = str(self._texts.iloc[left_row])
        right_text = str(self._texts.iloc[right_row])
        atoms = set(_text_atoms(left_text, move)) | set(_text_atoms(right_text, move))
        pool = self._donors(move, exclude=frozenset(atoms))
        if not pool:
            return None
        moved_to = pool[int(self._rng.integers(len(pool)))]
        replacement = {atom: moved_to for atom in atoms}
        left_move = token_move(left_text, move, replacement)
        right_move = token_move(right_text, move, replacement)
        if left_move is None and right_move is None:
            return None
        origin = left_move[1] if left_move is not None else right_move[1]
        return (
            left_move[0] if left_move is not None else left_text,
            right_move[0] if right_move is not None else right_text,
            origin,
            moved_to,
        )

    def _edited_positive_row(self, positive, outcome) -> PairRow:
        """The edited-positive row over a copied real base pair."""
        left_text, right_text, origin, move, moved_to = outcome
        return PairRow(
            anchor_row=int(self._row_by_gtin(str(positive.gtin1).strip())),
            partner_row=int(self._row_by_gtin(str(positive.gtin2).strip())),
            label=1,
            population=POPULATION_EDITED_POSITIVE,
            is_real=True,
            anchor_gtin=str(positive.gtin1).strip(),
            partner_gtin=str(positive.gtin2).strip(),
            anchor_text=left_text,
            partner_text=right_text,
            edit_field=move,
            edit_from=origin,
            edit_to=moved_to,
        )

    # ── attribute diff over canonical set columns ────────────────────────────
    def attribute_diff(self, anchor_gtin: str, partner_gtin: str) -> dict[str, bool]:
        """One bool per whitelisted dimension: both populated and different.

        An unpopulated dimension is never a diff — absence is not a verdict.
        """
        left = self.canonical_by_gtin(anchor_gtin)
        right = self.canonical_by_gtin(partner_gtin)
        outcome: dict[str, bool] = {}
        for dimension in _DIFF_DIMENSIONS:
            outcome[dimension] = self._dimension_diff(left, right, dimension)
        return outcome

    def _dimension_diff(self, left, right, dimension: str) -> bool:
        """One dimension's bool: both populated and different."""
        left_atoms = (
            read_set_column(left.get(_DIMENSION_COLUMNS[dimension]))
            if left is not None else frozenset()
        )
        right_atoms = (
            read_set_column(right.get(_DIMENSION_COLUMNS[dimension]))
            if right is not None else frozenset()
        )
        return bool(left_atoms and right_atoms
                    and left_atoms != right_atoms)

    # ── shadow-gate attach ───────────────────────────────────────────────────
    def attach_shadow_gate(self) -> None:
        """Lookup gate_design on (gtin1,gtin2) pairs — comparison columns only."""
        if not isinstance(self.pairs, pd.DataFrame) or self.pairs.empty or not self.spec.shadow_gate:
            return
        lookup = self._shadow_gate_lookup()
        decisions: list[str] = []
        reasons: list[str] = []
        rows = list(self.pairs.itertuples(index=False))
        for pair in _LOG.progress(rows, desc="shadow_gate_attach", unit="pair",
                                  total=len(rows)):
            result = self._gate_verdict(lookup, pair)
            decisions.append(result[0])
            reasons.append(result[1])
        out = self.pairs.copy()
        out["shadow_gate_decision"] = decisions
        out["shadow_gate_reason"] = reasons
        self.pairs = out

    def _shadow_gate_lookup(self) -> dict[tuple[str, str], tuple[str, str]]:
        """(gtin1, gtin2) -> (gate_decision, gate_reason) lookup (informational)."""
        lookup: dict[tuple[str, str], tuple[str, str]] = {}
        rows = list(self.gates.itertuples(index=False))
        for g in _LOG.progress(rows, desc="gate_lookup", unit="pair", total=len(rows)):
            lookup[(str(g.gtin1).strip(), str(g.gtin2).strip())] = (
                str(g.gate_decision), str(g.gate_reason),
            )
        return lookup

    @staticmethod
    def _gate_verdict(
        lookup: Mapping[tuple[str, str], tuple[str, str]], pair
    ) -> tuple[str, str]:
        """The gate's verdict for a pair (either endpoint order), or blanks."""
        result = (
            lookup.get((str(pair.anchor_gtin).strip(), str(pair.partner_gtin).strip()))
            or lookup.get((str(pair.partner_gtin).strip(), str(pair.anchor_gtin).strip()))
        )
        if result is None:
            return "", ""
        return result

    # ── real base pairs (always kept; not mint-only) ─────────────────────────
    def base_pairs(self, *, negative: bool) -> list[PairRow]:
        """The real base x base pairs for one label class (kept every run)."""
        self._prepare()
        wanted = 0 if negative else 1
        population = (
            POPULATION_BASE_NEGATIVE if negative else POPULATION_BASE_POSITIVE
        )
        rows: list[PairRow] = []
        candidates = self.labeled[self.labeled["true_label"].astype(int) == wanted]
        for pair in _LOG.progress(
            candidates.itertuples(index=False), desc="base_pairs",
            unit="pair", total=len(candidates),
        ):
            row = self._base_pair_row(pair, wanted, population)
            if row is not None:
                rows.append(row)
        self._score_pairs(rows)
        return rows

    def _base_pair_row(
        self, pair, wanted: int, population: _POPULATION_TYPE
    ) -> PairRow | None:
        """One real base-pair row (None when an endpoint has no anchor row)."""
        left_row = self._row_by_gtin(str(pair.gtin1).strip())
        right_row = self._row_by_gtin(str(pair.gtin2).strip())
        if left_row is None or right_row is None:
            return None
        return PairRow(
            anchor_row=left_row,
            partner_row=right_row,
            label=wanted,
            population=population,
            is_real=True,
            anchor_gtin=str(pair.gtin1).strip(),
            partner_gtin=str(pair.gtin2).strip(),
            anchor_text=str(self._texts.iloc[left_row]),
            partner_text=str(self._texts.iloc[right_row]),
        )

    # ── emit ─────────────────────────────────────────────────────────────────
    def emit(self, run_tag: str) -> dict:
        """Ordered pass: block -> mine -> coverage -> mint -> attach -> write.

        Writes pairs.csv + manifest.json and returns the manifest. The
        emitted table is the ONLY interface to the trainer/evaluator.
        """
        with _LOG.section("negative_supply.emit"):
            self._prepare()
            real_rows, covered, minted_rows = self._supply_stages()
            everything = self._collect_pairs(real_rows, minted_rows)
            frame = self._pairs_frame(everything)
            folder = self._run_folder(run_tag)
            frame = self._attached_gate_columns(frame)
            frame.to_csv(folder / "pairs.csv", index=False)
            anchors_total, covered_total = self._coverage_counts(covered)
            from core.coverage_contracts import NegativeSupplyCoverage
            coverage = NegativeSupplyCoverage(
                anchors_total=anchors_total,
                anchors_with_real_partner=covered_total,
                coverage_share=round(covered_total / max(anchors_total, 1), 4),
            )
            manifest = self._manifest(run_tag, coverage, frame, folder)
            (folder / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            self.pairs = frame
            return manifest

    def _supply_stages(self) -> tuple[list[PairRow], set[int], list[PairRow]]:
        """Ordered supply stages: block -> mine real partners -> mint the rest.

        The count of anchors WITH a real partner decides the mint's share;
        ``covered`` is the emitting funnel's coverage key downstream.
        """
        self.block()
        real_rows = self.mine_real_partners()
        covered = {pair.anchor_row for pair in real_rows}
        minted_rows = self.mint(covered)
        return real_rows, covered, minted_rows

    def _collect_pairs(
        self, real_rows: list[PairRow], minted_rows: list[PairRow]
    ) -> list[PairRow]:
        """Every emitted population: base rows first, real, minted, edited."""
        return (
            self.base_pairs(negative=True)
            + self.base_pairs(negative=False)
            + real_rows
            + minted_rows
            + self.edited_positives()
        )

    def _pairs_frame(self, everything: list[PairRow]) -> pd.DataFrame:
        """PairRow models -> the emitted table (one dump pass, one bar)."""
        return pd.DataFrame(
            [pair.model_dump() for pair in _LOG.progress(
                everything, desc="pair_dump", unit="pair", total=len(everything))]
        )

    def _run_folder(self, run_tag: str) -> Path:
        """The run's lane output folder, created on demand."""
        from core.common import RESULTS

        folder = RESULTS / "negative_supply" / run_tag
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _attached_gate_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The frame with shadow-gate columns attached (comparison only)."""
        if self.spec.shadow_gate:
            self.pairs = frame
            self.attach_shadow_gate()
            frame = self.pairs
        return frame

    def _manifest(self, run_tag: str, coverage, frame: pd.DataFrame, folder: Path) -> dict:
        """The run's manifest document (spec + coverage + populations + funnel)."""
        return {
            "schema": "er-negative-supply-v1",
            "run_tag": run_tag,
            "spec": self.spec.model_dump(mode="json"),
            "coverage": coverage.model_dump(mode="json"),
            "populations": frame["population"].value_counts().to_dict(),
            "funnel": self.funnel,
            "pairs_sha256": hashlib.sha256(
                (folder / "pairs.csv").read_bytes()
            ).hexdigest(),
        }

    def _coverage_counts(self, covered: set[int]) -> tuple[int, int]:
        """(anchors_total, covered_total) per the configured entity level."""
        if self.spec.mint.entity_level == "gtin":
            gtins = self.df["gtin"].fillna("").astype(str).str.strip().to_numpy()
            anchors_total = len(set(gtins[self.anchor_mask.to_numpy()]))
            covered_total = len({gtins[position] for position in covered})
        else:
            anchors_total = int(self.anchor_mask.sum())
            covered_total = len(covered)
        return anchors_total, covered_total


def _suppress_same_gtin(
    chunk: "np.ndarray", gtins: "np.ndarray", start: int
) -> None:
    """Suppress an anchor's own gtin in its candidate rows (in place).

    Duplicate listings are excluded BEFORE the top-k budget is allocated, so
    the budget is spent on other-gtin partners only.
    """
    for position in range(len(chunk)):
        chunk[position, gtins == gtins[start + position]] = -np.inf


@timed
def main() -> None:
    RunLogger.configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--spec", type=Path, default=None)
    args = parser.parse_args()
    from core.common import F, load_dataset_deduped

    supply = NegativeSupply(
        spec=load_spec(args.spec),
        df=load_dataset_deduped(),
        canonical=pd.read_csv(F["canonical_records"], dtype=str, keep_default_na=False),
        gates=pd.read_csv(F["gate_results"], dtype={"gtin1": str, "gtin2": str}, keep_default_na=False),
        labeled=pd.read_csv(F["labeled_pairs"], dtype={"gtin1": str, "gtin2": str}, keep_default_na=False),
    )
    manifest = supply.emit(args.run_tag)
    print(json.dumps({"coverage": manifest["coverage"],
                      "populations": manifest["populations"],
                      "pairs": manifest["populations"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
