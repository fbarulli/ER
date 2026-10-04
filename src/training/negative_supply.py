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
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    import os

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
_VOLUME_TOKEN_RE = re.compile(r"\bvolume_ml_\d+(?:\.\d+)?\b")
_FLAVOR_TOKEN_RE = re.compile(r"\bflavor_[a-z0-9_]+\b")
_SET_CHARS = str.maketrans("", "", "'\"{}[]()")
_EMPTY_CELL_NORMS = frozenset(
    {"", "frozenset()", "frozenset", "set()", "set", "nan", "none"}
)


def read_set_column(raw: object) -> frozenset[str]:
    """A canonical CSV cell like "{'355.0', '500 ml'}" -> frozenset."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return frozenset()
    text = str(raw).translate(_SET_CHARS).strip()
    if text.lower() in _EMPTY_CELL_NORMS:
        return frozenset()
    return frozenset(atom.strip() for atom in text.split(",") if atom.strip())


def finalized_texts(df: pd.DataFrame) -> pd.Series:
    """Finalized model-input text per row via the SSOT builder (no re-normalization)."""
    from core.record_linkage import finalized_texts as finalized

    return finalized(df)


# ── the ONE token move both labels share ────────────────────────────────────
def token_move(
    text: str, dimension: str, replacement: Mapping[str, str]
) -> tuple[str, str, str] | None:
    """ONE whitelisted token move on a text copy; returns (new, from, to).

    Deterministic: the leftmost whitelisted token with a replacement target
    lands once. No move is invented when the text carries no such surface.
    """
    if dimension == "volume":
        pattern: re.Pattern = _VOLUME_TOKEN_RE
        prefix = "volume_ml_"
    elif dimension == "flavor":
        pattern = _FLAVOR_TOKEN_RE
        prefix = "flavor_"
    else:
        raise ValueError(f"unwhitelisted move: {dimension}")
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


def _dimension_pool(
    canonical: pd.DataFrame, dimension: str
) -> frozenset[str]:
    """Every atom observed in the corpus for one dimension's set column."""
    if dimension not in _DIMENSION_COLUMNS:
        raise ValueError(f"unpooled dimension: {dimension}")
    pool: set[str] = set()
    for raw in canonical[_DIMENSION_COLUMNS[dimension]]:
        pool |= read_set_column(raw)
    return frozenset(pool)


def _donor_pool(
    canonical: pd.DataFrame, dimension: str, exclude: frozenset[str]
) -> tuple[str, ...]:
    """Sorted unique corpus atoms usable as move targets (token-shaped only)."""
    atoms = re.compile(r"\d+(?:\.\d+)?" if dimension == "volume" else r"[a-z0-9_]+")
    return tuple(
        sorted(
            value
            for value in _dimension_pool(canonical, dimension)
            if value not in exclude and atoms.fullmatch(value)
        )
    )


_HANDLER_TOKEN_LEN = len("volume_ml_")
_FLAVOR_PREFIX = "flavor_"


def _text_atoms(text: str, dimension: str) -> tuple[str, ...]:
    pattern = _VOLUME_TOKEN_RE if dimension == "volume" else _FLAVOR_TOKEN_RE
    length = _HANDLER_TOKEN_LEN if dimension == "volume" else len(_FLAVOR_PREFIX)
    return tuple(m.group(0)[length:] for m in pattern.finditer(text))


# ── the orchestrator ────────────────────────────────────────────────────────
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
    _records_by_gtin: Mapping = PrivateAttr(default_factory=dict)
    _texts: pd.Series | None = PrivateAttr(default=None)
    _rng: np.random.Generator | None = PrivateAttr(default=None)
    _rows_by_gtin: dict[str, int] | None = PrivateAttr(default=None)
    _donor_pools: dict[str, tuple[str, ...]] = PrivateAttr(default_factory=dict)
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
        if not self._records_by_gtin:
            self._records_by_gtin = {
                str(row["gtin"]).strip(): row
                for row in self.canonical.to_dict("records")
            }
        if self._rows_by_gtin is None:
            self._rows_by_gtin = {}
            for position, gtin in enumerate(self.df["gtin"].fillna("").astype(str).str.strip()):
                self._rows_by_gtin.setdefault(gtin, position)
        if self._texts is None:
            self._texts = finalized_texts(self.df)
        if self._rng is None:
            self._rng = np.random.default_rng(self.spec.seed)

    def canonical_by_gtin(self, gtin: str) -> Mapping | None:
        self._prepare()
        return self._records_by_gtin.get(gtin)

    @property
    def anchor_mask(self) -> pd.Series:
        from core.gtin import gtin_validity

        self._prepare()
        gtins = self.df["gtin"].fillna("").astype(str).str.strip()
        valid = pd.Series(gtin_validity(gtins).to_numpy(), index=self.df.index)
        titled = self._texts.str.strip().ne("")
        return valid & gtins.ne("") & titled

    def _row_by_gtin(self, gtin: str) -> int | None:
        self._prepare()
        return self._rows_by_gtin.get(gtin)

    def _donors(self, dimension: str, exclude: frozenset[str]) -> tuple[str, ...]:
        if dimension not in self._donor_pools:
            self._donor_pools[dimension] = _donor_pool(self.canonical, dimension, frozenset())
        return tuple(value for value in self._donor_pools[dimension] if value not in exclude)

    # ── stage 1: blocking ────────────────────────────────────────────────────
    def block(self) -> pd.DataFrame:
        """Real different-GTIN candidates per anchor, TF-IDF cosine top-k."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._prepare()
        anchors = np.flatnonzero(self.anchor_mask.to_numpy())
        pool_texts = self._texts.iloc[anchors].tolist()
        gtins = (
            self.df["gtin"].fillna("").astype(str).str.strip().to_numpy()[anchors]
        )
        top_k = self.spec.blocker.top_k
        if len(anchors) == 0:
            self.candidates = pd.DataFrame()
            self.funnel["block"] = {"anchors": 0, "candidates": 0}
            return self.candidates
        vectorizer = TfidfVectorizer(sublinear_tf=True)
        matrix = vectorizer.fit_transform(pool_texts)
        self._block_vectorizer = vectorizer
        self._block_matrix = matrix
        self._block_anchor_positions = {int(anchor): index for index, anchor in enumerate(anchors)}
        rows: list[tuple[int, int, float]] = []
        chunk_rows = self.spec.blocker.chunk_rows
        for start in range(0, len(anchors), chunk_rows):
            chunk = (matrix[start:start + chunk_rows] @ matrix.T).toarray()
            chunk = chunk.astype(np.float32)
            # Exclude duplicate listings before allocating the top-k budget.
            for position in range(len(chunk)):
                chunk[position, gtins == gtins[start + position]] = -np.inf
            limit = min(top_k, len(anchors))
            local = np.argpartition(-chunk, limit - 1, axis=1)[:, :limit]
            local = np.take_along_axis(local, np.argsort(-np.take_along_axis(chunk, local, axis=1), axis=1), axis=1)
            for position, nearest in enumerate(local):
                for column in nearest:
                    score = float(chunk[position, column])
                    if score <= self.spec.blocker.min_score:
                        break
                    if gtins[start + position] == gtins[column]:
                        continue
                    rows.append((
                        int(anchors[start + position]),
                        int(anchors[column]),
                        round(score, 6),
                    ))
        self.candidates = pd.DataFrame(
            rows, columns=["anchor_row", "candidate_row", "score"]
        ).drop_duplicates(subset=["anchor_row", "candidate_row"])
        self.funnel["block"] = {
            "anchors": int(len(anchors)),
            "candidates": int(len(self.candidates)),
            "min_score": self.spec.blocker.min_score,
            "top_k": top_k,
        }
        return self.candidates

    # ── stage 2: real one-diff partners ──────────────────────────────────────
    def mine_real_partners(self) -> list[PairRow]:
        """Real different-GTIN pairs with EXACTLY ONE whitelisted diff.

        Every candidate above the floor is evaluated; the funnel reports
        survivors by differing-dimension count, so 'why N?' is answerable
        from the trace without re-running the filters.
        """
        self._prepare()
        if self.candidates is None:
            self.block()
        passed: dict[tuple[str, str], dict[str, bool]] = {}
        rows: list[PairRow] = []
        census = {
            "no_canonical_record": 0, "diff_count_1": 0, "diff_count_other": 0,
        }
        for candidate in self.candidates.itertuples(index=False):
            left_gtin = str(self.df["gtin"].iloc[candidate.anchor_row]).strip()
            right_gtin = str(self.df["gtin"].iloc[candidate.candidate_row]).strip()
            if left_gtin not in self._records_by_gtin or right_gtin not in self._records_by_gtin:
                census["no_canonical_record"] += 1
                continue
            key = tuple(sorted((left_gtin, right_gtin)))
            if key not in passed:
                passed[key] = self.attribute_diff(left_gtin, right_gtin)
            diffs = passed[key]
            if sum(diffs.values()) == 1 and any(diffs[dim] for dim in self.spec.mint.moves):
                census["diff_count_1"] += 1
                rows.append(
                    PairRow(
                        anchor_row=candidate.anchor_row,
                        partner_row=candidate.candidate_row,
                        label=0,
                        population=POPULATION_REAL_PARTNER,
                        is_real=True,
                        anchor_gtin=left_gtin,
                        partner_gtin=right_gtin,
                        anchor_text=str(self._texts.iloc[candidate.anchor_row]),
                        partner_text=str(self._texts.iloc[candidate.candidate_row]),
                        score=float(candidate.score),
                        diff_dimension=next(
                            dim for dim in self.spec.mint.moves if diffs[dim]
                        ),
                    )
                )
            else:
                census["diff_count_other"] += 1
        rows.sort(key=lambda pair: (pair.anchor_row, pair.partner_row))
        self.funnel["mine_real"] = {**census, "real_partners": len(rows)}
        return rows

    # ── stage 3: minted partners for uncovered anchors ───────────────────────
    def mint(self, covered: set[int]) -> list[PairRow]:
        """ONE partner per uncovered anchor with a single whitelisted move.

        The partner is a synthetic token-move copy of its own anchor (the
        anchor text stays untouched); the donor value is drawn deterministically
        from the sorted corpus pool under the spec seed. When the anchor's
        text carries no surface for the preferred move, the other whitelisted
        move is tried before the anchor is skipped (reported, not silent).
        """
        self._prepare()
        gtins = self.df["gtin"].fillna("").astype(str).str.strip().to_numpy()
        covered_gtins = {gtins[position] for position in covered}
        uncovered = [
            int(position)
            for position, covered_row in
            enumerate(self.anchor_mask.to_numpy())
            if covered_row and (
                gtins[position] not in covered_gtins
                if self.spec.mint.entity_level == "gtin" else position not in covered
            )
        ]
        rows: list[PairRow] = []
        skipped = {"no_move_surface": 0, "empty_pool": 0, "target_cap": 0,
                   "same_entity": 0, "below_blocker_floor": 0}
        minted_gtins: set[str] = set()
        moves = self.spec.mint.moves or ("flavor",)
        for sequence, anchor in enumerate(sorted(uncovered)):
            if self.spec.mint.entity_level == "gtin" and gtins[anchor] in minted_gtins:
                skipped["same_entity"] += 1
                continue
            if len(rows) >= self.spec.mint.max_minted:
                skipped["target_cap"] += 1
                continue
            if self.spec.mint.prefer == "alternate":
                ordered = (moves[(sequence) % len(moves)],)
            elif self.spec.mint.prefer == "flavor":
                ordered = ("flavor", ) if "flavor" in moves else moves
            else:
                ordered = ("volume",) if "volume" in moves else moves
            # try the preferred move first, then any remaining whitelisted move
            attempted = list(ordered) + [
                move for move in moves if move not in ordered
            ]
            outcome = None
            dimension = ""
            moved_from = ""
            moved_to = ""
            for move in attempted:
                atoms = set(_text_atoms(str(self._texts.iloc[anchor]), move))
                pool = self._donors(move, exclude=frozenset(atoms))
                if not pool:
                    skipped["empty_pool"] += 1
                    continue
                moved_to = pool[int(self._rng.integers(len(pool)))]
                replacement = {atom: moved_to for atom in atoms}
                outcome = token_move(str(self._texts.iloc[anchor]), move, replacement)
                if outcome is not None:
                    dimension, moved_from = move, outcome[1]
                    outcome = (outcome[0], outcome[1], outcome[2])
                    break
            if outcome is None:
                skipped["no_move_surface"] += 1
                continue
            rows.append(
                PairRow(
                    anchor_row=anchor,
                    partner_row=-1,
                    label=0,
                    population=POPULATION_MINTED_PARTNER,
                    is_real=False,
                    anchor_gtin=str(self.df["gtin"].iloc[anchor]).strip(),
                    partner_gtin="",
                    anchor_text=str(self._texts.iloc[anchor]),
                    partner_text=outcome[0],
                    edit_field=dimension,
                    edit_from=moved_from,
                    edit_to=moved_to,
                )
            )
            minted_gtins.add(gtins[anchor])
        rows.sort(key=lambda pair: pair.anchor_row)
        # minted partners get the same blocker cosine so no texture feature
        # separates them from real arms downstream
        self._score_pairs(rows)
        eligible = [pair for pair in rows if pair.score > self.spec.blocker.min_score]
        skipped["below_blocker_floor"] = len(rows) - len(eligible)
        rows = eligible
        self.funnel["mint"] = {
            **skipped,
            "anchors_uncovered": (len({gtins[position] for position in uncovered})
                                  if self.spec.mint.entity_level == "gtin" else len(uncovered)),
            "uncovered_sku_rows": len(uncovered),
            "minted": len(rows),
        }
        return rows

    def _score_pairs(self, rows: list[PairRow]) -> None:
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
        for positive in self.labeled[self.labeled["true_label"].astype(int) == 1].itertuples(index=False):
            left_row = self._row_by_gtin(str(positive.gtin1).strip())
            right_row = self._row_by_gtin(str(positive.gtin2).strip())
            if left_row is None or right_row is None:
                continue
            left_record = self.canonical_by_gtin(str(positive.gtin1).strip())
            right_record = self.canonical_by_gtin(str(positive.gtin2).strip())
            outcome = None
            for move in self.spec.mint.moves or ("flavor",):
                atoms = set(
                    _text_atoms(str(self._texts.iloc[left_row]), move)
                ) | set(_text_atoms(str(self._texts.iloc[right_row]), move))
                pool = self._donors(move, exclude=frozenset(atoms))
                if not pool:
                    continue
                moved_to = pool[int(self._rng.integers(len(pool)))]
                replacement = {atom: moved_to for atom in atoms}
                left_move = token_move(str(self._texts.iloc[left_row]), move, replacement)
                right_move = token_move(str(self._texts.iloc[right_row]), move, replacement)
                if left_move is None and right_move is None:
                    continue
                origin = left_move[1] if left_move is not None else right_move[1]
                outcome = (
                    left_move[0] if left_move is not None else str(self._texts.iloc[left_row]),
                    right_move[0] if right_move is not None else str(self._texts.iloc[right_row]),
                    origin,
                    move,
                    moved_to,
                )
                break
            if outcome is None:
                continue
            rows.append(
                PairRow(
                    anchor_row=left_row,
                    partner_row=right_row,
                    label=1,
                    population=POPULATION_EDITED_POSITIVE,
                    is_real=True,
                    anchor_gtin=str(positive.gtin1).strip(),
                    partner_gtin=str(positive.gtin2).strip(),
                    anchor_text=outcome[0],
                    partner_text=outcome[1],
                    edit_field=outcome[3],
                    edit_from=outcome[2],
                    edit_to=outcome[4],
                )
            )
        return rows

    # ── attribute diff over canonical set columns ────────────────────────────
    def attribute_diff(self, anchor_gtin: str, partner_gtin: str) -> dict[str, bool]:
        left = self.canonical_by_gtin(anchor_gtin)
        right = self.canonical_by_gtin(partner_gtin)
        outcome: dict[str, bool] = {}
        for dimension in _DIFF_DIMENSIONS:
            left_atoms = (
                read_set_column(left.get(_DIMENSION_COLUMNS[dimension]))
                if left is not None else frozenset()
            )
            right_atoms = (
                read_set_column(right.get(_DIMENSION_COLUMNS[dimension]))
                if right is not None else frozenset()
            )
            outcome[dimension] = bool(left_atoms and right_atoms
                                      and left_atoms != right_atoms)
        return outcome

    # ── shadow-gate attach ───────────────────────────────────────────────────
    def attach_shadow_gate(self) -> None:
        """Lookup gate_design on (gtin1,gtin2) pairs — comparison columns only."""
        pairs = self.pairs
        if not isinstance(pairs, pd.DataFrame) or pairs.empty or not self.spec.shadow_gate:
            return
        lookup: dict[tuple[str, str], tuple[str, str]] = {}
        for g in self.gates.itertuples(index=False):
            lookup[(str(g.gtin1).strip(), str(g.gtin2).strip())] = (
                str(g.gate_decision), str(g.gate_reason),
            )
        decisions: list[str] = []
        reasons: list[str] = []
        for pair in pairs.itertuples(index=False):
            result = lookup.get((str(pair.anchor_gtin).strip(), str(pair.partner_gtin).strip())) \
                or lookup.get((str(pair.partner_gtin).strip(), str(pair.anchor_gtin).strip()))
            decisions.append(result[0] if result else "")
            reasons.append(result[1] if result else "")
        out = self.pairs.copy()
        out["shadow_gate_decision"] = decisions
        out["shadow_gate_reason"] = reasons
        self.pairs = out

    # ── real base pairs (always kept; not mint-only) ─────────────────────────
    def base_pairs(self, *, negative: bool) -> list[PairRow]:
        self._prepare()
        wanted = 0 if negative else 1
        population = (
            POPULATION_BASE_NEGATIVE if negative else POPULATION_BASE_POSITIVE
        )
        rows: list[PairRow] = []
        for pair in self.labeled[self.labeled["true_label"].astype(int) == wanted].itertuples(index=False):
            left_row = self._row_by_gtin(str(pair.gtin1).strip())
            right_row = self._row_by_gtin(str(pair.gtin2).strip())
            if left_row is None or right_row is None:
                continue
            rows.append(
                PairRow(
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
            )
        self._score_pairs(rows)
        return rows

    # ── emit ─────────────────────────────────────────────────────────────────
    def emit(self, run_tag: str) -> dict:
        """Ordered pass: block -> mine -> coverage -> mint -> attach -> write.

        Writes pairs.csv + manifest.json and returns the manifest. The
        emitted table is the ONLY interface to the trainer/evaluator.
        """
        from core.common import RESULTS

        self._prepare()
        self.block()
        real_rows = self.mine_real_partners()
        covered = {pair.anchor_row for pair in real_rows}
        minted_rows = self.mint(covered)
        everything = (
            self.base_pairs(negative=True)
            + self.base_pairs(negative=False)
            + real_rows
            + minted_rows
            + self.edited_positives()
        )
        frame = pd.DataFrame([pair.model_dump() for pair in everything])
        folder = RESULTS / "negative_supply" / run_tag
        folder.mkdir(parents=True, exist_ok=True)
        if self.spec.shadow_gate:
            self.pairs = frame
            self.attach_shadow_gate()
            frame = self.pairs
        frame.to_csv(folder / "pairs.csv", index=False)
        if self.spec.mint.entity_level == "gtin":
            gtins = self.df["gtin"].fillna("").astype(str).str.strip().to_numpy()
            anchors_total = len(set(gtins[self.anchor_mask.to_numpy()]))
            covered_total = len({gtins[position] for position in covered})
        else:
            anchors_total = int(self.anchor_mask.sum())
            covered_total = len(covered)
        from core.coverage_contracts import NegativeSupplyCoverage
        coverage = NegativeSupplyCoverage(
            anchors_total=anchors_total, anchors_with_real_partner=covered_total,
            coverage_share=round(covered_total / max(anchors_total, 1), 4),
        )
        manifest = {
            "schema": "er-negative-supply-v1",
            "run_tag": run_tag,
            "spec": self.spec.model_dump(mode="json"),
            "coverage": coverage.model_dump(mode='json'),
            "populations": frame["population"].value_counts().to_dict(),
            "funnel": self.funnel,
            "pairs_sha256": hashlib.sha256(
                (folder / "pairs.csv").read_bytes()
            ).hexdigest(),
        }
        (folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        self.pairs = frame
        return manifest


# ── GTIN-grouped split + discriminator ──────────────────────────────────────
def gtin_group_split(frame: pd.DataFrame, *, k: int = 4, seed: int = 1337) -> pd.Series:
    """Deterministic GTIN-grouped bucket index (0..k-1) per pair row.

    Entities are union components over the gtin namespace: pairs sharing an
    endpoint share a bucket. Minted synthetic rows carry -1 (never scored;
    evaluation is real-pairs-only by owner ruling). Components are ordered
    canonically (shared core.disjoint_sets), so the bucket assignment depends
    only on the component PARTITION, never on union order or root labels.
    """
    gtins = frame["anchor_gtin"].astype(str).str.strip()
    partners = frame["partner_gtin"].astype(str).str.strip()
    groups = DisjointSet()
    for left, right, minted in zip(gtins, partners, frame["population"]):
        if minted != POPULATION_MINTED_PARTNER and right and left:
            groups.union(left, right)
    components = groups.components()
    rng = np.random.default_rng(seed)
    bucket_of_component = np.asarray(rng.permutation(len(components))) % k
    fold_of = {
        item: int(bucket_of_component[index])
        for index, component in enumerate(components)
        for item in component
    }
    out = []
    for left, minted in zip(gtins, frame["population"]):
        out.append(-1 if minted == POPULATION_MINTED_PARTNER else fold_of.get(left, -1))
    return pd.Series(out, index=frame.index, name="group_fold")


# ── trainer bridge (config-gated; the gate path stays the default) ──────────
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


def build_lane_training_data(
    df: pd.DataFrame, *, payload_variant: str = "full",
    run_tag: str, mint_cap: float = 1.0,
) -> dict:
    """Trainer contract with negatives from the lane instead of the gate.

    Reuses ``pipeline.build_training_data`` for the payload and POSITIVES
    (positives are gate-independent — a row paired with its own canonical),
    then REPLACES the gate-derived negatives:

    * real negatives = the lane's ``base_negative`` + ``real_partner`` rows,
      mapped by GTIN through the trainer's own ``gtin_to_row`` /
      ``gtin_to_canon_idx`` (so a real-partner pair equals what the gate path
      would produce for the same anchors);
    * minted partners = appended as leaf payload rows (``row_bc``
      ``minted:<anchor>:<idx>`` — no canonical, no neighbours) and returned in
      ``neg_minted``, which is TRAINING-ONLY. They never enter ``neg``, so
      evaluation stays real-pairs-only and the augmentation stages (which read
      ``neg``) never re-edit an already-minted record.

    A minted partner's text is the lane's single whitelisted move replayed on
    the TRAINER's own payload text for that anchor (the lane's blocking text is
    title-only; the trainer's payload carries attributes), so text format
    matches every other payload row.

    The gate-derived miners (targeted-attribute, cross-brand) are DROPPED: the
    lane's real partners are their replacement, per the 2026-10-03 ruling.
    """
    from pipeline import build_training_data

    base = build_training_data(df, payload_variant=payload_variant)
    return assemble_lane_bundle(
        base, load_pairs(run_tag), n_sku=len(df), mint_cap=mint_cap
    )


def assemble_lane_bundle(
    base: Mapping, pairs: pd.DataFrame, *, n_sku: int, mint_cap: float = 1.0
) -> dict:
    """Pure assembler: base bundle + lane pairs -> trainer contract.

    Split out from :func:`build_lane_training_data` so the negative swap, the
    minted-leaf append and the cap are unit-testable without the live
    canonical/gate artifacts.
    """
    payload = list(base["payload"])
    row_bc = [str(value) for value in base["row_bc"]]
    structured = [list(row) for row in base["structured_features"]]
    gtin_to_row = dict(base["gtin_to_row"])
    canon_gtins = row_bc[n_sku:]
    gtin_to_canon_idx = {g: n_sku + i for i, g in enumerate(canon_gtins) if g}
    payload_source = ["sku"] * n_sku + ["canonical"] * len(canon_gtins)

    # ── real negatives (gate decision replaced by the lane) ──
    neg_rows: list[tuple[int, int]] = []
    neg_source: list[str] = []
    real_populations = (POPULATION_BASE_NEGATIVE, POPULATION_REAL_PARTNER)
    for pair in pairs.itertuples(index=False):
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

    # ── minted partners: leaf rows, TRAINING-ONLY, capped by mint_cap ──
    minted = pairs[pairs["population"] == POPULATION_MINTED_PARTNER].sort_values(
        "anchor_row", kind="stable"
    )
    n_real = len(neg_rows)
    if mint_cap >= 1.0:
        allowed_minted = len(minted)
    else:
        allowed_minted = (
            int((mint_cap * n_real) / (1.0 - mint_cap)) if mint_cap > 0 else 0
        )
    allowed_minted = min(allowed_minted, len(minted))
    dropped_cap = len(minted) - allowed_minted
    neg_minted_rows: list[tuple[int, int]] = []
    width = len(structured[0]) if structured else 0
    for pair in list(minted.itertuples(index=False))[:allowed_minted]:
        anchor = int(pair.anchor_row)
        if not (0 <= anchor < n_sku):
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

    stats = dict(base["stats"])
    stats.update({
        "negative_supply_mode": "lane",
        "n_lane_base_negative": neg_source.count("base_negative"),
        "n_lane_real_partner": neg_source.count("real_partner"),
        "n_lane_minted": len(neg_minted_rows),
        "n_lane_minted_dropped_cap": dropped_cap,
    })

    from core.schemas import TrainingData as _TrainingData

    bundle = _TrainingData(
        payload=payload,
        structured_features=structured,
        row_bc=np.array(row_bc),
        pos=np.asarray(base["pos"], dtype=int).reshape(-1, 2),
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


def main() -> None:
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
