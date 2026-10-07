"""06: Tiered exact-duplicate dedupe — deliberate representatives, not blind drops.

05b found 13,519 rows in retailer+title duplicate groups, but only 4,439 also
match on price — so ~9,080 same-title rows at one retailer are DISTINCT
marketplace offers (Gittigidiyor-style sellers), not scrape glitches.
Collapsing them is PRICE-AGGREGATION, not noise-removal, so the representative
is chosen deliberately using gtin presence and descriptor completeness;
price does not rank representatives.

Tiers (each operates on the rows surviving the previous tier):
  T1  retailer+gtin      same product at one retailer -> 1 row (gtin is
                           ground truth, highest confidence).
  T1.5 retailer+malformed   same retailer + same CHECKSUM-INVALID gtin +
      gtin               same product -> 1 row. Identity decided by the
                           descriptor bundle (core.sku_identity), never by
                           price or URL.
  T2  retailer+title+       matching identity partition -> 1 row. Price is
      identity partition    NOT part of the key (it is a seller attribute);
                            same retailer+title+gtin at two prices is the
                            same product offered twice, which is T3's
                            price-aggregation and is flagged, not silent.
  T3  retailer+title +      varying price -> one deliberate representative per
      identity partition   IDENTITY PARTITION, flagged as an ambiguous offer
                            (pv) so the price-collapse is auditable.

The identity partition in T3 is load-bearing, not cosmetic (measured
2026-09-30): keying the collapse on (retailer,title) ALONE merged two distinct
checksum-valid products whenever one retailer listed the same title string
under two gtins. 692 groups, 1,086 product-listings deleted, and 264
products lost their ONLY row in the corpus — present in canonical_records.csv,
absent from the deduped output, with a representative carrying a sibling's
gtin. T2 already detects exactly this hazard and deferred to T3; T3 then
merged them anyway. The partition closes it: rows carrying DIFFERENT trusted
gtins never collapse together. Rows without a gtin may share a populated
title for price aggregation; rows with missing titles stay separate in T2/T3.

Identity is decided by `core.sku_identity` (the SSOT) and by nothing else.
Representative choice deliberately ignores `price`, `url` and `image_url`:
price moves with the seller, and the URL columns are export noise — the
completeness score counts DESCRIPTOR fields only.

Class map (one owner per responsibility):
  - IdentityDecider   — adjudication, descriptor verdicts, review rows
  - FrameWorkbook     — work-frame prep: helpers, staging, protections
  - TieredCollapse    — the four tiers + representative bookkeeping
  - SanityGate        — the stage's hard invariants
  - StageWriter       — every CSV the stage publishes

Writes:
  data/dataset_deduped.csv             deduped dataset (pipeline input)
  data/sku_to_rep.csv                  raw SKU (sku_id) -> rep_id
  results/06_dedupe_summary.csv        per-tier counts
  results/06_ambiguous_offer_groups.csv  retailer+title >1 price
  results/06_dedupe_removals.csv       one review row per removed raw SKU
  results/training/dedupe_conflicts.csv  descriptor conflicts left unresolved
  results/manifests/dedupe.json        per-stage manifest, written LAST
                                        (SILENT_DROPS task 4 — the stage's
                                        completion marker: closure
                                        input_rows == output_rows + dropped,
                                        every output sha256-pinned, all
                                        writes atomic)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from core.common import DATA_PATH, SEED, F, data_cfg, ensure_parent, load_dataset
from core.deduplication import collapse_representatives
from core.gtin import gtin_validity
from core.manifest import atomic_write_csv, begin_manifest, finish_manifest
from core.run_log import RunLogger
from core.sku_identity import (
    completeness_frame,
    evaluate_sku_identity,
    row_identity,
)
from core.step_trace import timed

_LOG = RunLogger(__name__)

# output paths from the config SSOT (files.*) — were hardcoded here, the
# only filenames in the tree outside config/paths.yaml
CSV_SUMMARY = F["dedupe_summary"]
CSV_OFFERS = F["ambiguous_offer_groups"]
CSV_REMOVALS = F["removals"]
DEDUPED_PATH = F["dataset_deduped"]
SKU_TO_REP_PATH = F["sku_to_rep"]
CSV_CONFLICTS = F["dedupe_conflicts"]

HELPERS = ["_price", "_complete", "_has_bc", "_t2_bc", "_bc_valid", "_ident"]

# STEP-2 adjudicated verdicts (owner review 2026-09-29). The 10 residual
# groups that the descriptor bundle cannot decide unambiguously were reviewed
# on their ORIGINAL (unnormalized) data — full title, attributes, brand,
# category, price. 8 are genuine product splits (keep separate), 2 are
# same-product where a listing qualified a shared trait (collapse). Keyed by
# (retailer, gtin) in config/paths.yaml dedupe_adjudications so the T1.5
# loop can look up the reviewed verdict directly.
#
# This table is the reason the text predicate is allowed to decide T1.5 at
# all: measured on gtin-labeled ground truth (2026-09-30) the descriptor
# predicate alone merges 59.0% of provably-different pairs, because a
# missing descriptor reads as agreement. Absence of a conflict is therefore
# NOT sufficient — a byte-identical malformed gtin at one retailer plus a
# descriptor verdict is, and anything the verdict cannot settle is escalated
# here rather than merged.

_TIER_T1 = "T1 retailer+gtin"
_TIER_T15 = "T1.5 retailer+malformed-gtin+same-product"
_TIER_T2 = "T2 retailer+title+gtin"
_TIER_T3 = "T3 retailer+title+identity-partition (price-aggregation)"

# The only decisions core.sku_identity may return for two rows to collapse:
# "same" is proof, "compatible_unverified" is absence-of-contradiction —
# and absence is only enough when an owner verdict or trusted gtin backs it.
_COMPATIBLE_DECISIONS = frozenset({"same", "compatible_unverified"})


# ── product-identity decisions ──────────────────────────────────────────────
class IdentityDecider:
    """Whether a group's rows are the same product (the tier arbiter).

    STEP 0 — config-owned adjudicated overrides win outright.
    STEP 1 — the descriptor bundle decides: no dimension may PROVE the rows
      are different products. Category is deliberately not a veto: it is as
      noisy as the attribute cell on these rows (Reconstituted vs
      Not-from-Conjugate juice are the same product).
    STEP 2 — a group the bundle cannot settle is NOT merged. The absence of
      a conflict is not evidence of identity (measured 59.0% false-merge
      rate on gtin-labeled hard negatives), so the safe answer is to keep
      the rows apart and let the link lane adjudicate.

    `sub` frames must contain the descriptor columns.
    """

    @staticmethod
    def adjudicated(retailer: str, gtin: str) -> bool | None:
        """STEP 0 — the config-owned reviewed verdict for a group, if any.

        Read from data_cfg() on every call (the config SSOT is patched by
        tests and refreshable by design — never cached here).
        """
        for reviewed in data_cfg().dedupe_adjudications:
            if (retailer, gtin) == (reviewed.retailer, reviewed.gtin):
                return reviewed.decision == "collapse"
        return None

    @staticmethod
    def all_compatible(identities: list) -> bool:
        """No pair of identities may PROVE different products.

        Compatibility is not transitive: A={lemon,lime} can overlap B={lemon}
        and C={lime} while B conflicts with C. Check every pair before
        collapse. core.sku_identity owns the comparison, so the dedupe, the
        gate and the vetoes cannot drift apart.
        """
        return all(
            evaluate_sku_identity(left, right)["decision"] in _COMPATIBLE_DECISIONS
            for left, right in combinations(identities, 2)
        )

    @classmethod
    def same_product(cls, sub: pd.DataFrame, retailer: str, gtin: str) -> bool:
        """The full two-step decision: True when the rows are one product.

        Returns True when the group should collapse.
        """
        reviewed = cls.adjudicated(retailer, gtin)
        if reviewed is not None:
            return reviewed
        return cls.all_compatible(
            [row_identity(row) for row in sub.to_dict("records")]
        )

    @classmethod
    def pair_review_record(
        cls, anchor, other, retailer: str, gtin: str, titles: tuple[str, str]
    ) -> tuple[dict, bool]:
        """ONE pairing review row: (row, unresolved?).

        `label` distinguishes "descriptor proves different" from "descriptor
        had nothing to say" — the row is a review queue entry, not a silent
        either/or.
        """
        evaluation = evaluate_sku_identity(anchor, other)
        reasons = evaluation["identity_conflicts"]
        reviews = evaluation["review_dimensions"] + evaluation["unclassified_keys"]
        record = {
            "tier": "T1.5", "retailer": retailer, "gtin": gtin,
            "label": "proven_split" if reasons else "unresolved",
            "reasons": "|".join(reasons or [f"review:{r}" for r in reviews]) or "-",
            "title_a": titles[0], "title_b": titles[1],
            "brand_a": anchor.brand and " ".join(sorted(anchor.brand)) or "",
            "brand_b": other.brand and " ".join(sorted(other.brand)) or "",
        }
        return record, not reasons

    @classmethod
    def review_records(
        cls, sub: pd.DataFrame, retailer: str, gtin: str
    ) -> tuple[list[dict], int]:
        """Review rows for a group the descriptor bundle cannot settle.

        Anchors on the first row and records each pairing. Returns the rows
        and the count of unresolved (non-proven) ones.
        """
        identities = [row_identity(r) for r in sub.to_dict("records")]
        anchor = identities[0]
        titles = (sub["sku_name_eng"].iat[0], sub["sku_name_eng"].iat[1])
        records: list[dict] = []
        unresolved = 0
        for other in identities[1:]:
            record, is_unresolved = cls.pair_review_record(
                anchor, other, retailer, gtin, titles
            )
            records.append(record)
            unresolved += int(is_unresolved)
        return records, unresolved


# Pinned call sites (scripts/dedupe_invalid_gtin_groups.py and the identity
# tests) import this decision through the module face.
def _same_product_by_title(sub: pd.DataFrame, retailer: str, gtin: str) -> bool:
    """Two-step product-identity decision (see :class:`IdentityDecider`)."""
    return IdentityDecider.same_product(sub, retailer, gtin)


# ── work-frame preparation ──────────────────────────────────────────────────
class FrameWorkbook:
    """Every transformation the tiers key on, from raw export to the
    population split: helper columns, checksum staging, title protections
    and the raw-frame ambiguous-offer audit."""

    @classmethod
    def load(cls) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The raw export under identity links, plus its tier-ready copy."""
        from core.identity_policy import apply_identity_links

        df = apply_identity_links(load_dataset())
        return df, cls.helper_columns(df)

    @staticmethod
    def helper_columns(df: pd.DataFrame) -> pd.DataFrame:
        """The per-row work columns the tiers key and rank on.

        Descriptor completeness, NOT `df.notna().sum()`: the old score
        counted `url` and `image_url`, so a listing survived on the strength
        of two export-noise columns while a complete title with no image
        lost (measured 2026-09-30). price is excluded from ranking too —
        it is a seller attribute, not a description of the product.
        """
        return df.assign(
            _price=pd.to_numeric(df["sku_last_price"], errors="coerce"),
            _complete=completeness_frame(df),
            _has_bc=(df["gtin"].fillna("").str.len() > 0).astype(int),
        )

    @staticmethod
    def missing_title_mask(frame: pd.DataFrame) -> pd.Series:
        """Rows whose title is absent or blank (whitespace-only = absent)."""
        return frame["sku_name_eng"].fillna("").astype(str).str.strip().eq("")

    @classmethod
    def protect_missing_titles(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Give untitled listings separate partitions before title-based collapse.

        T1 has already consumed trustworthy retailer/gtin identity. An
        absent title cannot prove identity for the remaining T2/T3 rows,
        even when both also lack a gtin. Use row positions rather than a
        possibly missing or repeated listing ID, preserving every such
        source row independently.
        """
        missing = cls.missing_title_mask(frame)
        if not missing.any():
            return frame
        frame = frame.copy()
        positions = np.flatnonzero(missing.to_numpy())
        frame.loc[missing, "_ident"] = [f"untitled-row:{i}" for i in positions]
        return frame

    @staticmethod
    def descriptor_review_positions(frame: pd.DataFrame) -> set:
        """Row labels in title-groups that FAIL the same descriptor review as T1.5.

        Only rows with an empty identity partition (no trusted key) are
        reviewed; a group where any pair proves different — or simply
        cannot be settled — is protected whole (compatibility is not
        transitive).
        """
        candidates = frame[frame["_ident"].eq("")]
        protected: list = []
        groups = list(candidates.groupby(["retailer", "sku_name_eng"], sort=False))
        for _, group in _LOG.progress(
            groups, desc="descriptor review groups", unit="group"
        ):
            if len(group) < 2:
                continue
            identities = [row_identity(row) for row in group.to_dict("records")]
            if not IdentityDecider.all_compatible(identities):
                protected.extend(group.index)
        return set(protected)

    @classmethod
    def protect_untrusted_title_conflicts(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Do not let T2/T3 undo descriptor splits left unresolved by T1.5.

        Empty identity partitions mean no trusted key. Before price
        aggregation, require every pair to pass the same descriptor review
        as T1.5. Preserve the whole group on a conflict: compatibility is
        not transitive.
        """
        protected = cls.descriptor_review_positions(frame)
        if not protected:
            return frame
        result = frame.copy()
        positions = [
            position for position, index in enumerate(result.index)
            if index in protected
        ]
        column = result.columns.get_loc("_ident")
        result.iloc[positions, column] = [
            f"descriptor-review-row:{position}" for position in positions
        ]
        return result

    @staticmethod
    def ambiguous_offer_audit(work: pd.DataFrame) -> pd.DataFrame:
        """retailer+title groups with >1 row and >1 price, from the RAW frame.

        Computed BEFORE any tier consumes the rows: deriving it from
        whatever survived to T3 made a group T2 collapsed first silently
        vanish from the audit trail.
        """
        prices = work.groupby(["retailer", "sku_name_eng"])["_price"].agg(
            ["size", "nunique", "min", "max"]
        ).rename(columns={"size": "count"})
        return prices[(prices["count"] > 1) & (prices["nunique"] > 1)]

    @staticmethod
    def stage_trusted_gtins(work: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
        """Checksum-validity bookkeeping + the trusted-identity partition.

        _bc_valid marks rows whose gtin PASSES the GS1 checksum (owner
        ruling, src/core/gtin.py) and is not held by the identity review;
        an invalid gtin is export noise, not identity. _ident is a row's
        TRUSTED gtin, or "" when it has none: T1 collapses within a
        partition by construction and T3 reuses it so two different
        products sharing a title string can never collapse together.
        Reviewed rows get their own "review:" partition. Returns the
        augmented frame and the set of trusted gtins present in the INPUT
        (the identity invariant's before-side).
        """
        from core.identity_policy import reviewed_row_mask

        bc_stripped = work["gtin"].fillna("").astype(str).str.strip()
        valid = gtin_validity(bc_stripped)
        before_valid_gtins = set(bc_stripped[valid]) - {""}
        held = reviewed_row_mask(work)
        work = work.assign(_bc_valid=(valid & ~held).to_numpy())
        work = work.assign(_ident=np.where(work["_bc_valid"], bc_stripped, ""))
        work.loc[held, "_ident"] = "review:" + work.loc[held, "sku_id"].astype(str)
        return work, before_valid_gtins

    @staticmethod
    def partition_by_gtin_trust(
        work: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split rows into (trusted-gtin, malformed-gtin, no-gtin) populations.

        Trusted = has a gtin that passes the GS1 checksum and is not held by
        the identity review; malformed rows fall through to T1.5, gtin-less
        rows to the title tiers.
        """
        with_bc = work[(work["_has_bc"] == 1) & (work["_bc_valid"])]
        malformed = work[(work["_has_bc"] == 1) & (~work["_bc_valid"])]
        no_bc = work[work["_has_bc"] == 0]
        return with_bc, malformed, no_bc


# Pinned call sites (the dedupe tests) exercise the title protections and
# the frame prep through the module face.
def _protect_missing_titles(frame: pd.DataFrame) -> pd.DataFrame:
    """Give untitled listings separate partitions (see :class:`FrameWorkbook`)."""
    return FrameWorkbook.protect_missing_titles(frame)


def _protect_untrusted_title_conflicts(frame: pd.DataFrame) -> pd.DataFrame:
    """Require the T1.5 descriptor review on title groups (see FrameWorkbook)."""
    return FrameWorkbook.protect_untrusted_title_conflicts(frame)


def _load_work_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The raw export under identity links, plus its tier-ready work copy."""
    return FrameWorkbook.load()


def _ambiguous_offer_audit(work: pd.DataFrame) -> pd.DataFrame:
    """retailer+title groups with >1 row and >1 price (see FrameWorkbook)."""
    return FrameWorkbook.ambiguous_offer_audit(work)


def _stage_trusted_gtins(work: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Checksum staging + trusted-identity partition (see FrameWorkbook)."""
    return FrameWorkbook.stage_trusted_gtins(work)


def _partition_by_gtin_trust(work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by gtin trust (see :class:`FrameWorkbook`)."""
    return FrameWorkbook.partition_by_gtin_trust(work)


# ── the tiers + representative bookkeeping ──────────────────────────────────
class TieredCollapse:
    """The four collapse tiers and the representative pointers they update.

    parent[i] = original row index of the surviving representative for row
    i. Updated per tier and resolved transitively at the end (a T1 survivor
    may itself be collapsed by T2/T3). The collapse is deterministic; no
    RNG is consumed.
    """

    def __init__(self, index: pd.Index) -> None:
        self.parent = {i: i for i in index}
        self.drops: dict[str, list] = {}
        self._identity = IdentityDecider()

    def record(self, tier: str, dropped: list) -> None:
        """Register one tier's deliberately dropped rows for the removal ledger."""
        self.drops[tier] = dropped

    @timed
    def _tier_t1(self, with_bc: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """T1: retailer+gtin -> one row (ground-truth identity).

        ONLY for rows that actually have a gtin ("no gtin" is not "same
        gtin") and whose gtin passes the checksum: 103 retailer+gtin groups
        carried >1 distinct title on a checksum-fail gtin and would silently
        merge different products. Invalid-gtin rows are not dropped: they
        fall through to T1.5 and the T2/T3 title-based tiers.
        Returns (t1_survivors, dropped_indices).
        """
        t1, dropped = collapse_representatives(
            with_bc, ["retailer", "gtin"], ["_complete"], [False], parent=self.parent
        )
        return t1, dropped

    @timed
    def _tier_t15(self, malformed: pd.DataFrame):
        """T1.5: same retailer + same MALFORMED gtin + same product -> one row.

        T1 refuses to collapse on an invalid gtin, but a malformed gtin that
        is byte-identical at one retailer is still a strong candidate, and
        the descriptor bundle is the arbiter (IdentityDecider). A group it
        cannot settle is kept whole and recorded as a review row. This
        recovers the 97 groups measured in the dedupe invalid-gtin audit
        (2026-09-29) while never merging genuinely different products.
        Returns (work, dropped, collapsed_groups, unresolved_groups, conflicts).
        """
        kept: list[pd.DataFrame] = []
        dropped: list = []
        conflicts: list[dict] = []
        collapsed = unresolved = 0
        groups = list(malformed.groupby(["retailer", "gtin"], sort=False))
        for (retailer, gtin), sub in _LOG.progress(
            groups, desc="T1.5 malformed-gtin groups", unit="group"
        ):
            piece, piece_dropped, piece_conflicts, piece_unresolved = (
                self._settle_group(sub, retailer, gtin)
            )
            kept.append(piece)
            dropped.extend(piece_dropped)
            conflicts.extend(piece_conflicts)
            collapsed += bool(piece_dropped)
            unresolved += piece_unresolved
        kept_frame = pd.concat(kept)
        _LOG.info(f"[dedupe] T1.5 complete: {len(dropped):,} dropped across "
                  f"{collapsed:,} groups; {unresolved:,} unresolved escalations")
        return kept_frame, dropped, collapsed, unresolved, conflicts

    def _settle_group(
        self, sub: pd.DataFrame, retailer: str, gtin: str
    ) -> tuple[pd.DataFrame, list, list[dict], int]:
        """Settle ONE malformed-gtin group: collapse it or escalate it.

        Returns (kept_frame, dropped_indices, conflict_rows, unresolved_count);
        a group with a single row passes through untouched.
        """
        if len(sub) <= 1:
            return sub, [], [], 0
        if self._identity.same_product(sub, retailer, gtin):
            rep_idx, rep = self._most_informative_representative(sub)
            for idx in sub.index:
                self.parent[idx] = rep_idx
            return rep, list(sub.index.difference([rep_idx])), [], 0
        records, group_unresolved = self._identity.review_records(
            sub, retailer, gtin
        )
        return sub, [], records, group_unresolved

    @timed
    def _tier_t2(self, work: pd.DataFrame):
        """T2: retailer+title+gtin -> one row (lossless).

        `price` is NOT part of the key: it is a seller attribute, and two
        rows at one retailer with the same title and the same product gtin
        but different prices are the same product offered twice — that is
        T3's price-aggregation, and the ambiguous-offer audit (computed
        from the raw frame) still records it.

        "Lossless" is ENFORCED by the gtin-agreement guard: rows collapse
        only when their trusted gtins agree (same non-empty identity, or
        both without one). Two DIFFERENT trusted gtins under one title are
        different products; those groups flow to T3 instead of collapsing
        here. Returns (work, dropped_indices, deferred_frame).
        """
        work = FrameWorkbook.protect_missing_titles(work)
        work = FrameWorkbook.protect_untrusted_title_conflicts(work)
        keyed = work.assign(_t2_bc=work["_ident"])
        # Split T2 by gtin agreement WITHIN each (retailer,title) group: an
        # all-same-identity group collapses losslessly; a group with >1
        # distinct trusted gtin carries genuinely different products — those
        # rows all flow to T3 (kept here via the conflict mask).
        bc_agrees = keyed.groupby(
            ["retailer", "sku_name_eng"], sort=False, dropna=False
        )["_t2_bc"].transform("nunique").le(1)
        clean = keyed[bc_agrees]
        deferred = keyed[~bc_agrees]
        collapsed, dropped = collapse_representatives(
            clean, ["retailer", "sku_name_eng", "_t2_bc"], ["_complete"], [False],
            parent=self.parent,
        )
        work = pd.concat([collapsed, deferred.drop(columns=["_t2_bc"])])
        _LOG.info(f"[dedupe] T2 complete: {len(dropped):,} dropped, "
                  f"{len(deferred):,} deferred to T3")
        return work, dropped, deferred

    @timed
    def _tier_t3(self, work: pd.DataFrame):
        """T3: retailer+title WITHIN AN IDENTITY PARTITION -> one representative.

        The `_ident` partition is the fix for the measured identity loss:
        keying on (retailer,title) alone deleted 1,086 product-listings
        carrying a valid gtin and erased 264 products from the corpus
        entirely. Rows sharing a title with NO trusted gtin still collapse
        together — the genuine price-aggregation case — flagged by the
        ambiguous-offer audit, not silent. Returns (work, dropped_indices).
        """
        collapsed, dropped = collapse_representatives(
            work, ["retailer", "sku_name_eng", "_ident"],
            ["_has_bc", "_complete", "sku_name_eng"], [False, False, True],
            parent=self.parent,
        )
        _LOG.info(f"[dedupe] T3 complete: {len(dropped):,} dropped")
        return collapsed, dropped

    @timed
    def _resolve_representatives(self, index: pd.Index) -> None:
        """Compress every pointer to its FINAL surviving representative.

        T1/T2 survivors may themselves be dropped by a later tier, so the
        chain is resolved transitively before the raw-SKU -> rep mapping is
        emitted.
        """
        for row in _LOG.progress(index, desc="resolve_representatives", unit="row"):
            while self.parent[self.parent[row]] != self.parent[row]:
                self.parent[row] = self.parent[self.parent[row]]

    def _sku_to_rep_frame(self, df: pd.DataFrame, work: pd.DataFrame) -> pd.DataFrame:
        """The raw-SKU -> surviving-representative mapping.

        rep_id is a 0-based position into the deduped output, so ITEM_ID
        derives from this step instead of re-deduping.
        """
        rep_pos = {idx: pos for pos, idx in enumerate(work.index)}
        return pd.DataFrame({
            "sku_id": df["sku_id"].to_numpy(),
            "rep_id": [rep_pos[self.parent[i]] for i in df.index],
        })

    def tier_of(self) -> dict:
        """The tier name per deliberately dropped roster index.

        An index appears in exactly one tier's drop list (a tier only drops
        rows that survived the previous ones), so no entry is ever
        overwritten.
        """
        tier_of: dict = {}
        for tier in (_TIER_T1, _TIER_T15, _TIER_T2, _TIER_T3):
            for idx in self.drops[tier]:
                tier_of[idx] = tier
        return tier_of

    def _removal_frame(self, sku_to_rep: pd.DataFrame) -> pd.DataFrame:
        """One review row for every raw SKU a tier deliberately collapsed.

        Captured per-drop, before transitive representative resolution
        obscures where the removal happened.
        """
        tier_of = self.tier_of()
        removals = sku_to_rep.loc[list(tier_of)].copy()
        removals["tier"] = [tier_of[idx] for idx in removals.index]
        return removals[["sku_id", "rep_id", "tier"]].reset_index(drop=True)

    def row_accounting(
        self,
        n0: int,
        deduped: pd.DataFrame,
        ambiguous_out: pd.DataFrame,
        conflicts: pd.DataFrame,
        *,
        n_t1_skipped: int,
        n_deferred: int,
    ) -> dict:
        """The stage's closure ledger (SILENT_DROPS task 4; capture-only).

        The tier counters partition the frame at each step, so the closure
        input_rows == output_rows + sum(dropped) holds by construction
        (finish_manifest asserts it before publishing):
        71,623 == 61,529 + (1,943 + 2,245 + 5,906).

        The two "deferred" populations are NOT drops and deliberately
        excluded from `dropped`:
          skipped_checksum_invalid (T1) — 3,715 checksum-fail gtin rows are
            concatenated BACK into the work frame; T1.5 collapses the
            same-product ones (counted under dropped.t1_5_*), the rest fall
            through to the title tiers.
          deferred_to_t3 (T2) — 465 gtin-conflicting rows likewise re-enter
            the frame and are settled by T3's counter.
        Recording them under their own keys (outside `dropped`) keeps the
        audit trail complete without breaking the closure invariant.
        """
        return {
            "input_rows": n0,
            "output_rows": len(deduped),
            "dropped": {
                "t1_retailer_gtin": len(self.drops[_TIER_T1]),
                "t1_5_retailer_malformed_gtin_same_product":
                    len(self.drops[_TIER_T15]),
                "t2_retailer_title_gtin": len(self.drops[_TIER_T2]),
                "t3_retailer_title_identity_partition": len(self.drops[_TIER_T3]),
            },
            "skipped_checksum_invalid": n_t1_skipped,
            "deferred_to_t3": n_deferred,
            "ambiguous_offer_groups": len(ambiguous_out),
            "unresolved_identity_review_rows": len(conflicts),
        }

    def _most_informative_representative(self, sub: pd.DataFrame):
        """Rank a candidate group and return (rep_index, rep_row_frame).

        Most complete descriptors, then a trusted gtin, then the most
        informative title. NOT price — the cheapest listing is not the most
        truthful one.
        """
        order = sub.sort_values(
            ["_complete", "_ident", "sku_name_eng"],
            ascending=[False, False, True],
            na_position="last", kind="stable",
        )
        return order.index[0], order.iloc[[0]]


# ── hard invariants ─────────────────────────────────────────────────────────
class SanityGate:
    """Every sanity check the stage must pass before publishing."""

    @staticmethod
    def no_partition_duplicates(work: pd.DataFrame) -> None:
        """No (retailer,title,identity-partition) duplicates may remain.

        The partition is part of the key on PURPOSE: two genuinely
        different products may share a title string at one retailer, but no
        two rows sharing a trusted gtin can survive as duplicates. Checked
        on `work` because `deduped` drops the helper columns.
        """
        dups = int(
            work.duplicated(subset=["retailer", "sku_name_eng", "_ident"]).sum()
        )
        if dups:
            raise AssertionError(
                f"sanity FAILED: {dups} retailer+title+identity dupes remain"
            )

    @staticmethod
    def rep_mapping_complete(sku_to_rep: pd.DataFrame, deduped: pd.DataFrame) -> None:
        """Every raw SKU resolves to a representative, covering 0..n-1 exactly."""
        if sku_to_rep["rep_id"].isna().any():
            raise AssertionError(
                "sanity FAILED: some SKUs map to no representative"
            )
        if set(sku_to_rep["rep_id"].unique()) != set(range(len(deduped))):
            raise AssertionError(
                "sanity FAILED: rep_id coverage is not 0..n-1"
            )

    @staticmethod
    def identity_invariant(
        before_valid_gtins: set[str], deduped: pd.DataFrame
    ) -> None:
        """IDENTITY INVARIANT (2026-09-30): no product may lose its last row.

        Every trusted gtin present in the input must still be present in
        the output, or a product has been erased from the matching input.
        Measured 264 products failing this before the T3 identity partition
        landed. This is a hard gate, not a report: the failure mode it
        catches is silent and unrecoverable downstream, and no future tier
        may be allowed to reintroduce it.
        """
        after = set(deduped["gtin"].fillna("").astype(str).str.strip())
        lost = before_valid_gtins - after
        if lost:
            raise AssertionError(
                f"sanity FAILED: {len(lost):,} trusted gtins lost their last "
                f"row (e.g. {sorted(lost)[:3]}) — a product was deleted from "
                f"the matching input, not de-duplicated"
            )

    @staticmethod
    def removal_accounting(removals: pd.DataFrame, expected: int) -> None:
        """The removal ledger must close: one row per deliberately dropped SKU."""
        if len(removals) != expected:
            raise AssertionError(
                f"sanity FAILED: removals has {len(removals):,} rows, expected "
                f"{expected:,}"
            )


# ── output publication ──────────────────────────────────────────────────────
class StageWriter:
    """Every CSV the dedupe stage publishes (all writes atomic)."""

    @staticmethod
    def deduped(deduped: pd.DataFrame) -> None:
        """The deduped dataset (pipeline input)."""
        atomic_write_csv(deduped, DEDUPED_PATH, index=False)
        _LOG.info(f"wrote {DEDUPED_PATH} ({len(deduped):,} rows)")

    @staticmethod
    def sku_to_rep(sku_to_rep: pd.DataFrame) -> None:
        """The raw SKU -> rep_id mapping the notebook consumes."""
        atomic_write_csv(sku_to_rep, SKU_TO_REP_PATH, index=False)
        _LOG.info(f"wrote {SKU_TO_REP_PATH} ({len(sku_to_rep):,} rows)")

    @staticmethod
    def removals(removals: pd.DataFrame) -> None:
        """One review row per removed raw SKU."""
        atomic_write_csv(removals, CSV_REMOVALS, index=False)
        _LOG.info(f"wrote {CSV_REMOVALS} ({len(removals):,} review rows)")

    @staticmethod
    def summary(summary: list[dict]) -> None:
        """The per-tier counts display table."""
        atomic_write_csv(pd.DataFrame(summary), CSV_SUMMARY, index=False)
        _LOG.info(f"wrote {CSV_SUMMARY} (display table, {len(summary)} rows)")

    @classmethod
    def ambiguous(cls, ambiguous: pd.DataFrame) -> pd.DataFrame:
        """retailer+title groups flagged as ambiguous offers (pv)."""
        flattened = ambiguous.rename(
            columns={"count": "rows", "nunique": "distinct_prices"}
        ).reset_index()
        return cls.publish_offers(flattened)

    @staticmethod
    def publish_offers(flattened: pd.DataFrame) -> pd.DataFrame:
        """The atomic publication of one flattened-offers table."""
        atomic_write_csv(flattened, CSV_OFFERS, index=False)
        _LOG.info(
            f"wrote {CSV_OFFERS} (display table, {len(flattened):,} rows) "
            f"— {len(flattened):,} ambiguous-offer groups flagged"
        )
        return flattened

    @staticmethod
    def conflicts(conflicts: pd.DataFrame) -> None:
        """Descriptor conflicts left unresolved (the escalation queue)."""
        atomic_write_csv(conflicts, CSV_CONFLICTS, index=False)
        n_unresolved = (
            int((conflicts["label"] == "unresolved").sum()) if len(conflicts) else 0
        )
        _LOG.info(
            f"wrote {CSV_CONFLICTS} ({len(conflicts):,} rows) — "
            f"{n_unresolved:,} identity questions the descriptor bundle could "
            f"not settle, escalated instead of guessed"
        )


@timed
def main() -> None:
    for _out in (CSV_SUMMARY, CSV_OFFERS, CSV_REMOVALS, CSV_CONFLICTS,
                 DEDUPED_PATH, SKU_TO_REP_PATH):
        ensure_parent(_out)
    # Stage manifest (SILENT_DROPS task 4) — begin BEFORE the work: the
    # raw export is hashed now (53MB, chunked) so the record pins exactly
    # what this stage read. Seed = the SSOT seed (lib.common.SEED); the
    # tiered collapse below is deterministic, no RNG is consumed.
    manifest = begin_manifest("dedupe", inputs=[DATA_PATH], seed=SEED)
    df, work = FrameWorkbook.load()
    n0 = len(df)
    _LOG.info(f"[dedupe] {n0:,} raw rows under identity links")
    ambiguous = FrameWorkbook.ambiguous_offer_audit(work)

    # parent[i] = original row index of the surviving representative for
    # row i, owned by the tier runner (see TieredCollapse).
    tiers = TieredCollapse(work.index)
    summary: list[dict] = []

    work, before_valid_gtins = FrameWorkbook.stage_trusted_gtins(work)
    with _LOG.section("dedupe.tiers"):
        with_bc, malformed, no_bc = FrameWorkbook.partition_by_gtin_trust(work)
        t1, dropped1 = tiers._tier_t1(with_bc)
        n_t1_skipped = len(malformed)
        work = pd.concat([t1, malformed, no_bc])
        tiers.record(_TIER_T1, dropped1)
        summary.append({"tier": _TIER_T1,
                        "dropped_rows": len(dropped1),
                        "skipped_checksum_invalid": n_t1_skipped})

        t15_kept, t15_dropped, t15_groups, t15_unresolved, conflict_rows = (
            tiers._tier_t15(malformed)
        )
        work = pd.concat([t1, t15_kept, no_bc])
        tiers.record(_TIER_T15, t15_dropped)
        summary.append({"tier": _TIER_T15,
                        "dropped_rows": len(t15_dropped),
                        "collapsed_groups": t15_groups,
                        "unresolved_groups": t15_unresolved})

        work, dropped2, t2_deferred = tiers._tier_t2(work)
        tiers.record(_TIER_T2, dropped2)
        summary.append({"tier": _TIER_T2,
                        "dropped_rows": len(dropped2),
                        "deferred_to_t3": len(t2_deferred)})

        work, dropped3 = tiers._tier_t3(work)
        tiers.record(_TIER_T3, dropped3)
        summary.append({"tier": _TIER_T3,
                        "dropped_rows": len(dropped3)})

    # ---- outputs --------------------------------------------------------------
    with _LOG.section("dedupe.outputs"):
        tiers._resolve_representatives(df.index)
        deduped = work.drop(columns=HELPERS, errors="ignore").reset_index(drop=True)
        sku_to_rep = tiers._sku_to_rep_frame(df, work)

        with _LOG.section("dedupe.sanity"):
            SanityGate.no_partition_duplicates(work)
            SanityGate.rep_mapping_complete(sku_to_rep, deduped)
            _LOG.info("  [PASS] no retailer+title+identity duplicates remain; "
                      "SKU->rep mapping complete")
            SanityGate.identity_invariant(before_valid_gtins, deduped)
            _LOG.info(f"  [PASS] identity invariant: all "
                      f"{len(before_valid_gtins):,} trusted gtins still have "
                      f"a representative")

        removals = tiers._removal_frame(sku_to_rep)
        SanityGate.removal_accounting(removals, n0 - len(deduped))

        StageWriter.deduped(deduped)
        StageWriter.sku_to_rep(sku_to_rep)
        StageWriter.removals(removals)
        summary.append({"tier": "TOTAL dropped", "dropped_rows": n0 - len(deduped)})
        summary.append({"tier": "TOTAL remaining", "dropped_rows": len(deduped)})
        StageWriter.summary(summary)
        ambiguous_out = StageWriter.ambiguous(ambiguous)
        conflicts = pd.DataFrame(
            conflict_rows,
            columns=["tier", "retailer", "gtin", "label", "reasons",
                     "title_a", "title_b", "brand_a", "brand_b"],
        )
        StageWriter.conflicts(conflicts)

    row_accounting = tiers.row_accounting(
        n0, deduped, ambiguous_out, conflicts,
        n_t1_skipped=n_t1_skipped, n_deferred=len(t2_deferred),
    )
    manifest_path = finish_manifest(
        manifest,
        outputs=[DEDUPED_PATH, SKU_TO_REP_PATH, CSV_SUMMARY, CSV_OFFERS,
                 CSV_REMOVALS, CSV_CONFLICTS],
        row_accounting=row_accounting,
        expected_outputs=[
            F["dataset_deduped"].name, F["sku_to_rep"].name,
            F["dedupe_summary"].name, F["ambiguous_offer_groups"].name,
            F["removals"].name, F["dedupe_conflicts"].name,
        ],
    )
    _LOG.info(f"wrote {manifest_path} — stage manifest (closure "
              f"{row_accounting['input_rows']:,} == {row_accounting['output_rows']:,} "
              f"+ {sum(row_accounting['dropped'].values()):,} dropped)")

    _LOG.info(f"\n{len(df):,} rows -> {len(deduped):,} after tiered dedupe "
              f"(dropped {n0 - len(deduped):,}); "
              f"{len(ambiguous_out):,} retailer+title groups were "
              "price-varying offers (flagged, not silently merged)")


if __name__ == "__main__":
    main()
