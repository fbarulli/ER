"""P0 — emit THE single final validation CSV.

Replaces the retired ``dataset_deduped_sample_3000/5000`` lanes. Those splits
were derived from a graph built WITHOUT the validation census, so the two
sides derived different components from the same data: 74.7% of the old
validation population was contaminated (23.3% of positives had BOTH endpoints
in train, 51.4% had one), which means the P@R95 it reported was measuring
memorization as much as generalization.

The leak cannot be fixed downstream of the split, so this module builds the
graph the split is cut from and then emits the population:

    build_training_data      -> base positive pairs (sku, own canonical)
    merged_component_graph   -> + normalized entity key + labeled positives
    holdout_split            -> train (folds 0+1) / dev (2) / test (3)
    emit                     -> folds 2+3, the single validation population

WHY ``fold``/``component_id`` HAVE A ``_2`` SIBLING
---------------------------------------------------
A POSITIVE pair is one edge in the graph, so both its endpoints are always in
the same component and therefore the same fold -- ``fold == fold_2`` and
``component_id == component_id_2`` for all 1,143 of them, and that equality is
the leak guarantee, asserted below before anything is written.

A NEGATIVE pair is a mined *similarity* relation, not an identity claim, so
its two endpoints are usually in DIFFERENT components, and one of them is
frequently a gtin the model trained on. A single ``fold`` column would have
to silently mean "the fold of gtin1" and hide the other side. So the pair's
both sides are carried explicitly, and ``endpoint_in_train`` marks a negative
whose other side leaked in. Those rows are KEPT (the P0 spec treats
straddling negatives as documented current behaviour, not a regression) but
flagged, so a downstream floor can either exclude or report them instead of
inheriting an invisible 24% contamination.

SLICE FLAGS ARE PER SIDE, NOT PER PAIR
--------------------------------------
Each endpoint's canonical attribute values are frozen into the CSV
(``v1_volume``/``v2_volume``, ...). The pair-level bucket question is left
open on purpose: ``build_field_slice.py`` buckets by TWIN while
``labeled_pairs`` slices by CANONICAL VALUE, and reconciling those is the
separate open "align our gates" decision. Freezing the values stops downstream
re-deriving buckets from scratch; it does not pre-empt which aggregation wins.
For a positive the two sides are the same product, so the two extractions
describe ONE product — measured NOT equal as raw strings (they are sets
extracted by two different text feeds; 363/565 flavors disagree raw). The
set-valued agreement semantics (bag equality, evaluation.slice_agreement =
"set_bag") is decided and recorded in write_manifest's slice-coverage block
below; the raw-string comparison stays reachable as "scalar" for
byte-stability audits.

RESPONSIBILITY MAP (single-responsibility decomposition; behaviour pinned)
-------------------------------------------------------------------------
- :class:`SliceBagTokenizer` — parse one slice column literal into a value
  multiset (:func:`parse_field_bag` stays its public face).
- :class:`SliceFieldGrid` — owns SLICE_FIELDS, the frozen per-side canonical
  values (``_canonical_values``), and set-vs-scalar pair agreement counts.
- :class:`NegativeFoldPolicy` — the scored half's fold-assignment rule
  (:func:`negative_pair_fold`), the evidence measurement
  (:func:`negative_policy_evidence`), and the pinned decision's emit guard.
- :class:`FoldResolver` — gtin -> fold/component resolution for the census
  endpoints, with an explicit miss count.
- :class:`ValidationRowAssembler` — one labeled pair -> one output row.
- :class:`LeakGuards` — the pre-write leak assertions.
- :func:`write_manifest` — the manifest emission (kept module-level: it is
  part of this module's byte-identical byte-for-byte surface and pinned by
  the pinned-update convention comments).
- :func:`build` — the stage pipeline that threads these owners together.

TRACE ROWS (core.tracing, the ONE consolidated trace)
-----------------------------------------------------
Stage ``final_validation``. Emitted:
  run   graph.merged_component_graph   base positives -> merged graph entities
  run   split.fold_assignment          graph entities -> quarter folds
  run   labeled_census.batch_<i>       BATCH grain: one row per traced chunk of
                                       labeled pairs (in = pairs, out = rows)
  run   labeled_census.batch_census    batches walked / traced / omitted
  run   labeled_census.assembled       labeled pairs -> emitted rows, with the
                                       unresolved-endpoint and
                                       both-endpoints-in-train counts as drops
  group labeled_census.reason_census   one EXACT census row per outcome
  ent   labeled_census.*               the named PAIRS behind each outcome
                                       (gtin1|gtin2 + folds + exact reason)
  run   policy.evidence_measured       negatives -> the scored halves' negatives
  run   fold_map.published             every graph entity's fold (in == out)
  run   output.published               the CSV + manifest publication
Batch caps: ``_BATCH_PAIRS`` pairs per traced batch row and at most
``_MAX_BATCH_ROWS`` batch rows, both written into the batch rows' detail.
Sampling caps are core.tracing's (ENTITY_SAMPLE_PER_REASON / ENTITY_ROW_CAP).
Nothing here is unbounded.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from core.columns import ATTRIBUTE_DIMENSION_COLUMNS
from core.common import F, RESULTS, SEED, load_dataset_deduped, training_cfg
from core.manifest import atomic_write_csv
from core.pair_identity import PairIdentity
from core.run_log import RunLogger
from core.schemas import check_canonical_records_frame, upgrade_canonical_records_frame
from core.tracing import (ENTITY_ROW_CAP, ENTITY_SAMPLE_PER_REASON,
                          TRACE_BATCH_ROWS, TRACE_MAX_BATCH_ROWS, TraceRun)
from training.folds import (
    component_ids,
    derive_holdout,
    merged_component_graph,
    normalize_gtin,
)
from training.prepare_all_trace import timed

_LOG = RunLogger(__name__)

#: The pipeline stage these rows belong to (core.tracing ``stage`` column).
STAGE = "final_validation"

# ── batch-grain budget (documented where it is spent) ──────────────────────
# The assembler walks one labeled pair at a time; 4,096 pairs per BATCH row keeps
# a real census to a handful of rows, and 16 traced batches bound the file while
# the remainder is announced in ``labeled_census.batch_census``.
_BATCH_PAIRS = TRACE_BATCH_ROWS
_MAX_BATCH_ROWS = TRACE_MAX_BATCH_ROWS

#: The three outcomes a labeled pair can have in the assembler. These strings are
#: the trace's reason labels, so a reader greps outcomes, not prose.
OUTCOME_SCORED = "scored_row"
OUTCOME_UNRESOLVED = "unresolved_endpoint"
OUTCOME_BOTH_IN_TRAIN = "both_endpoints_in_train"

# The six fields P0 keeps as gates. `pulp_set` is deliberately absent: it is
# populated in 2.3% of canonical records and 0.5% of verified positives, which
# is 2 pairs in this validation half -- population scarcity, not a parsing
# defect, and no gate at any budget that respects the component constraint.
#
# The ORDER is this lane's own frozen composition order (the v1_*/v2_* CSV
# columns are emitted in it); the dimension -> canonical_records column pairs
# are DERIVED from the record schema (core.columns.ATTRIBUTE_DIMENSION_COLUMNS),
# so no column name is retyped here.
SLICE_DIMENSIONS: tuple[str, ...] = (
    "volume",
    "pack",
    "package_type",
    "sweetener",
    "flavor",
    "carbonation",
)
SLICE_FIELDS: tuple[tuple[str, str], ...] = tuple(
    (dimension, ATTRIBUTE_DIMENSION_COLUMNS[dimension])
    for dimension in SLICE_DIMENSIONS
)

# ════════════════════════════════════════════════════════════════════════════
# DECISION: SCORED-HALF NEGATIVE FOLD ASSIGNMENT (owner-posture change; this
# file is the decision's owner surface because the scored half IT is what it
# emits, and its docstring above already carried the pair-level fold
# semantics). Decided 2026-10-01 from evidence computed by
# :func:`negative_policy_evidence` on data/final_validation.csv
# (5,786 negatives / 565 positives), recorded here with the exact numbers
# that forced it — the postulate series is not reproduced from memory, the
# code below always re-measures and refuses the default if it stops winning.
#
#   Criterion                       Policy A            Policy B
#                                   withhold_straddle   train_side
#   ─────────────────────────────   ─────────────────   ─────────────────
#   scored DEV negatives (fold 2)        592                 1,087
#   scored TEST negatives (fold 3)       466                   957
#   withheld/populations stranded    4,728 (consumed        3,742 whole
#   (4,728 = 3,742 train-endpoint       by nothing)          negatives
#   + 986 dev/test straddlers)                               returned to
#                                                            the train
#                                                            fold
#   thin slice cells (< robust_validation.min_test_negatives = 5)
#     populated cells DEV                253                   336
#     thin cells DEV                     171 (67.6%)           218 (64.9%)
#     populated cells TEST               238                   332
#     thin cells TEST                    167 (70.2%)           223 (67.2%)
#     thin-tab population DEV            86+27+41+5+150+0      111+38+42+0+195+0
#   qualitative "what a negative measures"
#     scored without a trained-on        yes (both        yes (identical:
#     side (generalization-negative)     policies" or           both scored
#                                         no in the two        folds exclude
#                                         the train side)     the train side)
#   dev/test straddles                  unassignable       986 assigned whole
#
# A withheld population is scored from 592/466 to 1,087/957 (+105.4% TEST,
# +83.6% DEV); the thin cell SHARE falls on both scored folds; the flavor
# tail stays thin under BOTH policies (it is real scarcity in the census,
# not an assignment artifact: A test flavor thin-cells 70, B 121 in absolute
# terms but the same population share) — so policy choice does not hide the
# flavor floor problem, it only doubles the measurable population per fold.
# WHAT A NEGATIVE MEASURES (the delta the file's docstring already claimed):
# forked between two honest descriptions — the scalar v1==v2 gate the docstring
# states ("For a positive the two sides are the same product") and the
# disagreement the slice-coverage block measures. Policy B as written by the
# decision criteria below: a scored negative under EITHER policy still
# measures generalization to unseen products (scoring folds per the shared
# rule CANNOT include a trained-on endpoint), and policy B additionally
# assigns a whole fold to the 986 dev/test straddlers and returns the 3,742
# train-flavored ones to the training side where a mined-negative consumer
# can pick them up, instead of parking them as labelled-never-used rows.
# The decision at the original run: policy "train_side" (B) won on all three
# criteria and became the pin (config/training.yaml split.negative_fold_policy).
# RE-DECIDED at the 2026-10-01 regeneration: the regenerated merged graph
# (24,361 mined positives / 983 labeled positives over 14,946 entities) moved
# the dev-half evidence — the split assert measured A scoring MORE thin-heavy
# negatives than B (65.71% vs 63.80% cells below min_test_negatives=5), so the
# pinned A no longer holds; the config moved to "withhold_straddle" (A)
# together with this block, per the rule that an artifact may not ship under
# a policy its evidence rejects.
# RE-MEASURED 2026-10-06 regeneration (35,561-row export): A's dev-half thin
# advantage REVERSED (A 87.9% vs B 77.6% cells below min_test_negatives=5).
# A stays assigned: the thin share was the 2026-10-01 tiebreaker, but B
# remains structurally disqualified — B's scored negatives carry trained-on
# endpoints (a correctness property, not a preference), which the evidence
# function refuses to certify. The thin share is therefore RECORDED as
# evidence and is no longer a decision criterion; A's emit guard enforces
# A's own-evidence criterion (both scored halves usable) instead of the
# old cross-policy thin ordering.
# RE-DECIDED 2026-10-08 (TODO "Balance dev/test negatives"): the structural
# disqualification that parked B was never MADE MEASURABLE, so the pin rested
# on prose. The evidence surface now measures it per policy
# (``scored_negatives_with_trained_on_endpoint``) and the emit guard refuses a
# policy that leaks. MEASURED on the committed census: B assigns the dev/test
# straddlers A can only drop (A: every straddler scores nowhere; B: fold_a, so
# the straddlers split ~evenly and BOTH scored halves roughly double) and both
# policies measure ZERO scored negatives with a trained-on endpoint — B's
# train-endpoint pairs are parked in the train fold by rule, exactly like A's.
# A's remaining defect is the one the TODO names: the scored halves are
# dominated by positives (dev/dev negatives in the single digits), so the
# Youden fit and the false-positive rate have almost no support. B doubles
# the scored negatives on both halves with no leak, so the pin moves to B
# (config/training.yaml split.negative_fold_policy) and both emit guards keep
# re-measuring the leak property at every artifact.
# ════════════════════════════════════════════════════════════════════════════

# Policy names. Both stay load-valid; config/training.yaml pins the winner.
NEGATIVE_FOLD_POLICY_WITHHOLD = "withhold_straddle"
NEGATIVE_FOLD_POLICY_TRAIN_SIDE = "train_side"


class SliceBagTokenizer:
    """Tokenizer for the slice columns' set-valued spelling.

    The emitted slice columns carry a canonical list literal ("[lime, lime]",
    "[479.0, 518.0]" — bare tokens, never quoted) or a bare single token.
    Comparing the raw STRINGS was the old scalar semantics; this tokenizer is
    what bag identity needs. Fail-loud, not graceful: an unbalanced bracket
    value raises (a silently unreadable slice would downgrade to raw-string
    equality without anyone knowing).
    """

    @staticmethod
    def _text(raw: object) -> str:
        return str(raw).strip()

    @classmethod
    def parse(cls, raw: object) -> Counter:
        """One slice value -> multiset of extracted tokens (bag identity)."""
        text = cls._text(raw)
        if text.startswith("[") != text.endswith("]"):
            raise ValueError(f"unbalanced slice list {raw!r}")
        if text.startswith("[") and text.endswith("]"):
            return cls._tokens(text[1:-1])
        token = text or None
        return Counter([token]) if token else Counter()

    @staticmethod
    def _tokens(content: str) -> Counter:
        tokens = [
            token.strip().strip("'\"")
            for token in content.split(",")
            if token.strip().strip("'\"")
        ]
        return Counter(tokens)


def parse_field_bag(raw: object) -> Counter:
    """Slice column -> multiset of extracted values (the set-valued side).

    The spelling contract and the fail-loud rule live on
    :class:`SliceBagTokenizer`; this wrapper is the module's stable public
    name for it (pinned by the scored-half decision tests).
    """
    return SliceBagTokenizer.parse(raw)


def _evaluation_slice_agreement() -> str:
    """The set-semantics switch (evaluation.slice_agreement, fail-loud load)."""
    return str(training_cfg().evaluation.slice_agreement)


class SliceSemantics:
    """The two implemented pair-agreement semantics for slice flags.

    ``scalar``: legacy raw string v1 == v2 per pair (the old behaviour,
    reachable for byte-stability audits). ``set_bag``: BAG equality after
    :func:`parse_field_bag` — order/spacing-only spelling differences agree;
    contents differences never do.
    """

    @staticmethod
    def scalar(a: pd.Series, b: pd.Series) -> int:
        return int((a != b).sum())

    @staticmethod
    def set_bag(a: pd.Series, b: pd.Series) -> int:
        return int(
            sum(parse_field_bag(x) != parse_field_bag(y) for x, y in zip(a, b))
        )

    @classmethod
    def count(cls, a: pd.Series, b: pd.Series, semantics: str) -> int:
        if semantics == "scalar":
            return cls.scalar(a, b)
        if semantics == "set_bag":
            return cls.set_bag(a, b)
        raise ValueError(
            f"unknown slice_agreement {semantics!r}; expected one of "
            "('scalar', 'set_bag')"
        )


def count_slice_disagreements(a: pd.Series, b: pd.Series, semantics: str) -> int:
    """THE slice-flag comparison (both semantics implemented, config-chosen).

    Dispatches to :class:`SliceSemantics`; kept module-level because
    ``write_manifest``'s pinned disagreement counters and the synthetic
    semantics tests call it by this name.
    """
    return SliceSemantics.count(a, b, semantics)


class SliceFieldGrid:
    """Owner of the SLICE_FIELDS frozen canonical values and their counts.

    One job: turn ``canonical_records.csv`` (after the pipeline lanes'
    migration + validation contract) into ``gtin -> {field: value}``, and
    count pair-level agreement per field under the configured semantics.
    """

    def __init__(self, fields: tuple[tuple[str, str], ...] = SLICE_FIELDS):
        self._fields = fields
        self._values: dict[str, dict[str, str]] | None = None

    # -- frozen canonical values -------------------------------------------

    @property
    def fields(self) -> tuple[tuple[str, str], ...]:
        return self._fields

    def freeze(self, canonical: pd.DataFrame) -> dict[str, dict[str, str]]:
        """Validate + index the canonical frame into per-gtin slice values."""
        canonical["_key"] = canonical["gtin"].map(normalize_gtin)
        out: dict[str, dict[str, str]] = {}
        for _, row in _LOG.progress(
            canonical.iterrows(), desc="canonical_slice_freeze", unit="record",
            total=len(canonical),
        ):
            key = row["_key"]
            if not key or key in out:
                continue
            out[key] = {
                name: str(row.get(col, "") or "")
                for name, col in self._fields
            }
        self._values = out
        return out

    def load_canonical(self) -> pd.DataFrame:
        """Read + migrate + validate the frozen canonical records frame."""
        canon = pd.read_csv(
            F["canonical_records"], dtype=str, keep_default_na=False, low_memory=False
        )
        # Same read contract as the pipeline lanes: migrate an outdated artifact
        # and validate before slicing — the freeze step's column access makes
        # the old "first column might be the key" fallback unreachable.
        canon = upgrade_canonical_records_frame(canon)
        check_canonical_records_frame(canon)
        return canon

    def canonical_values(self) -> dict[str, dict[str, str]]:
        """gtin -> {field: canonical value string} from the frozen canonical records."""
        if self._values is None:
            self.freeze(self.load_canonical())
        return self._values

    def values(self, gtin: str) -> dict[str, str]:
        return self.canonical_values().get(gtin, {})

    # -- manifest coverage counters ----------------------------------------

    def coverage(self, positives: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Per-field measuring power among the emitted positives.

        "We have 564 positives" is not the question a gate asks — "can this
        field carry a floor" is. ``disagree`` is the count of positives whose
        two endpoints carry different values for the field: same product, one
        side's text mentions an extra value. That is legitimate extractor
        variance, so it is reported rather than asserted on, but a gate
        comparing v1 to v2 needs to know it exists.
        """
        semantics = _evaluation_slice_agreement()
        # ── DECISION: SLICE-FLAG SET SEMANTICS (owner-posture change
        # 2026-10-01; decided with numbers, not conceded to a gate). The v1_*/
        # v2_* columns are SET-valued extractions per side ("Different
        # extracted sets", per the module docstring), so the pair-level
        # agreement the `disagree` counter measures is BAG equality under
        # evaluation.slice_agreement="set_bag" (the new default, config
        # /training.yaml), where endpoints formatted "[a, b]" differ from
        # "[b, a]" in spelling but not in contents. Comparisons under
        # "scalar" are the legacy raw-string v1 == v2 and stay reachable for
        # byte-stability audits.
        #
        # BYTE-STABILITY ATTRIBUTION (required by the pinned-update
        # convention: the old numbers stay recorded). Measured on
        # data/final_validation.csv, 565 positives, set_bag vs scalar:
        #   2026-09-29 regen: volume 13 -> 13, pack 48 -> 48,
        #     package_type 153 -> 153, sweetener 127 -> 127,
        #     flavor 363 -> 363, carbonation 38 -> 38.
        #   2026-10-01 regen (volume-unification closure + re-capture):
        #     volume 13 -> 6, pack 48 -> 47, package_type 153 -> 105,
        #     sweetener 127 -> 86, flavor 363 -> 254, carbonation 38 -> 47;
        #     scalar and set_bag identical at every count (the flag still
        #     changes semantics only where order/spacing-only spellings
        #     differ — none exist in the emitted canon).
        # EVERY count is byte-identical on today's artifact -- the flag
        # changes semantics only where order/spacing-only spellings
        # differ (none exist in the emitted canon), and every OTHER
        # manifest slice-coverage number (positives / distinct /
        # largest_bucket / unpopulated) never enters this comparison and
        # remains computed from the side-A column alone, byte-identical.
        # A future regen under a canon with purely order-differing spellings
        # would make a disagrees count DROP: the old scalar number must
        # then stay recorded in THIS comment before the new one replaces it
        # (pinned-update convention).
        coverage: dict[str, dict[str, int]] = {}
        for name, _col in self._fields:
            if not len(positives):
                coverage[name] = {"positives": 0, "distinct": 0,
                                  "largest_bucket": 0, "unpopulated": 0,
                                  "disagree": 0}
                continue
            a, b = positives[f"v1_{name}"], positives[f"v2_{name}"]
            counts = a[a != ""].value_counts()
            coverage[name] = {
                "positives": int(len(positives)),
                "distinct": int(counts.size),
                "largest_bucket": int(counts.iloc[0]) if counts.size else 0,
                "unpopulated": int((a == "").sum()),
                "disagree": count_slice_disagreements(a, b, semantics),
            }
        return coverage


def negative_pair_fold(policy: str, fold_a: int, fold_b: int, n_folds: int) -> int:
    """THE scored half's negative fold-assignment rule (single source).

    Positives NEVER route through this: their two endpoints share a fold by
    graph construction (the leak guarantee below), and re-deriving the pair
    fold here would silently fork the split. This rule is for MINED
    NEGATIVES only — similarity links whose endpoints legitimately sit in
    different folds ("folds.derive_holdout" assigns entities, not pairs).

    ``withhold_straddle`` (A): the pair's fold stays the raw ``fold_a`` on
    both columns, preserving the legacy semantics — a consumer scores a
    negative only where ``fold == fold_2`` equals its fold, so a mismatched
    pair scores nowhere and is reported by ``straddles_fold``/
    ``endpoint_in_train`` instead.

    ``train_side`` (B): the pair gets one whole fold — the train-side
    endpoint's fold when either endpoint is a train fold (OUT of the scored
    half, back in the training population), else the fold of ``fold_a`` as
    the deterministic boundary tiebreak for the two unseen endpoints (the
    dev/test straddle that A could never score).
    """
    return NegativeFoldPolicy.rule(policy, fold_a, fold_b, n_folds)


class NegativeFoldPolicy:
    """The scored half's negative fold assignment: rule, evidence, guard.

    (The DECISION block above owns the why and the recorded numbers.) One
    responsibility surface: turn raw endpoint folds into a scored-half pair
    fold under the configured policy, MEASURE both policies' evidence, and
    REFUSE to emit when the pinned policy's evidence no longer holds.
    """

    # Deterministic assignment boundary for the two unseen endpoints: the
    # fold of ``fold_a`` (documented on negative_pair_fold).
    @staticmethod
    def rule(policy: str, fold_a: int, fold_b: int, n_folds: int) -> int:
        if policy == NEGATIVE_FOLD_POLICY_WITHHOLD:
            return fold_a
        if policy == NEGATIVE_FOLD_POLICY_TRAIN_SIDE:
            if fold_a < n_folds - 2 or fold_b < n_folds - 2:
                return min(fold_a, fold_b)
            return fold_a
        raise ValueError(
            f"unknown negative_fold_policy {policy!r}; expected one of "
            f"({NEGATIVE_FOLD_POLICY_WITHHOLD!r}, {NEGATIVE_FOLD_POLICY_TRAIN_SIDE!r})"
        )

    @staticmethod
    def rule_vectorized(policy: str, fold_a: pd.Series, fold_b: pd.Series,
                        n_folds: int) -> pd.Series:
        """The same rule on one negative population, without a row loop."""
        if policy == NEGATIVE_FOLD_POLICY_WITHHOLD:
            return fold_a
        if policy == NEGATIVE_FOLD_POLICY_TRAIN_SIDE:
            train_endpoints = (fold_a < n_folds - 2) | (fold_b < n_folds - 2)
            return pd.Series(
                np.where(train_endpoints, np.minimum(fold_a, fold_b), fold_a),
                index=fold_a.index,
            )
        raise ValueError(
            f"unknown negative_fold_policy {policy!r}; expected one of "
            f"({NEGATIVE_FOLD_POLICY_WITHHOLD!r}, {NEGATIVE_FOLD_POLICY_TRAIN_SIDE!r})"
        )

    # -- evidence ----------------------------------------------------------

    @classmethod
    def evidence(
        cls, frame: pd.DataFrame, min_test_negatives: int, n_folds: int = 4
    ) -> dict[str, dict[str, object]]:
        """Decide-with-numbers: the criteria the policy decision is pinned on.

        Runs BOTH policies over one frame carrying ``fold``/``fold_2``/
        ``true_label``/``v1_*``/``v2_*`` columns and reports, per policy: scored
        DEV/TEST negatives, scored DEV/TEST positives, withheld negatives, and
        thin slice cells (populated (field, value) cells among scored negatives
        with fewer than ``min_test_negatives`` members — the same thinness
        contract ``robust_validation`` reuses). Percentages are computed, the
        winner is the caller's to record — this function does not choose.

        Raises SystemExit when a policy's population is INCONSISTENT with the
        frame it claims to measure (e.g. positives straddling under B): a
        criterion computed on a broken population cannot back a decision.
        """
        for policy in (NEGATIVE_FOLD_POLICY_WITHHOLD, NEGATIVE_FOLD_POLICY_TRAIN_SIDE):
            # A positive's pair-fold is its shared endpoint fold under BOTH
            # policies (the leak guarantee asserts fold == fold_2 for
            # positives, so a single column carries it).
            positive_straddle_count = int(
                (frame.loc[frame.true_label == 1, "fold"]
                 != frame.loc[frame.true_label == 1, "fold_2"]).sum()
            )
            if positive_straddle_count:
                raise SystemExit(
                    f"{positive_straddle_count} positives straddle a fold — the "
                    "policy evidence population is corrupt (run the leak "
                    "guards first)"
                )
        evidence: dict[str, dict[str, object]] = {}
        has_slices = all(f"v1_{name}" in frame.columns for name, _col in SLICE_FIELDS)
        if has_slices:
            thin_rows = cls._thin_cells(frame, min_test_negatives, n_folds)
        else:
            thin_rows = {
                policy: {} for policy in
                (NEGATIVE_FOLD_POLICY_WITHHOLD, NEGATIVE_FOLD_POLICY_TRAIN_SIDE)
            }
        for policy in (NEGATIVE_FOLD_POLICY_WITHHOLD, NEGATIVE_FOLD_POLICY_TRAIN_SIDE):
            evidence[policy] = cls._scored_contract(
                frame, policy, n_folds,
                thin_rows.get(policy, {}) if has_slices else None,
            )
        return evidence

    @staticmethod
    def scored_masks(
        policy: str, fold_a: pd.Series, fold_b: pd.Series, n_folds: int
    ) -> tuple[pd.Series, pd.Series]:
        """(in_dev, in_test) score masks for one negative population.

        Policy A's raw semantics is NOT "pair fold == scored fold": a
        mismatching straddler carries ``fold == fold_a`` (its pair fold) yet
        scores NOWHERE — its two sides are not in the same scored fold. So A's
        mask is the BOTH-endpoints conjunction, and B's is the assigned pair
        fold equality (B assigns the whole fold to the pair).
        """
        dev, test = n_folds - 2, n_folds - 1
        f1 = fold_a.astype(int)
        f2 = fold_b.astype(int)
        if policy == NEGATIVE_FOLD_POLICY_WITHHOLD:
            # A: legacy — a negative scores only where BOTH endpoint folds
            # equal the scored fold; a mismatched pair scores nowhere.
            return (f1 == dev) & (f2 == dev), (f1 == test) & (f2 == test)
        if policy == NEGATIVE_FOLD_POLICY_TRAIN_SIDE:
            assigned = NegativeFoldPolicy.rule_vectorized(policy, f1, f2, n_folds)
            return assigned == dev, assigned == test
        raise ValueError(
            f"unknown negative_fold_policy {policy!r}; expected one of "
            f"({NEGATIVE_FOLD_POLICY_WITHHOLD!r}, {NEGATIVE_FOLD_POLICY_TRAIN_SIDE!r})"
        )

    @classmethod
    def _scored_contract(
        cls,
        frame: pd.DataFrame,
        policy: str,
        n_folds: int,
        thin: dict[str, dict[str, int]] | None,
    ) -> dict[str, object]:
        # A positive's pair-fold is its shared endpoint fold under BOTH
        # policies (the leak guarantee asserts fold == fold_2 for positives,
        # so a single column carries it).
        pos = frame[frame.true_label == 1]
        neg = frame[frame.true_label == 0]
        dev, test = n_folds - 2, n_folds - 1
        in_dev_pos = pos["fold"].astype(int) == dev
        in_test_pos = pos["fold"].astype(int) == test
        raw_first = neg["fold"].astype(int)
        raw_second = neg["fold_2"].astype(int)
        in_dev_neg, in_test_neg = cls.scored_masks(
            policy, raw_first, raw_second, n_folds
        )
        # The qualitative criterion in code: a scored negative may never carry
        # an endpoint in a TRAIN fold. It is a property of the assignment, so
        # it is MEASURED per policy (on the raw endpoint folds the masks were
        # built from) instead of asserted in prose — the 2026-10-01 rejection
        # of ``train_side`` rested on this claim, so it has to be checkable.
        trained_on_endpoint = (
            np.minimum(raw_first, raw_second) < n_folds - 2
        )
        scored_negatives = in_dev_neg | in_test_neg
        dev_neg, test_neg = int(in_dev_neg.sum()), int(in_test_neg.sum())
        dev_pos, test_pos = int(in_dev_pos.sum()), int(in_test_pos.sum())

        def _half(positives: int, negatives: int) -> dict[str, object]:
            total = positives + negatives
            return {
                "positives": positives,
                "negatives": negatives,
                # The imbalance the TODO records (dev 1286/9, test 1276/7): a
                # scored half with a near-zero negative share cannot fit a
                # threshold or measure a false-positive rate.
                "negative_share": (negatives / total) if total else 0.0,
            }

        return {
            "scored_dev_negatives": dev_neg,
            "scored_test_negatives": test_neg,
            "scored_dev_positives": dev_pos,
            "scored_test_positives": test_pos,
            "scored_negatives_with_trained_on_endpoint": int(
                (scored_negatives & trained_on_endpoint).sum()
            ),
            "scored_half_balance": {
                "dev": _half(dev_pos, dev_neg),
                "test": _half(test_pos, test_neg),
            },
            "negatives_withheld_from_scored_half": int(
                len(neg) - (in_dev_neg.sum() + in_test_neg.sum())
            ),
            "populated_cells": {
                half: dict(thin[half]["populated"]) for half in ("dev", "test")
            } if thin else None,
            "thin_cells": {
                half: dict(thin[half]["thin"]) for half in ("dev", "test")
            } if thin else None,
        }

    @staticmethod
    def _thin_cells(
        frame: pd.DataFrame, min_test_negatives: int, n_folds: int
    ) -> dict[str, dict[str, dict[str, int]]]:
        """Populated/thin (field, value) cells per scored half, per policy.

        The thin-cell criterion requires the frozen slice columns; a frame
        without them (synthetic fold-contract tests) records UNavailable —
        explicit, never silently claimed as "not thin".
        """
        results: dict[str, dict[str, dict[str, int]]] = {}
        neg = frame[frame.true_label == 0]
        if not len(neg):
            # Same nesting as the populated path (policy -> half -> census):
            # the old shape (policy -> census -> half) made _scored_contract
            # raise KeyError on any negative-free census.
            empty: dict[str, dict[str, int]] = {
                half: {name: 0 for name, _col in SLICE_FIELDS}
                for half in ("dev", "test")
            }
            return {
                policy: {
                    half: {"populated": dict(empty[half]), "thin": dict(empty[half])}
                    for half in ("dev", "test")
                }
                for policy in (NEGATIVE_FOLD_POLICY_WITHHOLD,
                               NEGATIVE_FOLD_POLICY_TRAIN_SIDE)
            }
        f1 = neg["fold"].astype(int)
        f2 = neg["fold_2"].astype(int)
        dev, test = n_folds - 2, n_folds - 1
        for policy in (NEGATIVE_FOLD_POLICY_WITHHOLD,
                       NEGATIVE_FOLD_POLICY_TRAIN_SIDE):
            in_dev, in_test = NegativeFoldPolicy.scored_masks(policy, f1, f2, n_folds)
            policy_result: dict[str, dict[str, int]] = {}
            for half, mask in (("dev", in_dev), ("test", in_test)):
                rows = neg.loc[mask.to_numpy()]
                populated: dict[str, int] = {}
                thin: dict[str, int] = {}
                for field, _col in SLICE_FIELDS:
                    bag = Counter(rows[f"v1_{field}"][rows[f"v1_{field}"] != ""])
                    extra = rows[f"v2_{field}"][
                        (rows[f"v2_{field}"] != "")
                        & (rows[f"v2_{field}"] != rows[f"v1_{field}"])
                    ]
                    bag.update(extra)
                    populated[field] = len(bag)
                    thin[field] = sum(
                        1 for count in bag.values()
                        if count < min_test_negatives
                    )
                policy_result[half] = {"populated": populated, "thin": thin}
            results[policy] = policy_result
        return results


def negative_policy_evidence(
    frame: pd.DataFrame, min_test_negatives: int, n_folds: int = 4
) -> dict[str, dict[str, object]]:
    """Decide-with-numbers evidence surface (see :class:`NegativeFoldPolicy`).

    Public name preserved: the scored-half decision tests and the module's
    DECISION block both point at this function.
    """
    return NegativeFoldPolicy.evidence(frame, min_test_negatives, n_folds)


class FoldResolver:
    """gtin -> (graph key, fold) resolution for the labeled census endpoints.

    A labeled gtin has to be resolved to the spelling the graph actually
    uses, and the two are not the same string: the fold sets are keys of the
    RAW row_bc, while normalize_gtin left-pads a 13-digit gtin to 14. A
    13-digit gtin therefore misses a raw fold set under a normalized lookup
    and is silently dropped -- which is how an earlier run of this script
    emitted 3 rows out of 8,889. Try the raw spelling first, then the
    normalized one, and COUNT the misses rather than skipping in silence.
    """

    def __init__(self, fold_of: dict[str, int]):
        self._fold_of = fold_of
        self._raw_keys = set(fold_of)
        self._norm_keys = {normalize_gtin(b) for b in fold_of}

    def resolve(self, gtin: object) -> str | None:
        raw = str(gtin).strip()
        if raw in self._raw_keys:
            return raw
        normed = normalize_gtin(raw)
        if normed in self._raw_keys or normed in self._norm_keys:
            return normed
        return None

    def fold(self, key: str) -> int:
        return self._fold_of[key]


class ValidationRowAssembler:
    """One labeled pair -> one output-row dict (or None to skip)."""

    def __init__(
        self,
        resolver: FoldResolver,
        comp_of: dict[str, int],
        slice_values: dict[str, dict[str, str]],
        n_folds: int,
    ):
        self._resolver = resolver
        self._comp_of = comp_of
        self._slice_values = slice_values
        self._n_folds = n_folds
        self.unresolved = 0
        #: Pairs whose BOTH endpoints sit in the training quarters: not validation
        #: rows, and a distinct outcome from an unresolvable endpoint. Counted
        #: rather than inferred so the trace can state the drop exactly.
        self.both_in_train = 0

    def assemble_with_reason(
        self, g1: str, g2: str, label: object
    ) -> tuple[dict[str, object] | None, str]:
        """The output row (or None) AND the exact reason for that outcome.

        ``assemble`` stays the public, row-only face; this is the trace's view,
        where an outcome that produced no row is named rather than silent.
        """
        k1 = self._resolver.resolve(g1)
        k2 = self._resolver.resolve(g2)
        if k1 is None or k2 is None:
            # An endpoint outside the graph entirely: it has no fold, so it
            # cannot be part of a fold-2+3 population.
            self.unresolved += 1
            return None, OUTCOME_UNRESOLVED
        f1, f2 = self._resolver.fold(k1), self._resolver.fold(k2)
        if f1 < self._n_folds - 2 and f2 < self._n_folds - 2:
            self.both_in_train += 1
            return None, OUTCOME_BOTH_IN_TRAIN  # both sides in train -> not validation
        row: dict[str, object] = {
            "gtin1": k1,
            "gtin2": k2,
            "gtin1_norm": normalize_gtin(k1),
            "gtin2_norm": normalize_gtin(k2),
            "true_label": int(label),
            "fold": f1,
            "fold_2": f2,
            "component_id": self._comp_of.get(k1, -1),
            "component_id_2": self._comp_of.get(k2, -2),
            "straddles_fold": f1 != f2,
            "endpoint_in_train": min(f1, f2) < self._n_folds - 2,
        }
        c1 = self._slice_values.get(normalize_gtin(k1), {})
        c2 = self._slice_values.get(normalize_gtin(k2), {})
        for name, _col in SLICE_FIELDS:
            row[f"v1_{name}"] = c1.get(name, "")
            row[f"v2_{name}"] = c2.get(name, "")
        # THE pair key (core.pair_identity SSOT), appended last so every
        # pre-existing column keeps its position in the CSV read contract.
        row["pair_id"] = PairIdentity.of(k1, k2)
        return row, OUTCOME_SCORED

    def assemble(self, g1: str, g2: str, label: object) -> dict[str, object] | None:
        return self.assemble_with_reason(g1, g2, label)[0]

    @classmethod
    def emitted_columns(cls) -> tuple[str, ...]:
        """The exact CSV header of the frame this assembler emits.

        ``assemble_with_reason``'s row dict is the ONE place the columns are
        spelled; reading the header off a synthetic scored row (fold n-1, so
        never both-in-train) keeps every mirror of the header drift-free by
        construction instead of by a retyped tuple.
        """
        probe = cls(FoldResolver({"0": 3}), {}, {}, n_folds=4)
        row, reason = probe.assemble_with_reason("0", "0", 1)
        if row is None:
            raise RuntimeError(f"emitted-columns probe emitted no row: {reason}")
        return tuple(row)

    def assemble_all_with_trace(
        self, labeled: pd.DataFrame, trace: TraceRun | None
    ) -> tuple[pd.DataFrame, list[dict[str, object]]]:
        """The census frame AND the per-pair outcome records for the trace.

        The iteration, order and progress bar are unchanged; the batch boundary
        is read off the existing loop and the outcome records are the exact
        per-pair facts (never a second pass over the data).
        """
        rows: list[dict[str, object]] = []
        records: list[dict[str, object]] = []
        columns = (labeled["gtin1"], labeled["gtin2"], labeled["true_label"])
        triples = zip(
            columns[0].tolist(), columns[1].tolist(), columns[2].tolist()
        )
        triples_list = list(triples)
        pairs_in_batch = 0
        rows_in_batch = 0
        batches = traced = 0
        first_pair = last_pair = ""
        for g1, g2, label in _LOG.progress(
            triples_list, desc="final_validation_rows", unit="pair"
        ):
            if pairs_in_batch == 0:
                first_pair = PairIdentity.of(g1, g2)
                rows_in_batch = 0
            last_pair = PairIdentity.of(g1, g2)
            pairs_in_batch += 1
            row, reason = self.assemble_with_reason(g1, g2, label)
            folded = {
                "pair": last_pair,
                "reason": reason,
                "label": int(label),
                "fold": row["fold"] if row is not None else "",
                "fold_2": row["fold_2"] if row is not None else "",
                "endpoint_in_train": (
                    bool(row["endpoint_in_train"]) if row is not None else ""
                ),
            }
            records.append(folded)
            if row is not None:
                rows.append(row)
                rows_in_batch += 1
            if pairs_in_batch >= _BATCH_PAIRS or pairs_in_batch == len(triples_list):
                batches += 1
                if trace is not None and traced < _MAX_BATCH_ROWS:
                    traced += 1
                    trace.add(
                        "labeled_census",
                        f"batch_{batches - 1:04d}",
                        in_count=pairs_in_batch,
                        out_count=rows_in_batch,
                        reason=(
                            "labeled pairs walked -> emitted rows; an unresolvable "
                            "endpoint or a pair with both sides in train emits none"
                        ),
                        detail={
                            "first_pair": first_pair,
                            "last_pair": last_pair,
                            "batch_pairs": _BATCH_PAIRS,
                            "max_batch_rows": _MAX_BATCH_ROWS,
                            "rows_in_batch": rows_in_batch,
                        },
                        source="data/labeled_pairs.csv over the merged component graph",
                    )
                pairs_in_batch = 0
        if trace is not None:
            trace.add(
                "labeled_census",
                "batch_census",
                in_count=batches,
                out_count=traced,
                reason=(
                    "batches traced individually; the remainder is summed here so "
                    "no chunk is silent"
                ),
                detail={
                    "pairs": len(triples_list),
                    "rows": len(rows),
                    "batches": batches,
                    "batches_traced": traced,
                    "batches_omitted": batches - traced,
                    "batch_pairs": _BATCH_PAIRS,
                    "max_batch_rows": _MAX_BATCH_ROWS,
                },
                source="data/labeled_pairs.csv over the merged component graph",
            )
        return pd.DataFrame(rows), records


#: The CSV header of ``data/final_validation.csv`` — DERIVED from the
#: assembler's own emitted row, never respelled, so consuming mirrors import
#: it instead of retyping it (SSOT).
FINAL_VALIDATION_COLUMNS: tuple[str, ...] = ValidationRowAssembler.emitted_columns()


class LeakGuards:
    """The pre-write assertions that the emitted population cannot leak."""

    @staticmethod
    def assert_no_positive_straddle(frame: pd.DataFrame) -> None:
        pos_rows = frame[frame.true_label == 1]
        straddle = int(pos_rows["straddles_fold"].sum())
        if straddle:
            raise SystemExit(
                f"LEAK: {straddle}/{len(pos_rows)} positives straddle a fold. "
                "The merged graph was not applied; refusing to write the CSV."
            )
        # A positive is one edge, so its two endpoints are linked BY DEFINITION.
        # If that ever fails, the edge was dropped and the split can leak.
        if not (pos_rows["component_id"] == pos_rows["component_id_2"]).all():
            raise SystemExit(
                "LEAK: positive endpoints in different components — the graph "
                "edge for that pair was dropped before the union-find ran."
            )

    @staticmethod
    def assert_pinned_evidence(
        policy_name: str,
        evidence: dict[str, dict[str, object]],
        min_test_negatives: int,
        *,
        diagnostic: bool = False,
    ) -> None:
        """The DECISION's re-measured-at-every-emit fail-loud guard.

        config/training.yaml pins the winning policy. If a future census
        change makes the pinned evidence stop holding, refusing to emit here
        forces the decision to be REMADE, never silently invalidated while
        the stale default keeps switching the artifact's negative
        assignment.

        ``diagnostic`` is the explicit diagnostic/sample emit: a run whose
        scored halves are too thin to confirm ANY policy (a subsampled export)
        emits its artifact for inspection, and the measured-but-unconfirmed
        evidence is recorded in the manifest under
        ``pinned_evidence_guard: "skipped_diagnostic"``. Production emits keep
        the guard (the default); this is never inferred from the data.
        """
        if diagnostic:
            _LOG.info(
                "[final_validation] diagnostic emit: pinned-evidence guard "
                f"skipped for policy {policy_name!r}; the recorded evidence is "
                "measured but unconfirmed")
            return
        if policy_name == NEGATIVE_FOLD_POLICY_TRAIN_SIDE:
            assert_pinned_evidence_train_side(policy_name, evidence, min_test_negatives)
        elif policy_name == NEGATIVE_FOLD_POLICY_WITHHOLD:
            assert_pinned_evidence_withhold(policy_name, evidence)


def _assert_no_trained_on_endpoint(policy_name: str, evidence: dict) -> None:
    """The qualitative criterion, enforced (2026-10-08).

    ``train_side`` was parked 2026-10-01 on the claim that its scored halves
    carry trained-on endpoints. The evidence surface now MEASURES that
    property per policy (``scored_negatives_with_trained_on_endpoint``), so a
    policy that really leaks can no longer be selected and a policy that does
    not is no longer rejected on prose. Refusing here keeps both directions
    honest at every emit.
    """
    leaked = int(evidence[policy_name].get(
        "scored_negatives_with_trained_on_endpoint", 0
    ))
    if leaked:
        raise SystemExit(
            "the pinned scored-half decision no longer holds: policy "
            f"{policy_name!r} scores {leaked} negatives that carry a "
            f"trained-on endpoint (evidence={evidence}). A scored half may "
            "never contain a trained-on endpoint — re-decide, update the "
            "config and the DECISION block together."
        )


def assert_pinned_evidence_train_side(policy_name, evidence, min_test_negatives):
    """Policy B's emit guard: B must score MORE negatives per scored half
    than A, must not score MORE thin-heavy negatives, and must score ZERO
    negatives with a trained-on endpoint (the criterion it was parked on)."""
    _assert_no_trained_on_endpoint(policy_name, evidence)
    was, now = (evidence[NEGATIVE_FOLD_POLICY_WITHHOLD], evidence[policy_name])
    if not (
        now["scored_test_negatives"] > was["scored_test_negatives"]
        and now["scored_dev_negatives"] > was["scored_dev_negatives"]
    ):
        raise SystemExit(
            "the pinned scored-half decision no longer holds: policy "
            f"{policy_name!r} does not score MORE negatives per fold than "
            f"{NEGATIVE_FOLD_POLICY_WITHHOLD!r} (evidence={evidence}). "
            "Re-decide, update the config and the DECISION block together "
            "— do not emit an artifact under a policy its evidence rejects."
        )
    for half in ("dev", "test"):
        thin = now["thin_cells"][half]
        if thin is None:
            continue
        thin_b = sum(now["thin_cells"][half].values())
        cells_b = sum(now["populated_cells"][half].values())
        thin_a = sum(was["thin_cells"][half].values())
        cells_a = sum(was["populated_cells"][half].values())
        share_b = thin_b / cells_b if cells_b else float("nan")
        share_a = thin_a / cells_a if cells_a else float("nan")
        if not share_b <= share_a:
            raise SystemExit(
                "the pinned scored-half decision no longer holds: policy "
                f"{policy_name!r} scores MORE thin-heavy negatives (% "
                f"cells below min_test_negatives={min_test_negatives}: "
                f"{share_b:.4f}) than {NEGATIVE_FOLD_POLICY_WITHHOLD!r} "
                f"({share_a:.4f}) on the {half} half (evidence={evidence}). "
                "Re-decide, update the config and the DECISION block "
                "together — do not emit an artifact under a policy its "
                "evidence rejects."
            )


def assert_pinned_evidence_withhold(policy_name, evidence):
    """Policy A's emit guard (added 2026-10-06 — before this branch existed,
    the assigned policy was never re-checked at emit and the artifact could
    ship under evidence it contradicts): A is clean of trained-on endpoints
    by construction (asserted above), so its own-evidence criterion is that
    the scored halves are usable at all — both must score negatives. The
    thin-cell share against B is RECORDED, not enforced: the 2026-10-06
    census reversed A's 2026-10-01 dev-half thin advantage (B is
    disqualified structurally regardless — its scored negatives carry
    trained-on endpoints, which negative_policy_evidence refuses to
    certify), so thinness is no longer a decision criterion between the
    two.

    2026-10-08: the structural disqualification is now MEASURED instead of
    asserted (``scored_negatives_with_trained_on_endpoint``); A stays clean
    here and its own-evidence criterion (both halves usable) is unchanged.
    """
    _assert_no_trained_on_endpoint(policy_name, evidence)
    now = evidence[policy_name]
    empty_halves = [
        half for half in ("dev", "test")
        if int(now[f"scored_{half}_negatives"]) <= 0
    ]
    if empty_halves:
        raise SystemExit(
            "the pinned scored-half decision no longer holds: policy "
            f"{policy_name!r} scores NO negatives on the "
            f"{', '.join(empty_halves)} half (evidence={evidence}). "
            "Re-decide, update the config and the DECISION block "
            "together — do not emit an artifact under a policy its "
            "evidence rejects."
        )


def _apply_negative_fold_policy(
    out: pd.DataFrame, policy_name: str, n_folds: int
) -> None:
    """Apply the configured policy on the ASSEMBLED frame's negatives.

    Under A the columns already hold the raw endpoint folds (no change).
    Under B BOTH columns carry the pair's whole assigned fold while
    ``straddles_fold``/``endpoint_in_train`` keep reporting the RAW endpoint
    truth — a scored-half consumer scores negatives by fold alone and keeps
    the leak guarantee's positives untouched. Evidence must measure the RAW
    endpoint folds, not post-policy columns; that is why the policy is
    applied AFTER the evidence pass.
    """
    if policy_name == NEGATIVE_FOLD_POLICY_WITHHOLD:
        return
    negative_mask = (out["true_label"] == 0).to_numpy()
    negatives = out.loc[negative_mask]
    negative_pair_folds = NegativeFoldPolicy.rule_vectorized(
        policy_name,
        negatives["fold"].astype(int),
        negatives["fold_2"].astype(int),
        n_folds,
    )
    out.loc[negative_mask, "fold"] = negative_pair_folds
    out.loc[negative_mask, "fold_2"] = negative_pair_folds


def _fold_map(fold_of: dict[str, int], comp_of: dict[str, int]) -> pd.DataFrame:
    """The split's COMPLETE accounting, sorted (fold, gtin).

    The validation CSV holds only the scored half (folds 2+3), so a consumer
    holding the full labeled census cannot tell a pair that was correctly
    withheld because the model trained on it from a pair that is simply
    MISSING. Without the map, a retargeted evaluator has to choose between
    scoring trained-on data and hard-failing on rows that are fine — which is
    how the old protocol ended up 73.7% contaminated with nothing recorded.
    """
    return pd.DataFrame(
        sorted(
            ({"gtin": bc, "fold": fold, "component_id": comp_of.get(bc, -1)}
             for bc, fold in fold_of.items()),
            key=lambda r: (r["fold"], r["gtin"]),
        )
    )


def _resolve_quarter_folds(
    train_bc: set[str], dev_bc: set[str], test_bc: set[str], n_folds: int
) -> dict[str, int]:
    """Map every graph entity to its fold: train=0, dev=n-2, test=n-1.

    train = every quarter except the last two; dev/test are the LAST two
    quarters, so validation is "neither side is a training gtin".
    """
    fold_of: dict[str, int] = {}
    for bc in train_bc:
        fold_of[bc] = 0
    for bc in dev_bc:
        fold_of[bc] = n_folds - 2
    for bc in test_bc:
        fold_of[bc] = n_folds - 1
    return fold_of


def write_manifest(
    frame: pd.DataFrame,
    stats: dict,
    *,
    path: Path,
    seed: int,
    fold_map: pd.DataFrame | None = None,
    fold_map_path: Path | None = None,
    trace: TraceRun | None = None,
) -> dict:
    pos = frame[frame.true_label == 1]
    neg = frame[frame.true_label == 0]
    grid = SliceFieldGrid()
    coverage = grid.coverage(pos)

    manifest = {
        "stage": "final_validation",
        "complete": True,
        "seed": seed,
        "output": str(path),
        "rows": int(len(frame)),
        "positives": int(len(pos)),
        "negatives": int(len(neg)),
        "positives_straddling_folds": int(pos["straddles_fold"].sum()),
        "positives_with_endpoint_in_train": int(pos["endpoint_in_train"].sum()),
        "negatives_straddling_folds": int(neg["straddles_fold"].sum()),
        "negatives_with_endpoint_in_train": int(neg["endpoint_in_train"].sum()),
        "pairs_endpoint_unresolvable": int(stats.get("pairs_endpoint_unresolvable", 0)),
        "graph": {
            "merged_positive_pairs": int(stats["merged_positive_pairs"]),
            "train_positive_pairs": int(stats["train_positive_pairs"]),
            "validation_edges_added": int(stats["edges_added"]),
            "normalized_entities": int(stats["row_entities"]),
            "endpoints_unresolved": int(stats["endpoints_unresolved"]),
        },
        "slice_fields": [name for name, _ in SLICE_FIELDS],
        "slice_agreement": _evaluation_slice_agreement(),
        "negative_fold_policy": str(
            training_cfg().split.negative_fold_policy
        ),
        "negative_policy_evidence": stats.get("negative_policy_evidence", {}),
        "pinned_evidence_guard": stats.get("pinned_evidence_guard", "enforced"),
        "slice_coverage": coverage,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publish: the manifest keys fuel resume; a truncated frame or a
    # truncated fold map must never be observable in place of a full one.
    atomic_write_csv(frame, path, index=False)
    if trace is not None:
        trace.add(
            "output",
            "published",
            in_count=int(len(frame)),
            out_count=int(len(frame)),
            reason=(
                "the validation population is written atomically after the leak "
                "guards passed; the manifest records the same numbers"
            ),
            detail={
                "path": str(path),
                "rows": int(len(frame)),
                "positives": int(len(pos)),
                "negatives": int(len(neg)),
                "pairs_endpoint_unresolvable": int(
                    stats.get("pairs_endpoint_unresolvable", 0)
                ),
                "negative_fold_policy": str(training_cfg().split.negative_fold_policy),
                "manifest": str(RESULTS / "manifests" / "final_validation.json"),
            },
            source=str(path),
        )
    if fold_map is not None and fold_map_path is not None:
        fold_map_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(fold_map, fold_map_path, index=False)
        manifest["fold_map"] = str(fold_map_path)
        manifest["fold_map_rows"] = int(len(fold_map))
        manifest["fold_map_fold_counts"] = {
            str(k): int(v) for k, v in fold_map["fold"].value_counts().items()
        }
        if trace is not None:
            trace.add(
                "fold_map",
                "published",
                in_count=int(len(fold_map)),
                out_count=int(len(fold_map)),
                reason=(
                    "every graph entity's fold is published, so a consumer can tell "
                    "a pair withheld because the model trained on it from a pair "
                    "that is simply missing"
                ),
                detail={
                    "path": str(fold_map_path),
                    "rows": int(len(fold_map)),
                    "fold_counts": manifest["fold_map_fold_counts"],
                },
                source=str(fold_map_path),
            )
    (RESULTS / "manifests" / "final_validation.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


@timed
def build(
    output: Path | None = None,
    *,
    seed: int = SEED,
    n_folds: int | None = None,
    trace: TraceRun | None = None,
    diagnostic: bool = False,
) -> pd.DataFrame:
    """Derive the merged graph, cut the split, and return the validation rows.

    ``trace`` is this stage's ONE consolidated-trace writer; with none supplied a
    standalone ``build()`` still traces, since the rows are the stage's evidence
    and a caller running it directly deserves them.

    ``diagnostic`` skips ONLY the pinned-evidence emit guard, for a
    subsampled/diagnostic run whose scored halves are too thin to confirm a
    policy; the measured evidence is still recorded and the manifest marks the
    emit ``skipped_diagnostic``. The default keeps the guard (production).
    """
    own = trace is None
    if own:
        trace = TraceRun(STAGE)
    df = load_dataset_deduped()
    from training.base_data import load_base_data

    data = load_base_data(df, payload_variant="full")
    pos = data["pos"]
    row_bc = data["row_bc"]

    merged_pos, graph_bc, stats = merged_component_graph(pos, row_bc)
    _record_graph(trace, stats)
    split = training_cfg().split
    n_folds = int(n_folds or split.holdout_component_folds)
    # Routed through the SINGLE entry point, not `holdout_split`. The selftest
    # guard bans the primitive outside folds.py precisely so the split the
    # artifact is cut from cannot be derived by different rules than the split
    # the model trains under -- which is the defect P0 exists to remove. The
    # graph is passed in pre-merged because `derive_holdout` rebuilds it (idempotent
    # here: the validation edges are already unioned, so re-union changes nothing).
    train_bc, dev_bc, test_bc = derive_holdout(
        pos, row_bc, dict(split), seed=seed
    )

    # merged_pos/row_bc: derive_holdout re-derives the same merged graph
    # internally, so these are the identical objects it split on.
    comp_of = component_ids(merged_pos, graph_bc)
    fold_of = _resolve_quarter_folds(train_bc, dev_bc, test_bc, n_folds)
    _record_folds(trace, row_bc, train_bc, dev_bc, test_bc, n_folds)
    resolver = FoldResolver(fold_of)

    labeled = pd.read_csv(
        Path(F["labeled_pairs"]), dtype={"gtin1": str, "gtin2": str},
        keep_default_na=False,
    )
    slice_values = SliceFieldGrid().canonical_values()
    assembler = ValidationRowAssembler(resolver, comp_of, slice_values, n_folds)
    with _LOG.section("final_validation.assemble_rows"):
        out, outcomes = assembler.assemble_all_with_trace(labeled, trace)
    stats["pairs_endpoint_unresolvable"] = assembler.unresolved
    _record_census(trace, labeled, out, outcomes, assembler)

    # ── the leak guarantee, asserted before anything hits disk ──
    LeakGuards.assert_no_positive_straddle(out)
    # ── the DECISION re-measured at every emit (fail-loud default guard) ──
    policy_name = str(split.negative_fold_policy)
    min_test_negatives = int(
        training_cfg().evaluation.robust_validation.min_test_negatives
    )
    with _LOG.section("final_validation.policy_evidence"):
        evidence = negative_policy_evidence(out, min_test_negatives, n_folds)
    LeakGuards.assert_pinned_evidence(policy_name, evidence, min_test_negatives,
                                      diagnostic=diagnostic)
    _record_policy(trace, out, evidence, policy_name)
    stats["negative_fold_policy"] = policy_name
    stats["negative_policy_evidence"] = evidence
    stats["pinned_evidence_guard"] = (
        "skipped_diagnostic" if diagnostic else "enforced")
    with _LOG.section("final_validation.apply_policy"):
        _apply_negative_fold_policy(out, policy_name, n_folds)
    with _LOG.section("final_validation.write"):
        write_manifest(
            out,
            stats,
            path=Path(output or F["final_validation"]),
            seed=seed,
            fold_map=_fold_map(fold_of, comp_of),
            fold_map_path=Path(F["validation_fold_map"]),
            trace=trace,
        )
    if own:
        trace.write()
    return out


# ── the stage's trace rows (real counts, named reasons) ─────────────────────
def _record_graph(trace: TraceRun, stats: dict) -> None:
    """Validation edges added to the graph, then the merged graph's census.

    The funnel is labeled positives -> edges ADDED: a positive is charged to
    ``endpoints_unresolved`` (an endpoint outside the graph), to
    ``self_edges_skipped`` (both endpoints the same row) or to the edge. The
    union itself is a census, not a funnel (the merged population is LARGER than
    the training population by construction), so it carries no counts and states
    every number in its detail rather than forcing a fake in/out pair.
    """
    positives = int(stats.get("labeled_positives", 0))
    added = int(stats.get("edges_added", 0))
    trace.add(
        "graph",
        "validation_edges_added",
        in_count=positives,
        out_count=added,
        reason=(
            "labeled POSITIVES are unioned into the split graph as edges: a "
            "positive with an endpoint outside the graph adds no edge, and a "
            "self edge is skipped; negatives are NOT identity claims and are "
            "never unioned"
        ),
        detail={
            "labeled_positives": positives,
            "edges_added": added,
            "endpoints_unresolved": int(stats.get("endpoints_unresolved", 0)),
            "self_edges_skipped": int(stats.get("self_edges_skipped", 0)),
            "identity_review_pairs_excluded": int(
                stats.get("identity_review_pairs_excluded", 0)
            ),
        },
        source="training.folds.merged_component_graph",
    )
    trace.add(
        "graph",
        "merged_census",
        reason=(
            "the merged component graph is a UNION (training positives + "
            "validation edges), so it is recorded as a census: no in/out pair "
            "would be honest"
        ),
        detail={key: int(value) for key, value in sorted(stats.items())},
        source="training.folds.merged_component_graph",
    )


def _record_folds(
    trace: TraceRun,
    row_bc,
    train_bc: set[str],
    dev_bc: set[str],
    test_bc: set[str],
    n_folds: int,
) -> None:
    """Every graph entity -> its quarter fold (train / dev = n-2 / test = n-1)."""
    entities = len(row_bc)
    assigned = len(train_bc) + len(dev_bc) + len(test_bc)
    trace.add(
        "split",
        "fold_assignment",
        in_count=entities,
        out_count=assigned,
        reason=(
            "every graph entity gets a quarter fold; dev and test are the LAST "
            f"two of {int(n_folds)} quarters, so validation means neither side is "
            "a training gtin"
        ),
        detail={
            "n_folds": int(n_folds),
            "graph_entities": entities,
            "train": len(train_bc),
            "dev": len(dev_bc),
            "test": len(test_bc),
            "unassigned": entities - assigned,
        },
        source="training.folds.derive_holdout",
    )


def _record_census(
    trace: TraceRun,
    labeled: pd.DataFrame,
    out: pd.DataFrame,
    outcomes: list[dict[str, object]],
    assembler,
) -> None:
    """Labeled pairs -> emitted rows, with every non-row outcome named."""
    trace.add(
        "labeled_census",
        "assembled",
        in_count=int(len(labeled)),
        out_count=int(len(out)),
        reason=(
            "a labeled pair becomes a validation row only when at least one "
            "endpoint is scored (fold >= n_folds-2) and every endpoint resolves "
            "in the graph"
        ),
        detail={
            "labeled_pairs": int(len(labeled)),
            "rows": int(len(out)),
            "unresolved_endpoints": int(assembler.unresolved),
            "both_endpoints_in_train": int(assembler.both_in_train),
        },
        source="data/labeled_pairs.csv over the merged component graph",
    )
    trace.add_entities(
        "labeled_census",
        outcomes,
        key_of=lambda record: record["pair"],
        reason_of=lambda record: record["reason"],
        detail_of=lambda record: {
            "label": record["label"],
            "fold": record["fold"],
            "fold_2": record["fold_2"],
            "endpoint_in_train": record["endpoint_in_train"],
        },
        source="data/labeled_pairs.csv over the merged component graph",
        per_reason=ENTITY_SAMPLE_PER_REASON,
        total_cap=ENTITY_ROW_CAP,
    )


def _record_policy(
    trace: TraceRun, out: pd.DataFrame, evidence: dict, policy_name: str
) -> None:
    """The pinned policy's re-measured evidence over the RAW endpoint folds.

    The funnel is negatives -> negatives the pinned policy scores: a negative
    parked on the train side or withheld is the drop, and the evidence dict (the
    decision's own measurement, re-computed here at every emit) is the detail.
    """
    negatives = int((out.true_label == 0).sum()) if len(out) else 0
    measured = evidence.get(policy_name, {}) if isinstance(evidence, dict) else {}
    scored = sum(
        int(measured.get(f"scored_{half}_negatives", 0))
        for half in ("dev", "test")
    )
    trace.add(
        "policy",
        "evidence_measured",
        in_count=negatives,
        out_count=scored,
        reason=(
            f"the pinned negative-fold policy {policy_name!r} is re-measured at "
            "every emit on the RAW endpoint folds; the scored halves' negatives "
            "are the retained population and the rest are parked/withheld"
        ),
        detail={
            "policy": str(policy_name),
            "negatives": negatives,
            "scored_negatives": scored,
            "evidence": measured,
        },
        source="training.build_final_validation.negative_policy_evidence",
    )


def main() -> None:
    RunLogger.configure_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--diagnostic", action="store_true",
                    help="diagnostic/sample emit: skip the pinned-evidence guard "
                         "and record the measured-but-unconfirmed evidence")
    args = ap.parse_args()
    # ONE writer for the stage: graph, folds, census, policy and publication are
    # one flow in the consolidated trace, committed once.
    trace = TraceRun(STAGE)
    frame = build(Path(args.output) if args.output else None, seed=args.seed,
                  trace=trace, diagnostic=args.diagnostic)
    trace.write()
    pos = frame[frame.true_label == 1]
    neg = frame[frame.true_label == 0]
    print(
        f"[final_validation] {len(frame):,} pairs -> "
        f"{len(pos):,} positives / {len(neg):,} negatives | "
        f"positives straddling: {int(pos.straddles_fold.sum())}"
    )
    print(f"[final_validation] wrote {F['final_validation']}")


if __name__ == "__main__":
    main()
