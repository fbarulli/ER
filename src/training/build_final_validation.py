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
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from core.common import F, RESULTS, SEED, load_dataset_deduped, training_cfg
from core.schemas import check_canonical_records_frame, upgrade_canonical_records_frame
from training.folds import (
    component_ids,
    derive_holdout,
    merged_component_graph,
    normalize_gtin,
)

# The six fields P0 keeps as gates. `pulp_set` is deliberately absent: it is
# populated in 2.3% of canonical records and 0.5% of verified positives, which
# is 2 pairs in this validation half -- population scarcity, not a parsing
# defect, and no gate at any budget that respects the component constraint.
SLICE_FIELDS: tuple[tuple[str, str], ...] = (
    ("volume", "volume_set"),
    ("pack", "pack_set"),
    ("package_type", "package_type_set"),
    ("sweetener", "sweetener_set"),
    ("flavor", "flavor_set"),
    ("carbonation", "carbonation_set"),
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
# ════════════════════════════════════════════════════════════════════════════

# Policy names. Both stay load-valid; config/training.yaml pins the winner.
NEGATIVE_FOLD_POLICY_WITHHOLD = "withhold_straddle"
NEGATIVE_FOLD_POLICY_TRAIN_SIDE = "train_side"


def negative_pair_fold(
    policy: str, fold_a: int, fold_b: int, n_folds: int
) -> int:
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


def negative_policy_evidence(
    frame: pd.DataFrame, min_test_negatives: int, n_folds: int = 4
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
    criteria: dict[str, dict[str, object]] = {}
    for policy in (NEGATIVE_FOLD_POLICY_WITHHOLD, NEGATIVE_FOLD_POLICY_TRAIN_SIDE):
        pos = frame[frame.true_label == 1]
        neg = frame[frame.true_label == 0]
        # A positive's pair-fold is its shared endpoint fold under BOTH
        # policies (the leak guarantee asserts fold == fold_2 for positives,
        # so a single column carries it).
        dev, test = n_folds - 2, n_folds - 1
        in_dev_pos = pos["fold"].astype(int) == dev
        in_test_pos = pos["fold"].astype(int) == test
        f1 = neg["fold"].astype(int)
        f2 = neg["fold_2"].astype(int)
        if policy == NEGATIVE_FOLD_POLICY_WITHHOLD:
            # A: legacy — a negative scores only where BOTH endpoint folds
            # equal the scored fold; a mismatched pair scores nowhere.
            in_dev_neg = (f1 == dev) & (f2 == dev)
            in_test_neg = (f1 == test) & (f2 == test)
        else:
            assigned = neg.apply(
                lambda row: negative_pair_fold(
                    policy, int(row["fold"]), int(row["fold_2"]), n_folds=n_folds
                ),
                axis=1,
            )
            in_dev_neg = assigned == dev
            in_test_neg = assigned == test
        scored = {
            "dev": in_dev_neg, "test": in_test_neg,
        }
        # thin-cell criterion requires the frozen slice columns; a frame
        # without them (synthetic fold-contract tests) records UNavailable —
        # explicit, never silently claimed as "not thin".
        has_slices = all(f"v1_{name}" in frame.columns for name, _ in SLICE_FIELDS)
        thin: dict[str, dict[str, int]] | None = (
            {} if has_slices else None
        )
        populated: dict[str, dict[str, int]] | None = (
            {} if has_slices else None
        )
        for half, mask in scored.items() if has_slices else ():
            rows = neg.loc[mask]
            for field, _col in SLICE_FIELDS:
                bag = Counter(rows[f"v1_{field}"][rows[f"v1_{field}"] != ""])
                extra = rows[f"v2_{field}"][
                    (rows[f"v2_{field}"] != "") & (rows[f"v2_{field}"] != rows[f"v1_{field}"])
                ]
                bag.update(extra)
                populated.setdefault(half, {})[field] = len(bag)
                thin.setdefault(half, {})[field] = sum(
                    1 for count in bag.values() if count < min_test_negatives
                )
        criteria[policy] = {
            "scored_dev_negatives": int(in_dev_neg.sum()),
            "scored_test_negatives": int(in_test_neg.sum()),
            "scored_dev_positives": int(in_dev_pos.sum()),
            "scored_test_positives": int(in_test_pos.sum()),
            "negatives_withheld_from_scored_half": int(
                len(neg) - (in_dev_neg.sum() + in_test_neg.sum())
            ),
            "populated_cells": populated,
            "thin_cells": thin,
        }
        pos_straddle = int((pos["fold"] != pos["fold_2"]).sum())
        if pos_straddle:
            raise SystemExit(
                f"{pos_straddle} positives straddle a fold — the policy "
                "evidence population is corrupt (run the leak guards first)"
            )
    return criteria


def _canonical_values() -> dict[str, dict[str, str]]:
    """gtin -> {field: canonical value string} from the frozen canonical records."""
    canon = pd.read_csv(
        F["canonical_records"], dtype=str, keep_default_na=False, low_memory=False
    )
    # Same read contract as the pipeline lanes: migrate a stale artifact and
    # validate before slicing — the column check below makes the old
    # "first column might be the key" fallback unreachable.
    canon = upgrade_canonical_records_frame(canon)
    check_canonical_records_frame(canon)
    canon["_key"] = canon["gtin"].map(normalize_gtin)
    out: dict[str, dict[str, str]] = {}
    for _, row in canon.iterrows():
        key = row["_key"]
        if not key or key in out:
            continue
        out[key] = {name: str(row.get(col, "") or "") for name, col in SLICE_FIELDS}
    return out


def build(
    output: Path | None = None,
    *,
    seed: int = SEED,
    n_folds: int | None = None,
) -> pd.DataFrame:
    """Derive the merged graph, cut the split, and return the validation rows."""
    df = load_dataset_deduped()
    from training.base_data import load_base_data

    data = load_base_data(df, payload_variant="full")
    pos = data["pos"]
    row_bc = data["row_bc"]

    merged_pos, graph_bc, stats = merged_component_graph(pos, row_bc)
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
    # train = every quarter except the last two; dev/test are the LAST two
    # quarters, so validation is "neither side is a training gtin".
    fold_of: dict[str, int] = {}
    for bc in train_bc:
        fold_of[bc] = 0
    for bc in dev_bc:
        fold_of[bc] = n_folds - 2
    for bc in test_bc:
        fold_of[bc] = n_folds - 1

    # A labeled gtin has to be resolved to the spelling the graph actually
    # uses, and the two are not the same string: the fold sets are keys of the
    # RAW row_bc, while normalize_gtin left-pads a 13-digit gtin to 14. A
    # 13-digit gtin therefore misses a raw fold set under a normalized lookup
    # and is silently dropped -- which is how an earlier run of this script
    # emitted 3 rows out of 8,889. Try the raw spelling first, then the
    # normalized one, and COUNT the misses rather than skipping in silence.
    raw_keys = set(fold_of)
    norm_keys = {normalize_gtin(b) for b in fold_of}

    def resolve(gtin: str) -> str | None:
        raw = str(gtin).strip()
        if raw in raw_keys:
            return raw
        normed = normalize_gtin(raw)
        return normed if normed in raw_keys or normed in norm_keys else None

    labeled = pd.read_csv(
        F["labeled_pairs"], dtype={"gtin1": str, "gtin2": str}, keep_default_na=False
    )
    canon = _canonical_values()

    rows: list[dict[str, object]] = []
    unresolvable = 0
    for g1, g2, label in zip(
        labeled["gtin1"], labeled["gtin2"], labeled["true_label"]
    ):
        k1, k2 = resolve(g1), resolve(g2)
        if k1 is None or k2 is None:
            # An endpoint outside the graph entirely: it has no fold, so it
            # cannot be part of a fold-2+3 population.
            unresolvable += 1
            continue
        if fold_of[k1] < n_folds - 2 and fold_of[k2] < n_folds - 2:
            continue  # both sides in train -> not validation
        f1, f2 = fold_of[k1], fold_of[k2]
        label = int(label)
        # SCORED-HALF POLICY: applied AFTER the evidence pass (below), on the
        # assembled frame's NEGATIVE rows only — a positive keeps its shared
        # endpoint fold (leak guarantee). Evidence must measure the RAW
        # endpoint folds, not post-policy columns; applying the policy here
        # would make policy A's evidence read the already-B-shaped frame.
        c1 = canon.get(normalize_gtin(k1), {})
        c2 = canon.get(normalize_gtin(k2), {})
        row: dict[str, object] = {
            "gtin1": k1,
            "gtin2": k2,
            "gtin1_norm": normalize_gtin(k1),
            "gtin2_norm": normalize_gtin(k2),
            "true_label": int(label),
            "fold": f1,
            "fold_2": f2,
            "component_id": comp_of.get(k1, -1),
            "component_id_2": comp_of.get(k2, -2),
            "straddles_fold": f1 != f2,
            "endpoint_in_train": min(f1, f2) < n_folds - 2,
        }
        for name, _col in SLICE_FIELDS:
            row[f"v1_{name}"] = c1.get(name, "")
            row[f"v2_{name}"] = c2.get(name, "")
        rows.append(row)

    out = pd.DataFrame(rows)

    # ── the leak guarantee, asserted before anything hits disk ──
    pos_rows = out[out.true_label == 1]
    straddle = int(pos_rows["straddles_fold"].sum())
    if straddle:
        raise SystemExit(
            f"LEAK: {straddle}/{len(pos_rows)} positives straddle a fold. The "
            "merged graph was not applied; refusing to write the CSV."
        )
    # A positive is one edge, so its two endpoints are linked BY DEFINITION.
    # If that ever fails, the edge was dropped and the split can leak.
    if not (pos_rows["component_id"] == pos_rows["component_id_2"]).all():
        raise SystemExit(
            "LEAK: positive endpoints in different components — the graph edge "
            "for that pair was dropped before the union-find ran."
        )

    stats["pairs_endpoint_unresolvable"] = unresolvable
    # ── the DECISION re-measured at every emit (fail-loud default guard) ──
    # config/training.yaml pins "train_side" as the winner of the decision
    # block above. If a future census change makes the pinned evidence stop
    # holding (policy B no longer beats A on scored half counts or the
    # thinness share), refusing to emit here forces the decision to be
    # REMADE, never silently invalidated while the stale default keeps
    # switching the artifact's negative assignment.
    policy_name = str(split.negative_fold_policy)
    min_test_negatives = int(
        training_cfg().evaluation.robust_validation.min_test_negatives
    )
    evidence = negative_policy_evidence(out, min_test_negatives, n_folds)
    if policy_name == NEGATIVE_FOLD_POLICY_TRAIN_SIDE:
        was, now = evidence[NEGATIVE_FOLD_POLICY_WITHHOLD], evidence[policy_name]
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
    stats["negative_fold_policy"] = policy_name
    stats["negative_policy_evidence"] = evidence
    # ── apply the configured policy on the ASSEMBLED frame's negatives ──
    # Under A the columns already hold the raw endpoint folds (no change).
    # Under B BOTH columns carry the pair's whole assigned fold while
    # `straddles_fold`/`endpoint_in_train` keep reporting the RAW endpoint
    # truth — a scored-half consumer scores negatives by fold alone and
    # keeps the leak guarantee's positives untouched.
    if policy_name != NEGATIVE_FOLD_POLICY_WITHHOLD:
        neg_mask = (out["true_label"] == 0).to_numpy()
        assigned = [
            negative_pair_fold(policy_name, int(f1), int(f2), n_folds)
            for f1, f2 in zip(out.loc[neg_mask, "fold"], out.loc[neg_mask, "fold_2"])
        ]
        out.loc[neg_mask, "fold"] = assigned
        out.loc[neg_mask, "fold_2"] = assigned
    # The fold map is the split's COMPLETE accounting, and it is not optional.
    # The validation CSV holds only the scored half (folds 2+3), so a consumer
    # holding the full labeled census cannot tell a pair that was correctly
    # withheld because the model trained on it from a pair that is simply
    # MISSING. Without the map, a retargeted evaluator has to choose between
    # scoring trained-on data and hard-failing on rows that are fine — which is
    # how the old protocol ended up 73.7% contaminated with nothing recorded.
    fold_map = pd.DataFrame(
        sorted(
            ({"gtin": bc, "fold": fold_of[bc], "component_id": comp_of.get(bc, -1)}
             for bc in fold_of),
            key=lambda r: (r["fold"], r["gtin"]),
        )
    )
    out_path = Path(output or F["final_validation"])
    write_manifest(
        out,
        stats,
        path=out_path,
        seed=seed,
        fold_map=fold_map,
        fold_map_path=Path(F["validation_fold_map"]),
    )
    return out


def parse_field_bag(raw: object) -> Counter:
    """Slice column -> multiset of extracted values (the set-valued side).

    The emitted slice columns carry a canonical list literal ("[lime, lime]",
    "[479.0, 518.0]" — bare tokens, never quoted) or a bare single token.
    Comparing the raw STRINGS was the old scalar semantics; this tokenizer is
    what bag identity needs. Fail-loud, not graceful: an unbalanced bracket
    value raises (a silently unreadable slice would downgrade to raw-string
    equality without anyone knowing).
    """
    text = str(raw).strip()
    if text.startswith("[") != text.endswith("]"):
        raise ValueError(f"unbalanced slice list {raw!r}")
    if text.startswith("[") and text.endswith("]"):
        content = text[1:-1]
        tokens = [
            token.strip().strip("'\"")
            for token in content.split(",")
            if token.strip().strip("'\"")
        ]
        return Counter(tokens)
    token = text or None
    return Counter([token]) if token else Counter()


def _evaluation_slice_agreement() -> str:
    """The set-semantics switch (evaluation.slice_agreement, fail-loud load)."""
    return str(training_cfg().evaluation.slice_agreement)


def count_slice_disagreements(a: pd.Series, b: pd.Series, semantics: str) -> int:
    """THE slice-flag comparison (both semantics implemented, config-chosen).

    ``scalar``: legacy raw string v1 == v2 per pair (the old behaviour,
    reachable for byte-stability audits). ``set_bag``: BAG equality after
    parse_field_bag — order/spacing-only spelling differences agree;
    contents differences never do.
    """
    if semantics == "scalar":
        return int((a != b).sum())
    if semantics == "set_bag":
        return int(
            sum(parse_field_bag(x) != parse_field_bag(y) for x, y in zip(a, b))
        )
    raise ValueError(
        f"unknown slice_agreement {semantics!r}; expected one of "
        "('scalar', 'set_bag')"
    )


def write_manifest(
    frame: pd.DataFrame,
    stats: dict,
    *,
    path: Path,
    seed: int,
    fold_map: pd.DataFrame | None = None,
    fold_map_path: Path | None = None,
) -> dict:
    pos = frame[frame.true_label == 1]
    neg = frame[frame.true_label == 0]

    # Per-field measuring power, because "we have 564 positives" is not the
    # question a gate asks -- "can this field carry a floor" is. `disagree` is
    # the count of positives whose two endpoints carry different values for
    # the field: same product, one side's text mentions an extra value. That is
    # legitimate extractor variance, so it is reported rather than asserted on,
    # but a gate comparing v1 to v2 needs to know it exists.
    coverage: dict[str, dict[str, int]] = {}
    for name, _col in SLICE_FIELDS:
        a, b = pos[f"v1_{name}"], pos[f"v2_{name}"]
        if not len(pos):
            coverage[name] = {"positives": 0, "distinct": 0, "largest_bucket": 0,
                              "unpopulated": 0, "disagree": 0}
            continue
        counts = a[a != ""].value_counts()
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
        disagreement = count_slice_disagreements(
            a, b, _evaluation_slice_agreement()
        )
        coverage[name] = {
            "positives": int(len(pos)),
            "distinct": int(counts.size),
            "largest_bucket": int(counts.iloc[0]) if counts.size else 0,
            "unpopulated": int((a == "").sum()),
            "disagree": disagreement,
        }

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
        "slice_coverage": coverage,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if fold_map is not None and fold_map_path is not None:
        fold_map_path.parent.mkdir(parents=True, exist_ok=True)
        fold_map.to_csv(fold_map_path, index=False)
        manifest["fold_map"] = str(fold_map_path)
        manifest["fold_map_rows"] = int(len(fold_map))
        manifest["fold_map_fold_counts"] = {
            str(k): int(v) for k, v in fold_map["fold"].value_counts().items()
        }
    (RESULTS / "manifests" / "final_validation.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    frame = build(Path(args.output) if args.output else None, seed=args.seed)
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
