"""Hard-negative mining for euromonitor entity resolution.

Mines cross-barcode pairs the bi-encoder finds confusing (the 0.45-0.80 cosine
band) while EXCLUDING known label errors: same-title+brand rows carrying
conflicting barcodes (the mislabeled-barcode groups). The output is auditable —
a CSV with title/brand/barcode/cosine per pair — so a human can hand-label a
sample and the exclusion is visible, never hidden.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
import re

import numpy as np
import pandas as pd


def normalized_product_name(title: object, brand: object) -> str:
    """Stable product-name key with pack/size/container evidence removed.

    Flavor, carbonation, sweetener/diet, and pulp words deliberately remain;
    they are identity-bearing and must not make two different variants look
    like the same product name.

    FUSED MULTIPLIER FORM (audit 2026-09-15): the two generic expressions
    below need a word boundary in front of the size, so a FUSED token such as
    ``18x33cl`` / ``12x330ml`` / ``6x1.5l`` was never normalized — ``x`` and
    ``3`` are both word characters, so the size part is unreachable and the
    whole token survived as if it were identity-bearing. On the live gate
    candidate window 2,396 candidates carried a fused token and **115 conflict
    candidates were rejected by name equality alone** (every one of them a gate
    ``hard_no`` with reason ``Pack blocker``; 101 volume, 39 pack conflicts) —
    i.e. the name key silently made those pack/volume negatives unreachable.
    The fused form is normalized first; ``N x M`` with spaces still falls
    through to the generic passes.
    """
    from core.critical_attributes import normalized_attribute_text

    text = normalized_attribute_text(title)
    brand_tokens = set(normalized_attribute_text(brand).split())
    text = re.sub(
        r"\b\d+\s*[x×]\s*\d+(?:[.,]\d+)?\s*[a-z]*\b",
        " ",
        text,
    )
    text = re.sub(
        r"\b\d+(?:[.,]\d+)?\s*(?:ml|cl|l|lt|ltr|liters?|litres?|"
        r"fl\s*oz|fluid\s+ounces?|oz|ounces?|qt|quarts?|pt|pints?|"
        r"gal|gallons?)\b",
        " ",
        text,
    )
    text = re.sub(
        r"\b(?:pack|case|count)\s*(?:of\s*)?\d+\b|"
        r"\b\d+\s*(?:pack|packs|pk|count|ct|units?|pieces?|pcs|"
        r"bottles?|cans?|tins?|cartons?|boxes?|packets?|sachets?|bags?)\b",
        " ",
        text,
    )
    container_noise = {
        "pack", "packs", "case", "count", "ct", "unit", "units",
        "bottle", "bottles", "can", "cans", "tin", "tins", "carton",
        "cartons", "box", "boxes", "packet", "packets", "sachet",
        "sachets", "bag", "bags",
    }
    return " ".join(
        token
        for token in text.split()
        if token not in brand_tokens and token not in container_noise
    )


def flavor_variant_product_name(name: object) -> str:
    """Product name with identity-bearing FLAVOR words removed.

    Used by the miner's flavour-variant rule: two canonicals that present the
    same name once flavour tokens are dropped are candidate flavour conflicts
    (``pear soda`` vs ``raspberry soda``), which strict name equality can
    never reach because the flavour word is part of the name by design.
    """
    from core.critical_attributes import FLAVOR_LEXICON

    return " ".join(
        token for token in str(name or "").split() if token not in FLAVOR_LEXICON
    )


@dataclass
class MiningFunnel:
    """Step-by-step accounting of one targeted-attribute mining pass.

    WHY THIS EXISTS (audit 2026-09-15): the miner used to return only its
    output, so "why did this lane stop at N pairs?" could not be answered from
    the run — the caller had to re-implement every filter to find out. On the
    live artifacts the answer was that the name-equality filter alone killed
    93% of the candidate window, which no output count reveals. The funnel is
    the miner's OWN readback: candidate -> similarity floor -> canonical
    resolution -> same-canonical guard -> brand -> name -> conflict -> emitted,
    with per-filter drop counts and a conflict-dimension census.

    DELIVERY CONTRACT (an out-of-tree caller can consume it):
    ``mine_targeted_attribute_negatives(..., funnel=<MiningFunnel instance>)``
    fills the passed instance in place and still returns ``(pairs, scores)``,
    so no existing caller changes. ``mine_targeted_attribute_negatives_with_funnel``
    is the same call returning the instance as a third value. ``stages()``
    yields ``(step, in_count, out_count, reason)`` rows ready for
    ``core.tracing.TraceRun.add``; ``to_dict()`` is the JSON detail payload.
    See the trace-owner call documented on the miner itself.

    The two ``*_census`` mappings are computed ONLY when a funnel is passed
    (they require an attribute evaluation per rejected candidate, ~40k on the
    live window); every count above them is exact either way.
    """

    miner: str = "targeted_attribute_negatives"
    n_target: int = 0
    min_similarity: float = 0.0
    volume_relative_tolerance: float = 0.0
    volume_absolute_tolerance_ml: float = 0.0
    name_match: str = "flavor_variant"
    skipped_reason: str = ""
    gate_rows: int = 0
    above_similarity_floor: int = 0
    above_floor_hard_no: int = 0
    above_floor_proceed: int = 0
    above_floor_fallback: int = 0
    # Candidate-row drop counters: one candidate = one (gtin1, gtin2) gate row.
    dropped_candidates_no_canonical_record: int = 0
    dropped_candidates_no_source_row: int = 0
    dropped_candidates_no_canonical_index: int = 0
    dropped_candidates_same_canonical: int = 0
    dropped_candidates_brand: int = 0
    dropped_candidates_name: int = 0
    dropped_candidates_no_conflict: int = 0
    flavor_variant_candidates: int = 0
    passed_candidates: int = 0
    # Pair-direction counters: a candidate emits up to two (anchor, target) rows.
    dropped_pairs_already_in_baseline: int = 0
    emitted_pairs: int = 0
    conflict_dimension_census: dict[str, int] = field(default_factory=dict)
    name_blocked_conflict_dimension_census: dict[str, int] = field(default_factory=dict)

    def stages(self) -> list[tuple[str, int, int, str]]:
        """Return the funnel as ordered ``(step, in_count, out_count, reason)``.

        Counts are cumulative down the funnel: each step's ``out_count`` is the
        next step's ``in_count``, so a trace read top-to-bottom is the data
        flow and no step can silently lie about its own attrition. The first
        block is counted in CANDIDATE ROWS (gate pairs); ``direction_expansion``
        switches to PAIR DIRECTIONS, because one candidate emits two
        ``(source SKU, other canonical)`` rows.
        """
        steps: list[tuple[str, int, int, str]] = []
        cursor = int(self.gate_rows)
        for name, dropped, reason in self._candidate_drops():
            steps.append((name, cursor, cursor - int(dropped), reason))
            cursor -= int(dropped)
        directions = 2 * cursor
        steps.append((
            "direction_expansion",
            cursor,
            directions,
            "one candidate emits both (source SKU -> other canonical) directions",
        ))
        steps.append((
            "baseline_deduplication",
            directions,
            directions - int(self.dropped_pairs_already_in_baseline),
            "pair direction already present in the baseline negative population",
        ))
        steps.append((
            "emitted",
            directions - int(self.dropped_pairs_already_in_baseline),
            int(self.emitted_pairs),
            "pair directions emitted after the target cap",
        ))
        return steps

    def _candidate_drops(self) -> list[tuple[str, int, str]]:
        return [
            (
                "gate_similarity_floor",
                self.gate_rows - self.above_similarity_floor,
                f"gate similarity not strictly above {self.min_similarity:g}",
            ),
            (
                "canonical_records_resolution",
                self.dropped_candidates_no_canonical_record,
                "endpoint GTIN has no canonical record",
            ),
            (
                "representative_row_resolution",
                self.dropped_candidates_no_source_row,
                "endpoint GTIN has no source row in the SKU frame",
            ),
            (
                "canonical_index_resolution",
                self.dropped_candidates_no_canonical_index,
                "endpoint GTIN has no payload canonical index",
            ),
            (
                "same_canonical_guard",
                self.dropped_candidates_same_canonical,
                "both GTINs resolve to the same canonical item (true match)",
            ),
            (
                "brand_equality",
                self.dropped_candidates_brand,
                "canonical brands differ or are empty",
            ),
            (
                "product_name_equality",
                self.dropped_candidates_name,
                f"normalized product names differ (name_match={self.name_match})",
            ),
            (
                "critical_attribute_conflict",
                self.dropped_candidates_no_conflict,
                "no critical attribute conflict under the gate's own tolerance",
            ),
        ]

    def bottleneck(self, *, exclude_similarity_floor: bool = False) -> str:
        """The candidate-level step that dropped the most candidates.

        ``exclude_similarity_floor=True`` answers the actionable question —
        which FILTER inside the candidate window is the real constraint —
        instead of naming the configured threshold, which is a knob rather
        than a defect.
        """
        drops = self._candidate_drops()
        if exclude_similarity_floor:
            drops = [item for item in drops if item[0] != "gate_similarity_floor"]
        return max(drops, key=lambda item: item[1])[0] if drops else ""

    def candidate_ceiling_pct(self) -> float:
        """Passing share of the candidates that cleared the similarity floor."""
        if not self.above_similarity_floor:
            return 0.0
        return 100.0 * self.passed_candidates / self.above_similarity_floor

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable readback for a trace ``detail`` column."""
        return {
            "miner": self.miner,
            "name_match": self.name_match,
            "target": int(self.n_target),
            "min_similarity": float(self.min_similarity),
            "volume_relative_tolerance": float(self.volume_relative_tolerance),
            "volume_absolute_tolerance_ml": float(self.volume_absolute_tolerance_ml),
            "skipped_reason": self.skipped_reason,
            "gate_rows": int(self.gate_rows),
            "above_similarity_floor": int(self.above_similarity_floor),
            "above_floor_by_decision": {
                "hard_no": int(self.above_floor_hard_no),
                "proceed": int(self.above_floor_proceed),
                "fallback": int(self.above_floor_fallback),
            },
            "dropped_candidates": {
                name: int(dropped) for name, dropped, _ in self._candidate_drops()
            },
            "flavor_variant_candidates": int(self.flavor_variant_candidates),
            "passed_candidates": int(self.passed_candidates),
            "dropped_pairs_already_in_baseline": int(self.dropped_pairs_already_in_baseline),
            "emitted_pairs": int(self.emitted_pairs),
            "bottleneck": self.bottleneck(),
            "candidate_bottleneck": self.bottleneck(exclude_similarity_floor=True),
            "candidate_to_emitted_pct": round(self.candidate_ceiling_pct(), 4),
            "target_reached": bool(self.emitted_pairs >= self.n_target > 0),
            "conflict_dimension_census": dict(sorted(self.conflict_dimension_census.items())),
            "name_blocked_conflict_dimension_census": dict(
                sorted(self.name_blocked_conflict_dimension_census.items())
            ),
        }


def mine_targeted_attribute_negatives(
    df: pd.DataFrame,
    gates: pd.DataFrame,
    canonical_records: pd.DataFrame,
    gtin_to_row: dict[str, int],
    gtin_to_canon_idx: dict[str, int],
    *,
    existing: np.ndarray | None = None,
    n_target: int,
    min_similarity: float,
    volume_relative_tolerance: float = 0.0,
    volume_absolute_tolerance_ml: float = 0.0,
    canonical_map: dict[str, str] | None = None,
    name_match: str = "flavor_variant",
    funnel: MiningFunnel | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mine same-brand/name critical-attribute negatives from gate evidence.

    ``similarity`` is the existing gate candidate's short-canonical-token
    Jaccard score. The threshold is strict (``score > min_similarity``).
    Both directions are emitted as source-SKU -> other-canonical pairs and
    all critical conflicts come from the same evaluator used at inference.

    SAME-CANONICAL GUARD (audit 2026-09-15): two GTINs can resolve to the
    SAME canonical item. ``build_training_data`` drops exactly those pairs
    with the documented rule "Those rows are true matches and must never be
    emitted as label-0 pairs", and reports the count as
    ``n_neg_same_canonical_dropped``. The gate's ``similarity`` score does not
    know about that identity, so without this guard the miner re-added 154 of
    the 350 committed targeted negatives — 44% — as label-0 pairs whose two
    GTINs share byte-identical canonical text, giving one anchor/target pair
    both label 1 and label 0. Passing ``canonical_map`` (gtin -> canonical
    string) applies the same identity rule here.

    SSOT (audit 2026-09-15): the conflict verdict MUST use the same volume
    tolerance as the training-label gate. Defaulting to exact equality here
    made the two lanes disagree on the same pair — a pair whose volumes sit
    inside the gate's ``vol_tolerance`` was labelled ``proceed`` by the gate
    (a positive) while this miner emitted it as a hard negative. Live check
    found exactly such a pair (`8002267004212`/`8002267025644`: identical
    canonical text, volumes 480 vs 500 = 4.0% <= 0.05, gate ``proceed``).

    NAME RULE (audit 2026-09-15): strict name equality is the filter that
    actually bounds this lane — on the live gate artifact it dropped 40,269 of
    the 43,209 candidates that reached it (93.2%), and because the flavour word
    is part of the name BY DESIGN, no flavour-conflict pair could ever be
    emitted (5,549 flavour-conflict candidates above the floor, 0 reachable,
    while the gate itself labels 876 rows "Critical attribute mismatch:
    flavor"). ``name_match="flavor_variant"`` (default) admits a candidate
    whose names are equal once FLAVOR_LEXICON tokens are dropped ONLY when the
    pair carries an explicit flavour conflict AND the training gate already
    labels that same row ``hard_no``. The gate is the label authority: this
    rule can only re-expose pairs the gate has itself called label 0, never
    relabel a ``proceed``/``fallback`` row. ``name_match="exact"`` restores the
    pre-audit behaviour (196 pairs on the live artifact vs 504 with the rule).

    FUNNEL (audit 2026-09-15): pass a :class:`MiningFunnel` through ``funnel=``
    to receive the miner's own step-by-step accounting. The consolidated-trace
    owner should call the miner as::

        from core.hard_negatives import (
            MiningFunnel, mine_targeted_attribute_negatives,
        )
        funnel = MiningFunnel()
        targeted_neg, targeted_scores = mine_targeted_attribute_negatives(
            df, gates, canonical_records, gtin_to_row, gtin_to_canon_idx,
            existing=neg, n_target=int(targeted_cfg["target"]),
            min_similarity=float(targeted_cfg["min_similarity"]),
            volume_relative_tolerance=float(training_cfg().gate.vol_tolerance),
            canonical_map=canon_map, funnel=funnel,
        )
        for step, in_count, out_count, reason in funnel.stages():
            pair_trace.add(
                "mining", f"targeted_attribute_funnel.{step}",
                in_count=in_count, out_count=out_count,
                reason=reason, detail=funnel.to_dict(),
                source="gate_results.csv + canonical_records.csv",
            )

    The returned pair/score arrays keep their historical shape and order.
    """
    name_match = str(name_match)
    if name_match not in {"exact", "flavor_variant"}:
        raise ValueError(
            "name_match must be 'exact' or 'flavor_variant'; "
            f"got {name_match!r}"
        )
    if funnel is not None:
        funnel.miner = "targeted_attribute_negatives"
        funnel.n_target = int(n_target)
        funnel.min_similarity = float(min_similarity)
        funnel.volume_relative_tolerance = float(volume_relative_tolerance)
        funnel.volume_absolute_tolerance_ml = float(volume_absolute_tolerance_ml)
        funnel.name_match = name_match
        funnel.gate_rows = int(len(gates))
    if n_target <= 0:
        if funnel is not None:
            funnel.skipped_reason = "n_target <= 0"
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)
    from core.attribute_conflicts import (
        canonical_attribute_info,
        critical_attribute_evaluation,
    )

    def _census(target: dict[str, int], dimensions: object) -> None:
        for dimension in dimensions or ():
            target[str(dimension)] = target.get(str(dimension), 0) + 1

    def _evaluate(left: object, right: object) -> dict[str, list[str]]:
        return critical_attribute_evaluation(
            canonical_attribute_info(left),
            canonical_attribute_info(right),
            volume_relative_tolerance=float(volume_relative_tolerance),
            volume_absolute_tolerance_ml=float(volume_absolute_tolerance_ml),
        )

    records = {
        str(row["gtin"]): row
        for row in canonical_records.to_dict("records")
    }
    existing_keys = {
        (int(a), int(b)) for a, b in (existing if existing is not None else [])
    }
    ranked = gates.assign(
        candidate_similarity=pd.to_numeric(gates["similarity"], errors="coerce")
    )
    ranked = ranked[ranked["candidate_similarity"].gt(float(min_similarity))]
    if funnel is not None:
        funnel.above_similarity_floor = int(len(ranked))
        decisions = ranked["gate_decision"].astype(str).value_counts().to_dict() if "gate_decision" in ranked else {}
        funnel.above_floor_hard_no = int(decisions.get("hard_no", 0))
        funnel.above_floor_proceed = int(decisions.get("proceed", 0))
        funnel.above_floor_fallback = int(decisions.get("fallback", 0))
    ranked = ranked.sort_values(
        ["candidate_similarity", "gtin1", "gtin2"],
        ascending=[False, True, True],
        kind="stable",
    )
    found: list[tuple[int, int, float]] = []
    for row in ranked.itertuples(index=False):
        left_gtin, right_gtin = str(row.gtin1), str(row.gtin2)
        if left_gtin not in records or right_gtin not in records:
            if funnel is not None:
                funnel.dropped_candidates_no_canonical_record += 1
            continue
        if left_gtin not in gtin_to_row or right_gtin not in gtin_to_row:
            if funnel is not None:
                funnel.dropped_candidates_no_source_row += 1
            continue
        if left_gtin not in gtin_to_canon_idx or right_gtin not in gtin_to_canon_idx:
            if funnel is not None:
                funnel.dropped_candidates_no_canonical_index += 1
            continue
        # Same canonical item => true match, never a label-0 pair.
        if canonical_map is not None:
            left_canon = canonical_map.get(left_gtin)
            right_canon = canonical_map.get(right_gtin)
            if left_canon is not None and left_canon == right_canon:
                if funnel is not None:
                    funnel.dropped_candidates_same_canonical += 1
                continue
        left_record, right_record = records[left_gtin], records[right_gtin]
        left_row, right_row = gtin_to_row[left_gtin], gtin_to_row[right_gtin]
        left_brand = str(left_record.get("mode_brand", "")).strip().casefold()
        right_brand = str(right_record.get("mode_brand", "")).strip().casefold()
        if not left_brand or left_brand != right_brand:
            if funnel is not None:
                funnel.dropped_candidates_brand += 1
            continue
        left_name = normalized_product_name(df.iloc[left_row].get("title", ""), left_brand)
        right_name = normalized_product_name(df.iloc[right_row].get("title", ""), right_brand)
        exact_name = bool(left_name) and left_name == right_name
        evaluation: dict[str, list[str]] | None = None
        if not exact_name:
            left_residual = flavor_variant_product_name(left_name)
            relaxed = (
                name_match == "flavor_variant"
                and bool(left_residual)
                and left_residual == flavor_variant_product_name(right_name)
                # The gate is the label authority: only a pair the gate has
                # ALREADY called hard_no may be re-exposed by this rule.
                and str(getattr(row, "gate_decision", "")) == "hard_no"
            )
            if funnel is not None:
                # Census of what the NAME filter blocks: the evidence that
                # made flavour/pulp negatives structurally unreachable.
                evaluation = _evaluate(left_record, right_record)
                _census(
                    funnel.name_blocked_conflict_dimension_census,
                    evaluation["conflicts"],
                )
            if not relaxed:
                if funnel is not None:
                    funnel.dropped_candidates_name += 1
                continue
            if evaluation is None:
                evaluation = _evaluate(left_record, right_record)
            if "flavor" not in evaluation["conflicts"]:
                # The names differ only by flavour words, but the shared
                # evaluator sees no flavour conflict: nothing identity-bearing
                # separates them, so this is not a negative.
                if funnel is not None:
                    funnel.dropped_candidates_name += 1
                continue
            if funnel is not None:
                funnel.flavor_variant_candidates += 1
        if evaluation is None:
            evaluation = _evaluate(left_record, right_record)
        if not evaluation["conflicts"]:
            if funnel is not None:
                funnel.dropped_candidates_no_conflict += 1
            continue
        if funnel is not None:
            _census(funnel.conflict_dimension_census, evaluation["conflicts"])
            funnel.passed_candidates += 1
        score = float(row.candidate_similarity)
        for pair in (
            (left_row, gtin_to_canon_idx[right_gtin]),
            (right_row, gtin_to_canon_idx[left_gtin]),
        ):
            pair = (int(pair[0]), int(pair[1]))
            if pair in existing_keys:
                if funnel is not None:
                    funnel.dropped_pairs_already_in_baseline += 1
                continue
            existing_keys.add(pair)
            found.append((pair[0], pair[1], score))
            if len(found) >= int(n_target):
                break
        if len(found) >= int(n_target):
            break
    if funnel is not None:
        funnel.emitted_pairs = int(len(found))
    if not found:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)
    return (
        np.asarray([(a, b) for a, b, _ in found], dtype=int),
        np.asarray([score for _, _, score in found], dtype=float),
    )


def mine_targeted_attribute_negatives_with_funnel(
    df: pd.DataFrame,
    gates: pd.DataFrame,
    canonical_records: pd.DataFrame,
    gtin_to_row: dict[str, int],
    gtin_to_canon_idx: dict[str, int],
    *,
    existing: np.ndarray | None = None,
    n_target: int,
    min_similarity: float,
    volume_relative_tolerance: float = 0.0,
    volume_absolute_tolerance_ml: float = 0.0,
    canonical_map: dict[str, str] | None = None,
    name_match: str = "flavor_variant",
) -> tuple[np.ndarray, np.ndarray, MiningFunnel]:
    """Same call as :func:`mine_targeted_attribute_negatives`, returning its funnel.

    This is the seam an out-of-tree caller (the consolidated-trace owner) uses
    to record the lane's attrition without re-implementing any filter. The
    signature is pinned to the miner's by
    ``tests/test_mining_hypotheses.py::test_funnel_wrapper_signature_matches_miner``.
    """
    funnel = MiningFunnel()
    pairs, scores = mine_targeted_attribute_negatives(
        df,
        gates,
        canonical_records,
        gtin_to_row,
        gtin_to_canon_idx,
        existing=existing,
        n_target=n_target,
        min_similarity=min_similarity,
        volume_relative_tolerance=volume_relative_tolerance,
        volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
        canonical_map=canonical_map,
        name_match=name_match,
        funnel=funnel,
    )
    return pairs, scores, funnel


def mine_attribute_conflict_negatives(
    df: pd.DataFrame,
    payload: list[str],
    row_barcodes: np.ndarray,
    emb: np.ndarray,
    *,
    existing: np.ndarray | None = None,
    n_target: int = 0,
    cosine_lo: float = 0.45,
    cosine_hi: float = 0.95,
    volume_relative_tolerance: float = 0.0,
    volume_absolute_tolerance_ml: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mine additional same-brand/category pairs that disagree on attributes.

    The gate hard-no population remains the baseline. This supplemental lane
    searches representative SKU rows against other canonical targets sharing
    the canonical brand/type, requires a volume, pack, or flavor conflict, and
    keeps only high-cosine pairs. It therefore expands coverage of the exact
    attribute-conflict population without relabeling the original 6,051 rows.

    ``volume_*_tolerance`` must be the training gate's own tolerance; see the
    SSOT note on :func:`mine_targeted_attribute_negatives`.
    """
    if n_target <= 0 or len(df) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    from core.common import F

    canon_path = F["canonical_records"]
    canon = pd.read_csv(canon_path, dtype=str, keep_default_na=False)
    from core.attribute_conflicts import attribute_conflict_types, canonical_attribute_info

    canon_by_gtin = {}
    for record in canon.to_dict("records"):
        gtin = str(record["gtin"])
        canon_by_gtin[gtin] = {
            "brand": str(record.get("mode_brand", "")).strip().lower(),
            "type": str(record.get("mode_type", "")).strip().lower(),
            "canonical": str(record.get("canonical", "")).strip(),
            **canonical_attribute_info(record),
        }
    # Canonicals occupy the first post-data block. Masked copies are appended
    # later with the same barcode, so first-wins is the canonical-only rule.
    canon_idx: dict[str, int] = {}
    for i in range(len(df), len(row_barcodes)):
        gtin = str(row_barcodes[i])
        if gtin in canon_by_gtin:
            canon_idx.setdefault(gtin, i)
    if not canon_idx:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for gtin, info in canon_by_gtin.items():
        key = (info["brand"], info["type"])
        if key[0] and key[1]:
            groups[key].append(gtin)

    # Longest representative row per barcode, stable on original row order.
    reps: dict[str, int] = {}
    for i, gtin in enumerate(row_barcodes[: len(df)]):
        gtin = str(gtin)
        if gtin not in canon_by_gtin:
            continue
        title_len = len(str(df.iloc[i].get("title", "")))
        old = reps.get(gtin)
        if old is None or title_len > len(str(df.iloc[old].get("title", ""))):
            reps[gtin] = i

    product_names = {
        gtin: normalized_product_name(
            df.iloc[row].get("title", ""), canon_by_gtin[gtin]["brand"]
        )
        for gtin, row in reps.items()
    }

    existing_keys = {
        (int(a), int(b)) for a, b in (existing if existing is not None else [])
    }
    found: list[tuple[int, int, float]] = []
    for source_gtin, source_row in reps.items():
        source = canon_by_gtin[source_gtin]
        candidates = groups.get((source["brand"], source["type"]), [])
        for target_gtin in candidates:
            if (
                target_gtin == source_gtin
                or target_gtin not in canon_idx
                or (
                    source["canonical"]
                    and source["canonical"] == canon_by_gtin[target_gtin]["canonical"]
                )
            ):
                continue
            target = canon_by_gtin[target_gtin]
            if (
                not product_names.get(source_gtin)
                or product_names.get(source_gtin) != product_names.get(target_gtin)
            ):
                continue
            if not attribute_conflict_types(
                source,
                target,
                volume_relative_tolerance=float(volume_relative_tolerance),
                volume_absolute_tolerance_ml=float(volume_absolute_tolerance_ml),
            ):
                continue
            target_row = canon_idx[target_gtin]
            score = float(np.dot(emb[source_row], emb[target_row]))
            if not float(cosine_lo) < score <= float(cosine_hi):
                continue
            pair = (int(source_row), int(target_row))
            if pair in existing_keys:
                continue
            existing_keys.add(pair)
            found.append((pair[0], pair[1], score))

    found.sort(key=lambda item: (-item[2], item[0], item[1]))
    found = found[: int(n_target)]
    if not found:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)
    return (
        np.asarray([(a, b) for a, b, _ in found], dtype=int),
        np.asarray([s for _, _, s in found], dtype=float),
    )


def conflicting_barcode_pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    """Row-index pairs with the same title but conflicting barcodes.

    These are label errors (same product, conflicting barcode): a pair like this
    must never enter the negative pool, or training teaches the model to push
    apart titles that are actually the same product.

    Keyed on ``title`` ALONE (not title+brand): brand is a noisy field — the
    same product can carry different brand spellings across retailers, and the
    miner's candidates are different-brand pairs by design, so a title+brand
    key would make every excluded pair unreachable at the filter (same-brand
    candidates are skipped before the exclusion check — that asymmetry was a
    live bug; the guard never fired). Pairs are stored order-normalized
    ``(min, max)`` so a non-monotonic index upstream can not silently break
    the membership lookup.
    """
    barcodes = df["barcode"].fillna("").astype(str)
    pairs: set[tuple[int, int]] = set()
    # fast path: only titles with >1 DISTINCT non-empty barcode can produce a
    # conflicting pair; everything else is skipped without a per-group Python
    # loop (the old full groupby walked every one of ~57k title groups with a
    # pandas .loc per group — 37s; this pre-filter leaves only the handful of
    # genuinely conflicted titles)
    bc = pd.DataFrame(
        {"title": df["title"], "barcode": barcodes, "row": np.arange(len(df))}
    )
    bc = bc[bc["title"].notna() & (bc["barcode"].str.len() > 0)]
    nuniq = bc.groupby("title")["barcode"].nunique()
    conflicted_titles = set(nuniq[nuniq > 1].index)
    if not conflicted_titles:
        return pairs
    cbc = bc[bc["title"].isin(conflicted_titles)]
    for title, g in cbc.groupby("title", sort=False):
        idx = sorted(int(i) for i in g["row"])
        bc_by_row = dict(zip(g["row"], g["barcode"], strict=True))
        for a, b in combinations(idx, 2):
            if bc_by_row[a] != bc_by_row[b]:
                pairs.add((a, b))
    return pairs


def pairs_in_set(
    pairs: np.ndarray, row_barcodes: np.ndarray, bc_set: set[str]
) -> np.ndarray:
    """Boolean mask over pairs whose BOTH endpoints' barcode is in bc_set.

    Held-out pair filtering: a pair is only in the split if both rows belong to
    it, so no train/test entity leaks across the boundary.
    """
    members = list(bc_set)
    return np.isin(row_barcodes[pairs[:, 0]], members) & np.isin(
        row_barcodes[pairs[:, 1]], members
    )


def build_triplets(
    train_pos: np.ndarray,
    hard_train: np.ndarray,
    payload: list[str],
    *,
    seed: int | None = None,
    max_triples: int | None = None,
) -> list:
    """Build (anchor, positive, hard-negative) triples for TripletLoss.

    Each hard-negative partner is drawn from the anchor's mined hard negatives
    (falling back to the positive partner's). Capped at max_triples so the
    fine-tune stays tractable.

    CONFIG SSOT (owner directive: read from configs, not declared): seed
    defaults to lib.common.SEED (root config/paths.yaml seed) and max_triples
    defaults to training.max_triples (config/training.yaml) when None;
    explicit values still win (training.py passes per-fold seed offsets).
    No inline literals in this signature.
    """
    from core.common import SEED, runtime

    if seed is None:
        seed = SEED
    if max_triples is None:
        max_triples = int(runtime("max_triples"))

    from sentence_transformers import InputExample

    hn_map: dict[int, list[int]] = defaultdict(list)
    for a, b in hard_train:
        # Index BOTH directions: mined pairs are unordered (a<b at build), so a
        # one-directional map silently discards any hard negative whose anchor
        # happens to be the second element of the mined pair.
        hn_map[int(a)].append(int(b))
        hn_map[int(b)].append(int(a))

    rng = np.random.default_rng(seed)
    triples: list[InputExample] = []
    for a, b in train_pos:
        partners = hn_map.get(int(a)) or hn_map.get(int(b))
        if not partners:
            continue
        c = int(partners[rng.integers(len(partners))])
        triples.append(InputExample(texts=[payload[a], payload[b], payload[c]]))
        if len(triples) >= max_triples:
            break
    return triples


def mine_hard_negatives(
    df: pd.DataFrame,
    emb: np.ndarray,
    *,
    seed: int | None = None,
    n_target: int | None = None,
    cosine_lo: float | None = None,
    cosine_hi: float | None = None,
    exclude_conflicting: bool | None = None,
    k: int | None = None,
    max_per_canonical: int | None = None,
    max_per_brand: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mine hard negatives: cross-barcode, different-brand, same-macro, mid-cosine.

    Uses cosine ANN within each macro-category block, then filters to the
    confusion band (cosine_lo..cosine_hi) with a DIFFERENT brand (the signature
    of the champion's false positives), a different non-empty barcode, and no
    conflicting-barcode label error. Returns (pairs, cosine) as an (N,2) int
    array and an (N,) float array, hardest-first.

    CONFIG SSOT: every miner parameter resolves from config/training.yaml when
    omitted — target, band, k, chunk size, and conflict exclusion are all
    under mining.ann. Explicit values still win. Endpoint diversity caps keep
    one canonical or brand cluster from consuming the entire target.
    """
    from core.common import SEED, category_macros, training_cfg

    if seed is None:
        seed = SEED
    ann_cfg = training_cfg().mining.ann
    if n_target is None:
        n_target = int(ann_cfg.target)
    if k is None:
        k = int(ann_cfg.k)
    if exclude_conflicting is None:
        exclude_conflicting = bool(ann_cfg.exclude_conflicting)
    if max_per_canonical is None:
        max_per_canonical = int(ann_cfg.max_per_canonical)
    if max_per_brand is None:
        max_per_brand = int(ann_cfg.max_per_brand)
    if cosine_lo is None or cosine_hi is None:
        lo, hi = ann_cfg.band.split("-")
        cosine_lo = float(lo) if cosine_lo is None else cosine_lo
        cosine_hi = float(hi) if cosine_hi is None else cosine_hi
    barcodes = df["barcode"].fillna("").astype(str).to_numpy()
    brands = df["brand"].fillna("").astype(str).to_numpy()
    # MACRO_MAP moved to config (SSOT): config/paths.yaml category_macros,
    # read via lib.common.category_macros() — no module-level copy.
    macro_map = category_macros()
    macro = df["category"].fillna("").map(lambda c: macro_map.get(c, "?")).to_numpy()
    # Barcode trust (owner ruling, see src/core/gtin.py): a checksum-fail barcode
    # cannot certify "known different" any more than a missing one can —
    # exclude from the negative population exactly like empty barcodes.
    from core.gtin import barcode_validity

    bc_valid = barcode_validity(df["barcode"].fillna("").astype(str)).to_numpy()
    excluded = conflicting_barcode_pairs(df) if exclude_conflicting else set()

    found: list[tuple[int, int, float]] = []
    n_band_seen = 0  # pairs reaching all filters except exclusion (audit denominator)
    n_excluded_in_band = 0  # pairs the label-error guard DROPPED (audit trail)
    for m in np.unique(macro):
        idx = np.flatnonzero(macro == m)
        if len(idx) < 2:
            continue
        # VECTORIALIZED neighbor search: one BLAS matmul per macro block replaces
        # sklearn's kneighbors (5x faster, identical top-k neighbor sets —
        # verified on the deduped corpus: block CONCENTRATES 7,881 rows, top-5
        # neighbor identities match exactly). Emb rows are L2-normalized so the
        # dot product IS cosine similarity.
        #
        # CHUNKED over block rows (OOM fix, owner audit 2026-09-07): the old
        # full-grid version materialized N x N arrays (sims + meshgrid + cand
        # + topk mask ~= 11-16 GB for JUICE's N=18,251) and the kernel OOM-
        # killed the full-corpus run (rc=137 after the zero-shot encode).
        # Chunking is candidate-IDENTICAL: np.argpartition(axis=1) is
        # row-independent, so per-row top-k over a (chunk, N) slice equals
        # the full matrix's, and the a<b order filter then selects the same
        # (i, j) pairs the grid's top-k membership mask did. Peak memory per
        # chunk = chunk x N float64 (~300 MB at chunk=2048, N=18k).
        k_eff = min(k, len(idx))
        n = len(idx)
        # rows of this block, reindexed 0..n-1 (local), global = idx[local]
        bc_blk = barcodes[idx]
        br_blk = brands[idx]
        bcv_blk = bc_valid[idx]
        chunk_size = int(ann_cfg.chunk_size)
        for c0 in range(0, n, chunk_size):
            c1 = min(c0 + chunk_size, n)
            sims_chunk = emb[idx[c0:c1]] @ emb[idx].T  # (c, n) cosine
            top = np.argpartition(-sims_chunk, kth=k_eff - 1, axis=1)[:, :k_eff]
            # candidate pairs from top-k membership: (local_i, local_j)
            li = np.repeat(np.arange(c0, c1), k_eff)
            lj = top.ravel()
            # same order filter as the full grid: a < b in LOCAL indices
            keep = li < lj
            # flat candidate scores over the SAME (li, lj) arrays — filtered
            # in lockstep with keep below so index spaces never mix
            s_flat = sims_chunk[li - c0, lj]
            keep &= (s_flat >= cosine_lo) & (s_flat <= cosine_hi)  # band
            bc_a = bc_blk[li[keep]]
            bc_b = bc_blk[lj[keep]]
            br_a = br_blk[li[keep]]
            br_b = br_blk[lj[keep]]
            # real, distinct, and BOTH trusted (GS1 checksum) — an invalid
            # barcode has unknown identity, not "known different"
            valid = (
                (bc_a != "")
                & (bc_b != "")
                & (bc_a != bc_b)
                & (br_a != br_b)  # different brand
                & bcv_blk[li[keep]]
                & bcv_blk[lj[keep]]
            )
            sel_local = np.flatnonzero(keep)[valid]
            li_sel = li[keep][valid]
            lj_sel = lj[keep][valid]
            sels = s_flat[keep][valid]
            ga = idx[li_sel]
            gb = idx[lj_sel]
            n_band_seen += len(sel_local)
            if excluded:
                # only check membership for pairs; keep the loop off the hot path
                # unless exclusions exist for this block's rows
                ex_rows = excluded  # set of (min,max) global pairs
                for a_, b_, s_ in zip(ga.tolist(), gb.tolist(), sels.tolist(), strict=True):
                    if (min(a_, b_), max(a_, b_)) in ex_rows:
                        n_excluded_in_band += 1
                    else:
                        found.append((a_, b_, s_))
            else:
                for a_, b_, s_ in zip(ga.tolist(), gb.tolist(), sels.tolist(), strict=True):
                    found.append((a_, b_, s_))

    if exclude_conflicting:
        # the exclusion is auditable, never hidden: the count is part of the
        # return so callers can report (and tests can pin) how many candidate
        # pairs the label-error guard dropped.
        print(
            f"mining audit: {n_band_seen:,} candidate pairs in band, "
            f"{n_excluded_in_band:,} excluded as conflicting-barcode label errors"
        )

    found.sort(key=lambda t: -t[2])  # hardest (highest cosine) first
    seen: set[tuple[int, int]] = set()
    canonical_counts: dict[str, int] = defaultdict(int)
    brand_counts: dict[str, int] = defaultdict(int)
    pairs_out: list[tuple[int, int]] = []
    cos_out: list[float] = []
    for a, b, s in found:
        if (a, b) in seen:
            continue
        endpoint_barcodes = (str(barcodes[a]), str(barcodes[b]))
        endpoint_brands = (
            str(brands[a]).strip().lower(),
            str(brands[b]).strip().lower(),
        )
        if any(
            value and canonical_counts[value] >= int(max_per_canonical)
            for value in endpoint_barcodes
        ):
            continue
        if any(
            value and brand_counts[value] >= int(max_per_brand)
            for value in endpoint_brands
        ):
            continue
        seen.add((a, b))
        pairs_out.append((a, b))
        cos_out.append(s)
        for value in endpoint_barcodes:
            if value:
                canonical_counts[value] += 1
        for value in endpoint_brands:
            if value:
                brand_counts[value] += 1
        if len(pairs_out) >= n_target:
            break

    if not pairs_out:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)
    return np.asarray(pairs_out, dtype=int), np.asarray(cos_out, dtype=float)


def calibrated_ann_band(
    scores: np.ndarray,
    configured_band: tuple[float, float],
    score_quantiles: tuple[float, float],
    band_mode: str,
) -> tuple[float, float, dict[str, object]]:
    """Select the ANN band using the explicitly configured mode.

    There is intentionally no implicit fallback. ``fixed`` uses the literal
    configured band, ``adaptive_quantile`` uses the configured score
    quantiles, and ``intersection`` uses only their overlap (which may be
    empty). The mode is required from the config SSOT by every caller.
    """
    scores = np.asarray(scores, dtype=float)
    lo, hi = (float(x) for x in configured_band)
    qlo, qhi = (float(x) for x in score_quantiles)
    if band_mode not in {"fixed", "adaptive_quantile", "intersection"}:
        raise ValueError(
            "mining.ann.band_mode must be one of fixed, adaptive_quantile, "
            f"intersection; got {band_mode!r}"
        )
    if scores.size == 0:
        return lo, hi, {
            "candidate_count": 0.0,
            "candidate_min": float("nan"),
            "candidate_max": float("nan"),
            "candidate_median": float("nan"),
            "band_overlap_pct": 0.0,
            "band_lo": lo,
            "band_hi": hi,
            "band_mode": band_mode,
        }
    q_values = np.quantile(scores, [qlo, qhi])
    overlap = (scores >= lo) & (scores <= hi)
    if band_mode == "fixed":
        band_lo, band_hi = lo, hi
    elif band_mode == "adaptive_quantile":
        band_lo, band_hi = (float(q_values[0]), float(q_values[1]))
    else:
        band_lo = max(lo, float(q_values[0]))
        band_hi = min(hi, float(q_values[1]))
    return band_lo, band_hi, {
        "candidate_count": float(scores.size),
        "candidate_min": float(np.min(scores)),
        "candidate_max": float(np.max(scores)),
        "candidate_median": float(np.median(scores)),
        "band_overlap_pct": float(np.mean(overlap)),
        "band_lo": float(band_lo),
        "band_hi": float(band_hi),
        "band_mode": band_mode,
    }
