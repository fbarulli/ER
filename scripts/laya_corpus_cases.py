"""scripts/laya_corpus_cases.py — render one corpus case from source truth.

A corpus case is the laya contract keys PLUS the traceability tags
(`difficulty_slice`/`gate_reason`/`attribute`). Every label is read off the
composed sides / attributes the caller hands in; a question the schema does
not declare is simply not emitted (never invented).

`CorpusCaseRenderer` owns that rendering: single/pair records, the schema-gated
expected-label dict, the top-level tags, the `better_match` pairwise cases.
"""
from __future__ import annotations

from scripts.laya_corpus_composer import PAIR_FIELDS, compose_side
from scripts.laya_corpus_rules import PackageStateRules, PairLabelRules


class CorpusCaseRenderer:
    """Renders corpus records and their expected labels from source truth."""

    @staticmethod
    def _record(state: str, questions: dict, expected: dict, *,
                difficulty_slice: str, gate_reason: str = "",
                attribute: str = "none") -> dict:
        """One corpus case: the laya contract keys PLUS the traceability tags.

        `difficulty_slice`/`gate_reason`/`attribute` ride at the top level: the
        laya trainer (`items_from_rows`) reads only `state`/`questions`/`gold`/
        `expected`, so the extra keys never affect training, and the eval
        report reads them back to slice accuracy/ECE without re-deriving
        membership.
        """
        return {"state": state, "questions": questions, "expected": expected,
                "difficulty_slice": difficulty_slice,
                "gate_reason": gate_reason, "attribute": attribute}

    @staticmethod
    def _pair_expected(questions: dict, side_one: dict[str, str],
                       side_two: dict[str, str], *,
                       attr_one: str | None = None,
                       attr_two: str | None = None,
                       identity: str | None = None,
                       brand_one: str | None = None,
                       brand_two: str | None = None,
                       counterfactual: str | None = None,
                       gate_verdict: str | None = None,
                       gate_reason: str | None = None) -> dict:
        """Every pair label this row's sources support, gated by the schema.

        A label is emitted only when the caller's `questions` dict declares the
        qid (the hermetic fixtures with a 3-question schema stay byte-identical)
        AND the source supplies the truth. All field labels come off the SAME
        composed sides the state was rendered from.
        """
        expected: dict[str, str] = {}

        def put(qid: str, label: str | None) -> None:
            if label is not None and qid in questions:
                expected[qid] = label

        put("identity_claim", identity)
        put("counterfactual", counterfactual)
        for field in PAIR_FIELDS:
            put(f"field_same:{field}",
                PairLabelRules._field_same_label(side_one, side_two, field))
        if attr_one is not None and attr_two is not None:
            put("pack_volume_equal",
                PackageStateRules.pack_volume_equal(attr_one, attr_two))
            put("pack_format_equivalent",
                PackageStateRules.pack_format_equivalent(attr_one, attr_two))
        put("evidence_sufficient",
            PairLabelRules.evidence_sufficient(side_one, side_two))
        put("same_brand_only",
            PairLabelRules.same_brand_only(brand_one, brand_two, identity))
        put("gate_verdict", gate_verdict)
        put("gate_reason", gate_reason)
        return expected

    @staticmethod
    def _pair_meta(side_one: dict[str, str], side_two: dict[str, str], *,
                   gate_reason: str = "") -> dict:
        return {"difficulty_slice":
                PairLabelRules._difficulty_slice(side_one, side_two),
                "gate_reason": gate_reason,
                "attribute":
                PairLabelRules._primary_attribute(side_one, side_two)}

    @staticmethod
    def _single_meta(attribute: str) -> dict:
        """A single-state row's tags: no pair slice, the first measured field."""
        side = compose_side(attribute)
        for field in PAIR_FIELDS:
            if side.get(field):
                return {"difficulty_slice": "single_state", "gate_reason": "",
                        "attribute": field}
        return {"difficulty_slice": "single_state", "gate_reason": "",
                "attribute": "none"}

    @staticmethod
    def _render_triple_side(side: dict[str, str]) -> str:
        """One listing's six-field literals for the better_match state."""
        return "[" + ", ".join(
            f"{field}:{side.get(field) or '-'}" for field in PAIR_FIELDS) + "]"

    @staticmethod
    def better_match_records(pairs: list[dict], by_sku: dict[str, dict],
                             questions: dict) -> list[dict]:
        """Pairwise `better_match` (choice(2)) cases from the GTIN truth.

        For every confirmed-different listing pair (A, B) that also has a
        confirmed-same partner C for A, the state carries the anchor A and two
        candidates (B, C); the answer is the candidate whose GTIN equals A's.
        The two candidates are placed deterministically by GTIN (the smaller
        GTIN is `candidate_1`) so position is not a free signal. Emitted only
        when the schema declares `better_match`.
        """
        if "better_match" not in questions:
            return []
        same_by_sku: dict[str, dict[str, dict]] = {}
        for pair in pairs:
            if int(pair["label"]) != 1:
                continue
            for anchor, partner in ((pair["sku_id1"], pair["sku_id2"]),
                                    (pair["sku_id2"], pair["sku_id1"])):
                same_by_sku.setdefault(anchor, {})[partner] = pair
        records: list[dict] = []
        seen: set[str] = set()
        for pair in pairs:
            if int(pair["label"]) != 0:
                continue
            anchor = pair["sku_id1"]
            different = pair["sku_id2"]
            for same in sorted(same_by_sku.get(anchor, {})):
                if same == different:
                    continue
                gtin_different = by_sku[different]["gtin"]
                gtin_same = by_sku[same]["gtin"]
                # deterministic candidate order by GTIN; GTIN truth decides
                first_is_same = gtin_same <= gtin_different
                cand_one = same if first_is_same else different
                cand_two = different if first_is_same else same
                side_anchor = compose_side(by_sku[anchor]["attribute"])
                side_one = compose_side(by_sku[cand_one]["attribute"])
                side_two = compose_side(by_sku[cand_two]["attribute"])
                state = (f"anchor: {CorpusCaseRenderer._render_triple_side(side_anchor)}; "
                         f"candidate_1: {CorpusCaseRenderer._render_triple_side(side_one)}; "
                         f"candidate_2: {CorpusCaseRenderer._render_triple_side(side_two)}")
                if state in seen:
                    continue
                seen.add(state)
                records.append(CorpusCaseRenderer._record(
                    state, questions,
                    {"better_match": "candidate_1" if first_is_same
                     else "candidate_2"},
                    difficulty_slice="pairwise",
                    attribute=PairLabelRules._primary_attribute(side_one, side_two)))
        return records
