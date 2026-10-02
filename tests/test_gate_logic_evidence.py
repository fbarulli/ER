"""Actual fired gates and independent diagnostics must remain distinct."""
from types import SimpleNamespace

import pytest

from scripts.evaluate_gate_logic import canonical_quality, evaluate_sample
from core.common import training_cfg


def record(gtin, volume):
    return {"gtin": gtin, "canonical": "sample", "volume_set": {volume},
            "pack_set": set(), "volume_confidence": 1., "pack_confidence": 0.,
            "volume_consistency": 1., "pack_consistency": 1.,
            "attribute_consistency_flags": set(), "source_rows": "[]"}


@pytest.mark.parametrize("committed_decision", ["hard_no", "proceed"])
def test_actual_gate_not_inferred_from_independent_engine(monkeypatch, committed_decision):
    from core.attribute_decision import AttributeDecisionEngine

    diagnostic = SimpleNamespace(conflicts=["flavour"], agreements=[], inconclusive=[],
                                 as_dict=lambda: {"flavour": {"result": "CONFLICT"}})
    monkeypatch.setattr(AttributeDecisionEngine, "evaluate", lambda *args, **kwargs: diagnostic)
    reason = training_cfg().gate.reasons.pack_blocker
    pair = {"gtin1": "001", "gtin2": "002", "similarity": .9,
            "gate_decision": committed_decision, "gate_reason": reason}
    actual = evaluate_sample(pair, {"001": record("001", 500), "002": record("002", 1000)},
                             listing_limit=1, endpoints={})
    assert actual["current_gate"] == {"decision": "hard_no", "reason": reason}
    assert actual["actual_fired_stage"] == "pack_blocker"
    assert actual["independent_attribute_diagnostics"]["conflicts"] == ["flavour"]
    assert ("current_gate_differs_from_committed" in actual["inspection_flags"]) == (committed_decision != "hard_no")
    assert actual["left"]["source_listing_count"] == 0
    assert "left_original_source_rows_missing" in actual["inspection_flags"]


def test_quality_distinguishes_missing_and_low_confidence_evidence():
    missing = record("001", 500)
    missing.update(volume_set=set(), volume_confidence=0.)
    observed = record("002", 500)
    observed.update(volume_confidence=0., volume_consistency=0.,
                    attribute_consistency_flags={"volume_sources_disagree"})
    quality = canonical_quality({"001": missing, "002": observed})
    assert quality["fields"]["volume"]["low_confidence_without_observed_evidence"] == 1
    assert quality["fields"]["volume"]["low_confidence_with_observed_evidence"] == 1
    assert quality["fields"]["volume"]["below_consistency_threshold"] == 1
    assert quality["source_disagreement_flag_frequencies"] == {"volume_sources_disagree": 1}
