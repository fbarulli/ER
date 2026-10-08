"""scripts/laya_compare.py — score models on the holdout with clustered CIs."""
from __future__ import annotations

from scripts.laya_compare import _pair_key, run_comparison


def _holdout():
    return [
        {"source": "final_validation", "stratum": "p0_disjoint", "label": "1",
         "component": "c1", "gtin1": "11111111111111", "gtin2": "22222222222222"},
        {"source": "final_validation", "stratum": "p0_disjoint", "label": "0",
         "component": "c2", "gtin1": "33333333333333", "gtin2": "44444444444444"},
        {"source": "listing_pairs", "stratum": "real_listing", "label": "1",
         "component": "c3", "gtin1": "55555555555555", "gtin2": "66666666666666"},
        {"source": "gate_results", "stratum": "gate_proceed", "label": "",
         "component": "c4", "gtin1": "77777777777777", "gtin2": "88888888888888"},
    ]


def test_pair_key_is_order_insensitive_and_normalized():
    assert _pair_key("11111111111111", "22222222222222") == \
        _pair_key("22222222222222", "11111111111111")
    assert _pair_key("11111111111111.0", "22222222222222") == \
        _pair_key("11111111111111", "22222222222222")


def test_run_comparison_scores_labelled_rows_and_summarizes_gate():
    holdout = _holdout()
    perfect = {
        _pair_key("11111111111111", "22222222222222"): 0.9,
        _pair_key("33333333333333", "44444444444444"): 0.1,
        _pair_key("55555555555555", "66666666666666"): 0.8,
        _pair_key("77777777777777", "88888888888888"): 0.6,
    }
    report = run_comparison(holdout, {"perfect": perfect}, n_boot=25)
    model = report["models"]["perfect"]
    # only the 3 labelled rows score; the gate row is label-less
    assert model["matched_labelled"] == 3
    assert model["unmatched_labelled"] == 0
    overall = model["truth"]["overall"]["metrics"]
    assert overall["accuracy"] == 1.0
    assert overall["pr_auc"] == 1.0
    assert set(model["truth"]["by_stratum"]) == {"p0_disjoint", "real_listing"}
    # gate difficulty is reported as a score summary, not truth
    assert model["gate_score_summary"]["gate_proceed"]["n"] == 1
    assert model["gate_score_summary"]["gate_proceed"]["above_threshold"] == 1


def test_run_comparison_counts_unmatched_predictions():
    holdout = _holdout()
    partial = {_pair_key("11111111111111", "22222222222222"): 0.9}
    model = run_comparison(holdout, {"partial": partial}, n_boot=10)[
        "models"]["partial"]
    assert model["matched_labelled"] == 1
    assert model["unmatched_labelled"] == 2
