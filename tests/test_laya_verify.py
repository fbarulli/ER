"""scripts/laya_verify.py — score a checkpoint on the holdout (no laya needed)."""
from __future__ import annotations

import csv

from scripts.laya_verify import (
    holdout_states,
    predictions_report,
    write_predictions,
)

CATALOG = [
    {"sku_id": "s1", "gtin": "11111111111111", "attribute": "Brand: Acme; Flavour: cola"},
    {"sku_id": "s2", "gtin": "22222222222222", "attribute": "Brand: Acme; Flavour: cola"},
    {"sku_id": "s3", "gtin": "33333333333333", "attribute": "Brand: Acme; Flavour: lemon"},
    {"sku_id": "s4", "gtin": "44444444444444", "attribute": "Brand: Other; Flavour: lemon"},
    {"sku_id": "s5", "gtin": "55555555555555", "attribute": "Brand: Third; Flavour: grape"},
    {"sku_id": "s6", "gtin": "66666666666666", "attribute": "Brand: Fourth; Flavour: grape"},
]


def _rows():
    return [
        {"gtin1": "11111111111111", "gtin2": "22222222222222", "label": "1",
         "component": "c1", "stratum": "p0_disjoint"},
        {"gtin1": "33333333333333", "gtin2": "44444444444444", "label": "0",
         "component": "c2", "stratum": "p0_disjoint"},
        {"gtin1": "55555555555555", "gtin2": "66666666666666", "label": "",
         "component": "c3", "stratum": "gate_proceed"},
    ]


def test_holdout_states_scores_labelled_pairs_and_skips_gate():
    scored, skipped = holdout_states(_rows(), CATALOG)
    assert len(scored) == 2 and len(skipped) == 1
    assert skipped[0]["stratum"] == "gate_proceed"
    # the composed state carries both sides' sliced fields (reused composer)
    assert all(row["state"] for row in scored)
    assert all(row["label"] in ("0", "1") for row in scored)


def test_holdout_states_skips_unknown_gtins():
    rows = [{"gtin1": "11111111111111", "gtin2": "99999999999999",
             "label": "1", "component": "c", "stratum": "p0_disjoint"}]
    scored, skipped = holdout_states(rows, CATALOG)
    assert scored == []
    assert skipped[0]["reason"] == "endpoint absent from catalog"


def test_predictions_report_stratifies_with_clustered_cis():
    scored, _ = holdout_states(_rows(), CATALOG)
    report = predictions_report(scored, [0.9, 0.1], threshold=0.5,
                                n_boot=50, seed=1729)
    overall = report["overall"]["metrics"]
    assert overall["n"] == 2 and overall["accuracy"] == 1.0
    assert "p0_disjoint" in report["by_stratum"]
    assert "f1" in report["overall"]["cis"]


def test_write_predictions_is_the_compare_contract(tmp_path):
    scored, _ = holdout_states(_rows(), CATALOG)
    out = tmp_path / "preds.csv"
    write_predictions(scored, [0.9, 0.1], out)
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["gtin1", "gtin2", "score"]
    assert [r["score"] for r in rows] == ["0.9", "0.1"]
