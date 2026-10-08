"""scripts/laya_holdout.py — component-disjoint, difficulty-tagged holdout."""
from __future__ import annotations

import csv
from pathlib import Path

from scripts.laya_holdout import build_holdout


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _fixtures(root: Path) -> dict[str, Path]:
    catalog = _write(root / "catalog.csv", ["sku_id", "gtin", "attribute"], [
        ["s1", "11111111111111", "Volume: 100"],
        ["s2", "22222222222222", "Volume: 100"],
        ["s3", "33333333333333", "Volume: 200"],
    ])
    listing = _write(root / "listing.csv",
                     ["sku_id1", "sku_id2", "label", "split"], [
                         ["s1", "s2", "1", "test"],   # links s1+s2 -> one component
                         ["s1", "s3", "0", "test"],
                     ])
    p0 = _write(root / "p0.csv",
                ["gtin1", "gtin2", "true_label", "endpoint_in_train",
                 "component_id"], [
                    ["11111111111111", "99999999999999", "0", "False", "c-p0"],
                    ["11111111111111", "88888888888888", "1", "True", "c-p0b"],
                ])
    gate = _write(root / "gate.csv",
                  ["gtin1", "gtin2", "gate_decision", "gate_reason", "similarity"], [
                      ["11111111111111", "22222222222222", "hard_no", "Pack blocker", "0.1"],
                      ["11111111111111", "33333333333333", "proceed", "Known critical attributes compatible", "0.8"],
                      ["11111111111111", "44444444444444", "fallback", "Missing flavor evidence", "0.5"],
                  ])
    labeled = _write(root / "labeled.csv", ["gtin1", "gtin2", "true_label"], [])
    return {"listing_path": listing, "catalog_path": catalog, "p0_path": p0,
            "gate_path": gate, "labeled_path": labeled}


def test_holdout_is_component_disjoint_and_difficulty_tagged(tmp_path):
    rows, receipt = build_holdout(**_fixtures(tmp_path))
    by_source = receipt["by_source"]
    assert by_source["listing_pairs"] == 2
    assert by_source["final_validation"] == 2
    # hard_no is the easy pipeline-verified mass -> never in the truth holdout
    assert by_source["gate_results"] == 2

    strata = receipt["by_stratum"]
    assert strata["gate_proceed"] == 1 and strata["gate_fallback"] == 1
    assert strata["p0_disjoint"] == 1 and strata["p0_overlap"] == 1

    # the positive listing pair links its two endpoints into ONE component
    listing_rows = [r for r in rows if r["source"] == "listing_pairs"]
    positive = next(r for r in listing_rows if r["label"] == "1")
    a = next(r for r in listing_rows if r["gtin1"] == positive["gtin1"])
    assert positive["component"]
    assert a["component"] == positive["component"]

    # gate rows are label-less (agreement/difficulty only), never scored as truth
    gate_rows = [r for r in rows if r["source"] == "gate_results"]
    assert all(r["label"] == "" for r in gate_rows)
    assert all(r["label_source"] == "gate_verdict" for r in gate_rows)


def test_holdout_label_census_counts_only_real_labels(tmp_path):
    rows, receipt = build_holdout(**_fixtures(tmp_path))
    labelled = [r for r in rows if r["label"] in ("0", "1")]
    assert receipt["labelled_rows"] == len(labelled)
    assert receipt["positives"] + receipt["negatives"] == receipt["labelled_rows"]
