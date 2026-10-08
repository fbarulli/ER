"""Offline pins for scripts/laya_build_dataset.py's growth folds.

Owner order "laya is overfitting" (2026-10-08): the tiny fine-tune corpus
(train 2,735) overfits, so the builder now folds the pipeline's minted
masking + augmentation data in on top of the base sources. These tests are
hermetic (a tiny tmp bundle + tmp labeled pairs) and pin the three things
that matter:

  (a) a rerun is byte-identical (deterministic seed + no set-order leaks);
  (b) the masked-positive states and the counterfactual/twin augmentation
      pairs are present and correctly labelled;
  (c) the gate `fallback` quarantine stays out of the corpus.

The base hermetic builder tests in tests/test_laya_lane.py (no bundle, no
labeled pairs) must keep producing the pre-growth corpus — the growth
sources are opt-in.
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import pickle
from pathlib import Path


def _builder():
    path = (Path(__file__).resolve().parents[1]
            / "scripts/laya_build_dataset.py")
    spec = importlib.util.spec_from_file_location("laya_build_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _sources(tmp_path: Path) -> dict:
    """A miniature, fully deterministic growth fixture.

    Catalog rows carry the standardized `attribute` the composer parses;
    the tmp prepared bundle carries the minted masking/augmentation audits
    in the pipeline's shape (cleaned payload texts + audit indices).
    """
    catalog = tmp_path / "catalog.csv"
    _write_csv(catalog, ["sku_id", "retailer", "gtin", "attribute"], [
        ["S1", "r", "5000000001", "Volume: 500; Flavour: apple"],
        ["S2", "r", "5000000002", "Flavour: lime"],
        ["S3", "r", "5000000003", "Pack Type: Can; 24x330ml"],
        ["S4", "r", "5000000004", "Flavour: cola; Carbonization: still"],
    ])
    pairs = tmp_path / "pairs.csv"
    _write_csv(pairs, ["sku_id1", "sku_id2", "label", "split"], [
        ["S1", "S2", "1", "train"],
        ["S3", "S1", "0", "dev"],
    ])
    gate = tmp_path / "gate.csv"
    _write_csv(
        gate,
        ["gtin1", "gtin2", "canon1", "canon2", "gate_decision",
         "gate_reason", "similarity"], [
            ["5000000002", "5000000003", "c1", "c2", "hard_no",
             "Critical attribute mismatch: flavor", "0.90"],
            ["5000000003", "5000000004", "c1", "c2", "fallback",
             "unresolved", "0.50"],
        ])
    labeled = tmp_path / "labeled_pairs.csv"
    _write_csv(labeled, ["gtin1", "gtin2", "true_label"], [
        ["5000000001", "5000000004", "1"],
        ["5000000002", "5000000004", "0"],
    ])
    question = tmp_path / "laya.question.json"
    questions = {
        "attribute_alignment": {"type": "choice", "instructions": "verdict?",
                                "criteria": {"aligned": None}},
        "identity_claim": {"type": "noul", "instructions": "same item?"},
        "package_state": {"type": "noul", "instructions": "has pack?"},
    }
    question.write_text(json.dumps({"schema": "test", "questions": questions}),
                        encoding="utf-8")

    import pandas as pd

    df = pd.DataFrame({
        "gtin": ["5000000001", "5000000002", "5000000004"],
        "attribute": ["Volume: 500; Flavour: apple", "Flavour: lime",
                      "Flavour: cola; Carbonization: still"],
    })
    payload = [
        "apple juice volume_ml_500 flavor_apple",
        "lime soda flavor_lime carbonation_carbonated",
        "cola drink flavor_cola carbonation_still",
    ]
    mask_audit = [
        {  # prose masked, structured volume retained -> package_state true
            "population": "positive", "target_mode": "targeted",
            "anchor_payload_idx": 0, "pair_payload_idx": 1,
            "copy_payload_idx": 3, "gtin": "5000000001",
            "anchor_text": payload[0],
            "masked_text": "apple [MASK] volume_ml_500 flavor_apple",
        },
        {  # no volume/pack evidence on the anchor -> package_state false
            "population": "positive", "target_mode": "targeted",
            "anchor_payload_idx": 1, "pair_payload_idx": 0,
            "copy_payload_idx": 4, "gtin": "5000000002",
            "anchor_text": payload[1],
            "masked_text": "[MASK] soda [MASK] flavor_lime carbonation_carbonated",
        },
    ]
    hard_negative_mask_audit = [
        {  # counterfactual twin: copy flavour swapped vs the paired side
            "population": "hard_negative", "target_mode": "counterfactual",
            "fields_hit": ["flavor"], "anchor_payload_idx": 0,
            "pair_payload_idx": 2, "copy_payload_idx": 5,
            "donor_anchor_payload_idx": 0, "gtin": "5000000001",
            "anchor_text": payload[2],
            "masked_text": "cola drink flavor_apple carbonation_still",
        },
    ]
    bundle = tmp_path / "bundle.pkl.gz"
    with gzip.open(bundle, "wb") as handle:
        pickle.dump({
            "df": df, "payload": payload, "mask_audit": mask_audit,
            "hard_negative_mask_audit": hard_negative_mask_audit,
        }, handle)
    return {"catalog": catalog, "pairs": pairs, "gate": gate,
            "question": question, "bundle": bundle, "labeled": labeled,
            "questions": questions}


def _build(builder, sources, out: Path, **extra) -> dict:
    return builder.build(
        catalog_path=sources["catalog"], pairs_path=sources["pairs"],
        gate_path=sources["gate"], question_path=sources["question"],
        output_dir=out, seed=1729,
        bundle_path=sources["bundle"], labeled_pairs_path=sources["labeled"],
        **extra)


def test_growth_fold_is_deterministic_byte_for_byte(tmp_path):
    builder = _builder()
    sources = _sources(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    receipt_one = _build(builder, sources, first)
    receipt_two = _build(builder, sources, second)
    assert receipt_one["corpus_sha256"] == receipt_two["corpus_sha256"]
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl",
                 "unknown_pairs.csv", "receipt.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_masked_positive_state_variants_are_folded_in(tmp_path):
    builder = _builder()
    sources = _sources(tmp_path)
    out = tmp_path / "out"
    receipt = _build(builder, sources, out)
    counts = receipt["counts"]
    assert counts["mask_cases"] == 2
    assert counts["mask_package_state_true"] == 1
    assert counts["mask_package_state_false"] == 1

    cases = [
        json.loads(line)
        for key in ("train", "dev", "test")
        for line in (out / f"{key}.jsonl").read_text().splitlines()
        if "package_state" in json.loads(line)["expected"]
        and "[MASK]" in json.loads(line)["state"]
    ]
    by_state = {case["state"]: case["expected"]["package_state"]
                for case in cases}
    assert by_state["apple [MASK] volume_ml_500 flavor_apple"] == "true"
    assert by_state[
        "[MASK] soda [MASK] flavor_lime carbonation_carbonated"] == "false"


def test_counterfactual_augmentation_pairs_are_folded_in(tmp_path):
    builder = _builder()
    sources = _sources(tmp_path)
    out = tmp_path / "out"
    receipt = _build(builder, sources, out)
    counts = receipt["counts"]
    assert counts["aug_pairs"] >= 2
    assert counts["aug_pairs_positive"] >= 1
    assert counts["aug_pairs_negative"] >= 1
    # the tmp labeled pair (5000000001, 5000000004) is a new positive
    assert receipt["growth"]["labeled_pairs_added"] == 2

    states = {
        json.loads(line)["state"]: json.loads(line)["expected"]["identity_claim"]
        for key in ("train", "dev", "test")
        for line in (out / f"{key}.jsonl").read_text().splitlines()
        if "identity_claim" in json.loads(line)["expected"]
    }
    assert "true" in states.values() and "false" in states.values()
    # the counterfactual twin's swapped flavour rides the side-by-side state
    twin_state = (
        "volume: v1= v2=; pack: v1= v2=; package_type: v1= v2=; "
        "sweetener: v1= v2=; flavor: v1=[apple] v2=[cola]; "
        "carbonation: v1=[still] v2=[still]")
    assert states[twin_state] == "false"


def test_fallback_quarantine_never_enters_the_corpus(tmp_path):
    builder = _builder()
    sources = _sources(tmp_path)
    out = tmp_path / "out"
    _build(builder, sources, out)
    with (out / "unknown_pairs.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["gate_decision"] == "fallback"
    assert rows[0]["attribute_pairs"]
    body = "".join((out / f"{key}.jsonl").read_text()
                   for key in ("train", "dev", "test"))
    assert rows[0]["attribute_pairs"] not in body


def test_every_emitted_case_parses_with_valid_labels(tmp_path):
    builder = _builder()
    sources = _sources(tmp_path)
    out = tmp_path / "out"
    receipt = _build(builder, sources, out)
    questions = sources["questions"]
    seen = 0
    for key in ("train", "dev", "test"):
        lines = (out / f"{key}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == receipt["split_sizes"][key]
        for line in lines:
            record = json.loads(line)
            assert set(record) == {"state", "questions", "expected",
                                   "difficulty_slice", "gate_reason",
                                   "attribute"}
            assert isinstance(record["state"], str) and record["state"]
            assert record["questions"] == questions
            assert record["difficulty_slice"] in (
                "all_same", "one_diff", "multi_diff", "insufficient",
                "single_state", "pairwise")
            for qid, label in record["expected"].items():
                qtype = record["questions"][qid]["type"]
                if qtype == "noul":
                    assert label in ("true", "false")
                else:
                    assert label in record["questions"][qid]["criteria"]
            seen += 1
    assert seen == sum(receipt["split_sizes"].values())


def test_growth_sources_are_opt_in(tmp_path):
    """Without a bundle/labeled pairs the corpus is the base corpus again."""
    builder = _builder()
    sources = _sources(tmp_path)
    base = tmp_path / "base"
    grown = tmp_path / "grown"
    receipt_base = builder.build(
        catalog_path=sources["catalog"], pairs_path=sources["pairs"],
        gate_path=sources["gate"], question_path=sources["question"],
        output_dir=base, seed=1729)
    _build(builder, sources, grown)
    assert receipt_base["counts"]["mask_cases"] == 0
    assert receipt_base["counts"]["aug_pairs"] == 0
    assert receipt_base["growth"]["bundle"] is None
    assert (base / "train.jsonl").read_text() != (
        grown / "train.jsonl").read_text()
