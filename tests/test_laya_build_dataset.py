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
from collections import defaultdict
from pathlib import Path

import pytest


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


# ── the config-owned corpus composition block ──────────────────────────────
# `config/laya.question.json`'s optional top-level "corpus" block owns the
# composition knobs (seed / hard_no_cap / identity_negative_target_ratio /
# sources). Every key defaults to the historical behaviour, so a schema with
# no block (or an all-null block) reproduces the landed corpus exactly.
_FULL_QUESTIONS = {
    "identity_claim": {"type": "noul", "instructions": "same item?"},
    "package_state": {"type": "noul", "instructions": "has pack?"},
    "evidence_sufficient": {"type": "noul", "instructions": "enough?"},
    "counterfactual": {"type": "noul", "instructions": "minted?"},
    "same_brand_only": {"type": "noul", "instructions": "brand only?"},
    "pack_volume_equal": {"type": "noul", "instructions": "same pack?"},
    "pack_format_equivalent": {"type": "noul", "instructions": "same form?"},
    "better_match": {"type": "choice", "instructions": "which?",
                     "criteria": {"candidate_1": None, "candidate_2": None}},
    "gate_verdict": {"type": "choice", "instructions": "verdict?",
                     "criteria": {"proceed": None, "hard_no": None,
                                  "fallback": None}},
    "gate_reason": {"type": "choice", "instructions": "reason?",
                    "criteria": {"pack_blocker": None,
                                 "critical_attribute": None,
                                 "compatible": None}},
}
for _field in ("volume", "pack", "package_type", "sweetener", "flavor",
               "carbonation"):
    _FULL_QUESTIONS[f"field_same:{_field}"] = {
        "type": "choice", "instructions": "same?",
        "criteria": {"same": None, "different": None, "unknown": None}}


def _write_question(path: Path, corpus: dict | None = None) -> None:
    document = {"schema": "test", "questions": _FULL_QUESTIONS}
    if corpus is not None:
        document["corpus"] = corpus
    path.write_text(json.dumps(document), encoding="utf-8")


def test_corpus_config_block_is_read_and_unknown_keys_fail_loud(tmp_path):
    builder = _builder()
    document = {"questions": _FULL_QUESTIONS,
                "corpus": {"hard_no_cap": 2, "seed": 99}}
    config = builder._corpus_config(document)
    assert config["hard_no_cap"] == 2 and config["seed"] == 99
    assert config["identity_negative_target_ratio"] is None
    with pytest.raises(ValueError, match="unknown laya corpus knob"):
        builder._corpus_config({"questions": {}, "corpus": {"nope": 1}})
    with pytest.raises(ValueError, match="unknown laya corpus source"):
        builder._corpus_config(
            {"questions": {}, "corpus": {"sources": {"nope": "x"}}})
    assert builder.corpus_config_is_default(builder._corpus_config(
        {"questions": {}, "corpus": {"seed": None, "hard_no_cap": None,
                                     "sources": {"catalog": None}}}))


def test_all_null_corpus_block_reproduces_the_default_corpus(tmp_path):
    """The committed all-null block is a no-op: byte-identical output."""
    builder = _builder()
    sources = _sources(tmp_path)
    bare = tmp_path / "bare"
    nulled = tmp_path / "nulled"
    question_bare = tmp_path / "bare.question.json"
    question_null = tmp_path / "null_corpus.question.json"
    _write_question(question_bare)
    _write_question(question_null, corpus={
        "seed": None, "hard_no_cap": None,
        "identity_negative_target_ratio": None,
        "sources": {"catalog": None, "pairs": None, "gate": None,
                    "bundle": None, "labeled_pairs": None,
                    "output_dir": None}})
    receipts = [
        builder.build(
            catalog_path=sources["catalog"], pairs_path=sources["pairs"],
            gate_path=sources["gate"], question_path=question,
            output_dir=out, seed=1729, bundle_path=sources["bundle"],
            labeled_pairs_path=sources["labeled"])
        for question, out in ((question_bare, bare), (question_null, nulled))]
    assert receipts[0]["sha256"] == receipts[1]["sha256"]
    assert "identity_rebalance" not in receipts[1]
    assert "corpus_config" not in receipts[1]
    assert receipts[0]["counts"] == receipts[1]["counts"]


def test_corpus_sources_override_resolves_against_the_repo_root(tmp_path):
    builder = _builder()
    sources = {
        "catalog": "data/track_setup/eligible_catalog.csv",
        "output_dir": str(tmp_path / "absolute"),
    }
    resolved = builder.resolve_corpus_sources({"sources": sources})
    assert resolved["catalog"] == builder.ROOT / sources["catalog"]
    assert resolved["output_dir"] == tmp_path / "absolute"
    # an unset key keeps the hardcoded default
    assert resolved["pairs"] == builder.PAIRS_PATH


def test_main_honours_the_config_owned_sources_and_knobs(tmp_path):
    """`main()` resolves BOTH the paths and the knobs from the ONE config file."""
    builder = _builder()
    sources = _rebalance_fixture(tmp_path)
    out = tmp_path / "main_out"
    document = {"schema": "test", "questions": _FULL_QUESTIONS, "corpus": {
        "identity_negative_target_ratio": 1.0,
        "sources": {
            "catalog": str(sources["catalog"]),
            "pairs": str(sources["pairs"]),
            "gate": str(sources["gate"]),
            "bundle": str(sources["bundle"]),
            "labeled_pairs": str(sources["labeled"]),
            "output_dir": str(out),
        }}}
    config_path = tmp_path / "main.question.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    original = builder.QUESTION_PATH
    builder.main.__globals__["QUESTION_PATH"] = config_path
    try:
        builder.main()
    finally:
        builder.main.__globals__["QUESTION_PATH"] = original
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["identity_rebalance"]["minted_negatives_dropped"] == 2
    assert receipt["corpus_config"]["identity_negative_target_ratio"] == 1.0
    assert receipt["counts"]["identity_negative_total_with_growth"] == 2
    assert (out / "train.jsonl").is_file()


def _rebalance_fixture(tmp_path: Path) -> dict:
    """4 pipeline-MINTED counterfactual negatives, 2 ground-truth positives.

    No gate rows and no ground-truth negatives, so the rebalance target is
    reachable and the exact trim is observable.
    """
    catalog = tmp_path / "catalog.csv"
    _write_csv(catalog, ["sku_id", "gtin", "attribute"], [
        ["S1", "5000000001", "Flavour: apple"],
        ["S2", "5000000002", "Flavour: lime"],
        ["S3", "5000000003", "Flavour: cola"],
        ["S4", "5000000004", "Flavour: pear"],
        ["S5", "5000000005", "Flavour: grape"],
        ["S6", "5000000006", "Volume: 500"],
    ])
    pairs = tmp_path / "pairs.csv"
    # the aug states (flavor v1=[apple] v2=[X]) never collide with these
    _write_csv(pairs, ["sku_id1", "sku_id2", "label", "split"], [
        ["S1", "S6", "1", "train"],
        ["S2", "S6", "1", "dev"],
    ])
    gate = tmp_path / "gate.csv"
    _write_csv(gate, ["gtin1", "gtin2", "gate_decision", "gate_reason"], [])
    labeled = tmp_path / "labeled_pairs.csv"
    _write_csv(labeled, ["gtin1", "gtin2", "true_label"], [])
    payload = ["flavor_apple", "flavor_lime", "flavor_cola", "flavor_pear",
               "flavor_grape", "volume_ml_500", "flavor_melon"]
    audits = [
        {"population": "hard_negative", "target_mode": "counterfactual",
         "pair_payload_idx": index, "masked_text": payload[0]}
        for index in (2, 3, 4, 6)
    ]
    bundle = tmp_path / "bundle.pkl.gz"
    import pandas as pd

    with gzip.open(bundle, "wb") as handle:
        pickle.dump({"df": pd.DataFrame({"gtin": ["5000000001"]}),
                     "payload": payload, "mask_audit": [],
                     "hard_negative_mask_audit": audits}, handle)
    question = tmp_path / "laya.question.json"
    _write_question(question)
    return {"catalog": catalog, "pairs": pairs, "gate": gate,
            "question": question, "bundle": bundle, "labeled": labeled}


def test_identity_rebalance_thins_only_minted_negatives(tmp_path):
    builder = _builder()
    sources = _rebalance_fixture(tmp_path)
    default_out = tmp_path / "default"
    trimmed_out = tmp_path / "trimmed"
    receipt_default = _build(builder, sources, default_out)
    receipt_trimmed = builder.build(
        catalog_path=sources["catalog"], pairs_path=sources["pairs"],
        gate_path=sources["gate"], question_path=sources["question"],
        output_dir=trimmed_out, seed=1729, bundle_path=sources["bundle"],
        labeled_pairs_path=sources["labeled"],
        identity_negative_target_ratio=1.0)
    # default: 4 minted negatives ride the corpus, no rebalance census
    assert receipt_default["counts"]["identity_positive_total_with_growth"] == 2
    assert receipt_default["counts"]["identity_negative_total_with_growth"] == 4
    assert "identity_rebalance" not in receipt_default
    # ratio 1.0 -> target 2 negatives; the minted population is thinned to 2
    rebalance = receipt_trimmed["identity_rebalance"]
    assert rebalance["enabled"] is True
    assert rebalance["target_ratio"] == 1.0
    assert rebalance["positives_total"] == 2
    assert rebalance["ground_truth_negatives"] == 0
    assert rebalance["minted_negatives_available"] == 4
    assert rebalance["minted_negatives_dropped"] == 2
    assert receipt_trimmed["counts"]["identity_negative_total_with_growth"] == 2
    assert receipt_trimmed["split_sizes"] != receipt_default["split_sizes"]
    # no fabricated labels: every emitted identity negative is still a real case
    negatives = [
        json.loads(line)
        for key in ("train", "dev", "test")
        for line in (trimmed_out / f"{key}.jsonl").read_text().splitlines()
        if json.loads(line)["expected"].get("identity_claim") == "false"]
    assert len(negatives) == 2
    assert all(line["expected"]["counterfactual"] == "true"
               for line in negatives)


def test_identity_rebalance_is_deterministic_and_config_driven(tmp_path):
    builder = _builder()
    sources = _rebalance_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    question = tmp_path / "config.question.json"
    _write_question(question, corpus={"identity_negative_target_ratio": 1.0})
    receipts = [
        builder.build(
            catalog_path=sources["catalog"], pairs_path=sources["pairs"],
            gate_path=sources["gate"], question_path=question,
            output_dir=out, seed=1729, bundle_path=sources["bundle"],
            labeled_pairs_path=sources["labeled"])
        for out in (first, second)]
    assert receipts[0]["corpus_sha256"] == receipts[1]["corpus_sha256"]
    # the resolved config rides the receipt when a knob is set
    assert receipts[0]["corpus_config"]["identity_negative_target_ratio"] == 1.0
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "receipt.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    # an explicit corpus_config block still wins over the file's config SSOT
    override = builder.build(
        catalog_path=sources["catalog"], pairs_path=sources["pairs"],
        gate_path=sources["gate"], question_path=question,
        output_dir=tmp_path / "override", seed=1729,
        bundle_path=sources["bundle"], labeled_pairs_path=sources["labeled"],
        corpus_config={"identity_negative_target_ratio": None})
    assert "identity_rebalance" not in override
    assert override["counts"]["identity_negative_total_with_growth"] == 4


# ── item: the new questions are labelled from existing sources only ────────
def test_new_question_labels_come_from_the_sources_never_fabricated(tmp_path):
    builder = _builder()
    catalog = tmp_path / "catalog.csv"
    _write_csv(catalog, ["sku_id", "gtin", "brand", "attribute"], [
        ["S1", "5000000001", "Acme", "Volume: 500; Flavour: apple"],
        ["S2", "5000000002", "Acme", "Volume: 500; Flavour: lime"],
        ["S3", "5000000003", "Acme", "Volume: 330; Flavour: cola"],
        ["S4", "5000000004", "Other", "Flavour: pear"],
    ])
    pairs = tmp_path / "pairs.csv"
    # (S1, S2) same item; (S1, S3) different items but the SAME brand; the
    # extra (S3, S4) positive leaves room for one gate hard_no negative.
    _write_csv(pairs, ["sku_id1", "sku_id2", "label", "split"], [
        ["S1", "S2", "1", "train"],
        ["S3", "S4", "1", "test"],
        ["S1", "S3", "0", "dev"],
    ])
    gate = tmp_path / "gate.csv"
    _write_csv(
        gate,
        ["gtin1", "gtin2", "gate_decision", "gate_reason"], [
            ["5000000001", "5000000004", "hard_no",
             "Pack blocker: pack size"],
            ["5000000002", "5000000004", "proceed",
             "Known critical attributes compatible"],
        ])
    question = tmp_path / "laya.question.json"
    _write_question(question)
    out = tmp_path / "out"
    builder.build(catalog_path=catalog, pairs_path=pairs, gate_path=gate,
                  question_path=question, output_dir=out, seed=1729)
    records = [
        json.loads(line)
        for key in ("train", "dev", "test")
        for line in (out / f"{key}.jsonl").read_text().splitlines()]

    by_key: dict[tuple, list] = defaultdict(list)
    for record in records:
        expected = record["expected"]
        key = (expected.get("identity_claim"), expected.get("gate_verdict"),
               expected.get("counterfactual"))
        by_key[key].append(expected)

    same_pair = by_key[("true", None, "false")][0]
    # the enumerated pair labels, read off the composed sides / attributes
    assert same_pair["field_same:volume"] == "same"
    assert same_pair["field_same:flavor"] == "different"
    assert same_pair["pack_volume_equal"] == "true"
    assert same_pair["pack_format_equivalent"] == "false"
    assert same_pair["evidence_sufficient"] == "true"
    assert same_pair["same_brand_only"] == "false"  # same brand AND same item
    assert "gate_verdict" not in same_pair  # not a gate row: never invented
    assert "counterfactual" in same_pair

    different_pair = by_key[("false", None, "false")][0]
    assert different_pair["same_brand_only"] == "true"  # same brand, not item

    gate_hard_no = by_key[("false", "hard_no", "false")][0]
    assert gate_hard_no["gate_reason"] == "pack_blocker"
    gate_proceed = by_key[(None, "proceed", "false")][0]
    assert gate_proceed["gate_reason"] == "compatible"
    # the gate verdict is not a GTIN truth: identity stays unlabelled there
    assert "identity_claim" not in gate_proceed

    # better_match: S1 is confirmed-same with S2 and confirmed-different from S3
    better = [record["expected"]["better_match"] for record in records
              if "better_match" in record["expected"]]
    assert better and set(better) <= {"candidate_1", "candidate_2"}

    # no fabrication: a catalog carrying NO brand column emits NO
    # same_brand_only label at all (the state holds no brand evidence).
    brandless = tmp_path / "brandless_catalog.csv"
    _write_csv(brandless, ["sku_id", "gtin", "attribute"], [
        ["S1", "5000000001", "Volume: 500; Flavour: apple"],
        ["S2", "5000000002", "Volume: 500; Flavour: lime"],
        ["S3", "5000000003", "Volume: 330; Flavour: cola"],
        ["S4", "5000000004", "Flavour: pear"],
    ])
    brandless_out = tmp_path / "brandless"
    builder.build(catalog_path=brandless, pairs_path=pairs, gate_path=gate,
                  question_path=question, output_dir=brandless_out, seed=1729)
    for key in ("train", "dev", "test"):
        for line in (brandless_out / f"{key}.jsonl").read_text().splitlines():
            assert "same_brand_only" not in json.loads(line)["expected"]

