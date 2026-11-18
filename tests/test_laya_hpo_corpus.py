"""Scoped pins for scripts/laya_hpo_corpus.py — the HPO 50/50 zero-leakage carve.

Owner directive (2026-10-10): the stopped HPO's dev eval was LEAKY. These pin
the two public guarantees of the carve, hermetically (a tiny synthetic corpus,
no bundle/network) plus the built artifact's receipt:

  (a) ZERO leakage — records sharing a product side signature co-locate in ONE
      component and ONE half, so no component / endpoint / pair straddles;
      a field-less single state falls back to a full-text singleton (never a
      shared empty hub);
  (b) STRATIFIED + deterministic — every attribute/difficulty proportion is
      preserved in both halves and a rerun is byte-identical.

The real 50/50 balance over the 14,308-record corpus rides the committed
receipt (data/laya_hpo5050/receipt.json); the largest text-leakage component is
atomic, so a stratum concentrated in it (multi_diff) keeps a wider, documented
band than the near-exact attribute marginals.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.smart_split import SmartSplitProfile, smart_split_profile
from scripts.laya_corpus_composer import compose_side, compose_state
from scripts.laya_hpo_corpus import SPLIT_FILES, HpoCorpusCarver

_PROFILE = SmartSplitProfile(ratios={"train": 0.5, "dev": 0.5},
                             strata=("difficulty_slice", "attribute"), seed=1729)


def _record(state: str, difficulty: str, attribute: str) -> dict:
    return {"state": state, "questions": {}, "expected": {},
            "difficulty_slice": difficulty, "gate_reason": "",
            "attribute": attribute}


def _pair(attr_one: str, attr_two: str, difficulty: str, attribute: str) -> dict:
    state = compose_state(compose_side(attr_one), compose_side(attr_two))
    return _record(state, difficulty, attribute)


def _write_source(directory: Path, records: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "train.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _carve(tmp_path: Path, records: list[dict]) -> dict:
    source = tmp_path / "source"
    _write_source(source, records)
    return HpoCorpusCarver(source_dir=source, output_dir=tmp_path / "out",
                           profile=_PROFILE).run()


def _halves(tmp_path: Path) -> dict[str, list[dict]]:
    out = tmp_path / "out"
    return {name: [json.loads(line)
                   for line in (out / f"{name}.jsonl").read_text().splitlines()
                   if line.strip()]
            for name in ("train", "dev")}


def _synthetic_corpus() -> list[dict]:
    """A leak cluster + field-less singletons + a stratified singleton mass."""
    records = [
        # One text-leakage cluster: these four share product sides, so they
        # MUST land in one component and one half.
        _pair("Volume: 500", "Volume: 330", "one_diff", "volume"),
        _pair("Volume: 500", "Volume: 330", "one_diff", "volume"),
        _pair("Volume: 330", "Volume: 500; Flavour: orange", "one_diff",
              "volume"),
        _record("Volume: 500", "single_state", "volume"),
        # Two field-less single states: all-empty signatures must NOT collapse
        # into one shared hub (distinct full-text singletons).
        _record("Health Claims: antioxidant", "single_state", "none"),
        _record("Diets: vegan", "single_state", "none"),
    ]
    # A stratified singleton mass (unique sides) so the proportions are
    # measurable: 2 attributes x 2 difficulties x 30 records.
    for index in range(30):
        records.append(_pair(f"Volume: {1000 + index}", f"Volume: {2000 + index}",
                             "one_diff", "volume"))
        records.append(_record(f"Volume: {3000 + index}", "single_state",
                               "volume"))
        records.append(_pair(f"Flavour: flav{index}", f"Flavour: flav{index + 900}",
                             "one_diff", "flavor"))
        records.append(_record(f"Flavour: flav{index + 500}", "single_state",
                               "flavor"))
    return records


def test_hpo_carve_colocates_entities_and_is_zero_leakage(tmp_path):
    """(a) shared-side records co-locate; no component/endpoint/pair straddles."""
    _carve(tmp_path, _synthetic_corpus())
    halves = _halves(tmp_path)
    train, dev = halves["train"], halves["dev"]

    # ZERO leakage on all three grains the owner named.
    assert {r["component"] for r in train}.isdisjoint(
        r["component"] for r in dev)
    assert {e for r in train for e in r["endpoints"]}.isdisjoint(
        e for r in dev for e in r["endpoints"])
    assert {r["state"] for r in train}.isdisjoint(r["state"] for r in dev)

    # The leak cluster (two identical 500/330 pairs, a 330/orange pair sharing
    # the 330 side, and a single 500 state) is ONE component in ONE half.
    by_state = {}
    for half in (train, dev):
        for record in half:
            by_state.setdefault(record["state"], []).append(record)
    cluster = [r for r in train + dev
               if "side:[500.0]|||||" in r["endpoints"]
               or "side:[330.0]|||||" in r["endpoints"]]
    assert len(cluster) == 4
    assert len({r["component"] for r in cluster}) == 1
    assert len({("train" if r in train else "dev") for r in cluster}) == 1

    # The two field-less singletons are DISTINCT entities (no shared empty hub).
    fieldless = [r for r in train + dev if r["attribute"] == "none"]
    assert len(fieldless) == 2
    assert fieldless[0]["component"] != fieldless[1]["component"]
    assert all(r["endpoints"][0].startswith("raw:") for r in fieldless)

    # The 50/50 carve has no test role: test.jsonl is written EMPTY.
    assert (tmp_path / "out" / "test.jsonl").read_text() == ""
    assert SPLIT_FILES == ("train.jsonl", "dev.jsonl", "test.jsonl")


def test_hpo_carve_is_stratified_and_deterministic(tmp_path):
    """(b) per-stratum proportions preserved in both halves; rerun identical."""
    receipt = _carve(tmp_path, _synthetic_corpus())

    total = receipt["records"]
    sizes = receipt["split_sizes"]
    assert sizes["train"] + sizes["dev"] == total
    # Overall 50/50 within a small-N band.
    assert 0.40 <= sizes["train"] / total <= 0.60
    # Every attribute/difficulty stratum with real support keeps ~half in train.
    for axis in ("attribute_proportions", "difficulty_proportions"):
        for value, row in receipt[axis].items():
            if row["total"] >= 20:
                assert 0.35 <= row["train"] <= 0.65, (axis, value, row)

    # Deterministic: a second carve of the same source is byte-identical.
    again = HpoCorpusCarver(source_dir=tmp_path / "source",
                            output_dir=tmp_path / "out2",
                            profile=_PROFILE).run()
    assert again["corpus_sha256"] == receipt["corpus_sha256"]


def test_built_hpo5050_corpus_is_zero_leakage_and_proportional():
    """The committed 50/50 carve: the declared SSOT profile + a zero-leak,
    proportional receipt over the real 14,308-record corpus."""
    profile = smart_split_profile("laya_hpo_5050")
    assert profile.ratios == {"train": 0.5, "dev": 0.5}
    assert profile.strata == ("difficulty_slice", "attribute")

    receipt_path = (Path(__file__).resolve().parents[1]
                    / "data/laya_hpo5050/receipt.json")
    if not receipt_path.is_file():
        pytest.skip("data/laya_hpo5050 not built (run scripts/laya_hpo_corpus)")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    # ZERO leakage on all three grains (the owner mandate).
    assert receipt["leakage"] == {"components": 0, "endpoints": 0, "pairs": 0}
    # The halves are 50/50 and the empty test role is recorded.
    total = receipt["records"]
    train = receipt["split_sizes"]["train"]
    assert 0.48 <= train / total <= 0.52
    # Attributes are near-exact; difficulty allows the atomic giant component's
    # fixed composition (multi_diff concentrates in the largest text-leak cluster).
    for value, row in receipt["attribute_proportions"].items():
        if row["total"] >= 200:
            assert 0.45 <= row["train"] <= 0.55, (value, row)
    for value, row in receipt["difficulty_proportions"].items():
        if row["total"] >= 200:
            assert 0.30 <= row["train"] <= 0.70, (value, row)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
