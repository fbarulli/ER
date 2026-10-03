"""Attribute universe guard tests — semantics only, never the live corpus.

The live 71,623-row reproduction is scripts/attribute_universe_census.py's
job (it verifies against MEASURED_BASELINE and writes the census JSON).
These tests pin the CONTRACT on synthetic frames: hand-computed census
counts, delegated parsers being exactly the existing extractors, the veto
band classification, and that verify_census detects doctored counts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.attribute_universe import (
    AttributeUniverse,
    attribute_registry,
    _canonical_band,
)


_UNCLASSIFIED_KEY = "unclassified_keys"

# Checksum-valid corpus gtins (same fixtures test_dedupe_identity.py uses,
# extended with a programmatic family so a synthetic frame can afford pair
# groups without hand-carrying check digits).
_KNOWN_VALID = ("8715600246377", "8715600248098")


def _gs1_check_digit(body: str) -> int:
    total = sum(int(d) * w for d, w in zip(reversed(body), (3, 1) * 6))
    return (10 - total % 10) % 10


def _valid_gtin(index: int) -> str:
    body = f"87156002{index:04d}"
    return body + str(_gs1_check_digit(body))


def _frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"attribute": attributes, "gtin": gtin} for gtin, attributes in rows] or
        {"attribute": pd.Series(dtype=str), "gtin": pd.Series(dtype=str)}
    )


def test_census_matches_hand_computed_counts():
    u = AttributeUniverse(_frame([
        (_KNOWN_VALID[0], "Juice Content: 100%"),
        (_KNOWN_VALID[0], "Juice Content: 100%"),
        (_KNOWN_VALID[1], "Juice Content: 0-2%"),
        (_KNOWN_VALID[1], "Juice Content: 0-2%, 100%"),
        ("12345-9", "Juice Content: 25-50%"),   # malformed length -> no pairs
        ("12345-9", "Juice Content: 100%"),
    ]))
    census = u.census()
    juice = census["keys"]["juice content"]
    # 6 populated rows, 4 distinct sets, 2 same-GTIN pairs (invalid-gtin
    # rows contribute none), 1 conflict ("0-2%" vs "0-2%, 100%") -> rate 0.5.
    assert juice["rows_populated"] == 6
    assert juice["distinct_value_sets"] == 4
    assert juice["same_gtin_pairs_both_populated"] == 2
    assert juice["conflict_pairs"] == 1
    assert juice["conflict_rate"] == 0.5


def test_parse_normalizes_and_buckets_deterministically():
    u = AttributeUniverse(_frame([]))
    parsed = u.parse("Flavour: banana; Count per Unit: 12; New Thing: x, y")
    parsed_again = u.parse("Flavour: banana; Count per Unit: 12; New Thing: x, y")
    assert parsed == parsed_again
    assert parsed["flavour"] == frozenset({"banana"})
    assert parsed["count per unit"] == frozenset({"12"})
    assert parsed[_UNCLASSIFIED_KEY] == ("new thing",)

    # ordered band parser keeps band text canonical, never invents values
    assert _canonical_band("0-2 %") == "0-2%"
    assert _canonical_band("200 + mg") == "200+mg"
    assert _canonical_band("nonsense") == "nonsense"


def test_registry_covers_all_37_keys():
    registry = attribute_registry()
    assert len(registry) == 37
    assert registry["pack material type"].kind == "SET_ENUM"
    assert registry["juice content"].kind == "NUMERIC_BAND"
    assert registry["special edition"].kind == "CONSTANT"
    for constant in ("special edition", "giftbox"):
        assert registry[constant].kind == "CONSTANT"


def test_delegated_fields_parse_identically_to_existing_extractors():
    u = AttributeUniverse(_frame([]))
    flavour_cell = "Flavour: blueberry, banana, made up word"
    from core.critical_attributes import extract_declared_flavor_tokens
    assert u.parse(flavour_cell)["flavour"] == extract_declared_flavor_tokens(flavour_cell)
    assert u.parse(flavour_cell)["flavour"] == frozenset({"blueberry", "banana"})

    sweetener_cell = "Sweetener: sugar, unsweetened, unknown sweetener"
    from core.sweetener_values import declared_sweeteners
    expected = set()
    for part in ("sweetener_type", "sweetening", "unmapped"):
        expected |= set(declared_sweeteners(sweetener_cell)[part])
    assert u.parse(sweetener_cell)["sweetener"] == frozenset(expected)

    for volume_cell in ("Volume: 355", "Volume: 2 l", "Volume: 12 fl oz"):
        from pipeline import parse_attribute_volume_pack
        extracted, *_ = parse_attribute_volume_pack(volume_cell)
        assert u.parse(volume_cell)["volume"] == frozenset({float(extracted)})
    assert u.parse("Volume: 355")["volume"] == frozenset({355.0})

    enum_cell = "Pack Material Type: Paper / Carton"
    assert u.parse(enum_cell)["pack material type"] == frozenset({"paper / carton"})


def _band_frame(conflict_groups: int, agree_groups: int) -> pd.DataFrame:
    """'pack material type' at exactly conflict_groups/(conflict+agree) rate.

    Every group is two rows on one checksum-valid GTIN: a conflicting group
    contributes 1 pair + 1 conflict, an agreeing group 1 pair + none.
    """
    materials = ["Glass", "Metal", "Plastic", "Paper / Carton", "Flexible Pack"]
    rows = []
    index = 0
    for _ in range(conflict_groups):
        gtin = _valid_gtin(index)
        index += 1
        rows.append((gtin, "Pack Material Type: Glass; Coffee Type: arabica"))
        rows.append((gtin, "Pack Material Type: Plastic; Coffee Type: arabica"))
    for _ in range(agree_groups):
        gtin = _valid_gtin(index)
        index += 1
        material = materials[index % len(materials)]
        rows.append((gtin, f"Pack Material Type: {material}; Coffee Type: arabica"))
        rows.append((gtin, f"Pack Material Type: {material}; Coffee Type: arabica"))
    return _frame(rows)


def test_datagen_budget_veto_band():
    # 69 conflicting groups + 431 agreeing groups = 69/500 = 13.8% exactly.
    u = AttributeUniverse(_band_frame(conflict_groups=69, agree_groups=431))
    census = u.census()
    assert census["keys"]["pack material type"]["conflict_rate"] == pytest.approx(0.138)
    assert census["valid_gtin_rows"] == 1000
    budget = u.datagen_budget(census=census, min_value_support=20)
    assert budget["pack material type"]["veto_candidate"] is True
    # coffee type agrees on every pair (0% rate): below the conflict-rate
    # floor, so it can never veto — the doctrine's absence-is-not-a-conflict.
    assert budget["coffee type"]["veto_candidate"] is False
    assert budget["special edition"]["kind"] == "CONSTANT"


def test_verify_census_detects_doctored_counts():
    u = AttributeUniverse(_band_frame(conflict_groups=69, agree_groups=431))
    census = u.census()
    # the baseline contract is the SCALAR metrics per key (top_sets are sets,
    # excluded by design: a non-numeric baseline entry would be ambiguous)
    scalar_metrics = (
        "rows_populated", "distinct_value_sets", "distinct_raw_strings",
        "same_gtin_pairs_both_populated", "conflict_pairs", "conflict_rate",
    )
    honest = {
        key: {metric: stats[metric] for metric in scalar_metrics}
        for key, stats in census["keys"].items()
    }
    # a clean baseline equals the measured census: no drift
    u.verify_census({"keys": honest}, baseline=honest)
    # doctoring one metric beyond +/-1% must fail loudly
    doctored = {key: dict(stats) for key, stats in honest.items()}
    doctored["pack material type"]["rows_populated"] = round(
        doctored["pack material type"]["rows_populated"] * 1.05
    )
    with pytest.raises(SystemExit):
        u.verify_census({"keys": doctored}, baseline=honest)
    with pytest.raises(SystemExit):
        u.verify_census({"keys": {}}, baseline=honest)


def test_verify_census_is_live_data_free_for_the_suite():
    """Pinned live expectations live in the census SCRIPT, not here.

    The module pins MEASURED_BASELINE (its own census semantics on the real
    corpus); this test only asserts the mechanism exists so the suite never
    pins stale 71,623-row counts itself.
    """
    from core.attribute_universe import MEASURED_BASELINE

    assert "pack material type" in MEASURED_BASELINE
    assert "juice content" in MEASURED_BASELINE
    u = AttributeUniverse(_frame([]))
    tiny = {key: {"rows_populated": 0} for key in MEASURED_BASELINE}
    with pytest.raises(SystemExit):
        u.verify_census({"keys": tiny})
