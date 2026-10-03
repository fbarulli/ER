"""Census-to-wiring parity for the attribute-universe capture.

The attribute census (results/attribute_universe_census.json) measured 28
never-parsed raw attribute keys; the highest-yield captures are wired here
from the EVIDENCE side only (pipeline.extract_all structured evidence
section -> structured_features captured groups -> training.masking donor
registry). The veto list itself stays config-owned
(config/training.yaml rand_matching.targeted_veto_gates.veto_dimensions) and
is untouched. These tests pin the CONTRACT, never the live corpus:

* capture semantics — attribute-cell material reaches package_materials
  (title scrape first byte-unchanged, attribute values unioned after),
  bands canonicalized to the census device, foreign keys ignored;
* masking donor registry — the two donor-capable fields (pack material
  type: 5 sets; juice content bands: ~27) become mask/swap targets and every
  unknown-field raise stays fail-loud;
* byte contract — a synthetic frame's existing [FIELD_*] token stream is
  unchanged byte-wise with and without the captured evidence (append-only);
* census parity — the keys capture_universe_attributes() emits are the
  AttributeUniverse registry names, and on a synthetic cell the captured
  value sets EQUAL AttributeUniverse.parse's, band canon included.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline import (
    ATTRIBUTE_UNIVERSE_CAPTURE_KEYS,
    _canonical_band,
    capture_universe_attributes,
    extract_all,
)

CELL = (
    "Pack Material Type: Metal, Paper / Carton; Volume: 355; "
    "Juice Content: 0-2%, 100%; Water Type: Mineral; "
    "Carbonization: Still; Naturally Derived: Natural; "
    "Made From: Orange"
)


def test_attributes_material_reaches_package_materials_from_the_attribute_cell():
    """(a) material evidence ships as package materials (title ∪ attribute).

    The title NER scrape keeps its convention first; census evidence is the
    only new source, and the raw census value joints stay unmodified —
    "paper / carton" is the byte-exact census top-set value.
    """
    attribute_only = extract_all("Acme Cola variety pack", "Pack Material Type: Glass")
    assert attribute_only["package_materials"] == ["glass"]
    assert attribute_only["attribute_universe_evidence"]["pack material type"] == [
        "glass"
    ]

    union = extract_all(
        "Acme glass bottle", "Pack Material Type: Plastic, Paper / Carton"
    )
    assert union["package_materials"] == ["glass", "paper / carton", "plastic"]

    # the numeric band field is captured through the census band canon
    bands = extract_all("Acme juice", "Juice Content: 100%, 15-25 mg")
    assert bands["attribute_universe_evidence"]["juice content"] == ["100%", "15-25mg"]

    # no evidence anywhere -> title scrape bytes unchanged, no capture rows
    title_only = extract_all("Acme glass bottle 6 pack", "")
    assert title_only["package_materials"] == ["glass"]
    assert title_only["attribute_universe_evidence"] == {}

    # the full synthetic battery is captured across all declared keys
    full = extract_all("Acme Cola", CELL)
    assert full["attribute_universe_evidence"]["pack material type"] == [
        "metal",
        "paper / carton",
    ]
    assert full["attribute_universe_evidence"]["juice content"] == ["0-2%", "100%"]
    assert full["attribute_universe_evidence"]["water type"] == ["mineral"]
    assert full["attribute_universe_evidence"]["carbonization"] == ["still"]
    assert full["attribute_universe_evidence"]["naturally derived"] == ["natural"]
    assert full["attribute_universe_evidence"]["made from"] == ["orange"]


def test_masking_registry_addresses_the_two_donor_capable_fields():
    """(b) donor vocabularies reach mask/swap lanes; unknown fields stay loud.

    Measured support (core.attribute_universe registry): pack material type
    is a 5-set SET_ENUM at a 9.64% conflict rate and juice content a
    27-band NUMERIC_BAND channel — both above the donor floor. The masking
    registry must resolve them like every other entry and must keep raising
    on a field outside the registry (fail loud, never silently mask nothing).
    """
    from training.masking import (
        _FIELD_PREFIXES, field_of, mask_targeted, swap_structured_field, MASK_TOKEN,
    )
    assert set(ATTRIBUTE_UNIVERSE_CAPTURE_KEYS) == {
        "pack material type",
        "juice content",
        "carbonization",
        "naturally derived",
        "water type",
        "made from",
    }
    assert {
        "volume", "pack", "package_type", "flavor", "carbonation",
        "sweetener_type", "sweetening", "sweetener", "pulp",
        "package_material", "juice_content",
    } <= set(_FIELD_PREFIXES)
    assert field_of("package_material_glass") == "package_material"
    assert field_of("package_material_paper_/_carton") == "package_material"
    assert field_of("juice_content_0-2%") == "juice_content"
    assert field_of("[FIELD_PACK_MATERIAL]") is None

    swapped, changed = swap_structured_field(
        "base [FIELD_PACKAGE_TYPE] package_type_can [FIELD_PACK_MATERIAL] package_material_glass",
        "donor package_material_metal",
        field="package_material",
    )
    assert changed and "package_material_metal" in swapped
    assert "package_type_can" in swapped

    masked, extent, hit = mask_targeted(
        "still [FIELD_JUICE_CONTENT_BAND] juice_content_0-2% juice_content_100%",
        fields=("juice_content",),
        background_prob=0.0,
    )
    assert hit == ["juice_content"]
    # the mask token ALWAYS lands (a masked copy is never an exact duplicate)
    assert masked.split() == [
        "still",
        "[FIELD_JUICE_CONTENT_BAND]",
        MASK_TOKEN,
        MASK_TOKEN,
    ]

    for bad_field in ("water_type", "pack_material"):
        with pytest.raises(ValueError, match="unknown swap field"):
            swap_structured_field("a", "b", field=bad_field)
        with pytest.raises(ValueError, match="unknown mask target fields"):
            mask_targeted("still", fields=[bad_field])


def test_existing_field_token_stream_is_unchanged_and_capture_appends():
    """(c) the byte contract: legacy groups keep every token; capture appends.

    The pinned expected stream is the pre-capture composition of a fully
    populated synthetic frame (nine legacy groups in the legacy marker
    order). The capture groups ([FIELD_PACK_MATERIAL],
    [FIELD_JUICE_CONTENT_BAND]) sit AFTER every legacy group, so an info
    without the new keys renders this exact byte stream, and with the keys
    the delta is ONLY the captured blocks — pre-capture tokens never move.
    """
    from core.structured_features import append_text, canonical_info, info_from_sets, text_tokens

    legacy_info = info_from_sets(
        [355.0],
        [6],
        {"bottle"},
        flavor={"cola", "lemon"},
        carbonation={"carbonated"},
        sweetener={"no_sugar"},
        pulp={"no_pulp"},
        sweetener_type={"stevia"},
        sweetening={"unsweetened"},
    )
    streams = {append_text("Acme soda", legacy_info) for _ in range(25)}
    assert len(streams) == 1
    legacy_text = streams.pop()
    assert legacy_text == (
        "Acme soda "
        "[FIELD_VOLUME] volume_ml_355 [FIELD_PACK_SIZE] pack_qty_6 "
        "[FIELD_PACKAGE_TYPE] package_type_bottle [FIELD_FLAVOR] flavor_cola "
        "flavor_lemon [FIELD_CARBONATION] carbonation_carbonated "
        "[FIELD_SWEETENER_DIET] sweetener_diet_no_sugar [FIELD_PULP] pulp_no_pulp "
        "[FIELD_SWEETENER_TYPE] sweetener_type_stevia "
        "[FIELD_SWEETENING] sweetening_unsweetened"
    )

    captured = dict(legacy_info)
    captured["package_material"] = {"glass", "paper / carton"}
    captured["juice_content"] = {"0-2%", "100%"}
    assert text_tokens(captured) == [
        "[FIELD_VOLUME]", "volume_ml_355",
        "[FIELD_PACK_SIZE]", "pack_qty_6",
        "[FIELD_PACKAGE_TYPE]", "package_type_bottle",
        "[FIELD_FLAVOR]", "flavor_cola", "flavor_lemon",
        "[FIELD_CARBONATION]", "carbonation_carbonated",
        "[FIELD_SWEETENER_DIET]", "sweetener_diet_no_sugar",
        "[FIELD_PULP]", "pulp_no_pulp",
        "[FIELD_SWEETENER_TYPE]", "sweetener_type_stevia",
        "[FIELD_SWEETENING]", "sweetening_unsweetened",
        "[FIELD_PACK_MATERIAL]",
        "package_material_glass", "package_material_paper_/_carton",
        "[FIELD_JUICE_CONTENT_BAND]", "juice_content_0-2%", "juice_content_100%",
    ]
    # strictly append-only: the legacy byte stream prefixes the captured one
    captured_text = append_text("Acme soda", captured)
    assert captured_text.startswith(legacy_text + " ")
    assert captured_text[len(legacy_text) + 1:].split() == [
        "[FIELD_PACK_MATERIAL]",
        "package_material_glass", "package_material_paper_/_carton",
        "[FIELD_JUICE_CONTENT_BAND]", "juice_content_0-2%", "juice_content_100%",
    ]

    # the canonical lane reads the SAME set convention through the
    # already-existing package_material_set column; absence stays absence
    canonical = canonical_info({"package_material_set": "['paper / carton', 'pet']"})
    assert text_tokens(canonical) == [
        "[FIELD_PACK_MATERIAL]",
        "package_material_paper_/_carton", "package_material_pet",
    ]
    assert canonical_info({})["package_material"] == set()
    assert canonical_info({})["juice_content"] == set()


def test_capture_keys_match_the_attribute_universe_registry():
    """(d) census — registered names, census band canon, and cell parity."""
    from core.attribute_universe import AttributeUniverse, _canonical_band as census_band, attribute_registry

    assert set(ATTRIBUTE_UNIVERSE_CAPTURE_KEYS) <= set(attribute_registry())

    universe = AttributeUniverse(
        pd.DataFrame({"attribute": [CELL], "gtin": ["8715600246377"]})
    )
    parsed = universe.parse(CELL)
    captured = capture_universe_attributes(CELL)
    for key in ATTRIBUTE_UNIVERSE_CAPTURE_KEYS:
        assert key in parsed, key
        assert parsed[key] == captured[key], (key, sorted(parsed[key]), sorted(captured[key]))

    # the band canon is the census device byte for byte across corpus-shaped
    # variants: spacing, en dash, plus band, exact value, decimals
    battery = ("0-2%", "0 - 2%", "0\u20132%", "100%", "100 %",
               "200+ mg", "200 + mg", "15-25 mg", "5.5%", "no band words")
    for token in battery:
        assert _canonical_band(token) == census_band(token)
