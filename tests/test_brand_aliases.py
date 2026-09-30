"""Brand-alias map guards (vocabulary.json brand_aliases, seeded 2026-09-30).

The map (scripts/seed_brand_aliases.py) is an explicit reviewed pair list from
the 2026-09-29 within-GTIN measurement (TODO.md:668-676): brand variants are
SEMANTIC rebrands/sister filings, not typos. These tests pin the four
properties the owner doctrine demands:

(a) every seeded alias folds BOTH directions of its measured pair consistently
    (the token-star map is symmetric at the SET level);
(b) the SSOT still loads through core.common.vocabulary() without schema
    errors (veto-asymmetry hinges on the map being datable, not code);
(c) DISTINCT brands are never merged by folding (162 unique within-group
    brand-string vetoes -> after seeding, the declined/mixed-GTIN groups must
    keep firing; the map holds 8 entries, no fuzzy keys);
(d) veto asymmetry: with the map live, a brand conflict MUST still fire where
    no alias applies — the fold never manufactures agreement between real
    competitors and never doses out a conflict.
"""

from __future__ import annotations

import pytest

from core.common import vocabulary
from core.product_identity import (
    brand_aliases,
    brand_conflict,
    identity_conflict,
    normalize_brand,
    row_identity,
)

# The measured granted families + their source spellings. Kept as literals so
# a silent vocabulary.json drift is caught (config as SSOT governs the map;
# these constants guard the SEEDING CONTRACT, same doctrine as
# common.PINNED_GATE_FALLBACK_PAIRS).
FAMILIES: dict[str, tuple[str, ...]] = {
    "shoc": ("a shoc", "adrenaline shoc", "accelerator"),
    "dg": ("dg", "ting"),
    "hiball": ("hi ball", "hiball energy"),
    "lifeaid": ("fitaid", "lifeaid"),
    "olvi": ("olvi", "kevytolo"),
    "biotech": ("biotech usa", "bio techusa"),
}

# Measured decline set: whole-barcode mixed groups or key-restricted families;
# every pair here must STILL fire a brand conflict (49 within-group vetoes, 22
# still live — all 27 dissolved are seeded-family pairs).
EXPECTED_CONFLICT = [
    ("Goat Fuel", "Sioux City"),
    ("Lofbergs", "Maxim"),
    ("Fonter", "Lanjaron"),
    ("La Casera", "May Tea"),
    ("Aquabona", "Royal Bliss"),
    ("Bifrutas", "Pascual"),
    ("Coteaux Nantais", "Planet bio"),
    ("Alcampo", "Via Nature"),
    ("Lambda", "Mondariz"),
    ("Eco", "Int-Salim"),
    ("Primavera", "Tampico"),
    ("Booster", "Faxe"),
    ("Kitu Super Espresso", "Sunniva"),
    ("3D", "Blue Energy"),
    ("K A", "Ka"),
    ("Eloa", "Aloe Drink for Life"),
]

# Seeded map, frozen as the SSOT content at seed time (8 entries; every key
# token's measured territory is exactly its own family).
EXPECTED_KEYS = ("accelerator", "adrenaline", "ball", "fitaid", "hi", "kevytolo", "techusa", "ting")
EXPECTED_TARGETS = {"shoc", "dg", "hiball", "lifeaid", "olvi", "biotech"}


# (a) ── both directions of each measured alias pair ────────────────────────
def test_every_seeded_pair_folds_consistently():
    """Both directions of every granted pair must fold to one shared key and
    be free of brand conflict."""
    for family, sources in FAMILIES.items():
        for i, left in enumerate(sources):
            for right in sources[i + 1:]:
                left_fold = normalize_brand(left)
                right_fold = normalize_brand(right)
                shared = left_fold & right_fold
                assert shared, (
                    f"{family} {left!r} <-> {right!r} does not share a token: "
                    f"{sorted(left_fold)} x {sorted(right_fold)}"
                )
                assert family in shared, (
                    f"{left!r}/{right!r} fold must reach family key {family!r}"
                )
                assert not brand_conflict(left_fold, right_fold), (
                    f"{family} {left!r}<->{right!r} still vetoed: "
                    "both-direction fold is not set-symmetric"
                )


def test_alias_map_crosses_only_through_its_targets():
    """Side-effect containment of the star map: each alias key folds ONLY to
    its family key, and no fold-either-direction raises a conflict on the
    post-fold sets."""
    assert tuple(sorted(brand_aliases())) == EXPECTED_KEYS
    assert set(brand_aliases().values()) == EXPECTED_TARGETS


# (b) ── the SSOT loads clean (schema errors would crash at import) ─────────
def test_vocabulary_loads_with_brand_aliases():
    vocab = vocabulary()
    raw = vocab["brand_aliases"]
    assert isinstance(raw, dict) and all(
        isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
        for k, v in raw.items()
    ), "brand_aliases must be a str->str normalized-token map"
    provenance = vocab["brand_aliases_provenance"]
    assert provenance["counts"]["alias_families"] == len(FAMILIES)
    assert provenance["counts"]["alias_entries"] == len(brand_aliases())
    assert provenance["measurement"].startswith("within-GTIN brand-variant")
    # provenance must record the measured decision numbers
    assert provenance["counts"]["within_group_brand_vetoes_before_seeding"] == 49


# (c) ── folding does NOT merge two clearly distinct brands ─────────────────
# ('coca'/'pepsi' do NOT occur in the measured list — per the seeding brief,
# fall back to still-conflicting members of the measured decline set).
@pytest.mark.parametrize(
    "left,right",
    [(a, b) for a, b in EXPECTED_CONFLICT],
)
def test_distinct_brands_still_conflict_after_folding(left, right):
    assert brand_conflict(normalize_brand(left), normalize_brand(right)), (
        f"The map folded {left!r} with {right!r} — a distinct-brand conflict "
        "must stay live (veto asymmetry: folds may only dissolve within-GTIN "
        "spurious vetoes of granted families)"
    )


def test_seed_kept_only_the_measured_entries():
    assert len(brand_aliases()) == 8


# (d) ── veto asymmetry guard: conflicts MUST still be recognized ───────────
def test_brand_conflict_evaluated_when_no_alias_applies():
    """No mapped token in either brand cell: full identity_conflict must hold
    the 'brand' dimension (the veto survives where no fold applies)."""
    frame_rows = [
        {
            "title": "Whatever Root Beer", "brand": "Fonter",
            "attributes": "Flavour: Vanilla", "volume": "", "barcode": "",
        },
        {
            "title": "Whatever Root Beer", "brand": "Lanjaron",
            "attributes": "Flavour: Vanilla", "volume": "", "barcode": "",
        },
    ]
    left, right = (row_identity(r) for r in frame_rows)
    reasons = identity_conflict(left, right)
    assert "brand" in reasons, (
        "a veto may never disappear for a pair the alias map is not about — "
        "folding must remain a one-sided relaxation, not a wholesale peace"
    )
    assert "brand" not in identity_conflict(
        row_identity({"title": "a", "brand": "A SHOC", "attributes": "", "volume": "", "barcode": ""}),
        row_identity({"title": "b", "brand": "Accelerator", "attributes": "", "volume": "", "barcode": ""}),
    ), "a seeded family pair should NOT raise a brand conflict post-seed"


def test_folding_never_creates_a_conflict():
    """Structural property: folds only ADD tokens, so fold(x) shares every
    token the base fold of x had — pair conflicts can only ever DEcrease; the
    whole map is inspected. (No dataset needed: the property holds for any
    pair of single-token brand sets before/after the fold.)"""
    from core.product_identity import brand_aliases as aliases

    for key, target in aliases().items():
        held = frozenset({key})
        folded = frozenset({key, target})
        # Any pair whose pre-state was already "no conflict" stays so.
        assert not brand_conflict(folded, folded | {target})
        assert brand_conflict(held, frozenset({"unrelated"})) == brand_conflict(
            folded, frozenset({"unrelated"})
        ), f"alias {key!r}->{target!r} minted a conflict against an outsider"
