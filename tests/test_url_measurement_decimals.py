"""Audited slug decimals survive noise stripping without merging count lists."""

import pytest

from core.text import extract_volume_match
from core.url_evidence import url_text


def test_sierra_original_url_keeps_seven_point_five_ounces():
    # dataset.csv SKU76118049, title and attributes independently say 222ml.
    url = 'https://www.amazon.com/Sierra-Mist-Lemon-7-5oz-Packaging/dp/B071F77SQX/ref=sr_1_348?c=ts'
    cleaned = url_text(url)
    assert '7.5oz' in cleaned.split()
    assert extract_volume_match(cleaned)[:2] == (7.5, 'oz')


@pytest.mark.parametrize('slug, expected', [
    ('drink-0-8l', (.8, 'l')),
    ('drink-7.5oz', (7.5, 'oz')),
    ('drink-24-500ml', (500, 'ml')),
    ('drink-12x355ml', (355, 'ml')),
    # AMBIGUOUS, and asserted as the implementation actually resolves it.
    # `NN-M-unit` (M one or two digits) is a slug decimal by the documented
    # rule, so `12-12oz` reconstructs to 12.12 oz even though retailer slugs
    # also use that shape for "12 pack of 12 oz". The two are surface-
    # identical; only plausibility could separate them, and a plausibility
    # heuristic here would shift the gate census for no downstream gain.
    #
    # MEASURED 2026-10-06 on the real corpus: this never reaches canonical
    # records. All 12,326 values in canonical_records.volume_set are integers,
    # and the 39 real `12-12-fl-oz` slugs (RiteAid/Walmart "12-12-fl-oz-
    # 355-ml-cans") land as 355/2000/591/222 ml with NO
    # volume_sources_disagree flag — cross-source merging discards the
    # spurious value. test_canonical_volumes_are_whole_ml_locks_that in.
    ('drink-12-12oz', (12.12, 'oz')),
    ('drink-0-8literary', (None, None)),
    ('code4100-5604', (None, None)),
])
def test_slug_decimal_controls(slug, expected):
    assert extract_volume_match(url_text('https://example.org/' + slug))[:2] == expected


def test_canonical_volumes_are_whole_ml_locks_that_in():
    """The guarantee the slug-decimal rule is NOT allowed to break.

    Every canonical volume is a whole millilitre. A slug decimal that welded a
    pack count onto the unit size would surface here as a fractional value.
    This is the assertion that actually protects the gate; the parameterised
    case above only pins url_text's intermediate wording.
    """
    import pandas as pd
    from core.common import F
    # Resolve through the config SSOT, not a repo-relative literal: the test
    # must hold wherever pytest is invoked from.
    canon = pd.read_csv(F["canonical_records"], dtype=str, keep_default_na=False)
    for value in canon["volume_set"]:
        for token in str(value).strip("[]").split(","):
            token = token.strip()
            if not token:
                continue
            number = float(token)
            assert number == int(number), f"fractional canonical volume {token}"


def test_opaque_image_basename_cannot_invent_eight_litres():
    # dataset.csv SKU71073123; opaque media code, not a product slug.
    image = 'https://images-na.ssl-images-amazon.com/images/I/51AFLZI--8L._AC_US160_.jpg'
    assert extract_volume_match(url_text(image))[0] is None
    assert '500ml' in url_text('https://example.org/images/500ml._AC_US160_.jpg')


def test_sibling_tails_never_corroborate_and_zero_pads_survive():
    """The three reconstruction defects measured on real dataset.csv URLs.

    riteaid "67-6-fl-oz-2-qt-3-6-fl-oz-2-lt": both `NN-M` candidates matched
    their SIBLING's tail scan ("3-6-fl-oz" scans "6 fl oz") at gap 0.0 and
    flipped audited decimals (67.6->6, 33.8->8): cross-candidate self-
    corroboration. mathem.se "…-1-75l-1-75l-…": the duplicated mention read
    its twin's tail ("75 l") as a corroborator and printed 75l 75l.
    iperdrive.iper.it "r-pet-25-05-l": the decimal text was built from the
    ints, printing 25.5 for the 25.05 the blind rule always produced.
    """

    def check(url, expected):
        got = url_text(url)
        assert got == expected, f"\n  got:      {got!r}\n  expected: {expected!r}"

    check('https://iperdrive.iper.it/product/levissima-acqua-naturale-r-pet'
          '-25-05-l-acqua-naturale-levissima-50cl',
          'levissima acqua naturale pet 25.05 acqua naturale levissima 50cl')
    check('https://iperdrive.iper.it/product/levissima-acqua-frizzante-r-pet'
          '-25-05-l-50-levissima',
          'levissima acqua frizzante pet 25.05 levissima')
    check('https://www.mathem.se/varor/apelsinjuice/juice-apelsin-1-75l-1-75l'
          '-garant',
          'varor apelsinjuice juice apelsin 1.75l 1.75l garant')
    check('https://www.riteaid.com/shop/dr-browns-cream-soda-original-67-6-fl'
          '-oz-2-qt-3-6-fl-oz-2-lt-5057132',
          'dr browns cream soda 67.6 fl oz 2 qt 3.6 fl oz 2 lt')
    check('https://www.riteaid.com/shop/aqua-hydrate-water-alkalized-with'
          '-electrolytes-33-8-fl-oz-1-qt-1-8-oz-1-lt-0337585',
          'aqua hydrate water alkalized with electrolytes 33.8 fl oz 1 qt 1.8 oz 1 lt')


def test_pack_reading_flips_only_on_a_free_standing_corroborator():
    """The pack flip the reconstruction exists for, on real slugs, and the
    audited decimals sharing their slug with corroborating mentions stay."""
    from core.url_evidence import url_text as read

    # RiteAid 12-pack: the independent 355-ml mention does the flipping.
    assert read('https://www.riteaid.com/shop/sierra-mist-soda-lemon-lime-12'
                '-12-fl-oz-355-ml-cans-144-fl-oz-4-26-lt-5067720') == (
        'sierra mist soda lemon lime 12 fl oz 355 ml cans 144 fl oz 4.26 lt')
    # Walmart 8-pack of 16-oz cans vs the pre-fix 8.16-oz reading.
    assert read('https://www.walmart.com/ip/Venom-Energy-Drink-Black-Cherry'
                '-Kiwi-16fl-oz-Pack-of-8-16-oz-Marked-Cans/991367607?'
                'classType=REGULAR') == (
        'venom energy drink black cherry kiwi 16fl oz pack of 16 oz marked cans')
