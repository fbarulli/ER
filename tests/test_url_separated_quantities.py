import pytest

from core.text import extract_volume_match
from core.url_evidence import url_text
from pipeline import extract_pack_from_title


@pytest.mark.parametrize("slug,expected", [("cola-330-ml", (330., "ml")),
                                         ("cola-12-fl-oz", (12., "fl oz")),
                                         ("cola-23.7-fl-oz", (23.7, "fl oz"))])
def test_separated_quantity_keeps_explicit_unit(slug, expected):
    text = url_text("https://example.org/" + slug + "/123456789")
    assert "123456789" not in text
    assert extract_volume_match(text)[:2] == expected


def test_explicit_pack_survives_while_standalone_id_is_removed():
    assert extract_pack_from_title(url_text("https://example.org/cola-24-pack/123456789"))[0] == 24
    assert url_text("https://example.org/cola/123456789") == "cola"


def test_media_dimensions_and_hashes_do_not_become_quantities():
    text = url_text("https://example.org/cache/1/small_image/220x/9df78eab33525d08d6e5fb8d27136e95/0")
    assert not text


@pytest.mark.parametrize("sku_url", ["https://example.org/shop/12x1-mineralwasser",
                                 "https://example.org/shop/12x1-pet-bottles",
                                 "https://example.org/shop/12x1 mineralwasser"])
def test_url_pack_span_requires_a_recognized_unit(sku_url):
    from pipeline import extract_pack_evidence, extract_pack_from_title
    text = url_text(sku_url)
    evidence = extract_pack_evidence(text)
    # No multiplier may fire without a shared recognized measurement unit, and
    # no raw span may swallow the first letter of the next word ('12x1 m', '12x1 p').
    assert not [e for e in evidence if e.get('rule') == 'multiplier'], evidence
    assert extract_pack_from_title(text) == (1, 0.0)


def test_url_multiplier_with_unit_still_counts():
    from pipeline import extract_pack_evidence
    evidence = extract_pack_evidence(url_text('https://example.org/shop/12x355ml-pet'))
    unit = [e for e in evidence if e.get('rule') == 'multiplier']
    assert unit and unit[0]['count'] == 12 and unit[0]['raw_match'].endswith('ml')


def test_decimal_pack_notation_survives_url_evidence():
    """Measured 2026-10-02: 26 sku_url rows carry decimal pack shapes
    ('6x1.5l', '4x0.25l', '12x50.7oz', '12x0.33l'); the bare-isdigit part
    check dropped them as media codes, deleting the URL's pack evidence."""
    from core.url_evidence import _has_unit_suffix, _spec, url_text
    spec = _spec()
    for token in ('6x1.5l', '4x0.25l', '12x50.7oz', '12x0.33l', '6x17.5cl', '8x0.25l'):
        assert _has_unit_suffix(token, spec), token
        assert url_text('https://x.example/' + token) == token
    assert _has_unit_suffix('12x33cl', spec)           # integer shape still exact
    assert not _has_unit_suffix('220x1280', spec)      # media dimensions stay dead
    assert not _has_unit_suffix('6x1.5x', spec)        # unitless suffix stays dead


def test_decimal_url_pack_evidence_is_retrievable():
    from pipeline import extract_pack_from_title
    assert extract_pack_from_title(url_text('https://example.org/pack-zilia-6x1.5l'))[0] == 6
    assert extract_pack_from_title(url_text('https://example.org/pack-zilia-12x0.33l'))[0] == 12


@pytest.mark.parametrize('slug,unit_end', [('link-24x330ml.html', 'ml'),
                                           ('uht-3x220ml.html', 'ml'),
                                           ('zilia-12x1l.article_id=2', 'l')])
def test_multiplier_span_dies_before_extension_separator(slug, unit_end):
    """The tail may not swallow the '.' before the next section: a raw span
    ending on a unit letter must never consume a URL extension separator
    ('24x330ml.html' previously yielded the raw span '24x330ml.')."""
    from pipeline import extract_pack_evidence
    evidence = extract_pack_evidence(slug)
    span = [e for e in evidence if e.get('rule') == 'multiplier']
    assert span and span[0]['raw_match'].endswith(unit_end) \
        and not span[0]['raw_match'].endswith('.'), evidence
