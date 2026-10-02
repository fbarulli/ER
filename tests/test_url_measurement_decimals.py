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
    ('drink-12-12oz', (12, 'oz')),
    ('drink-0-8literary', (None, None)),
    ('code4100-5604', (None, None)),
])
def test_slug_decimal_controls(slug, expected):
    assert extract_volume_match(url_text('https://example.org/' + slug))[:2] == expected


def test_opaque_image_basename_cannot_invent_eight_litres():
    # dataset.csv SKU71073123; opaque media code, not a product slug.
    image = 'https://images-na.ssl-images-amazon.com/images/I/51AFLZI--8L._AC_US160_.jpg'
    assert extract_volume_match(url_text(image))[0] is None
    assert '500ml' in url_text('https://example.org/images/500ml._AC_US160_.jpg')
