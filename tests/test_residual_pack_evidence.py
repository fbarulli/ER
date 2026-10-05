"""Real residuals: size fragments and stale slugs cannot replace title counts."""
import pytest
from core.url_evidence import url_text
from pipeline import extract_all, extract_pack_from_title, extract_volume_from_title


@pytest.mark.parametrize('title,url,count', [
    ('Amy & Brian Original Coconut Water - Case of 6 / 33.8 oz',
     'https://www.target.com/p/amy-brian-original-coconut-water-case-of-6-33-8-oz/-/A-90209305', 6),
    ('Volcano Organic Lime Burst Juice - Case of 12 - 6.7 fl oz',
     'https://www.target.com/p/volcano-organic-lime-burst-juice-case-of-12-6-7-fl-oz/-/A-92437397',12),
    ('Vivaloe Coconut Aloe, 16.9 fl. oz., 12 Count',
     'https://www.amazon.com/Vivaloe-Coconut-Aloe-16-9-Count/dp/B00KAQN4CU',12),
    ('Hiball Energy Wild Berry (8 pack of 16 Fl Oz)', '',8),
])
def test_real_title_counts_reach_structured_input(title,url,count):
    assert extract_pack_from_title(title)[0] == count
    result = extract_all(title, '', '', url)
    assert result['pack_qty'] == count
    if 'Vivaloe' in title:
        assert 'pack_sources_disagree' in result['attribute_consistency_flags']


@pytest.mark.parametrize('slug,count,ml', [
    ('water-case-of-6-33-8-oz',6,1000),
    ('juice-case-of-12-6-7-fl-oz',12,198),
    ('water-case-of-12-16-9-oz',12,500),
    ('water-case-of-24-500ml',24,500),
])
def test_slug_preserves_count_and_decimal_volume(slug,count,ml):
    text = url_text('https://example.org/'+slug)
    assert extract_pack_from_title(text)[0] == count
    assert extract_volume_from_title(text)['volume_ml'] == ml


def test_url_remains_fallback_when_title_has_no_count():
    assert extract_all('Cola', '', '', 'https://example.org/cola-24-pack')['pack_qty'] == 24


@pytest.mark.parametrize('title,ml', [
    ('Organic juice concentrate 1 + 4, 200ml',200),
    ('Seven Teas - Case of 12 / 16 oz',473),
    ('Jones Soda Green Apple, 6 / 4 / 12 Oz',355),
    ('Juice 1 / 2 gal',1893),
])
def test_dilution_and_case_counts_do_not_shrink_package_volume(title,ml):
    assert extract_volume_from_title(title)['volume_ml'] == ml
