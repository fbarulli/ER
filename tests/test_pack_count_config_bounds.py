"""Reconciliation pack extraction uses the config-owned plausible range."""

from types import SimpleNamespace

from core import text


def test_current_source_pack_count_no_longer_raises():
    # dataset.csv sku_id=68338116: the simple inner pack is observed here;
    # the nested product is separately handled by extract_pack_from_title.
    title = 'Scheckters | Sparkling Green Tea & Mint | 2 x 12 x 250ml (UK)'
    assert text.extract_pack_counts(title) == {12}


def test_pack_count_uses_configured_inclusive_bounds(monkeypatch):
    active = text._unit_spec()
    spec = SimpleNamespace(**active.model_dump())
    spec.volume = active.volume
    spec.pack_min, spec.pack_max = 3, 7
    monkeypatch.setattr(text, '_unit_spec', lambda: spec)
    text._volume_views.cache_clear()
    try:
        assert text.extract_pack_counts('2 pack, 3 pack, 7 pack, 8 pack') == {3, 7}
    finally:
        text._volume_views.cache_clear()
