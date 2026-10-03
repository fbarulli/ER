from dashboard.app import _gate_date_context_html


def test_review_keeps_ambiguous_dates_and_duration_visible():
    html = _gate_date_context_html([{"sku_id": "123", "sku_name_eng": "Best before 10/03/2023",
                                     "description_short_eng": "Shelf Life: 730 Days"}])
    assert "2023-03-10 or 2023-10-03" in html
    assert "730 days" in html
    assert "10/03/2023" in html
    assert "deciding gate clause is shown separately" in html
    assert "no collection timestamp" in html


def test_listing_identifier_is_escaped_and_missing_date_adds_no_card():
    html = _gate_date_context_html([{"sku_id": "<script>alert(1)</script>",
                                     "sku_name_eng": "Best BY: 11-13-2024"}])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert _gate_date_context_html([{"sku_name_eng": "Water 330ml"}]) == ""
