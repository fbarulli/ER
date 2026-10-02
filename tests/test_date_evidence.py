from core.date_evidence import extract_date_evidence


def test_original_expiry_forms_keep_exact_spans_and_order():
    for text, normalized in [("Best BY: 11-13-2024", "2024-11-13"),
                             ("best before 2018-12-31", "2018-12-31"),
                             ("best before 26.3.2024", "2024-03-26")]:
        entry, = extract_date_evidence(text)
        assert entry["role"] == "expiry"
        assert entry["normalized_candidates"] == [normalized]
        assert entry["parse_status"] == "parsed"
        assert text[entry["start"]:entry["end"]] == entry["raw_match"]


def test_ambiguous_date_is_not_guessed():
    entry, = extract_date_evidence("Best before 10 / 03 / 2023")
    assert entry["normalized_candidates"] == ["2023-03-10", "2023-10-03"]
    assert entry["parse_status"] == "ambiguous_order"


def test_month_precision_and_manufacture_role():
    entry, = extract_date_evidence("Best before 08.2020")
    assert entry["normalized_candidates"] == ["2020-08"]
    assert entry["precision"] == "month"
    entry, = extract_date_evidence("Manufactured 2020-03-18")
    assert entry["role"] == "manufacture"


def test_invalid_and_uncued_dates_stay_reviewable():
    entry, = extract_date_evidence("best before 2023-02-31")
    assert entry["parse_status"] == "invalid"
    entry, = extract_date_evidence("tarih-18-03-2020")
    assert entry["role"] == "unspecified_calendar_date"
    assert entry["gate_use"] == "stock_review_context"
    assert extract_date_evidence("cola 24 x 330 ml; lot 12345678") == []


def test_calendar_month_tail_is_not_a_second_date():
    assert len(extract_date_evidence("best before 18-03-2020")) == 1


def test_shelf_life_and_expiry_reference_do_not_invent_dates():
    entry, = extract_date_evidence("Shelf Life: 730 Days 25.4 Ounce")
    assert entry["role"] == "shelf_life"
    assert (entry["duration_value"], entry["duration_unit"]) == (730, "day")
    assert not entry["normalized_candidates"]
    entry, = extract_date_evidence("Best Before: (See Base)")
    assert entry["role"] == "expiry_reference"
    assert entry["parse_status"] == "reference_only"


def test_short_year_preserves_century_uncertainty():
    entry, = extract_date_evidence("Best before: 10 / 21")
    assert entry["parse_status"] == "ambiguous_components"
    assert entry["partial_candidates"]        # day/month kept, no year invented
    assert not entry["normalized_candidates"]


def test_date_format_guidance_does_not_claim_delivery_or_weight_as_expiry():
    for tail in ("8-12 DAYS DELIVERY", "52.79fl oz", "0.93lbs"):
        entries = extract_date_evidence("not Best Before / Expiration UK is DD / MM / YYYY\n" + tail)
        assert all(entry["role"] == "date_format_reference" for entry in entries)
        assert all(not entry["normalized_candidates"] for entry in entries)
    assert all(entry["parse_status"] == "reference_only" for entry in extract_date_evidence("Best before 8-12 DAYS DELIVERY"))
