"""tests/test_url_evidence.py — the product text a listing URL really carries.

Defect this pins (owner ruling 2026-10-01). `url` and `image_url` were ruled
OUT of the per-title evidence capture with the stated reason "listing
identifiers, no product semantics". That reason was never checked against a
value. Reading dataset.csv:

    walmart.com/ip/Concord-Foods-Smoothie-Banana-Drink-Mixes-2-oz-Shelf-Stable
    riteaid.com/shop/sierra-mist-soda-lemon-lime-12-12-fl-oz-355-ml-cans-144-fl-oz

The slug is the retailer's own slugification of the product name and carries
flavour, volume and pack tokens verbatim. Excluding the column discarded real
evidence on the strength of an assumption. All 13 columns are now captured;
this module is the reader for the two URL columns.

A URL is product text MIXED WITH retailer scaffolding, so the reader is pinned
on both halves. Keeping a hash is as much a defect as dropping the column: a
surviving "9df78eab33525d08d6e5fb8d27136e95" becomes a confident-looking token
that matches nothing — and these tests caught four such leaks in the
implementation itself (uuid fragments, "220x", "k6rmm", "nan"), each of which
is asserted below so it cannot come back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.url_evidence import PATH_SCHEMA_WORDS, is_evidentiary, url_text


def test_reads_product_tokens_out_of_a_product_url() -> None:
    text = url_text(
        "https://www.riteaid.com/shop/sierra-mist-soda-lemon-lime-12-12-fl-oz"
        "-355-ml-cans-144-fl-oz-4-26-lt-5067720"
    )
    assert "sierra" in text and "mist" in text
    assert "lemon" in text and "lime" in text
    assert "ml" in text and "cans" in text


def test_keeps_the_walmart_ip_slug() -> None:
    text = url_text(
        "https://www.walmart.com/ip/Concord-Foods-Smoothie-Banana-Drink-Mixes"
        "-2-oz-Shelf-Stable/17619697?classType=REGULAR"
    )
    assert "concord" in text and "banana" in text
    assert "shelf" in text and "stable" in text


@pytest.mark.parametrize(
    "token",
    [
        "ip", "shop", "media", "catalog", "cache", "small", "image", "seo",
        "product", "www", "com",
    ],
)
def test_drops_storefront_scaffolding(token: str) -> None:
    assert token in PATH_SCHEMA_WORDS
    assert token not in url_text(f"https://www.example.com/{token}/x/product-name").split()


@pytest.mark.parametrize(
    "sku_url",
    [
        # riteaid: all scaffolding + one long content hash -> nothing left
        "https://www.riteaid.com/shop/media/catalog/product/cache/1/small_image"
        "/220x/9df78eab33525d08d6e5fb8d27136e95/0",
        # peapod: two-char media code + a random filename stem
        "https://i5.peapod.com/c/K6/K6RMM.jpg",
    ],
)
def test_hash_only_url_yields_nothing_not_noise(sku_url: str) -> None:
    """Empty is the correct answer. A surviving hash is worse than no column."""
    assert url_text(sku_url) == ""
    assert not is_evidentiary(sku_url)


@pytest.mark.parametrize(
    "leaked",
    ["d6bc7f7c", "0e12", "4c1c", "b86f", "2de2f2d3", "9df78eab", "220x",
     "k6rmm", "17619697", "5067720", "c"],
)
def test_no_hash_or_id_survives(leaked: str) -> None:
    """The four leaks this module shipped with, pinned as regressions."""
    urls = [
        "https://i5.walmartimages.com/seo/Concord-Foods-Smoothie-Banana-Drink"
        "-Mix-2-oz_d6bc7f7c-0e12-4c1c-b86f-2de2f2d3",
        "https://www.riteaid.com/shop/media/catalog/product/cache/1/small_image"
        "/220x/9df78eab33525d08d6e5fb8d27136e95/0",
        "https://i5.peapod.com/c/K6/K6RMM.jpg",
        "https://www.walmart.com/ip/Concord-Foods-Smoothie-Banana-Drink-Mixes"
        "-2-oz-Shelf-Stable/17619697?classType=REGULAR",
        "https://www.riteaid.com/shop/sierra-mist-soda-lemon-lime-12-12-fl-oz"
        "-355-ml-cans-144-fl-oz-4-26-lt-5067720",
    ]
    for url in urls:
        assert leaked not in url_text(url).split(), f"{leaked} survived {url}"


def test_walmart_image_slug_keeps_its_words_but_not_its_uuid() -> None:
    """Partially evidentiary: the slug is real, the hash is not."""
    text = url_text(
        "https://i5.walmartimages.com/seo/Concord-Foods-Smoothie-Banana-Drink"
        "-Mix-2-oz_d6bc7f7c-0e12-4c1c-b86f-2de2f2d3"
    )
    assert "concord" in text and "banana" in text
    assert "d6bc7f7c" not in text


@pytest.mark.parametrize("missing", [None, "", "   ", float("nan"), np.nan])
def test_missing_is_empty_not_the_string_nan(missing: object) -> None:
    """The raw export's missing cells arrive as NaN; str(NaN) is "nan"."""
    assert url_text(missing) == ""
    assert not is_evidentiary(missing)


def test_query_string_and_fragment_are_dropped() -> None:
    text = url_text("https://example.com/p/cola-zero-sugar?classType=X&pid=99#top")
    assert text == "cola zero sugar"


def test_deterministic() -> None:
    # "250ml" is KEPT. This line used to assert the opposite
    # ("red bull energy drink pack"): the media-code rule deleted any
    # letter+digit token, which ate the size token — 34 occurrences of
    # "250ml" and 9 of "2l" in 8,000 sampled sku_url slugs. Size evidence
    # deleted as if it were a retailer media code.
    url = "https://example.com/p/red-bull-energy-drink-250ml-24-pack"
    assert url_text(url) == url_text(url) == "red bull energy drink 250ml 24 pack"


def test_size_tokens_survive() -> None:
    """The pack gate reads size; the URL reader must not delete it."""
    for url, expected in (
        ("https://x.com/p/cola-250ml", "cola 250ml"),
        ("https://x.com/p/cola-2l", "cola 2l"),
        ("https://x.com/p/juice-12x355ml", "juice 12x355ml"),
        ("https://x.com/p/juice-355ml", "juice 355ml"),
        # pack notation with a nested size: 12 x (8 x 355ml)
        ("https://x.com/p/juice-12x8x355ml", "juice 12x8x355ml"),
    ):
        assert url_text(url) == expected, f"{url} lost its size token"


def test_long_real_words_are_not_hashes() -> None:
    """The bare-hash rule needs a DIGIT, not just length.

    It dropped any 8+ char alphanumeric run, which matched ordinary product
    words: "sparkling" (70x) and "strawberry" (48x) across 8,000 sampled
    sku_url slugs. Length alone does not make a token random.
    """
    for word in ("sparkling", "strawberry", "packaging", "blueberry"):
        assert word in url_text(f"https://x.com/p/cola-{word}-330ml").split(), (
            f"{word} was deleted as a hash"
        )
    # ...while real hashes and media codes still die
    for junk in ("9df78eab33525d08", "d6bc7f7c", "k6rmm", "220x"):
        assert junk not in url_text(f"https://x.com/media/cache/{junk}/cola").split()


def test_never_raises_on_junk() -> None:
    for junk in ("http://", "https://a.b/", "///", "://x", 12345, object()):
        assert isinstance(url_text(junk), str)


def test_no_double_spaces_or_trailing_space() -> None:
    text = url_text("https://example.com/p/cola--zero___sugar__330ml")
    assert "  " not in text and text == text.strip()


# ── measured yield on the real export ────────────────────────────────────────


def test_yield_on_the_real_export() -> None:
    """The claim this module rests on, measured rather than asserted."""
    from core.common import DATA_PATH

    if not DATA_PATH.exists():
        pytest.skip("raw export not present in this environment")
    df = pd.read_csv(DATA_PATH, dtype=str, low_memory=False)
    sample = df["sku_url"].dropna().sample(2000, random_state=0)
    texts = [url_text(u) for u in sample]
    informative = sum(1 for t in texts if t)
    # MEASURED 1651/2000 = 82.6%. The floor is that measured truth, not a
    # round number: an earlier 94.4% figure counted scaffolding and hashes as
    # product text, which is the failure this module exists to prevent.
    # Every empty result was checked and is a genuinely text-free URL — 179
    # amazon /dp/<ASIN> and 170 EAN/UUID-keyed coop/meijer/wegmans links.
    assert informative / len(texts) > 0.80, (
        f"only {informative}/{len(texts)} urls yielded product text"
    )
    # and it must be more than a couple of tokens when it fires
    fired = [t for t in texts if t]
    assert sum(len(t.split()) for t in fired) / len(fired) >= 3.0
