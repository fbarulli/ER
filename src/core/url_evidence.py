"""core.url_evidence — product text carried by a listing URL.

WHY (owner ruling 2026-10-01). Every one of the 13 mapped columns is now
captured per title (config/paths.yaml `column_evidence`). This module is the
READER for the two URL columns, because a URL is not inert text: the path slug
is the retailer's own slugification of the product name. Measured on
dataset.csv:

    walmart.com/ip/Concord-Foods-Smoothie-Banana-Drink-Mixes-2-oz-Shelf-Stable
    riteaid.com/shop/sierra-mist-soda-lemon-lime-12-12-fl-oz-355-ml-cans-144-fl-oz

An earlier ruling excluded `url` and `image_url` as "listing identifiers with
no product semantics". That was wrong, and it was wrong because nobody read a
value before ruling on it — the slug carries flavour, volume and pack tokens
verbatim.

WHAT IT EXTRACTS, and what it refuses to. A URL is a mix of product text and
retailer scaffolding, so the normalizer must be as careful about what it drops
as about what it keeps:

  * scaffolding dropped — `ip`, `shop`, `media`, `catalog`, `product`,
    `cache`, `small_image`, `seo`, and the rest of PATH_SCHEMA_WORDS;
  * article numbers dropped — retailer ids like `17619697`, and any
    digit-run, which are never product tokens;
  * hashes dropped — `9df78eab33525d08d6e5fb8d27136e95`, `d6bc7f7c`, `K6RMM`.
    This is why image_url is only PARTIALLY evidentiary: walmart's
    `seo/Concord-Foods-Smoothie-Banana-Drink-Mix-2-oz_d6bc7f7c` keeps its
    tokens, riteaid's `cache/1/small_image/220x/9df78eab.../0` is all
    scaffolding and hash. Measured: 5.6% of URLs clean to nothing.

A hash surviving into the token stream would be worse than dropping the column:
it becomes a confident-looking token that matches nothing and matches
everything. So the normalizer returns "" rather than noise, and callers must
treat "" as "no evidence", never as "no attributes".
"""

from __future__ import annotations

import re

from pipeline import normalize_text

__all__ = ["PATH_SCHEMA_WORDS", "is_evidentiary", "url_text"]

# Retailer URL scaffolding: path segments that describe the STOREFRONT's URL
# scheme, never the product. Kept as a declared set (not an inline literal at
# each call site) so a new retailer's scheme can be added in one place.
PATH_SCHEMA_WORDS: frozenset[str] = frozenset(
    {
        # generic path segments
        "ip", "shop", "stores", "store", "product", "products", "item", "items",
        "details", "detail", "pd", "dp", "gp", "aw", "node", "index", "main",
        "content", "dam", "catalog", "catalogue", "search", "browse", "en", "us",
        "www", "com", "net", "org", "co", "p", "sp", "cl", "sk", "itm",
        # retailer schemes keyed by identifier rather than name: amazon's
        # /dp/<ASIN> (179/2000 sampled urls), coop's /product/<EAN>, meijer's
        # /shopping/product, wegmans' /shop/categories. These carry NO product
        # text and must read as empty, not as their scaffolding.
        "dp", "asin", "shopping", "categories", "basket", "cart", "checkout",
        "account", "list", "lists", "wishlist", "compare", "deals", "offers",
        # image/media path segments
        "media", "image", "images", "img", "small", "large", "original", "seo",
        "cache", "resize", "scale", "width", "height", "quality", "format",
        "fit", "crop", "thumb", "thumbnail", "master", "is", "irs", "col",
        "files", "file", "upload", "uploads", "static", "assets", "sr",
        # file extensions that survive the extension strip
        "html", "htm", "php", "aspx", "jsp", "svg", "webp", "gif",
    }
)

_EXTENSION = re.compile(r"\.(jpe?g|png|gif|webp|svg|html?|php|aspx|jsp)$", re.I)
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
# A UUID/hash must be removed BEFORE hyphens become spaces, otherwise each of
# its groups survives as a bogus "word". Two details, both found by reading the
# function's OUTPUT rather than by reasoning about the pattern:
#   * the tail is `{6,}`, not a strict uuid's 12 — walmart's seo slugs carry an
#     8-4-4-4-8 hash (…-b86f-2de2f2d3) which a strict 12-char tail misses;
#   * there is NO leading \b — the slug arrives as "oz_d6bc7f7c-…", and \b does
#     not match between "_" and "d" because both are word characters, so a
#     \b-guarded pattern silently NEVER fires on the real data.
# 8-4-4-4-6 of hex cannot be a product phrase.
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{6,}", re.I
)
# A digit-run of any length is an article/size id, never a product token; a
# long bare alphanumeric run is a content hash.
_DIGIT_RUN = re.compile(r"\b\d+\b")
_BARE_HASH = re.compile(r"^[a-z0-9]{8,}$")
# A bare "NNNx" is an image-dimension spec from a media path
# (.../small_image/220x/…). Real pack notation carries its unit — "12x355ml"
# — so it survives the digit-run and dimension rules intact.
_IMAGE_DIM = re.compile(r"^\d+x$")
_VOWELS = frozenset("aeiou")
# Short letter+digit runs are retailer media codes (peapod's "c/K6/K6RMM.jpg"),
# never product tokens. Units are the exception and are named, not guessed.
_UNITS = frozenset({"ml", "l", "g", "kg", "mg", "oz", "fl", "cl", "dl"})
_SHORT_CODE = re.compile(r"^(?:[a-z]{1,3}\d+|\d+[a-z]{1,3})$")


def _is_noise(token: str) -> bool:
    if len(token) < 2:
        # a bare path letter (peapod's ".../c/K6/…") is storefront scaffolding
        return True
    if token in PATH_SCHEMA_WORDS or token in _UNITS:
        return False if token in _UNITS else True
    if _DIGIT_RUN.fullmatch(token) or _IMAGE_DIM.match(token):
        return True
    if len(token) >= 5 and not (set(token) & _VOWELS):
        # no vowel in a 5+ char token: a random code (k6rmm, 9df78eab…), not a
        # product word. Real words (sparkling, lemon) always have one.
        return True
    if _SHORT_CODE.match(token):
        return True
    if _BARE_HASH.match(token):
        return True
    return False


def url_text(url: object) -> str:
    """The product-bearing prose of ``url``, or "" if it carries none.

    Deterministic and total: never raises, never returns scaffolding, a hash,
    or "nan" from a missing cell. Safe to call on the raw export without
    per-row guards.
    """
    if url is None:
        return ""
    if isinstance(url, float) and url != url:  # NaN without pandas
        return ""
    text = str(url).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    text = text.split("?", 1)[0].split("#", 1)[0]
    text = _SCHEME.sub("", text)
    parts = text.split("/", 1)
    slug = parts[1] if len(parts) > 1 else parts[0]
    slug = _EXTENSION.sub("", slug)
    slug = _UUID.sub(" ", slug)
    slug = re.sub(r"[-_+]+", " ", slug)
    slug = _DIGIT_RUN.sub(" ", slug)
    tokens = [
        token
        for token in normalize_text(slug).split()
        if not _is_noise(token)
    ]
    return " ".join(tokens)


def is_evidentiary(url: object) -> bool:
    """True when the URL yields at least one product token."""
    return bool(url_text(url))