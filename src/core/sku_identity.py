"""src/core/sku_identity.py — ONE product-identity decision, SSOT.

Every label-forming surface (dedupe T1.5, the gate, the vetoes) has to answer
the same question: are these two catalog rows the SAME product? Until now each
lane answered it with its own ad-hoc comparison, and the audit on 2026-09-30
measured what that costs, using gtin-labeled ground truth:

  GT-POS  17,753 cross-retailer same-GTIN pairs (same product by definition)
          -> the old comparison called 3,369 of them a CONFLICT (19.0%)
  GT-NEG  13,517 within-retailer different-GTIN pairs (different product)
          -> the old comparison called 3,965 of them COMPATIBLE (29.3%)

Both error rates are EXTRACTION defects, not missing fields: on every sampled
cause both sides' fields were populated (0% empty), the two rows just read the
same product two different ways ("10.5 fl oz" -> 200 ml here and 311 ml there;
"Peet's" / "Peet S" / `Peet""S`; "Black & White" vs "Black and White"; one
listing says "Sugar Free" and its sibling simply omits the claim).

Four rules fix the class, and they are the whole module:

1.  SETS, NEVER COUNTS. A descriptor that appears in the title AND the
    attribute cell AND the category path is ONE token. Union before comparing,
    so a repeated descriptor can neither manufacture a conflict nor
    manufacture a match, and token order and repetition never matter.

2.  ABSENCE IS NOT CONTRADICTION. A veto fires only when BOTH sides carry
    positive evidence and the evidence disagrees. The old asymmetry — one
    listing saying "Sugar Free" while the other omits it — produced 754 false
    vetoes on proven-same pairs, and an empty field on one side made the pair
    compatible with EVERYTHING, which is what let single-linkage chaining
    merge whole flavour families (the 29.3%).

3.  PRICE AND URLS ARE NOT IDENTITY. `price` moves with the seller, and
    `url`/`image_url` are export noise (the corpus carries a "Fashion
    Accessories" spam title under a real gtin). Neither may decide identity
    or rank a representative. Completeness is scored over descriptor columns
    only.

4.  AN ELIGIBLE GTIN OUTRANKS DESCRIPTORS. Reviewed identity holds remain
    ineligible even with valid check digits. `gtin_validity` decides whether a
    gtin may be trusted as identity at all; when both rows carry a trusted
    gtin the gtin IS the answer and no text comparison is consulted.

Aliases are data, not code: `config/vocabulary.json` (via the validated
`core.common.vocabulary` SSOT) owns the brand map and the concept folds, and
`core.critical_attributes.FLAVOR_ALIASES` owns flavor folding. A wrong alias
here merges two real products, which is the expensive direction, so the maps
are explicit reviewed pairs and never edit distance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping

import pandas as pd

from core.common import data_cfg, training_cfg, vocabulary
from core.critical_attributes import (
    DECLARED_FLAVOR_LEXICON,
    FLAVOR_ALIASES,
    FLAVOR_LEXICON,
    volumes_compatible,
)
# Aliased: this module exports its own `categorical_conflict` with an
# alias-expanded signature, and a bare import of the same name would recurse
# into itself (caught by the acceptance run, not by review).
from core.critical_attributes import categorical_conflict as _ssot_categorical_conflict
from core.record_linkage import strip_pack_multiplicity
from core.product_dimensions import DimensionEvidence, row_dimensions, evaluate_dimensions, evaluate_columns
from pipeline import normalize_text

# ── descriptor columns ─────────────────────────────────────────────────────
# The columns that DESCRIBE a product. `price`, `url`, `image_url` and
# `country` are excluded on purpose (rule 3).
DESCRIPTOR_COLUMNS: tuple[str, ...] = tuple(data_cfg().descriptor_columns)
NON_DESCRIPTOR_COLUMNS: frozenset[str] = frozenset(
    set(data_cfg().column_mapping.values()) - set(DESCRIPTOR_COLUMNS)
)

# The gate's 5% relative tolerance; `volumes_compatible` is the SSOT for the
# predicate, so reusing the number keeps "compatible" meaning one thing.
VOLUME_RELATIVE_TOLERANCE = float(training_cfg().gate.vol_tolerance)

# `sku_info` returns {1.0} as its explicit "no pack count observed" sentinel,
# so 1.0 is NOT evidence of a single-unit product.
PACK_SENTINEL = 1.0

# A POSITIVE claim that this listing is the reduced-sugar variant. Its ABSENCE
# is never evidence of the regular variant (rule 2) — only an explicit
# contradiction is. Mirrors training.dedupe._DIET_MARKER, widened to the
# attribute cell because a claim stated only there is still a claim.
DIET_CLAIM_RE = re.compile(
    r"\b(diet|zero|sugar\s*free|no\s*sugar|low\s*cal|unsweetened|"
    r"sugar\s*reduced)\b", re.I,
)
# An explicit statement that the product is the FULL-sugar / regular variant.
# Deliberately narrow: "original" and "classic" are naming, not a sugar claim,
# and they live in the flavor bundle instead.
SUGAR_CLAIM_RE = re.compile(
    r"\b(full\s*sugar|with\s*sugar|sugared|contains\s*sugar|not\s*diet)\b", re.I,
)

# Flavor words that name the SAME flavor in different words. Measured on GT-POS
# false vetoes: "Peet's Black & White" and "Peet's Iced Espresso Black and
# White" read as {dark} and {espresso} and vetoed a proven-same pair. Applied
# to the FLAVOR dimension only, so a package type is never folded by a flavor
# synonym.
QUALIFIER_ALIASES: dict[str, frozenset[str]] = {
    "black": frozenset({"dark"}),
    "dark": frozenset({"black"}),
    "regular": frozenset({"original"}),
    "original": frozenset({"regular"}),
    "ice": frozenset({"iced"}),
    "iced": frozenset({"ice"}),
    "sugarfree": frozenset({"diet", "sugarfree"}),
    "decaf": frozenset({"caffeinefree", "decaffeinated", "decaf"}),
    "caffeinefree": frozenset({"decaf", "decaffeinated", "caffeinefree"}),
    "decaffeinated": frozenset({"decaf", "caffeinefree", "decaffeinated"}),
}

# Corporate suffixes carrying no brand identity: stripped so "Stumptown Coffee
# Roasters" and "Stumptown" are one brand (335 measured brand false vetoes on
# proven-same pairs, mostly possessive and OCR variants).
_BRAND_SUFFIX = frozenset({
    "co", "inc", "incorporated", "ltd", "llc", "plc", "gmbh", "sa", "nv", "bv",
    "the", "and", "company", "corp", "corporation", "group", "brand", "brands",
})
_BRAND_NOISE_RE = re.compile(r"[^a-z ]+")
_ALPHA_RE = re.compile(r"[a-z]+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FLAVOR_FIELDS = (r"flavou?r", r"roast\s*type")

# Dimensions compared as categorical SETS after alias expansion. `pack` is
# handled numerically (it has a sentinel) and `volume` by the SSOT predicate.
CATEGORICAL_DIMENSIONS: tuple[str, ...] = (
    "flavor", "carbonation", "sweetener", "sweetener_type", "sweetening",
    "pulp", "package_type", "package_material",
)
_SWEETENER_FAMILY = frozenset({"sweetener", "sweetener_type", "sweetening"})


@dataclass(frozen=True)
class ProductIdentity:
    """The deduped descriptor bundle for one catalog row.

    Every field is a SET. That is the point: a descriptor repeated across
    title/attribute/category collapses to one token, so comparison cannot be
    moved by how many times a retailer restated it.
    """

    brand: frozenset[str] = frozenset()
    volume_ml: frozenset[float] = frozenset()
    pack: frozenset[float] = frozenset()
    flavor: frozenset[str] = frozenset()
    carbonation: frozenset[str] = frozenset()
    sweetener: frozenset[str] = frozenset()
    sweetener_type: frozenset[str] = frozenset()
    sweetening: frozenset[str] = frozenset()
    pulp: frozenset[str] = frozenset()
    package_type: frozenset[str] = frozenset()
    package_material: frozenset[str] = frozenset()
    diet_claim: bool = False
    sugar_claim: bool = False
    gtin_trusted: bool = False
    gtin_key: str = ""
    identity_review_reason: str = ""
    completeness: int = 0
    dimensions: DimensionEvidence | None = None

    def as_mapping(self) -> dict[str, set[Any]]:
        """Shape the bundle like a `sku_info` mapping for the SSOT predicates."""
        return {
            "volume": set(self.volume_ml), "pack": set(self.pack),
            "package_type": set(self.package_type), "flavor": set(self.flavor),
            "carbonation": set(self.carbonation), "sweetener": set(self.sweetener),
            "sweetener_type": set(self.sweetener_type),
            "sweetening": set(self.sweetening), "pulp": set(self.pulp),
        }


def graph_schema() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive (relations, numeric) straight from the extractor's own contract.

    The graph listing schema is NOT a hand-maintained subset: every
    `ProductIdentity` descriptor field whose values are strings is a typed
    relation, every float-valued field is a numeric feature, in declaration
    order. A descriptor added to the extractor (new field here, populated in
    `row_identity`) therefore enters the graph automatically, and a prepared
    listings file built with an older schema is refused at load time by
    comparing the manifest against this derivation — never silently stale.
    Non-descriptor bookkeeping (claims flags, gtin key/review reason,
    completeness, raw dimension evidence) is excluded by construction because
    it is not a set-descriptor field.
    """
    import dataclasses
    from typing import get_type_hints

    hints = get_type_hints(ProductIdentity)
    relations, numeric = tuple(), tuple()
    for field in dataclasses.fields(ProductIdentity):
        origin = hints[field.name]
        descriptor = getattr(origin, "__args__", ()) and origin.__args__[0]
        if getattr(origin, "__origin__", None) is frozenset and descriptor is str:
            relations += (field.name,)
        elif getattr(origin, "__origin__", None) is frozenset and descriptor is float:
            numeric += (field.name,)
    return relations, numeric


# ── vocabulary (validated SSOT) ────────────────────────────────────────────
@lru_cache(maxsize=1)
def brand_aliases() -> Mapping[str, str]:
    """Brand alias map, config-owned.

    Owner ruling (2026-09-29): brand variants are SEMANTIC (a rebrand, a sister
    brand), not typos, so this is an explicit reviewed map and NOT edit-distance
    fuzzy matching — a wrong pair here merges two real products.
    """
    raw = vocabulary().get("brand_aliases") or {}
    return {str(k): str(v) for k, v in raw.items() if k}


@lru_cache(maxsize=1)
def concept_folds() -> Mapping[str, str]:
    """`CONCEPT_FOLDS` from config: sparkling->carbonated, waters->water, ..."""
    return {str(k): str(v) for k, v in (vocabulary().get("CONCEPT_FOLDS") or {}).items()}


@lru_cache(maxsize=1)
def flavor_vocabulary() -> frozenset[str]:
    """Every token that may be read as a flavor, before alias folding."""
    return frozenset(FLAVOR_LEXICON) | frozenset(DECLARED_FLAVOR_LEXICON)


# ── normalization ──────────────────────────────────────────────────────────
@lru_cache(maxsize=65536)
def _normalize_brand_cached(raw: object) -> frozenset[str]:
    text = _BRAND_NOISE_RE.sub(" ", normalize_text(str(raw or "")))
    tokens = [t for t in _ALPHA_RE.findall(text) if t not in _BRAND_SUFFIX]
    if not tokens:
        return frozenset()
    folded = set(tokens)
    aliases = brand_aliases()
    for token in tokens:
        target = aliases.get(token)
        if target:
            # Conservative fold (veto-asymmetry doctrine, seeded 2026-09-30
            # from the within-GTIN measurement via scripts/seed_brand_aliases
            # .py): the target is ADDED, never swapped in — a fold can only
            # make two brand token sets share a token or nest, never go
            # disjoint, so brand-agreement evidence stops being spuriously
            # contradicted while real cross-brand vetoes keep firing
            # (measured: 49 within-group brand vetoes -> 22, all 27 dissolved
            # pairs being seeded alias families on one gtin).
            folded.add(target)
    return frozenset(folded)


def normalize_brand(raw: object) -> frozenset[str]:
    """Brand cell -> a comparable token set.

    "by Concord Foods" (the export prefixes some brands), "Peet's", `Peet""S`
    and "Peet S" all reduce to one key. The apostrophe and doubled-quote OCR
    damage are measured cases, not hypotheticals.

    Memoized: the token fold is a pure function of the cell and the same cell
    is re-read by the identity bundle and the conflict miner.
    """
    try:
        return _normalize_brand_cached(raw)
    except TypeError:
        return _normalize_brand_uncached(raw)


def _normalize_brand_uncached(raw: object) -> frozenset[str]:
    return _normalize_brand_cached.__wrapped__(raw)


def _alias_fold_impl(tokens: Iterable[str], *, qualifiers: bool = False) -> frozenset[str]:
    out: set[str] = set()
    add = out.add
    concepts = concept_folds()
    concepts_get = concepts.get
    flatten_get = FLAVOR_ALIASES.get
    for token in tokens:
        token = str(token).strip().lower()
        if not token:
            continue
        add(token)
        singular = flatten_get(token)
        if singular:
            add(singular)
        concept = concepts_get(token)
        if concept:
            add(concept)
        if qualifiers:
            out |= QUALIFIER_ALIASES.get(token, frozenset())
    return frozenset(out)


@lru_cache(maxsize=131072)
def _alias_fold_cached(tokens: frozenset[str], qualifiers: bool) -> frozenset[str]:
    return _alias_fold_impl(tokens, qualifiers=qualifiers)


def alias_fold(tokens: Iterable[str], *, qualifiers: bool = False) -> frozenset[str]:
    """Vocabulary folding: plural -> singular, concept -> canonical, synonyms.

    `qualifiers=True` additionally applies the flavor synonym map, which is
    deliberately NOT applied to package/carbonation dimensions.

    Memoized on the token SET (order and repetition are already irrelevant to
    the fold), keyed with the ``qualifiers`` flag; the result is a frozenset.
    """
    try:
        return _alias_fold_cached(frozenset(tokens), qualifiers)
    except TypeError:
        return _alias_fold_impl(tokens, qualifiers=qualifiers)


def _string_set(values: object) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, str):
        values = [values]
    return frozenset(str(v).strip().lower() for v in values if str(v).strip())


def _float_set(values: object) -> frozenset[float]:
    if values is None:
        return frozenset()
    if isinstance(values, (int, float)):
        values = [values]
    out: set[float] = set()
    for value in values:
        try:
            out.add(float(value))
        except (TypeError, ValueError):
            continue
    return frozenset(out)


def completeness(row: Mapping[str, Any] | Any) -> int:
    """Populated DESCRIPTOR fields (rule 3: url/image_url/price do not count).

    Picks which row of a duplicate group survives: the most informative
    listing, not the cheapest and not the one carrying the most URLs.
    """
    get = (lambda k: row.get(k, "")) if isinstance(row, Mapping) else (
        lambda k: getattr(row, k, "")
    )
    # One read per column: the two-access form read and stringified every
    # descriptor cell twice for a boolean.
    populated = 0
    for col in DESCRIPTOR_COLUMNS:
        value = get(col)
        if value is not None and not pd.isna(value) and str(value).strip():
            populated += 1
    return populated


def completeness_frame(frame: pd.DataFrame) -> pd.Series:
    """Descriptor counts with the same missing-value policy as completeness.

    Work column by column instead of materializing a dictionary per source row.
    Preserve the input index so counts align during representative selection.
    """
    scores = pd.Series(0, index=frame.index, dtype="int64")
    for column in DESCRIPTOR_COLUMNS:
        if column in frame:
            populated = frame[column].notna() & frame[column].fillna("").astype(str).str.strip().ne("")
            scores += populated.astype("int64")
    return scores


@lru_cache(maxsize=32)
def _attr_token_re(key: str) -> re.Pattern[str]:
    r"""`<key>\s*:\s*([^;]+)` compiled once per key.

    The key is the caller's field pattern (`flavou?r`, `roast\s*type`,
    `pack\s*material\s*type`), so the source text is constant per key: a
    per-key cache keeps the module-level `re.search` wrapper — and its string
    cache lookup — out of every row's attribute-cell scan.
    """
    return re.compile(key + r"\s*:\s*([^;]+)", re.I)


@lru_cache(maxsize=65536)
def _attr_token_set_cached(attr: object, key: str) -> frozenset[str]:
    match = _attr_token_re(key).search(str(attr or ""))
    if not match:
        return frozenset()
    return frozenset(_TOKEN_RE.findall(normalize_text(match.group(1))))


def attr_token_set(attr: object, key: str) -> frozenset[str]:
    """`Key: value; ...` cell -> token set, normalized THEN tokenized.

    Order matters: normalizing first is what makes "coffee, vanilla" and
    "vanilla coffee" the same set. Memoized; the result is a frozenset.
    """
    try:
        return _attr_token_set_cached(attr, key)
    except TypeError:
        return _attr_token_set_cached.__wrapped__(attr, key)


@lru_cache(maxsize=65536)
def _identity_tokens_set_cached(attributes: object) -> frozenset[str]:
    out: set[str] = set()
    for field_re in _FLAVOR_FIELDS:
        out |= _attr_token_set_cached(attributes, field_re)
    return frozenset(out)


def identity_tokens_set(attributes: object) -> frozenset[str]:
    """Declared flavor UNION roast-type tokens from the attribute cell.

    Roast is a product splitter that lives in `Roast Type`, not `Flavour`:
    "French Roast" and "Vanilla" both declare "Flavour: coffee", so flavor
    alone cannot separate them. Memoized; the result is a frozenset.
    """
    try:
        return _identity_tokens_set_cached(attributes)
    except TypeError:
        return _identity_tokens_set_cached.__wrapped__(attributes)


def _gtin_facts(gtin: object) -> tuple[bool, str]:
    """(trusted, normalized key) using structural validation and review policy.

    Deliberately NOT memoized: the reviewed-identity policy is allowed to
    change within a process (pinned by test_gtin_scalar_facts), so this must
    read the current policy on every call.
    """
    from core.gtin import normalize_gtin_value
    from core.identity_policy import held_keys

    key, valid = normalize_gtin_value(gtin)
    if not valid or key.zfill(14) in held_keys():
        return False, ""
    return True, "" if key is None else str(key)


@lru_cache(maxsize=65536)
def _row_dimensions_cached(items: tuple) -> DimensionEvidence:
    return row_dimensions(dict(items))


def _row_dimensions_memo(row: Mapping[str, Any] | Any) -> DimensionEvidence:
    """Memoized :func:`row_dimensions` for all-string mapping rows.

    Row-dimension parsing and context resolution are pure in the row's cells.
    Keying is restricted to rows whose values are all ``str`` so the cached
    call receives byte-identical inputs (the export path is dtype=str); any
    other row shape falls through to the uncached parser unchanged.
    """
    if isinstance(row, Mapping):
        items = tuple(row.items())
        if all(isinstance(value, str) for _, value in items):
            try:
                return _row_dimensions_cached(tuple(sorted(items)))
            except TypeError:
                pass
    return row_dimensions(row)


def row_identity(row: Mapping[str, Any] | Any) -> ProductIdentity:
    """Build the descriptor bundle for one catalog row.

    Accepts a Mapping or a pandas namedtuple. The extraction itself is the
    pipeline's own `sku_info` — this module never re-implements a parser, it
    only decides how parsed values are COMPARED.
    """
    from core.identity_policy import resolve_listing_row
    if isinstance(row, Mapping) or hasattr(row, "to_dict"):
        row = resolve_listing_row(dict(row) if isinstance(row, Mapping) else row.to_dict())
    get = (lambda k, d="": row.get(k, d)) if isinstance(row, Mapping) else (
        lambda k, d="": getattr(row, k, d)
    )
    title = str(get("sku_name_eng", "") or "")
    attributes = get("attribute", "")
    description = str(get("description_short_eng", "") or "")
    url = str(get("sku_url", "") or "")
    image_url = str(get("image_url", "") or "")
    category_text = " ".join(
        str(get(col, "") or "") for col in ("category", "breadcrumbs_eng")
    )

    from core.structured_features import sku_info

    info = sku_info(
        title, attributes, description,
        url=url, image_url=image_url,
        breadcrumbs_eng=str(get("breadcrumbs_eng", "") or ""),
        category=str(get("category", "") or ""),
    )
    declared = identity_tokens_set(attributes)
    title_tokens = set(_TOKEN_RE.findall(normalize_text(title)))
    flavor = alias_fold(set(info["flavor"]) | set(declared), qualifiers=True)
    # Title-side flavor words close the gap when the attribute cell declares
    # nothing (a listing that only says "Black & White" in its title).
    flavor |= alias_fold(title_tokens & flavor_vocabulary(), qualifiers=True)
    flavor |= alias_fold(title_tokens & declared, qualifiers=True)

    from core.identity_policy import review_reason, listing_review_reason
    held_reason = review_reason(get("gtin", "")) or listing_review_reason(get("sku_id", ""), get("gtin", ""))
    trusted, key = _gtin_facts(get("gtin", ""))
    haystack = " ".join(
        (title, str(attributes or ""), description, category_text)
    )
    return ProductIdentity(
        brand=normalize_brand(get("brand", "")),
        volume_ml=_float_set(info["volume"]),
        pack=_float_set(info["pack"]),
        flavor=frozenset(flavor),
        carbonation=alias_fold(info["carbonation"]),
        sweetener=alias_fold(info["sweetener"]),
        sweetener_type=alias_fold(info["sweetener_type"]),
        sweetening=alias_fold(info["sweetening"]),
        pulp=alias_fold(info["pulp"]),
        package_type=alias_fold(info["package_type"]),
        # Keyed on the REAL corpus key. `pack\s*material` compiles to
        # `pack\s*material\s*:`, which never matches `Pack Material Type:`
        # (the word "Type" intervenes) — package_material was silently always
        # empty across all 35,571 non-null cells. Verified against the
        # measured attribute-key census, 2026-09-30.
        package_material=alias_fold(
            attr_token_set(attributes, r"pack\s*material\s*type")
        ),
        diet_claim=bool(DIET_CLAIM_RE.search(haystack)),
        sugar_claim=bool(SUGAR_CLAIM_RE.search(haystack)),
        gtin_trusted=trusted and not held_reason,
        gtin_key=key,
        identity_review_reason=held_reason,
        completeness=completeness(row),
        dimensions=_row_dimensions_memo(row if isinstance(row, Mapping) else {
            name: get(name, "") for name in DESCRIPTOR_COLUMNS + tuple(NON_DESCRIPTOR_COLUMNS)
        }),
    )


# ── the decision ───────────────────────────────────────────────────────────
def brand_conflict(left: frozenset[str], right: frozenset[str]) -> bool:
    """Both brands present and neither explains the other (rule 2)."""
    if not left or not right or (left & right):
        return False
    # A listing may carry a longer brand string than its sibling ("Kiju" vs
    # "Kiju Organic"); one being a SUBSET of the other is truncation, not a
    # different brand.
    return not (left <= right or right <= left)


def categorical_conflict(
    dimension: str, left: frozenset[str], right: frozenset[str]
) -> bool:
    """Both sides carry values and the folded sets are DISJOINT.

    The sweetener family delegates to the SSOT predicate so the dedupe, the
    gate and the conflict miner cannot drift on what "sweetener conflict"
    means.
    """
    predicate_dimension = "sweetener" if dimension in _SWEETENER_FAMILY else dimension
    return _ssot_categorical_conflict(
        predicate_dimension, {predicate_dimension: set(left)}, {predicate_dimension: set(right)}
    )


def identity_conflict(
    left: ProductIdentity,
    right: ProductIdentity,
    *,
    volume_relative_tolerance: float = VOLUME_RELATIVE_TOLERANCE,
) -> list[str]:
    """Descriptor dimensions that PROVE two rows are different products.

    An empty list is NOT proof of sameness — it is the absence of a conflict,
    which is exactly why a caller still needs an identity key (a trusted
    gtin, or a reviewed adjudication) before collapsing anything.

    Rule 4: two trusted gtins settle it outright; the text is not consulted.
    """
    if left.gtin_trusted and right.gtin_trusted:
        return [] if left.gtin_key == right.gtin_key else ["gtin"]

    reasons: list[str] = []
    if brand_conflict(left.brand, right.brand):
        reasons.append("brand")
    if not volumes_compatible(
        left.volume_ml, right.volume_ml,
        volume_relative_tolerance=volume_relative_tolerance,
    ):
        reasons.append("volume")
    left_pack = {p for p in left.pack if p != PACK_SENTINEL}
    right_pack = {p for p in right.pack if p != PACK_SENTINEL}
    if left_pack and right_pack and not (left_pack & right_pack):
        reasons.append("pack")
    for dimension in CATEGORICAL_DIMENSIONS:
        if categorical_conflict(
            dimension, getattr(left, dimension), getattr(right, dimension)
        ):
            reasons.append(dimension)
    # Rule 2 in its sharpest form: a reduced-sugar claim contradicts an
    # EXPLICIT full-sugar claim and nothing else.
    if (left.diet_claim and right.sugar_claim) or (
        left.sugar_claim and right.diet_claim
    ):
        reasons.append("diet_claim")
    return reasons


def same_product(left: ProductIdentity, right: ProductIdentity) -> bool:
    """True when no descriptor dimension proves the rows are different."""
    return not (left.identity_review_reason or right.identity_review_reason or identity_conflict(left, right))


def evaluate_sku_identity(left: ProductIdentity, right: ProductIdentity) -> dict:
    """One identity evaluation with established conflicts and ALL raw evidence.

    Raw feed differences require review; they do not prove different products.
    Compatible descriptors do not authorize identity edges for splitting.
    """
    conflicts = identity_conflict(left, right)
    attributes = evaluate_dimensions(left.dimensions, right.dimensions) if (
        left.dimensions is not None and right.dimensions is not None) else {}
    columns = evaluate_columns(left.dimensions, right.dimensions) if (
        left.dimensions is not None and right.dimensions is not None) else {}
    unknown = sorted(set(left.dimensions.unclassified_keys if left.dimensions else ()) |
                     set(right.dimensions.unclassified_keys if right.dimensions else ()))
    malformed = list(left.dimensions.malformed_parts if left.dimensions else ()) + list(
        right.dimensions.malformed_parts if right.dimensions else ())
    from core.product_context import compare_context
    contextual = compare_context(left.dimensions.context, right.dimensions.context) if (
        left.dimensions and right.dimensions and left.dimensions.context and right.dimensions.context) else {}
    resolved = []
    if contextual.get("caffeine", {}).get("status") == "equal":
        resolved.append("Caffeine")
    packaging = contextual.get("inner_packaging", {})
    if packaging.get("types", {}).get("status") == "equal" and packaging.get("materials", {}).get("status") == "equal":
        resolved.extend(["Pack Type", "Pack Material Type"])
    review = [k for k, v in attributes.items() if v["review"] and k not in resolved]
    known = left.gtin_trusted and right.gtin_trusted
    decision = ("same" if not conflicts else "different") if known else (
        "different" if conflicts else "review" if review or unknown or malformed
        else "compatible_unverified")
    holds = sorted({r for r in (left.identity_review_reason, right.identity_review_reason) if r})
    if holds:
        decision = "review"
    return {"decision": decision, "identity_review_reasons": holds, "context_comparison": contextual, "resolved_review_dimensions": resolved, "identity_conflicts": conflicts, "attribute": attributes,
            "columns": columns, "review_dimensions": review, "unclassified_keys": unknown,
            "malformed_parts": malformed}


__all__ = [
    "CATEGORICAL_DIMENSIONS",
    "DESCRIPTOR_COLUMNS",
    "NON_DESCRIPTOR_COLUMNS",
    "PACK_SENTINEL",
    "VOLUME_RELATIVE_TOLERANCE",
    "ProductIdentity",
    "alias_fold",
    "attr_token_set",
    "brand_aliases",
    "brand_conflict",
    "categorical_conflict",
    "completeness",
    "concept_folds",
    "flavor_vocabulary",
    "identity_conflict",
    "evaluate_sku_identity",
    "identity_tokens_set",
    "normalize_brand",
    "row_identity",
    "same_product",
]
