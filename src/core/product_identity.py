"""src/core/product_identity.py — ONE product-identity decision, SSOT.

Every label-forming surface (dedupe T1.5, the gate, the vetoes) has to answer
the same question: are these two catalog rows the SAME product? Until now each
lane answered it with its own ad-hoc comparison, and the audit on 2026-09-30
measured what that costs, using barcode-labeled ground truth:

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
    Accessories" spam title under a real barcode). Neither may decide identity
    or rank a representative. Completeness is scored over descriptor columns
    only.

4.  AN ELIGIBLE BARCODE OUTRANKS DESCRIPTORS. Reviewed identity holds remain
    ineligible even with valid check digits. `barcode_validity` decides whether a
    barcode may be trusted as identity at all; when both rows carry a trusted
    barcode the barcode IS the answer and no text comparison is consulted.

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

from core.common import vocabulary
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
from core.gtin import barcode_validity
from core.record_linkage import strip_pack_multiplicity
from core.product_dimensions import DimensionEvidence, row_dimensions, evaluate_dimensions, evaluate_columns
from pipeline import normalize_text

# ── descriptor columns ─────────────────────────────────────────────────────
# The columns that DESCRIBE a product. `price`, `url`, `image_url` and
# `country` are excluded on purpose (rule 3).
DESCRIPTOR_COLUMNS: tuple[str, ...] = (
    "title", "brand", "category", "category_path", "attributes", "description",
)
NON_DESCRIPTOR_COLUMNS: frozenset[str] = frozenset(
    {"price", "url", "image_url", "country", "retailer", "product_id", "barcode"}
)

# The gate's 5% relative tolerance; `volumes_compatible` is the SSOT for the
# predicate, so reusing the number keeps "compatible" meaning one thing.
VOLUME_RELATIVE_TOLERANCE = 0.05

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
    barcode_trusted: bool = False
    barcode_key: str = ""
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
def normalize_brand(raw: object) -> frozenset[str]:
    """Brand cell -> a comparable token set.

    "by Concord Foods" (the export prefixes some brands), "Peet's", `Peet""S`
    and "Peet S" all reduce to one key. The apostrophe and doubled-quote OCR
    damage are measured cases, not hypotheticals.
    """
    text = _BRAND_NOISE_RE.sub(" ", normalize_text(str(raw or "")))
    tokens = [t for t in _ALPHA_RE.findall(text) if t not in _BRAND_SUFFIX]
    if not tokens:
        return frozenset()
    folded = set(tokens)
    aliases = brand_aliases()
    for token in tokens:
        target = aliases.get(token)
        if target:
            folded.add(target)
    return frozenset(folded)


def alias_fold(tokens: Iterable[str], *, qualifiers: bool = False) -> frozenset[str]:
    """Vocabulary folding: plural -> singular, concept -> canonical, synonyms.

    `qualifiers=True` additionally applies the flavor synonym map, which is
    deliberately NOT applied to package/carbonation dimensions.
    """
    out: set[str] = set()
    concepts = concept_folds()
    for token in tokens:
        token = str(token).strip().lower()
        if not token:
            continue
        out.add(token)
        singular = FLAVOR_ALIASES.get(token)
        if singular:
            out.add(singular)
        concept = concepts.get(token)
        if concept:
            out.add(concept)
        if qualifiers:
            out |= QUALIFIER_ALIASES.get(token, frozenset())
    return frozenset(out)


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
    return sum(1 for col in DESCRIPTOR_COLUMNS if str(get(col) or "").strip())


def attr_token_set(attr: object, key: str) -> frozenset[str]:
    """`Key: value; ...` cell -> token set, normalized THEN tokenized.

    Order matters: normalizing first is what makes "coffee, vanilla" and
    "vanilla coffee" the same set.
    """
    match = re.search(key + r"\s*:\s*([^;]+)", str(attr or ""), re.I)
    if not match:
        return frozenset()
    return frozenset(_TOKEN_RE.findall(normalize_text(match.group(1))))


def identity_tokens_set(attributes: object) -> frozenset[str]:
    """Declared flavor UNION roast-type tokens from the attribute cell.

    Roast is a product splitter that lives in `Roast Type`, not `Flavour`:
    "French Roast" and "Vanilla" both declare "Flavour: coffee", so flavor
    alone cannot separate them.
    """
    out: set[str] = set()
    for field_re in _FLAVOR_FIELDS:
        out |= attr_token_set(attributes, field_re)
    return frozenset(out)


def _barcode_facts(barcode: object) -> tuple[bool, str]:
    """(trusted, normalized key) using structural validation and review policy."""
    import pandas as pd

    from core.gtin import normalize_and_validate_gtin

    series = pd.Series([barcode], dtype="string")
    facts = normalize_and_validate_gtin(series)
    if not bool(barcode_validity(series).iloc[0]):
        return False, ""
    key = facts["gtin_clean"].iloc[0]
    return True, "" if key is None else str(key)


def row_identity(row: Mapping[str, Any] | Any) -> ProductIdentity:
    """Build the descriptor bundle for one catalog row.

    Accepts a Mapping or a pandas namedtuple. The extraction itself is the
    pipeline's own `sku_info` — this module never re-implements a parser, it
    only decides how parsed values are COMPARED.
    """
    get = (lambda k, d="": row.get(k, d)) if isinstance(row, Mapping) else (
        lambda k, d="": getattr(row, k, d)
    )
    title = str(get("title", "") or "")
    attributes = get("attributes", "")
    description = str(get("description", "") or "")
    category_text = " ".join(
        str(get(col, "") or "") for col in ("category", "category_path")
    )

    from core.structured_features import sku_info

    info = sku_info(title, attributes, description)
    declared = identity_tokens_set(attributes)
    title_tokens = set(_TOKEN_RE.findall(normalize_text(title)))
    flavor = alias_fold(set(info["flavor"]) | set(declared), qualifiers=True)
    # Title-side flavor words close the gap when the attribute cell declares
    # nothing (a listing that only says "Black & White" in its title).
    flavor |= alias_fold(title_tokens & flavor_vocabulary(), qualifiers=True)
    flavor |= alias_fold(title_tokens & declared, qualifiers=True)

    from core.identity_policy import review_reason
    held_reason = review_reason(get("barcode", ""))
    trusted, key = _barcode_facts(get("barcode", ""))
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
        barcode_trusted=trusted,
        barcode_key=key,
        identity_review_reason=held_reason,
        completeness=completeness(row),
        dimensions=row_dimensions(row if isinstance(row, Mapping) else {
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
    barcode, or a reviewed adjudication) before collapsing anything.

    Rule 4: two trusted barcodes settle it outright; the text is not consulted.
    """
    if left.barcode_trusted and right.barcode_trusted:
        return [] if left.barcode_key == right.barcode_key else ["barcode"]

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


def evaluate_product_identity(left: ProductIdentity, right: ProductIdentity) -> dict:
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
    known = left.barcode_trusted and right.barcode_trusted
    decision = ("same" if not conflicts else "different") if known else (
        "different" if conflicts else "review" if review or unknown or malformed
        else "compatible_unverified")
    holds = sorted({r for r in (left.identity_review_reason, right.identity_review_reason) if r})
    if holds:
        decision = "review"
    return {"decision": decision, "identity_review_reasons": holds, "context_comparison": contextual, "resolved_review_dimensions": resolved, "identity_conflicts": conflicts, "attributes": attributes,
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
    "evaluate_product_identity",
    "identity_tokens_set",
    "normalize_brand",
    "row_identity",
    "same_product",
]
