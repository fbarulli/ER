"""Source-grounded identity distinctions used to review otherwise clean pairs.

These channels supplement critical-attribute vetoes. They never infer a
negative from a missing claim, and category breadcrumbs are not declarations.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from collections.abc import Mapping

from core.critical_attributes import extract_critical_claims, extract_description_claims
from core.text import normalized_attribute_text


_GENERIC_FLAVORS = frozenset({'fruit', 'cola', 'coffee', 'tea', 'tonic'})

# Module-scope compiled patterns (PERF r15).
#
# `listing_identity` runs ~25 word-probe patterns per row through the `re`
# module, and every one of those calls re-hashes the pattern string and
# re-reads `re`'s internal cache (`re._compile`) before matching.  Binding the
# compiled patterns here removes that dispatch from a per-row hot path; the
# pattern text is copied verbatim from the call sites, so nothing about the
# match semantics changes.
_FLAVOR_VARIANT_RE = re.compile(
    r'\b(?:fruit punch|root beer|blood orange|black cherry|white grape|black grape|'
    r'grape white|grape black|cool blue|glacier freeze|red peak rush|carmel valley)\b')
_APPLE_RE = re.compile(r'\bapple\b')
_JUICE_RE = re.compile(r'\bjuice\b')
_CULTIVAR_RE = re.compile(r'\b(?:gala|braeburn|russet|golden delicious|cox)\b')
_FORMULATION_RE = re.compile(
    r'\b(?:protein boost|caffeine boost|double espresso|women fit|c mix|multi v|'
    r'immune strong|morgenstark|bcaa|collagen|pink beach|glow)\b')
_BATTERY_RE = re.compile(r'\bbattery\b')
_BATTERY_LINE_RE = re.compile(r'\b(?:black|rise|pearberry)\b')
_SOFT_DRINK_RE = re.compile(r'\b(?:cola|kola|soda|birch beer|energy drink)\b')
_RECIPE_VARIANT_RE = re.compile(r'\b(?:regular|original|light|lite|diet)\b')
_SPICE_PRODUCT_RE = re.compile(r'\b(?:turnip|ginger ale)\b')
_SPICY_RE = re.compile(r'\b(?:spicy|hot)\b')
_PLAIN_RE = re.compile(r'\b(?:plain|simple)\b')
_CAFFEINE_FREE_RE = re.compile(
    r'\b(?:zero caffeine|no caffeine|caffeine free|decaf|decaffeinated)\b')
_JUICE_CONTENT_RE = re.compile(
    r'(?:^|;)\s*juice content\s*:\s*(\d+)(?:\s*[-–]\s*(\d+))?\s*%', re.I)
_DESCRIPTION_FLAVOR_RE = re.compile(
    r"(?:[a-z]+\s+){0,2}[a-z]+\s+flavou?red|(?:flavou?r|taste)\s+of\s+(?:[a-z]+\s*){1,3}")
_WATER_RE = re.compile(r'\bwater\b')
_LIGHT_STRENGTH_RE = re.compile(r'\b(?:slight|lightly|light)\b')
_MEDIUM_STRENGTH_RE = re.compile(r'\bmedium\b')
_STRONG_STRENGTH_RE = re.compile(r'\b(?:strong|strongly)\b')
_COLA_FAMILY_RE = re.compile(r'\b(?:cola|kola)\b')
_MATE_FAMILY_RE = re.compile(r'\b(?:mate|yerba)\b')
_HERBAL_VARIANT_RE = re.compile(r'\barishta\s+([a-z]+)\b')
_NAMED_FAMILIES = (
    ('cream_soda', re.compile(r'\bcream soda\b')),
    ('dr_bob', re.compile(r'\bdr[.]? bob\b')),
    ('lemon_tea', re.compile(r'\blemon tea\b')),
    ('chai_concentrate', re.compile(r'\bchai\b.*\bconcentrate\b')),
    ('gingerbread_syrup', re.compile(r'\bgingerbread\b.*\bsyrup\b')),
)

# flavor -> the `\b<flavor>\b` probe, built once per lexicon entry instead of
# re-escaping and re-compiling it inside the per-phrase loop.
#
# The key space is CLOSED: `_flavor_probe` is only ever called with members of
# `core.critical_attributes.DECLARED_FLAVOR_LEXICON`, which is a frozenset built
# from `_VOCAB["declared_flavor_lexicon"]` in config/vocabulary.json — 89
# entries at the time of writing, and re-read only at import. Nothing derived
# from a row can enter the key. `lru_cache` is used anyway rather than a bare
# module dict so the bound is structural: if that vocabulary ever becomes
# dynamic, the cache can still not grow without limit on a per-row path.
@lru_cache(maxsize=256)
def _flavor_probe(flavor: str) -> "re.Pattern":
    return re.compile(r'\b' + re.escape(flavor) + r'\b')


@lru_cache(maxsize=65536)
def _listing_identity_cached(title: str, attributes: str, description: str) -> tuple:
    facts = _listing_identity_impl(title, attributes, description)
    return tuple((key, tuple(values)) for key, values in facts.items())


def listing_identity(title: str, attributes: str = '', description: str = '') -> dict[str, list[str]]:
    """Memoized view of :func:`_listing_identity_impl`.

    The declaration scan is pure in its three source cells and the same
    listing is re-walked by the extraction ledger and the identity lane. A
    fresh dict of fresh lists is returned so no caller can mutate the cached
    value.
    """
    key = (title, attributes, description)
    try:
        cached = _listing_identity_cached(*key)
    except TypeError:
        return _listing_identity_impl(*key)
    return {dimension: list(values) for dimension, values in cached}


def _listing_identity_impl(title: str, attributes: str = '', description: str = '') -> dict[str, list[str]]:
    from core.product_selection import selected_identity_inputs
    selected_title, selected_attributes, selected_variant = selected_identity_inputs(title, attributes)
    title_text = normalized_attribute_text(selected_title)
    claims = extract_critical_claims(selected_title, selected_attributes)
    facts = {}
    if selected_variant:
        facts['selected_variant'] = [selected_variant]
    # Preserve phrase identity instead of reducing it to a generic ingredient.
    flavor_variants = set(_FLAVOR_VARIANT_RE.findall(title_text))
    flavor_variants = {'white grape' if v == 'grape white' else 'black grape' if v == 'grape black' else v
                       for v in flavor_variants}
    if flavor_variants:
        facts['flavor_variant'] = sorted(flavor_variants)
    if _APPLE_RE.search(title_text) and _JUICE_RE.search(title_text):
        cultivars = set(_CULTIVAR_RE.findall(title_text))
        if cultivars:
            facts['cultivar'] = sorted(cultivars)
    formulations = set(_FORMULATION_RE.findall(title_text))
    if _BATTERY_RE.search(title_text):
        formulations.update(_BATTERY_LINE_RE.findall(title_text))
    if formulations:
        facts['formulation'] = sorted(formulations)
    if _SOFT_DRINK_RE.search(title_text):
        labels = set(_RECIPE_VARIANT_RE.findall(title_text))
        if labels:
            facts['recipe_variant'] = sorted('light' if v == 'lite' else 'original' if v == 'regular' else v for v in labels)
    if _SPICE_PRODUCT_RE.search(title_text):
        spice = 'not_as_hot' if 'not as hot' in title_text else 'spicy' if _SPICY_RE.search(title_text) else 'plain' if _PLAIN_RE.search(title_text) else ''
        if spice:
            facts['spice_variant'] = [spice]
    if claims['organic']:
        facts['organic'] = sorted(claims['organic'])
    declared_text = normalized_attribute_text(selected_title, selected_attributes)
    if _CAFFEINE_FREE_RE.search(declared_text):
        facts['caffeine_status'] = ['caffeine_free']
    # Only explicit concentration declarations, never category membership.
    content = _JUICE_CONTENT_RE.search(selected_attributes)
    if content:
        lo, hi = int(content.group(1)), int(content.group(2) or content.group(1))
        if 0 <= lo <= hi <= 100:
            facts['juice_concentration'] = ['pure' if lo == hi == 100 else 'diluted' if hi < 100 else 'unspecified']
    # The lexicon import stays lazy (and out of the phrase loop): it is only
    # needed when the description actually carries a flavor phrase.
    phrases = _DESCRIPTION_FLAVOR_RE.findall(normalized_attribute_text(description))
    if phrases:
        from core.critical_attributes import DECLARED_FLAVOR_LEXICON, FLAVOR_ALIASES
        description_flavors = {
            FLAVOR_ALIASES.get(flavor, flavor)
            for phrase in phrases
            for flavor in DECLARED_FLAVOR_LEXICON
            if _flavor_probe(flavor).search(phrase)
        }
    else:
        description_flavors = set()
    if description_flavors - _GENERIC_FLAVORS:
        facts['description_flavor'] = sorted(description_flavors - _GENERIC_FLAVORS)
    flavors = claims['flavor'] - _GENERIC_FLAVORS
    if flavors:
        facts['flavor'] = sorted(flavors)
    pulp = claims['pulp'] | extract_description_claims(description)['pulp']
    if pulp:
        facts['pulp'] = sorted(pulp)
    # Compare strength only when the product title identifies water.
    # Generic "strong" in coffee/health claims does not assert carbonation.
    if _WATER_RE.search(title_text):
        strengths = set()
        if _LIGHT_STRENGTH_RE.search(title_text):
            strengths.add('light')
        if _MEDIUM_STRENGTH_RE.search(title_text):
            strengths.add('medium')
        if _STRONG_STRENGTH_RE.search(title_text):
            strengths.add('strong')
        if strengths:
            facts['carbonation_strength'] = sorted(strengths)
    families = set()
    if _COLA_FAMILY_RE.search(title_text):
        families.add('cola')
    if _MATE_FAMILY_RE.search(title_text):
        families.add('mate')
    if families:
        facts['drink_family'] = sorted(families)
    for family, pattern in _NAMED_FAMILIES:
        if pattern.search(title_text):
            facts.setdefault('drink_family', []).append(family)
    carbonation = claims['carbonation'] | extract_description_claims(description)['carbonation']
    if len(carbonation) > 1:
        facts['_source_conflicts'] = ['carbonation']
    # A named herbal variant is the word immediately after its product line,
    # not a bag of arbitrary residual title words.
    variants = set(_HERBAL_VARIANT_RE.findall(title_text))
    if variants:
        facts['herbal_variant'] = sorted(variants)
    return facts


@lru_cache(maxsize=16384)
def _captured_identity(captured: str) -> dict[str, frozenset[str]]:
    try:
        rows = json.loads(captured)
    except (ValueError, TypeError):
        return {}
    if not isinstance(rows, list):
        return {}
    from core.columns import alias_names
    facts = defaultdict(set)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        def text(role):
            return ' '.join(str(row.get(name) or '') for name in alias_names(role))
        for dimension, values in listing_identity(text('sku_name_eng'), text('attribute'), text('description_short_eng')).items():
            facts[dimension].update(values)
    return {dimension: frozenset(values) for dimension, values in facts.items()}


def _record_identity(record: Mapping) -> dict[str, frozenset[str]]:
    retained = record.get('declared_identity')
    if retained is not None:
        return {dimension: frozenset(values) for dimension, values in retained.items()}
    # Original listings are authoritative, including on pre-fix snapshots.
    captured = record.get('source_rows')
    if captured:
        encoded = captured if isinstance(captured, str) else json.dumps(captured, sort_keys=True)
        return _captured_identity(encoded)
    ledger = record.get('evidence_ledger') or []
    if isinstance(ledger, str):
        ledger = json.loads(ledger)
    facts = defaultdict(set)
    for entry in ledger:
        if entry.get('field') == 'declared_identity':
            for dimension, values in entry['value'].items():
                facts[dimension].update(values)
    return {dimension: frozenset(values) for dimension, values in facts.items()}


def record_identity(record: Mapping) -> dict[str, frozenset[str]]:
    facts = dict(_record_identity(record))
    flags = record.get('attribute_consistency_flags') or ()
    if isinstance(flags, str):
        import ast
        flags = ast.literal_eval(flags)
    conflicts = set()
    for flag in flags:
        if flag in {'volume_sources_disagree', 'pack_sources_disagree', 'pack_hierarchy_ambiguous'}:
            conflicts.add(str(flag))
        elif str(flag).startswith('categorical_source_conflict:'):
            conflicts.add(str(flag).split(':', 1)[1])
    if conflicts:
        facts['_source_conflicts'] = frozenset(conflicts)
    return facts


def identity_review_dimensions(left: Mapping, right: Mapping) -> list[str]:
    a, b = record_identity(left), record_identity(right)
    review = []
    for dimension in sorted(a.keys() | b.keys()):
        av, bv = a.get(dimension, frozenset()), b.get(dimension, frozenset())
        # One-sided evidence is unknown, never a hard veto. A newly declared
        # flavor, texture or named variant still prevents confident approval.
        if dimension == '_source_conflicts':
            if av or bv:
                review.extend('source:' + value for value in sorted(av | bv))
            continue
        if dimension == 'juice_concentration' and (not av or not bv):
            continue
        if av != bv:
            review.append(dimension)
    return review
