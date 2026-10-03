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


def listing_identity(title: str, attributes: str = '', description: str = '') -> dict[str, list[str]]:
    from core.product_selection import selected_identity_inputs
    selected_title, selected_attributes, selected_variant = selected_identity_inputs(title, attributes)
    title_text = normalized_attribute_text(selected_title)
    claims = extract_critical_claims(selected_title, selected_attributes)
    facts = {}
    if selected_variant:
        facts['selected_variant'] = [selected_variant]
    # Preserve phrase identity instead of reducing it to a generic ingredient.
    flavor_variants = set(re.findall(
        r'\b(?:fruit punch|root beer|blood orange|black cherry|white grape|black grape|'
        r'grape white|grape black|cool blue|glacier freeze|red peak rush|carmel valley)\b', title_text))
    flavor_variants = {'white grape' if v == 'grape white' else 'black grape' if v == 'grape black' else v
                       for v in flavor_variants}
    if flavor_variants:
        facts['flavor_variant'] = sorted(flavor_variants)
    if re.search(r'\bapple\b', title_text) and re.search(r'\bjuice\b', title_text):
        cultivars = set(re.findall(r'\b(?:gala|braeburn|russet|golden delicious|cox)\b', title_text))
        if cultivars:
            facts['cultivar'] = sorted(cultivars)
    formulations = set(re.findall(
        r'\b(?:protein boost|caffeine boost|double espresso|women fit|c mix|multi v|'
        r'immune strong|morgenstark|bcaa|collagen|pink beach|glow)\b', title_text))
    if re.search(r'\bbattery\b', title_text):
        formulations.update(re.findall(r'\b(?:black|rise|pearberry)\b', title_text))
    if formulations:
        facts['formulation'] = sorted(formulations)
    if re.search(r'\b(?:cola|kola|soda|birch beer|energy drink)\b', title_text):
        labels = set(re.findall(r'\b(?:regular|original|light|lite|diet)\b', title_text))
        if labels:
            facts['recipe_variant'] = sorted('light' if v == 'lite' else 'original' if v == 'regular' else v for v in labels)
    if re.search(r'\b(?:turnip|ginger ale)\b', title_text):
        spice = 'not_as_hot' if 'not as hot' in title_text else 'spicy' if re.search(r'\b(?:spicy|hot)\b', title_text) else 'plain' if re.search(r'\b(?:plain|simple)\b', title_text) else ''
        if spice:
            facts['spice_variant'] = [spice]
    if claims['organic']:
        facts['organic'] = sorted(claims['organic'])
    declared_text = normalized_attribute_text(selected_title, selected_attributes)
    if re.search(r'\b(?:zero caffeine|no caffeine|caffeine free|decaf|decaffeinated)\b', declared_text):
        facts['caffeine_status'] = ['caffeine_free']
    # Only explicit concentration declarations, never category membership.
    content = re.search(r'(?:^|;)\s*juice content\s*:\s*(\d+)(?:\s*[-–]\s*(\d+))?\s*%', selected_attributes, re.I)
    if content:
        lo, hi = int(content.group(1)), int(content.group(2) or content.group(1))
        if 0 <= lo <= hi <= 100:
            facts['juice_concentration'] = ['pure' if lo == hi == 100 else 'diluted' if hi < 100 else 'unspecified']
    description_flavors = set()
    for phrase in re.findall(r"(?:[a-z]+\s+){0,2}[a-z]+\s+flavou?red|(?:flavou?r|taste)\s+of\s+(?:[a-z]+\s*){1,3}", normalized_attribute_text(description)):
        from core.critical_attributes import DECLARED_FLAVOR_LEXICON, FLAVOR_ALIASES
        for flavor in DECLARED_FLAVOR_LEXICON:
            if re.search(r'\b' + re.escape(flavor) + r'\b', phrase):
                description_flavors.add(FLAVOR_ALIASES.get(flavor, flavor))
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
    if re.search(r'\bwater\b', title_text):
        strengths = set()
        if re.search(r'\b(?:slight|lightly|light)\b', title_text):
            strengths.add('light')
        if re.search(r'\bmedium\b', title_text):
            strengths.add('medium')
        if re.search(r'\b(?:strong|strongly)\b', title_text):
            strengths.add('strong')
        if strengths:
            facts['carbonation_strength'] = sorted(strengths)
    families = set()
    if re.search(r'\b(?:cola|kola)\b', title_text):
        families.add('cola')
    if re.search(r'\b(?:mate|yerba)\b', title_text):
        families.add('mate')
    if families:
        facts['drink_family'] = sorted(families)
    for family, pattern in (
        ('cream_soda', r'\bcream soda\b'), ('dr_bob', r'\bdr[.]? bob\b'),
        ('lemon_tea', r'\blemon tea\b'), ('chai_concentrate', r'\bchai\b.*\bconcentrate\b'),
        ('gingerbread_syrup', r'\bgingerbread\b.*\bsyrup\b'),
    ):
        if re.search(pattern, title_text):
            facts.setdefault('drink_family', []).append(family)
    carbonation = claims['carbonation'] | extract_description_claims(description)['carbonation']
    if len(carbonation) > 1:
        facts['_source_conflicts'] = ['carbonation']
    # A named herbal variant is the word immediately after its product line,
    # not a bag of arbitrary residual title words.
    variants = set(re.findall(r'\barishta\s+([a-z]+)\b', title_text))
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
