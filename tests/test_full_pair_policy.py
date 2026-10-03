"""All-attribute approval policy and retained product-name regressions."""
import pytest

from core.attribute_decision import AttributeDecisionEngine
from core.attribute_universe import attribute_registry
from core.declared_identity import listing_identity
from core.pair_policy import assess_pair, identity_similarity
from core.critical_attributes import extract_critical_claims
from training.rand_matching import targeted_veto_gate


def info(**kwargs):
    result = dict(volume={330}, pack={1}, flavor_set={'lemon'},
                  universe_evidence={}, declared_identity={})
    result.update(kwargs)
    return result


def test_complete_registry_has_an_explicit_policy_trace_including_constants():
    a = info()
    evidence = AttributeDecisionEngine(volume_relative_tolerance=.05).evaluate(a, a)
    policy = assess_pair(evidence, a, a)
    assert set(policy['dimensions']) == set(attribute_registry())
    assert policy['review'] == []
    assert policy['positive_identity'] == ['flavour']


@pytest.mark.parametrize('key,left,right', [
    ('water type', {'mineral'}, {'spring'}),
    ('carbonization', {'still'}, {'sparkling'}),
    ('rtd coffee style', {'latte'}, {'espresso'}),
])
def test_nonveto_attribute_conflicts_change_candidate_route(key, left, right):
    a, b = info(universe_evidence={key:left}), info(universe_evidence={key:right})
    gate = targeted_veto_gate(a, b, sku_brand='Acme', candidate_brand='Acme', exact_gtin=False)
    assert gate['targeted_gate_route'] == 'human_review'
    assert key in gate['targeted_gate_reason']


def test_quantity_agreement_without_identity_proof_is_review():
    a = info(flavor_set=set())
    policy = assess_pair(AttributeDecisionEngine(volume_relative_tolerance=.05).evaluate(a, a), a, a)
    assert policy['review'] == ['positive_identity_missing']


def test_sku_identity_compounds_are_part_of_similarity():
    assert identity_similarity('lohilo carbonated pink_beach_bcaa_drink',
                               'lohilo carbonated glow_2022_collagen_containing') < .5
    assert identity_similarity('acme cream_soda', 'acme cream soda') == 1.


def test_description_declared_flavor_is_retained_without_ingredient_guessing():
    facts = listing_identity('BCAA drink 330ml', '', 'A guava flavored drink.')
    assert facts['description_flavor'] == ['guava']
    assert 'description_flavor' not in listing_identity('Drink 330ml', '', 'Ingredients: guava and apple.')


def test_carbonic_negation_and_added_co2_are_distinct():
    assert extract_critical_claims('Water without carbonic 1L')['carbonation'] == {'still'}
    assert 'still' not in extract_critical_claims('Mineral water without added carbon dioxide')['carbonation']


def test_cached_original_source_parse_preserves_every_dimension_verdict():
    import json
    from core.attribute_decision import _RAW_EVIDENCE_CACHE
    engine = AttributeDecisionEngine(volume_relative_tolerance=.05)
    a, b = info(flavor_set=set()), info(flavor_set=set())
    left = {'source_rows': json.dumps([{'sku_name_eng':'Vanilla water 330ml', 'attribute':'Water Type: Mineral'}])}
    right = {'source_rows': json.dumps([{'sku_name_eng':'Orange water 330ml', 'attribute':'Water Type: Spring'}])}
    _RAW_EVIDENCE_CACHE.clear()
    initial = engine.evaluate(a, b, left_raw=left, right_raw=right)
    assert len(_RAW_EVIDENCE_CACHE) == 2
    cached = engine.evaluate(a, b, left_raw=left, right_raw=right)
    assert initial == cached
    _RAW_EVIDENCE_CACHE.clear()
