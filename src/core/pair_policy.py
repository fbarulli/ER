"""Shared approval policy consuming the complete attribute comparison.

Configured vetoes retain their precedence. Every remaining conflict or
asymmetric populated comparison requires review. Missing metadata is recorded
as unknown; it cannot supply the positive identity evidence needed to approve.
"""
from core.attribute_decision import ComparisonResult

# Quantity, packaging and marketing agreement alone cannot identify a recipe.
IDENTITY_KEYS = frozenset({
    'flavour', 'water type', 'tea type', 'coffee type', 'rtd coffee style',
    'concentrate format', 'sports drink style', 'made from',
    'botanicals and functional ingredients', 'sports ingredients',
})


def assess_pair(evidence, left, right):
    from core.declared_identity import record_identity
    review, trace, positive = [], {}, []
    for key, entry in evidence.dimensions.items():
        if entry.result in {ComparisonResult.CONFLICT, ComparisonResult.SUBSET}:
            action = 'review'
            review.append(key)
        elif entry.fallback_from in {'source_conflict', 'claim_conflict'}:
            action = 'review'
            review.append(key)
        elif entry.result is ComparisonResult.MATCH:
            action = 'agreement'
            if key in IDENTITY_KEYS:
                positive.append(key)
        else:
            action = 'unknown'
        trace[key] = {'action': action, **entry.as_dict()}
    a, b = record_identity(left), record_identity(right)
    for key in sorted(a.keys() & b.keys()):
        if key not in {'_source_conflicts', 'organic', 'caffeine_status',
                       'juice_concentration', 'pulp', 'carbonation_strength'}:
            if a[key] and a[key] == b[key]:
                positive.append('declared:' + key)
    # An exact nonempty canonical name is affirmative evidence, provided
    # the full attribute checks above and source identity checks also pass.
    name_a, name_b = left.get('canonical'), right.get('canonical')
    if name_a and name_a == name_b and '_' in str(name_a):
        positive.append('canonical_name')
    if not positive:
        review.append('positive_identity_missing')
    return {'review': sorted(set(review)), 'positive_identity': sorted(positive),
            'dimensions': trace}


def identity_similarity(left, right):
    """Compare all canonical words, including words captured in compounds."""
    a = set(str(left or '').replace('_', ' ').split())
    b = set(str(right or '').replace('_', ' ').split())
    return len(a & b) / len(a | b) if a and b else 0.0
