"""Source-bound variant qualifiers and shared conservative matching review."""
from __future__ import annotations
import re
from collections import defaultdict
from core.critical_attributes import normalized_attribute_text

GENERIC_FLAVORS = frozenset({'fruit','juice','coffee','latte','tea'})

def extract_identity_variants(title: object) -> set[str]:
    text=normalized_attribute_text(title)
    result=set()
    axes={
        'spice': {'spicy':r'\bspicy\b', 'plain':r'\bplain\b'},
        'caffeine': {'decaf':r'\b(?:decaf|decaffeinated)\b'},
        'range': {'original':r'\boriginal\b','lite':r'\b(?:lite|light)\b'},
    }
    if re.search(r'\b(?:water|mineral)\b',text):
        axes['carbonation_strength']={'classic':r'\bclassic\b','medium':r'\bmedium\b','light':r'\b(?:lightly|slightly|finely)\s+(?:sparkling|carbonated)\b'}
    for axis,values in axes.items():
        for value,pattern in values.items():
            if re.search(pattern,text):result.add(f'{axis}:{value}')
    return result

def matching_review_reasons(left: dict, right: dict) -> list[str]:
    """Absence/containment is review evidence, never an invented conflict."""
    reasons=[]
    axes=[]
    for record in (left,right):
        by_axis=defaultdict(set)
        for token in record.get('identity_variant_set') or ():
            axis,_,value=str(token).partition(':')
            by_axis[axis].add(value)
        axes.append(by_axis)
    for axis in sorted(set(axes[0])|set(axes[1])):
        a,b=axes[0][axis],axes[1][axis]
        if len(a)>1 or len(b)>1 or a!=b:reasons.append('variant:'+axis)
    a=set(left.get('flavor_set') or ())-GENERIC_FLAVORS
    b=set(right.get('flavor_set') or ())-GENERIC_FLAVORS
    if a and b and a!=b and (a<=b or b<=a):reasons.append('specific_flavor_subset')
    if not a or not b:
        # Unflavored water can be described without a flavor field. Unknown
        # flavor in other product families cannot establish variant identity.
        water=left.get('mode_type')==right.get('mode_type')=='water'
        if a or b or (not water and (left.get('mode_type') or right.get('mode_type'))):
            reasons.append('missing_specific_flavor')
    if left.get('mode_type') and right.get('mode_type') and left['mode_type']!=right['mode_type']:
        reasons.append('product_type')
    return sorted(set(reasons))
