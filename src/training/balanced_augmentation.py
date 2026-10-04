"""Allocate measured attribute gaps, mint negatives, then mask context.

Original rows are retained. Donors and new supervision stay in the frozen
training split. Unsupported attributes get an explicit shortfall, not labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.schemas import BalancedAugmentationSpec, MaskAuditEntry
from core.coverage_contracts import Count, require_keys
from training.masking import (_FIELD_PREFIXES, _field_surfaces, _field_values_conflict,
    _coherent_splice_field, field_of, augment_pairs, extend_augmented_features,
    normalize_entity_key)


class AttributeAllocation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    positive_both_observed: Count
    negative_both_observed: Count
    eligible_anchors: Count
    donor_values: Count
    requested: Count
    minted: Count
    masked: Count
    shortfall: Count
    status: str

    @model_validator(mode='after')
    def accounting(self):
        if self.minted + self.shortfall != self.requested:
            raise ValueError('attribute mint allocation does not close')
        return self


class AugmentationCoverage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_train_positives: Count
    source_train_negatives: Count
    vendor_variation_positives: Count
    masked_positives: Count
    minted_negatives: Count
    masked_minted_negatives: Count
    attributes: dict[str, AttributeAllocation]
    rejections: dict[str, Count]
    requested_counts: dict[str, Count]

    @model_validator(mode='after')
    def complete(self):
        require_keys(self.attributes, _FIELD_PREFIXES, 'balanced augmentation attributes')
        if sum(e.minted for e in self.attributes.values()) != self.minted_negatives:
            raise ValueError('minted attribute population does not close')
        if sum(e.masked for e in self.attributes.values()) != self.masked_minted_negatives:
            raise ValueError('masked attribute population does not close')
        if self.vendor_variation_positives + self.masked_positives > self.minted_negatives + self.masked_minted_negatives:
            raise ValueError('positive augmentation exceeds negative augmentation')
        actual = {name:getattr(self,name) for name in self.requested_counts}
        if actual != self.requested_counts:
            raise ValueError(f'augmentation output shortfall: requested={self.requested_counts}, produced={actual}')
        return self


class ObjectiveBalanceCoverage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    original_unique: Count
    augmented_unique: Count
    original_repeat_presentations: Count
    total_presentations: Count
    required_original_share: float = Field(ge=0, lt=1)
    achieved_original_share: float = Field(ge=0, le=1)

    @model_validator(mode='after')
    def accounting(self):
        originals = self.original_unique + self.original_repeat_presentations
        if originals + self.augmented_unique != self.total_presentations:
            raise ValueError('objective presentation accounting does not close')
        actual = originals / max(self.total_presentations,1)
        if abs(actual-self.achieved_original_share)>1e-8 or actual < self.required_original_share:
            raise ValueError('original objective share is not satisfied')
        return self


def balance_objective(triples, populations, copy_ids, original_share):
    """Retain every unique triple; explicitly weight originals via repeats."""
    original = [i for i,triple in enumerate(triples) if not set(triple) & copy_ids]
    augmented = len(triples)-len(original)
    required = max(len(original), math.ceil(augmented*original_share/(1-original_share)))
    if required and not original:
        raise ValueError('cannot preserve original objective share: no licensed original triples')
    repeat = required-len(original)
    added=[original[i % len(original)] for i in range(repeat)]
    coverage=ObjectiveBalanceCoverage(original_unique=len(original),augmented_unique=augmented,
        original_repeat_presentations=repeat,total_presentations=len(triples)+repeat,
        required_original_share=original_share,achieved_original_share=required/max(len(triples)+repeat,1))
    return triples+[triples[i] for i in added],populations+[populations[i] for i in added],coverage


def context_mask(text, field, extent, rng):
    """Mask prose context while retaining structured evidence and lexical claims."""
    from core.critical_attributes import extract_critical_claims, extract_flavor_tokens
    import re
    tokens = text.split()
    protected = {'no', 'not', 'non', 'without', 'with', 'free', 'added', 'sugar',
        'pulp', 'carbonated', 'still', 'diet', 'soda', 'zero', 'low'}
    for values in _field_surfaces(text).values():
        for token in values:
            protected.update(re.findall(r'[a-z0-9]+', token.casefold()))
    eligible = [i for i, token in enumerate(tokens) if field_of(token) is None
        and not token.startswith('[FIELD_') and not re.search(r'\d', token)
        and not (set(re.findall(r'[a-z0-9]+', token.casefold())) & protected)
        and not extract_flavor_tokens(token)
        and not any(extract_critical_claims(token).values())]
    if not eligible:
        return None
    n = min(len(eligible), max(1, round(len(tokens) * extent)))
    for i in rng.sample(eligible, n):
        tokens[i] = '[MASK]'
    return ' '.join(tokens), n / max(len(tokens), 1)


def augment_balanced(*, pos, neg, payload, row_bc, features, df, train_indices,
                     canonical_indices, spec: BalancedAugmentationSpec, seed):
    rng = random.Random(seed)
    original_payload = list(payload)
    fields = [_field_surfaces(text) for text in payload]
    positives = [(int(a), int(b)) for a,b in pos if int(a) in train_indices and int(b) in train_indices]
    negatives = [(int(a), int(b)) for a,b in neg if int(a) in train_indices and int(b) in train_indices]
    positive_support, negative_support = Counter(), Counter()
    anchors, donors = defaultdict(list), defaultdict(list)
    for a,b in positives:
        for field in _FIELD_PREFIXES:
            if fields[a].get(field) and fields[b].get(field):
                positive_support[field] += 1
                if b in canonical_indices and not _field_values_conflict(field, fields[a][field], fields[b][field]):
                    anchors[field].append((a,b))
    for a,b in negatives:
        negative_support.update(f for f in _FIELD_PREFIXES if fields[a].get(f) and fields[b].get(f))
    for index in sorted(train_indices):
        for field, values in fields[index].items():
            donors[field].append((index, values))
    eligible_fields = [f for f in _FIELD_PREFIXES if len(anchors[f]) >= spec.min_attribute_pairs
        and len({tuple(v) for _,v in donors[f]}) >= 2]
    target = spec.counts.minted_negatives
    weights = {f: math.sqrt(len(anchors[f])) * (1 + max(0, positive_support[f]/max(len(positives),1)
        - negative_support[f]/max(len(negatives),1))) for f in eligible_fields}
    quota = {f: 0 for f in _FIELD_PREFIXES}
    if weights:
        total = sum(weights.values())
        raw = {f: target*w/total for f,w in weights.items()}
        for f,value in raw.items(): quota[f] = math.floor(value)
        for f in sorted(raw, key=lambda f: (-(raw[f]-quota[f]), f))[:target-sum(quota.values())]: quota[f] += 1
    new_payload, new_bc = list(payload), list(map(str,row_bc))
    minted, audits, rejection, value_use = [], [], Counter(), Counter()
    minted_per_field, masked_per_field = Counter(), Counter()
    dedup = set()
    value_cap = max(spec.min_attribute_pairs, math.ceil(target*.03))
    for field in sorted(eligible_fields, key=lambda f: (len(anchors[f]), f)):
        candidates = list(anchors[field]); rng.shuffle(candidates)
        # Round-robin source entities/vendors: each anchor gets one turn before
        # another variant; no large seller/product group monopolizes a field.
        groups = defaultdict(list)
        for a,b in candidates:
            vendor = str(df.iloc[a].get('retailer','')) if a < len(df) else ''
            groups[(normalize_entity_key(row_bc[a],str(a)),vendor)].append((a,b))
        candidates = []
        while any(groups.values()):
            for group in sorted(groups):
                if groups[group]: candidates.append(groups[group].pop())
        for variant in range(spec.max_variants_per_anchor_field):
            for a,b in candidates:
                if minted_per_field[field] >= quota[field]: break
                source_key = normalize_entity_key(row_bc[a],str(a))
                attempts = rng.sample(donors[field], min(spec.donor_attempts,len(donors[field])))
                source_vendor = str(df.iloc[a].get('retailer','')) if a < len(df) else ''
                attempts.sort(key=lambda item: item[0] >= len(df) or str(df.iloc[item[0]].get('retailer','')) == source_vendor)
                for donor, values in attempts:
                    if normalize_entity_key(row_bc[donor],str(donor)) == source_key: continue
                    if not _field_values_conflict(field, fields[b][field], values): continue
                    signature = (field,tuple(values))
                    if value_use[signature] >= value_cap: continue
                    text = _coherent_splice_field(original_payload[b],field,values)
                    if text is None: rejection['ambiguous_prose:'+field] += 1; continue
                    if (a,b,text) in dedup: continue
                    if not _field_values_conflict(field,_field_surfaces(text)[field],fields[b][field]): continue
                    copy = len(new_payload);new_payload.append(text);new_bc.append(str(row_bc[a]))
                    minted.append((copy,b));dedup.add((a,b,text));value_use[signature] += 1
                    minted_per_field[field] += 1
                    audits.append(MaskAuditEntry(anchor_payload_idx=a, pair_payload_idx=b,
                        copy_payload_idx=copy, copy_source_payload_idx=b, gtin=str(row_bc[a]),
                        realized_extent=len(fields[b][field])/max(len(original_payload[b].split()),1),
                        anchor_text=original_payload[b], masked_text=text, population='hard_negative',
                        target_mode='counterfactual', coherent_prose=True, generation_variant='minted',
                        fields_hit=[field], donor_split='train', donor_anchor_payload_idx=donor,
                        donor_pair_payload_idx=donor, donor_anchor_entity=str(row_bc[donor]),
                        donor_pair_entity=str(row_bc[donor]), fields_before={field:fields[b][field]},
                        fields_after={field:values}).model_dump())
                    break
                else: rejection['no_safe_donor:'+field] += 1
    # Mask AFTER minting. Keep the discriminating attribute and all numeric
    # features unchanged; the original clean negative view remains available.
    minted_count = len(minted)
    mask_candidates = list(audits); rng.shuffle(mask_candidates)
    for audit in mask_candidates:
        if sum(masked_per_field.values()) >= spec.counts.masked_minted_negatives: break
        field = audit['fields_hit'][0]
        result = context_mask(audit['masked_text'],field,spec.mask_extent,rng)
        if result is None: rejection['no_maskable_context:'+field] += 1; continue
        text,extent = result
        if (audit['anchor_payload_idx'],audit['pair_payload_idx'],text) in dedup: continue
        copy=len(new_payload);new_payload.append(text);new_bc.append(audit['gtin'])
        minted.append((copy,audit['pair_payload_idx']))
        audits.append(MaskAuditEntry.model_validate({**audit,'copy_payload_idx':copy,
            'masked_text':text,'realized_extent':extent,'generation_variant':'minted_masked'}).model_dump())
        masked_per_field[field] += 1
    new_features = extend_augmented_features(features,new_payload,audits)
    # Real same-entity, cross-vendor listing variations are licensed positives.
    by_entity=defaultdict(list)
    for index in sorted(train_indices):
        if index<len(df): by_entity[normalize_entity_key(row_bc[index],str(index))].append(index)
    vendor_pairs=[]
    for members in by_entity.values():
        if len(vendor_pairs)>=spec.counts.vendor_variation_positives: break
        for left in members:
            right=next((i for i in members if str(df.iloc[left].get('retailer','')) != str(df.iloc[i].get('retailer',''))
                and all(not _field_values_conflict(f,fields[left].get(f,[]),fields[i].get(f,[])) for f in _FIELD_PREFIXES)),None)
            if right is not None: vendor_pairs.append((left,right));break
    pos_with_vendors=np.vstack([pos,np.asarray(vendor_pairs,dtype=int)]) if vendor_pairs else np.asarray(pos)
    # Only originals that now have an explicit negative are masked: no dead
    # positive copies are minted speculatively.
    usable={a for a,b in negatives} | {r['anchor_payload_idx'] for r in audits}
    mask_pairs=np.asarray([(a,b) for a,b in positives if a in usable],dtype=int).reshape(-1,2)
    if len(mask_pairs) < spec.counts.masked_positives:
        raise ValueError(f'positive masking needs {spec.counts.masked_positives} licensed anchors; only {len(mask_pairs)} available')
    chosen = rng.sample(list(map(tuple,mask_pairs.tolist())),spec.counts.masked_positives)
    _,new_payload,new_bc,n_mask,pos_audit=augment_pairs(np.asarray(chosen,dtype=int).reshape(-1,2),new_payload,np.asarray(new_bc),
        frac=1.,seed=seed+1,population='positive',lo=.1,hi=.2)
    masked_pairs=np.asarray([(r['copy_payload_idx'],r['pair_payload_idx']) for r in pos_audit],dtype=int).reshape(-1,2)
    final_pos=np.vstack([pos_with_vendors,masked_pairs]) if len(masked_pairs) else pos_with_vendors
    new_features=extend_augmented_features(new_features,new_payload,pos_audit)
    coverage=AugmentationCoverage(source_train_positives=len(positives),source_train_negatives=len(negatives),
        vendor_variation_positives=len(vendor_pairs),masked_positives=n_mask,
        minted_negatives=minted_count,masked_minted_negatives=len(minted)-minted_count,
        attributes={f:AttributeAllocation(positive_both_observed=positive_support[f],negative_both_observed=negative_support[f],
            eligible_anchors=len(anchors[f]),donor_values=len({tuple(v) for _,v in donors[f]}),requested=quota[f],
            minted=minted_per_field[f],masked=masked_per_field[f],shortfall=quota[f]-minted_per_field[f],
            status='eligible' if f in eligible_fields else 'insufficient_comparable_evidence') for f in _FIELD_PREFIXES},
        rejections=dict(rejection),requested_counts=spec.counts.model_dump())
    return final_pos, np.vstack([neg,np.asarray(minted,dtype=int)]) if minted else np.asarray(neg),new_payload,np.asarray(new_bc),new_features,pos_audit,audits,coverage
