"""Census structural difficulty and attribute coverage of frozen supervision."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from core.portable_archive import ByteCount
import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.coverage_contracts import Count, require_keys
from training.difficulty import DifficultySpec, DifficultyEndpoint, measure_pair
from training.masking import _FIELD_PREFIXES
from training.prepared_bundle import load_prepared_bundle


class PopulationDifficultyCoverage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    pairs: Count
    difficulty: dict[str, Count]
    reasons: dict[str, Count]
    vendor_relation: dict[str, Count]
    attributes: dict[str, dict[str, Count]]

    @model_validator(mode='after')
    def complete(self):
        require_keys(self.difficulty, {'easy', 'medium', 'hard', 'unknown'}, 'difficulty')
        require_keys(self.vendor_relation, {'same_vendor', 'cross_vendor', 'unknown'}, 'vendor relation')
        require_keys(self.attributes, _FIELD_PREFIXES, 'difficulty attribute')
        for counts in (self.difficulty, self.reasons, self.vendor_relation):
            if sum(counts.values()) != self.pairs:
                raise ValueError('difficulty strata must account for every pair')
        for counts in self.attributes.values():
            require_keys(counts, {'both_observed', 'one_observed', 'neither_observed', 'conflict'}, 'attribute evidence')
            if sum(counts[k] for k in ('both_observed', 'one_observed', 'neither_observed')) != self.pairs:
                raise ValueError('attribute evidence must account for every pair')
            if counts['conflict'] > counts['both_observed']:
                raise ValueError('attribute conflicts exceed comparable evidence')
        return self


class DifficultyReport(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: int = 1
    bundle: str
    bundle_size: int = Field(ge=0)
    definition: DifficultySpec
    difficulty_kind: str = 'structural proxy; no model scores or errors used'
    attributes: list[str]
    catalog_rows: Count
    catalog_attribute_rows: dict[str, Count]
    populations: dict[str, PopulationDifficultyCoverage]
    notes: list[str]

    @model_validator(mode='after')
    def complete(self):
        require_keys(self.catalog_attribute_rows, _FIELD_PREFIXES, 'catalog attribute')
        require_keys(self.attributes, _FIELD_PREFIXES, 'report attribute registry')
        require_keys(self.populations, {'original_positive', 'original_negative',
            'augmented_positive', 'augmented_negative', 'consumed_positive', 'consumed_negative'}, 'difficulty populations')
        if any(n > self.catalog_rows for n in self.catalog_attribute_rows.values()):
            raise ValueError('catalog coverage exceeds catalog population')
        return self


def file_size(path) -> int:
    h = ByteCount()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--definition', type=Path, help='optional DifficultySpec JSON')
    args = parser.parse_args()
    spec = DifficultySpec.model_validate_json(args.definition.read_text()) if args.definition else DifficultySpec()
    before = file_size(args.bundle)
    _, bundle = load_prepared_bundle(args.bundle)
    endpoints = [DifficultyEndpoint.from_text(text) for text in bundle['payload']]
    copies = {}
    audits_by_copy = {}
    for audit in [*bundle['mask_audit'], *bundle['hard_negative_mask_audit']]:
        copies[int(audit['copy_payload_idx'])] = int(audit.get('copy_source_payload_idx')
            if audit.get('copy_source_payload_idx') is not None else audit['anchor_payload_idx'])
        audits_by_copy[int(audit['copy_payload_idx'])] = audit
        if audit.get('copy_pair_payload_idx') is not None:
            copies[int(audit['copy_pair_payload_idx'])] = int(audit['pair_payload_idx'])
            audits_by_copy[int(audit['copy_pair_payload_idx'])] = audit
    vendors = {}
    for i in range(len(endpoints)):
        parent = copies.get(i, i)
        if parent < len(bundle['df']):
            raw = str(bundle['df'].iloc[parent].get('retailer', '')).strip().casefold()
            vendors[i] = raw if raw not in {'', 'nan', 'none'} else None
        else:
            vendors[i] = vendors.get(parent)
    roles = {str(entity): role for role, entities in bundle['training_plan']['holdout'].items()
             for entity in entities}
    totals = defaultdict(Counter)
    attr_counts = defaultdict(lambda: defaultdict(Counter))
    rows, cache = [], {}

    def add(a, b, label, population, presentation_id):
        a, b = int(a), int(b)
        key = (a, b, label)
        if key not in cache:
            cache[key] = measure_pair(endpoints[a], endpoints[b], label, spec)
        evidence = cache[key]
        va, vb = vendors[a], vendors[b]
        relation = 'unknown' if not va or not vb else 'same_vendor' if va == vb else 'cross_vendor'
        totals[population]['pairs'] += 1
        totals[population]['difficulty:' + evidence.difficulty] += 1
        totals[population]['reason:' + evidence.reason] += 1
        totals[population]['vendor:' + relation] += 1
        for field in _FIELD_PREFIXES:
            state = ('both_observed' if field in evidence.both_observed else
                     'one_observed' if field in evidence.one_observed else 'neither_observed')
            attr_counts[population][field][state] += 1
            if field in evidence.conflicts:
                attr_counts[population][field]['conflict'] += 1
        pair_roles = {roles.get(str(bundle['row_bc'][i]), 'unknown') for i in (a, b)}
        rows.append(dict(population=population, presentation_id=presentation_id,
            payload_index1=a, payload_index2=b, label=label,
            split=next(iter(pair_roles)) if len(pair_roles) == 1 else 'mixed',
            vendor_relation=relation, difficulty=evidence.difficulty,
            reason=evidence.reason, text_overlap=evidence.text_overlap,
            identical_prose=evidence.identical_prose, comparable_attributes=len(evidence.both_observed),
            conflicts=json.dumps(evidence.conflicts), one_observed=json.dumps(evidence.one_observed),
            generation='augmented' if a in copies or b in copies else 'original',
            target_modes='|'.join(sorted({audits_by_copy[i]['target_mode'] for i in (a,b) if i in copies}))))

    for name, label in [('pos', 1), ('neg', 0)]:
        # Unique stored pairs, separately from presentation-weighted objective.
        for n, (a, b) in enumerate(sorted(set(map(tuple, bundle[name].tolist())))):
            lane = 'augmented' if int(a) in copies or int(b) in copies else 'original'
            add(a, b, label, lane + ('_positive' if label else '_negative'), n)
    for fold_i, fold in enumerate(bundle['training_plan']['inputs']['folds']):
        for n, (a, positive, negative) in enumerate(fold['objective']['triples']):
            add(a, positive, 1, 'consumed_positive', f'{fold_i}:{n}')
            add(a, negative, 0, 'consumed_negative', f'{fold_i}:{n}')
    populations = {}
    for name in ['original_positive', 'original_negative', 'augmented_positive',
                 'augmented_negative', 'consumed_positive', 'consumed_negative']:
        counts = totals[name]
        populations[name] = PopulationDifficultyCoverage(pairs=counts['pairs'],
            difficulty={d: counts['difficulty:' + d] for d in ('easy', 'medium', 'hard', 'unknown')},
            reasons={k.removeprefix('reason:'): v for k, v in counts.items() if k.startswith('reason:')},
            vendor_relation={r: counts['vendor:' + r] for r in ('same_vendor', 'cross_vendor', 'unknown')},
            attributes={field: {state: attr_counts[name][field][state]
                for state in ('both_observed', 'one_observed', 'neither_observed', 'conflict')}
                for field in _FIELD_PREFIXES})
    report = DifficultyReport(bundle=str(args.bundle), bundle_size=before, definition=spec,
        attributes=list(_FIELD_PREFIXES), catalog_rows=len(bundle['df']),
        catalog_attribute_rows={f: sum(bool(e.attributes[f]) for e in endpoints[:len(bundle['df'])])
                                for f in _FIELD_PREFIXES}, populations=populations,
        notes=['Catalog coverage measures source-listing encoder evidence, not raw attribute presence.',
               'Stored pair counts are unique within each label; consumed pairs are objective-weighted.',
               'Canonical endpoints have unknown vendor; source-copy lineage preserves listing vendors.',
               'Positive conflicts flag review; difficulty never changes labels.',
               'Whole-catalog singleton rows have attribute coverage but no pair difficulty.',
               'Structural thresholds are provisional; model error measurement remains separate.'])
    if file_size(args.bundle) != before:
        raise ValueError('bundle changed during difficulty measurement')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'report.json').write_text(report.model_dump_json(indent=2) + '\n')
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / 'pairs.csv', index=False)
    frame.groupby(['population', 'split', 'generation', 'target_modes', 'vendor_relation', 'difficulty'],
                  dropna=False).size().rename('pairs').reset_index().to_csv(
                      args.output_dir / 'strata.csv', index=False)
    table = []
    for field in _FIELD_PREFIXES:
        row = {'attribute': field, 'catalog_rows': report.catalog_attribute_rows[field],
               'catalog_share': report.catalog_attribute_rows[field] / max(report.catalog_rows, 1)}
        for name, population in populations.items():
            c = population.attributes[field]
            row.update({name + '_' + key: value for key, value in c.items()})
            row[name + '_both_share'] = c['both_observed'] / population.pairs if population.pairs else None
        table.append(row)
    pd.DataFrame(table).to_csv(args.output_dir / 'attribute_coverage.csv', index=False)
    print(json.dumps({name: {'pairs': p.pairs, 'difficulty': p.difficulty}
                      for name, p in populations.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
