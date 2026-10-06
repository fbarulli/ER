"""One exhaustive pair cohort, shared by all frozen model ablations."""
from __future__ import annotations

import copy
import io
import json

import pandas as pd

from model_tracks.shared_graph_data import CLEAN_BACKUP_SUFFIX
from model_tracks.training_data import augmentation_node_id, canonical_node_id


def prepare_cohort(setup, bundle):
    from graph_tracks.data import load_records
    from model_tracks.ablation import digest, write
    from model_tracks.shared_graph_data import _canonical_identity, _record, _copy_record
    from core.sku_identity import row_identity
    from training.folds import normalize_gtin

    folder = setup / 'ablation_cohort'
    folder.mkdir(parents=True, exist_ok=True)
    # The portable layout class owns this composition (package.py ships the
    # members with the SAME resolver; drift between ship/consume is dead).
    from model_tracks.portable_layout import PortableLayout
    clean = PortableLayout.consumer_clean_backup(setup)
    clean_pairs = pd.read_csv(clean / 'pairs.csv', dtype=str, keep_default_na=False)
    catalog = pd.read_csv(setup / 'eligible_catalog.csv', dtype=str, keep_default_na=False)
    rows = catalog.set_index('sku_id', drop=False).to_dict('index')
    records = {r['sku_id']: r for r in load_records(setup / 'prepared/listings.json')}
    canonical = pd.read_csv(io.BytesIO(bundle['canonical_records_csv']), dtype=str, keep_default_na=False)
    canonical_rows = canonical.set_index('gtin', drop=False).to_dict('index')
    gtins = sorted(canonical_rows)
    base = len(bundle['df'])
    canonical_end = base + len(gtins)
    audits = {}
    for audit in (*bundle['mask_audit'], *bundle['hard_negative_mask_audit']):
        copy_source = audit.get('copy_source_payload_idx')
        audits[int(audit['copy_payload_idx'])] = (int(audit['anchor_payload_idx'] if copy_source is None else copy_source), audit)
        if audit.get('copy_pair_payload_idx') is not None:
            audits[int(audit['copy_pair_payload_idx'])] = (int(audit['pair_payload_idx']), audit)
    if set(audits) != set(range(canonical_end, len(bundle['payload']))):
        raise ValueError('ablation requires complete mint lineage')
    node_ids = {}
    def endpoint(index):
        index = int(index)
        if index in node_ids:
            return node_ids[index]
        if index < base:
            row = bundle['df'].iloc[index].to_dict()
            node_id = str(row['sku_id'])
            record = copy.deepcopy(records.get(node_id) or _record(row_identity(row), node_id))
        elif index < canonical_end:
            gtin = gtins[index-base]
            node_id = canonical_node_id(gtin)
            row = {'sku_id': node_id, 'gtin': gtin}
            record = _record(_canonical_identity(canonical_rows[gtin]), node_id)
        else:
            parent, audit = audits[index]
            parent_id = endpoint(parent)
            node_id = augmentation_node_id(index)
            row = {'sku_id': node_id, 'gtin': str(bundle['row_bc'][index])}
            record = _copy_record(records[parent_id], bundle['payload'][index], audit, node_id)
        # Exact frozen input for every bundled endpoint; clean holdout rows keep
        # their normal composition. Existing clean IDs must retain that view.
        if node_id in rows and index < base:
            node_id = f'ablation_payload:{index}'
            row = dict(row, sku_id=node_id)
            record.update(sku_id=node_id)
        row['frozen_payload'] = bundle['payload'][index]
        rows[node_id] = row
        records[node_id] = record
        node_ids[index] = node_id
        return node_id

    fold = bundle['training_plan']['inputs']['folds'][0]
    from training.prepared_bundle import prepared_holdout
    from core.common import training_cfg, SEED
    populations = prepared_holdout(bundle, dict(training_cfg().split), seed=SEED)
    splits = {normalize_gtin(entity): split for split, values in zip(('train','dev','test'), populations, strict=True)
              for entity in values}
    consumed = {}
    triples = fold['objective']['triples']
    for n, (a,b,c) in enumerate(triples):
        for label, other in ((1,b),(0,c)):
            consumed.setdefault((int(a),int(other),label), []).append(n)
    cohort = []
    for n, pair in enumerate(clean_pairs.to_dict('records')):
        cohort.append({**pair, 'cohort_id':f'clean:{n}', 'population':'real',
            'evaluation_scope':'heldout' if pair['split'] != 'train' else 'training_diagnostic',
            'difficulty_slice':'unknown', 'mint_lineage':[], 'consumed_example_ids':[]})
    # Include the entire minted supply, including copies not selected by MNRL,
    # and every frozen objective pair (including easy sampled negatives).
    for name, label in [('pos',1),('neg',0),('train_neg',0)]:
        for n, (a,b) in enumerate(bundle[name]):
            a,b = int(a),int(b)
            lineage = [audits[i][1] for i in (a,b) if i in audits]
            entities = {splits.get(normalize_gtin(bundle['row_bc'][i]), 'unknown') for i in (a,b)}
            split = next(iter(entities)) if len(entities) == 1 else 'mixed'
            population = '|'.join(sorted({x['population'] for x in lineage})) or 'real_bundle'
            cohort.append({'sku_id1':endpoint(a), 'sku_id2':endpoint(b), 'label':str(label),
                'split':split, 'cohort_id':f'bundle:{name}:{n}', 'population':population,
                'evaluation_scope':'mint_diagnostic' if lineage else 'bundle_diagnostic',
                'payload_index1':a, 'payload_index2':b, 'mint_lineage':lineage,
                'difficulty_slice': 'unknown',
                'consumed_example_ids':consumed.get((a,b,label),[])})
    for n, ((a,b,c), population) in enumerate(zip(triples, fold['objective']['dataset']['population'], strict=True)):
        for label, other in ((1,b),(0,c)):
            a,other = int(a),int(other)
            cohort.append({'sku_id1':endpoint(a), 'sku_id2':endpoint(other), 'label':str(label),
                'split':'train', 'cohort_id':f'objective:{n}:{label}', 'population':population,
                'evaluation_scope':'training_diagnostic', 'payload_index1':a,'payload_index2':other,
                'mint_lineage':[audits[i][1] for i in (a,other) if i in audits],
                'difficulty_slice':'unknown',
                'consumed_example_ids':[n]})
    from training.difficulty import DifficultyEndpoint, measure_pair
    difficulty_endpoints = {}
    difficulty_pairs = {}
    for pair in cohort:
        if 'payload_index1' not in pair:
            pair['difficulty_reason'] = 'no_frozen_encoder_input'
            continue
        a, b = pair['payload_index1'], pair['payload_index2']
        for index in (a, b):
            if index not in difficulty_endpoints:
                difficulty_endpoints[index] = DifficultyEndpoint.from_text(bundle['payload'][index])
        key = (a, b, int(pair['label']))
        if key not in difficulty_pairs:
            difficulty_pairs[key] = measure_pair(difficulty_endpoints[a], difficulty_endpoints[b], key[2])
        evidence = difficulty_pairs[key]
        pair.update(difficulty_slice=evidence.difficulty, difficulty_reason=evidence.reason,
                    difficulty_text_overlap=evidence.text_overlap)
    frame = pd.DataFrame(cohort)
    for column in ('mint_lineage','consumed_example_ids'):
        frame[column] = frame[column].map(lambda value: json.dumps(value, sort_keys=True))
    from core.coverage_contracts import CohortCoverage
    coverage = CohortCoverage.model_validate({'cohort_sha256':digest(frame.fillna('').to_dict('records')),
        'pair_rows':len(frame), 'minted_endpoints_total':len(audits),
        'minted_endpoints_covered':len(set(node_ids)&set(audits)),
        'by_scope':frame.evaluation_scope.value_counts().to_dict(),
        'by_population':frame.population.value_counts().to_dict(),
        'by_difficulty':{name:int(frame.difficulty_slice.eq(name).sum())
                         for name in ('easy','medium','hard','unknown')},
        'unknown_difficulty_policy':'retain unknown; never invent easy/hard labels'})
    frame.fillna('').to_csv(folder/'pairs.csv', index=False)
    pd.DataFrame(list(rows.values())).fillna('').to_csv(folder/'catalog.csv', index=False)
    write(folder/'listings.json', {'schema':'er-graph-listings-v1','listings':list(records.values())})
    write(folder/'coverage.json', coverage.model_dump(mode='json'))
    return folder
