"""Read-only joins of saved decisions. Never recompute a gate or run a model."""
import csv
import io
import ast
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

from core.common import F, TRAIN_ROOT, resolve_model, training_cfg, data_cfg, CONFIG_PATH, TRAINING_CONFIG_PATH, VOCABULARY_CONFIG_PATH
from core.columns import read_column
from core.schemas import CANONICAL_RECORDS_COLUMNS
from graph_tracks.data import file_hash, load_records, load_text_cache
from model_tracks.config import load_config
from training.folds import normalize_gtin
from training.prepare_embeddings import input_identity, validate_prepared_provenance
from training.gate_replay import fired_stage
from graph_tracks.text_cache import texts_hash
from graph_tracks.config import load_config as graph_config
from graph_tracks.preflight import load_inputs as graph_inputs
from graph_tracks.report_attributes import load_inputs as report_attributes, FILENAME as REPORT_ATTRIBUTES
from core.identity_policy import POLICY_PATH, review_reason
from core.attribute_universe import attribute_registry
from core.attribute_conflicts import canonical_attribute_info, CRITICAL_NAME_BY_CENSUS_KEY, VETO_CENSUS_KEY_BY_DIMENSION, veto_eligibility_ledger, _universe_value
from core.attribute_decision import engine
from training.attribute_separation import ATTRIBUTE_SOURCES, ATTRIBUTE_UNAVAILABLE
from training.prepared_bundle import PreparedBundleManifest
from jev_reports import e, table
from training_reports import runs, entries, read, open_artifact, REPORT_ERRORS

router = APIRouter()
_LOADED_CONFIG_HASHES = {path:file_hash(path) for path in (
    CONFIG_PATH, TRAINING_CONFIG_PATH, VOCABULARY_CONFIG_PATH, POLICY_PATH,
    CONFIG_PATH.parent / 'identity_dimensions.yaml')}


def json_file(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def csv_rows(path):
    if not path.is_file():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def generation_rows(path, member):
    """Stream CSVs through the dashboard's safe archive reader."""
    with open_artifact(path,member) as handle:
        text = io.TextIOWrapper(handle)
        try:
            reader = csv.DictReader(text)
            if not set(reader.fieldnames or []) & {'fields_hit','presentations','augmentation','target_mode','difficulty_slice','difficulty'}:
                return
            yield from enumerate(reader,2)
        finally:
            text.detach()


def pair_key(a, b):
    return tuple(sorted((normalize_gtin(a), normalize_gtin(b))))


def ledger_path(name):
    """Ledger references are restricted to its directory, including symlinks."""
    base = F['decision_ledger'].parent.resolve()
    path = TRAIN_ROOT / name
    if not path.resolve().is_relative_to(base):
        raise ValueError('JEV ledger path escapes its evidence directory')
    return path


def audit_history(ledger):
    found = defaultdict(list)
    for item in ledger:
        sample_path = ledger_path(item['sample'])
        checkpoint_path = ledger_path(item['checkpoint'])
        sample = json_file(sample_path, [])
        staged = {(x.get('input_scope', ''), x['gtin1'], x['gtin2']) for x in sample}
        if not checkpoint_path.is_file():
            continue
        verified = bool(item.get('sample_sha256')) and file_hash(sample_path) == item['sample_sha256']
        for line_number, line in enumerate(checkpoint_path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            case = (row.get('input_scope', ''), row.get('gtin1'), row.get('gtin2'))
            if row.get('status') != 'ok' or case not in staged:
                continue
            found[pair_key(row['gtin1'], row['gtin2'])].append({
                **row, 'round': item['round'], 'kind': item.get('kind', 'fresh'),
                'artifact': checkpoint_path.relative_to(TRAIN_ROOT).as_posix(),
                'line': line_number, 'sample_checksum_matches': verified,
            })
    return found


def embedding_state(setup, listing_ids, model):
    cache = setup / 'shared_minilm__embeddings.npz'
    state = {'status': 'pending', 'reason': 'GPU result is absent', 'path': str(cache.relative_to(TRAIN_ROOT))}
    if not cache.is_file():
        return state, {}, None
    try:
        vectors, metadata = load_text_cache(cache, listing_ids)
        manifest = json_file(setup / 'prepared/input_manifest.json', {})
        validate_prepared_provenance(cache, metadata, manifest)
        if json_file(setup / 'embedding_inputs.json', {}).get('ids') != listing_ids:
            raise ValueError('Embedding input ID order/population differs from prepared listings')
        current = input_identity(setup, Path(resolve_model(model)))
        for key, value in current.items():
            if metadata.get(key) != value:
                raise ValueError(f'Current embedding input mismatch: {key}')
        state.update(status='valid', reason='Current inputs, policy, composition and checkpoint verified',
                     rows=len(vectors), dimensions=vectors.shape[1], metadata=metadata)
        return state, {sku: i for i, sku in enumerate(listing_ids)}, vectors
    except (ValueError, OSError, KeyError, TypeError) as exc:
        state.update(status='unusable', reason=str(exc))
        return state, {}, None


def request_state(setup, model, listing_ids):
    path = setup / 'embedding_inputs.json'
    if not path.is_file():
        path = F['decision_embedding_request']
    if not path.is_file():
        return {'status':'missing', 'reason':'No saved composed input texts'}, {}
    state = {'status':'valid', 'source':str(path.relative_to(TRAIN_ROOT))}
    try:
        request = json_file(path)
        if request.get('schema') != 'er-embedding-request-v2':
            raise ValueError('Unsupported prepared embedding request')
        if len(request['ids']) != len(request['texts']) or len(set(request['ids'])) != len(request['ids']):
            raise ValueError('Invalid composed text population')
        if request['ids'] != listing_ids:
            raise ValueError('Composed text IDs differ from prepared listing order/population')
        if request['metadata'].get('text_sha256') != texts_hash(request['texts']):
            raise ValueError('Composed text checksum mismatch')
        current = input_identity(setup, Path(resolve_model(model)))
        mismatch = [k for k,v in current.items() if request['metadata'].get(k) != v]
        if mismatch:
            state.update(status='historical', reason='Current input mismatch: ' + ', '.join(mismatch))
        state['metadata'] = request['metadata']
        return state, dict(zip(request['ids'],request['texts']))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        state.update(status='unusable', reason=str(exc))
        return state, {}


def signature(path):
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def attribute_evidence(left, right):
    """Existing engine interpretation of CURRENT evidence, never a saved verdict."""
    if not left or not right:
        return {'status':'missing canonical evidence', 'dimensions':{}}
    try:
        a,b = canonical_attribute_info(left), canonical_attribute_info(right)
        evidence = engine().evaluate(a,b,
                                     left_raw=left, right_raw=right)
        return {'status':'current shared-engine interpretation', 'dimensions':evidence.as_dict(),
                'values':{key:{'left':sorted(str(v) for v in _universe_value(a,key,spec)),
                               'right':sorted(str(v) for v in _universe_value(b,key,spec))}
                          for key,spec in attribute_registry().items()}}
    except (ValueError, KeyError, TypeError) as exc:
        return {'status':'unusable canonical evidence', 'reason':str(exc), 'dimensions':{}}


def tracking_attributes():
    return sorted(set(attribute_registry()) | {
        VETO_CENSUS_KEY_BY_DIMENSION.get(key,key) or key for key in
        set(ATTRIBUTE_SOURCES) | set(ATTRIBUTE_UNAVAILABLE) | set(CRITICAL_NAME_BY_CENSUS_KEY.values())})


def attribute_tracking(traces, reports, available):
    """Registry-led rows; model/JEV pair scores are context, not attribute weights."""
    registry = attribute_registry()
    try:
        census = json_file(F['decision_attribute_census'], {})
        policy = veto_eligibility_ledger(census=census.get('census',{}).get('keys',{}))
    except (ValueError, OSError, KeyError):
        policy = veto_eligibility_ledger(census={})
    keys = tracking_attributes()
    metrics = defaultdict(list)
    for report in reports:
        data = report['report']
        for name, values in data.get('attribute_errors',{}).items():
            metrics[name.split('/')[0]].append({'source':report['source'], 'kind':'attribute_errors',
                                               'slice':name, 'metrics':values})
        for name, values in data.get('robust_validation',{}).get('slice_aggregate',{}).items():
            dimension, _, label = name.partition('::')
            metrics[label if dimension == 'attribute' else dimension].append(
                {'source':report['source'], 'kind':'aggregate_validation_slice', 'slice':name, 'metrics':values})
    for run,path in available.items():
        try:
            for member in entries(path):
                if member.endswith(('attribute_separation_summary.csv','attribute_separation_values.csv')):
                    for line,row in enumerate(csv.DictReader(io.StringIO(read(path,member).decode())),2):
                        if row.get('attribute'):
                            metrics[row['attribute']].append({'source':run+'/'+member,'line':line,
                                'kind':'attribute_separation', 'metrics':row})
        except REPORT_ERRORS:
            continue
    rows = []
    for key in keys:
        dimension = CRITICAL_NAME_BY_CENSUS_KEY.get(key,key)
        canonical_field = ATTRIBUTE_SOURCES.get(dimension,(None,None))[0]
        cases = []
        for trace_index, trace in enumerate(traces):
            comparison = trace['attribute_evidence']['dimensions'].get(key)
            j = [{'history_index':i,'saved_attribute_state':h.get('attribute_states',{}).get(key,
                       h.get('attribute_states',{}).get(dimension))} for i,h in enumerate(trace['jev_history'])]
            cases.append({'trace_index':trace_index, 'pair':trace['pair'], 'current_comparison':comparison,
                'comparison_values':trace['attribute_evidence'].get('values',{}).get(key),
                'canonical_values':{g:(trace['canonical_evidence'][g] or {}).get(canonical_field) if canonical_field else None
                                    for g in trace['pair']},
                'jev_attribute_states':j,
                'controlled_ablation_results':[r for r in trace.get('controlled_ablation_results',[]) if r['attribute'] == key]})
        rows.append({'attribute':key,'dimension':dimension,'canonical_field':canonical_field,
                     'registry_kind':getattr(registry.get(key),'kind',None),
                     'unavailable_reason':ATTRIBUTE_UNAVAILABLE.get(dimension),
                     'gate_policy':policy.get(dimension,policy.get(key)),
                     'current_page_states':dict(Counter((x['current_comparison'] or {}).get('state','not registered') for x in cases)),
                     'evaluation_metrics':metrics.get(dimension,[]) + (metrics.get(key,[]) if key != dimension else []),
                     'cases':cases})
    return rows


def generation_tracking(suite, available, limit, traces):
    """Use frozen bundle headers and emitted CSVs; never unpickle or generate."""
    bundle = TRAIN_ROOT / suite.text_bundle
    header_path = bundle.with_suffix(bundle.suffix + '.json')
    header = {'source':str(header_path.relative_to(TRAIN_ROOT)), 'status':'missing'}
    if header_path.is_file():
        try:
            manifest = PreparedBundleManifest.model_validate_json(header_path.read_text())
            header.update(status='bundle checksum verified; current-input readiness not established' if bundle.is_file() and file_hash(bundle) == manifest.sha256 else 'checksum mismatch',
                          manifest=manifest.model_dump(mode='json'))
        except (ValueError, OSError) as exc:
            header.update(status='unusable', reason=str(exc))
    registry = attribute_registry()
    sources = dict(available)
    visibility = F['decision_visibility']
    if visibility.is_dir():
        sources[str(visibility.relative_to(TRAIN_ROOT))] = visibility
    groups = Counter()
    previews = []
    inventory = []
    linked = defaultdict(list)
    keys = {tuple(trace['pair']) for trace in traces}
    for run,path in sources.items():
        try:
            members = entries(path)
            context = saved_run_context(path,members)
            for member in members:
                if not member.endswith('.csv'):
                    continue
                count, attributed = 0, 0
                for line,row in generation_rows(path,member):
                    count += 1
                    hit = row.get('fields_hit','')
                    try:
                        hits = ast.literal_eval(hit) if hit else []
                    except (ValueError,SyntaxError):
                        hits = []
                    if not isinstance(hits,(list,tuple,set)):
                        hits = []
                    if row.get('attribute'):
                        hits = [*hits,row['attribute']]
                    mapped = set()
                    for field in hits:
                        field = str(field)
                        candidate = data_cfg().decision_attribute_aliases.get(field,
                            VETO_CENSUS_KEY_BY_DIMENSION.get(field,field.replace('_',' ')))
                        if candidate in registry or candidate in ATTRIBUTE_SOURCES:
                            mapped.add(candidate)
                    difficulty = row.get('difficulty_slice') or row.get('difficulty')
                    basis = 'explicit' if difficulty else 'saved population (not a measured difficulty grade)'
                    difficulty = difficulty or row.get('population') or 'unknown'
                    mode = row.get('augmentation') or row.get('target_mode') or 'unknown'
                    frozen = (context or {}).get('inputs') or {}
                    text_context = frozen.get('text',{}) if member.startswith('text/') else {}
                    profile = row.get('masking_profile') or text_context.get('masking_profile') or 'unknown'
                    variant = row.get('generation_variant') or row.get('target_mode') or 'unknown'
                    attributed += bool(mapped)
                    for key in mapped or {'unattributed'}:
                        groups[(key,difficulty,basis,profile,mode,variant,run,member,row.get('fold',''),row.get('epoch',''))] += 1
                    if len(previews) < limit:
                        previews.append({'source':run+'/'+member,'line':line,'attributes':sorted(mapped),'row':row})
                    a,b = row.get('a_gtin'),row.get('b_gtin')
                    targets = [pair_key(a,b)] if a and b else [k for k in keys if normalize_gtin(read_column(row,'gtin')) in k]
                    for key in targets:
                        if key in keys and len(linked[key]) < limit:
                            linked[key].append({'source':run+'/'+member,'line':line,'attributes':sorted(mapped),'row':row,
                                                'association':'saved pair endpoints' if a and b else 'entity-only; counterpart unavailable',
                                                'run_context':context})
                if count:
                    inventory.append({'source':run+'/'+member,'rows':count,'attribute_mapped_rows':attributed,
                                      'unattributed_rows':count-attributed})
        except REPORT_ERRORS as exc:
            inventory.append({'source':run,'error':str(exc)})
    dimensions = ('attribute','difficulty_slice','difficulty_basis','masking_profile','masking_mode',
                  'generated_data_variant','run','artifact','fold','epoch')
    combinations = [{**dict(zip(dimensions,key)),'rows':count} for key,count in sorted(groups.items())]
    for trace in traces:
        trace['generation_evidence'] = linked[tuple(trace['pair'])]
    return {'bundle':header,'combinations':combinations,'inventory':inventory,'preview':previews,
            'meaning':'Only explicitly saved field hits establish attribute membership. Payload indices remain bundle-scoped. Missing difficulty, masking profile or generated variant stays unknown. Pair and entity associations are distinguished; previews are bounded.'}


def report_inventory(available):
    """Use the dashboard's existing run/archive discovery and safe readers."""
    reports = []
    root = F['decision_training_report']
    if root.is_file():
        reports.append({'source': root.relative_to(TRAIN_ROOT).as_posix(),
                        'sha256': file_hash(root), 'report': json_file(root), 'run': None})
    for run, path in available.items():
        try:
            for member in entries(path):
                if Path(member).name == 'report.json':
                    reports.append({'source': run + '/' + member, 'run': run,
                                    'report': json.loads(read(path, member))})
        except REPORT_ERRORS:
            continue
    return reports


def saved_run_context(path, members):
    member = next((x for x in members if Path(x).name == 'suite_manifest.json'), None)
    if member is None:
        return None
    manifest = json.loads(read(path,member))
    return {'source':member,'run_tag':manifest.get('run_tag'),'config':manifest.get('config'),
            'inputs':manifest.get('inputs')}


def saved_model_evidence(keys, sku_gtin, available):
    """Join historical model scores through frozen source rows when available."""
    evidence = defaultdict(list)
    inventory = []
    for run, path in available.items():
        try:
            members = entries(path)
            context = saved_run_context(path,members)
            mapping = {}
            mapping_sources = []
            for member in members:
                if member.endswith('_prepared_inputs/canonical_records.csv'):
                    mapping_sources.append(member)
                    for row in csv.DictReader(io.StringIO(read(path, member).decode())):
                        for source in json.loads(row.get('source_rows') or '[]'):
                            sku = str(read_column(source, 'sku_id'))
                            gtin = normalize_gtin(read_column(source, 'gtin', row['gtin']))
                            if sku in mapping and mapping[sku] != gtin:
                                raise ValueError('Conflicting historical SKU→GTIN mapping')
                            mapping[sku] = gtin
            for member in members:
                if not member.endswith(('scored_pairs.csv', 'attribute_error_examples.csv', 'retrieval_queries.csv')):
                    continue
                manifest_member = next((x for x in members if Path(x).parent == Path(member).parent
                                        and x.endswith('report_manifest.json')), None)
                manifest = json.loads(read(path, manifest_member)) if manifest_member else None
                count, matched, unverified = 0, 0, 0
                for line, row in enumerate(csv.DictReader(io.StringIO(read(path, member).decode())), 2):
                    count += 1
                    a = row.get('sku_id1', row.get('product_id1', row.get('sku_id_a')))
                    b = row.get('sku_id2', row.get('product_id2', row.get('sku_id_b')))
                    direct_a, direct_b = row.get('gtin_a', row.get('gtin1')), row.get('gtin_b', row.get('gtin2'))
                    verified = bool(direct_a and direct_b) or bool(a in mapping and b in mapping)
                    lookup = mapping if verified and not direct_a else sku_gtin
                    if direct_a and direct_b:
                        key = pair_key(direct_a, direct_b)
                    elif a in lookup and b in lookup:
                        key = pair_key(lookup[a], lookup[b])
                    else:
                        # Retrieval rows describe a query, not an individual decision.
                        sku = str(read_column(row, 'sku_id'))
                        g = mapping.get(sku, sku_gtin.get(sku))
                        target_keys = [k for k in keys if g in k] if g else []
                        for k in target_keys:
                            evidence[k].append({'run':run, 'artifact':member, 'line':line,
                                                'kind':'retrieval_query_summary', 'row':row,
                                                'mapping_verified':sku in mapping,
                                                'mapping_sources':mapping_sources, 'manifest':manifest,
                                                'run_context_source':context['source'] if context else None})
                        matched += len(target_keys)
                        continue
                    if not verified:
                        unverified += 1
                    if key in keys:
                        matched += 1
                        evidence[key].append({'run':run, 'artifact':member, 'line':line, 'kind':'saved_pair_prediction',
                                              'row':row, 'mapping_verified':verified,
                                              'mapping_sources':mapping_sources, 'manifest':manifest,
                                              'run_context_source':context['source'] if context else None})
                inventory.append({'run':run, 'artifact':member, 'rows':count, 'matched_page_rows':matched,
                                  'rows_using_current_catalog_mapping':unverified})
        except REPORT_ERRORS as exc:
            inventory.append({'run':run, 'error':str(exc)})
    return evidence, inventory


def _controlled_report(path, attribute=''):
    if not path.is_file():
        return {'status':'not measured', 'rows':[], 'meaning':'Frozen-checkpoint ablation measures changes from removing declared inputs; it is not an intrinsic attribute weight'}
    try:
        from model_tracks.ablation import validate_sources, resolve, load_prepared, verify_threshold_binding, request_context
        payload = json_file(path)
        if payload.get('schema') != 'er-attribute-ablation-report-v1':
            raise ValueError('unsupported ablation report')
        request_path = resolve(payload['request_path'])
        request = json_file(request_path)
        with request_context(request_path):
            validate_sources(request)
            load_prepared(request_path,request).close()
            if file_hash(request_path) != payload['request_sha256'] or file_hash(resolve(payload['result_path'])) != payload['result_sha256']:
                raise ValueError('ablation request/result changed')
            if verify_threshold_binding(request,payload['threshold_provenance']) != payload.get('threshold_binding'):
                raise ValueError('threshold checkpoint binding missing or invalid')
            threshold = payload['threshold_provenance']
            if file_hash(resolve(threshold['path'])) != threshold['sha256']:
                raise ValueError('saved baseline threshold report changed')
        rows = [r for r in payload['rows'] if not attribute or r['attribute'] == attribute]
        return {**payload, 'status':'verified frozen-checkpoint intervention', 'rows':rows,
                'meaning':payload['intervention']+'; '+payload['retrieval_scope']}
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return {'status':'invalid or stale controlled ablation: '+str(exc), 'rows':[],
                'meaning':'Stale influence scores are withheld; prepare and evaluate current inputs'}


def controlled_influence(attribute=''):
    pointer = F['decision_ablation_report']
    paths = sorted(pointer.parent.glob('*/report.json')) if pointer.parent.is_dir() else []
    paths.extend(sorted((TRAIN_ROOT/'results/model_tracks').glob('*/*/ablation/report.json')))
    if pointer.is_file():
        paths.append(pointer)
    if not paths:
        return _controlled_report(pointer, attribute)
    seen, reports, rows, invalid = set(), [], [], []
    for path in paths:
        report = _controlled_report(path, attribute)
        if report['status'] != 'verified frozen-checkpoint intervention':
            invalid.append({'source':str(path),'status':report['status']})
            continue
        if report['request_sha256'] in seen:
            continue
        seen.add(report['request_sha256'])
        reports.append({k:v for k,v in report.items() if k != 'rows'})
        rows.extend({**row, 'track':report['track'], 'request_sha256':report['request_sha256'],
                     'threshold':report['threshold'], 'threshold_source':report['threshold_source']}
                    for row in report['rows'])
    return {'status':'verified frozen-checkpoint intervention' if reports else 'invalid or stale controlled ablation',
            'rows':rows,'reports':reports,'invalid_reports':invalid,
            'meaning':'Declared-input interventions at frozen checkpoints and thresholds; fixed candidate catalog with query-only interventions. Missing axes remain unknown'}


def inspect(gtin1='', gtin2='', gate='', scope='', round=None, offset=0, limit=50, attribute=''):
    for path, expected in _LOADED_CONFIG_HASHES.items():
        if not path.is_file() or file_hash(path) != expected:
            raise HTTPException(409, 'Configuration changed since startup; restart the dashboard before inspecting current decisions')
    if offset < 0 or not 1 <= limit <= 100:
        raise HTTPException(422, 'offset must be nonnegative; limit must be 1–100')
    if bool(gtin1) != bool(gtin2):
        raise HTTPException(422, 'Provide both GTINs for a pair trace')
    suite = load_config(F['decision_suite_config'])
    setup = TRAIN_ROOT / suite.setup_dir
    available = runs()
    ledger = json_file(F['decision_ledger'], [])
    watched = [F[k] for k in ('decision_ledger','decision_rebuild_report','decision_training_report',
               'decision_suite_config','decision_embedding_request','canonical_records','gate_results','labeled_pairs',
               'final_validation','embedding_similarities','package_gate_impact_pairs','dataset','dataset_deduped','field_ablation',
               'decision_attribute_census','decision_visibility','decision_ablation_report')]
    bundle = TRAIN_ROOT / suite.text_bundle
    watched += [bundle,bundle.with_suffix(bundle.suffix+'.json')]
    if F['decision_visibility'].is_dir():
        watched += [F['decision_visibility'] / member for member in entries(F['decision_visibility'])]
    watched += [setup / name for name in ('eligible_catalog.csv','gnn_only.yaml','hybrid.yaml','setup_manifest.json',
                'embedding_inputs.json','shared_minilm__embeddings.npz','prepared/listings.json',
                'prepared/input_manifest.json','prepared/pairs.csv','prepared/' + REPORT_ATTRIBUTES)]
    watched += list(_LOADED_CONFIG_HASHES)
    watched += [ledger_path(item[key]) for item in ledger for key in ('sample','checkpoint')]
    for path in available.values():
        watched.extend([path] if path.is_file() else [path / member for member in entries(path)])
    signatures = {path: signature(path) for path in watched}
    catalog = csv_rows(setup / 'eligible_catalog.csv')
    by_gtin = defaultdict(list)
    sku_gtin = {}
    for row in catalog:
        sku = str(read_column(row, 'sku_id'))
        gtin = normalize_gtin(read_column(row, 'gtin'))
        by_gtin[gtin].append(row)
        sku_gtin[sku] = gtin
    records = load_records(setup / 'prepared/listings.json')
    listing_records = {x['sku_id']: x for x in records}
    prepared = {'status':'valid'}
    try:
        graph_inputs(graph_config(setup / 'gnn_only.yaml'))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        prepared.update(status='unusable', reason=str(exc))
    try:
        attributes = report_attributes(setup / 'prepared/listings.json', records)
    except (ValueError, OSError, KeyError) as exc:
        attributes = {}
        prepared['report_attributes'] = {'status':'unusable','reason':str(exc)}
    text_state, texts = request_state(setup, suite.text_model, list(listing_records))
    state, vector_ids, vectors = embedding_state(setup, list(listing_records), suite.text_model)
    histories = audit_history(ledger)
    gates = csv_rows(F['gate_results'])
    requested = pair_key(gtin1, gtin2) if gtin1 else None
    selected = [x for x in gates if (not requested or pair_key(x['gtin1'], x['gtin2']) == requested)
                and (not gate or x['gate_decision'] == gate)
                and (not scope and round is None or any(
                    (not scope or h.get('input_scope', '') == scope) and (round is None or h['round'] == round)
                    for h in histories.get(pair_key(x['gtin1'], x['gtin2']), [])))]
    selected.sort(key=lambda x: not bool(histories.get(pair_key(x['gtin1'], x['gtin2']))))
    if requested and not selected and not gate and not scope and round is None:
        selected = [{'gtin1': gtin1, 'gtin2': gtin2, 'gate_decision': 'not in current gate population', 'gate_reason': ''}]
    page = selected[offset:offset + limit]
    keys = {pair_key(x['gtin1'], x['gtin2']) for x in page}
    models, model_inventory = saved_model_evidence(keys, sku_gtin, available)
    pair_artifacts = defaultdict(list)
    for artifact in ('final_validation', 'embedding_similarities', 'package_gate_impact_pairs'):
        for line, row in enumerate(csv_rows(F[artifact]), 2):
            if row.get('gtin1') and row.get('gtin2'):
                key = pair_key(row['gtin1'], row['gtin2'])
                if key in keys:
                    pair_artifacts[key].append({'source':str(F[artifact].relative_to(TRAIN_ROOT)), 'line':line, 'row':row})
    canonical = {normalize_gtin(x['gtin']): x for x in csv_rows(F['canonical_records'])
                 if normalize_gtin(x['gtin']) in {g for key in keys for g in key}}
    labels = {pair_key(x['gtin1'], x['gtin2']): x for x in csv_rows(F['labeled_pairs'])}
    prepared_pairs = defaultdict(list)
    for row in csv_rows(setup / 'prepared/pairs.csv'):
        # The versioned graph pair contract still spells these product_id1/2.
        a, b = row.get('sku_id1', row.get('product_id1')), row.get('sku_id2', row.get('product_id2'))
        if a in sku_gtin and b in sku_gtin:
            key = pair_key(sku_gtin[a], sku_gtin[b])
            if key in keys:
                prepared_pairs[key].append({**row, 'sku_id1': a, 'sku_id2': b})
    traces = []
    for row in page:
        key = pair_key(row['gtin1'], row['gtin2'])
        left, right = [by_gtin[g] for g in key]
        cosine = None
        if vectors is not None:
            a = [vector_ids[str(read_column(x, 'sku_id'))] for x in left]
            b = [vector_ids[str(read_column(x, 'sku_id'))] for x in right]
            if a and b:
                values = vectors[a] @ vectors[b].T
                cosine = {'min': float(values.min()), 'mean': float(values.mean()),
                          'max': float(values.max()), 'listing_combinations': int(values.size)}
        traces.append({'pair': key, 'gate': row, 'gate_source': str(F['gate_results'].relative_to(TRAIN_ROOT)),
                       'fired_stage':fired_stage(row.get('gate_reason','')),
                       'identity_review_reasons':{g:review_reason(g) for g in key},
                       'canonical_evidence': {g: canonical.get(g) for g in key},
                       'attribute_evidence':attribute_evidence(canonical.get(key[0]),canonical.get(key[1])),
                       'listings': {g: by_gtin[g] for g in key},
                       'graph_inputs': {g: [listing_records[str(read_column(x, 'sku_id'))]
                                           for x in by_gtin[g] if str(read_column(x, 'sku_id')) in listing_records] for g in key},
                       'prepared_pairs': prepared_pairs[key], 'derived_label': labels.get(key),
                       'prepared_model_texts':{str(read_column(x,'sku_id')):texts.get(str(read_column(x,'sku_id')))
                                               for x in left + right},
                       'report_attribute_classes':{str(read_column(x,'sku_id')):
                            {k:sorted(v) for k,v in attributes.get(str(read_column(x,'sku_id')),{}).items()}
                            for x in left + right},
                       'jev_history': histories.get(key, []), 'embedding_cosine': cosine,
                       'saved_model_evidence':models[key], 'validation_and_gate_artifacts':pair_artifacts[key]})
    rebuild = json_file(F['decision_rebuild_report'], {})
    provenance = []
    for name, expected in rebuild.get('source_sha256', {}).items():
        path = TRAIN_ROOT / name
        if not path.resolve().is_relative_to(TRAIN_ROOT.resolve()):
            continue
        actual = file_hash(path) if path.is_file() else None
        provenance.append({'source': name, 'expected': expected, 'actual': actual, 'matches': expected == actual})
    reports = report_inventory(available)
    run_contexts = {}
    for run,path in available.items():
        try:
            run_contexts[run] = saved_run_context(path,entries(path))
        except REPORT_ERRORS as exc:
            run_contexts[run] = {'status':'unavailable','reason':str(exc)}
    influence = controlled_influence(attribute)
    ablation_pairs = defaultdict(list)
    for result in influence['rows']:
        if result.get('gtin1') and result.get('gtin2'):
            ablation_pairs[pair_key(result['gtin1'],result['gtin2'])].append(result)
    for trace in traces:
        trace['controlled_ablation_results'] = ablation_pairs.get(tuple(trace['pair']), [])
    tracking = attribute_tracking(traces, reports, available)
    generation = generation_tracking(suite,available,limit,traces)
    for row in tracking:
        row['controlled_ablation_results'] = [r for r in influence['rows'] if r['attribute'] == row['attribute']]
        row['generation_slices'] = [x for x in generation['combinations'] if x['attribute'] in {row['attribute'],row['dimension']}]
    if attribute and attribute not in {x['attribute'] for x in tracking}:
        raise HTTPException(422, 'Unknown attribute; select a registered attribute')
    if attribute:
        tracking = [x for x in tracking if x['attribute'] == attribute]
    if any(signature(path) != before for path,before in signatures.items()):
        raise HTTPException(409, 'Evidence changed while building the trace; refresh to inspect a consistent snapshot')
    return {'total': len(selected), 'offset': offset, 'limit': limit, 'traces': traces,
            'gate_census': dict(Counter(x['gate_decision'] for x in gates)), 'embedding': state,
            'rebuild_report': rebuild, 'rebuild_provenance': provenance, 'reports': reports,
            'prepared_inputs':prepared, 'composed_texts':text_state,
            'current_gate_config':training_cfg().gate.model_dump(mode='json'),
            'attribute_tracking':tracking,
            'generation_tracking':generation,
            'run_contexts':run_contexts,
            'coverage_gaps':[
                {'item':'Historical attribute execution','status':'Saved gate reasons available; intermediate attribute engine metrics reconstructed only for current evidence'},
                {'item':'Joint difficulty/masking/generation slices','status':'Partial' if generation['inventory'] else 'Missing',
                 'unattributed_rows':sum(x.get('unattributed_rows',0) for x in generation['inventory'])},
                {'item':'Attribute influence','status':influence['status']},
                {'item':'Retrieval rank per candidate pair','status':'Saved query summaries available; individual neighbor ranks not established by these summaries'},
                {'item':'Independent human truth','status':'Gate-derived labels and JEV judgments have separate provenance; neither is presented as verified human truth'},
            ],
            'attribute_influence':influence,
            'payload_variant_comparisons':{'source':str(F['field_ablation'].relative_to(TRAIN_ROOT)), 'rows':csv_rows(F['field_ablation'])},
            'model_artifact_inventory':model_inventory}


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(x) for x in value]
    return value


@router.get('/api/decisions')
def decisions_json(gtin1: str = '', gtin2: str = '', gate: str = '', scope: str = '', round: int | None = None,
                   offset: int = 0, limit: int = 50, attribute: str = ''):
    return Response(json.dumps(clean_json(inspect(gtin1, gtin2, gate, scope, round, offset, limit, attribute))),
                    media_type='application/json', headers={'Cache-Control': 'no-store'})


@router.get('/decisions', response_class=HTMLResponse)
def decisions(gtin1: str = '', gtin2: str = '', gate: str = '', scope: str = '', round: int | None = None,
              offset: int = 0, limit: int = 50, attribute: str = ''):
    result = inspect(gtin1, gtin2, gate, scope, round, offset, limit, attribute)
    query = dict(gtin1=gtin1, gtin2=gtin2, gate=gate, scope=scope, offset=offset, limit=limit,attribute=attribute)
    if round is not None:
        query['round'] = round
    body = '<h1>Attribute decision tracking</h1><p>Attribute → extracted values, confidence and comparison metrics → saved gate reason → JEV evidence → model outcomes and attribute slices. Pair identifiers connect the evidence; attributes organize the inspection.</p>'
    body += '<p><a href="/gate">Gate</a> · <a href="/jev">JEV audits</a> · <a href="/training">Training runs</a> · <a href="/graphs">Graph tracks</a> · <a href="/api/decisions?' + e(urlencode(query)) + '">Download this trace as JSON</a></p>'
    body += '<form onsubmit="for(const input of this.querySelectorAll(\'input\')){if(!input.value)input.disabled=true}"><label>Attribute <select name="attribute"><option value="">All attributes</option>' + ''.join('<option value="'+e(key)+'" '+('selected' if key == attribute else '')+'>'+e(key)+'</option>' for key in tracking_attributes()) + '</select></label> ' + ''.join('<label>' + e(label) + ' <input name="' + name + '" value="' + e(value) + '"></label> ' for name, label, value in [('gtin1','GTIN A',gtin1),('gtin2','GTIN B',gtin2),('gate','Gate decision',gate),('scope','JEV input scope',scope),('round','JEV round',round or '')]) + '<button>Inspect</button></form>'
    body += '<h2>Attribute map</h2><p>Counts describe the displayed pair page. Comparison metrics use the existing shared attribute engine on current evidence; they are not recovered historical execution traces. Saved gate reasons remain separate. JEV/model scores assess pairs unless the artifact explicitly records an attribute assessment.</p>'
    for row in result['attribute_tracking']:
        policy = row['gate_policy'] or {}
        body += '<details><summary>'+e(row['attribute'])+' · '+e(policy.get('decision_participation','no saved policy mapping'))+' · '+e(row['current_page_states'])+'</summary>'
        body += '<p>Canonical field: '+e(row['canonical_field'])+'; registry kind: '+e(row['registry_kind'])+'.</p>'
        body += table(['Pair','Current state','Current comparison','Fallback source','Saved gate stage','JEV cases','Model results'], [[
            ' / '.join(case['pair']), (case['current_comparison'] or {}).get('state','missing'),
            (case['current_comparison'] or {}).get('result','missing'), (case['current_comparison'] or {}).get('fallback_from',''),
            result['traces'][case['trace_index']]['fired_stage'], len(case['jev_attribute_states']),
            len(result['traces'][case['trace_index']]['saved_model_evidence'])] for case in row['cases']])
        body += '<pre>'+e(json.dumps(row,indent=2))+'</pre></details>'
    body += '<h2>Model influence per attribute</h2><p>'+e(result['attribute_influence']['status'])+'. '+e(result['attribute_influence']['meaning'])+'.</p>'
    if result['attribute_influence']['rows']:
        body += '<pre>'+e(json.dumps(result['attribute_influence'],indent=2))+'</pre>'
    generation = result['generation_tracking']
    body += '<h2>Attribute × difficulty × masking × generated data</h2><p>Bundle provenance: '+e(generation['bundle']['status'])+'. '+e(generation['meaning'])+'</p>'
    combinations = [x for x in generation['combinations'] if not attribute or x['attribute'] == attribute]
    body += table(['Attribute','Difficulty / population','Basis','Masking profile','Masking mode','Generation variant','Source','Fold','Epoch','Rows'],
        [[x[k] for k in ('attribute','difficulty_slice','difficulty_basis','masking_profile','masking_mode','generated_data_variant','run','fold','epoch','rows')] for x in combinations])
    body += '<details><summary>Generated-data sources, lineage and coverage gaps</summary><pre>'+e(json.dumps(generation,indent=2))+'</pre></details>'
    body += '<h2>Current coverage</h2>' + table(['Gate decision','Pairs'], result['gate_census'].items())
    body += '<details><summary>Coverage gaps</summary>' + table(['Item','Status'],[[x['item'],x['status']] for x in result['coverage_gaps']]) + '</details>'
    body += '<p>Embeddings: ' + e(result['embedding']['status']) + ' — ' + e(result['embedding']['reason']) + '. Cosine is min/mean/max across all eligible listing combinations. Gate similarity is lexical similarity and uses a different scale.</p>'
    body += '<p>Prepared training inputs: ' + e(result['prepared_inputs']['status']) + ' ' + e(result['prepared_inputs'].get('reason','')) + '. Composed texts: ' + e(result['composed_texts']['status']) + ' ' + e(result['composed_texts'].get('reason','')) + '.</p>'
    body += '<details><summary>Current gate rules and thresholds (config)</summary><pre>' + e(json.dumps(result['current_gate_config'],indent=2)) + '</pre></details>'
    body += '<h2>Rebuild provenance</h2>' + table(['Source','Checksum matches current file'], [(x['source'],x['matches']) for x in result['rebuild_provenance']])
    body += '<details><summary>Saved gate rebuild report, transitions and label population</summary><pre>' + e(json.dumps(result['rebuild_report'], indent=2)) + '</pre></details>'
    body += f'<h2>Pair traces ({result["total"]}; showing {offset + 1 if result["traces"] else 0}–{offset + len(result["traces"])})</h2>'
    for trace in result['traces']:
        a,b = trace['pair']
        body += '<details><summary>' + e(a + ' / ' + b + ' · ' + trace['gate']['gate_decision'] + ' · ' + trace['gate'].get('gate_reason','')) + '</summary>'
        body += '<p>' + ' · '.join('<a href="/catalog?gtin=' + e(g) + '">Source listings ' + e(g) + '</a>' for g in trace['pair']) + '</p>'
        body += '<p>Decision stage: ' + e(trace['fired_stage']) + '; saved lexical similarity: ' + e(trace['gate'].get('similarity','missing')) + '; embedding cosine: ' + e(trace['embedding_cosine']) + '.</p>'
        fields = [field for field in CANONICAL_RECORDS_COLUMNS if field.endswith(('_set','_confidence','_consistency','_flags'))]
        body += '<h3>Extracted attributes at the gate</h3>' + table(['Field',a,b], [[field,
                   (trace['canonical_evidence'][a] or {}).get(field,'missing'),
                   (trace['canonical_evidence'][b] or {}).get(field,'missing')] for field in fields])
        body += table(['Round','Input','Order','Saved gate','Score','Stratum','Source line','Sample verified'], [[h['round'],h.get('input_scope',''),h.get('copy',''),h.get('gate',''),h.get('noul'),h.get('stratum',''),h['artifact'] + ':' + str(h['line']),h['sample_checksum_matches']] for h in trace['jev_history']])
        body += '<h3>Saved model decisions and retrieval</h3>' + table(['Run','Artifact','Kind','Score','Prediction','Split','Frozen mapping verified'], [[x['run'],x['artifact'],x['kind'],x['row'].get('score',''),x['row'].get('prediction',''),x['row'].get('split',''),x['mapping_verified']] for x in trace['saved_model_evidence']])
        body += '<p>Historical scores retain their saved threshold and checkpoint manifest. Unverified SKU mappings use the current catalog only as an inspection aid.</p>'
        body += '<p>Derived labels are gate-generated training targets. Saved JEV scores and report metrics do not establish human truth. Missing evidence stays missing.</p><pre>' + e(json.dumps(trace,indent=2)) + '</pre></details>'
    if offset + limit < result['total']:
        body += '<p><a href="/decisions?' + e(urlencode({**query,'offset':offset+limit})) + '">Next page</a></p>'
    if offset:
        body += '<p><a href="/decisions?' + e(urlencode({**query,'offset':max(0,offset-limit)})) + '">Previous page</a></p>'
    body += '<h2>Evaluation reports and slices</h2><p>Aggregate slice metrics are historical run evidence. They are not assigned to a pair without saved pair membership or prediction records. Pair inputs above expose brand, category, country, extracted attribute, split and saved JEV stratum.</p>'
    for report in result['reports']:
        body += '<details><summary>' + e(report['source']) + '</summary>'
        if report['run']:
            body += '<p><a href="/training?' + e(urlencode({'run':report['run']})) + '">Open run artifacts and plots</a></p>'
        data = report['report']
        body += '<h3>Attribute errors</h3>' + table(['Attribute / label','Examples','Errors','Error rate','Mean score'],
                   [[key,v.get('n'),v.get('errors'),v.get('error_rate'),v.get('mean_score')]
                    for key,v in data.get('attribute_errors',{}).items()])
        robust = data.get('robust_validation',{})
        body += '<h3>Validation slices</h3>' + table(['Dimension / slice','Fold observations','Error rate','PR AUC','Recall'],
                   [[key,v.get('n'),v.get('error_rate_mean','unavailable'),v.get('pr_auc_mean','unavailable'),v.get('recall_mean','unavailable')]
                    for key,v in robust.get('slice_aggregate',{}).items()])
        body += '<details><summary>Slice support, fold, repeat, threshold and status</summary>' + table(
                   ['Dimension','Slice','Fold','Repeat','N','Positive','Negative','Threshold','Status'],
                   [[v.get(k) for k in ('dimension','slice','fold','repeat','n','n_positive','n_negative','threshold','status')]
                    for v in robust.get('slices',[])]) + '</details>'
        body += '<pre>' + e(json.dumps(report['report'],indent=2)) + '</pre></details>'
    return HTMLResponse('<!doctype html><html><head><title>Decision evidence</title><style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:12px 0}input{max-width:180px}</style></head><body>' + body + '</body></html>', headers={'Cache-Control':'no-store'})
