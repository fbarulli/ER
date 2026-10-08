"""One exhaustive pair cohort, shared by all frozen model ablations."""
from __future__ import annotations

import copy
import io
import json

import pandas as pd

from core.coverage_contracts import (
    UNKNOWN_DIMENSION_VALUE,
    CohortCoverage,
    Difficulty,
    DimensionAccounting,
    ReportCoverageContract,
    TaggedDimensionRecord,
)
from core.eval_trace import canonical_dimension
from core.run_log import RunLogger
from core.step_trace import timed
from core.tracing import flush_stage_trace, stage_trace
from model_tracks.shared_graph_data import CLEAN_BACKUP_SUFFIX
from model_tracks.training_data import augmentation_node_id, canonical_node_id

_LOG = RunLogger(__name__)


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import training_cfg
    return training_cfg().preparation.graph_setup


#: The stage name this module owns in the ONE consolidated pipeline trace.
STAGE = "ablation_cohort"

#: The cohort folder member that carries the validated GENERAL coverage
#: contract (``ReportCoverageContract``) beside the frozen ``coverage.json``.
#: ADDITIVE: a new member, so the four frozen artifacts keep their bytes. The
#: contract is persisted because a validated contract that is thrown away proves
#: nothing to a later reader: the emitted cohort now ships the per-record census
#: it was validated against.
COHORT_CONTRACT_FILE = "coverage_contract.json"

#: The module's trace writer: the shared shim's slot (``None`` until first use;
#: see :func:`core.tracing.stage_trace`), so importing this module never touches
#: the trace layout.
_TRACE = None


def trace():
    """The ONE writer for the ``ablation_cohort`` stage of the current run."""
    global _TRACE
    _TRACE = stage_trace(STAGE, _TRACE)
    return _TRACE


def flush_trace():
    """Commit this process's cohort rows once; a no-op while empty."""
    return flush_stage_trace(_TRACE)


# ── the cohort frame's own traceability dimensions (ADDITIVE, contract-only) ──
# The frame below IS the registry of its dimensions: these names are its
# columns, tagged per record, never a second list of the data. They feed the
# project-wide ``ReportCoverageContract`` *alongside* the frozen
# ``CohortCoverage`` strata, which stays exactly as it was. The ``attribute``
# axis is declared only when the frame carries it, and is otherwise declared
# NOT APPLICABLE with an explicit reason rather than silently omitted: the
# per-attribute axis is measured by ``training.attribute_separation``.
COHORT_TAGGED_DIMENSIONS = ('population', 'difficulty_slice', 'evaluation_scope', 'split')
COHORT_ATTRIBUTE_DIMENSION = 'attribute'
#: Frame column -> the SHARED traceability dimension name. The vocabulary lives
#: in ``core.eval_trace`` (one name per axis), so the frame's
#: ``difficulty_slice`` column is censused as the ONE ``difficulty`` axis every
#: other producer reports, and a column that is not in the vocabulary fails loud
#: here rather than forming a private axis.
COHORT_DIMENSION_NAMES = {
    name: canonical_dimension(name) for name in COHORT_TAGGED_DIMENSIONS}
COHORT_PARTITION_UNKNOWN_POLICIES = {
    'population': (f'a pair declares the population it was drawn from; a pair '
                   f'without one is rejected, never tagged '
                   f'{UNKNOWN_DIMENSION_VALUE!r}'),
    'difficulty_slice': (f'an unmeasured pair is retained as '
                         f'{UNKNOWN_DIMENSION_VALUE!r}; easy/medium/hard are '
                         'never invented'),
    'evaluation_scope': (f'a pair declares exactly one evaluation scope; a blank '
                         f'scope fails rather than being tagged '
                         f'{UNKNOWN_DIMENSION_VALUE!r}'),
    'split': (f'a pair belongs to exactly one split; a pair spanned by two is '
              f'tagged mixed, never {UNKNOWN_DIMENSION_VALUE!r}'),
}
COHORT_ATTRIBUTE_UNKNOWN_POLICY = (
    f'a pair may carry several attribute slices, so membership (not partition) is '
    f'the right multiplicity; a pair unobservable for every attribute is tagged '
    f'{UNKNOWN_DIMENSION_VALUE!r} rather than dropped'
)
COHORT_ATTRIBUTE_ABSENT_REASON = (
    'the cohort frame carries no attribute column: pairs are stratified by '
    'population/scope/difficulty/split, and the per-attribute axis is measured '
    'separately by training.attribute_separation'
)


def _cohort_tag(value: object) -> str:
    """One dimension tag exactly as the contract will carry it.

    A missing value (None/NaN/NaT) becomes the empty string so the contract's
    blankness check rejects it -- never ``'nan'``, which would look tagged.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return ''
    if isinstance(value, float) and value != value:
        return ''
    return str(value)


def _census(tags: pd.Series) -> dict[str, int]:
    """The producer's own census of one tag column (independent of the records)."""
    return {value: int(count) for value, count in tags.value_counts().items()}


def cohort_report_coverage(frame: pd.DataFrame) -> ReportCoverageContract:
    """Build and validate the GENERAL coverage contract from the cohort frame.

    Every pair row carries its own dimension tags, so "every pair is accounted"
    is a property of the emitted frame rather than of two producer numbers
    agreeing: a missing column, a blank/NaN tag, a stratum census that disagrees
    with the carried tags, or an empty frame all fail loud here.

    The frame's COLUMN names are mapped to the shared dimension vocabulary
    (``COHORT_DIMENSION_NAMES``) for both the records and the declared
    dimensions, so the frame's ``difficulty_slice`` column and every other
    producer's ``difficulty`` axis are one stratum, never two.
    """
    required = (*COHORT_TAGGED_DIMENSIONS, 'cohort_id')
    missing = sorted(name for name in required if name not in frame.columns)
    if missing:
        raise ValueError(f'ablation cohort frame carries no {missing} column(s)')
    tagged = {COHORT_DIMENSION_NAMES[name]: frame[name].map(_cohort_tag)
              for name in COHORT_TAGGED_DIMENSIONS}
    has_attribute = COHORT_ATTRIBUTE_DIMENSION in frame.columns
    if has_attribute:
        tagged[COHORT_ATTRIBUTE_DIMENSION] = frame[COHORT_ATTRIBUTE_DIMENSION].map(_cohort_tag)
    else:
        tagged[COHORT_ATTRIBUTE_DIMENSION] = pd.Series(
            [UNKNOWN_DIMENSION_VALUE] * len(frame), index=frame.index)
    columns = {name: tags.to_numpy() for name, tags in tagged.items()}
    record_ids = frame['cohort_id'].map(_cohort_tag).to_numpy()
    contract = ReportCoverageContract(
        records=tuple(
            TaggedDimensionRecord(
                record_id=str(record_ids[position]),
                dimensions={name: (values[position],) for name, values in columns.items()},
            )
            for position in range(len(frame))
        ),
        dimensions={
            COHORT_DIMENSION_NAMES[name]: DimensionAccounting(
                policy='partition',
                counts=_census(tagged[COHORT_DIMENSION_NAMES[name]]),
                unknown_policy=COHORT_PARTITION_UNKNOWN_POLICIES[name],
            )
            for name in COHORT_TAGGED_DIMENSIONS
        } | {
            COHORT_ATTRIBUTE_DIMENSION: DimensionAccounting(
                policy='overlap' if has_attribute else 'not_applicable',
                counts=(_census(tagged[COHORT_ATTRIBUTE_DIMENSION]) if has_attribute
                        else {UNKNOWN_DIMENSION_VALUE: len(frame)}),
                unknown_policy=COHORT_ATTRIBUTE_UNKNOWN_POLICY,
                reason=None if has_attribute else COHORT_ATTRIBUTE_ABSENT_REASON,
            )
        },
    )
    return contract


def validate_cohort_coverage(frame: pd.DataFrame, coverage: CohortCoverage) -> ReportCoverageContract:
    """Adopt the general contract for the SAME frame and cross-check the strata.

    Additive by construction: nothing here is written, so the emitted
    ``pairs.csv``/``catalog.csv``/``listings.json``/``coverage.json`` bytes are
    untouched. The census the contract re-derives from the carried records must
    also equal the frozen strata ``coverage.json`` reports, so the general
    contract and the artifact can never disagree silently.
    """
    contract = cohort_report_coverage(frame)
    derived = contract.derived_counts()
    for column, strata in (('evaluation_scope', coverage.by_scope),
                           ('population', coverage.by_population),
                           ('difficulty_slice', coverage.by_difficulty)):
        if derived[COHORT_DIMENSION_NAMES[column]] != {
                value: count for value, count in strata.items() if count}:
            raise ValueError(
                f'ablation cohort {COHORT_DIMENSION_NAMES[column]} disagrees '
                'with the coverage contract')
    return contract


def adopt_cohort_coverage(frame: pd.DataFrame, coverage: CohortCoverage,
                          folder: Path) -> ReportCoverageContract:
    """Validate the general contract against the frame and PERSIST it.

    ONE seam (``prepare_cohort`` calls only this), so a test can neuter the
    adoption and prove the four frozen artifacts are byte-identical without it -
    a validated contract that is thrown away proves nothing to a later reader.
    The census the caller writes into the trace is read back from the SAME
    contract, so no second number is declared anywhere.
    """
    from model_tracks.ablation import write

    contract = validate_cohort_coverage(frame, coverage)
    write(folder / COHORT_CONTRACT_FILE, contract.model_dump(mode='json'))
    return contract


@timed
def prepare_cohort(setup, bundle):
    from graph_tracks.data import load_records
    from model_tracks.ablation import digest, source_name, write
    from model_tracks.shared_graph_data import _canonical_identity, _record, _copy_record
    from core.sku_identity import row_identity
    from training.folds import normalize_gtin

    with _LOG.section('ablation_cohort.load_inputs'):
        layout = _setup_layout()
        folder = setup / 'ablation_cohort'
        folder.mkdir(parents=True, exist_ok=True)
        # The portable layout class owns this composition (package.py ships the
        # members with the SAME resolver; drift between ship/consume is dead).
        from model_tracks.portable_layout import PortableLayout
        clean = PortableLayout.consumer_clean_backup(setup)
        clean_pairs = pd.read_csv(clean / 'pairs.csv', dtype=str, keep_default_na=False)
        catalog = pd.read_csv(setup / layout.catalog, dtype=str, keep_default_na=False)
        rows = catalog.set_index('sku_id', drop=False).to_dict('index')
        records = {r['sku_id']: r for r in load_records(setup / layout.prepared_dir / 'listings.json')}
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
        trace().add(
            "prepare_cohort", "inputs",
            in_count=int(len(clean_pairs)) + int(len(catalog)), out_count=len(rows),
            reason='the cohort is built from the clean holdout pairs, the eligible catalog and the '
                   'prepared bundle payload, and every bundled mint/derived endpoint must carry lineage',
            detail={'clean_pairs': int(len(clean_pairs)), 'catalog_rows': int(len(catalog)),
                    'catalog_rows_indexed': len(rows), 'listing_records': len(records),
                    'canonical_gtins': len(gtins), 'bundle_rows': base,
                    'canonical_end': canonical_end, 'payload_rows': len(bundle['payload']),
                    'mint_lineage_entries': len(audits),
                    'clean_backup': source_name(clean)},
            source=source_name(setup / layout.catalog),
        )
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

    with _LOG.section('ablation_cohort.holdout_splits'):
        fold = bundle['training_plan']['inputs']['folds'][0]
        from training.prepared_bundle import prepared_holdout
        from core.common import training_cfg, SEED
        populations = prepared_holdout(bundle, dict(training_cfg().split), seed=SEED)
        splits = {normalize_gtin(entity): split for split, values in zip(('train','dev','test'), populations, strict=True)
                  for entity in values}
    with _LOG.section('ablation_cohort.clean_pairs'):
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
        trace().add(
            "prepare_cohort", "clean_pairs",
            in_count=int(len(clean_pairs)), out_count=len(cohort),
            reason='every clean holdout pair carries its own cohort_id/population/scope tags',
            detail={'clean_pairs': int(len(clean_pairs)), 'cohort_rows': len(cohort),
                    'heldout': sum(1 for p in cohort if p['evaluation_scope'] == 'heldout'),
                    'training_diagnostic': sum(1 for p in cohort
                                               if p['evaluation_scope'] == 'training_diagnostic')},
            source=source_name(clean / 'pairs.csv'),
        )
    # Include the entire minted supply, including copies not selected by MNRL,
    # and every frozen objective pair (including easy sampled negatives).
    with _LOG.section('ablation_cohort.bundle_pairs'):
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
        trace().add(
            "prepare_cohort", "bundle_pairs",
            # A CUMULATIVE count, not a funnel: the cohort already holds the clean
            # pairs, so `out` is the running total (stated in detail).
            in_count=None,
            out_count=len(cohort),
            reason='the entire minted supply is included, including copies MNRL never selected',
            detail={name: len(bundle[name]) for name in ('pos', 'neg', 'train_neg')}
                   | {'bundle_pairs': sum(len(bundle[name]) for name in ('pos', 'neg', 'train_neg')),
                      'cohort_rows': len(cohort), 'clean_rows': int(len(clean_pairs))},
            source='prepared bundle pos/neg/train_neg triples',
        )
    with _LOG.section('ablation_cohort.objective_pairs'):
        for n, ((a,b,c), population) in enumerate(zip(triples, fold['objective']['dataset']['population'], strict=True)):
            for label, other in ((1,b),(0,c)):
                a,other = int(a),int(other)
                cohort.append({'sku_id1':endpoint(a), 'sku_id2':endpoint(other), 'label':str(label),
                    'split':'train', 'cohort_id':f'objective:{n}:{label}', 'population':population,
                    'evaluation_scope':'training_diagnostic', 'payload_index1':a,'payload_index2':other,
                    'mint_lineage':[audits[i][1] for i in (a,other) if i in audits],
                    'difficulty_slice':'unknown',
                    'consumed_example_ids':[n]})
        trace().add(
            "prepare_cohort", "objective_pairs",
            # A CUMULATIVE count, not a funnel (see bundle_pairs above).
            in_count=None, out_count=len(cohort),
            reason='every frozen objective triple contributes its positive and its sampled negative',
            detail={'objective_triples': len(triples), 'pair_rows_added': 2 * len(triples),
                    'cohort_rows': len(cohort)},
            source='prepared bundle objective triples',
        )
    with _LOG.section('ablation_cohort.difficulty'):
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
        unmeasured = [pair for pair in cohort
                      if pair.get('difficulty_reason') == 'no_frozen_encoder_input']
        trace().add(
            "prepare_cohort", "difficulty",
            in_count=len(cohort), out_count=len(cohort) - len(unmeasured),
            reason='difficulty is measured from the frozen encoder payload; a pair with no frozen '
                   'encoder input is retained and tagged, never invented as easy/hard',
            detail={'cohort_rows': len(cohort), 'measured': len(cohort) - len(unmeasured),
                    'unmeasured': len(unmeasured), 'reused_measurements': len(difficulty_pairs),
                    'distinct_endpoints': len(difficulty_endpoints)},
            source='training.difficulty.measure_pair',
        )
        # The EXACT skipped census is the GROUP rows; the entity rows sample it.
        trace().add_entities(
            "prepare_cohort.difficulty_skipped", unmeasured,
            key_of=lambda pair: pair['cohort_id'],
            reason_of=lambda pair: 'no_frozen_encoder_input',
            detail_of=lambda pair: {'cohort_id': pair['cohort_id'], 'split': pair['split'],
                                    'population': pair['population'],
                                    'evaluation_scope': pair['evaluation_scope'],
                                    'label': pair['label']},
            source='ablation cohort pairs without a frozen encoder payload',
        )
    with _LOG.section('ablation_cohort.frame_and_coverage'):
        frame = pd.DataFrame(cohort)
        for column in ('mint_lineage','consumed_example_ids'):
            frame[column] = frame[column].map(lambda value: json.dumps(value, sort_keys=True))
        coverage = CohortCoverage.model_validate({'cohort_sha256':digest(frame.fillna('').to_dict('records')),
            'pair_rows':len(frame), 'minted_endpoints_total':len(audits),
            'minted_endpoints_covered':len(set(node_ids)&set(audits)),
            'by_scope':frame.evaluation_scope.value_counts().to_dict(),
            'by_population':frame.population.value_counts().to_dict(),
            'by_difficulty':{name:int(frame.difficulty_slice.eq(name).sum())
                             for name in Difficulty.__args__},
            'unknown_difficulty_policy':'retain unknown; never invent easy/hard labels'})
        # The frame is complete and every tag exists HERE, so this is the one
        # point where the GENERAL per-record contract can be adopted: ONE call
        # validates it against the same frame the frozen strata above describe,
        # persists it as the cohort's ``coverage_contract.json`` member (the four
        # frozen artifacts keep their bytes), and returns the contract the trace
        # census below is read back from - no second declared number.
        contract = adopt_cohort_coverage(frame, coverage, folder)
        census = contract.derived_counts()
        trace().add(
            "prepare_cohort", "frame",
            in_count=len(cohort), out_count=len(frame),
            reason='the cohort frame is complete and every dimension tag exists before it is written',
            detail={'pair_rows': len(frame), 'cohort_sha256': coverage.cohort_sha256,
                    'minted_endpoints_total': coverage.minted_endpoints_total,
                    'minted_endpoints_covered': coverage.minted_endpoints_covered,
                    # the CARRIED tags decide these, not the frozen strata
                    'by_scope': census['evaluation_scope'],
                    'by_population': census['population'],
                    'by_difficulty': census['difficulty'],
                    'contract_dimensions': sorted(contract.dimensions)},
            source='ablation cohort frame',
        )
    with _LOG.section('ablation_cohort.persist'):
        frame.fillna('').to_csv(folder/'pairs.csv', index=False)
        pd.DataFrame(list(rows.values())).fillna('').to_csv(folder/'catalog.csv', index=False)
        write(folder/'listings.json', {'schema':'er-graph-listings-v1','listings':list(records.values())})
        write(folder/'coverage.json', coverage.model_dump(mode='json'))
        # ``coverage_contract.json`` was written by the adoption call above; the
        # row below records the member beside the four frozen artifacts.
        trace().add(
            "prepare_cohort", "persisted",
            in_count=None, out_count=5,
            reason='one cohort folder: labeled pairs, ablation catalog, listings, the frozen '
                   'coverage and the validated general coverage contract',
            detail={'pair_rows': len(frame), 'folder': source_name(folder),
                    'pairs': source_name(folder / 'pairs.csv'),
                    'catalog': source_name(folder / 'catalog.csv'),
                    'listings': source_name(folder / 'listings.json'),
                    'coverage': source_name(folder / 'coverage.json'),
                    'coverage_contract': source_name(folder / COHORT_CONTRACT_FILE),
                    'contract_records': contract.records_total},
            source=source_name(folder),
        )
    flush_trace()
    return folder
