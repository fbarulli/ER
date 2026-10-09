"""Prepare real catalog inputs for graph tracks without starting training.

Derives the same component split as text training. Labeled entity pairs use
the lexically first listing per entity; same-entity listings form positive
chains. Cross-split negatives are excluded and counted, never relabeled.

One responsibility per unit:

  _PairLedger                the supervised-pair accumulation contract
  _normalize_split_roles     entity->split role map
  _eligible_listing_groups   retained catalog + its listing groups
  _apply_source_labels       labeled-pair supervision, one row at a time
  _chain_pairs               trusted same-entity positive chains
  listing_contract           the pairing orchestrator
  setup                      the artifact orchestrator (stages below)

TRACE ROWS (core.tracing, the ONE consolidated trace)
-----------------------------------------------------
Stage ``graph_setup``. Emitted:
  run   catalog.listings_retained    catalog rows -> retained listings, with the
                                     unassigned-listing drop named in detail
  run   labels.source_rows_applied   labeled rows -> applied source-label pairs,
                                     with missing-endpoint / cross-split counts
  run   chains.batches_<i>           BATCH grain over entity groups (in =
                                     entities walked, out = chain pairs built)
  run   labels.batches_<i>           BATCH grain over labeled rows
  run   chains.batch_census /        batches walked / traced / omitted, so no
        labels.batch_census          chunk is silent
  ent   chains.untrusted_listing     each listing dropped from a chain because
                                     its gtin is not GS1-valid (named sku_id)
  run   pairs.supervision_built      source-label + chain pairs actually built
  group pairs.reason_census          the EXACT provenance census (origin kind)
  ent   pairs.*                      the named supervised pairs with their
                                     split/label/origins
  run   artifacts.published          the setup artifacts written + the census
Batch caps: ``_BATCH_ENTITIES`` / ``_BATCH_LABEL_ROWS`` per traced batch row,
at most ``_MAX_BATCH_ROWS`` batch rows each (both written into the rows'
detail). Entity caps are core.tracing's (ENTITY_SAMPLE_PER_REASON /
ENTITY_ROW_CAP). Nothing here is unbounded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from core.run_log import RunLogger
from core.tracing import (
    ENTITY_ROW_CAP,
    ENTITY_SAMPLE_PER_REASON,
    SCOPE_ENTITY,
    TRACE_MAX_BATCH_ROWS,
    TraceRun,
)
from graph_tracks.data import census, file_size, fit_vocabulary, load_records
from graph_tracks.prepare import prepare
from graph_tracks.text_cache import checkpoint_size
from graph_tracks.train import write_json

_LOG = RunLogger(__name__)

#: The pipeline stage these rows belong to (core.tracing ``stage`` column).
STAGE = "graph_setup"


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup).

    Every layout name this module writes (the catalog/split/pair CSVs, the
    lineage/manifest/census documents, the prepared dir and the rendered lane
    configs) is read from the ONE declared contract rather than spelled here.
    """
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


def write_setup_frames(output: Path, *, catalog: pd.DataFrame,
                       splits: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """The ONE writer of the setup's catalog/split/pair CSVs.

    ``model_tracks.smoke_inputs`` (the CPU smoke's subset setup) and
    ``model_tracks.shared_graph_data`` (the shared-objective projection rebuild)
    each used to re-emit these three frames with their own ``to_csv`` calls, so
    a write-flag change here silently forked the other two surfaces. Both call
    this instead; the producer owns the write.
    """
    layout = _setup_layout()
    catalog.to_csv(Path(output) / layout.catalog, index=False)
    splits.to_csv(Path(output) / layout.splits, index=False)
    pairs.to_csv(Path(output) / layout.pairs, index=False)


def write_track_config(output: Path, track: str, settings: dict) -> Path:
    """The ONE writer of a runnable per-track lane config into a setup tree."""
    path = Path(output) / _setup_layout().track_config(track)
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path


def write_text_config(output: Path, settings: dict) -> Path:
    """The ONE writer of the setup's text lane config (its declared name).

    The text lane's own declared retrieval/index contract, dumped through the
    same yaml writer as the graph lanes' configs, so the smoke setup and the
    production setup cannot render it two different ways.
    """
    path = Path(output) / _setup_layout().text_config
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path

# ── batch-grain budget (documented where it is spent) ──────────────────────
# The two per-row loops are entity groups (chains) and labeled rows (supervision).
# One batch row per 512 entities / 2,048 rows keeps a real cohort to a handful of
# rows; at most 16 batch rows each are traced and the remainder is announced in
# the matching ``batch_census`` row rather than vanishing.
_BATCH_ENTITIES = 512
_BATCH_LABEL_ROWS = 2048
_MAX_BATCH_ROWS = TRACE_MAX_BATCH_ROWS


class _PairLedger:
    """The listing-pair supervision ledger: dedupe, conflicts, lineage."""

    def __init__(self):
        self.pairs: dict[tuple[str, str], tuple[int, str]] = {}
        self.lineage: dict[tuple[str, str], list[dict]] = {}
        self.skipped: Counter = Counter()

    def add(self, a: str, b: str, label: int, split: str, origin: dict) -> None:
        """Accept one supervised pair; self/negatives and conflicts fail-loud."""
        key = tuple(sorted((a, b)))
        if a == b:
            if label == 0:
                raise ValueError('negative label within one normalized entity')
            self.skipped['self_positive'] += 1
            return
        value = (int(label), split)
        if key in self.pairs and self.pairs[key] != value:
            raise ValueError('conflicting listing-pair supervision')
        self.pairs[key] = value
        self.lineage.setdefault(key, []).append(origin)

    def frame(self) -> pd.DataFrame:
        """The supervised pairs as a fixed-column frame (sorted identity)."""
        return pd.DataFrame([
            {'sku_id1': a, 'sku_id2': b, 'label': label, 'split': split}
            for (a, b), (label, split) in sorted(self.pairs.items())
        ], columns=['sku_id1', 'sku_id2', 'label', 'split'])


def _normalize_split_roles(populations: dict) -> dict[str, str]:
    """Entity -> split roles; an entity in two splits is a loud error."""
    from training.folds import normalize_gtin
    roles: dict[str, str] = {}
    for split, values in populations.items():
        for value in values:
            key = normalize_gtin(value)
            if key in roles and roles[key] != split:
                raise ValueError('entity assigned to multiple splits')
            roles[key] = split
    return roles


def _eligible_listing_groups(catalog: pd.DataFrame, roles: dict[str, str]):
    """Retain the assigned listings, group by normalized GTIN, split frames."""
    from training.folds import normalize_gtin
    frame = catalog.fillna('').copy()
    _validate_catalog_skuids(frame)
    keys = frame.gtin.map(normalize_gtin)
    retained = keys.isin(roles) & keys.ne('')
    excluded = int((~retained).sum())
    frame = frame.loc[retained].sort_values('sku_id').reset_index(drop=True)
    groups: dict[str, list[str]] = {}
    keys = frame.gtin.map(normalize_gtin)
    for key, listing in (tracked := _tracked_groups(zip(keys, frame.sku_id))):
        groups.setdefault(key, []).append(listing)
    assignments = pd.DataFrame({'sku_id': frame.sku_id,
                                'split': keys.map(roles)})
    return frame, assignments, groups, excluded


def _tracked_groups(pairs_iter):
    """One tqdm/tracked pass over (gtin, sku_id) pairs to listing groups."""
    from core.progress import tracked
    return tracked(pairs_iter, "listing_groups")


def _validate_catalog_skuids(frame: pd.DataFrame) -> None:
    """The catalog contract: unique, nonempty sku_id."""
    if frame.sku_id.duplicated().any() or (frame.sku_id == '').any():
        raise ValueError('catalog requires unique nonempty sku_id')


def _chain_pairs(groups, roles, ledger: _PairLedger, frame: pd.DataFrame,
                 trace: TraceRun | None = None) -> int:
    """Trusted listings of one entity form its positive chain in listing order.

    Returns the number of chain pairs the ledger accepted (the trace's attempts
    count). ``trace`` adds BATCH-grain rows over the entity groups and an ENTITY
    row for every listing dropped from a chain (a listing whose gtin fails the
    GS1 check digit cannot assert the identity the chain claims), capped and
    named.
    """
    from core.gtin import gtin_validity
    trusted_ids = set(frame.loc[gtin_validity(frame.gtin)].sku_id)
    groups_list = list(groups.items())
    entities_in_batch = pairs_in_batch = 0
    batches = traced = 0
    untrusted_rows = 0
    chain_pairs = 0
    for key, listings in _LOG.progress(groups_list, desc='listing_chains',
                                       unit='entity'):
        eligible = [listing for listing in listings if listing in trusted_ids]
        dropped = len(listings) - len(eligible)
        ledger.skipped['untrusted_identity_chain_listings'] += dropped
        if trace is not None and dropped:
            for listing in listings:
                if listing in trusted_ids:
                    continue
                if untrusted_rows >= ENTITY_ROW_CAP:
                    break
                untrusted_rows += 1
                trace.add(
                    'chains', 'untrusted_listing', scope=SCOPE_ENTITY,
                    key=listing,
                    reason=(
                        'listing dropped from its positive chain: its gtin fails '
                        'the GS1 check digit, so it cannot assert the identity the '
                        'chain claims'
                    ),
                    detail={'gtin': key, 'chain_listings': len(listings)},
                    source=f'{_setup_layout().catalog} gtin column',
                )
        entities_in_batch += 1
        for a, b in zip(eligible, eligible[1:]):
            ledger.add(a, b, 1, roles[key], {'kind': 'trusted_same_entity_chain',
                                             'gtin': key, 'augmentation': 'not_applicable'})
            pairs_in_batch += 1
            chain_pairs += 1
        if entities_in_batch >= _BATCH_ENTITIES or entities_in_batch == len(groups_list):
            batches += 1
            if trace is not None and traced < _MAX_BATCH_ROWS:
                traced += 1
                # No in/out pair: one entity's k trusted listings form k-1
                # chain pairs, so "entities -> pairs" is a fan-out census.
                trace.add(
                    'chains', f'batch_{batches - 1:04d}',
                    reason=(
                        'entity groups walked; the same-entity chain pairs they '
                        'formed are a fan-out over their trusted listings (a '
                        'census, not a funnel), so no in/out pair is stated'
                    ),
                    detail={
                        'entity_groups': entities_in_batch,
                        'pairs': pairs_in_batch,
                        'batch_entities': _BATCH_ENTITIES,
                        'max_batch_rows': _MAX_BATCH_ROWS,
                    },
                    source=f'{_setup_layout().catalog} grouped by normalized gtin',
                )
            entities_in_batch = 0
            pairs_in_batch = 0
    if trace is not None:
        trace.add(
            'chains', 'batch_census', in_count=batches, out_count=traced,
            reason=(
                'entity-group batches traced individually; the remainder is summed '
                'here so no chunk is silent'
            ),
            detail={
                'entity_groups': len(groups_list),
                'batches': batches,
                'batches_traced': traced,
                'batches_omitted': batches - traced,
                'batch_entities': _BATCH_ENTITIES,
                'max_batch_rows': _MAX_BATCH_ROWS,
                'untrusted_chain_listings': int(
                    ledger.skipped['untrusted_identity_chain_listings']
                ),
            },
            source=f'{_setup_layout().catalog} grouped by normalized gtin',
        )
    return chain_pairs


def _apply_source_labels(labels: pd.DataFrame, groups, roles,
                         ledger: _PairLedger,
                         trace: TraceRun | None = None) -> list[str]:
    """Every labeled source row contributes its supervised listing pair.

    ``trace`` adds BATCH-grain rows over the labeled rows; the skip reasons are
    counted by the ledger and recorded by the caller's stage row.
    """
    from training.folds import normalize_gtin
    source_axes = [c for c in labels.columns if c not in {'gtin1', 'gtin2', 'true_label'}]
    rows_in_batch = pairs_in_batch = 0
    batches = traced = 0
    n_rows = int(len(labels))
    for source_row, row in enumerate(
            _LOG.progress(labels.itertuples(index=False), desc='source_labels',
                          unit='row'), 1):
        a, b = normalize_gtin(row.gtin1), normalize_gtin(row.gtin2)
        label = int(row.true_label)
        if label not in (0, 1):
            raise ValueError('invalid entity label')
        rows_in_batch += 1
        if a not in groups or b not in groups:
            ledger.skipped['missing_listing_endpoint'] += 1
            continue
        if roles[a] != roles[b]:
            if label:
                raise ValueError('positive label crosses shared split')
            ledger.skipped['cross_split_negative'] += 1
            continue
        # Read metadata by the original column names, not namedtuple's renamed
        # fields, so arbitrary source trace-axis names survive unchanged.
        metadata = labels.iloc[source_row - 1][source_axes].to_dict()
        ledger.add(groups[a][0], groups[b][0], label, roles[a],
                   {'kind': 'source_entity_label', 'source_row': source_row,
                    'gtin1': str(row.gtin1), 'gtin2': str(row.gtin2), 'metadata': metadata})
        pairs_in_batch += 1
        if rows_in_batch >= _BATCH_LABEL_ROWS or rows_in_batch == n_rows:
            batches += 1
            if trace is not None and traced < _MAX_BATCH_ROWS:
                traced += 1
                trace.add(
                    'labels', f'batch_{batches - 1:04d}',
                    in_count=rows_in_batch, out_count=pairs_in_batch,
                    reason=(
                        'labeled source rows walked -> supervised listing pairs; a '
                        'row with a missing endpoint or a cross-split negative '
                        'contributes none'
                    ),
                    detail={
                        'batch_label_rows': _BATCH_LABEL_ROWS,
                        'max_batch_rows': _MAX_BATCH_ROWS,
                        'pairs_in_batch': pairs_in_batch,
                    },
                    source='data/labeled_pairs.csv',
                )
            rows_in_batch = 0
            pairs_in_batch = 0
    if trace is not None:
        trace.add(
            'labels', 'batch_census', in_count=batches, out_count=traced,
            reason=(
                'labeled-row batches traced individually; the remainder is summed '
                'here so no chunk is silent'
            ),
            detail={
                'label_rows': n_rows,
                'batches': batches,
                'batches_traced': traced,
                'batches_omitted': batches - traced,
                'batch_label_rows': _BATCH_LABEL_ROWS,
                'max_batch_rows': _MAX_BATCH_ROWS,
            },
            source='data/labeled_pairs.csv',
        )
    return source_axes


def listing_contract(catalog, labels, populations, trace: TraceRun | None = None):
    """Pair the catalog under the shared split policy; return frames + accounts.

    ``trace`` (optional) receives this unit's BATCH-grain rows; the stage rows
    that summarize the accounting are emitted by :func:`setup`, which holds the
    frames those counts describe.
    """
    from training.folds import normalize_gtin
    roles = _normalize_split_roles(populations)
    frame, assignments, groups, excluded = _eligible_listing_groups(catalog, roles)
    ledger = _PairLedger()
    chain_pairs = _chain_pairs(groups, roles, ledger, frame, trace)
    source_axes = _apply_source_labels(labels, groups, roles, ledger, trace)
    pairs = ledger.frame()
    # Trace-only bookkeeping rides in DataFrame attrs, NOT in `accounting`: the
    # accounting dict is serialized verbatim into the setup manifest, so adding
    # a key here would change an emitted artifact's bytes.
    pairs.attrs['chain_pairs_accepted'] = int(chain_pairs)
    accounting = {
        'excluded_unassigned_listings': excluded,
        'skipped_labels': dict(ledger.skipped),
        'source_trace_columns': source_axes,
        'missing_axes': sorted({'difficulty', 'masking', 'gendata', 'gate_evidence'} - set(source_axes)),
        'augmentation': ('not_applicable: graph track uses fixed labels '
                         'without generated/masked pairs'),
        'pair_lineage': _pair_lineage_records(ledger),
    }
    return frame, assignments, pairs, accounting


def _pair_lineage_records(ledger: _PairLedger) -> list[dict]:
    """One lineage record per supervised pair (its accumulating origins)."""
    return [{'sku_id1': a, 'sku_id2': b, 'label': ledger.pairs[(a, b)][0],
             'split': ledger.pairs[(a, b)][1], 'origins': origins}
            for (a, b), origins in sorted(ledger.lineage.items())]


def setup(output: Path, checkpoint: Path, *, training_tensors: bool = True) -> Path:
    """Assemble every graph-track setup artifact (orchestrator only)."""
    from core.timing import Timing
    timing = Timing('graph_tracks.setup')
    from core.common import SEED, TRAIN_ROOT, F, git_revision, load_dataset_deduped, training_cfg
    from core.identity_policy import POLICY_PATH
    from training.base_data import load_base_data
    from training.folds import derive_holdout
    # ONE consolidated-trace writer for the stage. The rows are committed at the
    # end so the catalog, the supervision and the published artifacts read as one
    # flow (core.tracing: a stage replaces its own rows in place, so one writer).
    trace = TraceRun(STAGE)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = checkpoint.resolve()
    baseline_size = checkpoint_size(checkpoint)
    catalog = load_dataset_deduped().fillna('')
    data = load_base_data(catalog, payload_variant='full')
    timing.mark('checkpoint_catalog_and_base_data')
    train, dev, test = derive_holdout(data['pos'], data['row_bc'],
                                      dict(training_cfg().split), seed=SEED)
    labels = pd.read_csv(F['labeled_pairs'], dtype=str, keep_default_na=False)
    frame, assignments, pairs, accounting = listing_contract(
        catalog, labels, {'train': train, 'dev': dev, 'test': test}, trace)
    _record_supervision(trace, catalog, frame, labels, pairs, accounting)
    timing.mark('splits_and_listing_contract')
    # Validate before publishing any setup artifacts.
    from graph_tracks.train import load_pairs
    load_pairs_from = [{'sku_id': r.sku_id, 'split': r.split}
                       for r in assignments.itertuples(index=False)]
    output.mkdir(parents=True)
    layout = _setup_layout()
    write_setup_frames(output, catalog=frame, splits=assignments, pairs=pairs)
    write_json(output / layout.pair_lineage,
               _pair_lineage_document(accounting, output, F))
    load_pairs(output / layout.pairs, load_pairs_from)
    timing.mark('validate_and_write_pairs')
    listings = prepare(output / layout.catalog, output / layout.splits,
                       output / layout.pairs, output / layout.prepared_dir,
                       training_tensors=training_tensors)
    timing.mark('graph_prepare')
    records = load_records(listings, trace=trace)
    write_json(output / layout.census, census(records, fit_vocabulary(records, trace=trace)))
    timing.mark('graph_features_and_census')
    _write_setup_manifest(output, accounting, baseline_size, checkpoint, pairs,
                          F, git_revision())
    templates = _load_setup_templates(Path(TRAIN_ROOT))
    _write_track_configs(output, templates, listings, baseline_size)
    _write_text_config(output, templates)
    _record_artifacts(trace, output, frame, assignments, pairs, listings, records)
    timing.mark('sizes_manifest_and_track_configs')
    trace.write()
    return output


# ── the stage's trace rows (real counts, named reasons) ─────────────────────
def _record_supervision(
    trace: TraceRun,
    catalog: pd.DataFrame,
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    pairs: pd.DataFrame,
    accounting: dict,
) -> None:
    """Catalog retention, label application and the pair-provenance census.

    Every count comes from the accounting ``listing_contract`` returned (the
    same numbers the setup manifest ships) or from the frames themselves, so the
    trace and the manifest cannot disagree. Pair provenance is an ENTITY census:
    one row per supervised pair naming its split, its label and its origin kinds.
    """
    skipped = accounting.get('skipped_labels', {})
    trace.add(
        'catalog', 'listings_retained',
        in_count=int(len(catalog)), out_count=int(len(frame)),
        reason=(
            'a catalog listing is retained only when its normalized gtin belongs '
            'to an assigned split population; an unassigned listing is dropped '
            'here, never silently relabelled'
        ),
        detail={
            'catalog_rows': int(len(catalog)),
            'retained_listings': int(len(frame)),
            'excluded_unassigned_listings': int(
                accounting.get('excluded_unassigned_listings', 0)
            ),
        },
        source="dataset_deduped (core.common.load_dataset_deduped)",
    )
    missing_endpoint = int(skipped.get('missing_listing_endpoint', 0))
    cross_split = int(skipped.get('cross_split_negative', 0))
    applied = int(len(labels)) - missing_endpoint - cross_split
    trace.add(
        'labels', 'source_rows_applied',
        in_count=int(len(labels)), out_count=applied,
        reason=(
            'a labeled row contributes its supervised pair only when BOTH '
            'normalized gtins have a retained listing and both sit in one split; '
            'a cross-split positive is a loud error, a cross-split negative is '
            'dropped'
        ),
        detail={
            'label_rows': int(len(labels)),
            'applied_pairs': applied,
            'missing_listing_endpoint': missing_endpoint,
            'cross_split_negative': cross_split,
            'other_skips': {
                key: int(value) for key, value in skipped.items()
                if key not in {'missing_listing_endpoint', 'cross_split_negative'}
            },
        },
        source="data/labeled_pairs.csv",
    )
    lineage = accounting.get('pair_lineage', [])
    records = [
        {
            'pair': f"{record['sku_id1']}|{record['sku_id2']}",
            'kind': '+'.join(sorted({str(origin.get('kind', '')) for origin in record['origins']})),
            'label': record['label'],
            'split': record['split'],
            'origins': len(record['origins']),
        }
        for record in lineage
    ]
    trace.add(
        'pairs', 'supervision_built',
        in_count=applied
        + int(skipped.get('untrusted_identity_chain_listings', 0))
        + int(pairs.attrs.get('chain_pairs_accepted', 0)),
        out_count=int(len(pairs)),
        reason=(
            'source-label rows and same-entity chain pairs are DEDUPLICATED into '
            'one supervised pair per normalized listing pair; the drop is '
            'therefore collisions (a pair supervised twice counts once) and self '
            'positives, never a fabricated loss'
        ),
        detail={
            'applied_source_label_pairs': applied,
            'chain_pairs_accepted': int(pairs.attrs.get('chain_pairs_accepted', 0)),
            'untrusted_identity_chain_listings': int(
                skipped.get('untrusted_identity_chain_listings', 0)
            ),
            'self_positive_skipped': int(skipped.get('self_positive', 0)),
            'supervised_pairs': int(len(pairs)),
            'pair_counts': {
                str(split): {str(label): int(count) for label, count in group.label.value_counts().items()}
                for split, group in pairs.groupby('split')
            } if len(pairs) else {},
        },
        source=f'{_setup_layout().catalog} + data/labeled_pairs.csv',
    )
    trace.add_entities(
        'pairs', records,
        key_of=lambda record: record['pair'],
        reason_of=lambda record: record['kind'],
        detail_of=lambda record: {
            'label': record['label'],
            'split': record['split'],
            'origins': record['origins'],
        },
        source='graph_tracks.setup._pair_lineage_records',
        per_reason=ENTITY_SAMPLE_PER_REASON,
        total_cap=ENTITY_ROW_CAP,
    )


def _record_artifacts(
    trace: TraceRun,
    output: Path,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    pairs: pd.DataFrame,
    listings: Path,
    records: list[dict],
) -> None:
    """The published setup artifacts, with the prepared listing population."""
    layout = _setup_layout()
    trace.add(
        'artifacts', 'published',
        in_count=int(len(frame)), out_count=int(len(records)),
        reason=(
            'the retained catalog is published, the supervised pairs are '
            'validated, the graph listings are materialized and the census is '
            'written; the prepared listing population is what the count reports'
        ),
        detail={
            'output': str(output),
            'retained_listings': int(len(frame)),
            'listing_splits': int(len(assignments)),
            'supervised_pairs': int(len(pairs)),
            'prepared_listings': int(len(records)),
            'listings_json': str(Path(listings).parent / layout.listings),
            'files': [
                name for name in (
                    layout.catalog, layout.splits, layout.pairs,
                    layout.pair_lineage, layout.manifest, layout.census,
                    layout.track_config('gnn_only'),
                    layout.track_config('cascade'), layout.text_config,
                )
                if (output / name).exists()
            ],
        },
        source=str(output),
    )


def _load_setup_templates(train_root: Path) -> dict:
    """The per-track graph templates + the text track's declared contract."""
    from core.common import TRAIN_ROOT
    from graph_tracks.config import load_config as load_graph_config, load_text_config
    config_dir = Path(TRAIN_ROOT) / str(_config_layout())
    templates = {track: load_graph_config(config_dir / f'graph_tracks_{template}.yaml',
                                          expected_track=track).model_dump()
                 for track, template in [('gnn_only', 'gnn'), ('cascade', 'cascade')]}
    templates['text'] = load_text_config(config_dir / 'text_track.yaml').model_dump()
    return templates


def _config_layout() -> str:
    """The declared config neighborhood (paths.yaml layouts block)."""
    from core.common import LAYOUTS
    return str(LAYOUTS['config_dir'].template)


def _pair_lineage_document(accounting: dict, output: Path, F) -> dict:
    """The pair-lineage document: lineage + the input sizes an auditor re-verifies."""
    return {'schema': 'er-graph-pair-lineage-v1',
            'pairs': accounting.pop('pair_lineage'),
            'source_trace_columns': accounting['source_trace_columns'],
            'missing_axes': accounting['missing_axes'],
            'augmentation': accounting['augmentation'],
            'listing_pairs_size': file_size(output / _setup_layout().pairs),
            'source_labels_size': file_size(F['labeled_pairs'])}


def _write_setup_manifest(output: Path, accounting: dict, baseline_size: int,
                          checkpoint: Path, pairs: pd.DataFrame, F, revision: str) -> None:
    """The setup manifest: the run's identity + pairing policy + pair counts."""
    from core.common import SEED
    from core.identity_policy import POLICY_PATH
    write_json(output / _setup_layout().manifest, {
        'schema': 'er-track-setup-v1', 'git_revision': revision, 'seed': SEED,
        'source_catalog_size': file_size(F['dataset_deduped']),
        'labeled_pairs_size': file_size(F['labeled_pairs']),
        'identity_policy_size': file_size(POLICY_PATH),
        'text_checkpoint': str(checkpoint), 'text_checkpoint_size': baseline_size,
        'text_checkpoint_status': 'local baseline; fine-tuning history not inferred',
        'split_protocol': 'training.folds.derive_holdout',
        'pair_protocol': 'first listing per labeled entity plus same-entity positive chains',
        'negative_policy': 'same-split labeled negatives only',
        'pair_counts': {split: {str(label): int(count) for label, count in group.label.value_counts().items()}
                        for split, group in pairs.groupby('split')},
        **accounting,
    })


def _suite_report_test() -> bool | None:
    """The suite's own ``report_test`` switch, from the config SSOT.

    ``model_tracks.preflight`` refuses a prepared setup whose lane configs
    disagree with the suite switch, so the rendered configs must not invent one.
    They used to be hardcoded ``report_test=False`` here, which silently dropped
    the suite's request to score the held-out test split. Resolved through the
    declared layout (paths.yaml ``model_tracks_config``).

    Returns ``None`` when the suite config is not declared/readable, so the
    caller falls back to the LANE TEMPLATE's own declared switch instead of
    inventing a value. A malformed suite config still raises: this is a
    fallback for a missing file, not for a broken one.
    """
    from core.common import artifact

    try:
        path = Path(artifact('model_tracks_config'))
    except KeyError:
        return None
    if not path.is_file():
        return None
    from model_tracks.config import load_config as load_suite_config
    return bool(load_suite_config(path).report_test)


def _write_track_configs(output: Path, templates: dict, listings: Path,
                         baseline_size: int) -> None:
    """Render each graph track's runnable config against this setup tree."""
    from graph_tracks.artifacts import name
    from graph_tracks.config import GraphConfig
    # The held-out test switch comes FROM THE CONFIG SSOT (the suite's own
    # report_test), never from a literal here: preflight compares the prepared
    # lane configs against the suite and refuses a setup whose lanes disagree.
    report_test = _suite_report_test()
    for track in ('gnn_only', 'cascade'):
        cfg = templates[track].copy()
        cfg.update(listings=str(listings),
                   pairs=str(listings.parent / 'pairs.csv'),
                   input_manifest=str(listings.parent / _setup_layout().input_manifest))
        if report_test is not None:
            cfg['report_test'] = report_test
        if track == 'cascade':
            # The cascade consumes the trained text ANN and the trained
            # gnn_only scorer. Both are artifacts of the OTHER tracks, named
            # through the ONE artifact-naming SSOT (graph_tracks.artifacts.name)
            # under the declared results root — the exact names the worker's
            # resolver consumes. They used to point into THIS setup tree
            # (output/text_index, output/gnn_checkpoint.json), which is not
            # where a trained artifact ever lands, so the cascade lane raised
            # FileNotFoundError before ranking anything. No text_cache fusion is
            # ever declared.
            cfg.pop('text_cache', None)
            results_root = Path(str(cfg['output_dir']))
            cfg['text_index'] = str(results_root / name('text', 'index'))
            cfg['gnn_checkpoint'] = str(
                results_root / name('gnn_only', 'best_checkpoint.json')
            )
        cfg = GraphConfig.model_validate(cfg).model_dump()
        write_track_config(output, track, cfg)


def _write_text_config(output: Path, templates: dict) -> None:
    """The text lane's own declared retrieval/index contract.

    It used to borrow gnn_only.yaml's HNSW settings and recall ladder, so a
    graph-track retune silently changed the text track's reported recall@k. Its
    ``report_test`` comes from the same config SSOT as the graph lanes (the
    suite's switch), never from a literal here.
    """
    text_cfg = templates['text'].copy()
    report_test = _suite_report_test()
    if report_test is not None:
        text_cfg['report_test'] = report_test
    write_text_config(output, text_cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=default_setup_dir())
    parser.add_argument('--text-checkpoint', type=Path,
                        default=_default_text_checkpoint())
    parser.add_argument('--defer-training-tensors', action='store_true',
                        help='suite packaging builds tensors once after final supervision projection')
    args = parser.parse_args()
    print(setup(args.output, args.text_checkpoint,
                training_tensors=not args.defer_training_tensors))


def default_setup_dir() -> Path:
    """The declared setup output root: the SUITE config's own ``setup_dir``.

    The prepared-setup root is config-owned (``config/model_tracks.yaml``
    ``setup_dir``): the orchestrator (``training.prepare_all``), ``model_tracks``
    preflight/worker/finalize and the data gate all read it from there. The
    standalone setup CLI used to default to a hand-spelled ``data_dir/track_setup``
    copy, so retargeting the suite's ``setup_dir`` silently left this producer
    writing a tree no consumer reads. Resolve the ONE home instead, falling back
    to the historical literal only when the suite config is absent/unreadable (a
    fallback for a missing file, not for a broken one).
    """
    from core.common import _CFG

    fallback = Path(_CFG['paths']['data_dir']) / 'track_setup'
    from core.common import artifact

    try:
        path = Path(artifact('model_tracks_config'))
    except KeyError:
        return fallback
    if not path.is_file():
        return fallback
    from core.common import TRAIN_ROOT
    from model_tracks.config import load_config as load_suite_config
    return Path(TRAIN_ROOT) / load_suite_config(path).setup_dir


def _default_text_checkpoint() -> Path:
    """The declared baseline text checkpoint (paths.yaml models registry)."""
    from core.common import TRAIN_ROOT
    from core.common import _CFG
    models_dir = _CFG['paths']['models_dir']
    key = _CFG['embedding_model_keys'][0]
    return Path(TRAIN_ROOT) / models_dir / _CFG['models'][key]


if __name__ == '__main__':
    main()