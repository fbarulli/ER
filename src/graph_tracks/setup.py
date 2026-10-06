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
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from core.run_log import RunLogger
from graph_tracks.data import census, file_hash, fit_vocabulary, load_records
from graph_tracks.prepare import prepare
from graph_tracks.text_cache import checkpoint_hash
from graph_tracks.train import write_json

_LOG = RunLogger(__name__)


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


def _chain_pairs(groups, roles, ledger: _PairLedger, frame: pd.DataFrame) -> None:
    """Trusted listings of one entity form its positive chain in listing order."""
    from core.gtin import gtin_validity
    trusted_ids = set(frame.loc[gtin_validity(frame.gtin)].sku_id)
    for key, listings in _LOG.progress(groups.items(), desc='listing_chains',
                                       unit='entity'):
        eligible = [listing for listing in listings if listing in trusted_ids]
        ledger.skipped['untrusted_identity_chain_listings'] += len(listings) - len(eligible)
        for a, b in zip(eligible, eligible[1:]):
            ledger.add(a, b, 1, roles[key], {'kind': 'trusted_same_entity_chain',
                                             'gtin': key, 'augmentation': 'not_applicable'})


def _apply_source_labels(labels: pd.DataFrame, groups, roles,
                         ledger: _PairLedger) -> list[str]:
    """Every labeled source row contributes its supervised listing pair."""
    from training.folds import normalize_gtin
    source_axes = [c for c in labels.columns if c not in {'gtin1', 'gtin2', 'true_label'}]
    for source_row, row in enumerate(
            _LOG.progress(labels.itertuples(index=False), desc='source_labels',
                          unit='row'), 1):
        a, b = normalize_gtin(row.gtin1), normalize_gtin(row.gtin2)
        label = int(row.true_label)
        if label not in (0, 1):
            raise ValueError('invalid entity label')
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
    return source_axes


def listing_contract(catalog, labels, populations):
    """Pair the catalog under the shared split policy; return frames + accounts."""
    from training.folds import normalize_gtin
    roles = _normalize_split_roles(populations)
    frame, assignments, groups, excluded = _eligible_listing_groups(catalog, roles)
    ledger = _PairLedger()
    _chain_pairs(groups, roles, ledger, frame)
    source_axes = _apply_source_labels(labels, groups, roles, ledger)
    pairs = ledger.frame()
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
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = checkpoint.resolve()
    baseline_hash = checkpoint_hash(checkpoint)
    catalog = load_dataset_deduped().fillna('')
    data = load_base_data(catalog, payload_variant='full')
    timing.mark('checkpoint_catalog_and_base_data')
    train, dev, test = derive_holdout(data['pos'], data['row_bc'],
                                      dict(training_cfg().split), seed=SEED)
    labels = pd.read_csv(F['labeled_pairs'], dtype=str, keep_default_na=False)
    frame, assignments, pairs, accounting = listing_contract(
        catalog, labels, {'train': train, 'dev': dev, 'test': test})
    timing.mark('splits_and_listing_contract')
    # Validate before publishing any setup artifacts.
    from graph_tracks.train import load_pairs
    load_pairs_from = [{'sku_id': r.sku_id, 'split': r.split}
                       for r in assignments.itertuples(index=False)]
    output.mkdir(parents=True)
    frame.to_csv(output / 'eligible_catalog.csv', index=False)
    assignments.to_csv(output / 'listing_splits.csv', index=False)
    pairs.to_csv(output / 'listing_pairs.csv', index=False)
    write_json(output / 'pair_lineage.json',
               _pair_lineage_document(accounting, output, F))
    load_pairs(output / 'listing_pairs.csv', load_pairs_from)
    timing.mark('validate_and_write_pairs')
    listings = prepare(output / 'eligible_catalog.csv', output / 'listing_splits.csv',
                       output / 'listing_pairs.csv', output / 'prepared',
                       training_tensors=training_tensors)
    timing.mark('graph_prepare')
    records = load_records(listings)
    write_json(output / 'graph_census.json', census(records, fit_vocabulary(records)))
    timing.mark('graph_features_and_census')
    _write_setup_manifest(output, accounting, baseline_hash, checkpoint, pairs,
                          F, git_revision())
    templates = _load_setup_templates(Path(TRAIN_ROOT))
    _write_track_configs(output, templates, listings, baseline_hash)
    _write_text_config(output, templates)
    timing.mark('hashes_manifest_and_track_configs')
    return output


def _config_template_path(track_template: str) -> Path:
    """A config-neighborhood template path (paths.yaml layouts)."""
    from core.common import LAYOUTS, TRAIN_ROOT
    return Path(TRAIN_ROOT) / str(LAYOUTS['config_dir'].template)


def _load_setup_templates(train_root: Path) -> dict:
    """The per-track graph templates + the text track's declared contract."""
    from core.common import TRAIN_ROOT
    from graph_tracks.config import load_config as load_graph_config, load_text_config
    config_dir = Path(TRAIN_ROOT) / str(_config_layout())
    templates = {track: load_graph_config(config_dir / f'graph_tracks_{template}.yaml',
                                          expected_track=track).model_dump()
                 for track, template in [('gnn_only', 'gnn'), ('hybrid', 'hybrid')]}
    templates['text'] = load_text_config(config_dir / 'text_track.yaml').model_dump()
    return templates


def _config_layout() -> str:
    """The declared config neighborhood (paths.yaml layouts block)."""
    from core.common import LAYOUTS
    return str(LAYOUTS['config_dir'].template)


def _pair_lineage_document(accounting: dict, output: Path, F) -> dict:
    """pair_lineage.json: lineage + the input hashes an auditor re-verifies."""
    return {'schema': 'er-graph-pair-lineage-v1',
            'pairs': accounting.pop('pair_lineage'),
            'source_trace_columns': accounting['source_trace_columns'],
            'missing_axes': accounting['missing_axes'],
            'augmentation': accounting['augmentation'],
            'listing_pairs_sha256': file_hash(output / 'listing_pairs.csv'),
            'source_labels_sha256': file_hash(F['labeled_pairs'])}


def _write_setup_manifest(output: Path, accounting: dict, baseline_hash: str,
                          checkpoint: Path, pairs: pd.DataFrame, F, revision: str) -> None:
    """setup_manifest.json: the run's identity + pairing policy + pair counts."""
    from core.common import SEED
    from core.identity_policy import POLICY_PATH
    write_json(output / 'setup_manifest.json', {
        'schema': 'er-track-setup-v1', 'git_revision': revision, 'seed': SEED,
        'source_catalog_sha256': file_hash(F['dataset_deduped']),
        'labeled_pairs_sha256': file_hash(F['labeled_pairs']),
        'identity_policy_sha256': file_hash(POLICY_PATH),
        'text_checkpoint': str(checkpoint), 'text_checkpoint_sha256': baseline_hash,
        'text_checkpoint_status': 'local baseline; fine-tuning history not inferred',
        'split_protocol': 'training.folds.derive_holdout',
        'pair_protocol': 'first listing per labeled entity plus same-entity positive chains',
        'negative_policy': 'same-split labeled negatives only',
        'pair_counts': {split: {str(label): int(count) for label, count in group.label.value_counts().items()}
                        for split, group in pairs.groupby('split')},
        **accounting,
    })


def _write_track_configs(output: Path, templates: dict, listings: Path,
                         baseline_hash: str) -> None:
    """Render each graph track's runnable config against this setup tree."""
    from graph_tracks.config import GraphConfig
    for track in ('gnn_only', 'hybrid'):
        cfg = templates[track].copy()
        cfg.update(listings=str(listings),
                   pairs=str(listings.parent / 'pairs.csv'),
                   input_manifest=str(listings.parent / 'input_manifest.json'),
                   report_test=False)
        if track == 'hybrid':
            cfg['text_cache'] = str(output / 'shared_minilm__embeddings.npz')
            cfg['text_checkpoint_sha256'] = baseline_hash
        cfg = GraphConfig.model_validate(cfg).model_dump()
        (output / f'{track}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))


def _write_text_config(output: Path, templates: dict) -> None:
    """The text lane's own declared retrieval/index contract.

    It used to borrow gnn_only.yaml's HNSW settings and recall ladder, so a
    graph-track retune silently changed the text track's reported recall@k.
    """
    text_cfg = templates['text'].copy()
    text_cfg.update(report_test=False)
    (output / 'text.yaml').write_text(yaml.safe_dump(text_cfg, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=_default_setup_output())
    parser.add_argument('--text-checkpoint', type=Path,
                        default=_default_text_checkpoint())
    parser.add_argument('--defer-training-tensors', action='store_true',
                        help='suite packaging builds tensors once after final supervision projection')
    args = parser.parse_args()
    print(setup(args.output, args.text_checkpoint,
                training_tensors=not args.defer_training_tensors))


def _default_setup_output() -> Path:
    """The declared setup output root (paths.yaml data_dir)."""
    from core.common import _CFG
    return Path(_CFG['paths']['data_dir']) / 'track_setup'


def _default_text_checkpoint() -> Path:
    """The declared baseline text checkpoint (paths.yaml models registry)."""
    from core.common import TRAIN_ROOT
    from core.common import _CFG
    models_dir = _CFG['paths']['models_dir']
    key = _CFG['embedding_model_keys'][0]
    return Path(TRAIN_ROOT) / models_dir / _CFG['models'][key]


if __name__ == '__main__':
    main()