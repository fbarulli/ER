"""Validate graph worker inputs and runtime without creating a run or training."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from graph_tracks.config import load_config
from graph_tracks.data import file_hash, load_records, load_text_cache
from graph_tracks.train import load_pairs


def load_inputs(cfg):
    """Shared trainer/preflight validation; no output files or GPU allocation."""
    from core.common import TRAIN_ROOT
    from core.identity_policy import POLICY_PATH
    resolve = lambda raw: (TRAIN_ROOT / raw).resolve()
    manifest = None
    if cfg.input_manifest:
        manifest = json.loads(resolve(cfg.input_manifest).read_text())
    elif not cfg.allow_unmanifested_inputs:
        raise ValueError('prepared input_manifest required; unmanifested inputs are synthetic-smoke only')
    for key, path in [('listings_sha256', resolve(cfg.listings)),
                      ('pairs_sha256', resolve(cfg.pairs)),
                      ('identity_policy_sha256', POLICY_PATH),
                      ('identity_dimensions_sha256', TRAIN_ROOT / 'config/identity_dimensions.yaml')]:
        if manifest is not None and manifest.get(key) != file_hash(path):
            raise ValueError(f'prepared input mismatch: {key}')
    from graph_tracks.data import NUMERIC, RELATIONS
    prepared_schema = (manifest.get('relations'), manifest.get('numeric')) if manifest else (None, None)
    if prepared_schema[0] is not None and prepared_schema[1] is not None:
        if list(prepared_schema[0]) != list(RELATIONS) or list(prepared_schema[1]) != list(NUMERIC):
            raise ValueError(
                'prepared listings schema is stale: the extractor graph schema moved '
                '(relations/numeric derive from core.sku_identity.graph_schema); '
                're-run local graph setup before launch')
    records = load_records(resolve(cfg.listings))
    if manifest is not None and manifest.get('pair_lineage_sha256'):
        if file_hash(resolve(cfg.listings).parent / 'pair_lineage.json') != manifest['pair_lineage_sha256']:
            raise ValueError('prepared pair lineage mismatch')
    from graph_tracks.prepared_inputs import PLAN, load_plan
    if (resolve(cfg.listings).parent / PLAN).is_file():
        plan, arrays = load_plan(resolve(cfg.listings), resolve(cfg.pairs))
        arrays.close()
        if plan['ids'] != [r['sku_id'] for r in records]:
            raise ValueError('prepared graph ID order mismatch')
    elif cfg.device == 'cuda':
        raise ValueError('CUDA training requires locally prepared graph tensors')
    if manifest is not None and manifest.get('report_attributes_sha256'):
        from graph_tracks.report_attributes import FILENAME, load_inputs as load_report_inputs
        if file_hash(resolve(cfg.listings).parent / FILENAME) != manifest['report_attributes_sha256']:
            raise ValueError('prepared report attribute mismatch')
        load_report_inputs(resolve(cfg.listings), records)
    pairs = load_pairs(resolve(cfg.pairs), records)
    vectors, metadata = None, None
    if cfg.text_cache:
        vectors, metadata = load_text_cache(resolve(cfg.text_cache), [r['sku_id'] for r in records])
        if manifest is not None:
            from core.model_input import model_input_composition
            if metadata.get('composition') != model_input_composition().model_dump(mode='json'):
                raise ValueError('text cache composition differs from active model input')
        if cfg.text_checkpoint_sha256 and metadata.get('checkpoint_sha256') != cfg.text_checkpoint_sha256:
            raise ValueError('text cache checkpoint mismatch')
        for key in ('catalog_sha256', 'identity_policy_sha256', 'identity_dimensions_sha256'):
            if manifest is not None and metadata.get(key) != manifest.get(key):
                raise ValueError(f'text cache/prepared input mismatch: {key}')
        if manifest is not None:
            from training.prepare_embeddings import validate_prepared_provenance
            validate_prepared_provenance(resolve(cfg.text_cache), metadata, manifest)
    return manifest, records, pairs, vectors, metadata


def preflight(config: Path, *, check_device: bool = True) -> dict:
    cfg = load_config(config)
    if not cfg.input_manifest or cfg.allow_unmanifested_inputs:
        raise ValueError('production preflight requires manifested inputs')
    _, records, pairs, vectors, _ = load_inputs(cfg)
    text_dimension = None if vectors is None else int(vectors.shape[1])
    if check_device and cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    packages = ['torch', 'numpy', 'pandas', 'pydantic', 'scikit-learn', 'PyYAML']
    if cfg.build_index:
        packages.append('hnswlib')
    if cfg.postprocess:
        packages.append('matplotlib')
    if cfg.wandb.mode != 'disabled':
        packages.append('wandb')
    if cfg.dvc.enabled:
        packages.append('dvc')
    return {'track': cfg.track, 'listings': len(records), 'device': cfg.device,
            'text_dimension': text_dimension, 'report_test': cfg.report_test,
            'pairs': {s: {'positive': int(y.sum()), 'negative': int((y == 0).sum())}
                      for s, (_, y) in pairs.items()},
            'runtime': {package: importlib.metadata.version(package) for package in packages}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.config), indent=2))


if __name__ == '__main__':
    main()
