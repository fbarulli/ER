"""Validate graph worker inputs and runtime without creating a run or training."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from graph_tracks.config import load_config
from graph_tracks.data import load_records
from graph_tracks.train import load_pairs


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


def load_inputs(cfg):
    """Shared trainer/preflight validation; no output files or GPU allocation.

    Inputs are trusted by construction: no manifest size, fingerprint or
    provenance value is compared (owner directive: data is never checked). The
    manifest is still required unless the lane declares unmanifested
    (synthetic-smoke) inputs, and the declared extractor graph schema must match
    the code's, because that schema is a config contract.
    """
    from core.common import TRAIN_ROOT
    layout = _setup_layout()
    resolve = lambda raw: (TRAIN_ROOT / raw).resolve()
    manifest = None
    if cfg.input_manifest:
        manifest = json.loads(resolve(cfg.input_manifest).read_text())
    elif not cfg.allow_unmanifested_inputs:
        raise ValueError('prepared input_manifest required; unmanifested inputs are synthetic-smoke only')
    if manifest is not None:
        from graph_tracks.data import NUMERIC, RELATIONS
        prepared_schema = (manifest.get('relations'), manifest.get('numeric'))
        if prepared_schema[0] is not None and prepared_schema[1] is not None:
            if list(prepared_schema[0]) != list(RELATIONS) or list(prepared_schema[1]) != list(NUMERIC):
                raise ValueError(
                    'prepared listings schema is incompatible: the extractor graph schema moved '
                    '(relations/numeric derive from core.sku_identity.graph_schema); '
                    're-run local graph setup before launch')
    records = load_records(resolve(cfg.listings))
    if cfg.device == 'cuda':
        from graph_tracks.prepared_inputs import PLAN
        if not (resolve(cfg.listings).parent / PLAN).is_file():
            raise ValueError('CUDA training requires locally prepared graph tensors')
    if manifest is not None:
        from graph_tracks.report_attributes import load_inputs as load_report_inputs
        load_report_inputs(resolve(cfg.listings), records)
    pairs = load_pairs(resolve(cfg.pairs), records)
    vectors, metadata = None, None
    if cfg.text_cache:
        # Unreachable: GraphConfig rejects text_cache for both tracks
        # (config.py:131-132), so no graph config reaching load_inputs can
        # declare a fused text cache. Keep the retired hybrid's cache-load and
        # provenance contract here as a fail-closed guard rather than letting a
        # future text-cache track silently skip the provenance checks.
        raise AssertionError('text_cache is retired: no graph track may declare it')
    return manifest, records, pairs, vectors, metadata


def preflight(config: Path, *, check_device: bool = True, require_dvc: bool = True) -> dict:
    cfg = load_config(config)
    if not cfg.input_manifest or cfg.allow_unmanifested_inputs:
        raise ValueError('production preflight requires manifested inputs')
    _, records, pairs, vectors, _ = load_inputs(cfg)
    text_dimension = None if vectors is None else int(vectors.shape[1])
    if check_device and cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return {'track': cfg.track, 'listings': len(records), 'device': cfg.device,
            'text_dimension': text_dimension, 'report_test': cfg.report_test,
            'pairs': {s: {'positive': int(y.sum()), 'negative': int((y == 0).sum())}
                      for s, (_, y) in pairs.items()},
            'runtime': runtime_versions(cfg, require_dvc=require_dvc)}


def runtime_versions(cfg, *, require_dvc: bool = True) -> dict:
    """Assert dependencies for this lane's actual export/report switches."""
    packages = ['torch', 'numpy', 'pandas', 'pydantic', 'scikit-learn', 'PyYAML']
    # Reporting builds temporary catalogs even when persistence is disabled.
    if cfg.build_index or cfg.postprocess:
        packages.append('hnswlib')
    if cfg.postprocess:
        packages.append('matplotlib')
    if cfg.wandb.mode != 'disabled':
        packages.append('wandb')
    if require_dvc and cfg.dvc.enabled:
        packages.append('dvc')
    return {package: importlib.metadata.version(package) for package in packages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.config), indent=2))


if __name__ == '__main__':
    main()
