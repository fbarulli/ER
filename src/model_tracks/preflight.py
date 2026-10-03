"""Validate all three prepared populations before provisioning a shared VM."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import pandas as pd

from model_tracks.config import load_config


def preflight(config: Path) -> dict:
    from core.common import F, SEED, TRAIN_ROOT, resolve_model, training_cfg
    from graph_tracks.preflight import preflight as graph_preflight
    from graph_tracks.text_cache import checkpoint_hash
    from training.prepared_bundle import canonical_payload_rows, load_prepared_bundle, prepared_holdout
    from training.folds import normalize_gtin
    cfg = load_config(config)
    root = (TRAIN_ROOT / cfg.setup_dir).resolve()
    setup = json.loads((root / 'setup_manifest.json').read_text())
    is_smoke = setup.get('smoke', False)
    source_hash = hashlib.sha256(Path(F['dataset_deduped']).read_bytes()).hexdigest()
    if setup.get('source_catalog_sha256') != source_hash:
        raise ValueError('graph setup is stale: source catalog; rebuild locally before launch')
    labels_hash = hashlib.sha256(Path(F['labeled_pairs']).read_bytes()).hexdigest()
    if not is_smoke and setup.get('labeled_pairs_sha256') != labels_hash:
        raise ValueError('graph setup is stale: labeled pairs; rebuild locally before launch')
    model = Path(resolve_model(cfg.text_model))
    if checkpoint_hash(model) != setup['text_checkpoint_sha256']:
        raise ValueError('text baseline differs from frozen hybrid checkpoint')
    checks = {track: graph_preflight(root / f'{track}.yaml', check_device=False)
              for track in ('gnn_only', 'hybrid')}
    manifest, bundle = load_prepared_bundle((TRAIN_ROOT / cfg.text_bundle).resolve())
    from core.schemas import DataTuple
    DataTuple(n_df=len(bundle['df']), **{key: bundle[key] for key in
              ('payload', 'structured_features', 'row_bc', 'country', 'pos', 'hp_pairs', 'emb0')})
    n_payload = len(bundle['payload'])
    import numpy as np
    for pairs_key, sources_key in (('neg', 'neg_sources'), ('train_neg', 'train_neg_sources')):
        pairs = np.asarray(bundle[pairs_key])
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
            raise ValueError(f'{pairs_key} must contain integer endpoint pairs')
        if pairs.size and (pairs.min() < 0 or pairs.max() >= n_payload):
            raise ValueError(f'{pairs_key} endpoints are outside the payload')
        if len(pairs) != len(bundle[sources_key]):
            raise ValueError(f'{sources_key} does not align with {pairs_key}')
    if not is_smoke:
        for key in ('labeled_pairs', 'canonical_records', 'gate_results'):
            if hashlib.sha256(bundle[f'{key}_csv']).hexdigest() != hashlib.sha256(F[key].read_bytes()).hexdigest():
                raise ValueError(f'text bundle is stale: {key}; rebuild locally before launch')
    canonical_payload_rows(len(bundle['df']), bundle['payload'], bundle['row_bc'])
    train, dev, test = prepared_holdout(bundle, dict(training_cfg().split), seed=SEED)
    roles = {normalize_gtin(key): split for split, values in
             [('train', train), ('dev', dev), ('test', test)] for key in values}
    catalog = pd.read_csv(root / 'eligible_catalog.csv', dtype=str, keep_default_na=False, low_memory=False)
    splits = pd.read_csv(root / 'listing_splits.csv', dtype=str).set_index('sku_id').split
    for row in catalog.itertuples(index=False):
        if roles.get(normalize_gtin(row.gtin)) != splits[row.sku_id]:
            raise ValueError('text/graph prepared split mismatch')
    from core.identity_policy import reviewed_row_mask
    if reviewed_row_mask(bundle['df']).any():
        raise ValueError('text bundle contains held identity listings')
    # Run the same read-only diet contract locally and on the remote snapshot.
    # Strict drift checks prevent launch under a different augmentation config.
    diet = subprocess.run(
        [sys.executable, str(TRAIN_ROOT / 'scripts/diet_manifest.py'),
         str((TRAIN_ROOT / cfg.text_bundle).resolve())], cwd=TRAIN_ROOT,
        env={**os.environ, 'PYTHONPATH': str(TRAIN_ROOT / 'src'),
             'PREPARED_BUNDLE_DRIFT_STRICT': '1'},
        capture_output=True, text=True, timeout=120,
    )
    if diet.returncode:
        raise ValueError('text bundle diet preflight failed:\n' + diet.stdout + diet.stderr)
    return {'text': {'bundle_sha256': manifest.sha256, 'payload': manifest.payload_variant,
                     'masking_profile': manifest.masking_profile, 'rows': manifest.n_df,
                     'diet': {'status': 'pass', 'log': diet.stdout}},
            **checks, 'hybrid_text_mode': 'frozen shared baseline; independent of current text fine-tuning',
            'source_catalog_sha256': source_hash, 'labeled_pairs_sha256': labels_hash,
            'parallel_workers': 3, 'colab_sessions': 1, 'colab_control_channels': 1}
