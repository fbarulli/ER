"""Validate all three prepared populations before provisioning a shared VM."""
from pathlib import Path
import hashlib
import json
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
    model = Path(resolve_model(cfg.text_model))
    if checkpoint_hash(model) != setup['text_checkpoint_sha256']:
        raise ValueError('text baseline differs from frozen hybrid checkpoint')
    checks = {track: graph_preflight(root / f'{track}.yaml', check_device=False)
              for track in ('gnn_only', 'hybrid')}
    manifest, bundle = load_prepared_bundle((TRAIN_ROOT / cfg.text_bundle).resolve())
    for key in ('labeled_pairs', 'canonical_records', 'gate_results'):
        if hashlib.sha256(bundle[f'{key}_csv']).hexdigest() != hashlib.sha256(F[key].read_bytes()).hexdigest():
            raise ValueError(f'text bundle is stale: {key}; rebuild locally before launch')
    canonical_payload_rows(len(bundle['df']), bundle['payload'], bundle['row_bc'])
    train, dev, test = prepared_holdout(bundle, dict(training_cfg().split), seed=SEED)
    roles = {normalize_gtin(key): split for split, values in
             [('train', train), ('dev', dev), ('test', test)] for key in values}
    catalog = pd.read_csv(root / 'eligible_catalog.csv', dtype=str, keep_default_na=False, low_memory=False)
    splits = pd.read_csv(root / 'listing_splits.csv', dtype=str).set_index('product_id').split
    for row in catalog.itertuples(index=False):
        if roles.get(normalize_gtin(row.barcode)) != splits[row.product_id]:
            raise ValueError('text/graph prepared split mismatch')
    from core.identity_policy import reviewed_row_mask
    if reviewed_row_mask(bundle['df']).any():
        raise ValueError('text bundle contains held identity listings')
    return {'text': {'bundle_sha256': manifest.sha256, 'payload': manifest.payload_variant,
                     'masking_profile': manifest.masking_profile, 'rows': manifest.n_df},
            **checks, 'hybrid_text_mode': 'frozen shared baseline; independent of current text fine-tuning',
            'parallel_workers': 3, 'colab_sessions': 1, 'colab_control_channels': 1}
