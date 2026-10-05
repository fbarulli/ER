"""Share one immutable base payload across stages of a single preparation run."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle

from core.manifest import sha256_file


def fingerprint(df, variant):
    import pandas as pd
    from core.common import F, TRAIN_ROOT
    root = Path(TRAIN_ROOT)
    # The graph caller fills missing cells. Normalize only the fingerprint;
    # the original dataframe still goes through the established builder.
    frame = df.fillna('').reset_index(drop=True)
    rows = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    files = [Path(F[key]) for key in ('dataset_deduped', 'canonical_records',
                                    'gate_results', 'labeled_pairs', 'number_reference')]
    files += list((root / 'config').glob('*.yaml')) + list((root / 'config').glob('*.json'))
    files += list((root / 'src').rglob('*.py'))
    inputs = {str(path.resolve()): sha256_file(path) for path in sorted(set(files))}
    return {'variant': variant, 'columns': list(frame.columns), 'rows': len(frame),
            'frame_sha256': hashlib.sha256(rows).hexdigest(),
            'pandas_version': pd.__version__, 'inputs': inputs}


def load_base_data(df, *, payload_variant='full', cache_path=None):
    from core.common import training_cfg
    from training.preparation_run import active_preparation
    run = active_preparation()
    if run is not None:
        # One producer, isolated mutable views for augmentation consumers.
        import copy
        import pandas as pd
        normalized = df.fillna('').reset_index(drop=True)
        key = hashlib.sha256(pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
                             + repr(list(normalized.columns)).encode()
                             + payload_variant.encode()).hexdigest()
        if key not in run._base:
            spec = training_cfg().negative_supply
            if spec.mode == 'lane':
                from training.negative_supply import build_lane_training_data
                data = build_lane_training_data(df, payload_variant=payload_variant,
                    run_tag=spec.pairs_run_tag, mint_cap=spec.mint_cap)
            else:
                from pipeline import build_training_data
                data = build_training_data(df, payload_variant=payload_variant)
            run._base[key] = data
        return copy.deepcopy(run._base[key])

    # Negative-supply mode (owner ruling 2026-10-03). Default 'gate' keeps the
    # existing path byte-for-byte; 'lane' replaces the gate-derived negatives
    # with the real-partner-first lane and bypasses the shared-base cache (the
    # lane's pairs.csv is not part of the cache fingerprint).
    if training_cfg().negative_supply.mode == "lane":
        from training.negative_supply import build_lane_training_data

        spec = training_cfg().negative_supply
        return build_lane_training_data(
            df, payload_variant=payload_variant,
            run_tag=spec.pairs_run_tag, mint_cap=spec.mint_cap,
        )

    from pipeline import build_training_data
    requested = cache_path or os.environ.get('EUROMONITOR_SHARED_BASE_DATA')
    if not requested:
        return build_training_data(df, payload_variant=payload_variant)
    path = Path(requested)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = path.with_suffix(path.suffix + '.json')
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        expected = fingerprint(df, payload_variant)
        if path.exists() or header.exists():
            if not path.exists() or not header.exists():
                raise ValueError('Incomplete shared base payload; use a fresh preparation run')
            metadata = json.loads(header.read_text())
            if metadata['fingerprint'] != expected:
                raise ValueError('Stale shared base payload; inputs changed, use a fresh preparation run')
            if metadata['sha256'] != sha256_file(path):
                raise ValueError('Shared base payload checksum mismatch')
            print(f'[shared-base] verified reuse -> {path}', flush=True)
            with path.open('rb') as stream:
                return pickle.load(stream)
        data = build_training_data(df, payload_variant=payload_variant)
        if fingerprint(df, payload_variant) != expected:
            raise ValueError('Preparation inputs changed while building the shared base payload')
        partial = path.with_suffix(path.suffix + '.partial')
        with partial.open('wb') as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {'fingerprint': expected, 'sha256': sha256_file(partial)}
        partial.replace(path)
        temporary = header.with_suffix(header.suffix + '.partial')
        temporary.write_text(json.dumps(metadata, indent=2) + '\n')
        temporary.replace(header)
        print(f'[shared-base] built once -> {path}', flush=True)
        return data
