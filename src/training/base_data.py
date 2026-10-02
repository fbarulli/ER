"""Share one immutable base payload across stages of a single preparation run."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


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
    inputs = {str(path.resolve()): digest(path) for path in sorted(set(files))}
    return {'variant': variant, 'columns': list(frame.columns), 'rows': len(frame),
            'frame_sha256': hashlib.sha256(rows).hexdigest(),
            'pandas_version': pd.__version__, 'inputs': inputs}


def load_base_data(df, *, payload_variant='full', cache_path=None):
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
            if metadata['sha256'] != digest(path):
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
        metadata = {'fingerprint': expected, 'sha256': digest(partial)}
        partial.replace(path)
        temporary = header.with_suffix(header.suffix + '.partial')
        temporary.write_text(json.dumps(metadata, indent=2) + '\n')
        temporary.replace(header)
        print(f'[shared-base] built once -> {path}', flush=True)
        return data
