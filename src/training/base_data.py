"""Share one immutable base payload across stages of a single preparation run.

RESPONSIBILITY MAP (single-responsibility decomposition; behaviour pinned)
-------------------------------------------------------------------------
- :class:`RunScopedPayload` — the active-preparation cache: one producer,
  isolated mutable views for augmentation consumers (deep-copied handoffs).
- :class:`LaneDispatch` — lane mode vs gate mode builder selection
  (negative_supply 'lane' bypasses the shared-base cache).
- :class:`SharedBasePayload` — the on-disk pickle contract: fingerprint-keyed
  cache reuse, checksum guard, and atomic partial->final publish under a file
  lock, so concurrent consumers reuse one build and an incompatible cache is
  rebuilt (never a freshness failure).
- :func:`fingerprint` — the inputs fingerprint (frame hash + input files).
- :func:`load_base_data` — the dispatch facade every consumer routes through.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle

from core.manifest import sha256_file
from core.run_log import RunLogger

_LOG = RunLogger(__name__)


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
    inputs = {}
    for path in _LOG.progress(
        sorted(set(files)), desc="payload_fingerprint_files", unit="file",
        total=len(sorted(set(files))),
    ):
        inputs[str(path.resolve())] = sha256_file(path)
    return {'variant': variant, 'columns': list(frame.columns), 'rows': len(frame),
            'frame_sha256': hashlib.sha256(rows).hexdigest(),
            'pandas_version': pd.__version__, 'inputs': inputs}


class RunScopedPayload:
    """The active-preparation cache: one producer, isolated consumer views."""

    @staticmethod
    def frame_key(df, payload_variant) -> str:
        import pandas as pd
        normalized = df.fillna('').reset_index(drop=True)
        key = hashlib.sha256(
            pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
            + repr(list(normalized.columns)).encode()
            + payload_variant.encode()
        ).hexdigest()
        return key

    @classmethod
    def load(cls, run, df, payload_variant):
        import copy
        key = cls.frame_key(df, payload_variant)
        if key not in run._base:
            run._base[key] = LaneDispatch.build(df, payload_variant=payload_variant)
        return copy.deepcopy(run._base[key])


class LaneDispatch:
    """Gate mode vs negative-supply lane mode (owner ruling 2026-10-03).

    Default 'gate' keeps the existing path byte-for-byte; 'lane' replaces
    the gate-derived negatives with the real-partner-first lane and bypasses
    the shared-base cache (the lane's pairs.csv is not part of the cache
    fingerprint).
    """

    @staticmethod
    def build(df, *, payload_variant: str):
        from core.common import training_cfg
        spec = training_cfg().negative_supply
        if spec.mode == 'lane':
            from training.negative_supply import build_lane_training_data
            return build_lane_training_data(
                df, payload_variant=payload_variant,
                run_tag=spec.pairs_run_tag, mint_cap=spec.mint_cap,
            )
        from pipeline import build_training_data
        return build_training_data(df, payload_variant=payload_variant)


class SharedBasePayload:
    """The file-cached base payload: verify fingerprint + checksum, then reuse.

    Cross-checkpoint consumers reuse one build via a file lock; the header
    JSON pins the fingerprint AND the pickle sha256, and the fingerprint is
    re-derived after every build so inputs changed mid-build can never be
    published.
    """

    def __init__(self, path: Path):
        self._path = path
        self._header = path.with_suffix(path.suffix + '.json')

    def reuse_or_build(self, df, *, payload_variant: str):
        from pipeline import build_training_data
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.with_suffix(self._path.suffix + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            expected = fingerprint(df, payload_variant)
            if self._path.exists() or self._header.exists():
                reused = self._reuse(expected)
                if reused is not None:
                    return reused
            data = build_training_data(df, payload_variant=payload_variant)
            if fingerprint(df, payload_variant) != expected:
                raise ValueError('Preparation inputs changed while building the shared base payload')
            self._publish(data, expected)
            _LOG.info(f'[shared-base] built once -> {self._path}')
            return data

    def _reuse(self, expected):
        """The cached payload when it matches these inputs, else None (rebuild).

        NO FRESHNESS FAILURE (owner directive 2026-10-08): a cache whose
        fingerprint does not match the current inputs is not "stale" — it is
        simply incompatible, and it is rebuilt below instead of refusing the
        run. The remaining guard is the payload's own checksum, i.e. the file's
        integrity.
        """
        if not self._path.exists() or not self._header.exists():
            _LOG.info('[shared-base] incomplete cache; rebuilding')
            return None
        try:
            metadata = json.loads(self._header.read_text())
        except (OSError, ValueError):
            _LOG.info('[shared-base] unreadable cache header; rebuilding')
            return None
        if metadata.get('fingerprint') != expected:
            _LOG.info('[shared-base] cache fingerprint does not match these inputs; rebuilding')
            return None
        if metadata.get('sha256') != sha256_file(self._path):
            raise ValueError('Shared base payload checksum mismatch')
        _LOG.info(f'[shared-base] verified reuse -> {self._path}')
        with self._path.open('rb') as stream:
            return pickle.load(stream)

    def _publish(self, data, expected) -> None:
        partial = self._path.with_suffix(self._path.suffix + '.partial')
        with partial.open('wb') as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {'fingerprint': expected, 'sha256': sha256_file(partial)}
        partial.replace(self._path)
        header_partial = self._header.with_suffix(self._header.suffix + '.partial')
        header_partial.write_text(json.dumps(metadata, indent=2) + '\n')
        header_partial.replace(self._header)


def load_base_data(df, *, payload_variant='full', cache_path=None):
    """THE one base-payload dispatch: run cache -> lane build -> shared file.

    Every training-side consumer routes through here so a single preparation
    run builds the immutable payload once (never per stage), lane mode is
    honoured exactly once, and cross-stage file sharing can only reuse a
    payload whose fingerprint+checksum still match.
    """
    from core.common import training_cfg
    from training.preparation_run import active_preparation
    run = active_preparation()
    if run is not None:
        # One producer, isolated mutable views for augmentation consumers.
        return RunScopedPayload.load(run, df, payload_variant)
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
    return SharedBasePayload(Path(requested)).reuse_or_build(
        df, payload_variant=payload_variant
    )
