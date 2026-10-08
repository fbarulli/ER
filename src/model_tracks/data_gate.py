"""The data tests, run once before training.

Every prepared-input data enforcement the three tracks rely on executes here,
in the supervisor, before the track barrier releases a single worker. The gate
is the single authoritative answer to "may these immutable inputs be trained
on", so the training path re-reads its data without re-arguing it.

Scope is deliberately the *data* invariants: bundle attestation, frozen token
and epoch-batch plans, the shared training contract and its graph projections,
graph input manifests/tensors, split closure and held-out leakage. Runtime
invariants are not data enforcement and stay where they are — gradient flow,
optimizer progress, per-fold presentation coverage, resumable checkpoint state
and checkpoint selection can only be observed while training runs.

An attestation is a digest over the suite configuration and every input byte
this gate verified. A worker recomputes it and, on a match, treats the gate as
proof for its own bytes. A standalone worker launch, a different configuration,
or a single changed byte all leave enforcement active.

Every path and knob comes from the validated configuration models. Nothing
here substitutes a default for a declared input, and a declared input that is
absent fails loudly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core.perf_switches import perf_enabled
from core.run_log import RunLogger
from core.step_trace import timed
from model_tracks.config import load_config

_LOG = RunLogger(__name__)

#: Set by the supervisor for its children; holds the gate attestation.
ATTESTATION_ENV = 'ER_DATA_GATE'
#: Records that the suite gate ran against the declared GPU-pending baseline cache.
GPU_PENDING_ENV = 'ER_DATA_GATE_GPU_PENDING'
#: Ignores attestation trust entirely; every check stays live.
FORCE_ENV = 'ER_DATA_GATE_ENFORCE'
#: Suite configuration the attestation was computed over. Workers receive it so
#: a spawned trainer can recompute the same digest without extra arguments.
CONFIG_ENV = 'ER_DATA_GATE_CONFIG'

#: Digest recorded for a text cache the supervisor's baseline export still has
#: to produce. It is a declared state, not a missing input: `preflight` has
#: already verified the embedding request that will produce it, and the gate
#: digests that request's bytes below.
GPU_PENDING = 'declared GPU-pending; bound to the baseline embedding request'


class SplitPairCounts(BaseModel):
    model_config = ConfigDict(extra='forbid')
    positive: int = Field(ge=0)
    negative: int = Field(ge=0)


class TrackInputCensus(BaseModel):
    """What one track's executed configuration was validated against."""

    model_config = ConfigDict(extra='forbid')
    listings: int = Field(ge=1)
    pairs: dict[str, SplitPairCounts]
    text_dimension: int | None = Field(default=None, ge=1)
    device: str = Field(min_length=1)


class DataGateResult(BaseModel):
    """The gate's verdict and the proof a worker re-checks to trust it."""

    model_config = ConfigDict(extra='forbid')
    suite: dict
    tracks: dict[str, TrackInputCensus]
    attestation: str = Field(min_length=64, max_length=64)


def _resolve(config: Path) -> Path:
    from core.common import TRAIN_ROOT
    return Path(config) if Path(config).is_absolute() else TRAIN_ROOT / config


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


def suite_config() -> Path | None:
    """The suite configuration a worker must attest against, if one was given.

    Workers are separate processes that receive only the attestation, so the
    configuration travels beside it in the environment.  ``None`` means no suite
    gate published one -- the standalone gate and local preflight -- and callers
    must then verify everything rather than trust anything.
    """
    raw = os.environ.get(CONFIG_ENV)
    return None if not raw else _resolve(Path(raw))


def _owner_trusted(owner: str) -> bool:
    """Whether this worker may skip re-verifying the immutable inputs."""
    config = suite_config()
    return False if config is None else trusted(config, owner)


def _repo_relative(path: Path) -> str:
    from core.common import TRAIN_ROOT
    return path.resolve().relative_to(TRAIN_ROOT.resolve()).as_posix()


def _required(path: Path, owner: str) -> str:
    """Digest a declared input; its absence is a failure, never a skip."""
    from core.portable_archive import cached_file_digest
    if not path.is_file():
        raise FileNotFoundError(f'data gate input missing: {owner} -> {_repo_relative(path)}')
    return cached_file_digest(path)


def _suite_digest(config: Path) -> str:
    """Digest the suite configuration that decides how inputs are trained."""
    cfg = load_config(config)
    digest = hashlib.sha256()
    digest.update(json.dumps(cfg.model_dump(mode='json'), sort_keys=True,
                             separators=(',', ':')).encode())
    digest.update(b'\0')
    digest.update(str(config).encode())
    return digest.hexdigest()


@timed
def input_digests(config: Path, *, allow_gpu_pending: bool = False) -> dict[str, str]:
    """Digest every prepared input the gate verifies, keyed by owner.

    Binding the attestation to these bytes is what lets a worker skip the
    repeated re-verification: a changed input changes its digest and leaves
    enforcement active.
    """
    from core.common import TRAIN_ROOT, F
    from core.portable_archive import cached_file_digest
    from graph_tracks.config import load_config as load_graph_config
    cfg = load_config(config)
    layout = _setup_layout()
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    bundle = (TRAIN_ROOT / cfg.text_bundle).resolve()
    digests: dict[str, str] = {}
    # Inputs the suite gate reads directly. All are mandatory for this suite.
    # Every setup-tree name is the declared layout (training.preparation.
    # graph_setup), never re-spelled here.
    for owner, path in (('text_bundle', bundle),
                        ('text_bundle_manifest', Path(str(bundle) + '.json')),
                        ('setup_manifest', setup / layout.manifest),
                        ('text_config', setup / layout.text_config),
                        ('shared_training_data', setup / layout.shared_training_data),
                        ('text_training_binding', setup / layout.text_training_binding),
                        ('shared_training_projection', setup / layout.shared_training_projection),
                        ('text_export_request', setup / layout.text_export_request),
                        ('eligible_catalog', setup / layout.catalog),
                        ('listing_splits', setup / layout.splits),
                        # baseline export consumes these to produce the
                        # frozen embedding cache, so they are always gate inputs.
                        ('embedding_inputs', setup / layout.embedding_request),
                        ('prepared_text', setup / 'prepared_text.npz')):
        digests[owner] = _required(path, owner)
    for key in ('dataset_deduped', 'labeled_pairs', 'canonical_records', 'gate_results'):
        digests[key] = _required(Path(F[key]).resolve(), key)
    for track in ('gnn_only', 'cascade'):
        track_config = setup / layout.track_config(track)
        digests[f'{track}_config'] = _required(track_config, f'{track} config')
        settings = load_graph_config(track_config, expected_track=track)
        # These four are the graph model's own declared inputs; the schema says
        # which are optional for this track, so the configuration decides.
        declared = (('listings', settings.listings), ('pairs', settings.pairs),
                    ('input_manifest', settings.input_manifest),
                    ('text_cache', settings.text_cache))
        for key, raw in declared:
            if raw is None:
                continue
            path = (TRAIN_ROOT / raw).resolve()
            owner = f'{track}.{key}'
            if path.is_file():
                digests[owner] = cached_file_digest(path)
            elif key == 'text_cache' and allow_gpu_pending:
                digests[owner] = GPU_PENDING
            else:
                raise FileNotFoundError(
                    f'data gate input missing: {owner} -> {_repo_relative(path)}')
    return digests


@timed
def attestation(config: Path, *, allow_gpu_pending: bool = False) -> str:
    """The digest a worker must match to treat the gate as proof."""
    config = _resolve(config)
    digest = hashlib.sha256()
    digest.update(_suite_digest(config).encode())
    for owner, value in sorted(input_digests(config,
                                             allow_gpu_pending=allow_gpu_pending).items()):
        digest.update(b'\0')
        digest.update(owner.encode())
        digest.update(b'\0')
        digest.update(value.encode())
    return digest.hexdigest()


#: Per-process memo of `enforced` verdicts, keyed by (resolved config, pending).
#: A worker asks the same question several times (gate, preflight, trainer);
#: replaying `attestation` for each is redundant re-hashing of the same bytes.
_enforced: dict | None = None


def _gpu_pending() -> bool:
    """Whether the suite declared GPU-only caches as legitimately not built yet.

    The supervisor exports ER_DATA_GATE_GPU_PENDING=1 for every worker, so the
    environment is the one channel that reaches all three tracks.  The explicit
    keyword stays available for the supervisor's own call.
    """
    return os.environ.get(GPU_PENDING_ENV) == '1'


def enforced(config: Path, *, allow_gpu_pending: bool = False) -> bool:
    """Whether this process must run the data tests itself.

    False only inside a suite whose supervisor already ran the gate over exactly
    these configuration and input bytes.
    """
    if os.environ.get(FORCE_ENV) == '1':
        return True
    expected = os.environ.get(ATTESTATION_ENV)
    if not expected:
        return True
    resolved = _resolve(config)
    pending = allow_gpu_pending or _gpu_pending()
    if perf_enabled('data_gate.enforced_memo'):
        global _enforced
        if _enforced is None:
            _enforced = {}
        key = (str(resolved), pending)
        verdict = _enforced.get(key)
        if verdict is None:
            verdict = expected != attestation(resolved, allow_gpu_pending=pending)
            _enforced[key] = verdict
        return verdict
    return expected != attestation(resolved, allow_gpu_pending=pending)


def trusted(config: Path, owner: str, *, allow_gpu_pending: bool = False) -> bool:
    """Record that the gate already proved these inputs, and say so once."""
    if enforced(config, allow_gpu_pending=allow_gpu_pending):
        return False
    _LOG.info(f'[data-gate] {owner}: verified before training by the suite gate; '
              'configuration and input bytes unchanged')
    return True


@timed
def validate(config: Path, *, suite_inputs: dict | None = None,
             allow_gpu_pending: bool = False,
             native_token_model: Path | None = None) -> DataGateResult:
    """Run every data test the three tracks depend on. Raises on any failure.

    Returns the suite preflight summary plus each track's input census obtained
    under the exact configuration that track will execute. `suite_inputs` reuses
    a preflight summary the caller already holds; the Colab supervisor passes
    none so the gate re-runs it against the VM's own bytes.
    """
    from core.common import TRAIN_ROOT
    from graph_tracks.config import GraphConfig
    from graph_tracks.preflight import load_inputs
    from model_tracks.config import load_config as load_suite_config
    from model_tracks.preflight import preflight
    from model_tracks.worker import graph_worker_settings

    config = _resolve(config)
    cfg = load_suite_config(config)
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    if suite_inputs is None:
        suite_inputs = preflight(config, allow_gpu_pending=allow_gpu_pending,
                                 native_token_model=native_token_model)
    tracks: dict[str, TrackInputCensus] = {}
    # Validate each graph track's inputs under its executed configuration, not
    # the prepared one: the worker overrides device, epochs, postprocessing and
    # the GPU execution backends, and CUDA additionally requires prepared
    # tensors. Building that configuration through the worker's own helper is
    # what keeps this gate from passing on inputs the trainer would reject.
    for track in ('gnn_only',):
        settings = GraphConfig.model_validate(
            graph_worker_settings(setup, cfg, track, gpu_only=True))
        _, records, pairs, vectors, _ = load_inputs(settings, verify_inputs=True)
        tracks[track] = TrackInputCensus(
            listings=len(records),
            pairs={split: SplitPairCounts(positive=int(labels.sum()),
                                          negative=int((labels == 0).sum()))
                   for split, (_, labels) in pairs.items()},
            text_dimension=None if vectors is None else int(vectors.shape[1]),
            device=settings.device)
    # The cascade is a combinator over the same frozen graph population and the
    # trained text ANN. It trains nothing, so it declares no shared projection;
    # its trained artifacts are validated by the worker after training.
    cascade_settings = GraphConfig.model_validate(
        graph_worker_settings(setup, cfg, 'cascade', gpu_only=True))
    tracks['cascade'] = TrackInputCensus(
        listings=tracks['gnn_only'].listings, pairs=tracks['gnn_only'].pairs,
        text_dimension=None, device=cascade_settings.device)
    return DataGateResult(suite=suite_inputs, tracks=tracks,
                          attestation=attestation(config,
                                                  allow_gpu_pending=allow_gpu_pending))


def census_tracks(result: DataGateResult) -> dict[str, dict]:
    """The gate's per-track census as plain JSON-serializable values.

    The events/attestation boundary needs bytes, not model objects: emitting
    this dict through json.dump/events.emit must never raise on a census type.
    """
    return {track: census.model_dump(mode='json')
            for track, census in result.tracks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--allow-gpu-pending', action='store_true',
                        help='accept the declared pending baseline embedding cache')
    parser.add_argument('--attestation-only', action='store_true',
                        help='print this configuration/input digest and exit')
    args = parser.parse_args()
    if args.attestation_only:
        print(attestation(args.config))
        return
    result = validate(args.config, allow_gpu_pending=args.allow_gpu_pending)
    print(result.model_dump_json(indent=2))
    print(f'[data-gate] passed; attestation={result.attestation}', flush=True)


if __name__ == '__main__':
    main()
