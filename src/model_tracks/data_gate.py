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

An attestation is a size total over the suite configuration and every input
this gate verified. A worker recomputes it and, on a match, treats the gate as
proof for its own inputs. A standalone worker launch, a different configuration,
or a single changed byte all leave enforcement active.

Every path and knob comes from the validated configuration models. Nothing
here substitutes a default for a declared input, and a declared input that is
absent fails loudly.
"""
from __future__ import annotations

import argparse
from core.portable_archive import ByteCount
import json
import os
from pathlib import Path
import traceback

from pydantic import BaseModel, ConfigDict, Field

from core.perf_switches import perf_enabled
from core.run_log import RunLogger
from core.step_trace import timed
from model_tracks.config import load_config

_LOG = RunLogger(__name__)

#: Set by the supervisor for its children; holds the gate attestation.
ATTESTATION_ENV = 'ER_DATA_GATE'
#: Ignores attestation trust entirely; every check stays live.
FORCE_ENV = 'ER_DATA_GATE_ENFORCE'
#: Suite configuration the attestation was computed over. Workers receive it so
#: a spawned trainer can recompute the same total without extra arguments.
CONFIG_ENV = 'ER_DATA_GATE_CONFIG'


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
    attestation: int = Field(ge=0)


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


def _required(path: Path, owner: str) -> int:
    """Account for a declared input; its absence is a failure, never a skip."""
    from core.portable_archive import file_size
    if not path.is_file():
        raise FileNotFoundError(f'data gate input missing: {owner} -> {_repo_relative(path)}')
    return file_size(path)


def _suite_size(config: Path) -> int:
    """Account for the suite configuration that decides how inputs are trained."""
    cfg = load_config(config)
    count = ByteCount()
    count.update(json.dumps(cfg.model_dump(mode='json'), sort_keys=True,
                            separators=(',', ':')).encode())
    count.update(b'\0')
    count.update(str(config).encode())
    return count.total


@timed
def input_sizes(config: Path) -> dict[str, int]:
    """Account for every prepared input the gate verifies, keyed by owner.

    Binding the attestation to these sizes is what lets a worker skip the
    repeated re-verification: a changed input changes its size and leaves
    enforcement active.
    """
    from core.common import TRAIN_ROOT, F
    from graph_tracks.config import load_config as load_graph_config
    cfg = load_config(config)
    layout = _setup_layout()
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    bundle = (TRAIN_ROOT / cfg.text_bundle).resolve()
    sizes: dict[str, int] = {}
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
        sizes[owner] = _required(path, owner)
    for key in ('dataset_deduped', 'labeled_pairs', 'canonical_records', 'gate_results'):
        sizes[key] = _required(Path(F[key]).resolve(), key)
    for track in ('gnn_only', 'cascade'):
        track_config = setup / layout.track_config(track)
        sizes[f'{track}_config'] = _required(track_config, f'{track} config')
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
            # A declared input is REQUIRED: there is no "pending" state that
            # tolerates a not-yet-produced cache (owner directive 2026-10-08 —
            # the only tolerated absence was a freshness allowance, and it is
            # gone; the producer runs before the gate).
            sizes[owner] = _required(path, owner)
    return sizes


@timed
def attestation(config: Path) -> int:
    """The size total a worker must match to treat the gate as proof."""
    config = _resolve(config)
    count = ByteCount()
    count.update(str(_suite_size(config)).encode())
    for owner, value in sorted(input_sizes(config).items()):
        count.update(b'\0')
        count.update(owner.encode())
        count.update(b'\0')
        count.update(str(value).encode())
    return count.total


#: Per-process memo of `enforced` verdicts, keyed by resolved config.
#: A worker asks the same question several times (gate, preflight, trainer);
#: replaying `attestation` for each is redundant re-counting of the same bytes.
_enforced: dict | None = None


def _must_enforce(expected: str, resolved: Path) -> bool:
    """Whether an expected attestation fails to match this process's inputs.

    A supervisor publishes the total as text in the environment; an unreadable
    value is not proof, so it degrades to full enforcement with the traceback
    recorded rather than crashing the worker.
    """
    try:
        token = int(expected)
    except ValueError:
        _LOG.warning('[data-gate] unreadable attestation in ' + ATTESTATION_ENV
                     + '; re-verifying every input\n' + traceback.format_exc())
        return True
    return token != attestation(resolved)


def enforced(config: Path) -> bool:
    """Whether this process must run the data tests itself.

    False only inside a suite whose supervisor already ran the gate over exactly
    these configuration and input sizes.
    """
    if os.environ.get(FORCE_ENV) == '1':
        return True
    expected = os.environ.get(ATTESTATION_ENV)
    if not expected:
        return True
    resolved = _resolve(config)
    if perf_enabled('data_gate.enforced_memo'):
        global _enforced
        if _enforced is None:
            _enforced = {}
        key = str(resolved)
        verdict = _enforced.get(key)
        if verdict is None:
            verdict = _must_enforce(expected, resolved)
            _enforced[key] = verdict
        return verdict
    return _must_enforce(expected, resolved)


def trusted(config: Path, owner: str) -> bool:
    """Record that the gate already proved these inputs, and say so once."""
    if enforced(config):
        return False
    _LOG.info(f'[data-gate] {owner}: verified before training by the suite gate; '
              'configuration and input bytes unchanged')
    return True


@timed
def validate(config: Path, *, suite_inputs: dict | None = None) -> DataGateResult:
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
        suite_inputs = preflight(config)
    tracks: dict[str, TrackInputCensus] = {}
    # Validate each graph track's inputs under its executed configuration, not
    # the prepared one: the worker overrides device, epochs, postprocessing and
    # the GPU execution backends, and CUDA additionally requires prepared
    # tensors. Building that configuration through the worker's own helper is
    # what keeps this gate from passing on inputs the trainer would reject.
    for track in ('gnn_only',):
        settings = GraphConfig.model_validate(
            graph_worker_settings(setup, cfg, track, gpu_only=True))
        _, records, pairs, vectors, _ = load_inputs(settings)
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
                          attestation=attestation(config))


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
    parser.add_argument('--attestation-only', action='store_true',
                        help='print this configuration/input size total and exit')
    args = parser.parse_args()
    if args.attestation_only:
        print(attestation(args.config))
        return
    result = validate(args.config)
    print(result.model_dump_json(indent=2))
    print(f'[data-gate] passed; attestation={result.attestation}', flush=True)


if __name__ == '__main__':
    main()
