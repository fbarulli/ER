"""Rebuild every full-training input without training or changing smoke inputs.

Run from ER: PYTHONPATH=src .venv/bin/python -m training.prepare_all

Shape of this module (one responsibility per unit):

  PreparationState / PreparedFile    persisted preparation contract
  _RunContext                        one immutable load of owning configs
  hashing/verification primitives    sha256, file_inventory, copy ISLANDbundle
  resume primitives                  verify_reusable_outputs, verify_stage_manifest
  stage measurement                  refresh_gate_census
  PrepareRun                         the orchestrator; each method owns one job
  prepare_all / main                 public wrapper + CLI
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Literal
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
import yaml
from tqdm import tqdm

from core.portable_archive import Digest
from core.run_log import RunLogger
from core.schemas import PREPARATION_REUSABLE_KEYS
from training.prepare_all_trace import send, timed, trace_step

STAGES = ('dedupe', 'cross_country_pairs', 'number_reference', 'verify_reference',
          'canonical_and_gates', 'gate_census', 'labeled_pairs', 'validation',
          'graph_inputs', 'full_bundle', 'suite_inputs', 'verify_handoff')

REUSABLE_KEYS = PREPARATION_REUSABLE_KEYS

_LOG = RunLogger(__name__)
_HASH_CHUNK_BYTES = 1 << 20
_LINUX_FICLONE = 0x40049409
_CLONE_UNSUPPORTED = frozenset({errno.EXDEV, errno.EOPNOTSUPP, errno.ENOTTY,
                                errno.EINVAL, errno.ENOSYS})
_RUN_TAG_PATTERN = r'[A-Za-z0-9_-]+'
_EXTRA_LANE_STAGES = frozenset({'negative_supply', 'discriminator'})

_STAGE_MODULES = {
    'dedupe': ['training.dedupe'],
    'cross_country_pairs': ['training.build_second04_pairs'],
    'number_reference': ['training.build_reference'],
    'verify_reference': ['training.build_reference', '--verify'],
    'canonical_and_gates': ['training.data_prep'],
    'labeled_pairs': ['training.labeled_pairs'],
    'validation': ['training.build_final_validation'],
}


class PreparedFile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sha256: Digest
    bytes: StrictInt = Field(ge=0)


class PreparationState(BaseModel):
    """Persisted resume contract; old unverified states require regeneration."""
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    status: Literal['running', 'complete', 'failed', 'deferred']
    resume_from: Literal['dedupe', 'validation', 'full_bundle', 'suite_inputs']
    training_started: Literal[False]
    smoke_updated: Literal[False]
    stages: list[str]
    run_dir: str
    tracks_config: str
    negative_supply_mode: Literal['gate', 'lane']
    negative_supply_run_tag: str = Field(pattern=_RUN_TAG_PATTERN)
    provenance: dict[str, Digest]
    smoke_original: dict[str, Digest]
    reusable_outputs: dict[str, PreparedFile] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_stages(self):
        if len(self.stages) != len(set(self.stages)) or set(self.stages) - set(STAGES) - _EXTRA_LANE_STAGES:
            raise ValueError('Preparation stages must be unique known stages')
        if self.status == 'complete' and 'verify_handoff' not in self.stages:
            raise ValueError('Complete preparation requires verified handoff')
        return self


@dataclass(frozen=True)
class _RunContext:
    """One validated load of the preparation's owning configuration and paths.

    Every path here is config-owned (model_tracks.yaml, training.yaml,
    paths.yaml); no call-site path literals live below this layer.
    """
    root: Path
    results: Path
    config_path: Path
    suite: Any
    training: Any
    prep: Any
    lane: Any
    reusable_keys: Sequence[str]
    files: Any
    checkpoint: Path
    setup: Path
    text_bundle: Path
    bundle: Path
    smoke: Path


@timed
def preparation_provenance(root: Path, suite_config: Path, checkpoint: str | Path) -> dict[str, str]:
    """Pin source, owning configs, raw input and baseline checkpoint content."""
    from core.common import CONFIG_PATH, TRAINING_CONFIG_PATH, VOCABULARY_CONFIG_PATH, DATA_PATH, TRAIN_ROOT, artifact
    from graph_tracks.text_cache import checkpoint_hash
    identity = {}
    for path in _LOG.progress(_provenance_paths(root, suite_config), desc='provenance_hash', unit='file'):
        if path.resolve() == Path(TRAINING_CONFIG_PATH).resolve():
            # The run measures its own gate census; it is output, not a setting.
            config = yaml.safe_load(path.read_text())
            identity[str(path.resolve())] = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()).hexdigest()
        else:
            identity[str(path.resolve())] = sha256(path)
    identity['text_checkpoint'] = checkpoint_hash(Path(checkpoint), use_memo=False)
    return identity


def _provenance_paths(root: Path, suite_config: Path) -> set[Path]:
    """Every byte-relevant input of a full preparation, as one resolved set."""
    from core.common import CONFIG_PATH, TRAINING_CONFIG_PATH, VOCABULARY_CONFIG_PATH, DATA_PATH, TRAIN_ROOT, artifact
    layouts = _pipeline_layouts()
    paths = set((root / layouts['source_code_dir']).rglob('*.py')) | \
        set((root / layouts['scripts_dir']).rglob('*.py'))
    paths.update(path for path in (root / layouts['config_dir']).rglob('*')
                 if path.suffix in {'.yaml', '.yml', '.json'} and path.is_file())
    paths.update([Path(CONFIG_PATH), Path(TRAINING_CONFIG_PATH),
                  Path(VOCABULARY_CONFIG_PATH), Path(suite_config), Path(DATA_PATH)])
    # Measured-evidence inputs the pipeline consumes fail-loud: they are
    # tracked build inputs, so any regeneration must regenerate everything.
    paths.update([TRAIN_ROOT / 'artifacts/evidence/attribute_universe_census.json',
                  artifact('semantic_family_registry')])
    return paths


def _pipeline_layouts() -> dict[str, str]:
    """The pipeline's repo-layout bindings (paths.yaml layouts block)."""
    from core.common import LAYOUTS
    return {key: str(LAYOUTS[key].template)
            for key in ('source_code_dir', 'scripts_dir', 'config_dir',
                        'model_tracks_config', 'negative_supply_discriminator')}


@timed
def verify_reusable_outputs(entries: dict[str, PreparedFile]) -> None:
    """Verify every reusable resume artifact still matches its recorded bytes."""
    if not entries:
        raise ValueError('Resume has no verified prepared outputs; regenerate inputs')
    for path, entry in _LOG.progress(entries.items(), desc='verify_reusable', unit='file'):
        if Path(path).stat().st_size != entry.bytes or sha256(path) != entry.sha256:
            raise ValueError(f'Stale prepared resume input: {path}')


@timed
def sha256(path: str | Path) -> str:
    """Stream one file through sha256 with a byte-accurate progress bar."""
    digest = hashlib.sha256()
    source = Path(path)
    with source.open('rb') as stream, \
            _LOG.bar(total=source.stat().st_size, desc=f'sha256:{source.name}',
                     unit='B') as bar:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b''):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


@timed
def file_inventory(paths: Sequence[Path]) -> dict[str, dict[str, str | int]]:
    """Hash each resolved artifact once per snapshot; never cache across stages."""
    inventory: dict[str, dict[str, str | int]] = {}
    for path in _LOG.progress(dict.fromkeys(path.resolve() for path in paths),
                              desc='inventory', unit='file'):
        inventory[str(path)] = PreparedFile(sha256=sha256(path),
                                            bytes=path.stat().st_size).model_dump()
    return inventory


@timed
def copy_bundle(source: Path, destination: Path) -> None:
    """Use filesystem copy-on-write when available, keeping independent files.

    Never hard-link mutable bundle paths: a later write must not corrupt its peer.
    Unsupported filesystems use shutil's platform-accelerated copy instead.
    """
    try:
        with trace_step('copy_bundle.ioctl_clone'):
            with source.open('rb') as src, destination.open('wb') as dst:
                fcntl.ioctl(dst.fileno(), _LINUX_FICLONE, src.fileno())
    except OSError as error:
        if error.errno not in _CLONE_UNSUPPORTED:
            raise
        with trace_step('copy_bundle.shutil_fallback'):
            _copy_file_with_progress(source, destination)
        shutil.copystat(source, destination)
    else:
        shutil.copystat(source, destination)


def _copy_file_with_progress(source: Path, destination: Path) -> None:
    """Byte-stream one file with a size bar (the clone-unsupported fallback)."""
    with source.open('rb') as src, destination.open('wb') as dst, \
            _LOG.bar(total=source.stat().st_size, desc='copy_bundle', unit='B') as bar:
        for chunk in iter(lambda: src.read(_HASH_CHUNK_BYTES), b''):
            dst.write(chunk)
            bar.update(len(chunk))


@timed
def verify_stage_manifest(path: str | Path, *, required_inputs: Sequence[str | Path] = ()) -> None:
    """Fail loudly when a prerequisite manifest is incomplete or stale."""
    from core.manifest import source_tree_sha256
    from core.schemas import StageManifest
    from core.tracing import trace_path
    manifest = _load_stage_manifest(path)
    if manifest.status != 'complete':
        raise ValueError(f'Incomplete prerequisite: {path}')
    if required_inputs and manifest.environment.get('source_sha256') != source_tree_sha256():
        raise ValueError(f'Prerequisite source changed or was not recorded: {path}; regenerate inputs')
    recorded_inputs = {Path(entry.path).resolve() for entry in manifest.inputs}
    if set(Path(item).resolve() for item in required_inputs) - recorded_inputs:
        raise ValueError(f'Prerequisite lacks current config/reference provenance: {path}; regenerate inputs')
    _verify_manifest_hashes(manifest, trace_path)


def _load_stage_manifest(path: str | Path):
    """Parse one StageManifest, attaching the manifest path to parse errors."""
    from core.schemas import StageManifest
    try:
        with trace_step('verify_stage_manifest.load'):
            return StageManifest.model_validate_json(Path(path).read_text())
    except ValueError as exc:
        exc.add_note(f'Preparation prerequisite manifest: {path}')
        raise


def _verify_manifest_hashes(manifest, trace_path) -> None:
    """Rehash every recorded input/output; the append-only trace is exempt."""
    with trace_step('verify_stage_manifest.stale_hash_check',
                    outputs=len(manifest.inputs) + len(manifest.outputs)):
        entries = manifest.inputs + manifest.outputs
        for entry in _LOG.progress(entries, desc='verify_manifest', unit='entry',
                                   total=len(entries)):
            # The consolidated trace is append-only across stages, not frozen input.
            if Path(entry.path).resolve() == trace_path().resolve():
                continue
            if sha256(entry.path) != entry.sha256:
                raise ValueError(f'Stale prerequisite: {entry.path}')


@timed
def refresh_gate_census(gate_csv: Path, report_path: Path) -> dict[str, int]:
    """Measure the gate-decision census and record it as a run artifact.

    Keep the structural checks (no self-pairs, no duplicates, unknown
    decisions) — they guard the frame, not a pinned number. The counts land
    in run_dir/gate_census.json and the stage manifest as measured records;
    there is deliberately NO config rewrite and no pinned equality check
    (owner ruling 2026-10-06: the pin system is removed).
    """
    import pandas as pd
    from core.manifest import atomic_write_json
    frame = pd.read_csv(gate_csv, usecols=['gtin1', 'gtin2', 'gate_decision'],
                        dtype=str, keep_default_na=False)
    _validate_gate_frame(frame)
    counts = {'total_pairs': len(frame)}
    for key in _LOG.progress(('hard_no', 'proceed', 'fallback'), desc='gate_census',
                             unit='decision', total=3):
        counts[key] = int(frame.gate_decision.eq(key).sum())
    atomic_write_json(counts, report_path)
    return counts


def _validate_gate_frame(frame) -> None:
    """Structural checks guarding the census frame (fail-loud, no pins)."""
    if frame[['gtin1', 'gtin2']].apply(lambda column: column.str.strip().eq('')).any().any():
        raise ValueError('Empty gate pair endpoint')
    if frame.gtin1.eq(frame.gtin2).any():
        raise ValueError('Gate census contains self-pairs')
    if _ordered_identities(frame.gtin1, frame.gtin2).duplicated().any():
        raise ValueError('Duplicate candidate pairs')
    if set(frame.gate_decision) - {'hard_no', 'proceed', 'fallback'}:
        raise ValueError('Unknown gate decisions')


def _ordered_identities(left, right):
    """One canonical (small, large) identity per pair for duplicate checks."""
    import pandas as pd
    ordered = left.le(right)
    return pd.DataFrame({'left': left.where(ordered, right),
                         'right': right.where(ordered, left)})


def _load_run_context(tracks_config: Path | None) -> _RunContext:
    """Load the owning configs, cross-validate them, resolve derived paths."""
    with _LOG.section('prepare_all.load_configs'):
        config_path, suite, training = _load_preparation_configs(tracks_config)
    _validate_config_agreement(suite, training)
    return _derive_run_context(suite, training, config_path)


def _load_preparation_configs(tracks_config):
    """Read model_tracks.yaml + training.yaml afresh for long-lived callers."""
    from core.common import TRAIN_ROOT, TRAINING_CONFIG_PATH, training_cfg
    from model_tracks.config import load_config as load_suite
    root = Path(TRAIN_ROOT)
    config_path = Path(tracks_config or _default_tracks_config(root)).resolve()
    suite = load_suite(config_path)
    training = type(training_cfg()).model_validate(yaml.safe_load(Path(TRAINING_CONFIG_PATH).read_text()))
    return config_path, suite, training


def _validate_config_agreement(suite, training) -> None:
    """Fail before any work when suite and training configs disagree."""
    if suite.text_model != training.training.base_model:
        raise ValueError('suite text_model and training.base_model must agree before preparation')
    if suite.epochs > training.training.epochs:
        raise ValueError('suite epochs exceed locally prepared training epochs; update training.yaml first')


def _derive_run_context(suite, training, tracks_config) -> _RunContext:
    """Resolve the config-owned path map into the immutable run context."""
    from core.common import F, RESULTS, TRAIN_ROOT, resolve_model
    root, results = Path(TRAIN_ROOT), Path(RESULTS)
    prep = training.prep if hasattr(training, 'prep') else training.preparation
    return _RunContext(
        root=root, results=results,
        config_path=Path(tracks_config or _default_tracks_config(root)).resolve(),
        suite=suite, training=training, prep=prep, lane=training.negative_supply,
        reusable_keys=tuple(prep.reusable_keys), files=F,
        checkpoint=resolve_model(suite.text_model),
        setup=(root / suite.setup_dir).resolve(),
        text_bundle=(root / suite.text_bundle).resolve(),
        bundle=_full_baseline_bundle(root, training),
        smoke=root / prep.smoke_dir,
    )


def _default_tracks_config(root: Path) -> Path:
    """The declared default three-track configuration (paths.yaml layouts)."""
    return root / str(_pipeline_layouts()['model_tracks_config'])


def _full_baseline_bundle(root, training) -> Path:
    """The one configured full baseline bundle (the contract requires exactly one)."""
    bundles = training.colab.full_prepared_bundles
    if len(bundles) != 1:
        raise ValueError('prepare_all requires exactly one configured full baseline bundle')
    return (root / bundles[0]).resolve()


@contextmanager
def _acquire_prepare_lock(lock_path: Path):
    """Single-flight: one preparation at a time across processes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        with trace_step('prepare_all.lock_acquire'):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another training preparation is already running') from None
        yield lock


@timed
def _verify_resume_prerequisites(resume_from: str) -> None:
    """'validation' resume re-verifies the CSV stages' committed manifests."""
    if resume_from != 'validation':
        return
    from core.common import CONFIG_PATH, RESULTS, TRAINING_CONFIG_PATH, VOCABULARY_CONFIG_PATH, F
    for name in ('data_prep', 'labeled_pairs'):
        verify_stage_manifest(Path(RESULTS) / 'manifests' / (name + '.json'),
            required_inputs=([CONFIG_PATH, VOCABULARY_CONFIG_PATH, F['number_reference']]
                             if name == 'data_prep' else [TRAINING_CONFIG_PATH]))


@timed
def _hash_smoke_baseline(smoke: Path) -> dict[str, str]:
    """Hash the pristine smoke tree once; later stages prove it unmodified."""
    with trace_step('prepare_all.smoke_before_hash'):
        before: dict[str, str] = {}
        for path in _LOG.progress((path for path in smoke.rglob('*') if path.is_file()),
                                  desc='smoke_before_hash', unit='file'):
            before[str(path)] = sha256(path)
    return before


def _prepare_environment(root: Path, run_dir: Path, prep) -> tuple[dict[str, str], str]:
    """Child-stage environment contract; returns (env, shared_base_payload)."""
    env = os.environ.copy()
    env['PYTHONPATH'] = (str(root / _pipeline_layouts()['source_code_dir'])
                         + os.pathsep + str(root))
    env['EUROMONITOR_SHARED_BASE_DATA'] = _shared_base_payload_path(run_dir)
    _bind_cohort_tag(env)
    env.pop('WANDB_API_KEY', None)
    # Preparation mutates inputs; inherited worker attestations are invalid.
    env['ER_DATA_GATE_ENFORCE'] = '1'
    return env, env['EUROMONITOR_SHARED_BASE_DATA']


def _shared_base_payload_path(run_dir: Path) -> str:
    """One timestamped shared-base pickle path per preparation run."""
    return str(run_dir / ('shared_base_' +
                          datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.pkl'))


def _bind_cohort_tag(env: dict[str, str]) -> None:
    """Tag local preps with their mounted cohort (dataset.csv staged copy)."""
    from core.common import mounted_cohort
    # Cohort-scoped augmentation counts (training.yaml cohort_counts) key
    # off ER_COHORT_TAG: a local prep that stages a cohort export copy at
    # dataset.csv must tag itself so the full-dataset vendor quota can
    # never out-mine the cohort's supply (10k exhausts at 287 < 300).
    env.setdefault('ER_COHORT_TAG', '' if (cohort := mounted_cohort()) == 'full' else cohort)


def _initial_manifest(*, root: Path, resume_from: str, run_dir: Path, config_path: Path,
                      lane, run_tag: str, shared_base_payload: str, provenance: dict,
                      smoke_before: dict) -> dict[str, Any]:
    """The run's first persistence: running state, provenance, smoke baseline."""
    return {
        'status': 'running', 'resume_from': resume_from,
        'training_started': False, 'smoke_updated': False,
        'stages': [], 'run_dir': str(run_dir),
        'shared_base_payload': shared_base_payload,
        'tracks_config': str(config_path),
        'negative_supply_mode': lane.mode,
        'negative_supply_run_tag': run_tag,
        'provenance': provenance, 'smoke_original': smoke_before,
        'reusable_outputs': {},
        'hybrid_embeddings': 'GPU pending: frozen baseline forward before hybrid training',
    }


class PrepareRun:
    """One full preparation: owns its layout, manifest, stages and reports.

    Each method below owns exactly one responsibility; the orchestrator is
    execute() and nothing else.
    """

    def __init__(self, *, run_dir=None, resume_from='dedupe', tracks_config=None,
                 negative_supply_run_tag=None):
        self.resume_from = resume_from
        self.requested_tag = negative_supply_run_tag
        self.context = _load_run_context(tracks_config)
        self.run_dir, self.run_tag = self._resolve_layout(run_dir)
        self.suite_archive = self.run_dir / (
            f"{self.context.prep.suite_archive_name}.{self.context.suite.input_archive_format}")
        self.env, self.shared_base_payload = _prepare_environment(
            self.context.root, self.run_dir, self.context.prep)
        self.smoke_before: dict[str, str] | None = None
        self.manifest: dict[str, Any] = {}
        self.manifest_path = self.run_dir / self.context.prep.manifest_file

    def _resolve_layout(self, run_dir) -> tuple[Path, str]:
        """Resolve run directory + lane-locked negative-supply tag (config paths)."""
        from core.common import RESULTS
        with _LOG.section('prepare_all.resolve_paths'):
            run_tag = _resolve_run_tag(self.context.lane, self.requested_tag)
            directory = Path(run_dir or Path(RESULTS) / self.context.prep.run_dir_base /
                             datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')).resolve()
            run_tag = run_tag or ('prep_' + directory.name)
            if not re.fullmatch(_RUN_TAG_PATTERN, run_tag):
                raise ValueError('preparation directory name requires an explicit valid --negative-supply-run-tag')
            directory.mkdir(parents=True, exist_ok=True)
            os.environ['ER_TIMING_LOG'] = str(directory / self.context.prep.timings_log)
        return directory, run_tag

    # --- persistence ------------------------------------------------------

    @timed
    def publish(self) -> None:
        """One atomic manifest write + timing/offender report refresh."""
        self._write_manifest_atomically()
        self._write_timing_summary()
        self._write_offender_report()

    def _write_manifest_atomically(self) -> None:
        """Validate + swap the running manifest into place atomically."""
        temporary = self.manifest_path.with_suffix('.tmp')
        state = PreparationState.model_validate(self.manifest)
        temporary.write_text(state.model_dump_json(indent=2) + '\n')
        temporary.replace(self.manifest_path)

    def _write_timing_summary(self) -> None:
        """Materialize the run's timing JSON (status + per-stage seconds)."""
        prep = self.context.prep
        timing_path = self.run_dir / prep.timings_file
        timing_temporary = timing_path.with_suffix('.tmp')
        timing_temporary.write_text(json.dumps({
            'status': self.manifest['status'],
            'stages': self.manifest.get('stage_metrics', {}),
            'total_stage_seconds': round(sum(self.manifest.get('stage_seconds', {}).values()), 3),
        }, indent=2) + '\n')
        timing_temporary.replace(timing_path)

    def _write_offender_report(self) -> None:
        """One self-rewriting worst-offender report for the whole run."""
        from core.timing import collect_timing_entries, write_offender_report
        write_offender_report(self.run_dir / self.context.prep.offender_report,
                              collect_timing_entries(self.run_dir,
                                                     self.manifest.get('stage_seconds', {})))

    # --- stage primitives -------------------------------------------------

    @timed
    def run(self, name: str, arguments: list[str], *, check: bool = True):
        """Run one child stage under its own log + stage-timing environment."""
        _LOG.info(f'[prepare] {name} -> {self.run_dir / (name + self.context.prep.stage_log_suffix)}')
        self.env['ER_TIMING_OUT'] = str(self.run_dir / (name + self.context.prep.stage_timing_suffix))
        self.env['ER_TIMING_LOG'] = str(self.run_dir / self.context.prep.timings_log)
        with (self.run_dir / (name + self.context.prep.stage_log_suffix)).open('w') as log:
            from training.preparation_run import active_preparation
            result = active_preparation().run_stage(arguments, root=self.context.root,
                                                    env=self.env, log=log)
        self.manifest.setdefault('stage_metrics', {})[name]['returncode'] = result.returncode
        if check:
            result.check_returncode()
        return result

    @timed
    def archive(self, path: Path, name: str) -> None:
        """Move one superseded artifact into the run's archive directory."""
        from training.preparation_run import active_preparation
        active_preparation().invalidate(path)
        if path.exists():
            destination = self.run_dir / self.context.prep.archive_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination = destination.with_name(destination.name + '.' +
                    datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
            shutil.move(str(path), str(destination))

    # --- resume -----------------------------------------------------------

    @timed
    def replay_resume_state(self) -> None:
        """Validate the saved manifest against this run and replay its stages."""
        if self.resume_from not in {'full_bundle', 'suite_inputs'}:
            return
        with trace_step('prepare_all.resume_replay'):
            previous_state = PreparationState.model_validate_json(self.manifest_path.read_text())
            previous = previous_state.model_dump(mode='json')
            self._validate_resume(previous_state, previous)
            self.run_tag = previous_state.negative_supply_run_tag
            prerequisite = 'graph_inputs' if self.resume_from == 'full_bundle' else 'full_bundle'
            if prerequisite not in previous.get('stages', []) or \
                    previous.get('tracks_config') != str(self.context.config_path):
                raise ValueError(
                    f"{self.resume_from} resume requires this run's completed {prerequisite}")
            remaining = set(STAGES[STAGES.index(self.resume_from):])
            self.manifest['stages'] = [stage for stage in previous['stages']
                                       if stage not in remaining]
            self.manifest.update(previous, status='running', resume_from=self.resume_from,
                                 stages=self.manifest['stages'],
                                 shared_base_payload=self.shared_base_payload)
            self.manifest.pop('failed_stage', None)
            self.manifest.pop('error', None)

    def _validate_resume(self, previous_state, previous) -> None:
        """Fail loudly when this run differs from the saved one in any byte."""
        if previous_state.run_dir != str(self.run_dir):
            raise ValueError('Preparation manifest belongs to another run directory')
        if self.requested_tag and self.requested_tag != previous_state.negative_supply_run_tag:
            raise ValueError('Resume negative-supply run tag differs from the saved run')
        if previous_state.provenance != self.manifest['provenance']:
            raise ValueError('Preparation source/config/raw input/checkpoint changed; regenerate inputs')
        if previous_state.smoke_original != self.smoke_before:
            raise ValueError('Smoke files changed since this preparation began')
        verify_reusable_outputs(previous_state.reusable_outputs)

    # --- stage planning ---------------------------------------------------

    def plan_stages(self) -> list[str]:
        """The run's ordered stage list, including the negative-supply lane."""
        stages = list(STAGES)
        if self.run_tag:
            stages[stages.index('validation'):stages.index('validation')] = \
                list(_EXTRA_LANE_STAGES)
        return stages

    def _stage_command(self, name: str) -> list[str]:
        """The child command for a module-backed stage (config-owned paths)."""
        if name in _STAGE_MODULES:
            return ['-m', *_STAGE_MODULES[name]]
        if name == 'negative_supply':
            return ['-m', 'training.negative_supply', '--run-tag', self.run_tag]
        if name == 'graph_inputs':
            return ['-m', 'graph_tracks.setup', '--output', str(self.context.setup),
                    '--text-checkpoint', str(self.context.checkpoint),
                    '--defer-training-tensors']
        if name == 'full_bundle':
            F = self.context.files
            return ['-m', 'training.train', '--dataset', str(F['dataset_deduped']),
                    '--payload', 'full', '--prepare-bundle', str(self.context.bundle),
                    '--model', self.context.suite.text_model,
                    '--no-mask-effect', '--no-plot']
        if name == 'suite_inputs':
            return ['-m', 'model_tracks.package', '--config', str(self.context.config_path),
                    '--output', str(self.suite_archive)]
        raise KeyError(name)

    def _archive_stage_prefix(self, name: str) -> None:
        """Archive the superseded artifacts a stage is about to regenerate."""
        from core.common import RESULTS
        if name == 'negative_supply':
            lanes = Path(RESULTS) / self.context.prep.negative_supply_dir / self.run_tag
            with trace_step(f'prepare_all.{name}.archive'):
                self.archive(lanes, 'negative_supply')
        elif name == 'graph_inputs':
            with trace_step(f'prepare_all.{name}.archive'):
                self.archive(self.context.setup, 'track_setup')
        elif name == 'full_bundle':
            bundle = self.context.bundle
            with trace_step(f'prepare_all.{name}.archive'):
                self.archive(bundle, bundle.name)
                self.archive(bundle.with_suffix(bundle.suffix + '.json'), bundle.name + '.json')
        elif name == 'suite_inputs':
            with trace_step(f'prepare_all.{name}.archive'):
                self.archive(self.suite_archive, self.suite_archive.name)

    # --- stage specialists ------------------------------------------------

    def _stage_gate_census(self, name: str) -> None:
        """Measure the gate census inline (no child process)."""
        with trace_step('prepare_all.stage_refresh_gate_census', stage=name):
            self.manifest['gate_census'] = refresh_gate_census(
                self.context.files['gate_results'],
                self.run_dir / self.context.prep.gate_census_file)

    def _stage_discriminator(self, name: str) -> None:
        """Run + record the diagnostic verdict; gate mode decides enforcement."""
        from core.common import RESULTS
        arguments = [str(self.context.root /
                         _pipeline_layouts()['negative_supply_discriminator']),
                     str(Path(RESULTS) / self.context.prep.negative_supply_dir /
                         self.run_tag / 'pairs.csv'),
                     '--out', str(self.run_dir / self.context.prep.discriminator_file)]
        with trace_step(f'prepare_all.{name}.run_and_verdict'):
            result = self.run(name, arguments, check=False)
            verdict = json.loads((self.run_dir / self.context.prep.discriminator_file).read_text())
            self.manifest['discriminator'] = verdict
            if result.returncode and (self.context.lane.mode == 'lane' or
                                      verdict.get('verdict') != 'SEPARABLE'):
                raise subprocess.CalledProcessError(result.returncode, arguments)
            if result.returncode:
                _LOG.info('[prepare] diagnostic lane is SEPARABLE; active gate mode is unchanged')

    def _stage_full_bundle(self, name: str) -> None:
        """Train-side bundle build, then sync the text bundle when paths differ."""
        self._stage_module(name)
        if self.context.text_bundle != self.context.bundle:
            with trace_step(f'prepare_all.{name}.copy_text_bundle'):
                self._sync_text_bundle()

    def _sync_text_bundle(self) -> None:
        """Make the text bundle an independent copy of the full bundle."""
        bundle, text_bundle = self.context.bundle, self.context.text_bundle
        self.archive(text_bundle, 'text_bundle.pkl.gz')
        self.archive(text_bundle.with_suffix(text_bundle.suffix + '.json'),
                     'text_bundle.pkl.gz.json')
        text_bundle.parent.mkdir(parents=True, exist_ok=True)
        copy_bundle(bundle, text_bundle)
        from training.preparation_run import active_preparation
        active_preparation().alias_bundle(bundle, text_bundle)
        shutil.copy2(bundle.with_suffix(bundle.suffix + '.json'),
                     text_bundle.with_suffix(text_bundle.suffix + '.json'))

    def _stage_module(self, name: str) -> None:
        """One module-backed child stage under its trace section."""
        with trace_step(f'prepare_all.{name}.run'):
            self.run(name, self._stage_command(name))

    def _stage_verify_handoff(self, name: str) -> None:
        """The boundary: verify everything loads, then write the handoff report."""
        from training.handoff import verify_training_loads, write_handoff_report
        handoff_path = self.run_dir / self.context.prep.handoff_file
        with self._inline_stage_timing(name):
            with trace_step(f'prepare_all.{name}.verify_training_loads'):
                report = verify_training_loads(
                    root=self.context.root, suite=self.context.suite,
                    suite_config_path=self.context.config_path,
                    checkpoint=self.context.checkpoint, setup_dir=self.context.setup,
                    full_bundle=self.context.bundle, text_bundle=self.context.text_bundle,
                    suite_archive=self.suite_archive,
                    provenance=self.manifest['provenance'], smoke_dir=self.context.smoke,
                    smoke_original=self.smoke_before,
                    reusable_paths=self._handoff_inventory_paths())
        with trace_step(f'prepare_all.{name}.write_handoff_report'):
            write_handoff_report(report, handoff_path)
        self._record_handoff(report=report, handoff_path=handoff_path)

    def _record_handoff(self, *, report, handoff_path) -> None:
        """Fold the handoff report's summary into the run manifest."""
        self.manifest['handoff'] = {'path': str(handoff_path), 'status': report.status,
                                    'total_seconds': report.total_seconds}
        self.manifest['outputs'] = report.final_inventory
        self.manifest['bundle'] = report.bundle_header
        self.manifest['suite_package'] = report.suite_package
        self.manifest['smoke_unchanged_verified'] = True

    @contextmanager
    def _inline_stage_timing(self, name: str):
        """Apply the stage-timing contract to inline (non-child) stage work.

        The boundary runs in-process, not through run(): the same env contract
        keeps its Timing sections in the stage JSON and the offender report.
        """
        os.environ.update(
            ER_TIMING_OUT=str(self.run_dir / (name + self.context.prep.stage_timing_suffix)),
            ER_TIMING_LOG=str(self.run_dir / self.context.prep.timings_log))
        try:
            yield
        finally:
            os.environ.pop('ER_TIMING_OUT', None)
            os.environ.pop('ER_TIMING_LOG', None)

    # --- dispatch ---------------------------------------------------------

    def _execute_stage(self, name: str) -> None:
        """Dispatch exactly one stage to its specialist or module runner."""
        if name == 'gate_census':
            self._stage_gate_census(name)
        elif name == 'negative_supply':
            self._archive_stage_prefix(name)
            self._stage_module(name)
        elif name == 'discriminator':
            self._stage_discriminator(name)
        elif name == 'graph_inputs':
            self._archive_stage_prefix(name)
            self._stage_module(name)
        elif name == 'full_bundle':
            self._archive_stage_prefix(name)
            self._stage_full_bundle(name)
        elif name == 'suite_inputs':
            self._archive_stage_prefix(name)
            self._stage_module(name)
        elif name == 'verify_handoff':
            self._stage_verify_handoff(name)
        else:
            self._stage_module(name)

    # --- inventory + invalidation ----------------------------------------

    def _handoff_inventory_paths(self) -> list[Path]:
        """Every artifact the handoff boundary must prove loadable."""
        from core.common import RESULTS
        F = self.context.files
        inventory_paths = [Path(F[key]) for key in self.context.reusable_keys]
        inventory_paths += [path for path in self.context.setup.rglob('*') if path.is_file()]
        bundle = self.context.bundle
        inventory_paths += [bundle, bundle.with_suffix(bundle.suffix + '.json'),
                            self.context.text_bundle,
                            self.context.text_bundle.with_suffix(self.context.text_bundle.suffix + '.json'),
                            self.suite_archive]
        if self.run_tag:
            lanes = Path(RESULTS) / self.context.prep.negative_supply_dir / self.run_tag
            inventory_paths += [lanes / filename for filename in ('pairs.csv', 'manifest.json')]
            inventory_paths.append(self.run_dir / self.context.prep.discriminator_file)
        return inventory_paths

    def _reusable_candidate_paths(self, name: str) -> list[Path]:
        """The resume inventory a completed graph/bundle stage owns."""
        from core.common import RESULTS
        candidates = [Path(self.context.files[key]) for key in self.context.reusable_keys]
        candidates += [path for path in self.context.setup.rglob('*') if path.is_file()]
        if name == 'full_bundle':
            bundle = self.context.bundle
            candidates += [bundle, bundle.with_suffix(bundle.suffix + '.json'),
                           self.context.text_bundle,
                           self.context.text_bundle.with_suffix(self.context.text_bundle.suffix + '.json')]
        lanes = Path(RESULTS) / self.context.prep.negative_supply_dir / self.run_tag
        candidates += [lanes / filename for filename in ('pairs.csv', 'manifest.json')]
        candidates.append(self.run_dir / self.context.prep.discriminator_file)
        return candidates

    def _collect_reusable_outputs(self, name: str) -> None:
        """Record a post-stage inventory for the next resume's reuse."""
        with trace_step(f'prepare_all.{name}.collect_reusable_outputs'):
            self.manifest['reusable_outputs'] = file_inventory(self._reusable_candidate_paths(name))

    def _invalidate_caches(self, name: str) -> None:
        """Drop the expired in-memory artifacts a producer stage just superseded."""
        from core.common import F
        if name == 'full_bundle':
            from training.preparation_run import active_preparation
            with trace_step(f'prepare_all.{name}.invalidate'):
                active_preparation()._base.clear()
                active_preparation()._datasets.clear()
        elif name == 'dedupe':
            from training.preparation_run import active_preparation
            with trace_step(f'prepare_all.{name}.invalidate'):
                active_preparation().invalidate(Path(F['dataset_deduped']))
        elif name == 'number_reference':
            with trace_step(f'prepare_all.{name}.invalidate'):
                import pipeline
                pipeline._VERDICTS_CACHE = None
                pipeline._VERDICTS_LOADED = False

    # --- stage bookkeeping ------------------------------------------------

    def _begin_stage(self, name: str) -> float:
        """Mark one stage running in the manifest, timings log + live reporting."""
        stage_started = time.monotonic()
        self.manifest.setdefault('stage_metrics', {})[name] = {
            'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
            'detail_path': str(self.run_dir / (name + self.context.prep.stage_timing_suffix)),
        }
        from core.timing import emit_timing
        emit_timing(f'[timing] prepare.{name} state=started',
                    path=self.run_dir / self.context.prep.timings_log)
        return stage_started

    def _complete_stage(self, name: str, stage_started: float) -> None:
        """Record one stage's healthy completion across manifest + timings."""
        from core.timing import emit_timing
        elapsed = round(time.monotonic() - stage_started, 3)
        self.manifest.setdefault('stage_seconds', {})[name] = elapsed
        self.manifest['stage_metrics'][name].update(
            status='complete', seconds=elapsed,
            finished_at=datetime.now(timezone.utc).isoformat())
        self.manifest['stages'].append(name)
        emit_timing(f'[timing] prepare.{name} state=completed elapsed_seconds={elapsed:.3f}',
                    path=self.run_dir / self.context.prep.timings_log)

    def _fail_stage(self, name: str, stage_started: float) -> None:
        """Record one stage's failure across manifest + timings (no raise)."""
        from core.timing import emit_timing
        elapsed = round(time.monotonic() - stage_started, 3)
        self.manifest.setdefault('stage_seconds', {})[name] = elapsed
        self.manifest['stage_metrics'][name].update(
            status='failed', seconds=elapsed,
            finished_at=datetime.now(timezone.utc).isoformat())
        emit_timing(f'[timing] prepare.{name} state=failed elapsed_seconds={elapsed:.3f}',
                    path=self.run_dir / self.context.prep.timings_log)

    # --- orchestration ----------------------------------------------------

    def _run_stage_loop(self) -> None:
        """Execute every remaining stage once, each timed and published."""
        stages = self.plan_stages()
        first = 'negative_supply' if self.resume_from == 'validation' else self.resume_from
        remaining = stages[stages.index(first):]
        bar = _LOG.progress(remaining, desc='prepare_all', unit='stage')
        name, stage_started = None, None
        try:
            for name in bar:
                bar.set_postfix_str(name)
                stage_started = self._begin_stage(name)
                self.publish()
                self._execute_stage(name)
                if name in {'graph_inputs', 'full_bundle'}:
                    self._collect_reusable_outputs(name)
                self._invalidate_caches(name)
                self._complete_stage(name, stage_started)
                self.publish()
        except BaseException:
            if stage_started is not None and name in self.manifest.get('stage_metrics', {}):
                self._failed_stage = name
                self._fail_stage(name, stage_started)
            raise
        finally:
            bar.close()

    @timed
    def execute(self) -> Path:
        """Run the whole preparation under the single-flight lock."""
        from core.common import RESULTS
        prep = self.context.prep
        self.smoke_before = _hash_smoke_baseline(self.context.smoke)
        provenance = preparation_provenance(self.context.root, self.context.config_path,
                                            self.context.checkpoint)
        self.manifest = _initial_manifest(
            root=self.context.root, resume_from=self.resume_from, run_dir=self.run_dir,
            config_path=self.context.config_path, lane=self.context.lane,
            run_tag=self.run_tag, shared_base_payload=self.shared_base_payload,
            provenance=provenance, smoke_before=self.smoke_before)
        with _acquire_prepare_lock(Path(RESULTS) / prep.lock_file):
            _verify_resume_prerequisites(self.resume_from)
            self.replay_resume_state()
            self.publish()
            try:
                self._run_stage_loop()
            except BaseException as error:
                self.manifest.update(status='failed', error=str(error))
                if getattr(self, '_failed_stage', None):
                    self.manifest['failed_stage'] = self._failed_stage
                self.publish()
                raise
            self.manifest['status'] = 'complete'
            self.publish()
            _LOG.info(f'[prepare] complete -> {self.manifest_path}')
            return self.manifest_path


def _resolve_run_tag(lane, requested: str | None) -> str | None:
    """Resolve the lane-locked negative-supply tag before the run dir exists."""
    if lane.mode == 'lane':
        if requested and requested != lane.pairs_run_tag:
            raise ValueError('negative-supply run tag differs from the configured training lane')
        return lane.pairs_run_tag
    if requested and not re.fullmatch(_RUN_TAG_PATTERN, requested):
        raise ValueError('negative-supply run tag must contain only letters, digits, underscores, hyphens')
    return requested


def _prepare_all(*, run_dir=None, resume_from='dedupe', tracks_config=None,
                 negative_supply_run_tag=None):
    """Orchestrate one preparation through the PrepareRun owner."""
    return PrepareRun(run_dir=run_dir, resume_from=resume_from,
                      tracks_config=tracks_config,
                      negative_supply_run_tag=negative_supply_run_tag).execute()


@timed
def prepare_all(**kwargs):
    """Execute the CSV-to-training-input lifecycle as one owned Python run."""
    from training.preparation_run import TrainingPreparation
    return TrainingPreparation(**kwargs).execute()


@timed
def main():
    RunLogger.configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--tracks-config', type=Path,
                        help='three-track configuration (default: config/model_tracks.yaml)')
    parser.add_argument('--negative-supply-run-tag',
                        help='generate and audit the real-first lane; gate mode keeps this diagnostic-only')
    parser.add_argument('--resume-from', choices=['dedupe', 'validation', 'full_bundle', 'suite_inputs'], default='dedupe',
                        help='validation verifies CSV manifests; suite_inputs reuses the completed graph/text bundle')
    args = parser.parse_args()
    prepare_all(run_dir=args.run_dir, resume_from=args.resume_from,
                tracks_config=args.tracks_config, negative_supply_run_tag=args.negative_supply_run_tag)


if __name__ == '__main__':
    main()
