"""Rebuild every full-training input without training or changing smoke inputs.

Run from ER: PYTHONPATH=src .venv/bin/python -m training.prepare_all
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import subprocess
import sys
from typing import Literal
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
import yaml
from core.portable_archive import Digest
from core.schemas import PREPARATION_REUSABLE_KEYS
from training.prepare_all_trace import send, timed, trace_step

STAGES = ('dedupe', 'cross_country_pairs', 'number_reference', 'verify_reference',
        'canonical_and_gates', 'gate_census', 'labeled_pairs', 'validation',
          'graph_inputs', 'full_bundle', 'suite_inputs', 'verify_handoff')


REUSABLE_KEYS = PREPARATION_REUSABLE_KEYS


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
    negative_supply_run_tag: str = Field(pattern=r'^[A-Za-z0-9_-]+$')
    provenance: dict[str, Digest]
    smoke_original: dict[str, Digest]
    reusable_outputs: dict[str, PreparedFile] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_stages(self):
        if len(self.stages) != len(set(self.stages)) or set(self.stages) - set(STAGES) - {'negative_supply', 'discriminator'}:
            raise ValueError('Preparation stages must be unique known stages')
        if self.status == 'complete' and 'verify_handoff' not in self.stages:
            raise ValueError('Complete preparation requires verified handoff')
        return self


@timed
def preparation_provenance(root: Path, suite_config: Path, checkpoint: str | Path) -> dict[str, str]:
    """Pin source, owning configs, raw input and baseline checkpoint content."""
    from core.common import CONFIG_PATH, TRAINING_CONFIG_PATH, VOCABULARY_CONFIG_PATH, DATA_PATH, TRAIN_ROOT, artifact
    from graph_tracks.text_cache import checkpoint_hash
    paths = set((root / 'src').rglob('*.py')) | set((root / 'scripts').rglob('*.py'))
    paths.update(path for path in (root / 'config').rglob('*')
                 if path.suffix in {'.yaml', '.yml', '.json'} and path.is_file())
    paths.update([Path(CONFIG_PATH), Path(TRAINING_CONFIG_PATH),
                  Path(VOCABULARY_CONFIG_PATH), Path(suite_config), Path(DATA_PATH)])
    # Measured-evidence inputs the pipeline consumes fail-loud: they are
    # tracked build inputs, so any regeneration must regenerate everything.
    paths.update([TRAIN_ROOT / 'artifacts/evidence/attribute_universe_census.json',
                  artifact('semantic_family_registry')])
    identity = {}
    for path in sorted(paths):
        if path.resolve() == Path(TRAINING_CONFIG_PATH).resolve():
            # This run measures the census itself; it is output, not a setting.
            config = yaml.safe_load(path.read_text())
            config['rand_matching'].pop('gate_census_pin', None)
            identity[str(path.resolve())] = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()).hexdigest()
        else:
            identity[str(path.resolve())] = sha256(path)
    identity['text_checkpoint'] = checkpoint_hash(Path(checkpoint), use_memo=False)
    return identity


@timed
def verify_reusable_outputs(entries: dict[str, PreparedFile]) -> None:
    if not entries:
        raise ValueError('Resume has no verified prepared outputs; regenerate inputs')
    for path, entry in entries.items():
        if Path(path).stat().st_size != entry.bytes or sha256(path) != entry.sha256:
            raise ValueError(f'Stale prepared resume input: {path}')


@timed
def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@timed
def file_inventory(paths: Sequence[Path]) -> dict[str, dict[str, str | int]]:
    """Hash each resolved artifact once per snapshot; never cache across stages."""
    return {str(path): PreparedFile(sha256=sha256(path), bytes=path.stat().st_size).model_dump()
            for path in dict.fromkeys(path.resolve() for path in paths)}


@timed
def copy_bundle(source: Path, destination: Path) -> None:
    """Use filesystem copy-on-write when available, keeping independent files.

    Never hard-link mutable bundle paths: a later write must not corrupt its peer.
    Unsupported filesystems use shutil's platform-accelerated copy instead.
    """
    try:
        with trace_step('copy_bundle.ioctl_clone'):
            with source.open('rb') as src, destination.open('wb') as dst:
                fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # Linux FICLONE
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EOPNOTSUPP, errno.ENOTTY,
                              errno.EINVAL, errno.ENOSYS}:
            raise
        with trace_step('copy_bundle.shutil_fallback'):
            shutil.copy2(source, destination)
    else:
        shutil.copystat(source, destination)


@timed
def verify_stage_manifest(path: str | Path, *, required_inputs: Sequence[str | Path] = ()) -> None:
    from core.manifest import source_tree_sha256
    from core.schemas import StageManifest
    from core.tracing import trace_path
    try:
        with trace_step('verify_stage_manifest.load'):
            manifest = StageManifest.model_validate_json(Path(path).read_text())
    except ValueError as exc:
        exc.add_note(f'Preparation prerequisite manifest: {path}')
        raise
    if manifest.status != 'complete':
        raise ValueError(f'Incomplete prerequisite: {path}')
    if required_inputs and manifest.environment.get('source_sha256') != source_tree_sha256():
        raise ValueError(f'Prerequisite source changed or was not recorded: {path}; regenerate inputs')
    recorded_inputs = {Path(entry.path).resolve() for entry in manifest.inputs}
    if set(Path(item).resolve() for item in required_inputs) - recorded_inputs:
        raise ValueError(f'Prerequisite lacks current config/reference provenance: {path}; regenerate inputs')
    with trace_step('verify_stage_manifest.stale_hash_check',
                    outputs=len(manifest.inputs) + len(manifest.outputs)):
        for entry in manifest.inputs + manifest.outputs:
            # The consolidated trace is append-only across stages, not frozen input.
            if Path(entry.path).resolve() == trace_path().resolve():
                continue
            if sha256(entry.path) != entry.sha256:
                raise ValueError(f'Stale prerequisite: {entry.path}')


@timed
def refresh_gate_census(gate_csv: Path, config_path: Path, report_path: Path) -> dict[str, int]:
    import pandas as pd
    from core.manifest import atomic_write_json, atomic_write_text
    from core.schemas import RandMatchingSpec
    frame = pd.read_csv(gate_csv, usecols=['gtin1', 'gtin2', 'gate_decision'],
                        dtype=str, keep_default_na=False)
    if frame[['gtin1', 'gtin2']].apply(lambda column: column.str.strip().eq('')).any().any():
        raise ValueError('Empty gate pair endpoint')
    if frame.gtin1.eq(frame.gtin2).any():
        raise ValueError('Gate census contains self-pairs')
    left, right = frame.gtin1, frame.gtin2
    ordered = left.le(right)
    identities = pd.DataFrame({'left': left.where(ordered, right),
                               'right': right.where(ordered, left)})
    if identities.duplicated().any():
        raise ValueError('Duplicate candidate pairs')
    if set(frame.gate_decision) - {'hard_no', 'proceed', 'fallback'}:
        raise ValueError('Unknown gate decisions')
    counts = {'total_pairs': len(frame), **{
        key: int(frame.gate_decision.eq(key).sum())
        for key in ('hard_no', 'proceed', 'fallback')}}
    counts = RandMatchingSpec.GateCensusPinSpec.model_validate(counts).model_dump()
    pattern = r'(  gate_census_pin:\n)(    total_pairs: \d+\n    hard_no: \d+\n    proceed: \d+\n    fallback: \d+\n)'
    text, n = re.subn(pattern, lambda match: match.group(1) + ''.join(
        f'    {key}: {value}\n' for key, value in counts.items()), Path(config_path).read_text())
    if n != 1:
        raise ValueError('Expected exactly one configured gate census')
    atomic_write_json(counts, report_path)
    atomic_write_text(config_path, text)
    return counts


def _prepare_all(*, run_dir=None, resume_from='dedupe', tracks_config=None,
                negative_supply_run_tag=None):
    from core.timing import emit_timing
    requested_negative_supply_run_tag = negative_supply_run_tag
    from core.common import F, RESULTS, TRAIN_ROOT, CONFIG_PATH, TRAINING_CONFIG_PATH, VOCABULARY_CONFIG_PATH, resolve_model, training_cfg
    from model_tracks.config import load_config as load_suite
    with trace_step('prepare_all.load_configs'):
        root = Path(TRAIN_ROOT)
        config_path = Path(tracks_config or root / 'config/model_tracks.yaml').resolve()
        suite = load_suite(config_path)
        # Read bundle settings afresh for long-lived callers.
        training = type(training_cfg()).model_validate(yaml.safe_load(Path(TRAINING_CONFIG_PATH).read_text()))
        prep = training.preparation
        # The declared run contract owns the resumable key set; the module-level
        # constant is the defaulted public API (tests pin it), the run reads cfg.
        reusable_keys = tuple(prep.reusable_keys)
        if suite.text_model != training.training.base_model:
            raise ValueError('suite text_model and training.base_model must agree before preparation')
        if suite.epochs > training.training.epochs:
            raise ValueError('suite epochs exceed locally prepared training epochs; update training.yaml first')
    with trace_step('prepare_all.resolve_paths'):
        lane = training.negative_supply
        if lane.mode == 'lane':
            if negative_supply_run_tag and negative_supply_run_tag != lane.pairs_run_tag:
                raise ValueError('negative-supply run tag differs from the configured training lane')
            negative_supply_run_tag = lane.pairs_run_tag
        if negative_supply_run_tag and not re.fullmatch(r'[A-Za-z0-9_-]+', negative_supply_run_tag):
            raise ValueError('negative-supply run tag must contain only letters, digits, underscores, hyphens')
        run_dir = Path(run_dir or RESULTS / prep.run_dir_base / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')).resolve()
        negative_supply_run_tag = negative_supply_run_tag or ('prep_' + run_dir.name)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', negative_supply_run_tag):
            raise ValueError('preparation directory name requires an explicit valid --negative-supply-run-tag')
        run_dir.mkdir(parents=True, exist_ok=True)
        os.environ['ER_TIMING_LOG'] = str(run_dir / prep.timings_log)
    lock_path = RESULTS / prep.lock_file
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        with trace_step('prepare_all.lock_acquire'):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another training preparation is already running') from None
        with trace_step('prepare_all.resume_validation'):
            if resume_from == 'validation':
                for name in ('data_prep', 'labeled_pairs'):
                    verify_stage_manifest(RESULTS / 'manifests' / (name + '.json'),
                        required_inputs=([CONFIG_PATH, VOCABULARY_CONFIG_PATH, F['number_reference']]
                                         if name == 'data_prep' else [TRAINING_CONFIG_PATH]))
        checkpoint = resolve_model(suite.text_model)
        setup = (root / suite.setup_dir).resolve()
        text_bundle = (root / suite.text_bundle).resolve()
        if len(training.colab.full_prepared_bundles) != 1:
            raise ValueError('prepare_all requires exactly one configured full baseline bundle')
        bundle = (root / training.colab.full_prepared_bundles[0]).resolve()
        suite_archive = run_dir / f'{prep.suite_archive_name}.{suite.input_archive_format}'
        smoke = root / prep.smoke_dir
        with trace_step('prepare_all.smoke_before_hash'):
            smoke_before = {str(path): sha256(path) for path in smoke.rglob('*') if path.is_file()}
        env = os.environ.copy()
        env['PYTHONPATH'] = str(root / 'src') + os.pathsep + str(root)
        env['EUROMONITOR_SHARED_BASE_DATA'] = str(run_dir / (
            'shared_base_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.pkl'))
        env.pop('WANDB_API_KEY', None)
        # Preparation mutates inputs; inherited worker attestations are invalid.
        env['ER_DATA_GATE_ENFORCE'] = '1'
        completed = []
        with trace_step('prepare_all.manifest_init'):
            manifest = {'status': 'running', 'resume_from': resume_from,
                        'training_started': False, 'smoke_updated': False,
                        'stages': completed, 'run_dir': str(run_dir),
                        'shared_base_payload': env['EUROMONITOR_SHARED_BASE_DATA'],
                        'tracks_config': str(config_path),
                        'negative_supply_mode': lane.mode,
                        'negative_supply_run_tag': negative_supply_run_tag,
                        'provenance': preparation_provenance(root, config_path, checkpoint),
                        'smoke_original': smoke_before, 'reusable_outputs': {},
                        'hybrid_embeddings': 'GPU pending: frozen baseline forward before hybrid training'}
        manifest_path = run_dir / prep.manifest_file
        with trace_step('prepare_all.resume_replay'):
            if resume_from in {'full_bundle', 'suite_inputs'}:
                previous_state = PreparationState.model_validate_json(manifest_path.read_text())
                previous = previous_state.model_dump(mode='json')
                if previous_state.run_dir != str(run_dir):
                    raise ValueError('Preparation manifest belongs to another run directory')
                if requested_negative_supply_run_tag and requested_negative_supply_run_tag != previous_state.negative_supply_run_tag:
                    raise ValueError('Resume negative-supply run tag differs from the saved run')
                negative_supply_run_tag = previous_state.negative_supply_run_tag
                if previous_state.provenance != manifest['provenance']:
                    raise ValueError('Preparation source/config/raw input/checkpoint changed; regenerate inputs')
                if previous_state.smoke_original != smoke_before:
                    raise ValueError('Smoke files changed since this preparation began')
                verify_reusable_outputs(previous_state.reusable_outputs)
                prerequisite = 'graph_inputs' if resume_from == 'full_bundle' else 'full_bundle'
                if (prerequisite not in previous.get('stages', []) or
                        previous.get('tracks_config') != str(config_path)):
                    raise ValueError(f'{resume_from} resume requires this run\'s completed {prerequisite}')
                remaining = set(STAGES[STAGES.index(resume_from):])
                completed.extend(stage for stage in previous['stages']
                                 if stage not in remaining)
                manifest.update(previous, status='running', resume_from=resume_from,
                                stages=completed, shared_base_payload=env['EUROMONITOR_SHARED_BASE_DATA'])
                manifest.pop('failed_stage', None)
                manifest.pop('error', None)
        @timed
        def publish():
            temporary = manifest_path.with_suffix('.tmp')
            state = PreparationState.model_validate(manifest)
            temporary.write_text(state.model_dump_json(indent=2) + '\n')
            temporary.replace(manifest_path)
            timing_path = run_dir / prep.timings_file
            timing_temporary = timing_path.with_suffix('.tmp')
            timing_temporary.write_text(json.dumps({
                'status': manifest['status'],
                'stages': manifest.get('stage_metrics', {}),
                'total_stage_seconds': round(sum(manifest.get('stage_seconds', {}).values()), 3),
            }, indent=2) + '\n')
            timing_temporary.replace(timing_path)
            # One self-rewriting worst-offender report for the whole run.
            from core.timing import collect_timing_entries, write_offender_report
            write_offender_report(run_dir / prep.offender_report,
                                  collect_timing_entries(run_dir, manifest.get('stage_seconds', {})))
        @timed
        def run(name, arguments, *, check=True):
            print(f'[prepare] {name} -> {run_dir / (name + prep.stage_log_suffix)}', flush=True)
            env['ER_TIMING_OUT'] = str(run_dir / (name + prep.stage_timing_suffix))
            env['ER_TIMING_LOG'] = str(run_dir / prep.timings_log)
            with (run_dir / (name + prep.stage_log_suffix)).open('w') as log:
                from training.preparation_run import active_preparation
                result = active_preparation().run_stage(arguments, root=root, env=env, log=log)
            manifest['stage_metrics'][name]['returncode'] = result.returncode
            if check:
                result.check_returncode()
            return result
        @timed
        def archive(path, name):
            from training.preparation_run import active_preparation
            active_preparation().invalidate(path)
            if path.exists():
                destination = run_dir / prep.archive_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    destination = destination.with_name(destination.name + '.' +
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
                shutil.move(str(path), str(destination))
        publish()
        try:
            stages = list(STAGES)
            if negative_supply_run_tag:
                stages[stages.index('validation'):stages.index('validation')] = ['negative_supply', 'discriminator']
            first = 'negative_supply' if resume_from == 'validation' else resume_from
            from tqdm import tqdm
            bar = tqdm(stages[stages.index(first):], desc='prepare_all',
                       unit='stage', dynamic_ncols=True)
            for name in bar:
                stage_started = time.monotonic()
                manifest.setdefault('stage_metrics', {})[name] = {
                    'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
                    'detail_path': str(run_dir / (name + prep.stage_timing_suffix)),
                }
                emit_timing(f'[timing] prepare.{name} state=started', path=run_dir / prep.timings_log)
                publish()
                if name == 'dedupe':
                    with trace_step(f'prepare_all.{name}.run'):
                        run(name, ['-m', 'training.dedupe'])
                elif name == 'cross_country_pairs':
                    with trace_step(f'prepare_all.{name}.run'):
                        run(name, ['-m', 'training.build_second04_pairs'])
                elif name == 'number_reference':
                    with trace_step(f'prepare_all.{name}.run'):
                        run(name, ['-m', 'training.build_reference'])
                elif name == 'verify_reference':
                    with trace_step(f'prepare_all.{name}.run'):
                        run(name, ['-m', 'training.build_reference', '--verify'])
                elif name == 'canonical_and_gates':
                    with trace_step(f'prepare_all.{name}.run'):
                        run(name, ['-m', 'training.data_prep'])
                elif name == 'gate_census':
                    with trace_step('prepare_all.stage_refresh_gate_census', stage=name):
                        manifest['gate_census'] = refresh_gate_census(
                            F['gate_results'], TRAINING_CONFIG_PATH, run_dir / prep.gate_census_file)
                        from core.common import refresh_training_config
                        refresh_training_config()
                elif name == 'labeled_pairs':
                    with trace_step(f'prepare_all.{name}.run'):
                        run(name, ['-m', 'training.labeled_pairs'])
                elif name == 'negative_supply':
                    with trace_step(f'prepare_all.{name}.archive'):
                        archive(RESULTS / prep.negative_supply_dir / negative_supply_run_tag, 'negative_supply')
                    with trace_step(f'prepare_all.{name}.run'):
                        run(name, ['-m', 'training.negative_supply', '--run-tag', negative_supply_run_tag])
                elif name == 'discriminator':
                    with trace_step(f'prepare_all.{name}.run_and_verdict'):
                        arguments = [str(root / 'scripts/negative_supply_discriminator.py'),
                                     str(RESULTS / prep.negative_supply_dir / negative_supply_run_tag / 'pairs.csv'),
                                     '--out', str(run_dir / prep.discriminator_file)]
                        result = run(name, arguments, check=False)
                        verdict = json.loads((run_dir / prep.discriminator_file).read_text())
                        manifest['discriminator'] = verdict
                        if result.returncode and (lane.mode == 'lane' or verdict.get('verdict') != 'SEPARABLE'):
                            raise subprocess.CalledProcessError(result.returncode, arguments)
                        if result.returncode:
                            print('[prepare] diagnostic lane is SEPARABLE; active gate mode is unchanged', flush=True)
                elif name == 'validation':
                    with trace_step(f'prepare_all.{name}.run'):
                        run(name, ['-m', 'training.build_final_validation'])
                elif name == 'graph_inputs':
                    with trace_step(f'prepare_all.{name}.archive'):
                        archive(setup, 'track_setup')
                    with trace_step(f'prepare_all.{name}.run'):
                        run(name, ['-m', 'graph_tracks.setup', '--output', str(setup),
                                   '--text-checkpoint', str(checkpoint), '--defer-training-tensors'])
                elif name == 'full_bundle':
                    with trace_step(f'prepare_all.{name}.archive'):
                        archive(bundle, bundle.name)
                        archive(bundle.with_suffix(bundle.suffix + '.json'), bundle.name + '.json')
                    with trace_step(f'prepare_all.{name}.run'):
                        run(name, ['-m', 'training.train', '--dataset', str(F['dataset_deduped']),
                                   '--payload', 'full', '--prepare-bundle', str(bundle),
                                   '--model', suite.text_model, '--no-mask-effect', '--no-plot'])
                    if text_bundle != bundle:
                        with trace_step(f'prepare_all.{name}.copy_text_bundle'):
                            archive(text_bundle, 'text_bundle.pkl.gz')
                            archive(text_bundle.with_suffix(text_bundle.suffix + '.json'), 'text_bundle.pkl.gz.json')
                            text_bundle.parent.mkdir(parents=True, exist_ok=True)
                            copy_bundle(bundle, text_bundle)
                            from training.preparation_run import active_preparation
                            active_preparation().alias_bundle(bundle, text_bundle)
                            shutil.copy2(bundle.with_suffix(bundle.suffix + '.json'),
                                         text_bundle.with_suffix(text_bundle.suffix + '.json'))
                elif name == 'suite_inputs':
                    with trace_step(f'prepare_all.{name}.archive'):
                        archive(suite_archive, suite_archive.name)
                    with trace_step(f'prepare_all.{name}.run'):
                        run(name, ['-m', 'model_tracks.package', '--config', str(config_path),
                                   '--output', str(suite_archive)])
                else:
                    from training.handoff import verify_training_loads, write_handoff_report
                    inventory_paths = [Path(F[key]) for key in reusable_keys]
                    inventory_paths += [p for p in setup.rglob('*') if p.is_file()]
                    inventory_paths += [bundle, bundle.with_suffix(bundle.suffix + '.json'),
                                        text_bundle, text_bundle.with_suffix(text_bundle.suffix + '.json'),
                                        suite_archive]
                    if negative_supply_run_tag:
                        inventory_paths += [RESULTS / prep.negative_supply_dir / negative_supply_run_tag / filename
                                            for filename in ('pairs.csv', 'manifest.json')]
                        inventory_paths.append(run_dir / prep.discriminator_file)
                    handoff_path = run_dir / prep.handoff_file
                    # The boundary runs inline, not through run(): apply the
                    # same stage-timing environment so its Timing sections
                    # land in the stage JSON and the offender report.
                    os.environ.update(ER_TIMING_OUT=str(run_dir / (name + prep.stage_timing_suffix)),
                                      ER_TIMING_LOG=str(run_dir / prep.timings_log))
                    try:
                        with trace_step(f'prepare_all.{name}.verify_training_loads'):
                            report = verify_training_loads(
                                root=root, suite=suite, suite_config_path=config_path,
                                checkpoint=checkpoint, setup_dir=setup, full_bundle=bundle,
                                text_bundle=text_bundle, suite_archive=suite_archive,
                                provenance=manifest['provenance'], smoke_dir=smoke,
                                smoke_original=smoke_before, reusable_paths=inventory_paths)
                    finally:
                        os.environ.pop('ER_TIMING_OUT', None)
                        os.environ.pop('ER_TIMING_LOG', None)
                    with trace_step(f'prepare_all.{name}.write_handoff_report'):
                        write_handoff_report(report, handoff_path)
                    manifest['handoff'] = {'path': str(handoff_path), 'status': report.status,
                                           'total_seconds': report.total_seconds}
                    manifest['outputs'] = report.final_inventory
                    manifest['bundle'] = report.bundle_header
                    manifest['suite_package'] = report.suite_package
                    manifest['smoke_unchanged_verified'] = True
                if name in {'graph_inputs', 'full_bundle'}:
                    with trace_step(f'prepare_all.{name}.collect_reusable_outputs'):
                        reusable = [Path(F[key]) for key in reusable_keys]
                        reusable += [path for path in setup.rglob('*') if path.is_file()]
                        if name == 'full_bundle':
                            reusable += [bundle, bundle.with_suffix(bundle.suffix + '.json'), text_bundle,
                                         text_bundle.with_suffix(text_bundle.suffix + '.json')]
                        reusable += [RESULTS / prep.negative_supply_dir / negative_supply_run_tag / filename
                                     for filename in ('pairs.csv', 'manifest.json')]
                        reusable.append(run_dir / prep.discriminator_file)
                        manifest['reusable_outputs'] = file_inventory(reusable)
                if name == 'full_bundle':
                    with trace_step(f'prepare_all.{name}.invalidate'):
                        from training.preparation_run import active_preparation
                        active_preparation()._base.clear()
                        active_preparation()._datasets.clear()
                if name == 'dedupe':
                    with trace_step(f'prepare_all.{name}.invalidate'):
                        from training.preparation_run import active_preparation
                        active_preparation().invalidate(Path(F['dataset_deduped']))
                if name == 'number_reference':
                    with trace_step(f'prepare_all.{name}.invalidate'):
                        import pipeline
                        pipeline._VERDICTS_CACHE = None
                        pipeline._VERDICTS_LOADED = False
                completed.append(name)
                elapsed = round(time.monotonic() - stage_started, 3)
                manifest.setdefault('stage_seconds', {})[name] = elapsed
                manifest['stage_metrics'][name].update(status='complete', seconds=elapsed,
                    finished_at=datetime.now(timezone.utc).isoformat())
                emit_timing(f'[timing] prepare.{name} state=completed elapsed_seconds={elapsed:.3f}', path=run_dir / prep.timings_log)
                publish()
            bar.close()
            manifest['status'] = 'complete'
            publish()
            print(f'[prepare] complete -> {manifest_path}', flush=True)
            return manifest_path
        except BaseException as error:
            if 'bar' in locals():
                bar.close()
            if name in manifest.get('stage_metrics', {}):
                elapsed = round(time.monotonic() - stage_started, 3)
                manifest.setdefault('stage_seconds', {})[name] = elapsed
                manifest['stage_metrics'][name].update(status='failed', seconds=elapsed,
                    finished_at=datetime.now(timezone.utc).isoformat())
                emit_timing(f'[timing] prepare.{name} state=failed elapsed_seconds={elapsed:.3f}', path=run_dir / prep.timings_log)
            manifest.update(status='failed', failed_stage=name, error=str(error))
            publish()
            raise


@timed
def prepare_all(**kwargs):
    """Execute the CSV-to-training-input lifecycle as one owned Python run."""
    from training.preparation_run import TrainingPreparation
    return TrainingPreparation(**kwargs).execute()


@timed
def main():
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
