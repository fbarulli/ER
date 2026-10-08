"""Run the frozen CPU finalize step from a packaged runtime snapshot.

Lane change (owner ruling): the operator box is no longer a finalize surface.
Finalize is a remote CPU lane job — a Kaggle CPU kernel or a Colab CPU stage
runs :mod:`model_tracks.bundle_steps` from a sparse checkout. This module is the
in-process form of that job: it materializes the packaged source/config
inventory into a temporary sparse checkout and runs the ONE finalize entry
(:func:`model_tracks.local_complete.complete`) against the two verified bundles.

It stays a thin wrapper: no post-processing logic lives here, and the archives
are verified once each through :meth:`core.bundle.Bundle.load`, whose streaming
pass also yields the whole-file digests the receipt records.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from model_tracks.package import package_member
import subprocess
import sys
import tempfile

from pydantic import BaseModel, ConfigDict, Field

from core.portable_archive import Digest
from core.archive_reader import archive_sidecar
from core.tracing import flush_stage_trace, stage_trace

#: The stage name this module owns in the ONE consolidated pipeline trace.
STAGE = "snapshot_completion"

#: The module's trace writer: the shared shim's slot (``None`` until first use;
#: see :func:`core.tracing.stage_trace`), so importing this module never touches
#: the trace layout. This wrapper owns no post-processing logic, so it records
#: the runtime inventory it materializes, which working-tree files differ, and
#: the receipt it returns.
_TRACE = None


def trace():
    """The ONE writer for the ``snapshot_completion`` stage of the current run."""
    global _TRACE
    _TRACE = stage_trace(STAGE, _TRACE)
    return _TRACE


def flush_trace():
    """Commit this process's snapshot-completion rows once; a no-op while empty."""
    return flush_stage_trace(_TRACE)


def _spec():
    from core.bundle import bundle_spec
    return bundle_spec()


class SnapshotCompletionReceipt(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    run_tag: str = Field(pattern=r'^[A-Za-z0-9_-]+$')
    input_archive_sha256: Digest
    training_archive_sha256: Digest
    final_archive_sha256: Digest
    source_inventory: dict[str, Digest]
    working_tree_mismatches: list[str]


def complete(training_archive: Path, input_archive: Path, run_tag: str, *, publish: bool = True) -> Path:
    """Run the frozen CPU finalize step, then publish from the original workspace."""
    from core.bundle import Bundle, BundleRole
    from core.common import TRAIN_ROOT
    from model_tracks.config import SuiteConfig
    import yaml

    spec = _spec()
    training_archive = Path(training_archive).resolve()
    input_archive = Path(input_archive).resolve()
    # Two boundary checks, one per archive; every later read is trusted.
    training = Bundle.load(training_archive, BundleRole.result)
    inputs = Bundle.load(input_archive, BundleRole.inputs)
    if training.run_tag() != run_tag:
        raise ValueError('snapshot completion run mismatch')
    settings = SuiteConfig.model_validate(
        yaml.safe_load(inputs.read(package_member('suite_package_config'))))
    from model_tracks.resume import runtime_source_inventory
    inventory = runtime_source_inventory(
        inputs.manifest[spec.files_key], ablation_config=settings.ablation_config)
    if not inventory or 'src/model_tracks/local_complete.py' not in inventory:
        raise ValueError('prepared inputs lack the frozen completion runtime')
    from graph_tracks.data import file_hash
    mismatches = [relative for relative, expected in inventory.items()
                  if not (TRAIN_ROOT / relative).is_file()
                  or file_hash(TRAIN_ROOT / relative) != expected]
    trace().add(
        'complete', 'runtime_inventory',
        in_count=len(inputs.manifest[spec.files_key]), out_count=len(inventory),
        reason='the packaged runtime inventory is the only source materialized into the sparse '
               'checkout, so the frozen source is what runs',
        detail={'run_tag': run_tag, 'packaged_files': len(inputs.manifest[spec.files_key]),
                'inventory_files': len(inventory),
                'local_complete_present': 'src/model_tracks/local_complete.py' in inventory,
                'working_tree_mismatches': len(mismatches),
                'mismatch_sample': mismatches[:5],
                'train_root': str(TRAIN_ROOT)},
        source=str(input_archive),
    )
    # The exact census of working-tree drift is the GROUP row; the entity rows
    # name each file whose live bytes differ from the frozen inventory.
    trace().add_entities(
        'complete.working_tree_drift', mismatches,
        key_of=lambda relative: relative,
        reason_of=lambda relative: 'live_checkout_differs_from_frozen_inventory',
        detail_of=lambda relative: {'relative': relative,
                                    'frozen_sha256': inventory.get(relative),
                                    'live_present': (TRAIN_ROOT / relative).is_file()},
        source=str(TRAIN_ROOT),
    )
    with inputs.reader() as source:
        with tempfile.TemporaryDirectory(prefix='er-suite-completion-') as temporary:
            snapshot = Path(temporary)
            written = 0
            for relative in inventory:
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(relative))
                written += 1
            # Root discovery needs project markers; packaging owns all executable
            # source and config above, so this marker carries no runtime settings.
            (snapshot / 'pyproject.toml').write_text(
                '[project]\nname = "er-frozen-completion"\nversion = "0.0.0"\n')
            models = snapshot / 'artifacts' / 'models'
            models.parent.mkdir(parents=True, exist_ok=True)
            models.symlink_to(TRAIN_ROOT / 'artifacts' / 'models', target_is_directory=True)
            result = snapshot / 'completion_result.json'
            code = (
                'import json, pathlib, sys\n'
                'from model_tracks.local_complete import complete\n'
                'final = complete(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), '
                'sys.argv[3], publish=False)\n'
                'pathlib.Path(sys.argv[4]).write_text(json.dumps({"final": str(final)}))\n'
            )
            env = {**os.environ, 'PYTHONPATH': str(snapshot / 'src'),
                   'EUROMONITOR_PROJECT_ROOT': str(snapshot),
                   'EUROMONITOR_RESULTS_DIR': str(snapshot / 'training_results'),
                   'MPLCONFIGDIR': str(snapshot / 'matplotlib'),
                   'CUDA_VISIBLE_DEVICES': '', 'PYTHONUNBUFFERED': '1'}
            print(f'[snapshot-completion] frozen CPU runtime; '
                  f'{len(mismatches)} working-tree differences retained', flush=True)
            subprocess.run([sys.executable, '-c', code, str(training_archive),
                            str(input_archive), run_tag, str(result)],
                           cwd=snapshot, env=env, check=True)
            final = Path(json.loads(result.read_text())['final'])
            trace().add(
                'complete', 'frozen_runtime',
                in_count=written, out_count=1,
                reason='the frozen source/config inventory runs the ONE finalize step in a '
                       'subprocess, so completion never depends on the live checkout',
                detail={'snapshot': str(snapshot), 'files_written': written,
                        'mismatches_retained': len(mismatches),
                        'result_json': str(result), 'final': str(final),
                        'publish_in_subprocess': False},
                source=str(input_archive),
            )
    from model_tracks.resume import validate_completed_suite_archive
    sealed = Bundle.load(final, BundleRole.result)
    validate_completed_suite_archive(final, run_tag, settings=settings, bundle=sealed)
    if sealed.run_tag() != run_tag:
        raise ValueError('snapshot completion produced a different run')
    receipt = SnapshotCompletionReceipt(
        run_tag=run_tag, input_archive_sha256=inputs.digest,
        training_archive_sha256=training.digest,
        final_archive_sha256=sealed.digest, source_inventory=inventory,
        working_tree_mismatches=mismatches)
    archive_sidecar(final, '.snapshot_completion.json').write_text(
        receipt.model_dump_json(indent=2) + '\n')
    from model_tracks.local_complete import _publish
    trace().add(
        'complete', 'receipt',
        in_count=1, out_count=1,
        reason='the receipt pins both incoming archives and the sealed completion archive, with '
               'the inventory and the retained working-tree drift',
        detail={'run_tag': run_tag, 'receipt': str(final) + '.snapshot_completion.json',
                'input_archive_sha256': inputs.digest,
                'training_archive_sha256': training.digest,
                'final_archive_sha256': sealed.digest,
                'inventory_files': len(inventory),
                'working_tree_mismatches': len(mismatches),
                'publish': bool(publish)},
        source=str(final),
    )
    published = _publish(final, settings, run_tag, ablation_done=True) if publish else final
    flush_trace()
    return published
