"""Complete a downloaded suite using its immutable packaged runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
from model_tracks.package import package_member
import subprocess
import sys
import tempfile

from pydantic import BaseModel, ConfigDict, Field

from core.portable_archive import Digest, verify_archive
from graph_tracks.data import file_hash
from core.archive_reader import archive_sidecar, open_archive


class SnapshotCompletionReceipt(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    run_tag: str = Field(pattern=r'^[A-Za-z0-9_-]+$')
    input_archive_sha256: Digest
    training_archive_sha256: Digest
    final_archive_sha256: Digest
    source_inventory: dict[str, Digest]
    working_tree_mismatches: list[str]


def complete(training_archive: Path, input_archive: Path, run_tag: str, *, publish: bool = True) -> Path:
    """Run frozen CPU reporting, then publish from the original workspace."""
    from core.common import TRAIN_ROOT
    from model_tracks.config import SuiteConfig
    import yaml

    training_archive = training_archive.resolve()
    input_archive = input_archive.resolve()
    training = verify_archive(training_archive, 'suite_bundle_manifest.json')
    inputs = verify_archive(input_archive, 'model_tracks_package.json')
    if training['run_tag'] != run_tag:
        raise ValueError('snapshot completion run mismatch')
    with open_archive(input_archive) as archive:
        settings = SuiteConfig.model_validate(
            yaml.safe_load(archive.read(package_member('suite_package_config'))))
    inventory = {relative: digest for relative, digest in inputs['files'].items()
                 if relative.startswith(('src/', 'config/', 'scripts/'))}
    if settings.ablation_config in inputs['files']:
        inventory[settings.ablation_config] = inputs['files'][settings.ablation_config]
    if not inventory or 'src/model_tracks/local_complete.py' not in inventory:
        raise ValueError('prepared inputs lack the frozen completion runtime')
    mismatches = [relative for relative, expected in inventory.items()
                  if not (TRAIN_ROOT / relative).is_file()
                  or file_hash(TRAIN_ROOT / relative) != expected]
    with open_archive(input_archive) as archive:
        with tempfile.TemporaryDirectory(prefix='er-suite-completion-') as temporary:
            snapshot = Path(temporary)
            for relative in inventory:
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(relative))
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
    from model_tracks.resume import validate_completed_suite_archive
    completed = validate_completed_suite_archive(final, run_tag, settings=settings)
    if completed.get('run_tag') != run_tag:
        raise ValueError('snapshot completion produced a different run')
    receipt = SnapshotCompletionReceipt(
        run_tag=run_tag, input_archive_sha256=file_hash(input_archive),
        training_archive_sha256=file_hash(training_archive),
        final_archive_sha256=file_hash(final), source_inventory=inventory,
        working_tree_mismatches=mismatches)
    archive_sidecar(final, '.snapshot_completion.json').write_text(
        receipt.model_dump_json(indent=2) + '\n')
    from model_tracks.local_complete import _publish
    return _publish(final, settings, run_tag, ablation_done=True) if publish else final
