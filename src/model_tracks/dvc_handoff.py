"""Verified Colab-to-DVC transport, followed by a local pull after release."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import yaml

from core.portable_archive import verify_archive
from graph_tracks.data import file_hash


def publish(archive: Path, run_tag: str) -> Path:
    from core import common
    from model_tracks.publish import persist_results
    receipt = persist_results(archive, run_tag)
    data = json.loads(receipt.read_text())
    data.update(schema='er-training-dvc-handoff-v1', archive_name=archive.name,
                archive_size=archive.stat().st_size,
                remote=common.training_cfg().colab.dvc_remote_url,
                pointer=(archive.with_suffix('.publication') / (archive.name + '.dvc')).read_text())
    profile = archive.with_suffix('.profile.json')
    if profile.is_file():
        data['archive_profile'] = json.loads(profile.read_text())
    events = archive.with_suffix('.publication') / common.training_cfg().colab.dvc_events_file
    if events.is_file():
        data['dvc_profile_events'] = events.read_text()
    receipt.write_text(json.dumps(data, indent=2) + '\n')
    return receipt


def validate(data: dict, run_tag: str, digest: str, remote: str) -> dict:
    """Reject a wrong run, remote or unsafe pointer before releasing Colab."""
    if (data.get('schema') != 'er-training-dvc-handoff-v1'
            or data.get('run_tag') != run_tag or data.get('archive_sha256') != digest
            or data.get('remote') != remote or data.get('verified_download') is not True
            or data.get('archive_name') != run_tag + '.zip'
            or not isinstance(data.get('archive_size'), int) or data['archive_size'] <= 0):
        raise ValueError('training DVC handoff identity mismatch')
    pointer = yaml.safe_load(data['pointer'])
    outs = pointer.get('outs', []) if isinstance(pointer, dict) else []
    if len(outs) != 1 or outs[0].get('path') != data['archive_name'] or not outs[0].get('md5'):
        raise ValueError('training DVC handoff pointer mismatch')
    return data


def pull(data: dict, destination: Path) -> Path:
    from training import dvc_store
    validate(data, data['run_tag'], data['archive_sha256'], data['remote'])
    token = os.environ.get('DVC_API_KEY')
    if not token:
        raise RuntimeError('DVC_API_KEY is required for local training-result pull')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if 'archive_profile' in data:
        destination.with_suffix('.profile.json').write_text(json.dumps(data['archive_profile'], indent=2) + '\n')
    if 'dvc_profile_events' in data:
        destination.with_suffix('.dvc_profile.jsonl').write_text(data['dvc_profile_events'])
    # An isolated cache forces collection to exercise the durable remote.
    pull_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='.training-dvc-pull-', dir=destination.parent) as temporary:
        root = Path(temporary)
        pointer = root / (data['archive_name'] + '.dvc')
        pointer.write_text(data['pointer'])
        dvc_store._configure_workspace(root, token=token, remote=data['remote'])
        dvc_store._run(['dvc', 'pull', '--force', pointer.name], root)
        payload = root / data['archive_name']
        if payload.stat().st_size != data['archive_size'] or file_hash(payload) != data['archive_sha256']:
            raise ValueError('training DVC pull archive checksum mismatch')
        manifest = verify_archive(payload, 'suite_bundle_manifest.json')
        if manifest.get('run_tag') != data['run_tag']:
            raise ValueError('training DVC pull run mismatch')
        partial = destination.with_suffix('.zip.partial')
        shutil.copy2(payload, partial)
        partial.replace(destination)
    with destination.with_suffix('.dvc_profile.jsonl').open('a') as handle:
        handle.write(json.dumps({'event': 'local_pull_verified',
            'timestamp_unix': time.time(), 'elapsed_seconds': time.monotonic() - pull_started,
            'archive_bytes': destination.stat().st_size}) + '\n')
    return destination
