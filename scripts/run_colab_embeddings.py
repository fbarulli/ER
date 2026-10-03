"""Run the standalone hybrid embedding worker on Colab T4 and retrieve its cache.

From ER: PYTHONPATH=src .venv/bin/python scripts/run_colab_embeddings.py
"""
import json
import os
from pathlib import Path
from cli import colab as backend
from core.common import TRAIN_ROOT
from graph_tracks.data import file_hash

backend.GPU = 'T4'
os.environ['EUROMONITOR_KEEP_ALIVE_ALLOWED'] = '1'
backend.check_colab_cli()
lock = backend.acquire_colab_launch_lock()
backend.start_live_log()
try:
    backend.ensure_session()
    backend.stop_keep_alive_daemon(reason='GPU embedding job')
    backend.prepare_remote_layout(minimal_runtime=True)
    backend.install_deps(minimal_runtime=True, graph_runtime=True)
    archive = TRAIN_ROOT / 'results/embedding_job/colab_embeddings_inputs.zip'
    remote_archive = backend.REMOTE_ROOT + '/embedding_inputs.zip'
    backend._upload_with_retries(archive, remote_archive, timeout=backend._RESULT_DOWNLOAD_TIMEOUT_SECONDS)
    script = f'''import os, pathlib, subprocess, sys, zipfile
root = pathlib.Path({backend.REMOTE_ROOT!r})
with zipfile.ZipFile({remote_archive!r}) as archive:
    for member in archive.infolist():
        relative = pathlib.Path(member.filename).relative_to('ER')
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError('unsafe archive path')
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
subprocess.run([sys.executable, '-m', 'training.prepare_embeddings', '--setup-dir', 'data/track_setup', '--device', 'cuda', '--batch-size', '256'], cwd=root, env={{**os.environ, 'PYTHONPATH': str(root/'src'), 'PYTHONUNBUFFERED': '1'}}, check=True)
'''
    backend.log_gpu_profile()
    backend.run_detached_stage('hybrid_embeddings', ['/usr/bin/python3', '-c', script], timeout=backend._WORKER_TIMEOUT_SECONDS)
    remote = backend.REMOTE_ROOT + '/data/track_setup/shared_minilm__embeddings.npz'
    expected = backend.run_colab_exec_capture(backend.SESSION, f"import hashlib\nprint(hashlib.sha256(open({remote!r}, 'rb').read()).hexdigest())", timeout=120).strip()
    local = TRAIN_ROOT / 'data/track_setup/shared_minilm__embeddings.npz'
    partial = local.with_suffix('.npz.partial')
    backend._download_one_remote_file(remote, partial)
    if file_hash(partial) != expected:
        raise ValueError('download hash mismatch')
    partial.replace(local)
    print(f'GPU embedding cache retrieved and verified: {local}', flush=True)
finally:
    backend.stop()
    backend.close_live_log()
    backend.release_colab_launch_lock(lock)
