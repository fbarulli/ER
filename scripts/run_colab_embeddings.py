"""Run the standalone hybrid embedding worker on Colab T4 and retrieve its cache.

From ER: PYTHONPATH=src .venv/bin/python scripts/run_colab_embeddings.py
"""
import json
import os
from pathlib import Path
from cli import colab as backend
from core.common import TRAIN_ROOT
from graph_tracks.data import file_hash

def main():
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
        # Code, frozen checkpoint, and prepared embedding inputs come from Git.
        # Avoid duplicating the tracked model in a large Colab upload request.
        script = f'''import os, pathlib, subprocess, sys
    root = pathlib.Path({backend.REMOTE_ROOT!r})
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


if __name__ == "__main__":
    main()
