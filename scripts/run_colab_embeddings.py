"""Compose and validate locally; use Colab only for CUDA encoding."""
import json
import os
from pathlib import Path
import tempfile
import uuid
import zipfile
from cli import colab as backend
from core.common import TRAIN_ROOT, resolve_model
from graph_tracks.data import file_hash
from training.prepare_embeddings import input_identity, prepare_request, validate_result


def complete_local_handoff(local, request):
    from model_tracks.preflight import preflight
    validate_result(local, request)
    print('[embeddings/local] checking text, GNN and hybrid handoff', flush=True)
    checks = preflight(TRAIN_ROOT / 'config/model_tracks.yaml')
    report = TRAIN_ROOT / 'results/embedding_job/local_handoff.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({'status': 'complete', 'cache': str(local),
                                 'sha256': file_hash(local), 'preflight': checks}, indent=2) + '\n')
    print(f'[embeddings/local] handoff verified: {report}', flush=True)


def main():
    setup = TRAIN_ROOT / 'data/track_setup'
    checkpoint = Path(resolve_model('minilm_l6'))
    local = setup / 'shared_minilm__embeddings.npz'
    # Finish CPU work before allocating a GPU. Never reuse a cached text request.
    request = prepare_request(setup, checkpoint)
    if local.exists():
        validate_result(local, request)  # Legacy/stale results fail closed.
        print('[embeddings/local] current cache verified against freshly composed texts', flush=True)
        complete_local_handoff(local, request)
        return
    backend.GPU = 'T4'
    os.environ['EUROMONITOR_KEEP_ALIVE_ALLOWED'] = '1'
    backend.check_colab_cli()
    lock = backend.acquire_colab_launch_lock()
    backend.start_live_log()
    try:
        with tempfile.TemporaryDirectory(prefix='embedding-job-', dir=setup) as temporary:
            temporary = Path(temporary)
            request_path = temporary / 'request.json'
            request_path.write_text(json.dumps(request, ensure_ascii=False, sort_keys=True))
            request_digest = file_hash(request_path)
            package = temporary / 'gpu_inputs.zip'
            with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_STORED) as archive:
                archive.write(request_path, 'request.json')
                archive.write(TRAIN_ROOT / 'scripts/encode_prepared_embeddings.py', 'encode.py')
                for path in sorted(checkpoint.rglob('*')):
                    if path.is_file():
                        archive.write(path, 'checkpoint/' + path.relative_to(checkpoint).as_posix())
            current = input_identity(setup, checkpoint)
            if any(current[key] != request['metadata'][key] for key in current):
                raise ValueError('Embedding inputs changed during packaging')
            backend.ensure_session()
            backend.stop_keep_alive_daemon(reason='GPU embedding job')
            backend.prepare_remote_layout(minimal_runtime=True)
            backend.install_deps(minimal_runtime=True, graph_runtime=True)
            job = backend.REMOTE_ROOT + '/prepared_training/embeddings_' + uuid.uuid4().hex
            backend.run_colab_exec_stream(backend.SESSION,
                f'import pathlib\npathlib.Path({job!r}).mkdir(parents=True)\n',
                timeout=120, log_name='embedding_directory', retry_safe=True)
            backend._upload_with_retries(package, job + '/inputs.zip', timeout=600)
            script = (
                'import hashlib, pathlib, subprocess, sys, zipfile\n'
                f'root = pathlib.Path({job!r})\n'
                f"assert hashlib.sha256((root/'inputs.zip').read_bytes()).hexdigest() == {file_hash(package)!r}, 'Input upload checksum mismatch'\n"
                "with zipfile.ZipFile(root/'inputs.zip') as archive:\n"
                "    archive.extractall(root)\n"
                "subprocess.run([sys.executable, str(root/'encode.py'), '--request', str(root/'request.json'), "
                "'--checkpoint', str(root/'checkpoint'), '--output', str(root/'vectors.npz')], check=True)\n"
            )
            backend.run_detached_stage('hybrid_embeddings', ['/usr/bin/python3', '-c', script],
                                       timeout=backend._WORKER_TIMEOUT_SECONDS)
            expected = backend._read_remote_text(job + '/vectors.sha256').strip()
            candidate = temporary / 'vectors.npz'
            print('[embeddings/local] downloading GPU result', flush=True)
            backend._download_one_remote_file(job + '/vectors.npz', candidate)
            if file_hash(candidate) != expected:
                raise ValueError('Embedding download checksum mismatch')
            validate_result(candidate, request, request_sha256=request_digest)
            current = input_identity(setup, checkpoint)
            if any(current[key] != request['metadata'][key] for key in current):
                raise ValueError('Local inputs changed during GPU encoding; refusing publication')
            # Persist provenance first. Readers require it and fail on any mismatch.
            request_path.replace(setup / 'embedding_inputs.json')
            candidate.replace(local)
            print(f'[embeddings/local] validated cache published: {local}', flush=True)
    finally:
        backend.stop()
        backend.close_live_log()
        backend.release_colab_launch_lock(lock)
    complete_local_handoff(local, request)


if __name__ == '__main__':
    main()
