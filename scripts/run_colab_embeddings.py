"""Compose/validate locally; Colab encoding supports explicit CPU smoke tests."""
import json
import argparse
import os
from pathlib import Path
import tempfile
import uuid
import tarfile
from core.archive_reader import tar_archive
from cli import colab as backend
from core.common import TRAIN_ROOT, resolve_model, runtime
from graph_tracks.data import file_size
from training.prepare_embeddings import input_identity, prepare_request, validate_result


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import training_cfg
    return training_cfg().preparation.graph_setup


def publish_results(paths, message: str) -> None:
    """Default publisher: DVC-track, push and point at the paths (run_retention)."""
    from model_tracks.run_retention import publish_result_paths
    publish_result_paths(paths)


def complete_local_handoff(local, request):
    from model_tracks.preflight import preflight
    validate_result(local, request)
    print('[embeddings/local] checking text, GNN and hybrid handoff', flush=True)
    try:
        checks = preflight(TRAIN_ROOT / 'config/model_tracks.yaml')
        status = 'complete'
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        checks = {'error':str(exc),'type':type(exc).__name__}
        status = 'blocked'
    report = TRAIN_ROOT / 'results/embedding_job/local_handoff.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({'status': status, 'cache': str(local),
                                 'size': file_size(local), 'preflight': checks}, indent=2) + '\n')
    print(f'[embeddings/local] handoff {status}: {report}', flush=True)
    return report


def persist_embeddings(local, handoff, publisher=None, *, additional_files=None, namespace="embedding_job", prefix="embeddings"):
    """Save the verified embedding tar.zst through the DVC-backed publisher."""
    from core.portable_archive import ByteCount
    setup = local.parent
    layout = _setup_layout()
    files = {p.relative_to(setup).as_posix(): p for p in (
        local, setup / layout.embedding_request, setup / layout.catalog,
        setup / 'encoding.log',setup / 'prepared_text.npz',
        setup / layout.manifest, *sorted((setup / layout.prepared_dir).glob('*')))
        if p.is_file()}
    files.update(additional_files or {})
    files['local_handoff.json'] = handoff
    # Structural identity (names + byte sizes): the retired sha256 token became a
    # ByteCount total, so render it as text before truncating it into the tag
    # (4cc0af9 left the slice on an int and broke every persist call).
    identity = str(ByteCount(json.dumps({key: file_size(path) for key, path in files.items()},
                                        sort_keys=True).encode()).total)
    run_tag = prefix + '-' + identity[:24]
    folder = TRAIN_ROOT / 'results' / namespace / run_tag
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / backend._RESULT_ARCHIVE_NAME
    manifest = {'schema_version': '1', 'run_id': run_tag, 'workers': 1,
                'included': [{'worker': 1, 'path': key, 'size': path.stat().st_size,
                              'size': file_size(path)} for key, path in files.items()],
                'excluded': []}
    if not archive.exists():
        manifest_path = folder / backend._RESULT_MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        with tar_archive(archive, 'x') as result:
            for key, path in files.items():
                result.add(path, arcname='worker_1/' + key)
            result.add(manifest_path, arcname=backend._RESULT_MANIFEST_NAME)
    # Reuse the existing Colab tar.zst extractor and checksum/coverage validator.
    with tempfile.TemporaryDirectory(dir=folder) as temporary:
        verified = backend._extract_result_archive(archive, Path(temporary), run_tag, 1)
        expected = {(item['path'], item['size']) for item in manifest['included']}
        if {(item.path, item.size) for item in verified.included} != expected:
            raise ValueError('existing embedding archive differs from validated result')
    print(f'[{prefix}/local] saving verified tar.zst through DVC: {archive}', flush=True)
    kind = 'cache' if prefix == 'embeddings' else 'results'
    (publisher or publish_results)([archive], f'{prefix}: save verified {kind} {run_tag}')
    print(f'[{prefix}/local] DVC save complete: {archive}', flush=True)
    return archive


def main(prepared_request=None, *, device='cuda', smoke_size=None, publisher=None):
    if device == 'cpu' and smoke_size is None:
        raise ValueError('CPU is permitted only for an explicit embedding smoke test')
    if smoke_size is not None and smoke_size < 1:
        raise ValueError('Smoke size must be positive')
    setup = TRAIN_ROOT / 'data/track_setup'
    layout = _setup_layout()
    checkpoint = Path(resolve_model('minilm_l6'))
    local = setup / layout.shared_embeddings
    # Finish CPU work before allocating a GPU. Never reuse a cached text request.
    if smoke_size is not None:
        if prepared_request is not None:
            raise ValueError('Smoke tests compose fresh local texts')
        import pandas as pd
        from graph_tracks.text_cache import compose_texts, texts_size
        source_identity = input_identity(setup, checkpoint)
        smoke = TRAIN_ROOT / 'results/embedding_job' / ('smoke-' + uuid.uuid4().hex)
        smoke.mkdir(parents=True)
        catalog = smoke / layout.catalog
        rows = pd.read_csv(setup / layout.catalog, dtype=str, keep_default_na=False).head(smoke_size)
        if len(rows) != smoke_size:
            raise ValueError('Smoke sample exceeds eligible listing population')
        rows.to_csv(catalog,index=False)
        ids,texts = compose_texts(catalog)
        request = {'schema':'er-embedding-request-v2','ids':ids,'texts':texts,
                   'metadata':{**source_identity,'text_size':texts_size(texts),'scope':'smoke',
                               'sample_size':smoke_size,'sample_catalog_size':file_size(catalog)}}
        local = smoke / layout.shared_embeddings
        print(f'[embeddings/local] smoke={smoke_size} device={device} output={smoke}',flush=True)
    elif prepared_request is None:
        request = prepare_request(setup, checkpoint)
    else:
        from graph_tracks.text_cache import texts_size
        from graph_tracks.data import load_records
        request = json.loads(Path(prepared_request).read_text())
        if (request.get('schema') != 'er-embedding-request-v2'
                or request['metadata'].get('text_size') != texts_size(request['texts'])
                or len(request['ids']) != len(request['texts'])
                or len(set(request['ids'])) != len(request['ids'])
                or set(request['ids']) != {r['sku_id'] for r in load_records(setup/layout.prepared_dir/'listings.json')}):
            raise ValueError('Prepared text request is corrupt; recompose locally')
        print('[embeddings/local] resuming prepared texts; no CPU recomposition needed', flush=True)
    # Prepare native token IDs locally, including full-length validation.
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from core.encoding_inputs import prepare_token_batches
    arrays = {}
    tokenizer_model = SentenceTransformer(str(checkpoint),device='cpu',local_files_only=True)
    token_plan = prepare_token_batches(tokenizer_model,request['texts'],arrays,batch_size=runtime('batch_size_embed'))
    del tokenizer_model
    tokens = local.parent/'prepared_text.npz'
    tokens.parent.mkdir(parents=True,exist_ok=True)
    with tokens.open('wb') as handle:
        np.savez_compressed(handle,**arrays)
    request['prepared_text'] = {**token_plan,'size':file_size(tokens)}
    print(f'[embeddings/local] tokenized={len(request["texts"])} truncated=0 max_tokens={max(token_plan["token_lengths"],default=0)}',flush=True)
    saved = local.parent / layout.embedding_request if smoke_size is not None else TRAIN_ROOT / 'results/embedding_job/prepared_request.json'
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.with_suffix('.tmp').write_text(json.dumps(request, ensure_ascii=False, sort_keys=True))
    saved.with_suffix('.tmp').replace(saved)
    if local.exists():
        validate_result(local, request)  # identity only; no recorded-size recomparison
        print('[embeddings/local] current cache verified against freshly composed texts', flush=True)
        handoff = complete_local_handoff(local, request)
        persist_embeddings(local, handoff,publisher=publisher)
        if json.loads(handoff.read_text())['status'] != 'complete':
            raise RuntimeError('Embedding result saved; local training handoff is blocked: '+json.loads(handoff.read_text())['preflight']['error'])
        return
    backend.GPU = 'CPU' if device == 'cpu' else 'T4'
    os.environ['EUROMONITOR_KEEP_ALIVE_ALLOWED'] = '1'
    backend.check_colab_cli()
    lock = backend.acquire_colab_launch_lock()
    backend.start_live_log()
    try:
        with tempfile.TemporaryDirectory(prefix='embedding-job-', dir=setup) as temporary:
            temporary = Path(temporary)
            request_path = temporary / 'request.json'
            request_path.write_text(json.dumps(request, ensure_ascii=False, sort_keys=True))
            request_size = file_size(request_path)
            package = temporary / 'gpu_inputs.tar.zst'
            with tar_archive(package, 'w') as archive:
                archive.add(request_path, arcname='request.json')
                archive.add(tokens,arcname='prepared_text.npz')
                archive.add(TRAIN_ROOT/'src/core/encoding_inputs.py',arcname='encoding_inputs.py')
                archive.add(TRAIN_ROOT / 'scripts/encode_prepared_embeddings.py', arcname='encode.py')
            stored = TRAIN_ROOT / 'results/embedding_job/inputs' / f'embeddings-{request_size}.tar.zst'
            stored.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(package, stored)
            (publisher or publish_results)([stored], f'embeddings: save prepared {device} inputs {request_size}')
            remote_package = backend.REMOTE_ROOT + '/' + stored.relative_to(TRAIN_ROOT).as_posix()
            # The frozen checkpoint already ships in Git. The worker verifies
            # its actual bytes against the local request before GPU encoding.
            remote_checkpoint = backend.REMOTE_ROOT + '/' + checkpoint.relative_to(TRAIN_ROOT).as_posix()
            current = input_identity(setup, checkpoint)
            if any(current[key] != request['metadata'][key] for key in current):
                raise ValueError('Embedding inputs changed during packaging')
            backend.ensure_session()
            backend.stop_keep_alive_daemon(reason=f'{device} embedding job')
            backend.prepare_remote_layout(minimal_runtime=True)
            backend.install_deps(minimal_runtime=True, graph_runtime=True)
            job = backend.REMOTE_ROOT + '/prepared_training/embeddings_' + uuid.uuid4().hex
            backend.run_colab_exec_stream(backend.SESSION,
                f'import pathlib\npathlib.Path({job!r}).mkdir(parents=True)\n',
                timeout=120, log_name='embedding_directory', retry_safe=True)
            # The input package is an INPUT the lane consumes: materialize it on
            # the VM by direct upload. DVC is write-only storage (no lane loads).
            backend._upload_with_retries(
                stored, remote_package,
                timeout=backend._RESULT_DOWNLOAD_TIMEOUT_SECONDS)
            script = (
                'import pathlib, subprocess, sys, tarfile\n'
                f'sys.path.insert(0, {backend.REMOTE_ROOT + "/src"!r})\n'
                'from core.archive_reader import tar_archive\n'
                'from graph_tracks.data import file_size\n'
                f'root = pathlib.Path({job!r})\n'
                f"package = pathlib.Path({remote_package!r})\n"
                f"assert file_size(package) == {file_size(package)!r}, 'input package size mismatch'\n"
                "with tar_archive(package) as archive:\n"
                "    archive.extractall(root, filter='data')\n"
                "subprocess.run([sys.executable, str(root/'encode.py'), '--request', str(root/'request.json'), "
                f"'--checkpoint', {remote_checkpoint!r}, '--output', str(root/'vectors.npz'), '--device', {device!r}], check=True)\n"
            )
            backend.run_detached_stage('hybrid_embeddings', ['/usr/bin/python3', '-c', script],
                                       timeout=backend._WORKER_TIMEOUT_SECONDS)
            size_probe = ('import json, pathlib\n'
                              f"print(json.dumps({{'size': pathlib.Path({(job + '/vectors.size')!r}).read_text().strip()}}))\n")
            expected = backend._parse_remote_json(backend.run_colab_exec_capture(
                backend.SESSION, size_probe, timeout=backend._PROBE_TIMEOUT_SECONDS))['size']
            candidate = temporary / 'vectors.npz'
            print(f'[embeddings/local] downloading {device} result', flush=True)
            backend._download_one_remote_file(job + '/vectors.npz', candidate)
            if file_size(candidate) != expected:
                raise ValueError('Embedding download size mismatch')
            validate_result(candidate, request)
            # Persist provenance first. Readers require it and fail on any mismatch.
            request_path.replace(local.parent / layout.embedding_request)
            candidate.replace(local)
            print(f'[embeddings/local] validated cache published: {local}', flush=True)
    finally:
        backend.stop()
        backend.close_live_log()
        backend.release_colab_launch_lock(lock)
    import re
    log = backend.LIVE_LOG_PATH.read_text() if backend.LIVE_LOG_PATH.is_file() else ''
    log = re.sub(r'(?i)(colab-runtime-proxy-token[= :]+)[^\s&\"\']+',r'\1[REDACTED]',log)
    (local.parent / 'encoding.log').write_text(log)
    if smoke_size is not None:
        handoff = local.parent / 'smoke_report.json'
        handoff.write_text(json.dumps({'status':'passed','scope':'smoke','device':device,
            'sample_size':smoke_size,'shape':validate_result(local,request), 'size':file_size(local)},indent=2))
        persist_embeddings(local,handoff,publisher=publisher)
    else:
        handoff = complete_local_handoff(local, request)
        persist_embeddings(local, handoff,publisher=publisher)
        if json.loads(handoff.read_text())['status'] != 'complete':
            raise RuntimeError('Embedding result saved; local training handoff is blocked: '+json.loads(handoff.read_text())['preflight']['error'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-request', type=Path,
                        help='resume only when current input/code sizes match the preserved request')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--smoke-size',type=int)
    args = parser.parse_args()
    main(args.prepared_request,device=args.device,smoke_size=args.smoke_size)
