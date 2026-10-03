"""Run one prepared frozen-checkpoint job through the existing Git/Colab flow."""
import argparse
import json
import re
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path
from cli import colab as backend
from core.common import TRAIN_ROOT
from graph_tracks.data import file_hash
from model_tracks.ablation import resolve, validate_sources, report, save_report, frozen_threshold, Settings, load_prepared, verify_threshold_binding, validate_vectors
from model_tracks.publish import push_artifacts
from model_tracks.package import runtime_snapshot_files
from run_colab_embeddings import persist_embeddings


def main(request_path, *, threshold, threshold_source, publisher=None):
    request_path = request_path.resolve()
    request = json.loads(request_path.read_text())
    validate_sources(request)
    load_prepared(request_path,request).close()
    verify_threshold_binding(request,frozen_threshold(threshold_source, threshold))
    # Every source and the worker must be available from the existing clone.
    for name in request['sources']:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Colab inputs must be inside the Git repository')
        # Directory sources (e.g. text checkpoints) are too large for the
        # tar and stay in the clone; the ablation bootstrap has no dvc
        # pull. Verify the current branch's tree actually carries them so
        # the remote clone cannot miss them, before any accelerator is
        # provisioned (the publication push below ships this branch).
        if resolve(name).is_dir():
            tracked = subprocess.run(
                ['git','ls-tree','-r','--name-only','HEAD','--',name],
                cwd=TRAIN_ROOT, capture_output=True, text=True)
            if tracked.returncode != 0 or not tracked.stdout.strip():
                raise ValueError(
                    f'directory source {name} is not committed to the current branch; '
                    'the remote clone cannot provide it (no dvc pull in the ablation '
                    'bootstrap) — commit and push it as Git content first')
    folder = request_path.parent
    package = folder/'inputs.tar.gz'
    files = runtime_snapshot_files()
    # Catalog/pairs/config/graph checkpoint are shipped through Git in the same tar.
    # Directory text checkpoints stay in the existing clone and are hash checked.
    files.update({name:resolve(name) for name in request['sources'] if resolve(name).is_file()})
    inventory = {name:file_hash(path) for name,path in files.items()}
    if not package.exists():
        with tarfile.open(package,'x:gz') as archive:
            archive.add(request_path,arcname='request.json')
            archive.add(folder/'prepared_inputs.npz',arcname='prepared_inputs.npz')
            archive.add(TRAIN_ROOT/'src/model_tracks/ablation.py',arcname='ablation.py')
            for name,path in files.items():
                archive.add(path,arcname='runtime/'+name)
    with tarfile.open(package,'r:gz') as archive:
        expected = {'prepared_inputs.npz':file_hash(folder/'prepared_inputs.npz'),'request.json':file_hash(request_path),'ablation.py':file_hash(TRAIN_ROOT/'src/model_tracks/ablation.py'),
                    **{'runtime/'+name:sha for name,sha in inventory.items()}}
        import hashlib
        if len(archive.getnames()) != len(expected) or set(archive.getnames()) != set(expected) or any(
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() != sha for name,sha in expected.items()):
            raise ValueError('ablation input package is stale')
    if package.stat().st_size >= 100*1024**2:
        raise ValueError('ablation inputs exceed GitHub file limit; use the existing DVC artifact flow')
    publish = publisher or push_artifacts
    publish([package],f'ablation: save frozen inputs {file_hash(request_path)[:24]}')
    remote_package = backend.REMOTE_ROOT+'/'+package.relative_to(TRAIN_ROOT).as_posix()
    job = backend.REMOTE_ROOT+'/prepared_training/ablation_'+uuid.uuid4().hex
    result = folder/'vectors.npz'
    if result.exists():
        # A retry after download or publication must finish persistence without
        # allocating another accelerator. Full reporting verifies provenance.
        validated = report(request_path,result,threshold,threshold_source=threshold_source,save=False)
        return persist_result(request_path,result,validated,threshold_source,publisher=publish)
    accelerator = Settings.model_validate(request.get('settings', {})).accelerator
    if accelerator.upper() == 'CPU':
        raise ValueError('ablation requires a GPU accelerator')
    backend.GPU = accelerator
    backend.check_colab_cli()
    lock = backend.acquire_colab_launch_lock()
    backend.start_live_log()
    try:
        backend.ensure_session()
        backend.stop_keep_alive_daemon(reason='frozen checkpoint ablation')
        backend.prepare_remote_layout(minimal_runtime=True)
        backend.install_deps(minimal_runtime=True,graph_runtime=True)
        script = ('import hashlib, pathlib, subprocess, sys, tarfile, os, shutil\n'
            f'root = pathlib.Path({job!r}); root.mkdir(parents=True)\n'
            f'package = pathlib.Path({remote_package!r})\n'
            f'assert hashlib.sha256(package.read_bytes()).hexdigest() == {file_hash(package)!r}\n'
            "with tarfile.open(package, 'r:gz') as archive:\n    archive.extractall(root, filter='data')\n"
            f"shutil.copytree(root/'runtime', {backend.REMOTE_ROOT!r}, dirs_exist_ok=True)\n"
            f"os.environ['PYTHONPATH'] = {backend.REMOTE_ROOT + '/src'!r}\n"
            f"os.chdir({backend.REMOTE_ROOT!r})\n"
            "subprocess.run([sys.executable, str(root/'ablation.py'), 'encode', '--request', str(root/'request.json'), '--output', str(root/'vectors.npz')], check=True)\n")
        backend.run_detached_stage('attribute_ablation',['/usr/bin/python3','-c',script],timeout=backend._WORKER_TIMEOUT_SECONDS)
        probe = 'import json, pathlib\n'+f"print(json.dumps({{'sha256': pathlib.Path({job + '/vectors.sha256'!r}).read_text().strip()}}))\n"
        expected = backend._parse_remote_json(backend.run_colab_exec_capture(backend.SESSION,probe,timeout=backend._PROBE_TIMEOUT_SECONDS))['sha256']
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            downloaded = Path(tmp)/'vectors.npz'
            backend._download_one_remote_file(job+'/vectors.npz',downloaded)
            if file_hash(downloaded) != expected:
                raise ValueError('ablation download checksum mismatch')
            validate_sources(request)
            # Validate full shape/provenance before installing the downloaded result.
            validate_vectors(request_path,downloaded)
            downloaded.replace(result)
    finally:
        backend.stop()
        backend.close_live_log()
        backend.release_colab_launch_lock(lock)
    log = backend.LIVE_LOG_PATH.read_text() if backend.LIVE_LOG_PATH.is_file() else ''
    log = re.sub(r'(?i)(colab-runtime-proxy-token[= :]+)[^\s&\"\']+',r'\1[REDACTED]',log)
    (folder/'encoding.log').write_text(log)
    # Persist the already-validated report once: the installed result is
    # the hash-checked download, so recomputing it would duplicate work.
    validated = report(request_path,result,threshold,threshold_source=threshold_source,save=False)
    return persist_result(request_path,result,validated,threshold_source,publisher=publish)


def persist_result(request_path,result,validated,threshold_source,*,publisher=None):
    handoff = save_report(request_path, validated)
    return persist_embeddings(result,handoff,publisher=publisher,
        additional_files={'request.json':request_path,'prepared_inputs.npz':request_path.parent/'prepared_inputs.npz','report.json':request_path.parent/'report.json',
                          'baseline_threshold_report'+resolve(threshold_source).suffix:resolve(threshold_source)},
        namespace='attribute_ablation',prefix='ablation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--threshold',type=float,required=True)
    parser.add_argument('--threshold-source',required=True)
    args = parser.parse_args()
    main(args.request,threshold=args.threshold,threshold_source=args.threshold_source)
