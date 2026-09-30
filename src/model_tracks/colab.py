"""All-track adapter; provisioning, locks, polling and teardown stay in cli.colab."""
import json
from pathlib import Path
import zipfile

from core.portable_archive import verify_archive
from graph_tracks.data import file_hash
from model_tracks.package import verify


def run(archive: Path, run_tag: str):
    from cli import colab as backend
    from core.common import RESULTS
    metadata = verify(archive)
    remote_zip = f'{backend.REMOTE_ROOT}/prepared_training/{run_tag}__all_tracks.zip'
    backend.run_colab_exec_stream(backend.SESSION,
        f'import pathlib\npathlib.Path({remote_zip!r}).parent.mkdir(parents=True,exist_ok=True)\n',
        timeout=120,log_name='tracks_upload_directory',retry_safe=True)
    backend._upload_with_retries(archive,remote_zip,timeout=backend._RESULT_DOWNLOAD_TIMEOUT_SECONDS)
    remote_output = f'{backend.REMOTE_ROOT}/results/model_tracks/{run_tag}'
    with zipfile.ZipFile(archive) as source:
        import yaml
        settings = yaml.safe_load(source.read('data/model_tracks/suite.yaml'))
    needs_dvc = settings['publish_git'] or settings.get('publish_dvc', False)
    auth = backend._remote_auth_env_script(include_wandb=True, force_dvc=needs_dvc)
    script = backend._BOOTSTRAP + auth + f'''
import hashlib, json, os, pathlib, subprocess, sys, zipfile
root=pathlib.Path({backend.REMOTE_ROOT!r})
archive_path=pathlib.Path({remote_zip!r})
if hashlib.sha256(archive_path.read_bytes()).hexdigest() != {file_hash(archive)!r}:
    raise ValueError("prepared all-track upload mismatch")
subprocess.run(["git","fetch",{backend.GIT_REMOTE_NAME!r},{metadata['revision']!r}],cwd=root,check=True)
subprocess.run(["git","checkout","--detach",{metadata['revision']!r}],cwd=root,check=True)
with zipfile.ZipFile(archive_path) as archive:
    for member in archive.infolist():
        if not (root/member.filename).resolve().is_relative_to(root.resolve()):
            raise ValueError("unsafe input package member")
    archive.extractall(root)
from core.portable_archive import verify_archive
verify_archive(archive_path,"model_tracks_package.json")
env={{**os.environ,"PYTHONPATH":str(root/"src"),"PYTHONUNBUFFERED":"1"}}
result_archive=pathlib.Path({remote_output!r}+".zip")
if result_archive.exists():
    # Collection/publication retry must never restart completed training.
    verified=verify_archive(result_archive,"suite_bundle_manifest.json")
    if verified["run_tag"] != {run_tag!r}:
        raise ValueError("existing suite result run mismatch")
    from model_tracks.config import load_config
    if load_config(root/"data/model_tracks/suite.yaml").dvc_enabled:
        from model_tracks.publish import persist_results
        persist_results(result_archive,{run_tag!r})
else:
    subprocess.run([sys.executable,"-m","model_tracks.run","--config","data/model_tracks/suite.yaml",
                   "--output",{remote_output!r},"--run-tag",{run_tag!r}],cwd=root,env=env,check=True)
'''
    backend.run_detached_stage('all_tracks',['/usr/bin/python3','-c',script],
                               timeout=backend._WORKER_TIMEOUT_SECONDS)
    expected = backend._read_remote_text(remote_output+'.sha256').strip()
    local = RESULTS/'model_tracks'/f'{run_tag}.zip'
    local.parent.mkdir(parents=True,exist_ok=True)
    if not local.exists() or file_hash(local) != expected:
        partial = local.with_suffix('.zip.partial')
        backend._download_one_remote_file(remote_output+'.zip',partial)
        if file_hash(partial) != expected:
            raise ValueError('all-track result download mismatch; partial retained for diagnosis')
        partial.replace(local)
    if file_hash(local)!=expected:
        raise ValueError('all-track result download mismatch')
    manifest=verify_archive(local,'suite_bundle_manifest.json')
    with zipfile.ZipFile(local) as archive:
        for track in ('text','gnn_only','hybrid'):
            marker=json.loads(archive.read(f'{track}/track_complete.json'))
            if marker.get('status')!='ok' or not marker.get('postprocess_complete'):
                raise ValueError(f'{track} incomplete in downloaded results')
    from model_tracks.publish import materialize
    with zipfile.ZipFile(local) as archive:
        suite=json.loads(archive.read('suite_manifest.json'))
    if suite['config']['publish_git'] or suite['config'].get('publish_dvc', False):
        from core.common import TRAIN_ROOT
        receipt = json.loads(backend._read_remote_text(remote_output+'.publication.json'))
        if (receipt.get('run_tag') != run_tag or receipt.get('archive_sha256') != expected
                or receipt.get('verified_download') is not True):
            raise ValueError('suite DVC publication receipt mismatch')
        prefix = f'dvc_refs/{run_tag}/worker_1/'
        receipts = [receipt]
        with zipfile.ZipFile(local) as archive:
            receipts.extend(json.loads(archive.read(member)) for member in archive.namelist()
                if '/_artifact_publications/' in member and member.endswith('.publication.json'))
        for incremental in receipts:
            if not incremental['run_tag'].startswith(run_tag):
                raise ValueError('incremental publication run mismatch')
            prefix = f"dvc_refs/{incremental['run_tag']}/worker_1/"
            for relative, contents in incremental['references'].items():
                target = TRAIN_ROOT / relative
                if not relative.startswith(prefix) or not target.resolve().is_relative_to((TRAIN_ROOT/prefix).resolve()):
                    raise ValueError('unsafe suite DVC reference')
                if target.exists() and target.read_text() != contents:
                    raise ValueError('existing suite DVC reference differs')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(contents)
        local.with_suffix('.publication.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if suite['config']['publish_git']:
            materialize(local,run_tag,push=True)
    return local
