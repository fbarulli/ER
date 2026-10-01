"""All-track adapter; provisioning, locks, polling and teardown stay in cli.colab."""
import json
from pathlib import Path
import zipfile

from core.portable_archive import verify_archive
from graph_tracks.data import file_hash
from model_tracks.package import verify


def run(archive: Path, run_tag: str, *, resume: bool = False, resume_archive: Path | None = None):
    from cli import colab as backend
    from core.common import RESULTS
    metadata = verify(archive)
    with zipfile.ZipFile(archive) as source:
        import yaml
        settings = yaml.safe_load(source.read('data/model_tracks/suite.yaml'))
    from model_tracks.config import SuiteConfig
    settings = SuiteConfig.model_validate(settings).model_dump()
    runtime = getattr(backend, 'GPU', None)
    if runtime is not None and settings['device'] != ('cpu' if runtime.upper() == 'CPU' else 'cuda'):
        raise ValueError('packaged suite device differs from requested Colab runtime')
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_tag):
        raise ValueError('invalid run tag')
    recovery_local = RESULTS / 'model_tracks' / f'{run_tag}.recovery.zip'
    if resume_archive is not None and not resume:
        raise ValueError('recovery archive requires resume')
    if resume and resume_archive is None and recovery_local.exists():
        resume_archive = recovery_local
    remote_recovery = f'{backend.REMOTE_ROOT}/prepared_training/{run_tag}__recovery.zip'
    remote_zip = f'{backend.REMOTE_ROOT}/prepared_training/{run_tag}__all_tracks.zip'
    backend.run_colab_exec_stream(backend.SESSION,
        f'import pathlib\npathlib.Path({remote_zip!r}).parent.mkdir(parents=True,exist_ok=True)\n',
        timeout=120,log_name='tracks_upload_directory',retry_safe=True)
    backend._upload_with_retries(archive,remote_zip,timeout=backend._RESULT_DOWNLOAD_TIMEOUT_SECONDS)
    if resume_archive is not None:
        recovery = verify_archive(resume_archive, 'suite_recovery_manifest.json')
        if recovery.get('run_tag') != run_tag:
            raise ValueError('recovery suite run mismatch')
        original = recovery.get('input_package')
        if not isinstance(original, dict) or any(original.get(key) != metadata.get(key)
                                               for key in ('revision', 'files')):
            raise ValueError('resume package differs from interrupted suite sources or inputs')
        backend._upload_with_retries(resume_archive, remote_recovery,
                                     timeout=backend._RESULT_DOWNLOAD_TIMEOUT_SECONDS)
    remote_output = f'{backend.REMOTE_ROOT}/results/model_tracks/{run_tag}'
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
output_path=pathlib.Path({remote_output!r})
if {resume_archive is not None!r} and not output_path.exists():
    from model_tracks.package import restore_recovery
    restore_recovery(pathlib.Path({remote_recovery!r}),output_path,{run_tag!r})
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
    if {resume!r} and not output_path.exists():
        raise FileNotFoundError("interrupted suite state is unavailable; refusing to restart under its run tag")
    command=[sys.executable,"-m","model_tracks.run","--config","data/model_tracks/suite.yaml",
             "--output",{remote_output!r},"--run-tag",{run_tag!r}]
    if {resume!r} and output_path.exists():
        command.append("--resume")
    subprocess.run(command,cwd=root,env=env,check=True)
'''
    try:
        backend.run_detached_stage('all_tracks',['/usr/bin/python3','-c',script],
                                   timeout=backend._WORKER_TIMEOUT_SECONDS)
    except BaseException:
        # The backend tears down the VM after this returns: collect stopped
        # workers' checkpoints and completed reports first when still reachable.
        try:
            recovery_remote = remote_output + '.recovery.zip'
            recovery_script = backend._BOOTSTRAP + f'''
import json, os, pathlib, signal, time
from model_tracks.package import recovery_package
from graph_tracks.data import file_hash
# Stop only this suite's supervisor and separately owned worker groups before
# reading checkpoints. A lost poll connection can leave detached work alive.
owned=[]
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():
        continue
    try:
        args=proc.joinpath('cmdline').read_bytes().split(b'\\0')
        environment=proc.joinpath('environ').read_bytes().split(b'\\0')
        supervisor=b'model_tracks.run' in args and {remote_output.encode()!r} in args
        worker=any(b'EUROMONITOR_RESULTS_DIR='+{remote_output.encode()!r}+b'/'+track in environment
                   for track in (b'text',b'gnn_only',b'hybrid'))
        if supervisor or worker:
            owned.append(int(proc.name))
            if worker and os.getpgid(int(proc.name)) == int(proc.name):
                os.killpg(int(proc.name),signal.SIGTERM)
            else:
                os.kill(int(proc.name),signal.SIGTERM)
    except (OSError,ValueError):
        pass
for _ in range(100):
    if not any(pathlib.Path('/proc',str(pid)).exists() for pid in owned):
        break
    time.sleep(.1)
for pid in owned:
    try:
        if pathlib.Path('/proc',str(pid)).exists():
            if os.getpgid(pid) == pid:
                os.killpg(pid,signal.SIGKILL)
            else:
                os.kill(pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
destination=pathlib.Path({recovery_remote!r})
if destination.exists():
    destination.unlink()
input_package=json.loads(pathlib.Path({backend.REMOTE_ROOT!r},'model_tracks_package.json').read_text())
recovery_package(pathlib.Path({remote_output!r}),destination,{run_tag!r},input_package=input_package)
destination.with_suffix('.sha256').write_text(file_hash(destination)+'\\n')
'''
            backend.run_colab_exec_stream(backend.SESSION, recovery_script, timeout=300,
                                         log_name='tracks_recovery', retry_safe=True)
            expected_recovery = backend._read_remote_text(str(Path(recovery_remote).with_suffix('.sha256'))).strip()
            recovery_local.parent.mkdir(parents=True, exist_ok=True)
            partial = recovery_local.with_suffix('.zip.partial')
            backend._download_one_remote_file(recovery_remote, partial)
            if file_hash(partial) != expected_recovery:
                raise ValueError('suite recovery download mismatch')
            verify_archive(partial, 'suite_recovery_manifest.json')
            partial.replace(recovery_local)
            print(f'Interrupted suite recovery saved: {recovery_local}', flush=True)
        except BaseException as recovery_error:
            print(f'Interrupted suite recovery unavailable: {recovery_error}', flush=True)
        raise
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
