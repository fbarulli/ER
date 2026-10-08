"""All-track adapter; provisioning, locks, polling and teardown stay in cli.colab."""
import json
from pathlib import Path
import hashlib

from core.portable_archive import verify_archive, verify_archive_digest, verified_archive
from core.archive_reader import open_archive, tar_archive
from graph_tracks.data import file_hash
from model_tracks.package import verify, package_member

RESULT_MANIFEST = 'suite_bundle_manifest.json'


def verify_result_archive(path: Path) -> tuple[dict, str]:
    """Verify a downloaded result archive once and return (manifest, sha256).

    This is the single boundary the result download trusts: the whole-file
    digest is folded into the same streaming pass that checks the manifest and
    every member, so the result archive is read exactly once. Tests stub this
    named seam rather than the underlying core call.
    """
    return verify_archive_digest(path, RESULT_MANIFEST)


def _publish_git_inputs(paths, message: str) -> None:
    """Publish the input transport to the branch the Colab VM actually clones.

    ``push_artifacts`` commits on the current branch and pushes its upstream.
    The VM clones ``colab.branch``, so when the working branch differs from it
    the clone would miss the transport and the remote stage would abort with a
    FileNotFoundError. Re-point the publication at the configured branch with a
    fast-forward push (never forced).
    """
    import subprocess
    from core.common import TRAIN_ROOT, training_cfg
    from model_tracks.publish import push_artifacts
    push_artifacts(paths, message)
    branch = training_cfg().colab.branch
    try:
        head = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                              cwd=TRAIN_ROOT, text=True, capture_output=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return
    if head and head != 'HEAD' and head != branch:
        subprocess.run(['git', 'push', 'origin', f'HEAD:{branch}'],
                       cwd=TRAIN_ROOT, check=True)


def prepare_git_inputs(archive: Path, run_tag: str, *, resume_archive=None, publisher=None):
    """Save immutable inputs through the existing Git artifact publisher."""
    from core.common import TRAIN_ROOT
    metadata = verify(archive)
    files = {'inputs.tar.zst':archive}
    if resume_archive is not None:
        recovery = verify_archive(resume_archive,'suite_recovery_manifest.json')
        if recovery.get('run_tag') != run_tag:
            raise ValueError('recovery suite run mismatch')
        original = recovery.get('input_package')
        if not isinstance(original,dict) or any(original.get(key) != metadata.get(key)
                                               for key in ('revision','files')):
            raise ValueError('resume package differs from interrupted suite sources or inputs')
        files['recovery.tar.zst'] = resume_archive
    inventory = {name:{'sha256':file_hash(path),'size':path.stat().st_size}
                 for name,path in files.items()}
    identity = hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    folder = TRAIN_ROOT/'results/model_tracks/inputs'
    folder.mkdir(parents=True,exist_ok=True)
    transport = folder/f'{identity}.tar.zst'
    if not transport.exists():
        partial = transport.with_suffix('.partial')
        with tar_archive(partial, 'x') as package:
            for name,path in files.items():
                package.add(path,arcname=name,recursive=False)
        partial.replace(transport)
    with tar_archive(transport) as package:
        if set(package.getnames()) != set(files):
            raise ValueError('Git input transport inventory mismatch')
        for name,expected in inventory.items():
            member = package.getmember(name)
            if not member.isfile() or member.size != expected['size']:
                raise ValueError('Git input transport member mismatch')
            with package.extractfile(member) as source:
                if hashlib.file_digest(source,'sha256').hexdigest() != expected['sha256']:
                    raise ValueError('Git input transport checksum mismatch')
    if transport.stat().st_size >= 100*1024**2:
        raise ValueError('Suite input transport exceeds GitHub regular-file limit; reduce the input package size')
    (publisher or _publish_git_inputs)([transport],f'tracks: save immutable GPU inputs {identity[:24]}')
    return transport


def _collect_failure_logs(backend, remote_output: str, run_tag: str):
    """Collect diagnostics even when preflight never produced a suite manifest."""
    from core.common import RESULTS
    from model_tracks.resume import TRACKS
    names = ['suite_events.jsonl']
    for track in TRACKS:
        names.extend([f'{track}__worker.log', f'{track}/worker_events.jsonl'])
    script = f'''import hashlib, json, pathlib
root=pathlib.Path({remote_output!r})
print(json.dumps({{name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                  for name in {names!r} if (root/name).is_file()}}))
'''
    inventory = json.loads(backend.run_colab_exec_capture(backend.SESSION, script, timeout=120))
    folder = RESULTS / 'model_tracks' / f'{run_tag}__logs'
    for name, digest in inventory.items():
        if name not in names:
            raise ValueError('unexpected failure diagnostic path')
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + '.partial')
        backend._download_one_remote_file(remote_output + '/' + name, partial)
        if file_hash(partial) != digest:
            raise ValueError(f'failure log changed during collection: {name}; partial retained')
        partial.replace(destination)
        print(f'Failure log verified: {destination}', flush=True)
    if not inventory:
        print('No remote suite log files were available for collection', flush=True)


def run(archive: Path, run_tag: str, *, resume: bool = False, resume_archive: Path | None = None,
        git_inputs: Path | None = None):
    from cli import colab as backend
    from core.common import RESULTS, TRAIN_ROOT
    with verified_archive(archive, 'model_tracks_package.json') as (source, metadata):
        import yaml
        settings = yaml.safe_load(source.read(package_member('suite_package_config')))
    from model_tracks.config import SuiteConfig
    settings = SuiteConfig.model_validate(settings).model_dump()
    result_suffix = '.' + settings['result_archive_format']
    runtime = getattr(backend, 'GPU', None)
    if runtime is not None and settings['device'] != ('cpu' if runtime.upper() == 'CPU' else 'cuda'):
        raise ValueError('packaged suite device differs from requested Colab runtime')
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_tag):
        raise ValueError('invalid run tag')
    recovery_local = RESULTS / 'model_tracks' / f'{run_tag}.recovery.tar.zst'
    if resume_archive is not None and not resume:
        raise ValueError('recovery archive requires resume')
    if resume and resume_archive is None and recovery_local.exists():
        resume_archive = recovery_local
    remote_recovery = f'{backend.REMOTE_ROOT}/prepared_training/{run_tag}__recovery.tar.zst'
    remote_zip = f"{backend.REMOTE_ROOT}/prepared_training/{run_tag}__all_tracks.{settings['input_archive_format']}"
    git_inputs = git_inputs or prepare_git_inputs(archive,run_tag,resume_archive=resume_archive)
    remote_inputs = backend.REMOTE_ROOT+'/'+git_inputs.resolve().relative_to(TRAIN_ROOT.resolve()).as_posix()
    if resume_archive is not None:
        recovery = verify_archive(resume_archive, 'suite_recovery_manifest.json')
        if recovery.get('run_tag') != run_tag:
            raise ValueError('recovery suite run mismatch')
        original = recovery.get('input_package')
        if not isinstance(original, dict) or any(original.get(key) != metadata.get(key)
                                               for key in ('revision', 'files')):
            raise ValueError('resume package differs from interrupted suite sources or inputs')
    remote_output = f'{backend.REMOTE_ROOT}/results/model_tracks/{run_tag}'
    auth = backend._wandb_env_script()
    script = backend._BOOTSTRAP + auth + f'''
import hashlib, json, os, pathlib, subprocess, sys
from core.archive_reader import tar_archive
from graph_tracks.data import file_hash
root=pathlib.Path({backend.REMOTE_ROOT!r})
archive_path=pathlib.Path({remote_zip!r})
transport=pathlib.Path({remote_inputs!r})
if file_hash(transport) != {file_hash(git_inputs)!r}:
    raise ValueError("cloned Git input transport mismatch")
archive_path.parent.mkdir(parents=True,exist_ok=True)
with tar_archive(transport) as package:
    expected_members={{'inputs.tar.zst'}} | ({{'recovery.tar.zst'}} if {resume_archive is not None!r} else set())
    if set(package.getnames()) != expected_members:
        raise ValueError("cloned Git input inventory mismatch")
    for member_name,destination in [('inputs.tar.zst',archive_path),('recovery.tar.zst',pathlib.Path({remote_recovery!r}))]:
        if member_name not in expected_members:
            continue
        member=package.getmember(member_name)
        if not member.isfile():
            raise ValueError("unsafe Git input member")
        with package.extractfile(member) as source,destination.open('wb') as target:
            import shutil
            shutil.copyfileobj(source,target)
if file_hash(archive_path) != {file_hash(archive)!r}:
    raise ValueError("prepared all-track Git input mismatch")
# The immutable package can predate its transport publication commit. Fetch
# only that revision: a depth-one branch checkout need not contain its parent.

from core.portable_archive import verified_archive, verify_archive, install_data_members
with verified_archive(archive_path,"model_tracks_package.json") as (archive, _):
    for member in archive.infolist():
        if not (root/member.filename).resolve().is_relative_to(root.resolve()):
            raise ValueError("unsafe input package member")
    # The pinned checkout (git checkout --detach <package revision>) is
    # authoritative for src/config/scripts: the package's embedded snapshot is
    # built from the packaging working tree, which can drift from this revision
    # (the transport clone itself is newer), so installing it would clobber the
    # checkout exactly like the Kaggle train-kernel bundle did. Only data
    # installs.
    install_data_members(archive, root)
output_path=pathlib.Path({remote_output!r})
if {resume_archive is not None!r} and not output_path.exists():
    from model_tracks.package import restore_recovery
    restore_recovery(pathlib.Path({remote_recovery!r}),output_path,{run_tag!r})
env={{**os.environ,"PYTHONPATH":str(root/"src"),"PYTHONUNBUFFERED":"1", "ER_GPU_TRAINING_ONLY":"1"}}
result_archive=pathlib.Path({remote_output!r}+{result_suffix!r})
if result_archive.exists():
    # Collection/publication retry must never restart completed training.
    verified=verify_archive(result_archive,"suite_bundle_manifest.json")
    if verified["run_tag"] != {run_tag!r}:
        raise ValueError("existing suite result run mismatch")

else:
    if {resume!r} and not output_path.exists():
        raise FileNotFoundError("interrupted suite state is unavailable; refusing to restart under its run tag")
    command=[sys.executable,"-m","model_tracks.run","--config",{package_member("suite_package_config")!r},
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
            recovery_remote = remote_output + '.recovery.tar.zst'
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
                   for track in (b'text',b'gnn_only',b'cascade'))
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
            partial = recovery_local.with_name(recovery_local.name + '.partial')
            backend._download_one_remote_file(recovery_remote, partial)
            if file_hash(partial) != expected_recovery:
                raise ValueError('suite recovery download mismatch')
            verify_archive(partial, 'suite_recovery_manifest.json')
            partial.replace(recovery_local)
            print(f'Interrupted suite recovery saved: {recovery_local}', flush=True)
        except BaseException as recovery_error:
            print(f'Interrupted suite recovery unavailable: {recovery_error}', flush=True)
        try:
            _collect_failure_logs(backend, remote_output, run_tag)
        except Exception as log_error:
            print(f'Failure diagnostic collection unavailable: {log_error}; inspect local Colab stage log', flush=True)
        raise
    expected = backend._read_remote_text(remote_output+'.sha256').strip()
    local = RESULTS/'model_tracks'/f'{run_tag}.training{result_suffix}'
    local.parent.mkdir(parents=True,exist_ok=True)
    # One download + one streaming verify (owner #5): verify_archive_digest folds
    # the whole-file SHA256 into the manifest/member verification pass, so the
    # ~1 GB result archive is read exactly once. A matching local archive skips
    # the download entirely; a corrupt partial is retained for diagnosis.
    manifest = None
    observed = None
    if local.exists():
        try:
            manifest, observed = verify_result_archive(local)
        except ValueError:
            manifest, observed = None, None
    if observed != expected:
        partial = local.with_name(local.name + '.partial')
        backend._download_one_remote_file(remote_output+result_suffix, partial)
        manifest, observed = verify_result_archive(partial)
        if observed != expected:
            raise ValueError('all-track result download mismatch; partial retained for diagnosis')
        partial.replace(local)
    if manifest.get('run_tag') != run_tag:
        raise ValueError('all-track result archive run mismatch')
    print(f'[tracks] Direct result archive download verified: {local}', flush=True)
    # Release GPU quota before local inference, indexing, reporting or publishing.
    # stop() is deliberately non-raising for launcher finally blocks. Require
    # a verified release here so reporting cannot overlap an idle GPU session.
    import time
    for release_attempt in range(3):
        if backend.stop() is True:
            break
        if release_attempt < 2:
            time.sleep(2 * (release_attempt + 1))
    else:
        raise RuntimeError(
            f"Colab session {backend.SESSION!r} termination could not be verified; "
            f"CPU postprocessing refused. Result handoff retained beside {local}. "
            f"Release the session with colab stop -s {backend.SESSION} before completing locally."
        )
    from model_tracks.resume import TRACKS, expected_postprocess, validate_archived_track
    with open_archive(local) as result:
        for track in TRACKS:
            # The remote stage always trains GPU-only: trained lanes defer their
            # CPU reports, the cascade already finished its composed report.
            validate_archived_track(result, manifest, track,
                                    postprocess_complete=expected_postprocess(track, gpu_only=True))
    from model_tracks.snapshot_completion import complete
    return complete(local, archive, run_tag, publish=False)
