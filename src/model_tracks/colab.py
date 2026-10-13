"""All-track Colab adapter; provisioning, locks, polling and teardown stay in cli.colab.

``TracksLane`` owns the all-track Colab lane's BEHAVIOR: it is the one place a
suite archive crosses to/from the VM — the immutable inputs transport
(``prepare_git_inputs``), the result download (``run``), the recovery receipt
(``run`` + ``_check_recovery_run``) and the interrupted-run log receipt
(``collect_failure_logs``). Consumers build the lane and call it; nothing
re-derives a transport name, a member inventory or a download token at a call
site.

Every archive is opened through its ONE ``Bundle`` load. A bundle is IMMUTABLE
and trusted: no content hash, byte size or existence value is compared anywhere
in this lane (owner directive: data is never checked).

Module level keeps only the thin entry points the launcher and the resume
builder call (``run`` / ``prepare_git_inputs``).
"""
import json
from pathlib import Path

from core.bundle import Bundle, BundleRole, bundle_spec, manifest_name
from core.archive_reader import tar_archive
from core.manifest import publish_replacing
from model_tracks.package import verify, package_member


class TracksLane:
    """The all-track Colab lane: transport, download, install and receipt.

    One instance describes one suite run (``archive`` + ``run_tag``); every
    crossing of the VM boundary is a method here, so the lane's behavior is
    owned in one place instead of being re-spelled at each call site.
    """

    def __init__(self, *, archive: Path, run_tag: str, resume: bool = False,
                 resume_archive: Path | None = None,
                 git_inputs: Path | None = None) -> None:
        self.archive = Path(archive)
        self.run_tag = run_tag
        self.resume = resume
        self.resume_archive = resume_archive
        self.git_inputs = git_inputs

    # ------------------------------------------------------------- result read
    @staticmethod
    def verify_result_archive(path: Path) -> tuple[dict, str]:
        """Open a downloaded result archive once and return (manifest, size).

        This is the single boundary the result download trusts: the manifest is read
        once and the archive is then trusted (no member byte is compared).
        """
        handle = Bundle.load(path, BundleRole.result)
        return handle.manifest, handle.path.stat().st_size

    # ---------------------------------------------------------------- recovery
    def _check_recovery_run(self, recovery: dict) -> None:
        """The ONE resume-provenance check: the recovery belongs to this run.

        Only the run tag is compared (a structural contract). The recorded
        ``input_package`` is a RECORD of what the interrupted run used, never
        compared to refuse: a bundle is immutable, so a change in data yields a new
        bundle instead (owner directive: data is never checked).
        """
        if recovery.get('run_tag') != self.run_tag:
            raise ValueError('recovery suite run mismatch')

    # --------------------------------------------------------------- publishing
    def _publish_git_inputs(self, paths, message: str) -> None:
        """Archive the input transport, then push its pointer to the cloned branch.

        The transport bytes are archived in the dagshub DVC remote (write-only
        storage) while git keeps only the ``*.dvc`` pointer; the VM never pulls
        from DVC, the launcher uploads the local archive to it. The pointer must
        still reach the branch the VM clones, so the commit is pushed explicitly
        to ``colab.branch`` (a fast-forward, never forced).
        """
        import subprocess
        from core.common import TRAIN_ROOT, training_cfg
        from model_tracks.run_retention import publish_result_paths

        publish_result_paths(paths)
        branch = training_cfg().colab.branch
        subprocess.run(['git', 'push', 'origin', f'HEAD:{branch}'],
                       cwd=TRAIN_ROOT, check=True)

    def prepare_git_inputs(self, recovery: dict | None = None, publisher=None) -> Path:
        """Save this lane's immutable inputs through the DVC-backed publisher.

        ``recovery`` is the ALREADY loaded recovery manifest (the caller that
        opened the same archive for its provenance check passes it down), so a
        resume run reads the recovery archive exactly once.

        Every archive name and the transport directory come from the ONE data
        bundle declaration (``ColabSpec.data_bundle``), the same declaration both
        Colab lanes read. The transport is named by the run tag (a structural
        value); nothing about its bytes is compared. The transport is identified
        only by its run tag and reused when it already exists, so relaunching a
        run neither rebuilds nor re-pushes it.
        """
        from core.common import TRAIN_ROOT, training_cfg
        bundle = training_cfg().colab.data_bundle
        verify(self.archive)
        files = {bundle.transport_member: self.archive}
        resume_archive = self.resume_archive
        if resume_archive is not None:
            recovery = (recovery if recovery is not None
                        else Bundle.load(resume_archive, BundleRole.recovery).manifest)
            self._check_recovery_run(recovery)
            files[bundle.recovery_member] = resume_archive
        identity = self.run_tag
        folder = TRAIN_ROOT/bundle.git_transport_dir
        folder.mkdir(parents=True,exist_ok=True)
        transport = folder/bundle.git_transport_name(identity)
        if not transport.exists():
            partial = transport.with_suffix('.partial')
            with tar_archive(partial, 'x') as package:
                for name,path in files.items():
                    package.add(path,arcname=name,recursive=False)
            # Publish the completed sibling through the ONE publish helper (fsync +
            # os.replace), never a bare rename of a possibly-unflushed file.
            publish_replacing(partial, transport)
        (publisher or self._publish_git_inputs)(
            [transport], f'tracks: save immutable GPU inputs {identity[:24]}')
        return transport

    # ----------------------------------------------------------- failure receipt
    def collect_failure_logs(self, backend, remote_output: str) -> None:
        """Collect diagnostics even when preflight never produced a suite manifest.

        The remote probe reports which log files exist; each is downloaded and
        published as the interrupted run's receipt. No downloaded byte count is
        compared (owner directive: data is never checked).
        """
        from core.common import RESULTS
        from core.bundle import bundle_spec
        from model_tracks.resume import TRACKS
        spec = bundle_spec()
        names = [spec.suite_events_file]
        for track in TRACKS:
            names.extend([f'{track}__worker.log', f'{track}/{spec.worker_events_file}'])
        # PINNED STANDALONE COPY: `run_colab_exec_capture` executes this probe
        # verbatim (no `_BOOTSTRAP`, so the checkout is not on sys.path) and it
        # reports the present log names, which the local side receipts.
        script = f'''import json, pathlib
root=pathlib.Path({remote_output!r})
print(json.dumps([name for name in {names!r} if (root/name).is_file()]))
'''
        present = json.loads(backend.run_colab_exec_capture(backend.SESSION, script, timeout=120))
        folder = RESULTS / 'model_tracks' / f'{self.run_tag}__logs'
        for name in present:
            if name not in names:
                raise ValueError('unexpected failure diagnostic path')
            destination = folder / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + '.partial')
            backend._download_one_remote_file(remote_output + '/' + name, partial)
            publish_replacing(partial, destination)
            print(f'Failure log saved: {destination}', flush=True)
        if not present:
            print('No remote suite log files were available for collection', flush=True)

    # -------------------------------------------------------------------- lane
    def run(self):
        """Run one all-track suite on the VM and collect its result archive."""
        from cli import colab as backend
        from core.common import RESULTS, TRAIN_ROOT, training_cfg
        bundle = training_cfg().colab.data_bundle
        inputs_bundle = Bundle.load(self.archive, BundleRole.inputs)
        with inputs_bundle.reader() as source:
            metadata = inputs_bundle.manifest
            import yaml
            settings = yaml.safe_load(source.read(package_member('suite_package_config')))
        from model_tracks.config import SuiteConfig
        settings = SuiteConfig.model_validate(settings).model_dump()
        result_suffix = '.' + settings['result_archive_format']
        import re
        run_tag = self.run_tag
        if not re.fullmatch(r'[A-Za-z0-9_-]+', run_tag):
            raise ValueError('invalid run tag')
        recovery_local = RESULTS / 'model_tracks' / bundle.recovery_archive_name(run_tag)
        resume_archive = self.resume_archive
        if resume_archive is not None and not self.resume:
            raise ValueError('recovery archive requires resume')
        if self.resume and resume_archive is None and recovery_local.exists():
            resume_archive = recovery_local
        remote_recovery = f'{backend.REMOTE_ROOT}/{bundle.remote_dir}/{bundle.remote_recovery_name(run_tag)}'
        remote_zip = f"{backend.REMOTE_ROOT}/{bundle.remote_dir}/{bundle.remote_archive_name(run_tag, settings['input_archive_format'])}"
        # Opened ONCE here (the boundary that owns the recovery role) and handed to
        # the transport builder, so a resume run never reads the archive twice.
        recovery = (Bundle.load(resume_archive, BundleRole.recovery).manifest
                    if resume_archive is not None else None)
        git_inputs = self.git_inputs or TracksLane(
            archive=self.archive, run_tag=run_tag,
            resume_archive=resume_archive).prepare_git_inputs(recovery=recovery)
        remote_inputs = backend.REMOTE_ROOT+'/'+git_inputs.resolve().relative_to(TRAIN_ROOT.resolve()).as_posix()
        if resume_archive is not None:
            self._check_recovery_run(recovery)
        remote_output = f'{backend.REMOTE_ROOT}/results/model_tracks/{run_tag}'
        auth = backend._wandb_env_script()
        script = backend._BOOTSTRAP + auth + f'''
import json, os, pathlib, subprocess, sys
from core.archive_reader import tar_archive
root=pathlib.Path({backend.REMOTE_ROOT!r})
archive_path=pathlib.Path({remote_zip!r})
transport=pathlib.Path({remote_inputs!r})
transport.parent.mkdir(parents=True,exist_ok=True)
archive_path.parent.mkdir(parents=True,exist_ok=True)
with tar_archive(transport) as package:
    expected_members={{{bundle.transport_member!r}}} | ({{{bundle.recovery_member!r}}} if {resume_archive is not None!r} else set())
    for member_name,destination in [({bundle.transport_member!r},archive_path),({bundle.recovery_member!r},pathlib.Path({remote_recovery!r}))]:
        if member_name not in expected_members:
            continue
        member=package.getmember(member_name)
        if not member.isfile():
            raise ValueError("unsafe Git input member")
        with package.extractfile(member) as source,destination.open('wb') as target:
            import shutil
            shutil.copyfileobj(source,target)
# The immutable package can predate its transport publication commit. Fetch
# only that revision: a depth-one branch checkout need not contain its parent.

from core.bundle import Bundle, BundleRole
from core.portable_archive import install_data_members
with Bundle.load(archive_path,BundleRole.inputs).reader() as archive:
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
    verified=Bundle.load(result_archive,BundleRole.result).manifest
    if verified["run_tag"] != {run_tag!r}:
        raise ValueError("existing suite result run mismatch")

else:
    if {self.resume!r} and not output_path.exists():
        raise FileNotFoundError("interrupted suite state is unavailable; refusing to restart under its run tag")
    command=[sys.executable,"-m","model_tracks.run","--config",{package_member("suite_package_config")!r},
             "--output",{remote_output!r},"--run-tag",{run_tag!r}]
    if {self.resume!r} and output_path.exists():
        command.append("--resume")
    subprocess.run(command,cwd=root,env=env,check=True)
'''
        # The transport is an INPUT the lane consumes: materialize it directly on
        # the VM. DVC is write-only storage (owner mandate 2026-10-09), so no lane
        # pulls an input back from it; the launcher uploads the local archive that
        # prepare_git_inputs produced and archived.
        backend._upload_with_retries(
            git_inputs, remote_inputs, timeout=backend._RESULT_DOWNLOAD_TIMEOUT_SECONDS)
        try:
            backend.run_detached_stage('all_tracks',['/usr/bin/python3','-c',script],
                                       timeout=backend._WORKER_TIMEOUT_SECONDS)
        except BaseException:
            # The backend tears down the VM after this returns: collect stopped
            # workers' checkpoints and completed reports first when still reachable.
            try:
                recovery_remote = f'{backend.REMOTE_ROOT}/results/model_tracks/{bundle.recovery_archive_name(run_tag)}'
                recovery_script = backend._BOOTSTRAP + f'''
import json, os, pathlib, signal, time
from model_tracks.package import recovery_package
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
input_package=json.loads(pathlib.Path({backend.REMOTE_ROOT!r},{manifest_name(BundleRole.inputs)!r}).read_text())
recovery_package(pathlib.Path({remote_output!r}),destination,{run_tag!r},input_package=input_package)
'''
                backend.run_colab_exec_stream(backend.SESSION, recovery_script, timeout=300,
                                             log_name='tracks_recovery', retry_safe=True)
                recovery_local.parent.mkdir(parents=True, exist_ok=True)
                partial = recovery_local.with_name(recovery_local.name + '.partial')
                backend._download_one_remote_file(recovery_remote, partial)
                verify_recovery=Bundle.load(partial,BundleRole.recovery)
                if verify_recovery.run_tag() != run_tag:
                    raise ValueError('suite recovery run mismatch')
                publish_replacing(partial, recovery_local)
                print(f'Interrupted suite recovery saved: {recovery_local}', flush=True)
            except BaseException as recovery_error:
                print(f'Interrupted suite recovery unavailable: {recovery_error}', flush=True)
            try:
                self.collect_failure_logs(backend, remote_output)
            except Exception as log_error:
                print(f'Failure diagnostic collection unavailable: {log_error}; inspect local Colab stage log', flush=True)
            raise
        local = RESULTS/'model_tracks'/f'{run_tag}.training{result_suffix}'
        local.parent.mkdir(parents=True,exist_ok=True)
        # One download + one boundary read (owner directive: data is never checked):
        # an existing local archive is read once and reused when its manifest parses;
        # otherwise it is downloaded and read once.
        manifest = None
        if local.exists():
            try:
                manifest, _ = self.verify_result_archive(local)
            except ValueError:
                manifest = None
        if manifest is None:
            partial = local.with_name(local.name + '.partial')
            backend._download_one_remote_file(remote_output+result_suffix, partial)
            manifest, _ = self.verify_result_archive(partial)
            publish_replacing(partial, local)
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
        # The download boundary read these bytes (``verify_result_archive``); the
        # track contract reads the SAME archive through the trusted handle, so the
        # result archive is parsed exactly once per VM crossing.
        with Bundle.trusted(local, BundleRole.result, manifest).reader() as result:
            for track in TRACKS:
                # The remote stage always trains GPU-only: trained lanes defer their
                # CPU reports, the cascade already finished its composed report.
                validate_archived_track(result, manifest, track,
                                        postprocess_complete=expected_postprocess(track, gpu_only=True))
        from model_tracks.snapshot_completion import complete
        final = complete(local, self.archive, run_tag, publish=False)
        # Run RESULT retention (owner goal 2026-10-09): the downloadable training
        # archive and the sealed completion archive go to the dagshub DVC remote;
        # git keeps only their *.dvc pointers, and the local payloads are freed so
        # a finished run no longer occupies local disk (dvc pull restores either).
        from model_tracks.run_retention import publish_result_paths
        publish_result_paths([local, final], drop_local=True)
        return final


def run(archive: Path, run_tag: str, *, resume: bool = False, resume_archive: Path | None = None,
        git_inputs: Path | None = None):
    """Entry point: build the one all-track lane and run it (behavior lives there)."""
    return TracksLane(archive=archive, run_tag=run_tag, resume=resume,
                      resume_archive=resume_archive, git_inputs=git_inputs).run()


def prepare_git_inputs(archive: Path, run_tag: str, *, resume_archive=None,
                       recovery: dict | None = None, publisher=None):
    """Entry point: build the lane and publish its immutable inputs transport."""
    return TracksLane(archive=archive, run_tag=run_tag,
                      resume_archive=resume_archive).prepare_git_inputs(
                          recovery=recovery, publisher=publisher)
