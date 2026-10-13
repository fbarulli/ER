"""Publish immutable artifact generations while model workers continue."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil

from core.archive_reader import archive_sidecar, archive_settings
from core.bundle import bundle_spec
from core.run_log import RunLogger
from model_tracks.publish import persist_results

_LOG = RunLogger(__name__)


def _publish(bundle, tag):
    # Linux priorities are per-thread; children launched by this upload
    # thread inherit its lower priority, without slowing the trainer thread.
    if hasattr(os, 'setpriority'):
        import threading
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
    # ``bundle`` is the writer's verified handle: persist_results reads the
    # archive's size and run tag off it, so the bytes this thread just sealed
    # are never re-opened (one integrity check per archive per VM crossing).
    receipt = persist_results(bundle.path, tag, bundle=bundle)
    # Verified remote storage and the receipt retain the generation; avoid
    # accumulating duplicate checkpoint bytes on the Colab disk.
    shutil.rmtree(archive_sidecar(bundle.path, bundle_spec().publication_sidecar_suffix))
    bundle.path.unlink()
    return receipt


class ArtifactPublisher:
    def __init__(self, output: Path):
        self.output = output
        self.enabled = os.environ.get('ER_INCREMENTAL_DVC') == '1'
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='artifact-dvc') if self.enabled else None
        self.futures = []

    def submit(self, generation: str, paths: list[Path]):
        if not self.enabled:
            return
        self.check()
        tag = f"{os.environ['EUROMONITOR_RUN_ID']}-{generation}"
        files = {}
        for path in paths:
            for item in ([path] if path.is_file() else path.rglob('*')):
                if item.is_file() and not item.is_symlink():
                    files[item.relative_to(self.output).as_posix()] = item
        if not files:
            # A generation with no files has nothing to publish, and an empty
            # bundle is not a thing (Bundle.seal_archive refuses it): the call
            # is an explicit no-op, never a crash on a smoke invocation.
            _LOG.info(f'[incremental] generation {generation} has no files; '
                      'nothing to publish')
            return
        directory = self.output / bundle_spec().artifact_publications_dir
        directory.mkdir(exist_ok=True)
        # Snapshot before returning: checkpoint rotation and mutable reports
        # cannot change the bytes read by the background DVC publisher. The
        # Bundle owns the writer (size-while-writing + one verify).
        from core.bundle import Bundle, BundleRole
        with _LOG.section('incremental.archive', files=len(files), generation=generation):
            sealed = Bundle.seal_archive(
                directory / f'{tag}.{archive_settings().format}', files,
                role=BundleRole.result, metadata={bundle_spec().run_tag_key: tag})
        # Ship the writer's handle, not its path: the background publisher reads
        # the transport size and run tag from it without re-verifying bytes the
        # sealing writer already sized while writing.
        self.futures.append(self.executor.submit(_publish, sealed, tag))

    def check(self):
        for future in self.futures:
            if future.done():
                future.result()

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        for future in self.futures:
            future.result()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
