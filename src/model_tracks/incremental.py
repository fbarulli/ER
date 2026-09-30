"""Publish immutable artifact generations while model workers continue."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil

from core.portable_archive import write_archive
from model_tracks.publish import persist_results


def _publish(archive, tag):
    # Linux priorities are per-thread; children launched by this upload
    # thread inherit its lower priority, without slowing the trainer thread.
    if hasattr(os, 'setpriority'):
        import threading
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
    receipt = persist_results(archive, tag)
    # Verified remote storage and the receipt retain the generation; avoid
    # accumulating duplicate checkpoint bytes on the Colab disk.
    shutil.rmtree(archive.with_suffix('.publication'))
    archive.unlink()
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
        directory = self.output / '_artifact_publications'
        directory.mkdir(exist_ok=True)
        files = {}
        for path in paths:
            for item in ([path] if path.is_file() else path.rglob('*')):
                if item.is_file() and not item.is_symlink():
                    files[item.relative_to(self.output).as_posix()] = item
        # Snapshot before returning: checkpoint rotation and mutable reports
        # cannot change the bytes read by the background DVC publisher.
        archive = write_archive(directory / f'{tag}.zip', files,
                                manifest_name='suite_bundle_manifest.json', metadata={'run_tag': tag})
        self.futures.append(self.executor.submit(_publish, archive, tag))

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
