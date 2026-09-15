"""Incremental result sync: fetch finished artifacts while training runs.

The syncer is a latency optimisation, never a correctness dependency.  These
tests pin both halves of that contract: it must move artifacts as they become
complete, and it must never capture a half-written file or fail the run when
the remote misbehaves.
"""

from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest import mock

from cli import colab


class _FakeRemote:
    """A scripted remote: known files and sizes, plus a download recorder."""

    def __init__(self, files: dict[str, int]) -> None:
        self.files = files
        self.downloaded: list[str] = []
        self.download_fails: set[str] = set()

    def list_remote(self, _directory: str) -> list[str]:
        return sorted(self.files)

    def file_size(self, remote: str) -> int:
        if remote not in self.files:
            raise RuntimeError(f"no such remote file: {remote}")
        return self.files[remote]

    def download_one(self, remote: str, local: Path) -> None:
        if remote in self.download_fails:
            raise RuntimeError("simulated download failure")
        self.downloaded.append(remote)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("payload")

    def patched(self):
        return mock.patch.multiple(
            colab,
            _list_remote=self.list_remote,
            _remote_file_size=self.file_size,
            _download_one_remote_file=self.download_one,
        )


class IncrementalResultSyncTests(unittest.TestCase):
    REMOTE = "/content/EuromonitoR/results/concurrent_train_20260101T000000Z"
    RUN_ID = "20260101T000000Z"

    def _syncer(self, remote: _FakeRemote):
        return colab._IncrementalResultSync(self.REMOTE, self.RUN_ID, workers=1)

    def test_a_settled_file_is_transferred_once(self):
        remote = _FakeRemote({
            f"{self.REMOTE}/worker_1/dvc_manifest.json": 120,
        })
        with remote.patched(), mock.patch.object(colab, "TRAINING_RESULTS", Path("/tmp/sync-test")):
            syncer = self._syncer(remote)
            # First pass observes the file; a second confirms it is not growing.
            syncer._pass()
            self.assertEqual(remote.downloaded, [], "must not fetch on first sighting")
            syncer._pass()
            self.assertEqual(len(remote.downloaded), 1)
            # A third pass must not re-fetch what it already has.
            syncer._pass()
            self.assertEqual(len(remote.downloaded), 1)
            self.assertEqual(syncer.synced_bytes(), 120)

    def test_a_growing_file_is_never_captured_mid_write(self):
        remote = _FakeRemote({
            f"{self.REMOTE}/worker_1/checkpoint/trainer_state.json": 10,
        })
        with remote.patched(), mock.patch.object(colab, "TRAINING_RESULTS", Path("/tmp/sync-test")):
            syncer = self._syncer(remote)
            syncer._pass()
            remote.files[f"{self.REMOTE}/worker_1/checkpoint/trainer_state.json"] = 99
            syncer._pass()
            self.assertEqual(remote.downloaded, [], "size changed: still being written")
            syncer._pass()
            self.assertEqual(len(remote.downloaded), 1, "stable size: now safe to fetch")

    def test_excluded_directories_are_not_transferred(self):
        remote = _FakeRemote({
            f"{self.REMOTE}/worker_1/_checkpoints/checkpoint-1/optimizer.pt": 5,
            f"{self.REMOTE}/worker_1/.dvc-cache/abc": 5,
            f"{self.REMOTE}/worker_1/wandb/run.log": 5,
            f"{self.REMOTE}/worker_1/sku_predictions.csv": 5,
        })
        with remote.patched(), mock.patch.object(colab, "TRAINING_RESULTS", Path("/tmp/sync-test")):
            syncer = self._syncer(remote)
            syncer._pass()
            syncer._pass()
            self.assertEqual(
                [Path(p).name for p in remote.downloaded],
                ["sku_predictions.csv"],
            )

    def test_a_download_failure_never_propagates(self):
        target = f"{self.REMOTE}/worker_1/sku_predictions.csv"
        remote = _FakeRemote({target: 5})
        remote.download_fails.add(target)
        with remote.patched(), mock.patch.object(colab, "TRAINING_RESULTS", Path("/tmp/sync-test")):
            syncer = self._syncer(remote)
            syncer._pass()
            syncer._pass()  # must not raise
            self.assertEqual(syncer.synced_bytes(), 0)

    def test_a_listing_failure_never_propagates(self):
        with mock.patch.object(
            colab, "_list_remote", side_effect=RuntimeError("control channel down"),
        ):
            syncer = colab._IncrementalResultSync(self.REMOTE, self.RUN_ID, workers=1)
            syncer._pass()  # must not raise

    def test_stop_is_prompt_and_idempotent(self):
        remote = _FakeRemote({})
        with remote.patched(), mock.patch.object(colab, "TRAINING_RESULTS", Path("/tmp/sync-test")):
            syncer = self._syncer(remote)
            syncer.start()
            syncer.stop()
            syncer.stop()  # a second stop must not hang or raise


if __name__ == "__main__":
    unittest.main()
