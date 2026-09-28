"""Incremental result sync: fetch finished artifacts while training runs.

The syncer is a latency optimisation, never a correctness dependency.  These
tests pin both halves of that contract against the live heartbeat design: a
remote checkpoint is copied only once its manifest exists (the trainer
writes it last), only when its score beats what is held, and every failure
is swallowed so the authoritative end-of-run download stays the source of
truth.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cli import colab


class _FakeRemote:
    """A scripted remote: heartbeat text, directory listings, downloads."""

    def __init__(
        self,
        texts: dict[str, str] | None = None,
        listings: dict[str, list[str]] | None = None,
    ) -> None:
        self.texts = texts or {}
        self.listings = listings or {}
        self.downloaded: list[str] = []
        self.download_fails: set[str] = set()
        self.listed: list[str] = []

    def read_text(self, remote: str) -> str:
        if remote not in self.texts:
            raise RuntimeError(f"no such remote file: {remote}")
        return self.texts[remote]

    def list_remote(self, directory: str, max_depth: int | None = None) -> list[str]:
        self.listed.append(directory)
        if directory in self.listings:
            return sorted(self.listings[directory])
        # The syncer first lists the worker's _checkpoints root to find the
        # heartbeat's checkpoint, then lists that checkpoint directory to
        # check for its manifest and download its files. A root listing should
        # therefore also answer narrower directory queries.
        prefix = directory.rstrip("/") + "/"
        for root, files in self.listings.items():
            if directory.startswith(root.rstrip("/") + "/"):
                return sorted(name for name in files if name.startswith(prefix))
        return []

    def download_one(self, remote: str, local: Path) -> None:
        if remote in self.download_fails:
            raise RuntimeError("simulated download failure")
        self.downloaded.append(remote)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("payload")

    def patched(self):
        return mock.patch.multiple(
            colab,
            _read_remote_text=self.read_text,
            _list_remote=self.list_remote,
            _download_one_remote_file=self.download_one,
        )


class IncrementalResultSyncTests(unittest.TestCase):
    REMOTE = "/content/EuromonitoR/results/concurrent_train_20260101T000000Z"
    RUN_ID = "20260101T000000Z"

    def _syncer(self, remote: _FakeRemote):
        return colab._IncrementalResultSync(self.REMOTE, self.RUN_ID, workers=1)

    def _checkpoint_files(self, step: int) -> tuple[str, list[str]]:
        directory = (
            f"{self.REMOTE}/worker_1/_checkpoints/model/run_f0/checkpoint-{step}"
        )
        return directory, [
            f"{directory}/model.safetensors",
            f"{directory}/checkpoint_manifest.json",
        ]

    def _heartbeat(self, step: int, score: float) -> dict[str, str]:
        import json

        return {
            f"{self.REMOTE}/worker_1/live_status.json": json.dumps(
                {"step": step, "dev_average_precision": score}
            )
        }

    def _root_listing(self, files: list[str]) -> dict[str, list[str]]:
        return {f"{self.REMOTE}/worker_1/_checkpoints": files}

    def test_manifest_gated_checkpoint_is_copied_once(self):
        directory, files = self._checkpoint_files(10)
        remote = _FakeRemote(
            texts=self._heartbeat(10, 0.90),
            listings=self._root_listing(files),
        )
        with tempfile.TemporaryDirectory() as tmp, remote.patched(), \
                mock.patch.object(colab, "TRAINING_RESULTS", Path(tmp)):
            syncer = self._syncer(remote)
            # First pass copies a manifest-complete checkpoint...
            syncer._pass()
            self.assertEqual(len(remote.downloaded), 2)
            self.assertGreater(syncer.synced_bytes(), 0)
            # ...later passes must not re-fetch what is already held.
            syncer._pass()
            syncer._pass()
            self.assertEqual(len(remote.downloaded), 2)

    def test_checkpoint_without_manifest_is_never_captured(self):
        directory, files = self._checkpoint_files(10)
        files = [name for name in files if not name.endswith("checkpoint_manifest.json")]
        remote = _FakeRemote(
            texts=self._heartbeat(10, 0.90),
            listings=self._root_listing(files),
        )
        with tempfile.TemporaryDirectory() as tmp, remote.patched(), \
                mock.patch.object(colab, "TRAINING_RESULTS", Path(tmp)):
            syncer = self._syncer(remote)
            syncer._pass()
            syncer._pass()
            syncer._pass()
            self.assertEqual(
                remote.downloaded, [],
                "manifest absent: checkpoint still being written",
            )
            self.assertEqual(syncer.synced_bytes(), 0)

    def test_only_the_checkpoint_subtree_is_listed_and_worse_scores_do_not_replace(self):
        old_dir, old_files = self._checkpoint_files(10)
        new_dir, new_files = self._checkpoint_files(11)
        remote = _FakeRemote(
            texts=self._heartbeat(10, 0.90),
            listings=self._root_listing(old_files + new_files),
        )
        with tempfile.TemporaryDirectory() as tmp, remote.patched(), \
                mock.patch.object(colab, "TRAINING_RESULTS", Path(tmp)):
            syncer = self._syncer(remote)
            syncer._pass()
            self.assertEqual(len(remote.downloaded), 2)
            # A worse score must not replace the held best...
            remote.texts.update(self._heartbeat(11, 0.50))
            syncer._pass()
            self.assertEqual(len(remote.downloaded), 2)
            # ...and no pass may walk outside the checkpoint subtree.
            root = f"{self.REMOTE}/worker_1/_checkpoints"
            self.assertTrue(remote.listed)
            self.assertTrue(
                all(entry == root or entry.startswith(root + "/") for entry in remote.listed),
                f"run-tree walk detected: {remote.listed}",
            )
            # A better score supersedes through the staging swap.
            remote.texts.update(self._heartbeat(12, 0.95))
            latest_dir, latest_files = self._checkpoint_files(12)
            remote.listings[root] = sorted(old_files + new_files + latest_files)
            syncer._pass()
            self.assertEqual(len(remote.downloaded), 4)

    def test_a_download_failure_never_propagates(self):
        directory, files = self._checkpoint_files(10)
        remote = _FakeRemote(
            texts=self._heartbeat(10, 0.90),
            listings=self._root_listing(files),
        )
        remote.download_fails.update(files)
        with tempfile.TemporaryDirectory() as tmp, remote.patched(), \
                mock.patch.object(colab, "TRAINING_RESULTS", Path(tmp)):
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
        with tempfile.TemporaryDirectory() as tmp, remote.patched(), \
                mock.patch.object(colab, "TRAINING_RESULTS", Path(tmp)):
            syncer = self._syncer(remote)
            syncer.start()
            syncer.stop()
            syncer.stop()  # a second stop must not hang or raise


if __name__ == "__main__":
    unittest.main()
