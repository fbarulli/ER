"""A default-suite launch reuses the canonical sealed package.

The operator line `er-colab --what tracks --tracks-config config/model_tracks.yaml`
names the DEFAULT suite explicitly (config SSOT: bundle.suite_config). It must
still take the verify+copy reuse path when a full-cohort prepared package was
discovered — the 2026-10-12 crash class was exactly this launch repackaging
locally for ~26 minutes because the canonical branch only fired when
--tracks-config was absent. Public behavior pinned: with discovery returning a
run dir, main() verifies+copies its archive and never calls suite_package; the
GitHub-release route (the transport default) then derives the published-bundle
reference from the canonical package and stops at the pre-provision branch sync.

No VM, no network, no real 447MB file: discovery is steered to a tmp dir carrying
the SSOT archive name, resolve runs for real (it must accept what discovery
returns), and the release-route branch-sync boundary (sync_launch_branch) stops
the launch before provisioning.
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest

from cli import colab


def test_explicit_default_suite_reuses_the_canonical_package(tmp_path):
    run_dir = tmp_path / "1012T000000000000Z"
    run_dir.mkdir()
    archive = run_dir / colab._BUNDLE_ARCHIVE
    archive.write_bytes(b"sealed")
    calls: dict[str, object] = {}

    def fake_package(_config, _dest):
        calls["packaged"] = True
        return _dest

    def stop_at_branch_sync(*_args, **_kwargs):
        # The release route's only pre-provision remote step is the branch sync;
        # raising here proves the launch reached it with the reuse path taken.
        raise RuntimeError("unit-c-stop")

    argv = ["colab.py", "--what", "tracks",
            "--tracks-config", str(colab._suite_config_path()),
            "--gpu", "T4", "--allow-gpu"]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.object(colab, "acquire_colab_launch_lock", return_value=None), \
         mock.patch.object(colab, "release_colab_launch_lock"), \
         mock.patch.object(colab, "start_live_log"), \
         mock.patch.object(colab, "default_prepared_input_package", return_value=run_dir), \
         mock.patch("model_tracks.package.package", fake_package), \
         mock.patch("model_tracks.package.verify", return_value={}), \
         mock.patch("model_tracks.package.packaged_suite_identity", return_value={"i": 1}), \
         mock.patch("model_tracks.package.suite_identity", return_value={"i": 1}), \
         mock.patch("model_tracks.colab.sync_launch_branch", stop_at_branch_sync), \
         mock.patch("model_tracks.colab.prepare_git_inputs",
                    side_effect=AssertionError("release route must not build a transport")), \
         mock.patch.object(colab.shutil, "copy2",
                           side_effect=lambda src, dst: calls.setdefault(
                               "copied", (src, dst))), \
         pytest.raises(RuntimeError, match="unit-c-stop"):
        colab.main()

    copied_src, copied_dst = calls["copied"]  # type: ignore[misc]
    assert copied_src == archive
    assert str(copied_dst).startswith(str(colab.RESULTS / "model_tracks"))
    # the reuse path ran: the canonical package was copied, never repackaged
    assert "packaged" not in calls
    assert "packaged" not in calls
