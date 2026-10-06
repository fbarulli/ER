"""Bundle-lane delivery root satisfies the result-transfer path contract.

Regression: the first `--what bundle` lane placed its delivery archive under
repo-root ``results/colab_bundle/<run_id>/`` while
``_download_file_with_visibility`` resolves every display/event path with
``local.relative_to(TRAINING_RESULTS / run_id)`` — a ``ValueError`` raised
by contract at delivery time, after the multi-hour VM preparation, with the
default teardown destroying the archive. These tests pin the local-root
contract WITHOUT any Colab transport: ``relative_to`` and the retention
sweep are pure filesystem/path algebra.
"""
from __future__ import annotations

from pathlib import Path

from cli import colab


RUN_ID = "bundle_1005T171234567890Z"


def test_bundle_delivery_local_is_colab_bundle_sibling_in_training_results():
    delivery = colab._bundle_delivery_local(RUN_ID)
    assert delivery == colab.TRAINING_RESULTS / ("colab_bundle_" + RUN_ID)
    assert delivery.is_relative_to(colab.TRAINING_RESULTS)


def test_bundle_delivery_satisfies_download_relative_to_contract():
    # Exactly the pair run_bundle hands to _download_file_with_visibility:
    # the download run_id is the delivery root's own name, so the contract
    # root TRAINING_RESULTS / run_id IS the delivery directory itself.
    delivery_dir = colab._bundle_delivery_local(RUN_ID)
    assert delivery_dir == colab.TRAINING_RESULTS / delivery_dir.name
    local = delivery_dir / "bundle_delivery.tar.zst"
    relative = local.relative_to(colab.TRAINING_RESULTS / delivery_dir.name)
    assert relative == Path("bundle_delivery.tar.zst")


def test_bundle_delivery_call_shape_survives_pure_path_algebra():
    # Collapse the contract into one expression the run_bundle call must keep:
    # local.relative_to(TRAINING_RESULTS / passed_run_id) resolves to the
    # archive name. Guarded by is_relative_to first so the failure mode is
    # the named contract, not an opaque ValueError.
    run_id = colab._bundle_delivery_local(RUN_ID).name
    local = colab._bundle_delivery_local(RUN_ID) / "bundle_delivery.tar.zst"
    assert local.is_relative_to(colab.TRAINING_RESULTS / run_id)
    assert local.relative_to(colab.TRAINING_RESULTS / run_id) == (
        Path("bundle_delivery.tar.zst")
    )


def test_run_retention_never_sweeps_the_bundle_delivery_root(tmp_path):
    # run_retention recognizes (and prunes) only completed runs carrying
    # track markers, and replace_smoke's overwrite rule targets smoke_ names.
    # A bundle delivery root must be invisible to both retention lanes.
    from model_tracks.run_retention import _looks_like_run

    delivery_dir = tmp_path / colab._bundle_delivery_local(RUN_ID).name
    delivery_dir.mkdir(parents=True)
    (delivery_dir / "bundle_delivery.tar.zst").write_bytes(b"")
    assert not _looks_like_run(delivery_dir)
    assert not delivery_dir.name.startswith("smoke_")
    assert not _looks_like_run(colab._bundle_delivery_local(RUN_ID))
