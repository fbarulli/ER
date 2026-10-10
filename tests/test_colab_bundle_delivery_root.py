"""Result-download root contract: receipts key to the root the file lands in.

Regression (commit 4d40d1e, re-broken per-lane): the first `--what bundle`
lane placed its delivery archive under repo-root ``results/colab_bundle/
<run_id>/`` while ``_download_file_with_visibility`` resolved every
display/event path with ``local.relative_to(TRAINING_RESULTS / run_id)`` — a
``ValueError`` raised by contract at delivery time, after the multi-hour VM
preparation, with the default teardown destroying the archive. The same class
struck every lane-qualified root: the CPU delivery lane handed the transport
the BARE run_id while the archive landed under ``colab_bundle_<run_id>``
(fb10b68), and the smoke lane under ``smoke_<run_id>`` (4186da7).

The transport now DERIVES the run root from the destination itself
(``local.relative_to(TRAINING_RESULTS).parts[0]``), so a caller cannot key
receipts to a root other than the one the file actually lands in. These tests
pin that contract WITHOUT any Colab transport: the algebra test is pure path
arithmetic, and the two lane tests run the real derivation behind the standard
offline command fake every lane test uses.
"""
from __future__ import annotations

from pathlib import Path

from cli import colab


RUN_ID = "bundle_1005T171234567890Z"


def test_bundle_delivery_local_is_colab_bundle_sibling_in_training_results():
    delivery = colab._bundle_delivery_local(RUN_ID)
    assert delivery == colab.TRAINING_RESULTS / ("colab_bundle_" + RUN_ID)
    assert delivery.is_relative_to(colab.TRAINING_RESULTS)


def test_bundle_delivery_destination_carries_its_own_event_root():
    # The transport's derivation on exactly the pair run_bundle hands it: the
    # first component under TRAINING_RESULTS IS the delivery root's own name,
    # and the remainder is the archive — one root, no caller-passed run_id.
    local = colab._bundle_delivery_local(RUN_ID) / "bundle_delivery.tar.zst"
    run_relative = local.relative_to(colab.TRAINING_RESULTS)
    assert run_relative.parts[0] == "colab_bundle_" + RUN_ID
    assert Path(*run_relative.parts[1:]) == Path("bundle_delivery.tar.zst")


def _fake_download_command(monkeypatch, payload: bytes) -> None:
    """The standard offline `colab` CLI fake: a download writes its local file."""

    def fake(*args, **kwargs):
        Path(args[4]).write_bytes(payload)

    monkeypatch.setattr(colab, "colab", fake)
    monkeypatch.setattr(colab, "REMOTE_ROOT", "/content/ER")
    monkeypatch.setattr(colab, "SESSION", "test-delivery-vm")
    monkeypatch.setattr(colab, "_read_remote_text", lambda remote: "22\n")


def test_delivery_download_keeps_archive_and_receipts_in_one_prefixed_root(
    tmp_path, monkeypatch
):
    """The reported CPU-lane regression: archive AND its receipts land in the
    ``colab_bundle_``-prefixed delivery root; the bare root never appears."""
    from cli.colab_lane import ColabCPULane
    from cli.colab_lane_contracts import DELIVERY_ARCHIVE_NAME

    results = tmp_path / "training_results"
    monkeypatch.setattr(colab, "TRAINING_RESULTS", results)
    _fake_download_command(monkeypatch, b"delivered bundle bytes")

    ColabCPULane()._download_delivery(RUN_ID)

    delivery = results / ("colab_bundle_" + RUN_ID)
    assert (delivery / DELIVERY_ARCHIVE_NAME).is_file()
    events = (delivery / colab._RESULT_EVENTS_FILE).read_text(encoding="utf-8")
    assert "download_file" in events  # the transport's own receipt
    assert "measured" in events  # the lane's size receipt
    assert not (results / RUN_ID).exists()


def test_smoke_result_download_keys_every_receipt_to_the_smoke_root(
    tmp_path, monkeypatch
):
    """Same class, smoke lane: the local root carries the ``smoke_`` prefix
    (4186da7), so every receipt of the verified download keys to that root."""
    from types import SimpleNamespace

    from cli.colab_result_sync import download_verified_training_results
    from model_tracks import run_retention

    results = tmp_path / "training_results"
    monkeypatch.setattr(colab, "TRAINING_RESULTS", results)
    _fake_download_command(monkeypatch, b"result archive bytes")
    monkeypatch.setattr(
        colab, "_prepare_remote_result_archive",
        lambda remote_base, workers: f"{remote_base}/results.tar.zst")
    monkeypatch.setattr(
        colab, "_extract_result_archive",
        lambda *a, **k: SimpleNamespace(included=[], excluded=[]))
    monkeypatch.setattr(run_retention, "replace_smoke", lambda root: None)
    events: list[str] = []
    monkeypatch.setattr(
        colab, "_result_event", lambda root, *a, **k: events.append(root))

    download_verified_training_results(
        f"/content/ER/results/concurrent_train_{RUN_ID}", 1, smoke=True)

    assert events and set(events) == {"smoke_" + RUN_ID}
    assert (results / ("smoke_" + RUN_ID) / colab._RESULT_ARCHIVE_NAME).is_file()
    assert not (results / RUN_ID).exists()


def test_smoke_remote_run_metadata_lands_in_the_smoke_root(tmp_path, monkeypatch):
    """Metadata side of the same class: a smoke run's ``remote_run.json`` rides
    in the ``smoke_``-prefixed root the verified download uses — never orphaned
    in a bare ``<run_id>/`` root the download never writes to."""
    results = tmp_path / "training_results"
    monkeypatch.setattr(colab, "TRAINING_RESULTS", results)

    colab._record_remote_run(
        f"/content/ER/results/concurrent_train_{RUN_ID}",
        workers=1, lane="train", smoke=True)

    assert (results / ("smoke_" + RUN_ID) / "remote_run.json").is_file()
    assert not (results / RUN_ID).exists()


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


def test_install_delivery_routes_data_to_checkout_and_prep_to_results(tmp_path):
    """The install step maps each delivery member to its consumable home.

    ``data/*`` lands under the checkout root (the prepared trees a launch
    packages); ``training_prep/*`` lands under ``results/`` (where the prepared
    ``all_tracks_inputs`` package is scanned). A member under any other
    top-level name is not part of the delivery contract and is skipped, never
    guessed at. No VM, no transport — pure local extraction through the ONE
    ``core.archive_reader.tar_archive`` reader.
    """
    from cli.colab_lane import ColabCPULane
    from core.archive_reader import tar_archive

    checkout = tmp_path / "checkout"
    results = tmp_path / "results"
    checkout.mkdir()
    results.mkdir()
    # build a delivery archive with the two contract roots + one stray member
    staging = tmp_path / "staging"
    (staging / "data" / "track_setup").mkdir(parents=True)
    (staging / "data" / "track_setup" / "cascade.yaml").write_text("track: cascade\n")
    (staging / "data" / "prepared").mkdir(parents=True)
    (staging / "data" / "prepared" / "full.bin").write_bytes(b"prepared")
    (staging / "training_prep" / "20261010T160841744159").mkdir(parents=True)
    (staging / "training_prep" / "20261010T160841744159"
     / "all_tracks_inputs.tar.zst").write_bytes(b"package")
    (staging / "stray").mkdir()
    (staging / "stray" / "ignored.txt").write_text("not part of the contract")
    archive = tmp_path / "bundle_delivery.tar.zst"
    with tar_archive(archive, "w") as tar:
        for rel in ("data", "training_prep", "stray"):
            tar.add(staging / rel, arcname=rel, recursive=True)

    installed = ColabCPULane().install_delivery(
        archive, checkout=checkout, results=results)

    # data/* under the checkout, training_prep/* under results
    assert (checkout / "data" / "track_setup" / "cascade.yaml").is_file()
    assert (checkout / "data" / "prepared" / "full.bin").is_file()
    assert (results / "training_prep" / "20261010T160841744159"
            / "all_tracks_inputs.tar.zst").is_file()
    # the stray top-level member is never written anywhere
    assert not (checkout / "stray").exists()
    assert not (results / "stray").exists()
    # FILE counts only (the two data files, the one prep package)
    assert installed == {"data": 2, "training_prep": 1}
