"""Kaggle dataset/export transport lane — packaging, transport contract, fail-loud.

NEW-lane pins (branch kaggle-lane). Everything runs offline: no test touches
the network or the kaggle executable; upload/download are pinned through a
staged subprocess fake (the staged-fake precedent of the colab-lane tests).
The live invocation contract (--execute + credentials + slug) is fail-loud.
"""
from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from core import common
from cli import kaggle_lane


def _write_export(path: Path, rows: int = 3) -> str:
    frame = pd.DataFrame({
        "sku_id": [f"SKU{index}" for index in range(rows)],
        "retailer": ["store"] * rows,
        "country": ["USA"] * rows,
        "sku_name_eng": ["Test Drink 500 ml"] * rows,
    })
    frame.to_csv(path, index=False)
    return "sku_id,retailer,country,sku_name_eng"


def _spec(tmp_path, monkeypatch, *, staging="kaggle_stage", slug=None):
    from core.schemas import KaggleSpec

    spec = KaggleSpec(slug=slug, staging_dir=staging)
    monkeypatch.setattr(kaggle_lane, "_spec", lambda: spec)
    monkeypatch.setattr(kaggle_lane, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(kaggle_lane, "staging_dir",
                        lambda: (tmp_path / staging).resolve())
    return spec


def test_export_csvs_block_additive_and_yaml_consistent():
    cfg = common.training_cfg()
    # The committed YAML carries the same values the schema defaults to;
    # neither a missing block nor a flipped default may change colab/prep.
    assert cfg.kaggle.slug is None
    assert cfg.kaggle.export_csvs == ("dataset.csv", "dataset_50pct.csv",
                                      "dataset_10k.csv")
    assert cfg.kaggle.submission_id_columns == ("sku_id", "item_id")
    raw = yaml.safe_load((common.TRAIN_ROOT / "config/training.yaml").read_text())
    assert raw["kaggle"]["staging_dir"] == cfg.kaggle.staging_dir
    assert raw["kaggle"]["slug"] is None


@pytest.mark.parametrize("updates", [
    {"staging_dir": "/abs/path"},
    {"staging_dir": "../escape"},
    {"submission_id_columns": ("sku_id",)},
    {"submission_id_columns": ("sku_id", "sku_id")},
    {"unexpected": True},
])
def test_kaggle_spec_rejects_non_portable_and_bad_contract(tmp_path, updates):
    from core.schemas import KaggleSpec

    with pytest.raises(ValidationError):
        KaggleSpec(**updates)


def test_package_export_measures_census_and_writes_archive(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    export = tmp_path / "dataset_50pct.csv"
    header = _write_export(export, rows=4)
    package = kaggle_lane.package_export(export)
    stage = tmp_path / "kaggle_stage" / "50pct"
    assert Path(package.archive_path).read_bytes()[:2] == b"PK"
    with zipfile.ZipFile(package.archive_path) as bundle:
        assert bundle.namelist() == ["dataset.csv"]
        member = bundle.read("dataset.csv")
    assert member.decode().splitlines()[0] == header
    assert package.census.rows == 4
    assert package.census.columns == header.split(",")
    receipt = json.loads((stage / "50pct.receipt.json").read_text())
    assert receipt["export_rows"] == 4
    assert receipt["cohort"] == "50pct"
    assert len(receipt["archive_sha256"]) == 64
    assert receipt["metadata"]["licenses"] == [{"name": "other"}]


def test_package_export_deterministic_receipt_for_same_bytes(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    export = tmp_path / "dataset.csv"
    _write_export(export)
    first = kaggle_lane.package_export(export)
    second = kaggle_lane.package_export(export)
    assert first.census == second.census
    first_receipt = json.loads(
        (tmp_path / "kaggle_stage/full/full.receipt.json").read_text())
    # The receipt archive hash is the transport-identity contract; the
    # census (export) hash is identical across packages of the same bytes.
    assert len(first_receipt["archive_sha256"]) == 64
    assert first.census.sha256 == second.census.sha256


def test_package_export_rejects_missing_and_empty(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        kaggle_lane.package_export(tmp_path / "absent.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("sku_id\n")
    with pytest.raises(ValueError, match="no data rows"):
        kaggle_lane.package_export(empty)


def test_cohort_labels_match_shared_tags(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    assert kaggle_lane.cohort_label(Path("dataset.csv")) == "full"
    assert kaggle_lane.cohort_label(Path("dataset_50pct.csv")) == "50pct"
    assert kaggle_lane.cohort_label(Path("weird name.csv")) == "weird_name"


def test_upload_dry_run_never_touches_network(tmp_path, monkeypatch):
    spec = _spec(tmp_path, monkeypatch, slug="owner/slug")
    export = tmp_path / "dataset.csv"
    _write_export(export)
    package = kaggle_lane.package_export(export)
    called = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: called.append(a) or pytest.fail("network"))
    plan = kaggle_lane.upload_dataset(package, execute=False)
    assert plan["mode"] == "dry-run" and plan["slug"] == "owner/slug"
    assert not called


def test_upload_requires_configured_slug(tmp_path, monkeypatch):
    spec = _spec(tmp_path, monkeypatch, slug=None)
    export = tmp_path / "dataset.csv"
    _write_export(export)
    package = kaggle_lane.package_export(export)
    with pytest.raises(RuntimeError, match="kaggle.slug is unset"):
        kaggle_lane.upload_dataset(package, execute=True)


def test_upload_requires_kaggle_executable(tmp_path, monkeypatch):
    spec = _spec(tmp_path, monkeypatch, slug="owner/slug")
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: None)
    export = tmp_path / "dataset.csv"
    _write_export(export)
    package = kaggle_lane.package_export(export)
    with pytest.raises(RuntimeError, match="kaggle.json"):
        kaggle_lane.upload_dataset(package, execute=True)


def test_upload_executed_runs_configured_cli(tmp_path, monkeypatch):
    spec = _spec(tmp_path, monkeypatch, slug="owner/slug")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    export = tmp_path / "dataset.csv"
    _write_export(export)
    package = kaggle_lane.package_export(export)
    plan = kaggle_lane.upload_dataset(package, execute=True)
    assert plan["mode"] == "executed" and plan["returncode"] == 0
    assert commands and commands[0][0].endswith("kaggle")
    assert "create" in commands[0]


def test_upload_executed_fails_loud_on_nonzero(tmp_path, monkeypatch):
    spec = _spec(tmp_path, monkeypatch, slug="owner/slug")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 3)

    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    export = tmp_path / "dataset.csv"
    _write_export(export)
    package = kaggle_lane.package_export(export)
    with pytest.raises(RuntimeError, match="rc=3"):
        kaggle_lane.upload_dataset(package, execute=True)


def test_download_requires_package_receipt(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch, slug="owner/slug")
    missing = kaggle_lane.KagglePackage(
        export_path=str(tmp_path / "dataset.csv"),
        archive_path=str(tmp_path / "x.zip"),
        metadata_path=str(tmp_path / "m.json"),
        census=kaggle_lane.ExportCensus(
            rows=1, bytes=1, sha256="0" * 64, columns=["sku_id"]),
    )
    with pytest.raises(FileNotFoundError, match="receipt"):
        kaggle_lane.download_dataset(missing, execute=False)


def test_download_dry_run_reports_expected_identity(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch, slug="owner/slug")
    export = tmp_path / "dataset.csv"
    _write_export(export)
    package = kaggle_lane.package_export(export)
    plan = kaggle_lane.download_dataset(package, execute=False)
    assert plan["expected_archive_sha256"] and len(plan["expected_archive_sha256"]) == 64


def test_download_verifies_fetched_archive_identity(tmp_path, monkeypatch):
    spec = _spec(tmp_path, monkeypatch, slug="owner/slug")
    export = tmp_path / "dataset.csv"
    _write_export(export)
    package = kaggle_lane.package_export(export)
    stage = tmp_path / "kaggle_stage" / "full"
    fetched = stage / "slug.zip"
    fetched.write_bytes(Path(package.archive_path).read_bytes())
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        kaggle_lane.subprocess, "run",
        lambda command, **kw: subprocess.CompletedProcess(command, 0))
    plan = kaggle_lane.download_dataset(package, execute=True)
    assert plan["verified"] is True


def test_download_rejects_drifted_fetchback(tmp_path, monkeypatch):
    spec = _spec(tmp_path, monkeypatch, slug="owner/slug")
    export = tmp_path / "dataset.csv"
    _write_export(export)
    package = kaggle_lane.package_export(export)
    stage = tmp_path / "kaggle_stage" / "full"
    (stage / "slug.zip").write_bytes(b"tampered")
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        kaggle_lane.subprocess, "run",
        lambda command, **kw: subprocess.CompletedProcess(command, 0))
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        kaggle_lane.download_dataset(package, execute=True)


def test_submission_packaging_matches_external_contract(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({
        "SKU_ID": ["S1", "S2", "S3"],
        "ITEM_ID": ["ITEM-A", "ITEM-B", "UNMATCHED_1"],
    }).to_csv(predictions, index=False)
    output = tmp_path / "packaged" / "submission.csv"
    kaggle_lane.package_submission(predictions, output)
    frame = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert list(frame.columns) == ["sku_id", "item_id"]
    assert len(frame) == 3
    receipt = json.loads(
        (tmp_path / "kaggle_stage" / "submission.receipt.json").read_text())
    assert receipt["rows"] == 3 and receipt["unique_items"] == 3
    assert receipt["unmatched_items"] == 1
    assert receipt["columns"] == ["sku_id", "item_id"]


def test_submission_packaging_rejects_missing_columns(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    predictions = tmp_path / "bad.csv"
    pd.DataFrame({"sku": ["S1"], "item": ["I1"]}).to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        kaggle_lane.package_submission(predictions, tmp_path / "out.csv")


# ── remote bundle-generation kernel (owner ruling 2026-10-06: CPU-only) ─────

def _kernel_spec(tmp_path, monkeypatch, **updates):
    from core.schemas import KaggleSpec

    values = {"staging_dir": "kaggle_stage", "username": "owner",
              "cpu_kernel_slug": "owner/er-bundle-cpu"}
    values.update(updates)
    spec = KaggleSpec(**values)
    monkeypatch.setattr(kaggle_lane, "_spec", lambda: spec)
    monkeypatch.setattr(kaggle_lane, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(kaggle_lane, "staging_dir",
                        lambda: (tmp_path / "kaggle_stage").resolve())
    return spec


def test_credentials_dry_run_never_writes(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    target = tmp_path / "home" / ".kaggle" / "kaggle.json"
    tokens = tmp_path / "home" / ".kaggle" / "access_token"
    monkeypatch.setattr(kaggle_lane, "CREDENTIALS_PATH", target)
    monkeypatch.setattr(kaggle_lane, "ACCESS_TOKEN_PATH", tokens)
    monkeypatch.setenv("KAGGLE_API_KEY", "token-abc")
    plan = kaggle_lane.write_credentials(execute=False)
    assert plan["mode"] == "dry-run" and plan["key_present"] is True
    assert plan["username"] == "owner"
    assert not target.exists() and not tokens.exists()


def test_credentials_execute_writes_0600_and_fail_loud(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    target = tmp_path / "home" / ".kaggle" / "kaggle.json"
    tokens = tmp_path / "home" / ".kaggle" / "access_token"
    monkeypatch.setattr(kaggle_lane, "CREDENTIALS_PATH", target)
    monkeypatch.setattr(kaggle_lane, "ACCESS_TOKEN_PATH", tokens)
    monkeypatch.setenv("KAGGLE_API_KEY", "token-abc")
    plan = kaggle_lane.write_credentials(execute=True)
    assert plan["written"] is True
    document = json.loads(target.read_text())
    assert document == {"username": "owner", "key": "token-abc"}
    assert target.stat().st_mode & 0o777 == 0o600
    # kaggle.json is the single credential; the token file is never written.
    assert not tokens.exists()
    monkeypatch.setenv("KAGGLE_API_KEY", "")
    with pytest.raises(RuntimeError, match="empty or unset"):
        kaggle_lane.write_credentials(execute=True)


def test_credentials_require_configured_username(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, username=None)
    with pytest.raises(RuntimeError, match="kaggle.username is unset"):
        kaggle_lane.write_credentials(execute=False)


def test_stage_bundle_kernel_pins_revision_and_metadata(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    receipt = kaggle_lane.stage_bundle_kernel()
    stage = tmp_path / "kaggle_stage" / "bundle_kernel"
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    assert metadata["id"] == "owner/er-bundle-cpu"
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is True
    assert metadata["kernel_type"] == "script"
    assert metadata["code_file"] == "bundle_cpu.py"
    script = (stage / "bundle_cpu.py").read_text()
    assert 'REVISION = "abc123def"' in script
    assert "training.prepare_all" in script
    assert '"src"' in script and "artifacts/models" in script
    assert "all_tracks_inputs.tar.zst" in script
    assert receipt["revision"] == "abc123def" and receipt["gpu"] is False


def test_stage_bundle_kernel_requires_slug(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, cpu_kernel_slug=None)
    with pytest.raises(RuntimeError, match="cpu_kernel_slug is unset"):
        kaggle_lane.stage_bundle_kernel()


def test_push_bundle_kernel_invokes_cli_with_staged_dir(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    stage = tmp_path / "kaggle_stage" / "bundle_kernel"
    stage.mkdir(parents=True)
    (stage / "kernel-metadata.json").write_text(
        json.dumps({"id": "owner/er-bundle-cpu", "code_file": "bundle_cpu.py"}),
        encoding="utf-8")
    (stage / "bundle_cpu.py").write_text(
        "REPOSITORY = 'https://example.invalid/ER.git'\n"
        "BRANCH = 'kaggle-lane'\n"
        "REVISION = '0000000000000000000000000000000000000000'\n"
        "_runtime_files = ()\n",
        encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    import importlib
    monkeypatch.setattr(importlib.import_module("core.runtime_inputs"),
                        "staged_kernel_preflight", lambda stage_dir: None)
    result = kaggle_lane.push_bundle_kernel(stage)
    assert result["pushed"] is True
    assert Path(calls[0][0]).name == "kaggle"
    assert calls[0][1:3] == ["kernels", "push"]
    assert str(stage) in calls[0]


def test_kernel_status_parses_state(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout='owner/er-bundle-cpu status is "running"\n', stderr="")

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    status = kaggle_lane.kernel_status()
    assert status["status"] == "running"


def test_fetch_bundle_output_verifies_sha_and_installs(tmp_path, monkeypatch):
    import hashlib

    _kernel_spec(tmp_path, monkeypatch)
    archive_bytes = b"fake archive bytes"

    def fake_run(command, **kwargs):
        stage = Path(command[command.index("-p") + 1])
        bundle = stage / "bundle"
        bundle.mkdir(parents=True)
        (bundle / "all_tracks_inputs.tar.zst").write_bytes(archive_bytes)
        receipt = {"revision": "abc123", "branch": "kaggle-lane", "run_dir": "r",
                   "archive": "all_tracks_inputs.tar.zst",
                   "archive_bytes": len(archive_bytes),
                   "archive_sha256": hashlib.sha256(archive_bytes).hexdigest()}
        (bundle / "bundle.receipt.json").write_text(json.dumps(receipt))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    plan = kaggle_lane.fetch_bundle_output(execute=True)
    assert plan["verified"] is True and plan["cohort"] == "full"
    installed = tmp_path / "kaggle_stage" / "full" / "bundle" / "all_tracks_inputs.tar.zst"
    assert installed.read_bytes() == archive_bytes


def test_fetch_bundle_output_rejects_sha_drift(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)

    def fake_run(command, **kwargs):
        stage = Path(command[command.index("-p") + 1])
        bundle = stage / "bundle"
        bundle.mkdir(parents=True)
        (bundle / "all_tracks_inputs.tar.zst").write_bytes(b"tampered")
        receipt = {"archive": "all_tracks_inputs.tar.zst", "archive_sha256": "0" * 64}
        (bundle / "bundle.receipt.json").write_text(json.dumps(receipt))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        kaggle_lane.fetch_bundle_output(execute=True)


def test_fetch_bundle_output_dry_run_never_touches_network(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: called.append(a) or pytest.fail("network"))
    plan = kaggle_lane.fetch_bundle_output(execute=False)
    assert plan["mode"] == "dry-run"
    assert not called


def test_kernel_slugs_tracked_in_config():
    # Owner ruling 2026-10-06: the GPU training kernel stays tracked via the
    # config SSOT even while only the CPU bundle kernel is live.
    cfg = common.training_cfg()
    assert cfg.kaggle.cpu_kernel_slug == "fbarulli/er-bundle-cpu"
    assert cfg.kaggle.gpu_kernel_slug == "fbarulli/er-train-gpu"


def test_kernel_status_resolves_gpu_slug(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="queued", stderr="")

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    status = kaggle_lane.kernel_status(which="gpu")
    assert status["kernel"] == "owner/er-train-gpu"
    assert "owner/er-train-gpu" in calls[0]


def test_kernel_status_fail_loud_without_slug(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug=None)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    with pytest.raises(RuntimeError, match="gpu_kernel_slug is unset"):
        kaggle_lane.kernel_status(which="gpu")


def test_fetch_failed_kernel_log_keeps_session_log_despite_contract_fail(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")

    def fake_run(command, **kwargs):
        stage = Path(command[command.index("-p") + 1])
        (stage / "er-train-gpu.log").write_text(
            "Traceback (most recent call last):")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    plan = kaggle_lane.fetch_failed_kernel_log("train")
    assert plan["error_log"] is not None
    kept = Path(plan["error_log"])
    assert kept.read_text().startswith("Traceback")
    assert kept.name == "er-train-gpu.log"
    assert kept.parent == tmp_path / "logs" / "kaggle"


def test_supervise_records_kernel_log_on_error(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    monkeypatch.setattr(kaggle_lane, "kernel_status",
                        lambda *a, **kw: {"status": "error",
                                          "raw": "KernelWorkerStatus.ERROR"})
    monkeypatch.setattr(kaggle_lane, "stream_kernel_logs",
                        lambda *a, **kw: {"kernel": "owner/er-train-gpu"})
    monkeypatch.setattr(
        kaggle_lane, "fetch_failed_kernel_log",
        lambda kind: {"kind": kind, "mode": "executed",
                      "error_log": "/tmp/opc/diag.log"})
    plan = kaggle_lane.supervise_kernels(kinds=("train",), execute=True)
    failure = plan["failures"]["train"]
    assert failure["status"] == "error"
    assert failure["error_log"] == "/tmp/opc/diag.log"


def test_supervise_releases_session_on_error(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    monkeypatch.setattr(kaggle_lane, "kernel_status",
                        lambda *a, **kw: {"status": "error",
                                          "raw": "KernelWorkerStatus.ERROR"})
    monkeypatch.setattr(kaggle_lane, "stream_kernel_logs",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kaggle_lane, "fetch_failed_kernel_log",
                        lambda kind: {"kind": kind, "mode": "executed",
                                      "error_log": None})
    pushes = []

    def fake_run(command, **kwargs):
        pushes.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    plan = kaggle_lane.supervise_kernels(kinds=("train",), execute=True)
    assert plan["failures"]["train"]["stop"]["stopped"] is True
    assert any("kernels" in parts and "push" in parts and "-p" in parts
               for parts in pushes), "session release must replace the version"


def test_stream_kernel_logs_replays_whole_session_on_reconnect(tmp_path, monkeypatch):
    import types
    import requests
    import kagglesdk.kaggle_client
    import kagglesdk.kernels.types.kernels_api_service

    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    frames = [
        'data: {"stream_name":"stdout","time":1,"data":"+ git clone\\n"}',
        'data: {"stream_name":"stderr","time":2,"data":"[timing] mark 1s\\n"}',
        'data: {"stream_name":"stdout","time":3,"data":"phase complete\\n"}',
    ]
    pulls = []

    class Stream:
        state = {"dropped": False}

        def iter_lines(self):
            pulls.append(1)
            yield frames[0]
            yield frames[1]
            if not self.state["dropped"]:
                self.state["dropped"] = True
                raise requests.exceptions.ChunkedEncodingError(
                    "Response ended prematurely")
            yield frames[2]

    fake_api = types.SimpleNamespace(
        get_kernel_session_logs_stream=lambda request: Stream())
    monkeypatch.setattr(
        kagglesdk.kaggle_client, "KaggleClient",
        lambda env: types.SimpleNamespace(kernels=types.SimpleNamespace(
            kernels_api_client=fake_api)))
    monkeypatch.setattr(kaggle_lane.time, "sleep", lambda seconds: None)

    kaggle_lane.stream_kernel_logs("owner/er-train-gpu")
    # One roof (owner order 2026-10-07): every transcript landmark lands on
    # the single logs/kaggle/lane.log (files.stream_log default).
    destination = tmp_path / "logs" / "kaggle" / "lane.log"
    content = destination.read_text().splitlines()
    assert content == ["+ git clone", "[timing] mark 1s", "phase complete"], \
        "decoded data payloads must be written as plain lines"
    assert len(pulls) >= 2, "the dropped SSE connection must reconnect"


def test_stream_kernel_logs_expands_cr_frames_and_tags_last_bar(tmp_path, monkeypatch):
    import types
    import requests
    import kagglesdk.kaggle_client
    import kagglesdk.kernels.types.kernels_api_service

    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    frames = [
        'data: {"stream_name":"stdout","time":1,"data":"12%\\r35%\\r60%\\r"}',
        'data: {"stream_name":"stdout","time":2,"data":"[timing] done\\n"}',
    ]

    class Stream:
        def iter_lines(self):
            yield from frames

    fake_api = types.SimpleNamespace(
        get_kernel_session_logs_stream=lambda request: Stream())
    monkeypatch.setattr(
        kagglesdk.kaggle_client, "KaggleClient",
        lambda env: types.SimpleNamespace(kernels=types.SimpleNamespace(
            kernels_api_client=fake_api)))
    monkeypatch.setattr(kaggle_lane.time, "sleep", lambda seconds: None)

    kaggle_lane.stream_kernel_logs("owner/er-train-gpu")
    # One roof: stream transcripts append to logs/kaggle/lane.log (files.stream_log).
    content = (tmp_path / "logs" / "kaggle"
               / "lane.log").read_text().splitlines()
    # every \r frame is its own grep-able line, and the last bar stays tagged
    # at the end of its chunk so the log tail shows the training tqdm strip
    assert content == ["12%", "35%", "60%", "[tqdm] 60%", "[timing] done"]


def test_lane_logs_dir_is_under_canonical_logs_root(tmp_path, monkeypatch):
    _spec(tmp_path, monkeypatch)
    assert kaggle_lane.lane_logs_dir() == (tmp_path / "logs" / "kaggle").resolve()



def _fixed_paris_datetime():
    from datetime import datetime as real_datetime
    from zoneinfo import ZoneInfo

    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 6, 21, 12, 47, tzinfo=ZoneInfo("Europe/Paris"))

    return FixedDatetime


def test_log_lane_console_and_file_carry_local_stamp(tmp_path, monkeypatch, capsys):
    spec = _spec(tmp_path, monkeypatch)
    monkeypatch.setattr(kaggle_lane, "datetime", _fixed_paris_datetime())
    expected = "2026-10-06T21:12:47 CEST"
    kaggle_lane._log_lane("$ kaggle kernels push")
    console = capsys.readouterr().out.splitlines()[0]
    assert console == f"[kaggle-lane {expected}] $ kaggle kernels push"
    log_path = (tmp_path / "logs" / "kaggle" / "lane.log")
    assert log_path.read_text().splitlines()[0] == \
        f"{expected} $ kaggle kernels push"


def test_stamp_helper_format():
    from zoneinfo import ZoneInfo
    monkeypatch = _fixed_paris_datetime()
    original = kaggle_lane.datetime
    kaggle_lane.datetime = monkeypatch
    try:
        assert kaggle_lane._stamp() == "[kaggle-lane 2026-10-06T21:12:47 CEST]"
    finally:
        kaggle_lane.datetime = original


def test_stamp_matches_paris_local_format():
    import re
    assert re.fullmatch(
        r"\[kaggle-lane \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} (CET|CEST)\]",
        kaggle_lane._stamp())


# ── publish default + chain op (owner order 2026-10-07) ─────────────────────

def _verified_bundle_install(tmp_path: Path, revision="abc123def") -> Path:
    import hashlib

    install = tmp_path / "kaggle_stage" / "10k" / "bundle"
    install.mkdir(parents=True, exist_ok=True)
    archive_bytes = b"bundle bytes"
    (install / "all_tracks_inputs.tar.zst").write_bytes(archive_bytes)
    (install / "bundle.receipt.json").write_text(json.dumps(
        {"revision": revision, "branch": "kaggle-lane", "run_dir": "r",
         "cohort": "10k", "cohort_dataset": "dataset_10k.csv",
         "archive": "all_tracks_inputs.tar.zst", "archive_bytes": 12,
         "archive_sha256": hashlib.sha256(archive_bytes).hexdigest()}))
    (install / "manifest.json").write_text("{}\n")
    (install / "timings.json").write_text("{}\n")
    return install


def _isolate_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_lane, "ACCESS_TOKEN_PATH",
                        tmp_path / "home" / ".kaggle" / "access_token")


def _hermetic_staging(monkeypatch):
    """The stage/push helpers inventory the REAL repo checkout; these pins
    only need the metadata/script contracts, so the fakes close that door."""
    import importlib

    def members(*extra, lane="bundle"):
        flat = {name for group in extra for name in
                (group if isinstance(group, (tuple, list)) else (group,))}
        return tuple(sorted(flat))

    monkeypatch.setattr(kaggle_lane, "checkout_members", members)
    monkeypatch.setattr(kaggle_lane, "checkout_inventory", members)
    monkeypatch.setattr(kaggle_lane, "checkout_preflight_script",
                        lambda files, root_expression="root":
                        "_runtime_files = ()\n")
    monkeypatch.setattr(importlib.import_module("core.runtime_inputs"),
                        "staged_kernel_preflight", lambda stage_dir: None)


def test_publish_bundle_dataset_builds_stage_and_versions(tmp_path, monkeypatch):
    spec = _kernel_spec(tmp_path, monkeypatch,
                        gpu_kernel_slug="owner/er-train-gpu",
                        bundle_dataset_slug="owner/er-10k-bundle")
    _isolate_credentials(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    install = _verified_bundle_install(tmp_path)
    archive_bytes = (install / "all_tracks_inputs.tar.zst").read_bytes()
    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        if "status" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout='{"current_version_number": 12}\n', stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    plan = kaggle_lane.publish_bundle_dataset("bundle", execute=True)
    # the stage dir mirrors the previous manual flow (10k_bundle_dataset)
    stage = tmp_path / "kaggle_stage" / "10k_bundle_dataset"
    metadata = json.loads((stage / "dataset_metadata.json").read_text())
    assert metadata["id"] == "owner/er-10k-bundle"
    assert metadata["title"] == "ER 10k bundle"
    assert metadata["licenses"] == [{"name": "other"}]
    assert (stage / "all_tracks_inputs.tar.zst").read_bytes() == archive_bytes
    receipt = json.loads((stage / "publish.receipt.json").read_text())
    assert receipt["published"] is True
    assert receipt["revision"] == "abc123def"
    assert receipt["dataset_version"] == 12
    # the train mount pin rides the plan; `datasets version` ran via the CLI
    assert plan["train_stage_mount"]["dataset_sources_pinned"] == [
        "owner/er-10k-bundle/12"]
    assert plan["train_stage_mount"]["dataset_sources_default"] == [
        "owner/er-10k-bundle"]
    assert commands[0][:3] == ["/usr/bin/kaggle", "datasets", "version"]
    assert commands[1][:3] == ["/usr/bin/kaggle", "datasets", "status"]
    # train/embed outputs have no SSOT dataset to publish — recorded skip
    assert kaggle_lane.publish_bundle_dataset(
        "train", execute=True)["published"] is False
    # a drifted install fail-louds BEFORE anything is staged: the tampered
    # archive stops matching its own kernel receipt and may not be staged
    (install / "all_tracks_inputs.tar.zst").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="no verified bundle install"):
        kaggle_lane.publish_bundle_dataset("bundle", execute=True)


def test_chain_runs_supervised_with_one_spawn_per_kernel(tmp_path, monkeypatch):
    spec = _kernel_spec(tmp_path, monkeypatch,
                        cpu_kernel_slug="owner/er-bundle-cpu",
                        gpu_kernel_slug="owner/er-train-gpu",
                        embedding_kernel_slug="owner/er-embed-gpu",
                        embedding_dataset_slug="owner/er-embed-requests",
                        bundle_dataset_slug="owner/er-10k-bundle")
    _isolate_credentials(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="complete", stderr="")

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    spawns: list[str] = []

    def fake_spawn(watcher):
        watcher = kaggle_lane.AUTOWATCH_WHICH.get(watcher, watcher)
        spawns.append(watcher)
        kind = {"cpu": "bundle", "gpu": "train", "embed": "embed"}[watcher]
        receipts = tmp_path / "kaggle_stage" / f"autowatch_{kind}.receipt.json"
        receipts.write_text(json.dumps({
            "status": "complete", "polls": 2,
            "fetch": {"verified": True, "archive_sha256": "d" * 64,
                      "cohort": "10k",
                      "publish": {"published": True, "slug": "owner/er-10k-bundle",
                                  "dataset_version": 12}},
            "stop": {"stopped": True},
        }))
        return {"autowatch": "spawned", "kernel": watcher, "log": str(receipts)}

    monkeypatch.setattr(kaggle_lane, "_spawn_autowatch", fake_spawn)
    # dry-run: the entire plan prints (rooted paths), nothing staged/written
    dry = kaggle_lane.run_chain(cohort="10k", with_embed=True, execute=False)
    assert dry["mode"] == "dry-run" and set(dry["steps"]) == {"bundle", "train", "embed"}
    assert dry["revision"] == "abc123def"
    assert dry["steps"]["bundle"]["publish"]["slug"] == "owner/er-10k-bundle"
    # a dry-run chain never stages a kernel or writes a receipt
    assert not list(tmp_path.rglob("kernel-metadata.json"))
    assert not (tmp_path / "kaggle_stage" / "chain.receipt.json").exists()
    # executed: each push path spawns EXACTLY its own watcher — never doubled
    plan = kaggle_lane.run_chain(cohort="10k", with_embed=True, execute=True)
    assert spawns == ["cpu", "gpu", "embed"], \
        "one watcher per kernel: bundle via its push, train/embed via their paths"
    assert plan["mode"] == "executed"
    for step in ("bundle", "train", "embed"):
        assert plan["steps"][step]["stage"]["revision"] == "abc123def"
        assert plan["steps"][step]["fetched_sha256"] == "d" * 64
    train_metadata = json.loads((tmp_path / "kaggle_stage" / "train_kernel"
                                 / "kernel-metadata.json").read_text())
    assert train_metadata["dataset_sources"] == ["owner/er-10k-bundle/12"], \
        "the train stage must attach the fresh published version"
    assert json.loads((tmp_path / "kaggle_stage" / "chain.receipt.json")
                      .read_text())["steps"]["train"]["stage"]["revision"] == "abc123def"


def _fake_sdk_cancel(monkeypatch, cancels):
    import types
    import kagglesdk.kaggle_client

    class FakeKernelsApi:
        def cancel_kernel_session(self, request):
            cancels.append(request.kernel_session_id)
            return types.SimpleNamespace()

    client = types.SimpleNamespace(kernels=types.SimpleNamespace(
        kernels_api_client=FakeKernelsApi()))
    monkeypatch.setattr(kagglesdk.kaggle_client, "KaggleClient", lambda env: client)


def test_stop_kernel_sdk_cancel_reaches_terminal_stopped(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    session_file = tmp_path / "logs" / "kaggle" / "er-train-gpu.session_id"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text("123456\n")
    cancels = []
    _fake_sdk_cancel(monkeypatch, cancels)
    monkeypatch.setattr(kaggle_lane, "kernel_status",
                        lambda *a, **kw: {"status": "error",
                                          "raw": "KernelWorkerStatus.ERROR"})
    plan = kaggle_lane.stop_kernel("owner/er-train-gpu", which="gpu", execute=True)
    assert cancels == [123456], "the recorded session id must drive the SDK cancel"
    assert plan["cancel_method"] == "sdk_cancel_kernel_session"
    assert plan["verdict"] == "stopped"
    assert plan["terminal_state"] == "error"
    assert plan["stopped"] is True


def test_stop_kernel_verify_window_expires_reports_still_running(tmp_path, monkeypatch):
    import types
    import cli.kaggle_kernels

    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    session_file = tmp_path / "logs" / "kaggle" / "er-train-gpu.session_id"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text("987654")
    cancels = []
    _fake_sdk_cancel(monkeypatch, cancels)
    monkeypatch.setattr(kaggle_lane, "kernel_status",
                        lambda *a, **kw: {"status": "running", "raw": "RUNNING"})
    ticks = iter([0.0, 10.0, 10_000.0])
    sleeps = []
    fake_time = types.SimpleNamespace(monotonic=lambda: next(ticks),
                                      sleep=lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(cli.kaggle_kernels, "time", fake_time)
    with pytest.raises(RuntimeError, match="stop did not reach a terminal state"):
        kaggle_lane.stop_kernel("owner/er-train-gpu", which="gpu", execute=True)
    assert cancels == [987654]
    assert sleeps == [15.0], "verify polls run logs_poll_seconds apart"


def test_stop_kernel_sdk_failure_falls_back_to_stub_push(tmp_path, monkeypatch):
    import types
    import kagglesdk.kaggle_client

    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    session_file = tmp_path / "logs" / "kaggle" / "er-train-gpu.session_id"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text("555000")
    monkeypatch.setattr(kagglesdk.kaggle_client, "KaggleClient",
                        lambda env: (_ for _ in ()).throw(
                            ConnectionError("proxy down")))
    pushes = []

    def fake_run(command, **kwargs):
        pushes.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane, "_run_kaggle", fake_run)
    monkeypatch.setattr(kaggle_lane, "_require_kaggle_executable",
                        lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(kaggle_lane, "kernel_status",
                        lambda *a, **kw: {"status": "error", "raw": "ERROR"})
    plan = kaggle_lane.stop_kernel("owner/er-train-gpu", which="gpu", execute=True)
    assert plan["cancel_method"] == "version_replace"
    assert "proxy down" in plan["cancel_error"], "SDK failure is recorded, not hidden"
    assert plan["verdict"] == "stopped" and plan["stopped"] is True
    assert any("kernels" in parts and "push" in parts and "-p" in parts
               for parts in pushes), "fallback must still replace the version"


def test_stop_kernel_without_session_id_falls_back_to_stub_push(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    cancels = []
    _fake_sdk_cancel(monkeypatch, cancels)
    pushes = []

    def fake_run(command, **kwargs):
        pushes.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane, "_run_kaggle", fake_run)
    monkeypatch.setattr(kaggle_lane, "_require_kaggle_executable",
                        lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(kaggle_lane, "kernel_status",
                        lambda *a, **kw: {"status": "complete", "raw": "COMPLETE"})
    plan = kaggle_lane.stop_kernel("owner/er-train-gpu", which="gpu", execute=True)
    assert cancels == [], "no recorded session id -> no SDK cancel attempt"
    assert plan["cancel_method"] == "version_replace"
    assert plan["verdict"] == "stopped" and plan["stopped"] is True
    assert plan["terminal_state"] == "complete"
    assert any("kernels" in parts and "push" in parts for parts in pushes), \
        "the stub replace must still be pushed"
