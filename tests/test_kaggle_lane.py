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
    # The 2.x CLI token file: no trailing newline, 0600.
    assert tokens.read_text() == "token-abc"
    assert tokens.stat().st_mode & 0o777 == 0o600
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
    assert kept.parent == tmp_path / "kaggle_stage" / "logs"


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
    destination = tmp_path / "kaggle_stage" / "logs" / "er-train-gpu.stream.log"
    content = destination.read_text().splitlines()
    assert content == frames, "replayed session must deduplicate, never append"
    assert len(pulls) >= 2, "the dropped SSE connection must reconnect"

