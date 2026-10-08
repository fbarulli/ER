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


def _fake_published_tip(monkeypatch, tip: str):
    """Offline fake for the staged-pin guard: `git fetch origin` is a
    no-op rc=0; `git rev-parse origin/<branch>` prints the fake tip."""
    def fake_run(command, **kwargs):
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout=tip,
                                               stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_stage_bundle_kernel_pins_revision_and_metadata(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    _fake_published_tip(monkeypatch, "abc123def")
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
    # the published-tip invariant: the receipt records the origin tip the
    # guard verified against the pin (origin/kaggle-lane == HEAD)
    assert receipt["published_tip"] == "abc123def"


def test_stage_bundle_kernel_refuses_stale_published_tip(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    _fake_published_tip(monkeypatch, "fed321cba9")
    with pytest.raises(RuntimeError, match="pull or push"):
        kaggle_lane.stage_bundle_kernel()


def test_stage_gpu_kernel_refuses_stale_published_tip(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    _fake_published_tip(monkeypatch, "fed321cba9")
    with pytest.raises(RuntimeError, match="origin/kaggle-lane"):
        kaggle_lane.stage_gpu_kernel(kind="train")


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


def test_kernel_identities_resolve_every_kind_from_config(tmp_path, monkeypatch):
    """Finding 1: ONE registry resolves each kind's slug, code file and result.

    Every surface that used to rebuild a kind->slug (or kind->code-file) table
    reads this instead, so the finalize job's identity shares the CPU slug while
    keeping its own kind, watcher identity, code file and result name.
    """
    spec = _kernel_spec(tmp_path, monkeypatch,
                        gpu_kernel_slug="owner/er-train-gpu",
                        embedding_kernel_slug="owner/er-embed-gpu")
    identities = kaggle_lane.kernel_identities(spec)
    assert set(identities) == {"bundle", "train", "embed", "finalize"}
    assert {kind: identity.slug(spec) for kind, identity in identities.items()} == {
        "bundle": "owner/er-bundle-cpu", "train": "owner/er-train-gpu",
        "embed": "owner/er-embed-gpu", "finalize": "owner/er-bundle-cpu"}
    assert {kind: identity.which for kind, identity in identities.items()} == {
        "bundle": "cpu", "train": "gpu", "embed": "embed", "finalize": "finalize"}
    # values come from kaggle.files, not per-surface literals (bundle names its
    # own manifest+archive pair; the others template the result name)
    assert identities["train"].code_file == spec.files.code_files["train"]
    assert identities["embed"].result_name == spec.files.result_names["embed"]
    assert identities["finalize"].result_name == spec.files.result_names["finalize"]
    assert identities["finalize"].bundle_role == "result"
    assert identities["bundle"].bundle_role == "inputs"
    # The train kernel ships the suite's own sealed result Bundle, so its fetched
    # output is role-loaded at the same boundary the finalize job consumes; the
    # embed output is not a Bundle role.
    assert identities["train"].bundle_role == "result"
    assert identities["embed"].bundle_role is None
    assert identities["bundle"].manifest_name(spec.files) == spec.files.bundle_receipt
    assert identities["finalize"].manifest_name(spec.files) == \
        spec.files.result_manifest.format(kind=spec.files.result_names["finalize"])
    # push_kernel reverse-maps a staged code file back to ONE kind
    code_files = [identity.code_file for identity in identities.values()]
    assert len(set(code_files)) == len(code_files) == 4
    # a renaming in the config flows through the registry (no baked literals)
    spec.files.code_files["bundle"] = "renamed_cpu.py"
    assert kaggle_lane.kernel_identity("bundle", spec).code_file == "renamed_cpu.py"
    # a watcher alias resolves to the kind it watches
    assert kaggle_lane.kernel_identity("cpu", spec).kind == "bundle"
    assert kaggle_lane.kernel_identity("finalize", spec).kind == "finalize"
    assert kaggle_lane.AUTOWATCH_WHICH["finalize"] == "finalize"


def test_fetch_kernel_output_rejects_an_unknown_kind(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unknown kernel output kind"):
        kaggle_lane.fetch_kernel_output(kind="bogus", execute=False)


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
            if self.state["dropped"]:
                # The midtier SSE proxy re-sends the WHOLE session from line 0
                # on a reconnect; the follower must rewrite, never duplicate.
                for frame in frames:
                    yield frame
                return
            yield frames[0]
            yield frames[1]
            self.state["dropped"] = True
            raise requests.exceptions.ChunkedEncodingError(
                "Response ended prematurely")

    fake_api = types.SimpleNamespace(
        get_kernel_session_logs_stream=lambda request: Stream())
    monkeypatch.setattr(
        kagglesdk.kaggle_client, "KaggleClient",
        lambda env: types.SimpleNamespace(kernels=types.SimpleNamespace(
            kernels_api_client=fake_api)))
    monkeypatch.setattr(kaggle_lane.time, "sleep", lambda seconds: None)

    kaggle_lane.stream_kernel_logs("owner/er-train-gpu")
    # One roof (owner order 2026-10-07): every transcript landmark — decoded
    # stdout AND the watcher's status lines — lands on the single
    # logs/kaggle/lane.log (files.stream_log default).
    destination = tmp_path / "logs" / "kaggle" / "lane.log"
    content = destination.read_text().splitlines()
    assert [line for line in content
            if line.startswith(("+ git", "[timing]", "phase"))] == \
        ["+ git clone", "[timing] mark 1s", "phase complete"], \
        "a whole-session replay must be rewritten exactly once, never duplicated"
    assert any("reconnect attempt 1" in line for line in content), \
        "the reconnect status line shares the same transcript"
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


def test_one_transcript_per_run_stream_does_not_concatenate(tmp_path, monkeypatch):
    """Two consecutive runs overwrite, never append-sprawl (owner order)."""
    import types
    import kagglesdk.kaggle_client
    import cli.kaggle_runtime as runtime

    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    monkeypatch.delenv("ER_KAGGLE_LANE_APPEND", raising=False)

    def run(tag):
        class Stream:
            def iter_lines(self):
                yield (f'data: {{"stream_name":"stdout","time":1,'
                       f'"data":"{tag} [timing] 1s\\n"}}')

        fake_api = types.SimpleNamespace(
            get_kernel_session_logs_stream=lambda request: Stream())
        monkeypatch.setattr(
            kagglesdk.kaggle_client, "KaggleClient",
            lambda env: types.SimpleNamespace(kernels=types.SimpleNamespace(
                kernels_api_client=fake_api)))
        # Each run is a fresh process: its first _log_lane truncates the file.
        monkeypatch.setattr(runtime, "_LANE_LOG_STARTED", False)
        kaggle_lane._log_lane(f"{tag} push rc=0")
        kaggle_lane.stream_kernel_logs("owner/er-train-gpu")

    run("first")
    run("second")
    content = (tmp_path / "logs" / "kaggle" / "lane.log").read_text()
    assert "first" not in content, "a new run must overwrite the old transcript"
    assert "second push rc=0" in content
    assert "second [timing] 1s" in content


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


# ── train-kernel bundle install: pinned checkout stays authoritative ─────────

def _train_install_loop_source() -> str:
    """The exact Bundle-boundary install block from the template.

    Sliced out so this pin runs the shipped source, not a re-typed copy: a
    regression that drops the config/scripts skip must fail the test below.
    The boundary load itself (exactly ONE ``Bundle.load`` per crossing) is
    pinned by test_train_kernel_verifies_the_attached_bundle_exactly_once.
    """
    body = kaggle_lane.TRAIN_KERNEL_BODY
    start = body.index("with inputs_bundle.reader() as archive:")
    end = body.index("package_manifest = root /", start)
    return body[start:end]


def test_train_install_loop_skips_code_and_config_keeps_data(tmp_path):
    """Regression for KeyError: 'source_code_dir' (owner order 2026-10-07).

    The bundle's embedded config/paths.yaml predates the source_code_dir
    layout; extracting it over the pinned checkout crashed
    model_tracks.package. The install loop must skip src/, config/ and
    scripts/ and install only the bundle's data.
    """
    import contextlib
    import zipfile

    source = _train_install_loop_source()
    # the pin names the crash and the owner order date, so the rationale
    # cannot be silently deleted without failing here
    assert "source_code_dir" in source
    assert "2026-10-07" in source
    assert '"config/"' in source and '"scripts/"' in source

    names = [
        "src/model_tracks/package.py",
        "src/core/common.py",
        "config/paths.yaml",
        "config/training.yaml",
        "scripts/encode_prepared_embeddings.py",
        "data/model_tracks/suite.yaml",
        "data/cohort/dataset.csv",
        "artifacts/models/checkpoint.bin",
        "model_tracks_package.json",
    ]
    archive_path = tmp_path / "all_tracks_inputs.zip"
    with zipfile.ZipFile(archive_path, "w") as bundle:
        for name in names:
            bundle.writestr(name, b"payload")

    root = tmp_path / "checkout"
    root.mkdir()
    lane = {"files": {"source_dir": "src",
                      "package_manifest": "model_tracks_package.json"}}

    @contextlib.contextmanager
    def _reader(path):
        with zipfile.ZipFile(path) as archive:
            yield archive

    class _InputsBundle:
        """The trusted handle the boundary load returns on the VM."""

        def reader(self):
            return _reader(archive_path)

    namespace = {"inputs_bundle": _InputsBundle(),
                 "archive_path": archive_path, "LANE": lane, "root": root}
    exec(compile(source, "<train-kernel-install>", "exec"), namespace)

    # pinned checkout neighborhoods never get clobbered by the bundle
    assert not (root / "src").exists()
    assert not (root / "config").exists()
    assert not (root / "scripts").exists()
    # only the bundle's data installs
    assert (root / "data" / "model_tracks" / "suite.yaml").read_bytes() == b"payload"
    assert (root / "data" / "cohort" / "dataset.csv").read_bytes() == b"payload"
    assert (root / "artifacts" / "models" / "checkpoint.bin").read_bytes() == b"payload"
    assert (root / "model_tracks_package.json").read_bytes() == b"payload"


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
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout="abc123def",
                                               stderr="")
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
    assert plan["published_tip"] == "abc123def"
    train_metadata = json.loads((tmp_path / "kaggle_stage" / "train_kernel"
                                 / "kernel-metadata.json").read_text())
    assert train_metadata["dataset_sources"] == ["owner/er-10k-bundle/12"], \
        "the train stage must attach the fresh published version"
    assert json.loads((tmp_path / "kaggle_stage" / "chain.receipt.json")
                      .read_text())["steps"]["train"]["stage"]["revision"] == "abc123def"


def test_chain_refuses_executed_run_on_stale_tip(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch,
                 gpu_kernel_slug="owner/er-train-gpu",
                 embedding_kernel_slug="owner/er-embed",
                 bundle_dataset_slug="owner/er-10k-bundle")
    _isolate_credentials(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    _fake_published_tip(monkeypatch, "fed321cba9")
    with pytest.raises(RuntimeError, match="pull or push"):
        kaggle_lane.run_chain(execute=True)
    # a refused chain never staged a payload or wrote a receipt
    assert not list(tmp_path.rglob("kernel-metadata.json"))


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


# ── session-id capture: launch path records the id, stop consumes it ────────

def _fake_stream_client(monkeypatch, url: str, closed: list[str]):
    """A fake KaggleClient whose log-stream probe returns ``url`` and closes."""
    import types
    import kagglesdk.kaggle_client

    class FakeApi:
        def get_kernel_session_logs_stream(self, request):
            return types.SimpleNamespace(url=url, close=lambda: closed.append(url))

    client = types.SimpleNamespace(kernels=types.SimpleNamespace(
        kernels_api_client=FakeApi()))
    monkeypatch.setattr(kagglesdk.kaggle_client, "KaggleClient", lambda env: client)


def test_clear_kernel_session_id_removes_existing_and_is_noop(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    session_file = tmp_path / "logs" / "kaggle" / "er-bundle-cpu.session_id"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text("111222\n")
    kaggle_lane.clear_kernel_session_id("owner/er-bundle-cpu")
    assert not session_file.exists(), "a fresh push drops the previous run's id"
    # absent id file -> no-op, never raises (a push path clears unconditionally)
    kaggle_lane.clear_kernel_session_id("owner/er-bundle-cpu")
    assert not session_file.exists()


def test_capture_kernel_session_id_writes_id_from_stream_url(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    closed: list[str] = []
    url = "https://www.kaggle.com/api/i/kernels/987654321?x=1"
    _fake_stream_client(monkeypatch, url, closed)
    monkeypatch.setattr(kaggle_lane.time, "sleep", lambda seconds: None)
    plan = kaggle_lane.capture_kernel_session_id("owner/er-bundle-cpu", attempts=1)
    assert plan["session_id"] == 987654321
    session_file = tmp_path / "logs" / "kaggle" / "er-bundle-cpu.session_id"
    assert session_file.read_text() == "987654321\n"
    assert plan["session_id_file"] == str(session_file)
    assert closed == [url], "capture must close the response without following it"


def test_capture_kernel_session_id_none_when_url_has_no_id(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    closed: list[str] = []
    _fake_stream_client(
        monkeypatch,
        "https://www.kaggle.com/api/i/kernels.GetKernelSessionLogsStream", closed)
    monkeypatch.setattr(kaggle_lane.time, "sleep", lambda seconds: None)
    plan = kaggle_lane.capture_kernel_session_id("owner/er-bundle-cpu", attempts=2)
    assert plan["session_id"] is None
    assert not (tmp_path / "logs" / "kaggle" / "er-bundle-cpu.session_id").exists(), \
        "no id in the URL -> no file, no fabricated session"
    assert len(closed) == 2, "each bounded attempt still closes its response"


def test_push_and_record_session_clears_before_push_and_captures_after(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    stage = tmp_path / "kaggle_stage" / "bundle_kernel"
    stage.mkdir(parents=True)
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(kaggle_lane, "_require_kaggle_executable",
                        lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(kaggle_lane, "clear_kernel_session_id",
                        lambda slug: events.append(("clear", slug)))
    monkeypatch.setattr(kaggle_lane, "_run_kaggle",
                        lambda command: events.append(("push", list(command))) or (0, ""))
    monkeypatch.setattr(kaggle_lane, "capture_kernel_session_id",
                        lambda slug: events.append(("capture", slug)))
    kaggle_lane.KaggleKernels._push_and_record_session(stage, "owner/er-bundle-cpu")
    assert [kind for kind, _ in events] == ["clear", "push", "capture"]
    assert events[0] == ("clear", "owner/er-bundle-cpu")
    assert events[2] == ("capture", "owner/er-bundle-cpu")


def test_push_kernel_records_session_around_push(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    stage = tmp_path / "kaggle_stage" / "bundle_kernel"
    stage.mkdir(parents=True)
    (stage / "kernel-metadata.json").write_text(json.dumps(
        {"id": "owner/er-bundle-cpu", "code_file": "bundle_cpu.py"}))
    (stage / "bundle_cpu.py").write_text("# stub\n")
    import importlib
    monkeypatch.setattr(importlib.import_module("core.runtime_inputs"),
                        "staged_kernel_preflight", lambda stage_dir: None)
    monkeypatch.setattr(kaggle_lane, "_spawn_autowatch", lambda *a, **kw: {})
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(kaggle_lane, "_require_kaggle_executable",
                        lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(kaggle_lane, "clear_kernel_session_id",
                        lambda slug: events.append(("clear", slug)))
    monkeypatch.setattr(kaggle_lane, "_run_kaggle",
                        lambda command: events.append(("push", list(command))) or (0, ""))
    monkeypatch.setattr(kaggle_lane, "capture_kernel_session_id",
                        lambda slug: events.append(("capture", slug)))
    plan = kaggle_lane.push_kernel(stage)
    assert plan["pushed"] is True
    assert [kind for kind, _ in events] == ["clear", "push", "capture"]


def test_push_and_record_session_capture_failure_does_not_abort_push(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch)
    stage = tmp_path / "kaggle_stage" / "bundle_kernel"
    stage.mkdir(parents=True)
    logged: list[str] = []
    monkeypatch.setattr(kaggle_lane, "_require_kaggle_executable",
                        lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(kaggle_lane, "clear_kernel_session_id", lambda slug: None)
    monkeypatch.setattr(kaggle_lane, "_run_kaggle", lambda command: (0, ""))
    monkeypatch.setattr(kaggle_lane, "capture_kernel_session_id",
                        lambda slug: (_ for _ in ()).throw(RuntimeError("proxy down")))
    monkeypatch.setattr(kaggle_lane, "_log_lane", lambda line: logged.append(line))
    kaggle_lane.KaggleKernels._push_and_record_session(stage, "owner/er-bundle-cpu")
    assert any("session-id capture skipped" in line for line in logged), \
        "a best-effort capture failure is logged, never raised onto the push"


def test_stop_kernel_no_wait_with_session_id_requests_sdk_cancel(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    session_file = tmp_path / "logs" / "kaggle" / "er-train-gpu.session_id"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text("424242\n")
    cancels: list[int] = []
    _fake_sdk_cancel(monkeypatch, cancels)
    monkeypatch.setattr(kaggle_lane, "kernel_status",
                        lambda *a, **kw: pytest.fail("wait=False must not poll status"))
    plan = kaggle_lane.stop_kernel("owner/er-train-gpu", which="gpu",
                                   execute=True, wait=False)
    assert plan["verdict"] == "requested"
    assert plan["stopped"] is None
    assert plan["cancel_method"] == "sdk_cancel_kernel_session"
    assert cancels == [424242], "the recorded id must still drive the cancel"


def test_stop_kernel_no_wait_without_session_id_pushes_stub_and_requests(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    pushes: list[list[str]] = []
    monkeypatch.setattr(kaggle_lane, "_run_kaggle",
                        lambda command: pushes.append(list(command)) or (0, ""))
    monkeypatch.setattr(kaggle_lane, "_require_kaggle_executable",
                        lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(kaggle_lane, "kernel_status",
                        lambda *a, **kw: pytest.fail("wait=False must not poll status"))
    plan = kaggle_lane.stop_kernel("owner/er-train-gpu", which="gpu",
                                   execute=True, wait=False)
    assert plan["verdict"] == "requested"
    assert plan["stopped"] is None
    assert plan["cancel_method"] == "version_replace"
    assert any("kernels" in parts and "push" in parts for parts in pushes), \
        "no id -> the stub replace is still issued"
    # the stop stub push is not a real launch: it must never capture an id
    assert not (tmp_path / "logs" / "kaggle" / "er-train-gpu.session_id").exists()


def test_stop_kernel_wait_true_version_replace_fails_loud(tmp_path, monkeypatch):
    import types
    import cli.kaggle_kernels

    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu")
    monkeypatch.setattr(kaggle_lane, "_run_kaggle", lambda command: (0, ""))
    monkeypatch.setattr(kaggle_lane, "_require_kaggle_executable",
                        lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(kaggle_lane, "kernel_status",
                        lambda *a, **kw: {"status": "running", "raw": "RUNNING"})
    ticks = iter([0.0, 10.0, 10_000.0])
    sleeps: list[float] = []
    monkeypatch.setattr(cli.kaggle_kernels, "time", types.SimpleNamespace(
        monotonic=lambda: next(ticks),
        sleep=lambda seconds: sleeps.append(seconds)))
    with pytest.raises(RuntimeError, match="stop did not reach a terminal state"):
        kaggle_lane.stop_kernel("owner/er-train-gpu", which="gpu", execute=True)
    assert sleeps == [15.0], "verify polls run logs_poll_seconds apart"


# ── finalize lane job (bundle_steps role=result, remote CPU) ────────────────

def _finalize_spec(tmp_path, monkeypatch, **updates):
    values = {"gpu_kernel_slug": "owner/er-train-gpu",
              "bundle_dataset_slug": "owner/er-10k-bundle"}
    values.update(updates)
    return _kernel_spec(tmp_path, monkeypatch, **values)


def test_stage_finalize_kernel_pins_revision_and_attaches_both_bundles(tmp_path, monkeypatch):
    spec = _finalize_spec(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    _fake_published_tip(monkeypatch, "abc123def")
    receipt = kaggle_lane.stage_finalize_kernel(revision="abc123def", run_tag="gpu_x")
    stage = tmp_path / "kaggle_stage" / "finalize_kernel"
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    # the finalize job is a second version of the BUNDLING CPU kernel slug
    assert metadata["id"] == "owner/er-bundle-cpu"
    assert metadata["code_file"] == "finalize_cpu.py"
    assert metadata["enable_gpu"] is False and metadata["enable_internet"] is True
    # both verified inputs attach: the published inputs bundle + the trained result
    assert metadata["dataset_sources"] == ["owner/er-10k-bundle"]
    assert metadata["kernel_sources"] == ["owner/er-train-gpu"]
    script = (stage / "finalize_cpu.py").read_text()
    assert 'REVISION = "abc123def"' in script
    assert "Bundle.load(" in script and "BundlePipeline" in script
    assert '"result"' in script and "expected_digest=" in script
    assert "finalized_bundle" in script
    assert "sparse-checkout" in script and "--no-cone" in script
    assert receipt["kind"] == "finalize" and receipt["role"] == "result"
    assert receipt["gpu"] is False and receipt["revision"] == "abc123def"
    assert receipt["published_tip"] == "abc123def"
    assert receipt["bundle_dataset"] == "owner/er-10k-bundle"
    assert receipt["result_kernel"] == "owner/er-train-gpu"


def test_stage_finalize_kernel_refuses_stale_published_tip(tmp_path, monkeypatch):
    _finalize_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    _fake_published_tip(monkeypatch, "fed321cba9")
    with pytest.raises(RuntimeError, match="pull or push"):
        kaggle_lane.stage_finalize_kernel()


def test_finalize_kernel_runs_bundling_from_a_verified_sparse_checkout(tmp_path, monkeypatch):
    """Item 3: the finalize job's sparse checkout must carry everything the
    bundling step reads — verified against the REAL repo inventory (no fakes:
    `checkout_members`/`checkout_inventory` run their git ls-files), so an
    untracked or missing finalize input fails this test instead of the VM."""
    import ast

    from core import runtime_inputs

    spec = _finalize_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    # The published-tip guard is exercised by its own tests; here the REAL
    # `git ls-files` inventory must run, so only the tip lookup is faked
    # (patching subprocess.run would break check_output's ls-files).
    monkeypatch.setattr(runtime_inputs, "published_tip", lambda repository, branch: "abc123def")
    kaggle_lane.stage_finalize_kernel(revision="abc123def")
    script = (tmp_path / "kaggle_stage" / "finalize_kernel" / "finalize_cpu.py").read_text()
    literals = {}
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                        "CHECKOUT_PATHS", "_runtime_files"}:
                    literals[target.id] = ast.literal_eval(node.value)
    assert tuple(literals["CHECKOUT_PATHS"]) == runtime_inputs.checkout_members(
        spec.checkout_paths, lane="bundle"), \
        "the staged sparse selection must be the shared runtime selection"
    selection = set(literals["CHECKOUT_PATHS"])
    # every neighborhood the finalize step reads (src model_tracks/graph_tracks,
    # config incl. the ablation config, the wheels/evidence artifacts, the
    # git-shipped models, the prepared smoke tree, the repo metadata)
    assert {"src", "scripts", "config", "requirements", "artifacts/wheels",
            "artifacts/evidence", "artifacts/models", "pyproject.toml",
            "requirements.txt", "colab_backend.py", "dataset.csv"} <= selection
    # ... and the inventory the job verifies BEFORE it installs dependencies
    inventory = set(literals["_runtime_files"])
    assert tuple(literals["_runtime_files"]) == runtime_inputs.checkout_inventory(
        spec.checkout_paths, lane="bundle")
    assert {"src/model_tracks/bundle_steps.py", "src/core/bundle.py",
            "config/model_tracks.yaml"} <= inventory
    # the step itself is told the sparse selection it runs from (item 3)
    assert "sparse_paths=tuple(CHECKOUT_PATHS)" in script


def test_chain_plan_places_the_finalize_step_after_train(tmp_path, monkeypatch):
    _finalize_spec(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    _isolate_credentials(tmp_path, monkeypatch)
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    plan = kaggle_lane.run_chain(cohort="10k", with_finalize=True, execute=False)
    assert plan["with_finalize"] is True and plan["mode"] == "dry-run"
    assert list(plan["steps"]) == ["bundle", "train", "finalize"], \
        "the finalize job is the last step and only runs when asked for"
    assert plan["slugs"]["finalize"] == "owner/er-bundle-cpu"
    finalize = plan["steps"]["finalize"]
    assert finalize["stage"]["kernel"] == "owner/er-bundle-cpu"
    assert finalize["role"] == "result"
    assert finalize["mount"] == {
        "dataset_sources": ["owner/er-10k-bundle/<fresh version>"],
        "kernel_sources": ["owner/er-train-gpu"],
    }
    assert not list(tmp_path.rglob("kernel-metadata.json")), \
        "a dry-run chain stages nothing at all"


def test_chain_runs_the_finalize_job_with_its_own_watcher(tmp_path, monkeypatch):
    _finalize_spec(tmp_path, monkeypatch,
                   embedding_kernel_slug="owner/er-embed-gpu",
                   embedding_dataset_slug="owner/er-embed-requests")
    _isolate_credentials(tmp_path, monkeypatch)
    _hermetic_staging(monkeypatch)
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")

    def fake_run(command, **kwargs):
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout="abc123def", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="complete", stderr="")

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    spawns: list[str] = []

    def fake_spawn(watcher):
        watcher = kaggle_lane.AUTOWATCH_WHICH.get(watcher, watcher)
        spawns.append(watcher)
        kind = {"cpu": "bundle", "gpu": "train", "embed": "embed",
                "finalize": "finalize"}[watcher]
        receipt = tmp_path / "kaggle_stage" / f"autowatch_{kind}.receipt.json"
        receipt.write_text(json.dumps({
            "status": "complete", "polls": 2,
            "fetch": {"verified": True, "archive_sha256": "d" * 64, "cohort": "10k",
                      "publish": ({"published": True, "slug": "owner/er-10k-bundle",
                                   "dataset_version": 12} if kind == "bundle" else {})},
            "stop": {"stopped": True},
        }))
        return {"autowatch": "spawned", "kernel": watcher, "log": str(receipt)}

    monkeypatch.setattr(kaggle_lane, "_spawn_autowatch", fake_spawn)
    plan = kaggle_lane.run_chain(cohort="10k", with_finalize=True, execute=True)
    assert spawns == ["cpu", "gpu", "finalize"], \
        "one watcher per pushed kernel, the finalize job included"
    finalize = plan["steps"]["finalize"]
    assert finalize["stage"]["kind"] == "finalize"
    assert finalize["stage"]["role"] == "result"
    assert finalize["fetched_sha256"] == "d" * 64
    assert finalize["push"]["kernel"] == "finalize"
    metadata = json.loads((tmp_path / "kaggle_stage" / "finalize_kernel"
                           / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == ["owner/er-10k-bundle/12"], \
        "the finalize job attaches the version this chain published"
    assert metadata["kernel_sources"] == ["owner/er-train-gpu"]
    staged = (tmp_path / "kaggle_stage" / "finalize_kernel" / "finalize_cpu.py").read_text()
    assert "sparse-checkout" in staged and "BundlePipeline" in staged


def test_train_kernel_verifies_the_attached_bundle_exactly_once(tmp_path, monkeypatch):
    """Item 1a (Kaggle): the GPU kernel's install does ONE integrity check of
    the attached inputs Bundle at its boundary — no second whole-archive hash,
    no per-member re-verification."""
    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu",
                 bundle_dataset_slug="owner/er-10k-bundle")
    _hermetic_staging(monkeypatch)
    _fake_published_tip(monkeypatch, "abc123def")
    kaggle_lane.stage_gpu_kernel(kind="train", revision="abc123def")
    script = (tmp_path / "kaggle_stage" / "train_kernel" / "train_gpu.py").read_text()
    assert script.count("Bundle.load(") == 1
    assert 'expected_digest=receipt.get("archive_sha256")' in script
    assert "verified_archive" not in script
    assert "sha256_file(archive_path)" not in script


def test_fetch_finalize_output_identifies_the_sealed_result_bundle(tmp_path, monkeypatch):
    """Item 1a (Kaggle fetch): the fetched sealed result bundle is named by the
    ONE boundary load, so the operator box gets a trusted run-tagged handle."""
    import hashlib

    from core.bundle import Bundle

    _finalize_spec(tmp_path, monkeypatch)
    member = tmp_path / "member.txt"
    member.write_text("sealed", encoding="utf-8")
    sealed = Bundle.seal_archive(tmp_path / "sealed.tar.zst", {"tracks/a.txt": member},
                                 role="result", metadata={"run_tag": "gpu_test"})
    archive_bytes = (tmp_path / "sealed.tar.zst").read_bytes()

    def fake_run(command, **kwargs):
        stage = Path(command[command.index("-p") + 1])
        bundle = stage / "bundle"
        bundle.mkdir(parents=True)
        (bundle / "finalized_bundle.tar.zst").write_bytes(archive_bytes)
        (bundle / "finalized_bundle.tar.zst.sha256").write_text(sealed.digest + "\n")
        (bundle / "finalized_bundle.manifest.json").write_text(json.dumps({
            "kind": "finalized_bundle", "role": "result", "run_tag": "gpu_test",
            "cohort": "10k", "archive": "finalized_bundle.tar.zst",
            "archive_sha256": hashlib.sha256(archive_bytes).hexdigest()}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    plan = kaggle_lane.fetch_kernel_output(kind="finalize", execute=True)
    assert plan["verified"] is True and plan["cohort"] == "10k"
    assert plan["bundle"] == {"identified": True, "role": "result",
                              "sha256": sealed.digest, "members": 2,
                              "run_tag": "gpu_test"}, \
        "the sealed tree's one member plus the container manifest"
    installed = tmp_path / "kaggle_stage" / "10k" / "finalize" / "finalized_bundle.tar.zst"
    assert installed.read_bytes() == archive_bytes


def test_role_archive_fetch_hashes_the_archive_exactly_once(tmp_path, monkeypatch):
    """Finding 2: a bundle-role fetch performs ONE whole-archive integrity read.

    The role's boundary load verifies the archive digest AND its member
    inventory in a single pass, so the fetch must not hash the archive again
    itself — a second ``sha256_file`` here would be the redundant read the audit
    flagged.
    """
    import hashlib

    from core.bundle import Bundle

    _finalize_spec(tmp_path, monkeypatch)
    member = tmp_path / "member.txt"
    member.write_text("sealed", encoding="utf-8")
    sealed = Bundle.seal_archive(tmp_path / "sealed.tar.zst", {"tracks/a.txt": member},
                                 role="result", metadata={"run_tag": "gpu_test"})
    archive_bytes = (tmp_path / "sealed.tar.zst").read_bytes()

    def fake_run(command, **kwargs):
        stage = Path(command[command.index("-p") + 1])
        bundle = stage / "bundle"
        bundle.mkdir(parents=True)
        (bundle / "finalized_bundle.tar.zst").write_bytes(archive_bytes)
        (bundle / "finalized_bundle.tar.zst.sha256").write_text(sealed.digest + "\n")
        (bundle / "finalized_bundle.manifest.json").write_text(json.dumps({
            "kind": "finalized_bundle", "role": "result", "run_tag": "gpu_test",
            "cohort": "10k", "archive": "finalized_bundle.tar.zst",
            "archive_sha256": hashlib.sha256(archive_bytes).hexdigest()}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(kaggle_lane, "sha256_file",
                        lambda path: pytest.fail(f"redundant archive hash of {path}"))
    plan = kaggle_lane.fetch_kernel_output(kind="finalize", execute=True)
    assert plan["verified"] is True
    assert plan["archive_sha256"] == sealed.digest
    assert plan["bundle"]["identified"] is True


def test_non_role_archive_fetch_hashes_once_without_a_bundle_load(tmp_path, monkeypatch):
    """The complement: an embed output has no bundle role, so the fetch's own
    digest IS its single integrity check (one read, no boundary load)."""
    import hashlib

    _kernel_spec(tmp_path, monkeypatch,
                 embedding_kernel_slug="owner/er-embed-gpu")
    archive_bytes = b"fake embed vectors archive bytes"

    def fake_run(command, **kwargs):
        stage = Path(command[command.index("-p") + 1])
        out = stage / "embed"
        out.mkdir(parents=True)
        (out / "vectors.tar.zst").write_bytes(archive_bytes)
        (out / "vectors.manifest.json").write_text(json.dumps({
            "kind": "vectors", "cohort": "10k",
            "archive": "vectors.tar.zst",
            "archive_sha256": hashlib.sha256(archive_bytes).hexdigest()}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    real_sha256_file = kaggle_lane.sha256_file
    hashed: list[str] = []

    def counting_sha256(path):
        hashed.append(str(path))
        return real_sha256_file(path)

    monkeypatch.setattr(kaggle_lane, "sha256_file", counting_sha256)
    plan = kaggle_lane.fetch_kernel_output(kind="embed", execute=True)
    assert plan["verified"] is True
    assert len(hashed) == 1, "exactly one whole-archive digest for a non-role kind"
    assert plan["bundle"] == {"identified": False, "role": None,
                              "note": "fetched 'embed' output is not a bundle role archive"}


def test_fetch_train_output_role_loads_the_sealed_result_bundle(tmp_path, monkeypatch):
    """The train kernel's identity carries the result role (the handoff fix).

    The train kernel ships ``model_tracks.run``'s sealed result Bundle, so its
    fetched output is named by the SAME boundary load the finalize job performs:
    the whole-archive digest and the member inventory are verified in that one
    pass, and no second whole-archive hash runs at the fetch.
    """
    import hashlib

    from core.bundle import Bundle

    _kernel_spec(tmp_path, monkeypatch, gpu_kernel_slug="owner/er-train-gpu",
                 bundle_dataset_slug="owner/er-10k-bundle")
    assert kaggle_lane.kernel_identity("train").bundle_role == "result"
    member = tmp_path / "member.txt"
    member.write_text("sealed", encoding="utf-8")
    sealed = Bundle.seal_archive(tmp_path / "sealed.tar.zst", {"tracks/a.txt": member},
                                 role="result", metadata={"run_tag": "gpu_test"})
    archive_bytes = (tmp_path / "sealed.tar.zst").read_bytes()

    def fake_run(command, **kwargs):
        stage = Path(command[command.index("-p") + 1])
        out = stage / "train"
        out.mkdir(parents=True)
        (out / "result_bundle.tar.zst").write_bytes(archive_bytes)
        (out / "result_bundle.tar.zst.sha256").write_text(sealed.digest + "\n")
        (out / "result_bundle.manifest.json").write_text(json.dumps({
            "kind": "result_bundle", "role": "result", "run_tag": "gpu_test",
            "cohort": "10k", "archive": "result_bundle.tar.zst",
            "archive_sha256": hashlib.sha256(archive_bytes).hexdigest()}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(kaggle_lane, "sha256_file",
                        lambda path: pytest.fail(f"redundant archive hash of {path}"))
    plan = kaggle_lane.fetch_kernel_output(kind="train", execute=True)
    assert plan["verified"] is True and plan["archive_sha256"] == sealed.digest
    assert plan["bundle"] == {"identified": True, "role": "result",
                              "sha256": sealed.digest, "members": 2,
                              "run_tag": "gpu_test"}


# ── the embed objective: explicit, never silently absent ────────────────────

def test_embed_objective_absent_is_explicit_and_never_silent(tmp_path, monkeypatch):
    _kernel_spec(tmp_path, monkeypatch, cpu_kernel_slug="owner/er-bundle-cpu",
                 embedding_kernel_slug=None, embedding_dataset_slug=None)
    verdict = kaggle_lane.embed_objective(execute=False)
    assert verdict["configured"] is False and verdict["available"] is False
    assert "embedding_kernel_slug" in verdict["reason"]
    with pytest.raises(RuntimeError) as error:
        kaggle_lane.require_embed_objective(execute=False)
    # the two named fixes: push the kernel, or drop the step (never skip it)
    assert "--what embed-kernel --execute" in str(error.value)
    assert "never silently skipped" in str(error.value)


def test_embed_objective_configured_but_missing_on_the_account(tmp_path, monkeypatch):
    """The phantom-kernel case: config names fbarulli/er-embed-gpu, the account
    has no such kernel, and the step must fail loud (not disappear)."""
    _kernel_spec(tmp_path, monkeypatch,
                 embedding_kernel_slug="owner/er-embed-gpu",
                 embedding_dataset_slug="owner/er-embed-requests")
    monkeypatch.setattr(kaggle_lane.shutil, "which", lambda name: "/usr/bin/kaggle")

    def fail(command):
        raise RuntimeError("kaggle command failed (rc=1): kernels status owner/er-embed-gpu")

    monkeypatch.setattr(kaggle_lane, "_run_kaggle", fail)
    verdict = kaggle_lane.embed_objective(execute=True)
    assert verdict["configured"] is True and verdict["available"] is False
    assert "could not be reached on the account" in verdict["reason"]
    with pytest.raises(RuntimeError, match="embed objective unavailable"):
        kaggle_lane.require_embed_objective(execute=True)
    # a dry run never probes the account, so it reports the unknown state instead
    assert kaggle_lane.embed_objective(execute=False)["available"] is None


def test_chain_with_embed_refuses_an_unconfigured_objective(tmp_path, monkeypatch):
    """`--with-embed` with a half-configured objective (kernel named, request
    dataset missing) fails loud naming the objective, never skipping the step."""
    _finalize_spec(tmp_path, monkeypatch,
                   embedding_kernel_slug="owner/er-embed-gpu",
                   embedding_dataset_slug=None)
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    with pytest.raises(RuntimeError, match="embed objective"):
        kaggle_lane.run_chain(cohort="10k", with_embed=True, execute=False)

