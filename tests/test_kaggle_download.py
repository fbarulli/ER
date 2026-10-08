"""Public-API tests for the fail-loud, 429-aware download fetchers.

No test touches the network or the kaggle CLI: both downloaders take an
injected runner + sleep, so 429 pacing, empty-output, retry and verification
are exercised deterministically.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from cli.kaggle_download import (
    DatasetDownloader,
    DownloadError,
    EmptyDownloadError,
    KernelOutputFetcher,
)
from cli.kaggle_monitor import ReconnectBackoff


def _backoff() -> ReconnectBackoff:
    return ReconnectBackoff(base_seconds=1.0, cap_seconds=60.0,
                            rate_limit_cap_seconds=300.0, jitter=0.0)


def _dest_of(command: list[str]) -> Path:
    return Path(command[command.index("-p") + 1])


def _fw(command: list[str], *names: str) -> "subprocess.CompletedProcess[str]":
    dest = _dest_of(command)
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        (dest / name).write_bytes(b"payload")
    return subprocess.CompletedProcess(command, 0, stdout="")


def _ok(command: list[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.CompletedProcess(command, 0, stdout="")


# ── KernelOutputFetcher ────────────────────────────────────────────────────

def test_kernel_fetch_verifies_the_landed_archive(tmp_path):
    result = KernelOutputFetcher(
        argv_prefix=("kaggle",), runner=lambda command: _fw(command, "laya.tar.gz"),
        sleep=lambda _: None, backoff=_backoff()).fetch(
        "owner/slug", tmp_path / "out")
    assert result.attempts == 1 and not result.empty
    assert [path.name for path in result.files] == ["laya.tar.gz"]
    assert result.command[:2] == ("kaggle", "kernels")


def test_kernel_fetch_retries_429_then_succeeds(tmp_path):
    calls = {"n": 0}
    delays: list[float] = []

    def runner(command):
        calls["n"] += 1
        if calls["n"] == 1:
            return subprocess.CompletedProcess(
                command, 1,
                stdout="429 Client Error: Too Many Requests for url: https://x")
        return _fw(command, "a.tar.gz")

    result = KernelOutputFetcher(
        argv_prefix=("kaggle",), runner=runner, sleep=delays.append,
        backoff=_backoff(), max_attempts=3).fetch("owner/slug", tmp_path / "out")
    assert result.attempts == 2 and calls["n"] == 2
    assert len(delays) == 1


def test_kernel_fetch_honours_retry_after(tmp_path):
    calls = {"n": 0}
    delays: list[float] = []

    def runner(command):
        calls["n"] += 1
        if calls["n"] == 1:
            return subprocess.CompletedProcess(
                command, 1,
                stdout="429 Too Many Requests Retry-After: 120")
        return _fw(command, "a.tar.gz")

    KernelOutputFetcher(
        argv_prefix=("kaggle",), runner=runner, sleep=delays.append,
        backoff=_backoff(), max_attempts=3).fetch("owner/slug", tmp_path / "out")
    assert delays and delays[0] >= 120.0


def test_empty_kernel_output_fails_loud(tmp_path):
    delays: list[float] = []
    with pytest.raises(EmptyDownloadError, match="wrote no files") as excinfo:
        KernelOutputFetcher(
            argv_prefix=("kaggle",), runner=_ok, sleep=delays.append,
            backoff=_backoff(), max_attempts=2).fetch(
            "owner/slug", tmp_path / "out")
    # one retry before the loud failure; the dir is created but stays empty
    assert len(delays) == 1
    assert not any((tmp_path / "out").iterdir())
    assert "kaggle_download.py" in excinfo.value.traceback_text


def test_non_rate_limit_failure_is_not_retried(tmp_path):
    calls = {"n": 0}

    def runner(command):
        calls["n"] += 1
        return subprocess.CompletedProcess(command, 1, stdout="403 Forbidden")

    with pytest.raises(DownloadError) as excinfo:
        KernelOutputFetcher(
            argv_prefix=("kaggle",), runner=runner, sleep=lambda _: None,
            backoff=_backoff(), max_attempts=3).fetch(
            "owner/slug", tmp_path / "out")
    assert calls["n"] == 1
    assert not isinstance(excinfo.value, EmptyDownloadError)


def test_failure_keeps_the_full_cli_output(tmp_path):
    body = "Z" * 5000

    def runner(command):
        return subprocess.CompletedProcess(command, 1, stdout=body)

    with pytest.raises(DownloadError) as excinfo:
        KernelOutputFetcher(
            argv_prefix=("kaggle",), runner=runner, sleep=lambda _: None,
            backoff=_backoff()).fetch("owner/slug", tmp_path / "out")
    assert body in str(excinfo.value)  # never truncated to a one-liner


def test_kernel_fetch_requires_an_archive(tmp_path):
    with pytest.raises(DownloadError, match="none matched required"):
        KernelOutputFetcher(
            argv_prefix=("kaggle",), runner=lambda command: _fw(command, "run.log"),
            sleep=lambda _: None, backoff=_backoff()).fetch(
            "owner/slug", tmp_path / "out")


def test_kernel_fetch_reports_resume_into_an_existing_dir(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "existing.tar.gz").write_bytes(b"old")
    result = KernelOutputFetcher(
        argv_prefix=("kaggle",), runner=_ok, sleep=lambda _: None,
        backoff=_backoff()).fetch("owner/slug", dest, require_globs=())
    assert result.resumed is True and result.attempts == 1


def test_runner_exception_carries_the_full_traceback(tmp_path):
    def runner(command):
        raise RuntimeError("socket reset while downloading")

    with pytest.raises(DownloadError) as excinfo:
        KernelOutputFetcher(
            argv_prefix=("kaggle",), runner=runner, sleep=lambda _: None,
            backoff=_backoff(), max_attempts=1).fetch(
            "owner/slug", tmp_path / "out")
    assert "Traceback" in excinfo.value.traceback_text
    assert "RuntimeError: socket reset while downloading" in excinfo.value.traceback_text


def test_bad_kernel_slug_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="owner/slug"):
        KernelOutputFetcher(argv_prefix=("kaggle",), runner=_ok).fetch(
            "noslug", tmp_path / "out")


# ── DatasetDownloader ──────────────────────────────────────────────────────

def _zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)


def test_dataset_download_unzips_and_verifies_sha(tmp_path):
    source = tmp_path / "source.zip"
    _zip(source, {"train.jsonl": b"row\n", "receipt.json": b"{}"})
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    dest = tmp_path / "dl"

    def runner(command):
        _dest_of(command).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, _dest_of(command) / "source.zip")
        return _ok(command)

    result = DatasetDownloader(
        argv_prefix=("kaggle",), runner=runner, sleep=lambda _: None,
        backoff=_backoff()).download("owner/data", dest, sha256=expected,
                                     unzip=True)
    assert (dest / "train.jsonl").is_file() and (dest / "receipt.json").is_file()
    assert any(path.name == "source.zip" for path in result.files)
    assert result.command[:2] == ("kaggle", "datasets")


def test_dataset_download_rejects_sha_drift(tmp_path):
    source = tmp_path / "source.zip"
    _zip(source, {"a.txt": b"x"})
    dest = tmp_path / "dl"

    def runner(command):
        _dest_of(command).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, _dest_of(command) / "source.zip")
        return _ok(command)

    with pytest.raises(DownloadError, match="sha256 mismatch"):
        DatasetDownloader(
            argv_prefix=("kaggle",), runner=runner, sleep=lambda _: None,
            backoff=_backoff()).download("owner/data", dest, sha256="0" * 64)


def test_dataset_download_retries_429(tmp_path):
    calls = {"n": 0}
    delays: list[float] = []

    def runner(command):
        calls["n"] += 1
        if calls["n"] == 1:
            return subprocess.CompletedProcess(command, 1, stdout="429")
        _dest_of(command).mkdir(parents=True, exist_ok=True)
        _zip(_dest_of(command) / "data.zip", {"x.txt": b"y"})
        return _ok(command)

    result = DatasetDownloader(
        argv_prefix=("kaggle",), runner=runner, sleep=delays.append,
        backoff=_backoff(), max_attempts=3).download("owner/data", tmp_path / "dl")
    assert result.attempts == 2 and len(delays) == 1


def test_dataset_download_empty_output_fails_loud(tmp_path):
    with pytest.raises(EmptyDownloadError, match="wrote no files"):
        DatasetDownloader(
            argv_prefix=("kaggle",), runner=_ok, sleep=lambda _: None,
            backoff=_backoff(), max_attempts=1).download(
            "owner/data", tmp_path / "dl")


def test_bad_dataset_slug_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="owner/slug"):
        DatasetDownloader(argv_prefix=("kaggle",), runner=_ok).download(
            "noslug", tmp_path / "dl")
