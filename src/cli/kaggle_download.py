"""src/cli/kaggle_download.py — fail-loud, 429-aware Kaggle downloads.

The JOB 1 regression: ``kaggle kernels output`` returns exit code 0 even when
it downloads ZERO files (the upstream CLI never checks its own returned
file list), and the status/output endpoints answer ``429 Too Many Requests``
under load. The lane's old fetch path trusted ``rc == 0`` and reported
success, so a dead/empty kernel output looked like a clean download and the
checkpoint was silently lost.

Two single-responsibility classes fix that at the transport boundary:

* :class:`KernelOutputFetcher` — ``kaggle kernels output`` with 429-aware
  exponential backoff (+ ``Retry-After``), resume-into-the-same-directory
  retries, post-download verification and a fail-LOUD contract: an rc=0 with
  no files raises :class:`EmptyDownloadError`, never a silent empty success.
* :class:`DatasetDownloader` — ``kaggle datasets download`` with the same
  pacing, plus unzip and sha256 verification. This is the ONE dataset-archive
  transport: :meth:`cli.kaggle_datasets.KaggleDatasets.download_dataset`
  delegates here (receipt identity included) instead of re-spelling its own
  ``kaggle datasets download`` call, so there is no second downloader.

Every failure carries the FULL CLI output and the FULL Python traceback
(``DownloadError.traceback_text``); nothing is truncated to a one-liner.
"""
from __future__ import annotations

import fnmatch
import hashlib
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from cli.kaggle_monitor import ReconnectBackoff

StrPath = str | Path
Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class DownloadResult:
    """What a verified download actually landed on disk."""

    slug: str
    dest: Path
    files: tuple[Path, ...]
    attempts: int
    command: tuple[str, ...]
    stdout: str = ""
    resumed: bool = False
    archive: Path | None = None

    @property
    def empty(self) -> bool:
        return not self.files


class DownloadError(RuntimeError):
    """A download failed; ``traceback_text`` carries the FULL traceback."""

    def __init__(self, message: str, *, traceback_text: str = "",
                 stdout: str = ""):
        super().__init__(message)
        self.traceback_text = traceback_text
        self.stdout = stdout


class EmptyDownloadError(DownloadError):
    """rc=0 but no files were written — the silent-success bug."""


def _matches_any(path: Path, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def _stack() -> str:
    """A full call stack for failures with no live exception (never ``NoneType``)."""
    return "".join(traceback.format_stack())


class _RetryingDownloader:
    """Shared 429-aware retry/verify engine for the two download verbs."""

    def __init__(self, *, argv_prefix: Sequence[str] | None = None,
                 cwd: StrPath | None = None, max_attempts: int = 5,
                 backoff: ReconnectBackoff | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 runner: Runner | None = None):
        self._prefix = list(argv_prefix or (sys.executable, "-m", "kaggle"))
        self._cwd = Path(cwd) if cwd is not None else None
        self._max_attempts = max(1, int(max_attempts))
        self._backoff = backoff or ReconnectBackoff()
        self._sleep = sleep
        self._runner = runner

    # ── transport ─────────────────────────────────────────────────────────
    def _run_once(self, command: list[str]) -> "subprocess.CompletedProcess[str]":
        if self._runner is not None:
            return self._runner(command)
        return subprocess.run(
            command, cwd=self._cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)

    def _command(self, *tail: str) -> list[str]:
        return [*self._prefix, *tail]

    # ── pacing ────────────────────────────────────────────────────────────
    @staticmethod
    def _rate_limited(result: "subprocess.CompletedProcess[str]") -> bool:
        text = result.stdout or ""
        return ReconnectBackoff.is_rate_limited(RuntimeError(text))

    def _pace(self, attempt: int, error: BaseException | None) -> None:
        self._sleep(self._backoff.delay(attempt, error))

    # ── verification ──────────────────────────────────────────────────────
    @staticmethod
    def _files_under(dest: Path) -> tuple[Path, ...]:
        if not dest.exists():
            return ()
        return tuple(sorted(path for path in dest.rglob("*") if path.is_file()))

    def _verify(self, slug: str, dest: Path, files: tuple[Path, ...],
                require_globs: Sequence[str]) -> None:
        if not files:
            raise EmptyDownloadError(
                f"kaggle download for {slug!r} exited 0 but wrote no files "
                f"under {dest} (the CLI's silent-empty success); nothing was "
                "fetched — the remote may be a stop-stub version or not yet "
                "finalized",
                stdout="", traceback_text=_stack())
        if require_globs and not any(
                _matches_any(path, require_globs) for path in files):
            observed = sorted(path.name for path in files)
            raise DownloadError(
                f"kaggle download for {slug!r} landed files {observed} but "
                f"none matched required {list(require_globs)} under {dest}",
                traceback_text=_stack())

    # ── engine ────────────────────────────────────────────────────────────
    def _download(self, *, slug: str, dest: Path, tail: Sequence[str] = (),
                  command: Sequence[str] | None = None,
                  require_globs: Sequence[str] = (),
                  allow_empty: bool = False) -> DownloadResult:
        dest = Path(dest)
        resumed = dest.is_dir() and any(dest.iterdir())
        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)
        # A caller may hand the full argv (built by the canonical builder) so
        # the download verbs never re-spell a token here.
        command = list(command) if command is not None else self._command(*tail)
        stdout = ""
        last_error: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                result = self._run_once(command)
            except Exception as error:  # transport blew up (runner raised)
                if attempt < self._max_attempts:
                    self._pace(attempt, error)
                    continue
                raise DownloadError(
                    f"kaggle download for {slug!r} raised after {attempt} "
                    f"attempts: {type(error).__name__}: {error}",
                    traceback_text=traceback.format_exc()) from error
            stdout = result.stdout or ""
            if result.returncode != 0:
                if self._rate_limited(result) and attempt < self._max_attempts:
                    self._pace(attempt, RuntimeError(stdout))
                    continue
                raise DownloadError(
                    f"kaggle download for {slug!r} failed (rc="
                    f"{result.returncode}) after {attempt} attempts:\n"
                    f"--- full kaggle output ---\n{stdout}",
                    stdout=stdout, traceback_text=_stack())
            files = self._files_under(dest)
            if files:
                self._verify(slug, dest, files, require_globs)
                return DownloadResult(
                    slug=slug, dest=dest, files=files, attempts=attempt,
                    command=tuple(command), stdout=stdout, resumed=resumed)
            if allow_empty:
                return DownloadResult(
                    slug=slug, dest=dest, files=files, attempts=attempt,
                    command=tuple(command), stdout=stdout, resumed=resumed)
            # rc == 0, zero files: the upstream silent success. Retry (output
            # may not be ready yet), then fail LOUD.
            last_error = EmptyDownloadError(
                f"kaggle download for {slug!r} exited 0 with no files under "
                f"{dest}",
                stdout=stdout, traceback_text=_stack())
            if attempt < self._max_attempts:
                self._pace(attempt, last_error)
        raise EmptyDownloadError(
            f"kaggle download for {slug!r} exited 0 but wrote no files under "
            f"{dest} after {self._max_attempts} attempts (the CLI's "
            "silent-empty success); nothing was fetched",
            stdout=stdout, traceback_text=_stack())


class KernelOutputFetcher(_RetryingDownloader):
    """Fetch a kernel's ``/kaggle/working`` output with verification."""

    def fetch(self, slug: str, dest: StrPath, *,
              require_globs: Sequence[str] = ("*.tar.gz", "*.zip"),
              allow_empty: bool = False) -> DownloadResult:
        self._validate_slug(slug)
        # ONE argv home: the canonical `kernels output` token shape
        # (KaggleKernels.kernels_output_argv), never re-spelled here.
        from cli.kaggle_kernels import KaggleKernels

        return self._download(
            slug=slug, dest=Path(dest),
            command=KaggleKernels.kernels_output_argv(
                self._prefix, slug, Path(dest)),
            require_globs=require_globs, allow_empty=allow_empty)

    @staticmethod
    def _validate_slug(slug: str) -> None:
        owner, slash, kernel = slug.rpartition("/")
        if not slash or not owner or not kernel:
            raise ValueError(f"kernel slug must be owner/slug, got {slug!r}")


class DatasetDownloader(_RetryingDownloader):
    """Download a Kaggle dataset with unzip + optional sha256 verification."""

    def download(self, slug: str, dest: StrPath, *,
                 require_globs: Sequence[str] = (),
                 sha256: str | None = None,
                 unzip: bool = False) -> DownloadResult:
        owner, slash, name = slug.rpartition("/")
        if not slash or not owner or not name:
            raise ValueError(f"dataset slug must be owner/slug, got {slug!r}")
        result = self._download(
            slug=slug, dest=Path(dest),
            tail=("datasets", "download", slug, "-p", str(dest)),
            require_globs=require_globs)
        # Verify the transport identity BEFORE unpacking: the receipt's sha256
        # is the archive contract, so a drifted archive never has its (possibly
        # zip-slip/archive-bomb) bytes extracted, and the archive stays
        # unambiguous among the members an earlier unpack may have left behind.
        archive = self._archive(result.files) if (sha256 is not None or unzip) else None
        if sha256 is not None:
            observed = self._sha256(archive)
            if observed != sha256:
                raise DownloadError(
                    f"dataset {slug!r} sha256 mismatch for {archive}: expected "
                    f"{sha256} observed {observed}",
                    traceback_text=_stack())
        result = DownloadResult(
            slug=result.slug, dest=result.dest, files=result.files,
            attempts=result.attempts, command=result.command,
            stdout=result.stdout, resumed=result.resumed, archive=archive)
        if unzip:
            self._unzip(archive)
            result = DownloadResult(
                slug=result.slug, dest=result.dest,
                files=self._files_under(result.dest), attempts=result.attempts,
                command=result.command, stdout=result.stdout,
                resumed=result.resumed, archive=archive)
        return result

    @staticmethod
    def _archive(files: tuple[Path, ...]) -> Path:
        archives = [
            path for path in files
            if path.suffix in {".zip", ".gz", ".zst", ".tar"} or path.name.endswith(".tar.gz")
        ]
        if not archives:
            raise DownloadError(
                f"dataset download landed no archive among "
                f"{[p.name for p in files]}",
                traceback_text=_stack())
        # Kaggle dataset downloads are zips; prefer one deterministically, and
        # among equals take the most recently landed so a resumed stage dir's
        # older archive is never the one verified (the fetch-back contract).
        return max(archives, key=lambda path: (
            path.suffix == ".zip", path.stat().st_mtime, str(path)))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _unzip(archive: Path) -> None:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(archive.parent)
