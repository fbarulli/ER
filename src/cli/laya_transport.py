"""Laya lane transport: addressed kernel slugs, push, stop, and fetch.

``LayaTransportFactory`` is the Facade over the kaggle owners
(``KaggleKernels`` / ``KaggleDatasets`` / ``KernelOutputFetcher``): the laya
lane never re-implements a push argv, a session cancel, or a download; it only
resolves the laya spec's slug and feeds the staged payload.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cli.kaggle_kernels import KaggleKernels

if TYPE_CHECKING:
    from cli.kaggle_watcher import KernelWatcher, KernelWatcherSpec
from cli.laya_recipe import (
    DECISION_BINDINGS,
    FINETUNE_DECISION,
    FINETUNE_EVAL_DECISION,
    FINETUNE_SMOKE_DECISION,
    HOLDOUT_EVAL_DECISION,
)
from cli.laya_runtime import LayaRuntimeFactory
from cli.laya_staging import LayaPayloadPreflight
from cli.laya_traceability import fetched_traceability, read_decision_rows
from core.laya_config import LayaSpec

#: decision kind -> the LayaSpec attribute holding its pushed kernel slug.
_KIND_KERNEL_SLUG_ATTR = {
    FINETUNE_DECISION: "finetune_kernel_slug",
    FINETUNE_EVAL_DECISION: "finetune_eval_kernel_slug",
    HOLDOUT_EVAL_DECISION: "holdout_eval_kernel_slug",
}


class LayaTransportFactory:
    """Resolves kernel slugs and drives the kaggle transport for one spec."""

    def __init__(self, spec: LayaSpec, runtime: LayaRuntimeFactory):
        self._spec = spec
        self._runtime = runtime

    def kernel_slug(self, decision_kind: str) -> str:
        """The pushed Kaggle kernel slug a decision kind runs on (stop target)."""
        spec = self._spec
        if decision_kind == FINETUNE_SMOKE_DECISION:
            slug = spec.finetune_smoke.kernel_slug
            attr = "finetune_smoke.kernel_slug"
        elif decision_kind in _KIND_KERNEL_SLUG_ATTR:
            attr = _KIND_KERNEL_SLUG_ATTR[decision_kind]
            slug = getattr(spec, attr, None)
        elif decision_kind in DECISION_BINDINGS:
            attr = "export_dataset_slug"
            slug = getattr(spec, attr, None)
        else:
            raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                             f"expected {list(DECISION_BINDINGS)}")
        if not slug:
            raise RuntimeError(
                f"config laya.{attr} is unset; name the target kernel (owner/slug) "
                f"before addressing {decision_kind!r}")
        return slug

    @staticmethod
    def container_session_id(container: str) -> int | None:
        """The kernel session id inside ``KAGGLE_CONTAINER_NAME``."""
        parts = str(container or "").rsplit("-", 2)
        if len(parts) == 3 and parts[1].isdigit():
            return int(parts[1])
        return None

    def recorded_session_id(self, slug: str) -> int | None:
        """The recorded session id for a pushed kernel (``None`` if it holds
        only the kernel-name fallback handle). ONE reader: the shared owner."""
        from cli.kaggle_monitor import KaggleMonitor

        kernel = slug.rpartition("/")[2]
        if not kernel:
            raise ValueError(f"slug must be owner/slug, got {slug!r}")
        handle = KaggleMonitor.recorded_kernel_handle(slug)
        return int(handle) if handle and handle.isdigit() else None

    @staticmethod
    def stop_kaggle_kernel(slug: str, *, execute: bool, wait: bool = True,
                           which: str = "laya") -> dict[str, Any]:
        """First-class teardown of a pushed laya kernel's running session.

        ``which`` is the harvest contract's slot (the watcher passes its own
        identity); the slug is always explicit here, so it only names the stop
        stage. The persisted handle — session id, else the kernel name — is read
        by ``KaggleKernels.stop_kernel`` (SDK cancel when numeric, version-
        replace fallback otherwise).
        """
        return KaggleKernels.stop_kernel(slug, which=which, execute=execute,
                                         wait=wait)

    def push_kaggle_kernel(self, stage_dir: Path, *, execute: bool,
                           activate: bool = True) -> dict[str, Any]:
        """`kaggle kernels push` a staged payload, `--execute`-gated."""
        argv = KaggleKernels.kernels_push_argv(
            [sys.executable, "-m", "kaggle"], stage_dir)
        plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                                "argv": argv, "stage": str(stage_dir)}
        if not execute:
            self._runtime.log_lane(f"dry-run: would run {' '.join(argv)}")
            return plan
        metadata_file = Path(stage_dir) / "kernel-metadata.json"
        if not metadata_file.is_file():
            raise RuntimeError(
                "--activate gate: no staged kernel at "
                f"{stage_dir} (kernel-metadata.json is missing); stage first "
                "(--what stage-kernel)")
        LayaPayloadPreflight.verify(Path(stage_dir))
        slug = json.loads(metadata_file.read_text(encoding="utf-8"))["id"]

        def run_push() -> None:
            result = subprocess.run(argv, cwd=self._runtime.train_root,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            output = result.stdout or ""
            plan["returncode"] = result.returncode
            # `kaggle kernels push` returns rc=0 even on a soft error — e.g. the
            # GPU quota message "Maximum batch GPU session count of 2 reached" —
            # so the exit code alone is NOT fail-loud.
            lowered = output.lower()
            if (result.returncode != 0 or "error" in lowered
                    or "successfully pushed" not in lowered):
                tail = output.strip()[-4000:] or "(kaggle produced no output)"
                raise RuntimeError(
                    f"kaggle kernels push failed (rc={result.returncode}): "
                    f"{' '.join(argv)}\n--- kaggle output ---\n{tail}")

        # ONE clear->push->capture sequence, shared with the kaggle lane. The
        # capture persists a usable handle: the session id when the proxy is up,
        # else the kernel name (the persisted-name fallback), so stop/status/
        # output are always targetable.
        captured = KaggleKernels.push_with_session_capture(slug, run_push)
        plan["pushed"] = True
        plan["session_id"] = captured.get("session_id")
        plan["handle"] = captured.get("handle")
        self._runtime.log_lane(
            f"pushed kernel payload: {' '.join(argv)} rc=0 "
            f"session_id={plan.get('session_id')} handle={plan.get('handle')}")
        return plan

    def collect_kaggle_result(self, kind: str, slug: str | None = None, *,
                              execute: bool = False) -> dict[str, Any]:
        """`kaggle kernels output` for a staged/decided kernel, then traceability.

        Downloads to ``results/laya_lane/fetch/<kind>/`` via the canonical
        ``kernels_output_argv`` + the 429-aware ``KernelOutputFetcher``. The
        per-kind receipt name (``laya_<kind>.receipt.json``; smoke =
        ``laya_finetune-smoke.receipt.json``) is accepted whether it rides the
        kernel's ``laya_finetune.tar.gz`` or lands as a top-level download, so a
        smoke and the prod kind can never be confused. The archive and every
        top-level download (expanded checkpoint/base_model/wandb trees included)
        are left in the stage for the harvest.
        """
        slug = slug or self.kernel_slug(kind)
        plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                                "kind": kind, "slug": slug}
        if not execute:
            self._runtime.log_lane(f"dry-run: would fetch kernel output for {slug}")
            return plan
        stage = self._runtime.staging_dir() / "fetch" / kind
        if stage.exists():
            import shutil

            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        # The fail-loud, 429-aware boundary: an rc=0 CLI call that wrote zero
        # files (the upstream `kernels output` silent-empty success) must NEVER
        # be reported as a successful fetch.
        from cli.kaggle_download import DownloadError, KernelOutputFetcher

        try:
            download = KernelOutputFetcher(cwd=self._runtime.train_root).fetch(
                slug, stage, require_globs=("*.tar.gz", "*.zip"))
        except DownloadError as error:
            raise RuntimeError(
                f"kaggle kernels output for {slug} failed: {error}\n"
                f"{error.traceback_text}") from error
        plan["download"] = {
            "files": [str(path) for path in download.files],
            "attempts": download.attempts,
            "resumed": download.resumed,
        }
        archives = sorted(stage.glob("*.tar.gz")) or sorted(stage.glob("*.zip"))
        if not archives:
            raise RuntimeError(f"kaggle kernels output staged no archive "
                               f"under {stage} (slug {slug})")
        receipt_name = f"laya_{kind}.receipt.json"
        decisions_name = f"{kind}.decisions.jsonl"
        reports: dict[str, Any] = {}
        # ONE streaming pass: `r|*` never seeks, so the receipt, every JSON report
        # member and the per-row `<kind>.decisions.jsonl` member are consumed in a
        # single forward walk. Bodies are buffered so a missing receipt still
        # fails loud BEFORE any report lands (the pre-streaming order).
        members: list[str] = []
        decision_rows: list[dict] = []
        payload: Any = None
        pending: list[tuple[str, bytes]] = []
        with tarfile.open(archives[0], "r|*") as tar:
            for member in tar:
                members.append(member.name)
                if Path(member.name).name == receipt_name:
                    payload = json.loads(tar.extractfile(member).read().decode())
                    continue
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if name.endswith(".json"):
                    body = tar.extractfile(member).read()
                    pending.append((name, body))
                    try:
                        reports[name] = json.loads(body.decode())
                    except (ValueError, UnicodeDecodeError):
                        continue
                elif name == decisions_name:
                    body = tar.extractfile(member).read()
                    pending.append((name, body))
                    decision_rows = read_decision_rows(body)
        # The kernel also writes the receipt beside the tar, so `kernels output`
        # leaves it top-level; accept it there too (the tar member wins).
        if payload is None:
            receipt_file = stage / receipt_name
            if receipt_file.is_file():
                payload = json.loads(receipt_file.read_text(encoding="utf-8"))
        if payload is None:
            raise RuntimeError(
                f"fetched archive {archives[0].name} and {stage} carry no "
                f"{receipt_name}; the kernel receipt contract failed")
        for name, body in pending:
            (stage / name).write_bytes(body)
        plan.update({"archive": str(archives[0]), "members": members,
                     "receipt": payload, "reports": reports})
        if decision_rows:
            plan["decision_rows"] = len(decision_rows)
        # Validate the fetched reports against the shared traceability contract
        # and EMIT the per-row grain through the declared layout. Store the JSON
        # form: this plan is the printed `--fetch` document, so a pydantic model
        # left in it would make json.dumps raise TypeError.
        documents, artifacts = fetched_traceability(
            payload, reports, decision_kind=kind,
            decision_rows=decision_rows,
            staging_root=self._runtime.staging_dir())
        plan["traceability"] = documents
        if artifacts:
            plan["traceability_artifacts"] = artifacts
        self._runtime.log_lane(
            f"fetched kernel output for {slug}: "
            f"archive={archives[0].name} members={len(members)} "
            f"reports={sorted(reports)} rows={len(decision_rows)}")
        return plan

    def fetch_failed_result(self, kind: str, *, slug: str | None = None
                            ) -> dict[str, Any]:
        """Download and report a FAILED kernel's retained partial artifacts.

        Mirrors ``KaggleOutputs.fetch_failed_kernel_log`` for the laya lane: a
        missing receipt is recorded fail-closed so it never blocks the session
        release the harvest performs next.
        """
        plan: dict[str, Any] = {"kind": kind, "mode": "executed",
                                "error_log": None}
        try:
            plan["fetch"] = self.collect_kaggle_result(kind, slug, execute=True)
        except (RuntimeError, FileNotFoundError, OSError) as error:
            plan["fail_closed"] = str(error)[-800:]
        return plan

    def watcher(self, kind: str, *, slug: str | None = None) -> "KernelWatcher":
        """The consolidated detached watcher bound to this lane's owners."""
        from cli.kaggle_watcher import KernelWatcher

        return KernelWatcher(self.watcher_spec(kind, slug=slug))

    def watcher_spec(self, kind: str, *, slug: str | None = None
                     ) -> "KernelWatcherSpec":
        """Resolve this lane's watcher parameters into the shared spec.

        The receipt lives under ``results/laya_lane/fetch/<kind>/`` (the same
        roof the harvest downloads to) and the progress roof is
        ``logs/laya/lane.log``; the detached entry re-derived the spec from the
        ``--watch`` op.
        """
        from cli.kaggle_monitor import KaggleMonitor
        from cli.kaggle_watcher import KernelWatcherSpec
        from core.common import training_cfg
        from core.manifest import atomic_write_json

        # An explicit slug needs no config; otherwise resolve the configured
        # one (fails loud when unset). Leaving ``configured_slug`` None for an
        # explicit slug makes the harvest pass the slug through instead of
        # re-resolving a config that may be unset.
        if slug:
            configured: str | None = None
            resolved = slug
        else:
            configured = self.kernel_slug(kind)
            resolved = configured
        kaggle = training_cfg().kaggle
        stage = self._runtime.staging_dir() / "fetch" / kind
        return KernelWatcherSpec(
            which=kind, kind=kind, configured_slug=configured,
            fetch_output=self.collect_kaggle_result,
            fetch_failure=self.fetch_failed_result,
            stop=self.stop_kaggle_kernel,
            stream_logs=KaggleMonitor.stream_kernel_logs,
            kernel_status=KaggleKernels.kernel_status,
            capture_session=KaggleMonitor.capture_kernel_session_id,
            log_lane=self._runtime.log_lane,
            write_json=atomic_write_json,
            # Distinct from the kernel's own per-kind receipt
            # (`laya_<kind>.receipt.json`) that the fetch downloads here.
            receipt_path=stage / "autowatch.receipt.json",
            log_path=self._runtime.lane_logs_dir() / "lane.log",
            poll_seconds=kaggle.logs_poll_seconds,
            stream_join_seconds=kaggle.limits.stream_join_seconds,
            append=True,
            entry_argv=(sys.executable, "-m", "cli.laya_lane", "--watch",
                        "--decision", kind, "--slug", resolved, "--execute"),
            cwd=self._runtime.train_root,
            source_dir=self._runtime.train_root / "src",
        )
