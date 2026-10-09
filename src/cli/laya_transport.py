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
from typing import Any

from cli.kaggle_kernels import KaggleKernels
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
        """The launch-recorded session id for a pushed kernel (``None`` if none)."""
        from cli import kaggle_lane as lane

        kernel = slug.rpartition("/")[2]
        if not kernel:
            raise ValueError(f"slug must be owner/slug, got {slug!r}")
        path = (lane.lane_logs_dir()
                / lane._spec().files.session_id_file.format(kernel=kernel))
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def stop_kaggle_kernel(slug: str, *, execute: bool,
                           wait: bool = True) -> dict[str, Any]:
        """First-class teardown of a pushed laya kernel's running session."""
        return KaggleKernels.stop_kernel(slug, which="laya", execute=execute,
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
        from cli import kaggle_lane as lane

        slug = json.loads(metadata_file.read_text(encoding="utf-8"))["id"]
        # Record the session id at launch (the capture point the verified stop
        # reads): a later stop then cancels the EXACT session via the SDK instead
        # of a blind version replace. Drop any stale id first.
        lane.clear_kernel_session_id(slug)
        result = subprocess.run(argv, cwd=self._runtime.train_root,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        output = result.stdout or ""
        rc = result.returncode
        plan["returncode"] = rc
        # `kaggle kernels push` returns rc=0 even on a soft error — e.g. the
        # GPU-session quota message "Kernel push error: Maximum batch GPU session
        # count of 2 reached" — so the exit code alone is NOT fail-loud.
        lowered = (output or "").lower()
        if rc != 0 or "error" in lowered or "successfully pushed" not in lowered:
            tail = output.strip()[-4000:] or "(kaggle produced no output)"
            raise RuntimeError(
                f"kaggle kernels push failed (rc={rc}): {' '.join(argv)}\n"
                f"--- kaggle output ---\n{tail}")
        plan["pushed"] = True
        try:
            captured = lane.capture_kernel_session_id(slug)
            plan["session_id"] = captured.get("session_id")
        except Exception as error:  # noqa: BLE001 - best-effort launch aid
            plan["session_id"] = None
            self._runtime.log_lane(f"[{slug}] session-id capture skipped: {error}")
        self._runtime.log_lane(f"pushed kernel payload: {' '.join(argv)} rc=0 "
                               f"session_id={plan.get('session_id')}")
        return plan

    def collect_kaggle_result(self, decision_kind: str, slug: str, *,
                              execute: bool = False) -> dict[str, Any]:
        """`kaggle kernels output` for a staged/decided kernel."""
        plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                                "decision_kind": decision_kind, "slug": slug}
        if not execute:
            self._runtime.log_lane(f"dry-run: would fetch kernel output for {slug}")
            return plan
        stage = self._runtime.staging_dir() / "fetch" / decision_kind
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
        receipt_name = f"laya_{decision_kind}.receipt.json"
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
                if member.name == receipt_name:
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
                elif name == f"{decision_kind}.decisions.jsonl":
                    body = tar.extractfile(member).read()
                    pending.append((name, body))
                    decision_rows = read_decision_rows(body)
        if payload is None:
            raise RuntimeError(
                f"fetched archive {archives[0].name} carries no "
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
            payload, reports, decision_kind=decision_kind,
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
