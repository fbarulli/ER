"""Failure artifact compression, verification, and terminal session cleanup."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core.manifest import file_size


class KernelLifecycle:
    """Preserve remote failure artifacts and release compute after harvesting."""

    _failure_finalizer = r"""
    except BaseException:
        import traceback
        import zipfile
        failure_traceback = traceback.format_exc()
        print(failure_traceback, flush=True)
        WORKING.mkdir(parents=True, exist_ok=True)
        (WORKING / LANE["files"]["failure_log"]).write_text(failure_traceback, encoding="utf-8")
        # Keep all generated output trees, including partial preparation products.
        # The checkout itself can contain injected credentials and is not output.
        failure_root = SCRATCH / LANE["files"]["checkout_dir"]
        failure_files = {}
        for directory in dict.fromkeys([
                *LANE["paths"].values(), *LANE["remote"]["extra_artifact_dirs"]]):
            source = failure_root / directory
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    failure_files[LANE["files"]["checkout_dir"] + "/" + str(path.relative_to(failure_root))] = path
        for path in sorted(WORKING.rglob("*")):
            if path.is_file() and path.name not in {
                    LANE["files"]["failure_archive"], LANE["files"]["failure_zip"], LANE["files"]["failure_manifest"]}:
                failure_files["working/" + str(path.relative_to(WORKING))] = path
        sys.path.insert(0, str(failure_root / LANE["files"]["source_dir"]))
        try:
            from core.archive_reader import tar_archive
            failure_archive = WORKING / LANE["files"]["failure_archive"]
            with tar_archive(failure_archive, "w") as archive:
                for name, path in failure_files.items():
                    archive.add(path, arcname=name)
        except Exception:
            # Setup itself may have failed before the zstd dependencies arrived.
            # Use the lane's existing ZIP_DEFLATED transport in that case.
            (WORKING / LANE["files"]["failure_archive"]).unlink(missing_ok=True)
            failure_archive = WORKING / LANE["files"]["failure_zip"]
            with zipfile.ZipFile(failure_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, path in failure_files.items():
                    archive.write(path, arcname=name)
        failure_manifest = {
            "status": "error", "archive": failure_archive.name,
            "archive_size": file_size(failure_archive),
            "files": {name: {"bytes": path.stat().st_size,
                             "size": file_size(path)}
                      for name, path in failure_files.items()},
        }
        (WORKING / LANE["files"]["failure_manifest"]).write_text(
            json.dumps(failure_manifest, indent=2), encoding="utf-8")
        raise
    """

    @classmethod
    def wrap_script(cls, script: str) -> str:
        """Install failure archival before checkout, setup, or worker execution."""
        import textwrap

        return ("try:\n" + textwrap.indent(script, "    ")
                + textwrap.dedent(cls._failure_finalizer))

    @staticmethod
    def failure_output(stage: Path) -> dict[str, Any] | None:
        from cli import kaggle_lane as lane

        files = lane._spec().files
        manifests = sorted(stage.rglob(files.failure_manifest))
        if not manifests:
            return None
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        name = manifest.get("archive")
        if name not in {files.failure_archive, files.failure_zip}:
            raise RuntimeError("invalid failure archive name")
        archive = manifests[0].parent / name
        observed = file_size(archive)
        return {"failed": True, "verified": True,
                "fetched_archive": str(archive), "archive_size": observed}

    @staticmethod
    def harvest_and_stop(*, kind: str, slug: str, which: str,
                         status: str, fetch_output, fetch_failure,
                         stop, configured_slug: str | None) -> dict[str, Any]:
        from cli import kaggle_lane as lane

        limits = lane._spec().limits
        plan: dict[str, Any] = {}
        target = {"slug": slug} if slug != configured_slug else {}
        try:
            # Output may take a short time to become downloadable after the
            # terminal status appears. Retry before replacing that version.
            for attempt in range(limits.transfer_attempts):
                try:
                    if status == "complete":
                        plan["fetch"] = fetch_output(kind=kind, execute=True, **target)
                    else:
                        diagnostic = fetch_failure(kind, **target)
                        plan["failures"] = {kind: diagnostic}
                        if diagnostic.get("fail_closed") and attempt < limits.transfer_attempts - 1:
                            time.sleep(limits.retry_seconds)
                            continue
                    break
                except Exception as error:
                    if attempt == limits.transfer_attempts - 1:
                        plan["fetch_error"] = str(error)
                    else:
                        time.sleep(limits.retry_seconds)
        finally:
            for attempt in range(limits.stop_attempts):
                try:
                    plan["stop"] = stop(slug=slug, which=which, execute=True)
                    break
                except Exception as error:
                    plan["stop"] = {"stopped": False, "error": str(error)}
                    if attempt < limits.stop_attempts - 1:
                        time.sleep(limits.retry_seconds)
        return plan


