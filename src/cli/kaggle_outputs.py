"""Verified output downloads and failed-run artifact recovery."""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any



class KaggleOutputs:
    """Verified output downloads and failed-run artifact recovery."""

    @staticmethod
    def fetch_kernel_output(*, kind: str = "bundle", execute: bool,
                            cohort: str | None = None,
                            slug: str | None = None) -> dict[str, Any]:
        """Download a kernel's output and hash-verify its manifest archive.

        bundle — bundle.receipt.json contract (all_tracks_inputs.tar.zst)
        train  — result_manifest.json contract (result_bundle.tar.zst)
        embed  — result_manifest.json contract (vectors.tar.zst)
        Verified artifacts install under staging_dir/<cohort>/<kind>/; the
        destination cohort tag comes from the explicit override (for the bundle
        lane's per-cohort fetch) or from the config dataset binding, matching
        the kernel receipt's own cohort tag when it declares one.
        Publish default (owner order 2026-10-07): after the verification a
        fresh dataset version publishes automatically (`plan["publish"]`,
        publish_bundle_dataset) — for bundles the `kaggle.bundle_dataset_slug`
        SSOT target; train/embed outputs record a skip note (no SSOT dataset).
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        slugs = {"bundle": spec.cpu_kernel_slug, "train": spec.gpu_kernel_slug,
                 "embed": spec.embedding_kernel_slug}
        if kind not in slugs:
            raise ValueError(f"unknown kernel output kind: {kind!r}")
        slug = slug or slugs[kind]
        if not slug:
            raise RuntimeError(f"config kaggle kernel slug for {kind!r} is unset")
        result_kind = spec.files.result_names.get(kind, kind)
        manifest_name = (spec.files.bundle_receipt if kind == "bundle"
                         else spec.files.result_manifest.format(kind=result_kind))
        archive_name = (spec.files.bundle_archive if kind == "bundle"
                        else spec.files.result_archive.format(kind=result_kind))
        stage = lane.staging_dir() / lane._spec().files.fetch_stage.format(kind=kind)
        plan: dict[str, Any] = {
            "mode": "executed" if execute else "dry-run",
            "kernel": slug,
            "kind": kind,
            "stage": str(stage),
        }
        if not execute:
            return plan
        executable = lane._require_kaggle_executable(spec.kaggle_executable)
        if stage.exists():
            retained = stage.with_name(f"{stage.name}.{time.time_ns()}")
            stage.rename(retained)
        stage.mkdir(parents=True)
        command = [executable, "kernels", "output", slug, "-p", str(stage)]
        _, _ = lane._run_kaggle(command)
        failure = lane.KernelLifecycle.failure_output(stage)
        if failure:
            plan.update(failure)
            return plan
        manifests = sorted(stage.rglob(manifest_name))
        if not manifests:
            zips = sorted(stage.glob("*.zip"))
            if zips:
                with zipfile.ZipFile(zips[0]) as bundle:
                    bundle.extractall(stage / lane._spec().files.unpack_dir)
                failure = lane.KernelLifecycle.failure_output(stage / lane._spec().files.unpack_dir)
                if failure:
                    plan.update(failure)
                    return plan
                manifests = sorted((stage / lane._spec().files.unpack_dir).rglob(manifest_name))
        if not manifests:
            raise RuntimeError(
                f"kernel output under {stage} contained no {manifest_name}")
        manifest_dir = manifests[0].parent
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        archive = manifest_dir / archive_name
        if not archive.is_file():
            raise FileNotFoundError(
                f"kernel output manifest names {archive_name} but it is missing "
                f"under {manifest_dir}")
        observed = lane.sha256_file(archive)
        sidecar = manifest_dir / (archive_name + spec.files.hash_suffix)
        expected = (sidecar.read_text().strip() if sidecar.is_file()
                    else manifest.get("archive_sha256"))
        if not expected or observed != expected:
            raise RuntimeError(
                f"fetched {kind} sha256 mismatch: expected {expected} observed "
                f"{observed}")
        from core.common import F

        receipt_cohort = (manifest if kind == "bundle"
                          else json.loads(manifests[0].read_text())).get("cohort")
        resolved_cohort = cohort or receipt_cohort or lane.cohort_label(Path(F["dataset"]))
        destination = lane.staging_dir() / resolved_cohort / spec.files.install_dir.format(kind=kind)
        destination.mkdir(parents=True, exist_ok=True)
        installed = {}
        shutil.copy2(archive, destination / archive.name)
        installed[archive_name] = str(destination / archive.name)
        sidecar_names = [manifest_name, (archive_name + spec.files.hash_suffix)]
        if kind == "bundle":
            sidecar_names += list(spec.files.bundle_sidecars)
        for name in sidecar_names:
            sidecar_path = manifest_dir / name
            if sidecar_path.is_file():
                shutil.copy2(sidecar_path, destination / name)
                installed[name] = str(destination / name)
        plan.update({
            "fetched_archive": str(archive),
            "archive_sha256": observed,
            "verified": True,
            "cohort": resolved_cohort,
            "installed": installed,
        })
        # Publish default (owner order 2026-10-07): every successful verified
        # fetch ends with the publish step — no operator hand-invoke. The plan
        # entry records what was published (or why it was skipped/failed).
        plan["publish"] = lane._publish_after_verified_fetch(kind)
        return plan

    @staticmethod
    def fetch_failed_kernel_log(kind: str, *, slug: str | None = None) -> dict[str, Any]:
        """Download and verify partial artifacts, retaining legacy session logs.

        Older kernels may lack the failure finalizer; keep their downloaded
        output and report the missing archive without blocking session release.
        """
        from cli import kaggle_lane as lane

        plan: dict[str, Any] = {"kind": kind, "mode": "executed", "error_log": None}
        try:
            plan["fetch"] = lane.fetch_kernel_output(
                kind=kind, execute=True, **({"slug": slug} if slug else {}))
        except (RuntimeError, FileNotFoundError, OSError) as error:
            plan["fail_closed"] = str(error)[-lane._spec().limits.error_tail_chars:]
        stage = lane.staging_dir() / lane._spec().files.fetch_stage.format(kind=kind)
        if stage.is_dir():
            for candidate in sorted(stage.rglob(lane._spec().files.log_glob)):
                destination = lane.lane_logs_dir() / candidate.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, destination)
                plan["error_log"] = str(destination)
        return plan

    @staticmethod
    def fetch_bundle_output(*, execute: bool) -> dict[str, Any]:
        """Back-compat entry point — delegates to the generalized fetcher."""
        from cli import kaggle_lane as lane

        return lane.fetch_kernel_output(kind="bundle", execute=execute)

