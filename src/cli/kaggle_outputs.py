"""Verified output downloads and failed-run artifact recovery."""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from core.archive_reader import archive_sidecar



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
        finalize — result_manifest.json contract (finalized_bundle.tar.zst, the
        sealed result bundle the remote CPU finalize job wrote)
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
        # One registry (config SSOT) resolves the slug, the bundle role, and the
        # manifest/archive names — no per-surface kind->slug table here.
        try:
            identity = lane.kernel_identity(kind, spec)
        except RuntimeError as error:
            raise ValueError(f"unknown kernel output kind: {kind!r}") from error
        slug = slug or identity.slug(spec)
        if not slug:
            raise RuntimeError(
                f"config kaggle.{identity.slug_attr} for {kind!r} is unset")
        manifest_name = identity.manifest_name(spec.files)
        archive_name = identity.archive_name(spec.files)
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
        # ONE download transport: the 429-aware, fail-loud, silent-empty-loud
        # KernelOutputFetcher — never a bare `kaggle kernels output` through
        # _run_kaggle. It creates the stage dir, paces 429s and raises on an
        # rc=0/zero-file run (the CLI's silent-empty success); the manifest/sha
        # verification below is unchanged.
        from cli.kaggle_download import KernelOutputFetcher

        KernelOutputFetcher(
            argv_prefix=(executable,), cwd=lane.TRAIN_ROOT).fetch(
            slug, stage, require_globs=())
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
        sidecar = archive_sidecar(manifest_dir / archive_name, spec.files.hash_suffix)
        expected = (sidecar.read_text().strip() if sidecar.is_file()
                    else manifest.get("archive_sha256"))
        if not expected:
            raise RuntimeError(
                f"fetched {kind} output records no sha256 for {archive_name} "
                f"(neither {sidecar.name} nor the "
                "manifest carries one)")
        # ONE read of the archive. A bundle role is named by its boundary load,
        # which verifies the whole-archive sha256 against the recorded receipt
        # AND the archive's own member inventory in that same pass — so a role
        # archive is never hashed twice. Kinds with no bundle role (train,
        # embed) get the plain whole-archive digest. Either way this crossing
        # performs exactly one integrity hash of the fetched archive.
        bundle = KaggleOutputs.identify_bundle(kind, archive, expected)
        observed = bundle.get("sha256") or lane.sha256_file(archive)
        if observed != expected:
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
        sidecar_names = [manifest_name, sidecar.name]
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
        # ``bundle`` above IS this crossing's single integrity check: the role
        # load verified the whole-archive sha256 against the receipt and, in the
        # same pass, the archive's own member inventory. A non-role kind was
        # named and hashed once; either way the artifact is reported for what it
        # is and nothing downstream re-reads the archive to re-verify it.
        plan["bundle"] = bundle
        # Publish default (owner order 2026-10-07): every successful verified
        # fetch ends with the publish step — no operator hand-invoke. The plan
        # entry records what was published (or why it was skipped/failed).
        plan["publish"] = lane._publish_after_verified_fetch(kind)
        return plan

    @staticmethod
    def identify_bundle(kind: str, archive: Path, expected_digest: str) -> dict[str, Any]:
        """Name the fetched archive's ``core.bundle`` role (and load ONE handle).

        The role comes from the lane identity registry (``bundle`` fetches the
        prepared-inputs Bundle, ``finalize`` the sealed result Bundle); a role
        archive is loaded exactly once through :meth:`core.bundle.Bundle.load`,
        whose single pass verifies the whole-archive sha256 against
        ``expected_digest`` AND the archive's own member inventory — so the
        caller gets the observed digest and never re-reads the archive. Any
        other kind is not a Bundle role. An archive whose role manifest is
        absent is reported as unidentified — never silently treated as a bundle
        (the enforcing stage fails loud on its own load instead).
        """
        from cli import kaggle_lane as lane

        from core.bundle import Bundle, BundleRole, manifest_name

        role_value = lane.kernel_identity(kind).bundle_role
        if role_value is None:
            return {"identified": False, "role": None,
                    "note": f"fetched {kind!r} output is not a bundle role archive"}
        role = BundleRole(role_value)
        try:
            handle = Bundle.load(archive, role, expected_digest=expected_digest)
        except (ValueError, KeyError, EOFError, OSError) as error:
            return {"identified": False, "role": role.value,
                    "expected_manifest": manifest_name(role),
                    "note": f"{type(error).__name__}: {str(error)[:300]}"}
        return {"identified": True, "role": role.value, "sha256": handle.digest,
                "members": len(handle.members()),
                "run_tag": handle.run_tag() or None}

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

