"""Kernel staging, launch, status, and explicit session release."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any



class KaggleKernels:
    """Kernel staging, launch, status, and explicit session release."""

    @staticmethod
    def _bake_wandb_key_missing() -> str:
        raise RuntimeError(
            'wandb is always on (owner order 2026-10-07): kernel staging '
            'requires WANDB_API_KEY in the environment or .env; refusing to '
            'bake a silent local-only train run')

    @staticmethod
    def _kernel_script_gate(script: str) -> None:
        """Staging-time AST gate: never push an unparseable kernel or one that
        references an undeclared template constant (the v4 NameError class)."""
        from cli import kaggle_lane as lane

        import ast
        parsed = ast.parse(script)
        defined = {node.id for stmt in ast.walk(parsed)
                   if isinstance(stmt, ast.Assign)
                   for node in stmt.targets if isinstance(node, ast.Name)}
        undeclared = {expr.id for expr in ast.walk(parsed)
                      if isinstance(expr, ast.Name) and isinstance(expr.ctx, ast.Load)
                      and expr.id.isupper() and expr.id not in defined}
        if undeclared:
            raise ValueError(f"staged kernel uses undeclared constants: "
                             f"{sorted(undeclared)}; regenerate the template")

    @staticmethod
    def _attachment_gate(script: str, metadata: dict[str, Any]) -> None:
        """Staging-time wiring gate: a script that reads the /kaggle/input
        mount must have its datasets attached in the metadata.

        The laya session died exactly so (FileNotFoundError at session time):
        its staged metadata carried dataset_sources: [] while the script read
        /kaggle/input at runtime — staging succeeded, the session did not.
        The gate refuses to stage a kernel whose rendered script reads the
        input mount (a Load of the INPUTS template constant or a string
        literal naming /kaggle/input) while dataset_sources AND
        kernel_sources are both empty; the spec-driven fix is naming the
        dataset slug in config and re-staging, never a silent attach-nothing
        push. String constants excluded: the serialized LANE JSON rides in
        every script and itself embeds remote.input_dir — it is configured
        data, not a reader.
        """
        import ast
        reads_input_mount = False
        for expr in ast.walk(ast.parse(script)):
            if isinstance(expr, ast.Name) and isinstance(expr.ctx, ast.Load) \
                    and expr.id == "INPUTS":
                reads_input_mount = True
                break
            if (isinstance(expr, ast.Constant) and isinstance(expr.value, str)
                    and "/kaggle/input" in expr.value
                    and not expr.value.lstrip().startswith("{")):
                reads_input_mount = True
                break
        attached = (metadata.get("dataset_sources")
                    or metadata.get("kernel_sources"))
        if reads_input_mount and not attached:
            raise RuntimeError(
                "staged kernel reads /kaggle/input but attaches no dataset "
                f"(dataset_sources={metadata.get('dataset_sources')}, "
                f"kernel_sources={metadata.get('kernel_sources')}); set the "
                "dataset/kernel slug in config (spec-driven SSOT) and "
                "re-stage — a session with nothing attached dies on the "
                "first input read")

    @staticmethod
    def stage_bundle_kernel(*, revision: str | None = None,
                            cohort: str | None = None) -> dict[str, Any]:
        """Stage the CPU bundle-generation kernel: metadata + script + receipt.

        Dry-safe: writes into the staging area only; `push_bundle_kernel` makes
        the network call. The revision is pinned at staging time so the kernel
        clones exactly the source the package provenance will record. The
        cohort chooses which root-level export drives prepare_all in the
        kernel (colab replace-on-checkout pattern, no upload round-trip).
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        slug = spec.cpu_kernel_slug
        if not slug:
            raise RuntimeError(
                "config kaggle.cpu_kernel_slug is unset; name the CPU kernel "
                "(owner/slug) before staging")
        pinned = revision or lane._git_revision()
        cohort = cohort or spec.default_cohort
        cohort_dataset = lane.cohort_export_csv(cohort)
        stage = lane.staging_dir() / lane._spec().files.kernel_stage.format(kind="bundle")
        stage.mkdir(parents=True, exist_ok=True)
        metadata = {
            "id": slug,
            "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
            "code_file": spec.files.code_files["bundle"],
            "language": "python",
            "kernel_type": "script",
            "enable_gpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "is_private": True,
        }
        script = (lane.BUNDLE_KERNEL_SCRIPT
                  .replace("@REPOSITORY@", spec.repository)
                  .replace("@BRANCH@", spec.branch)
                  .replace("@REVISION@", pinned)
                  .replace("@CHECKOUT_PATHS@",
                           json.dumps(lane.checkout_members((*spec.checkout_paths, cohort_dataset))))
                  .replace("@RUNTIME_PREFLIGHT@", lane.checkout_preflight_script(
                      lane.checkout_inventory((*spec.checkout_paths, cohort_dataset))))
                  .replace("@REQUIREMENTS@", spec.bundle_requirements)
                  .replace("@COHORT@", cohort)
                  .replace("@COHORT_DATASET@", cohort_dataset))
        lane.atomic_write_json(metadata, stage / lane._spec().files.kernel_metadata)
        script = lane.KernelTemplates.render_runtime(script, spec)
        script = lane.KernelLifecycle.wrap_script(script)
        lane._kernel_script_gate(script)
        lane._attachment_gate(script, metadata)
        (stage / spec.files.code_files["bundle"]).write_text(script, encoding="utf-8")
        receipt = {
            "kernel": slug,
            "gpu": False,
            "branch": spec.branch,
            "revision": pinned,
            "cohort": cohort,
            "cohort_dataset": cohort_dataset,
            "staged": str(stage),
            "code_file": spec.files.code_files["bundle"],
        }
        lane.atomic_write_json(receipt, stage / lane._spec().files.kernel_receipt.format(kind="bundle"))
        return receipt

    @staticmethod
    def push_bundle_kernel(stage_dir: Path) -> dict[str, Any]:
        """Push the staged CPU kernel via the configured kaggle executable.

        Every live push spawns its own detached autowatch (default since the
        owner order: no session may outlive a terminal run); the watcher
        downloads and releases, no operator arg involved.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        slug = spec.cpu_kernel_slug
        if not slug:
            raise RuntimeError(
                "config kaggle.cpu_kernel_slug is unset; name the CPU kernel "
                "(owner/slug) before pushing")
        from core.runtime_inputs import staged_kernel_preflight
        staged_kernel_preflight(Path(stage_dir))
        executable = lane._require_kaggle_executable(spec.kaggle_executable)
        command = [executable, "kernels", "push", "-p", str(stage_dir)]
        _, _ = lane._run_kaggle(command)
        plan = {"mode": "executed", "kernel": slug, "pushed": True,
                "staged": str(stage_dir)}
        plan.update(lane._spawn_autowatch("cpu"))
        return plan

    @staticmethod
    def stage_gpu_kernel(*, kind: str, slug: str | None = None,
                         revision: str | None = None,
                         run_tag: str | None = None,
                         checkpoint: str | None = None,
                         checkout_paths: list[str] | None = None,
                         bundle_dataset_version: str | None = None) -> dict[str, Any]:
        """Stage a GPU kernel (train | embed) with the CPU bundle attached.

        Dry-safe: metadata + generated script + receipt under the staging area;
        `--execute` makes push_kernel() perform the network call. The train
        kernel attaches the CPU bundle kernel via kernel_sources (Kaggle mounts
        its output under /kaggle/input); the embed kernel additionally needs an
        embedding request dataset slug (kaggle.embedding_dataset_slug).
        `bundle_dataset_version` optionally pins the attached bundle dataset to
        `owner/slug/version` (the form kaggle's kernel metadata accepts);
        unpinned, the newest published version mounts automatically — the chain
        passes the version its publish step recorded.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        if kind not in lane.GPU_KERNEL_KINDS:
            raise RuntimeError(f"unknown GPU kernel kind: {kind}")
        _, body = lane.GPU_KERNEL_KINDS[kind]
        code_file = spec.files.code_files[kind]
        resolved_slug = slug or (spec.embedding_kernel_slug if kind == "embed"
                                 else spec.gpu_kernel_slug)
        if not resolved_slug:
            raise RuntimeError(
                f"config kaggle.{kind}_kernel_slug is unset; name the {kind} "
                "kernel (owner/slug) before staging")
        bundle_slug = spec.cpu_kernel_slug
        if not bundle_slug:
            raise RuntimeError("config kaggle.cpu_kernel_slug is unset; the GPU "
                               "kernel attaches the CPU bundle kernel output")
        pinned = revision or lane._git_revision()
        tag = run_tag or (spec.run_tag_prefix + time.strftime(spec.limits.run_tag_format, time.gmtime()))
        resolved_checkpoint = checkpoint or spec.checkpoint
        stage = lane.staging_dir() / lane._spec().files.kernel_stage.format(kind=kind)
        stage.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {
            "id": resolved_slug,
            "title": resolved_slug.rsplit("/", 1)[-1].replace("-", " ").title(),
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "is_private": True,
        }
        if kind == "embed":
            request_dataset = spec.embedding_dataset_slug
            if not request_dataset:
                raise RuntimeError(
                    "config kaggle.embedding_dataset_slug is unset; package + "
                    "upload the embedding request dataset first (--what package "
                    "--dataset-csv ... ; then set the slug)")
            metadata["dataset_sources"] = [request_dataset]
        else:
            metadata["kernel_sources"] = [bundle_slug]
            bundle_dataset = spec.bundle_dataset_slug
            if bundle_dataset:
                # Unpinned mount = newest published version (the publish default
                # keeps it fresh); a known version pins explicitly.
                bundle_dataset_entry = bundle_dataset
                if bundle_dataset_version:
                    bundle_dataset_entry = f"{bundle_dataset}/{bundle_dataset_version}"
                metadata["dataset_sources"] = [bundle_dataset_entry]
        template = lane.TRAIN_KERNEL_SHARED + body
        script = (template
                  .replace("@REPOSITORY@", spec.repository)
                  .replace("@BRANCH@", spec.branch)
                  .replace("@REVISION@", pinned)
                  .replace("@CHECKOUT_PATHS@",
                           json.dumps(lane.checkout_members(checkout_paths or spec.checkout_paths, lane="training")))
                  .replace("@BUNDLE_DATASET_NAME@",
                           (spec.bundle_dataset_slug or "").rsplit("/", 1)[-1])
                  .replace("@RUNTIME_PREFLIGHT@", "\n".join(
                      "    " + line for line in lane.checkout_preflight_script(
                          lane.checkout_inventory(checkout_paths or spec.checkout_paths, lane="training")).splitlines()))
                  .replace("@REQUIREMENTS@", spec.bundle_requirements)
                  .replace("@RUN_TAG@", tag)
                  .replace("@SUITE_CONFIG@", spec.train_suite_config)
                  .replace("@BUNDLE_KERNEL_SLUG@", bundle_slug)
                  .replace("@CHECKPOINT@", resolved_checkpoint)
        .replace("@WANDB_API_KEY@", os.environ.get("WANDB_API_KEY")
                 or lane._env_dot_value("WANDB_API_KEY")
                 or self._bake_wandb_key_missing()))
        lane.atomic_write_json(metadata, stage / lane._spec().files.kernel_metadata)
        script = lane.KernelTemplates.render_runtime(script, spec)
        script = lane.KernelLifecycle.wrap_script(script)
        lane._kernel_script_gate(script)
        lane._attachment_gate(script, metadata)
        (stage / code_file).write_text(script, encoding="utf-8")
        receipt = {
            "kernel": resolved_slug,
            "kind": kind,
            "gpu": True,
            "branch": spec.branch,
            "revision": pinned,
            "run_tag": tag,
            "bundle_kernel": bundle_slug,
            "bundle_dataset_version": bundle_dataset_version if kind == "train" else None,
            "checkpoint": resolved_checkpoint if kind == "embed" else None,
            "staged": str(stage),
            "code_file": code_file,
        }
        lane.atomic_write_json(receipt, stage / lane._spec().files.kernel_receipt.format(kind=kind))
        return receipt

    @staticmethod
    def push_kernel(stage_dir: Path) -> dict[str, Any]:
        """Push any staged kernel (bundle | train | embed) via the CLI."""
        from cli import kaggle_lane as lane

        spec = lane._spec()
        from core.runtime_inputs import staged_kernel_preflight
        staged_kernel_preflight(Path(stage_dir))
        executable = lane._require_kaggle_executable(spec.kaggle_executable)
        metadata = json.loads((Path(stage_dir) / lane._spec().files.kernel_metadata).read_text())
        kind = next((kind for kind, code_file in spec.files.code_files.items()
                     if code_file == metadata["code_file"]), None)
        if kind is None:
            raise RuntimeError("pushed kernel has no configured watcher kind")
        command = [executable, "kernels", "push", "-p", str(stage_dir)]
        _, _ = lane._run_kaggle(command)
        configured_slug = {"bundle": spec.cpu_kernel_slug,
                           "train": spec.gpu_kernel_slug,
                           "embed": spec.embedding_kernel_slug}[kind]
        target = {"slug": metadata["id"]} if metadata["id"] != configured_slug else {}
        plan = {"mode": "executed", "kernel": metadata["id"], "pushed": True,
                "staged": str(stage_dir)}
        plan.update(lane._spawn_autowatch(kind, **target))
        return plan

    @staticmethod
    def kernel_status(slug: str | None = None, *, which: str = "cpu") -> dict[str, Any]:
        from cli import kaggle_lane as lane

        spec = lane._spec()
        resolved = slug or (spec.embedding_kernel_slug if which == "embed"
                            else spec.gpu_kernel_slug if which == "gpu"
                            else spec.cpu_kernel_slug)
        if not resolved:
            raise RuntimeError(
                f"config kaggle.{which}_kernel_slug is unset; pass a slug or name "
                f"the {which} kernel in config")
        executable = lane._require_kaggle_executable(spec.kaggle_executable)
        command = [executable, "kernels", "status", resolved]
        _, output = lane._run_kaggle(command)
        # CLI 2.x prints  "KernelWorkerStatus.COMPLETE"; 1.x printed bare words.
        normalized = output.replace("KernelWorkerStatus.", "")
        status = "unknown"
        for candidate in ("cancelAcknowledged", "cancelRequested", "complete",
                          "running", "queued", "error"):
            if candidate.lower() in normalized.lower():
                status = candidate
                break
        return {"kernel": resolved, "status": status, "raw": output.strip()}

    @staticmethod
    def stop_kernel(slug: str | None = None, *, which: str = "cpu",
                    execute: bool) -> dict[str, Any]:
        """Stop a kernel's running session with a verified verdict.

        "Replacement is the kill" is not a verified kill: the plan must report
        stopped / still_running and success is only ever a terminal status.
        Preferred mechanism: the stream follower records the session's
        kernel_session_id (files.session_id_file under logs/kaggle/); the SDK
        cancels that exact session (cancel_kernel_session). Without a recorded
        id — or when the SDK cancel raises — the fallback is the version
        replace: push a trivial stub that prints and exits, and the platform
        tears down the current session to run version N+1. Both paths are
        verified by bounded status polls (limits.stop_verify_polls x
        logs_poll_seconds); no terminal status inside the window degrades the
        verdict to still_running and the stop fails loud. Dry-run by default;
        --execute performs the cancel/replace.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        resolved = slug or (spec.embedding_kernel_slug if which == "embed"
                            else spec.gpu_kernel_slug if which == "gpu"
                            else spec.cpu_kernel_slug)
        if not resolved:
            raise RuntimeError(
                f"config kaggle.{which}_kernel_slug is unset; pass a slug or "
                f"name the {which} kernel in config")
        stage = lane.staging_dir() / lane._spec().files.stop_stage.format(which=which)
        plan: dict[str, Any] = {
            "mode": "executed" if execute else "dry-run",
            "kernel": resolved,
            "staged": str(stage),
        }
        if not execute:
            return plan
        session_id: int | None = None
        session_file = (lane.lane_logs_dir()
                        / spec.files.session_id_file.format(
                            kernel=resolved.rsplit("/", 1)[-1]))
        try:
            session_id = int(session_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            session_id = None
        if session_id is not None:
            # Preferred: the SDK cancel — kills the exact recorded session.
            try:
                from kagglesdk.kaggle_client import KaggleClient
                from kagglesdk.kaggle_env import KaggleEnv
                from kagglesdk.kernels.types.kernels_api_service import (
                    ApiCancelKernelSessionRequest)
                request = ApiCancelKernelSessionRequest()
                request.kernel_session_id = session_id
                KaggleClient(env=KaggleEnv.PROD).kernels.kernels_api_client \
                    .cancel_kernel_session(request)
                plan["cancel_method"] = "sdk_cancel_kernel_session"
            except Exception as err:
                plan["cancel_error"] = f"{type(err).__name__}: {str(err)[:200]}"
        if "cancel_method" not in plan:
            # Fallback: version replace — the stub push tears the session down.
            stage.mkdir(parents=True, exist_ok=True)
            title = resolved.rsplit("/", 1)[-1].replace("-", " ").title()
            lane.atomic_write_json({
                "id": resolved,
                "title": title,
                "code_file": lane._spec().files.stop_code,
                "language": "python",
                "kernel_type": "script",
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [],
                "kernel_sources": [],
                "competition_sources": [],
                "is_private": True,
            }, stage / lane._spec().files.kernel_metadata)
            (stage / lane._spec().files.stop_code).write_text(
                'print("[kaggle-lane] run cancelled by owner; session released")\n',
                encoding="utf-8")
            executable = lane._require_kaggle_executable(spec.kaggle_executable)
            command = [executable, "kernels", "push", "-p", str(stage)]
            lane._run_kaggle(command)
            plan["cancel_method"] = "version_replace"
        # Verified stop: bounded status polls; only a terminal state is a stop.
        verdict = "still_running"
        deadline = (time.monotonic()
                    + max(spec.limits.stop_verify_polls, 1) * spec.logs_poll_seconds)
        while time.monotonic() < deadline:
            state = lane.kernel_status(resolved)["status"]
            if state in ("complete", "error"):
                verdict = "stopped"
                plan["terminal_state"] = state
                break
            time.sleep(spec.logs_poll_seconds)
        plan["verdict"] = verdict
        plan["stopped"] = verdict == "stopped"
        if not plan["stopped"]:
            raise RuntimeError(
                f"stop did not reach a terminal state within the verify window "
                f"({verdict}; session {resolved}, method {plan['cancel_method']})")
        return plan

