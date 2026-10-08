"""Kernel staging, launch, status, and explicit session release."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any



#: The train kernel ships the suite's OWN sealed result Bundle (role handoff).
#:
#: ``model_tracks.run`` already seals ``<run_tag>.<result_archive_format>``
#: through ``Bundle.seal_result``: the ``result`` role owns the member set (the
#: selected checkpoint only) and the run-tag manifest the finalize boundary
#: checks. That archive IS the role bundle the finalize job's boundary load
#: accepts. The shared template helper re-tarred the output TREE instead — no
#: role manifest, no transport digest in the manifest the fetch reads — and the
#: finalize boundary rejected it with "archive manifest missing" (the proven
#: broken train->finalize role handoff). Splice this override into the TRAIN
#: kernel only: it copies the sealed archive and records its whole-file sha256
#: as the manifest transport token, so ``kaggle_outputs.fetch_kernel_output``
#: and the finalize job both verify the one archive that crossed. The embed
#: kernel keeps the tree tarball (its vectors output is not a Bundle role).
TRAIN_RESULT_BUNDLE_SHIP = '''
def stage_result_archive(output, *, kind, extra):
    """Ship ``model_tracks.run``'s sealed result Bundle, never a tree tarball.

    Redefines the shared tree-tar helper for the train kernel so the fetched
    artifact carries the role manifest the finalize boundary verifies, and the
    manifest records the archive's whole-file sha256 (the transport token the
    fetch and the finalize boundary pin).
    """
    import yaml as _yaml
    from model_tracks.config import SuiteConfig as _SuiteConfig
    config_path = Path(SUITE_CONFIG)
    if not config_path.is_absolute():
        config_path = root / config_path
    settings = _SuiteConfig.model_validate(_yaml.safe_load(config_path.read_text()))
    sealed_archive = output.with_suffix("." + settings.result_archive_format)
    if not sealed_archive.is_file():
        raise SystemExit(
            "model_tracks.run sealed no result bundle at " + str(sealed_archive))
    result_archive = WORKING / LANE["files"]["result_archive"].format(kind=kind)
    shutil.copy2(sealed_archive, result_archive)
    digest = sha256_file(result_archive)
    (WORKING / (result_archive.name + LANE["files"]["hash_suffix"])).write_text(
        digest + "\\n", encoding="utf-8")
    (WORKING / LANE["files"]["result_manifest"].format(kind=kind)).write_text(
        json.dumps({"kind": kind, "run_tag": RUN_TAG, "revision": REVISION,
                    "archive": result_archive.name, "archive_sha256": digest,
                    **extra}, indent=2), encoding="utf-8")
    print("[%s] shipped sealed result bundle %s sha256=%s"
          % (kind, result_archive.name, digest), flush=True)
    return digest
'''


class KaggleKernels:
    """Kernel staging, launch, status, and explicit session release."""

    @staticmethod
    def _bake_wandb_key_missing() -> str:
        raise RuntimeError(
            'wandb is always on (owner order 2026-10-07): kernel staging '
            'requires WANDB_API_KEY in the environment or .env; refusing to '
            'bake a silent local-only train run')

    @staticmethod
    def _wandb_api_key() -> str:
        """The key a staged kernel bakes, or a loud refusal (never silent).

        The environment wins; otherwise the repository's own .env is read
        through ``core.common.TRAIN_ROOT`` — the stable project root, NOT the
        lane's ``TRAIN_ROOT``, which alternate checkouts and tests re-point and
        which must never hide the operator's key from the bake.
        """
        from cli import kaggle_lane as lane
        from core.common import TRAIN_ROOT as repository_root

        return (os.environ.get("WANDB_API_KEY")
                or lane._env_dot_value("WANDB_API_KEY", repository_root)
                or KaggleKernels._bake_wandb_key_missing())

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
    def embed_objective(*, execute: bool = False) -> dict[str, Any]:
        """The embed objective's EXPLICIT state — never a silent absence.

        Owner context: ``kaggle.embedding_kernel_slug`` names
        ``fbarulli/er-embed-gpu`` and no embed kernel exists on the account.
        Configuration alone cannot see that, so the verdict is two-stage:
        ``configured`` is the config contract (both the kernel and its request
        dataset slug are named), and — only with ``execute`` — ``available``
        probes the account through ``kaggle kernels status``. A dry run never
        contacts Kaggle and reports ``available: None``.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        verdict: dict[str, Any] = {
            "objective": "embed",
            "kernel": spec.embedding_kernel_slug,
            "request_dataset": spec.embedding_dataset_slug,
            "configured": bool(spec.embedding_kernel_slug
                               and spec.embedding_dataset_slug),
        }
        if not verdict["configured"]:
            verdict.update(
                available=False,
                reason=("the embed objective is absent: config "
                        "kaggle.embedding_kernel_slug and "
                        "kaggle.embedding_dataset_slug must both name owner/slug "
                        "targets before any embed step can run"),
            )
            return verdict
        if not execute:
            verdict.update(
                available=None,
                reason="dry-run: account presence is not probed; pass --execute "
                       "to verify the configured embed kernel exists",
            )
            return verdict
        try:
            status = lane.kernel_status(slug=spec.embedding_kernel_slug, which="embed")
        except (RuntimeError, OSError) as error:
            verdict.update(
                available=False,
                reason=(f"configured embed kernel {spec.embedding_kernel_slug!r} "
                        f"could not be reached on the account: "
                        f"{str(error)[-400:]}"),
            )
            return verdict
        verdict["status"] = status.get("status")
        verdict["available"] = status.get("status") != "unknown"
        verdict["reason"] = (
            f"account reports the configured embed kernel "
            f"{spec.embedding_kernel_slug!r} as {status.get('status')!r}"
            if verdict["available"] else
            (f"the account does not report the configured embed kernel "
             f"{spec.embedding_kernel_slug!r} (status {status.get('status')!r})"))
        return verdict

    @staticmethod
    def require_embed_objective(*, execute: bool = False) -> dict[str, Any]:
        """The embed gate: return the verdict or fail loud with the two fixes.

        Every surface that would run the embed objective (the chain, the
        standalone check) calls this FIRST, so a missing kernel is a named
        error and never a skipped, silent step.
        """
        verdict = KaggleKernels.embed_objective(execute=execute)
        if verdict.get("available") is False:
            raise RuntimeError(
                f"embed objective unavailable: {verdict.get('reason')}\n"
                "push the embed kernel first (er-kaggle --what embed-kernel "
                "--execute stages and pushes kaggle.embedding_kernel_slug with "
                "its request dataset attached), or drop the embed step so the "
                "objective is explicitly absent — it is never silently skipped")
        return verdict

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
        identity = lane.kernel_identity("bundle", spec)
        slug = identity.slug(spec)
        if not slug:
            raise RuntimeError(
                f"config kaggle.{identity.slug_attr} is unset; name the CPU kernel "
                "(owner/slug) before staging")
        pinned = revision or lane._git_revision()
        from core import runtime_inputs
        tip = runtime_inputs.require_published_tip_match(
            pinned, spec.repository, spec.branch)
        cohort = cohort or spec.default_cohort
        cohort_dataset = lane.cohort_export_csv(cohort)
        stage = lane.staging_dir() / lane._spec().files.kernel_stage.format(kind=identity.kind)
        stage.mkdir(parents=True, exist_ok=True)
        metadata = {
            "id": slug,
            "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
            "code_file": identity.code_file,
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
        (stage / identity.code_file).write_text(script, encoding="utf-8")
        receipt = {
            "kernel": slug,
            "gpu": False,
            "branch": spec.branch,
            "revision": pinned,
            "published_tip": tip,
            "cohort": cohort,
            "cohort_dataset": cohort_dataset,
            "staged": str(stage),
            "code_file": identity.code_file,
        }
        lane.atomic_write_json(
            receipt, stage / lane._spec().files.kernel_receipt.format(kind=identity.kind))
        return receipt

    @staticmethod
    def _push_and_record_session(stage_dir: Path, slug: str) -> None:
        """Push a staged kernel and record its session id for in-place cancel.

        Every live push gets its own session id written to
        ``logs/kaggle/<kernel>.session_id`` so ``stop`` can use the SDK's
        in-place ``cancel_kernel_session`` instead of a version-replace stub.
        The stale id is cleared first (a push invalidates any prior session),
        and capture is best-effort — the autowatch stream follower is the
        backup writer and must never gate the launch.
        """
        from cli import kaggle_lane as lane

        executable = lane._require_kaggle_executable(lane._spec().kaggle_executable)
        lane.clear_kernel_session_id(slug)
        lane._run_kaggle([executable, "kernels", "push", "-p", str(stage_dir)])
        try:
            lane.capture_kernel_session_id(slug)
        except Exception as error:  # noqa: BLE001 - best-effort launch aid
            lane._log_lane(f"[{slug}] session-id capture skipped: {error}")

    @staticmethod
    def push_bundle_kernel(stage_dir: Path) -> dict[str, Any]:
        """Push the staged CPU kernel via the configured kaggle executable.

        Every live push spawns its own detached autowatch (default since the
        owner order: no session may outlive a terminal run); the watcher
        downloads and releases, no operator arg involved.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        identity = lane.kernel_identity("bundle", spec)
        slug = identity.slug(spec)
        if not slug:
            raise RuntimeError(
                f"config kaggle.{identity.slug_attr} is unset; name the CPU kernel "
                "(owner/slug) before pushing")
        from core.runtime_inputs import staged_kernel_preflight
        staged_kernel_preflight(Path(stage_dir))
        KaggleKernels._push_and_record_session(Path(stage_dir), slug)
        plan = {"mode": "executed", "kernel": slug, "pushed": True,
                "staged": str(stage_dir)}
        plan.update(lane._spawn_autowatch(identity.which))
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
        identity = lane.kernel_identity(kind, spec)
        code_file = identity.code_file
        resolved_slug = slug or identity.slug(spec)
        if not resolved_slug:
            raise RuntimeError(
                f"config kaggle.{identity.slug_attr} is unset; name the {kind} "
                "kernel (owner/slug) before staging")
        bundle_identity = lane.kernel_identity("bundle", spec)
        bundle_slug = bundle_identity.slug(spec)
        if not bundle_slug:
            raise RuntimeError(
                f"config kaggle.{bundle_identity.slug_attr} is unset; the GPU "
                "kernel attaches the CPU bundle kernel output")
        pinned = revision or lane._git_revision()
        from core import runtime_inputs
        tip = runtime_inputs.require_published_tip_match(
            pinned, spec.repository, spec.branch)
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
        # The train kernel ships the suite's own sealed result Bundle, so the
        # finalize job's `Bundle.load(..., "result")` boundary accepts it (the
        # proven role handoff fix). The embed kernel keeps the shared tree-tar
        # helper: its vectors output is not a Bundle role.
        ship = TRAIN_RESULT_BUNDLE_SHIP if kind == "train" else ""
        template = lane.TRAIN_KERNEL_SHARED + ship + body
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
        .replace("@WANDB_API_KEY@", KaggleKernels._wandb_api_key()))
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
            "published_tip": tip,
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
    def stage_finalize_kernel(*, revision: str | None = None,
                              run_tag: str | None = None,
                              bundle_dataset_version: str | None = None,
                              checkout_paths: list[str] | None = None,
                              slug: str | None = None) -> dict[str, Any]:
        """Stage the CPU finalize kernel: metadata + script + receipt.

        The finalize role (``model_tracks.bundle_steps.finalize``) is the third
        bundling step and runs as ONE remote CPU lane job — the operator box is
        no longer a finalize surface. It reuses the bundling CPU kernel slug
        (``kaggle.cpu_kernel_slug``): generation and finalize are the two
        ``bundle_steps`` roles, so Kaggle mounts them as two versions of the
        same account kernel under two code files (exactly like the stop-stub
        version replace); ``slug`` overrides it for a dedicated kernel.

        It attaches BOTH verified inputs: the published prepared-inputs bundle
        dataset (``kaggle.bundle_dataset_slug``, immutable) and the trained
        result kernel output (``kaggle.gpu_kernel_slug``). The revision is
        pinned at staging time and must be the published origin tip, so the
        finalize job runs exactly the source the staged receipt names.

        Dry-safe: writes only into the staging area; ``push_kernel`` performs
        the network call.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        identity = lane.kernel_identity(lane.FINALIZE_KERNEL_KIND, spec)
        train_identity = lane.kernel_identity("train", spec)
        resolved_slug = slug or identity.slug(spec)
        if not resolved_slug:
            raise RuntimeError(
                f"config kaggle.{identity.slug_attr} is unset; the finalize job runs "
                "on the bundling CPU kernel (generation + finalize are the two "
                "bundle_steps roles) — name it (owner/slug) before staging")
        train_slug = train_identity.slug(spec)
        if not train_slug:
            raise RuntimeError(
                f"config kaggle.{train_identity.slug_attr} is unset; the finalize kernel "
                "attaches the trained result kernel output and cannot run "
                "without it")
        bundle_dataset = spec.bundle_dataset_slug
        if not bundle_dataset:
            raise RuntimeError(
                "config kaggle.bundle_dataset_slug is unset; the finalize "
                "kernel attaches the published prepared-inputs bundle and "
                "cannot run without it")
        code_file = identity.code_file
        pinned = revision or lane._git_revision()
        from core import runtime_inputs
        tip = runtime_inputs.require_published_tip_match(
            pinned, spec.repository, spec.branch)
        tag = run_tag or (spec.run_tag_prefix
                          + time.strftime(spec.limits.run_tag_format, time.gmtime()))
        stage = lane.staging_dir() / lane._spec().files.kernel_stage.format(kind=identity.kind)
        stage.mkdir(parents=True, exist_ok=True)
        bundle_dataset_entry = bundle_dataset
        if bundle_dataset_version:
            bundle_dataset_entry = f"{bundle_dataset}/{bundle_dataset_version}"
        metadata: dict[str, Any] = {
            "id": resolved_slug,
            "title": resolved_slug.rsplit("/", 1)[-1].replace("-", " ").title(),
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "enable_gpu": False,
            "enable_internet": True,
            "dataset_sources": [bundle_dataset_entry],
            "kernel_sources": [train_slug],
            "competition_sources": [],
            "is_private": True,
        }
        checkout = list(checkout_paths or spec.checkout_paths)
        template = lane.TRAIN_KERNEL_SHARED + lane.FINALIZE_KERNEL_BODY
        script = (template
                  .replace("@REPOSITORY@", spec.repository)
                  .replace("@BRANCH@", spec.branch)
                  .replace("@REVISION@", pinned)
                  .replace("@CHECKOUT_PATHS@",
                           json.dumps(lane.checkout_members(checkout, lane="bundle")))
                  .replace("@BUNDLE_DATASET_NAME@", bundle_dataset.rsplit("/", 1)[-1])
                  .replace("@RUNTIME_PREFLIGHT@", "\n".join(
                      "    " + line for line in lane.checkout_preflight_script(
                          lane.checkout_inventory(checkout, lane="bundle")).splitlines()))
                  .replace("@REQUIREMENTS@", spec.bundle_requirements)
                  .replace("@RUN_TAG@", tag)
                  .replace("@SUITE_CONFIG@", spec.train_suite_config)
                  .replace("@BUNDLE_KERNEL_SLUG@", resolved_slug)
                  .replace("@FINALIZE_RESULT_NAME@", identity.result_name))
        lane.atomic_write_json(metadata, stage / lane._spec().files.kernel_metadata)
        script = lane.KernelTemplates.render_runtime(script, spec)
        script = lane.KernelLifecycle.wrap_script(script)
        lane._kernel_script_gate(script)
        lane._attachment_gate(script, metadata)
        (stage / code_file).write_text(script, encoding="utf-8")
        receipt = {
            "kernel": resolved_slug,
            "kind": identity.kind,
            "gpu": False,
            "role": identity.bundle_role,
            "branch": spec.branch,
            "revision": pinned,
            "published_tip": tip,
            "run_tag": tag,
            "bundle_kernel": resolved_slug,
            "result_kernel": train_slug,
            "bundle_dataset": bundle_dataset,
            "bundle_dataset_version": bundle_dataset_version,
            "checkout_paths": list(checkout),
            "staged": str(stage),
            "code_file": code_file,
        }
        lane.atomic_write_json(
            receipt, stage / lane._spec().files.kernel_receipt.format(kind=identity.kind))
        return receipt

    @staticmethod
    def push_kernel(stage_dir: Path) -> dict[str, Any]:
        """Push any staged kernel (bundle | train | embed | finalize) via the CLI."""
        from cli import kaggle_lane as lane

        spec = lane._spec()
        from core.runtime_inputs import staged_kernel_preflight
        staged_kernel_preflight(Path(stage_dir))
        metadata = json.loads((Path(stage_dir) / lane._spec().files.kernel_metadata).read_text())
        # The pushed script name is the kind key: one registry entry per kind.
        identities = lane.kernel_identities(spec)
        kind = next((identity.kind for identity in identities.values()
                     if identity.code_file == metadata["code_file"]), None)
        if kind is None:
            raise RuntimeError("pushed kernel has no configured watcher kind")
        identity = identities[kind]
        KaggleKernels._push_and_record_session(Path(stage_dir), metadata["id"])
        configured_slug = identity.slug(spec)
        target = {"slug": metadata["id"]} if metadata["id"] != configured_slug else {}
        plan = {"mode": "executed", "kernel": metadata["id"], "pushed": True,
                "staged": str(stage_dir)}
        plan.update(lane._spawn_autowatch(identity.kind, **target))
        return plan

    @staticmethod
    def _kernel_slug(spec, which: str) -> str | None:
        """The configured slug for one kernel identity, or None.

        ``finalize`` deliberately resolves to the bundling CPU slug: the
        finalize job is the second ``bundle_steps`` role on the same account
        kernel, so it has no slug of its own to configure.
        """
        from cli import kaggle_lane as lane

        return lane.kernel_identity(which, spec).slug(spec)

    @staticmethod
    def kernel_status(slug: str | None = None, *, which: str = "cpu") -> dict[str, Any]:
        from cli import kaggle_lane as lane

        spec = lane._spec()
        resolved = slug or KaggleKernels._kernel_slug(spec, which)
        if not resolved:
            config_key = lane.kernel_identity(which, spec).slug_attr
            raise RuntimeError(
                f"config kaggle.{config_key} is unset; pass a slug or name "
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
                    execute: bool, wait: bool = True) -> dict[str, Any]:
        """Stop a kernel's running session with a verified verdict.

        Preferred mechanism: the recorded session id (written on every push and
        by the stream follower) feeds the SDK's in-place
        ``cancel_kernel_session`` — no new run. Without a recorded id — or when
        the SDK cancel raises — the fallback is the version replace: push a
        trivial stub that prints and exits, and the platform tears down the
        current session to run version N+1. Both paths are verified by bounded
        status polls (limits.stop_verify_polls x logs_poll_seconds); no terminal
        status inside the window degrades the verdict to still_running and the
        stop fails loud. Dry-run by default; ``--execute`` performs the
        cancel/replace. With ``wait=False`` the cancel/replace is issued and the
        plan is returned immediately (verdict ``requested``) — the caller's
        detached watcher owns the terminal confirmation, so the CLI never blocks
        silently for minutes.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        resolved = slug or KaggleKernels._kernel_slug(spec, which)
        if not resolved:
            config_key = lane.kernel_identity(which, spec).slug_attr
            raise RuntimeError(
                f"config kaggle.{config_key} is unset; pass a slug or "
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
        if not wait:
            # Fire-and-forget: the cancel/replace is issued; a detached watcher
            # (or the operator) confirms terminal. Never block silently.
            plan["verdict"] = "requested"
            plan["stopped"] = None
            return plan
        # Verified stop: bounded status polls; only a terminal state is a stop.
        verdict = "still_running"
        deadline = (time.monotonic()
                    + max(spec.limits.stop_verify_polls, 1) * spec.logs_poll_seconds)
        last_state = None
        while time.monotonic() < deadline:
            state = lane.kernel_status(resolved)["status"]
            if state != last_state:
                print(f"[stop] {resolved} state={state} (method="
                      f"{plan['cancel_method']})", flush=True)
                last_state = state
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

