"""Sequential preparation, training, and embedding orchestration."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any



class KaggleChain:
    """Sequential preparation, training, and embedding orchestration."""

    @staticmethod
    def _await_autowatch_receipt(kind: str, *,
                                 deadline_polls: int | None = None) -> dict[str, Any]:
        """Wait for the auto-spawned watcher to finish its terminal-handler pass.

        The chain reuses each push path exactly as-is (push_bundle_kernel /
        push_kernel + single _spawn_autowatch) — every push carries its own
        watcher, so the chain waits on the watcher's receipt
        (results/kaggle_lane/autowatch_<kind>.receipt.json) instead of running a
        competing second watcher (two actors racing into the rmtree'd fetch dir
        is exactly the failure class this prevents). Past the supervise harvest
        ceiling the chain fails loud; the watcher keeps running independently.
        """
        from cli import kaggle_lane as lane

        deadline_polls = lane._spec().limits.max_polls if deadline_polls is None else deadline_polls
        poll = lane._spec().logs_poll_seconds
        receipt_path = lane.staging_dir() / lane._spec().files.autowatch_receipt.format(kind=kind)
        for _ in range(deadline_polls + 1):
            if receipt_path.is_file():
                return json.loads(receipt_path.read_text(encoding="utf-8"))
            time.sleep(poll)
        raise RuntimeError(
            f"chain gave up waiting for the {kind} watcher receipt "
            f"({receipt_path}) after {deadline_polls} polls; the spawned "
            "watcher keeps running — check logs/kaggle/autowatch_*.log")

    @staticmethod
    def _clear_stale_autowatch_receipt(kind: str) -> str | None:
        from cli import kaggle_lane as lane

        stale = lane.staging_dir() / lane._spec().files.autowatch_receipt.format(kind=kind)
        if stale.exists():
            stale.unlink()
            return str(stale)
        return None

    @staticmethod
    def _verify_chain_step(*, kind: str, watch: dict[str, Any],
                           plan_entry: dict[str, Any],
                           expect_publish: bool = False) -> None:
        """Chain gate: terminal-complete status, sha-verified fetch, and (for
        the bundle step) the publish default actually published."""
        from cli import kaggle_lane as lane

        if watch.get("status") != "complete":
            raise RuntimeError(
                f"chain {kind} watcher reported status {watch.get('status')!r} "
                f"(expected complete); failures={json.dumps(watch.get('failures', {}))[:800]}")
        fetch = watch.get("fetch") or {}
        if not fetch.get("verified") or fetch.get("failed"):
            raise RuntimeError(
                f"chain {kind} fetch was not verified: {json.dumps(fetch)[:800]}")
        plan_entry["fetched_size"] = fetch.get("archive_size")
        publish = fetch.get("publish") or {}
        plan_entry["publish"] = publish
        if expect_publish and not publish.get("published"):
            raise RuntimeError(
                f"chain {kind} publish step did not publish: {publish.get('error') or json.dumps(publish)[:800]}")

    @staticmethod
    def run_chain(*, cohort: str | None = None, with_embed: bool = False,
                  with_finalize: bool = False, execute: bool) -> dict[str, Any]:
        """One command runs the whole kaggle loop supervised end-to-end.

        Steps and their fail-loud gates, in order:
        1. bundle-kernel: staged + pushed (cohort pinned at the chain's HEAD
           revision); push_bundle_kernel's own watcher supervises → verifies the
           fetch (sha vs kernel receipt) → releases the session; the publish
           default inside the verified fetch publishes a fresh bundle-dataset
           version and records its number.
        2. train-kernel: staged (SAME revision — any drift fails loud with both
           named) + pushed via the standard push path; its single spawned
           watcher runs the terminal-handler loop; the chain waits on the
           watcher receipt (never double-spawns, never double-watches). The
           train stage attaches the fresh bundle dataset version the publish
           recorded (owner/slug/version) when the mount pin is available.
        3. (--with-embed) embed-kernel: identical pattern after the train
           watcher reports completion; the step runs only when the embed
           objective is configured AND present on the account (an absent kernel
           fails loud with the push path named, never silently skipped).
        4. (--with-finalize) finalize-kernel: the remote CPU job that runs
           model_tracks.bundle_steps role=result from a sparse checkout — the
           operator box is no longer a finalize surface. It reuses the bundling
           CPU kernel slug and attaches the published inputs bundle plus the
           trained result kernel output; the sealed result bundle is fetched
           and digest-verified like every other step.
        Dry-run prints the entire plan (no staging writes, no subprocesses).
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        cohort = cohort or spec.default_cohort
        stage_root = lane.staging_dir()
        head = lane._git_revision()
        # One identity registry (config SSOT) resolves every step's slug and the
        # config field it came from — no per-surface kind->slug table here.
        identities = lane.kernel_identities(spec)
        slugs = {kind: identity.slug(spec) for kind, identity in identities.items()}
        publish_slug = spec.bundle_dataset_slug
        plan: dict[str, Any] = {
            "what": "chain", "mode": "executed" if execute else "dry-run",
            "cohort": cohort, "with_embed": with_embed,
            "with_finalize": with_finalize, "revision": head,
            "slugs": slugs, "publish_slug": publish_slug, "steps": {},
        }
        kinds = ["bundle", "train"] + (["embed"] if with_embed else []) \
            + (["finalize"] if with_finalize else [])
        for step_kind in kinds:
            if not slugs[step_kind]:
                raise RuntimeError(
                    f"chain requires config kaggle.{identities[step_kind].slug_attr}; "
                    "name the kernel (owner/slug) before chaining")
        if not publish_slug:
            raise RuntimeError(
                "chain requires config kaggle.bundle_dataset_slug; the publish "
                "step and the train stage's bundle mount depend on it")
        if not execute:
            for step_kind in kinds:
                plan["steps"][step_kind] = {
                    "stage": {"kernel": slugs[step_kind],
                              "revision": head, "cohort": cohort,
                              "staged": str(stage_root / spec.files.kernel_stage.format(kind=step_kind))},
                    "push": "planned (reuses the standard push path)",
                    "autowatch": "planned (single spawn per push path)",
                    "mount": ({"dataset_sources": [f"{publish_slug}/<fresh version>"]}
                              if step_kind == "train" else None),
                }
                if step_kind == "bundle":
                    plan["steps"][step_kind]["publish"] = {
                        "slug": publish_slug,
                        "stage": str(lane._bundle_dataset_stage(cohort)),
                        "mode": "planned",
                    }
                if step_kind == "embed":
                    # The embed objective is stated, never implied: an
                    # unconfigured objective fails here (dry run included), and
                    # the verdict rides the plan so the operator sees exactly
                    # which kernel/dataset the step would use.
                    plan["steps"][step_kind]["objective"] = \
                        lane.require_embed_objective(execute=False)
                if step_kind == "finalize":
                    plan["steps"][step_kind]["mount"] = {
                        "dataset_sources": [f"{publish_slug}/<fresh version>"],
                        "kernel_sources": [identities["train"].slug(spec)],
                    }
                    plan["steps"][step_kind]["role"] = "result"
            return plan
        # ── the published-tip invariant: the chain pins per-step revisions
        # from `head`; before any staging write, the head must BE the
        # published origin tip (the staged-race guard shared with the
        # stage surfaces, which re-check as they pin).
        from core import runtime_inputs
        tip = runtime_inputs.require_published_tip_match(
            head, spec.repository, spec.branch)
        plan["published_tip"] = tip

        def assert_revision(pinned: str, where: str) -> None:
            if pinned != head:
                raise RuntimeError(
                    f"chain revision drift at {where}: chain pinned {head} but "
                    f"the stage pinned {pinned} (named mismatch — re-run the "
                    "chain from the same HEAD)")

        for step_index, step_kind in enumerate(kinds):
            expect_publish = step_kind == "bundle"
            entry: dict[str, Any] = {}
            if step_kind == "embed":
                # Live embed step: the configured kernel must EXIST on the
                # account (config alone cannot tell — the phantom
                # fbarulli/er-embed-gpu case), or the step fails loud with the
                # push path named instead of running a missing kernel.
                entry["objective"] = lane.require_embed_objective(execute=True)
            if step_kind == "bundle":
                receipt = lane.stage_bundle_kernel(revision=head, cohort=cohort)
            elif step_kind == "finalize":
                # The remote CPU finalize job: bundle_steps role=result from a
                # sparse checkout, attaching the published inputs bundle (the
                # version this chain's publish recorded) and the trained
                # result kernel output. Same published-tip guard as every
                # other stage (stage_finalize_kernel re-checks it).
                receipt = lane.stage_finalize_kernel(
                    revision=head,
                    bundle_dataset_version=(publish_plan or {}).get("dataset_version"),
                    run_tag=(plan["steps"]["train"]["stage"]["run_tag"]
                             if "train" in plan["steps"] else None))
            else:
                receipt = lane.stage_gpu_kernel(
                    kind=step_kind, revision=head,
                    bundle_dataset_version=(
                        (publish_plan or {}).get("dataset_version")
                        if step_kind == "train" else None))
            assert_revision(receipt["revision"], f"{step_kind} stage")
            entry["stage"] = receipt
            stale = lane._clear_stale_autowatch_receipt(step_kind)
            if stale:
                entry["cleared_stale_receipt"] = stale
            if step_kind == "bundle":
                entry["push"] = lane.push_bundle_kernel(Path(receipt["staged"]))
                entry["autowatch"] = entry["push"].get("autowatch")
            else:
                # Exact main()-lineage push path: push_kernel + ONE watcher spawn
                # (the finalize step reuses it, with its own watcher identity).
                entry["push"] = lane.push_kernel(Path(receipt["staged"]))
                entry["autowatch"] = entry["push"].get("autowatch")
            watch = lane._await_autowatch_receipt(step_kind)
            entry["watch"] = watch
            lane._verify_chain_step(kind=step_kind, watch=watch, plan_entry=entry,
                               expect_publish=expect_publish)
            if step_kind == "bundle":
                publish_plan = entry["publish"]
            plan["steps"][step_kind] = entry

        receipt_path = stage_root / lane._spec().files.chain_receipt
        try:
            lane.atomic_write_json(plan, receipt_path)
        except OSError as error:
            print(lane._stamp(), f"[kaggle-lane] chain receipt write failed ({error}); "
                  "continuing", flush=True)
        plan["receipt"] = str(receipt_path)
        return plan

