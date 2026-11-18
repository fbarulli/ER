"""The laya training-run owner: ONE class for the end-to-end run lifecycle.

``LayaTrainingRun`` is the lane's Facade over the existing owners. A training
setup is a class call, never scattered ad-hoc assembly:

  * :meth:`resolve_study` — the ONE Optuna study (remote PostgreSQL when
    ``OPTUNA_STORAGE_URL`` is configured, else the config-declared local
    SQLite study), through the canonical ``training.hpo_study.StudyOwner``;
  * :meth:`stage` — the fine-tune / HPO payload from the config SSOT
    (``LayaSpec`` / ``FinetuneSpec`` / ``config/laya_hpo_space.yaml``) through
    ``LayaStagingFactory`` + ``LayaRecipeFactory`` (and ``LayaHpoStager``);
  * :meth:`publish` — create-or-version the dataset the kernel attaches, from
    the same config SSOT, BEFORE the push (``LayaPublishFactory``);
  * :meth:`push` — ``kaggle kernels push`` with the shared session capture
    (``LayaTransportFactory`` -> ``KaggleKernels.push_with_session_capture``);
  * :meth:`launch` — the ONE ``--execute`` lifecycle: publish -> push -> spawn
    the detached terminal watcher (the fine-tune lane's proven execute path);
  * :meth:`track` — real-time via the canonical readers (the W&B run reader,
    degrading to the ONE execution-log reader; the removed SSE follower is
    never used);
  * :meth:`harvest` — fetch the output/receipt (``KernelOutputFetcher`` /
    ``LayaTransportFactory.collect_kaggle_result``);
  * :meth:`teardown` — stop the running session, THEN delete the kernel.

Every collaborator is injected (DI); :class:`LayaTrainingRunFactory` is the ONE
assembly site that builds the owner from the config SSOT. Illegal run kinds are
unrepresentable (:class:`LayaRunKind` is an enum over the SSOT decision kinds).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from cli.laya_hpo import HPO_DECISION, LANES as HPO_LANES
from cli.laya_publish import LayaPublishFactory
from cli.laya_recipe import (
    FINETUNE_DECISION,
    FINETUNE_EVAL_DECISION,
    FINETUNE_SMOKE_DECISION,
    HOLDOUT_EVAL_DECISION,
    LayaRecipeFactory,
)
from cli.laya_runtime import LayaRuntimeFactory
from cli.laya_staging import LayaStagingFactory
from cli.laya_transport import LayaTransportFactory
from core.laya_config import LayaSpec
from training.hpo_study import ResolvedStudy


class LayaRunKind(StrEnum):
    """The laya run kinds the owner stages (values are the SSOT decision kinds)."""

    FINETUNE = FINETUNE_DECISION
    FINETUNE_SMOKE = FINETUNE_SMOKE_DECISION
    FINETUNE_EVAL = FINETUNE_EVAL_DECISION
    HOLDOUT_EVAL = HOLDOUT_EVAL_DECISION
    HPO = HPO_DECISION


class LayaTrainingRun:
    """Facade over the lane owners for ONE laya training run (lifecycle SRP).

    Construction is a factory's job (see :class:`LayaTrainingRunFactory`); every
    collaborator is passed in so no method re-reads a global or instantiates a
    dependency it should be given.
    """

    def __init__(
        self,
        *,
        staging: LayaStagingFactory,
        transport: LayaTransportFactory,
        publisher: LayaPublishFactory,
        study_resolver: Callable[..., ResolvedStudy],
        hpo_stager: Callable[..., Any],
        tracker: Callable[..., dict[str, Any]],
    ) -> None:
        self._staging = staging
        self._transport = transport
        self._publisher = publisher
        self._study_resolver = study_resolver
        self._hpo_stager = hpo_stager
        self._tracker = tracker

    # ── study ──────────────────────────────────────────────────────────────
    def resolve_study(self, *, generation_id: str | None = None,
                      model_key: str | None = None) -> ResolvedStudy:
        """The ONE study storage (remote URL wins; else local SQLite).

        Delegates to the canonical ``StudyOwner`` resolver; this owner never
        reads ``OPTUNA_STORAGE_URL`` itself and carries no study token.
        """
        return self._study_resolver(generation_id=generation_id,
                                    model_key=model_key)

    # ── staging ────────────────────────────────────────────────────────────
    def stage(self, kind: LayaRunKind, *, lane: str = HPO_LANES[0],
              revision: str | None = None, run_tag: str | None = None,
              generation_id: str | None = None, n_trials: int | None = None,
              n_jobs: int | None = None, space_config: str | Path | None = None,
              kernel_slug: str | None = None,
              checkpoint_path: Path | None = None) -> dict[str, Any]:
        """Stage the fine-tune / HPO kernel payload from the config SSOT."""
        if kind is LayaRunKind.HPO:
            return self._hpo_stager(
                lane, revision=revision, run_tag=run_tag,
                generation_id=generation_id, n_trials=n_trials, n_jobs=n_jobs,
                space_config=space_config, kernel_slug=kernel_slug).stage()
        if kind is LayaRunKind.FINETUNE:
            return self._staging.stage_finetune_kernel(
                revision=revision, run_tag=run_tag)
        if kind is LayaRunKind.FINETUNE_SMOKE:
            return self._staging.stage_finetune_kernel(
                revision=revision, run_tag=run_tag, smoke=True)
        if kind is LayaRunKind.FINETUNE_EVAL:
            return self._staging.stage_finetune_eval_kernel(
                revision=revision, run_tag=run_tag,
                checkpoint_path=checkpoint_path)
        if kind is LayaRunKind.HOLDOUT_EVAL:
            return self._staging.stage_holdout_eval_kernel(
                revision=revision, run_tag=run_tag)
        raise ValueError(f"unknown laya run kind: {kind!r}")

    # ── publish / transport ────────────────────────────────────────────────
    def publish(self, kind: LayaRunKind, *, run_tag: str,
                execute: bool = False) -> dict[str, Any]:
        """Create-or-version the dataset the kernel attaches (``--execute`` gated)."""
        return self._publisher.publish_laya_dataset(
            kind.value, run_tag=run_tag, execute=execute)

    def push(self, stage_dir: Path, *, execute: bool = False,
             activate: bool = True) -> dict[str, Any]:
        """Push a staged payload with the shared session capture."""
        return self._transport.push_kaggle_kernel(
            Path(stage_dir), execute=execute, activate=activate)

    def watch(self, kind: LayaRunKind, *, slug: str,
              run_tag: str | None = None) -> dict[str, Any]:
        """Spawn the ONE detached terminal watcher for a pushed kernel.

        ``run_tag`` binds the canonical live-log reader so the watcher streams
        the run's real console into the deterministic local transcript.
        """
        return self._transport.watcher(
            kind.value, slug=slug, run_tag=run_tag).spawn()

    def launch(self, kind: LayaRunKind, *, stage_dir: Path, run_tag: str,
               execute: bool = False) -> dict[str, Any]:
        """The ONE ``--execute`` lifecycle: publish dataset -> push -> watch.

        Mirrors the fine-tune lane's execute path exactly: the attached inputs
        travel as the dataset BEFORE the push (a stale remote dataset would be
        attached silently otherwise), then ONE detached watcher owns progress,
        download and release. Dry-run stops after the gated publish/push plans.
        """
        result: dict[str, Any] = {
            "publish": self.publish(kind, run_tag=run_tag, execute=execute),
            "push": self.push(Path(stage_dir), execute=execute),
        }
        if not execute:
            return result
        kernel_id = json.loads(
            (Path(stage_dir) / "kernel-metadata.json").read_text(encoding="utf-8")
        )["id"]
        result["watch"] = self.watch(kind, slug=kernel_id, run_tag=run_tag)
        return result

    def track(self, *, run_tag: str | None = None, slug: str | None = None,
              follow: bool = True) -> dict[str, Any]:
        """Real-time tracking: W&B primary, the ONE execution-log reader as fallback."""
        return self._tracker(run_tag=run_tag, slug=slug, follow=follow)

    def harvest(self, kind: str, slug: str | None = None, *,
                execute: bool = False) -> dict[str, Any]:
        """Fetch a finished kernel's output/receipt."""
        return self._transport.collect_kaggle_result(kind, slug, execute=execute)

    def teardown(self, slug: str, *, execute: bool = False,
                 wait: bool = True) -> dict[str, Any]:
        """Stop the running session, THEN delete the kernel (dry-run by default)."""
        stop = self._transport.stop_kaggle_kernel(slug, execute=execute, wait=wait)
        delete = self._transport.delete_kaggle_kernel(slug, execute=execute)
        return {"kernel": slug, "stop": stop, "delete": delete}


class LayaTrainingRunFactory:
    """The ONE assembly site for a :class:`LayaTrainingRun` from the config SSOT."""

    @classmethod
    def build_staging(cls, spec: LayaSpec, runtime: LayaRuntimeFactory,
                      training_config: Callable[[], Any]) -> LayaStagingFactory:
        """Build the staging factory with the lane's boundary lookups injected."""
        from cli import laya_lane

        return LayaStagingFactory(
            spec=spec, runtime=runtime, recipe=LayaRecipeFactory(spec),
            training_cfg=training_config,
            git_revision=laya_lane._git_revision,
            env_value=laya_lane._env_value,
            wandb_project=laya_lane._wandb_project,
            decision_preflight=laya_lane.LAYA_RUNTIME_PREFLIGHT)

    @classmethod
    def from_config(cls, *, spec: LayaSpec | None = None,
                    train_root: Path | None = None,
                    training_config: Callable[[], Any] | None = None,
                    ) -> LayaTrainingRun:
        """Build the owner from the config SSOT (the ONE wiring site)."""
        from cli import laya_hpo
        from cli.kaggle_monitor import KaggleMonitor
        from core.common import TRAIN_ROOT, training_cfg

        training_config = training_config or training_cfg
        root = Path(train_root) if train_root is not None else TRAIN_ROOT
        spec = spec if spec is not None else training_config().laya
        runtime = LayaRuntimeFactory(spec, root)
        return LayaTrainingRun(
            staging=cls.build_staging(spec, runtime, training_config),
            transport=LayaTransportFactory(spec, runtime),
            publisher=LayaPublishFactory(spec, runtime),
            study_resolver=laya_hpo.resolve_study_storage,
            hpo_stager=laya_hpo.LayaHpoStager,
            tracker=KaggleMonitor.track_run)
