"""Laya lane dataset packaging + `--execute`-gated remote publish.

``LayaPublishFactory`` builds the sealed base-model archive and create-or-versions
the attached input datasets. Distinct from staging (which only writes local
payload dirs): every method here either seals an archive or drives the kaggle
dataset CLI behind the ``--execute`` gate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cli.laya_recipe import (
    BASE_MODEL_MANIFEST_FILE,
    DATASET_METADATA_FILE,
    DATASET_PAYLOAD_DIR,
    FINETUNE_CKPT_DECISION,
    FINETUNE_DECISION,
    FINETUNE_EVAL_DECISION,
    FINETUNE_SMOKE_DECISION,
    HOLDOUT_EVAL_DECISION,
)
from cli.laya_runtime import LayaRuntimeFactory
from cli.laya_transport import LayaTransportFactory
from core.bundle import Bundle, BundleRole
from core.laya_config import LayaSpec
from core.manifest import atomic_write_json


class LayaPublishFactory:
    """Packages the base model and publishes the attached input datasets."""

    def __init__(self, spec: LayaSpec, runtime: LayaRuntimeFactory):
        self._spec = spec
        self._runtime = runtime

    def package_base_model(self, *, source_dir: Path, dataset_slug: str,
                           archive_name: str, member_name: str,
                           output_dir: Path | None = None) -> dict[str, Any]:
        """Package the local base checkpoint tree as a sealed `.tar.zst` payload."""
        source_dir = Path(source_dir)
        if not (source_dir / "rl_agent_config.json").is_file():
            raise FileNotFoundError(
                f"base-model source {source_dir} carries no rl_agent_config.json")
        if not dataset_slug:
            raise RuntimeError(
                "config laya.base_model_dataset is unset; name the base-model "
                "dataset (owner/slug) before packaging")
        stage = (Path(output_dir) if output_dir
                 else self._runtime.staging_dir() / "base_model")
        stage.mkdir(parents=True, exist_ok=True)
        archive_path = stage / archive_name
        if archive_path.exists():
            archive_path.unlink()
        files = {f"{member_name}/{path.relative_to(source_dir).as_posix()}": path
                 for path in sorted(source_dir.rglob("*")) if path.is_file()}
        if not files:
            raise FileNotFoundError(
                f"base-model source {source_dir} carries no files to package")
        sealed = Bundle.seal_archive(
            archive_path, files, role=BundleRole.inputs,
            manifest_name=BASE_MODEL_MANIFEST_FILE,
            metadata={"schema": "er-laya-base-model-v1", "member": member_name,
                      "source": str(source_dir)})
        metadata = {"title": "er laya base", "id": dataset_slug,
                    "licenses": [{"name": "other"}]}
        atomic_write_json(metadata, stage / DATASET_METADATA_FILE)
        receipt = {
            "dataset": dataset_slug,
            "payload": str(stage),
            "archive": archive_name,
            "member": member_name,
            "source": str(source_dir),
            "bundle_role": BundleRole.inputs.value,
            "manifest": BASE_MODEL_MANIFEST_FILE,
            "bytes": archive_path.stat().st_size,
            "sha256": sealed.digest,
            "metadata": metadata,
        }
        atomic_write_json(receipt, stage / "base_model.receipt.json")
        self._runtime.log_lane(
            f"packaged base model {dataset_slug} member={member_name} "
            f"archive={archive_name} bytes={receipt['bytes']} -> {stage}")
        return receipt

    def _dataset_target(self, decision_kind: str) -> tuple[str, str]:
        """``(slug, config_key)`` the decision kind's attached dataset resolves to.

        The external-kind registry wins first (a sibling lane whose dataset is
        not a ``LayaSpec`` field of the shared bindings, e.g. HPO attaches the
        fine-tune corpus); otherwise the landed spec bindings apply. Fails loud
        on an unset slug so a dry run and an executed attach agree on the target.
        """
        spec = self._spec
        external_attr = LayaTransportFactory.external_dataset_attr(decision_kind)
        if external_attr:
            slug, key = getattr(spec, external_attr), external_attr
        elif decision_kind == HOLDOUT_EVAL_DECISION:
            slug, key = spec.holdout_dataset_slug, "holdout_dataset_slug"
        elif decision_kind == FINETUNE_CKPT_DECISION:
            slug, key = spec.finetune_ckpt_dataset, "finetune_ckpt_dataset"
        elif decision_kind == FINETUNE_SMOKE_DECISION:
            slug = spec.finetune_smoke.dataset_slug
            key = "finetune_smoke.dataset_slug"
        elif decision_kind in (FINETUNE_DECISION, FINETUNE_EVAL_DECISION):
            slug, key = spec.finetune_dataset_slug, "finetune_dataset_slug"
        else:
            slug, key = spec.dataset_slug, "dataset_slug"
        if not slug:
            raise RuntimeError(
                f"config laya.{key} is unset; name the dataset (owner/slug) "
                "before an attach")
        return slug, key

    def publish_laya_dataset(self, decision_kind: str, *, run_tag: str,
                             execute: bool) -> dict[str, Any]:
        """`--execute`-gated create-or-version of the laya inputs dataset."""
        payload = (self._runtime.staging_dir() / "kaggle" / decision_kind
                   / DATASET_PAYLOAD_DIR)
        metadata_file = payload / DATASET_METADATA_FILE
        slug, _ = self._dataset_target(decision_kind)
        plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                                "payload": str(payload), "slug": slug}
        if not execute:
            plan["note"] = ("the dataset attach rides --execute only "
                            "(mirroring the kernels-push gate)")
            self._runtime.log_lane(
                f"dry-run: dataset {slug} for {decision_kind} would "
                f"publish to the remote surface at {payload}")
            return plan
        if not metadata_file.is_file():
            raise RuntimeError(
                "--activate gate: no staged dataset payload at "
                f"{payload} ({DATASET_METADATA_FILE} is missing); stage first")
        from cli import kaggle_lane as lane
        from cli.kaggle_datasets import KaggleDatasets

        executable = lane._require_kaggle_executable(
            lane._spec().kaggle_executable)
        current = KaggleDatasets._dataset_current_version(slug)
        version = current.get("dataset_version")
        # ONE argv home: the canonical builder owns the create/version shape;
        # the laya lane only names its ``-r zip`` directory mode.
        commands = KaggleDatasets.dataset_publish_commands(
            executable, payload, message=f"laya inputs {run_tag}",
            dir_mode_args=["-r", "zip"])
        plan["action"] = "version" if version else "create"
        command = commands[plan["action"]]
        plan["command"] = command
        _, _ = lane._run_kaggle(command)
        plan["returncode"] = 0
        refreshed = KaggleDatasets._dataset_current_version(slug)
        plan["dataset_version"] = refreshed.get("dataset_version") or version
        plan["published"] = True
        receipt_path = (self._runtime.staging_dir() / "kaggle" / decision_kind
                        / f"{decision_kind}.receipt.json")
        if receipt_path.is_file():
            body = json.loads(receipt_path.read_text(encoding="utf-8"))
            body["dataset"].update({"action": plan["action"],
                                    "version": plan["dataset_version"]})
            atomic_write_json(body, receipt_path)
        self._runtime.log_lane(
            f"published dataset {slug} | action={plan['action']} "
            f"version={plan['dataset_version']} rc=0")
        return plan
