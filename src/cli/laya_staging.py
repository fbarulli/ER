"""Laya lane payload + kernel staging (dry-safe; fail-loud on a missing contract).

The ``LayaStagingFactory`` carries ONE resolved ``LayaSpec`` plus the injected
lane deps (runtime paths, trainer recipe, git revision, kaggle config, .env and
wandb lookups), so every staged surface renders from explicit inputs and no
caller re-reads a global. Per-kind methods are the staging strategies; the
laya lane entry point builds the factory at its boundary.
"""
from __future__ import annotations

import ast
import json
import shutil
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from cli.kaggle_kernel_templates import KernelTemplates
from cli.kaggle_kernels import KaggleKernels
from cli.laya_kernel_text_edge import (
    DECISION_KERNEL_SCRIPT,
    EVAL_KERNEL_SCRIPT,
    HOLDOUT_EVAL_KERNEL_SCRIPT,
    NOTEBOOK_SCRIPT,
)
from cli.laya_kernel_text_finetune import (
    FINETUNE_DEVICE_PATCH_SOURCE,
    FINETUNE_EVAL_KERNEL_SCRIPT,
    FINETUNE_EVAL_RUNTIME_PREFLIGHT,
    FINETUNE_KERNEL_SCRIPT,
    FINETUNE_RUNTIME_PREFLIGHT,
)
from cli.laya_recipe import (
    COLAB_NOTEBOOK_NAME,
    DATASET_CSV_NAME,
    DATASET_METADATA_FILE,
    DATASET_PAYLOAD_DIR,
    DECISION_BINDINGS,
    DECISION_KERNEL_CODE_FILE,
    EVAL_KERNEL_CODE_FILE,
    FINETUNE_CODE_FILE,
    FINETUNE_CORPUS_FILES,
    FINETUNE_CORPUS_RECEIPT,
    FINETUNE_DECISION,
    FINETUNE_EVAL_CODE_FILE,
    FINETUNE_EVAL_DECISION,
    FINETUNE_EVAL_SPLIT_FILES,
    FINETUNE_SMOKE_DECISION,
    HOLDOUT_EVAL_CODE_FILE,
    HOLDOUT_EVAL_DECISION,
    HOLDOUT_JSONL,
    QUESTION_SCHEMA_FILE,
    LayaRecipeFactory,
)
from cli.laya_runtime import _METRIC_EXPECTATION_KEYS, LayaRuntimeFactory
from cli.laya_train_patch_loop import FINETUNE_PERF_PATCH_SOURCE
from core.laya_config import LayaSpec
from core.manifest import atomic_write_json, sha256_file


@lru_cache(maxsize=8)
def _parse(script: str) -> ast.Module:
    return ast.parse(script)


class LayaStagingFactory:
    """Stages every laya payload/kernel surface from injected deps."""

    def __init__(self, *, spec: LayaSpec, runtime: LayaRuntimeFactory,
                 recipe: LayaRecipeFactory,
                 training_cfg: Callable[[], Any],
                 git_revision: Callable[[], str],
                 env_value: Callable[[str], str | None],
                 wandb_project: Callable[[], str],
                 decision_preflight: str):
        self._spec = spec
        self._runtime = runtime
        self._recipe = recipe
        self._training_cfg = training_cfg
        self._git_revision = git_revision
        self._env_value = env_value
        self._wandb_project = wandb_project
        self._decision_preflight = decision_preflight

    # ── shared rendering ───────────────────────────────────────────────────
    @staticmethod
    def template(script: str, values: dict[str, str]) -> str:
        return KernelTemplates.substitute(script, values)

    @staticmethod
    def kernel_script_gate(script: str) -> None:
        KaggleKernels._kernel_script_gate(script)

    @staticmethod
    def module_scope_gate(script: str) -> None:
        import builtins

        tree = _parse(script)
        compile(tree, "<laya-payload>", "exec")
        bound = set(dir(builtins))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Import):
                bound.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                bound.update(a.asname or a.name for a in node.names
                             if a.name != "*")
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
        loaded = {node.id
                  for stmt in tree.body
                  if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                           ast.ClassDef))
                  for node in ast.walk(stmt)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                  and not (node.id.startswith("__") and node.id.endswith("__"))}
        undeclared = sorted(loaded - bound)
        if undeclared:
            raise ValueError(f"staged kernel loads undefined top-level names: "
                             f"{undeclared}; a NameError-class payload must "
                             "never stage again")

    @staticmethod
    def decision_tag() -> str:
        """UTC-stamped run tag (the payload stamp stays UTC: the remote session
        may not share this box's zone; the console/lane.log stamp stays
        Europe/Paris per the landed kaggle_lane convention)."""
        return datetime.now(ZoneInfo("UTC")).strftime("%m%dT%H%M%SZ")

    def _finetune_smoke_recipe(self) -> dict[str, Any]:
        """The smoke's ``TrainConfig`` kwargs: the landed recipe with the smoke
        dials overlaid from ``laya.finetune_smoke`` (SSOT), never a literal."""
        smoke = self._spec.finetune_smoke
        recipe = self._recipe.finetune_config()
        recipe.update({"epochs": smoke.epochs, "micro_batch": smoke.micro_batch,
                       "grad_accum": smoke.grad_accum})
        return recipe

    # ── decision-input staging (dry-safe; fail-loud on a missing contract) ──
    def stage_decision_input(self, kind: str, *, decision_kind: str,
                             override: Path | None = None) -> dict[str, Any]:
        """Stage ONE decision CSV under results/laya_lane/<kind>/<decision>/."""
        if decision_kind not in DECISION_BINDINGS:
            raise ValueError(f"unknown decision kind: {decision_kind!r}")
        from core.common import F

        entry = DECISION_BINDINGS[decision_kind]
        source = (Path(override) if override
                  else F[self._recipe.decision_binding(decision_kind)])
        stage = self._runtime.staging_dir() / kind / decision_kind
        stage.mkdir(parents=True, exist_ok=True)
        census = self._runtime.census_csv(source, entry["wanted_columns"])
        destination = stage / source.name
        shutil.copy2(source, destination)
        receipt = {
            "kind": kind, "decision_kind": decision_kind,
            "binding": self._recipe.decision_binding(decision_kind),
            "source": str(source), "staged": str(destination),
            "rows": census["rows"], "columns": census["columns"],
            "sha256": census["sha256"], "bytes": census["bytes"],
            "description": entry["description"],
            **census["expectation"],
        }
        atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
        self._runtime.log_lane(
            f"staged decision input [{kind}/{decision_kind}] "
            f"{source.name} rows={census['rows']} "
            f"sha256={census['sha256'][:12]} -> {destination}")
        return receipt

    def stage_question_schema(self, kind: str, *,
                              override: Path | None = None) -> dict[str, Any]:
        """Stage the laya.question schema under results/laya_lane/<kind>/."""
        spec = self._spec
        source = (Path(override) if override
                  else self._runtime.train_root / spec.question_schema)
        stage = self._runtime.staging_dir() / kind / "question"
        stage.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            raise FileNotFoundError(f"laya.question schema not found: {source}")
        schema = json.loads(source.read_text(encoding="utf-8"))
        questions = schema.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError(
                f"laya.question schema at {source} carries no 'questions' dict")
        destination = stage / QUESTION_SCHEMA_FILE
        shutil.copy2(source, destination)
        receipt = {
            "question_schema": spec.question_schema,
            "staged": str(destination),
            "questions": sorted(questions),
            "sha256": sha256_file(source),
        }
        atomic_write_json(receipt, stage / "question.receipt.json")
        self._runtime.log_lane(
            f"staged question schema [{kind}] {source.name} "
            f"({len(questions)} questions) -> {destination}")
        return receipt

    # ── dataset payloads ───────────────────────────────────────────────────
    def stage_dataset_payload(self, decision_kind: str, *, dataset_slug: str,
                              question_source: Path,
                              decision_source: Path) -> dict[str, Any]:
        """Stage the DATASET payload for spec.dataset_slug (dry-safe)."""
        if not dataset_slug:
            raise RuntimeError(
                "config laya.dataset_slug is unset; name the input dataset "
                "(owner/slug) before staging")
        stage = (self._runtime.staging_dir() / "kaggle" / decision_kind
                 / DATASET_PAYLOAD_DIR)
        stage.mkdir(parents=True, exist_ok=True)
        metadata = {"title": "er laya requests", "id": dataset_slug,
                    "licenses": [{"name": "other"}]}
        atomic_write_json(metadata, stage / DATASET_METADATA_FILE)
        shutil.copy2(question_source, stage / QUESTION_SCHEMA_FILE)
        shutil.copy2(decision_source, stage / DATASET_CSV_NAME)
        payload_files = (QUESTION_SCHEMA_FILE, DATASET_CSV_NAME)
        receipt = {
            "dataset": dataset_slug,
            "payload": str(stage),
            "metadata": metadata,
            "files": {name: sha256_file(stage / name) for name in payload_files},
        }
        atomic_write_json(receipt, stage / "dataset_payload.receipt.json")
        self._runtime.log_lane(
            f"staged dataset payload [{decision_kind}] {dataset_slug} "
            f"files={list(payload_files)} -> {stage}")
        return receipt

    # ── decision kernel ────────────────────────────────────────────────────
    def stage_decision_kernel(self, *, decision_kind: str,
                              revision: str | None = None,
                              run_tag: str | None = None,
                              input_override: Path | None = None,
                              checkpoint_path: Path | None = None
                              ) -> dict[str, Any]:
        """Stage the kaggle decision kernel payload (dry-safe)."""
        spec = self._spec
        if decision_kind not in DECISION_BINDINGS:
            raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                             f"expected {list(DECISION_BINDINGS)}")
        if decision_kind == FINETUNE_DECISION:
            return self.stage_finetune_kernel(revision=revision, run_tag=run_tag)
        if decision_kind == FINETUNE_SMOKE_DECISION:
            return self.stage_finetune_kernel(revision=revision, run_tag=run_tag,
                                              smoke=True)
        if decision_kind == FINETUNE_EVAL_DECISION:
            return self.stage_finetune_eval_kernel(
                revision=revision, run_tag=run_tag,
                checkpoint_path=checkpoint_path)
        if spec.laya_decision_epochs <= 0:
            raise RuntimeError(
                "config laya.laya_decision_epochs <= 0: the decision lane is "
                "disabled (no payload may stage a GPU session)")
        slug = spec.export_dataset_slug
        if not slug:
            raise RuntimeError(
                "config laya.export_dataset_slug is unset; name the target "
                "kernel (owner/slug) before staging")
        dataset_slug = spec.dataset_slug
        if not dataset_slug:
            raise RuntimeError(
                "config laya.dataset_slug is unset; the kernel inputs travel "
                "as that dataset (owner/slug) — the staged play_500.csv + "
                "laya.question.json do NOT ride `kaggle kernels push`; "
                "name it before staging")
        # ── the published-tip invariant ('origin/<branch> == HEAD'): the pin
        # resolves BEFORE any payload write.
        repository = self._training_cfg().kaggle.repository
        branch = self._training_cfg().kaggle.branch
        revision = revision or self._git_revision()
        from core import runtime_inputs
        tip = runtime_inputs.require_published_tip_match(
            revision, repository, branch)
        question = self.stage_question_schema("kaggle")
        input_receipt = self.stage_decision_input(
            "kaggle", decision_kind=decision_kind, override=input_override)
        dataset_receipt = self.stage_dataset_payload(
            decision_kind, dataset_slug=dataset_slug,
            question_source=Path(question["staged"]),
            decision_source=Path(input_receipt["staged"]))
        stage = self._runtime.staging_dir() / "kaggle" / decision_kind
        stage.mkdir(parents=True, exist_ok=True)
        code_file = (DECISION_KERNEL_CODE_FILE if decision_kind != "laya-cli-eval"
                     else EVAL_KERNEL_CODE_FILE)
        template = (DECISION_KERNEL_SCRIPT if decision_kind != "laya-cli-eval"
                    else EVAL_KERNEL_SCRIPT)
        tag = run_tag or spec.run_tag_prefix + self.decision_tag()
        # single T4: the payload never requests the double accelerator; the
        # script itself pins CUDA_VISIBLE_DEVICES=0. THE INPUTS TRAVEL AS THE
        # DATASET: kernels push does NOT ship the co-located csv/schema files,
        # so resolve_input would FileNotFoundError once boot passes.
        metadata = KaggleKernels.kernel_metadata(
            slug, code_file, enable_gpu=True, dataset_sources=[dataset_slug])
        entry = DECISION_BINDINGS[decision_kind]
        staged_csv = input_receipt["staged"]
        values = {
            "LAYA_PACKAGE": spec.laya_package,
            "CHECKPOINT_HUB": spec.checkpoint_hub,
            "DECISION_KIND": decision_kind,
            "RUN_TAG": tag,
            "DECISION_CSV": DATASET_CSV_NAME,
            "STATE_COLUMN": entry["state_column"],
            "BATCH_SIZE": str(spec.laya_decision_batch_size),
            "MIN_CONFIDENCE": repr(spec.min_router_confidence),
            "QUESTION_SCHEMA_FILE": repr(QUESTION_SCHEMA_FILE),
            "REPOSITORY": repository,
            "BRANCH": branch,
            "REVISION": revision,
        }
        # two-pass substitution (a nested value's @tokens@ are never
        # re-scanned once it is inserted): the preflight bakes its own
        # literal tuple FIRST, then drops into the script — the push gate
        # (_staged_laya_push_preflight) literal-evals `_runtime_files`.
        preflight = self.template(self._decision_preflight, values)
        script = self.template(template, {**values,
                                          "RUNTIME_PREFLIGHT": preflight})
        self.kernel_script_gate(script)
        self.module_scope_gate(script)
        atomic_write_json(metadata, stage / "kernel-metadata.json")
        (stage / code_file).write_text(script, encoding="utf-8")
        receipt = {
            "kernel": slug,
            "kind": decision_kind,
            "gpu": "T4 (single)",
            "run_tag": tag,
            "staged": str(stage),
            "code_file": code_file,
            "question_schema": question["staged"],
            "question_sha256": question["sha256"],
            "decision_input": staged_csv,
            "decision_sha256": input_receipt["sha256"],
            "dataset": {"slug": dataset_slug,
                        "payload": dataset_receipt["payload"],
                        "files": dataset_receipt["files"]},
            "checkpoint_hub": spec.checkpoint_hub,
            "batch_size": spec.laya_decision_batch_size,
            "min_confidence": spec.min_router_confidence,
            "epochs": spec.laya_decision_epochs,
            "state_column": entry["state_column"],
            "evals_enabled": bool(spec.laya_evals_enabled),
            "calibration": bool(spec.calibration),
            "onnx": bool(spec.onnx),
            "published_pin": {"repository": repository, "branch": branch,
                              "revision": revision},
            "published_tip": tip,
        }
        # the labeled decision csv's metric contract rides the staged receipt
        receipt.update({key: input_receipt[key]
                        for key in _METRIC_EXPECTATION_KEYS
                        if key in input_receipt})
        atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
        shutil.copy2(question["staged"], stage / QUESTION_SCHEMA_FILE)
        self._runtime.log_lane(
            f"staged kaggle kernel [{decision_kind}] ({spec.gpu}) "
            f"run_tag={tag} -> {stage}")
        return receipt

    # ── corpus dataset + finetune kernels ──────────────────────────────────
    def stage_finetune_dataset_payload(self, *, dataset_slug: str,
                                       corpus_dir: Path,
                                       kind: str = FINETUNE_DECISION,
                                       title: str = "er laya train"
                                       ) -> dict[str, Any]:
        """Stage the fine-tune CORPUS as a kaggle dataset payload (dry-safe)."""
        if not dataset_slug:
            raise RuntimeError(
                "config laya.finetune_dataset_slug is unset; name the corpus "
                "dataset (owner/slug) before staging")
        corpus_dir = Path(corpus_dir)
        stage = self._runtime.staging_dir() / "kaggle" / kind / DATASET_PAYLOAD_DIR
        # Boundary (deliberately NOT a `Bundle`): a kaggle dataset DIRECTORY.
        stage.mkdir(parents=True, exist_ok=True)
        metadata = {"title": title, "id": dataset_slug,
                    "licenses": [{"name": "other"}]}
        atomic_write_json(metadata, stage / DATASET_METADATA_FILE)
        files = list(FINETUNE_CORPUS_FILES) + [FINETUNE_CORPUS_RECEIPT]
        for name in files:
            source = corpus_dir / name
            if not source.is_file():
                raise FileNotFoundError(
                    f"fine-tune corpus file not found: {source} (build it with "
                    "scripts/laya_build_dataset.py)")
            shutil.copy2(source, stage / name)
        receipt = {
            "dataset": dataset_slug,
            "payload": str(stage),
            "metadata": metadata,
            "files": {name: sha256_file(stage / name) for name in files},
        }
        atomic_write_json(receipt, stage / "dataset_payload.receipt.json")
        self._runtime.log_lane(
            f"staged finetune dataset payload {dataset_slug} "
            f"files={files} -> {stage}")
        return receipt

    def _pairs_composer(self):
        """The corpus pair composer (scripts/laya_metrics_pairs.py; reused)."""
        import importlib.util

        path = self._runtime.train_root / "scripts/laya_metrics_pairs.py"
        spec = importlib.util.spec_from_file_location("laya_metrics_pairs", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _csv_rows(path: Path) -> list[dict]:
        import csv

        with Path(path).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _norm_gtin(value: object) -> str:
        from training.folds import normalize_gtin

        return normalize_gtin(value)

    def stage_holdout_dataset_payload(self, *, dataset_slug: str,
                                      holdout_csv: Path, catalog_path: Path,
                                      question_path: Path) -> dict[str, Any]:
        """Stage the component-disjoint holdout as a kaggle dataset payload."""
        if not dataset_slug:
            raise RuntimeError(
                "config laya.holdout_dataset_slug is unset; name the holdout "
                "dataset (owner/slug) before staging")
        questions = json.loads(
            Path(question_path).read_text(encoding="utf-8"))["questions"]
        composer = self._pairs_composer()
        rows = self._csv_rows(Path(holdout_csv))
        by_gtin: dict[str, dict] = {}
        for row in self._csv_rows(Path(catalog_path)):
            by_gtin.setdefault(self._norm_gtin(row.get("gtin")), row)
        lines, skipped = [], 0
        for row in rows:
            label = str(row.get("label", "")).strip()
            one = by_gtin.get(self._norm_gtin(row.get("gtin1")))
            two = by_gtin.get(self._norm_gtin(row.get("gtin2")))
            if label not in ("0", "1") or one is None or two is None:
                skipped += 1
                continue
            state = composer.compose_state(
                composer.compose_side(one["attribute"]),
                composer.compose_side(two["attribute"]))
            lines.append({
                "state": state, "questions": questions,
                "expected": {"identity_claim": "true" if label == "1" else "false"},
                "stratum": str(row.get("stratum", "")),
                "component": str(row.get("component", "")),
            })
        if not lines:
            raise RuntimeError(
                f"holdout {holdout_csv} produced no labelled rows; build it with "
                "scripts/laya_holdout.py first")
        stage = (self._runtime.staging_dir() / "kaggle" / HOLDOUT_EVAL_DECISION
                 / DATASET_PAYLOAD_DIR)
        stage.mkdir(parents=True, exist_ok=True)
        metadata = {"title": "er laya holdout", "id": dataset_slug,
                    "licenses": [{"name": "other"}]}
        atomic_write_json(metadata, stage / DATASET_METADATA_FILE)
        body = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)
        (stage / HOLDOUT_JSONL).write_text(body, encoding="utf-8")
        receipt = {
            "dataset": dataset_slug, "payload": str(stage), "rows": len(lines),
            "skipped": skipped, "question_schema": str(question_path),
            "files": {HOLDOUT_JSONL: sha256_file(stage / HOLDOUT_JSONL)},
        }
        atomic_write_json(receipt, stage / "dataset_payload.receipt.json")
        self._runtime.log_lane(
            f"staged holdout dataset payload {dataset_slug} "
            f"rows={len(lines)} (skipped {skipped}) -> {stage}")
        return receipt

    def stage_holdout_eval_kernel(self, *, revision: str | None = None,
                                  run_tag: str | None = None) -> dict[str, Any]:
        """Stage the holdout-verification kaggle kernel (dry-safe)."""
        spec = self._spec
        slug = spec.holdout_eval_kernel_slug
        if not slug:
            raise RuntimeError(
                "config laya.holdout_eval_kernel_slug is unset; name the target "
                "kernel (owner/slug) before staging")
        dataset_slug = spec.holdout_dataset_slug
        if not dataset_slug:
            raise RuntimeError(
                "config laya.holdout_dataset_slug is unset; the holdout travels as "
                "that dataset (owner/slug) — name it before staging")
        ckpt_dataset = spec.finetune_ckpt_dataset
        repository = self._training_cfg().kaggle.repository
        branch = self._training_cfg().kaggle.branch
        revision = revision or self._git_revision()
        from core import runtime_inputs

        tip = runtime_inputs.require_published_tip_match(
            revision, repository, branch)
        question_path = self._runtime.train_root / spec.question_schema
        dataset_receipt = self.stage_holdout_dataset_payload(
            dataset_slug=dataset_slug,
            holdout_csv=self._runtime.train_root / spec.holdout_csv,
            catalog_path=self._runtime.train_root
            / "data/track_setup/eligible_catalog.csv",
            question_path=question_path)
        stage = self._runtime.staging_dir() / "kaggle" / HOLDOUT_EVAL_DECISION
        stage.mkdir(parents=True, exist_ok=True)
        tag = run_tag or spec.run_tag_prefix + self.decision_tag()
        dataset_sources = [dataset_slug]
        if ckpt_dataset:
            dataset_sources.append(ckpt_dataset)
        metadata = KaggleKernels.kernel_metadata(
            slug, HOLDOUT_EVAL_CODE_FILE, enable_gpu=True,
            dataset_sources=dataset_sources)
        values = {
            "LAYA_PACKAGE": spec.finetune_package,
            "RUN_TAG": tag,
            "HOLDOUT_JSONL": HOLDOUT_JSONL,
            "CKPT_DIR": spec.finetune_ckpt_dir,
            "BATCH_SIZE": str(spec.holdout_eval_batch_size),
            "THRESHOLD": repr(spec.holdout_eval_threshold),
            "N_BOOT": str(spec.holdout_eval_bootstrap),
            "SEED": "1729",
            "REPOSITORY": repository,
            "BRANCH": branch,
            "REVISION": revision,
        }
        preflight = self.template(self._decision_preflight, {
            **values, "DECISION_CSV": HOLDOUT_JSONL,
            "QUESTION_SCHEMA_FILE": repr(QUESTION_SCHEMA_FILE)})
        script = self.template(HOLDOUT_EVAL_KERNEL_SCRIPT,
                               {**values, "RUNTIME_PREFLIGHT": preflight})
        self.kernel_script_gate(script)
        self.module_scope_gate(script)
        atomic_write_json(metadata, stage / "kernel-metadata.json")
        (stage / HOLDOUT_EVAL_CODE_FILE).write_text(script, encoding="utf-8")
        receipt = {
            "kernel": slug,
            "kind": HOLDOUT_EVAL_DECISION,
            "gpu": "T4 (single)",
            "run_tag": tag,
            "staged": str(stage),
            "code_file": HOLDOUT_EVAL_CODE_FILE,
            "dataset": {"slug": dataset_slug,
                        "payload": dataset_receipt["payload"],
                        "files": dataset_receipt["files"]},
            "checkpoint_dataset": ckpt_dataset,
            "threshold": spec.holdout_eval_threshold,
            "n_boot": spec.holdout_eval_bootstrap,
            "published_pin": {"repository": repository, "branch": branch,
                              "revision": revision},
            "published_tip": tip,
        }
        atomic_write_json(receipt, stage / f"{HOLDOUT_EVAL_DECISION}.receipt.json")
        self._runtime.log_lane(
            f"staged kaggle kernel [{HOLDOUT_EVAL_DECISION}] ({spec.gpu}) "
            f"run_tag={tag} -> {stage}")
        return receipt

    def stage_finetune_kernel(self, *, revision: str | None = None,
                              run_tag: str | None = None,
                              smoke: bool = False) -> dict[str, Any]:
        """Stage the kaggle fine-tune kernel payload (dry-safe)."""
        spec = self._spec
        if spec.laya_decision_epochs <= 0:
            raise RuntimeError(
                "config laya.laya_decision_epochs <= 0: the laya lane is "
                "disabled (no payload may stage a GPU session)")
        smoke_spec = spec.finetune_smoke if smoke else None
        kind = FINETUNE_SMOKE_DECISION if smoke else FINETUNE_DECISION
        slug = smoke_spec.kernel_slug if smoke else spec.finetune_kernel_slug
        if not slug:
            which = "laya.finetune_smoke.kernel_slug" if smoke \
                else "laya.finetune_kernel_slug"
            raise RuntimeError(
                f"config {which} is unset; name the target kernel (owner/slug) "
                "before staging")
        dataset_slug = (smoke_spec.dataset_slug if smoke
                        else spec.finetune_dataset_slug)
        if not dataset_slug:
            which = "laya.finetune_smoke.dataset_slug" if smoke \
                else "laya.finetune_dataset_slug"
            raise RuntimeError(
                f"config {which} is unset; the corpus travels as that dataset "
                "(owner/slug) — name it before staging")
        base_dataset = spec.base_model_dataset
        if not base_dataset:
            raise RuntimeError(
                "config laya.base_model_dataset is unset; the base checkpoint "
                "travels as that dataset (owner/slug) — the finetune kernel "
                "must extract a LOCAL dir, never fetch from the Hub")
        repository = self._training_cfg().kaggle.repository
        branch = self._training_cfg().kaggle.branch
        revision = revision or self._git_revision()
        from core import runtime_inputs
        tip = runtime_inputs.require_published_tip_match(
            revision, repository, branch)
        corpus_dir = self._runtime.train_root / spec.finetune_corpus_dir
        if smoke:
            from cli.laya_smoke import FinetuneSmokeCorpus

            corpus_dir = self._runtime.train_root / smoke_spec.corpus_dir
            FinetuneSmokeCorpus(smoke_spec,
                                source_dir=self._runtime.train_root
                                / spec.finetune_corpus_dir,
                                dest_dir=corpus_dir).build(
                seed=spec.finetune.seed)
        dataset_receipt = self.stage_finetune_dataset_payload(
            dataset_slug=dataset_slug, corpus_dir=corpus_dir, kind=kind,
            title="er laya train smoke" if smoke else "er laya train")
        stage = self._runtime.staging_dir() / "kaggle" / kind
        stage.mkdir(parents=True, exist_ok=True)
        tag = run_tag or spec.run_tag_prefix + self.decision_tag()
        # ONE device source: the smoke's `laya.finetune_smoke.device` (default
        # "cpu" = the original CPU smoke) or the prod `laya.finetune.device`.
        device = smoke_spec.device if smoke else spec.finetune.device
        # enable_gpu derives from the SAME device: a cpu smoke never requests a
        # GPU; prod keeps its single T4 (the script pins CUDA_VISIBLE_DEVICES=0
        # either way). THE CORPUS + THE BASE CHECKPOINT TRAVEL AS DATASETS.
        metadata = KaggleKernels.kernel_metadata(
            slug, FINETUNE_CODE_FILE, enable_gpu=(not smoke or device != "cpu"),
            dataset_sources=[dataset_slug, base_dataset])
        recipe = self._finetune_smoke_recipe() if smoke \
            else self._recipe.finetune_config()
        values = {
            "LAYA_PACKAGE": spec.finetune_package,
            "BASE_MODEL_ARCHIVE": spec.base_model_archive,
            "BASE_MODEL_DIR": spec.base_model_dir,
            "RUN_TAG": tag,
            "TRAIN_JSONL": FINETUNE_CORPUS_FILES[0],
            "DEV_JSONL": FINETUNE_CORPUS_FILES[1],
            "TEST_JSONL": FINETUNE_CORPUS_FILES[2],
            # The FULL TrainConfig surface rides one repr-baked Python literal.
            "FINETUNE_CONFIG": repr(recipe),
            # The training controls ride a SEPARATE repr literal (TrainConfig
            # rejects unknown kwargs): the perf patch reads this global.
            "FINETUNE_CONTROL": repr(self._recipe.finetune_control()),
            "FINETUNE_DEVICE": device,
            "HELD_OUT_BATCH": str(spec.laya_decision_batch_size),
            "RECEIPT_NAME": f"laya_{kind}.receipt.json",
            # wandb mirror: the key is read from .env at staging and baked in
            # (never committed); empty key -> the kernel logs nothing.
            "WANDB_API_KEY": self._env_value("WANDB_API_KEY") or "",
            "WANDB_PROJECT": self._wandb_project(),
            "REPOSITORY": repository,
            "BRANCH": branch,
            "REVISION": revision,
            "DEVICE_PATCH": FINETUNE_DEVICE_PATCH_SOURCE,
            "PERF_PATCH": FINETUNE_PERF_PATCH_SOURCE,
        }
        # two-pass substitution (the preflight bakes its own literal tuple first).
        preflight = self.template(FINETUNE_RUNTIME_PREFLIGHT, values)
        script = self.template(FINETUNE_KERNEL_SCRIPT,
                               {**values, "RUNTIME_PREFLIGHT": preflight})
        self.kernel_script_gate(script)
        self.module_scope_gate(script)
        atomic_write_json(metadata, stage / "kernel-metadata.json")
        (stage / FINETUNE_CODE_FILE).write_text(script, encoding="utf-8")
        receipt = {
            "kernel": slug,
            "kind": kind,
            "gpu": "CPU (smoke)" if smoke and device == "cpu" else "T4 (single)",
            "run_tag": tag,
            "staged": str(stage),
            "code_file": FINETUNE_CODE_FILE,
            "dataset": {"slug": dataset_slug,
                        "payload": dataset_receipt["payload"],
                        "files": dataset_receipt["files"]},
            "laya_package": spec.finetune_package,
            # The base checkpoint is the attached er-laya-base dataset archive,
            # extracted in-kernel; the Hub id is NOT passed as --base anymore.
            "base_model": {"dataset": base_dataset,
                           "archive": spec.base_model_archive,
                           "dir": spec.base_model_dir},
            "recipe": recipe,
            "control": self._recipe.finetune_control(),
            "device": device,
            "smoke": smoke,
            "corpus_dir": str(corpus_dir),
            "published_pin": {"repository": repository, "branch": branch,
                              "revision": revision},
            "published_tip": tip,
        }
        atomic_write_json(receipt, stage / f"{kind}.receipt.json")
        self._runtime.log_lane(
            f"staged kaggle finetune kernel ({receipt['gpu']}) run_tag={tag} "
            f"-> {stage}")
        return receipt

    def stage_finetune_eval_kernel(self, *, revision: str | None = None,
                                   run_tag: str | None = None,
                                   checkpoint_path: Path | None = None
                                   ) -> dict[str, Any]:
        """Stage the kaggle fine-tune EVAL-ONLY kernel payload (dry-safe)."""
        spec = self._spec
        if spec.laya_decision_epochs <= 0:
            raise RuntimeError(
                "config laya.laya_decision_epochs <= 0: the laya lane is "
                "disabled (no payload may stage a GPU session)")
        slug = spec.finetune_eval_kernel_slug
        if not slug:
            raise RuntimeError(
                "config laya.finetune_eval_kernel_slug is unset; name the target "
                "eval kernel (owner/slug) before staging")
        dataset_slug = spec.finetune_dataset_slug
        if not dataset_slug:
            raise RuntimeError(
                "config laya.finetune_dataset_slug is unset; the corpus travels "
                "as that dataset (owner/slug) — name it before staging")
        ckpt_dataset = spec.finetune_ckpt_dataset
        if not ckpt_dataset and not checkpoint_path:
            raise RuntimeError(
                "config laya.finetune_ckpt_dataset is unset and no checkpoint "
                "path was given; the eval-only kernel needs a fine-tuned "
                "checkpoint dataset (owner/slug) or an explicit path")
        split = spec.finetune_eval_split
        if split not in FINETUNE_EVAL_SPLIT_FILES:
            raise ValueError(
                f"config laya.finetune_eval_split {split!r} is not one of "
                f"{sorted(FINETUNE_EVAL_SPLIT_FILES)}")
        repository = self._training_cfg().kaggle.repository
        branch = self._training_cfg().kaggle.branch
        revision = revision or self._git_revision()
        from core import runtime_inputs
        tip = runtime_inputs.require_published_tip_match(
            revision, repository, branch)
        dataset_receipt = self.stage_finetune_dataset_payload(
            dataset_slug=dataset_slug,
            corpus_dir=self._runtime.train_root / spec.finetune_corpus_dir,
            kind=FINETUNE_EVAL_DECISION)
        stage = self._runtime.staging_dir() / "kaggle" / FINETUNE_EVAL_DECISION
        stage.mkdir(parents=True, exist_ok=True)
        tag = run_tag or spec.run_tag_prefix + self.decision_tag()
        dataset_sources = [dataset_slug]
        if ckpt_dataset and not checkpoint_path:
            dataset_sources.append(ckpt_dataset)
        # THE HELD-OUT SPLIT + THE CHECKPOINT TRAVEL AS DATASETS.
        metadata = KaggleKernels.kernel_metadata(
            slug, FINETUNE_EVAL_CODE_FILE, enable_gpu=True,
            dataset_sources=dataset_sources)
        eval_jsonl = FINETUNE_EVAL_SPLIT_FILES[split]
        calibration = self._recipe.eval_calibration_config()
        values = {
            "LAYA_PACKAGE": spec.finetune_package,
            "RUN_TAG": tag,
            "EVAL_JSONL": eval_jsonl,
            "EVAL_SPLIT": split,
            "CKPT_DIR": spec.finetune_ckpt_dir,
            "CHECKPOINT_PATH": str(checkpoint_path) if checkpoint_path else "",
            "BATCH_SIZE": str(spec.finetune_eval_batch_size),
            # The eval path's calibration selection rides one repr literal.
            "EVAL_CALIBRATION": repr(calibration),
            "REPOSITORY": repository,
            "BRANCH": branch,
            "REVISION": revision,
        }
        preflight = self.template(FINETUNE_EVAL_RUNTIME_PREFLIGHT, values)
        script = self.template(FINETUNE_EVAL_KERNEL_SCRIPT,
                               {**values, "RUNTIME_PREFLIGHT": preflight})
        self.kernel_script_gate(script)
        self.module_scope_gate(script)
        atomic_write_json(metadata, stage / "kernel-metadata.json")
        (stage / FINETUNE_EVAL_CODE_FILE).write_text(script, encoding="utf-8")
        receipt = {
            "kernel": slug,
            "kind": FINETUNE_EVAL_DECISION,
            "gpu": "T4 (single)",
            "run_tag": tag,
            "staged": str(stage),
            "code_file": FINETUNE_EVAL_CODE_FILE,
            "dataset": {"slug": dataset_slug,
                        "payload": dataset_receipt["payload"],
                        "files": dataset_receipt["files"]},
            "checkpoint_dataset": ckpt_dataset,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_dir_hint": spec.finetune_ckpt_dir,
            "eval_split": split,
            "eval_jsonl": eval_jsonl,
            "eval_calibration": calibration,
            "laya_package": spec.finetune_package,
            "published_pin": {"repository": repository, "branch": branch,
                              "revision": revision},
            "published_tip": tip,
        }
        atomic_write_json(receipt,
                          stage / f"{FINETUNE_EVAL_DECISION}.receipt.json")
        self._runtime.log_lane(
            f"staged kaggle finetune-eval kernel ({spec.gpu}) "
            f"split={split} ckpt={ckpt_dataset or checkpoint_path} "
            f"run_tag={tag} -> {stage}")
        return receipt

    def stage_colab_notebook(self, *, decision_kind: str,
                             run_tag: str | None = None) -> dict[str, Any]:
        """Colab notebook payload in the receipts style (dry-safe)."""
        spec = self._spec
        if decision_kind not in DECISION_BINDINGS:
            raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                             f"expected {list(DECISION_BINDINGS)}")
        if spec.laya_decision_epochs <= 0:
            raise RuntimeError(
                "config laya.laya_decision_epochs <= 0: the decision lane is "
                "disabled (no payload may stage a session)")
        if not spec.export_dataset_slug and not spec.dataset_slug:
            raise RuntimeError(
                "config laya.export_dataset_slug / dataset_slug both unset; "
                "name the target surface (owner/slug) before staging")
        question = self.stage_question_schema("colab")
        input_receipt = self.stage_decision_input("colab",
                                                  decision_kind=decision_kind)
        stage = self._runtime.staging_dir() / "colab" / decision_kind
        stage.mkdir(parents=True, exist_ok=True)
        tag = run_tag or spec.run_tag_prefix + self.decision_tag()
        entry = DECISION_BINDINGS[decision_kind]
        staged_csv = input_receipt["staged"]
        script = self.template(NOTEBOOK_SCRIPT, {
            "LAYA_PACKAGE": spec.laya_package,
            "DECISION_KIND": decision_kind,
            "RUN_TAG": tag,
            "DECISION_CSV": Path(staged_csv).name,
            "STATE_COLUMN": entry["state_column"],
        })
        self.kernel_script_gate(script)
        self.module_scope_gate(script)
        notebook = stage / COLAB_NOTEBOOK_NAME
        notebook.write_text(script, encoding="utf-8")
        receipt = {
            "kernel": COLAB_NOTEBOOK_NAME,
            "kind": decision_kind,
            "gpu": "T4 (single)",
            "run_tag": tag,
            "staged": str(stage),
            "notebook": str(notebook),
            "receipt_style": "results/laya_lane/<kind>/<op>/...",
            "question_schema": question["staged"],
            "decision_input": staged_csv,
            "epochs": spec.laya_decision_epochs,
            "note": "the colab CLI surface is unchanged; delivery contract only",
        }
        atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
        self._runtime.log_lane(
            f"staged colab notebook payload [{decision_kind}] "
            f"run_tag={tag} -> {notebook}")
        return receipt


class LayaPayloadPreflight:
    """The laya push gate over a staged payload's ATTACHED-inputs inventory."""

    @staticmethod
    def verify(stage_dir: Path) -> None:
        """Verify the staged dataset payload against the kernel's inventory and
        the publish pin; the inventory rides the dataset, never the checkout."""
        from core.runtime_inputs import remote_revision_preflight

        metadata = json.loads((stage_dir / "kernel-metadata.json").read_text())
        script = (stage_dir / metadata["code_file"]).read_text()
        values = {}
        for node in ast.walk(ast.parse(script)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {
                            "REPOSITORY", "BRANCH", "REVISION", "_runtime_files"}:
                        values[target.id] = ast.literal_eval(node.value)
        required = {"REPOSITORY", "BRANCH", "REVISION", "_runtime_files"}
        if required - values.keys():
            raise ValueError(
                "Staged kernel lacks runtime preflight inventory; regenerate it")
        payload = stage_dir / "dataset_payload"
        missing = [name for name in values["_runtime_files"]
                   if not (payload / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Staged dataset payload {payload} is missing attached inputs: "
                + ", ".join(missing) + "; stage the payload first")
        remote_revision_preflight(values["REPOSITORY"], values["BRANCH"], (),
                                  revision=values["REVISION"])
