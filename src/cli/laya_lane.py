"""Laya decision lane (branch laya-lane).

Typed-decision surface for the `laya` package (PyPI `laya`: non-
autoregressive decision engine, Python >= 3.10, torch/transformers wheel
stack): stage the laya.question schema + ER decision inputs on THIS box,
then run the typed decision questions on a REMOTE GPU session:
  * kind="kaggle" — a Kaggle GPU kernel payload (metadata + script +
    receipt under results/laya_lane/kaggle/<decision>/); `--execute`
    drives `kaggle kernels push`, everything else is offline staging.
  * kind="colab"  — a Colab notebook payload + receipt under
    results/laya_lane/colab/<decision>/, delivered in the kaggle-lane
    receipts style. The notebook is the delivery CONTRACT: this lane
    never imports or edits cli.colab / cli.colab_lane and never opens a
    session.

Owner rulings honored (docs/laya-lane.md):
* 2xT4 -> SINGLE T4 per owner ruling: no double accelerator (the session
  pins one CUDA device; the staged payload never requests a second GPU);
* laya installs over pip (`pip install laya`), never vendored;
* exports return via /kaggle/working tar + a hashed receipt.

Contract + evidence: tests/test_laya_lane.py (offline, no network).

This module is the lane entry point + public surface: the behavior lives in
the injected factories (`laya_recipe` / `laya_runtime` / `laya_staging` /
`laya_publish` / `laya_transport` / `laya_local_eval` / `laya_traceability`);
the thin wrappers here read the lane's own globals so tests can re-point
``_spec`` / ``TRAIN_ROOT`` / ``_git_revision`` / ``training_cfg`` / the env
lookups / the decision preflight ONCE at this boundary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from cli.kaggle_kernel_templates import KernelTemplates
from cli.kaggle_kernels import KaggleKernels
from cli.laya_kernel_text_edge import (  # noqa: F401  (public lane surface)
    DECISION_KERNEL_SCRIPT,
    EVAL_KERNEL_SCRIPT,
    HOLDOUT_EVAL_KERNEL_SCRIPT,
    HOLDOUT_RUNTIME_PREFLIGHT,
    LAYA_RUNTIME_PREFLIGHT,
    NOTEBOOK_SCRIPT,
)
from cli.laya_kernel_text_finetune import (  # noqa: F401
    FINETUNE_DEVICE_PATCH_SOURCE,
    FINETUNE_EVAL_KERNEL_SCRIPT,
    FINETUNE_EVAL_RUNTIME_PREFLIGHT,
    FINETUNE_KERNEL_SCRIPT,
    FINETUNE_RUNTIME_PREFLIGHT,
)
from cli.laya_local_eval import LayaLocalEvalRunner
from cli.laya_publish import LayaPublishFactory

# ── the contract registry (SSOT constants + recipe factory) ─────────────────
from cli.laya_recipe import (  # noqa: F401  (public lane surface)
    BASE_MODEL_MANIFEST_FILE,
    COLAB_NOTEBOOK_NAME,
    DATASET_CSV_NAME,
    DATASET_METADATA_FILE,
    DATASET_PAYLOAD_DIR,
    DECISION_BINDINGS,
    DECISION_KERNEL_CODE_FILE,
    EVAL_CALIBRATION_FIELDS,
    EVAL_KERNEL_CODE_FILE,
    FINETUNE_CKPT_DECISION,
    FINETUNE_CODE_FILE,
    FINETUNE_CONFIG_FIELDS,
    FINETUNE_CONTROL_FIELDS,
    FINETUNE_CORPUS_FILES,
    FINETUNE_CORPUS_RECEIPT,
    FINETUNE_DECISION,
    FINETUNE_EVAL_CODE_FILE,
    FINETUNE_EVAL_DECISION,
    FINETUNE_EVAL_RECEIPT_FILE,
    FINETUNE_EVAL_REPORT_FILE,
    FINETUNE_EVAL_SPLIT_FILES,
    FINETUNE_LAYA_PACKAGE,
    FINETUNE_SMOKE_DECISION,
    GPU_KINDS,
    HOLDOUT_CATALOG_FILE,
    HOLDOUT_EVAL_CODE_FILE,
    HOLDOUT_EVAL_DECISION,
    HOLDOUT_EVAL_RECEIPT_FILE,
    HOLDOUT_EVAL_REPORT_FILE,
    HOLDOUT_JSONL,
    KINDS,
    QUESTION_SCHEMA_FILE,
    LayaRecipeFactory,
)
from cli.laya_runtime import LayaRuntimeFactory
from cli.laya_staging import LayaPayloadPreflight, LayaStagingFactory
from cli.laya_traceability import (  # noqa: F401  (public lane surface)
    RecordGrainGap,
    abstention_thresholds,
    corpus_digest,
    corpus_skip_census,
    corpus_traceability,
    coverage_from_records,
    decision_csv_records,
    decision_record_columns,
    decision_row_tags,
    emit_traceability,
    eval_case_records,
    metric_block_or_none,
    overlap_items,
    read_decision_rows,
    records_traceability,
)
from cli.laya_traceability import (
    fetched_traceability as _fetched_traceability,
)
from cli.laya_traceability import (
    staged_question_schema as _staged_question_schema,
)
from cli.laya_train_patch_ddp import FINETUNE_CONTROL_LOGIC_SOURCE  # noqa: F401
from cli.laya_train_patch_loop import FINETUNE_PERF_PATCH_SOURCE  # noqa: F401
from cli.laya_transport import LayaTransportFactory
from core.common import TRAIN_ROOT, training_cfg
from core.manifest import sha256_file  # noqa: F401  (public surface re-export)


# ── lane boundary: the global lookups tests re-point once ──────────────────
def _spec():
    return training_cfg().laya


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TRAIN_ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"git rev-parse failed in {TRAIN_ROOT}: {result.stderr.strip()}")
    return result.stdout.strip()


def _env_value(name: str) -> str | None:
    """The Colab lane's .env lookup (ONE shared implementation).

    The secret is baked into the STAGED kernel only, never written to the
    repo. This lane used to carry a near-verbatim copy; it now delegates so the
    two lanes cannot drift on lookup order.
    """
    from cli.colab_runtime import _env_value as _shared

    return _shared(name)


def _wandb_project() -> str:
    """The wandb project (`tracking.wandb.project`; ER default `e-r`)."""
    try:
        return training_cfg().tracking.wandb.project
    except Exception:
        return "e-r"


def _template(script: str, values: dict[str, str]) -> str:
    """The shared ``@TOKEN@`` substitution (ONE loop: KernelTemplates)."""
    return KernelTemplates.substitute(script, values)


def _kernel_script_gate(script: str) -> None:
    """The shared staging-time AST gate (ONE home: KaggleKernels)."""
    KaggleKernels._kernel_script_gate(script)


def _module_scope_gate(script: str) -> None:
    """The staged payload's top-level NameError gate."""
    LayaStagingFactory.module_scope_gate(script)


#: ``scripts/laya_metrics_pairs.py`` probes a built pair CSV through the lane's
#: own measurement (the canonical implementation lives on the runtime factory).
_measure_csv = LayaRuntimeFactory.measure_csv


def _runtime() -> LayaRuntimeFactory:
    return LayaRuntimeFactory(_spec(), TRAIN_ROOT)


def staging_dir() -> Path:
    """The lane staging root (TRAIN_ROOT-relative; SSOT laya.staging_dir)."""
    return _runtime().staging_dir()


def _stamp() -> str:
    return _runtime()._stamp()


def _log_lane(line: str) -> None:
    _runtime().log_lane(line)


def _log_local(line: str) -> None:
    """Console-only line for a laya surface that is NOT a Kaggle run."""
    _runtime().log_local(line)


def _staging_factory() -> LayaStagingFactory:
    from cli.laya_training_run import LayaTrainingRunFactory

    spec = _spec()
    runtime = LayaRuntimeFactory(spec, TRAIN_ROOT)
    return LayaTrainingRunFactory.build_staging(spec, runtime, training_cfg)


def _register_external_lanes() -> None:
    """Load external lane kinds into the transport registry (idempotent).

    The HPO lane owns its external-kind descriptor (kernel slug, receipt name,
    attached dataset) and registers it on import. A detached ``--watch`` process
    starts in THIS module, so the owning lane must be imported at this boundary
    or ``collect_kaggle_result`` falls back to the guessed receipt name.
    """
    from cli import laya_hpo

    if laya_hpo.HPO_DECISION not in LayaTransportFactory.EXTERNAL_KIND_DATASETS:
        laya_hpo.register_dispatch()


def _publish_factory() -> LayaPublishFactory:
    _register_external_lanes()
    return LayaPublishFactory(_spec(), _runtime())


def _transport_factory() -> LayaTransportFactory:
    _register_external_lanes()
    return LayaTransportFactory(_spec(), _runtime())


def _training_run():
    """The ONE run owner, wired from the lane boundary (see its factory)."""
    from cli.laya_training_run import LayaTrainingRunFactory

    return LayaTrainingRunFactory.from_config(
        spec=_spec(), train_root=TRAIN_ROOT, training_config=training_cfg)


# ── recipe surface ─────────────────────────────────────────────────────────
def finetune_config(spec: Any | None = None) -> dict[str, Any]:
    """The full `laya.train.TrainConfig` kwargs from `laya.finetune` (SSOT)."""
    return LayaRecipeFactory(spec if spec is not None
                             else _spec()).finetune_config()


def finetune_control(spec: Any | None = None) -> dict[str, Any]:
    """The training-control kwargs the perf patch bakes (SSOT)."""
    return LayaRecipeFactory(spec if spec is not None
                             else _spec()).finetune_control()


def eval_calibration_config(spec: Any | None = None) -> dict[str, Any]:
    """The `laya.eval_calibration` selection the eval kernels bake (SSOT)."""
    return LayaRecipeFactory(spec if spec is not None
                             else _spec()).eval_calibration_config()


def decision_binding(decision_kind: str) -> str:
    """SSOT F-binding for a decision kind (fail-loud on an unknown kind)."""
    return LayaRecipeFactory(_spec()).decision_binding(decision_kind)


# ── traceability surface ───────────────────────────────────────────────────
def staged_question_schema() -> dict:
    """The ``questions`` dict of the schema THIS BOX staged, never re-derived."""
    return _staged_question_schema(staging_dir())


def fetched_traceability(receipt: dict, reports: dict, *,
                         decision_kind: str | None = None,
                         decision_rows=(), questions=None
                         ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a fetched kernel's reports and EMIT the per-row grain."""
    return _fetched_traceability(
        receipt, reports, decision_kind=decision_kind,
        decision_rows=decision_rows, questions=questions,
        staging_root=staging_dir())


# ── staging surface ────────────────────────────────────────────────────────
def stage_decision_input(kind: str, *, decision_kind: str,
                         override: Path | None = None) -> dict[str, Any]:
    return _staging_factory().stage_decision_input(
        kind, decision_kind=decision_kind, override=override)


def stage_question_schema(kind: str, *,
                          override: Path | None = None) -> dict[str, Any]:
    return _staging_factory().stage_question_schema(kind, override=override)


def stage_dataset_payload(decision_kind: str, *, dataset_slug: str,
                          question_source: Path,
                          decision_source: Path) -> dict[str, Any]:
    return _staging_factory().stage_dataset_payload(
        decision_kind, dataset_slug=dataset_slug,
        question_source=question_source, decision_source=decision_source)


def stage_decision_kernel(*, decision_kind: str,
                          revision: str | None = None,
                          run_tag: str | None = None,
                          input_override: Path | None = None,
                          checkpoint_path: Path | None = None
                          ) -> dict[str, Any]:
    return _staging_factory().stage_decision_kernel(
        decision_kind=decision_kind, revision=revision, run_tag=run_tag,
        input_override=input_override, checkpoint_path=checkpoint_path)


def stage_finetune_dataset_payload(*, dataset_slug: str, corpus_dir: Path,
                                   kind: str = FINETUNE_DECISION,
                                   title: str = "er laya train"
                                   ) -> dict[str, Any]:
    return _staging_factory().stage_finetune_dataset_payload(
        dataset_slug=dataset_slug, corpus_dir=corpus_dir, kind=kind,
        title=title)


def stage_finetune_ckpt_dataset_payload(*, dataset_slug: str,
                                        checkpoint_dir: Path,
                                        member: str = "checkpoint"
                                        ) -> dict[str, Any]:
    return _staging_factory().stage_finetune_ckpt_dataset_payload(
        dataset_slug=dataset_slug, checkpoint_dir=checkpoint_dir,
        member=member)


def stage_holdout_dataset_payload(*, dataset_slug: str, holdout_csv: Path,
                                  catalog_path: Path, question_path: Path
                                  ) -> dict[str, Any]:
    return _staging_factory().stage_holdout_dataset_payload(
        dataset_slug=dataset_slug, holdout_csv=holdout_csv,
        catalog_path=catalog_path, question_path=question_path)


def stage_holdout_eval_kernel(*, revision: str | None = None,
                              run_tag: str | None = None) -> dict[str, Any]:
    return _staging_factory().stage_holdout_eval_kernel(
        revision=revision, run_tag=run_tag)


def stage_finetune_kernel(*, revision: str | None = None,
                          run_tag: str | None = None,
                          smoke: bool = False) -> dict[str, Any]:
    return _staging_factory().stage_finetune_kernel(
        revision=revision, run_tag=run_tag, smoke=smoke)


def stage_finetune_eval_kernel(*, revision: str | None = None,
                               run_tag: str | None = None,
                               checkpoint_path: Path | None = None
                               ) -> dict[str, Any]:
    return _staging_factory().stage_finetune_eval_kernel(
        revision=revision, run_tag=run_tag, checkpoint_path=checkpoint_path)


def stage_colab_notebook(*, decision_kind: str,
                         run_tag: str | None = None) -> dict[str, Any]:
    return _staging_factory().stage_colab_notebook(
        decision_kind=decision_kind, run_tag=run_tag)


def _staged_laya_push_preflight(stage_dir: Path) -> None:
    """Laya push gate over the staged payload's attached-inputs inventory."""
    LayaPayloadPreflight.verify(stage_dir)


# ── publish surface ────────────────────────────────────────────────────────
def package_base_model(*, source_dir: Path, dataset_slug: str,
                       archive_name: str, member_name: str,
                       output_dir: Path | None = None) -> dict[str, Any]:
    return _publish_factory().package_base_model(
        source_dir=source_dir, dataset_slug=dataset_slug,
        archive_name=archive_name, member_name=member_name,
        output_dir=output_dir)


def publish_laya_dataset(decision_kind: str, *, run_tag: str,
                         execute: bool) -> dict[str, Any]:
    return _publish_factory().publish_laya_dataset(
        decision_kind, run_tag=run_tag, execute=execute)


# ── transport surface ──────────────────────────────────────────────────────
def kernel_slug(decision_kind: str) -> str:
    return _transport_factory().kernel_slug(decision_kind)


def recorded_session_id(slug: str) -> int | None:
    return _transport_factory().recorded_session_id(slug)


def stop_kaggle_kernel(slug: str, *, execute: bool,
                       wait: bool = True) -> dict[str, Any]:
    return LayaTransportFactory.stop_kaggle_kernel(slug, execute=execute,
                                                   wait=wait)


def delete_kaggle_kernel(slug: str, *, execute: bool = False) -> dict[str, Any]:
    return LayaTransportFactory.delete_kaggle_kernel(slug, execute=execute)


def push_kaggle_kernel(stage_dir: Path, *, execute: bool,
                       activate: bool = True) -> dict[str, Any]:
    return _transport_factory().push_kaggle_kernel(
        stage_dir, execute=execute, activate=activate)


def collect_kaggle_result(kind: str, slug: str | None = None, *,
                          execute: bool = False) -> dict[str, Any]:
    return _transport_factory().collect_kaggle_result(
        kind, slug, execute=execute)


def watcher(kind: str, *, slug: str | None = None):
    """The consolidated detached watcher for a laya decision kind."""
    return _transport_factory().watcher(kind, slug=slug)


# ── local eval surface ─────────────────────────────────────────────────────
def fit_eval_calibration(laya_train: Any, records, calibration: dict) -> dict:
    """Fit laya's OWN calibration for the held-out eval path (SSOT selection)."""
    return LayaLocalEvalRunner.fit_eval_calibration(
        laya_train, records, calibration)


def local_eval_checkpoint(checkpoint_dir: Path, *,
                          eval_data: Path | None = None,
                          out_dir: Path | None = None,
                          split: str | None = None,
                          batch_size: int | None = None,
                          limit: int | None = None) -> dict[str, Any]:
    """Local (CPU) held-out eval of a fetched fine-tuned checkpoint."""
    return LayaLocalEvalRunner(_runtime(),
                               LayaRecipeFactory(_spec())).local_eval_checkpoint(
        checkpoint_dir, eval_data=eval_data, out_dir=out_dir, split=split,
        batch_size=batch_size, limit=limit)


class LayaLane:
    """The lane surface: results/laya_lane/<kind>/<decision>/ receipts.

    ONE class, TWO kinds: kind names the remote surface ("kaggle",
    "colab"); anything else fails loud.
    """

    kind: str

    def __init__(self, kind: str):
        if kind not in KINDS:
            raise ValueError(f"unknown laya lane kind: {kind!r}; "
                             f"expected {list(KINDS)}")
        self.kind = kind
        self._spec = _spec()

    def stage(self, decision_kind: str, *,
              input_override: Path | None = None,
              checkpoint_path: Path | None = None) -> dict[str, Any]:
        """Stage the payload (offline, dry-safe)."""
        if self.kind == "kaggle":
            if decision_kind == HOLDOUT_EVAL_DECISION:
                return stage_holdout_eval_kernel()
            return stage_decision_kernel(
                decision_kind=decision_kind, input_override=input_override,
                checkpoint_path=checkpoint_path)
        return stage_colab_notebook(decision_kind=decision_kind)

    def push(self, stage_dir: Path, *, execute: bool = False,
             activate: bool = True) -> dict[str, Any]:
        """`--execute` gated push (kaggle kind only)."""
        if self.kind != "kaggle":
            raise RuntimeError("push is a kaggle-lane operation")
        return push_kaggle_kernel(stage_dir, execute=execute,
                                  activate=activate)


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, default="kaggle")
    parser.add_argument("--decision", choices=GPU_KINDS, default="attribute",
                        help="which typed decision run to stage "
                             "(default: attribute)")
    parser.add_argument("--execute", action="store_true",
                        help="make the remote call (kaggle kernels push); "
                             "the default is an offline dry-run")
    parser.add_argument("--decision-input", type=Path, default=None,
                        help="alternate decision source CSV on this box "
                             "(forwarded as the staging override; "
                             "kaggle staging path)")
    parser.add_argument("--fetch", action="store_true",
                        help="fetch a pushed kernel's output (kaggle "
                             "kernels output) instead of staging; combine "
                             "with --decision + --slug and --execute")
    parser.add_argument("--slug", default=None,
                        help="the pushed kernel slug (owner/slug) for "
                             "--fetch / --stop (default: the config slug "
                             "for --decision)")
    parser.add_argument("--stop", action="store_true",
                        help="tear down the running session for --decision's "
                             "kernel (kaggle only); the launch-recorded "
                             "session id feeds the SDK cancel")
    parser.add_argument("--delete", action="store_true",
                        help="delete --decision's kernel (kaggle only), "
                             "releasing a live session first; dry-run unless "
                             "--execute")
    parser.add_argument("--track", action="store_true",
                        help="real-time tracking via the W&B run reader, "
                             "degrading to the ONE kaggle execution-log reader "
                             "(kaggle only)")
    parser.add_argument("--run-tag", default=None,
                        help="the W&B run tag to track (--track)")
    parser.add_argument("--watch", action="store_true",
                        help="run the terminal watcher for a pushed kernel: "
                             "poll to terminal, download via `kaggle kernels "
                             "output`, then release the session (kaggle only). "
                             "This is the detached entry the push spawns")
    parser.add_argument("--session-id", action="store_true",
                        help="print the launch-recorded kernel session id for "
                             "--decision's kernel (or --slug); no network")
    parser.add_argument("--local-eval", action="store_true",
                        help="run the CPU held-out eval of a fetched "
                             "fine-tuned checkpoint instead of staging")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="the fine-tuned checkpoint dir for "
                             "--local-eval (must carry rl_agent_config.json)")
    parser.add_argument("--eval-data", type=Path, default=None,
                        help="the eval JSONL for --local-eval (default: the "
                             "config split under data/laya)")
    parser.add_argument("--eval-out", type=Path, default=None,
                        help="output dir for --local-eval "
                             "(default: results/laya_lane/local_eval)")
    parser.add_argument("--eval-split", choices=tuple(FINETUNE_EVAL_SPLIT_FILES),
                        default=None,
                        help="the corpus split for --local-eval "
                             "(default: config laya.finetune_eval_split)")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="cap the number of eval rows for --local-eval")
    args = parser.parse_args()

    if args.local_eval:
        # Local CPU eval path: no staging, no network, no kernel.
        if args.checkpoint is None:
            parser.error("--local-eval requires --checkpoint PATH")
        report = local_eval_checkpoint(
            args.checkpoint, eval_data=args.eval_data, out_dir=args.eval_out,
            split=args.eval_split, limit=args.eval_limit)
        print(json.dumps(report, indent=2), flush=True)
        return

    if args.fetch:
        # Fetch path: kaggle kernels output for a pushed kernel; the eval
        # report JSONs land under results/laya_lane/fetch/<decision>/.
        if not args.slug:
            parser.error("--fetch requires --slug owner/slug")
        plan = _training_run().harvest(args.decision, args.slug,
                                       execute=args.execute)
        print(json.dumps(plan, indent=2), flush=True)
        return

    if args.session_id:
        # First-class reader of the launch-recorded session id (the verified
        # in-place stop's target); offline, no network.
        slug = args.slug or kernel_slug(args.decision)
        print(json.dumps({"kernel": slug,
                          "session_id": recorded_session_id(slug)},
                         indent=2), flush=True)
        return

    if args.stop:
        # First-class teardown: resolve the kernel the decision ran on (or an
        # explicit --slug), then cancel its running session. Dry-run by default.
        if args.kind != "kaggle":
            parser.error("--stop is a kaggle-lane operation")
        slug = args.slug or kernel_slug(args.decision)
        plan = stop_kaggle_kernel(slug, execute=args.execute)
        print(json.dumps(plan, indent=2, default=str), flush=True)
        return

    if args.delete:
        # Teardown: resolve the kernel the decision ran on (or an explicit
        # --slug), stop its running session, THEN delete the kernel. Dry-run by
        # default (the owner composes the two canonical transport calls).
        if args.kind != "kaggle":
            parser.error("--delete is a kaggle-lane operation")
        slug = args.slug or kernel_slug(args.decision)
        plan = _training_run().teardown(slug, execute=args.execute)
        print(json.dumps(plan, indent=2, default=str), flush=True)
        return

    if args.track:
        # Real-time tracking: the W&B run reader first (no run tag -> the ONE
        # kaggle execution-log reader for the resolved kernel). Kaggle only.
        if args.kind != "kaggle":
            parser.error("--track is a kaggle-lane operation")
        slug = args.slug or kernel_slug(args.decision)
        plan = _training_run().track(run_tag=args.run_tag, slug=slug,
                                     follow=True)
        print(json.dumps(plan, indent=2, default=str), flush=True)
        return

    if args.watch:
        # The detached terminal watcher's entry: poll to terminal, download via
        # `kaggle kernels output` and release the session. It carries its own
        # receipt under results/laya_lane/fetch/<kind>/.
        if args.kind != "kaggle":
            parser.error("--watch is a kaggle-lane operation")
        plan = watcher(args.decision, slug=args.slug).autowatch(
            execute=args.execute, slug=args.slug)
        print(json.dumps(plan, indent=2, default=str), flush=True)
        return

    lane = LayaLane(args.kind)
    receipt = lane.stage(args.decision, input_override=args.decision_input,
                         checkpoint_path=args.checkpoint)
    print(_stamp(), f"[laya-lane] staged {args.kind}/{args.decision} payload: "
          f"{json.dumps(receipt, indent=2)}", flush=True)
    if args.execute and args.kind == "kaggle":
        stage_dir = Path(receipt["staged"])
        # the inputs travel as the dataset BEFORE the push (the kernel
        # metadata attaches spec.dataset_slug; a missing/drifting dataset
        # would FileNotFoundError resolve_input once boot passes)
        dataset_plan = publish_laya_dataset(args.decision,
                                            run_tag=receipt["run_tag"],
                                            execute=True)
        print(json.dumps(dataset_plan, indent=2), flush=True)
        push_plan = lane.push(stage_dir, execute=True)
        print(json.dumps(push_plan, indent=2), flush=True)
        # ONE detached terminal watcher now owns progress, download and release:
        # it streams the run into the shared logs/kaggle/lane.log, fetches the
        # per-kind output into results/laya_lane/fetch/<kind>/ and stops the
        # session. Its spawn is the same owner the ER lane uses, so `--execute`
        # runs AND retrieves.
        kernel_id = json.loads(
            (stage_dir / "kernel-metadata.json").read_text())["id"]
        watch_plan = watcher(args.decision, slug=kernel_id).spawn()
        print(json.dumps(watch_plan, indent=2), flush=True)
    elif args.execute:
        _log_local("colab payloads are a delivery contract only; nothing "
                   "to --execute")
    else:
        _log_local("dry-run only; pass --execute to touch the remote surface")


if __name__ == "__main__":
    main()
