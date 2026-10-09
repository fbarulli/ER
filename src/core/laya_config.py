"""src/core/laya_config.py — LayaSpec, the laya decision lane's config SSOT.

Relocated 2026-10-07 from core.schemas (one lane one file; schemas.py stays
the megafile's shared core). core.schemas re-exports LayaSpec so every
existing import surface stays byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.laya_datasets import LayaDatasets


class FinetuneSpec(BaseModel):
    """laya.finetune — the fine-tune recipe SSOT (additive).

    Every `laya.train.TrainConfig` field the finetune kernel drives is named
    here, so the whole trainer surface is YAML-driven. The defaults reproduce
    the landed research recipe (epochs 8, micro_batch 8, grad_accum 8,
    encoder_lr 2.5e-5, head_lr 1e-4, loss "soft-ce", seed 1729) and the
    upstream `TrainConfig` defaults for every other knob, so a
    config/training.yaml without this block stages byte-identically.

    `eval_data` is not a knob: the kernel sets it to the attached dev split.
    `device` is the runtime resolver input (`"auto"` -> cuda when the pinned
    single T4 is present).
    """

    model_config = ConfigDict(extra="forbid")

    # optimization
    epochs: int = Field(default=8, ge=1, le=128)
    micro_batch: int = Field(default=8, ge=1, le=1024)
    grad_accum: int = Field(default=8, ge=1, le=1024)
    encoder_lr: float = Field(default=2.5e-5, gt=0.0)
    head_lr: float = Field(default=1e-4, gt=0.0)
    min_lr: float = Field(default=1e-6, ge=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    # loss
    loss: Literal["soft-ce", "rlcd"] = "soft-ce"
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    rl_samples: int = Field(default=4, ge=1, le=1024)
    sigma_start: float = Field(default=0.4, ge=0.0)
    sigma_end: float = Field(default=0.1, ge=0.0)
    w_sph: float = Field(default=0.75, ge=0.0)
    w_rps: float = Field(default=1.0, ge=0.0)
    # layout / data
    shuffle_options: list[str] = Field(default_factory=list)
    option_layout: Literal["sequential", "parallel"] | None = None
    max_len: int | None = Field(default=None, ge=1)
    head_max_len: int | None = Field(default=None, ge=1)
    text_column: str = "text"
    label_column: str = "label"
    question_id: str = "label"
    instructions: str | None = None
    freeze_encoder: bool = False
    # calibration / abstention
    calib_max: int = Field(default=400, ge=0)
    calib_frac: float = Field(default=0.1, ge=0.0, lt=1.0)
    calib_seed: int = 20260922
    target_error: float = Field(default=0.10, ge=0.0, le=1.0)
    min_abstain_n: int = Field(default=10, ge=1)
    # runtime
    seed: int = 1729
    amp: bool | None = None
    gradient_checkpointing: bool | None = None
    log_every: int = Field(default=100, ge=0)
    device: str = "auto"

    # ── training controls (NOT TrainConfig fields; baked as FINETUNE_CONTROL) ─
    # These knobs drive the staged perf patch, never `laya.train.TrainConfig`
    # (which rejects unknown kwargs). They are OUT of FINETUNE_CONFIG_FIELDS so
    # the baked TrainConfig surface is byte-identical to the landed recipe.
    # Phase 1 (DEFAULT-ON): per-epoch dev eval, early stop, best tracking,
    # per-epoch checkpoint/resume and the LR-scheduler menu.
    eval_dev: bool = True
    early_stop: bool = True
    early_stop_patience: int = Field(default=2, ge=0, le=64)
    early_stop_min_delta: float = Field(default=0.0, ge=0.0)
    early_stop_metric: Literal["dev_accuracy", "dev_loss"] = "dev_accuracy"
    keep_best: bool = True
    save_each_epoch: bool = True
    resume: bool = True
    lr_scheduler: Literal["cosine", "linear", "constant", "onecycle",
                          "plateau"] = "cosine"
    warmup_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    warmup_steps: int = Field(default=0, ge=0)
    plateau_patience: int = Field(default=2, ge=0, le=64)
    plateau_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    onecycle_pct_start: float = Field(default=0.3, gt=0.0, lt=1.0)
    # Confidence cut used only to derive the dev abstain_rate/coverage wandb
    # metrics (laya's evaluate_records emits neither).
    abstain_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Phase 2 (default-OFF): gradual unfreezing, layer-wise LR decay, EMA,
    # SWA/SWA-LR and the optimizer menu.
    unfreeze_after_epoch: int | None = Field(default=None, ge=0)
    layer_decay: float = Field(default=1.0, gt=0.0, le=1.0)
    ema: bool = False
    ema_decay: float = Field(default=0.999, gt=0.0, lt=1.0)
    swa: bool = False
    swa_lr: float | None = Field(default=None, gt=0.0)
    swa_start_frac: float = Field(default=0.75, ge=0.0, lt=1.0)
    optimizer: Literal["adamw", "adafactor", "lamb"] = "adamw"
    # Phase 3 (default-OFF): class weighting / balanced sampling, hard-example
    # mining, curriculum, adversarial training and post-hoc temperature scaling.
    class_weight: bool = False
    balanced_sample: bool = False
    hard_example_frac: float = Field(default=0.0, ge=0.0, lt=1.0)
    curriculum: bool = False
    adv_eps: float = Field(default=0.0, ge=0.0)
    adv_kind: Literal["fgm", "awp"] = "fgm"
    temperature_scale: bool = False
    # Phase 4 (default-OFF, except log_grad_norm): torch.compile, AMP dtype,
    # TF32, grad-norm logging, ECE/confusion artifacts and determinism.
    compile_model: bool = False
    amp_dtype: Literal["fp16", "bf16"] = "fp16"
    tf32: bool = False
    log_grad_norm: bool = True
    write_error_artifacts: bool = False
    deterministic: bool = False
    # ── torch.profiler (default-OFF; opt in per run) ────────────────────────
    # A bounded schedule profiles only a slice of an epoch; the chrome trace
    # lands under `<output_dir>/<profile_dir>/epoch_<n>.json` (rank 0 only).
    profile: bool = False
    profile_dir: str = "profiler"
    profile_schedule: dict[str, int] = Field(
        default_factory=lambda: {"wait": 1, "warmup": 1, "active": 1,
                                 "repeat": 1})
    # ── extended knobs (all default-OFF; HPO-searchable dials) ──────────────
    # no_decay_bias_norm: exclude bias/norm (ndim<=1) params from weight_decay.
    no_decay_bias_norm: bool = False
    # optim_state_dtype: bf16 optimizer states/master weights (composes with
    # amp_dtype, which only governs the forward compute).
    optim_state_dtype: Literal["fp32", "bf16"] = "fp32"
    # lr_scaling: scale the peak LR from effective batch (micro*accum*world)
    # over base_batch; explicit encoder_lr/head_lr win when "none".
    lr_scaling: Literal["none", "linear", "sqrt"] = "none"
    base_batch: int = Field(default=0, ge=0)
    # r_drop: two dropout-masked forwards + alpha*KL (auto-off without dropout).
    r_drop: bool = False
    r_drop_alpha: float = Field(default=0.5, ge=0.0)
    # drop_path: stochastic depth over the head's transformer blocks.
    drop_path: bool = False
    drop_path_rate: float = Field(default=0.1, ge=0.0, lt=1.0)
    drop_path_schedule: Literal["constant", "linear"] = "linear"
    # dynamic_padding: round the collated batch length to pad_to_multiple
    # (laya already pads to the batch longest; max_len only truncates).
    dynamic_padding: bool = False
    pad_to_multiple: int = Field(default=8, ge=0)
    # batch_size_ramp: linearly ramp grad_accum (rank-symmetric) to target.
    batch_size_ramp: bool = False
    batch_ramp_start_frac: float = Field(default=0.25, gt=0.0, le=1.0)
    batch_ramp_epochs: int = Field(default=2, ge=0)
    # loss_schedule: ONE schedule for sigma/w_sph/w_rps/margin. "laya"
    # reproduces laya.train.sigma_at exactly (byte-identical default);
    # "constant" holds every term at its start value (no schedule).
    loss_schedule: Literal["laya", "linear", "cosine", "constant"] = "laya"
    w_sph_end: float | None = Field(default=None, ge=0.0)
    w_rps_end: float | None = Field(default=None, ge=0.0)
    contrastive_margin: float = Field(default=0.0, ge=0.0)
    contrastive_margin_end: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _profile_dir_is_a_portable_name(self) -> "FinetuneSpec":
        candidate = Path(self.profile_dir)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                f"finetune.profile_dir must be a portable name: "
                f"{self.profile_dir!r}")
        allowed = {"wait", "warmup", "active", "repeat"}
        extra = set(self.profile_schedule) - allowed
        if extra:
            raise ValueError(
                f"finetune.profile_schedule carries unknown keys {sorted(extra)}; "
                f"expected {sorted(allowed)}")
        for key in allowed:
            value = self.profile_schedule.get(key)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(
                    f"finetune.profile_schedule.{key} must be a non-negative "
                    f"int, got {value!r}")
        return self


class EvalCalibrationSpec(BaseModel):
    """laya.eval_calibration — the held-out EVAL path's calibration knobs.

    The fine-tune EVAL-ONLY path (the remote `finetune-eval` kernel and the
    local `--local-eval` CPU helper) fits laya's own calibration on the scored
    split: `fit_temperature_map` (the per-type temperature sequence + the
    per-bucket map) and, opt-in, `fit_abstention_thresholds` (the per-bucket
    `min_confidence` gate). Both live in the external `laya` package and are
    CONSUMED here, never reimplemented; this block only selects them.

    The defaults reproduce the landed eval byte-for-byte: the temperature map
    was always fitted and reported (`temperature=True`), abstention threshold
    fitting was never called (`abstention=False`, so the report carries no
    `abstention_thresholds` key), and no explicit runtime scalar was pinned
    (`min_confidence=None`). A config/training.yaml without this block
    therefore stages exactly as before.

    `min_confidence` is the runtime scalar the eval report/provenance echoes
    (the `laya.router`-style gate): when set it also overrides the fitted
    map's `"default"` sentinel, because an explicit operator pin beats a fit.
    `target_error`/`min_abstain_n` are the two `fit_abstention_thresholds`
    kwargs (`target_error` as named; `min_abstain_n` is passed as laya's
    `min_bucket_n` floor, the same role `TrainConfig.min_abstain_n` plays) and
    mirror the sibling `FinetuneSpec` defaults, never re-derived.
    """

    model_config = ConfigDict(extra="forbid")

    temperature: bool = True
    abstention: bool = False
    target_error: float = Field(default=0.10, ge=0.0, le=1.0)
    min_abstain_n: int = Field(default=10, ge=1)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _abstention_needs_temperature(self) -> "EvalCalibrationSpec":
        # `fit_abstention_thresholds(records, temperature, ...)` scales the
        # logits by the fitted per-type map before choosing the cut, so
        # abstention without the temperature fit has no calibrated scale.
        if self.abstention and not self.temperature:
            raise ValueError(
                "laya.eval_calibration.abstention=True requires temperature="
                "True: laya's fit_abstention_thresholds cuts on the "
                "calibrated confidence scale")
        return self


class FinetuneSmokeSpec(BaseModel):
    """laya.finetune_smoke — the tiny end-to-end smoke of the finetune kernel.

    The NEW finetune kernel (dials + profiler + early-stop/dev-eval + the
    fail-loud fetchers) only ever ran on a T4. This block selects the smallest
    honest end-to-end validation: a tiny subset corpus, 1 epoch, micro-batch 1
    / grad-accum 1, and DEDICATED dataset + kernel slugs so a smoke never
    versions or overwrites the production corpus/kernel. Every dial is config,
    never a code literal.

    ``device`` is the ONE source of truth for the smoke's runtime: ``"cpu"``
    (the default) keeps the original CPU smoke byte-for-byte, while ``"cuda"``
    selects the same single-T4 path as the prod kind (the kernel metadata's
    ``enable_gpu`` and the baked ``FINETUNE_DEVICE`` both derive from it), so a
    GPU smoke proves the GPU code path end-to-end. There is deliberately no
    separate ``enable_gpu`` bool: a bool + a device could disagree.

    There is deliberately no ``enabled`` flag: the dedicated ``finetune-smoke``
    decision kind IS the selector, so an illegal "smoke on the prod kind" state
    cannot be expressed. A missing slug fails loud at staging.
    """

    model_config = ConfigDict(extra="forbid")

    kernel_slug: str | None = None
    dataset_slug: str | None = None
    # Where the tiny subsets are generated (TRAIN_ROOT-relative, gitignored).
    corpus_dir: str = "results/laya_lane/smoke_corpus"
    train_rows: int = Field(default=200, ge=1, le=5000)
    dev_rows: int = Field(default=100, ge=1, le=5000)
    test_rows: int = Field(default=100, ge=1, le=5000)
    epochs: int = Field(default=1, ge=1, le=8)
    micro_batch: int = Field(default=1, ge=1, le=64)
    grad_accum: int = Field(default=1, ge=1, le=64)
    device: Literal["cpu", "cuda"] = "cpu"


class LayaSpec(BaseModel):
    """training.laya — the laya decision lane's SSOT (additive).

    Additive exactly like the sibling KaggleSpec (kaggle: above): a
    config/training.yaml without this block loads byte-identically and no
    existing default flips. The block declares what the lane stages and
    what its preconditions check; the SSOT `config/paths.yaml` `files:`
    bindings are REFERENCES (resolved through core.common.F), never
    duplicated.

    Whitespace/careat: `laya_decision_epochs <= 0` DISABLES the lane (no
    payload may stage a GPU session); the knob makes "laya decisions off"
    reachable by config alone.
    """

    model_config = ConfigDict(extra="forbid")

    # The laya.question typed-question schema (checked at staging; a
    # missing file fails the stage, never a silent empty placeholder).
    question_schema: str = "config/laya.question.json"
    # Which SSOT binding (config/paths.yaml `files:`) the decision CSV
    # resolves from per run kind — resolved through core.common.F at
    # staging, never duplicated. Keys are the lane's decision kinds.
    decision_csv_bindings: dict[str, str] = Field(
        default_factory=lambda: {"attribute": "dataset",
                                 "identity": "final_validation",
                                 "laya-cli-eval": "final_validation"},
    )
    # laya checkpoint hub source (convaiinnovations/laya on the Hugging
    # Face hub). Kept for the decision kinds that still load a checkpoint
    # by id; the FINE-TUNE path no longer uses it (see base_model_* below).
    checkpoint_hub: str = "convaiinnovations/laya"
    # The fine-tune BASE checkpoint travels as its OWN kaggle dataset: the
    # 647 MB local snake_local tree ships as a `.tar.zst` (plain git caps
    # at 100 MB), the transport the project already uses for large
    # payloads. The finetune kernel attaches this dataset, extracts the
    # archive under /kaggle/input to a local dir, and passes the extracted
    # DIRECTORY as `--base` — so resolve_checkpoint_dir sees a local dir
    # carrying rl_agent_config.json and never calls the Hub.
    base_model_dataset: str | None = LayaDatasets.BASE.slug
    base_model_archive: str = "convaiinnovations-laya.tar.zst"
    # The single top-level member of base_model_archive (extraction yields
    # a dir of this name); used as a deterministic hint before the rglob.
    base_model_dir: str = "convaiinnovations-laya"
    # Staging root (TRAIN_ROOT-relative). Receipts land under
    # results/laya_lane/<kind>/<op>/...
    staging_dir: str = "results/laya_lane"
    # PyPI package (installed over pip on the session, never vendored).
    laya_package: str = "laya"
    # The fine-tune/eval kernels' pin (the version their flags + the PERF/
    # device monkeypatches were verified against). Distinct from
    # `laya_package` (the DECISION kernel's unpinned install) and YAML-driven,
    # so a version bump is config, never a code literal.
    finetune_package: str = "laya>=0.3.29"
    # The JSONL corpus root the fine-tune/eval payloads stage from
    # (TRAIN_ROOT-relative; scripts/laya_build_dataset.py writes it). A path
    # knob, not a literal: relocating the corpus is config alone.
    finetune_corpus_dir: str = "data/laya"
    # Both kaggle dataset slugs ('owner/slug') are owner-picked
    # 2026-10-07 (no silent default account; kaggle_slug=None sibling
    # precedent).
    dataset_slug: str | None = LayaDatasets.REQUESTS.slug
    export_dataset_slug: str | None = LayaDatasets.DECISIONS.slug
    # The fine-tune corpus (data/laya/{train,dev,test}.jsonl + receipt.json,
    # scripts/laya_build_dataset.py) travels as its OWN kaggle dataset and the
    # trained checkpoint is its own kernel — distinct slugs from the decision
    # payloads so a dataset version never drops the decision inputs.
    finetune_dataset_slug: str | None = LayaDatasets.CORPUS.slug
    finetune_kernel_slug: str | None = "fbarulli/er-laya-finetune"
    # ── fine-tune EVAL-only path (held-out score, no retrain) ─────────────
    # A dedicated eval-only kernel scores a fine-tuned checkpoint on the
    # corpus held-out split: it attaches the SAME corpus dataset
    # (finetune_dataset_slug) + the fine-tuned checkpoint dataset
    # (finetune_ckpt_dataset), loads the checkpoint, and runs
    # `laya.train.calibration_records` + `evaluate_records`. No training,
    # no Hub fetch. The checkpoint dataset is optional when the operator
    # bakes an explicit local/attached path instead.
    finetune_eval_kernel_slug: str | None = "fbarulli/er-laya-finetune-eval"
    finetune_ckpt_dataset: str | None = LayaDatasets.FINETUNE_CKPT.slug
    # Deterministic member-dir hint inside the attached checkpoint dataset
    # (mirrors base_model_dir); the kernel rglobs `rl_agent_config.json` as
    # a fallback when the hint misses.
    finetune_ckpt_dir: str = "checkpoint"
    # The corpus split the eval-only kernel scores. Held out by construction:
    # the fine-tune trains on train and evaluates/calibrates on dev.
    finetune_eval_split: Literal["train", "dev", "test"] = "test"
    finetune_eval_batch_size: int = Field(default=16, ge=1, le=256)
    # ── holdout-eval path (component-disjoint verification ON Kaggle) ──────
    # Scores a fine-tuned checkpoint on the component-disjoint holdout (real
    # pairs + P0 + gate strata; scripts/laya_holdout.py) IN-SESSION and writes
    # the clustered, gate-stratified report, so verification never runs
    # locally. Attaches the staged holdout dataset (holdout_dataset_slug) + the
    # checkpoint dataset (finetune_ckpt_dataset).
    holdout_eval_kernel_slug: str | None = "fbarulli/er-laya-holdout-eval"
    holdout_dataset_slug: str | None = LayaDatasets.HOLDOUT.slug
    holdout_csv: str = "data/laya/holdout.csv"
    holdout_eval_batch_size: int = Field(default=16, ge=1, le=256)
    holdout_eval_bootstrap: int = Field(default=2000, ge=0, le=100000)
    holdout_eval_threshold: float = 0.5
    # The tiny CPU end-to-end smoke selection (dedicated slugs + dials; see
    # FinetuneSmokeSpec). Off the production kind entirely: only the dedicated
    # `finetune-smoke` decision kind reads it.
    finetune_smoke: FinetuneSmokeSpec = Field(default_factory=FinetuneSmokeSpec)
    # The FULL `laya.train.TrainConfig` recipe the finetune kernel builds
    # (additive; defaults reproduce the landed recipe exactly). YAML-driven
    # so every trainer knob is SSOT config, never a code literal.
    finetune: FinetuneSpec = Field(default_factory=FinetuneSpec)
    # The held-out EVAL path's calibration/abstention selection (additive;
    # defaults reproduce the landed eval exactly). Consumed by the
    # `finetune-eval` kernel AND the local `--local-eval` CPU helper, so both
    # report the same per-type temperature + abstention `min_confidence`.
    eval_calibration: EvalCalibrationSpec = Field(
        default_factory=EvalCalibrationSpec)
    run_tag_prefix: str = "laya_"
    # SINGLE T4 per owner ruling; the meta never requests 2xT4.
    gpu: Literal["T4"] = "T4"
    laya_decision_batch_size: int = Field(default=8, ge=1, le=128)
    # 0 DISABLES the decision lane (no payload may stage a session).
    laya_decision_epochs: int = Field(default=1, ge=0, le=8)
    laya_decision_max_rows: int = Field(default=2500, ge=2, le=50000)
    # Router confidence gate: >0 wires min_confidence into the session's
    # Router.predict calls (abstention/low-confidence answers flagged).
    min_router_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Laya-side calibration (recorded as the session-side knob).
    calibration: bool = False
    # laya-evals harness score over the same identity decision samples.
    laya_evals_enabled: bool = False
    # ONNX export path (Agent backend='onnx'); the kernel wiring is
    # pending (see docs/laya-lane.md 'Caveats'). Recorded, not hidden.
    onnx: bool = False

    @model_validator(mode="after")
    def _declared_paths_are_portable(self) -> "LayaSpec":
        def walk(fragment: Any, where: str) -> None:
            if isinstance(fragment, dict):
                for key, value in fragment.items():
                    walk(value, f"{where}.{key}")
                return
            if not isinstance(fragment, str):
                return
            candidate = Path(fragment)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(
                    f"laya.{where} must be a portable name or relative path: {fragment!r}"
                )

        walk(self.model_dump(), "paths")
        return self
