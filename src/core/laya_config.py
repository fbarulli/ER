"""src/core/laya_config.py — LayaSpec, the laya decision lane's config SSOT.

Relocated 2026-10-07 from core.schemas (one lane one file; schemas.py stays
the megafile's shared core). core.schemas re-exports LayaSpec so every
existing import surface stays byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    # Face hub; the kernel loads it explicitly).
    checkpoint_hub: str = "convaiinnovations/laya"
    # Staging root (TRAIN_ROOT-relative). Receipts land under
    # results/laya_lane/<kind>/<op>/...
    staging_dir: str = "results/laya_lane"
    # PyPI package (installed over pip on the session, never vendored).
    laya_package: str = "laya"
    # Both kaggle dataset slugs ('owner/slug') are owner-picked
    # 2026-10-07 (no silent default account; kaggle_slug=None sibling
    # precedent).
    dataset_slug: str | None = "fbarulli/er-laya-requests"
    export_dataset_slug: str | None = "fbarulli/er-laya-decisions"
    # The fine-tune corpus (data/laya/{train,dev,test}.jsonl + receipt.json,
    # scripts/laya_build_dataset.py) travels as its OWN kaggle dataset and the
    # trained checkpoint is its own kernel — distinct slugs from the decision
    # payloads so a dataset version never drops the decision inputs.
    finetune_dataset_slug: str | None = "fbarulli/er-laya-train"
    finetune_kernel_slug: str | None = "fbarulli/er-laya-finetune"
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
