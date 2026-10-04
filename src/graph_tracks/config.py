"""Validated configuration for isolated graph experiments."""
from pathlib import Path
from urllib.parse import urlsplit
from typing import Literal

import yaml
from core.execution_policy import AggregationBackend, OptimizerBackend
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, model_validator


def _default_ann_recall_ks() -> tuple[int, ...]:
    """Resolve the ANN ladder from SSOT; missing dependencies/config fail."""
    from core.common import ann_retrieval_ks
    return ann_retrieval_ks()


class RetrievalConfig(BaseModel):
    """Shared validated ANN/index contract; explicit lane overrides are allowed."""
    model_config = ConfigDict(extra="forbid", validate_default=True)
    retrieval_ks: list[StrictInt] = Field(
        default_factory=lambda: list(_default_ann_recall_ks()), min_length=1
    )
    hnsw_m: int = Field(default=16, ge=2)
    hnsw_ef_construction: int = Field(default=200, ge=2)
    hnsw_ef_search: int = Field(default=100, ge=1)
    report_test: bool = True
    build_index: bool = True

    @model_validator(mode="after")
    def check_retrieval(self):
        if any(k < 1 for k in self.retrieval_ks) or len(set(self.retrieval_ks)) != len(self.retrieval_ks):
            raise ValueError("retrieval_ks must contain unique positive integers")
        return self


class RetrievalReportContext(RetrievalConfig):
    """Validated retrieval settings plus the provenance of scored vectors."""
    checkpoint: Path
    listings_sha256: str = Field(min_length=1)

    @classmethod
    def from_config(cls, cfg: RetrievalConfig, checkpoint: Path, listings_sha256: str):
        return cls.model_validate({
            **{key: getattr(cfg, key) for key in RetrievalConfig.model_fields},
            'checkpoint': checkpoint, 'listings_sha256': listings_sha256,
        })


class TextConfig(RetrievalConfig):
    track: Literal["text"]
    output_dir: str = Field(min_length=1)


class WandbSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: str = Field(default="e-r", min_length=1)
    mode: Literal["offline", "online", "disabled"] = "offline"


class DvcSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    remote: str | None = None
    push: bool = False

    @model_validator(mode="after")
    def check_remote(self):
        if self.remote:
            parsed = urlsplit(self.remote)
            if parsed.username or parsed.password:
                raise ValueError("DVC remote URL must not contain credentials; use environment/config.local")
        if self.push and (not self.enabled or not self.remote):
            raise ValueError("DVC push requires enabled and an explicit remote")
        return self


class GraphConfig(RetrievalConfig):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    track: Literal["gnn_only", "hybrid"]
    listings: str
    pairs: str
    output_dir: str
    text_cache: str | None = None
    text_checkpoint_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    input_manifest: str | None = None
    allow_unmanifested_inputs: bool = False
    wandb: WandbSpec = Field(default_factory=WandbSpec)
    dvc: DvcSpec = Field(default_factory=DvcSpec)
    include_inputs: bool = True
    postprocess: bool = True
    inference_batch_size: int = Field(default=1024, ge=1)
    hidden_dim: int = Field(default=64, ge=4)
    output_dim: int = Field(default=128, ge=4)
    epochs: int = Field(default=10, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    optimizer_backend: OptimizerBackend = 'auto'
    aggregation_backend: AggregationBackend = 'index_add'
    early_stopping_patience: int = Field(default=3, ge=1)
    early_stopping_threshold: float = Field(default=0.001, ge=0)
    lr_scheduler: Literal["plateau", "constant"] = "plateau"
    lr_patience: int = Field(default=1, ge=0)
    lr_factor: float = Field(default=0.5, gt=0, lt=1)
    min_lr: float = Field(default=0.00001, ge=0)
    seed: int = Field(default=1337, ge=0)
    device: Literal["cpu", "cuda"] = "cpu"
    graph_enabled: bool = True
    metric_weight: float = Field(default=1., ge=0)
    negative_margin: float = Field(default=0.3, ge=-1, lt=1)
    max_grad_norm: float = Field(default=1., gt=0)

    @model_validator(mode="after")
    def check_track(self):
        if (self.track == "hybrid") != bool(self.text_cache):
            raise ValueError("hybrid requires text_cache; gnn_only forbids it")
        if self.track == 'gnn_only' and self.text_checkpoint_sha256:
            raise ValueError('gnn_only forbids a text checkpoint reference')
        if not all((self.listings, self.pairs, self.output_dir)):
            raise ValueError("input and output paths must not be empty")
        return self


def load_config(path: Path, *, expected_track: str | None = None) -> GraphConfig:
    try:
        cfg = GraphConfig.model_validate(yaml.safe_load(path.read_text()))
    except (ValidationError, yaml.YAMLError) as exc:
        exc.add_note(f"Graph lane configuration: {path}")
        raise
    if expected_track is not None and cfg.track != expected_track:
        raise ValueError(f"{path}: expected {expected_track} lane, got {cfg.track}")
    return cfg


def load_text_config(path: Path) -> TextConfig:
    if not path.is_file():
        raise FileNotFoundError(f"text lane config {path} is missing; regenerate the track setup")
    try:
        return TextConfig.model_validate(yaml.safe_load(path.read_text()))
    except (ValidationError, yaml.YAMLError) as exc:
        exc.add_note(f"Text lane configuration: {path}")
        raise
