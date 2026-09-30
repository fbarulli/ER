"""Validated configuration for isolated graph experiments."""
from pathlib import Path
from urllib.parse import urlsplit
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class WandbSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: str = Field(default="e-r", min_length=1)
    mode: Literal["offline", "online", "disabled"] = "offline"


class DvcSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
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


class GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    report_test: bool = True
    build_index: bool = True
    inference_batch_size: int = Field(default=1024, ge=1)
    retrieval_ks: list[int] = Field(default_factory=lambda: [1, 5, 10, 50], min_length=1)
    hnsw_m: int = Field(default=16, ge=2)
    hnsw_ef_construction: int = Field(default=200, ge=2)
    hnsw_ef_search: int = Field(default=100, ge=1)
    hidden_dim: int = Field(default=64, ge=4)
    output_dim: int = Field(default=128, ge=4)
    epochs: int = Field(default=10, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0001, ge=0)
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
        if any(k < 1 for k in self.retrieval_ks) or len(set(self.retrieval_ks)) != len(self.retrieval_ks):
            raise ValueError("retrieval_ks must contain unique positive integers")
        if not all((self.listings, self.pairs, self.output_dir)):
            raise ValueError("input and output paths must not be empty")
        return self


def load_config(path: Path) -> GraphConfig:
    return GraphConfig.model_validate(yaml.safe_load(path.read_text()))
