"""Validated configuration for isolated graph experiments."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    track: Literal["gnn_only", "hybrid"]
    listings: str
    pairs: str
    output_dir: str
    text_cache: str | None = None
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
        if not all((self.listings, self.pairs, self.output_dir)):
            raise ValueError("input and output paths must not be empty")
        return self


def load_config(path: Path) -> GraphConfig:
    return GraphConfig.model_validate(yaml.safe_load(path.read_text()))
