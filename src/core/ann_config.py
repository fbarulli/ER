"""Validated SSOT for the standalone ANN embedding/index lane."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from core.common import TRAIN_ROOT

ANN_CONFIG_PATH = TRAIN_ROOT / "config" / "training_ANN.yaml"


class AnnEmbeddingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    loss: Literal["mnrl"]
    device: Literal["cpu", "cuda"]
    batch_size: int = Field(ge=1)
    encode_batch_size: int = Field(ge=1)
    max_sequence_length: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    warmup_ratio: float = Field(ge=0.0, le=1.0)
    weight_decay: float = Field(ge=0.0)


class AnnSmokeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample: int = Field(ge=2)
    epochs: int = Field(ge=1)


class HnswIndexSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["hnswlib"]
    output_dir: str = Field(min_length=1)
    space: Literal["ip"]
    ef_construction: int = Field(ge=1)
    M: int = Field(ge=2)
    ef_search: int = Field(ge=1)
    top_k: int = Field(ge=1)


class AnnTrainingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding: AnnEmbeddingSpec
    smoke: AnnSmokeSpec
    index: HnswIndexSpec


@lru_cache(maxsize=1)
def load_ann_config(path: Path = ANN_CONFIG_PATH) -> AnnTrainingSpec:
    if not path.is_file():
        raise FileNotFoundError(f"ANN config does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AnnTrainingSpec.model_validate(raw)


__all__ = ["ANN_CONFIG_PATH", "AnnTrainingSpec", "HnswIndexSpec", "load_ann_config"]
