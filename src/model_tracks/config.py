from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    setup_dir: str
    text_bundle: str
    text_model: str = 'minilm_l6'
    epochs: int = Field(default=10, ge=1)
    device: Literal['cpu', 'cuda'] = 'cuda'
    schedule: Literal['parallel'] = 'parallel'
    max_parallel: Literal[3] = 3
    gpu_parallel_backend: Literal['mps'] = 'mps'
    memory_reservations_gb: dict[str, float] = Field(default_factory=dict)
    gpu_headroom_gb: float = Field(default=2.0, ge=0)
    report_test: bool = False
    publish_dvc: bool = False
    publish_git: bool = True
    profiling: bool = False

    @property
    def dvc_enabled(self) -> bool:
        return self.publish_dvc or self.publish_git

    @model_validator(mode='after')
    def check_memory(self):
        if set(self.memory_reservations_gb) - {'text', 'gnn_only', 'hybrid'}:
            raise ValueError('unknown track memory reservation')
        if any(v <= 0 for v in self.memory_reservations_gb.values()):
            raise ValueError('memory reservations must be positive measured peaks')
        if self.memory_reservations_gb and set(self.memory_reservations_gb) != {'text', 'gnn_only', 'hybrid'}:
            raise ValueError('provide measured memory peaks for all three tracks together')
        return self


def load_config(path: Path) -> SuiteConfig:
    return SuiteConfig.model_validate(yaml.safe_load(path.read_text()))
