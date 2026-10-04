from pathlib import Path
from typing import Literal
import yaml
from core.execution_policy import AggregationBackend, OptimizerBackend
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_default=True, allow_inf_nan=False)
    setup_dir: str = Field(min_length=1)
    text_bundle: str = Field(min_length=1)
    text_model: str = 'minilm_l6'
    epochs: int = Field(default=10, ge=1)
    device: Literal['cpu', 'cuda'] = 'cuda'
    schedule: Literal['parallel'] = 'parallel'
    max_parallel: Literal[3] = 3
    gpu_parallel_backend: Literal['mps'] = 'mps'
    # Missing fields in old suite manifests preserve their original execution.
    gpu_optimizer_backend: OptimizerBackend = 'auto'
    gpu_graph_aggregation_backend: AggregationBackend = 'index_add'
    memory_reservations_gb: dict[str, float] = Field(default_factory=dict)
    gpu_headroom_gb: float = Field(default=2.0, ge=0)
    report_test: bool = False
    publish_dvc: bool = False
    publish_git: bool = True
    profiling: bool = False
    post_training_ablation: bool = False
    ablation_config: str = Field(default='config/attribute_ablation.yaml', min_length=1)

    @field_validator('ablation_config')
    @classmethod
    def portable_ablation_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError('ablation_config must be a project-relative file path')
        return path.as_posix()

    def graph_execution_overrides(self) -> dict[str, str]:
        if self.device != 'cuda':
            return {}
        return {'optimizer_backend': self.gpu_optimizer_backend,
                'aggregation_backend': self.gpu_graph_aggregation_backend}

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
    try:
        return SuiteConfig.model_validate(yaml.safe_load(path.read_text()))
    except (ValidationError, yaml.YAMLError) as exc:
        exc.add_note(f"Model-track suite configuration: {path}")
        raise
