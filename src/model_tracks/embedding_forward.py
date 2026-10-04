"""Shared contract for forwarding frozen native tokens on the suite device."""
from pathlib import Path
from typing import ClassVar, Literal
import json
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from graph_tracks.data import file_hash
from graph_tracks.text_cache import checkpoint_hash


EmbeddingDevice = Literal['cpu', 'cuda']


def validate_embedding_device(device):
    import torch
    device = TypeAdapter(EmbeddingDevice).validate_python(device)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('configured embedding device requires CUDA')
    return device


class PreparedEmbeddingForward(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    device: EmbeddingDevice
    checkpoint: Path
    request_path: Path
    tokens_path: Path
    plan: dict
    row_count: int = Field(gt=0)
    tokens_sha256: str
    embedding_dtype: ClassVar[str] = 'float32'
    normalization_atol: ClassVar[float] = 1e-4

    @property
    def export_location(self):
        return 'Colab GPU' if self.device == 'cuda' else 'Colab CPU'

    def forward(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from core.encoding_inputs import PreparedTokenInputs, tokenization_policy, load_token_features
        validate_embedding_device(self.device)
        request_hash = file_hash(self.request_path)
        checkpoint_digest = checkpoint_hash(self.checkpoint)
        if file_hash(self.tokens_path) != self.tokens_sha256:
            raise ValueError('embedding tokens changed before forwarding')
        model = SentenceTransformer(str(self.checkpoint), device=self.device, local_files_only=True)
        model.eval()
        if tokenization_policy(model) != self.plan['tokenization']:
            raise ValueError('checkpoint native tokenizer differs from prepared export')
        from core.performance import PerformanceRecorder
        perf = PerformanceRecorder('text')
        from model_tracks.embedding_staging import EmbeddingOutputBuffer
        with np.load(self.tokens_path, allow_pickle=False) as arrays, torch.no_grad(), perf.section("encode"), EmbeddingOutputBuffer(self.row_count, self.device) as outputs:
            PreparedTokenInputs(plan=self.plan, arrays=arrays, row_count=self.row_count)
            for batch in self.plan['token_batches']:
                vectors = model(load_token_features(arrays, batch, self.device))['sentence_embedding']
                outputs.append(torch.nn.functional.normalize(vectors, p=2, dim=1))
        if (file_hash(self.request_path) != request_hash
                or file_hash(self.tokens_path) != self.tokens_sha256
                or checkpoint_hash(self.checkpoint) != checkpoint_digest):
            raise ValueError('embedding inputs changed during forwarding')
        model._er_forward_performance = perf.summary()
        return outputs.numpy(), model, checkpoint_digest, request_hash

    @staticmethod
    def write(output, ids, vectors, metadata, validate):
        candidate = output.with_suffix('.npz.partial')
        with candidate.open('wb') as handle:
            np.savez_compressed(handle, ids=np.asarray(ids, dtype=str), embeddings=vectors,
                                metadata=json.dumps(metadata, sort_keys=True))
        validate(candidate)
        candidate.replace(output)
        return output
