"""Bounded input transfers and a single host allocation for frozen outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch


@dataclass
class EmbeddingOutputBuffer:
    row_count: int
    device: str
    storage: torch.Tensor | None = None
    written: int = 0
    stream: torch.cuda.Stream | None = None
    pending_batches: int = 0
    maximum_inflight_batches: ClassVar[int] = 16

    def __enter__(self) -> EmbeddingOutputBuffer:
        if self.device == 'cuda':
            self.stream = torch.cuda.current_stream()
        return self

    def append(self, vectors: torch.Tensor) -> None:
        if vectors.ndim != 2 or self.written + len(vectors) > self.row_count:
            raise ValueError('embedding output population or dimensions differ')
        if self.storage is None:
            self.storage = torch.empty((self.row_count, vectors.shape[1]), dtype=torch.float32,
                                       device='cpu', pin_memory=self.device == 'cuda')
        if vectors.shape[1] != self.storage.shape[1]:
            raise ValueError('embedding output width changed between batches')
        end = self.written + len(vectors)
        self.storage[self.written:end].copy_(vectors.detach().float(), non_blocking=self.device == 'cuda')
        self.written = end
        self.pending_batches += 1
        if self.stream is not None and self.pending_batches >= self.maximum_inflight_batches:
            # Bound queued input pinned allocations and outstanding output
            # copies even when CPU submission outruns the encoder.
            self.stream.synchronize()
            self.pending_batches = 0

    def __exit__(self, *exc) -> None:
        # Complete outstanding copies even on failure before the pinned host
        # buffer can be destroyed. Consumers never read partially copied data.
        if self.stream is not None:
            self.stream.synchronize()

    def numpy(self) -> np.ndarray:
        if self.storage is None or self.written != self.row_count:
            raise ValueError('embedding output population incomplete')
        if self.stream is not None:
            self.stream.synchronize()
        return self.storage.numpy()
