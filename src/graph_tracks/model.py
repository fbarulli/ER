"""Two-hop typed listing -> attribute -> listing message passing in PyTorch.

Attribute states aggregate TRAIN listings only. Queries read those fixed
states and their own features, so query batches cannot inform one another.
This is a small full-batch baseline, not a neighbor-sampled GraphSAGE package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from graph_tracks.data import GraphBatch, NUMERIC, RELATIONS
from graph_tracks.pooling import pool, segment_pool, topology


def mean_pool(values: torch.Tensor, indices: torch.Tensor, count: int) -> torch.Tensor:
    total = values.new_zeros((count, values.shape[-1]))
    total.index_add_(0, indices, values)
    sizes = values.new_zeros(count)
    sizes.index_add_(0, indices, values.new_ones(len(indices)))
    return total / sizes.clamp_min(1).unsqueeze(1)


class AttributeGNN(nn.Module):
    def __init__(self, vocabulary: dict[str, list[str]], hidden: int = 64,
                 output: int = 128, text_dim: int = 0, graph_enabled: bool = True,
                 aggregation_backend: Literal['index_add', 'segment'] = 'index_add'):
        if aggregation_backend not in {'index_add', 'segment'}:
            raise ValueError('unknown graph aggregation backend')
        super().__init__()
        self.aggregation_backend = aggregation_backend
        self.graph_enabled = graph_enabled
        self.text_dim = text_dim
        self.tokens = nn.ModuleDict({r: nn.Embedding(len(vocabulary[r]) + 1, hidden)
                                    for r in RELATIONS})
        self.input = nn.Linear(len(NUMERIC) * 3 + len(RELATIONS) * hidden + text_dim, hidden)
        self.to_attribute = nn.ModuleDict({r: nn.Linear(hidden, hidden) for r in RELATIONS})
        self.to_listing = nn.ModuleDict({r: nn.Linear(hidden, hidden, bias=False) for r in RELATIONS})
        self.output = nn.Linear(hidden * 2, output)
        self.norm = nn.LayerNorm(hidden)

    def pool(self, values: torch.Tensor, target: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        operation = segment_pool if self.aggregation_backend == 'segment' else pool
        return operation(values, target, sizes)

    def initial(self, batch: GraphBatch, text: torch.Tensor | None = None) -> torch.Tensor:
        if bool(self.text_dim) != (text is not None):
            raise ValueError("text input is required only for the hybrid model")
        features = [batch.numeric]
        for relation in RELATIONS:
            value, listing, sizes = topology(batch, relation, dtype=self.tokens[relation].weight.dtype)
            features.append(self.pool(self.tokens[relation](value), listing, sizes))
        if text is not None:
            if text.shape != (len(batch.numeric), self.text_dim):
                raise ValueError("text vector shape mismatch")
            features.append(F.normalize(text, dim=-1))
        return F.relu(self.norm(self.input(torch.cat(features, dim=-1))))

    def context(self, support: GraphBatch, text: torch.Tensor | None = None, *, initial=None) -> dict:
        if not self.graph_enabled:
            return {}
        h = self.initial(support, text) if initial is None else initial
        states = {}
        for relation in RELATIONS:
            listing, value, sizes = topology(
                support, relation, attribute=True,
                count=self.tokens[relation].num_embeddings, dtype=h.dtype)
            states[relation] = F.relu(self.to_attribute[relation](self.pool(
                h[listing], value, sizes)))
            # An unknown attribute is not a shared relation.
            states[relation] = states[relation] * (torch.arange(
                len(states[relation]), device=h.device) != 0).unsqueeze(1)
        return states

    def encode(self, batch: GraphBatch, states: dict,
               text: torch.Tensor | None = None, *, initial=None) -> torch.Tensor:
        h = self.initial(batch, text) if initial is None else initial
        messages = []
        for relation in RELATIONS if self.graph_enabled else ():
            value, listing, sizes = topology(batch, relation, dtype=h.dtype)
            transformed = self.to_listing[relation](states[relation][value])
            # Autocast may change the linear output dtype. Keep the original
            # pooling arithmetic in that dtype as well.
            messages.append(self.pool(transformed, listing, sizes.to(transformed.dtype)))
        message = torch.stack(messages).mean(0) if self.graph_enabled else torch.zeros_like(h)
        return F.normalize(self.output(torch.cat([h, message], dim=-1)), dim=-1)


class PairScorer(nn.Module):
    """Trainable graph cosine calibration, with a direct text path in hybrid."""
    def __init__(self, hybrid: bool):
        super().__init__()
        self.head = nn.Linear(2 if hybrid else 1, 1)

    def forward(self, embeddings: torch.Tensor, pairs: torch.Tensor,
                text: torch.Tensor | None = None) -> torch.Tensor:
        return self.score(embeddings, pairs, text).logits

    @staticmethod
    def text_cosine(text: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        text = F.normalize(text, dim=-1)
        left, right = pairs.unbind(1)
        return (text[left] * text[right]).sum(-1)

    def score(self, embeddings: torch.Tensor, pairs: torch.Tensor,
              text: torch.Tensor | None = None, *, text_cosine: torch.Tensor | None = None) -> PairScores:
        left, right = pairs.unbind(1)
        cosine = (embeddings[left] * embeddings[right]).sum(-1)
        features = [cosine]
        if text_cosine is None and text is not None:
            text_cosine = self.text_cosine(text, pairs)
        if text_cosine is not None:
            if text_cosine.shape != cosine.shape or text_cosine.device != cosine.device:
                raise ValueError('prepared text pair cosine differs from graph pairs')
            features.append(text_cosine)
        return PairScores(self.head(torch.stack(features, dim=1)).squeeze(1), cosine)


@dataclass(frozen=True)
class PairScores:
    logits: torch.Tensor
    cosine: torch.Tensor
