"""Retrieve-then-rerank cascade over a trained text ranker and a trained GNN decider.

The cascade is a COMBINATOR, not a third trained model.  The text encoder is
consumed exactly as trained to generate recall-oriented ANN candidates, and the
``gnn_only`` pair scorer is consumed exactly as trained to make the
precision-oriented decision over *those* candidates.  Nothing is re-fused and
no embedding is re-projected here: the two roles stay separate and measurable.

Evidence for the split: the text ranker leads retrieval geometry while
``gnn_only`` is the strongest pair scorer but a weak retriever; a fixed 50/50
hybrid fusion of the two landed below both.  The cascade keeps each component
in the role it wins.

Device-agnostic (``cpu`` or ``cuda``) and config-free: callers pass the values
(``k``, thresholds, k-ladders, bins).  The text ANN is the existing
:class:`training.hnsw_index.PersistentHnswIndex` (hnswlib-backed) and the
decider is any module with ``forward(embeddings, pairs, text)`` returning
logits -- normally :class:`graph_tracks.model.PairScorer`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from training.hnsw_index import PersistentHnswIndex


# ---------------------------------------------------------------------------
# Query / catalog containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Query:
    """One or more queries carrying BOTH retrieval text and decider geometry.

    ``text`` is the ranker-space embedding the text ANN searches over.
    ``graph`` is the ``gnn_only`` encoding the pair scorer consumes.  They are
    deliberately separate spaces: the cascade never fuses them.
    """

    ids: tuple[str, ...]
    text: np.ndarray
    graph: torch.Tensor
    text_pair_cosine: torch.Tensor | None = None

    def __post_init__(self):
        text = np.asarray(self.text, dtype=np.float32)
        if text.ndim != 2 or len(text) != len(self.ids):
            raise ValueError('query text must be (n_queries, dim) aligned with ids')
        if self.graph.ndim != 2 or len(self.graph) != len(self.ids):
            raise ValueError('query graph must be (n_queries, decider_dim) aligned with ids')
        object.__setattr__(self, 'text', text)

    @property
    def count(self) -> int:
        return len(self.ids)


@dataclass
class CascadeIndex:
    """A text ANN plus the catalog ``gnn_only`` geometry it retrieves into.

    The ANN is built and queried by :class:`PersistentHnswIndex`; the graph
    rows are indexed by the same catalog IDs so a retrieved candidate can be
    handed to the pair scorer without re-encoding anything.
    """

    text_index: PersistentHnswIndex
    graph: torch.Tensor
    ids: tuple[str, ...]
    _lookup: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.graph = self.graph.detach().float()
        if self.graph.ndim != 2:
            raise ValueError('catalog graph embeddings must be 2-D')
        if len(self.ids) != len(self.graph):
            raise ValueError('catalog ids and graph embeddings must be aligned')
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('catalog ids must be unique')
        self._lookup = {value: i for i, value in enumerate(self.ids)}

    @classmethod
    def from_arrays(cls, text_vectors: np.ndarray, graph_vectors: torch.Tensor,
                    ids: Sequence[str], *, directory: Path, checkpoint: Path | None = None,
                    model_name: str = 'text', ef_construction: int = 200, M: int = 16,
                    ef_search: int = 100, build_index: bool = True) -> 'CascadeIndex':
        """Build the text ANN from embeddings, reusing the persistent index type."""
        index = PersistentHnswIndex(directory, ef_construction=ef_construction, M=M,
                                    ef_search=ef_search)
        if build_index:
            index.build(np.asarray(text_vectors, dtype=np.float32), list(ids),
                        checkpoint=checkpoint or Path(directory) / 'text_checkpoint',
                        model_name=model_name)
        else:
            index.ids = [str(value) for value in ids]
        return cls(text_index=index, graph=torch.as_tensor(graph_vectors), ids=tuple(ids))

    def row(self, identifier: str) -> int:
        try:
            return self._lookup[identifier]
        except KeyError as exc:
            raise KeyError(f'catalog does not contain {identifier!r}') from exc


# ---------------------------------------------------------------------------
# Role outputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ranked:
    """Text-ranker candidates; higher ``similarities`` means closer."""

    query_ids: tuple[str, ...]
    candidate_ids: np.ndarray          # (n_queries, k) object array
    similarities: np.ndarray           # (n_queries, k) float32


@dataclass(frozen=True)
class Decisions:
    """Decider scores over the retrieved candidates, with a reranked order."""

    query_ids: tuple[str, ...]
    candidate_ids: np.ndarray          # (n_queries, k) object array
    scores: np.ndarray                 # (n_queries, k) decision probability
    order: np.ndarray                  # (n_queries, k) argsort, best first
    similarities: np.ndarray | None = None

    def ranked_ids(self, query: int = 0) -> list[str]:
        row = np.asarray(self.candidate_ids[query], dtype=object)[self.order[query]]
        return [value for value in row.tolist() if value is not None]


@dataclass(frozen=True)
class CandidateBatch:
    """Everything the pair scorer needs for one retrieved candidate set."""

    query_ids: tuple[str, ...]
    candidate_ids: np.ndarray
    pairs: torch.Tensor
    embeddings: torch.Tensor
    similarities: np.ndarray | None = None
    text: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Role components
# ---------------------------------------------------------------------------
def rank(query, index: CascadeIndex, k: int, *, exclude_self: bool = False) -> Ranked:
    """Text ANN candidate generation (recall-oriented).

    Returns the top-``k`` catalog IDs per query together with the ANN
    similarities.  This is the ranker role only: no graph geometry is read
    here and no decision is made.
    """
    if k < 1:
        raise ValueError('k must be positive')
    if isinstance(query, Query):
        text = query.text
        query_ids = query.ids
    else:
        text = np.asarray(query, dtype=np.float32)
        if text.ndim == 1:
            text = text[None, :]
        query_ids = tuple(f'q{i}' for i in range(len(text)))
    request = k + 1 if exclude_self else k
    labels, distances = index.text_index.query(text, top_k=request)
    catalog = index.ids
    width = min(k, labels.shape[1])
    candidate_ids = np.full((len(query_ids), width), None, dtype=object)
    similarities = np.full((len(query_ids), width), np.nan, dtype=np.float32)
    for i in range(len(query_ids)):
        picked = 0
        for j in range(labels.shape[1]):
            candidate = catalog[int(labels[i, j])]
            if exclude_self and candidate == query_ids[i]:
                continue
            candidate_ids[i, picked] = candidate
            similarities[i, picked] = float(distances[i, j])
            picked += 1
            if picked == width:
                break
    return Ranked(query_ids=query_ids, candidate_ids=candidate_ids, similarities=similarities)


def pair_batch(query: Query, index: CascadeIndex, ranked: Ranked) -> CandidateBatch:
    """Bridge retrieved IDs to the pair scorer's geometry, without re-encoding."""
    count, width = ranked.candidate_ids.shape
    if count != query.count:
        raise ValueError('retrieved queries differ from the supplied query population')
    query_rows = np.repeat(np.arange(count), width)
    catalog_rows = np.empty((count, width), dtype=np.int64)
    for i in range(count):
        for j in range(width):
            identifier = ranked.candidate_ids[i, j]
            if identifier is None:
                raise ValueError('candidate set is not rectangular; increase the catalog or k')
            catalog_rows[i, j] = index.row(str(identifier))
    # Query rows occupy the first block of the combined geometry; catalog rows follow.
    right = len(query.graph) + catalog_rows.reshape(-1)
    pairs = torch.stack([torch.as_tensor(query_rows, dtype=torch.long),
                         torch.as_tensor(right, dtype=torch.long)], dim=1)
    embeddings = torch.cat([query.graph.detach().float(), index.graph], dim=0)
    return CandidateBatch(query_ids=query.ids, candidate_ids=ranked.candidate_ids,
                          pairs=pairs, embeddings=embeddings,
                          similarities=ranked.similarities)


def _module_device(module) -> torch.device:
    try:
        return next(module.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device('cpu')


def decide(candidates: CandidateBatch, scorer) -> Decisions:
    """GNN pair scorer over the retrieved candidates (precision-oriented).

    ``scorer`` is any module with ``forward(embeddings, pairs, text)`` returning
    logits, normally :class:`graph_tracks.model.PairScorer` in its ``gnn_only``
    (single-cosine) configuration.  The output is a probability per retrieved
    candidate plus the reranked order -- the decision stage only.
    """
    count, width = candidates.candidate_ids.shape
    device = _module_device(scorer)
    embeddings = candidates.embeddings.to(device)
    pairs = candidates.pairs.to(device)
    text = candidates.text.to(device) if candidates.text is not None else None
    with torch.no_grad():
        logits = scorer(embeddings, pairs, text)
    scores = torch.sigmoid(logits.reshape(count, width)).float().cpu().numpy()
    order = np.argsort(-scores, axis=1, kind='stable')
    return Decisions(query_ids=candidates.query_ids, candidate_ids=candidates.candidate_ids,
                     scores=scores, order=order, similarities=candidates.similarities)


def cascade(query, index: CascadeIndex, scorer, k: int) -> Decisions:
    """Retrieve-then-rerank composition returning ranked decisions.

    ``text`` ANN retrieves ``k`` recall-oriented candidates, then the ``gnn_only``
    scorer makes the precision-oriented decision over exactly those candidates.
    Self-retrieval is excluded so the decision is always between distinct listings.
    """
    if not isinstance(query, Query):
        raise TypeError('cascade requires a Query carrying both text and graph geometry')
    ranked = rank(query, index, k, exclude_self=True)
    return decide(pair_batch(query, index, ranked), scorer)


# ---------------------------------------------------------------------------
# Per-role evaluation
# ---------------------------------------------------------------------------
def candidate_recall(retrieved: Sequence[Collection[str]],
                     relevant: Sequence[Collection[str]], k: int) -> float:
    """Fraction of queries with at least one relevant ID in their top-``k``.

    The ranker's recall-oriented view: a query counts once whether one or many
    of its truths were retrieved.
    """
    if len(retrieved) != len(relevant):
        raise ValueError('retrieved and relevant must be aligned per query')
    if not retrieved:
        raise ValueError('candidate recall needs at least one query')
    if k < 1:
        raise ValueError('k must be positive')
    hits = sum(1 for ids, truth in zip(retrieved, relevant)
               if set(list(ids)[:k]) & set(truth))
    return hits / len(retrieved)


def micro_recall_at_k(retrieved: Sequence[Collection[str]],
                      relevant: Sequence[Collection[str]], k: int) -> float:
    """Relevant candidates recovered across all queries divided by all truths."""
    recovered = sum(len(set(list(ids)[:k]) & set(truth))
                    for ids, truth in zip(retrieved, relevant))
    total = sum(len(truth) for truth in relevant)
    return recovered / total if total else 0.0


def ranker_report(retrieved: Sequence[Collection[str]],
                  relevant: Sequence[Collection[str]],
                  ks: Iterable[int]) -> dict[str, float]:
    """Ranker role metrics: candidate recall and micro recall at each k."""
    report: dict[str, float] = {}
    for k in ks:
        report[f'candidate_recall_at_{k}'] = candidate_recall(retrieved, relevant, k)
        report[f'recall_at_{k}'] = micro_recall_at_k(retrieved, relevant, k)
    return report


def precision_at_recall(labels, scores, target: float) -> float | None:
    """Highest precision holding recall at or above ``target`` (``None`` if unreachable).

    ``precision_recall_curve`` appends a ``(precision=1, recall=0)`` sentinel;
    it is sliced off so an unreachable target is reported honestly.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if set(labels.tolist()) != {0, 1}:
        return None
    precision, recall, _ = precision_recall_curve(labels, scores)
    body = slice(0, len(precision) - 1)
    reachable = recall[body] >= target
    return float(precision[body][reachable].max()) if reachable.any() else None


def expected_calibration_error(labels, scores, bins: int = 10) -> float:
    """Equal-width expected calibration error of the decider probabilities.

    ONE implementation: delegates to ``training.advanced.expected_calibration_error``
    (audit MED 5 — a second binning routine had drifted from it) while keeping
    this module's loud input contract (aligned 1-D arrays, non-empty).
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError('labels and scores must be aligned one-dimensional arrays')
    if not len(labels):
        raise ValueError('calibration needs at least one decision')
    if bins < 1:
        raise ValueError('bins must be positive')
    from training.advanced import expected_calibration_error as _ece

    return _ece(scores, labels, n_bins=bins)


def decider_report(labels, scores, *, recall_targets: Sequence[float] = (0.95,),
                   bins: int = 10) -> dict[str, float | int | bool | None]:
    """Decider role metrics on the RETRIEVED candidate set.

    Inputs are the relevance labels and decision probabilities of the
    candidates the ranker handed over; PR-AUC, precision at the requested
    recall targets, and calibration (ECE) are all measured here and nowhere
    else.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    supported = set(labels.tolist()) == {0, 1}
    report: dict[str, float | int | bool | None] = {
        'rows': int(len(labels)),
        'positive_pairs': int(labels.sum()),
        'both_classes': supported,
        'pr_auc': float(average_precision_score(labels, scores)) if supported else None,
        'ece': expected_calibration_error(labels, scores, bins=bins),
    }
    for target in recall_targets:
        report[f'precision_at_recall_{target:g}'] = precision_at_recall(labels, scores, target)
    return report


def retrieved_relevance(decisions: Decisions, relevant: Sequence[Collection[str]]):
    """Flatten retrieved candidates into (labels, scores) for the decider role."""
    if len(relevant) != len(decisions.query_ids):
        raise ValueError('relevant sets must align with the decisions')
    labels: list[int] = []
    scores: list[float] = []
    for i, truth in enumerate(relevant):
        truth = set(truth)
        for j in range(len(decisions.candidate_ids[i])):
            identifier = decisions.candidate_ids[i, j]
            if identifier is None:
                continue
            labels.append(1 if identifier in truth else 0)
            scores.append(float(decisions.scores[i, j]))
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


__all__ = [
    'CandidateBatch', 'CascadeIndex', 'Decisions', 'Query', 'Ranked',
    'cascade', 'candidate_recall', 'decide', 'decider_report',
    'expected_calibration_error', 'micro_recall_at_k', 'pair_batch',
    'precision_at_recall', 'rank', 'ranker_report', 'retrieved_relevance',
]
