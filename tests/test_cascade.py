"""Cascade composition and per-role metric checks on hand-built cases.

Two properties only:
  (a) retrieve-then-rerank surfaces a candidate the text ranker placed low but
      the decider scores high -- proving the ranker/decider split works and
      that the cascade differs from either role alone;
  (b) the ranker (recall) and decider (PR-AUC / precision@recall / ECE) helpers
      return the hand-checked values.
"""
import numpy as np
import pytest
import torch

from graph_tracks.model import PairScorer
from model_tracks.cascade import (
    CascadeIndex, Query, candidate_recall, cascade, decide, decider_report,
    expected_calibration_error, micro_recall_at_k, precision_at_recall,
    ranker_report, retrieved_relevance,
)

CATALOG = ('A', 'B', 'C', 'D', 'E')


class CosineFakeScorer(torch.nn.Module):
    """A scorer with no learned head: logits are the graph cosine itself."""

    def forward(self, embeddings, pairs, text=None):
        left, right = pairs.unbind(1)
        return (embeddings[left] * embeddings[right]).sum(-1)


def synthetic_index(tmp_path):
    # Text geometry: A and B are the only candidates the ranker will retrieve
    # at k=2; the truly best graph match C sits far down the text ranking.
    text = np.asarray([
        [1.0, 0.0],    # A: cos 1.00
        [0.98, 0.2],   # B: cos ~0.98
        [0.1, 0.995],  # C: cos ~0.10
        [0.0, 1.0],    # D: cos 0
        [-1.0, 0.0],   # E: cos -1
    ], dtype=np.float32)
    # Graph geometry: within the text-retrieved {A, B} the decider prefers B,
    # while globally it prefers C (which text never retrieved).
    graph = torch.tensor([
        [0.2, 0.9798],  # A
        [0.9, 0.4359],  # B
        [1.0, 0.0],     # C
        [0.0, 1.0],     # D
        [-1.0, 0.0],    # E
    ])
    index = CascadeIndex.from_arrays(text, graph, CATALOG, directory=tmp_path / 'ann')
    query = Query(ids=('Q',), text=np.asarray([[1.0, 0.0]], dtype=np.float32),
                  graph=torch.tensor([[1.0, 0.0]]))
    return index, query


def test_cascade_reranks_beyond_the_text_ranking(tmp_path):
    index, query = synthetic_index(tmp_path)
    k = 2
    ranked = cascade(query, index, CosineFakeScorer(), k)

    # Ranker alone: top-2 by text similarity is A then B; C is not retrieved.
    text_ranked = sorted(range(len(CATALOG)),
                         key=lambda i: -float(np.dot(query.text[0], index.text_index.embeddings[i])))
    assert [CATALOG[i] for i in text_ranked[:k]] == ['A', 'B']
    assert ranked.ranked_ids(0) == ['B', 'A']

    # Decider alone (global graph cosine) would pick C, which the ranker missed.
    graph_ranked = sorted(range(len(CATALOG)),
                          key=lambda i: -float(torch.cosine_similarity(
                              query.graph[0], index.graph[i], dim=0)))
    assert CATALOG[graph_ranked[0]] == 'C'

    # The cascade disagrees with both roles alone.
    assert ranked.ranked_ids(0)[0] == 'B'
    assert ranked.ranked_ids(0)[0] != 'A'   # text-only
    assert ranked.ranked_ids(0)[0] != 'C'   # gnn-only global
    best, runner_up = ranked.order[0][:2]
    assert ranked.scores[0][best] > ranked.scores[0][runner_up]  # B beats A after rerank


def test_cascade_reuses_the_trained_pair_scorer(tmp_path):
    index, query = synthetic_index(tmp_path)
    # PairScorer in gnn_only mode has weight 1 and bias 0, so its monotonic
    # logits reproduce the cosine order -- the cascade consumes it unchanged.
    fake = cascade(query, index, CosineFakeScorer(), 2)
    reused = cascade(query, index, PairScorer(False), 2)
    assert reused.ranked_ids(0) == fake.ranked_ids(0) == ['B', 'A']


def test_decide_is_scorer_agnostic():
    from model_tracks.cascade import CandidateBatch
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pairs = torch.tensor([[0, 1], [0, 2]])
    batch = CandidateBatch(query_ids=('Q',), candidate_ids=np.asarray([['X', 'Y']], dtype=object),
                           pairs=pairs, embeddings=embeddings)
    result = decide(batch, CosineFakeScorer())
    assert result.ranked_ids(0) == ['X', 'Y']
    assert result.scores[0][0] > result.scores[0][1]


def test_ranker_role_metrics_hand_checked():
    retrieved = [['A', 'B', 'C'], ['D', 'E', 'F']]
    relevant = [{'B'}, {'D'}]
    assert candidate_recall(retrieved, relevant, 1) == pytest.approx(0.5)
    assert candidate_recall(retrieved, relevant, 2) == pytest.approx(1.0)
    assert micro_recall_at_k(retrieved, relevant, 1) == pytest.approx(0.5)
    assert micro_recall_at_k(retrieved, relevant, 2) == pytest.approx(1.0)
    report = ranker_report(retrieved, relevant, (1, 2))
    assert report['candidate_recall_at_1'] == pytest.approx(0.5)
    assert report['candidate_recall_at_2'] == pytest.approx(1.0)
    assert report['recall_at_2'] == pytest.approx(1.0)


def test_decider_role_metrics_hand_checked():
    labels = [1, 0, 0, 1]
    scores = [0.9, 0.8, 0.2, 0.1]
    # AP = (0.5-0)*1.0 + (1.0-0.5)*0.5 = 0.75 (positives at ranks 1 and 4).
    # Recall 1.0 is only reachable by keeping all four candidates (precision 0.5).
    assert precision_at_recall(labels, scores, 0.95) == pytest.approx(0.5)
    # ECE with 2 equal-width bins: two bins, each |0.15-0.5|=0.35, weighted 0.5.
    assert expected_calibration_error(labels, scores, bins=2) == pytest.approx(0.35)
    report = decider_report(labels, scores, recall_targets=(0.95,), bins=2)
    assert report['pr_auc'] == pytest.approx(0.75)
    assert report['precision_at_recall_0.95'] == pytest.approx(0.5)
    assert report['ece'] == pytest.approx(0.35)
    assert report['positive_pairs'] == 2
    assert report['both_classes'] is True
    # A single-class candidate set has no defined PR-AUC, not a fake 0.0.
    assert decider_report([1, 1], [0.9, 0.8])['pr_auc'] is None


def test_retrieved_relevance_flattens_the_decider_view(tmp_path):
    index, query = synthetic_index(tmp_path)
    result = cascade(query, index, CosineFakeScorer(), 2)
    labels, scores = retrieved_relevance(result, [{'B'}])
    assert labels.tolist() == [0, 1]        # A irrelevant, B relevant
    assert scores.shape == (2,)
    assert scores[1] > scores[0]
