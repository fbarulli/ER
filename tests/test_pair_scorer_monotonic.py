"""Regression checks for the single-cosine gnn_only pair scorer.

The fused graph/text hybrid head is retired; these checks pin the surviving
monotonic single-cosine head and that the fused two-input head can no longer be
built at all.
"""
import json

import pytest
import torch

from graph_tracks.model import PairScorer


def test_higher_similarity_increases_new_match_score():
    torch.manual_seed(1337)
    scorer = PairScorer()
    vectors = torch.tensor([[1., 0.], [0.6, 0.8], [1., 0.]])
    pairs = torch.tensor([[0, 1], [0, 2]])
    logits = scorer.score(vectors, pairs).logits
    assert logits[1] > logits[0]


def test_optimizer_projection_prevents_similarity_inversion():
    scorer = PairScorer()
    optimizer = torch.optim.SGD(scorer.parameters(), lr=10.)
    # An adverse update would invert the similarity coefficient.
    scorer.head.weight.sum().backward()
    optimizer.step()
    assert (scorer.head.weight < 0).all()
    scorer.project_similarity_weights()
    assert (scorer.head.weight >= 0).all()
    # Projection must not disconnect the weights from learning.
    optimizer.zero_grad()
    (-scorer.head.weight.sum()).backward()
    optimizer.step()
    scorer.project_similarity_weights()
    assert (scorer.head.weight > 0).all()


def test_fused_hybrid_head_is_retired():
    with pytest.raises(ValueError, match='fused-hybrid pair scorer is retired'):
        PairScorer(True)


def test_calibration_metrics_report_only_the_graph_cosine():
    scorer = PairScorer()
    metrics = scorer.calibration_metrics()
    assert metrics['policy'] == 'gnn_single_cosine_v1'
    assert 'graph_cosine_weight' in metrics
    assert 'text_cosine_weight' not in metrics


def test_training_checkpoints_and_epoch_metrics_keep_monotonic_head(tmp_path, monkeypatch):
    from test_graph_tracks import inputs, disable_tracking
    from graph_tracks.train import train
    disable_tracking(monkeypatch)
    _, _, _, config = inputs(tmp_path)
    # Force an inverted update at each epoch to check training integration,
    # rather than relying on this small fixture to naturally invert a head.
    original_step = torch.optim.AdamW.step

    def adverse_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.shape == (1, 1):
                        parameter.fill_(-1.)
        return result

    monkeypatch.setattr(torch.optim.AdamW, 'step', adverse_step)
    checkpoint = train(config, run_tag='monotonic')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert (payload['scorer']['head.weight'] >= 0).all()
    metric_path = next((tmp_path/'run').rglob('*epoch_metrics.jsonl'))
    metrics = [json.loads(line) for line in metric_path.read_text().splitlines()]
    assert len(metrics) == 2
    for epoch in metrics:
        head = epoch['scorer_calibration']
        assert head['graph_cosine_weight'] >= 0
        assert 'text_cosine_weight' not in head
