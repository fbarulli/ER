"""Regression checks for the inverted similarity heads found in the pair audit."""
import json

import pytest
import torch

from graph_tracks.model import PairScorer


@pytest.mark.parametrize('hybrid', [False, True])
@pytest.mark.parametrize('seed', [7, 42, 1337])
def test_higher_similarity_increases_new_match_score(hybrid, seed):
    torch.manual_seed(seed)
    scorer = PairScorer(hybrid)
    vectors = torch.tensor([[1., 0.], [0.6, 0.8], [1., 0.]])
    pairs = torch.tensor([[0, 1], [0, 2]])
    logits = scorer.score(vectors, pairs,
                          text_cosine=torch.tensor([.7, .7]) if hybrid else None).logits
    assert logits[1] > logits[0]
    if hybrid:
        logits = scorer.score(vectors, pairs,
                              text_cosine=torch.tensor([.7, 1.])).logits
        same_graph = scorer.score(vectors, pairs,
                                  text_cosine=torch.tensor([.7, .7])).logits
        assert logits[1] > same_graph[1]


@pytest.mark.parametrize('hybrid', [False, True])
def test_optimizer_projection_prevents_similarity_inversion(hybrid):
    scorer = PairScorer(hybrid)
    optimizer = torch.optim.SGD(scorer.parameters(), lr=10.)
    # An adverse update would invert every similarity coefficient.
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


def test_historical_signed_checkpoint_scores_are_preserved():
    scorer = PairScorer(True)
    scorer.load_state_dict({'head.weight': torch.tensor([[.546, -.163]]),
                            'head.bias': torch.tensor([.231])})
    vectors = torch.tensor([[1., 0.], [1., 0.]])
    pairs = torch.tensor([[0, 1]])
    result = scorer.score(vectors, pairs, text_cosine=torch.tensor([.99])).logits
    torch.testing.assert_close(result, torch.tensor([.546 - .163 * .99 + .231]))
    assert scorer.head.weight[0, 1] < 0


@pytest.mark.parametrize('hybrid', [False, True])
def test_training_checkpoints_and_epoch_metrics_keep_monotonic_head(tmp_path, monkeypatch, hybrid):
    from test_graph_tracks import inputs, disable_tracking
    from graph_tracks.train import train
    disable_tracking(monkeypatch)
    _, _, _, config = inputs(tmp_path, hybrid)
    # Force an inverted update at each epoch to check training integration,
    # rather than relying on this small fixture to naturally invert a head.
    original_step = torch.optim.AdamW.step

    def adverse_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.shape == (1, 2 if hybrid else 1):
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
        assert head['text_cosine_weight'] is None or head['text_cosine_weight'] >= 0
