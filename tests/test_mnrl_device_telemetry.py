import pytest
import torch

from training.losses import _tracking_mnrl_loss
from sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking import MultipleNegativesRankingLoss


def make_loss(warmup=False):
    return _tracking_mnrl_loss(None, monitoring_enabled=True, warmup_enabled=warmup,
                               warmup_epochs=2, twin_weight=0.3)


def test_monitoring_preserves_objective_and_gradients_with_complete_epoch_counts():
    torch.manual_seed(7)
    loss = make_loss()
    loss.set_triple_populations(['base', 'twin', 'masked'])
    loss.set_epoch(1)
    expected = dict(base=[], twin=[], masked=[], unknown=[])
    buffer = None
    for batch in range(5):
        embeddings = [torch.randn(4, 8, dtype=torch.float64, requires_grad=True) for _ in range(2)]
        references = [value.detach().clone().requires_grad_() for value in embeddings]
        ids = torch.tensor([0, 1, 2, 99])
        loss.set_batch_pair_ids(ids)
        actual = loss.compute_loss_from_embeddings(embeddings, None)
        plain_loss = MultipleNegativesRankingLoss(None)
        plain = plain_loss.compute_loss_from_embeddings(references, None)
        torch.testing.assert_close(actual, plain, rtol=0, atol=0)
        actual.backward()
        plain.backward()
        for left, right in zip(embeddings, references):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        scores = plain_loss.similarity_fct(references[0], references[1]) * plain_loss.scale
        values = -(scores.diag() - torch.logsumexp(scores, dim=1))
        for population, value in zip(expected, values.detach().tolist()):
            expected[population].append(value)
        if buffer is None:
            buffer = loss._pending_subset
        assert loss._pending_subset is buffer
        assert buffer.shape == (2, 4) and not buffer.requires_grad and buffer.grad_fn is None
        assert loss._subset_totals == {}
    loss.set_epoch(2)
    assert loss._pending_subset is None
    rows = loss.mnrl_subset_rows_by_epoch()
    for row in rows:
        values = expected[row['population']]
        assert row['epoch'] == 1 and row['triple_count'] == 5
        assert row['mean_loss'] == pytest.approx(sum(values) / len(values))
    # Reporting is idempotent: no double counting when the dashboard refreshes.
    assert loss.mnrl_subset_rows_by_epoch() == rows


def test_vectorized_warmup_preserves_weighted_objective_and_gradients():
    torch.manual_seed(19)
    embeddings = [torch.randn(3, 8, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    references = [value.detach().clone().requires_grad_() for value in embeddings]
    loss = make_loss(warmup=True)
    loss.set_triple_populations(['base', 'twin', 'masked'])
    loss.set_epoch(1)
    loss.set_batch_pair_ids(torch.tensor([0, 1, 2]))
    actual = loss.compute_loss_from_embeddings(embeddings, None)
    plain_loss = MultipleNegativesRankingLoss(None)
    scores = plain_loss.similarity_fct(references[0], references[1]) * plain_loss.scale
    row_losses = -(scores.diag() - torch.logsumexp(scores, dim=1))
    expected = torch.stack([row_losses[0], row_losses[1] * 0.3, row_losses[2]]).mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    for left, right in zip(embeddings, references):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
