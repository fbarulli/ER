import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from graph_tracks.data import fit_vocabulary, tensorize
from graph_tracks.model import AttributeGNN, PairScorer
from test_graph_tracks import population


def test_split_only_encoding_preserves_scores_losses_and_gradients(hybrid=False):
    records = population()
    vocabulary = fit_vocabulary(records)
    support_ids = [i for i, row in enumerate(records) if row['split'] == 'train']
    dev_ids = [i for i, row in enumerate(records) if row['split'] == 'dev']
    full = tensorize(records, vocabulary, 'cpu')
    support = tensorize([records[i] for i in support_ids], vocabulary, 'cpu')
    dev = tensorize([records[i] for i in dev_ids], vocabulary, 'cpu')
    torch.manual_seed(7)
    text = torch.randn(len(records), 6) if hybrid else None
    support_text = text[support_ids] if hybrid else None
    dev_text = text[dev_ids] if hybrid else None
    original = AttributeGNN(vocabulary, 8, 8, 6 if hybrid else 0)
    optimized = copy.deepcopy(original)
    scorer = PairScorer(hybrid)
    optimized_scorer = copy.deepcopy(scorer)
    global_pairs = torch.tensor([[support_ids[0], support_ids[1]], [support_ids[0], support_ids[2]]])
    local_pairs = torch.tensor([[0, 1], [0, 2]])
    labels = torch.tensor([1., 0.])

    def step(model, head, batch, pairs, query_text):
        embeddings = model.encode(batch, model.context(support, support_text), query_text)
        logits = head(embeddings, pairs, query_text)
        left, right = pairs.unbind(1)
        cos = (embeddings[left] * embeddings[right]).sum(-1)
        metric = (labels * (1 - cos) + (1 - labels) * F.relu(cos - .2)).mean()
        loss = F.binary_cross_entropy_with_logits(logits, labels) + .1 * metric
        loss.backward()
        return logits, loss

    old_scores, old_loss = step(original, scorer, full, global_pairs, text)
    new_scores, new_loss = step(optimized, optimized_scorer, support, local_pairs, support_text)
    torch.testing.assert_close(old_scores, new_scores, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(old_loss, new_loss, rtol=1e-5, atol=1e-6)
    for old, new in zip(list(original.parameters()) + list(scorer.parameters()),
                        list(optimized.parameters()) + list(optimized_scorer.parameters())):
        if old.grad is not None:
            torch.testing.assert_close(old.grad, new.grad, rtol=1e-4, atol=1e-6)
    with torch.no_grad():
        context = original.context(support, support_text)
        old_dev = original.encode(full, context, text)[dev_ids]
        new_dev = original.encode(dev, context, dev_text)
    torch.testing.assert_close(old_dev, new_dev, rtol=1e-5, atol=1e-6)
