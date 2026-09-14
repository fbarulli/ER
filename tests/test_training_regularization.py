from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from core.common import training_cfg
from training import training


class TrainingRegularizationTests(unittest.TestCase):
    def test_regularization_config_is_conservative_and_validated(self) -> None:
        cfg = training_cfg().training
        self.assertAlmostEqual(cfg.weight_decay, 0.01)
        self.assertAlmostEqual(cfg.projection_dropout, 0.10)
        self.assertAlmostEqual(cfg.label_smoothing, 0.05)
        self.assertTrue(cfg.random_easy_negatives.enabled)
        self.assertAlmostEqual(cfg.random_easy_negatives.ratio_to_hard, 1.0)

    def test_projection_dropout_is_idempotent_and_eval_safe(self) -> None:
        model = nn.Sequential()
        self.assertTrue(training._configure_projection_dropout(model, 0.10))
        self.assertFalse(training._configure_projection_dropout(model, 0.20))
        self.assertEqual(len(model), 1)
        model.eval()
        features = {"sentence_embedding": torch.ones(2, 4)}
        self.assertTrue(
            torch.equal(model(features)["sentence_embedding"], torch.ones(2, 4))
        )

    def test_zero_smoothing_preserves_online_contrastive_arithmetic(self) -> None:
        positives = torch.tensor([0.10, 0.30])
        negatives = torch.tensor([0.05, 0.30])
        positive_loss, negative_loss, hinge = training._smoothed_contrastive_losses(
            positives, negatives, margin=0.20, label_smoothing=0.0
        )
        self.assertAlmostEqual(positive_loss.item(), float((positives**2).sum()))
        self.assertAlmostEqual(negative_loss.item(), float((hinge**2).sum()))

    def test_random_easy_mixing_is_deterministic_ratio_exact_and_split_safe(
        self,
    ) -> None:
        candidates = np.asarray([[2, 3], [3, 4]], dtype=int)
        hard = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=int)
        sources = np.asarray(["gate", "gate", "attribute_conflict"], dtype=object)
        row_bc = np.asarray(["a", "b", "c", "d", "e"], dtype=object)
        kwargs = {
            "df": pd.DataFrame({"barcode": row_bc}),
            "row_bc": row_bc,
            "train_barcodes": set(row_bc),
            "seed": 17,
            "enabled": True,
            "ratio_to_hard": 1.0,
            "candidate_pool_size": 10,
        }
        with patch.object(
            training,
            "_split_safe_random_negative_pairs",
            return_value=candidates.copy(),
        ):
            first = training._mix_random_easy_training_negatives(
                hard, sources, **kwargs
            )
            second = training._mix_random_easy_training_negatives(
                hard, sources, **kwargs
            )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[0][: len(hard)], hard)
        self.assertEqual(first[2], 2)
        self.assertEqual(len(first[0]), 6)
        self.assertEqual(list(first[1]).count("random_easy"), len(hard))
        self.assertTrue(training.pairs_in_set(first[0], row_bc, set(row_bc)).all())


if __name__ == "__main__":
    unittest.main()
