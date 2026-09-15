from __future__ import annotations

import unittest

import pandas as pd

from training.sample_deduped_dataset import _sample


class SampleDedupedDatasetTests(unittest.TestCase):
    def test_stratified_sample_is_deterministic_and_unique(self) -> None:
        population = pd.DataFrame(
            [
                {
                    "product_id": f"p{index}",
                    "retailer": "r1",
                    "country": "gb",
                    "brand": "a" if index < 3 else "b",
                    "category": "juice" if index < 3 else "water",
                    "attributes": "flavor:orange" if index < 3 else "volume:500ml",
                }
                for index in range(6)
            ]
        )
        first, allocation = _sample(population, 3, 42)
        second, _ = _sample(population, 3, 42)
        self.assertEqual(first["product_id"].tolist(), second["product_id"].tolist())
        self.assertEqual(len(first), 3)
        self.assertEqual(sum(allocation.values()), 3)
        self.assertEqual(first["product_id"].nunique(), 3)

    def test_sample_rejects_size_larger_than_population(self) -> None:
        population = pd.DataFrame(
            {
                "product_id": ["a", "b", "c", "d"],
                "retailer": ["r"] * 4,
                "country": ["gb"] * 4,
                "brand": ["a"] * 4,
                "category": ["juice"] * 4,
                "attributes": [""] * 4,
            }
        )
        with self.assertRaisesRegex(ValueError, "sample size"):
            _sample(population, 5, 42)


if __name__ == "__main__":
    unittest.main()
