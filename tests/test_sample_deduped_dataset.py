from __future__ import annotations

import unittest

import pandas as pd

from training.sample_deduped_dataset import _sample_half


class SampleDedupedDatasetTests(unittest.TestCase):
    def test_half_sample_is_deterministic_and_audits_odd_strata(self) -> None:
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
        first, allocation, audit = _sample_half(population, 3, 42)
        second, _, _ = _sample_half(population, 3, 42)
        self.assertEqual(first["product_id"].tolist(), second["product_id"].tolist())
        self.assertEqual(len(first), 3)
        self.assertEqual(sum(row["retained_rows"] for row in audit), 3)
        self.assertTrue(
            all(
                row["retained_rows"]
                in {row["population_rows"] // 2, (row["population_rows"] + 1) // 2}
                for row in audit
            )
        )
        self.assertEqual(sum(allocation.values()), 3)

    def test_half_sample_rejects_non_half_size(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "exactly half"):
            _sample_half(population, 3, 42)


if __name__ == "__main__":
    unittest.main()
