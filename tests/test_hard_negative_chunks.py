"""Chunked mining must preserve full-block candidates and gather only once."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.hard_negatives import mine_hard_negatives


class GatherCounter(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values).view(cls)
        obj.gathers = []
        return obj

    def __array_finalize__(self, parent):
        self.gathers = getattr(parent, "gathers", [])

    def __getitem__(self, key):
        if isinstance(key, np.ndarray):
            self.gathers.append(len(key))
        return np.asarray(super().__getitem__(key))


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 40])
@pytest.mark.parametrize("k", [3, 40])
def test_mining_chunks_equal_full_block(monkeypatch, chunk_size, k):
    from core import common, gtin

    cfg = SimpleNamespace(chunk_size=40)
    monkeypatch.setattr(common, "training_cfg", lambda: SimpleNamespace(
        mining=SimpleNamespace(ann=cfg)))
    monkeypatch.setattr(common, "category_macros", lambda: {"a": "A", "b": "B"})
    monkeypatch.setattr(gtin, "barcode_validity", lambda values: values.ne(""))
    n = 23
    frame = pd.DataFrame({
        "barcode": [str(i // 2) if i != 4 else "" for i in range(n)],
        "brand": [f"brand{i % 4}" for i in range(n)],
        "category": ["a" if i % 3 else "b" for i in range(n)],
    })
    rng = np.random.default_rng(23)
    values = rng.normal(size=(n, 9))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    options = dict(n_target=1000, cosine_lo=-1., cosine_hi=1.,
                   exclude_conflicting=False, k=k,
                   max_per_canonical=1000, max_per_brand=1000)
    expected_pairs, expected_scores = mine_hard_negatives(frame, values, **options)
    cfg.chunk_size = chunk_size
    counted = GatherCounter(values)
    pairs, scores = mine_hard_negatives(frame, counted, **options)
    np.testing.assert_array_equal(pairs, expected_pairs)
    np.testing.assert_allclose(scores, expected_scores, atol=1e-15, rtol=0)
    # Interleaved macro rows require advanced indexing; each macro is gathered
    # once, irrespective of how many query chunks it contains.
    assert sorted(counted.gathers) == [8, 15]
