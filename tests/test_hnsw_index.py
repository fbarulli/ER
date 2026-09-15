from pathlib import Path

import numpy as np

from core.ann_config import load_ann_config
from training.hnsw_index import PersistentHnswIndex


def test_ann_config_is_normalized_ip_hnsw() -> None:
    cfg = load_ann_config()
    assert cfg.embedding.loss == "mnrl"
    assert cfg.smoke.sample == 100
    assert cfg.smoke.epochs == 1
    assert cfg.index.space == "ip"
    assert cfg.index.ef_construction == 200
    assert cfg.index.M == 32
    assert cfg.index.ef_search == 100
    assert cfg.index.top_k == 50


def test_hnsw_round_trip_preserves_catalog_identity(tmp_path: Path) -> None:
    embeddings = np.eye(4, dtype=np.float32)
    ids = ["gtin-a", "gtin-b", "gtin-c", "gtin-d"]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    index = PersistentHnswIndex(
        tmp_path / "index",
        ef_construction=200,
        M=32,
        ef_search=100,
    )
    index.build(
        embeddings,
        ids,
        checkpoint=checkpoint,
        model_name="ann-smoke",
    )

    restored = PersistentHnswIndex(
        tmp_path / "index",
        ef_construction=200,
        M=32,
        ef_search=100,
    )
    restored.load(
        ids=ids,
        dim=4,
        checkpoint=checkpoint,
        model_name="ann-smoke",
    )
    labels, distances = restored.query(embeddings, top_k=2)

    assert labels.shape == (4, 2)
    assert labels[:, 0].tolist() == [0, 1, 2, 3]
    assert np.all(distances[:, 0] < 1e-5)
