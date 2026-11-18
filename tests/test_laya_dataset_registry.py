"""Pin: the laya lane's corpus selection exposes EXACTLY smoke + full.

Owner directive (2026-10-09): the laya lane uses exactly two corpora — the
carved full corpus (production) and the smoke subset — and nothing else. This
pins the public contract of :class:`core.laya_datasets.LayaCorpora` (the
selection SSOT) and the ``LayaSpec`` bindings that read it.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.laya_config import LayaSpec
from core.laya_datasets import LayaCorpora


def test_laya_corpus_selection_offers_only_smoke_or_full() -> None:
    assert LayaCorpora.keys() == ("full", "smoke")
    assert LayaCorpora.slugs() == (LayaCorpora.FULL.slug, LayaCorpora.SMOKE.slug)
    assert LayaCorpora.select("full") is LayaCorpora.FULL
    assert LayaCorpora.select("smoke") is LayaCorpora.SMOKE
    for stale in ("3k", "50pct", "10k", "nope"):
        with pytest.raises(ValueError, match="unknown laya corpus kind"):
            LayaCorpora.select(stale)
    spec = LayaSpec()
    assert spec.finetune_dataset_slug == LayaCorpora.FULL.slug
    assert spec.finetune_smoke.dataset_slug == LayaCorpora.SMOKE.slug
    # a rogue corpus binding is unrepresentable (fail loud at parse).
    with pytest.raises(ValidationError, match="not a registered corpus"):
        LayaSpec(finetune_dataset_slug="fbarulli/er-laya-3k")
