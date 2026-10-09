"""Public-API tests for the class-based ER/laya dataset registry (SSOT)."""
from __future__ import annotations

import pytest

from core.laya_datasets import LayaDataset, LayaDatasets


def test_registry_declares_the_full_regression_surface():
    assert LayaDatasets.slugs() == (
        "fbarulli/er-laya-base",
        "fbarulli/er-laya-train",
        "fbarulli/er-laya-requests",
        "fbarulli/er-10k-bundle",
        "fbarulli/reviews",
        "fbarulli/er-laya-decisions",
        "fbarulli/er-laya-holdout",
        "fbarulli/er-laya-finetune-ckpt",
    )


def test_every_registered_slug_is_unique():
    slugs = LayaDatasets.slugs()
    assert len(set(slugs)) == len(slugs)
    assert len(set(LayaDatasets.keys())) == len(slugs)


def test_existence_flags_match_the_observed_kaggle_state():
    # `kaggle datasets list --mine` (2026-10-08): only these five exist.
    assert {member.slug for member in LayaDatasets.existing()} == {
        "fbarulli/er-laya-base", "fbarulli/er-laya-train",
        "fbarulli/er-laya-requests", "fbarulli/er-10k-bundle",
        "fbarulli/reviews",
    }
    # the checkpoint transport the eval-only kernels reference is NOT published
    assert {member.slug for member in LayaDatasets.missing()} == {
        "fbarulli/er-laya-decisions", "fbarulli/er-laya-holdout",
        "fbarulli/er-laya-finetune-ckpt",
    }
    assert LayaDatasets.FINETUNE_CKPT.exists is False


def test_get_and_by_slug_round_trip_every_member():
    for member in LayaDatasets.all():
        assert LayaDatasets.get(member.key) is member
        assert LayaDatasets.by_slug(member.slug) is member
        assert member.owner == "fbarulli"
        assert member.name and "/" not in member.name
        assert isinstance(member, LayaDataset)


def test_unknown_key_and_slug_fail_loud():
    with pytest.raises(KeyError, match="unknown laya dataset"):
        LayaDatasets.get("nope")
    with pytest.raises(KeyError, match="not a registered laya dataset"):
        LayaDatasets.by_slug("owner/nope")


def test_registry_is_a_static_namespace_never_instantiated():
    with pytest.raises(TypeError, match="static registry"):
        LayaDatasets()
