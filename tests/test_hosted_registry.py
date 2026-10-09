"""The hosted-dataset registry owns the Kaggle/Colab dataset surface.

Every baked-in behavior of ``core.hosted_dataset`` is pinned here by ONE
focused test per public behavior: the declaration resolves (slugs, roles,
directions, members, attach, on_kaggle), lookups answer from the class, the
mount + staged-local addresses come from the one declared template, and the
module itself spells no slug (SSOT: the yaml is the only home).
"""
from pathlib import Path

import pytest

from core.hosted_dataset import hosted_registry, hosted_registry_spec


def test_the_registry_resolves_every_declared_entry_and_speaks_the_vocabularies():
    """The class is the SSOT it declares: entries, roles and direction facts match."""
    registry, spec = hosted_registry(), hosted_registry_spec()
    assert registry.slugs() == tuple(entry.slug for entry in spec.datasets)
    assert [entry.role for entry in registry.entries.values()] == list(spec.roles)
    for entry in registry.entries.values():
        # Kaggle mounts a dataset under its handle; identity derives from the slug.
        assert entry.mount == entry.handle
        assert entry.slug == f"{entry.owner}/{entry.handle}"
        assert entry.is_input or entry.is_output


def test_lookups_answer_by_slug_role_direction_and_kernel():
    registry = hosted_registry()
    assert registry.entry("fbarulli/er-laya-train").role == "corpus"
    assert registry.by_role("ckpt").slug == "fbarulli/er-laya-finetune-ckpt"
    assert [e.role for e in registry.by_direction("output")] == ["decisions", "ckpt"]
    assert [e.role for e in registry.kernel_inputs("finetune-eval")] == ["corpus", "ckpt"]
    assert [e.role for e in registry.kernel_outputs("finetune")] == ["ckpt"]
    # The two datasets that do NOT exist on Kaggle yet are declared, not gated.
    assert {e.slug for e in registry.pending_on_kaggle()} == {
        "fbarulli/er-laya-holdout", "fbarulli/er-laya-finetune-ckpt"}


def test_addresses_render_from_the_one_template_and_refuse_unknown_members():
    registry = hosted_registry()
    slug = "fbarulli/er-laya-train"
    assert registry.mount_path(slug) == Path("/kaggle/input/er-laya-train")
    assert registry.mount_member(slug, "train.jsonl") == Path(
        "/kaggle/input/er-laya-train/train.jsonl")
    assert registry.local_path(slug) == (
        registry.staging_root / "er-laya-train")
    with pytest.raises(KeyError, match="declares no member"):
        registry.local_member(slug, "not_a_member.jsonl")
    with pytest.raises(KeyError, match="unknown hosted dataset"):
        registry.entry("owner/not-declared")
    with pytest.raises(KeyError, match="unknown kernel kind"):
        registry.kernel_inputs("not-a-kernel")


def test_the_module_spells_no_slug_the_yaml_declares():
    """SSOT guard: the registry code must never re-spell a declared slug."""
    source = (Path(__file__).parents[1] / "src" / "core" / "hosted_dataset.py").read_text(
        encoding="utf-8")
    for entry in hosted_registry().entries.values():
        assert entry.slug not in source
        assert entry.handle not in source
