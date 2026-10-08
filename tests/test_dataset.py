"""The project Dataset owns the data surface: members, read spec, identity, transport.

Every baked-in behavior of ``core.dataset.Dataset`` is pinned here by ONE test:
member resolution equals the current SSOT accessors, the declared read spec
drives ``load``, ``identity`` is a content digest (stable, portable, content-
following, absence-tolerant), ``as_bundle`` hands off to ``Bundle.load``, and
the class carries no validity gate.
"""
import inspect
from pathlib import Path

import pandas as pd
import pytest

from core import common
from core.dataset import (
    SOURCE,
    Dataset,
    DatasetMemberSpec,
    DatasetSpec,
    dataset,
    dataset_spec,
)


def _injected(store: Dataset, **members: Path) -> Dataset:
    """A copy of ``store`` with one line-item member pointed at a fixture path."""
    return store.model_copy(update={"members": {**store.members, **members}})


# ── resolution: equals the current accessors ────────────────────────────────

def test_members_resolve_through_the_same_ssot_accessors():
    """``member(name)`` IS the ``files``/``layouts`` accessor, not a copy of it."""
    store, spec = dataset(), dataset_spec()
    for name, binding in spec.members.items():
        assert store.member(name) == common.F[binding.key]
    for name, binding in spec.layout.items():
        assert store.member(name) == common.artifact(binding.key)
    assert store.member(SOURCE) == common.F[spec.source.key]


def test_the_source_export_is_the_declared_raw_export():
    assert dataset().member(SOURCE) == common.DATA_PATH
    assert dataset().source.is_absolute()


def test_paths_returns_source_members_and_trees():
    store = dataset()
    assert set(store.paths()) == {SOURCE, *store.members, *store.layout}
    assert store.paths()[SOURCE] == store.source


def test_an_unknown_member_names_the_declared_set():
    with pytest.raises(KeyError, match="unknown dataset member"):
        dataset().member("not_a_member")


def test_a_tree_root_is_not_a_loadable_frame():
    with pytest.raises(ValueError, match="tree root, not a loadable frame"):
        dataset().load("prepared")


# ── the declared read spec ──────────────────────────────────────────────────

def test_the_read_spec_is_the_declared_shared_dataset_csv_read_spec():
    assert dataset().read == common.data_cfg().dataset_csv_read
    assert dataset().read.dtype == "str"


def test_load_applies_the_declared_read_spec_and_honors_overrides(tmp_path):
    store = dataset()
    frame = pd.DataFrame({"gtin": ["1", "nan", ""], "sku_name_eng": ["water", "nan", ""]})
    path = tmp_path / "frame.csv"
    frame.to_csv(path, index=False)
    injected = _injected(store, dataset_deduped=path)

    declared = injected.load("dataset_deduped")
    # keep_default_na=True from the declared spec coerces "" and "nan" to NA ...
    assert declared.sku_name_eng.isna().sum() == 2
    assert pd.api.types.is_string_dtype(declared.sku_name_eng)  # dtype=str from the spec
    # ... and a caller override steers the read without re-spelling the SSOT.
    assert injected.load("dataset_deduped", keep_default_na=False).sku_name_eng.tolist() == [
        "water", "nan", "",
    ]


def test_load_source_and_load_deduped_read_their_declared_members(tmp_path):
    store = dataset()
    source = tmp_path / "source.csv"
    deduped = tmp_path / "deduped.csv"
    pd.DataFrame({"sku_name_eng": ["a"]}).to_csv(source, index=False)
    pd.DataFrame({"sku_name_eng": ["b"]}).to_csv(deduped, index=False)
    injected = store.model_copy(update={
        "source": source,
        "members": {**store.members, "dataset_deduped": deduped},
    })
    assert injected.load_source().sku_name_eng.tolist() == ["a"]
    assert injected.load_deduped().sku_name_eng.tolist() == ["b"]


def test_load_canonical_records_delegates_to_the_validated_reader(monkeypatch):
    """The class does not re-implement the artifact's own frame contract."""
    marker = pd.DataFrame({"gtin": ["1"]})
    recorded = []
    monkeypatch.setattr(common, "canonical_records_frame",
                        lambda: recorded.append(1) or marker)
    assert dataset().load_canonical_records() is marker
    assert recorded == [1]


# ── identity: a content digest, and only that ───────────────────────────────

def _store_at(root: Path, body: bytes) -> Dataset:
    root.mkdir(parents=True, exist_ok=True)
    (root / "source.csv").write_bytes(body)
    (root / "deduped.csv").write_bytes(b"deduped")
    return Dataset(
        name="unit",
        read=common.data_cfg().dataset_csv_read,
        source=root / "source.csv",
        members={"dataset_deduped": root / "deduped.csv"},
        layout={},
    )


def test_identity_is_stable_portable_and_follows_content(tmp_path):
    first = _store_at(tmp_path / "a", b"source")
    second = _store_at(tmp_path / "b", b"source")
    # the digest is the bytes' identity, never the checkout root's: same bytes
    # at two paths share one digest.
    assert first.identity() == second.identity()
    assert first.identity() == _store_at(tmp_path / "c", b"source").identity()
    # a changed member byte changes the digest.
    (tmp_path / "b" / "deduped.csv").write_bytes(b"deduped-changed")
    assert first.identity() != second.identity()
    # a changed source byte changes the digest too.
    assert first.identity() != _store_at(tmp_path / "d", b"OTHER").identity()


def test_identity_records_an_absent_member_without_raising(tmp_path):
    """A missing member is content (absence), not a validity verdict."""
    store = Dataset(
        name="unit",
        read=common.data_cfg().dataset_csv_read,
        source=tmp_path / "never-written.csv",
        members={"dataset_deduped": tmp_path / "also-missing.csv"},
        layout={},
    )
    assert len(store.identity()) == 64
    # and presence changes the digest, so absence is distinguishable from bytes.
    (tmp_path / "never-written.csv").write_bytes(b"x")
    assert store.identity() != _store_at(tmp_path, b"x").identity()


def test_identity_ignores_tree_roots(tmp_path):
    """Tree roots are addressing, not line items: they stay out of the digest."""
    store = _store_at(tmp_path / "e", b"source")
    with_tree = store.model_copy(update={"layout": {"prepared": tmp_path / "e"}})
    assert with_tree.identity() == store.identity()


def test_the_class_carries_no_validity_gate():
    """No age/validity verdict lives here; the digest is the whole surface."""
    import core.dataset as module

    source = inspect.getsource(module).lower()
    for forbidden in ("stale", "freshness", "is_fresh", "expired", "mtime", "getmtime"):
        assert forbidden not in source, f"validity wording leaked into core.dataset: {forbidden!r}"


# ── transport hand-off ──────────────────────────────────────────────────────

def test_as_bundle_hands_off_to_the_verified_bundle_boundary(monkeypatch, tmp_path):
    import core.bundle as bundle

    recorded = {}

    def fake_load(path, role, *, expected_digest=None, manifest_name=None):
        recorded.update(path=Path(path), role=role,
                        expected_digest=expected_digest, manifest_name=manifest_name)
        return "trusted-handle"

    monkeypatch.setattr(bundle.Bundle, "load", staticmethod(fake_load))
    archive = tmp_path / "inputs.tar.zst"
    assert dataset().as_bundle(archive, "result", expected_digest="a" * 64) == "trusted-handle"
    assert recorded == {"path": archive, "role": "result",
                        "expected_digest": "a" * 64, "manifest_name": None}


# ── declaration validation ──────────────────────────────────────────────────

def test_a_binding_the_phone_book_does_not_declare_is_refused():
    spec = DatasetSpec(
        name="bad",
        source=DatasetMemberSpec(via="files", key="dataset"),
        members={"ghost": DatasetMemberSpec(via="files", key="not_a_declared_file")},
    )
    with pytest.raises(ValueError, match="not declared"):
        Dataset.from_config(spec)


def test_a_layout_binding_the_phone_book_does_not_declare_is_refused():
    spec = DatasetSpec(
        name="bad",
        source=DatasetMemberSpec(via="files", key="dataset"),
        members={"dataset_deduped": DatasetMemberSpec(via="files", key="dataset_deduped")},
        layout={"ghost": DatasetMemberSpec(via="layouts", key="not_a_declared_layout")},
    )
    with pytest.raises(ValueError, match="not declared"):
        Dataset.from_config(spec)


def test_the_source_name_is_reserved_and_a_member_cannot_be_a_line_item_and_tree():
    with pytest.raises(ValueError, match="reserved"):
        DatasetSpec(
            name="bad",
            source=DatasetMemberSpec(via="files", key="dataset"),
            members={SOURCE: DatasetMemberSpec(via="files", key="dataset")},
        )
    with pytest.raises(ValueError, match="line item and tree"):
        DatasetSpec(
            name="bad",
            source=DatasetMemberSpec(via="files", key="dataset"),
            members={"dup": DatasetMemberSpec(via="files", key="dataset_deduped")},
            layout={"dup": DatasetMemberSpec(via="layouts", key="dataset_prepared")},
        )


def test_the_declaration_covers_the_dataset_surface():
    """The dataset's member set is exactly the declared line items + trees."""
    spec = dataset_spec()
    assert set(spec.members) == {
        "dataset_deduped", "sku_to_rep", "canonical_records", "gate_results",
        "labeled_pairs", "final_validation", "number_tokens_reference",
    }
    assert set(spec.layout) == {"prepared", "track_setup"}
    assert spec.source == DatasetMemberSpec(via="files", key="dataset")


def test_the_derived_artifacts_have_a_loadable_path():
    store = dataset()
    for name in ("canonical_records", "gate_results", "dataset_deduped",
                 "labeled_pairs", "final_validation", "number_tokens_reference",
                 "sku_to_rep"):
        assert store.member(name).name.endswith(".csv")
    assert store.member("prepared").name == "prepared"
    assert store.member("track_setup").name == "track_setup"


# ── the migrated slice resolves identically to the accessors it replaced ─────

def test_the_migrated_producers_resolve_to_the_paths_they_replaced():
    """The producer modules now name the dataset; F[...] returns the same path."""
    import training.data_prep as data_prep
    import training.dedupe as dedupe

    assert dedupe.DEDUPED_PATH == common.F["dataset_deduped"]
    assert dedupe.SKU_TO_REP_PATH == common.F["sku_to_rep"]
    assert data_prep._stage_outputs()[0][:2] == [
        common.F["canonical_records"], common.F["gate_results"],
    ]


def test_the_dataset_read_equals_the_direct_read_the_consumer_replaced(tmp_path):
    """``load(name, **read)`` matches the bypassed ``pd.read_csv(path, ...)``."""
    path = tmp_path / "dataset_deduped.csv"
    pd.DataFrame({"gtin": ["1", "nan", ""], "sku_id": ["a", "b", "c"]}).to_csv(
        path, index=False)
    store = dataset().model_copy(
        update={"members": {**dataset().members, "dataset_deduped": path}})
    direct = pd.read_csv(path, dtype=str, keep_default_na=False)  # the replaced read
    pd.testing.assert_frame_equal(
        store.load("dataset_deduped", dtype=str, keep_default_na=False), direct)
