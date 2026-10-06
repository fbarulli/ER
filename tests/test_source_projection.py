"""CSV projection preserves NA semantics and the column SSOT contract."""

import pandas as pd
import pytest

from core import common
from core.columns import CANONICAL_COLUMNS, COLUMN_MAPPING, raw_of


@pytest.fixture
def source_export(tmp_path, monkeypatch):
    path = tmp_path / "source.csv"
    values = {column: ["", "", ""] for column in COLUMN_MAPPING}
    values.update({
        raw_of("sku_id"): ["001", "002", "003"],
        raw_of("gtin"): ["00012345678905", "NA", ""],
        raw_of("brand"): ["Brand", "None", ""],
        raw_of("sku_name_eng"): ["Tea", "Coffee", "Water"],
    })
    pd.DataFrame(values).to_csv(path, index=False)
    monkeypatch.setattr(common, "DATA_PATH", path)
    return path


@pytest.mark.parametrize("raw", [False, True])
def test_projected_source_matches_full(source_export, raw):
    loader = common.load_raw_export if raw else common.load_dataset
    columns = ["sku_id", "gtin", "brand"]
    if raw:
        columns = [raw_of(column) for column in columns]
    full = loader()
    projected = loader(columns=columns)
    pd.testing.assert_frame_equal(projected, full.loc[:, full.columns.isin(columns)])
    id_column = raw_of("sku_id") if raw else "sku_id"
    gtin_column = raw_of("gtin") if raw else "gtin"
    brand_column = raw_of("brand") if raw else "brand"
    assert projected[id_column].tolist() == ["001", "002", "003"]
    assert projected[gtin_column].iloc[0] == "00012345678905"
    assert projected[gtin_column].iloc[1:].isna().all()
    assert projected[brand_column].iloc[1:].isna().all()


@pytest.mark.parametrize("raw", [False, True])
def test_empty_projection_rejected(source_export, raw):
    loader = common.load_raw_export if raw else common.load_dataset
    with pytest.raises(ValueError, match="at least one column"):
        loader(columns=[])


def test_unknown_canonical_projection_rejected(source_export):
    with pytest.raises(ValueError, match="unknown canonical source columns"):
        common.load_dataset(columns=["missing_column"])


def test_raw_default_and_override_share_csv_and_identity_contract(source_export, monkeypatch):
    from core import identity_policy

    canonical = common.load_dataset()
    catalog = source_export.with_name("catalog.csv")
    canonical.to_csv(catalog, index=False)
    monkeypatch.setitem(common.F, "dataset_deduped", catalog)
    calls = []

    def identity_view(frame):
        calls.append(frame.copy())
        return frame.iloc[1:].copy()

    monkeypatch.setattr(identity_policy, "exclude_reviewed_rows", identity_view)
    default = common.load_dataset_deduped()
    override = common.load_dataset_deduped(catalog)
    pd.testing.assert_frame_equal(default, override)
    pd.testing.assert_frame_equal(default, canonical.iloc[1:])
    assert len(calls) == 2  # One identity-policy application per load.
    for parsed in calls:
        pd.testing.assert_frame_equal(parsed, canonical)


def test_override_schema_uses_column_ssot_without_default_csv(source_export, monkeypatch):
    from core import identity_policy

    monkeypatch.setattr(identity_policy, "exclude_reviewed_rows", lambda frame: frame)
    monkeypatch.setitem(common.F, "dataset_deduped", source_export.with_name("missing.csv"))
    canonical = common.load_dataset()
    catalog = source_export.with_name("override.csv")
    canonical.to_csv(catalog, index=False)
    pd.testing.assert_frame_equal(common.load_dataset_deduped(catalog), canonical)
    missing = CANONICAL_COLUMNS[0]
    canonical.drop(columns=[missing]).to_csv(catalog, index=False)
    with pytest.raises(ValueError, match=f"canonical columns.*{missing}"):
        common.load_dataset_deduped(catalog)


@pytest.mark.parametrize("lane", ["raw", "canonical", "deduped", "override"])
def test_all_dataset_lanes_obey_configured_na_policy(source_export, monkeypatch, lane):
    from core import identity_policy

    cfg = common.data_cfg().model_copy(deep=True)
    cfg.dataset_csv_read.keep_default_na = False
    monkeypatch.setattr(common, "data_cfg", lambda: cfg)
    monkeypatch.setattr(identity_policy, "exclude_reviewed_rows", lambda frame: frame)
    raw = pd.read_csv(source_export, dtype=str, keep_default_na=False)
    catalog = source_export.with_name("catalog.csv")
    raw.rename(columns=COLUMN_MAPPING).to_csv(catalog, index=False)
    monkeypatch.setitem(common.F, "dataset_deduped", catalog)
    if lane == "raw":
        frame, column = common.load_raw_export(), raw_of("gtin")
    elif lane == "canonical":
        frame, column = common.load_dataset(), "gtin"
    else:
        frame = common.load_dataset_deduped(catalog if lane == "override" else None)
        column = "gtin"
    assert frame[column].tolist() == ["00012345678905", "NA", ""]
