"""Rejection coverage for single-constraint config/contract validators.

Each test starts from a payload the SSOT models ACCEPT (the live config for
``DataConfig``, or a minimal well-formed document for the small specs), then
violates exactly ONE constraint and asserts the boundary raises
``ValidationError`` naming that constraint. These pin validator ``raise``
branches that no acceptance-path test can reach, so a deleted guard fails here
instead of silently accepting a bad config.
"""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from core.common import data_cfg
from core.schemas import (
    AuditSpec,
    CalibrationPartition,
    DataConfig,
    DeclarationDropoutSpec,
    RandStratumSweepSpec,
    RandTruthSplitsSpec,
    SweepSpec,
    UnitsSpec,
)


# ── DataConfig: canonical_optional_columns / column_aliases ─────────────────

def test_canonical_optional_columns_rejects_blank_unknown_literal():
    """An empty unknown-value literal makes 'unknown' inexpressible."""
    raw = data_cfg().model_dump()
    column = sorted(raw["canonical_optional_columns"])[0]
    raw["canonical_optional_columns"][column] = "   "
    with pytest.raises(ValidationError, match="canonical_optional_columns"):
        DataConfig.model_validate(raw)


def test_column_aliases_rejects_non_canonical_key():
    """An alias key must be a real canonical column, not a stray name."""
    raw = data_cfg().model_dump()
    raw["column_aliases"]["not_a_canonical_column"] = ["stray"]
    with pytest.raises(ValidationError, match="is not a canonical column"):
        DataConfig.model_validate(raw)


def test_column_aliases_rejects_alias_colliding_with_real_column():
    """An alias equal to a real column name makes resolution ambiguous."""
    raw = data_cfg().model_dump()
    owner = sorted(raw["column_aliases"])[0]
    real_column = next(
        name for name in raw["column_mapping"].values()
        if name != owner and name not in raw["column_aliases"].get(owner, [])
    )
    raw["column_aliases"][owner] = [*raw["column_aliases"][owner], real_column]
    with pytest.raises(ValidationError, match="already a real column name"):
        DataConfig.model_validate(raw)


def test_column_aliases_rejects_one_alias_claimed_twice():
    """Two columns claiming the same alias must be refused, not last-wins.

    NOTE: the "claimed by both" branch is shadowed by the earlier ``taken``
    check (every recorded alias is added to ``taken``), so the refusal surfaces
    as "already a real column name". The contract is the same: a duplicate
    alias is a loud ValidationError, never a silent last-writer-wins.
    """
    raw = data_cfg().model_dump()
    left, right = sorted(raw["column_aliases"])[:2]
    raw["column_aliases"][left] = [*raw["column_aliases"][left], "shared_alias"]
    raw["column_aliases"][right] = [*raw["column_aliases"][right], "shared_alias"]
    with pytest.raises(ValidationError, match="aliases must be distinct"):
        DataConfig.model_validate(raw)


# ── UnitsSpec: pack bounds + one spelling, one spec ─────────────────────────

def _units_payload() -> dict:
    return data_cfg().units.model_dump()


def test_units_rejects_inverted_pack_bounds():
    units = _units_payload()
    units["pack_min"] = units["pack_max"]
    with pytest.raises(ValidationError, match=r"pack_min .* must be < pack_max"):
        UnitsSpec.model_validate(units)


def test_units_rejects_one_spelling_with_conflicting_specs():
    """A spelling may not be claimed by two entries with different numbers."""
    units = _units_payload()
    first, second = units["volume"][0], units["volume"][1]
    second = dict(second)
    second["spellings"] = [first["spellings"][0]]
    second["ml_per_unit"] = first["ml_per_unit"] + 1.0
    units["volume"] = [first, second, *units["volume"][2:]]
    with pytest.raises(ValidationError, match="claimed by two"):
        UnitsSpec.model_validate(units)


def test_units_rejects_stripped_key_collision_across_entries():
    """Distinct spellings that strip to the same key must agree on the spec."""
    units = _units_payload()
    first = dict(units["volume"][0])
    second = dict(units["volume"][1])
    first["spellings"] = ["fl oz-"]
    second["spellings"] = ["floz"]
    second["ml_per_unit"] = first["ml_per_unit"] + 1.0
    units["volume"] = [first, second, *units["volume"][2:]]
    with pytest.raises(ValidationError, match="collide with another entry"):
        UnitsSpec.model_validate(units)


# ── RandTruthSplitsSpec: size relations ─────────────────────────────────────

def _truth_splits(**overrides) -> dict:
    payload = {
        "output_dir": "out",
        "calibration_output": "calibration.csv",
        "holdout_output": "holdout.csv",
        "sample_size": 10,
        "calibration_size": 5,
        "calibration_folds": 3,
        "seed": 0,
    }
    payload.update(overrides)
    return payload


def test_truth_splits_rejects_calibration_covering_the_sample():
    payload = _truth_splits(sample_size=6, calibration_size=6)
    with pytest.raises(ValidationError, match="must be smaller than sample_size"):
        RandTruthSplitsSpec.model_validate(payload)


def test_truth_splits_rejects_fewer_than_three_holdout_rows():
    payload = _truth_splits(sample_size=7, calibration_size=5)
    with pytest.raises(ValidationError, match="at least three holdout rows"):
        RandTruthSplitsSpec.model_validate(payload)


def test_truth_splits_rejects_a_fold_without_truth():
    payload = _truth_splits(sample_size=10, calibration_size=3, calibration_folds=4)
    with pytest.raises(ValidationError, match="at least one truth per calibration fold"):
        RandTruthSplitsSpec.model_validate(payload)


# ── RandStratumSweepSpec / DeclarationDropoutSpec ───────────────────────────

def test_stratum_sweep_rejects_identities_below_fold_count():
    payload = {
        "output_dir": "out",
        "output": "sweep.csv",
        "identities_per_status": 3,
        "skus_per_identity": 2,
        "calibration_folds": 5,
        "seed": 0,
    }
    with pytest.raises(ValidationError, match="at least one identity per calibration fold"):
        RandStratumSweepSpec.model_validate(payload)


def test_declaration_dropout_rejects_inverted_drop_bounds():
    with pytest.raises(ValidationError, match=r"min_drop .* must be <= max_drop"):
        DeclarationDropoutSpec(frac=0.5, min_drop=3, max_drop=2)


# ── AuditSpec: the strip-audit ladder covers [0, 1] contiguously ────────────

def _audit_payload(**overrides) -> dict:
    payload = {
        "strip_audit_sample": 10,
        "strip_ladder_bands": [{"lo": 0.0, "hi": 0.5}, {"lo": 0.5, "hi": 1.0}],
        "blocking_budget": 10,
        "blocking_min_recall": 0.5,
    }
    payload.update(overrides)
    return payload


def test_strip_ladder_must_start_at_zero():
    payload = _audit_payload(strip_ladder_bands=[{"lo": 0.1, "hi": 0.5}])
    with pytest.raises(ValidationError, match=r"must start at lo=0\.0"):
        AuditSpec.model_validate(payload)


def test_strip_ladder_must_be_contiguous():
    payload = _audit_payload(
        strip_ladder_bands=[{"lo": 0.0, "hi": 0.5}, {"lo": 0.6, "hi": 1.0}]
    )
    with pytest.raises(ValidationError, match="must be contiguous"):
        AuditSpec.model_validate(payload)


# ── SweepSpec: train fractions are strictly inside (0, 1) ───────────────────

def test_sweep_rejects_train_fraction_outside_the_open_unit_interval():
    payload = {
        "payload_variants": ["full"],
        "train_fracs": [0.0, 0.8],
        "smoke_sample": 1,
        "sweep_sample": 1,
        "rerank_model": "cross-encoder",
    }
    with pytest.raises(ValidationError, match=r"must be in \(0,1\)"):
        SweepSpec.model_validate(payload)


# ── CalibrationPartition: shape, populations, boundary ──────────────────────

def _partition(**overrides) -> dict:
    payload = {
        "positive_fit": np.array([[0, 1]]),
        "positive_reserved": np.array([[2, 3]]),
        "negative_fit": np.array([[0, 2]]),
        "negative_reserved": np.array([[1, 3]]),
        "row_bc": np.array(["g0", "g1", "g2", "g3"]),
        "n_positive_pairs": 2,
        "n_negative_pairs": 2,
    }
    payload.update(overrides)
    return payload


def test_calibration_partition_rejects_wrong_pool_shape():
    payload = _partition(positive_fit=np.array([[0, 1, 2]]))
    with pytest.raises(ValidationError, match=r"must be an \(n, 2\) pair array"):
        CalibrationPartition(**payload)


def test_calibration_partition_rejects_changed_positive_population():
    payload = _partition(n_positive_pairs=3)
    with pytest.raises(ValidationError, match="changed its population"):
        CalibrationPartition(**payload)


def test_calibration_partition_rejects_identity_crossing_the_boundary():
    payload = _partition(
        positive_fit=np.array([[0, 1]]),
        positive_reserved=np.array([[1, 2]]),
        negative_fit=np.zeros((0, 2), dtype=int),
        negative_reserved=np.zeros((0, 2), dtype=int),
        n_positive_pairs=2,
        n_negative_pairs=0,
    )
    with pytest.raises(ValidationError, match="cross the calibration boundary"):
        CalibrationPartition(**payload)


def test_calibration_partition_rejects_changed_negative_population():
    payload = _partition(n_negative_pairs=3)
    with pytest.raises(ValidationError, match="negative calibration partition changed its population"):
        CalibrationPartition(**payload)


def test_calibration_partition_rejects_reserved_positive_identity_in_fit_negatives():
    payload = _partition(negative_fit=np.array([[2, 0]]))
    with pytest.raises(ValidationError, match="reserved positive identities occur in fit negatives"):
        CalibrationPartition(**payload)


def test_calibration_partition_rejects_fit_positive_identity_in_reserved_negatives():
    payload = _partition(
        negative_fit=np.array([[0, 1]]), negative_reserved=np.array([[0, 2]])
    )
    with pytest.raises(ValidationError, match="fit positive identities occur in reserved negatives"):
        CalibrationPartition(**payload)


def test_calibration_partition_rejects_mirrored_negative_pair_across_the_boundary():
    """The mirrored orientation of a reserved negative pair must not stay fit."""
    payload = _partition(
        row_bc=np.array(["g0", "g1", "g2", "g3", "g4", "g5"]),
        negative_fit=np.array([[4, 5]]),
        negative_reserved=np.array([[5, 4]]),
    )
    with pytest.raises(ValidationError, match="negative identity pairs cross the calibration boundary"):
        CalibrationPartition(**payload)
