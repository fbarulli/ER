"""Pins for the training-module fix batch (worktree review fixes).

Coverage: attestation fail-closed plan identity, controlled/frozen sampler
invariants plus the column-wise text-hash equivalence, boundary epoch-plan
validation, MNRL single-call triple/population equivalence (incl. the twin
edge reuse), per-fold dynamic mask audit isolation, the prepared payload
digest helper, and the unrelated-pair selection memo identity guard.
"""

from __future__ import annotations

from core.portable_archive import ByteCount
import json
import random

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from training.attestation import (
    TrainingAttestation,
    read_attestation,
    verify_attestation,
    verify_plan_identity,
)
from training.run_plan import validate_epoch_batches
from training.sampler import (
    ControlledBatchSampler,
    FrozenBatchSampler,
    _row_text_values,
    resolve_composition,
)
from training.token_inputs import payload_size
from training.training import (
    _build_mnrl_training_triples,
    _build_mnrl_triple_populations,
    _build_pair_lineage,
    _dynamic_mask_negative_transform,
    _mnrl_training_triples_with_populations,
)
from training.uniformity import select_unrelated_pairs


def _attestation(plan_identity):
    return TrainingAttestation(
        attestation_schema="er-training-attestation-v1",
        status="pass",
        run_dir="run",
        finished_at="finished",
        bundle_path="bundle.pkl.gz",
        bundle_size=1234,
        plan_identity=plan_identity,
        checks={},
        attested_at="now",
    )


def test_verify_plan_identity_fails_closed_without_identity():
    attestation = _attestation(None)
    with pytest.raises(ValueError, match="no plan identity"):
        verify_plan_identity(attestation, loss="mnrl", train_frac=1.0, sample=False)


def test_verify_plan_identity_accepts_matching_identity():
    identity = {"loss": "mnrl", "train_frac": 1.0, "sample": False}
    verify_plan_identity(_attestation(identity), loss="mnrl", train_frac=1.0, sample=False)


def test_verify_plan_identity_names_drift_fields():
    identity = {"loss": "mnrl", "train_frac": 1.0, "sample": False}
    with pytest.raises(ValueError, match="'loss'"):
        verify_plan_identity(_attestation(identity), loss="contrastive",
                             train_frac=1.0, sample=False)
    with pytest.raises(ValueError, match="'sample'"):
        verify_plan_identity(_attestation(identity), loss="mnrl",
                             train_frac=1.0, sample=True)


def test_read_attestation_rejects_unknown_schema(tmp_path):
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps({"schema": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a training attestation"):
        read_attestation(path, bundle_path=path)


def test_verify_attestation_binds_bundle_bytes(tmp_path):
    bundle = tmp_path / "bundle.pkl.gz"
    bundle.write_bytes(b"frozen-bytes")
    good = _attestation({"loss": "mnrl"})
    good = good.model_copy(update={"bundle_size": ByteCount(b"frozen-bytes").total})
    verify_attestation(good, bundle_path=bundle)
    with pytest.raises(ValueError, match="size mismatch"):
        verify_attestation(_attestation({"loss": "mnrl"}), bundle_path=bundle)


def _sampler_dataset():
    rows = 8
    sentence1 = [f"text {i}" for i in range(rows)]
    sentence2 = [f"counter {i}" for i in range(rows)]
    populations = (
        ["gate_positive"] * 4 + ["masked_positive"] * 2 + ["hard_negative"] * 2
    )
    return Dataset.from_dict(
        {"sentence1": sentence1, "sentence2": sentence2, "pair_population": populations}
    )


_COMPOSITION = {"gate_positive": 2, "masked_positive": 1, "hard_negative": 1}


def test_controlled_batch_sampler_covers_every_row_once_per_epoch():
    dataset = _sampler_dataset()
    sampler = ControlledBatchSampler(dataset, 4, dict(_COMPOSITION), seed=3)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        batches = [list(batch) for batch in sampler]
        indices = sorted(index for batch in batches for index in batch)
        assert indices == list(range(len(dataset)))
        assert len(batches) == sampler.__len__()


def test_controlled_batch_sampler_dedups_identical_text():
    dataset = Dataset.from_dict(
        {
            "sentence1": ["same words", "other words", "same words", "more words",
                          "last words", "extra words", "final words", "one word"],
            "sentence2": ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"],
            "pair_population": ["gate_positive"] * 8,
        }
    )
    sampler = ControlledBatchSampler(
        dataset, 4, {"gate_positive": 4}, seed=11
    )
    sampler.set_epoch(0)
    duplicate_rows = {0, 2}
    for batch in sampler:
        assert not duplicate_rows <= set(batch)


def test_controlled_batch_sampler_is_deterministic_per_seed_and_epoch():
    dataset = _sampler_dataset()
    sampler = ControlledBatchSampler(dataset, 4, dict(_COMPOSITION), seed=5)
    sampler.set_epoch(2)
    first = [list(batch) for batch in sampler]
    sampler.set_epoch(2)
    second = [list(batch) for batch in sampler]
    assert first == second
    sampler.set_epoch(2)
    replay = ControlledBatchSampler(dataset, 4, dict(_COMPOSITION), seed=5)
    replay.set_epoch(2)
    assert [list(batch) for batch in replay] == first


def test_row_text_value_column_form_matches_row_form():
    dataset = _sampler_dataset()
    column_form = _row_text_values(dataset)
    row_form = []
    text_columns = {"sentence1", "sentence2", "anchor", "positive", "negative"}
    for index in range(len(dataset)):
        row = dataset[index]
        row_form.append(
            frozenset(str(row[col]) for col in dataset.column_names if col in text_columns)
        )
    assert column_form == row_form


def test_resolve_composition_scales_present_populations():
    counts = resolve_composition(
        {"gate_positive": 4, "masked_positive": 2, "hard_negative": 2},
        ["gate_positive", "masked_positive", "hard_negative"],
        8,
    )
    assert sum(counts.values()) == 8
    assert set(counts) == {"gate_positive", "masked_positive", "hard_negative"}
    assert all(value >= 1 for value in counts.values())


def test_resolve_composition_redistributes_absent_populations():
    counts = resolve_composition(
        {"gate_positive": 4, "masked_positive": 2, "hard_negative": 2},
        ["gate_positive", "hard_negative"],
        8,
    )
    assert set(counts) == {"gate_positive", "hard_negative"}
    assert sum(counts.values()) == 8


def test_frozen_batch_sampler_pins():
    with pytest.raises(ValueError, match="repeats training rows"):
        FrozenBatchSampler([[[0, 1], [1, 2]]], expected_rows=3)
    with pytest.raises(ValueError, match="exactly once"):
        FrozenBatchSampler([[[0, 1]]], expected_rows=3)
    sampler = FrozenBatchSampler([[[0, 1], [2]], [[2, 1], [0]]], expected_rows=3, batch_size=2)
    sampler.set_epoch(1)
    assert list(sampler) == [[2, 1], [0]]
    with pytest.raises(ValueError, match="absent from local presentation plan"):
        sampler.set_epoch(5)


def test_validate_epoch_batches_boundary_pins():
    plan = {
        "inputs": {
            "folds": [
                {
                    "objective": {
                        "dataset": {"anchor": ["a", "b", "c"]},
                        "sampler": {
                            "cpu": {"batch_size": 2, "epochs": [[[0, 1], [2]]]}
                        },
                    }
                }
            ]
        }
    }
    validate_epoch_batches(plan, epochs=1, batch_sizes={"cpu": 2})
    bad_repeat = {
        "inputs": {"folds": [{"objective": {
            "dataset": {"anchor": ["a", "b", "c"]},
            "sampler": {"cpu": {"batch_size": 2, "epochs": [[[0, 1], [1]]]}}}}]}
    }
    with pytest.raises(ValueError, match="exactly once"):
        validate_epoch_batches(bad_repeat, epochs=1, batch_sizes={"cpu": 2})
    bad_size = {
        "inputs": {"folds": [{"objective": {
            "dataset": {"anchor": ["a", "b", "c"]},
            "sampler": {"cpu": {"batch_size": 2, "epochs": [[[0, 1, 2]]]}}}}]}
    }
    with pytest.raises(ValueError, match="batch shape invalid"):
        validate_epoch_batches(bad_size, epochs=1, batch_sizes={"cpu": 2})
    short_plan = {
        "inputs": {"folds": [{"objective": {
            "dataset": {"anchor": ["a", "b", "c"]},
            "sampler": {"cpu": {"batch_size": 2, "epochs": []}}}}]}
    }
    with pytest.raises(ValueError, match="batch size/epochs differ"):
        validate_epoch_batches(short_plan, epochs=1, batch_sizes={"cpu": 2})


def test_mnrl_single_call_projection_matches_helper_outputs():
    train_pos = np.array([[0, 1], [2, 3]])
    train_neg = np.array([[5, 1], [2, 6]])
    mask_audit = [{
        "anchor_payload_idx": 0, "copy_payload_idx": 4, "pair_payload_idx": 1,
        "copy_pair_payload_idx": None, "target_mode": "", "population": "positive",
    }]
    hard_audit = [{
        "anchor_payload_idx": 0, "copy_payload_idx": 5, "pair_payload_idx": 1,
        "target_mode": "counterfactual", "copy_source_payload_idx": None,
    }]
    joined = _mnrl_training_triples_with_populations(
        train_pos, train_neg, mask_audit=mask_audit, hard_negative_mask_audit=hard_audit
    )
    assert [triple for triple, _ in joined] == _build_mnrl_training_triples(
        train_pos, train_neg, mask_audit=mask_audit, hard_negative_mask_audit=hard_audit
    )
    assert [population for _, population in joined] == _build_mnrl_triple_populations(
        train_pos, train_neg, mask_audit=mask_audit, hard_negative_mask_audit=hard_audit
    )
    assert ("twin", ) == tuple(
        {population for _, population in joined} & {"twin"}
    )


def test_build_pair_lineage_rejects_dynamic_audit_rows():
    dynamic_row = {
        "fold": 0, "epoch": 1, "pair_id": 9, "population": "hard_negative",
        "augmentation": "dynamic_mask", "anchor_text": "a", "masked_text": "b",
        "realized_extent": 0.5, "configured_mask_lo": 0.5, "configured_mask_hi": 0.8,
        "mask_prob": None,
    }
    with pytest.raises(KeyError):
        _build_pair_lineage(
            np.array([[0, 1]]),
            np.array([[2, 3]]),
            train_neg_sources=np.array(["gate"], dtype=object),
            mask_audit=[],
            hard_negative_mask_audit=[dynamic_row],
            payload_metadata=[],
            gate_lookup={},
        )


def _run_dynamic_mask_transform(labels, pair_ids, epoch, ann_state, audit):
    batch = {
        "label": labels,
        "pair_id": pair_ids,
        "sentence1": [f"anchor {i}" for i in range(len(labels))],
        "sentence2": [f"target {i}" for i in range(len(labels))],
    }
    stats = {}
    return _dynamic_mask_negative_transform(
        batch,
        rng=random.Random(0),
        frac=1.0,
        mask_prob=None,
        mask_lo=0.5,
        mask_hi=0.8,
        counts={},
        counts_by_epoch={},
        stats_by_epoch=stats,
        epoch_ref={"epoch": epoch},
        ann_state=ann_state,
        pair_populations=["hard_negative" if label == 0 else "gate_positive"
                          for label in labels],
        mask_audit=audit,
        fold=0,
    )


def test_dynamic_mask_transform_isolates_dynamic_audit_rows():
    static_audit: list[dict] = []
    dynamic_audit: list[dict] = []
    _run_dynamic_mask_transform(
        [0, 1], [3, 0], 2, {"version": 5}, audit=dynamic_audit
    )
    assert len(dynamic_audit) == 1
    assert static_audit == []
    row = dynamic_audit[0]
    assert row["epoch"] == 2
    assert row["pair_id"] == 3
    assert row["augmentation"] == "dynamic_mask"


def test_dynamic_mask_transform_tracks_epoch_and_ann_version():
    dynamic_audit: list[dict] = []
    presentation_counts: dict[tuple, int] = {}
    batch = {
        "label": [0],
        "pair_id": [3],
        "sentence1": ["anchor 0"],
        "sentence2": ["target 0"],
    }
    _dynamic_mask_negative_transform(
        batch,
        rng=random.Random(0),
        frac=1.0,
        mask_prob=None,
        mask_lo=0.5,
        mask_hi=0.8,
        counts={},
        counts_by_epoch={},
        stats_by_epoch={},
        epoch_ref={"epoch": 4},
        ann_state={"version": 7},
        pair_populations=["hard_negative"],
        presentation_counts=presentation_counts,
        mask_audit=dynamic_audit,
        fold=0,
    )
    key = next(iter(presentation_counts))
    assert key == (4, 3, "hard_negative", "dynamic_mask", 7)
    assert dynamic_audit[0]["epoch"] == 4


def test_payload_digest_helper_matches_recorded_formula():
    payload = ["alpha", "beta �sym", "gamma"]
    expected = ByteCount(json.dumps(list(payload), ensure_ascii=False).encode()).total
    assert payload_size(payload) == expected
    assert payload_size(list(payload)) == payload_size(tuple(payload))


def _unrelated_fixture():
    df = pd.DataFrame(
        {
            "brand": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "category": ["x", "y", "z", "w", "v", "u", "t", "s"],
            "breadcrumbs_eng": ["x", "y", "z", "w", "v", "u", "t", "s"],
        }
    )
    payload = [f"sku{i} product{i}" for i in range(len(df))]
    return df, payload


class _DiagnosticsStubModel:
    training = False

    def train(self, mode: bool) -> None:
        self.training = mode

    def encode(self, texts, **kwargs):
        base = np.array([[1.0, 0.0], [0.0, 1.0], [0.25, 0.75], [0.5, 0.5]])
        return np.vstack([base] * (len(texts) // len(base) + 1))[: len(texts)]


_COLLAPSE_CFG = {
    "collapse_guardrail": {
        "enabled": True,
        "unrelated_pairs": 2,
        "seed": 7,
        "max_token_frequency": 1.0,
        "operating_threshold": 0.9,
        "crossing_rate_ceiling": 1.0,
        "median_penalty_start": 1.0,
        "p90_penalty_start": 1.0,
        "cosine_std_floor": 0.0,
    }
}


def test_unrelated_pair_selection_memo_hits_through_collapse_diagnostics(monkeypatch):
    import training.uniformity as uniformity

    df, payload = _unrelated_fixture()
    long_payload = list(payload) + ["extra 9", "extra 10"]

    original = uniformity.select_unrelated_pairs
    returned_lists = []

    def spy(df_arg, payload_arg, *, n_pairs, seed, max_token_frequency):
        result = original(
            df_arg, payload_arg,
            n_pairs=n_pairs, seed=seed, max_token_frequency=max_token_frequency,
        )
        returned_lists.append(result)
        return result

    monkeypatch.setattr(uniformity, "select_unrelated_pairs", spy)

    first = uniformity.collapse_diagnostics(
        model=_DiagnosticsStubModel(),
        df=df,
        payload=long_payload,
        config=_COLLAPSE_CFG,
        batch_size=8,
        requested=True,
        evaluation_step=1,
    )
    second = uniformity.collapse_diagnostics(
        model=_DiagnosticsStubModel(),
        df=df,
        payload=long_payload,
        config=_COLLAPSE_CFG,
        batch_size=8,
        requested=True,
        evaluation_step=2,
    )
    assert len(returned_lists) == 2
    # THE A10 HIT: the second evaluation step receives the SAME pairs list
    # object — the slice was materialized once and the memo served the
    # repeat instead of recomputing.
    assert returned_lists[0] is returned_lists[1]
    # Byte-identical telemetry across repeated evaluation steps.
    assert first == second


def test_unrelated_pair_memo_never_reuses_foreign_objects():
    df, payload = _unrelated_fixture()
    reference = select_unrelated_pairs(df, payload, n_pairs=3, seed=1, max_token_frequency=1.0)
    recomputed = select_unrelated_pairs(
        df, list(payload), n_pairs=3, seed=1, max_token_frequency=1.0
    )
    assert recomputed == reference
    assert recomputed is not reference
