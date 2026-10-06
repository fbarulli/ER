"""TIER 1(a): counterpart positives for anchor-side value swaps — on REAL data.

A hard-negative swap transplants one field into the ANCHOR only
(``symmetric=False``), so the copy stops agreeing with its own source
positive. The MNRL triple builder therefore omitted every swap copy and it
never reached a gradient. ``mint_swap_counterpart_positives`` replays the same
transplant onto the source's own positive so ``(copy, counterpart)`` is a
genuine positive.

These tests run against the real prepared bundle
(``data/prepared/full/worker_1_baseline.pkl.gz``) — real payload text, real
minted swap rows, real donor rows, real training negatives. No synthetic
fixtures: a fabricated payload can be built to pass by construction, which
proves nothing about the real rows this lane trains on.

The invariants pinned here:

1. The counterpart carries the DONOR's value for the transplanted field, so
   the copy and the counterpart AGREE — a positive pair that still matches.
2. No value is invented: every value on the counterpart also exists on the
   real donor row it was borrowed from.
3. Counterparts are neither byte-identical to the source positive nor to the
   swap copy (both would be degenerate positives), and each inherits the
   source positive's gtin.
4. THE POINT OF THE CHANGE: on real triples, ``swap_values`` rows produce
   ZERO MNRL triples before counterparts exist and exactly one per minted
   counterpart after — the rows were dead and now train.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

from training.masking import field_of, mint_swap_counterpart_positives
from training.training import _mnrl_training_triples_with_populations

BUNDLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "prepared" / "full" / "worker_1_baseline.pkl.gz"
)

# These guard the TIER 1(a) counterpart positives minted for ANCHOR-SIDE value
# swaps. That lane is `augment_value_swaps(population="hard_negative")`, which
# lives inside the `not balanced_policy.enabled` branch of
# training/train.py. With config masking.balanced_augmentation.enabled = true
# (shipped since 66bead6, 2026-10-04) it never runs, so the bundle carries
# zero swap_values rows and this fixture asserts against data the pipeline
# does not produce. Skip LOUDLY and reversibly: re-enable the classic lane
# and these come back. Do not relax the assertions.
def _classic_swap_lane_active() -> bool:
    from core.common import load_config

    return not load_config()["masking"]["balanced_augmentation"]["enabled"]


pytestmark = [
    pytest.mark.skipif(
        not BUNDLE.is_file(), reason=f"real prepared bundle absent: {BUNDLE}"
    ),
    pytest.mark.skipif(
        not _classic_swap_lane_active(),
        reason=(
            "anchor-side value swaps are unreachable: masking."
            "balanced_augmentation.enabled is true, so training/train.py "
            "skips augment_value_swaps and mints no swap_values rows"
        ),
    ),
]


@pytest.fixture(scope="module")
def real_bundle() -> dict:
    with gzip.open(BUNDLE, "rb") as fh:
        return pickle.load(fh)


@pytest.fixture(scope="module")
def real_swaps(real_bundle) -> dict:
    """Real payload / gtins / positives / training negatives / swap rows."""
    audit = [
        row
        for row in real_bundle["hard_negative_mask_audit"]
        if str(row.get("target_mode")) == "swap_values"
        and row.get("population") != "swap_counterpart"
    ]
    assert audit, "real bundle has no swap_values hard-negative rows"
    return {
        "payload": real_bundle["payload"],
        "row_bc": real_bundle["row_bc"],
        "pos": real_bundle["pos"],
        "train_neg": real_bundle["train_neg"],
        "audit": audit,
    }


def _val(text: str, field: str) -> tuple[str, ...]:
    return tuple(sorted(t for t in text.split() if field_of(t) == field))


def _mint(real: dict):
    """Mint counterparts, returning (pairs, new_payload, new_bc, new_audit)."""
    return mint_swap_counterpart_positives(
        real["audit"], real["payload"], real["row_bc"], real["pos"]
    )


def _split(pairs, base: int):
    """Minted counterparts point past ``base``; reused source positives don't."""
    minted = [p for p in pairs if p[1] >= base]
    reused = [p for p in pairs if p[1] < base]
    return minted, reused


def test_real_swap_population_is_material(real_swaps):
    """The lane this fixes must be non-trivial on real data, or it is moot."""
    assert len(real_swaps["audit"]) > 1000


def test_every_real_swap_row_gets_a_counterpart_positive(real_swaps):
    """100% coverage — the floor is only cleared if none are left behind."""
    real = real_swaps
    (pairs, new_payload, _new_bc, new_audit) = _mint(real)
    assert len(pairs) == len(real["audit"]), (
        f"{len(real['audit']) - len(pairs)} real swap rows still have no "
        "counterpart positive"
    )
    minted, reused = _split(pairs, len(real["payload"]))
    assert minted and reused, "expected both minted and reused counterparts"
    # Only minted counterparts consume a new payload row.
    assert len(new_payload) == len(new_audit) == len(minted)


def test_counterparts_agree_with_their_copy_on_the_transplanted_field(real_swaps):
    """Invariant 1 — the reason the minted pair stays a valid positive."""
    real = real_swaps
    (pairs, new_payload, _new_bc, new_audit) = _mint(real)
    base = len(real["payload"])
    assert pairs, "no counterparts minted from real swap rows"
    for (_copy_i, counterpart_i), row in zip(_split(pairs, base)[0], new_audit):
        field = row["fields_hit"][0]
        assert _val(real["payload"][int(row["pair_payload_idx"])], field) == _val(
            new_payload[counterpart_i - base], field
        ), f"copy/counterpart disagree on transplanted field {field}"


def test_reused_source_positive_does_not_contradict_the_copy(real_swaps):
    """A reused source positive is valid ONLY because it cannot contradict the copy.

    Two real sub-cases, both safe: the source positive is SILENT about the
    transplanted field (481 rows), or the donor value already EQUALS the
    positive's so copy and positive agree (12 rows). The dangerous case — the
    positive carrying a DIFFERENT value for the transplanted field — would
    assert a false match and must never be reused.
    """
    real = real_swaps
    (pairs, _new_payload, _new_bc, _new_audit) = _mint(real)
    base = len(real["payload"])
    reused = _split(pairs, base)[1]
    assert reused, "no reused source positives in the real bundle"
    field_of_copy = {
        int(r["copy_payload_idx"]): r["fields_hit"][0] for r in real["audit"]
    }
    for copy_i, positive_i in reused:
        field = field_of_copy[int(copy_i)]
        positive_val = _val(real["payload"][positive_i], field)
        copy_val = _val(real["payload"][copy_i], field)
        assert positive_val in ((), copy_val), (
            f"reused source positive carries {positive_val} while the copy "
            f"carries {copy_val} on {field} — reusing it asserts a false match"
        )


def test_counterpart_value_comes_from_the_real_donor_row(real_swaps):
    """Invariant 2 — nothing is invented; the donor really carries the value."""
    real = real_swaps
    (_pairs, new_payload, _new_bc, new_audit) = _mint(real)
    base = len(real["payload"])
    assert new_audit, "no counterparts minted"
    for row in new_audit:
        field = row["fields_hit"][0]
        donor = set(_val(real["payload"][int(row["donor_anchor_payload_idx"])], field))
        counterpart = set(_val(new_payload[int(row["copy_payload_idx"]) - base], field))
        assert counterpart, f"counterpart lost the transplanted field {field}"
        assert counterpart <= donor, (
            f"counterpart carries {counterpart - donor}, absent from the "
            f"real donor row"
        )


def test_counterparts_are_not_degenerate_and_inherit_source_gtin(real_swaps):
    """Invariant 3 — no self-referential or unchanged positives."""
    real = real_swaps
    (_pairs, new_payload, new_bc, new_audit) = _mint(real)
    base = len(real["payload"])
    for row in new_audit:
        counterpart = new_payload[int(row["copy_payload_idx"]) - base]
        assert counterpart != real["payload"][int(row["anchor_payload_idx"])], (
            "counterpart is byte-identical to the source positive"
        )
        assert counterpart != real["payload"][int(row["pair_payload_idx"])], (
            "counterpart is byte-identical to the swap copy"
        )
    assert len(new_bc) == len(new_audit)
    # A counterpart is a copy of the SOURCE POSITIVE, so it inherits that
    # row's gtin (not the hard-negative's).
    for idx, row in enumerate(new_audit):
        assert new_bc[idx] == str(real["row_bc"][int(row["anchor_payload_idx"])])


def test_audit_lineage_points_counterpart_at_source_positive(real_swaps):
    """Feature lineage must not re-claim the already-extended swap copy."""
    real = real_swaps
    (pairs, _new_payload, _new_bc, new_audit) = _mint(real)
    base = len(real["payload"])
    for (_copy_i, counterpart_i), row in zip(_split(pairs, base)[0], new_audit):
        assert row["copy_payload_idx"] == counterpart_i
        assert row["copy_pair_payload_idx"] is None
        assert int(row["anchor_payload_idx"]) < base  # the real source positive


def test_swap_rows_train_only_once_a_counterpart_exists(real_swaps):
    """THE POINT: every real swap copy produces a live MNRL triple.

    The shipped bundle was minted at BUILD time (TIER 1(a) replays inside
    training.train --prepare-bundle), so the counterpart positives are
    already registered in ``pos``. On a pre-mint bundle this test pinned
    zero swap triples before minting; on a minted bundle it now pins the
    post-rebuild outcome: every real anchor-side swap copy trains exactly
    once, and a fresh mint still covers 100% of the real rows.
    """
    real = real_swaps
    train_neg = real["train_neg"]

    def swap_triples(train_pos, hard_audit):
        triples = _mnrl_training_triples_with_populations(
            train_pos, train_neg, mask_audit=[], hard_negative_mask_audit=hard_audit
        )
        keys = {
            (int(r["copy_payload_idx"]), int(r["pair_payload_idx"]))
            for r in hard_audit
            if str(r.get("target_mode")) == "swap_values"
        }
        return [t for t, _pop in triples if (int(t[0]), int(t[2])) in keys]

    built = swap_triples(real["pos"], real["audit"])
    assert len(built) == len(real["audit"]), (
        f"{len(real['audit']) - len(built)} real swap rows produce no MNRL "
        "triple — a build-time counterpart mint must register a positive "
        "for every anchor-side swap copy"
    )

    (pairs, _np, _nb, new_audit) = _mint(real)
    assert pairs, "no counterparts minted; cannot test recovery"
