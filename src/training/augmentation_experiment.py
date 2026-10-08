"""Augmentation on/off experiment harness + minted-negative label quality.

Two TODO items in one surface:

ON/OFF EXPERIMENT. ``masking.frac`` is the single switch every static
augmentation lane hangs off (``train.py``: the balanced lane runs when
``args.mask_frac > 0 and balanced_augmentation.enabled``, the legacy
mask/swap/dropout/twin lanes when ``args.mask_frac > 0`` and it is disabled), so
``--mask-frac 0`` is a complete augmentation-OFF arm with no other config
change. :func:`experiment_plan` names the two comparable arms (same seed, same
frozen data, one switch apart) so the A/B pair is reproducible rather than
improvised, and :func:`arm_arguments` renders the trainer arguments for an arm.

LABEL QUALITY FOR MINTED NEGATIVES. Every minted negative is a label-0 row by
construction: the anchored listing (the copy keeps its anchor's GTIN) had one
attribute field transplanted from a DIFFERENT entity, and that transplant is
what makes the pair a non-match. That claim is checkable from the audit alone —
the recorded ``fields_after`` must still conflict with the pair side's own
declaration on the target field, the masked variant must keep that field, and
the transplanted values must exist in the recorded donor's text (nothing is
invented). :func:`minted_negative_label_quality` measures exactly that;
:func:`assert_label_quality` refuses a bundle whose minted rows do not hold up.

Usage:
    augmentation_experiment.py [--bundle TEXT_BUNDLE] [--assert-quality]
                              [--mask-frac F]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: The two arms of the augmentation experiment. ``on`` keeps the configured
#: masking fraction; ``off`` zeroes it (all static augmentation lanes off).
AUGMENTATION_ARMS = ("on", "off")

#: Audit marker for a minted negative (``balanced_augmentation``): the ready
#: row and its context-masked variant.
_MINTED_VARIANTS = ("minted", "minted_masked")


def arm_arguments(arm: str, *, configured_frac: float, seed: int | None = None) -> dict:
    """The trainer arguments for one augmentation arm (same data, one switch).

    ``on`` replays the configured fraction; ``off`` passes ``--mask-frac 0``,
    which disables the balanced lane AND the legacy mask/swap/dropout/twin
    lanes in ``train.py``. The seed is carried through untouched so the two
    arms differ only by augmentation.
    """
    if arm not in AUGMENTATION_ARMS:
        raise ValueError(
            f"unknown augmentation arm {arm!r}; expected one of {AUGMENTATION_ARMS}"
        )
    arguments = {
        "mask_frac": 0.0 if arm == "off" else float(configured_frac),
        "augmentation": arm,
    }
    if seed is not None:
        arguments["seed"] = int(seed)
    return arguments


def experiment_plan(configured_frac: float, *, seed: int | None = None) -> dict:
    """The A/B pair rendered together, ready to record next to the runs."""
    return {
        "arms": {
            arm: arm_arguments(arm, configured_frac=configured_frac, seed=seed)
            for arm in AUGMENTATION_ARMS
        },
        "identical": ["data bundle", "model", "split", "seed"],
        "differ": ["masking.frac (static augmentation on/off)"],
        "criterion": "held-out dev selection, then test metrics (report_test)",
    }


def _minted_rows(audit: list[dict]) -> list[dict]:
    return [
        row for row in audit or []
        if str(row.get("population", "hard_negative")) == "hard_negative"
        and str(row.get("generation_variant", "")) in _MINTED_VARIANTS
    ]


def minted_negative_label_quality(
    audit: list[dict], payload: list[str], row_bc, *, fail_loud: bool = False
) -> dict:
    """Measure whether each minted negative is a genuine, licensed label-0 row.

    Per row:
      * exactly one discriminating field (``fields_hit``) was transplanted;
      * the copy's OWN declared value for that field conflicts with the pair
        side's declaration — without this the row is a silent false negative;
      * the masked variant still carries the transplanted (conflicting) value:
        context masking may never erase the field that makes the row a 0;
      * the copy keeps its ANCHOR's entity/GTIN (the pair is rejected for the
        transplanted attribute, not for a different listing);
      * every transplanted value exists in the recorded donor's text and the
        donor is a different entity — no value is invented, none is self-donated.

    Returns ``{'checked', 'genuine', 'quality', 'per_field', 'violations'}``.
    ``fail_loud`` raises :class:`ValueError` listing the violations.
    """
    from training.masking import _field_surfaces, _field_values_conflict

    rows = _minted_rows(audit)
    report = {
        "checked": 0,
        "genuine": 0,
        "quality": 1.0,
        "per_field": {},
        "violations": [],
    }

    def violation(row, reason: str) -> None:
        report["violations"].append({
            "copy_payload_idx": row.get("copy_payload_idx"),
            "field": (row.get("fields_hit") or [None])[0],
            "reason": reason,
        })

    for row in rows:
        report["checked"] += 1
        fields = list(row.get("fields_hit") or [])
        field = fields[0] if fields else None
        copy_i = int(row["copy_payload_idx"])
        pair_i = int(row["pair_payload_idx"])
        anchor_i = int(row["anchor_payload_idx"])
        report["per_field"][field] = report["per_field"].get(field, 0) + 1
        if len(fields) != 1 or field is None:
            violation(row, "minted negative must name exactly one transplanted field")
            continue
        copy_text = payload[copy_i]
        pair_text = payload[pair_i]
        transplanted = (row.get("fields_after") or {}).get(field) or []
        if not transplanted:
            violation(row, "audit records no transplanted value")
            continue
        # The transplant IS the label: the copy declares a value the pair side
        # does not, on the field the audit names. Without this the row is a
        # silent false negative (identical listings labeled 0).
        if not _field_values_conflict(
            field,
            _field_surfaces(copy_text).get(field, []),
            _field_surfaces(pair_text).get(field, []),
        ):
            violation(row, "copy does not conflict with the pair side on its field")
            continue
        # Same listing, so the 0 comes from the attribute and not from identity.
        if str(row_bc[copy_i]) != str(row["gtin"]) or str(row_bc[anchor_i]) != str(row["gtin"]):
            violation(row, "minted copy does not keep its anchor entity")
            continue
        # Donor provenance: the value exists elsewhere, in a different entity.
        donor_i = row.get("donor_anchor_payload_idx")
        if donor_i is None or not 0 <= int(donor_i) < len(payload):
            violation(row, "minted negative records no usable donor")
            continue
        donor_values = _field_surfaces(payload[int(donor_i)]).get(field, [])
        if not set(transplanted).issubset(set(donor_values)):
            violation(row, "transplanted value is absent from the recorded donor")
            continue
        if str(row_bc[int(donor_i)]) == str(row["gtin"]):
            violation(row, "donor shares the anchor entity (self-donated value)")
            continue
        # The recorded transplant must be the value the copy actually carries:
        # this also catches context masking that erased the field that makes
        # the row a 0 (``minted_masked`` rows).
        if not set(transplanted).issubset(set(_field_surfaces(copy_text).get(field, []))):
            violation(row, "recorded transplanted value is absent from the copy text")
            continue
        report["genuine"] += 1
    if report["checked"]:
        report["quality"] = report["genuine"] / report["checked"]
    if fail_loud and report["violations"]:
        raise ValueError(
            f"minted-negative label quality failed: {len(report['violations'])}/"
            f"{report['checked']} rows are not genuine label-0 rows: "
            + json.dumps(report["violations"][:5])
        )
    return report


def assert_label_quality(report: dict) -> None:
    """Refuse a bundle whose minted negatives are not all genuine label-0 rows."""
    if report.get("violations"):
        raise ValueError(
            f"minted-negative label quality failed: {len(report['violations'])}/"
            f"{report['checked']} rows are not genuine label-0 rows: "
            + json.dumps(report["violations"][:5])
        )


def main(argv: list[str] | None = None) -> int:
    from core.common import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None,
                        help="prepared text bundle to audit (minted negatives)")
    parser.add_argument("--assert-quality", action="store_true",
                        help="exit non-zero when any minted negative fails its check")
    parser.add_argument("--mask-frac", type=float, default=None,
                        help="configured masking.frac (default: config/training.yaml)")
    args = parser.parse_args(argv)
    configured = (
        float(args.mask_frac) if args.mask_frac is not None
        else float(load_config()["training"]["masking"]["frac"])
    )
    plan = experiment_plan(configured)
    print(f"[augmentation-experiment] plan={json.dumps(plan, sort_keys=True)}", flush=True)
    if args.bundle is None:
        return 0
    from training.prepared_bundle import load_prepared_bundle
    _, data = load_prepared_bundle(args.bundle)
    report = minted_negative_label_quality(
        data.get("hard_negative_mask_audit", []), data["payload"], data["row_bc"]
    )
    print(f"[augmentation-experiment] label_quality={json.dumps(report, sort_keys=True)}",
          flush=True)
    if args.assert_quality:
        assert_label_quality(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
