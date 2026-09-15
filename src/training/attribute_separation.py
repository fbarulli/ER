"""Attribute separation metrics: how well each attribute separates pairs.

The defect this makes measurable: brand was separating true pairs from false
ones by only +0.0249 on the model score — a number produced once, by hand. This
module turns that into a repeatable, per-attribute and per-VALUE measurement
computed from labelled pairs plus canonical attributes alone: **no model, no
training run, CPU only.**

Definition.  For an attribute A and a labelled pair population, with the
attribute value set ``v(side)`` on each side of a pair:

* ``observable`` — at least one side carries a value for A.  Pairs where NEITHER
  side carries one hold no information about A and are counted separately as
  ``n_unobservable`` rather than silently scored as agreement.
* ``agree``      — both sides carry a value AND the sets intersect.
* ``separation`` — ``P(agree | positive) - P(agree | negative)`` over the
  observable pairs.  0 means the attribute does not separate at all.

Per value, anchored on the FIRST side: ``separation_v`` is
``P(v also on the other side | positive, v on side 1)`` minus the same
conditional on negatives — i.e. "when this brand appears, how much more often
is it a true match than a false one".

STATISTICAL HONESTY.  Every row carries its support counts, and a score is only
``reportable`` — and only ever ``flagged_weak`` — when BOTH classes clear the
configured support floor.  A brand observed in two pairs is reported with its
counts and is never called a defect; the schemas reject a flagged-but-
unsupported row outright.
"""
from __future__ import annotations

import argparse

import pandas as pd

from core.common import F, ensure_parent, load_config
from core.schemas import (
    SEPARATION_SUMMARY_COLUMNS,
    SEPARATION_VALUE_COLUMNS,
    AttributeSeparationSpec,
    SeparationSummaryRow,
    SeparationValueRow,
)

# Attribute -> (canonical_records column, is_numeric).  This mirrors the
# structured-attribute schema in core.structured_features; it is a
# schema-bound constant, not a tunable, so it is deliberately not config.
# ``category`` has no column in canonical_records.csv (verified), so it is not
# silently omitted from the table — see ATTRIBUTE_UNAVAILABLE.
ATTRIBUTE_SOURCES: dict[str, tuple[str, str]] = {
    "brand": ("mode_brand", "scalar"),
    "volume": ("volume_set", "volume_set"),
    "pack": ("pack_set", "pack_set"),
    "package_type": ("package_type_set", "string_set"),
    "flavor": ("flavor_set", "string_set"),
    "carbonation": ("carbonation_set", "string_set"),
    "sweetener": ("sweetener_set", "string_set"),
    "pulp": ("pulp_set", "string_set"),
}
ATTRIBUTE_UNAVAILABLE: dict[str, str] = {
    "category": (
        "canonical_records.csv carries no category column; the per-SKU "
        "category in dataset.csv is not a GTIN-level attribute and does not "
        "join to the pair population"
    )
}


def separation_spec() -> AttributeSeparationSpec:
    """The validated separation settings (config SSOT)."""
    return AttributeSeparationSpec.model_validate(
        load_config()["evaluation"]["attribute_separation"]
    )


def _values(raw: object, *, kind: str) -> frozenset[str]:
    """Parse one canonical attribute cell into a normalized value set.

    The cell kinds differ in the artifact and must not be conflated: the
    ``*_set`` columns are Python set reprs, ``mode_brand`` is a plain scalar
    string, and volume/pack each need their OWN canonicalizer.
    """
    from core.structured_features import _as_set, _as_string_set
    from pipeline import normalize_text

    if kind == "scalar":
        normalized = " ".join(normalize_text(raw).split())
        return frozenset({normalized} if normalized else ())
    if kind == "string_set":
        return frozenset(_as_string_set(raw, kind="attribute"))
    unit = "volume" if kind == "volume_set" else "pack"
    return frozenset(f"{value:g}" for value in _as_set(raw, kind=unit))


def separation_population(
    labeled_pairs: pd.DataFrame, canonical_records: pd.DataFrame
) -> pd.DataFrame:
    """Join labelled pairs to both sides' canonical attributes.

    Raises if a GTIN is missing from the canonical records: dropping such a
    pair would silently shrink the population the scores are computed over.
    """
    required = {"gtin1", "gtin2", "true_label"}
    missing = required - set(labeled_pairs.columns)
    if missing:
        raise ValueError(f"labeled pairs missing columns {sorted(missing)}")
    records = canonical_records.set_index(canonical_records["gtin"].astype(str))
    gtin1 = labeled_pairs["gtin1"].astype(str)
    gtin2 = labeled_pairs["gtin2"].astype(str)
    absent = sorted((set(gtin1) | set(gtin2)) - set(records.index))
    if absent:
        raise ValueError(
            f"{len(absent)} labelled-pair GTINs have no canonical record "
            f"(e.g. {absent[:5]}) — refusing to score a silently smaller population"
        )
    frame = pd.DataFrame(
        {"true_label": labeled_pairs["true_label"].astype(int).to_numpy()}
    )
    for attribute, (column, kind) in ATTRIBUTE_SOURCES.items():
        if column not in canonical_records.columns:
            raise ValueError(f"canonical records missing column {column!r}")
        frame[f"{attribute}__1"] = [
            _values(records.at[g, column], kind=kind) for g in gtin1
        ]
        frame[f"{attribute}__2"] = [
            _values(records.at[g, column], kind=kind) for g in gtin2
        ]
    return frame


def attribute_separation(
    population: pd.DataFrame, *, spec: AttributeSeparationSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary rows, per-value rows) for every attribute."""
    positive = population["true_label"].eq(1)
    negative = population["true_label"].eq(0)
    summary: list[dict[str, object]] = []
    by_value: list[dict[str, object]] = []

    for attribute in ATTRIBUTE_SOURCES:
        left = population[f"{attribute}__1"]
        right = population[f"{attribute}__2"]
        observable = left.ne(frozenset()) | right.ne(frozenset())
        agree = pd.Series(
            [bool(a & b) for a, b in zip(left, right)], index=population.index
        )
        n_pos = int((positive & observable).sum())
        n_neg = int((negative & observable).sum())
        p_pos = float((positive & observable & agree).sum() / n_pos) if n_pos else 0.0
        p_neg = float((negative & observable & agree).sum() / n_neg) if n_neg else 0.0
        reportable = (
            n_pos >= spec.min_pairs_per_class and n_neg >= spec.min_pairs_per_class
        )
        separation = p_pos - p_neg
        summary.append(
            SeparationSummaryRow(
                attribute=attribute,
                n_positive=n_pos,
                n_negative=n_neg,
                n_unobservable=int((~observable).sum()),
                p_agree_positive=p_pos,
                p_agree_negative=p_neg,
                separation=separation,
                reportable=reportable,
                flagged_weak=bool(reportable and separation <= spec.flag_below),
                negative_class_saturated=bool(n_neg and p_neg == 1.0),
            ).model_dump()
        )

        for value in sorted({v for values in left for v in values}):
            carries = left.map(lambda vs, v=value: v in vs)
            n_pos_v = int((positive & carries).sum())
            n_neg_v = int((negative & carries).sum())
            if not n_pos_v and not n_neg_v:
                continue
            p_pos_v = (
                float((positive & carries & right.map(lambda vs, v=value: v in vs)).sum() / n_pos_v)
                if n_pos_v
                else 0.0
            )
            p_neg_v = (
                float((negative & carries & right.map(lambda vs, v=value: v in vs)).sum() / n_neg_v)
                if n_neg_v
                else 0.0
            )
            reportable_v = (
                n_pos_v >= spec.min_value_support and n_neg_v >= spec.min_value_support
            )
            separation_v = p_pos_v - p_neg_v
            by_value.append(
                SeparationValueRow(
                    attribute=attribute,
                    value=value,
                    n_positive=n_pos_v,
                    n_negative=n_neg_v,
                    p_match_positive=p_pos_v,
                    p_match_negative=p_neg_v,
                    separation=separation_v,
                    reportable=reportable_v,
                    flagged_weak=bool(reportable_v and separation_v <= spec.flag_below),
                ).model_dump()
            )

    return (
        pd.DataFrame(summary, columns=list(SEPARATION_SUMMARY_COLUMNS)),
        pd.DataFrame(by_value, columns=list(SEPARATION_VALUE_COLUMNS)).sort_values(
            ["separation", "attribute", "value"], ignore_index=True
        ),
    )


def write_separation_reports(
    labeled_pairs: pd.DataFrame,
    canonical_records: pd.DataFrame,
    *,
    spec: AttributeSeparationSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and persist both artifacts through the registered paths."""
    resolved = separation_spec() if spec is None else spec
    summary, by_value = attribute_separation(
        separation_population(labeled_pairs, canonical_records), spec=resolved
    )
    for frame, key in (
        (summary, "attribute_separation_summary"),
        (by_value, "attribute_separation_values"),
    ):
        path = ensure_parent(F[key])
        frame.to_csv(path, index=False)
    return summary, by_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=F["labeled_pairs"])
    parser.add_argument("--canonicals", default=F["canonical_records"])
    args = parser.parse_args()
    spec = separation_spec()
    if not spec.enabled:
        raise SystemExit("evaluation.attribute_separation.enabled is false")
    labeled = pd.read_csv(args.pairs, dtype={"gtin1": str, "gtin2": str})
    canonicals = pd.read_csv(args.canonicals, dtype=str, keep_default_na=False)
    summary, by_value = write_separation_reports(labeled, canonicals, spec=spec)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    flagged = by_value[by_value["flagged_weak"]]
    print(f"\n[separation] {len(by_value):,} values scored; "
          f"{len(flagged):,} flagged weak with support >= {spec.min_value_support}")
    for attribute in ATTRIBUTE_SOURCES:
        worst = by_value[
            by_value["attribute"].eq(attribute) & by_value["reportable"]
        ].head(5)
        if not worst.empty:
            print(f"\n  worst-separated {attribute!r} values (support-cleared):")
            print(worst[["value", "n_positive", "n_negative", "separation"]]
                  .to_string(index=False))


if __name__ == "__main__":
    main()
