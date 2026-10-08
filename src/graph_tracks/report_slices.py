"""Unseen / sparse-neighborhood / isolated / missing-field generalization slices.

The model plan requires these four slices next to the attribute slices
("Generalization and coverage"). The attribute slices were implemented in
``graph_tracks.report_attributes``; these were not implemented at all, in any
lane.

All four are properties of the *listing catalog and the pair population*, so
they are computed once per run from the records plus the shared attribute
registry rather than per track. That is deliberate: the catalog is identical
across lanes, so deriving them per lane would produce three identical CSVs and
a fourth place for the definitions to drift.

Definitions (all derived from the normalized attribute values already produced
by ``training.attribute_separation``):

``unseen``
    The endpoint carries at least one critical attribute value that appears on
    no listing in the training-support split (train). This is the
    plan's "unseen values ... include examples of cross-flavor/cross-pack
    neighborhoods": a value the model has provably never been trained on.
``sparse_neighborhood``
    The endpoint shares at least one normalized attribute value with at most
    ``sparse_neighborhood_max_peers`` other catalog listings. Thin graph
    neighborhoods are where two-hop aggregation has nothing to aggregate.
``isolated``
    The endpoint shares no normalized attribute value with any other listing.
``missing_field``
    The endpoint has an empty value set for at least one critical attribute.

A pair is in a slice when *either* endpoint qualifies, which matches how the
attribute slices treat a pair (``attribute__1`` / ``attribute__2`` columns).
Slice membership is reported, never used to drop rows from the headline
metrics.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from core.common import generalization_slice_cfg, precision_at_recall_key
from core.eval_trace import SliceMetricsRow, UnmeasuredSliceRow, row_from_csv
from core.ranking_metrics import POOLED_METRIC_PREFIX
from graph_tracks.report import PAIR_METRIC_KEYS

SLICES = ("unseen", "sparse_neighborhood", "isolated", "missing_field")

#: The identity/accounting columns every slice row carries, DERIVED from the
#: contract model rather than retyped here. ``SliceMetricsRow`` and
#: ``UnmeasuredSliceRow`` declare the same names (the populated row only adds
#: metric columns through ``extra='allow'``), so one subtraction base covers
#: both; a rename in core.eval_trace moves this set at once.
ROW_KEY_COLUMNS: tuple[str, ...] = tuple(SliceMetricsRow.model_fields)


def _recall_pattern() -> re.Pattern[str]:
    """``p_at_r<recall>``, with the static prefix taken from the SSOT function.

    ``precision_at_recall_key()`` renders ``evaluation.operating_recall`` into a
    column name, so the numeric tail is what changes on a retune; only the
    static part is pinned here, and it is read from the function, not typed.
    """
    static = re.sub(r"[0-9.]+$", "", precision_at_recall_key())
    return re.compile(rf"^{re.escape(static)}[0-9.]+$")


def _pooled_pattern() -> re.Pattern[str]:
    """The ``ranking_at_k`` ladder as ``pair_metrics`` forwards it.

    Only the ``pooled_`` family reaches the emitter (the bare aliases are
    filtered out), and a slice with no positive pair emits none of them, so the
    ladder is matched by SHAPE rather than enumerated from a second registry.
    """
    prefix = re.escape(POOLED_METRIC_PREFIX)
    return re.compile(rf"^{prefix}(hits_at_1|(precision|recall)_at_[0-9]+)$")


#: The metric column families whose names are decided by config at emission
#: time. Everything else a populated row carries must come from
#: ``PAIR_METRIC_KEYS`` (derived from ``pair_metrics`` itself).
DYNAMIC_METRIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    _recall_pattern(), _pooled_pattern())


def _is_dynamic(key: str) -> bool:
    return any(pattern.match(key) for pattern in DYNAMIC_METRIC_PATTERNS)


def _blank(value) -> bool:
    """True for a cell a CSV-shape unmeasured row leaves unpopulated."""
    return value is None or value is False or (
        isinstance(value, str) and not value.strip())


def _check_metric_columns(row: dict, index: int, *, unmeasured: bool) -> None:
    """Check the metric surface of ONE row against the owning emitter.

    The fixed key set is ``PAIR_METRIC_KEYS`` (derived from ``pair_metrics`` in
    its OWN module -- never a list retyped here) and the two config-decided
    families are matched by pattern. An undeclared column fails; a populated row
    missing a fixed column fails; an unmeasured row carrying a metric value
    fails.
    """
    fixed = set(PAIR_METRIC_KEYS)
    extras = set(row) - set(ROW_KEY_COLUMNS)
    unknown = sorted(key for key in extras
                     if key not in fixed and not _is_dynamic(key))
    if unknown:
        raise ValueError(
            f"slice row {index} carries metric columns not in the emitter's "
            f"surface: {unknown}; pair_metrics owns {sorted(fixed)} and the "
            f"dynamic families are "
            f"{[pattern.pattern for pattern in DYNAMIC_METRIC_PATTERNS]}")
    if unmeasured:
        populated = sorted(key for key in extras if not _blank(row[key]))
        if populated:
            raise ValueError(
                f"slice row {index} is evaluated=False but carries metric "
                f"values for {populated}")
        return
    missing = sorted(fixed - set(row))
    if missing:
        raise ValueError(
            f"slice row {index} is evaluated=True but misses pair_metrics "
            f"columns {missing}")


def assert_slice_rows(rows: list[dict]) -> list[dict]:
    """Validate ``report()``'s rows; return the ORIGINAL dicts, untouched.

    The contract is a PREDICATE over the emitter's own rows: dispatch on
    ``evaluated`` into ``SliceMetricsRow`` / ``UnmeasuredSliceRow`` (the two
    key sets are distinct on purpose), then check the metric columns against
    the owning emitter's surface. No key is renamed, no row is rebuilt and no
    value is coerced in place, so a caller still hands the ORIGINAL dicts to
    ``pd.DataFrame(rows).to_csv(...)`` and the emitted bytes are unchanged.

    Rows read back from a persisted CSV are string-typed, so the disk -> emitter
    mapping (``row_from_csv``) is applied to a COPY for validation only.
    """
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"slice row {index} is {type(row).__name__}, not a dict")
        coerced = row_from_csv(row)   # copy; ``row`` itself is never touched
        evaluated = coerced.get("evaluated")
        if evaluated is True:
            SliceMetricsRow.model_validate(coerced)
            _check_metric_columns(row, index, unmeasured=False)
        elif evaluated is False:
            UnmeasuredSliceRow.model_validate(coerced)
            _check_metric_columns(row, index, unmeasured=True)
        else:
            raise ValueError(
                f"slice row {index}: evaluated must be True or False, "
                f"got {evaluated!r}")
    return rows


def _normalized_values(records: list[dict]) -> dict[str, dict[str, frozenset[str]]]:
    from training.attribute_separation import ATTRIBUTE_SOURCES

    values: dict[str, dict[str, frozenset[str]]] = {}
    for record in records:
        sku = record["sku_id"]
        values[sku] = {}
        for attribute in ATTRIBUTE_SOURCES:
            raw = record["numeric"].get("volume_ml" if attribute == "volume" else attribute)
            if raw is None:
                raw = record["attribute"].get(attribute, [])
            # Numeric attributes are stored as bare scalars, so normalize both
            # shapes rather than assuming a list.
            if raw is None:
                raw = ()
            elif isinstance(raw, (str, int, float)):
                raw = (raw,)
            values[sku][attribute] = frozenset(
                f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in raw
            )
    return values


def classify(records: list[dict]) -> dict[str, set[str]]:
    """Return ``{slice_name: {sku_id, ...}}`` for the whole catalog."""
    cfg = generalization_slice_cfg()
    max_peers = int(cfg["sparse_neighborhood_max_peers"])
    observed_split = str(cfg["observed_split"])
    values = _normalized_values(records)

    observed: set[tuple[str, str]] = set()
    for record in records:
        if record.get("split") != observed_split:
            continue
        for attribute, attribute_values in values[record["sku_id"]].items():
            observed.update((attribute, value) for value in attribute_values)

    # peer[s] = number of OTHER listings sharing at least one attribute value
    inverted: dict[tuple[str, str], set[str]] = {}
    own_values: dict[str, set[tuple[str, str]]] = {}
    for sku, per_attribute in values.items():
        mine: set[tuple[str, str]] = set()
        for attribute, attribute_values in per_attribute.items():
            mine.update((attribute, value) for value in attribute_values)
        own_values[sku] = mine
        for value in mine:
            inverted.setdefault(value, set()).add(sku)
    peers: dict[str, int] = {}
    for sku, mine in own_values.items():
        shared: set[str] = set()
        for value in mine:
            shared |= inverted.get(value, set())
        shared.discard(sku)
        peers[sku] = len(shared)

    result: dict[str, set[str]] = {name: set() for name in SLICES}
    for record in records:
        sku = record["sku_id"]
        per_attribute = values[sku]
        every_value = own_values[sku]
        if any(value not in observed for value in every_value):
            result["unseen"].add(sku)
        if peers[sku] == 0:
            result["isolated"].add(sku)
        elif peers[sku] <= max_peers:
            result["sparse_neighborhood"].add(sku)
        if any(len(attribute_values) == 0 for attribute_values in per_attribute.values()):
            result["missing_field"].add(sku)
    return result


def slice_frame(records: list[dict], scored: pd.DataFrame) -> pd.DataFrame:
    """Attach a boolean column per slice to the scored-pair frame.

    ``scored`` carries ``sku_id1``/``sku_id2``/``true_label``/``score``/
    ``split``.  Returns a copy with ``slice__<name>`` columns appended.
    """
    membership = classify(records)
    frame = scored.copy()
    for name in SLICES:
        members = membership[name]
        frame[f"slice__{name}"] = (
            frame["sku_id1"].isin(members) | frame["sku_id2"].isin(members)
        )
    return frame


def report(records: list[dict], scored: pd.DataFrame, *, track: str, output: Path,
           pair_metrics, threshold: float, ks=()) -> list[dict]:
    """Score every slice and write ``<track>__slice_metrics.csv``.

    ``pair_metrics`` is injected rather than imported so the caller controls
    the metric definition and so this module never becomes a second one.
    ``ks`` is the caller's retrieval ladder, passed through so a retuned
    ``evaluation.ann_recall_ks`` reaches the slice rows too.
    """
    from graph_tracks.artifacts import name

    frame = slice_frame(records, scored)
    rows: list[dict] = []
    for split, group in frame.groupby("split", sort=True):
        for name_slice in SLICES:
            population = group[group[f"slice__{name_slice}"]]
            if not len(population):
                # An absent slice is a coverage fact, not a missing metric:
                # record it with zero rows and both_classes False so the CSV
                # shows the slice was evaluated and found empty.
                rows.append({
                    "model": track, "split": split, "slice": name_slice,
                    "rows": 0, "positive_pairs": 0, "negative_pairs": 0,
                    "both_classes": False, "evaluated": False,
                })
                continue
            metrics = pair_metrics(
                population["true_label"].to_numpy(),
                population["score"].to_numpy(),
                threshold,
                ks,
            )
            rows.append({"model": track, "split": split, "slice": name_slice,
                         "evaluated": True, **metrics})
    # The emitter checks its OWN rows before persisting them; validation is a
    # predicate, so the dicts handed to pandas below are the same objects.
    assert_slice_rows(rows)
    pd.DataFrame(rows).to_csv(output / name(track, "slice_metrics.csv"), index=False)
    return rows


__all__ = ["DYNAMIC_METRIC_PATTERNS", "ROW_KEY_COLUMNS", "SLICES",
           "assert_slice_rows", "classify", "report", "slice_frame"]