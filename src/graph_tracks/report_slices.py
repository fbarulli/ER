"""Unseen / sparse-neighborhood / isolated / missing-field generalization slices.

MODEL_TRACKS_PLAN.md requires these four slices next to the attribute slices
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
    no listing in the *observed* split (dev -- the training half). This is the
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

from pathlib import Path

import numpy as np
import pandas as pd

from core.common import generalization_slice_cfg

SLICES = ("unseen", "sparse_neighborhood", "isolated", "missing_field")


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

    observed: set[str] = set()
    for record in records:
        if record.get("split") != observed_split:
            continue
        for attribute_values in values[record["sku_id"]].values():
            observed |= attribute_values

    # peer[s] = number of OTHER listings sharing at least one attribute value
    inverted: dict[str, set[str]] = {}
    own_values: dict[str, set[str]] = {}
    for sku, per_attribute in values.items():
        mine: set[str] = set()
        for attribute_values in per_attribute.values():
            mine |= attribute_values
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
    pd.DataFrame(rows).to_csv(output / name(track, "slice_metrics.csv"), index=False)
    return rows


__all__ = ["SLICES", "classify", "report", "slice_frame"]