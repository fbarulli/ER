"""Per-fold coverage of the masked-positive copies a frozen objective trains.

The MNRL objective trains a masked-positive copy only when the copy reaches a
triple IN ITS OWN FOLD. A bundle-level "the copy's source anchor owns a
negative somewhere in the training pool" heuristic therefore MIS-MEASURES
survival: it ignores the minted-twin lineage (``training.training
._negatives_by_anchor`` adds the counterfactual twin copies to their source
anchor's negatives), so on the shipped bundle it reports 980 of 1,200 masked
positives dead while the frozen objective trains all 1,200. A gate that
subtracts the 980 non-existent dead rows ("backfill-then-remove") understates
the positive side and hides which copies genuinely never reached a gradient.

This module measures the quantity the loss actually sees: for each fold of the
frozen objective, how many of the masked-positive copies appear as triple
anchors — the (copy, positive, negative) MNRL presentation is exactly one
gradient-bearing row. Coverage is reported PER FOLD plus the union, so a copy
that only survives in some folds is visible instead of averaged away, and
``never_trained`` names the copies no fold trains (the rows a
generate-only-if-covered policy must refuse to mint rather than backfill).
Bundles without a complete frozen plan return no folds; the diet gate keeps its
documented bundle-level fallback for them.
"""
from __future__ import annotations


def folded_objectives(data: dict) -> list[tuple[object, list]]:
    """The frozen objective's ``(fold label, triples)`` pairs, [] without a plan.

    Mirrors the completeness contract the diet gate already enforces on the
    frozen MNRL objective (``training_plan.identity.loss == 'mnrl'``, no
    skipped folds, one dataset row per triple). A missing or non-MNRL plan
    returns ``[]`` so the caller can keep the bundle-level fallback; an
    INCOMPLETE frozen plan raises, exactly like the gate's own accounting.
    """
    plan = data.get("training_plan")
    if plan is None or plan.get("identity", {}).get("loss") != "mnrl":
        return []
    inputs = plan.get("inputs") or {}
    if inputs.get("skipped") or not inputs.get("folds"):
        raise ValueError("MNRL diet requires a complete frozen objective")
    folds: list[tuple[object, list]] = []
    for fold in inputs["folds"]:
        objective = fold["objective"]
        triples = objective["triples"]
        dataset = objective["dataset"]
        if any(len(dataset[key]) != len(triples) for key in ("anchor", "positive", "negative")):
            raise ValueError("frozen MNRL triple and dataset rows differ")
        folds.append((fold["fold_i"], triples))
    return folds


def masked_positive_coverage(
    mask_audit: list[dict] | None,
    folds: list[tuple[object, list]],
    *,
    source_negatives=None,
) -> dict:
    """Per-fold survival of the masked-positive copies, plus the union.

    ``folds`` is :func:`folded_objectives` output. A copy is TRAINED in a fold
    when its ``copy_payload_idx`` is one of that fold's triple anchors. The
    returned mapping carries every fold's count/coverage, the union across
    folds, ``never_trained`` (union complement — the copies no fold trains),
    and ``source_anchor_covered`` (copies whose source anchor owns an explicit
    negative in the training pool). The last one is the lower bound the
    bundle-level heuristic sees; it is NOT survival, which is why both are
    reported side by side.
    """
    copies = [int(row["copy_payload_idx"]) for row in mask_audit or []]
    total = len(copies)
    per_fold = []
    trained_union: set[int] = set()
    for label, triples in folds:
        anchors = {int(triple[0]) for triple in triples}
        trained = [copy for copy in copies if copy in anchors]
        trained_union.update(trained)
        per_fold.append({
            "fold": label,
            "presentations": len(triples),
            "trained": len(trained),
            "coverage": (len(trained) / total) if total else 0.0,
        })
    covered_sources = 0
    if source_negatives is not None:
        negatives = {
            int(anchor) for anchor, _negative in
            _pairs(source_negatives)
        }
        covered_sources = sum(
            1 for row in mask_audit or []
            if int(row["anchor_payload_idx"]) in negatives
        )
    return {
        "copies": total,
        "folds": per_fold,
        "trained_union": len(trained_union),
        "never_trained": total - len(trained_union),
        "source_anchor_covered": covered_sources,
    }


def _pairs(values):
    """Iterate a 2-column pair array without importing numpy here."""
    for row in values:
        yield row[0], row[1]
