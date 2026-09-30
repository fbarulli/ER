"""P0 — emit THE single final validation CSV.

Replaces the retired ``dataset_deduped_sample_3000/5000`` lanes. Those splits
were derived from a graph built WITHOUT the validation census, so the two
sides derived different components from the same data: 74.7% of the old
validation population was contaminated (23.3% of positives had BOTH endpoints
in train, 51.4% had one), which means the P@R95 it reported was measuring
memorization as much as generalization.

The leak cannot be fixed downstream of the split, so this module builds the
graph the split is cut from and then emits the population:

    build_training_data      -> base positive pairs (sku, own canonical)
    merged_component_graph   -> + normalized entity key + labeled positives
    holdout_split            -> train (folds 0+1) / dev (2) / test (3)
    emit                     -> folds 2+3, the single validation population

WHY ``fold``/``component_id`` HAVE A ``_2`` SIBLING
---------------------------------------------------
A POSITIVE pair is one edge in the graph, so both its endpoints are always in
the same component and therefore the same fold -- ``fold == fold_2`` and
``component_id == component_id_2`` for all 1,143 of them, and that equality is
the leak guarantee, asserted below before anything is written.

A NEGATIVE pair is a mined *similarity* relation, not an identity claim, so
its two endpoints are usually in DIFFERENT components, and one of them is
frequently a barcode the model trained on. A single ``fold`` column would have
to silently mean "the fold of gtin1" and hide the other side. So the pair's
both sides are carried explicitly, and ``endpoint_in_train`` marks a negative
whose other side leaked in. Those rows are KEPT (the P0 spec treats
straddling negatives as documented current behaviour, not a regression) but
flagged, so a downstream floor can either exclude or report them instead of
inheriting an invisible 24% contamination.

SLICE FLAGS ARE PER SIDE, NOT PER PAIR
--------------------------------------
Each endpoint's canonical attribute values are frozen into the CSV
(``v1_volume``/``v2_volume``, ...). The pair-level bucket question is left
open on purpose: ``build_field_slice.py`` buckets by TWIN while
``labeled_pairs`` slices by CANONICAL VALUE, and reconciling those is the
separate open "align our gates" decision. Freezing the values stops downstream
re-deriving buckets from scratch; it does not pre-empt which aggregation wins.
For a positive the two sides are the same product, so ``v1_* == v2_*``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.common import F, RESULTS, SEED, load_dataset_deduped, training_cfg
from training.folds import (
    component_ids,
    derive_holdout,
    merged_component_graph,
    normalize_gtin,
)

# The six fields P0 keeps as gates. `pulp_set` is deliberately absent: it is
# populated in 2.3% of canonical records and 0.5% of verified positives, which
# is 2 pairs in this validation half -- population scarcity, not a parsing
# defect, and no gate at any budget that respects the component constraint.
SLICE_FIELDS: tuple[tuple[str, str], ...] = (
    ("volume", "volume_set"),
    ("pack", "pack_set"),
    ("package_type", "package_type_set"),
    ("sweetener", "sweetener_set"),
    ("flavor", "flavor_set"),
    ("carbonation", "carbonation_set"),
)


def _canonical_values() -> dict[str, dict[str, str]]:
    """gtin -> {field: canonical value string} from the frozen canonical records."""
    canon = pd.read_csv(
        F["canonical_records"], dtype=str, keep_default_na=False, low_memory=False
    )
    gtin_col = "gtin" if "gtin" in canon.columns else canon.columns[0]
    canon["_key"] = canon[gtin_col].map(normalize_gtin)
    out: dict[str, dict[str, str]] = {}
    for _, row in canon.iterrows():
        key = row["_key"]
        if not key or key in out:
            continue
        out[key] = {name: str(row.get(col, "") or "") for name, col in SLICE_FIELDS}
    return out


def build(
    output: Path | None = None,
    *,
    seed: int = SEED,
    n_folds: int | None = None,
) -> pd.DataFrame:
    """Derive the merged graph, cut the split, and return the validation rows."""
    df = load_dataset_deduped()
    from pipeline import build_training_data

    data = build_training_data(df, payload_variant="full")
    pos = data["pos"]
    row_bc = data["row_bc"]

    merged_pos, graph_bc, stats = merged_component_graph(pos, row_bc)
    split = training_cfg().split
    n_folds = int(n_folds or split.holdout_component_folds)
    # Routed through the SINGLE entry point, not `holdout_split`. The selftest
    # guard bans the primitive outside folds.py precisely so the split the
    # artifact is cut from cannot be derived by different rules than the split
    # the model trains under -- which is the defect P0 exists to remove. The
    # graph is passed in pre-merged because `derive_holdout` rebuilds it (idempotent
    # here: the validation edges are already unioned, so re-union changes nothing).
    train_bc, dev_bc, test_bc = derive_holdout(
        pos, row_bc, dict(split), seed=seed
    )

    # merged_pos/row_bc: derive_holdout re-derives the same merged graph
    # internally, so these are the identical objects it split on.
    comp_of = component_ids(merged_pos, graph_bc)
    # train = every quarter except the last two; dev/test are the LAST two
    # quarters, so validation is "neither side is a training barcode".
    fold_of: dict[str, int] = {}
    for bc in train_bc:
        fold_of[bc] = 0
    for bc in dev_bc:
        fold_of[bc] = n_folds - 2
    for bc in test_bc:
        fold_of[bc] = n_folds - 1

    # A labeled gtin has to be resolved to the spelling the graph actually
    # uses, and the two are not the same string: the fold sets are keys of the
    # RAW row_bc, while normalize_gtin left-pads a 13-digit barcode to 14. A
    # 13-digit gtin therefore misses a raw fold set under a normalized lookup
    # and is silently dropped -- which is how an earlier run of this script
    # emitted 3 rows out of 8,889. Try the raw spelling first, then the
    # normalized one, and COUNT the misses rather than skipping in silence.
    raw_keys = set(fold_of)
    norm_keys = {normalize_gtin(b) for b in fold_of}

    def resolve(gtin: str) -> str | None:
        raw = str(gtin).strip()
        if raw in raw_keys:
            return raw
        normed = normalize_gtin(raw)
        return normed if normed in raw_keys or normed in norm_keys else None

    labeled = pd.read_csv(
        F["labeled_pairs"], dtype={"gtin1": str, "gtin2": str}, keep_default_na=False
    )
    canon = _canonical_values()

    rows: list[dict[str, object]] = []
    unresolvable = 0
    for g1, g2, label in zip(
        labeled["gtin1"], labeled["gtin2"], labeled["true_label"]
    ):
        k1, k2 = resolve(g1), resolve(g2)
        if k1 is None or k2 is None:
            # An endpoint outside the graph entirely: it has no fold, so it
            # cannot be part of a fold-2+3 population.
            unresolvable += 1
            continue
        if fold_of[k1] < n_folds - 2 and fold_of[k2] < n_folds - 2:
            continue  # both sides in train -> not validation
        f1, f2 = fold_of[k1], fold_of[k2]
        c1 = canon.get(normalize_gtin(k1), {})
        c2 = canon.get(normalize_gtin(k2), {})
        row: dict[str, object] = {
            "gtin1": k1,
            "gtin2": k2,
            "gtin1_norm": normalize_gtin(k1),
            "gtin2_norm": normalize_gtin(k2),
            "true_label": int(label),
            "fold": f1,
            "fold_2": f2,
            "component_id": comp_of.get(k1, -1),
            "component_id_2": comp_of.get(k2, -2),
            "straddles_fold": f1 != f2,
            "endpoint_in_train": min(f1, f2) < n_folds - 2,
        }
        for name, _col in SLICE_FIELDS:
            row[f"v1_{name}"] = c1.get(name, "")
            row[f"v2_{name}"] = c2.get(name, "")
        rows.append(row)

    out = pd.DataFrame(rows)

    # ── the leak guarantee, asserted before anything hits disk ──
    pos_rows = out[out.true_label == 1]
    straddle = int(pos_rows["straddles_fold"].sum())
    if straddle:
        raise SystemExit(
            f"LEAK: {straddle}/{len(pos_rows)} positives straddle a fold. The "
            "merged graph was not applied; refusing to write the CSV."
        )
    # A positive is one edge, so its two endpoints are linked BY DEFINITION.
    # If that ever fails, the edge was dropped and the split can leak.
    if not (pos_rows["component_id"] == pos_rows["component_id_2"]).all():
        raise SystemExit(
            "LEAK: positive endpoints in different components — the graph edge "
            "for that pair was dropped before the union-find ran."
        )

    stats["pairs_endpoint_unresolvable"] = unresolvable
    # The fold map is the split's COMPLETE accounting, and it is not optional.
    # The validation CSV holds only the scored half (folds 2+3), so a consumer
    # holding the full labeled census cannot tell a pair that was correctly
    # withheld because the model trained on it from a pair that is simply
    # MISSING. Without the map, a retargeted evaluator has to choose between
    # scoring trained-on data and hard-failing on rows that are fine — which is
    # how the old protocol ended up 73.7% contaminated with nothing recorded.
    fold_map = pd.DataFrame(
        sorted(
            ({"gtin": bc, "fold": fold_of[bc], "component_id": comp_of.get(bc, -1)}
             for bc in fold_of),
            key=lambda r: (r["fold"], r["gtin"]),
        )
    )
    out_path = Path(output or F["final_validation"])
    write_manifest(
        out,
        stats,
        path=out_path,
        seed=seed,
        fold_map=fold_map,
        fold_map_path=Path(F["validation_fold_map"]),
    )
    return out


def write_manifest(
    frame: pd.DataFrame,
    stats: dict,
    *,
    path: Path,
    seed: int,
    fold_map: pd.DataFrame | None = None,
    fold_map_path: Path | None = None,
) -> dict:
    pos = frame[frame.true_label == 1]
    neg = frame[frame.true_label == 0]

    # Per-field measuring power, because "we have 564 positives" is not the
    # question a gate asks -- "can this field carry a floor" is. `disagree` is
    # the count of positives whose two endpoints carry different values for
    # the field: same product, one side's text mentions an extra value. That is
    # legitimate extractor variance, so it is reported rather than asserted on,
    # but a gate comparing v1 to v2 needs to know it exists.
    coverage: dict[str, dict[str, int]] = {}
    for name, _col in SLICE_FIELDS:
        a, b = pos[f"v1_{name}"], pos[f"v2_{name}"]
        if not len(pos):
            coverage[name] = {"positives": 0, "distinct": 0, "largest_bucket": 0,
                              "unpopulated": 0, "disagree": 0}
            continue
        counts = a[a != ""].value_counts()
        coverage[name] = {
            "positives": int(len(pos)),
            "distinct": int(counts.size),
            "largest_bucket": int(counts.iloc[0]) if counts.size else 0,
            "unpopulated": int((a == "").sum()),
            "disagree": int((a != b).sum()),
        }

    manifest = {
        "stage": "final_validation",
        "complete": True,
        "seed": seed,
        "output": str(path),
        "rows": int(len(frame)),
        "positives": int(len(pos)),
        "negatives": int(len(neg)),
        "positives_straddling_folds": int(pos["straddles_fold"].sum()),
        "positives_with_endpoint_in_train": int(pos["endpoint_in_train"].sum()),
        "negatives_straddling_folds": int(neg["straddles_fold"].sum()),
        "negatives_with_endpoint_in_train": int(neg["endpoint_in_train"].sum()),
        "pairs_endpoint_unresolvable": int(stats.get("pairs_endpoint_unresolvable", 0)),
        "graph": {
            "merged_positive_pairs": int(stats["merged_positive_pairs"]),
            "train_positive_pairs": int(stats["train_positive_pairs"]),
            "validation_edges_added": int(stats["edges_added"]),
            "normalized_entities": int(stats["row_entities"]),
            "endpoints_unresolved": int(stats["endpoints_unresolved"]),
        },
        "slice_fields": [name for name, _ in SLICE_FIELDS],
        "slice_coverage": coverage,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if fold_map is not None and fold_map_path is not None:
        fold_map_path.parent.mkdir(parents=True, exist_ok=True)
        fold_map.to_csv(fold_map_path, index=False)
        manifest["fold_map"] = str(fold_map_path)
        manifest["fold_map_rows"] = int(len(fold_map))
        manifest["fold_map_fold_counts"] = {
            str(k): int(v) for k, v in fold_map["fold"].value_counts().items()
        }
    (RESULTS / "manifests" / "final_validation.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    frame = build(Path(args.output) if args.output else None, seed=args.seed)
    pos = frame[frame.true_label == 1]
    neg = frame[frame.true_label == 0]
    print(
        f"[final_validation] {len(frame):,} pairs -> "
        f"{len(pos):,} positives / {len(neg):,} negatives | "
        f"positives straddling: {int(pos.straddles_fold.sum())}"
    )
    print(f"[final_validation] wrote {F['final_validation']}")

if __name__ == "__main__":
    main()
