"""Prioritize connected false-merge components for targeted gate review.

The input is a scored fold pair dump.  Negative source-to-canonical edges at
or above the supplied baseline threshold are connected into bipartite
components, enriched from the source and canonical catalogs, and replayed
through the current ANN gate.  Components below ``--min-component-size`` are
omitted so reviewers can start with chained, high-impact failures.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from core.common import rand_matching_cfg
from core.attribute_conflicts import sku_attribute_info
from training.rand_matching import _annotate_candidates, candidate_gate_fields


def _components(edges: pd.DataFrame) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        left = f"sku:{row.sku_id_a}"
        right = f"gtin:{str(row.sku_id_b).removeprefix('canon#')}"
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(adjacency)
    result: list[set[str]] = []
    while unseen:
        root = min(unseen)
        component: set[str] = set()
        queue = deque([root])
        unseen.remove(root)
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        result.append(component)
    return sorted(result, key=lambda nodes: (-len(nodes), sorted(nodes)))


def _bridge_scores(component_edges: pd.DataFrame) -> list[float]:
    """Return scores of edges whose removal disconnects their component."""
    records = list(component_edges.itertuples(index=False))
    bridge_scores: list[float] = []
    for removed_index, removed in enumerate(records):
        adjacency: dict[str, set[str]] = defaultdict(set)
        nodes: set[str] = set()
        for index, row in enumerate(records):
            left = f"sku:{row.sku_id_a}"
            right = f"gtin:{str(row.sku_id_b).removeprefix('canon#')}"
            nodes.update((left, right))
            if index != removed_index:
                adjacency[left].add(right)
                adjacency[right].add(left)
        start = min(nodes)
        reached = {start}
        queue = deque([start])
        while queue:
            for neighbor in adjacency[queue.popleft()]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        if len(reached) != len(nodes):
            bridge_scores.append(float(removed.score))
    return bridge_scores


def build_prioritized_component_report(
    pairs: pd.DataFrame,
    source: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    baseline_threshold: float = 0.60,
    min_component_size: int = 3,
) -> pd.DataFrame:
    required = {
        "label",
        "sku_id_a",
        "sku_id_b",
        "score",
        "volume_conflict",
        "pack_conflict",
        "package_type_conflict",
        "flavor_conflict",
        "attribute_conflict_type",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"pair dump missing required columns: {missing}")
    false_edges = pairs.loc[
        pairs["label"].eq(0) & pairs["score"].ge(float(baseline_threshold))
    ].copy()
    source_map = {
        str(row["product_id"]): row for row in source.to_dict(orient="records")
    }
    canonical_map = {
        str(row["gtin"]): row for row in canonical.to_dict(orient="records")
    }
    output: list[dict[str, object]] = []
    component_number = 0
    for nodes in _components(false_edges):
        if len(nodes) < min_component_size:
            continue
        component_number += 1
        sku_ids = sorted(
            node.removeprefix("sku:") for node in nodes if node.startswith("sku:")
        )
        gtins = sorted(
            node.removeprefix("gtin:") for node in nodes if node.startswith("gtin:")
        )
        component_edges = false_edges.loc[
            false_edges["sku_id_a"].astype(str).isin(sku_ids)
            & false_edges["sku_id_b"].astype(str).str.removeprefix("canon#").isin(gtins)
        ].copy()
        bridge_scores = _bridge_scores(component_edges)
        weakest_bridge = min(bridge_scores) if bridge_scores else float("nan")
        component_id = f"fold0_component_{component_number:03d}"
        for edge in component_edges.sort_values("score", ascending=False).itertuples(
            index=False
        ):
            sku_id = str(edge.sku_id_a)
            candidate_gtin = str(edge.sku_id_b).removeprefix("canon#")
            if sku_id not in source_map:
                raise KeyError(f"source SKU absent from live catalog: {sku_id}")
            if candidate_gtin not in canonical_map:
                raise KeyError(
                    f"candidate GTIN absent from canonical catalog: {candidate_gtin}"
                )
            source_row = source_map[sku_id]
            canonical_row = canonical_map[candidate_gtin]
            gate = candidate_gate_fields(
                pd.Series(source_row),
                sku_attribute_info(
                    source_row.get("title"), source_row.get("attributes")
                ),
                candidate_gtin,
                canonical_row,
                float(edge.score),
                sku_id=sku_id,
                source_row_index=sku_id,
                retrieval_source="fold0_false_merge_replay",
            )
            annotated = _annotate_candidates(
                pd.DataFrame([gate]),
                0.61,
                threshold_by_gtin_status=rand_matching_cfg()[
                    "threshold_by_gtin_status"
                ],
            ).iloc[0]
            output.append(
                {
                    "priority_rank": component_number,
                    "component_id": component_id,
                    "component_size": len(nodes),
                    "component_edge_count": len(component_edges),
                    "weakest_bridge_score": weakest_bridge,
                    "component_sku_ids": json.dumps(sku_ids),
                    "component_gtins": json.dumps(gtins),
                    "sku_id": sku_id,
                    "sku_gtin": gate["sku_gtin"],
                    "candidate_gtin": candidate_gtin,
                    "score_raw": float(edge.score),
                    "score_after_penalties": float(gate["score"]),
                    "effective_threshold": float(annotated["effective_threshold"]),
                    "accepted_after_gates": int(bool(annotated["accepted"])),
                    "assignment_gate": annotated["assignment_gate"],
                    "rejection_reason": annotated["rejection_reason"],
                    "targeted_gate_decision": gate["targeted_gate_decision"],
                    "targeted_gate_route": gate["targeted_gate_route"],
                    "targeted_gate_reason": gate["targeted_gate_reason"],
                    "gtin_status": gate["gtin_status"],
                    "source_brand": gate["sku_brand"],
                    "candidate_brand": gate["candidate_brand"],
                    "source_pack": gate["targeted_pack_a"],
                    "candidate_pack": gate["targeted_pack_b"],
                    "source_volume_ml": gate["targeted_volume_ml_a"],
                    "candidate_volume_ml": gate["targeted_volume_ml_b"],
                    "source_package_type": gate["targeted_package_type_a"],
                    "candidate_package_type": gate["targeted_package_type_b"],
                    "source_flavor": gate["sku_flavor"],
                    "candidate_flavor": gate["candidate_flavor"],
                    "prior_attribute_conflict_type": edge.attribute_conflict_type,
                    "pack_conflict": gate["targeted_pack_conflict"],
                    "volume_conflict": gate["targeted_volume_conflict"],
                    "package_type_conflict": gate[
                        "targeted_package_type_conflict"
                    ],
                    "brand_conflict": gate["targeted_brand_conflict"],
                    "flavor_conflict": int(edge.flavor_conflict),
                }
            )
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-threshold", type=float, default=0.60)
    parser.add_argument("--min-component-size", type=int, default=3)
    args = parser.parse_args()
    report = build_prioritized_component_report(
        pd.read_csv(args.pairs, low_memory=False),
        pd.read_csv(args.source, dtype=str, keep_default_na=False),
        pd.read_csv(args.canonical, dtype=str, keep_default_na=False),
        baseline_threshold=args.baseline_threshold,
        min_component_size=args.min_component_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(
        f"wrote {len(report)} edges across "
        f"{report['component_id'].nunique() if not report.empty else 0} components "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
