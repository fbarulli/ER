"""Diagnostics for the non-submission SKU-to-canonical candidate graph."""

from __future__ import annotations

import json

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class CandidateGraphFrameSpec(BaseModel):
    """Input-column contract for candidate graph diagnostics."""

    model_config = ConfigDict(extra="forbid")

    required_columns: frozenset[str] = Field(min_length=1)

    def validate_frame(self, frame: pd.DataFrame) -> None:
        missing = sorted(self.required_columns - set(frame.columns))
        if missing:
            raise ValueError(
                "candidate graph input contract violated: "
                f"missing={missing}"
            )


_CANDIDATE_GRAPH_FRAME_SPEC = CandidateGraphFrameSpec(
    required_columns=frozenset(
        {"SKU_ID", "candidate_gtin", "score", "gtin_status", "exact_gtin", "rule_ok"}
    )
)


def candidate_graph_diagnostics(
    candidates: pd.DataFrame,
    threshold: float,
) -> dict[str, float | int | str]:
    """Describe accepted bipartite candidates without assigning components.

    ``plausible_group_count`` is the number of canonical GTINs with at least
    one candidate passing the configured gates at ``threshold``.  It is a
    candidate-space diagnostic, not a replacement for direct assignment.
    """
    _CANDIDATE_GRAPH_FRAME_SPEC.validate_frame(candidates)
    accepted = candidates[
        candidates["gtin_status"].ne("different")
        & (
            candidates["exact_gtin"].astype(bool)
            | (
                candidates["rule_ok"].astype(bool)
                & candidates["score"].ge(float(threshold))
            )
        )
    ]
    if accepted.empty:
        return {
            "diagnostic_edge_count": 0,
            "diagnostic_component_count": 0,
            "diagnostic_component_size_distribution": "{}",
            "diagnostic_max_component_size": 0,
            "diagnostic_score_diameter": 0.0,
            "diagnostic_bridge_edge_count": 0,
            "diagnostic_weakest_bridge_score": float("nan"),
            "plausible_group_count": 0,
        }

    nodes = sorted(
        {
            *(f"sku:{value}" for value in accepted["SKU_ID"].astype(str)),
            *(f"gtin:{value}" for value in accepted["candidate_gtin"].astype(str)),
        }
    )
    index = {node: number for number, node in enumerate(nodes)}
    edges = [
        (index[f"sku:{sku}"], index[f"gtin:{gtin}"], float(score))
        for sku, gtin, score in accepted[
            ["SKU_ID", "candidate_gtin", "score"]
        ].itertuples(index=False)
    ]
    adjacency: list[list[tuple[int, int]]] = [[] for _ in nodes]
    for edge_id, (left, right, _) in enumerate(edges):
        adjacency[left].append((right, edge_id))
        adjacency[right].append((left, edge_id))

    discovery = [-1] * len(nodes)
    low = [-1] * len(nodes)
    bridge_edge_ids: set[int] = set()
    components: list[list[int]] = []
    time_counter = 0

    def visit(node: int, parent_edge: int, component: list[int]) -> None:
        nonlocal time_counter
        discovery[node] = low[node] = time_counter
        time_counter += 1
        component.append(node)
        for neighbour, edge_id in adjacency[node]:
            if edge_id == parent_edge:
                continue
            if discovery[neighbour] < 0:
                visit(neighbour, edge_id, component)
                low[node] = min(low[node], low[neighbour])
                if low[neighbour] > discovery[node]:
                    bridge_edge_ids.add(edge_id)
            else:
                low[node] = min(low[node], discovery[neighbour])

    for node in range(len(nodes)):
        if discovery[node] < 0:
            component: list[int] = []
            visit(node, -1, component)
            components.append(component)

    component_ids = {
        node: component_id
        for component_id, component in enumerate(components)
        for node in component
    }
    edge_scores = [
        [
            score
            for left, right, score in edges
            if component_ids[left] == component_id
            or component_ids[right] == component_id
        ]
        for component_id in range(len(components))
    ]
    diameters = [max(scores) - min(scores) for scores in edge_scores if scores]
    sizes = pd.Series([len(component) for component in components]).value_counts()
    size_distribution = json.dumps(
        {str(int(size)): int(count) for size, count in sizes.items()},
        sort_keys=True,
    )
    weakest_bridge = (
        min(edges[edge_id][2] for edge_id in bridge_edge_ids)
        if bridge_edge_ids
        else float("nan")
    )
    return {
        "diagnostic_edge_count": int(len(edges)),
        "diagnostic_component_count": int(len(components)),
        "diagnostic_component_size_distribution": size_distribution,
        "diagnostic_max_component_size": int(max(map(len, components))),
        "diagnostic_score_diameter": float(max(diameters)) if diameters else 0.0,
        "diagnostic_bridge_edge_count": int(len(bridge_edge_ids)),
        "diagnostic_weakest_bridge_score": float(weakest_bridge),
        "plausible_group_count": int(accepted["candidate_gtin"].nunique()),
    }
