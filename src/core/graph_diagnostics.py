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


#: The gate columns :func:`candidate_graph_diagnostics` requires of the frame it
#: is handed.
#:
#: This tuple is this module's definition of that contract; the frame spec below
#: derives its required set from it instead of re-spelling the names.
#:
#: The producer of those frames, ``training.rand_matching``, imports this tuple
#: and derives its candidate-gate contract from it, so the required set has one
#: definition.  The import runs in the legal ``training`` -> ``core`` direction;
#: the reverse would be an import cycle, because ``training.rand_matching``
#: imports this module at module level while no ``core`` module imports
#: ``training``.  A producer-side rename or drop still makes ``validate_frame``
#: fail loudly here instead of quietly changing the diagnostic.
CANDIDATE_GATE_COLUMNS: tuple[str, ...] = (
    "SKU_ID",
    "candidate_gtin",
    "score",
    "gtin_status",
    "exact_gtin",
    "rule_ok",
)


_CANDIDATE_GRAPH_FRAME_SPEC = CandidateGraphFrameSpec(
    required_columns=frozenset(CANDIDATE_GATE_COLUMNS)
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

    for root in range(len(nodes)):
        if discovery[root] >= 0:
            continue

        component: list[int] = []
        discovery[root] = low[root] = time_counter
        time_counter += 1
        component.append(root)
        # Each frame stores the node, the edge used to enter it, and the next
        # adjacency position to inspect. Completing a frame performs the
        # recursive function's post-order low-link update before returning to
        # its parent.
        stack: list[tuple[int, int, int]] = [(root, -1, 0)]
        while stack:
            node, parent_edge, next_adjacency = stack[-1]
            if next_adjacency < len(adjacency[node]):
                neighbour, edge_id = adjacency[node][next_adjacency]
                stack[-1] = (node, parent_edge, next_adjacency + 1)
                if edge_id == parent_edge:
                    continue
                if discovery[neighbour] < 0:
                    discovery[neighbour] = low[neighbour] = time_counter
                    time_counter += 1
                    component.append(neighbour)
                    stack.append((neighbour, edge_id, 0))
                else:
                    low[node] = min(low[node], discovery[neighbour])
                continue

            stack.pop()
            if parent_edge >= 0 and stack:
                parent = stack[-1][0]
                low[parent] = min(low[parent], low[node])
                if low[node] > discovery[parent]:
                    bridge_edge_ids.add(parent_edge)
        components.append(component)

    component_ids = {
        node: component_id
        for component_id, component in enumerate(components)
        for node in component
    }
    edge_scores = [[] for _ in components]
    for left, _, score in edges:
        edge_scores[component_ids[left]].append(score)
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
