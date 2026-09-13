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


CANDIDATE_GRAPH_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "diagnostic_edge_count",
    "diagnostic_component_count",
    "diagnostic_component_size_distribution",
    "diagnostic_max_component_size",
    "diagnostic_score_diameter",
    "diagnostic_bridge_edge_count",
    "diagnostic_weakest_bridge_score",
)
PLAUSIBLE_GROUP_COUNT_COLUMN = "plausible_group_count"
CANDIDATE_GRAPH_METRIC_COLUMNS: tuple[str, ...] = (
    *CANDIDATE_GRAPH_DIAGNOSTIC_COLUMNS,
    PLAUSIBLE_GROUP_COUNT_COLUMN,
)


def empty_candidate_graph_diagnostics() -> dict[str, float | int | str]:
    """Return the stable empty payload for candidate-graph diagnostics."""
    return dict(
        zip(
            CANDIDATE_GRAPH_METRIC_COLUMNS,
            (
                0,
                0,
                "{}",
                0,
                0.0,
                0,
                float("nan"),
                0,
            ),
            strict=True,
        )
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
        return empty_candidate_graph_diagnostics()

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
    return dict(
        zip(
            CANDIDATE_GRAPH_METRIC_COLUMNS,
            (
                int(len(edges)),
                int(len(components)),
                size_distribution,
                int(max(map(len, components))),
                float(max(diameters)) if diameters else 0.0,
                int(len(bridge_edge_ids)),
                float(weakest_bridge),
                int(accepted["candidate_gtin"].nunique()),
            ),
            strict=True,
        )
    )
