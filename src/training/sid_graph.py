"""Graph + SID joint inference (spike, analysis-only).

Pointwise top-1 assignment decides every SKU alone and leaves corroboration
on the floor (two SKUs of the same product never vouch for each other);
unconstrained clustering chains (A~B~C~D drift merges). This module does the
textbook middle: greedy correlation clustering — merge along strong links,
never across a cannot-link veto.

Nodes are SKU rows [0, n) and canonicals [n, n+m). Edge kinds:
  * LOCK (weight +inf): exact GTIN lock the lane itself honors. Only emitted
    when the caller allows it (both_equal stratum) — never for blind strata.
  * COS (weight = cosine): SKU-canon pairs above threshold; SKU-SKU pairs
    above a higher corroboration threshold.
  * SID (weight = bonus): SKU-canon or SKU-SKU pairs sharing the coarse SID
    code. The corroborating signal global scoring misused: here it only
    *adds* merge evidence, and cannot-link vetoes still outrank everything.
  * VETO (cannot-link): attribute conflicts (volume/pack) or brand mismatch
    with evidence on both sides. Vetoes block merges transitively through
    union-find forbidden sets — this is the anti-chaining mechanism the old
    connected-components lane lacked.

Pure numpy (no ``core`` imports). Deterministic: edges sorted by
(-weight, i, j); ties can never flip a verdict between runs.
"""

from __future__ import annotations

import numpy as np

INF_WEIGHT = float("inf")

__all__ = [
    "INF_WEIGHT",
    "greedy_constrained_clusters",
    "assign_clusters_to_canonicals",
]


def greedy_constrained_clusters(
    n_nodes: int,
    pos_edges: list[tuple[int, int, float]],
    cannot_link: set[tuple[int, int]] | list[tuple[int, int]],
) -> np.ndarray:
    """Greedy correlation clustering over explicit edge lists.

    Processes positive edges strongest-first and unions endpoints unless
    their components are cannot-linked (directly or transitively). Returns
    integer cluster labels aligned with node ids (compacted 0..C-1 in order
    of first appearance, so output is deterministic).
    """
    parent = list(range(n_nodes))
    members: dict[int, set[int]] = {i: {i} for i in range(n_nodes)}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ban: set[tuple[int, int]] = set()
    for a, b in cannot_link:
        if a != b:
            ban.add((min(a, b), max(a, b)))

    def blocked(ra: int, rb: int) -> bool:
        ma, mb = members[ra], members[rb]
        small, big = (ma, mb) if len(ma) <= len(mb) else (mb, ma)
        for x in small:
            for y in big:
                if (min(x, y), max(x, y)) in ban:
                    return True
        return False

    for a, b, w in sorted(pos_edges, key=lambda e: (-e[2], e[0], e[1])):
        if a == b:
            continue
        ra, rb = find(a), find(b)
        if ra == rb or blocked(ra, rb):
            continue
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra] |= members[rb]
        del members[rb]
    labels = np.asarray([find(i) for i in range(n_nodes)])
    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int64)


def assign_clusters_to_canonicals(
    sku_labels: np.ndarray,
    canon_labels: np.ndarray,
    sku_canon_weight: np.ndarray,
    locked: dict[int, int] | None = None,
) -> dict[int, int | None]:
    """Map each SKU node to a canonical index (or None = unmatched).

    A cluster containing a GTIN-locked SKU maps to the locked canonical.
    Otherwise the cluster maps to the canonical maximizing total edge weight
    from member SKUs (ties: smallest canonical index). Clusters with no
    canonical member and no positive SKU-canon weight map to None.
    """
    locked = locked or {}
    n_sku = int(sku_labels.shape[0])
    cluster_members: dict[int, list[int]] = {}
    for i, lab in enumerate(sku_labels.tolist()):
        cluster_members.setdefault(int(lab), []).append(i)
    canon_by_label: dict[int, list[int]] = {}
    for j, lab in enumerate(canon_labels.tolist()):
        canon_by_label.setdefault(int(lab), []).append(j)
    out: dict[int, int | None] = {}
    for lab, members in cluster_members.items():
        locked_js = [locked[i] for i in members if i in locked]
        if locked_js:
            out.update({i: int(locked_js[0]) for i in members})
            continue
        cands = canon_by_label.get(lab, [])
        if not cands:
            out.update({i: None for i in members})
            continue
        totals = {j: float(sku_canon_weight[members, j].sum()) for j in cands}
        best = max(sorted(totals), key=lambda j: totals[j])
        out.update({i: (int(best) if totals[best] > 0 else None) for i in members})
    return out
