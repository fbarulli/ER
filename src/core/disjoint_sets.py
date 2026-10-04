"""Union-find (disjoint-set) primitive shared by the fold/mining lanes.

Inline reimplementations of the same find/union loop had scattered across
training/folds (twice in one module), training/robust_validation and
training/negative_supply, each with its own path-compression style and union
orientation. This is the one shared implementation.

CALLER CONTRACT: group by MEMBERSHIP, never by root identity. The root is an
internal bookkeeping label — which element becomes the root of a merged set
depends on union ORDER and carries no meaning. Use :meth:`components` (sets,
canonically ordered) or membership tests; outputs then stay independent of
the order unions happen to arrive in.
"""
from __future__ import annotations


class DisjointSet:
    """Union-find with path compression over hashable string labels."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self._parent.setdefault(value, value)

    def find(self, value: str) -> str:
        """Root of ``value``'s set (adding it as a singleton if new)."""
        self.add(value)
        root = value
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[value] != root:  # path compression
            self._parent[value], value = root, self._parent[value]
        return root

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root

    def members(self) -> set[str]:
        return set(self._parent)

    def components(self) -> list[set[str]]:
        """The partition as sets, canonically ordered by sorted member lists
        so the result is independent of union order (and of which label is
        the root of any merged set)."""
        groups: dict[str, set[str]] = {}
        for value in self._parent:
            groups.setdefault(self.find(value), set()).add(value)
        return sorted(groups.values(), key=lambda members: sorted(members))
