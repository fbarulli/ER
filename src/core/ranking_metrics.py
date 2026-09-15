"""Deterministic retrieval metrics **and the evaluation-pool contract they need**.

This module owns BOTH halves of one protocol, because a ranking number is
meaningless without the pool it was measured on:

1. ``ranking_at_k`` and ``ranking_at_k_by_query`` score a candidate list.
2. ``build_evaluation_pool`` builds the per-query candidate list that
   ``ranking_at_k_by_query`` is *allowed* to consume, and ``ranking_coverage``
   reports the pool shape in the SAME record as the metric, so a degenerate
   pool can never again produce a confident-looking 1.0.

ONE NAME, ONE MEANING
---------------------
The same column names used to carry two different concepts:

* ``ranking_at_k`` scores ONE pooled binary candidate list (all scored pairs
  concatenated). Its precision/recall are properties of that single list.
* ``ranking_at_k_by_query`` scores ONE candidate list per source query and
  averages the per-query values.

Those are different quantities of different populations, so they are now
named apart:

* ``pooled_hits_at_1`` / ``pooled_precision_at_{k}`` / ``pooled_recall_at_{k}``
  — the POOLED definition, the primary keys returned by ``ranking_at_k``.
* ``hits_at_1`` / ``precision_at_{k}`` / ``recall_at_{k}``
  — RESERVED for the PER-QUERY retrieval definition, returned by
  ``ranking_at_k_by_query`` (and pinned by ``core.schemas.FoldMetrics`` /
  consumed by ``training/generate_training_report.py``).
* ``query_hits_at_1`` / ``query_precision_at_{k}`` / ``query_recall_at_{k}``
  — explicit aliases of the per-query values, for readers who want the
  concept in the name.

``ranking_at_k`` still also returns the bare keys so that two consumers
outside this module's ownership keep working unchanged
(``training/evaluate_models.py``, which writes the schema-pinned
``model_evaluation_summary.csv``, and ``training/selftest.py``).  Those keys
are listed by ``deprecated_pooled_bare_aliases(ks)`` and are byte-identical
aliases of the ``pooled_`` values — never re-derived — and this docstring is
the record of the migration still owed in those two files.

TIE HANDLING
------------
``ranking_at_k_by_query`` breaks score ties by stable input order.  When a
pool places the positive FIRST (the historical ``np.vstack([pos, neg])``
layout) an informationless constant scorer reads Hits@1 = 1.0 for every
query.  ``build_evaluation_pool`` therefore emits each query's block in a
seeded permutation, so an all-tie scorer scores at chance instead of at the
protocol's positional bias.  The tie-break policy is reported in the record
via ``ranking_coverage(..., tie_break=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np

#: ``ranking_at_k`` keys that mean the POOLED concept.  The ``pooled_`` prefix
#: is the canonical spelling; the bare spelling is retained only for the two
#: un-migrated consumers named in the module docstring.
POOLED_METRIC_PREFIX = "pooled_"


def deprecated_pooled_bare_aliases(ks: Sequence[int]) -> tuple[str, ...]:
    """The bare keys ``ranking_at_k`` still emits for its POOLED values.

    Listed so a reader can see exactly which names are the ambiguity: every
    one of them is also the name of a PER-QUERY value returned by
    ``ranking_at_k_by_query`` in the training fold record.
    """
    return (
        "hits_at_1",
        *(f"{metric}_at_{int(k)}" for k in ks for metric in ("precision", "recall")),
    )

#: A chance-level scorer's Recall@max(ks) on a pool of ``HEADROOM * max(ks)``
#: candidates is exactly ``1 / HEADROOM``.  The pool is built with that many
#: candidates, so the metric has one decade of usable range below 1.0.
POOL_HEADROOM = 10


def minimum_competitors(ks: Sequence[int]) -> int:
    """The FLOOR on competitors per query: pool size must exceed max(ks).

    With ``pool = 1 + N`` candidates and a single relevant candidate, a pool
    of size ``max(ks)`` makes the top-``max(ks)`` list the WHOLE pool, so
    Recall@max(ks) is 1.0 for every scorer that can order anything at all —
    the metric stops measuring retrieval.  ``1 + N > max(ks)`` is therefore
    the hard floor; ``competitors_per_query`` buys headroom above it.
    """
    ks = tuple(int(k) for k in ks)
    if not ks:
        raise ValueError("retrieval K list must not be empty")
    if any(k < 1 for k in ks):
        raise ValueError(f"K must be >= 1, got {min(ks)}")
    return max(ks)


def competitors_per_query(ks: Sequence[int], *, headroom: int = POOL_HEADROOM) -> int:
    """Competitors per query: ``headroom * max(ks) - 1``.

    The floor is ``minimum_competitors(ks) == max(ks)`` (pool strictly larger
    than max(ks)).  ``headroom`` buys range instead of just legality: with
    ``pool = headroom * max(ks)`` an informationless scorer's
    Recall@max(ks) is ``1 / headroom``, so a reported 1.0 can be told apart
    from a chance-level 0.10.  Derived from the configured
    ``evaluation.retrieval_ks``, so retuning K retunes the pool with it.
    """
    if headroom < 1:
        raise ValueError(f"pool headroom must be >= 1, got {headroom}")
    return headroom * minimum_competitors(ks) - 1


def ranking_at_k(
    labels: np.ndarray, scores: np.ndarray, ks: tuple[int, ...]
) -> dict[str, float]:
    """Return precision/recall at each K plus Hits@1 for ONE POOLED list.

    POOLED definition: positives are relevant candidates and negatives are
    non-relevant candidates in one concatenated population; metrics describe
    the top of that single score ranking, not a thresholded classifier
    decision.  Ties use stable input order, making every reported rank
    reproducible.  A short candidate list uses its available prefix for
    Precision@K while Recall@K still divides by all relevant candidates in
    the population.

    Returns the canonical ``pooled_*`` names plus the deprecated bare aliases
    (see the module docstring); the alias values ARE the pooled values.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if len(labels) == 0:
        raise ValueError("ranking metrics require at least one candidate")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("ranking metric labels must be binary")
    if any(k < 1 for k in ks):
        raise ValueError(f"K must be >= 1, got {min(ks)}")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    n_relevant = int(ranked.sum())
    result: dict[str, float] = {
        f"{POOLED_METRIC_PREFIX}hits_at_1": float(ranked[0] == 1)
    }
    for k in ks:
        if k < 1:
            raise ValueError(f"K must be >= 1, got {k}")
        top = ranked[:k]
        hits = int(top.sum())
        result[f"{POOLED_METRIC_PREFIX}precision_at_{k}"] = hits / len(top)
        result[f"{POOLED_METRIC_PREFIX}recall_at_{k}"] = (
            hits / n_relevant if n_relevant else 0.0
        )
    # Historical spellings, kept for training/evaluate_models.py (which feeds
    # the schema-pinned model_evaluation_summary.csv) and
    # training/selftest.py.  Byte-identical aliases, never recomputed.
    result["hits_at_1"] = result[f"{POOLED_METRIC_PREFIX}hits_at_1"]
    for k in ks:
        result[f"precision_at_{k}"] = result[f"{POOLED_METRIC_PREFIX}precision_at_{k}"]
        result[f"recall_at_{k}"] = result[f"{POOLED_METRIC_PREFIX}recall_at_{k}"]
    return result


def ranking_at_k_by_query(
    labels: np.ndarray,
    scores: np.ndarray,
    query_ids: np.ndarray,
    ks: tuple[int, ...],
) -> dict[str, float]:
    """Aggregate PER-QUERY retrieval metrics over independent source queries.

    PER-QUERY definition (the meaning reserved for the bare names): every
    query owns its own candidate list; Recall@K is the fraction of queries
    whose relevant candidate appears in their top K; Precision@K is the mean
    over queries of the relevant share of that query's top K.  A query with
    no relevant candidate is EXCLUDED (and counted by
    ``ranking_coverage``); Recall@K is intentionally not divided by the
    number of positive pair rows globally.

    The returned values also appear under ``query_*`` aliases so a reader can
    name the concept explicitly.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    query_ids = np.asarray(query_ids)
    if labels.ndim != 1 or scores.ndim != 1 or query_ids.ndim != 1:
        raise ValueError("ranking inputs must be aligned one-dimensional arrays")
    if not (len(labels) == len(scores) == len(query_ids)) or len(labels) == 0:
        raise ValueError("ranking inputs must be non-empty and aligned")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("ranking metric labels must be binary")
    # Same argument contract as ranking_at_k: an empty or non-positive K is a
    # caller bug. Rejecting it here matters because the alternative is a
    # confident-looking miss — hits_by_k is only populated per requested K, so
    # an empty ks used to return hits_at_1 == 0.0 for a population whose top
    # candidate IS relevant.
    if not ks:
        raise ValueError("ranking metrics require at least one K")
    if any(k < 1 for k in ks):
        raise ValueError(f"K must be >= 1, got {min(ks)}")

    result: dict[str, float] = {}
    hits_by_k: dict[int, list[int]] = {k: [] for k in ks}
    precision_by_k: dict[int, list[float]] = {k: [] for k in ks}
    top1_hits: list[int] = []
    for query in pd_unique(query_ids):
        mask = query_ids == query
        group = labels[mask][np.argsort(-scores[mask], kind="stable")]
        if not group.any():
            continue
        top1_hits.append(int(group[0] == 1))
        for k in ks:
            top = group[:k]
            hits_by_k[k].append(int(top.sum() > 0))
            precision_by_k[k].append(float(top.mean()) if len(top) else 0.0)
    # Hits@1 is defined on every query holding a relevant candidate, so it no
    # longer depends on 1 happening to be one of the requested K values.
    result["hits_at_1"] = float(np.mean(top1_hits)) if top1_hits else 0.0
    for k in ks:
        hits = hits_by_k[k]
        result[f"precision_at_{k}"] = (
            float(np.mean(precision_by_k[k])) if hits else 0.0
        )
        result[f"recall_at_{k}"] = float(np.mean(hits)) if hits else 0.0
    for key, value in tuple(result.items()):
        result[f"query_{key}"] = value
    return result


def ranking_coverage(
    labels: np.ndarray,
    query_ids: np.ndarray,
    ks: Sequence[int],
    *,
    prefix: str = "",
    tie_break: str = "stable_input_order",
) -> dict[str, float | int | str]:
    """Candidate-pool accounting for one per-query ranking population.

    Derived from the metric's OWN inputs (labels x query_ids), never from a
    builder's claim about the pool it intended to emit, so the record and the
    number cannot drift apart.  A reader must be able to tell from this
    record alone whether the metric was measurable:

    * ``n_queries`` / ``n_queries_scored`` / ``n_queries_excluded`` (+ the
      reason) — a query with no relevant candidate is dropped by
      ``ranking_at_k_by_query`` and shows up here instead of vanishing.
    * ``candidates_min|median|max|mean`` — the per-query pool size.
    * ``n_queries_pool_le_max_k`` / ``share_queries_pool_le_max_k`` — the
      share whose pool is not STRICTLY larger than max(ks); for those
      queries Recall@max(ks) is 1.0 for ANY scorer, i.e. meaningless.
    * ``chance_hits_at_1`` / ``chance_recall_at_{k}`` — what an
      informationless (all-ties) scorer would read on this exact pool
      shape, i.e. the baseline the reported recall must beat.
    * ``trustworthy`` — 1 only when nothing was excluded and no query's pool
      is <= max(ks). Any 0 here invalidates the accompanying recall numbers.
    """
    labels = np.asarray(labels, dtype=int)
    query_ids = np.asarray(query_ids)
    ks = tuple(int(k) for k in ks)
    if labels.ndim != 1 or query_ids.ndim != 1 or len(labels) != len(query_ids):
        raise ValueError("ranking coverage inputs must be aligned one-dimensional arrays")
    if not ks:
        raise ValueError("ranking coverage requires at least one K")
    max_k = max(ks)
    queries = pd_unique(query_ids)
    sizes = np.asarray([int((query_ids == q).sum()) for q in queries], dtype=int)
    relevant = np.asarray(
        [int(labels[query_ids == q].sum()) for q in queries], dtype=int
    )
    scored = relevant > 0
    n_total = int(len(queries))
    n_scored = int(scored.sum())
    n_excluded = n_total - n_scored
    safe = np.maximum(sizes[scored], 1)
    at_risk = int((sizes[scored] <= max_k).sum()) if n_scored else 0
    record: dict[str, float | int | str] = {
        f"{prefix}n_queries": n_total,
        f"{prefix}n_queries_scored": n_scored,
        f"{prefix}n_queries_excluded": n_excluded,
        # The only exclusion rule ranking_at_k_by_query has: no relevant
        # candidate, so Recall@K is undefined for that query.
        f"{prefix}n_queries_excluded_no_relevant": n_excluded,
        f"{prefix}candidates_min": int(sizes.min()) if n_total else 0,
        f"{prefix}candidates_median": float(np.median(sizes)) if n_total else 0.0,
        f"{prefix}candidates_max": int(sizes.max()) if n_total else 0,
        f"{prefix}candidates_mean": float(sizes.mean()) if n_total else 0.0,
        f"{prefix}max_k": max_k,
        f"{prefix}n_queries_pool_le_max_k": at_risk,
        f"{prefix}share_queries_pool_le_max_k": (
            at_risk / n_scored if n_scored else 1.0
        ),
        f"{prefix}chance_hits_at_1": (
            float(np.mean(1.0 / safe)) if n_scored else 0.0
        ),
        f"{prefix}tie_break": tie_break,
    }
    for k in ks:
        record[f"{prefix}chance_recall_at_{k}"] = (
            float(np.mean(np.minimum(k, safe) / safe)) if n_scored else 0.0
        )
    record[f"{prefix}trustworthy"] = int(
        n_scored > 0 and n_excluded == 0 and at_risk == 0
    )
    return record


def pd_unique(values: np.ndarray) -> np.ndarray:
    """Stable unique values without adding a pandas dependency to this module."""
    return np.asarray(list(dict.fromkeys(values.tolist())))


def component_index(pos_pairs: np.ndarray, row_bc: np.ndarray) -> np.ndarray:
    """Connected-component id per payload row, deterministic and hash-free.

    Union-find over the barcode graph induced by ``pos_pairs`` — the SAME
    relation ``training.folds.component_folds`` splits on, so a component is
    the transitive closure of "positive-pair linked".  The id of a row is
    derived from the lexicographically smallest barcode in its component and
    from a sorted id table, so the result never depends on dict/set iteration
    order or on ``PYTHONHASHSEED``.  Rows with an empty barcode get ``-1``
    (``folds.component_folds`` likewise never makes them graph nodes).
    """
    pos_pairs = np.asarray(pos_pairs, dtype=int).reshape(-1, 2)
    row_bc = np.asarray(row_bc)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for bc in row_bc:
        node = str(bc)
        if node and node not in parent:
            parent[node] = node
    for left, right in pos_pairs:
        a, b = str(row_bc[int(left)]), str(row_bc[int(right)])
        if not a or not b or a not in parent or b not in parent:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    roots = {node: find(node) for node in parent}
    smallest: dict[str, str] = {}
    for node, root in roots.items():
        if root not in smallest or node < smallest[root]:
            smallest[root] = node
    ids = {root: index for index, root in enumerate(sorted(smallest.values()))}
    result = np.full(len(row_bc), -1, dtype=int)
    for position, bc in enumerate(row_bc):
        node = str(bc)
        if node in roots:
            result[position] = ids[roots[node]]
    return result


@dataclass(frozen=True)
class EvaluationPool:
    """One per-query competitor pool, ready to be scored.

    ``pairs[k] == (query_row, candidate_row)`` are PAYLOAD row indices in the
    same space as the fold's positive/hard-negative pairs, so the caller can
    encode them through its own fused-embedding cache and take the cosine of
    each pair exactly as it does for the classification populations.
    """

    pairs: np.ndarray
    labels: np.ndarray
    query_keys: np.ndarray
    query_rows: np.ndarray
    coverage: dict[str, float | int]

    def __len__(self) -> int:
        return int(len(self.labels))


def build_evaluation_pool(
    queries: np.ndarray,
    query_keys: Sequence[Hashable],
    *,
    competitor_rows: np.ndarray,
    row_component: np.ndarray,
    row_bc: np.ndarray,
    n_competitors: int,
    seed: int,
    ks: Sequence[int],
    priority_pairs: np.ndarray | None = None,
    excluded_barcode_pairs: Iterable[tuple[str, str]] = (),
    max_probes: int | None = None,
) -> EvaluationPool:
    """Build the fold-safe per-query competitor pool for retrieval metrics.

    Every evaluated query gets EXACTLY ONE positive — its own canonical — plus
    up to ``n_competitors`` competing canonicals drawn from the same fold.
    Competitors are fold-safe by construction:

    * a competitor whose barcode shares the query's COMPONENT is excluded
      (the component is the unit the holdout split deals out, so a
      competitor from another component can never be linked to the query by
      any chain of positive pairs — the query's positive stays the only
      relevant candidate in its own pool);
    * a competitor that is a KNOWN TRUE MATCH of the query
      (``excluded_barcode_pairs``: labeled positives, gate-proceed matches,
      identical canonical identity) is excluded, because scoring it as a
      non-relevant candidate would silently count a correct ranking as a
      miss.

    Determinism: the competitor universe is SORTED, queries are processed in
    sorted-key order, the sweep follows a seeded permutation of the sorted
    universe behind a cursor that persists across queries (so exposure is
    spread evenly instead of concentrating on a few queries), and the
    within-query block order is a second seeded permutation. Nothing depends
    on dict/set iteration order, so repeated calls and different
    ``PYTHONHASHSEED`` values produce identical pools.

    Every query either receives a complete pool, a short pool that is COUNTED
    (``pool_queries_capped_below_target``), or a counted exclusion
    (``pool_queries_excluded`` + ``pool_excluded_no_eligible_competitor`` /
    ``pool_excluded_missing_positive``). Nothing is dropped silently.

    ``priority_pairs`` ((N,2) ``(query_row, competitor_row)``, e.g. the fold's
    mined hard negatives) are seated FIRST when they are eligible, so the
    hardest distractors the pipeline knows about stay inside the retrieval
    comparison instead of being replaced by easy random ones.
    ``max_probes`` caps the wrap-around scan (defaults to one full sweep of
    the universe, which provably finds every eligible candidate).
    """
    queries = np.asarray(queries, dtype=int).reshape(-1, 2)
    keys = np.asarray([str(key) for key in query_keys])
    if len(keys) != len(queries):
        raise ValueError(
            "query_keys must align with queries: "
            f"{len(keys)} keys for {len(queries)} queries"
        )
    if len(np.unique(keys)) != len(keys):
        raise ValueError(
            "query_keys must be unique — the per-query metric groups by key, "
            "so two queries sharing a key would silently share one pool"
        )
    floor = minimum_competitors(ks)
    if n_competitors < floor:
        raise ValueError(
            f"n_competitors={n_competitors} is below the pool floor "
            f"n_competitors >= max(ks)={floor}: a pool of {1 + n_competitors} "
            f"candidates is not strictly larger than max(ks), so "
            f"Recall@{max(ks)} would be 1.0 for every scorer"
        )
    row_bc = np.asarray(row_bc)
    row_component = np.asarray(row_component, dtype=int)
    if len(row_component) < len(row_bc):
        raise ValueError(
            f"row_component covers {len(row_component)} rows but row_bc has "
            f"{len(row_bc)} — component ids must cover the payload"
        )

    universe = np.unique(np.asarray(competitor_rows, dtype=int))
    if len(universe) == 0:
        raise ValueError("competitor_rows is empty: no candidate universe")

    forbidden: dict[str, set[str]] = {}
    for left, right in excluded_barcode_pairs:
        a, b = str(left).strip(), str(right).strip()
        if not a or not b or a == b:
            continue
        forbidden.setdefault(a, set()).add(b)
        forbidden.setdefault(b, set()).add(a)

    priority: dict[int, list[int]] = {}
    for left, right in (
        np.asarray(priority_pairs, dtype=int).reshape(-1, 2)
        if priority_pairs is not None and len(priority_pairs)
        else np.empty((0, 2), dtype=int)
    ):
        priority.setdefault(int(left), []).append(int(right))
    for row in priority:
        priority[row] = sorted(set(priority[row]))

    barcode_of_row = np.asarray([str(bc).strip() for bc in row_bc], dtype=object)

    def eligible(query_row: int, candidate_row: int) -> bool:
        if candidate_row < 0 or candidate_row >= len(barcode_of_row):
            return False
        if row_component[candidate_row] == row_component[query_row]:
            return False
        query_bc = barcode_of_row[query_row]
        candidate_bc = barcode_of_row[candidate_row]
        if not query_bc or not candidate_bc:
            return False
        return candidate_bc not in forbidden.get(query_bc, ())

    rng_sweep = np.random.default_rng(int(seed))
    permutation = rng_sweep.permutation(len(universe))
    rng_block = np.random.default_rng(int(seed) + 1)
    probes_cap = int(max_probes) if max_probes is not None else len(universe)

    key_of_row = {
        int(queries[index, 0]): keys[index] for index in range(len(queries))
    }
    pair_queries: list[int] = []
    pair_candidates: list[int] = []
    pair_labels: list[int] = []
    evaluated_rows: list[int] = []
    pool_sizes: list[int] = []
    exposure: dict[int, int] = {}
    n_missing_positive = 0
    n_no_eligible = 0
    n_capped = 0
    n_priority_used = 0
    n_priority_rejected = 0
    cursor = 0

    for index in sorted(range(len(queries)), key=lambda i: (keys[i], i)):
        query_row = int(queries[index, 0])
        positive_row = int(queries[index, 1])
        if positive_row < 0 or positive_row >= len(barcode_of_row):
            n_missing_positive += 1
            continue
        chosen: list[int] = []
        seen: set[int] = set()
        for candidate in priority.get(query_row, ()):
            if len(chosen) >= n_competitors:
                break
            if candidate in seen:
                continue
            if eligible(query_row, candidate):
                chosen.append(candidate)
                seen.add(candidate)
                n_priority_used += 1
            else:
                n_priority_rejected += 1
        probes = 0
        while len(chosen) < n_competitors and probes < probes_cap:
            candidate = int(universe[int(permutation[cursor % len(universe)])])
            cursor += 1
            probes += 1
            if candidate in seen or not eligible(query_row, candidate):
                continue
            chosen.append(candidate)
            seen.add(candidate)
        if not chosen:
            n_no_eligible += 1
            continue
        if len(chosen) < n_competitors:
            n_capped += 1
        block = np.asarray([positive_row, *chosen], dtype=int)
        labels = np.zeros(len(block), dtype=int)
        labels[0] = 1
        order = rng_block.permutation(len(block))
        block, labels = block[order], labels[order]
        pair_queries.extend([query_row] * len(block))
        pair_candidates.extend(int(value) for value in block)
        pair_labels.extend(int(value) for value in labels)
        evaluated_rows.append(query_row)
        pool_sizes.append(int(len(block)))
        for candidate in chosen:
            exposure[candidate] = exposure.get(candidate, 0) + 1

    sizes = np.asarray(pool_sizes, dtype=int) if pool_sizes else np.zeros(0, dtype=int)
    exposure_values = np.asarray(list(exposure.values()), dtype=int) if exposure else np.zeros(0, dtype=int)
    coverage: dict[str, float | int] = {
        "pool_target_competitors": int(n_competitors),
        "pool_floor_competitors": int(floor),
        "pool_headroom": int(POOL_HEADROOM),
        "pool_queries_requested": int(len(queries)),
        "pool_queries_evaluated": int(len(evaluated_rows)),
        "pool_queries_excluded": int(len(queries) - len(evaluated_rows)),
        "pool_excluded_no_eligible_competitor": int(n_no_eligible),
        "pool_excluded_missing_positive": int(n_missing_positive),
        "pool_queries_capped_below_target": int(n_capped),
        "pool_size_min": int(sizes.min()) if len(sizes) else 0,
        "pool_size_median": float(np.median(sizes)) if len(sizes) else 0.0,
        "pool_size_max": int(sizes.max()) if len(sizes) else 0,
        "pool_unique_candidate_rows": int(len(exposure)),
        "pool_draw_incidences": int(len(pair_candidates) - len(evaluated_rows)),
        "pool_candidate_exposure_min": (
            int(exposure_values.min()) if len(exposure_values) else 0
        ),
        "pool_candidate_exposure_max": (
            int(exposure_values.max()) if len(exposure_values) else 0
        ),
        "pool_candidate_exposure_mean": (
            float(exposure_values.mean()) if len(exposure_values) else 0.0
        ),
        "pool_priority_used": int(n_priority_used),
        "pool_priority_rejected": int(n_priority_rejected),
        "pool_ks_max": int(minimum_competitors(ks)),
    }
    return EvaluationPool(
        pairs=np.asarray(
            list(zip(pair_queries, pair_candidates, strict=True)), dtype=int
        ).reshape(-1, 2),
        labels=np.asarray(pair_labels, dtype=int),
        query_keys=np.asarray([key_of_row[row] for row in pair_queries], dtype=object),
        query_rows=np.asarray(evaluated_rows, dtype=int),
        coverage=coverage,
    )


def added_encode_rows(pool_rows: np.ndarray, already_encoded: np.ndarray) -> int:
    """How many payload rows a pool adds to an existing encode set."""
    return int(
        len(
            np.setdiff1d(
                np.unique(np.asarray(pool_rows, dtype=int)),
                np.unique(np.asarray(already_encoded, dtype=int)),
            )
        )
    )
