"""ER-346 — holdout retrieval-pool construction and pool-coverage accounting.

The metric in ``core.ranking_metrics`` was arithmetically correct but was fed a
pool that was ONE candidate wide for 90.5% of holdout queries, which made a
perfect oracle and an informationless constant scorer numerically identical.
These tests pin the pool contract that makes the metric measurable:

* every query's pool contains exactly its own positive;
* the pool is strictly larger than ``max(ks)`` (the floor), so Recall@max(ks)
  is not trivially 1.0;
* competitors come from OTHER COMPONENTS of the same fold — the fold-safety
  property, because the component is the unit the split deals out;
* competitors are never a known true match of the query;
* the draw is deterministic across repeated calls AND across processes with
  different ``PYTHONHASHSEED``;
* every query either gets a full pool, a counted short pool, or a counted
  exclusion — nothing is dropped silently;
* the metric separates an oracle from a constant on the synthetic pool, and
  ``ranking_coverage`` reports whether it was allowed to.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import numpy as np

from core.ranking_metrics import (
    POOLED_METRIC_PREFIX,
    added_encode_rows,
    build_evaluation_pool,
    competitors_per_query,
    component_index,
    deprecated_pooled_bare_aliases,
    minimum_competitors,
    ranking_at_k,
    ranking_at_k_by_query,
    ranking_coverage,
)

KS = (1, 5, 10)
SRC = Path(__file__).resolve().parents[1] / "src"


class Fixture:
    """A tiny payload with ONE genuinely multi-barcode component.

    Layout mirrors the real one: source rows first, then one canonical per
    GTIN.  ``g000``/``g001`` are linked by positive-pair edges in BOTH
    directions so they share a component; every other barcode is a singleton.
    The candidate universe (190 canonicals) is wider than the query set, so a
    full ``competitors_per_query((1, 5, 10)) == 99`` pool fits.
    """

    def __init__(self, n_source: int = 70, n_extra_canonical: int = 120) -> None:
        self.n_source = n_source
        self.n_extra = n_extra_canonical
        self.n_rows = n_source + n_source + n_extra_canonical
        # source rows
        self.row_bc = [f"g{i:03d}" for i in range(n_source)]
        # canonical rows: one per source GTIN ...
        self.row_bc += [f"g{i:03d}" for i in range(n_source)]
        # ... plus extra canonical-only GTINs (more candidates than queries)
        self.row_bc += [f"x{i:03d}" for i in range(n_extra_canonical)]
        self.row_bc = np.asarray(self.row_bc, dtype=object)
        canon = np.arange(n_source, 2 * n_source, dtype=int)
        pairs = [[i, int(canon[i])] for i in range(n_source)]
        # link g000 <-> g001 into ONE two-barcode component
        pairs.append([0, int(canon[1])])
        pairs.append([1, int(canon[0])])
        self.pos = np.asarray(pairs, dtype=int)
        self.component = component_index(self.pos, self.row_bc)
        self.canonical_rows = np.arange(n_source, self.n_rows, dtype=int)

    def query_table(self, rows: list[int]) -> np.ndarray:
        return np.asarray([[row, self.n_source + row] for row in rows], dtype=int)


def pool_for(fixture: Fixture, rows: list[int], n: int = 12, **kwargs):
    queries = fixture.query_table(rows)
    keys = [f"p{row}" for row in rows]
    return build_evaluation_pool(
        queries,
        keys,
        competitor_rows=fixture.canonical_rows,
        row_component=fixture.component,
        row_bc=fixture.row_bc,
        n_competitors=n,
        seed=17,
        ks=KS,
        **kwargs,
    )


class PoolFloorTests(unittest.TestCase):
    def test_floor_is_pool_strictly_larger_than_max_k(self) -> None:
        # pool size is 1 + N, so "pool > max(ks)" is exactly N >= max(ks).
        self.assertEqual(minimum_competitors((1, 5, 10)), 10)
        self.assertEqual(minimum_competitors((1, 5, 20)), 20)
        self.assertEqual(minimum_competitors((1,)), 1)
        with self.assertRaises(ValueError):
            minimum_competitors(())
        with self.assertRaises(ValueError):
            minimum_competitors((0, 5))

    def test_competitors_per_query_is_derived_from_ks_with_headroom(self) -> None:
        n = competitors_per_query(KS)
        self.assertEqual(n, 99)  # ks=[1,5,10] -> pool of 100
        self.assertGreater(n, minimum_competitors(KS))
        # a chance-level scorer's Recall@max(ks) is 1/headroom, so the metric
        # keeps a decade of range below 1.0 instead of sitting on the ceiling
        self.assertAlmostEqual(max(KS) / (1 + n), 0.10, places=12)
        self.assertEqual(competitors_per_query((1, 5, 20)), 199)

    def test_build_rejects_a_pool_at_or_below_max_k(self) -> None:
        fixture = Fixture()
        with self.assertRaises(ValueError) as ctx:
            pool_for(fixture, [2, 3], n=minimum_competitors(KS) - 1)
        self.assertIn("below the pool floor", str(ctx.exception))

    def test_every_query_gets_a_pool_strictly_larger_than_max_k(self) -> None:
        pool = pool_for(Fixture(), [2, 3, 4, 5], n=12)
        self.assertGreater(pool.coverage["pool_size_min"], max(KS))
        self.assertEqual(pool.coverage["pool_size_min"], 1 + 12)
        self.assertEqual(pool.coverage["pool_queries_evaluated"], 4)


class PoolContractTests(unittest.TestCase):
    def test_every_query_gets_exactly_its_own_positive(self) -> None:
        fixture = Fixture()
        rows = [2, 3, 4, 5, 6]
        pool = pool_for(fixture, rows)
        positive_rows = pool.pairs[pool.labels == 1, 1]
        self.assertEqual(len(positive_rows), len(rows))
        # the single relevant candidate of row r is its own canonical
        for row in rows:
            self.assertIn(fixture.n_source + row, positive_rows.tolist())
        self.assertEqual(
            sorted(pool.query_rows.tolist()),
            sorted(fixture.query_table(rows)[:, 0].tolist()),
        )
        # every query's block carries exactly one relevant candidate
        for key in np.unique(pool.query_keys):
            self.assertEqual(int(pool.labels[pool.query_keys == key].sum()), 1)

    def test_competitors_never_come_from_the_query_own_component(self) -> None:
        """FOLD SAFETY: the component is the unit the split deals out."""
        fixture = Fixture()
        # g000 and g001 share a two-barcode component; query both of them plus
        # singletons, and let every canonical be a candidate.
        rows = list(range(0, 12))
        pool = pool_for(fixture, rows, n=20)
        competitor = pool.labels == 0
        self.assertTrue(competitor.any())
        shared = (
            fixture.component[pool.pairs[:, 0]]
            == fixture.component[pool.pairs[:, 1]]
        )
        violations = int(np.sum(shared & competitor))
        self.assertEqual(
            violations,
            0,
            "a competitor shares the query's component — the pool is not "
            "fold-safe and a positive relationship could cross it",
        )
        # and the component really is wider than one barcode, or the assertion
        # above would be vacuous
        self.assertEqual(int(np.unique(fixture.component[:2]).size), 1)
        self.assertNotEqual(int(fixture.component[0]), int(fixture.component[2]))
        # BOTH members of the shared component are queried, so a competitor
        # from that component would have had to be drawn for one of them
        keys = [str(k) for k in np.unique(pool.query_keys)]
        self.assertIn("p0", keys)
        self.assertIn("p1", keys)
        # the ONLY same-component row in any block is the query's own positive
        for (a, b), label in zip(pool.pairs, pool.labels, strict=True):
            if label == 1:
                self.assertEqual(int(b), fixture.n_source + int(a))

    def test_competitors_never_are_known_true_matches(self) -> None:
        fixture = Fixture()
        rows = list(range(0, 8))
        excluded = frozenset(
            {("g002", "x000"), ("x000", "g002")}
        )
        pool = pool_for(fixture, rows, n=15, excluded_barcode_pairs=excluded)
        barcodes = [str(b) for b in fixture.row_bc]
        pairs = {
            (barcodes[int(a)], barcodes[int(b)])
            for (a, b), label in zip(pool.pairs, pool.labels, strict=True)
            if label == 0
        }
        self.assertNotIn(("g002", "x000"), pairs)
        # a query whose whole competing universe is excluded is COUNTED
        self.assertEqual(pool.coverage["pool_queries_excluded"], 0)
        self.assertGreater(len(pairs), 0)

    def test_query_with_no_eligible_competitor_is_counted_not_dropped(self) -> None:
        fixture = Fixture()
        # The candidate universe is the canonical pair of the ONE shared
        # component {g000, g001}.  Query p0 belongs to that component, so BOTH
        # candidates are own-component and p0 has no eligible competitor —
        # it must be COUNTED, never silently dropped.  Query p2 (a different
        # component) may use both, and is CAPPED (2 < n_competitors).
        universe = np.asarray([fixture.n_source, fixture.n_source + 1], dtype=int)
        queries = np.asarray(
            [[0, fixture.n_source], [2, fixture.n_source + 2]], dtype=int
        )
        pool = build_evaluation_pool(
            queries,
            ["p0", "p2"],
            competitor_rows=universe,
            row_component=fixture.component,
            row_bc=fixture.row_bc,
            n_competitors=10,
            seed=1,
            ks=KS,
        )
        coverage = pool.coverage
        self.assertEqual(coverage["pool_queries_requested"], 2)
        self.assertEqual(coverage["pool_queries_evaluated"], 1)
        self.assertEqual(coverage["pool_queries_excluded"], 1)
        self.assertEqual(coverage["pool_excluded_no_eligible_competitor"], 1)
        # evaluated + excluded == requested, and the cap is counted too
        self.assertEqual(
            coverage["pool_queries_evaluated"] + coverage["pool_queries_excluded"],
            coverage["pool_queries_requested"],
        )
        self.assertEqual(coverage["pool_queries_capped_below_target"], 1)
        self.assertEqual(coverage["pool_size_min"], 3)
        self.assertEqual(coverage["pool_size_max"], 3)
        # the surviving query is the one outside the shared component
        self.assertEqual([str(k) for k in pool.query_keys[pool.labels == 1]], ["p2"])

    def test_missing_positive_is_counted(self) -> None:
        fixture = Fixture()
        queries = np.asarray([[2, fixture.n_source + 2], [3, -1]], dtype=int)
        pool = build_evaluation_pool(
            queries,
            ["p2", "p3"],
            competitor_rows=fixture.canonical_rows,
            row_component=fixture.component,
            row_bc=fixture.row_bc,
            n_competitors=10,
            seed=1,
            ks=KS,
        )
        self.assertEqual(pool.coverage["pool_excluded_missing_positive"], 1)
        self.assertEqual(pool.coverage["pool_queries_evaluated"], 1)

    def test_duplicate_query_keys_are_rejected(self) -> None:
        fixture = Fixture()
        with self.assertRaises(ValueError) as ctx:
            build_evaluation_pool(
                fixture.query_table([2, 3]),
                ["same", "same"],
                competitor_rows=fixture.canonical_rows,
                row_component=fixture.component,
                row_bc=fixture.row_bc,
                n_competitors=10,
                seed=1,
                ks=KS,
            )
        self.assertIn("unique", str(ctx.exception))

    def test_priority_competitors_are_seated_first(self) -> None:
        fixture = Fixture()
        rows = [2, 3, 4]
        preferred = fixture.n_source + 20
        pool = pool_for(fixture, rows, n=15, priority_pairs=np.asarray([[2, preferred]]))
        block = pool.pairs[pool.query_keys == "p2"]
        self.assertIn(preferred, block[:, 1].tolist())
        self.assertEqual(pool.coverage["pool_priority_used"], 1)

    def test_priority_competitor_from_own_component_is_rejected(self) -> None:
        fixture = Fixture()
        # row 0's own canonical is in its own component -> never a competitor
        pool = pool_for(fixture, [0], n=10, priority_pairs=np.asarray([[0, fixture.n_source + 1]]))
        block = pool.pairs[pool.query_keys == "p0"]
        self.assertNotIn(fixture.n_source + 1, block[:, 1].tolist())
        self.assertEqual(pool.coverage["pool_priority_rejected"], 1)

    def test_draw_is_balanced_across_queries(self) -> None:
        fixture = Fixture()
        rows = list(range(0, 12))
        pool = pool_for(fixture, rows, n=12)
        # every query has the SAME pool size ...
        self.assertEqual(pool.coverage["pool_size_min"], pool.coverage["pool_size_max"])
        # ... and candidate exposure is spread, not concentrated
        exposure = pool.coverage
        self.assertLessEqual(
            exposure["pool_candidate_exposure_max"],
            exposure["pool_candidate_exposure_min"] + 2,
        )


class PoolDeterminismTests(unittest.TestCase):
    def test_repeated_calls_are_identical(self) -> None:
        fixture = Fixture()
        first = pool_for(fixture, list(range(0, 16)), n=12)
        second = pool_for(fixture, list(range(0, 16)), n=12)
        np.testing.assert_array_equal(first.pairs, second.pairs)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.query_keys, second.query_keys)
        self.assertEqual(first.coverage, second.coverage)

    def test_shuffled_input_order_does_not_change_the_pool(self) -> None:
        fixture = Fixture()
        rows = list(range(0, 16))
        pool = pool_for(fixture, rows, n=12)
        shuffled = pool_for(fixture, rows[::-1], n=12)
        # keys are matched, not positions: the block of each query must match
        first = {str(k): sorted(pool.pairs[pool.query_keys == k, 1].tolist())
                 for k in np.unique(pool.query_keys)}
        second = {str(k): sorted(shuffled.pairs[shuffled.query_keys == k, 1].tolist())
                  for k in np.unique(shuffled.query_keys)}
        self.assertEqual(first, second)

    def test_draw_is_identical_across_pythonhashseed_values(self) -> None:
        """Sets/dicts iterate in hash order — the draw must not depend on it."""
        script = textwrap.dedent(
            """
            import hashlib, sys
            import numpy as np
            from core.ranking_metrics import build_evaluation_pool

            n = 30
            row_bc = np.asarray(
                [f"g{i:03d}" for i in range(n)] + [f"g{i:03d}" for i in range(n)]
                + [f"x{i:03d}" for i in range(20)], dtype=object,
            )
            pos = np.asarray([[i, n + i] for i in range(n)], dtype=int)
            from core.ranking_metrics import component_index
            component = component_index(pos, row_bc)
            queries = np.asarray([[i, n + i] for i in range(n)], dtype=int)
            pool = build_evaluation_pool(
                queries, [f"p{i}" for i in range(n)],
                competitor_rows=np.arange(n, 2 * n + 20, dtype=int),
                row_component=component, row_bc=row_bc,
                n_competitors=12, seed=17, ks=(1, 5, 10),
                excluded_barcode_pairs={("g002", "x000"), ("x000", "g002")},
            )
            digest = hashlib.sha256()
            digest.update(pool.pairs.tobytes())
            digest.update(pool.labels.tobytes())
            digest.update("".join(pool.query_keys.tolist()).encode())
            print(digest.hexdigest())
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SRC), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        digests = []
        for seed in ("0", "1", "12345"):
            env["PYTHONHASHSEED"] = seed
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, check=True, env=env,
            )
            digests.append(result.stdout.strip())
        self.assertEqual(len(set(digests)), 1, f"hash-seed dependent draw: {digests}")

    def test_component_index_is_order_and_hash_independent(self) -> None:
        fixture = Fixture()
        first = component_index(fixture.pos, fixture.row_bc)
        shuffled = component_index(fixture.pos[::-1], fixture.row_bc)
        # labels are ids of the smallest barcode in each component: stable
        np.testing.assert_array_equal(first, shuffled)
        np.testing.assert_array_equal(
            first, component_index(fixture.pos, list(fixture.row_bc))
        )
        # g000/g001 share a component, everything else is a singleton
        self.assertEqual(first[0], first[1])
        self.assertNotEqual(first[0], first[2])
        self.assertEqual(int(np.unique(first[:2]).size), 1)

    def test_component_index_ignores_blank_barcodes(self) -> None:
        row_bc = np.asarray(["g1", "", "g1"], dtype=object)
        pos = np.asarray([[0, 2]], dtype=int)
        ids = component_index(pos, row_bc)
        self.assertEqual(ids[0], ids[2])
        self.assertEqual(ids[1], -1)


class CoverageTests(unittest.TestCase):
    def test_coverage_marks_a_degenerate_pool_untrustworthy(self) -> None:
        """The exact shape that broke the old protocol: ONE candidate."""
        labels = np.ones(5, dtype=int)
        queries = np.asarray([f"q{i}" for i in range(5)])
        coverage = ranking_coverage(labels, queries, KS, prefix="old_protocol_")
        self.assertEqual(coverage["old_protocol_n_queries"], 5)
        self.assertEqual(coverage["old_protocol_candidates_max"], 1)
        self.assertEqual(coverage["old_protocol_share_queries_pool_le_max_k"], 1.0)
        self.assertEqual(coverage["old_protocol_n_queries_pool_le_max_k"], 5)
        self.assertEqual(coverage["old_protocol_trustworthy"], 0)
        # a chance-level scorer is at the ceiling here — that is the tell
        self.assertEqual(coverage["old_protocol_chance_recall_at_10"], 1.0)
        self.assertEqual(coverage["old_protocol_chance_hits_at_1"], 1.0)

    def test_coverage_marks_a_real_pool_trustworthy(self) -> None:
        labels = np.asarray([1] + [0] * 99, dtype=int)
        queries = np.asarray(["q0"] * 100)
        coverage = ranking_coverage(labels, queries, KS, prefix="retrieval_")
        self.assertEqual(coverage["retrieval_candidates_min"], 100)
        self.assertEqual(coverage["retrieval_share_queries_pool_le_max_k"], 0.0)
        self.assertEqual(coverage["retrieval_trustworthy"], 1)
        self.assertAlmostEqual(coverage["retrieval_chance_recall_at_10"], 0.10)
        self.assertAlmostEqual(coverage["retrieval_chance_hits_at_1"], 0.01)

    def test_coverage_counts_excluded_queries_and_their_reason(self) -> None:
        labels = np.asarray([1, 0, 1], dtype=int)
        queries = np.asarray(["a", "a", "b"])
        coverage = ranking_coverage(labels, queries, KS)
        self.assertEqual(coverage["n_queries"], 2)
        self.assertEqual(coverage["n_queries_scored"], 2)
        self.assertEqual(coverage["n_queries_excluded"], 0)
        # a query with NO relevant candidate is excluded, and the reason
        # column accounts for exactly that many queries
        labels = np.asarray([1, 0, 0, 0], dtype=int)
        queries = np.asarray(["a", "a", "b", "b"])
        coverage = ranking_coverage(labels, queries, KS)
        self.assertEqual(coverage["n_queries"], 2)
        self.assertEqual(coverage["n_queries_scored"], 1)
        self.assertEqual(coverage["n_queries_excluded"], 1)
        self.assertEqual(
            coverage["n_queries_excluded"], coverage["n_queries_excluded_no_relevant"]
        )
        self.assertEqual(coverage["trustworthy"], 0)

    def test_coverage_reports_the_candidate_distribution(self) -> None:
        labels = np.asarray([1, 0, 0, 1, 0], dtype=int)
        queries = np.asarray(["a", "a", "a", "b", "b"])
        coverage = ranking_coverage(labels, queries, KS)
        self.assertEqual(coverage["candidates_min"], 2)
        self.assertEqual(coverage["candidates_max"], 3)
        self.assertEqual(coverage["candidates_median"], 2.5)
        self.assertAlmostEqual(coverage["candidates_mean"], 2.5)
        self.assertEqual(coverage["tie_break"], "stable_input_order")

    def test_pool_coverage_is_reported_in_the_metric_record(self) -> None:
        fixture = Fixture()
        pool = pool_for(fixture, list(range(0, 10)), n=12)
        record = ranking_at_k_by_query(
            pool.labels, pool.labels.astype(float), pool.query_keys, KS
        )
        record.update(
            ranking_coverage(pool.labels, pool.query_keys, KS, prefix="retrieval_")
        )
        record.update(
            {f"retrieval_{k}": v for k, v in pool.coverage.items()}
        )
        for required in (
            "retrieval_n_queries",
            "retrieval_candidates_min",
            "retrieval_candidates_median",
            "retrieval_candidates_max",
            "retrieval_share_queries_pool_le_max_k",
            "retrieval_n_queries_excluded",
            "retrieval_n_queries_excluded_no_relevant",
            "retrieval_chance_recall_at_10",
            "retrieval_trustworthy",
            "retrieval_pool_queries_evaluated",
            "retrieval_pool_size_min",
        ):
            self.assertIn(required, record)
        self.assertEqual(record["retrieval_trustworthy"], 1)
        self.assertEqual(record["retrieval_pool_size_min"], 13)


class MetricDiscriminationTests(unittest.TestCase):
    def test_metric_distinguishes_constant_from_oracle_on_the_pool(self) -> None:
        fixture = Fixture()
        n = competitors_per_query(KS)
        pool = pool_for(fixture, list(range(0, fixture.n_source)), n=n)
        oracle = ranking_at_k_by_query(
            pool.labels, pool.labels.astype(float), pool.query_keys, KS
        )
        constant = ranking_at_k_by_query(
            pool.labels, np.zeros(len(pool.labels)), pool.query_keys, KS
        )
        self.assertNotEqual(oracle, constant)
        self.assertEqual(oracle["recall_at_10"], 1.0)
        self.assertEqual(oracle["hits_at_1"], 1.0)
        # an informationless scorer reads about chance, NOT 1.0: the pool
        # order is a seeded permutation, so the positive is not first by rule
        coverage = ranking_coverage(pool.labels, pool.query_keys, KS)
        self.assertAlmostEqual(
            constant["recall_at_10"], coverage["chance_recall_at_10"], delta=0.15
        )
        self.assertAlmostEqual(
            constant["hits_at_1"], coverage["chance_hits_at_1"], delta=0.1
        )
        self.assertGreater(oracle["recall_at_10"] - constant["recall_at_10"], 0.5)
        self.assertGreater(oracle["recall_at_5"] - constant["recall_at_5"], 0.5)
        self.assertGreater(oracle["hits_at_1"] - constant["hits_at_1"], 0.5)
        self.assertEqual(coverage["trustworthy"], 1)

    def test_constant_scorer_is_at_the_ceiling_on_a_one_candidate_pool(self) -> None:
        """The reproduced defect, in miniature: pool of one == pool of one."""
        labels = np.ones(5, dtype=int)
        queries = np.asarray([f"q{i}" for i in range(5)])
        oracle = ranking_at_k_by_query(labels, labels.astype(float), queries, KS)
        constant = ranking_at_k_by_query(labels, np.zeros(5), queries, KS)
        self.assertEqual(oracle, constant)
        self.assertEqual(constant["recall_at_10"], 1.0)
        self.assertEqual(constant["hits_at_1"], 1.0)

    def test_real_model_shaped_scores_beat_chance(self) -> None:
        fixture = Fixture()
        pool = pool_for(fixture, list(range(0, 20)), n=12)
        rng = np.random.default_rng(3)
        # a "model" that ranks the positive well but not perfectly
        scores = rng.normal(0.0, 1.0, len(pool.labels))
        scores = np.where(pool.labels == 1, scores + 1.2, scores)
        metrics = ranking_at_k_by_query(pool.labels, scores, pool.query_keys, KS)
        coverage = ranking_coverage(pool.labels, pool.query_keys, KS)
        self.assertGreater(metrics["recall_at_10"], coverage["chance_recall_at_10"])


class PooledVersusPerQueryTests(unittest.TestCase):
    def test_ranking_at_k_exposes_pooled_names_and_keeps_aliases(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=int)
        scores = np.asarray([0.9, 0.8, 0.7, 0.6])
        metrics = ranking_at_k(labels, scores, (1, 5, 10))
        self.assertEqual(metrics[f"{POOLED_METRIC_PREFIX}hits_at_1"], 0.0)
        self.assertEqual(metrics[f"{POOLED_METRIC_PREFIX}precision_at_5"], 0.5)
        self.assertEqual(metrics[f"{POOLED_METRIC_PREFIX}recall_at_5"], 1.0)
        for bare in deprecated_pooled_bare_aliases((1, 5, 10)):
            self.assertIn(bare, metrics)
            # aliases are EXACT, never re-derived
            self.assertEqual(metrics[bare], metrics[f"{POOLED_METRIC_PREFIX}{bare}"])

    def test_bare_names_are_reserved_for_the_per_query_meaning(self) -> None:
        """One concept per name: the pooled and per-query values DIFFER."""
        fixture = Fixture()
        pool = pool_for(fixture, list(range(0, 12)), n=12)
        scores = pool.labels.astype(float)
        per_query = ranking_at_k_by_query(
            pool.labels, scores, pool.query_keys, KS
        )
        pooled = ranking_at_k(pool.labels, scores, KS)
        # per-query recall asks "is the query's positive in its own top K?"
        for k in KS:
            self.assertEqual(per_query[f"recall_at_{k}"], 1.0)
        # pooled recall divides by EVERY relevant row in the ONE concatenated
        # list — a different quantity on a different population, which is
        # exactly why the two spellings had to be split apart
        n_relevant = int(pool.labels.sum())
        self.assertEqual(pooled[f"{POOLED_METRIC_PREFIX}recall_at_10"], 10 / n_relevant)
        self.assertNotEqual(
            pooled[f"{POOLED_METRIC_PREFIX}recall_at_10"],
            per_query["recall_at_10"],
        )
        # precision@1 is the sharpest illustration of the split: the POOLED
        # value reads ONE position of the concatenated list; the PER-QUERY
        # value averages the first position of every query's own block.
        constant = np.zeros(len(pool.labels))
        per_query_c = ranking_at_k_by_query(
            pool.labels, constant, pool.query_keys, KS
        )
        pooled_c = ranking_at_k(pool.labels, constant, KS)
        self.assertEqual(
            pooled_c[f"{POOLED_METRIC_PREFIX}precision_at_1"],
            float(pool.labels[0] == 1),
        )
        self.assertAlmostEqual(
            per_query_c["precision_at_1"], 1.0 / (1 + 12), delta=0.1
        )
        # spelled out: the per-query value averages EVERY block's first slot
        top1 = [
            float(pool.labels[pool.query_keys == key][0])
            for key in np.unique(pool.query_keys)
        ]
        self.assertEqual(per_query_c["precision_at_1"], float(np.mean(top1)))
        # explicit aliases carry the per-query concept in the name
        for key, value in per_query.items():
            if not key.startswith("query_"):
                self.assertEqual(per_query[f"query_{key}"], value)


class EncodeCostTests(unittest.TestCase):
    def test_added_encode_rows_counts_only_new_payload_rows(self) -> None:
        fixture = Fixture()
        pool = pool_for(fixture, list(range(0, 10)), n=12)
        pool_rows = np.unique(pool.pairs.ravel())
        # everything already encoded => nothing added
        self.assertEqual(added_encode_rows(pool_rows, pool_rows), 0)
        # nothing encoded => every unique row is added
        self.assertEqual(
            added_encode_rows(pool_rows, np.zeros(0, dtype=int)), len(pool_rows)
        )
        half = pool_rows[: len(pool_rows) // 2]
        self.assertEqual(added_encode_rows(pool_rows, half), len(pool_rows) - len(half))

    def test_encode_cost_is_bounded_by_the_universe_not_by_queries_times_n(self) -> None:
        """The pool draws Q*N competitors but encodes only UNIQUE rows."""
        fixture = Fixture()
        n = competitors_per_query(KS)
        rows = list(range(0, fixture.n_source))
        pool = pool_for(fixture, rows, n=n)
        pool_rows = np.unique(pool.pairs.ravel())
        incidences = pool.coverage["pool_draw_incidences"]
        self.assertEqual(incidences, len(rows) * n)  # Q x N competitor slots
        # ... but the encode set is at most queries + positives + the whole
        # candidate universe, i.e. it does NOT grow with Q x N
        bound = 2 * len(rows) + len(fixture.canonical_rows)
        self.assertLessEqual(len(pool_rows), bound)
        self.assertLess(len(pool_rows), incidences // 10)
        # and what the fold actually pays is only what it had not encoded yet
        already = np.unique(np.r_[fixture.query_table(rows).ravel()])
        self.assertLess(added_encode_rows(pool_rows, already), len(pool_rows))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
