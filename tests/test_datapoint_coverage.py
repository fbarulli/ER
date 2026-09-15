"""Datapoint-population coverage and negative-source accounting (audit A4-2).

The defect these tests lock down: ``src/training/train.py`` emits negatives
whose provenance tag is ``targeted_attribute_conflict`` (measured 350 rows per
fold), but ``training.training.KNOWN_DATAPOINT_POPULATIONS`` was a hand-kept
literal list that did not contain it. A population outside the registry, the
fold's expected set and the presented set is never visited by the coverage
audit — so the audit that exists to prove nothing drops silently was itself
dropping a whole producer lane, and the negative-source census printed
percentages that summed to less than 100% while claiming to account for every
train-fold negative.

Three layers are covered here, all driven DIRECTLY (no training run needed —
the coverage computation is pure):

1. REGISTRY FROM PRODUCERS — the tag set discovered by scanning the producer
   sources must equal the registry, so a future divergence fails a test rather
   than a review.
2. COVERAGE OUTPUT — an unregistered population can no longer be silently
   omitted: it is written to the artifact and then raises loudly. Registered
   populations keep their exact ``ok`` / ``missing`` / ``eval_only`` /
   ``not_reached`` / ``unavailable`` semantics, and a registered population
   that contributes zero pairs still gets an explicit row.
3. ACCOUNTING IDENTITIES — the per-fold negative-source census enumerates
   every producer and its registered counts sum to the fold's negatives.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import core.common as core_common
import training.training as training


REPO_ROOT = Path(__file__).resolve().parents[1]

# Producer modules that can write a datapoint population or a negative-source
# tag. Every one of them is scanned below.
PRODUCER_FILES = (
    "src/training/train.py",
    "src/training/training.py",
    "src/training/ann_refresh.py",
    "src/training/train_prepared.py",
    "src/training/prepared_bundle.py",
    "src/training/masking.py",
    "src/core/hard_negatives.py",
    "src/pipeline.py",
)

# The three idioms that actually produce a datapoint/source tag.
SOURCE_ARRAY_TAG = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*,\s*dtype=object')
APPENDED_POPULATION_TAG = re.compile(
    r'populations\.append\(\s*"([A-Za-z_][A-Za-z0-9_]*)"'
)
AUDIT_SOURCE_TAG = re.compile(r'"source"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"')
LINEAGE_POPULATION_TAG = re.compile(r'population="([A-Za-z_][A-Za-z0-9_]*)"')

# The mask-audit lineage vocabulary is a DIFFERENT domain from the
# datapoint-population registry (_usage_row's docstring warns about exactly
# this); it is not a producer of coverage populations.
LINEAGE_DOMAIN = frozenset({"positive", "negative"})

# The registry exactly as it stood before the fix — kept here so the
# regression it caused can be executed rather than described.
PRE_FIX_REGISTRY = (
    "gate_positive",
    "hard_positive",
    "masked_positive",
    "gate",
    "attribute_conflict",
    "random_easy",
    "ann_finetuned",
)


def _scan(pattern: re.Pattern[str]) -> set[str]:
    found: set[str] = set()
    for relative in PRODUCER_FILES:
        found.update(pattern.findall((REPO_ROOT / relative).read_text(encoding="utf-8")))
    return found


class _CapturingVisibilityLog(unittest.TestCase):
    """Base: capture visibility artifacts instead of writing results/."""

    def setUp(self) -> None:
        self.frames: dict[str, pd.DataFrame] = {}
        self._original = core_common.write_visibility_log

        def capture(df, name, run_tag, sample):  # noqa: ANN001 - test double
            self.frames[name] = df

        core_common.write_visibility_log = capture
        self.addCleanup(
            lambda: setattr(core_common, "write_visibility_log", self._original)
        )

    def coverage_frame(self, fold: int = 0) -> pd.DataFrame:
        return self.frames[f"datapoint_type_coverage_fold{fold}.csv"]

    def run_coverage(
        self,
        pair_populations: list[str],
        *,
        presented_pairs: int | None = None,
        dynamic_populations: set[str] | None = None,
        fold: int = 0,
    ) -> dict[str, int]:
        """Drive _write_datapoint_usage with synthetic inputs."""
        if presented_pairs is None:
            presented_pairs = len(pair_populations)
        counts: dict[tuple, int] = {}
        for pair_id, population in enumerate(pair_populations[:presented_pairs]):
            counts[(0, pair_id, population, "none", 0)] = 1
        return training._write_datapoint_usage(
            fold_i=fold,
            pair_populations=pair_populations,
            presentation_counts=counts,
            pair_lineage=None,
            dynamic_populations=(
                {"ann_finetuned", "attribute_conflict"}
                if dynamic_populations is None
                else dynamic_populations
            ),
            run_tag="test-coverage",
            sample=True,
        )


class ProducerTagInventoryTests(_CapturingVisibilityLog):
    """Layer 1 — the registry is derived from the producers."""

    def test_every_producer_tag_is_registered(self) -> None:
        scanned = (
            _scan(SOURCE_ARRAY_TAG)
            | _scan(APPENDED_POPULATION_TAG)
            | _scan(AUDIT_SOURCE_TAG)
        )
        registry = set(training.KNOWN_DATAPOINT_POPULATIONS)
        fallbacks = set(training.DATAPOINT_FALLBACK_TAGS)
        self.assertTrue(scanned, "producer scan found no tags at all")
        undeclared = scanned - registry - fallbacks
        self.assertEqual(
            undeclared,
            set(),
            "producer(s) emit datapoint population tag(s) missing from "
            "DATAPOINT_POPULATION_SPEC; a new tag must be registered (with its "
            "emitter) or it escapes the coverage audit: "
            f"{sorted(undeclared)}",
        )
        # The reverse direction: a registry entry no producer emits is stale.
        self.assertEqual(scanned - fallbacks, registry)

    def test_targeted_attribute_conflict_is_registered(self) -> None:
        """The exact tag this audit exists to fix."""
        self.assertIn("targeted_attribute_conflict", _scan(SOURCE_ARRAY_TAG))
        self.assertIn(
            "targeted_attribute_conflict", training.KNOWN_DATAPOINT_POPULATIONS
        )
        self.assertIn(
            "targeted_attribute_conflict",
            training.NEGATIVE_SOURCE_DATAPOINT_POPULATIONS,
        )

    def test_lineage_population_vocabulary_is_declared(self) -> None:
        scanned = _scan(LINEAGE_POPULATION_TAG)
        allowed = (
            set(training.KNOWN_DATAPOINT_POPULATIONS)
            | set(training.DATAPOINT_FALLBACK_TAGS)
            | LINEAGE_DOMAIN
        )
        self.assertEqual(scanned - allowed, set())

    def test_registry_roles_match_the_derived_constants(self) -> None:
        spec = training.DATAPOINT_POPULATION_SPEC
        self.assertEqual(tuple(spec), training.KNOWN_DATAPOINT_POPULATIONS)
        self.assertEqual(
            training.DYNAMIC_DATAPOINT_POPULATIONS,
            frozenset({"ann_finetuned", "attribute_conflict"}),
        )
        self.assertEqual(
            training.NEGATIVE_SOURCE_DATAPOINT_POPULATIONS,
            ("gate", "targeted_attribute_conflict", "attribute_conflict",
             "random_easy"),
        )
        for name, entry in spec.items():
            self.assertTrue(entry.get("emitter"), f"{name} has no declared emitter")


class CoverageOutputTests(_CapturingVisibilityLog):
    """Layer 2 — nothing is silently omitted; statuses keep their meaning."""

    MIXED = (
        ["gate_positive"] * 8
        + ["hard_positive"] * 2
        + ["masked_positive"] * 1
        + ["gate"] * 20
        + ["targeted_attribute_conflict"] * 7
        + ["attribute_conflict"] * 3
        + ["random_easy"] * 20
    )

    def test_pre_fix_registry_dropped_the_population_from_the_artifact(self) -> None:
        """Execute the defect: with the old registry the row is missing."""
        original = training.KNOWN_DATAPOINT_POPULATIONS
        training.KNOWN_DATAPOINT_POPULATIONS = PRE_FIX_REGISTRY
        try:
            # The producer lane produced zero targeted pairs, so the tag can
            # only reach the audit through the registry.
            populations = [p for p in self.MIXED if p != "targeted_attribute_conflict"]
            self.run_coverage(populations)
            before = set(self.coverage_frame()["population"])
            self.assertNotIn("targeted_attribute_conflict", before)
        finally:
            training.KNOWN_DATAPOINT_POPULATIONS = original
        # Closed: the registered population now gets an explicit row even at
        # zero pairs, so a lane that silently contributes nothing is visible.
        self.run_coverage(populations)
        after = self.coverage_frame()
        self.assertIn("targeted_attribute_conflict", set(after["population"]))
        row = after.set_index("population").loc["targeted_attribute_conflict"]
        self.assertEqual(int(row["expected_pairs"]), 0)
        self.assertEqual(int(row["presentations"]), 0)
        self.assertEqual(row["status"], "unavailable")
        self.assertTrue(bool(row["registered"]))

    def test_unregistered_population_is_visible_then_loud(self) -> None:
        """A producer tag outside the registry must not vanish silently."""
        populations = self.MIXED + ["brand_new_lane"] * 4
        with self.assertRaises(training.UnregisteredDatapointPopulationError) as ctx:
            self.run_coverage(populations)
        message = str(ctx.exception)
        self.assertIn("brand_new_lane", message)
        # The artifact was written BEFORE the raise, so the evidence survives.
        frame = self.coverage_frame()
        row = frame.set_index("population").loc["brand_new_lane"]
        self.assertFalse(bool(row["registered"]))
        self.assertEqual(row["status"], "unregistered")
        self.assertEqual(int(row["expected_pairs"]), 4)

    def test_registered_zero_pair_population_has_an_honest_status(self) -> None:
        result = self.run_coverage(
            ["gate_positive"] * 3 + ["gate"] * 5,
            dynamic_populations={"attribute_conflict"},
        )
        frame = self.coverage_frame().set_index("population")
        # dynamic + enabled + empty -> not_reached (unchanged semantics)
        self.assertEqual(frame.loc["attribute_conflict", "status"], "not_reached")
        # dynamic + disabled -> absent, because it is not configured at all
        self.run_coverage(
            ["gate_positive"] * 3 + ["gate"] * 5, dynamic_populations=set()
        )
        self.assertNotIn(
            "attribute_conflict", set(self.coverage_frame()["population"])
        )
        # non-dynamic registered + no pairs -> unavailable (unchanged semantics)
        self.assertEqual(frame.loc["random_easy", "status"], "unavailable")
        self.assertEqual(frame.loc["targeted_attribute_conflict", "status"],
                         "unavailable")
        self.assertEqual(result["n_missing_datapoint_populations"], 0)
        self.assertEqual(result["n_unregistered_datapoint_populations"], 0)

    def test_missing_status_is_preserved_for_non_empty_unpresented(self) -> None:
        """`missing` stays load-bearing: non-empty population, zero shows."""
        result = self.run_coverage(
            ["gate"] * 6 + ["targeted_attribute_conflict"] * 2,
            presented_pairs=6,
        )
        frame = self.coverage_frame().set_index("population")
        self.assertEqual(frame.loc["targeted_attribute_conflict", "status"], "missing")
        self.assertEqual(result["n_missing_datapoint_populations"], 1)

    def test_identity_holds_for_a_realistic_mixed_population(self) -> None:
        result = self.run_coverage(self.MIXED)
        frame = self.coverage_frame()
        self.assertEqual(int(frame["expected_pairs"].sum()), len(self.MIXED))
        self.assertEqual(int(frame["presentations"].sum()), len(self.MIXED))
        self.assertTrue(frame["registered"].all())
        self.assertEqual(result["n_unregistered_datapoint_populations"], 0)
        self.assertEqual(result["n_missing_datapoint_populations"], 0)
        self.assertEqual(result["n_coverage_populations"],
                         int(result["n_coverage_registered_populations"]))

    def test_coverage_identity_assertions_are_live_and_loud(self) -> None:
        """The identity guards are asserted, not decorative.

        Driven directly, because with the real call path the two sums are equal
        BY CONSTRUCTION (the rows are built from the same counters). That is
        exactly why the reachable breach is a producer tag outside the registry
        — covered below — and why these guards exist to catch a future
        refactor that stops aggregating a key or truncates the pair list.
        """
        rows = [
            {"population": "gate", "expected_pairs": 3, "presentations": 2},
        ]
        with self.assertRaises(ValueError) as closure:
            training._assert_datapoint_coverage_identity(
                fold_i=0,
                coverage_rows=rows,
                presentation_counts={(0, 0, "gate", "none", 0): 5},
                observed={"gate": 5},
                pair_populations=["gate"] * 3,
                unregistered=[],
                by_population={"gate": {"presentations": 2, "pair_ids": {0}}},
            )
        self.assertIn("identity 1", str(closure.exception))
        with self.assertRaises(ValueError) as census:
            training._assert_datapoint_coverage_identity(
                fold_i=0,
                coverage_rows=rows,
                presentation_counts={(0, 0, "gate", "none", 0): 2},
                observed={"gate": 2},
                pair_populations=["gate"] * 4,
                unregistered=[],
                by_population={"gate": {"presentations": 2, "pair_ids": {0}}},
            )
        self.assertIn("identity 3", str(census.exception))
        with self.assertRaises(training.UnregisteredDatapointPopulationError):
            training._assert_datapoint_coverage_identity(
                fold_i=0,
                coverage_rows=rows,
                presentation_counts={(0, 0, "gate", "none", 0): 2},
                observed={"gate": 2},
                pair_populations=["gate"] * 3,
                unregistered=["brand_new_lane"],
                by_population={"gate": {"presentations": 2, "pair_ids": {0}}},
            )

    def test_ambiguous_pair_attribution_is_visible(self) -> None:
        """A pair presented under two labels is reported, not assumed away."""
        rows = [{"population": p, "expected_pairs": 1, "presentations": 1}
                for p in ("attribute_conflict", "ann_finetuned")]
        training._assert_datapoint_coverage_identity(
            fold_i=0,
            coverage_rows=rows,
            presentation_counts={(0, 0, "attribute_conflict", "none", 0): 1,
                                (1, 0, "ann_finetuned", "none", 1): 1},
            observed={"attribute_conflict": 1, "ann_finetuned": 1},
            pair_populations=["attribute_conflict"] * 2,
            unregistered=[],
            by_population={
                "attribute_conflict": {"presentations": 1, "pair_ids": {0}},
                "ann_finetuned": {"presentations": 1, "pair_ids": {0}},
            },
        )
        self.assertEqual(
            training._ambiguous_pair_attributions(
                {
                    "attribute_conflict": {"presentations": 1, "pair_ids": {0, 1}},
                    "ann_finetuned": {"presentations": 1, "pair_ids": {0}},
                }
            ),
            1,
        )


class NegativeSourceAccountingTests(_CapturingVisibilityLog):
    """Layer 3 — the negative-source census enumerates every producer."""

    # Exactly what train.py can emit, with the measured 350 targeted rows/fold.
    SYNTHETIC_SOURCES = (
        ["gate"] * 10263
        + ["targeted_attribute_conflict"] * 350
        + ["attribute_conflict"] * 40
        + ["random_easy"] * 10263
    )

    def _account(self, sources, usage_rows=None):
        array = np.asarray(sources, dtype=object)
        pairs = np.zeros((len(array), 2), dtype=int)
        if usage_rows is None:
            usage_rows = [
                {
                    "pair_id": index,
                    "population": str(source),
                    "present_count": 1,
                    "hard_selected_count": 1,
                    "backprop_count": 1,
                }
                for index, source in enumerate(array)
            ]
        return training._negative_source_accounting(
            fold_i=0, tr_negs=pairs, tr_neg_sources=array, usage_rows=usage_rows
        )

    def test_every_producer_source_is_counted_and_sums_to_the_fold(self) -> None:
        coverage = self._account(self.SYNTHETIC_SOURCES)
        total = len(self.SYNTHETIC_SOURCES)
        self.assertEqual(
            coverage["n_train_neg_source_targeted_attribute_conflict"], 350
        )
        # These three counters did not exist for this population before the
        # fix: the hardcoded {"gate","attribute_conflict","random_easy"} list
        # skipped it, so present/selected/backprop silently under-counted.
        for suffix in ("_present", "_selected", "_backprop"):
            self.assertEqual(
                coverage[f"n_train_neg_source_targeted_attribute_conflict{suffix}"],
                350,
            )
        self.assertIn("pct_train_neg_source_targeted_attribute_conflict", coverage)
        census = sum(
            coverage[f"n_train_neg_source_{name}"]
            for name in training.NEGATIVE_SOURCE_DATAPOINT_POPULATIONS
        )
        self.assertEqual(census, total)
        self.assertEqual(coverage["n_train_neg_source_total"], total)
        self.assertEqual(coverage["n_train_neg_source_registered"], total)
        self.assertEqual(coverage["n_train_neg_source_unregistered"], 0)
        self.assertEqual(coverage["n_train_neg_usage_rows_unattributed"], 0)
        self.assertAlmostEqual(
            sum(
                coverage[f"pct_train_neg_source_{name}"]
                for name in training.NEGATIVE_SOURCE_DATAPOINT_POPULATIONS
            ),
            1.0,
        )

    def test_unregistered_source_tag_is_loud(self) -> None:
        for tag in ("brand_new_lane", "hard_negative"):
            with self.assertRaises(
                training.UnregisteredDatapointPopulationError
            ) as ctx:
                self._account(["gate"] * 3 + [tag] * 2)
            self.assertIn(tag, str(ctx.exception))

    def test_unattributable_usage_label_is_loud(self) -> None:
        with self.assertRaises(training.UnregisteredDatapointPopulationError) as ctx:
            self._account(
                ["gate"] * 2,
                usage_rows=[
                    {"pair_id": 0, "population": "gate", "present_count": 1,
                     "hard_selected_count": 1, "backprop_count": 1},
                    {"pair_id": 1, "population": "hard_negative", "present_count": 1,
                     "hard_selected_count": 1, "backprop_count": 1},
                ],
            )
        self.assertIn("hard_negative", str(ctx.exception))

    def test_provenance_length_mismatch_is_loud(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            training._negative_source_accounting(
                fold_i=0,
                tr_negs=np.zeros((5, 2), dtype=int),
                tr_neg_sources=np.asarray(["gate"] * 4, dtype=object),
                usage_rows=[],
            )
        self.assertIn("N1", str(ctx.exception))

    def test_funnel_breach_is_loud(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._account(
                ["gate"] * 2,
                usage_rows=[
                    {"pair_id": index, "population": "gate", "present_count": 1,
                     "hard_selected_count": 1, "backprop_count": 0}
                    for index in range(3)
                ],
            )
        self.assertIn("N2", str(ctx.exception))

    def test_ann_label_is_attributed_without_becoming_a_source(self) -> None:
        coverage = self._account(
            ["gate"] * 4,
            usage_rows=[
                {"pair_id": 0, "population": "gate", "present_count": 1,
                 "hard_selected_count": 1, "backprop_count": 1},
                {"pair_id": 1, "population": "ann_finetuned", "present_count": 2,
                 "hard_selected_count": 1, "backprop_count": 1},
            ],
        )
        self.assertEqual(coverage["n_train_neg_source_gate"], 4)
        self.assertEqual(coverage["n_train_neg_source_ann_finetuned_present"], 1)
        self.assertEqual(coverage["n_train_neg_usage_rows_unattributed"], 0)


class ReportCoverageSurfacingTests(unittest.TestCase):
    """Layer 2/3 in the report: the invisible population becomes visible."""

    def _frame(self, drop: tuple[str, ...] = (), **overrides) -> pd.DataFrame:
        base = {
            "fold": [0],
            "status": ["ok"],
            "n_presented_gate_positive": [10],
            "n_presented_gate": [100],
            "n_presented_random_easy": [100],
            "n_presented_masked_positive": [3],
            "n_presented_hard_positive": [2],
            "n_presented_targeted_attribute_conflict": [0],
            "n_presented_attribute_conflict": [0],
            "n_presented_ann_finetuned": [0],
            "n_missing_datapoint_populations": [0],
            "n_coverage_ambiguous_pair_attributions": [0],
            "n_train_neg_source_gate": [2600],
            "n_train_neg_source_gate_present": [2500],
            "n_train_neg_source_gate_selected": [2400],
            "n_train_neg_source_gate_backprop": [2400],
            "n_train_neg_source_targeted_attribute_conflict": [350],
            "n_train_neg_source_targeted_attribute_conflict_present": [300],
            "n_train_neg_source_targeted_attribute_conflict_selected": [300],
            "n_train_neg_source_targeted_attribute_conflict_backprop": [300],
            "n_train_neg_source_attribute_conflict": [0],
            "n_train_neg_source_attribute_conflict_present": [0],
            "n_train_neg_source_attribute_conflict_selected": [0],
            "n_train_neg_source_attribute_conflict_backprop": [0],
            "n_train_neg_source_random_easy": [2600],
            "n_train_neg_source_random_easy_present": [2500],
            "n_train_neg_source_random_easy_selected": [2400],
            "n_train_neg_source_random_easy_backprop": [2400],
            "n_train_neg_source_total": [5550],
            "n_train_neg_usage_rows_unattributed": [0],
        }
        base.update(overrides)
        for column in drop:
            base.pop(column, None)
        return pd.DataFrame(base)

    def _section(self, frame: pd.DataFrame) -> dict:
        from training import generate_training_report as report

        return report._datapoint_coverage_section(
            frame, report._datapoint_population_registry()
        )

    def test_zero_presentation_populations_are_surfaced(self) -> None:
        section = self._section(self._frame())
        self.assertTrue(section["registry_available"])
        fold = section["folds"]["fold_0"]
        self.assertTrue(fold["tracked"])
        self.assertIn(
            "targeted_attribute_conflict", fold["populations"]
        )
        self.assertEqual(
            fold["populations"]["targeted_attribute_conflict"], 0
        )
        self.assertIn(
            "targeted_attribute_conflict",
            fold["populations_with_zero_presentations"],
        )
        self.assertIn(
            "fold_0", section["populations_with_zero_presentations"]
        )

    def test_negative_source_identity_is_reported_per_fold(self) -> None:
        section = self._section(self._frame())
        fold = section["folds"]["fold_0"]
        self.assertEqual(fold["negative_source_census"], 5550)
        self.assertEqual(fold["negative_source_declared_total"], 5550)
        self.assertTrue(fold["negative_source_identity_holds"])
        self.assertEqual(fold["unregistered_negative_sources"], [])
        self.assertEqual(section["negative_source_identity_breaches"], [])
        self.assertEqual(fold["negative_sources"]["targeted_attribute_conflict"],
                         {"total": 350, "present": 300, "selected": 300,
                          "backprop": 300})

    def test_identity_breach_is_reported_not_hidden(self) -> None:
        section = self._section(self._frame(n_train_neg_source_total=[9000]))
        fold = section["folds"]["fold_0"]
        self.assertFalse(fold["negative_source_identity_holds"])
        self.assertEqual(len(section["negative_source_identity_breaches"]), 1)

    def test_untracked_run_is_labelled_not_silently_empty(self) -> None:
        frame = pd.DataFrame({"fold": [0], "status": ["ok"]})
        section = self._section(frame)
        fold = section["folds"]["fold_0"]
        self.assertFalse(fold["tracked"])
        # The registry still lists what should have been reported.
        self.assertIn("targeted_attribute_conflict", section["registry_populations"])
        self.assertIn(
            "targeted_attribute_conflict", section["registered_but_absent"]
        )

    def test_unregistered_column_is_flagged(self) -> None:
        section = self._section(self._frame(n_presented_brand_new_lane=[1]))
        fold = section["folds"]["fold_0"]
        self.assertEqual(fold["populations"]["brand_new_lane"], 1)

    def test_flattened_rows_include_registry_populations_never_reported(self) -> None:
        """The CSV companion is complete, not just what the run happened to emit."""
        from training import generate_training_report as report

        # (a) the run DID report the population, at zero -> reported=1, zero row.
        rows = {
            row["population"]: row
            for row in report._datapoint_coverage_rows(self._section(self._frame()))
        }
        for name in ("targeted_attribute_conflict", "ann_finetuned"):
            self.assertEqual(rows[name]["reported"], 1)
            self.assertEqual(rows[name]["presentations"], 0)
            self.assertEqual(rows[name]["zero_presented"], 1)
            self.assertEqual(rows[name]["registered"], 1)
        # A reported population keeps reported=1 and its real counts.
        self.assertEqual(rows["gate"]["reported"], 1)
        self.assertEqual(rows["gate"]["presentations"], 100)
        self.assertEqual(rows["gate"]["negative_source_pairs"], 2600)
        self.assertEqual(rows["gate"]["negative_source_selected"], 2400)

        # (b) the run never reported it at all (no column anywhere) -> the row
        # still exists, with reported=0. This is the real shape of every run on
        # disk today: 0/28 coverage CSVs ever mentioned the targeted lane.
        silent = self._section(
            self._frame(
                drop=(
                    "n_presented_targeted_attribute_conflict",
                    "n_presented_ann_finetuned",
                    "n_train_neg_source_targeted_attribute_conflict",
                    "n_train_neg_source_targeted_attribute_conflict_present",
                    "n_train_neg_source_targeted_attribute_conflict_selected",
                    "n_train_neg_source_targeted_attribute_conflict_backprop",
                )
            )
        )
        silent_rows = {
            row["population"]: row
            for row in report._datapoint_coverage_rows(silent)
        }
        for name in ("targeted_attribute_conflict", "ann_finetuned"):
            self.assertEqual(silent_rows[name]["reported"], 0)
            self.assertEqual(silent_rows[name]["presentations"], 0)
            self.assertEqual(silent_rows[name]["registered"], 1)
        # Dropping the targeted SOURCE column while the fold still declares
        # 5,550 negatives is precisely the defect shape: the census no longer
        # reconciles. The report now says so instead of silently omitting 350
        # negatives (the old hardcoded three-source list produced exactly this
        # 5,200-of-5,550 picture with NO warning).
        self.assertEqual(
            silent["negative_source_identity_breaches"],
            ["fold 0: census=5200 != declared_total=5550"],
        )
        self.assertFalse(
            silent["folds"]["fold_0"]["negative_source_identity_holds"]
        )


if __name__ == "__main__":
    unittest.main()
