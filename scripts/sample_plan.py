#!/usr/bin/env python3
"""scripts/sample_plan.py — emit the sample plan + build the validation set.

Reads the REAL prevalences (bounded: only the declared slice columns of the
canonical entity frame) and the REAL validation census, computes the required
N per slice, per attribute value, and per (attribute value x difficulty) cell
with :class:`core.sample_plan.SamplePlan`, prints the plan, and MATERIALIZES
the validation set the plan asks for onto the declared artifact path + manifest
(``config/sampling.yaml`` ``validation_set:``). The emitted set carries the
``difficulty`` column, so the labels TRAVEL in the bundle/package.

Reads:
  data/canonical_records.csv   (entity prevalences; the slice columns only)
  data/final_validation.csv    (the LIVE/REDUCED scored census: 28 pairs, 4 pos)
  data/labeled_pairs.csv       (the full labelled census: 41 pairs)
  reports/difficulty/full/pairs.csv  (canonical per-pair difficulty; local-only)

Writes:
  data/validation/sample_plan_validation.csv      (the sized validation set)
  results/validation/sample_plan_manifest.json    (targets, required N, realized N)

Usage:
  PYTHONPATH=src .venv/bin/python scripts/sample_plan.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.common import TRAIN_ROOT, F, training_cfg
from core.sample_plan import SamplePlan, SubgroupCensus

#: The live/reduced scored census: final_validation.csv is the scored half
#: after the fold deal, NOT the labelled population (that is labeled_pairs).
SCORED_CENSUS = "final_validation"
LABELLED_CENSUS = "labeled_pairs"
ID_KEY = "_id"
LABEL_KEY = "_label"
#: Difficulty labels come ONLY from the canonical producer; a pair with no
#: measured label is retained as `unknown`, never invented as easy/hard.
UNKNOWN_DIFFICULTY = "unknown"


def _binding(key: str) -> Path:
    """Resolve one declared file binding to an absolute path."""
    path = Path(F[key])
    return path if path.is_absolute() else TRAIN_ROOT / path


class SamplePlanRun:
    """The one script entry point: read real prevalences, plan, build, emit.

    Orchestration only — every decision comes from :class:`SamplePlan` and the
    selection from :class:`core.sample_plan.ValidationSetBuilder`; this class
    reads the frames, attaches the canonical difficulty labels, and writes the
    declared artifacts.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._plan = SamplePlan.from_config()
        self._census = SubgroupCensus.from_config(self._plan.spec)
        self._difficulty: dict[str, str] = {}
        self._difficulty_source = ""

    # ── real inputs (bounded reads) ──
    def _attribute_slices(self) -> list[str]:
        """Every declared slice except the pair-only difficulty axis."""
        return [name for name in self._plan.spec.slices if name != "difficulty"]

    def _canonical_censuses(self):
        """Entity prevalences: read ONLY the declared attribute columns."""
        columns = self._attribute_slices()
        frame = pd.read_csv(
            self._args.canonicals,
            dtype=str,
            keep_default_na=False,
            usecols=columns,
        )
        return self._census.census(frame.to_dict("records"), columns=columns)

    def _load_difficulty(self) -> None:
        """Read the canonical per-pair difficulty labels, or retain `unknown`.

        The measured labels are the local-only output of
        ``training.difficulty.measure_pair`` (spec ``training.yaml difficulty``);
        when the file is absent every pair stays ``unknown`` with the reason
        recorded in the manifest — a level is never invented.
        """
        definition = self._plan.spec.difficulty_definition
        path = self._plan.difficulty_labels_path()
        if definition is None or path is None:
            self._difficulty_source = "unknown: no difficulty definition declared"
            return
        if not path.is_file():
            self._difficulty_source = (
                f"unknown: {path.relative_to(TRAIN_ROOT).as_posix()} absent "
                "(local-only canonical difficulty labels)"
            )
            return
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        column = next(
            (name for name in ("difficulty", "difficulty_slice") if name in frame.columns),
            None,
        )
        if column is None or not {"gtin1", "gtin2"} <= set(frame.columns):
            self._difficulty_source = (
                f"unknown: {definition.measured_labels_path} lacks a difficulty "
                "column / pair key"
            )
            return
        self._difficulty = {
            f"{row.gtin1}|{row.gtin2}": str(row[column]).strip().lower()
            for _, row in frame.iterrows()
        }
        self._difficulty_source = str(path)

    def _pair_samples(self, frame: pd.DataFrame) -> list[dict]:
        """Scored pairs as samples: a slice is the union of BOTH pair sides.

        The pair frame carries per-side bags (``v1_<field>``/``v2_<field>``); a
        pair carries a slice value when EITHER side declares it, plus its
        canonical difficulty label. A slice with no pair-side columns (brand) is
        absent from the pool, never fabricated.
        """
        samples: list[dict] = []
        for _, row in frame.iterrows():
            key = f"{row.gtin1}|{row.gtin2}"
            record: dict = {
                "difficulty": self._difficulty.get(key, UNKNOWN_DIFFICULTY),
                ID_KEY: key,
                LABEL_KEY: str(row.true_label),
            }
            for name in self._attribute_slices():
                base = name.removesuffix("_set")
                columns = [f"v1_{base}", f"v2_{base}"]
                if not all(column in frame.columns for column in columns):
                    continue
                values: list[str] = []
                for column in columns:
                    values.extend(SubgroupCensus.values(row[column], "set_literal"))
                record[name] = list(dict.fromkeys(values))
            samples.append(record)
        return samples

    @staticmethod
    def _row_count(path: Path) -> int:
        """The frame's data-row count (no full read: header + one pass)."""
        return max(sum(1 for _ in path.open(encoding="utf-8")) - 1, 0)

    # ── plan + build + emit ──
    def run(self) -> int:
        self._load_difficulty()
        censuses = list(self._canonical_censuses())
        scored = pd.read_csv(self._args.validation, dtype=str, keep_default_na=False)
        scored_n = len(scored)
        labelled_n = self._row_count(self._args.labeled)
        pool = self._pair_samples(scored)
        # The pair census adds the difficulty axis and the (attribute x
        # difficulty) cells; the attribute slices are already censused over the
        # entity catalog, so only the difficulty column is re-censused here.
        censuses.extend(self._census.census(pool, columns=["difficulty"]))
        folds = int(training_cfg().split.holdout_component_folds)
        report = self._plan.plan(
            censuses,
            labeled_census=labelled_n,
            current_validation=scored_n,
            component_folds=folds,
        )
        self._print_plan(report, scored_n, labelled_n)
        definition = self._plan.spec.difficulty_definition
        difficulty_definition = (
            None
            if definition is None
            else {**definition.model_dump(), "version": training_cfg().difficulty.version}
        )
        selection = self._plan.build_validation_set(
            report,
            pool,
            id_key=ID_KEY,
            label_key=LABEL_KEY,
            source=str(self._args.validation),
            difficulty_definition=difficulty_definition,
            difficulty_source=self._difficulty_source,
        )
        self._emit(scored, pool, selection)
        return 0

    def _print_plan(self, report, scored_n: int, labelled_n: int) -> None:
        spec = self._plan.spec
        print(f"[sample-plan] headline population {report.population} (# "
              "slices carry their own); targets "
              f"c={report.confidence} h={report.ci_half_width} "
              f"alpha={report.alpha} power={report.power} "
              f"effect={report.target_effect}")
        print(f"[sample-plan] per-subgroup N = {report.per_subgroup_n} "
              f"(max of CI n={self._plan.required_proportion_n(spec.targets.worst_case_proportion)} "
              f"and MDE n={self._plan.required_paired_n(report.target_effect)})")
        print(f"[sample-plan] MDE at N: population({report.population})="
              f"{report.mde_at_population*100:.2f}pp  "
              f"scored({scored_n})={report.mde_at_current_validation*100:.2f}pp  "
              f"labelled({labelled_n})={report.mde_at_labeled_census*100:.2f}pp")
        print("[sample-plan] slice / values / measurable / oversample / "
              "whole-field reqN / smallest meaningful")
        for plan in report.slices:
            measurable = sum(1 for item in plan.requirements if item.measurable)
            oversample = sum(1 for item in plan.requirements if item.needs_oversampling)
            binding = plan.smallest_meaningful
            label = "-" if binding is None else f"{binding.value} ({binding.support})"
            print(f"[sample-plan]   {plan.slice:32s} {plan.declared_values:5d} "
                  f"{measurable:5d} {oversample:5d} "
                  f"{plan.populated_required_n!s:>9s}   {label}")
        print(f"[sample-plan] difficulty source: {self._difficulty_source or 'unknown'}")
        binding = report.binding
        if binding is None:
            print(f"[sample-plan] BINDING: none meaningful; whole-population "
                  f"N={report.recommended_n}")
        else:
            print(f"[sample-plan] BINDING smallest meaningful subgroup: "
                  f"{binding.slice}={binding.value!r} support={binding.support} "
                  f"share={binding.share:.6g} -> required N={binding.required_n}")
        print(f"[sample-plan] recommended split.validation_size="
              f"{report.recommended_validation_size} ({report.validation_size_unit}); "
              f"reachable={report.validation_size_reachable}")
        for note in report.notes:
            print(f"[sample-plan] note: {note}")

    def _emit(self, scored: pd.DataFrame, pool: list[dict], selection) -> None:
        spec = self._plan.spec.validation_set
        if spec is None:
            raise ValueError(
                "config/sampling.yaml declares no validation_set: block; "
                "nothing to emit"
            )
        path = TRAIN_ROOT / spec.path
        manifest_path = TRAIN_ROOT / spec.manifest_path
        difficulty = {sample[ID_KEY]: sample["difficulty"] for sample in pool}
        selected = scored[scored.apply(
            lambda row: f"{row.gtin1}|{row.gtin2}" in set(selection.ids), axis=1
        )].copy()
        # The difficulty labels TRAVEL with the set (a column, not a local CSV).
        selected["difficulty"] = [
            difficulty.get(f"{row.gtin1}|{row.gtin2}", UNKNOWN_DIFFICULTY)
            for _, row in selected.iterrows()
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(path, index=False)
        manifest = {
            "bundle_member": spec.bundle_member,
            "artifact": spec.path,
            "validation_set": selection.manifest.model_dump(),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        realized = selection.manifest
        print(f"[sample-plan] emitted {path} ({realized.realized_n} rows; "
              f"{realized.positive} pos / {realized.negative} neg) + {manifest_path}")
        print(f"[sample-plan] realized coverage: "
              f"{len(realized.coverage) - len(realized.uncovered)}/"
              f"{len(realized.coverage)} targets covered; "
              f"{len(realized.uncovered)} uncovered (pool={realized.pool_size}, "
              f"target={realized.required_n})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonicals", type=Path, default=_binding("canonical_records"))
    parser.add_argument("--validation", type=Path, default=_binding(SCORED_CENSUS))
    parser.add_argument("--labeled", type=Path, default=_binding(LABELLED_CENSUS))
    args = parser.parse_args()
    return SamplePlanRun(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
