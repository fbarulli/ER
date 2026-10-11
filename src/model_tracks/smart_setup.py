"""The smart 50/50 zero-leakage partition of an EXISTING prepared track setup.

Owner mandate (2026-10-11): the full track trains against a 50% train / 50%
validate carve that is stratified per attribute and per difficulty and splits at
the COMPONENT/ENTITY level, so no GTIN endpoint, component or positive pair is
shared across the halves. This module PARTITIONS the prepared data — it never
re-bundles or re-prepares it from raw: the catalog, pairs, embeddings, token
table and frozen text bundle all pass through byte-for-byte; only the split
declaration and the graph artifacts derived from it are rewritten.

SCOPE — the graph lane (the ``model_tracks`` setup: ``eligible_catalog.csv``,
``listing_splits.csv``, ``listing_pairs.csv`` and the ``prepared/`` package the
``gnn_only`` + ``cascade`` tracks read). The text bundle's frozen holdout is NOT
re-bundled here (owner rule: never re-bundle); :meth:`partition` emits the
gtin->split ``populations`` as ``smart_partition_populations.json`` so the text
lane (a sibling owner) can adopt the SAME carve. Today the graph split and the
text bundle populations are one shared split; moving only the graph lane would
de-sync them, so a coordinated Gate-2 run re-freezes the text holdout from these
populations before launch (see the module README note in the run report).

Canonical reuse (no second implementation of anything):

* the carve (roles, ratios, strata, seed) is ``core.smart_split.SmartSplit``
  reading ``config/smart_split.yaml`` — train 0.5 vs validate = dev ∪ test 0.5,
  stratified per ``difficulty_slice`` × ``attribute``;
* the component graph is ``training.folds.merged_component_graph`` (the P0
  merged positive graph: the catalog's positive listing pairs + the labeled-pair
  census edges) mapped through ``training.folds.ComponentIndex``;
* the setup CSVs are written by ``graph_tracks.setup.write_setup_frames``, the
  prepared package by ``graph_tracks.prepare.prepare``, the lane configs by
  ``graph_tracks.setup.write_track_config`` / ``write_text_config`` with the ONE
  path rebaser (``model_tracks.smoke_inputs.repoint_lane_paths``).

Entry point: ``python -m model_tracks.smart_setup <setup_dir> <output_dir>``
publishes the partitioned setup and its runnable ``suite.yaml`` (the config SSOT
knob the full-track launch consumes: ``--tracks-config <output>/suite.yaml``).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.run_log import RunLogger
from core.smart_split import SmartSplit

_LOG = RunLogger(__name__)

#: The two declared strata axes this partition measures per component. They
#: must equal ``config/smart_split.yaml`` ``strata`` (checked at construction):
#: the partition cannot honor an axis it does not measure.
DIFFICULTY_AXIS = "difficulty_slice"
ATTRIBUTE_AXIS = "attribute"

#: Honest tags for an unmeasured axis value (never invented as easy/hard).
UNMEASURED_DIFFICULTY = "unknown"
NO_ATTRIBUTE = "none"

#: The cohort reason that marks a difficulty row as NOT measured.
UNMEASURED_DIFFICULTY_REASON = "no_frozen_encoder_input"

#: The partition census schema (embedded in the published setup manifest).
CENSUS_SCHEMA = "er-smart-setup-partition-v1"

#: The published gtin->split populations the text lane adopts (sibling owner).
POPULATIONS_NAME = "smart_partition_populations.json"


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


@dataclass(frozen=True)
class SetupPartition:
    """The carve result: the frames to write, the gtin populations, the census."""

    #: sku_id -> split over the retained catalog (the new listing_splits frame).
    assignments: pd.DataFrame
    #: The re-split supervised pairs (cross-half negatives excluded, counted).
    pairs: pd.DataFrame
    #: split -> sorted normalized GTINs (the entity-level carve; the text lane
    #: re-freezes its holdout from exactly these populations).
    populations: dict[str, list[str]]
    #: The published census (per-split counts, per-axis share preservation).
    census: dict[str, Any]


class SmartSetupPartition:
    """One prepared setup -> its smart 50/50 stratified zero-leakage carve.

    Construction takes the setup directory and the canonical ``SmartSplit``
    owner (``from_config`` reads the declared document); :meth:`partition`
    computes the carve; :meth:`publish` writes the new graph-lane setup tree.
    """

    def __init__(self, setup_dir: Path, split: SmartSplit, *,
                 suite_config: Path | None = None) -> None:
        self._setup = Path(setup_dir).resolve()
        self._split = split
        self._suite_config = suite_config
        axes = set(split.spec.strata)
        if axes != {DIFFICULTY_AXIS, ATTRIBUTE_AXIS}:
            raise ValueError(
                f"the declared strata {sorted(axes)} are not the axes this "
                f"partition measures ({DIFFICULTY_AXIS}, {ATTRIBUTE_AXIS})")
        self._catalog: pd.DataFrame | None = None
        self._pairs: pd.DataFrame | None = None
        self._tags: dict[str, str] | None = None
        self._component_tags: dict[int, tuple[str, str]] = {}

    @classmethod
    def from_config(cls, setup_dir: Path, *,
                    suite_config: Path | None = None) -> "SmartSetupPartition":
        """Build from the declared smart-split document (the ONE config home)."""
        return cls(setup_dir, SmartSplit.from_config(), suite_config=suite_config)

    # ── inputs ────────────────────────────────────────────────────────────

    def _inputs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The retained catalog and the supervised pairs (read once)."""
        if self._catalog is None or self._pairs is None:
            layout = _setup_layout()
            self._catalog = pd.read_csv(self._setup / layout.catalog, dtype=str,
                                        keep_default_na=False, low_memory=False)
            self._pairs = pd.read_csv(self._setup / layout.pairs, dtype=str,
                                      keep_default_na=False)
        return self._catalog, self._pairs

    def _attribute_tags(self) -> dict[str, str]:
        """sku -> its dominant declared attribute field (the model-input axis).

        Read from the setup's OWN prepared listings (the extractor's attribute
        registry), never re-extracted: the first observed non-``brand`` field in
        the registry's sorted order. ``brand`` is declared by ~every listing, so
        leading with it would collapse the axis to one stratum; it is the
        last-resort tag (a listing declaring only ``brand``), with ``none`` for a
        listing declaring nothing.
        """
        if self._tags is None:
            listings_path = (self._setup / _setup_layout().prepared_dir
                             / _setup_layout().listings)
            if not listings_path.is_file():
                raise ValueError(f"prepared listings missing: {listings_path}")
            document = json.loads(listings_path.read_text())
            self._tags = {}
            for record in document["listings"]:
                observed = sorted(field for field, values in record["attribute"].items()
                                  if values)
                discriminative = [field for field in observed if field != "brand"]
                self._tags[str(record["sku_id"])] = (
                    discriminative[0] if discriminative
                    else ("brand" if observed else NO_ATTRIBUTE))
        return self._tags

    def _difficulty_votes(self) -> dict[tuple[str, str], str]:
        """(sku, sku) unordered -> the cohort's MEASURED difficulty (its mode).

        The ablation cohort already measured pair difficulty from the frozen
        encoder payload; this partition reuses that measurement instead of
        re-deriving it. The join is the SKU pair, NOT ``example_id``: the
        cohort's measured rows (the bundle/minted populations) carry no
        example_id, while its unmeasured ``real`` rows (the only ones that do)
        cast no vote — a component with no measured pair stays honestly
        ``unknown``, never invented as easy/hard.
        """
        cohort = self._setup / "ablation_cohort" / "pairs.csv"
        if not cohort.is_file():
            _LOG.info("[smart-setup] no ablation cohort in the source setup; "
                      "every component stays difficulty=unknown")
            return {}
        frame = pd.read_csv(cohort, usecols=["sku_id1", "sku_id2", "difficulty_slice",
                                             "difficulty_reason"], dtype=str,
                            keep_default_na=False)
        measured = frame[frame.difficulty_reason != UNMEASURED_DIFFICULTY_REASON]
        counters: dict[tuple[str, str], Counter] = {}
        for row in measured.itertuples(index=False):
            key = tuple(sorted((str(row.sku_id1), str(row.sku_id2))))
            counters.setdefault(key, Counter())[str(row.difficulty_slice)] += 1
        return {key: self._modal(counter, UNMEASURED_DIFFICULTY)
                for key, counter in counters.items()}

    # ── the component graph ───────────────────────────────────────────────

    def _component_graph(self, catalog: pd.DataFrame,
                         pairs: pd.DataFrame) -> tuple[dict[str, int], dict]:
        """(gtin -> component id, edge stats) over the canonical merged graph.

        The component authority is ``merged_component_graph`` over the CATALOG
        listings: the catalog's positive listing pairs are the cross-gtin edges,
        and the labeled-pair census positives are folded in (the P0 fix). The
        text bundle's own positives are same-gtin singletons (two views of one
        product), so they merge nothing across gtins and are not needed here —
        the catalog graph is the one component authority for both lanes.
        """
        from training.folds import (
            ComponentIndex,
            merged_component_graph,
            normalize_gtin,
        )
        gtin_of_sku = {row.sku_id: normalize_gtin(row.gtin)
                       for row in catalog.itertuples(index=False)}
        if not all(gtin_of_sku.values()):
            raise ValueError("the eligible catalog carries an empty gtin; "
                             "identity is required to partition without leakage")
        row_of_sku = {sku: index for index, sku in enumerate(catalog.sku_id)}
        edges = [[row_of_sku[row.sku_id1], row_of_sku[row.sku_id2]]
                 for row in pairs.itertuples(index=False)
                 if row.label == "1" and row.sku_id1 in row_of_sku
                 and row.sku_id2 in row_of_sku]
        pos = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        row_bc = np.array([gtin_of_sku[sku] for sku in catalog.sku_id], dtype=object)
        merged_pos, _, stats = merged_component_graph(pos, row_bc)
        component_of_gtin = ComponentIndex(merged_pos, row_bc).ids()
        return component_of_gtin, stats

    # ── the carve ─────────────────────────────────────────────────────────

    def partition(self) -> SetupPartition:
        """Compute the stratified component carve over the existing setup."""
        from core.timing import Timing
        from training.folds import normalize_gtin
        timing = Timing("smart_setup.partition")
        catalog, pairs = self._inputs()
        component_of_gtin, edge_stats = self._component_graph(catalog, pairs)
        timing.mark("component_graph")

        gtin_of_sku = {row.sku_id: normalize_gtin(row.gtin)
                       for row in catalog.itertuples(index=False)}
        skus_per_component: dict[int, list[str]] = {}
        for sku, gtin in gtin_of_sku.items():
            skus_per_component.setdefault(component_of_gtin[gtin], []).append(sku)
        self._tag_components(skus_per_component, component_of_gtin, gtin_of_sku, pairs)
        split_of_sku = self._assign_components(skus_per_component)
        timing.mark("stratified_allocation")

        assignments = pd.DataFrame({"sku_id": catalog.sku_id,
                                    "split": [split_of_sku[sku] for sku in catalog.sku_id]})
        repartitioned, excluded = self._resplit_pairs(pairs, split_of_sku)
        populations = self._populations(gtin_of_sku, component_of_gtin, split_of_sku)
        census = self._census(assignments, repartitioned, excluded, skus_per_component,
                              component_of_gtin, gtin_of_sku, edge_stats)
        timing.mark("census")
        return SetupPartition(assignments=assignments, pairs=repartitioned,
                              populations=populations, census=census)

    def _tag_components(self, skus_per_component: dict[int, list[str]],
                        component_of_gtin: dict[str, int], gtin_of_sku: dict[str, str],
                        pairs: pd.DataFrame) -> None:
        """The (difficulty, attribute) stratum tag per component (voted by members)."""
        attribute_tags = self._attribute_tags()
        difficulty_votes = self._difficulty_votes()
        difficulties: dict[int, Counter] = {component: Counter()
                                            for component in skus_per_component}
        for row in pairs.itertuples(index=False):
            vote = difficulty_votes.get(tuple(sorted((str(row.sku_id1), str(row.sku_id2)))))
            if not vote:
                continue
            for sku in (row.sku_id1, row.sku_id2):
                gtin = gtin_of_sku.get(sku)
                if gtin is not None:
                    difficulties[component_of_gtin[gtin]][vote] += 1
        self._component_tags = {
            component: (self._modal(difficulties[component], UNMEASURED_DIFFICULTY),
                        self._modal(Counter(attribute_tags.get(sku, NO_ATTRIBUTE)
                                            for sku in skus), NO_ATTRIBUTE))
            for component, skus in skus_per_component.items()}

    def _assign_components(self, skus_per_component: dict[int, list[str]]) -> dict[str, str]:
        """Allocate the components through the canonical SmartSplit; sku -> split."""
        items = [{
            "component": component,
            "stratum": "|".join(self._component_tags[component]),
            "weight": len(skus),
        } for component, skus in sorted(skus_per_component.items())]
        allocated = self._split.allocate_stratified(
            items, stratum_of=lambda item: item["stratum"],
            weight_of=lambda item: item["weight"])
        split_of_component = {item["component"]: name
                              for name, members in allocated.items() for item in members}
        return {sku: split_of_component[component]
                for component, skus in skus_per_component.items() for sku in skus}

    @staticmethod
    def _modal(counter: Counter, default: str) -> str:
        """The deterministic mode (ties break on the sorted value)."""
        return max(sorted(counter), key=counter.get) if counter else default

    def _resplit_pairs(self, pairs: pd.DataFrame,
                       split_of_sku: dict[str, str]) -> tuple[pd.DataFrame, int]:
        """Re-carry the pair split; a cross-half pair is excluded and counted.

        A POSITIVE pair can never cross (both endpoints sit in one component by
        construction) — if one does, the carve is broken and this raises rather
        than silently leaking. Negatives are similarity relations, not identity
        claims: a cross-half negative is dropped exactly like the producer's
        ``cross_split_negative`` policy, never relabeled.

        ``example_id`` marks REPEATED training relationships (the source carries
        the same SKU pair many times with distinct ids, all on train), so the
        multiplicity is meaningful only on train. A pair that leaves train is one
        held-out relationship: its id is cleared (``SupervisedPairs`` rejects a
        non-empty id off train) AND the repeats collapse to a single row — kept,
        they would hand the scanner identical duplicates and it would raise
        "duplicate or conflicting pair". Lineage and the difficulty vote both key
        on the SKU pair, never on ``example_id``, so nothing is lost.
        """
        has_example_id = "example_id" in pairs.columns
        kept, excluded = [], 0
        seen_off_train: set[tuple[tuple[str, str], str]] = set()
        for row in pairs.itertuples(index=False):
            left = split_of_sku[row.sku_id1]
            right = split_of_sku[row.sku_id2]
            if left != right:
                if row.label == "1":
                    raise ValueError(
                        f"positive pair {row.sku_id1}/{row.sku_id2} crosses the "
                        "component carve — the split would leak")
                excluded += 1
                continue
            record = row._asdict()
            record["split"] = left
            if left != "train":
                key = (tuple(sorted((str(row.sku_id1), str(row.sku_id2)))), str(row.label))
                if key in seen_off_train:
                    continue
                seen_off_train.add(key)
                if has_example_id:
                    record["example_id"] = ""
            kept.append(record)
        return pd.DataFrame(kept, columns=list(pairs.columns)), excluded

    def _populations(self, gtin_of_sku: dict[str, str], component_of_gtin: dict[str, int],
                     split_of_sku: dict[str, str]) -> dict[str, list[str]]:
        """split -> sorted normalized GTINs (the entity-level carve).

        This is the SAME populations the text lane's frozen holdout must adopt to
        stay one shared split with the graph lane; it is published verbatim so a
        sibling owner re-freezes the text bundle from it rather than re-deriving
        a divergent carve.
        """
        split_of_gtin = {gtin: split_of_sku[sku] for sku, gtin in gtin_of_sku.items()}
        populations: dict[str, list[str]] = {name: [] for name in self._split.splits}
        for gtin, split in split_of_gtin.items():
            populations[split].append(gtin)
        return {name: sorted(values) for name, values in populations.items()}

    def _census(self, assignments: pd.DataFrame, pairs: pd.DataFrame, excluded: int,
                skus_per_component: dict[int, list[str]], component_of_gtin: dict[str, int],
                gtin_of_sku: dict[str, str], edge_stats: dict) -> dict[str, Any]:
        """The published census: per-split counts + per-axis share preservation."""
        from core.common import portable_path
        split_of_sku = dict(zip(assignments.sku_id, assignments.split))
        per_split: dict[str, Counter] = {
            name: Counter({"listings": 0, "gtins": 0, "components": 0,
                           "positive_pairs": 0, "negative_pairs": 0})
            for name in self._split.splits}
        gtins_seen: dict[str, set[str]] = {name: set() for name in self._split.splits}
        components_seen: dict[str, set[int]] = {name: set() for name in self._split.splits}
        for sku, split in split_of_sku.items():
            per_split[split]["listings"] += 1
            gtins_seen[split].add(gtin_of_sku[sku])
            components_seen[split].add(component_of_gtin[gtin_of_sku[sku]])
        for name in self._split.splits:
            per_split[name]["gtins"] = len(gtins_seen[name])
            per_split[name]["components"] = len(components_seen[name])
        for row in pairs.itertuples(index=False):
            per_split[row.split]["positive_pairs" if row.label == "1" else "negative_pairs"] += 1
        # Axis shares are the COMPONENT tags the carve stratified on, carried by
        # each split's listings: share(axis value) per split vs the population.
        listing_tag: dict[str, tuple[str, str]] = {}
        for component, skus in skus_per_component.items():
            for sku in skus:
                listing_tag[sku] = self._component_tags[component]
        axis_index = {ATTRIBUTE_AXIS: 1, DIFFICULTY_AXIS: 0}
        axis_shares: dict[str, dict[str, dict[str, float]]] = {}
        for axis, position in sorted(axis_index.items()):
            axis_shares[axis] = {}
            for name in self._split.splits:
                tags = Counter(listing_tag[sku][position] for sku, split
                               in split_of_sku.items() if split == name)
                total = max(sum(tags.values()), 1)
                axis_shares[axis][name] = {value: round(count / total, 6)
                                           for value, count in sorted(tags.items())}
        return {
            "schema": CENSUS_SCHEMA,
            "source_setup": portable_path(self._setup),
            "ratios": dict(self._split.spec.ratios),
            "seed": self._split.spec.seed,
            "strata": list(self._split.spec.strata),
            "components": len(skus_per_component),
            "merged_graph": {key: int(value) for key, value in edge_stats.items()},
            "per_split": {name: dict(per_split[name]) for name in self._split.splits},
            "excluded_cross_split_negatives": int(excluded),
            "axis_shares": axis_shares,
            "text_lane": ("holdout NOT re-bundled here (owner: never re-bundle); the "
                          f"gtin populations are published as {POPULATIONS_NAME} for the "
                          "text lane to adopt the SAME carve before a coordinated run"),
        }

    # ── publication ───────────────────────────────────────────────────────

    def publish(self, output: Path, *, training_tensors: bool = True) -> Path:
        """Write the partitioned graph-lane setup tree; return its suite.yaml."""
        from core.timing import Timing
        timing = Timing("smart_setup.publish")
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError(output)
        partition = self.partition()
        output.mkdir(parents=True)
        catalog, _ = self._inputs()
        from graph_tracks.setup import write_setup_frames
        write_setup_frames(output, catalog=catalog, splits=partition.assignments,
                           pairs=partition.pairs)
        self._write_pair_lineage(output, partition)
        self._link_unchanged(output)
        timing.mark("frames_and_links")
        self._prepare_dir(output, training_tensors=training_tensors)
        timing.mark("prepared")
        self._write_lane_configs(output)
        self._write_documents(output, partition)
        timing.mark("configs_and_documents")
        _LOG.info(f"[smart-setup] published {output}")
        return output / "suite.yaml"

    def _write_pair_lineage(self, output: Path, partition: SetupPartition) -> None:
        """Re-carry the pair lineage onto the re-split pairs (or skip, recorded)."""
        from graph_tracks.data import file_size
        from graph_tracks.train import write_json
        layout = _setup_layout()
        source = self._setup / layout.pair_lineage
        if not source.is_file():
            _LOG.info("[smart-setup] source setup carries no pair lineage; none published")
            return
        document = json.loads(source.read_text())
        split_of_pair = {(row.sku_id1, row.sku_id2): row.split
                         for row in partition.pairs.itertuples(index=False)}
        document["pairs"] = [
            {**record, "split": split_of_pair[(record["sku_id1"], record["sku_id2"])]}
            for record in document.get("pairs", [])
            if (record["sku_id1"], record["sku_id2"]) in split_of_pair]
        document["listing_pairs_size"] = file_size(output / layout.pairs)
        write_json(output / layout.pair_lineage, document)

    def _link_unchanged(self, output: Path) -> None:
        """Hardlink every split-independent member (fall back to a copy).

        The frozen text bundle and its sidecar pass through UNCHANGED (owner:
        never re-bundle); the text lane adopts the carve from the published
        populations, not from a rewrite here.
        """
        layout = _setup_layout()
        regenerated = {layout.catalog, layout.splits, layout.pairs, layout.pair_lineage,
                       layout.manifest, layout.census, layout.text_config,
                       layout.track_config("gnn_only"), layout.track_config("cascade"),
                       "suite.yaml", POPULATIONS_NAME}
        for path in sorted(self._setup.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self._setup)
            if relative.parts[0] == layout.prepared_dir or str(relative) in regenerated:
                continue
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError:
                import shutil
                shutil.copy2(path, target)

    def _prepare_dir(self, output: Path, *, training_tensors: bool) -> None:
        """Regenerate the prepared package through the canonical stage."""
        from graph_tracks.prepare import prepare
        layout = _setup_layout()
        prepare(output / layout.catalog, output / layout.splits, output / layout.pairs,
                output / layout.prepared_dir, training_tensors=training_tensors)

    def _write_lane_configs(self, output: Path) -> None:
        """Re-render the lane configs onto the new tree (the ONE rebaser)."""
        from core.common import TRAIN_ROOT
        from graph_tracks.config import load_config as load_graph_config
        from graph_tracks.config import load_text_config
        from graph_tracks.setup import write_text_config, write_track_config
        from model_tracks.smoke_inputs import repoint_lane_paths
        layout = _setup_layout()
        train_root = Path(TRAIN_ROOT).resolve()
        for track in ("gnn_only", "cascade"):
            settings = load_graph_config(self._setup / layout.track_config(track),
                                         expected_track=track).model_dump()
            settings = repoint_lane_paths(settings, setup=self._setup, output=output,
                                          train_root=train_root)
            write_track_config(output, track, settings)
        write_text_config(output,
                          load_text_config(self._setup / layout.text_config).model_dump())

    def _write_documents(self, output: Path, partition: SetupPartition) -> None:
        """The setup manifest, the census, the populations and the suite.yaml."""
        import yaml

        from core.bundle import bundle_spec
        from core.common import portable_path
        from graph_tracks.train import write_json
        from model_tracks.config import load_config
        layout = _setup_layout()
        manifest = json.loads((self._setup / layout.manifest).read_text())
        pair_counts: dict[str, dict[str, int]] = {}
        for row in partition.pairs.itertuples(index=False):
            counts = pair_counts.setdefault(row.split, {})
            counts[row.label] = counts.get(row.label, 0) + 1
        manifest.update(
            pair_counts=pair_counts,
            split_protocol=("core.smart_split.SmartSplit.allocate_stratified "
                            "(component grain, weighted by listings)"),
            smart_partition=partition.census)
        write_json(output / layout.manifest, manifest)
        census_path = self._setup / layout.census
        if census_path.is_file():
            census = json.loads(census_path.read_text())
            census["splits"] = {name: int((partition.assignments.split == name).sum())
                                for name in self._split.splits}
            write_json(output / layout.census, census)
        write_json(output / POPULATIONS_NAME, partition.populations)
        parent = load_config(self._suite_config or Path(bundle_spec().suite_config))
        settings = parent.model_dump()
        settings["setup_dir"] = portable_path(output)
        bundle = output / "text_prepared.pkl.gz"
        if bundle.is_file():
            settings["text_bundle"] = portable_path(bundle)
        (output / "suite.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("setup_dir", type=Path, help="the existing prepared setup")
    parser.add_argument("output_dir", type=Path,
                        help="the new partitioned setup (must not exist)")
    parser.add_argument("--suite-config", type=Path, default=None,
                        help="the parent suite config (default: the bundle SSOT suite_config)")
    parser.add_argument("--no-training-tensors", action="store_true",
                        help="skip the graph tensor stage (partition frames only)")
    args = parser.parse_args(argv)
    RunLogger.configure_console()
    suite_path = SmartSetupPartition.from_config(
        args.setup_dir, suite_config=args.suite_config).publish(
            args.output_dir, training_tensors=not args.no_training_tensors)
    manifest = json.loads((args.output_dir / _setup_layout().manifest).read_text())
    print(json.dumps({"suite_yaml": str(suite_path),
                      "census": manifest["smart_partition"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
