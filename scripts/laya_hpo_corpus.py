"""scripts/laya_hpo_corpus.py — carve the laya HPO 50/50 corpus (ZERO leakage).

Owner directive (2026-10-10): the just-stopped HPO trained the FULL corpus
against a LEAKY dev eval (``overlapping_eval``, ``is_held_out=false``, "24/27305
items overlap training"). This carves ``data/laya_hpo5050`` = 50% train / 50%
validate from the EXISTING laya corpus (``data/laya/{train,dev,test}.jsonl`` —
SAMPLED, never regenerated: the SAME records, re-partitioned), at the
COMPONENT/ENTITY grain so ZERO endpoint / component / pair crosses the halves,
STRATIFIED so every attribute/difficulty proportion is preserved.

The component-aware stratified carve is the ONE canonical smart-split owner
(:meth:`core.smart_split.SmartSplit.allocate_components`) driven by the declared
profile ``laya_hpo_5050`` (``config/smart_split.yaml``); no ratio/stratum/seed
is a code literal.

ENTITY = the product SIDE SIGNATURE (the six slice fields ``compose_state``
renders — the exact product text the model sees), never the GTIN. A GTIN
union-find is provably the WRONG grain here: one side signature is shared by
many catalog GTINs, so endpoint sets explode (measured: a single GTIN component
absorbs 94.8% of the corpus -> an unsplittable 50/50). The side signature is
the leakage-honest, splittable entity — two records that expose the same product
text are unioned (within a pair, and across any records sharing a side), so a
component never straddles the carve and the model never sees one product's text
in both halves. A record with NO measured slice field (an all-empty signature)
falls back to its full text as a singleton entity, so 1,214 field-less single
states never collapse into one false hub.

Outputs (``data/laya_hpo5050/``): ``train.jsonl`` + ``dev.jsonl`` (the 50/50
halves) and an EMPTY ``test.jsonl`` (the 50/50 carve has no test role, so the
kernel's session-end held-out eval records a reasoned skip), plus
``receipt.json`` (the per-attribute/difficulty proportions + the zero-leakage
census). Every record rides its derived ``component`` + ``endpoints`` tags (the
laya trainer reads only state/questions/expected, so the tags never train).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from core.disjoint_sets import DisjointSet
from core.smart_split import SmartSplit, SmartSplitProfile, smart_split_profile
from scripts.laya_corpus_composer import PAIR_FIELDS, compose_side
from scripts.laya_corpus_io import Digest, JsonLine

#: The three physical split files the fine-tune/HPO kernel attaches. The 50/50
#: carve populates train/dev and writes test EMPTY (its no-test-role contract).
SPLIT_FILES = ("train.jsonl", "dev.jsonl", "test.jsonl")
#: The all-empty six-field signature (a product with no measured slice field).
_EMPTY_SIDE = tuple("" for _ in PAIR_FIELDS)


def _side_key(side: dict[str, str]) -> tuple[str, ...]:
    """A side dict -> its order-stable six-field signature (the entity text)."""
    return tuple(side.get(field, "") for field in PAIR_FIELDS)


def _parse_sides(state: str) -> tuple[dict[str, str], dict[str, str]] | None:
    """Invert ``compose_state``: the joined pair state -> its two side dicts.

    The state is ``"<field>: v1=<A> v2=<B>; ..."``; a side value is ``""`` or a
    bracketed list, so the first ``" v2="`` splits each field unambiguously.
    ``None`` when the state is not in the pair shape (a ``better_match``
    triple), which the caller routes to its full-text singleton entity.
    """
    side_one: dict[str, str] = {}
    side_two: dict[str, str] = {}
    for segment in state.split("; "):
        if ": " not in segment:
            return None
        field, rest = segment.split(": ", 1)
        if not rest.startswith("v1=") or " v2=" not in rest:
            return None
        first, second = rest.split(" v2=", 1)
        side_one[field] = first[3:]
        side_two[field] = second
    return side_one, side_two


class RecordEntity:
    """The leakage entity of one corpus record: its product side signature(s).

    One job: map a record to the atomic text-unit(s) the model sees, so records
    exposing the same product text union into one component. A pair contributes
    BOTH sides (a record is atomic, so its two products co-locate); a single
    state contributes its one side; an all-empty signature (no measured slice
    field) or a non-pair shape falls back to the record's full text as a
    singleton entity, so field-less records never collapse into one false hub.
    """

    #: The entity-string prefixes (a side signature vs a full-text singleton).
    SIDE_PREFIX = "side:"
    RAW_PREFIX = "raw:"

    @classmethod
    def _side_entity(cls, side: dict[str, str]) -> str:
        return cls.SIDE_PREFIX + "|".join(_side_key(side))

    @classmethod
    def of(cls, record: dict[str, Any]) -> list[str]:
        """The record's entity keys (>= 1); the first is its component anchor."""
        state = record["state"]
        raw = cls.RAW_PREFIX + state
        if record.get("difficulty_slice") == "single_state":
            key = _side_key(compose_side(state))
            return [raw if key == _EMPTY_SIDE else cls.SIDE_PREFIX + "|".join(key)]
        sides = _parse_sides(state)
        if sides is None:
            return [raw]
        entities = [cls._side_entity(side) for side in sides
                    if _side_key(side) != _EMPTY_SIDE]
        return entities or [raw]


class HpoCorpusCarver:
    """Carve the 50/50 zero-leakage HPO corpus from the existing laya corpus.

    One job per method: read the existing records, index their entities into
    components, tag each record, deal the components through the canonical
    smart-split profile, then write the halves + the receipt. The records are
    SAMPLED (re-emitted verbatim plus the entity tags), never regenerated.
    """

    def __init__(self, *, source_dir: Path, output_dir: Path,
                 profile: SmartSplitProfile) -> None:
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.profile = profile

    # ── read + entity index ────────────────────────────────────────────────
    def _read_source(self) -> tuple[list[dict], dict[str, int]]:
        records: list[dict] = []
        sizes: dict[str, int] = {}
        for name in SPLIT_FILES:
            path = self.source_dir / name
            rows = ([json.loads(line)
                     for line in path.read_text(encoding="utf-8").splitlines()
                     if line.strip()] if path.is_file() else [])
            sizes[name.removesuffix(".jsonl")] = len(rows)
            records.extend(rows)
        if not records:
            raise RuntimeError(
                f"no corpus records under {self.source_dir}; build the laya "
                "corpus (scripts/laya_build_dataset.py) before the HPO carve")
        return records, sizes

    def _tag_components(self, records: list[dict]) -> int:
        """Attach ``component`` + ``endpoints`` tags; return the component count.

        Entities are unioned within each record (a pair's two products
        co-locate) and across records sharing an entity, then grouped by
        MEMBERSHIP (the ``DisjointSet`` contract) so the ids are independent of
        union order. The largest component is recorded: it caps how evenly the
        halves can balance (a component is atomic), so it is surfaced, not hid.
        """
        entities = [RecordEntity.of(record) for record in records]
        graph = DisjointSet()
        for group in entities:
            for entity in group:
                graph.add(entity)
            for other in group[1:]:
                graph.union(group[0], other)
        ordered = graph.components()
        entity_component = {entity: f"c{index}"
                            for index, members in enumerate(ordered)
                            for entity in members}
        for record, group in zip(records, entities):
            record["component"] = entity_component[group[0]]
            record["endpoints"] = sorted(set(group))
        self._component_sizes = Counter(record["component"]
                                        for record in records)
        return len(ordered)

    def _stratum_of(self, record: dict) -> str:
        return "|".join(str(record.get(axis, ""))
                        for axis in self.profile.strata)

    # ── the carve ──────────────────────────────────────────────────────────
    def run(self) -> dict[str, Any]:
        records, source_sizes = self._read_source()
        components = self._tag_components(records)
        # The ONE canonical smart-split owner (component-aware + stratified).
        halves = SmartSplit.allocate_components(
            records, lambda r: r["component"], self._stratum_of,
            self.profile.ratios, self.profile.seed)
        return self._emit(records, halves, source_sizes, components)

    def _emit(self, records: list[dict], halves: dict[str, list],
              source_sizes: dict[str, int], components: int) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name in SPLIT_FILES:
            split = name.removesuffix(".jsonl")
            path = self.output_dir / name
            rows = halves.get(split, [])
            with path.open("w", encoding="utf-8") as handle:
                for record in rows:
                    handle.write(JsonLine.dump(record) + "\n")
            written[split] = path
        return self._receipt(records, halves, written, source_sizes, components)

    # ── census + receipt ───────────────────────────────────────────────────
    def _axis_proportions(self, halves: dict[str, list], axis: str) -> dict:
        names = list(halves)
        totals: Counter = Counter()
        per_half: dict[str, Counter] = {name: Counter() for name in names}
        for name in names:
            for record in halves[name]:
                value = str(record.get(axis, ""))
                per_half[name][value] += 1
                totals[value] += 1
        return {
            value: {
                "total": totals[value],
                **{name: (per_half[name][value] / totals[value]
                          if totals[value] else 0.0) for name in names},
            }
            for value in sorted(totals)
        }

    @staticmethod
    def _leakage_census(halves: dict[str, list]) -> dict[str, int]:
        """The zero-leakage proof: no component / endpoint / pair in two halves.

        Recorded for auditability (never a gate — the component-grain deal is
        zero-leakage by construction); the scoped test pins the same property.
        """
        names = list(halves)
        shared = {"components": 0, "endpoints": 0, "pairs": 0}
        for axis, key_of in (
            ("components", lambda r: [r["component"]]),
            ("endpoints", lambda r: r["endpoints"]),
            ("pairs", lambda r: [r["state"]]),
        ):
            seen: dict[str, str] = {}
            for name in names:
                for record in halves[name]:
                    for key in key_of(record):
                        if key in seen and seen[key] != name:
                            shared[axis] += 1
                        seen.setdefault(key, name)
        return shared

    def _receipt(self, records: list[dict], halves: dict[str, list],
                 written: dict[str, Path], source_sizes: dict[str, int],
                 components: int) -> dict[str, Any]:
        largest = max(self._component_sizes.values())
        receipt = {
            "carve": "laya_hpo_5050",
            "entity": "product side signature (six slice fields); GTIN over-"
                      "merges (one side text <-> many GTINs) so it is not the "
                      "leakage grain",
            "output_dir": str(self.output_dir),
            "profile": {"ratios": dict(self.profile.ratios),
                        "strata": list(self.profile.strata),
                        "seed": self.profile.seed},
            "source_dir": str(self.source_dir),
            "source_split_sizes": source_sizes,
            "split_sizes": {name: len(rows) for name, rows in halves.items()},
            "records": len(records),
            "components": components,
            "largest_component": largest,
            "largest_component_pct": round(100.0 * largest / len(records), 4),
            "leakage": self._leakage_census(halves),
            "attribute_proportions": self._axis_proportions(halves, "attribute"),
            "difficulty_proportions":
                self._axis_proportions(halves, "difficulty_slice"),
            "sha256": {name: Digest.sha256(path)
                       for name, path in written.items()},
        }
        corpus_digest = hashlib.sha256()
        for name in SPLIT_FILES:
            split = name.removesuffix(".jsonl")
            if split in written:
                corpus_digest.update(written[split].read_bytes())
        receipt["corpus_sha256"] = corpus_digest.hexdigest()
        path = self.output_dir / "receipt.json"
        path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        return receipt


class HpoCorpusFactory:
    """Resolve the carve's config SSOT and build the carver (factory paradigm).

    The output dir + profile name ride the HPO space YAML's ``corpus`` block
    (``config/laya_hpo_space.yaml``); the source corpus is the LayaSpec
    ``finetune_corpus_dir``; the ratios/strata/seed are the named smart-split
    profile (``config/smart_split.yaml``). No path or fraction is a literal.
    """

    #: The carve profile when the space YAML does not name one.
    DEFAULT_PROFILE = "laya_hpo_5050"

    @classmethod
    def resolve(cls, *, profile_name: str | None = None,
                source_dir: Path | None = None,
                output_dir: Path | None = None) -> HpoCorpusCarver:
        from cli.laya_hpo_space import load_space
        from core.common import TRAIN_ROOT, training_cfg

        corpus = load_space().get("corpus") or {}
        name = (profile_name or corpus.get("split_profile")
                or cls.DEFAULT_PROFILE)
        resolved_output = output_dir or (
            TRAIN_ROOT / str(corpus.get("dir") or "data/laya_hpo5050"))
        resolved_source = source_dir or (
            TRAIN_ROOT / str(training_cfg().laya.finetune_corpus_dir))
        return HpoCorpusCarver(
            source_dir=resolved_source, output_dir=resolved_output,
            profile=smart_split_profile(name))

    @classmethod
    def build(cls, *, profile_name: str | None = None,
              source_dir: Path | None = None,
              output_dir: Path | None = None) -> dict[str, Any]:
        return cls.resolve(profile_name=profile_name, source_dir=source_dir,
                           output_dir=output_dir).run()


def main() -> None:
    import time

    started = time.perf_counter()
    receipt = HpoCorpusFactory.build()
    elapsed = time.perf_counter() - started
    print("[laya-hpo-corpus] carve=%s ratios=%s strata=%s seed=%d"
          % (receipt["carve"], json.dumps(receipt["profile"]["ratios"]),
             receipt["profile"]["strata"], receipt["profile"]["seed"]))
    print("[laya-hpo-corpus] source_split_sizes=%s -> split_sizes=%s records=%d"
          % (json.dumps(receipt["source_split_sizes"]),
             json.dumps(receipt["split_sizes"]), receipt["records"]))
    print("[laya-hpo-corpus] components=%d largest=%d (%.2f%% of the corpus; "
          "an atomic component caps the balance)"
          % (receipt["components"], receipt["largest_component"],
             receipt["largest_component_pct"]))
    print("[laya-hpo-corpus] leakage=%s (ZERO required)"
          % json.dumps(receipt["leakage"]))
    print("[laya-hpo-corpus] attribute_proportions=%s"
          % json.dumps(receipt["attribute_proportions"]))
    print("[laya-hpo-corpus] difficulty_proportions=%s"
          % json.dumps(receipt["difficulty_proportions"]))
    for name, digest in receipt["sha256"].items():
        print(f"[laya-hpo-corpus] sha256 {name} {digest}")
    print("[laya-hpo-corpus] corpus_sha256=%s" % receipt["corpus_sha256"])
    print("[laya-hpo-corpus] built in %.1fs -> %s"
          % (elapsed, Path(receipt["output_dir"]) / "receipt.json"))


if __name__ == "__main__":
    main()
