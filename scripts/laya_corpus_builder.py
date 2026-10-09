"""scripts/laya_corpus_builder.py — build the laya FINE-TUNE corpus (JSONL).

Owner order: "we will finetune laya with the correct dataset". The laya
trainer (`laya-train`, `laya.train.read_data`/`read_jsonl`/`items_from_rows`)
consumes ONE JSON case per line:

    {"state": <str>,
     "questions": {<config/laya.question.json "questions" verbatim>},
     "expected": {<qid>: <label>}}

Sources (read, never invented):
  * data/track_setup/eligible_catalog.csv — `attribute` = the standardized
    state string (one STATE case per row);
  * data/track_setup/listing_pairs.csv — the labelled pairs joined to the
    catalog and composed side-by-side exactly like scripts/laya_metrics_pairs.py
    (reused, not copied);
  * data/gate_results.csv — the verified DIFFERENT population: the `hard_no`
    rows are identity negatives; the `fallback` rows are UNKNOWN and are
    quarantined to data/laya/unknown_pairs.csv, never in the corpus;
  * data/final_validation.csv — mirrored only through
    scripts/laya_metrics_pairs.py's composed `attribute_pairs` shape.

Growth sources (owner order "laya is overfitting" 2026-10-08) are folded in by
`GrowthFoldIngestor`; both are OPT-IN so the hermetic builder tests are
unchanged.

Config (config/laya.question.json, optional top-level "corpus" block — the
config SSOT for this builder; every key defaults to the historical behaviour,
so a schema without the block reproduces the landed corpus byte-for-byte):
seed, hard_no_cap, identity_negative_target_ratio, sources.

Outputs (data/laya/): train.jsonl, dev.jsonl, test.jsonl, unknown_pairs.csv,
receipt.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from core.laya_config import LayaSplitRoles
from scripts.laya_corpus_cases import CorpusCaseRenderer
from scripts.laya_corpus_composer import compose_side, compose_state
from scripts.laya_corpus_growth import GrowthFoldIngestor
from scripts.laya_corpus_io import CsvSource, Digest, JsonLine
from scripts.laya_corpus_rules import (
    PACKAGE_STATE_RULE,
    GateReasonRules,
    PackageStateRules,
    PairLabelRules,
)

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data/track_setup/eligible_catalog.csv"
PAIRS_PATH = ROOT / "data/track_setup/listing_pairs.csv"
GATE_PATH = ROOT / "data/gate_results.csv"
QUESTION_PATH = ROOT / "config/laya.question.json"
OUTPUT_DIR = ROOT / "data/laya"
# The pipeline's minted masking/augmentation live in the prepared text
# bundle. `main()` passes these when present; `build()` defaults to None so
# hermetic callers (the builder tests) keep the pre-growth corpus exactly.
BUNDLE_PATH = ROOT / "data/prepared/full/worker_1_baseline.pkl.gz"
LABELED_PAIRS_PATH = ROOT / "data/labeled_pairs.csv"

SEED = 1729
# Balanced identity negatives: total negatives target the positive count,
# capped here (the "up to ~1000" ceiling), then the ground-truth listing
# negatives are subtracted to get the hard_no sample size.
HARD_NO_CAP = 1000
SPLIT_ORDER = ("train", "dev", "test")

# ── the corpus-composition config (config-owned; every knob defaults to the
#    historical behaviour) ───────────────────────────────────────────────────
CORPUS_CONFIG_DEFAULTS: dict = {
    "seed": None,
    "hard_no_cap": None,
    "identity_negative_target_ratio": None,
    "sources": None,
}
CORPUS_SOURCE_KEYS = ("catalog", "pairs", "gate", "bundle", "labeled_pairs",
                      "output_dir")
CORPUS_SOURCE_DEFAULTS = {
    "catalog": CATALOG_PATH,
    "pairs": PAIRS_PATH,
    "gate": GATE_PATH,
    "bundle": BUNDLE_PATH,
    "labeled_pairs": LABELED_PAIRS_PATH,
    "output_dir": OUTPUT_DIR,
}


class CorpusConfig:
    """Resolve the `corpus` composition block from the question-schema file."""

    @staticmethod
    def from_document(document: dict, override: dict | None = None) -> dict:
        """Resolve the `corpus` composition block from the question-schema file.

        `override` (when given) replaces the file's block wholesale — the
        explicit caller wins over config SSOT. Unknown keys fail loud (a typo
        must never silently fall back to a default), a null value means "keep
        the default", and a config with no block resolves to
        `CORPUS_CONFIG_DEFAULTS` exactly.
        """
        config = dict(CORPUS_CONFIG_DEFAULTS)
        block = override if override is not None else (document.get("corpus") or {})
        if not isinstance(block, dict):
            raise ValueError(
                f"laya corpus config must be a mapping, got {type(block).__name__}")
        unknown = sorted(set(block) - set(config))
        if unknown:
            raise ValueError(f"unknown laya corpus knob(s): {unknown}")
        for key, value in block.items():
            if value is not None:
                config[key] = value
        if isinstance(config["sources"], dict):
            unknown = sorted(set(config["sources"]) - set(CORPUS_SOURCE_KEYS))
            if unknown:
                raise ValueError(f"unknown laya corpus source(s): {unknown}")
        return config

    @staticmethod
    def resolve_sources(config: dict) -> dict:
        """The builder's input paths: the hardcoded defaults + config overrides.

        A config path is resolved against the repo root (`ROOT`), never the
        cwd, so the same config resolves identically from any working directory.
        """
        sources = dict(CORPUS_SOURCE_DEFAULTS)
        for key, value in (config.get("sources") or {}).items():
            if value is None:
                continue
            candidate = Path(value)
            sources[key] = (candidate if candidate.is_absolute()
                            else ROOT / candidate)
        return sources

    @staticmethod
    def is_default(config: dict) -> bool:
        """True when NO knob (nor any source path) is set — the null block case."""
        if any(config.get(key) is not None for key in
               ("seed", "hard_no_cap", "identity_negative_target_ratio")):
            return False
        return not any(value is not None
                       for value in (config.get("sources") or {}).values())


class SplitAllocator:
    """Deterministic corpus split allocation and negative rebalancing."""

    @staticmethod
    def _rebalance_identity_negatives(
        aug_records: list[dict], origins: dict[str, str], *,
        positives_total: int, ground_negative_total: int, ratio: float | None,
        seed: int,
    ) -> tuple[list[dict], dict]:
        """Thin the pipeline-MINTED identity negatives toward `ratio` x positives.

        `ratio is None` (the default) is a no-op, so the landed corpus is
        reproduced byte-for-byte. When set, only records the growth fold MINTED
        (origin `bundle`: counterfactual/twin/swap augmentation) may be dropped;
        ground-truth negatives (listing pairs, gate `hard_no`, labeled pairs) are
        never touched. The drop is deterministic (`seed` over a sorted state
        list), so a rerun reproduces every byte, and a target already satisfied
        drops nothing.
        """
        if ratio is None:
            return aug_records, {"enabled": False}
        if ratio < 0:
            raise ValueError(
                f"identity_negative_target_ratio must be >= 0, got {ratio!r}")
        minted = sorted(
            record["state"] for record in aug_records
            if origins.get(record["state"]) == "bundle"
            and record["expected"].get("identity_claim") == "false")
        target_negatives = int(round(ratio * positives_total))
        keep = max(0, target_negatives - ground_negative_total)
        census = {
            "enabled": True,
            "target_ratio": ratio,
            "positives_total": positives_total,
            "ground_truth_negatives": ground_negative_total,
            "target_negatives": target_negatives,
            "minted_negatives_available": len(minted),
            "minted_negatives_kept": min(len(minted), keep),
            "minted_negatives_dropped": max(0, len(minted) - keep),
        }
        if len(minted) <= keep:
            return aug_records, census
        dropped = set(random.Random(seed).sample(minted, len(minted) - keep))
        kept = [record for record in aug_records
                if record["state"] not in dropped]
        return kept, census

    @staticmethod
    def _allocate(total: int, ratios: dict[str, float]) -> dict[str, int]:
        """Largest-remainder allocation of `total` at `ratios` (sums to total)."""
        raw = {key: total * ratios[key] for key in ratios}
        allocation = {key: int(value) for key, value in raw.items()}
        remainder = total - sum(allocation.values())
        for key in sorted(ratios, key=lambda k: (-(raw[k] - allocation[k]), k)):
            if remainder <= 0:
                break
            allocation[key] += 1
            remainder -= 1
        return allocation

    @staticmethod
    def _split_ratios(listing_counts: dict[str, int]) -> dict[str, float]:
        """The documented ratios: the listing_pairs split proportions, so the
        state and gate-negative draws land in the same train/dev/test shape."""
        total = sum(listing_counts.get(key, 0) for key in SPLIT_ORDER)
        return {key: listing_counts.get(key, 0) / total for key in SPLIT_ORDER}

    @staticmethod
    def _assign_splits(items: list, ratios: dict[str, float], seed: int,
                       stratum_of: Callable[[dict], str]) -> dict[str, list]:
        """Deterministically shuffle (seed) and slice into the three splits.

        STRATIFIED by ``stratum_of`` (the corpus's documented subgroup key):
        each stratum is allocated independently at the same ratios, so a rare
        stratum is represented proportionally in EVERY split instead of being
        concentrated by one global shuffle. ``_allocate`` still sums each
        stratum to its own total, so every record lands in exactly one split
        and the split sizes are unchanged.
        """
        by_stratum: dict[str, list] = defaultdict(list)
        for item in items:
            by_stratum[stratum_of(item)].append(item)
        out: dict[str, list] = {key: [] for key in SPLIT_ORDER}
        for stratum in sorted(by_stratum):
            members = list(by_stratum[stratum])
            random.Random(f"{seed}:{stratum}").shuffle(members)
            allocation = SplitAllocator._allocate(len(members), ratios)
            cursor = 0
            for key in SPLIT_ORDER:
                out[key].extend(members[cursor:cursor + allocation[key]])
                cursor += allocation[key]
        return out

    @staticmethod
    def _difficulty_stratum(record: dict) -> str:
        """The rendered corpus record's documented difficulty stratum."""
        return record["difficulty_slice"]

    @staticmethod
    def _gate_reason_stratum(row: dict) -> str:
        """A raw gate row's bounded reason family (its key subgroup)."""
        return GateReasonRules.gate_reason_family(row["gate_reason"])

    @staticmethod
    def _stratified_sample(rows: list[dict], target: int,
                           reason_of: Callable[[dict], str],
                           seed: int) -> list[dict]:
        """Proportional-by-stratum sample (largest remainder), deterministic."""
        by_reason: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_reason[reason_of(row)].append(row)
        for members in by_reason.values():
            members.sort(key=lambda r: (r["gtin1"], r["gtin2"]))
        ratios = {reason: len(members) / len(rows)
                  for reason, members in by_reason.items()}
        allocation = SplitAllocator._allocate(min(target, len(rows)), ratios)
        rng = random.Random(seed)
        sampled: list[dict] = []
        for reason in sorted(by_reason):
            count = min(allocation[reason], len(by_reason[reason]))
            sampled.extend(rng.sample(by_reason[reason], count))
        return sampled


class CorpusBuilder:
    """Builds the corpus from resolved sources and a resolved config."""

    def __init__(self, *, catalog_path: Path, pairs_path: Path, gate_path: Path,
                 question_path: Path, output_dir: Path,
                 explicit_seed: int | None, explicit_hard_no_cap: int | None,
                 explicit_ratio: float | None, bundle_path: Path | None,
                 labeled_pairs_path: Path | None,
                 corpus_config: dict | None) -> None:
        self.catalog_path = catalog_path
        self.pairs_path = pairs_path
        self.gate_path = gate_path
        self.question_path = question_path
        self.output_dir = output_dir
        self.explicit_seed = explicit_seed
        self.explicit_hard_no_cap = explicit_hard_no_cap
        self.explicit_ratio = explicit_ratio
        self.bundle_path = bundle_path
        self.labeled_pairs_path = labeled_pairs_path
        self.corpus_config = corpus_config

    @staticmethod
    def build(*, catalog_path: Path = CATALOG_PATH,
              pairs_path: Path = PAIRS_PATH, gate_path: Path = GATE_PATH,
              question_path: Path = QUESTION_PATH,
              output_dir: Path = OUTPUT_DIR, seed: int | None = None,
              hard_no_cap: int | None = None, bundle_path: Path | None = None,
              labeled_pairs_path: Path | None = None,
              identity_negative_target_ratio: float | None = None,
              corpus_config: dict | None = None) -> dict:
        """Build the corpus.

        Every composition knob resolves `explicit argument > config/laya.question
        .json "corpus" block > historical default`, so an `int`/`float` argument
        wins over config SSOT and a caller that passes NOTHING (or a schema with
        no block) reproduces the landed corpus byte-for-byte.
        """
        builder = CorpusBuilder(
            catalog_path=Path(catalog_path), pairs_path=Path(pairs_path),
            gate_path=Path(gate_path), question_path=Path(question_path),
            output_dir=Path(output_dir), explicit_seed=seed,
            explicit_hard_no_cap=hard_no_cap, explicit_ratio=
            identity_negative_target_ratio, bundle_path=bundle_path,
            labeled_pairs_path=labeled_pairs_path, corpus_config=corpus_config)
        return builder.run()

    def run(self) -> dict:
        self._load_catalog()
        self._resolve_question_and_config()
        self._load_pairs()
        self._build_state_cases()
        self._build_listing_pair_cases()
        self._load_gate()
        self._build_gate_and_proceed_cases()
        self._fold_growth_and_rebalance()
        self._build_better_match_cases()
        self._emit_splits()
        self._quarantine_fallback()
        self._build_traceability_census()
        return self._write_receipt()

    # ── sources ────────────────────────────────────────────────────────────
    def _load_catalog(self) -> None:
        catalog_header, catalog = CsvSource.read(self.catalog_path)
        required = ("sku_id", "gtin", "attribute")
        missing_cols = [key for key in required if key not in catalog_header]
        if missing_cols:
            raise RuntimeError(
                f"eligible_catalog is missing required columns {missing_cols}")
        by_sku = {row["sku_id"]: row for row in catalog}
        # A gtin may repeat across sku_id rows; pick the smallest sku_id so the
        # join is deterministic (documented; the catalog keeps both spellings).
        by_gtin: dict[str, dict] = {}
        for row in sorted(catalog, key=lambda r: r["sku_id"]):
            by_gtin.setdefault((row["gtin"] or "").strip(), row)
        self.catalog_header = catalog_header
        self.catalog = catalog
        self.by_sku = by_sku
        self.by_gtin = by_gtin

    def _resolve_question_and_config(self) -> None:
        document = json.loads(self.question_path.read_text(encoding="utf-8"))
        self.questions = document["questions"]
        self.question_sha = Digest.sha256(self.question_path)
        # corpus composition: explicit argument > config block > default.
        corpus = CorpusConfig.from_document(document, self.corpus_config)
        self.corpus = corpus
        self.seed = (self.explicit_seed if self.explicit_seed is not None
                     else (corpus["seed"] if corpus["seed"] is not None
                           else SEED))
        self.hard_no_cap = (
            self.explicit_hard_no_cap if self.explicit_hard_no_cap is not None
            else (corpus["hard_no_cap"] if corpus["hard_no_cap"] is not None
                  else HARD_NO_CAP))
        self.identity_negative_target_ratio = (
            self.explicit_ratio if self.explicit_ratio is not None
            else corpus["identity_negative_target_ratio"])

    def _load_pairs(self) -> None:
        pairs_header, pairs = CsvSource.read(self.pairs_path)
        if pairs_header != ["sku_id1", "sku_id2", "label", "split"]:
            raise RuntimeError(f"listing_pairs header drifted: {pairs_header}")
        missing_skus = sorted(
            {sku for pair in pairs
             for sku in (pair["sku_id1"], pair["sku_id2"])
             if sku not in self.by_sku})
        if missing_skus:
            raise RuntimeError(
                f"{len(missing_skus)} listing_pair sku_id(s) resolve to no "
                f"catalog row: {missing_skus[:10]}")
        listing_counts = Counter(pair["split"] for pair in pairs)
        self.pairs = pairs
        self.ratios = SplitAllocator._split_ratios(listing_counts)

    # ── base cases ─────────────────────────────────────────────────────────
    def _build_state_cases(self) -> None:
        state_records = []
        state_pkg = Counter()
        for row in self.catalog:
            attribute = row["attribute"]
            labeled = PackageStateRules.package_state(attribute)
            state_pkg[str(labeled).lower()] += 1
            expected = {"package_state": "true" if labeled else "false"}
            if "evidence_sufficient" in self.questions:
                expected["evidence_sufficient"] = (
                    "true" if PairLabelRules._has_evidence(
                        compose_side(attribute)) else "false")
            state_records.append(CorpusCaseRenderer._record(
                attribute, self.questions, expected,
                **CorpusCaseRenderer._single_meta(attribute)))
        self.state_pkg = state_pkg
        self.state_splits = SplitAllocator._assign_splits(
            state_records, self.ratios, self.seed,
            SplitAllocator._difficulty_stratum)

    def _build_listing_pair_cases(self) -> None:
        pair_by_split: dict[str, list[dict]] = {key: [] for key in SPLIT_ORDER}
        identity_labels = Counter()
        for pair in self.pairs:
            row_one = self.by_sku[pair["sku_id1"]]
            row_two = self.by_sku[pair["sku_id2"]]
            one, two = row_one["attribute"], row_two["attribute"]
            side_one, side_two = compose_side(one), compose_side(two)
            label = "true" if int(pair["label"]) == 1 else "false"
            identity_labels[label] += 1
            expected = CorpusCaseRenderer._pair_expected(
                self.questions, side_one, side_two, attr_one=one, attr_two=two,
                identity=label, brand_one=row_one.get("brand"),
                brand_two=row_two.get("brand"), counterfactual="false")
            pair_by_split[pair["split"]].append(CorpusCaseRenderer._record(
                compose_state(side_one, side_two), self.questions, expected,
                difficulty_slice=PairLabelRules._difficulty_slice(
                    side_one, side_two),
                attribute=PairLabelRules._primary_attribute(side_one, side_two)))
        self.pair_by_split = pair_by_split
        self.positives = identity_labels["true"]
        self.listing_negatives = identity_labels["false"]

    # ── gate cases ─────────────────────────────────────────────────────────
    def _load_gate(self) -> None:
        gate_header, gate = CsvSource.read(self.gate_path)
        hard_no, proceed, fallback = [], [], []
        dropped_missing = Counter()
        for row in gate:
            both = ((row["gtin1"] or "").strip() in self.by_gtin
                    and (row["gtin2"] or "").strip() in self.by_gtin)
            if not both:
                dropped_missing[row["gate_decision"]] += 1
            if row["gate_decision"] == "hard_no":
                if both:
                    hard_no.append(row)
            elif row["gate_decision"] == "proceed":
                if both:
                    proceed.append(row)
            elif row["gate_decision"] == "fallback":
                fallback.append(row)
        self.gate_header = gate_header
        self.gate = gate
        self.hard_no = hard_no
        self.proceed = proceed
        self.fallback = fallback
        self.dropped_missing = dropped_missing

    def _gate_sides(self, row: dict) -> tuple[dict, dict]:
        one = self.by_gtin[(row["gtin1"] or "").strip()]["attribute"]
        two = self.by_gtin[(row["gtin2"] or "").strip()]["attribute"]
        return compose_side(one), compose_side(two)

    def _gate_state(self, row: dict) -> str:
        side_one, side_two = self._gate_sides(row)
        return compose_state(side_one, side_two)

    def _build_gate_and_proceed_cases(self) -> None:
        target_total_negatives = min(self.hard_no_cap, self.positives)
        hard_no_target = max(0, target_total_negatives - self.listing_negatives)
        sampled = SplitAllocator._stratified_sample(
            self.hard_no, hard_no_target, lambda row: row["gate_reason"],
            self.seed)
        self.sampled = sampled
        self.gate_reason_sample = Counter(row["gate_reason"] for row in sampled)
        self.gate_splits = SplitAllocator._assign_splits(
            sampled, self.ratios, self.seed,
            SplitAllocator._gate_reason_stratum)
        # Every joinable `proceed` gate row rides the corpus for its gate
        # verdict / reason labels (identity_claim stays unlabelled: the gate
        # verdict is not a GTIN truth). Deterministic row order, then split.
        self.proceed_splits = SplitAllocator._assign_splits(
            list(self.proceed), self.ratios, self.seed,
            SplitAllocator._gate_reason_stratum)
        self.gate_reason_families = Counter(
            GateReasonRules.gate_reason_family(row["gate_reason"])
            for row in self.gate)

        gate_records_by_split: dict[str, list[dict]] = {
            key: [] for key in SPLIT_ORDER}
        for key in SPLIT_ORDER:
            for row in self.gate_splits[key]:
                side_one, side_two = self._gate_sides(row)
                family = GateReasonRules.gate_reason_family(row["gate_reason"])
                expected = CorpusCaseRenderer._pair_expected(
                    self.questions, side_one, side_two, identity="false",
                    gate_verdict="hard_no", gate_reason=family,
                    attr_one=self.by_gtin[
                        (row["gtin1"] or "").strip()]["attribute"],
                    attr_two=self.by_gtin[
                        (row["gtin2"] or "").strip()]["attribute"],
                    brand_one=self.by_gtin[
                        (row["gtin1"] or "").strip()].get("brand"),
                    brand_two=self.by_gtin[
                        (row["gtin2"] or "").strip()].get("brand"),
                    counterfactual="false")
                gate_records_by_split[key].append(CorpusCaseRenderer._record(
                    compose_state(side_one, side_two), self.questions, expected,
                    difficulty_slice=PairLabelRules._difficulty_slice(
                        side_one, side_two),
                    gate_reason=family,
                    attribute=PairLabelRules._primary_attribute(
                        side_one, side_two)))

        proceed_records_by_split: dict[str, list[dict]] = {
            key: [] for key in SPLIT_ORDER}
        for key in SPLIT_ORDER:
            for row in self.proceed_splits[key]:
                side_one, side_two = self._gate_sides(row)
                family = GateReasonRules.gate_reason_family(row["gate_reason"])
                expected = CorpusCaseRenderer._pair_expected(
                    self.questions, side_one, side_two,
                    gate_verdict="proceed", gate_reason=family,
                    attr_one=self.by_gtin[
                        (row["gtin1"] or "").strip()]["attribute"],
                    attr_two=self.by_gtin[
                        (row["gtin2"] or "").strip()]["attribute"],
                    counterfactual="false")
                proceed_records_by_split[key].append(CorpusCaseRenderer._record(
                    compose_state(side_one, side_two), self.questions, expected,
                    difficulty_slice=PairLabelRules._difficulty_slice(
                        side_one, side_two),
                    gate_reason=family,
                    attribute=PairLabelRules._primary_attribute(
                        side_one, side_two)))
        self.gate_records_by_split = gate_records_by_split
        self.proceed_records_by_split = proceed_records_by_split

    # ── growth + rebalance ─────────────────────────────────────────────────
    def _fold_growth_and_rebalance(self) -> None:
        # The bundles/labeled pairs are opt-in (None in the hermetic builder
        # tests), so a caller without them reproduces the pre-growth corpus.
        existing_pair_states = {
            record["state"]
            for key in SPLIT_ORDER
            for record in (self.pair_by_split[key]
                           + self.gate_records_by_split[key])
        }
        mask_records, aug_records, growth_census, aug_origins = (
            GrowthFoldIngestor.ingest(
                bundle_path=self.bundle_path,
                labeled_pairs_path=self.labeled_pairs_path,
                by_gtin=self.by_gtin, existing_pair_states=existing_pair_states,
                questions=self.questions))
        self.growth_census = growth_census
        # ── IDENTITY REBALANCE (config-owned; no-op by default) ────────────
        # The minted counterfactual/twin negatives dominate the identity prior
        # (~1:14). The knob thins ONLY that minted population, deterministically,
        # never the ground-truth negatives.
        identity_negatives_growth = growth_census["aug_pair_negative"]
        positives_growth = growth_census["aug_pair_positive"]
        minted_negatives = sum(
            1 for record in aug_records
            if aug_origins.get(record["state"]) == "bundle"
            and record["expected"].get("identity_claim") == "false")
        # Ground truth = every negative that is NOT pipeline-minted: the listing
        # pairs, the gate hard_no sample and the labeled pairs.
        ground_negative_total = (self.listing_negatives + len(self.sampled)
                                 + identity_negatives_growth - minted_negatives)
        aug_records, rebalance_census = (
            SplitAllocator._rebalance_identity_negatives(
                aug_records, aug_origins,
                positives_total=self.positives + positives_growth,
                ground_negative_total=ground_negative_total,
                ratio=self.identity_negative_target_ratio, seed=self.seed))
        self.rebalance_census = rebalance_census
        self.positives_growth = positives_growth
        identity_negatives_emitted = sum(
            1 for record in aug_records
            if record["expected"].get("identity_claim") == "false")
        self.identity_negatives_total = (self.listing_negatives
                                         + len(self.sampled)
                                         + identity_negatives_emitted)
        self.mask_splits = SplitAllocator._assign_splits(
            mask_records, self.ratios, self.seed,
            SplitAllocator._difficulty_stratum)
        self.aug_splits = SplitAllocator._assign_splits(
            aug_records, self.ratios, self.seed,
            SplitAllocator._difficulty_stratum)

    def _build_better_match_cases(self) -> None:
        better_records = CorpusCaseRenderer.better_match_records(
            self.pairs, self.by_sku, self.questions)
        self.better_splits = SplitAllocator._assign_splits(
            better_records, self.ratios, self.seed,
            SplitAllocator._difficulty_stratum)

    # ── emit ───────────────────────────────────────────────────────────────
    def _emit_splits(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        split_sizes: dict[str, int] = {}
        split_counts: dict[str, dict] = {}
        split_paths: dict[str, Path] = {}
        split_records: dict[str, list] = {}
        for key in SPLIT_ORDER:
            path = self.output_dir / f"{key}.jsonl"
            lines = (self.pair_by_split[key] + self.gate_records_by_split[key]
                     + self.proceed_records_by_split[key] + self.aug_splits[key]
                     + self.better_splits[key] + self.state_splits[key]
                     + self.mask_splits[key])
            with path.open("w", encoding="utf-8") as handle:
                for record in lines:
                    handle.write(JsonLine.dump(record) + "\n")
            split_paths[key] = path
            split_records[key] = lines
            split_sizes[key] = len(lines)
            split_counts[key] = {
                "listing_positive": sum(
                    1 for r in self.pair_by_split[key]
                    if r["expected"].get("identity_claim") == "true"),
                "listing_negative": sum(
                    1 for r in self.pair_by_split[key]
                    if r["expected"].get("identity_claim") == "false"),
                "gate_negative": len(self.gate_records_by_split[key]),
                "gate_proceed": len(self.proceed_records_by_split[key]),
                "aug_pair_positive": sum(
                    1 for r in self.aug_splits[key]
                    if r["expected"].get("identity_claim") == "true"),
                "aug_pair_negative": sum(
                    1 for r in self.aug_splits[key]
                    if r["expected"].get("identity_claim") == "false"),
                "better_match": len(self.better_splits[key]),
                "state": len(self.state_splits[key]),
                "mask_state": len(self.mask_splits[key]),
            }
        self.split_paths = split_paths
        self.split_sizes = split_sizes
        self.split_counts = split_counts
        self.split_records = split_records

    def _quarantine_fallback(self) -> None:
        unknown_path = self.output_dir / "unknown_pairs.csv"
        unknown_columns = list(self.gate_header) + [
            "gtin1_in_catalog", "gtin2_in_catalog", "attribute_pairs"]
        with unknown_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=unknown_columns,
                                    lineterminator="\n")
            writer.writeheader()
            for row in self.fallback:
                in_one = (row["gtin1"] or "").strip() in self.by_gtin
                in_two = (row["gtin2"] or "").strip() in self.by_gtin
                out = dict(row)
                out["gtin1_in_catalog"] = str(in_one)
                out["gtin2_in_catalog"] = str(in_two)
                out["attribute_pairs"] = (
                    self._gate_state(row) if (in_one and in_two) else "")
                writer.writerow(out)
        self.unknown_path = unknown_path

    def _build_traceability_census(self) -> None:
        all_records = [
            record
            for key in SPLIT_ORDER
            for record in (self.pair_by_split[key]
                           + self.gate_records_by_split[key]
                           + self.proceed_records_by_split[key]
                           + self.aug_splits[key]
                           + self.better_splits[key]
                           + self.state_splits[key]
                           + self.mask_splits[key])
        ]
        question_label_census: dict[str, Counter] = defaultdict(Counter)
        difficulty_slice_census: Counter = Counter()
        gate_reason_census: Counter = Counter()
        attribute_census: Counter = Counter()
        for record in all_records:
            for qid, label in record["expected"].items():
                question_label_census[qid][label] += 1
            difficulty_slice_census[record["difficulty_slice"]] += 1
            gate_reason_census[record["gate_reason"] or "none"] += 1
            attribute_census[record["attribute"]] += 1
        self.question_label_census = question_label_census
        self.difficulty_slice_census = difficulty_slice_census
        self.gate_reason_census = gate_reason_census
        self.attribute_census = attribute_census
        self.all_records = all_records
        self.strata_coverage = self._strata_coverage()
        self.corpus_split_plan = self._corpus_split_plan()

    #: The corpus subgroup axes the carve is stratified/sized against. Both
    #: ride every record as top-level tags (CorpusCaseRenderer._record), so the
    #: census reads the emitted truth, never a re-derivation.
    CORPUS_SLICES: dict[str, str] = {
        "difficulty_slice": "scalar",
        "attribute": "scalar",
    }

    def _strata_coverage(self) -> dict:
        """Per-split counts of each corpus subgroup (the carve's coverage)."""
        return {
            key: {
                slice_name: dict(sorted(Counter(
                    record[slice_name] for record in self.split_records[key]
                ).items()))
                for slice_name in self.CORPUS_SLICES
            }
            for key in SPLIT_ORDER
        }

    def _corpus_split_plan(self) -> dict:
        """The power-consistent carve plan (targets: config/sampling.yaml).

        Reuses the canonical ``SamplePlan`` over the corpus's own subgroups so
        dev (select) and validation (held-out report) are sized to the SAME
        declared MDE the rest of the project measures against. The binding
        meaningful subgroup sets the per-subgroup floor; ``reachable`` states
        whether the carve can deliver it (`False` = the census, not the split,
        is the constraint). Read-only: it measures, it never gates.
        """
        from core.common import training_cfg
        from core.sample_plan import SamplePlan, SubgroupCensus

        plan = SamplePlan.from_config()
        census = SubgroupCensus(self.CORPUS_SLICES)
        censuses = census.census(self.all_records)
        folds = int(training_cfg().split.holdout_component_folds)
        report = plan.plan(
            censuses, labeled_census=len(self.all_records),
            component_folds=folds)
        binding = report.binding
        return {
            "per_subgroup_n": report.per_subgroup_n,
            "binding": None if binding is None else {
                "slice": binding.slice, "value": binding.value,
                "support": binding.support, "share": binding.share,
                "required_n": binding.required_n},
            "recommended_n": report.recommended_n,
            "recommended_validation_size": report.recommended_validation_size,
            "validation_size_unit": report.validation_size_unit,
            "reachable": report.validation_size_reachable,
            "mde_paired": {
                "dev": plan.mde_paired(self.split_sizes["dev"]),
                "validation": plan.mde_paired(self.split_sizes["test"]),
                "at_per_subgroup_n": plan.mde_paired(report.per_subgroup_n),
            },
            "targets": {
                name: getattr(report, name) for name in (
                    "confidence", "ci_half_width", "alpha", "power",
                    "target_effect", "min_subgroup_support")},
            "slices": [
                {"slice": slice_plan.slice,
                 "population": slice_plan.population,
                 "populated": slice_plan.populated,
                 "values": len(slice_plan.requirements)}
                for slice_plan in report.slices
            ],
        }

    def _write_receipt(self) -> dict:
        receipt = {
            "seed": self.seed,
            "package_state_rule": PACKAGE_STATE_RULE,
            "identity_label_values": {
                "true": self.positives, "false": self.listing_negatives},
            "counts": {
                "state_cases": len(self.catalog),
                "state_package_state_true": self.state_pkg["true"],
                "state_package_state_false": self.state_pkg["false"],
                "state_pack_type_only_no_quantity": sum(
                    1 for row in self.catalog
                    if PackageStateRules._pack_type_only(row["attribute"])),
                "listing_pairs_positive": self.positives,
                "listing_pairs_negative": self.listing_negatives,
                "gate_hard_no_available": len(self.hard_no),
                "gate_hard_no_sampled": len(self.sampled),
                "gate_proceed_available": len(self.proceed),
                "gate_proceed_in_corpus": sum(
                    len(self.proceed_records_by_split[key])
                    for key in SPLIT_ORDER),
                "gate_fallback_quarantined": len(self.fallback),
                "gate_hard_no_cap": self.hard_no_cap,
                "better_match_cases": sum(
                    len(self.better_splits[key]) for key in SPLIT_ORDER),
                "identity_positive_total": self.positives,
                "identity_negative_total": (self.listing_negatives
                                            + len(self.sampled)),
                "dropped_missing": {
                    "hard_no": self.dropped_missing["hard_no"],
                    "fallback": self.dropped_missing["fallback"],
                    "total": sum(self.dropped_missing.values()),
                },
                # ── the growth censuses (masking + augmentation) ──────────
                "mask_cases": self.growth_census["mask_cases"],
                "mask_package_state_true":
                    self.growth_census["mask_package_state_true"],
                "mask_package_state_false":
                    self.growth_census["mask_package_state_false"],
                "aug_pairs": self.growth_census["aug_pair_cases"],
                "aug_pairs_positive": self.growth_census["aug_pair_positive"],
                "aug_pairs_negative": self.growth_census["aug_pair_negative"],
                "aug_pairs_counterfactual":
                    self.growth_census.get("aug_pair_counterfactual", 0),
                "labeled_pairs_added": self.growth_census["labeled_pairs_added"],
                # corpus-wide identity prior AFTER folding the augmentation in:
                # the counterfactual/twin negatives are hard and numerous, so
                # the operator can see the balance at a glance.
                "identity_positive_total_with_growth": (
                    self.positives + self.positives_growth),
                "identity_negative_total_with_growth":
                    self.identity_negatives_total,
            },
            "growth": self.growth_census,
            "gate_reason_sample": dict(sorted(self.gate_reason_sample.items())),
            "gate_reason_families":
                dict(sorted(self.gate_reason_families.items())),
            "question_label_census": {
                qid: dict(sorted(labels.items()))
                for qid, labels in sorted(self.question_label_census.items())
            },
            "difficulty_slice_census":
                dict(sorted(self.difficulty_slice_census.items())),
            "attribute_census": dict(sorted(self.attribute_census.items())),
            "tag_gate_reason_census":
                dict(sorted(self.gate_reason_census.items())),
            "split_ratios": {key: self.ratios[key] for key in SPLIT_ORDER},
            "split_sizes": self.split_sizes,
            "split_counts": self.split_counts,
            # The carve's SSOT: which split SELECTS (HPO objective/early-stop)
            # and which VALIDATES (the held-out report) — never re-spelled in a
            # consumer. `test.jsonl` is the validation role, unchanged.
            "split_roles": dict(LayaSplitRoles.ROLES),
            "strata_coverage": self.strata_coverage,
            "sample_plan": self.corpus_split_plan,
            "question_schema_sha256": self.question_sha,
            "sha256": {
                **{f"{key}.jsonl": Digest.sha256(self.split_paths[key])
                   for key in SPLIT_ORDER},
                "unknown_pairs.csv": Digest.sha256(self.unknown_path),
            },
        }
        # One stable digest over the whole corpus body (the split files in
        # frozen SPLIT_ORDER), for the report and the determinism test.
        corpus_digest = hashlib.sha256()
        for key in SPLIT_ORDER:
            corpus_digest.update(self.split_paths[key].read_bytes())
        receipt["corpus_sha256"] = corpus_digest.hexdigest()
        # Additive: the composition-census blocks land ONLY when a knob is set,
        # so a default config keeps the landed receipt bytes exactly.
        if self.rebalance_census.get("enabled"):
            receipt["identity_rebalance"] = self.rebalance_census
        if not CorpusConfig.is_default(self.corpus):
            receipt["corpus_config"] = self.corpus
        receipt_path = self.output_dir / "receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        return receipt


def main() -> None:
    # The question schema is the config SSOT for this builder: its optional
    # `corpus` block owns the composition knobs AND the input/output paths
    # (hardcoded defaults otherwise), so a run never depends on code literals.
    if not Path(QUESTION_PATH).is_file():
        raise FileNotFoundError(f"required source missing: {QUESTION_PATH}")
    corpus_cfg = CorpusConfig.from_document(
        json.loads(Path(QUESTION_PATH).read_text(encoding="utf-8")))
    sources = CorpusConfig.resolve_sources(corpus_cfg)
    for key in ("catalog", "pairs", "gate"):
        if not Path(sources[key]).is_file():
            raise FileNotFoundError(
                f"required source missing ({key}): {sources[key]}")
    # Fold the pipeline's minted masking/augmentation in when the artifacts
    # are present (read, never invented); absent sources are reported, not
    # fabricated. `build()` still runs the base corpus without them.
    bundle = sources["bundle"] if Path(sources["bundle"]).is_file() else None
    labeled = (sources["labeled_pairs"]
               if Path(sources["labeled_pairs"]).is_file() else None)
    if bundle is None:
        print(f"[laya-build-dataset] masking/augmentation source absent: "
              f"{sources['bundle']} (base corpus only)")
    if labeled is None:
        print(f"[laya-build-dataset] labeled pairs source absent: "
              f"{sources['labeled_pairs']} (base corpus only)")
    if not CorpusConfig.is_default(corpus_cfg):
        print("[laya-build-dataset] corpus_config=" + json.dumps(corpus_cfg))
    receipt = CorpusBuilder.build(
        catalog_path=sources["catalog"], pairs_path=sources["pairs"],
        gate_path=sources["gate"], output_dir=sources["output_dir"],
        question_path=QUESTION_PATH, corpus_config=corpus_cfg,
        bundle_path=bundle, labeled_pairs_path=labeled)
    counts = receipt["counts"]
    print(
        "[laya-build-dataset] states=%d (pkg true=%d false=%d) "
        "listing_pairs=%d+,%d- gate_hard_no_sampled=%d/%d "
        "fallback_quarantined=%d dropped_missing=%d"
        % (counts["state_cases"], counts["state_package_state_true"],
           counts["state_package_state_false"], counts["listing_pairs_positive"],
           counts["listing_pairs_negative"], counts["gate_hard_no_sampled"],
           counts["gate_hard_no_available"], counts["gate_fallback_quarantined"],
           counts["dropped_missing"]["total"]))
    print(
        "[laya-build-dataset] growth: mask_cases=%d (pkg true=%d false=%d) "
        "aug_pairs=%d (+%d/-%d) labeled_pairs_added=%d"
        % (counts["mask_cases"], counts["mask_package_state_true"],
           counts["mask_package_state_false"], counts["aug_pairs"],
           counts["aug_pairs_positive"], counts["aug_pairs_negative"],
           receipt["growth"]["labeled_pairs_added"]))
    print("[laya-build-dataset] split_sizes="
          + json.dumps(receipt["split_sizes"]))
    print("[laya-build-dataset] split_counts="
          + json.dumps(receipt["split_counts"]))
    print("[laya-build-dataset] split_roles="
          + json.dumps(receipt["split_roles"]))
    print("[laya-build-dataset] strata_coverage="
          + json.dumps(receipt["strata_coverage"]))
    print("[laya-build-dataset] sample_plan="
          + json.dumps(receipt["sample_plan"]))
    if receipt["sample_plan"]["reachable"] is False:
        print("[laya-build-dataset] WARNING: sample plan unreachable for this "
              "corpus size: the census, not the split, is the binding "
              "constraint (see sample_plan.binding)")
    print("[laya-build-dataset] question_label_census="
          + json.dumps(receipt["question_label_census"]))
    print("[laya-build-dataset] difficulty_slice_census="
          + json.dumps(receipt["difficulty_slice_census"]))
    print("[laya-build-dataset] attribute_census="
          + json.dumps(receipt["attribute_census"]))
    print("[laya-build-dataset] tag_gate_reason_census="
          + json.dumps(receipt["tag_gate_reason_census"]))
    print("[laya-build-dataset] growth=" + json.dumps(receipt["growth"]))
    if "identity_rebalance" in receipt:
        print("[laya-build-dataset] identity_rebalance="
              + json.dumps(receipt["identity_rebalance"]))
    for name, digest in receipt["sha256"].items():
        print(f"[laya-build-dataset] sha256 {name} {digest}")
    print(f"[laya-build-dataset] corpus_sha256={receipt['corpus_sha256']}")
    print("[laya-build-dataset] -> "
          + str(Path(sources["output_dir"]) / "receipt.json"))


if __name__ == "__main__":
    main()
