"""scripts/laya_build_dataset.py — thin facade over the corpus builder modules.

Owner order: "we will finetune laya with the correct dataset". This module
used to hold the whole builder (1294 lines, over the ≤1000 hard limit). It is
now a re-export facade: the behaviour lives in the focused modules below, and
this facade keeps EVERY module-level name the old file exposed (tests and
callers load this file by path and read/call those symbols).

Layers:
  * scripts/laya_corpus_composer.py — the reused pair-state composer boundary;
  * scripts/laya_corpus_io.py       — CSV / sha256 / JSONL / bundle boundaries;
  * scripts/laya_corpus_rules.py    — the documented scoring rules;
  * scripts/laya_corpus_cases.py    — render one corpus case from source truth;
  * scripts/laya_corpus_growth.py   — fold the pipeline's minted data in;
  * scripts/laya_corpus_builder.py  — config, split allocation, orchestration.

The corpus build is byte-identical for the same inputs; the facade adds no
behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Run-directly bootstrap: `python scripts/laya_build_dataset.py` puts only the
# scripts dir on sys.path, so the `scripts.` package imports below would fail.
# The builder now also reads the core layer (the role map + the sample plan),
# so this checkout's `src/` must resolve too. Importing it as a package
# (`scripts.laya_build_dataset`) skips this branch.
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT))

from scripts.laya_corpus_builder import (
    BUNDLE_PATH,
    CATALOG_PATH,
    CORPUS_CONFIG_DEFAULTS,
    CORPUS_SOURCE_DEFAULTS,
    CORPUS_SOURCE_KEYS,
    GATE_PATH,
    HARD_NO_CAP,
    LABELED_PAIRS_PATH,
    OUTPUT_DIR,
    PAIRS_PATH,
    QUESTION_PATH,
    ROOT,
    SEED,
    SPLIT_ORDER,
    CorpusBuilder,
    CorpusConfig,
    SplitAllocator,
    main,
)
from scripts.laya_corpus_cases import CorpusCaseRenderer
from scripts.laya_corpus_composer import (
    _PAIRS_BUILDER,
    PAIR_FIELDS,
    PairsComposerLoader,
    compose_side,
    compose_state,
)
from scripts.laya_corpus_growth import (
    _PAYLOAD_ATTR_ORDER,
    _PAYLOAD_ATTR_PREFIXES,
    GrowthFoldIngestor,
)
from scripts.laya_corpus_io import CsvSource, Digest, JsonLine, PreparedBundle
from scripts.laya_corpus_rules import (
    FIELD_SAME_QIDS,
    GATE_REASON_FAMILIES,
    MULTIPACK_RE,
    NUMERIC_VALUE_RE,
    PACK_COUNT_KEYS,
    PACKAGE_STATE_RULE,
    GateReasonRules,
    PackageStateRules,
    PairLabelRules,
)

# The full module-level surface of the original file, preserved for callers
# that load this facade by path and read/call its symbols.
__all__ = [
    "BUNDLE_PATH",
    "CATALOG_PATH",
    "CORPUS_CONFIG_DEFAULTS",
    "CORPUS_SOURCE_DEFAULTS",
    "CORPUS_SOURCE_KEYS",
    "FIELD_SAME_QIDS",
    "GATE_PATH",
    "GATE_REASON_FAMILIES",
    "HARD_NO_CAP",
    "LABELED_PAIRS_PATH",
    "MULTIPACK_RE",
    "NUMERIC_VALUE_RE",
    "OUTPUT_DIR",
    "PACKAGE_STATE_RULE",
    "PACK_COUNT_KEYS",
    "PAIRS_PATH",
    "PAIR_FIELDS",
    "QUESTION_PATH",
    "ROOT",
    "SEED",
    "SPLIT_ORDER",
    "_PAIRS_BUILDER",
    "_PAYLOAD_ATTR_ORDER",
    "_PAYLOAD_ATTR_PREFIXES",
    "CorpusBuilder",
    "CorpusCaseRenderer",
    "CorpusConfig",
    "CsvSource",
    "Digest",
    "GateReasonRules",
    "GrowthFoldIngestor",
    "JsonLine",
    "PackageStateRules",
    "PairLabelRules",
    "PairsComposerLoader",
    "Path",
    "PreparedBundle",
    "SplitAllocator",
    "_attr_from_payload",
    "_corpus_config",
    "_difficulty_slice",
    "_dump_line",
    "_field",
    "_field_same_label",
    "_has_evidence",
    "_ingest_masking_and_augmentation",
    "_load_pairs_builder",
    "_load_prepared_bundle",
    "_pack_type_only",
    "_package_signature",
    "_pair_expected",
    "_pair_meta",
    "_primary_attribute",
    "_read_csv",
    "_rebalance_identity_negatives",
    "_record",
    "_render_triple_side",
    "_sha256",
    "_side_from_payload",
    "_single_meta",
    "_stratified_sample",
    "better_match_records",
    "build",
    "compose_side",
    "compose_state",
    "corpus_config_is_default",
    "evidence_sufficient",
    "gate_reason_family",
    "main",
    "pack_format_equivalent",
    "pack_volume_equal",
    "package_state",
    "resolve_corpus_sources",
    "same_brand_only",
]

# ── the pair composer (reused from scripts/laya_metrics_pairs.py) ──────────
_load_pairs_builder = PairsComposerLoader.load

# ── I/O boundaries ─────────────────────────────────────────────────────────
_read_csv = CsvSource.read
_sha256 = Digest.sha256
_dump_line = JsonLine.dump
_load_prepared_bundle = PreparedBundle.load

# ── rules ──────────────────────────────────────────────────────────────────
_field = PackageStateRules._field
package_state = PackageStateRules.package_state
_pack_type_only = PackageStateRules._pack_type_only
_package_signature = PackageStateRules._package_signature
pack_volume_equal = PackageStateRules.pack_volume_equal
pack_format_equivalent = PackageStateRules.pack_format_equivalent
_field_same_label = PairLabelRules._field_same_label
_has_evidence = PairLabelRules._has_evidence
evidence_sufficient = PairLabelRules.evidence_sufficient
same_brand_only = PairLabelRules.same_brand_only
_difficulty_slice = PairLabelRules._difficulty_slice
_primary_attribute = PairLabelRules._primary_attribute
gate_reason_family = GateReasonRules.gate_reason_family

# ── config ─────────────────────────────────────────────────────────────────
_corpus_config = CorpusConfig.from_document
resolve_corpus_sources = CorpusConfig.resolve_sources
corpus_config_is_default = CorpusConfig.is_default

# ── split allocation ───────────────────────────────────────────────────────
_rebalance_identity_negatives = SplitAllocator._rebalance_identity_negatives
_stratified_sample = SplitAllocator._stratified_sample

# ── case rendering ─────────────────────────────────────────────────────────
_record = CorpusCaseRenderer._record
_pair_expected = CorpusCaseRenderer._pair_expected
_pair_meta = CorpusCaseRenderer._pair_meta
_single_meta = CorpusCaseRenderer._single_meta
_render_triple_side = CorpusCaseRenderer._render_triple_side
better_match_records = CorpusCaseRenderer.better_match_records

# ── growth fold ────────────────────────────────────────────────────────────
_attr_from_payload = GrowthFoldIngestor._attr_from_payload
_side_from_payload = GrowthFoldIngestor._side_from_payload
_ingest_masking_and_augmentation = GrowthFoldIngestor.ingest

# ── orchestration ──────────────────────────────────────────────────────────
build = CorpusBuilder.build

if __name__ == "__main__":
    main()
