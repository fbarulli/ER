"""Laya decision lane (branch laya-lane).

Typed-decision surface for the `laya` package (PyPI `laya`: non-
autoregressive decision engine, Python >= 3.10, torch/transformers wheel
stack): stage the laya.question schema + ER decision inputs on THIS box,
then run the typed decision questions on a REMOTE GPU session:
  * kind="kaggle" — a Kaggle GPU kernel payload (metadata + script +
    receipt under results/laya_lane/kaggle/<decision>/); `--execute`
    drives `kaggle kernels push`, everything else is offline staging.
  * kind="colab"  — a Colab notebook payload + receipt under
    results/laya_lane/colab/<decision>/, delivered in the kaggle-lane
    receipts style. The notebook is the delivery CONTRACT: this lane
    never imports or edits cli.colab / cli.colab_lane and never opens a
    session.

Owner rulings honored (docs/laya-lane.md):
* 2xT4 -> SINGLE T4 per owner ruling: no double accelerator (the session
  pins one CUDA device; the staged payload never requests a second GPU);
* laya installs over pip (`pip install laya`), never vendored;
* exports return via /kaggle/working tar + a hashed receipt.

Contract + evidence: tests/test_laya_lane.py (offline, no network).
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_PARIS = ZoneInfo("Europe/Paris")  # build once, not per log line

from core.bundle import Bundle, BundleRole
from core.common import TRAIN_ROOT, training_cfg
from core.coverage_contracts import UNKNOWN_DIMENSION_VALUE
from core.eval_trace import (
    AbstentionBlock,
    CORE_RECORD_DIMENSIONS,
    EvalProvenance,
    EvalRowKey,
    MetricBlock,
    MinConfidenceKey,
    Share,
    TaggedRecord,
    TraceabilityCoverage,
    TraceabilityReport,
    derived_dimension_policies,
    derived_record_census,
    write_report,
)
from core.laya_config import LayaSpec
from core.manifest import atomic_write_json, sha256_file

# One roof (kaggle_lane precedent: TRAIN_ROOT/logs/<lane>/).
KINDS = ("kaggle", "colab")
GPU_KINDS = ("attribute", "identity", "laya-cli-eval", "finetune",
             "finetune-eval", "holdout-eval")
LANE_LOG_NAME = "lane.log"
# One fresh lane.log per run: first write of this process truncates, later
# writes append (owner order 2026-10-07: overwrite, never append-sprawl).
_LANE_LOG_STARTED = False

DECISION_KERNEL_CODE_FILE = "laya_decision.py"
EVAL_KERNEL_CODE_FILE = "laya_evals.py"
COLAB_NOTEBOOK_NAME = "laya_decision_colab.py"
QUESTION_SCHEMA_FILE = "laya.question.json"
# The kernel's inputs do NOT travel with `kaggle kernels push`: they are
# the er-laya-requests dataset (spec.dataset_slug), staged as its own
# payload dir (moved on --execute, mirroring the kernels-push path).
DATASET_PAYLOAD_DIR = "dataset_payload"
DATASET_METADATA_FILE = "dataset-metadata.json"
DATASET_CSV_NAME = "dataset.csv"  # DECISION_CSV resolves THIS name
# The base-model archive travels as a sealed `inputs` Bundle: this is the
# surface-owned role manifest name (the graph bundlers own theirs the same way
# via graph_tracks.artifacts.name); `Bundle.seal_archive` writes it as the one
# container manifest and verifies the archive as it seals.
BASE_MODEL_MANIFEST_FILE = "base_model_manifest.json"

# ── fine-tune kind (owner: "lets run laya") ────────────────────────────────
# The JSONL corpus built by scripts/laya_build_dataset.py (data/laya/
# {train,dev,test}.jsonl + receipt.json) travels as its OWN kaggle dataset
# (spec.finetune_dataset_slug), distinct from the decision payloads'
# er-laya-requests dataset: a corpus version must never drop the decision
# inputs (and vice versa). The kernel wraps the REAL `laya-train` CLI on a
# single T4 and tars the checkpoint back.
FINETUNE_DECISION = "finetune"
FINETUNE_CODE_FILE = "laya_finetune.py"
FINETUNE_CORPUS_FILES = ("train.jsonl", "dev.jsonl", "test.jsonl")
FINETUNE_CORPUS_RECEIPT = "receipt.json"

# ── fine-tune EVAL-only kind (held-out score, no retrain) ──────────────────
# A dedicated `--decision finetune-eval` kernel loads a fine-tuned checkpoint
# and scores the corpus HELD-OUT split (default test.jsonl) with
# `laya.train.load_checkpoint` -> `calibration_records` -> `evaluate_records`.
# It attaches the SAME corpus dataset (er-laya-train) + the fine-tuned
# checkpoint dataset, installs laya, and writes eval_report.json + a receipt
# into /kaggle/working for fetch-back. NO training, NO Hub.
FINETUNE_EVAL_DECISION = "finetune-eval"
FINETUNE_EVAL_CODE_FILE = "laya_finetune_eval.py"
FINETUNE_EVAL_REPORT_FILE = "eval_report.json"
FINETUNE_EVAL_RECEIPT_FILE = "laya_finetune-eval.receipt.json"

# ── holdout-eval kind: component-disjoint verification, run on Kaggle ───────
# Scores a fine-tuned checkpoint on the staged holdout (real pairs + P0 + gate
# strata) in-session and writes the clustered, gate-stratified report, so the
# honest verification never runs on the operator box. The holdout travels as a
# staged JSONL dataset (one composed identity state + label + stratum per row);
# the checkpoint rides the finetune_ckpt_dataset.
HOLDOUT_EVAL_DECISION = "holdout-eval"
HOLDOUT_EVAL_CODE_FILE = "laya_holdout_eval.py"
HOLDOUT_EVAL_REPORT_FILE = "holdout_report.json"
HOLDOUT_EVAL_RECEIPT_FILE = "laya_holdout-eval.receipt.json"
HOLDOUT_JSONL = "holdout.jsonl"
HOLDOUT_CATALOG_FILE = "holdout_catalog.csv"
# The corpus split name is a config literal; map it to the JSONL file the
# corpus dataset carries. Never duplicated: the tuple above is the SSOT.
FINETUNE_EVAL_SPLIT_FILES = {
    "train": FINETUNE_CORPUS_FILES[0],
    "dev": FINETUNE_CORPUS_FILES[1],
    "test": FINETUNE_CORPUS_FILES[2],
}
# `pip install laya`; pin laya>=0.3.29 (the version the flags were verified
# against: /tmp/opc/laya_pkg329/bin/laya-train --help).
# Deprecated module alias (the ``laya_package`` template value the legacy
# tests/test_lane_fixes.py renderer passes). The SSOT is
# ``laya.finetune_package``; every staged payload resolves it from the spec.
FINETUNE_LAYA_PACKAGE = LayaSpec().finetune_package
# The FULL `laya.train.TrainConfig` field surface the finetune kernel
# builds from `laya.finetune` (SSOT): every trainer knob is YAML-driven,
# never a code literal. `eval_data` is supplied by the kernel (the attached
# dev split); `device` is a runtime resolver input, not a TrainConfig field.
FINETUNE_CONFIG_FIELDS = (
    "epochs", "micro_batch", "grad_accum", "encoder_lr", "head_lr",
    "min_lr", "weight_decay", "grad_clip",
    "loss", "label_smoothing", "rl_samples", "sigma_start", "sigma_end",
    "w_sph", "w_rps",
    "shuffle_options", "option_layout", "max_len", "head_max_len",
    "text_column", "label_column", "question_id", "instructions",
    "freeze_encoder",
    "calib_max", "calib_frac", "calib_seed", "target_error", "min_abstain_n",
    "seed", "amp", "gradient_checkpointing", "log_every",
)


def finetune_config(spec: Any | None = None) -> dict[str, Any]:
    """The full `laya.train.TrainConfig` kwargs from `laya.finetune` (SSOT).

    `shuffle_options` is normalised to a tuple (the TrainConfig annotation)
    while staying JSON/repr-bakeable. Never a second recipe registry: the
    field list above names exactly the `FinetuneSpec` surface.
    """
    ft = (spec or _spec()).finetune
    config = {name: getattr(ft, name) for name in FINETUNE_CONFIG_FIELDS}
    config["shuffle_options"] = tuple(config["shuffle_options"])
    return config


# The `EvalCalibrationSpec` surface the held-out eval kernels consume: the
# per-type temperature fit + the optional abstention (`min_confidence`) fit.
# One tuple, so the shadowing lint (`finetune_config` precedent) proves every
# declared knob reaches the baked literal.
EVAL_CALIBRATION_FIELDS = (
    "temperature", "abstention", "target_error", "min_abstain_n",
    "min_confidence",
)


def eval_calibration_config(spec: Any | None = None) -> dict[str, Any]:
    """The `laya.eval_calibration` selection the eval kernels bake (SSOT).

    Never a second registry: the field list above names exactly the
    `EvalCalibrationSpec` surface. The defaults reproduce the landed eval
    exactly (temperature fit on, abstention fit off, no pinned scalar).
    """
    ec = (spec or _spec()).eval_calibration
    return {name: getattr(ec, name) for name in EVAL_CALIBRATION_FIELDS}


def fit_eval_calibration(laya_train: Any, records, calibration: dict) -> dict:
    """Fit laya's OWN calibration for the held-out eval path (SSOT selection).

    The CPU `--local-eval` twin of the `finetune-eval` kernel's
    `fit_eval_calibration` (the kernel is a staged string and cannot import
    this module — keep the two in lockstep). Consumes laya's
    `fit_temperature_map` / `fit_abstention_thresholds`, never reimplementing
    either; the defaults reproduce the landed eval exactly.
    """
    level = calibration or {}
    temperature = temperature_by_options = n_by_bucket = None
    if level.get("temperature", True):
        fitted = laya_train.fit_temperature_map(records)
        temperature = fitted.get("temperature")
        temperature_by_options = fitted.get("temperature_by_options")
        n_by_bucket = fitted.get("n_by_bucket")
    thresholds: dict[str, float] = {}
    if level.get("abstention"):
        thresholds = dict(laya_train.fit_abstention_thresholds(
            records, temperature, temperature_by_options or {},
            target_error=level.get("target_error", 0.10),
            min_bucket_n=level.get("min_abstain_n", 10)) or {})
    min_confidence = level.get("min_confidence")
    if min_confidence is not None:
        thresholds["default"] = min_confidence
    return {
        "temperature": temperature,
        "temperature_by_options": temperature_by_options,
        "n_by_bucket": n_by_bucket,
        "abstention_thresholds": thresholds,
        "min_confidence": min_confidence,
    }


# Per decision kind: the required header columns, the state column the
# decision state is built from, and what the run decides. THE CSV BINDING
# ITSELF lives in the SSOT (LayaSpec.decision_csv_bindings, resolved
# through core.common.F here) — never duplicated in a second registry.
DECISION_BINDINGS: dict[str, dict[str, Any]] = {
    "attribute": {
        "wanted_columns": ("sku_id", "sku_name_eng", "attribute"),
        "state_column": "attribute",
        # The columns that address one row (the kernel's ``_row`` tags).
        "record_columns": ("sku_id", "sku_name_eng"),
        "description": ("attribute-channel typed decision over the export's "
                        "attribute text. NOT a replacement for the frozen "
                        "SKU_ITEM/GTIN attribution path — it asks whether "
                        "laya's calibrated route agrees on the same "
                        "attributes the task layout already fixed"),
    },
    "identity": {
        "wanted_columns": ("gtin1", "gtin2", "true_label"),
        "state_column": "attribute_pairs",
        "record_columns": ("gtin1", "gtin2"),
        "description": ("identity typed decision over the frozen P0 "
                        "validation population. NOT a replacement for the "
                        "candidate-generation + gate + ann/rerank path — "
                        "its questions ask whether the candidates available "
                        "from the routes agree with the same identity "
                        "evidence the task layout already fixed"),
    },
    "laya-cli-eval": {
        "wanted_columns": ("gtin1", "gtin2", "true_label"),
        "state_column": "attribute_pairs",
        "record_columns": ("gtin1", "gtin2"),
        "description": ("the laya-evals harness score over the same "
                        "identity decision samples, verifying the shared "
                        "transport + recall identity"),
    },
    "finetune": {
        # Not a per-row decision CSV: the corpus row keys are the contract
        # (the JSONL the laya trainer consumes). Kept in the same registry so
        # `--decision finetune` rides the lane's existing binding surface.
        "wanted_columns": ("state", "questions", "expected"),
        "state_column": "state",
        # Not a per-row decision CSV: the corpus row is the unit, so no
        # decision record columns exist (the corpus adapter owns this grain).
        "record_columns": None,
        "description": ("fine-tune the convaiinnovations/laya checkpoint on "
                        "the verified-label JSONL corpus (state + identity "
                        "cases) via the real laya-train CLI on a single T4"),
    },
    "finetune-eval": {
        # Not a per-row decision CSV either: the eval-only kernel scores an
        # attached fine-tuned checkpoint against the corpus split, so the
        # corpus row keys are the contract (mirrors the finetune entry).
        "wanted_columns": ("state", "questions", "expected"),
        "state_column": "state",
        "record_columns": None,
        "description": ("HELD-OUT eval-only score of an attached fine-tuned "
                        "laya checkpoint on the corpus test split: loads the "
                        "checkpoint, runs calibration_records + "
                        "evaluate_records, writes eval_report.json. No "
                        "training, no Hub fetch."),
    },
}


def _spec():
    return training_cfg().laya


def decision_binding(decision_kind: str) -> str:
    """SSOT F-binding for a decision kind (fail-loud on an unknown kind)."""
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    bindings = _spec().decision_csv_bindings
    binding = bindings.get(decision_kind)
    if not binding:
        raise RuntimeError(
            f"config laya.decision_csv_bindings carries no entry for "
            f"{decision_kind!r}; name the config/paths.yaml files: binding "
            "before staging")
    return binding


# ── evaluation traceability adapters (lane-side; core owns the vocabulary) ──
# core.eval_trace owns the contract; THIS lane owns its row shapes, the
# record-id derivations and the corpus skip census. Every adapter is offline: it
# consumes the dict the kernel already wrote, so a fetched report is validated
# without laya installed and without re-deriving a single number.

# The one decision kind whose report is the laya-evals harness (its cases are the
# per-row grain); the rest of the decision CSV kinds report through the staged
# decision CSV itself.
_DECISION_EVAL_SOURCE = {"laya-cli-eval": "laya_cli_eval"}
# The kernel members that carry an evaluate_records payload.
_CORPUS_REPORT_NAMES = ("eval_report.json", "train_report.json")

# ── the FETCHED per-row grain (the record adapters' production caller) ──────
# `--fetch` is the one production path that sees both the remote run's own
# artifacts and this box's staged inputs, so it is where the per-row identity
# decision CSV and the laya.evals case list are built into the shared contract
# by ``decision_csv_records`` / ``eval_case_records`` / ``records_traceability``
# and WRITTEN through the declared ``traceability_report`` layout by
# ``emit_traceability`` (before this path those three adapters had zero
# production callers). A grain the fetched archive does not carry is recorded as
# an explicit ``not_applicable`` entry with its reason, never skipped; a grain it
# DOES carry but cannot be stamped or addressed fails loud.
_DECISION_ROWS_SUFFIX = ".decisions.jsonl"
_EVALS_DECISION_KIND = "laya-cli-eval"
_EVALS_REPORT_MEMBER = "report.json"
_RECORD_GRAIN_KEY = "record_grain"
_NOT_APPLICABLE = "not_applicable"
#: Every mandatory ``MetricBlock`` field: an aggregate block is only coerced into
#: one when it exposes the whole surface (see :func:`metric_block_or_none`).
_METRIC_BLOCK_FIELDS = ("items", "loss", "accuracy", "mean_confidence", "ece",
                        "brier", "brier_top1")


class RecordGrainGap(RuntimeError):
    """The fetched archive carries no per-row grain to validate (not an error in
    the run: an explicit ``not_applicable`` reason travels instead)."""


def decision_record_columns(decision_kind: str) -> tuple[str, ...]:
    """The columns that address one decision row (the kernel's ``_row`` tags).

    The kernel attaches ``{k: v for k, v in row.items() if k != STATE_COLUMN}``
    as ``answer['_row']``; the binding's declared ``record_columns`` are the
    subset of those that identify the row. Corpus kinds declare none: their grain
    is the corpus row, owned by the corpus adapter.
    """
    binding = DECISION_BINDINGS.get(decision_kind)
    if binding is None:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    columns = binding.get("record_columns")
    if not columns:
        raise ValueError(
            f"decision kind {decision_kind!r} is not a per-row decision CSV; "
            "its grain is the corpus row (see corpus_traceability)")
    return tuple(columns)


def decision_row_tags(row: dict) -> dict:
    """A decision row's tags: the kernel's ``_row`` when present, else the row."""
    tags = row.get("_row")
    return tags if isinstance(tags, dict) else row


def decision_source(decision_kind: str) -> str:
    """The eval source a per-row decision kind reports through."""
    decision_record_columns(decision_kind)  # fail loud on unknown/corpus kinds
    return _DECISION_EVAL_SOURCE.get(decision_kind, "identity_decision_csv")


def decision_csv_records(decision_kind: str, rows: list[dict], *,
                         split: str, population: str,
                         questions: dict[str, dict]) -> tuple[list, list]:
    """The ``identity_decision_csv`` grain: one TaggedRecord per CSV row, one
    EvalRowKey per (row, qid) the question schema declares.

    No identity is invented: the row id is the join of the binding's own
    ``record_columns`` over the ``_row`` tags the kernel already attaches, so a
    decision row that cannot be addressed fails loud instead of collapsing onto
    a neighbour.
    """
    columns = decision_record_columns(decision_kind)
    source = decision_source(decision_kind)
    qtypes = {qid: question["type"] for qid, question in questions.items()}
    records: list[TaggedRecord] = []
    keys: list[EvalRowKey] = []
    seen: set[str] = set()
    for row in rows:
        tags = decision_row_tags(row)
        record_id = "|".join(str(tags.get(column, "")) for column in columns)
        if record_id in seen:
            raise ValueError(f"duplicate decision record id {record_id!r}")
        seen.add(record_id)
        records.append(TaggedRecord(
            record_id=record_id, source=source, split=split,
            population=population, difficulty="unknown",
            difficulty_reason="the decision CSV carries no difficulty axis"))
        for qid, qtype in qtypes.items():
            keys.append(EvalRowKey(record_id=record_id, question_id=qid,
                                   question_type=qtype))
    return records, keys


def eval_case_records(cases: list[dict], *, split: str, population: str,
                      questions: dict[str, dict]) -> tuple[list, list]:
    """The ``laya_cli_eval`` grain: the ``laya.evals`` ``EvalReport.cases`` list.

    The harness carries no row id, so the case ordinal is the only stable
    identity; its free dimensions (language, model, tags) become slices.
    """
    qtypes = {qid: question["type"] for qid, question in questions.items()}
    records: list[TaggedRecord] = []
    keys: list[EvalRowKey] = []
    for index, case in enumerate(cases):
        record_id = f"case-{index:05d}"
        records.append(TaggedRecord(
            record_id=record_id, source="laya_cli_eval", split=split,
            population=population,
            slices=tuple(str(tag) for tag in (case.get("tags") or ())),
            difficulty="unknown",
            difficulty_reason="the evals harness carries no difficulty axis"))
        qid = str(case["qid"])
        keys.append(EvalRowKey(record_id=record_id, question_id=qid,
                               question_type=qtypes[qid]))
    return records, keys


def corpus_skip_census(report: dict) -> dict[str, int]:
    """``items_from_rows``' skip census: labelled questions that could not become
    items, counted by reason."""
    skipped = report.get("skipped") or {}
    if not isinstance(skipped, dict):
        raise ValueError(
            f"skipped census must be a mapping, got {type(skipped).__name__}")
    return {str(reason): int(count) for reason, count in skipped.items()}


@lru_cache(maxsize=1)
def _threshold_adapter():
    from pydantic import TypeAdapter

    return TypeAdapter(dict[MinConfidenceKey, Share])


def abstention_thresholds(mapping: dict | None) -> dict[str, float]:
    return dict(_threshold_adapter().validate_python(mapping or {}))


# The two shapes the finetune note uses to name the train/eval overlap:
# "10/1259 items overlap training data" and "10 items overlap".
_OVERLAP_PATTERNS = (r"(\d+)\s*/\s*(\d+)\s+items?", r"(\d+)\s+items?\s+overlap")


def overlap_items(note: Any) -> int | None:
    """The train/eval overlap count the finetune report names in its note."""
    text = str(note or "")
    for pattern in _OVERLAP_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


# The unknown policy per CARRIED core dimension. Each one cites the explicit
# unknown value, because the census reports it: the shared contract refuses a
# policy that leaves the explicit value unexplained.
_CARRIED_UNKNOWN_POLICIES: dict[str, str] = {
    "source": ("the producer tags every record with the eval source it was "
               "measured under; 'unknown' is never a source a producer assigns"),
    "split": ("every record declares exactly one split; a record without one is "
              "tagged 'unknown' rather than omitted from the census"),
    "population": ("every record declares the population it was drawn from; a "
                   "record without one is tagged 'unknown', never dropped"),
    "slice": ("a record carrying no slice is counted under 'unknown' so the "
              "membership census still accounts for the whole population"),
    "difficulty": ("difficulty is measured per record; a record with no "
                   "measurement keeps 'unknown' and carries its own "
                   "difficulty_reason, never an invented easy/hard label"),
}


def coverage_from_records(records, *, items_total: int,
                          split: str | None = None) -> TraceabilityCoverage:
    """The carried-record coverage: EVERY census COUNTED from the records.

    ``records_total``, ``by_source``, the five core strata (source/split/
    population/slice/difficulty) and each dimension's multiplicity are DERIVED
    from the records here, and the report validator re-derives the same numbers
    from the same records, so a declared census no record supports cannot pass.
    ``items_total`` is the metric grain (one record may score several questions)
    and is checked against the carried keys by the report. ``split`` is optional:
    when a producer still names it, it must be the split the records actually
    carry.
    """
    records = tuple(records)
    if not records:
        raise ValueError("coverage_from_records needs the records it censuses")
    derived = derived_record_census(records)
    policies = derived_dimension_policies(records)
    dimensions = {name: derived[name] for name in CORE_RECORD_DIMENSIONS}
    carried_splits = set(dimensions["split"])
    if split is not None and carried_splits != {split}:
        raise ValueError(
            f"declared split {split!r} is not the split the carried records "
            f"hold ({sorted(carried_splits)})")
    return TraceabilityCoverage(
        records_total=len(records), items_total=items_total,
        by_source=dimensions["source"],
        dimension_values={name: set(counts) for name, counts in dimensions.items()},
        dimension_multiplicity={name: policies[name] for name in dimensions},
        by_dimension=dimensions,
        unknown_policy=dict(_CARRIED_UNKNOWN_POLICIES),
        slice_coverage=policies["slice"])


def records_traceability(source: str, records, keys, *, model_id: str,
                         digests: dict, overall: MetricBlock | None = None,
                         split: str | None = None,
                         **provenance) -> TraceabilityReport:
    """A report whose identity is CARRIED and whose coverage is derived from it.

    ``overall`` is optional because the per-row decision grain measures no metrics
    of its own: an IDENTITY-ONLY report then carries the records and their keys,
    and its item count is the number of carried keys (derived, never a second
    declared number). When ``overall`` is given, its ``items`` must equal the key
    count, so the two can never disagree silently.
    """
    records, keys = tuple(records), tuple(keys)
    if not records:
        raise ValueError(
            "records_traceability needs the records it reports; an aggregate-only "
            "report is built by corpus_traceability, never here")
    carried_sources = {record.source for record in records}
    if carried_sources != {source}:
        raise ValueError(
            f"report source {source!r} is not what the carried records hold "
            f"({sorted(carried_sources)})")
    if not keys:
        raise ValueError(
            "carried records must carry their row keys: the item grain is the "
            "key population, never a declared number")
    items_total = len(keys) if overall is None else overall.items
    return TraceabilityReport(
        provenance=EvalProvenance(source=source, model_id=model_id,
                                  digests=digests, **provenance),
        overall=overall,
        by_type=dict(overall.by_type) if overall else {},
        records=records, keys=keys,
        coverage=coverage_from_records(records, items_total=items_total,
                                       split=split))


def corpus_traceability(report: dict, *, model_id: str, digests: dict,
                        split: str | None = None,
                        records: tuple = (), keys: tuple = ()) -> TraceabilityReport:
    """The fine-tune corpus producer -> the traceability contract.

    ``before``/``after`` are the ``evaluate_records`` blocks; ``rows``/``items``/
    ``skipped`` come from ``items_from_rows``; ``is_held_out`` and the overlap
    note are the traceability facts that matter most for a fine-tune eval.

    Two modes, and the report SAYS which one it is:

    * ``records`` + ``keys`` carried: every census is re-derived from them, and
      the aggregate numbers must agree with the carried population.
    * neither carried: the report is AGGREGATE-ONLY, declares why
      (``aggregate_reason``), and keeps the producer's own measured numbers; it
      does not pretend to a record-grain census (``slice`` is ``not_applicable``
      with a reason).
    """
    metrics = dict(report.get("after") or report.get("before") or {})
    if not metrics:
        raise ValueError("corpus report carries no before/after metric block")
    overall = MetricBlock.model_validate(metrics)
    rows = int(report.get("rows", report.get("eval_items", overall.items)))
    items = int(report.get("items", overall.items))
    if split is None:
        split = (report.get("eval_split")
                 or Path(str(report.get("eval_source") or "")).stem
                 or UNKNOWN_DIMENSION_VALUE)
    if bool(records) != bool(keys):
        raise ValueError(
            "the corpus grain carries its records and their keys together, or "
            "neither: rows=" + str(len(records)) + " keys=" + str(len(keys)))
    thresholds = abstention_thresholds(report.get("abstention_thresholds"))
    census = report.get("abstention")
    # The eval path may pin the runtime scalar explicitly (`min_confidence`);
    # otherwise the fitted map's "default" sentinel is the gate. Additive: a
    # report carrying neither keeps provenance.min_confidence None as before.
    min_confidence = report.get("min_confidence")
    if min_confidence is None:
        min_confidence = thresholds.get("default")
    if records:
        coverage = coverage_from_records(records, items_total=items, split=split)
    else:
        # Aggregate-only: the declared numbers are the producer's own measured
        # aggregates and the reason is mandatory, so this mode is never silent.
        coverage = TraceabilityCoverage(
            records_total=rows, items_total=items,
            by_source={"finetune_corpus": rows},
            dimension_values={"split": {split}},
            dimension_multiplicity={"split": "partition"},
            by_dimension={"split": {split: rows}},
            unknown_policy={"split": (
                f"the corpus split is declared by the producer ({split!r}); "
                f"a report that declares none is reported as "
                f"{UNKNOWN_DIMENSION_VALUE!r} rather than guessed")},
            slice_coverage="not_applicable",
            slice_coverage_reason=("the corpus producer reports aggregates; "
                                   "per-slice rows are not emitted yet"),
            aggregate_reason=("the corpus receipt carries measured aggregate "
                              "counts and no per-row population; the record-grain "
                              "census arrives only when the rows are carried"))
    return TraceabilityReport(
        provenance=EvalProvenance(
            source="finetune_corpus", model_id=model_id, digests=digests,
            is_held_out=report.get("is_held_out"),
            eval_overlap_items=overlap_items(report.get("note")),
            min_confidence=min_confidence,
            batch_size=int(report.get("batch_size") or 0)),
        overall=overall,
        by_type=dict(overall.by_type),
        skipped=corpus_skip_census(report),
        abstention=(AbstentionBlock.model_validate(census) if census else None),
        records=tuple(records),
        keys=tuple(keys),
        coverage=coverage)


def corpus_digest(receipt: dict) -> str:
    """The corpus digest a fetched lane receipt carries.

    The finetune receipt keys ``corpus_sha256`` by corpus file name; the eval-only
    receipt names the scored held-out split directly.
    """
    corpus = receipt.get("corpus_sha256")
    if isinstance(corpus, dict):
        for name in FINETUNE_CORPUS_FILES:
            if isinstance(corpus.get(name), str):
                return corpus[name]
    if isinstance(corpus, str):
        return corpus
    for key in ("eval_jsonl_sha256", "eval_data_sha256"):
        value = receipt.get(key)
        if isinstance(value, str):
            return value
    raise KeyError("receipt carries no corpus digest (corpus_sha256 / "
                   "eval_jsonl_sha256 / eval_data_sha256)")


def metric_block_or_none(payload: dict) -> MetricBlock | None:
    """The harness's aggregate block, ONLY when the whole surface is there.

    The evals harness may report cases without a metric block; an aggregate that
    exposes every ``MetricBlock`` field is validated as one (and fails loud when
    it disagrees with itself), while a partial block is NOT coerced: the report
    then carries its records alone, and the artifact says so (``overall: null``)
    instead of shipping a half-filled metric surface.
    """
    fields = {name: (payload or {})[name] for name in MetricBlock.model_fields
              if name in (payload or {})}
    if not set(_METRIC_BLOCK_FIELDS) <= set(fields):
        return None
    return MetricBlock.model_validate(fields)


def read_decision_rows(body: bytes) -> list[dict]:
    """The kernel's ``<kind>.decisions.jsonl`` bytes -> the per-row answer dicts."""
    rows: list[dict] = []
    for number, line in enumerate(body.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"decision row {number} is a {type(payload).__name__}, not an object")
        rows.append(payload)
    return rows


def staged_question_schema() -> dict:
    """The ``questions`` dict of the schema THIS BOX staged, never re-derived.

    A fetched kernel ships no question schema: the lane staged
    ``laya.question.json`` (``stage_question_schema``) before the push, so the
    local staged copy is the schema the remote run was fed. Absent => fail loud,
    because neither the ``(row, qid)`` keys nor the question TYPES can be derived
    without it, and a guessed qtype would silently satisfy the key contract.
    """
    for kind in KINDS:
        path = staging_dir() / kind / "question" / QUESTION_SCHEMA_FILE
        if not path.is_file():
            continue
        schema = json.loads(path.read_text(encoding="utf-8"))
        questions = schema.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError(
                f"staged question schema {path} carries no 'questions' dict")
        return questions
    raise FileNotFoundError(
        f"no staged {QUESTION_SCHEMA_FILE} under {staging_dir()}; stage the payload "
        "before fetching, or the per-row grain cannot be addressed")


def record_grain_traceability(decision_kind: str, *, receipt: dict,
                              reports: dict, decision_rows=(), questions=None
                              ) -> TraceabilityReport:
    """Build the PER-ROW report from the fetched grain (identity CSV / evals cases).

    * every per-row decision kind: the kernel's ``<kind>.decisions.jsonl`` rows
      are the grain (``decision_csv_records``) and the report is IDENTITY-ONLY
      (``overall=None``), so its item count IS the carried key count;
    * ``laya-cli-eval``: the ``laya.evals`` report's ``cases`` list is the grain
      (``eval_case_records``), and the harness's aggregate block rides along when
      it exposes the full metric surface.

    Both go through ``records_traceability``, so every census is re-derived from
    the fetched rows, and the digest the receipt names is stamped as provenance.
    A grain the fetched archive does not carry raises :class:`RecordGrainGap`
    (the caller records the explicit ``not_applicable`` reason); anything the
    archive DOES carry but cannot be addressed or stamped fails loud.
    """
    binding = DECISION_BINDINGS.get(decision_kind)
    if binding is None:
        raise ValueError(f"unknown decision kind: {decision_kind!r}")
    if not binding.get("record_columns"):
        raise RecordGrainGap(
            f"decision kind {decision_kind!r} reports through the corpus grain, "
            "not a per-row decision CSV (see corpus_traceability)")
    source = decision_source(decision_kind)
    digests = {key: receipt[key] for key in
               ("decision_csv_sha256", "evals_dataset_sha256")
               if isinstance(receipt.get(key), str)}
    # What the receipt can name, and the explicit unknown when it names nothing:
    # a guessed 'test' split would be a fabricated tag.
    split = str(receipt.get("split") or receipt.get("eval_split")
                or UNKNOWN_DIMENSION_VALUE)
    population = str(receipt.get("population") or decision_kind)
    model_id = str(receipt.get("checkpoint") or receipt.get("checkpoint_hub")
                   or receipt.get("output_dir") or receipt.get("gpu_kind")
                   or decision_kind)
    if decision_kind == _EVALS_DECISION_KIND:
        payload = reports.get(_EVALS_REPORT_MEMBER)
        cases = payload.get("cases") if isinstance(payload, dict) else None
        if not isinstance(cases, list) or not cases:
            raise RecordGrainGap(
                f"the fetched {_EVALS_REPORT_MEMBER} carries no 'cases' list, so the "
                "laya.evals per-case grain cannot be validated")
        records, keys = eval_case_records(
            cases, split=split, population=population,
            questions=questions or staged_question_schema())
        overall = metric_block_or_none(payload)
    else:
        if not decision_rows:
            raise RecordGrainGap(
                f"the fetched archive carries no {decision_kind}"
                f"{_DECISION_ROWS_SUFFIX}, so the per-row decision grain cannot be "
                "validated")
        records, keys = decision_csv_records(
            decision_kind, list(decision_rows), split=split,
            population=population, questions=questions or staged_question_schema())
        overall = None
    return records_traceability(
        source, records, keys, model_id=model_id, digests=digests,
        overall=overall, split=split,
        batch_size=int(receipt.get("batch_size") or 0))


def fetched_traceability(receipt: dict, reports: dict, *,
                         decision_kind: str | None = None,
                         decision_rows=(), questions=None
                         ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a fetched kernel's reports and EMIT the per-row grain it carries.

    Never silent (falsified 2026-10-08: a receipt naming no digest returned ``{}``
    and a member without a before/after block was skipped without trace):

    * a member that IS an ``evaluate_records`` payload but whose receipt names no
      corpus digest RAISES: its provenance cannot be stamped, so "validating" it
      would be a fiction;
    * a member that carries no such block is recorded as an explicit
      ``{'not_applicable': reason}`` entry;
    * when the fetched decision kind reports a per-row grain, it is built by
      :func:`record_grain_traceability` from the fetched rows, WRITTEN through the
      declared ``traceability_report`` layout (``emit_traceability``) and listed
      in the returned artifacts; when the archive does not carry that grain, the
      entry states ``not_applicable`` and why.

    Returns ``(documents, artifacts)``: JSON-ready documents keyed by member name
    (``_RECORD_GRAIN_KEY`` for the per-row grain) and emitted artifact paths.
    """
    model_id = str(receipt.get("output_dir") or receipt.get("checkpoint")
                   or receipt.get("gpu_kind") or "laya")
    documents: dict[str, Any] = {}
    metric_members = {
        name: payload for name, payload in reports.items()
        if isinstance(payload, dict) and (payload.get("before") or payload.get("after"))}
    if metric_members:
        try:
            digest = corpus_digest(receipt)
        except KeyError as error:
            raise KeyError(
                f"fetched members {sorted(metric_members)} are evaluate_records "
                f"payloads but the receipt names no corpus digest, so their "
                f"provenance cannot be stamped: {error}") from error
        for name in sorted(metric_members):
            documents[name] = corpus_traceability(
                metric_members[name], model_id=model_id,
                digests={"corpus_sha256": digest}).model_dump(mode="json")
    else:
        for name in _CORPUS_REPORT_NAMES:
            if isinstance(reports.get(name), dict):
                documents[name] = {_NOT_APPLICABLE: (
                    f"{name} carries no before/after evaluate_records block to "
                    "validate")}
    artifacts: dict[str, Any] = {}
    if decision_kind is not None:
        try:
            document = record_grain_traceability(
                decision_kind, receipt=receipt, reports=reports,
                decision_rows=decision_rows, questions=questions)
        except RecordGrainGap as gap:
            documents[_RECORD_GRAIN_KEY] = {_NOT_APPLICABLE: str(gap)}
        else:
            artifacts[_RECORD_GRAIN_KEY] = str(
                emit_traceability(decision_kind, document))
            documents[_RECORD_GRAIN_KEY] = document.model_dump(mode="json")
    return documents, artifacts


def emit_traceability(track: str, document: TraceabilityReport, *,
                      lane: str = "laya_lane") -> Path:
    """Write a lane traceability document to the layout ``traceability_report``
    resolves (core.common owns the template; core.eval_trace stamps the write)."""
    return write_report("traceability_report", {"lane": lane, "track": track},
                        document)


def staging_dir() -> Path:
    """The lane staging root (TRAIN_ROOT-relative; SSOT laya.staging_dir)."""
    return (TRAIN_ROOT / _spec().staging_dir).resolve()


def lane_logs_dir() -> Path:
    """The lane transcript dir (one canonical roof: TRAIN_ROOT/logs)."""
    return TRAIN_ROOT / "logs" / "laya"


def _stamp() -> str:
    """Bracketed Europe/Paris (CET/CEST) wall-clock prefix.

    Mirrors kaggle_lane._stamp: the CET convention landed there (owner
    order 2026-10-07) and this lane follows it; the "[laya-lane UTC-stamp]"
    phrasing in the relaunch brief predates that convention.
    """
    return (f"[laya-lane "
            f"{datetime.now(_PARIS):%Y-%m-%dT%H:%M:%S %Z}]")


def _log_lane(line: str) -> None:
    """Timestamped lane logging: console plus one fresh lane log per run.

    The file is truncated on the first write of this process and appended
    afterwards, so a new run writes over the previous run's transcript
    (owner order 2026-10-07: fresh file per run, never append-sprawl).
    Best-effort on the file side — a log-write failure is printed and
    never allowed to mask the operation's own outcome.
    """
    global _LANE_LOG_STARTED
    stamp = f"{datetime.now(_PARIS):%Y-%m-%dT%H:%M:%S %Z}"
    print(f"[laya-lane {stamp}] {line}", flush=True)
    try:
        log_dir = lane_logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if _LANE_LOG_STARTED else "w"
        with (log_dir / LANE_LOG_NAME).open(mode, encoding="utf-8") as handle:
            handle.write(f"{stamp} {line}\n")
        _LANE_LOG_STARTED = True
    except OSError as error:
        print(_stamp(), f"[laya-lane] lane.log write failed ({error}); "
              "continuing", flush=True)


# ── decision-input staging (dry-safe; fail-loud on a missing contract) ─────
def _measure_csv(path: Path, wanted_columns: tuple[str, ...]) -> dict[str, Any]:
    """Stdlib CSV census: header check + row count + sha256 + bytes.

    Deliberately NOT pandas: staging must run anywhere (including a box
    without the frame stack), and the contract is only the columns.
    """
    import csv as _csv

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"decision input not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = _csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"decision input has no header row: {path}")
        missing = [column for column in wanted_columns if column not in header]
        if missing:
            raise ValueError(
                f"decision input {path.name} is missing columns {missing} "
                f"(header: {header})")
        rows = sum(1 for _ in reader)
    if rows == 0:
        raise ValueError(f"decision input has no data rows: {path}")
    return {"rows": rows, "columns": list(header),
            "sha256": sha256_file(path), "bytes": path.stat().st_size}


# The accuracy/F1 metric contract a harvest agent needs when a decision
# CSV carries ground-truth labels (owner order 2026-10-07: "add accuracy
# + f1 ... give it all the pairs we know are the same"): the EXPECTED
# row count + label distribution computed from the csv itself + the gold
# columns the harvest reads — so the harvest computes accuracy/F1 against
# these WITHOUT re-deriving the expectation.
_METRIC_EXPECTATION_KEYS = ("expected_rows", "expected_label_distribution",
                            "metric_expectation")


def _metric_expectation(path: Path, columns: list[str]) -> dict[str, Any]:
    """Expected-metric contract fields for a labeled decision CSV.

    Computes `expected_rows` + `expected_label_distribution` from the
    csv's own `true_label` column (stdlib read; a csv without the column
    returns {} — the contract only ever attaches to labeled decisions).
    Fail-loud: a `true_label` column carrying values outside {0, 1}
    raises before any receipt lands.
    """
    if "true_label" not in columns:
        return {}
    import csv as _csv
    from collections import Counter

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = _csv.DictReader(handle)
        labels = Counter(row["true_label"] for row in rows)
    unknown = sorted(set(labels) - {"0", "1"})
    if unknown:
        raise ValueError(
            f"decision input {path.name} carries true_label values "
            f"outside {{0, 1}}: {unknown}")
    return {
        "expected_rows": sum(labels.values()),
        "expected_label_distribution": {
            label: labels[label] for label in sorted(labels)},
        "metric_expectation": {
            "accuracy_gold": "label",
            "f1_gold": "identity_claim-vs-true_label",
        },
    }


def _census_csv(path: Path, wanted_columns: tuple[str, ...],
                label_column: str = "true_label") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"decision input not found: {path}")
    digest = hashlib.sha256()
    labels: Counter = Counter()
    with path.open("rb") as handle:
        def lines():
            for raw in handle:
                digest.update(raw)
                yield raw.decode("utf-8")

        reader = csv.reader(lines())
        header = next(reader, None)
        if header is None:
            raise ValueError(f"decision input has no header row: {path}")
        missing = [column for column in wanted_columns if column not in header]
        if missing:
            raise ValueError(
                f"decision input {path.name} is missing columns {missing} "
                f"(header: {header})")
        label_at = header.index(label_column) if label_column in header else None
        rows = 0
        for record in reader:
            rows += 1
            if label_at is not None and record:
                labels[record[label_at] if label_at < len(record) else ""] += 1
    if rows == 0:
        raise ValueError(f"decision input has no data rows: {path}")
    expectation: dict[str, Any] = {}
    if label_at is not None:
        unknown = sorted(set(labels) - {"0", "1"})
        if unknown:
            raise ValueError(
                f"decision input {path.name} carries true_label values "
                f"outside {{0, 1}}: {unknown}")
        expectation = {
            "expected_rows": sum(labels.values()),
            "expected_label_distribution": {
                label: labels[label] for label in sorted(labels)},
            "metric_expectation": {
                "accuracy_gold": "label",
                "f1_gold": "identity_claim-vs-true_label",
            },
        }
    return {"rows": rows, "columns": list(header),
            "sha256": digest.hexdigest(), "bytes": path.stat().st_size,
            "expectation": expectation}


def stage_decision_input(kind: str, *, decision_kind: str,
                         override: Path | None = None) -> dict[str, Any]:
    """Stage ONE decision CSV under results/laya_lane/<kind>/<decision>/.

    The staged copy is receipted with the measured census (rows, columns,
    sha256, bytes) — the transport-identity contract the kaggle-lane
    package receipts carry. `override` names an alternate source CSV on
    this box (e.g. dataset_50pct.csv for the half cohort).
    """
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}")
    from core.common import F

    entry = DECISION_BINDINGS[decision_kind]
    source = Path(override) if override else F[decision_binding(decision_kind)]
    stage = staging_dir() / kind / decision_kind
    stage.mkdir(parents=True, exist_ok=True)
    census = _census_csv(source, entry["wanted_columns"])
    destination = stage / source.name
    shutil.copy2(source, destination)
    receipt = {
        "kind": kind, "decision_kind": decision_kind,
        "binding": decision_binding(decision_kind),
        "source": str(source), "staged": str(destination),
        "rows": census["rows"], "columns": census["columns"],
        "sha256": census["sha256"], "bytes": census["bytes"],
        "description": entry["description"],
        **census["expectation"],
    }
    atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
    _log_lane(f"staged decision input [{kind}/{decision_kind}] "
              f"{source.name} rows={census['rows']} "
              f"sha256={census['sha256'][:12]} -> {destination}")
    return receipt


def stage_question_schema(kind: str, *,
                          override: Path | None = None) -> dict[str, Any]:
    """Stage the laya.question schema under results/laya_lane/<kind>/.

    Dry-safe + fail-loud: a missing schema file raises FileNotFoundError
    and a schema without a 'questions' dict raises ValueError (no silent
    empty schema placeholder is ever staged).
    """
    spec = _spec()
    source = Path(override) if override else TRAIN_ROOT / spec.question_schema
    stage = staging_dir() / kind / "question"
    stage.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(f"laya.question schema not found: {source}")
    schema = json.loads(source.read_text(encoding="utf-8"))
    questions = schema.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError(
            f"laya.question schema at {source} carries no 'questions' dict")
    destination = stage / QUESTION_SCHEMA_FILE
    shutil.copy2(source, destination)
    receipt = {
        "question_schema": spec.question_schema,
        "staged": str(destination),
        "questions": sorted(questions),
        "sha256": sha256_file(source),
    }
    atomic_write_json(receipt, stage / "question.receipt.json")
    _log_lane(f"staged question schema [{kind}] {source.name} "
              f"({len(questions)} questions) -> {destination}")
    return receipt


# The laya payloads ride the ATTACHED er-laya-requests dataset, never a
# repo clone: the shipped core.runtime_inputs.checkout_preflight_script
# emits `_runtime_root = Path(root)` for clone lanes and a laya payload
# defines no root (NameError at line 36 killed the first remote boot).
# This dedicated template instead verifies THE ATTACHED INPUTS: files
# land under /kaggle/input/<slug>/ and resolve_input rglob's by name.
# REPOSITORY/BRANCH/REVISION/_runtime_files stay assigned at top level
# so the laya push gate still literal-evals them (_staged_laya_push_
# preflight — the push gate lives in this lane; the shared clone-lane
# staged_kernel_preflight checked the git tree for the attached-inputs
# inventory, which never matches a dataset-carried payload).
LAYA_RUNTIME_PREFLIGHT = '''\
_runtime_files = ("@DECISION_CSV@", @QUESTION_SCHEMA_FILE@)
INPUT_ROOT = Path("/kaggle/input")


def laya_runtime_preflight():
    """Verify the ATTACHED dataset inputs (the er-laya-requests dataset
    mounts under /kaggle/input/<slug>/ and resolve_input searches INPUTS
    recursively by name); fail loud before pip touches anything."""
    missing = [name for name in _runtime_files
               if not any(INPUT_ROOT.rglob(name))]
    if missing:
        raise FileNotFoundError(
            "Runtime preflight missing attached inputs: "
            + ", ".join(missing))
    print("[runtime-preflight] verified %d required files"
          % len(_runtime_files), flush=True)


laya_runtime_preflight()
'''


# The fine-tune corpus travels as its own attached dataset: this preflight
# verifies THE ATTACHED INPUTS (train/dev/test JSONL land under
# /kaggle/input/<slug>/ and rglob finds them by name). It bakes the same
# REPOSITORY/BRANCH/REVISION/_runtime_files inventory the lane push gate
# literal-evals.
FINETUNE_RUNTIME_PREFLIGHT = '''\
_runtime_files = ("@TRAIN_JSONL@", "@DEV_JSONL@", "@TEST_JSONL@")
INPUT_ROOT = Path("/kaggle/input")


def laya_runtime_preflight():
    """Verify the ATTACHED corpus inputs (the finetune dataset mounts under
    /kaggle/input/<slug>/ and rglob searches recursively by name); fail
    loud before pip touches anything."""
    missing = [name for name in _runtime_files
               if not any(INPUT_ROOT.rglob(name))]
    if missing:
        raise FileNotFoundError(
            "Runtime preflight missing attached inputs: "
            + ", ".join(missing))
    print("[runtime-preflight] verified %d required files"
          % len(_runtime_files), flush=True)


laya_runtime_preflight()
'''


# The eval-only kernel attaches the SAME corpus dataset and verifies the ONE
# held-out split it scores (rglob finds the JSONL under /kaggle/input/<slug>/).
# The CHECKPOINT is a separate attached dataset, resolved in-kernel by the
# `rl_agent_config.json` rglob (never vendored here): the push gate's
# `_runtime_files` inventory can only verify files that live in the staged
# dataset_payload, so the checkpoint inventory stays out of it and fails loud
# in `resolve_checkpoint()` instead.
FINETUNE_EVAL_RUNTIME_PREFLIGHT = '''\
_runtime_files = ("@EVAL_JSONL@",)
INPUT_ROOT = Path("/kaggle/input")


def laya_runtime_preflight():
    """Verify the ATTACHED corpus split (the corpus dataset mounts under
    /kaggle/input/<slug>/ and rglob searches recursively by name); fail
    loud before pip touches anything."""
    missing = [name for name in _runtime_files
               if not any(INPUT_ROOT.rglob(name))]
    if missing:
        raise FileNotFoundError(
            "Runtime preflight missing attached inputs: "
            + ", ".join(missing))
    print("[runtime-preflight] verified %d required files"
          % len(_runtime_files), flush=True)


laya_runtime_preflight()
'''


# ── kernel / notebook payload composition ──────────────────────────────────
@lru_cache(maxsize=8)
def _parse(script: str) -> ast.Module:
    return ast.parse(script)


def _kernel_script_gate(script: str) -> None:
    defined: set[str] = set()
    loaded: set[str] = set()
    for node in ast.walk(_parse(script)):
        if isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets
                           if isinstance(t, ast.Name))
        elif (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
              and node.id.isupper()):
            loaded.add(node.id)
    undeclared = loaded - defined
    if undeclared:
        raise ValueError(f"staged kernel uses undeclared constants: "
                         f"{sorted(undeclared)}; regenerate the template")


def _module_scope_gate(script: str) -> None:
    import builtins

    tree = _parse(script)  # shared with _kernel_script_gate (cache hit)
    compile(tree, "<laya-payload>", "exec")
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Import):
            bound.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            bound.update(a.asname or a.name for a in node.names
                         if a.name != "*")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    loaded = {node.id
              for stmt in tree.body
              if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef))
              for node in ast.walk(stmt)
              if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
              and not (node.id.startswith("__") and node.id.endswith("__"))}
    undeclared = sorted(loaded - bound)
    if undeclared:
        raise ValueError(f"staged kernel loads undefined top-level names: "
                         f"{undeclared}; a NameError-class payload must "
                         "never stage again")


def _template(script: str, values: dict[str, str]) -> str:
    for token, replacement in values.items():
        script = script.replace(f"@{token}@", replacement)
    return script


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TRAIN_ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"git rev-parse failed in {TRAIN_ROOT}: {result.stderr.strip()}")
    return result.stdout.strip()


def _env_value(name: str) -> str | None:
    """Read KEY=VALUE from .env (TRAIN_ROOT, then its parent), then the env.

    The same lookup the Colab lane uses (cli.colab_runtime._env_value), plus
    the box's project .env (`$HOME/ONE/.env`, the path .bashrc sources) so the
    lane finds the keys from any worktree. The secret is baked into the STAGED
    kernel only, never written to the repo.
    """
    from pathlib import Path as _Path

    for env_path in (TRAIN_ROOT / ".env", TRAIN_ROOT.parent / ".env",
                     _Path.home() / "ONE" / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == name:
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    return os.environ.get(name) or None


def _wandb_project() -> str:
    """The wandb project (`tracking.wandb.project`; ER default `e-r`)."""
    try:
        return training_cfg().tracking.wandb.project
    except Exception:
        return "e-r"


def decision_tag() -> str:
    """UTC-stamped run tag (the laya-lane stamp SURFACE stays UTC inside
    the payload because the remote session may not share this box's zone;
    the console/lane.log stamp itself stays Europe/Paris per the landed
    kaggle_lane convention)."""
    return datetime.now(ZoneInfo("UTC")).strftime("%m%dT%H%M%SZ")


DECISION_KERNEL_SCRIPT = '''\
"""ER typed decisions via laya on a Kaggle GPU session (cli.laya_lane).

Single T4 per owner ruling (2xT4 -> 1xT4; never requests the double
accelerator): pins one CUDA device, installs laya over pip, reads the
question schema + decision CSV attached as the kaggle dataset inputs,
runs the router's typed questions per row, and writes the decision
results + receipt into /kaggle/working for hash-verified fetch-back.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
CHECKPOINT_HUB = "@CHECKPOINT_HUB@"
DECISION_KIND = "@DECISION_KIND@"
RUN_TAG = "@RUN_TAG@"
DECISION_CSV = "@DECISION_CSV@"
STATE_COLUMN = "@STATE_COLUMN@"
BATCH_SIZE = @BATCH_SIZE@
MIN_CONFIDENCE = @MIN_CONFIDENCE@
QUESTION_SCHEMA_FILE = @QUESTION_SCHEMA_FILE@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_upgrade_laya():
    """laya installs over pip; torch must already be 2.14 cu13x."""
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               "--disable-pip-version-check", LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    """SINGLE T4 ruling: pin the FIRST cuda device only (never 2xT4)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    log("device pinned: " + torch.cuda.get_device_name(0)
        + " (single GPU, never a second one)")
    return "cuda"


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name + " (expected the staged "
        "laya_lane payload dataset)")


def predict_batch(agent, states, questions):
    """One batched call per chunk; min_confidence only when gated."""
    if MIN_CONFIDENCE > 0:
        return agent.predict_batch(states, questions,
                                   min_confidence=MIN_CONFIDENCE)
    return agent.predict_batch(states, questions)


def main():
    pip_upgrade_laya()
    device = pick_device()
    import laya
    questions_path = resolve_input(QUESTION_SCHEMA_FILE)
    questions = json.loads(questions_path.read_text())["questions"]
    decision_csv = resolve_input(DECISION_CSV)
    agent = laya.load(CHECKPOINT_HUB, device=device)
    rows = []
    with decision_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    log("loaded " + str(len(rows)) + " " + DECISION_KIND
        + " state rows from " + decision_csv.name)
    results = []
    for start in range(0, len(rows), BATCH_SIZE):
        chunk = rows[start:start + BATCH_SIZE]
        states = [row.get(STATE_COLUMN, "") for row in chunk]
        answers = predict_batch(agent, states, questions)
        for row, answer in zip(chunk, answers):
            answer["_row"] = {key: value for key, value in row.items()
                              if key != STATE_COLUMN}
            results.append(answer)
        log("decided " + str(min(start + BATCH_SIZE, len(rows))) + "/"
            + str(len(rows)) + " rows")
    WORKING.mkdir(parents=True, exist_ok=True)
    out = WORKING / (DECISION_KIND + ".decisions.jsonl")
    with out.open("w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(item) + "\\n" for item in results))
    log("wrote " + str(out) + " (" + str(len(results)) + " decisions)")
    receipt = {
        "gpu_kind": DECISION_KIND,
        "gpu": "T4 (single)",
        "run_tag": RUN_TAG,
        "laya_package": LAYA_PACKAGE,
        "checkpoint_hub": CHECKPOINT_HUB,
        "batch_size": BATCH_SIZE,
        "min_confidence": MIN_CONFIDENCE,
        "question_schema_sha256": hashlib.sha256(
            questions_path.read_bytes()).hexdigest(),
        "decision_csv_sha256": hashlib.sha256(
            decision_csv.read_bytes()).hexdigest(),
    }
    (WORKING / "laya_decision.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    with tarfile.open(WORKING / "laya_decision.tar.gz", "w:gz",
                      compresslevel=1) as tar:
        for item in sorted(WORKING.iterdir()):
            if item.name != "laya_decision.tar.gz":
                tar.add(item, arcname=item.name)
    log("staged laya_decision.tar.gz + receipt in /kaggle/working")


if __name__ == "__main__":
    main()
'''

EVAL_KERNEL_SCRIPT = '''\
"""laya-evals harness score on a Kaggle GPU session (cli.laya_lane).

Single T4 per owner ruling; installs laya over pip, derives the laya-evals
JSONL (state, questions, expected) from the staged identity decision CSV +
question schema, runs `laya-evals run`, and stages the report.json +
report.md + receipt into /kaggle/working.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
DECISION_CSV = "@DECISION_CSV@"
QUESTION_SCHEMA_FILE = @QUESTION_SCHEMA_FILE@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name
        + " (expected the staged laya_lane payload dataset)")


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-input", "--disable-pip-version-check",
                    LAYA_PACKAGE], check=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    questions_path = resolve_input(QUESTION_SCHEMA_FILE)
    questions = json.loads(questions_path.read_text())["questions"]
    decision_csv = resolve_input(DECISION_CSV)
    WORKING.mkdir(parents=True, exist_ok=True)
    dataset_jsonl = WORKING / "identity_evals.jsonl"
    with decision_csv.open(newline="") as handle, dataset_jsonl.open(
            "w", encoding="utf-8") as out:
        for row in csv.DictReader(handle):
            expected = int(row["true_label"])
            out.write(json.dumps({
                "state": row.get("attribute_pairs", ""),
                "questions": questions,
                "expected": {"identity_claim": expected},
            }) + "\\n")
    log("staged " + str(dataset_jsonl) + " from " + decision_csv.name)
    command = [sys.executable, "-m", "laya.evals", "run",
               str(dataset_jsonl), "--json", str(WORKING / "report.json")]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    receipt = {
        "gpu_kind": "laya-cli-eval",
        "gpu": "T4 (single)",
        "run_tag": RUN_TAG,
        "laya_package": LAYA_PACKAGE,
        "question_schema_sha256": hashlib.sha256(
            questions_path.read_bytes()).hexdigest(),
        "evals_dataset_sha256": hashlib.sha256(
            dataset_jsonl.read_bytes()).hexdigest(),
    }
    (WORKING / "laya_evals.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    log("staged laya-evals report + receipt in /kaggle/working")


if __name__ == "__main__":
    main()
'''

NOTEBOOK_SCRIPT = '''\
"""Laya typed-decision Colab payload (cli.laya_lane; delivery contract).

BOOK-END CONTRACT ONLY — no cli.colab import, NO session call. Operator
instruction set for the notebook wrapper:
  1. pip install laya (torch stack per laya's docs; python >= 3.10);
  2. upload the staged laya.question.json + the staged decision CSV as
     the notebook's own payload mounts;
  3. run the decision loop on a SINGLE GPU (never the double accelerator);
  4. write the receipt json + the export tar under the notebook's own
     export mount, then read it back into results/laya_lane/colab/<op>/.
"""

from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
DECISION_KIND = "@DECISION_KIND@"
RUN_TAG = "@RUN_TAG@"
DECISION_CSV = "@DECISION_CSV@"
STATE_COLUMN = "@STATE_COLUMN@"

WORKING = Path("/content/laya_out")


def main() -> None:
    print("[laya-lane] colab payload staged; delivery contract only",
          flush=True)
    WORKING.mkdir(parents=True, exist_ok=True)
    print("[laya-lane] decision kind: " + DECISION_KIND
          + " batch: " + str(len(list(WORKING.iterdir()))), flush=True)


if __name__ == "__main__":
    main()
'''


# Runtime device patch for the finetune kernel (laya<=0.4.0). `finetune()`
# evaluates the base checkpoint via `calibration_records()` BEFORE
# `train_model()` calls `model.to(device)`, so `load_checkpoint()`'s CPU
# model meets cuda `input_ids` and `index_select` raises "index is on
# cuda:0, different from other tensors on cpu" on the T4. Injected into the
# kernel below at the `@DEVICE_PATCH@` marker; it leaves the recipe, flags
# and the single-T4 rule untouched.
FINETUNE_DEVICE_PATCH_SOURCE = '''\
def force_model_to_device(model, device):
    # Move every module, and every registered buffer (non-persistent ones
    # included), onto `device` before any forward pass.
    import torch
    device = torch.device(device)
    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = buffer.to(device)
        module.to(device)
    return model.to(device)


def apply_device_patch():
    # Wrap the two forward entrypoints so the model is on the training
    # device before any forward pass. `calibration_records` is the crash:
    # it runs the base checkpoint on device inputs while the model is
    # still CPU. `train_model` is wrapped for the same invariant.
    from laya import train as laya_train

    original_calibration_records = laya_train.calibration_records

    def calibration_records(model, tok, items, device, *args, **kwargs):
        force_model_to_device(model, device)
        return original_calibration_records(
            model, tok, items, device, *args, **kwargs)

    laya_train.calibration_records = calibration_records

    original_train_model = laya_train.train_model

    def train_model(model, tok, items, config, device, *args, **kwargs):
        force_model_to_device(model, device)
        return original_train_model(
            model, tok, items, config, device, *args, **kwargs)

    laya_train.train_model = train_model
'''


# Movement/redundancy PERF_PATCH for the finetune kernel (laya>=0.3.29).
# Replaces laya.train.train_model with a faithful copy carrying exactly three
# recipe-neutral changes:
#   (1) the running loss is accumulated as a 0-dim CUDA tensor and synced with
#       ONE .item() per epoch (the stock loop synced twice per micro-step,
#       train.py:696 and :698);
#   (2) the batch tensors the loop consumes are moved to the device ONCE (the
#       stock loop re-moved marker_mask and qtype after _forward had already
#       moved them, train.py:608 vs :671);
#   (3) encode_item is memoized per (item id, option order) so steady-state
#       epochs skip re-tokenizing (train.py:665). draw_option_order is still
#       called in the same per-step order, so the RNG stream is identical.
# The SINGLE-PROCESS path is byte-for-byte the stock loop (ITEM order included);
# when WORLD_SIZE>1 the same loop additionally DDP-wraps the model and shards
# the items via `build_distributed_sampler` (see the DDP helpers below). The
# recipe flags, grad-accum window, clipping, scheduler and seed paths are
# unchanged. Opt out (patch AND sampler) with ER_LAYA_PERF_PATCH=0; force the
# single-process fallback with ER_LAYA_DDP=0. Injected at `@PERF_PATCH@`.
FINETUNE_PERF_PATCH_SOURCE = '''\
PERF_PATCH_ENV = "ER_LAYA_PERF_PATCH"


def perf_patch_enabled():
    # one env flag disables BOTH the movement patch and the GPU sampler.
    return os.environ.get(PERF_PATCH_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


# Distributed-data-parallel wiring (2xT4). `finetune` is a black box that
# calls train_model once per rank, so the wrap + the per-rank shard live IN
# the patched loop (DDP averages the gradients for us). One env flag forces
# the single-process fallback: ER_LAYA_DDP=0.
DDP_ENV = "ER_LAYA_DDP"


def ddp_enabled():
    return os.environ.get(DDP_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def dist_env():
    # LOCAL_RANK drives the per-rank CUDA device; RANK the global rank; the
    # spawn/torchrun launcher sets both (world_size <= 1 => single process).
    world_size = int(os.environ.get("WORLD_SIZE") or "1")
    rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK") or "0")
    return local_rank, rank, world_size


def is_distributed():
    return ddp_enabled() and dist_env()[2] > 1


def is_rank0():
    return dist_env()[1] == 0


def build_distributed_sampler(items, seed):
    # One disjoint shard per rank, reseeded every epoch via set_epoch(). The
    # pad-to-even behaviour keeps every rank's step count equal (the DDP
    # allreduce needs matching backward counts); a corpus whose item count
    # divides world_size then covers every item exactly once per epoch.
    import torch
    local_rank, rank, world_size = dist_env()
    return torch.utils.data.DistributedSampler(
        items, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)


def init_distributed(backend=None):
    # nccl on the T4 pair, gloo for the CPU test; idempotent per process.
    if not is_distributed():
        return False
    import torch
    import torch.distributed as dist
    local_rank, rank, world_size = dist_env()
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # env:// rendezvous needs a master; torchrun sets these, plain
        # torch.multiprocessing.spawn does not (single box => localhost).
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group(backend)
    return True


def barrier_if_distributed():
    if not is_distributed():
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()


def destroy_if_distributed():
    if not is_distributed():
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def run_on_rank0(fn):
    # rank 0 ONLY stages: the barrier first guarantees every rank finished
    # training before the single writer touches /kaggle/working.
    barrier_if_distributed()
    if is_rank0():
        return fn()
    return None


def resolve_nprocs():
    # 2 ranks only when DDP AND the perf loop are on and 2+ cuda devices
    # exist; an already-launched torchrun-style WORLD_SIZE is honoured as-is.
    if not (ddp_enabled() and perf_patch_enabled()):
        return 1
    env_world = int(os.environ.get("WORLD_SIZE", "0") or "0")
    if env_world > 1:
        return env_world
    try:
        import torch
    except Exception:
        return 1
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            return torch.cuda.device_count()
    except Exception:
        return 1
    return 1


def launch_finetune(worker):
    # Already launched torchrun-style (WORLD_SIZE>1): this process IS a rank,
    # run it directly. Otherwise spawn one process per visible device (nccl);
    # the caller falls back to the in-process single-device session when this
    # returns 1. spawn (not fork) because resolve_nprocs touched CUDA.
    if is_distributed():
        _, rank, world_size = dist_env()
        worker(rank, world_size)
        return world_size
    nprocs = resolve_nprocs()
    if nprocs > 1:
        import torch.multiprocessing as mp
        print("[perf-patch] distributed launch: %d ranks (nccl)" % nprocs,
              flush=True)
        mp.spawn(worker, args=(nprocs,), nprocs=nprocs, join=True,
                 start_method="spawn")
    return nprocs


def _perf_train_model(model, tok, items, config, device, max_len, head_max_len,
                      on_epoch_end=None, parallel=False):
    # Faithful copy of laya.train.train_model (0.3.29) with the three
    # recipe-neutral changes described in the source header.
    import torch
    from laya import train as laya_train

    config.validate()
    if not items:
        raise ValueError("no training items")
    amp = (device.type == "cuda") if config.amp is None else bool(config.amp)
    checkpointing = (amp if config.gradient_checkpointing is None
                     else bool(config.gradient_checkpointing))
    if config.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    elif checkpointing and hasattr(model.encoder,
                                   "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = checkpointing
    model.to(device).train()
    if config.freeze_encoder:
        model.encoder.eval()

    groups = [{"params": [p for n, p in model.named_parameters()
                          if not n.startswith("encoder.") and p.requires_grad],
               "lr": config.head_lr}]
    if not config.freeze_encoder:
        groups.insert(0, {
            "params": [p for n, p in model.named_parameters()
                       if n.startswith("encoder.") and p.requires_grad],
            "lr": config.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay,
                                 fused=(device.type == "cuda"))
    # DDP: wrap the model (grads averaged across ranks) and shard the items
    # with a per-rank DistributedSampler. The shard length is equal on every
    # rank (pad-to-even), so the grad-accum window and the optimizer steps
    # stay in lockstep across the DDP allreduce.
    ddp_sampler = None
    if is_distributed():
        local_rank, rank, world_size = dist_env()
        ddp_sampler = build_distributed_sampler(items, config.seed)
        # find_unused_parameters=True: laya's model has parameters that do not
        # contribute to every loss (frozen encoder / unused head paths), so the
        # default reduction bucket never completes and DDP raises "Expected to
        # have finished reduction in the prior iteration".
        if device.type == "cuda":
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)
        print("[perf-patch] ddp: rank %d/%d, %d local items"
              % (rank, world_size, len(ddp_sampler)), flush=True)
    epoch_len = len(ddp_sampler) if ddp_sampler is not None else len(items)
    steps_per_epoch = math.ceil(epoch_len / config.micro_batch)
    updates = max(1, math.ceil(steps_per_epoch / config.grad_accum)
                  * config.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=config.min_lr)
    scaler = (torch.amp.GradScaler("cuda")
              if amp and device.type == "cuda" else None)

    torch.manual_seed(config.seed)
    order_rng = random.Random(config.seed)
    params = [p for g in groups for p in g["params"]]
    history = []
    cache = {}
    hits = lookups = 0
    for epoch in range(config.epochs):
        if ddp_sampler is not None:
            # Per-epoch reseed: every rank shuffles identically then takes a
            # disjoint stride slice, so no item is trained twice per epoch.
            ddp_sampler.set_epoch(epoch)
            epoch_items = [items[i] for i in ddp_sampler]
        else:
            epoch_items = list(items)
            random.Random(config.seed + epoch).shuffle(epoch_items)
        sigma = laya_train.sigma_at(epoch, config.epochs, config.sigma_start,
                                    config.sigma_end)
        total, n_steps = None, 0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(epoch_items), config.micro_batch):
            chunk = []
            for it in epoch_items[start:start + config.micro_batch]:
                order = laya_train.draw_option_order(
                    it, order_rng, config.shuffle_options)
                key = (id(it), tuple(order) if order is not None else None,
                       max_len, head_max_len, parallel)
                encoded = cache.get(key)
                if encoded is None:
                    encoded = laya_train.encode_item(
                        tok, it, max_len, head_max_len, order, parallel)
                    cache[key] = encoded
                else:
                    hits += 1
                lookups += 1
                chunk.append(encoded)
            batch = laya_train.collate_items([chunk], tok.pad_token_id)
            # (2) one device move for what the loop consumes; _forward's own
            # .to(device) on the same device is then a no-op.
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            qtype = batch["qtype"].to(device)
            logits = laya_train._forward(model, batch, device, amp,
                                         config.freeze_encoder)
            if config.loss == "rlcd":
                loss = laya_train.rlcd_loss(logits, target, mask, qtype, sigma,
                                            config.rl_samples, config.w_sph,
                                            config.w_rps)
            else:
                loss = laya_train.soft_ce_loss(logits, target, mask)
            window_start = (n_steps // config.grad_accum) * config.grad_accum
            window_size = min(config.grad_accum,
                              steps_per_epoch - window_start)
            scaled = loss / window_size
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            n_steps += 1
            if (n_steps % config.grad_accum == 0
                    or start + config.micro_batch >= len(epoch_items)):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            # (1) stay on-GPU: accumulate the loss, sync once per epoch.
            detached = loss.detach()
            total = detached if total is None else total + detached
            if config.log_every and n_steps % config.log_every == 0:
                print("epoch %d/%d step %d" % (epoch + 1, config.epochs,
                                               n_steps), flush=True)
        mean = (float(total.item() / max(1, n_steps))
                if total is not None else 0.0)
        # DDP already averages the gradients; report the rank-averaged scalar
        # too so every rank logs the same honest epoch loss.
        if is_distributed():
            import torch.distributed as dist
            mean_tensor = torch.tensor(mean, device=device)
            dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
            mean = float(mean_tensor.item()) / dist_env()[2]
        history.append(mean)
        print("epoch %d/%d mean loss %.4f (encode memo hits %d/%d)"
              % (epoch + 1, config.epochs, mean, hits, lookups), flush=True)
        wandb_log_epoch(epoch, mean)
        if on_epoch_end is not None:
            on_epoch_end(epoch, mean)
    model.eval()
    return history


def apply_perf_patch():
    # Apply BEFORE the device patch so the device wrapper closes over (and
    # preserves) this loop; opt out with ER_LAYA_PERF_PATCH=0.
    if not perf_patch_enabled():
        print("[perf-patch] disabled via " + PERF_PATCH_ENV, flush=True)
        return False
    from laya import train as laya_train
    laya_train.train_model = _perf_train_model
    print("[perf-patch] laya.train.train_model patched: on-GPU loss (1 sync/"
          "epoch), single device move, encode memoization", flush=True)
    return True


def start_gpu_sampler():
    if not perf_patch_enabled():
        return None
    if shutil.which("nvidia-smi") is None:
        log("gpu sampler: nvidia-smi absent; skipping")
        return None
    path = WORKING / "gpu_usage.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader", "-l", "1", "-f", str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("gpu sampler: 1 Hz -> " + str(path))
    return proc


def stop_gpu_sampler(proc):
    if not proc:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def summarize_gpu_usage(path):
    if not path.is_file():
        return None
    utils, mems, total_mb = [], [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            util = float(parts[0].rstrip("%").strip())
            used = float(parts[1].split()[0])
            total = float(parts[2].split()[0])
        except (ValueError, IndexError):
            continue
        utils.append(util)
        mems.append(used)
        total_mb = total
    if not utils:
        return None
    return {
        "samples": len(utils),
        "util_min_pct": min(utils),
        "util_max_pct": max(utils),
        "util_mean_pct": sum(utils) / len(utils),
        "mem_used_peak_mb": max(mems),
        "mem_total_mb": total_mb,
    }
'''


FINETUNE_KERNEL_SCRIPT = '''\
"""ER laya fine-tune on a Kaggle GPU session (cli.laya_lane).

Distributed data-parallel over the 2xT4 pair: when two CUDA devices are
visible `main()` spawns one process per device (nccl), wraps the model in
`DistributedDataParallel` and shards the training items with a per-rank
`DistributedSampler`; a single GPU / CPU falls back to the original
one-process path unchanged (opt out with ER_LAYA_DDP=0). Installs laya over
pip (pinned `laya>=0.3.29`), reads the attached JSONL corpus (train/dev/test
+ receipt, the er-laya-train dataset), extracts the attached base-model
archive (the er-laya-base dataset; the shipped convaiinnovations/laya
checkpoint) to a local dir, builds the FULL `laya.train.TrainConfig` from
the YAML-driven `FINETUNE_CONFIG` (every trainer knob is config SSOT), and
calls `laya.train.finetune(...)` directly with the extracted DIRECTORY as
the base -- so `resolve_checkpoint_dir` takes the isdir branch and NEVER
calls the Hub. Rank 0 alone scores the held-out split, writes the receipt
and stages the checkpoint tar into /kaggle/working for hash-verified
fetch-back.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
TRAIN_JSONL = "@TRAIN_JSONL@"
DEV_JSONL = "@DEV_JSONL@"
TEST_JSONL = "@TEST_JSONL@"
BASE_MODEL_ARCHIVE = "@BASE_MODEL_ARCHIVE@"
BASE_MODEL_DIR = "@BASE_MODEL_DIR@"
FINETUNE_DEVICE = "@FINETUNE_DEVICE@"
FINETUNE_CONFIG = @FINETUNE_CONFIG@
HELD_OUT_BATCH = @HELD_OUT_BATCH@
WANDB_API_KEY = "@WANDB_API_KEY@"
WANDB_PROJECT = "@WANDB_PROJECT@"

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WANDB_RUN = None


def wandb_init():
    """Start the optional wandb mirror (project `tracking.wandb.project`).

    The API key travels baked (read from .env at staging, never from the repo);
    with no key the run stays local and artifacts are the record, exactly like
    the ER tracking contract. Rank 0 only (the caller gates it)."""
    global WANDB_RUN
    if not WANDB_API_KEY:
        print("[wandb] no WANDB_API_KEY; tracking disabled", flush=True)
        return None
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    try:
        import wandb
        WANDB_RUN = wandb.init(project=WANDB_PROJECT, name=RUN_TAG,
                               config=FINETUNE_CONFIG)
        print("[wandb] run " + str(getattr(WANDB_RUN, "id", ""))
              + " -> " + WANDB_PROJECT, flush=True)
    except Exception as error:  # tracking is best-effort, never fatal
        print("[wandb] init skipped: " + type(error).__name__ + ": "
              + str(error)[:200], flush=True)
        WANDB_RUN = None
    return WANDB_RUN


def wandb_log_epoch(epoch, mean):
    if WANDB_RUN is not None:
        WANDB_RUN.log({"epoch": epoch + 1, "train/mean_loss": mean}, step=epoch)


def wandb_log_metrics(report):
    if WANDB_RUN is None or not isinstance(report, dict):
        return
    flat = {}
    for phase in ("before", "after"):
        block = report.get(phase) or {}
        for key in ("accuracy", "loss", "ece", "brier", "brier_top1",
                    "mean_confidence"):
            if block.get(key) is not None:
                flat[phase + "/" + key] = block[key]
    if flat:
        WANDB_RUN.log(flat)


def wandb_finish():
    if WANDB_RUN is not None:
        try:
            WANDB_RUN.finish()
        except Exception:
            pass


@DEVICE_PATCH@

@PERF_PATCH@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_install_laya():
    """laya installs over pip, pinned; torch is already on the session."""
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               "--disable-pip-version-check", LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    """SINGLE T4 ruling: pin the FIRST cuda device only (never 2xT4).

    `auto`/`cuda` require a live cuda session and resolve to device 0; an
    explicit other device (e.g. `cpu`) is passed through while cuda stays
    pinned to device 0, so a second accelerator is never visible."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if FINETUNE_DEVICE not in ("auto", "cuda"):
        log("device configured: " + FINETUNE_DEVICE
            + " (cuda pinned to device 0)")
        return FINETUNE_DEVICE
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    log("device pinned: " + torch.cuda.get_device_name(0)
        + " (single GPU, never a second one)")
    return "cuda"


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name + " (expected the staged "
        "laya finetune dataset)")


def open_zstd(path):
    """Open a `.tar.zst` stream with whichever zstd binding the session has.

    The base checkpoint ships as a zstd tar (the project transport); never
    falls back to the network for the checkpoint itself. Python 3.14 exposes
    `compression.zstd`; the Kaggle 3.13 image needs `zstandard` (installed
    on demand only when neither binding is importable)."""
    try:
        from compression import zstd
        return zstd.open(path, "rb")
    except ImportError:
        pass
    try:
        import zstandard
        return zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-input", "--disable-pip-version-check",
                    "zstandard"], check=True)
    import zstandard
    return zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))


def extract_base_model(archive):
    """Extract the attached base-model tar.zst and return the directory that
    carries rl_agent_config.json.

    `--base` then points at a LOCAL dir, so laya's resolve_checkpoint_dir
    takes the isdir branch and NEVER calls snapshot_download (the HF
    dependency is gone from this path)."""
    destination = WORKING / "base_model"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stream = open_zstd(str(archive))
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            try:
                tar.extractall(destination, filter="data")
            except TypeError:
                tar.extractall(destination)
    finally:
        stream.close()
    candidate = destination / BASE_MODEL_DIR
    if (candidate / "rl_agent_config.json").is_file():
        return candidate
    for found in sorted(destination.rglob("rl_agent_config.json")):
        return found.parent
    raise FileNotFoundError(
        "base-model archive carried no rl_agent_config.json")


def run_laya_finetune(train_path, dev_path, base_model, out_dir, device):
    """Apply the PERF patch then the device patch, build the FULL
    `TrainConfig` from FINETUNE_CONFIG, and call `laya.train.finetune`
    directly.

    The `laya-train` CLI only exposes a subset of the trainer surface, so
    the non-CLI knobs are set by constructing the config here and calling
    `finetune` in-process (the monkeypatches reach the same
    `train_model`/`calibration_records` entrypoints finetune calls). PERF
    first so the device wrapper closes over the patched train_model (see
    FINETUNE_PERF_PATCH_SOURCE)."""
    apply_perf_patch()
    apply_device_patch()
    from laya import train as laya_train
    config = laya_train.TrainConfig(**FINETUNE_CONFIG,
                                    eval_data=str(dev_path))
    config.validate()
    log("TrainConfig: " + json.dumps(FINETUNE_CONFIG, sort_keys=True))
    # laya evaluates the dev split before training and prints nothing while it
    # does (27k items here -> several minutes of silence). Say so, so the quiet
    # stretch is not mistaken for a hang.
    log("starting laya.train.finetune: the pre-train dev evaluation runs "
        "silently until the first 'epoch 1/8 step' line (minutes on the full "
        "corpus)")
    return laya_train.finetune(
        data=str(train_path), model_dir=str(base_model),
        output_dir=str(out_dir), config=config, device=device)


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_held_out(test_path, checkpoint, device):
    """Score the just-trained checkpoint on the HELD-OUT test split.

    DEFAULT-ON (owner order): the training-time eval is the dev split and
    overlaps training/calibration, so this is the run's honest generalization
    number. The raw (uncalibrated) metrics are reported — the calibration was
    already fitted on the calibration split during training. Opt out with
    ER_LAYA_HELD_OUT=0.
    """
    import torch
    from laya import train as laya_train

    model, tok, cfg = laya_train.load_checkpoint(str(checkpoint))
    model = model.to(torch.device(device)).eval()
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    parallel = laya_train.uses_parallel_layout(cfg)
    rows = laya_train.read_jsonl(str(test_path))
    items, skipped = laya_train.items_from_rows(
        tok, rows, max_len, head_max_len, label_smoothing=0.0)
    if not items:
        raise SystemExit(
            "held-out split " + test_path.name + " produced no usable items "
            "(skipped: " + repr(skipped) + ")")
    log("held-out rows " + str(len(rows)) + " -> items " + str(len(items)))
    records = laya_train.calibration_records(
        model, tok, items, device, max_len, head_max_len,
        batch_size=HELD_OUT_BATCH, parallel=parallel)
    return {
        "eval_mode": "held_out",
        "is_held_out": True,
        "eval_source": test_path.name,
        "eval_split": "test",
        "rows": len(rows),
        "items": len(items),
        "skipped": skipped,
        "checkpoint": str(checkpoint),
        "metrics": laya_train.evaluate_records(records),
    }


def session_env():
    # Self-report the container session identity so the host-side log follower
    # can persist logs/kaggle/<kernel>.session_id for the verified in-place
    # stop. Kaggle sets NO KAGGLE_KERNEL_RUN_ID/KAGGLE_SESSION_ID in the
    # container; the only per-run id is the numeric suffix of
    # KAGGLE_CONTAINER_NAME ("kaggle_<token>-<session_id>-webtier"), which the
    # SDK cancel_kernel_session accepts.
    _container = os.environ.get("KAGGLE_CONTAINER_NAME", "")
    _parts = _container.rsplit("-", 2)
    _session_id = (_parts[1] if len(_parts) == 3 and _parts[1].isdigit()
                   else "")
    print("[kaggle-session] session_id=" + _session_id
          + " container=" + _container, flush=True)
    return {"KAGGLE_CONTAINER_NAME": _container, "session_id": _session_id}


def _finetune_session(distributed, session):
    if distributed:
        local_rank, rank, world_size = dist_env()
        device = "cuda"
        log("rank %d/%d on cuda:%d (DDP)" % (rank, world_size, local_rank))
    else:
        device = pick_device()
    train = resolve_input(TRAIN_JSONL)
    dev = resolve_input(DEV_JSONL)
    test = resolve_input(TEST_JSONL)
    log("corpus: " + train.name + " + " + dev.name + " (+ " + test.name + ")")
    WORKING.mkdir(parents=True, exist_ok=True)
    archive = resolve_input(BASE_MODEL_ARCHIVE)
    log("base-model archive: " + str(archive))
    base_model = extract_base_model(archive)
    log("base model: " + str(base_model))
    out_dir = WORKING / "checkpoint"
    if distributed and not is_rank0():
        # Non-rank-0 ranks must NEVER write the canonical checkpoint: suppress
        # laya's per-epoch + final save_checkpoint and send any incidental
        # byte to a private scratch dir (never /kaggle/working/checkpoint).
        from laya import train as laya_train
        laya_train.save_checkpoint = lambda *args, **kwargs: None
        out_dir = WORKING / ("checkpoint.rank" + str(dist_env()[1]))
    if is_rank0():
        wandb_init()
    gpu_handle = start_gpu_sampler() if is_rank0() else None
    try:
        summary = run_laya_finetune(train, dev, base_model, out_dir, device)
    finally:
        stop_gpu_sampler(gpu_handle)

    def rank0_work():
        receipt = {
            "gpu_kind": "finetune",
            "gpu": "T4 (single)",
            "session_env": session,
            "run_tag": RUN_TAG,
            "laya_package": LAYA_PACKAGE,
            "base_model": str(base_model),
            "base_model_archive": str(archive),
            "device": device,
            "perf_patch_enabled": perf_patch_enabled(),
            "ddp": distributed,
            "world_size": dist_env()[2],
            "gpu_usage": summarize_gpu_usage(WORKING / "gpu_usage.log"),
            "recipe": FINETUNE_CONFIG,
            "output_dir": str(WORKING / "checkpoint"),
            "corpus_sha256": {TRAIN_JSONL: sha256_of(train),
                              DEV_JSONL: sha256_of(dev),
                              TEST_JSONL: sha256_of(test)},
        }
        if isinstance(summary, dict):
            for key in ("train_items", "calibration_items", "eval_items",
                        "temperature", "epoch_loss"):
                if key in summary:
                    receipt[key] = summary[key]
        report = WORKING / "checkpoint" / "train_report.json"
        if report.is_file():
            receipt["train_report"] = json.loads(report.read_text())
        # DEFAULT-ON held-out validation (owner order): every fine-tune also
        # scores the just-trained checkpoint on the held-out `test` split, so
        # the receipt always carries the honest generalization number. Opt out
        # with ER_LAYA_HELD_OUT=0; a failure never discards the checkpoint.
        if os.environ.get("ER_LAYA_HELD_OUT", "1").strip().lower() not in (
                "0", "false", "off", "no"):
            try:
                held_out = evaluate_held_out(test, WORKING / "checkpoint",
                                             device)
                (WORKING / "checkpoint" / "held_out_report.json").write_text(
                    json.dumps(held_out, indent=2) + "\\n", encoding="utf-8")
                receipt["held_out"] = held_out
                log("held-out " + held_out["eval_source"] + " items="
                    + str(held_out["items"]) + " accuracy="
                    + str(held_out["metrics"].get("accuracy")))
            except Exception as error:  # keep the checkpoint; surface failure
                receipt["held_out_error"] = (
                    type(error).__name__ + ": " + str(error)[:400])
                log("held-out evaluation FAILED: " + receipt["held_out_error"])
        # Mirror the run to wandb (rank 0 only; no-op without WANDB_API_KEY).
        if isinstance(receipt.get("train_report"), dict):
            wandb_log_metrics(receipt["train_report"])
        if isinstance(receipt.get("held_out"), dict):
            wandb_log_metrics({"after": receipt["held_out"].get("metrics", {})})
        wandb_finish()
        (WORKING / "laya_finetune.receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
        with tarfile.open(WORKING / "laya_finetune.tar.gz", "w:gz",
                          compresslevel=1) as tar:
            for item in sorted(WORKING.iterdir()):
                if item.name not in ("laya_finetune.tar.gz", "base_model"):
                    tar.add(item, arcname=item.name)
        log("staged laya_finetune.tar.gz + receipt in /kaggle/working")

    # Barrier + rank-0-only gate: only rank 0 evaluates the held-out split,
    # writes the receipt and tars /kaggle/working; every rank then tears the
    # process group down.
    run_on_rank0(rank0_work)


def finetune_worker(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    init_distributed()
    try:
        _finetune_session(distributed=True, session=session_env())
    finally:
        destroy_if_distributed()


def main():
    session = session_env()
    pip_install_laya()
    if launch_finetune(finetune_worker) > 1:
        return
    _finetune_session(distributed=False, session=session)


if __name__ == "__main__":
    main()
'''


FINETUNE_EVAL_KERNEL_SCRIPT = '''\
"""ER laya fine-tune EVAL-ONLY on a Kaggle GPU session (cli.laya_lane).

Single T4 per owner ruling: pins one CUDA device, installs laya over pip
(pinned), reads the attached corpus HELD-OUT split + the attached fine-tuned
checkpoint dataset, loads the checkpoint (`laya.train.load_checkpoint`), runs
`calibration_records` + `evaluate_records` on the held-out split, and writes
eval_report.json (before vs after temperature calibration;
eval_mode=held_out, is_held_out=true) + a receipt into /kaggle/working for
hash-verified fetch-back. NO training, NO Hub.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
EVAL_JSONL = "@EVAL_JSONL@"
EVAL_SPLIT = "@EVAL_SPLIT@"
CKPT_DIR_HINT = "@CKPT_DIR@"
CHECKPOINT_PATH = "@CHECKPOINT_PATH@"
BATCH_SIZE = @BATCH_SIZE@
# The YAML-driven calibration/abstention selection (`laya.eval_calibration`):
# `temperature` fits laya's per-type temperature map (on by default), the
# opt-in `abstention` fits the per-bucket `min_confidence` gate, and
# `min_confidence` pins the runtime scalar. Baked as ONE repr literal.
EVAL_CALIBRATION = @EVAL_CALIBRATION@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_install_laya():
    """laya installs over pip, pinned; torch is already on the session."""
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               "--disable-pip-version-check", LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    """SINGLE T4 ruling: pin the FIRST cuda device only (never 2xT4)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    log("device pinned: " + torch.cuda.get_device_name(0)
        + " (single GPU, never a second one)")
    return "cuda"


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name + " (expected the staged "
        "laya eval corpus dataset)")


def resolve_checkpoint():
    """The fine-tuned checkpoint dir: an explicit CHECKPOINT_PATH when baked,
    else the CKPT_DIR_HINT match, else the rl_agent_config.json rglob under
    /kaggle/input. Never a Hub fetch."""
    if CHECKPOINT_PATH:
        candidate = Path(CHECKPOINT_PATH)
        if candidate.is_dir() and (candidate / "rl_agent_config.json").is_file():
            return candidate
        raise FileNotFoundError(
            "CHECKPOINT_PATH carries no rl_agent_config.json: "
            + CHECKPOINT_PATH)
    if CKPT_DIR_HINT:
        for found in sorted(INPUTS.rglob(CKPT_DIR_HINT)):
            if (found.is_dir()
                    and (found / "rl_agent_config.json").is_file()):
                return found
    for found in sorted(INPUTS.rglob("rl_agent_config.json")):
        return found.parent
    raise FileNotFoundError(
        "attached inputs carried no fine-tuned checkpoint "
        "(rl_agent_config.json); attach the checkpoint dataset")


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_eval_calibration(laya_train, records):
    """Fit laya's OWN calibration per the baked `EVAL_CALIBRATION` (SSOT).

    Consumes `fit_temperature_map` (per-type temperature sequence + the
    per-bucket map) and, opt-in, `fit_abstention_thresholds` (the per-bucket
    `min_confidence` gate); nothing is reimplemented here. The kernel sets
    `default` on the thresholds map only when `min_confidence` is pinned,
    because an explicit operator pin beats the fit's implied default.
    """
    level = EVAL_CALIBRATION or {}
    temperature = temperature_by_options = n_by_bucket = None
    if level.get("temperature", True):
        fitted = laya_train.fit_temperature_map(records)
        temperature = fitted.get("temperature")
        temperature_by_options = fitted.get("temperature_by_options")
        n_by_bucket = fitted.get("n_by_bucket")
    thresholds = {}
    if level.get("abstention"):
        thresholds = dict(laya_train.fit_abstention_thresholds(
            records, temperature, temperature_by_options or {},
            target_error=level.get("target_error", 0.10),
            min_bucket_n=level.get("min_abstain_n", 10)) or {})
    min_confidence = level.get("min_confidence")
    if min_confidence is not None:
        thresholds["default"] = min_confidence
    return {
        "temperature": temperature,
        "temperature_by_options": temperature_by_options,
        "n_by_bucket": n_by_bucket,
        "abstention_thresholds": thresholds,
        "min_confidence": min_confidence,
    }


def main():
    pip_install_laya()
    device = pick_device()
    import torch
    from laya import train as laya_train
    eval_path = resolve_input(EVAL_JSONL)
    checkpoint = resolve_checkpoint()
    log("checkpoint: " + str(checkpoint))
    log("held-out split: " + str(eval_path) + " (" + EVAL_SPLIT + ")")
    model, tok, cfg = laya_train.load_checkpoint(str(checkpoint))
    model = model.to(torch.device(device)).eval()
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    parallel = laya_train.uses_parallel_layout(cfg)
    rows = laya_train.read_jsonl(str(eval_path))
    items, skipped = laya_train.items_from_rows(
        tok, rows, max_len, head_max_len, label_smoothing=0.0)
    if not items:
        raise SystemExit(
            "eval split " + eval_path.name + " produced no usable items "
            "(skipped: " + repr(skipped) + ")")
    log("held-out rows " + str(len(rows)) + " -> items " + str(len(items)))
    records = laya_train.calibration_records(
        model, tok, items, device, max_len, head_max_len,
        batch_size=BATCH_SIZE, parallel=parallel)
    before = laya_train.evaluate_records(records)
    calibration = fit_eval_calibration(laya_train, records)
    after = laya_train.evaluate_records(
        records, calibration["temperature"],
        calibration["temperature_by_options"])
    comparison = {
        "delta_accuracy": round(
            after["accuracy"] - before["accuracy"], 4),
        "delta_ece": (round(after["ece"] - before["ece"], 4)
                      if after["ece"] is not None
                      and before["ece"] is not None else None),
        "delta_brier": (round(after["brier"] - before["brier"], 4)
                        if after["brier"] is not None
                        and before["brier"] is not None else None),
        "delta_mean_confidence": round(
            after["mean_confidence"] - before["mean_confidence"], 4),
    }
    report = {
        "eval_mode": "held_out",
        "is_held_out": True,
        "eval_source": eval_path.name,
        "eval_split": EVAL_SPLIT,
        "rows": len(rows),
        "items": len(items),
        "skipped": skipped,
        "checkpoint": str(checkpoint),
        "run_tag": RUN_TAG,
        "before": before,
        "after": after,
        "comparison": comparison,
        "temperature": calibration["temperature"],
        "temperature_by_options": calibration["temperature_by_options"],
    }
    # Additive: the opt-in knobs alone add keys, so a default config keeps the
    # landed report shape byte-for-byte.
    if EVAL_CALIBRATION.get("abstention") or calibration["min_confidence"] is not None:
        report["n_by_bucket"] = calibration["n_by_bucket"] or {}
    if calibration["abstention_thresholds"]:
        report["abstention_thresholds"] = calibration["abstention_thresholds"]
    if calibration["min_confidence"] is not None:
        report["min_confidence"] = calibration["min_confidence"]
    WORKING.mkdir(parents=True, exist_ok=True)
    report_path = WORKING / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\\n",
                           encoding="utf-8")
    log("wrote " + str(report_path) + " (accuracy before/after "
        + str(before["accuracy"]) + "/" + str(after["accuracy"]) + ")")
    receipt = {
        "gpu_kind": "finetune-eval",
        "gpu": "T4 (single)",
        "run_tag": RUN_TAG,
        "laya_package": LAYA_PACKAGE,
        "eval_split": EVAL_SPLIT,
        "eval_mode": "held_out",
        "is_held_out": True,
        "eval_jsonl_sha256": sha256_of(eval_path),
        "checkpoint": str(checkpoint),
        "eval_calibration": EVAL_CALIBRATION,
        "report_sha256": sha256_of(report_path),
    }
    (WORKING / "laya_finetune-eval.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    with tarfile.open(WORKING / "laya_finetune_eval.tar.gz", "w:gz",
                      compresslevel=1) as tar:
        for item in sorted(WORKING.iterdir()):
            if item.name != "laya_finetune_eval.tar.gz":
                tar.add(item, arcname=item.name)
    log("staged eval_report.json + receipt in /kaggle/working")


if __name__ == "__main__":
    main()
'''


def stage_dataset_payload(decision_kind: str, *, dataset_slug: str,
                          question_source: Path,
                          decision_source: Path) -> dict[str, Any]:
    """Stage the DATASET payload for spec.dataset_slug (dry-safe).

    Builds results/laya_lane/kaggle/<decision>/dataset_payload/: the
    kaggle `dataset-metadata.json` (title/id/licenses per the kaggle-lane
    payload shape) + copies of the staged laya.question.json and the
    staged decision CSV RENAMED to dataset.csv (the DECISION_CSV name the
    kernel resolves via INPUTS.rglob once the dataset attaches).

    Boundary (deliberately NOT a `Bundle`): a kaggle dataset payload is a
    DIRECTORY, so there is no single container to carry a Bundle manifest;
    integrity is the receipt's `files` inventory (the same name -> sha256
    shape `Bundle`'s manifest uses), re-verified by the consumer after the
    dataset attaches. Forcing an archive here would change what kaggle
    versions and what the kernel resolves by name.
    """
    if not dataset_slug:
        raise RuntimeError(
            "config laya.dataset_slug is unset; name the input dataset "
            "(owner/slug) before staging")
    stage = staging_dir() / "kaggle" / decision_kind / DATASET_PAYLOAD_DIR
    stage.mkdir(parents=True, exist_ok=True)
    metadata = {"title": "er laya requests", "id": dataset_slug,
                "licenses": [{"name": "other"}]}
    atomic_write_json(metadata, stage / DATASET_METADATA_FILE)
    shutil.copy2(question_source, stage / QUESTION_SCHEMA_FILE)
    shutil.copy2(decision_source, stage / DATASET_CSV_NAME)
    payload_files = (QUESTION_SCHEMA_FILE, DATASET_CSV_NAME)
    receipt = {
        "dataset": dataset_slug,
        "payload": str(stage),
        "metadata": metadata,
        "files": {name: sha256_file(stage / name) for name in payload_files},
    }
    atomic_write_json(receipt, stage / "dataset_payload.receipt.json")
    _log_lane(f"staged dataset payload [{decision_kind}] {dataset_slug} "
              f"files={list(payload_files)} -> {stage}")
    return receipt


def package_base_model(*, source_dir: Path, dataset_slug: str,
                       archive_name: str, member_name: str,
                       output_dir: Path | None = None) -> dict[str, Any]:
    """Package the local base checkpoint tree as a sealed `.tar.zst` dataset payload.

    The fine-tune base checkpoint is 647 MB (plain git caps at 100 MB), so
    it ships the way the project ships large payloads: a zstd tar attached
    as a kaggle dataset. The archive seals with every source under ONE
    top-level member named `member_name`, so extraction yields a dir
    carrying `rl_agent_config.json` (exactly what laya's
    resolve_checkpoint_dir needs to take the local-dir branch).

    Sealing goes through `Bundle.seal_archive` (the shared writer): it hashes
    each source once while writing, verifies the written bytes, and returns
    the whole-file digest on the handle, so the receipt's integrity token is
    the boundary token — no second read of the 647 MB archive. The extra
    `BASE_MODEL_MANIFEST_FILE` member is the Bundle role manifest;
    `extract_base_model` resolves the model dir by `rl_agent_config.json`, so
    the manifest never disturbs extraction. The kaggle
    `dataset-metadata.json` lands beside the archive. Results live under
    results/laya_lane/base_model (never committed).
    """
    source_dir = Path(source_dir)
    if not (source_dir / "rl_agent_config.json").is_file():
        raise FileNotFoundError(
            f"base-model source {source_dir} carries no rl_agent_config.json")
    if not dataset_slug:
        raise RuntimeError(
            "config laya.base_model_dataset is unset; name the base-model "
            "dataset (owner/slug) before packaging")
    stage = Path(output_dir) if output_dir else staging_dir() / "base_model"
    stage.mkdir(parents=True, exist_ok=True)
    archive_path = stage / archive_name
    if archive_path.exists():
        archive_path.unlink()
    files = {f"{member_name}/{path.relative_to(source_dir).as_posix()}": path
             for path in sorted(source_dir.rglob("*")) if path.is_file()}
    if not files:
        raise FileNotFoundError(
            f"base-model source {source_dir} carries no files to package")
    sealed = Bundle.seal_archive(
        archive_path, files, role=BundleRole.inputs,
        manifest_name=BASE_MODEL_MANIFEST_FILE,
        metadata={"schema": "er-laya-base-model-v1", "member": member_name,
                  "source": str(source_dir)})
    metadata = {"title": "er laya base", "id": dataset_slug,
                "licenses": [{"name": "other"}]}
    atomic_write_json(metadata, stage / DATASET_METADATA_FILE)
    receipt = {
        "dataset": dataset_slug,
        "payload": str(stage),
        "archive": archive_name,
        "member": member_name,
        "source": str(source_dir),
        "bundle_role": BundleRole.inputs.value,
        "manifest": BASE_MODEL_MANIFEST_FILE,
        "bytes": archive_path.stat().st_size,
        "sha256": sealed.digest,
        "metadata": metadata,
    }
    atomic_write_json(receipt, stage / "base_model.receipt.json")
    _log_lane(f"packaged base model {dataset_slug} member={member_name} "
              f"archive={archive_name} bytes={receipt['bytes']} -> {stage}")
    return receipt


def publish_laya_dataset(decision_kind: str, *, run_tag: str,
                         execute: bool) -> dict[str, Any]:
    """`--execute`-gated create-or-version of the laya inputs dataset.

    The staged play_500.csv + laya.question.json do NOT travel with
    `kaggle kernels push`: the kernel attaches spec.dataset_slug
    (fbarulli/er-laya-requests), so the dataset must exist remotely
    BEFORE the push. Dry run: returns the plan, never spawns a kaggle
    subprocess. Executed: datasets create when the dataset does not
    exist remotely, else datasets version (-r --dir-mode zip -m
    "laya inputs <tag>"); the helpers are IMPORTED from the kaggle lane
    (cli.kaggle_datasets / cli.kaggle_lane), never copied. The dataset
    version is recorded in the decision receipt.
    """
    payload = staging_dir() / "kaggle" / decision_kind / DATASET_PAYLOAD_DIR
    metadata_file = payload / DATASET_METADATA_FILE
    plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                            "payload": str(payload)}
    if not execute:
        plan["note"] = ("the dataset attach rides --execute only "
                        "(mirroring the kernels-push gate)")
        _log_lane(f"dry-run: dataset payload for {decision_kind} would "
                  f"publish to the remote surface at {payload}")
        return plan
    if not metadata_file.is_file():
        raise RuntimeError(
            "--activate gate: no staged dataset payload at "
            f"{payload} ({DATASET_METADATA_FILE} is missing); stage first")
    corpus_kind = decision_kind in (FINETUNE_DECISION, FINETUNE_EVAL_DECISION)
    if decision_kind == HOLDOUT_EVAL_DECISION:
        slug, key = _spec().holdout_dataset_slug, "holdout_dataset_slug"
    else:
        slug = (_spec().finetune_dataset_slug if corpus_kind
                else _spec().dataset_slug)
        key = "finetune_dataset_slug" if corpus_kind else "dataset_slug"
    plan["slug"] = slug
    if not slug:
        raise RuntimeError(
            f"config laya.{key} is unset; name the dataset (owner/slug) "
            "before an executed attach")
    from cli import kaggle_lane as lane
    from cli.kaggle_datasets import KaggleDatasets

    executable = lane._require_kaggle_executable(
        lane._spec().kaggle_executable)
    current = KaggleDatasets._dataset_current_version(slug)
    version = current.get("dataset_version")
    if version:
        plan["action"] = "version"
        # `-r` and `--dir-mode` are one argparse option: `-r --dir-mode
        # zip` fails with "argument -r/--dir-mode: expected one
        # argument" (fail-loud met live on the version path).
        command = [executable, "datasets", "version", "-r", "zip",
                   "-m", f"laya inputs {run_tag}",
                   "-p", str(payload)]
    else:
        plan["action"] = "create"
        command = [executable, "datasets", "create", "-p", str(payload)]
    plan["command"] = command
    _, _ = lane._run_kaggle(command)
    plan["returncode"] = 0
    refreshed = KaggleDatasets._dataset_current_version(slug)
    plan["dataset_version"] = refreshed.get("dataset_version") or version
    plan["published"] = True
    receipt_path = staging_dir() / "kaggle" / decision_kind \
        / f"{decision_kind}.receipt.json"
    if receipt_path.is_file():
        body = json.loads(receipt_path.read_text(encoding="utf-8"))
        body["dataset"].update({"action": plan["action"],
                                "version": plan["dataset_version"]})
        atomic_write_json(body, receipt_path)
    _log_lane(f"published dataset {slug} | action={plan['action']} "
              f"version={plan['dataset_version']} rc=0")
    return plan


def stage_decision_kernel(*, decision_kind: str, revision: str | None = None,
                          run_tag: str | None = None,
                          input_override: Path | None = None,
                          checkpoint_path: Path | None = None
                          ) -> dict[str, Any]:
    """Stage the kaggle decision kernel payload (dry-safe).

    Writes under results/laya_lane/kaggle/<decision_kind>/:
      kernel-metadata.json + <code_file>.py + <decision_kind>.receipt.json
      (+ the staged question schema + decision input receipts).
    Fail-loud preconditions (no silent skip):
      * spec.laya_decision_epochs > 0 (0 = disabled, nothing may stage);
      * spec.export_dataset_slug set (the target kernel owner/slug);
      * the question schema + decision CSV stage from their SSOT bindings.
    """
    spec = _spec()
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    if decision_kind == FINETUNE_DECISION:
        # The fine-tune kind is corpus-driven (JSONL dataset), not a
        # per-row decision CSV: it has its own staging surface, reached
        # through the same `--decision` dispatch.
        return stage_finetune_kernel(revision=revision, run_tag=run_tag)
    if decision_kind == FINETUNE_EVAL_DECISION:
        # The eval-only kind is corpus- + checkpoint-driven: it has its
        # own staging surface, reached through the same `--decision`
        # dispatch. No training, no Hub.
        return stage_finetune_eval_kernel(revision=revision, run_tag=run_tag,
                                          checkpoint_path=checkpoint_path)
    if spec.laya_decision_epochs <= 0:
        raise RuntimeError(
            "config laya.laya_decision_epochs <= 0: the decision lane is "
            "disabled (no payload may stage a GPU session)")
    slug = spec.export_dataset_slug
    if not slug:
        raise RuntimeError(
            "config laya.export_dataset_slug is unset; name the target "
            "kernel (owner/slug) before staging")
    dataset_slug = spec.dataset_slug
    if not dataset_slug:
        raise RuntimeError(
            "config laya.dataset_slug is unset; the kernel inputs travel "
            "as that dataset (owner/slug) — the staged play_500.csv + "
            "laya.question.json do NOT ride `kaggle kernels push`; "
            "name it before staging")
    # ── the published-tip invariant ('origin/<branch> == HEAD'): the pin
    # resolves BEFORE any payload write; a pin that misses the fetched
    # branch tip never stages (the 84ce2d0-vs-02dec14 staged-race class).
    repository = training_cfg().kaggle.repository
    branch = training_cfg().kaggle.branch
    revision = revision or _git_revision()
    from core import runtime_inputs
    tip = runtime_inputs.require_published_tip_match(
        revision, repository, branch)
    question = stage_question_schema("kaggle")
    input_receipt = stage_decision_input("kaggle",
                                         decision_kind=decision_kind,
                                         override=input_override)
    dataset_receipt = stage_dataset_payload(
        decision_kind, dataset_slug=dataset_slug,
        question_source=Path(question["staged"]),
        decision_source=Path(input_receipt["staged"]))
    stage = staging_dir() / "kaggle" / decision_kind
    stage.mkdir(parents=True, exist_ok=True)
    code_file = (DECISION_KERNEL_CODE_FILE if decision_kind != "laya-cli-eval"
                 else EVAL_KERNEL_CODE_FILE)
    template = (DECISION_KERNEL_SCRIPT if decision_kind != "laya-cli-eval"
                else EVAL_KERNEL_SCRIPT)
    tag = run_tag or spec.run_tag_prefix + decision_tag()
    metadata: dict[str, Any] = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        # single T4: the payload never requests the double accelerator;
        # the script itself pins CUDA_VISIBLE_DEVICES=0.
        "enable_internet": True,
        # THE INPUTS TRAVEL AS THE DATASET: kernels push does NOT ship
        # the co-located csv/schema files, so resolve_input would
        # FileNotFoundError once boot passes — attach the dataset slug.
        "dataset_sources": [dataset_slug],
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    entry = DECISION_BINDINGS[decision_kind]
    staged_csv = input_receipt["staged"]
    values = {
        "LAYA_PACKAGE": spec.laya_package,
        "CHECKPOINT_HUB": spec.checkpoint_hub,
        "DECISION_KIND": decision_kind,
        "RUN_TAG": tag,
        "DECISION_CSV": DATASET_CSV_NAME,
        "STATE_COLUMN": entry["state_column"],
        "BATCH_SIZE": str(spec.laya_decision_batch_size),
        "MIN_CONFIDENCE": repr(spec.min_router_confidence),
        "QUESTION_SCHEMA_FILE": repr(QUESTION_SCHEMA_FILE),
        "REPOSITORY": repository,
        "BRANCH": branch,
        "REVISION": revision,
    }
    # two-pass substitution (a nested value's @tokens@ are never
    # re-scanned once it is inserted): the preflight bakes its own
    # literal tuple FIRST, then drops into the script — the push gate
    # (_staged_laya_push_preflight) literal-evals `_runtime_files`.
    preflight = _template(LAYA_RUNTIME_PREFLIGHT, values)
    script = _template(template, {**values,
                                  "RUNTIME_PREFLIGHT": preflight})
    _kernel_script_gate(script)
    _module_scope_gate(script)
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / code_file).write_text(script, encoding="utf-8")
    receipt = {
        "kernel": slug,
        "kind": decision_kind,
        "gpu": "T4 (single)",
        "run_tag": tag,
        "staged": str(stage),
        "code_file": code_file,
        "question_schema": question["staged"],
        "question_sha256": question["sha256"],
        "decision_input": staged_csv,
        "decision_sha256": input_receipt["sha256"],
        "dataset": {"slug": dataset_slug,
                    "payload": dataset_receipt["payload"],
                    "files": dataset_receipt["files"]},
        "checkpoint_hub": spec.checkpoint_hub,
        "batch_size": spec.laya_decision_batch_size,
        "min_confidence": spec.min_router_confidence,
        "epochs": spec.laya_decision_epochs,
        "state_column": entry["state_column"],
        "evals_enabled": bool(spec.laya_evals_enabled),
        "calibration": bool(spec.calibration),
        "onnx": bool(spec.onnx),
        "published_pin": {"repository": repository, "branch": branch,
                          "revision": revision},
        "published_tip": tip,
    }
    # the labeled decision csv's metric contract rides the staged receipt
    # (expected rows + label distribution + the gold columns) so the
    # harvest computes accuracy/F1 without re-deriving the expectation
    receipt.update({key: input_receipt[key]
                    for key in _METRIC_EXPECTATION_KEYS
                    if key in input_receipt})
    atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
    # The decision csv is already co-located in the payload dir (the
    # stage-decision-input destination IS staging/<kind>/<decision>/);
    # the question schema lands beside it for the payload dataset bind.
    shutil.copy2(question["staged"], stage / QUESTION_SCHEMA_FILE)
    _log_lane(f"staged kaggle kernel [{decision_kind}] ({spec.gpu}) "
              f"run_tag={tag} -> {stage}")
    return receipt


def stage_finetune_dataset_payload(*, dataset_slug: str,
                                   corpus_dir: Path,
                                   kind: str = FINETUNE_DECISION
                                   ) -> dict[str, Any]:
    """Stage the fine-tune CORPUS as a kaggle dataset payload (dry-safe).

    Builds results/laya_lane/kaggle/<kind>/dataset_payload/: the kaggle
    `dataset-metadata.json` + the three split JSONL + the builder receipt.
    Distinct from the decision datasets (spec.dataset_slug) so a corpus
    version never drops the decision inputs (and vice versa). `kind` names
    the staging surface (`finetune` or the eval-only `finetune-eval`), so
    each kernel's push gate finds its own `dataset_payload` beside it.
    """
    if not dataset_slug:
        raise RuntimeError(
            "config laya.finetune_dataset_slug is unset; name the corpus "
            "dataset (owner/slug) before staging")
    corpus_dir = Path(corpus_dir)
    stage = staging_dir() / "kaggle" / kind / DATASET_PAYLOAD_DIR
    # Boundary (deliberately NOT a `Bundle`): like the decision payload, this
    # is a kaggle dataset DIRECTORY (the corpus JSONL + the corpus receipt),
    # not a sealed archive - kaggle versions the directory, so the receipt's
    # `files` inventory is the integrity token the kernel re-checks once the
    # dataset attaches.
    stage.mkdir(parents=True, exist_ok=True)
    metadata = {"title": "er laya train", "id": dataset_slug,
                "licenses": [{"name": "other"}]}
    atomic_write_json(metadata, stage / DATASET_METADATA_FILE)
    files = list(FINETUNE_CORPUS_FILES) + [FINETUNE_CORPUS_RECEIPT]
    for name in files:
        source = corpus_dir / name
        if not source.is_file():
            raise FileNotFoundError(
                f"fine-tune corpus file not found: {source} (build it with "
                "scripts/laya_build_dataset.py)")
        shutil.copy2(source, stage / name)
    receipt = {
        "dataset": dataset_slug,
        "payload": str(stage),
        "metadata": metadata,
        "files": {name: sha256_file(stage / name) for name in files},
    }
    atomic_write_json(receipt, stage / "dataset_payload.receipt.json")
    _log_lane(f"staged finetune dataset payload {dataset_slug} "
              f"files={files} -> {stage}")
    return receipt


def _pairs_composer():
    """The corpus pair composer (scripts/laya_metrics_pairs.py; reused, not copied)."""
    import importlib.util

    path = TRAIN_ROOT / "scripts/laya_metrics_pairs.py"
    spec = importlib.util.spec_from_file_location("laya_metrics_pairs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv_rows(path: Path) -> list[dict]:
    import csv

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _norm_gtin(value: object) -> str:
    from training.folds import normalize_gtin

    return normalize_gtin(value)


def stage_holdout_dataset_payload(*, dataset_slug: str, holdout_csv: Path,
                                  catalog_path: Path, question_path: Path
                                  ) -> dict[str, Any]:
    """Stage the component-disjoint holdout as a kaggle dataset payload (dry-safe).

    Builds results/laya_lane/kaggle/<decision>/dataset_payload/: the kaggle
    `dataset-metadata.json`, ONE ``holdout.jsonl`` (each labelled pair composed
    into laya's identity state with its stratum/component tags), plus a receipt.
    Only labelled rows travel — the gate strata are label-less difficulty tags,
    not truth, and must not enter the checkpoint's held-out score.
    """
    if not dataset_slug:
        raise RuntimeError(
            "config laya.holdout_dataset_slug is unset; name the holdout "
            "dataset (owner/slug) before staging")
    questions = json.loads(
        Path(question_path).read_text(encoding="utf-8"))["questions"]
    composer = _pairs_composer()
    rows = _csv_rows(Path(holdout_csv))
    by_gtin: dict[str, dict] = {}
    for row in _csv_rows(Path(catalog_path)):
        by_gtin.setdefault(_norm_gtin(row.get("gtin")), row)
    lines, skipped = [], 0
    for row in rows:
        label = str(row.get("label", "")).strip()
        one = by_gtin.get(_norm_gtin(row.get("gtin1")))
        two = by_gtin.get(_norm_gtin(row.get("gtin2")))
        if label not in ("0", "1") or one is None or two is None:
            skipped += 1
            continue
        state = composer.compose_state(
            composer.compose_side(one["attribute"]),
            composer.compose_side(two["attribute"]))
        lines.append({
            "state": state, "questions": questions,
            "expected": {"identity_claim": "true" if label == "1" else "false"},
            "stratum": str(row.get("stratum", "")),
            "component": str(row.get("component", "")),
        })
    if not lines:
        raise RuntimeError(
            f"holdout {holdout_csv} produced no labelled rows; build it with "
            "scripts/laya_holdout.py first")
    stage = staging_dir() / "kaggle" / HOLDOUT_EVAL_DECISION / DATASET_PAYLOAD_DIR
    stage.mkdir(parents=True, exist_ok=True)
    metadata = {"title": "er laya holdout", "id": dataset_slug,
                "licenses": [{"name": "other"}]}
    atomic_write_json(metadata, stage / DATASET_METADATA_FILE)
    body = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)
    (stage / HOLDOUT_JSONL).write_text(body, encoding="utf-8")
    receipt = {
        "dataset": dataset_slug, "payload": str(stage), "rows": len(lines),
        "skipped": skipped, "question_schema": str(question_path),
        "files": {HOLDOUT_JSONL: sha256_file(stage / HOLDOUT_JSONL)},
    }
    atomic_write_json(receipt, stage / "dataset_payload.receipt.json")
    _log_lane(f"staged holdout dataset payload {dataset_slug} "
              f"rows={len(lines)} (skipped {skipped}) -> {stage}")
    return receipt


HOLDOUT_EVAL_KERNEL_SCRIPT = '''\
"""ER laya holdout verification on Kaggle (cli.laya_lane).

Single T4: installs laya over pip, attaches the staged component-disjoint
holdout (JSONL of composed identity states + labels + strata) and the fine-tuned
checkpoint dataset, scores each pair's identity_claim with the checkpoint, and
writes holdout_report.json: overall + per-stratum precision/recall/F1/PR-AUC at
the configured threshold, each with a COMPONENT-clustered bootstrap CI (real
held-out verification, never the in-sample training eval). NO training, NO Hub.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
HOLDOUT_JSONL = "@HOLDOUT_JSONL@"
CKPT_DIR_HINT = "@CKPT_DIR@"
BATCH_SIZE = @BATCH_SIZE@
THRESHOLD = @THRESHOLD@
N_BOOT = @N_BOOT@
SEED = @SEED@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_install_laya():
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               "--disable-pip-version-check", LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    log("device pinned: " + torch.cuda.get_device_name(0))
    return "cuda"


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError("attached inputs carried no " + name)


def resolve_checkpoint():
    if CKPT_DIR_HINT:
        for found in sorted(INPUTS.rglob(CKPT_DIR_HINT)):
            if found.is_dir() and (found / "rl_agent_config.json").is_file():
                return found
    for found in sorted(INPUTS.rglob("rl_agent_config.json")):
        return found.parent
    raise FileNotFoundError("attached inputs carried no checkpoint")


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(labels, scores, threshold):
    tp = fp = tn = fn = 0
    for y, s in zip(labels, scores):
        pred = 1 if s >= threshold else 0
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and y:
            fn += 1
        else:
            tn += 1
    def pr(tp, fp):
        return tp / (tp + fp) if (tp + fp) else 0.0
    def rc(tp, fn):
        return tp / (tp + fn) if (tp + fn) else 0.0
    precision, recall = pr(tp, fp), rc(tp, fn)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    n = tp + fp + tn + fn
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / n if n else None,
            "precision": precision, "recall": recall, "f1": f1}


def pr_auc(labels, scores):
    import numpy as np
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return None
    order = np.argsort(-np.asarray(scores, dtype=float))
    hits = labels[order]
    cum = np.cumsum(hits)
    precision = cum / (np.arange(len(hits)) + 1)
    return float((precision * hits).sum() / hits.sum())


def bootstrap_ci(labels, scores, components, stat, n_boot, seed, alpha=0.05):
    import numpy as np
    components = np.asarray(components, dtype=object)
    labels, scores = np.asarray(labels), np.asarray(scores)
    unique = np.unique(components)
    rows_of = {c: np.where(components == c)[0] for c in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(n_boot)):
        picks = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([rows_of[c] for c in picks])
        value = stat(labels[idx], scores[idx])
        if value is not None:
            samples.append(float(value))
    point = stat(labels, scores)
    if not samples:
        return {"point": point, "lo": None, "hi": None}
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi)}


def block(labels, scores, components, threshold):
    m = binary_metrics(labels, scores, threshold)
    def _f1(y, s):
        return binary_metrics(y, s, threshold)["f1"]
    def _precision(y, s):
        return binary_metrics(y, s, threshold)["precision"]
    def _recall(y, s):
        return binary_metrics(y, s, threshold)["recall"]
    return {
        "metrics": m,
        "pr_auc": pr_auc(labels, scores),
        "cis": {
            "precision": bootstrap_ci(labels, scores, components, _precision, N_BOOT, SEED),
            "recall": bootstrap_ci(labels, scores, components, _recall, N_BOOT, SEED),
            "f1": bootstrap_ci(labels, scores, components, _f1, N_BOOT, SEED),
            "pr_auc": bootstrap_ci(labels, scores, components, pr_auc, N_BOOT, SEED),
        },
    }


def main():
    pip_install_laya()
    device = pick_device()
    import numpy as np
    import torch
    from laya import train as laya_train
    holdout = resolve_input(HOLDOUT_JSONL)
    checkpoint = resolve_checkpoint()
    log("holdout: " + str(holdout))
    log("checkpoint: " + str(checkpoint))
    rows = [json.loads(line) for line in holdout.read_text().splitlines()
            if line.strip()]
    model, tok, cfg = laya_train.load_checkpoint(str(checkpoint))
    model = model.to(torch.device(device)).eval()
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    parallel = laya_train.uses_parallel_layout(cfg)
    items, skipped = laya_train.items_from_rows(
        tok, rows, max_len, head_max_len, label_smoothing=0.0)
    if len(items) != len(rows):
        raise SystemExit("holdout items %d != rows %d (skipped %s)"
                         % (len(items), len(rows), repr(skipped)))
    records = laya_train.calibration_records(
        model, tok, items, device, max_len, head_max_len,
        batch_size=BATCH_SIZE, parallel=parallel)
    scores = []
    for _qt, logits, _t, _k in records:
        z = np.asarray(logits, dtype=float)
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        scores.append(float(p[1]))
    labels = [1 if str(r.get("expected", {}).get("identity_claim")).lower()
              in ("true", "1") else 0 for r in rows]
    strata = [str(r.get("stratum") or "unknown") for r in rows]
    components = [str(r.get("component") or r["state"][:24]) for r in rows]
    by_stratum = {}
    for name in sorted(set(strata)):
        idx = [i for i, s in enumerate(strata) if s == name]
        by_stratum[name] = block([labels[i] for i in idx],
                                 [scores[i] for i in idx],
                                 [components[i] for i in idx], THRESHOLD)
    report = {
        "eval_mode": "holdout",
        "is_held_out": True,
        "checkpoint": str(checkpoint),
        "run_tag": RUN_TAG,
        "rows": len(rows),
        "skipped": skipped,
        "threshold": THRESHOLD,
        "n_boot": N_BOOT,
        "overall": block(labels, scores, components, THRESHOLD),
        "by_stratum": by_stratum,
    }
    WORKING.mkdir(parents=True, exist_ok=True)
    out = WORKING / "holdout_report.json"
    out.write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
    receipt = {
        "gpu_kind": "holdout-eval",
        "run_tag": RUN_TAG,
        "device": device,
        "checkpoint": str(checkpoint),
        "holdout": str(holdout),
        "holdout_sha256": sha256_of(holdout),
        "rows": len(rows),
        "overall_accuracy": report["overall"]["metrics"]["accuracy"],
        "overall_f1": report["overall"]["metrics"]["f1"],
        "report": str(out),
    }
    (WORKING / "laya_holdout-eval.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    with tarfile.open(WORKING / "laya_holdout-eval.tar.gz", "w:gz",
                      compresslevel=1) as tar:
        for item in sorted(WORKING.iterdir()):
            if item.name != "laya_holdout-eval.tar.gz":
                tar.add(item, arcname=item.name)
    log("overall accuracy=" + str(report["overall"]["metrics"]["accuracy"])
        + " f1=" + str(report["overall"]["metrics"]["f1"]))
    print("[holdout-eval] report: " + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''


def stage_holdout_eval_kernel(*, revision: str | None = None,
                              run_tag: str | None = None) -> dict[str, Any]:
    """Stage the holdout-verification kaggle kernel (dry-safe).

    Writes under results/laya_lane/kaggle/holdout-eval/: kernel-metadata.json +
    laya_holdout_eval.py + holdout-eval.receipt.json (+ the staged holdout
    dataset payload). Attaches the holdout dataset + the fine-tuned checkpoint
    dataset.
    """
    spec = _spec()
    slug = spec.holdout_eval_kernel_slug
    if not slug:
        raise RuntimeError(
            "config laya.holdout_eval_kernel_slug is unset; name the target "
            "kernel (owner/slug) before staging")
    dataset_slug = spec.holdout_dataset_slug
    if not dataset_slug:
        raise RuntimeError(
            "config laya.holdout_dataset_slug is unset; the holdout travels as "
            "that dataset (owner/slug) — name it before staging")
    ckpt_dataset = spec.finetune_ckpt_dataset
    repository = training_cfg().kaggle.repository
    branch = training_cfg().kaggle.branch
    revision = revision or _git_revision()
    from core import runtime_inputs

    tip = runtime_inputs.require_published_tip_match(
        revision, repository, branch)
    question_path = TRAIN_ROOT / spec.question_schema
    dataset_receipt = stage_holdout_dataset_payload(
        dataset_slug=dataset_slug,
        holdout_csv=TRAIN_ROOT / spec.holdout_csv,
        catalog_path=TRAIN_ROOT / "data/track_setup/eligible_catalog.csv",
        question_path=question_path)
    stage = staging_dir() / "kaggle" / HOLDOUT_EVAL_DECISION
    stage.mkdir(parents=True, exist_ok=True)
    tag = run_tag or spec.run_tag_prefix + decision_tag()
    dataset_sources = [dataset_slug]
    if ckpt_dataset:
        dataset_sources.append(ckpt_dataset)
    metadata: dict[str, Any] = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
        "code_file": HOLDOUT_EVAL_CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    values = {
        "LAYA_PACKAGE": spec.finetune_package,
        "RUN_TAG": tag,
        "HOLDOUT_JSONL": HOLDOUT_JSONL,
        "CKPT_DIR": spec.finetune_ckpt_dir,
        "BATCH_SIZE": str(spec.holdout_eval_batch_size),
        "THRESHOLD": repr(spec.holdout_eval_threshold),
        "N_BOOT": str(spec.holdout_eval_bootstrap),
        "SEED": "1729",
        "REPOSITORY": repository,
        "BRANCH": branch,
        "REVISION": revision,
    }
    preflight = _template(LAYA_RUNTIME_PREFLIGHT, {
        **values, "DECISION_CSV": HOLDOUT_JSONL,
        "QUESTION_SCHEMA_FILE": repr(QUESTION_SCHEMA_FILE)})
    script = _template(HOLDOUT_EVAL_KERNEL_SCRIPT,
                       {**values, "RUNTIME_PREFLIGHT": preflight})
    _kernel_script_gate(script)
    _module_scope_gate(script)
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / HOLDOUT_EVAL_CODE_FILE).write_text(script, encoding="utf-8")
    receipt = {
        "kernel": slug,
        "kind": HOLDOUT_EVAL_DECISION,
        "gpu": "T4 (single)",
        "run_tag": tag,
        "staged": str(stage),
        "code_file": HOLDOUT_EVAL_CODE_FILE,
        "dataset": {"slug": dataset_slug,
                    "payload": dataset_receipt["payload"],
                    "files": dataset_receipt["files"]},
        "checkpoint_dataset": ckpt_dataset,
        "threshold": spec.holdout_eval_threshold,
        "n_boot": spec.holdout_eval_bootstrap,
        "published_pin": {"repository": repository, "branch": branch,
                          "revision": revision},
        "published_tip": tip,
    }
    atomic_write_json(receipt, stage / f"{HOLDOUT_EVAL_DECISION}.receipt.json")
    _log_lane(f"staged kaggle kernel [{HOLDOUT_EVAL_DECISION}] ({spec.gpu}) "
              f"run_tag={tag} -> {stage}")
    return receipt


def stage_finetune_kernel(*, revision: str | None = None,
                          run_tag: str | None = None) -> dict[str, Any]:
    """Stage the kaggle fine-tune kernel payload (dry-safe).

    Writes under results/laya_lane/kaggle/finetune/:
      kernel-metadata.json + laya_finetune.py + finetune.receipt.json
      (+ the staged corpus dataset payload).
    Fail-loud preconditions (no silent skip):
      * spec.laya_decision_epochs > 0 (0 = disabled, nothing may stage);
      * spec.finetune_kernel_slug + spec.finetune_dataset_slug set;
      * the corpus JSONL + receipt stage from data/laya (spec constant).
    """
    spec = _spec()
    if spec.laya_decision_epochs <= 0:
        raise RuntimeError(
            "config laya.laya_decision_epochs <= 0: the laya lane is "
            "disabled (no payload may stage a GPU session)")
    slug = spec.finetune_kernel_slug
    if not slug:
        raise RuntimeError(
            "config laya.finetune_kernel_slug is unset; name the target "
            "kernel (owner/slug) before staging")
    dataset_slug = spec.finetune_dataset_slug
    if not dataset_slug:
        raise RuntimeError(
            "config laya.finetune_dataset_slug is unset; the corpus travels "
            "as that dataset (owner/slug) — name it before staging")
    base_dataset = spec.base_model_dataset
    if not base_dataset:
        raise RuntimeError(
            "config laya.base_model_dataset is unset; the base checkpoint "
            "travels as that dataset (owner/slug) — the finetune kernel "
            "must extract a LOCAL dir, never fetch from the Hub")
    # The published-tip invariant ('origin/<branch> == HEAD') resolves
    # BEFORE any payload write, exactly like stage_decision_kernel.
    repository = training_cfg().kaggle.repository
    branch = training_cfg().kaggle.branch
    revision = revision or _git_revision()
    from core import runtime_inputs
    tip = runtime_inputs.require_published_tip_match(
        revision, repository, branch)
    dataset_receipt = stage_finetune_dataset_payload(
        dataset_slug=dataset_slug,
        corpus_dir=TRAIN_ROOT / spec.finetune_corpus_dir)
    stage = staging_dir() / "kaggle" / FINETUNE_DECISION
    stage.mkdir(parents=True, exist_ok=True)
    tag = run_tag or spec.run_tag_prefix + decision_tag()
    metadata: dict[str, Any] = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
        "code_file": FINETUNE_CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        # single T4: the payload never requests the double accelerator;
        # the script itself pins CUDA_VISIBLE_DEVICES=0.
        "enable_internet": True,
        # THE CORPUS + THE BASE CHECKPOINT TRAVEL AS DATASETS: kernels push
        # does NOT ship the co-located JSONL files, and the 647 MB base
        # checkpoint cannot ride git — attach the corpus slug AND the
        # base-model archive dataset (er-laya-base).
        "dataset_sources": [dataset_slug, base_dataset],
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    recipe = finetune_config(spec)
    values = {
        "LAYA_PACKAGE": spec.finetune_package,
        "BASE_MODEL_ARCHIVE": spec.base_model_archive,
        "BASE_MODEL_DIR": spec.base_model_dir,
        "RUN_TAG": tag,
        "TRAIN_JSONL": FINETUNE_CORPUS_FILES[0],
        "DEV_JSONL": FINETUNE_CORPUS_FILES[1],
        "TEST_JSONL": FINETUNE_CORPUS_FILES[2],
        # The FULL TrainConfig surface rides one repr-baked Python literal:
        # the kernel constructs `TrainConfig(**FINETUNE_CONFIG)` directly.
        "FINETUNE_CONFIG": repr(recipe),
        "FINETUNE_DEVICE": spec.finetune.device,
        "HELD_OUT_BATCH": str(spec.laya_decision_batch_size),
        # wandb mirror: the key is read from .env at staging and baked in
        # (never committed); empty key -> the kernel logs nothing.
        "WANDB_API_KEY": _env_value("WANDB_API_KEY") or "",
        "WANDB_PROJECT": _wandb_project(),
        "REPOSITORY": repository,
        "BRANCH": branch,
        "REVISION": revision,
        "DEVICE_PATCH": FINETUNE_DEVICE_PATCH_SOURCE,
        "PERF_PATCH": FINETUNE_PERF_PATCH_SOURCE,
    }
    # two-pass substitution (a nested value's @tokens@ are never re-scanned
    # once it is inserted): the preflight bakes its own literal tuple FIRST,
    # then drops into the script — the push gate
    # (_staged_laya_push_preflight) literal-evals `_runtime_files`.
    preflight = _template(FINETUNE_RUNTIME_PREFLIGHT, values)
    script = _template(FINETUNE_KERNEL_SCRIPT, {**values,
                                                "RUNTIME_PREFLIGHT": preflight})
    _kernel_script_gate(script)
    _module_scope_gate(script)
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / FINETUNE_CODE_FILE).write_text(script, encoding="utf-8")
    receipt = {
        "kernel": slug,
        "kind": FINETUNE_DECISION,
        "gpu": "T4 (single)",
        "run_tag": tag,
        "staged": str(stage),
        "code_file": FINETUNE_CODE_FILE,
        "dataset": {"slug": dataset_slug,
                    "payload": dataset_receipt["payload"],
                    "files": dataset_receipt["files"]},
        "laya_package": spec.finetune_package,
        # The base checkpoint is the attached er-laya-base dataset archive,
        # extracted in-kernel; the Hub id is NOT passed as --base anymore.
        "base_model": {"dataset": base_dataset,
                       "archive": spec.base_model_archive,
                       "dir": spec.base_model_dir},
        "recipe": recipe,
        "device": spec.finetune.device,
        "corpus_dir": str(TRAIN_ROOT / spec.finetune_corpus_dir),
        "published_pin": {"repository": repository, "branch": branch,
                          "revision": revision},
        "published_tip": tip,
    }
    atomic_write_json(receipt, stage / f"{FINETUNE_DECISION}.receipt.json")
    _log_lane(f"staged kaggle finetune kernel ({spec.gpu}) run_tag={tag} "
              f"-> {stage}")
    return receipt


def stage_finetune_eval_kernel(*, revision: str | None = None,
                               run_tag: str | None = None,
                               checkpoint_path: Path | None = None
                               ) -> dict[str, Any]:
    """Stage the kaggle fine-tune EVAL-ONLY kernel payload (dry-safe).

    Writes under results/laya_lane/kaggle/finetune-eval/:
      kernel-metadata.json + laya_finetune_eval.py +
      finetune-eval.receipt.json (+ the SAME staged corpus dataset payload
      as the finetune kind).
    Fail-loud preconditions (no silent skip):
      * spec.laya_decision_epochs > 0 (0 = disabled, nothing may stage);
      * spec.finetune_eval_kernel_slug + spec.finetune_dataset_slug set;
      * a checkpoint source: spec.finetune_ckpt_dataset OR checkpoint_path;
      * spec.finetune_eval_split names a corpus split.

    The kernel loads the attached checkpoint and scores the attached split:
    no training, no Hub. The checkpoint dataset is attached as a second
    `dataset_sources` entry; an explicit `checkpoint_path` is baked as
    CHECKPOINT_PATH and takes precedence in-kernel.
    """
    spec = _spec()
    if spec.laya_decision_epochs <= 0:
        raise RuntimeError(
            "config laya.laya_decision_epochs <= 0: the laya lane is "
            "disabled (no payload may stage a GPU session)")
    slug = spec.finetune_eval_kernel_slug
    if not slug:
        raise RuntimeError(
            "config laya.finetune_eval_kernel_slug is unset; name the target "
            "eval kernel (owner/slug) before staging")
    dataset_slug = spec.finetune_dataset_slug
    if not dataset_slug:
        raise RuntimeError(
            "config laya.finetune_dataset_slug is unset; the corpus travels "
            "as that dataset (owner/slug) — name it before staging")
    ckpt_dataset = spec.finetune_ckpt_dataset
    if not ckpt_dataset and not checkpoint_path:
        raise RuntimeError(
            "config laya.finetune_ckpt_dataset is unset and no checkpoint "
            "path was given; the eval-only kernel needs a fine-tuned "
            "checkpoint dataset (owner/slug) or an explicit path")
    split = spec.finetune_eval_split
    if split not in FINETUNE_EVAL_SPLIT_FILES:
        raise ValueError(
            f"config laya.finetune_eval_split {split!r} is not one of "
            f"{sorted(FINETUNE_EVAL_SPLIT_FILES)}")
    # The published-tip invariant resolves BEFORE any payload write.
    repository = training_cfg().kaggle.repository
    branch = training_cfg().kaggle.branch
    revision = revision or _git_revision()
    from core import runtime_inputs
    tip = runtime_inputs.require_published_tip_match(
        revision, repository, branch)
    dataset_receipt = stage_finetune_dataset_payload(
        dataset_slug=dataset_slug,
        corpus_dir=TRAIN_ROOT / spec.finetune_corpus_dir,
        kind=FINETUNE_EVAL_DECISION)
    stage = staging_dir() / "kaggle" / FINETUNE_EVAL_DECISION
    stage.mkdir(parents=True, exist_ok=True)
    tag = run_tag or spec.run_tag_prefix + decision_tag()
    dataset_sources = [dataset_slug]
    if ckpt_dataset and not checkpoint_path:
        dataset_sources.append(ckpt_dataset)
    metadata: dict[str, Any] = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
        "code_file": FINETUNE_EVAL_CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        # single T4: the payload never requests the double accelerator.
        "enable_internet": True,
        # THE HELD-OUT SPLIT + THE CHECKPOINT TRAVEL AS DATASETS: the corpus
        # dataset carries the JSONL split, the checkpoint dataset carries
        # the fine-tuned checkpoint dir (rl_agent_config.json).
        "dataset_sources": dataset_sources,
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    eval_jsonl = FINETUNE_EVAL_SPLIT_FILES[split]
    calibration = eval_calibration_config(spec)
    values = {
        "LAYA_PACKAGE": spec.finetune_package,
        "RUN_TAG": tag,
        "EVAL_JSONL": eval_jsonl,
        "EVAL_SPLIT": split,
        "CKPT_DIR": spec.finetune_ckpt_dir,
        "CHECKPOINT_PATH": str(checkpoint_path) if checkpoint_path else "",
        "BATCH_SIZE": str(spec.finetune_eval_batch_size),
        # The eval path's calibration/abstention selection rides one
        # repr-baked literal (the FINETUNE_CONFIG precedent), so every knob
        # is YAML-driven and the kernel never re-derives a default.
        "EVAL_CALIBRATION": repr(calibration),
        "REPOSITORY": repository,
        "BRANCH": branch,
        "REVISION": revision,
    }
    # two-pass substitution (the preflight bakes its own literal tuple
    # first; the push gate literal-evals `_runtime_files`).
    preflight = _template(FINETUNE_EVAL_RUNTIME_PREFLIGHT, values)
    script = _template(FINETUNE_EVAL_KERNEL_SCRIPT,
                       {**values, "RUNTIME_PREFLIGHT": preflight})
    _kernel_script_gate(script)
    _module_scope_gate(script)
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / FINETUNE_EVAL_CODE_FILE).write_text(script, encoding="utf-8")
    receipt = {
        "kernel": slug,
        "kind": FINETUNE_EVAL_DECISION,
        "gpu": "T4 (single)",
        "run_tag": tag,
        "staged": str(stage),
        "code_file": FINETUNE_EVAL_CODE_FILE,
        "dataset": {"slug": dataset_slug,
                    "payload": dataset_receipt["payload"],
                    "files": dataset_receipt["files"]},
        "checkpoint_dataset": ckpt_dataset,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_dir_hint": spec.finetune_ckpt_dir,
        "eval_split": split,
        "eval_jsonl": eval_jsonl,
        "eval_calibration": calibration,
        "laya_package": spec.finetune_package,
        "published_pin": {"repository": repository, "branch": branch,
                          "revision": revision},
        "published_tip": tip,
    }
    atomic_write_json(receipt,
                      stage / f"{FINETUNE_EVAL_DECISION}.receipt.json")
    _log_lane(f"staged kaggle finetune-eval kernel ({spec.gpu}) "
              f"split={split} ckpt={ckpt_dataset or checkpoint_path} "
              f"run_tag={tag} -> {stage}")
    return receipt


def stage_colab_notebook(*, decision_kind: str,
                         run_tag: str | None = None) -> dict[str, Any]:
    """Colab notebook payload in the receipts style (dry-safe).

    Returns a receipt dict; the payload script lands at
    results/laya_lane/colab/<decision_kind>/laya_decision_colab.py.
    Fail-loud preconditions match stage_decision_kernel (epochs, slug).
    """
    spec = _spec()
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    if spec.laya_decision_epochs <= 0:
        raise RuntimeError(
            "config laya.laya_decision_epochs <= 0: the decision lane is "
            "disabled (no payload may stage a session)")
    if not spec.export_dataset_slug and not spec.dataset_slug:
        raise RuntimeError(
            "config laya.export_dataset_slug / dataset_slug both unset; "
            "name the target surface (owner/slug) before staging")
    question = stage_question_schema("colab")
    input_receipt = stage_decision_input("colab",
                                         decision_kind=decision_kind)
    stage = staging_dir() / "colab" / decision_kind
    stage.mkdir(parents=True, exist_ok=True)
    tag = run_tag or spec.run_tag_prefix + decision_tag()
    entry = DECISION_BINDINGS[decision_kind]
    staged_csv = input_receipt["staged"]
    script = _template(NOTEBOOK_SCRIPT, {
        "LAYA_PACKAGE": spec.laya_package,
        "DECISION_KIND": decision_kind,
        "RUN_TAG": tag,
        "DECISION_CSV": Path(staged_csv).name,
        "STATE_COLUMN": entry["state_column"],
    })
    _kernel_script_gate(script)
    _module_scope_gate(script)
    notebook = stage / COLAB_NOTEBOOK_NAME
    notebook.write_text(script, encoding="utf-8")
    receipt = {
        "kernel": COLAB_NOTEBOOK_NAME,
        "kind": decision_kind,
        "gpu": "T4 (single)",
        "run_tag": tag,
        "staged": str(stage),
        "notebook": str(notebook),
        "receipt_style": "results/laya_lane/<kind>/<op>/...",
        "question_schema": question["staged"],
        "decision_input": staged_csv,
        "epochs": spec.laya_decision_epochs,
        "note": "the colab CLI surface is unchanged; delivery contract only",
    }
    atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
    _log_lane(f"staged colab notebook payload [{decision_kind}] "
              f"run_tag={tag} -> {notebook}")
    return receipt


def _staged_laya_push_preflight(stage_dir: Path) -> None:
    """Laya push gate (the clone-lane staged_kernel_preflight shape, with
    the laya payload semantics): the ATTACHED-inputs inventory
    (`_runtime_files`) rides the er-laya-requests DATASET, never the git
    checkout — so the inventory is verified against the STAGED
    dataset_payload, and remote_revision_preflight (a core helper, never
    edited here) checks only the publish pin with an empty inventory."""
    import ast
    import json

    from core.runtime_inputs import remote_revision_preflight

    metadata = json.loads((stage_dir / "kernel-metadata.json").read_text())
    script = (stage_dir / metadata['code_file']).read_text()
    values = {}
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                        'REPOSITORY', 'BRANCH', 'REVISION', '_runtime_files'}:
                    values[target.id] = ast.literal_eval(node.value)
    required = {'REPOSITORY', 'BRANCH', 'REVISION', '_runtime_files'}
    if required - values.keys():
        raise ValueError(
            'Staged kernel lacks runtime preflight inventory; regenerate it')
    payload = stage_dir / "dataset_payload"
    missing = [name for name in values['_runtime_files']
               if not (payload / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Staged dataset payload {payload} is missing attached inputs: "
            + ", ".join(missing) + "; stage the payload first")
    remote_revision_preflight(values['REPOSITORY'], values['BRANCH'], (),
                              revision=values['REVISION'])


# ── the executed ops (fail-loud, --execute gated) ─────────────────────────
#: decision kind -> the LayaSpec attribute holding its pushed kernel slug.
_KIND_KERNEL_SLUG_ATTR = {
    FINETUNE_DECISION: "finetune_kernel_slug",
    FINETUNE_EVAL_DECISION: "finetune_eval_kernel_slug",
    HOLDOUT_EVAL_DECISION: "holdout_eval_kernel_slug",
}


def kernel_slug(decision_kind: str) -> str:
    """The pushed Kaggle kernel slug a decision kind runs on (stop target).

    Decision kinds publish to ``export_dataset_slug`` (their kernel id IS the
    export slug); the two fine-tune kinds carry dedicated kernel slugs. Fail
    loud when the knob is unset — never guess an account.
    """
    spec = _spec()
    if decision_kind in _KIND_KERNEL_SLUG_ATTR:
        attr = _KIND_KERNEL_SLUG_ATTR[decision_kind]
    elif decision_kind in DECISION_BINDINGS:
        attr = "export_dataset_slug"
    else:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    slug = getattr(spec, attr, None)
    if not slug:
        raise RuntimeError(
            f"config laya.{attr} is unset; name the target kernel (owner/slug) "
            f"before addressing {decision_kind!r}")
    return slug


def container_session_id(container: str) -> int | None:
    """The kernel session id inside ``KAGGLE_CONTAINER_NAME``.

    Kaggle sets NO ``KAGGLE_KERNEL_RUN_ID``/``KAGGLE_SESSION_ID`` in the
    container; the only per-run id is the numeric suffix of
    ``kaggle_<token>-<session_id>-webtier``, which the SDK
    ``cancel_kernel_session`` accepts (verified live). Returns ``None`` for an
    absent/malformed name.
    """
    parts = str(container or "").rsplit("-", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return int(parts[1])
    return None


def recorded_session_id(slug: str) -> int | None:
    """The launch-recorded session id for a pushed kernel (``None`` if none).

    The stream follower persists the kernel's self-reported id at
    ``logs/kaggle/<kernel>.session_id``; this is the first-class reader the
    verified in-place stop (and ``--session-id``) shares.
    """
    from cli import kaggle_lane as lane

    kernel = slug.rpartition("/")[2]
    if not kernel:
        raise ValueError(f"slug must be owner/slug, got {slug!r}")
    path = (lane.lane_logs_dir()
            / lane._spec().files.session_id_file.format(kernel=kernel))
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def stop_kaggle_kernel(slug: str, *, execute: bool,
                       wait: bool = True) -> dict[str, Any]:
    """First-class teardown of a pushed laya kernel's running session.

    Delegates to the kaggle lane's verified stop: the session id the push
    recorded (see ``push_kaggle_kernel``) feeds the SDK's in-place
    ``cancel_kernel_session``; with no recorded id it falls back to the
    version-replace stub. ``which='laya'`` only names the staging dir — the
    slug addresses the kernel, so the stop label is honest (never ``cpu``).
    """
    from cli.kaggle_kernels import KaggleKernels

    return KaggleKernels.stop_kernel(slug, which="laya", execute=execute,
                                     wait=wait)


def push_kaggle_kernel(stage_dir: Path, *, execute: bool,
                       activate: bool = True) -> dict[str, Any]:
    """`kaggle kernels push` a staged payload, `--execute`-gated.

    Dry run: returns the plan + argv, never spawns the kaggle subprocess.
    Executed: requires the staged metadata file (--activate gate), runs
    the laya push preflight (_staged_laya_push_preflight), pushes, and
    embeds the CLI's own output in the raised RuntimeError on a failing
    returncode.
    """
    argv = [sys.executable, "-m", "kaggle", "kernels", "push",
            "-p", str(stage_dir)]
    plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                            "argv": argv, "stage": str(stage_dir)}
    if not execute:
        _log_lane(f"dry-run: would run {' '.join(argv)}")
        return plan
    metadata_file = Path(stage_dir) / "kernel-metadata.json"
    if not metadata_file.is_file():
        raise RuntimeError(
            "--activate gate: no staged kernel at "
            f"{stage_dir} (kernel-metadata.json is missing); stage first "
            "(--what stage-kernel)")
    _staged_laya_push_preflight(Path(stage_dir))
    from cli import kaggle_lane as lane

    slug = json.loads(metadata_file.read_text(encoding="utf-8"))["id"]
    # Record the session id at launch (the capture point the verified stop
    # reads): a later stop then cancels the EXACT session via the SDK instead
    # of a blind version replace. Drop any stale id first.
    lane.clear_kernel_session_id(slug)
    result = subprocess.run(argv, cwd=TRAIN_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    output = result.stdout or ""
    rc = result.returncode
    plan["returncode"] = rc
    # `kaggle kernels push` returns rc=0 even on a soft error — e.g. the
    # GPU-session quota message "Kernel push error: Maximum batch GPU session
    # count of 2 reached" — so the exit code alone is NOT fail-loud. Require the
    # CLI's success line and reject any error text in its output.
    lowered = (output or "").lower()
    if rc != 0 or "error" in lowered or "successfully pushed" not in lowered:
        tail = output.strip()[-4000:] or "(kaggle produced no output)"
        raise RuntimeError(
            f"kaggle kernels push failed (rc={rc}): {' '.join(argv)}\n"
            f"--- kaggle output ---\n{tail}")
    plan["pushed"] = True
    try:
        captured = lane.capture_kernel_session_id(slug)
        plan["session_id"] = captured.get("session_id")
    except Exception as error:  # noqa: BLE001 - best-effort launch aid
        plan["session_id"] = None
        _log_lane(f"[{slug}] session-id capture skipped: {error}")
    _log_lane(f"pushed kernel payload: {' '.join(argv)} rc=0 "
              f"session_id={plan.get('session_id')}")
    return plan


def collect_kaggle_result(decision_kind: str, slug: str, *,
                          execute: bool = False) -> dict[str, Any]:
    """`kaggle kernels output` for a staged/decided kernel.

    Dry run: returns the plan only. Executed: pulls the kernel's output
    archive and requires the receipt member
    (laya_<decision_kind>.receipt.json) — fail-loud when it is absent. The
    JSON report members land beside it, and the receipt's corpus digest feeds
    the traceability contract. Results install under
    results/laya_lane/fetch/<decision>/.

    Boundary (deliberately NOT a `Bundle`): the kernel stages a plain
    `*.tar.gz` of /kaggle/working, so there is no per-role manifest member and
    `Bundle.load` could never accept it. The integrity contract is the
    in-archive receipt itself (written last by the kernel, like the NER
    artifact manifest); `plan["traceability"]` carries the JSON form of each
    validated report so the returned plan is exactly what is printed.

    Traceability is not read-only any more: when the fetched decision kind
    carries its PER-ROW grain (``<kind>.decisions.jsonl`` / the ``laya.evals``
    case list), that grain is validated against the shared contract AND written
    through the declared ``traceability_report`` layout, and
    ``plan["traceability_artifacts"]["record_grain"]`` names the artifact. A
    member that carries no grain is reported as an explicit ``not_applicable``
    entry, never skipped.
    """
    plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                            "decision_kind": decision_kind, "slug": slug}
    if not execute:
        _log_lane(f"dry-run: would fetch kernel output for {slug}")
        return plan
    stage = staging_dir() / "fetch" / decision_kind
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "output", slug,
         "-p", str(stage)], cwd=TRAIN_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle kernels output failed (rc={result.returncode}) for "
            f"{slug}: {result.stdout.strip()[-4000:]}")
    archives = sorted(stage.glob("*.tar.gz")) or sorted(stage.glob("*.zip"))
    if not archives:
        raise RuntimeError(f"kaggle kernels output staged no archive "
                           f"under {stage} (slug {slug})")
    receipt_name = f"laya_{decision_kind}.receipt.json"
    reports: dict[str, Any] = {}
    # ONE streaming pass: `r|*` never seeks, so the receipt, every JSON report
    # member and the per-row `<kind>.decisions.jsonl` member are consumed in a
    # single forward walk. The JSON payloads the kernel wrote (eval_report.json
    # and siblings) land in the fetch dir so the report path is reproducible
    # offline: the returned plan carries them keyed by member name. The per-row
    # decision kernel writes ``<kind>.decisions.jsonl``, which is NOT a JSON
    # member: it is read here because the record grain has no other production
    # source. Bodies are buffered so a missing receipt still fails loud BEFORE
    # any report lands (the pre-streaming order).
    members: list[str] = []
    decision_rows: list[dict] = []
    payload: Any = None
    pending: list[tuple[str, bytes]] = []
    with tarfile.open(archives[0], "r|*") as tar:
        for member in tar:
            members.append(member.name)
            if member.name == receipt_name:
                payload = json.loads(tar.extractfile(member).read().decode())
                continue
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name.endswith(".json"):
                body = tar.extractfile(member).read()
                pending.append((name, body))
                try:
                    reports[name] = json.loads(body.decode())
                except (ValueError, UnicodeDecodeError):
                    continue
            elif name == f"{decision_kind}{_DECISION_ROWS_SUFFIX}":
                body = tar.extractfile(member).read()
                pending.append((name, body))
                decision_rows = read_decision_rows(body)
    if payload is None:
        raise RuntimeError(
            f"fetched archive {archives[0].name} carries no "
            f"{receipt_name}; the kernel receipt contract failed")
    for name, body in pending:
        (stage / name).write_bytes(body)
    plan.update({"archive": str(archives[0]), "members": members,
                 "receipt": payload, "reports": reports})
    if decision_rows:
        plan["decision_rows"] = len(decision_rows)
    # Validate the fetched reports against the shared traceability contract and
    # EMIT the per-row grain through the declared layout. Store the JSON form:
    # this plan is the printed `--fetch` document, so a pydantic model left in it
    # would make json.dumps raise TypeError.
    documents, artifacts = fetched_traceability(
        payload, reports, decision_kind=decision_kind,
        decision_rows=decision_rows)
    plan["traceability"] = documents
    if artifacts:
        plan["traceability_artifacts"] = artifacts
    _log_lane(f"fetched kernel output for {slug}: "
              f"archive={archives[0].name} members={len(members)} "
              f"reports={sorted(reports)} rows={len(decision_rows)}")
    return plan


def local_eval_checkpoint(checkpoint_dir: Path, *,
                          eval_data: Path | None = None,
                          out_dir: Path | None = None,
                          split: str | None = None,
                          batch_size: int | None = None,
                          limit: int | None = None) -> dict[str, Any]:
    """Local (CPU) held-out eval of a fetched fine-tuned checkpoint.

    The offline twin of the `finetune-eval` kernel: loads the checkpoint with
    `laya.train.load_checkpoint` on CPU and runs `calibration_records` +
    `evaluate_records` on the corpus split (default `data/laya/test.jsonl`),
    writing the same `eval_report.json` (+ receipt) under `out_dir` (default
    results/laya_lane/local_eval). Requires `laya` + torch installed locally;
    fails loud before any work when they are missing. Never touches the
    network and never trains.
    """
    spec = _spec()
    split = split or spec.finetune_eval_split
    if split not in FINETUNE_EVAL_SPLIT_FILES:
        raise ValueError(
            f"eval split {split!r} is not one of "
            f"{sorted(FINETUNE_EVAL_SPLIT_FILES)}")
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "rl_agent_config.json").is_file():
        raise FileNotFoundError(
            f"checkpoint {checkpoint_dir} carries no rl_agent_config.json")
    if eval_data is None:
        eval_data = (TRAIN_ROOT / spec.finetune_corpus_dir
                     / FINETUNE_EVAL_SPLIT_FILES[split])
    eval_data = Path(eval_data)
    if not eval_data.is_file():
        raise FileNotFoundError(f"eval data not found: {eval_data}")
    if out_dir is None:
        out_dir = staging_dir() / "local_eval"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size = batch_size or spec.finetune_eval_batch_size
    try:
        import torch
        from laya import train as laya_train
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "local eval needs the laya package + torch installed on this "
            f"box (pip install {spec.laya_package}): {error}") from error
    device = torch.device("cpu")
    model, tok, cfg = laya_train.load_checkpoint(str(checkpoint_dir))
    model = model.to(device).eval()
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    parallel = laya_train.uses_parallel_layout(cfg)
    rows = laya_train.read_jsonl(str(eval_data))
    if limit:
        rows = rows[:limit]
    items, skipped = laya_train.items_from_rows(
        tok, rows, max_len, head_max_len, label_smoothing=0.0)
    if not items:
        raise RuntimeError(
            f"eval data {eval_data} produced no usable items "
            f"(skipped: {skipped!r})")
    records = laya_train.calibration_records(
        model, tok, items, device, max_len, head_max_len,
        batch_size=batch_size, parallel=parallel)
    before = laya_train.evaluate_records(records)
    calibration = eval_calibration_config(spec)
    fitted = fit_eval_calibration(laya_train, records, calibration)
    after = laya_train.evaluate_records(
        records, fitted["temperature"], fitted["temperature_by_options"])
    report = {
        "eval_mode": "held_out",
        "is_held_out": True,
        "device": "cpu",
        "eval_source": eval_data.name,
        "eval_split": split,
        "rows": len(rows),
        "items": len(items),
        "skipped": skipped,
        "checkpoint": str(checkpoint_dir),
        "before": before,
        "after": after,
        "temperature": fitted["temperature"],
        "temperature_by_options": fitted["temperature_by_options"],
    }
    # Additive: the opt-in knobs alone add keys, so a default config keeps the
    # landed report shape byte-for-byte (mirrors the eval kernel).
    if calibration.get("abstention") or fitted["min_confidence"] is not None:
        report["n_by_bucket"] = fitted["n_by_bucket"] or {}
    if fitted["abstention_thresholds"]:
        report["abstention_thresholds"] = fitted["abstention_thresholds"]
    if fitted["min_confidence"] is not None:
        report["min_confidence"] = fitted["min_confidence"]
    atomic_write_json(report, out_dir / FINETUNE_EVAL_REPORT_FILE)
    receipt = {
        "gpu_kind": FINETUNE_EVAL_DECISION,
        "device": "cpu",
        "eval_split": split,
        "eval_mode": "held_out",
        "is_held_out": True,
        "eval_data": str(eval_data),
        "eval_data_sha256": sha256_file(eval_data),
        "checkpoint": str(checkpoint_dir),
        "eval_calibration": calibration,
        "report": str(out_dir / FINETUNE_EVAL_REPORT_FILE),
    }
    atomic_write_json(receipt, out_dir / FINETUNE_EVAL_RECEIPT_FILE)
    _log_lane(f"local cpu eval [{split}] items={len(items)} "
              f"accuracy={after['accuracy']} -> {out_dir}")
    return report


class LayaLane:
    """The lane surface: results/laya_lane/<kind>/<decision>/ receipts.

    ONE class, TWO kinds: kind names the remote surface ("kaggle",
    "colab"); anything else fails loud.
    """

    kind: str

    def __init__(self, kind: str):
        if kind not in KINDS:
            raise ValueError(f"unknown laya lane kind: {kind!r}; "
                             f"expected {list(KINDS)}")
        self.kind = kind
        self._spec = _spec()

    def stage(self, decision_kind: str, *,
              input_override: Path | None = None,
              checkpoint_path: Path | None = None) -> dict[str, Any]:
        """Stage the payload (offline, dry-safe)."""
        if self.kind == "kaggle":
            if decision_kind == HOLDOUT_EVAL_DECISION:
                return stage_holdout_eval_kernel()
            return stage_decision_kernel(
                decision_kind=decision_kind, input_override=input_override,
                checkpoint_path=checkpoint_path)
        return stage_colab_notebook(decision_kind=decision_kind)

    def push(self, stage_dir: Path, *, execute: bool = False,
             activate: bool = True) -> dict[str, Any]:
        """`--execute` gated push (kaggle kind only)."""
        if self.kind != "kaggle":
            raise RuntimeError("push is a kaggle-lane operation")
        return push_kaggle_kernel(stage_dir, execute=execute,
                                  activate=activate)

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Dispatch the parsed main() args through this lane."""
        if args.decision not in DECISION_BINDINGS:
            raise ValueError(f"unknown decision kind: {args.decision!r}")
        if args.kind != self.kind:
            raise ValueError(f"--kind {args.kind!r} does not match the "
                             f"lane kind {self.kind!r}")
        return self.stage(args.decision, input_override=args.decision_input)


# ── main ───────────────────────────────────────────────────────────────────
def _spawn_stream_follower(slug: str) -> None:
    """Follow a pushed kernel's live session log into the laya lane transcript.

    The laya lane otherwise has no visibility into the remote session (it never
    opens a stream), so the training tqdm never reaches ``logs/laya/lane.log``.
    This spawns the kaggle lane's SSE follower against the pushed slug so the
    live output lands there. Detached (setsid) so a wrapper/shell death cannot
    orphan or kill the follower.
    """
    from core.common import TRAIN_ROOT

    log = TRAIN_ROOT / "logs/laya/lane.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "from pathlib import Path\n"
        "from cli.kaggle_lane import stream_kernel_logs\n"
        f"stream_kernel_logs({slug!r}, log_path=Path({str(log)!r}))\n"
    )
    # The follower writes the transcript itself (``log_path``); discard its own
    # stdout/stderr so its console echo cannot double-write every line into the
    # same lane.log (the duplicate `[stream ...]` prefix regression).
    subprocess.Popen(
        [sys.executable, "-c", code], cwd=TRAIN_ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "PYTHONPATH": str(TRAIN_ROOT / "src")},
        start_new_session=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, default="kaggle")
    parser.add_argument("--decision", choices=GPU_KINDS, default="attribute",
                        help="which typed decision run to stage "
                             "(default: attribute)")
    parser.add_argument("--execute", action="store_true",
                        help="make the remote call (kaggle kernels push); "
                             "the default is an offline dry-run")
    parser.add_argument("--decision-input", type=Path, default=None,
                        help="alternate decision source CSV on this box "
                             "(forwarded as the staging override; "
                             "kaggle staging path)")
    parser.add_argument("--fetch", action="store_true",
                        help="fetch a pushed kernel's output (kaggle "
                             "kernels output) instead of staging; combine "
                             "with --decision + --slug and --execute")
    parser.add_argument("--slug", default=None,
                        help="the pushed kernel slug (owner/slug) for "
                             "--fetch / --stop (default: the config slug "
                             "for --decision)")
    parser.add_argument("--stop", action="store_true",
                        help="tear down the running session for --decision's "
                             "kernel (kaggle only); the launch-recorded "
                             "session id feeds the SDK cancel")
    parser.add_argument("--session-id", action="store_true",
                        help="print the launch-recorded kernel session id for "
                             "--decision's kernel (or --slug); no network")
    parser.add_argument("--local-eval", action="store_true",
                        help="run the CPU held-out eval of a fetched "
                             "fine-tuned checkpoint instead of staging")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="the fine-tuned checkpoint dir for "
                             "--local-eval (must carry rl_agent_config.json)")
    parser.add_argument("--eval-data", type=Path, default=None,
                        help="the eval JSONL for --local-eval (default: the "
                             "config split under data/laya)")
    parser.add_argument("--eval-out", type=Path, default=None,
                        help="output dir for --local-eval "
                             "(default: results/laya_lane/local_eval)")
    parser.add_argument("--eval-split", choices=tuple(FINETUNE_EVAL_SPLIT_FILES),
                        default=None,
                        help="the corpus split for --local-eval "
                             "(default: config laya.finetune_eval_split)")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="cap the number of eval rows for --local-eval")
    args = parser.parse_args()

    if args.local_eval:
        # Local CPU eval path: no staging, no network, no kernel.
        if args.checkpoint is None:
            parser.error("--local-eval requires --checkpoint PATH")
        report = local_eval_checkpoint(
            args.checkpoint, eval_data=args.eval_data, out_dir=args.eval_out,
            split=args.eval_split, limit=args.eval_limit)
        print(json.dumps(report, indent=2), flush=True)
        return

    if args.fetch:
        # Fetch path: kaggle kernels output for a pushed kernel; the eval
        # report JSONs land under results/laya_lane/fetch/<decision>/.
        if not args.slug:
            parser.error("--fetch requires --slug owner/slug")
        plan = collect_kaggle_result(args.decision, args.slug,
                                     execute=args.execute)
        print(json.dumps(plan, indent=2), flush=True)
        return

    if args.session_id:
        # First-class reader of the launch-recorded session id (the verified
        # in-place stop's target); offline, no network.
        slug = args.slug or kernel_slug(args.decision)
        print(json.dumps({"kernel": slug,
                          "session_id": recorded_session_id(slug)},
                         indent=2), flush=True)
        return

    if args.stop:
        # First-class teardown: resolve the kernel the decision ran on (or an
        # explicit --slug), then cancel its running session. Dry-run by default.
        if args.kind != "kaggle":
            parser.error("--stop is a kaggle-lane operation")
        slug = args.slug or kernel_slug(args.decision)
        plan = stop_kaggle_kernel(slug, execute=args.execute)
        print(json.dumps(plan, indent=2, default=str), flush=True)
        return

    lane = LayaLane(args.kind)
    receipt = lane.stage(args.decision, input_override=args.decision_input,
                         checkpoint_path=args.checkpoint)
    print(_stamp(), f"[laya-lane] staged {args.kind}/{args.decision} payload: "
          f"{json.dumps(receipt, indent=2)}", flush=True)
    if args.execute and args.kind == "kaggle":
        stage_dir = Path(receipt["staged"])
        # the inputs travel as the dataset BEFORE the push (the kernel
        # metadata attaches spec.dataset_slug; a missing/drifting dataset
        # would FileNotFoundError resolve_input once boot passes)
        dataset_plan = publish_laya_dataset(args.decision,
                                            run_tag=receipt["run_tag"],
                                            execute=True)
        print(json.dumps(dataset_plan, indent=2), flush=True)
        push_plan = lane.push(stage_dir, execute=True)
        print(json.dumps(push_plan, indent=2), flush=True)
        # Follow the pushed kernel's live session log into logs/laya/lane.log
        # (the lane otherwise has no remote visibility and never shows a tqdm).
        _spawn_stream_follower(
            json.loads((stage_dir / "kernel-metadata.json").read_text())["id"])
    elif args.execute:
        _log_lane("colab payloads are a delivery contract only; nothing "
                  "to --execute")
    else:
        _log_lane("dry-run only; pass --execute to touch the remote surface")


if __name__ == "__main__":
    main()
