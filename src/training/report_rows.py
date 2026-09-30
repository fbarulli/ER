"""Idempotent report-row persistence and calibration diagnostic traces."""

from __future__ import annotations

from collections import Counter
import json
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

from core.common import ensure_parent

def _append_csv(
    path: Path, new_rows: list[dict], key_fields: str | list[str]
) -> None:
    """Append with replace-by-key semantics: re-running the SAME variant
    updates its row IN PLACE (row order preserved — plots read the sweep
    order from it; the old concat+drop_duplicates moved re-run rows to the
    END, silently corrupting the 07 series order) instead of duplicating
    it. Idempotent regeneration — the reproducibility contract."""
    import pandas as pd

    ensure_parent(path)
    if isinstance(key_fields, str):
        key_fields = [key_fields]
    if not key_fields or len(set(key_fields)) != len(key_fields):
        raise ValueError(f"replace key must contain unique fields: {key_fields!r}")
    new_df = pd.DataFrame(new_rows)
    missing_incoming = [field for field in key_fields if field not in new_df.columns]
    if missing_incoming:
        raise ValueError(
            f"incoming rows for {path} are missing replace-key fields "
            f"{missing_incoming}"
        )
    if not path.exists():
        new_df.to_csv(path, index=False)
        return
    old = pd.read_csv(path)
    missing_legacy = [field for field in key_fields if field not in old.columns]
    if missing_legacy:
        legacy_marker = "__legacy_unknown__"
        for field in missing_legacy:
            old[field] = [
                f"{legacy_marker}:{field}:{record_number}"
                for record_number in range(len(old))
            ]
        print(
            f"[07-migration] {path}: retained {len(old):,} pre-existing row(s); "
            f"marked missing replace-key field(s) {missing_legacy} as "
            f"{legacy_marker}:<field>:<row>",
            flush=True,
        )
    key = list(key_fields)
    if not key:
        raise ValueError(
            f"{path} has no shared replace key from {key_fields}; refusing "
            "to append without provenance identity"
        )
    # index the old rows by key tuple -> row position. Keys compare on
    # STR — pandas reads "0.25" back as float 0.25, so a raw-tuple match
    # would miss on every numeric-looking key (07d's fraction column).
    old_rows = old.to_dict("records")
    for row_number, row in enumerate(old_rows):
        for field in key:
            if pd.isna(row[field]):
                row[field] = f"__legacy_unknown__:{field}:{row_number}"
    old_keys = [tuple(str(row[k]) for k in key) for row in old_rows]
    duplicate_old_keys = {
        item for item, count in Counter(old_keys).items() if count > 1
    }
    if duplicate_old_keys:
        raise ValueError(
            f"{path} contains duplicate replace keys; refusing ambiguous "
            f"replacement: {sorted(duplicate_old_keys)[:3]}"
        )
    pos = {key_value: i for i, key_value in enumerate(old_keys)}
    out_rows = old_rows
    appended = []
    new_rows_records = new_df.to_dict("records")
    missing_values = [
        (row_number, field)
        for row_number, row in enumerate(new_rows_records)
        for field in key
        if pd.isna(row[field])
    ]
    if missing_values:
        raise ValueError(
            f"incoming rows for {path} contain null replace-key values: "
            f"{missing_values[:3]}"
        )
    new_keys = [tuple(str(r[k]) for k in key) for r in new_rows_records]
    duplicate_new_keys = {
        item for item, count in Counter(new_keys).items() if count > 1
    }
    if duplicate_new_keys:
        raise ValueError(
            f"incoming rows for {path} contain duplicate replace keys; "
            f"refusing ambiguous replacement: {sorted(duplicate_new_keys)[:3]}"
        )
    for r, key_value in zip(new_rows_records, new_keys, strict=True):
        if key_value in pos:
            out_rows[pos[key_value]] = {**out_rows[pos[key_value]], **r}
        else:
            appended.append(r)
    expected_rows = len(out_rows) + len(appended)
    pd.DataFrame(out_rows + appended).to_csv(path, index=False)
    written = pd.read_csv(path)
    if len(written) != expected_rows:
        raise RuntimeError(
            f"{path} row-count changed during append: expected "
            f"{expected_rows}, wrote {len(written)}"
        )


def _nonnumeric_calibration_fields() -> tuple[str, ...]:
    """Return diagnostic contract fields that must be carried, not averaged."""
    from training.hpo_metrics import (
        CALIBRATION_AGGREGATE_FIELDS,
        CalibrationMetricRow,
    )

    numeric_fields = set(CALIBRATION_AGGREGATE_FIELDS)
    return tuple(
        name
        for name in CalibrationMetricRow.model_fields
        if name.startswith(
            ("calibration_", "collapse_", "diagnostic_", "attribute_conflict_")
        )
        and name not in numeric_fields
    )


def _trace_nonnumeric_calibration_fields(
    rows: list[dict], fields: tuple[str, ...]
) -> dict[str, str]:
    """Serialize per-fold string diagnostics so 07-series rows retain them."""
    trace: dict[str, str] = {}
    for field in fields:
        values = [row.get(field) for row in rows]
        trace[field] = json.dumps(values, sort_keys=True, default=str)
    if fields:
        print(
            f"[07] retained {len(fields)} nonnumeric calibration diagnostic "
            "field(s) as per-fold JSON traces",
            flush=True,
        )
    return trace



def emit_07_series(
    ok_rows: list[dict], args, *, report_paths, seed: int,
    train_config, recall_suffix, append_csv,
) -> None:
    """07b/07c/07d CSV emission — the report_plots inputs (owner ruling).

    07c (field ablation) and 07d (data scaling) are APPEND-MODE: each
    src/training/train invocation adds its variant's row, so a 07-series
    sweep accumulates instead of overwriting (the plots read whatever rows
    exist).
    07b (four-population scores) is written by the rerank lane, which has
    the trained embeddings; a plain run notes its absence.
    """

    from training.hpo_metrics import CALIBRATION_AGGREGATE_FIELDS

    # 05-03/06-3: the recall-tied aggregates below are a function of the
    # CONFIGURED recall target, so their column names are derived from it with
    # the same SSOT helper training.py's producer uses — a retune renames both
    # sides at once instead of filing a 95%-recall number under the old
    # fixed-suffix header (the literal stayed put while
    # rand_matching.target_recall became config-driven in 41cd50e).
    _train_cfg = train_config()
    _target_recall = float(_train_cfg.rand_matching.target_recall)
    _recall_key = recall_suffix(
        _target_recall
    )
    _prec_col = f"precision_at_{_recall_key}_recall"
    _tp_col = f"tp_at_{_recall_key}_recall"
    _fp_col = f"fp_at_{_recall_key}_recall"
    _thr_col = f"threshold_at_{_recall_key}_recall"
    # 05-05: every aggregate in these rows is a function of the split and the
    # seed, and 41cd50e is the commit that made both steerable from config.
    # Record them, and (below) key on them: two runs that differ only in these
    # inputs must not merge into one row whose numbers cannot be attributed.
    _split = _train_cfg.split
    _provenance = {
        "split": args.split,
        "holdout_component_folds": int(_split.holdout_component_folds),
        "calibration_seed_offset": int(_split.calibration_seed_offset),
        "seed": int(seed),
        # Keep the numeric value for analysis, but key rows on the canonical
        # label.  CSV round-tripping can coerce numeric values and erase the
        # identity boundary between closely spaced recall targets.
        "target_recall": _target_recall,
        "target_recall_key": _recall_key,
    }

    calibration_rows = [
        row for row in ok_rows
        if row.get("calibration_status", "available") == "available"
    ]
    missing_fields = sorted(
        {
            field
            for field in CALIBRATION_AGGREGATE_FIELDS
            if any(field not in row for row in calibration_rows)
        }
    )
    if missing_fields and calibration_rows:
        raise ValueError(
            "calibration metric contract missing from fold rows: "
            f"{missing_fields}"
        )
    if not calibration_rows:
        print(
            "[calibration] unavailable for all completed folds; "
            "07-series calibration aggregates will be NaN",
            flush=True,
        )

    # ---- 07c: one aggregate row per payload variant ----
    aggregate_cache: dict[str, float] = {}

    def agg(field: str) -> float:
        """Compute each fold aggregate once for both 07-series rows."""
        if field not in aggregate_cache:
            vals = [r.get(field) for r in ok_rows if r.get(field) is not None]
            invalid = [value for value in vals if not isinstance(value, numbers.Real)]
            if invalid:
                raise TypeError(
                    f"calibration field {field!r} is declared numeric but has "
                    f"non-numeric values: {invalid[:2]!r}"
                )
            aggregate_cache[field] = float(np.mean(vals)) if vals else float("nan")
        return aggregate_cache[field]

    nonnumeric_fields = _nonnumeric_calibration_fields()
    diagnostic_trace = _trace_nonnumeric_calibration_fields(
        ok_rows, nonnumeric_fields
    )

    row_07c = {
        "variant": args.payload,
        "average_precision": round(agg("pr_auc"), 4),
        "precision_at_1": round(agg("precision_at_1"), 4),
        "recall_at_1": round(agg("recall_at_1"), 4),
        "precision_at_5": round(agg("precision_at_5"), 4),
        "recall_at_5": round(agg("recall_at_5"), 4),
        "precision_at_10": round(agg("precision_at_10"), 4),
        "recall_at_10": round(agg("recall_at_10"), 4),
        "hits_at_1": round(agg("hits_at_1"), 4),
        _prec_col: round(agg(_prec_col), 4),
        _tp_col: round(agg(_tp_col), 1),
        _fp_col: round(agg(_fp_col), 1),
        _thr_col: round(agg(_thr_col), 4),
        "auc": round(agg("auc"), 4),
        "n_folds": len(ok_rows),
        "model": args.model,
        **_provenance,
    }
    row_07c.update(
        {
            field: round(agg(field), 4)
            for field in CALIBRATION_AGGREGATE_FIELDS
        }
    )
    row_07c.update(diagnostic_trace)
    append_csv(
        report_paths["field_ablation"],
        [row_07c],
        ["variant", *_provenance],
    )

    # ---- 07d: one row per train fraction ----
    row_07d = {
        "fraction": args.train_frac,
        "n_triples": round(agg("n_train")),
        "average_precision": round(agg("pr_auc"), 4),
        "precision_at_1": round(agg("precision_at_1"), 4),
        "recall_at_1": round(agg("recall_at_1"), 4),
        "precision_at_5": round(agg("precision_at_5"), 4),
        "recall_at_5": round(agg("recall_at_5"), 4),
        "precision_at_10": round(agg("precision_at_10"), 4),
        "recall_at_10": round(agg("recall_at_10"), 4),
        "hits_at_1": round(agg("hits_at_1"), 4),
        _prec_col: round(agg(_prec_col), 4),
        _tp_col: round(agg(_tp_col), 1),
        _fp_col: round(agg(_fp_col), 1),
        _thr_col: round(agg(_thr_col), 4),
        "repeat": "single",
        "model": args.model,
        "payload": args.payload,
        **_provenance,
    }
    row_07d.update(
        {
            field: round(agg(field), 4)
            for field in CALIBRATION_AGGREGATE_FIELDS
        }
    )
    row_07d.update(diagnostic_trace)
    append_csv(
        report_paths["data_scaling"], [row_07d], ["fraction", "payload", *_provenance]
    )


