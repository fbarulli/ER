#!/usr/bin/env python3
"""Stream extraction snapshots and compare every raw listing before/after.

Snapshots contain extraction outputs, not recomputed canonical/gate results.
Frozen-canonical gate replay must be reported separately: it cannot measure
regex changes until affected raw listings are re-extracted and re-aggregated.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from core.common import DATA_PATH, TRAIN_ROOT, _validate_source_export, data_cfg, training_cfg
from core.manifest import sha256_file
if __package__ in (None, ""):
    # Permit the documented direct script command as well as test imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_gate_logic import json_ready


def _extraction_census(record: dict, counts: Counter, threshold: float) -> None:
    if "error" in record:
        counts["extraction_errors"] += 1
        return
    extracted = record["extraction"]
    counts["successful_rows"] += 1
    for field in ("volume", "pack"):
        confidence = float(extracted.get(f"{field}_confidence", 0.))
        observed = bool(extracted.get("volume_ml", 0)) if field == "volume" else confidence > 0.
        counts[f"{field}:observed_evidence" if observed else f"{field}:no_observed_evidence"] += 1
        if confidence < threshold:
            counts[f"{field}:low_confidence_with_evidence" if observed else f"{field}:low_confidence_without_evidence"] += 1
    counts.update(f"flag:{flag}" for flag in extracted.get("attribute_consistency_flags", []))
    counts.update(f"measurement_role:{claim['role']}" for claim in extracted.get("measurement_evidence", []))
    counts.update(f"pack_role:{claim['role']}" for claim in extracted.get("pack_evidence", []))


def snapshot(target: Path, *, label: str, chunk_rows: int) -> dict:
    from pipeline import extract_all

    mapping = data_cfg().column_mapping
    meta = {"kind": "er.raw_extraction_snapshot.v1", "label": label,
            "raw_dataset_path": str(DATA_PATH), "raw_dataset_sha256": sha256_file(DATA_PATH),
            "code_fingerprints": {str(path): sha256_file(TRAIN_ROOT / path)
                                  for path in ["src/pipeline.py", "src/core/text.py", "src/core/critical_attributes.py",
                                               "src/core/url_evidence.py", "src/core/sweetener_values.py",
                                               "src/ner/ner_product_attributes.py", "src/core/date_evidence.py",
                                               "src/core/attribute_conflicts.py", "src/core/attribute_decision.py",
                                               "src/core/attribute_universe.py", "config/paths.yaml", "config/training.yaml"]}}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".pending")
    count, errors = 0, Counter()
    with gzip.open(temporary, "wt", encoding="utf-8") as output:
        output.write(json.dumps(meta) + "\n")
        chunks = pd.read_csv(DATA_PATH, chunksize=chunk_rows, **data_cfg().dataset_csv_read.model_dump())
        for chunk in chunks:
            rows = chunk.rename(columns=mapping).fillna("").to_dict("records")
            for row in rows:
                record = {"row_index": count, "sku_id": row["sku_id"]}
                try:
                    record["extraction"] = json_ready(extract_all(
                        row["sku_name_eng"], row["attribute"], row["description_short_eng"], row["sku_url"],
                        row["image_url"], row["breadcrumbs_eng"], row["category"]))
                except Exception as exc:
                    record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    errors[type(exc).__name__] += 1
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
            print(f"[{label}] extracted {count:,} raw rows; errors={sum(errors.values())}", flush=True)
    _validate_source_export(pd.DataFrame(index=pd.RangeIndex(count)), DATA_PATH)
    if sha256_file(DATA_PATH) != meta["raw_dataset_sha256"]:
        raise RuntimeError("raw dataset changed during extraction snapshot")
    temporary.replace(target)
    summary = {**meta, "rows": count, "errors": dict(errors), "snapshot": str(target)}
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def compare(before_path: Path, after_path: Path, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    changed_path = target.with_suffix(".changed.jsonl.gz")
    fields, new_fields, decisions = Counter(), Counter(), Counter()
    examples, checked, changed, existing_changes, numeric_changes = {}, 0, 0, 0, 0
    before_counts, after_counts = Counter(), Counter()
    threshold = float(training_cfg().gate.raw_conf_threshold)
    with gzip.open(before_path, "rt") as before, gzip.open(after_path, "rt") as after, gzip.open(changed_path, "wt") as changes:
        old_meta, new_meta = json.loads(next(before)), json.loads(next(after))
        if old_meta["raw_dataset_sha256"] != new_meta["raw_dataset_sha256"]:
            raise ValueError("snapshots describe different raw source datasets")
        for old_line, new_line in zip(before, after, strict=True):
            old, new = json.loads(old_line), json.loads(new_line)
            if (old["row_index"], old["sku_id"]) != (new["row_index"], new["sku_id"]):
                raise ValueError("snapshot rows are misaligned")
            checked += 1
            _extraction_census(old, before_counts, threshold)
            _extraction_census(new, after_counts, threshold)
            if "error" in old or "error" in new:
                if "error" not in new:
                    state = "baseline_error_resolved"
                elif "error" not in old:
                    state = "new_extraction_error"
                else:
                    state = "baseline_error_persisted" if old["error"] == new["error"] else "baseline_error_changed"
                decisions[state] += 1
            left, right = old.get("extraction", {}), new.get("extraction", {})
            moved = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
            if moved or old.get("error") != new.get("error"):
                changed += 1
                fields.update(moved)
                new_fields.update(key for key in moved if key not in left)
                existing_changes += any(key in left for key in moved)
                numeric_changes += any(key in moved for key in ("volume_ml", "pack_qty"))
                result = {"row_index": old["row_index"], "sku_id": old["sku_id"],
                          "changed_fields": moved, "before": old, "after": new}
                changes.write(json.dumps(result, ensure_ascii=False) + "\n")
                for field in moved:
                    if len(examples.setdefault(field, [])) < 5:
                        examples[field].append({"sku_id": old["sku_id"], "row_index": old["row_index"],
                                                "before": left.get(field), "after": right.get(field)})
    report = {"schema_version": "er.raw_extraction_delta.v1",
              "scope": "Full raw-source extraction delta. Not a frozen-gate replay or fresh canonical/gate census; changed regex fields require canonical rebuild before gate impact can be measured.",
              "before": old_meta, "after": new_meta, "rows_checked": checked, "changed_rows": changed,
              "existing_extraction_field_changed_rows": existing_changes,
              "volume_or_pack_assignment_changed_rows": numeric_changes,
              "newly_added_field_counts": dict(sorted(new_fields.items())),
              "raw_extraction_census": {"scope": "Raw listing extraction quality, not canonical/gate census. Pack evidence is observed when pack_confidence > 0; default pack_qty=1 is not evidence.",
                                        "confidence_threshold": threshold,
                                        "before": dict(sorted(before_counts.items())), "after": dict(sorted(after_counts.items())),
                                        "delta": {key: after_counts[key] - before_counts[key] for key in sorted(set(before_counts) | set(after_counts))}},
              "error_transitions": dict(decisions), "changed_field_counts": dict(sorted(fields.items())),
              "examples_per_changed_field": examples, "all_changed_rows": str(changed_path)}
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Raw extraction before/after", "", report["scope"], "",
             f"Compared {checked:,} rows; {changed:,} changed, including additive provenance fields. "
             f"Existing fields changed on {existing_changes:,} rows; volume/pack assignments changed on {numeric_changes:,}.", "",
             "| Field | Changed rows |", "|---|---:|"]
    lines.extend(f"| {field} | {count:,} |" for field, count in sorted(fields.items()))
    lines += ["", "## Error transitions", "", "```json", json.dumps(report["error_transitions"], indent=2), "```", "",
              "## Raw extraction quality census (separate from canonical/gate results)", "", "```json", json.dumps(report["raw_extraction_census"], indent=2), "```", "",
              "## Auditable examples", "", "```json", json.dumps(examples, indent=2, ensure_ascii=False), "```", ""]
    target.with_suffix(".md").write_text("\n".join(lines))
    return report


def regression_report(fixture_path: Path, target: Path) -> dict:
    """Evaluate tracked explicit source-evidence expectations without rewriting them."""
    from pipeline import extract_all, three_way_gate
    from training.gate_replay import fired_stage

    fixture = json.loads(fixture_path.read_text())
    samples = []
    for sample in fixture["listings"]:
        row, expected = sample["source_row"], sample["expected_corrected"]
        actual = json_ready(extract_all(row["sku_name_eng"], row["attribute"], row["description_short_eng"], row["sku_url"],
                                       row["image_url"], row["breadcrumbs_eng"], row["category"]))
        failed = []
        if "volume_ml" in expected and abs(actual["volume_ml"] - expected["volume_ml"]) > expected["volume_absolute_tolerance_ml"]:
            failed.append("volume_ml")
        if "pack_qty" in expected and actual["pack_qty"] != expected["pack_qty"]:
            failed.append("pack_qty")
        if not set(expected.get("required_flags", [])).issubset(actual["attribute_consistency_flags"]):
            failed.append("required_flags")
        if "volume_confidence_below" in expected and actual["volume_confidence"] >= expected["volume_confidence_below"]:
            failed.append("volume_confidence_below")
        roles = {entry["role"] for entry in actual.get("measurement_evidence", [])}
        if not set(expected.get("measurement_roles", [])).issubset(roles):
            failed.append("measurement_roles")
        if not set(expected.get("negated_sweetener_type_includes", [])).issubset(actual.get("negated_sweetener_type_set", [])):
            failed.append("negated_sweetener_type_includes")
        for entry in expected.get("required_pack_evidence", []):
            if not any(all(claim.get(key) == value for key, value in entry.items()) for claim in actual.get("pack_evidence", [])):
                failed.append("required_pack_evidence")
        samples.append({**sample, "current_extraction": actual, "failed_expectations": failed,
                        "status": "pass" if not failed else "fail"})
    pairs = []
    for sample in fixture["pairs"]:
        records = []
        for side in ("baseline_left", "baseline_right"):
            record = dict(sample[side])
            for key, value in record.items():
                if key.endswith("_set") or key.endswith("_flags"):
                    record[key] = set(value)
            records.append(record)
        current = three_way_gate(*records)
        expected = sample["expected_corrected"]
        passed = current["decision"] == expected["decision"] and fired_stage(current["reason"]) == expected["reason_stage"]
        pairs.append({**sample, "current_gate": current, "status": "pass" if passed else "fail"})
    report = {"schema_version": "er.gate_regex_regression_results.v1",
              "scope": "Tracked actual source sample re-extraction against independently specified corrections; pair gate evaluates frozen canonical inputs separately.",
              "fixture": str(fixture_path), "fixture_sha256": sha256_file(fixture_path),
              "passed_listings": sum(sample["status"] == "pass" for sample in samples),
              "failed_listings": sum(sample["status"] == "fail" for sample in samples),
              "listings": samples, "frozen_canonical_pairs": pairs}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Tracked gate/regex correction samples", "", report["scope"], "",
             "| Product ID | Status | Before volume | Current volume | Before pack | Current pack | Failed expectations |", "|---|---|---:|---:|---:|---:|---|"]
    lines.extend(f"| {sample['sku_id']} | {sample['status']} | {sample['baseline_extraction']['volume_ml']} | {sample['current_extraction']['volume_ml']} | "
                 f"{sample['baseline_extraction']['pack_qty']} | {sample['current_extraction']['pack_qty']} | {','.join(sample['failed_expectations'])} |"
                 for sample in samples)
    lines += ["", "## Frozen canonical gate regression", "", "```json", json.dumps(pairs, indent=2, ensure_ascii=False), "```", ""]
    target.with_suffix(".md").write_text("\n".join(lines))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    capture = sub.add_parser("snapshot")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--label", required=True)
    capture.add_argument("--chunk-rows", type=int, default=2048)
    diff = sub.add_parser("compare")
    diff.add_argument("--before", type=Path, required=True)
    diff.add_argument("--after", type=Path, required=True)
    diff.add_argument("--output", type=Path, required=True)
    regressions = sub.add_parser("regressions")
    regressions.add_argument("--fixture", type=Path, default=TRAIN_ROOT / "tests/fixtures/gate_regex_regressions.json")
    regressions.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "snapshot":
        if args.chunk_rows < 1:
            parser.error("chunk rows must be positive")
        summary = snapshot(args.output, label=args.label, chunk_rows=args.chunk_rows)
    elif args.action == "compare":
        summary = compare(args.before, args.after, args.output)
    else:
        summary = regression_report(args.fixture, args.output)
    print(json.dumps({key: value for key, value in summary.items() if key in {"rows", "errors", "rows_checked", "changed_rows", "passed_listings", "failed_listings"}}, indent=2))


if __name__ == "__main__":
    main()
