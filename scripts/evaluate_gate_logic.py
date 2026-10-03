#!/usr/bin/env python3
"""Surface deterministic gate samples with original evidence and live extraction.

PYTHONPATH=src .venv/bin/python scripts/evaluate_gate_logic.py

This is a sampled consistency review, not a labeled accuracy estimate. The
actual gate's fired reason is kept separate from independently run attribute
diagnostics, which can describe stages the actual gate never reached.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from core.common import F, RESULTS, TRAIN_ROOT, data_cfg, training_cfg
from core.manifest import sha256_file
from training.gate_replay import canonical_records_from_csv, fired_stage


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_ready(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def endpoint_evidence(record: dict, *, listing_limit: int) -> dict:
    from pipeline import extract_all

    raw = record.get("source_rows", "")
    rows = json.loads(raw) if isinstance(raw, str) and raw else raw or []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{record['gtin']}: source_rows must contain listing objects")
    extracted = []
    for index, row in enumerate(rows[:listing_limit]):
        live = extract_all(
            str(row.get("sku_name_eng", "")), str(row.get("attribute", "")),
            description_short_eng=str(row.get("description_short_eng", "")),
            sku_url=str(row.get("sku_url", "")), image_url=str(row.get("image_url", "")),
            breadcrumbs_eng=str(row.get("breadcrumbs_eng", "")),
            category=str(row.get("category", "")),
        )
        extracted.append({"listing_index": index, "current_extraction": json_ready(live)})
    ledger = record.get("evidence_ledger", "")
    ledger = json.loads(ledger) if isinstance(ledger, str) and ledger else ledger or []
    return {
        "gtin": record["gtin"], "canonical": record.get("canonical", ""),
        "extracted_sets": json_ready({key: value for key, value in record.items()
                                      if key.endswith("_set") or key.endswith("_flags")}),
        "confidence_and_consistency": {key: value for key, value in record.items()
                                       if key.endswith("_confidence") or key.endswith("_consistency")},
        "source_rows": rows, "persisted_evidence_ledger": ledger,
        "source_listing_count": len(rows),
        "live_extraction_listing_count": len(extracted),
        "live_extraction_is_complete": len(extracted) == len(rows),
        "live_listing_extractions": extracted,
    }


def evaluate_sample(pair: dict, canon: dict, *, listing_limit: int, endpoints: dict) -> dict:
    from core.attribute_conflicts import canonical_attribute_info
    from core.attribute_decision import AttributeDecisionEngine
    from pipeline import three_way_gate

    keys = str(pair["gtin1"]), str(pair["gtin2"])
    missing = [key for key in keys if key not in canon]
    committed = {"decision": pair["gate_decision"], "reason": pair["gate_reason"]}
    result = {"gtin1": keys[0], "gtin2": keys[1], "similarity": pair["similarity"],
              "committed_gate": committed, "inspection_flags": []}
    if missing:
        result.update(status="suspect", current_gate=None, missing_canonical_gtins=missing)
        result["inspection_flags"].append("missing_canonical_endpoint")
        return result
    left, right = (canon[key] for key in keys)
    current = three_way_gate(left, right)
    result["current_gate"] = current
    result["actual_fired_stage"] = fired_stage(current["reason"])
    changed = current["decision"] != committed["decision"] or current["reason"] != committed["reason"]
    if changed:
        result["inspection_flags"].append("current_gate_differs_from_committed")
    cfg = training_cfg().gate
    diagnostic = AttributeDecisionEngine(
        volume_relative_tolerance=float(cfg.vol_tolerance),
        volume_absolute_tolerance_ml=float(cfg.vol_abs_tolerance),
    ).evaluate(canonical_attribute_info(left), canonical_attribute_info(right),
               left_raw=left, right_raw=right)
    result["independent_attribute_diagnostics"] = {
        "scope": "Independently evaluated; these results do not identify which actual gate branch executed.",
        "conflicts": diagnostic.conflicts, "agreements": diagnostic.agreements,
        "inconclusive": diagnostic.inconclusive, "dimensions": diagnostic.as_dict(),
    }
    for side, key in zip(("left", "right"), keys, strict=True):
        if key not in endpoints:
            endpoints[key] = endpoint_evidence(canon[key], listing_limit=listing_limit)
        result[side] = endpoints[key]
        if not endpoints[key]["source_rows"]:
            result["inspection_flags"].append(f"{side}_original_source_rows_missing")
    result["status"] = "suspect" if result["inspection_flags"] else "consistent_on_sampled_replay"
    return result


def build_report(gates: pd.DataFrame, canon: dict, *, samples_per_reason: int, listing_limit: int) -> dict:
    if gates.duplicated(["gtin1", "gtin2"]).any():
        raise ValueError("gate artifact contains duplicate candidate pairs")
    frame = gates.copy()
    frame["similarity"] = pd.to_numeric(frame["similarity"], errors="raise")
    frame["stage"] = frame.gate_reason.map(fired_stage)
    selected = frame.sort_values(["similarity", "gtin1", "gtin2"], ascending=[False, True, True])
    selected = selected.groupby(["gate_decision", "gate_reason"], sort=True).head(samples_per_reason)
    endpoints = {}
    samples = [evaluate_sample(row, canon, listing_limit=listing_limit, endpoints=endpoints)
               for row in selected.to_dict("records")]
    counts = Counter(frame.stage)
    stages = [{"stage": stage, "configured_reason": reason,
               "committed_population": int(counts.get(stage, 0)),
               "coverage": "sampled" if counts.get(stage, 0) else "not_observed_in_committed_artifact"}
              for stage, reason in training_cfg().gate.reasons.model_dump().items()]
    report = {
        "schema_version": "er.gate_logic_review.v1",
        "scope": "Deterministic highest-similarity samples per committed decision/reason; not a full replay or labeled accuracy estimate.",
        "status_definition": "consistent_on_sampled_replay means the current actual gate matches the committed decision/reason and original listing evidence is present; it does not certify the label. Suspect means drift or missing evidence requires inspection, not that the label is incorrect.",
        "dashboard_limitation": "The existing /gate dimension table compares first-listing attributes; aggregated canonical evidence may differ. This report includes every persisted canonical source row, and separate live extraction for the stated listing limit.",
        "population_pairs": len(frame), "sampled_pairs": len(samples),
        "samples_per_reason": samples_per_reason, "live_listing_limit_per_endpoint": listing_limit,
        "population_decisions": dict(sorted(Counter(frame.gate_decision).items())),
        "population_decision_reasons": [{"decision": decision, "reason": reason, "pairs": int(count)}
                                       for (decision, reason), count in frame.groupby(["gate_decision", "gate_reason"]).size().items()],
        "sample_status_counts": dict(sorted(Counter(sample["status"] for sample in samples).items())),
        "configured_stages": stages, "unmapped_population_reasons": sorted(set(frame.loc[frame.stage.eq("unknown"), "gate_reason"])),
        "gate_thresholds": training_cfg().gate.model_dump(), "samples": json_ready(samples),
    }
    report.update(wiring_inventory(endpoints))
    report["canonical_evidence_quality"] = canonical_quality(canon)
    return report


def canonical_quality(canon: dict) -> dict:
    frame = pd.DataFrame(canon.values())
    cfg = training_cfg().gate
    confidence = float(cfg.raw_conf_threshold)
    consistency = float(cfg.consistency_fallback_threshold)
    fields = {}
    for field in ("volume", "pack"):
        observed = frame[f"{field}_set"].map(bool)
        low = pd.to_numeric(frame[f"{field}_confidence"], errors="raise").lt(confidence)
        inconsistent = pd.to_numeric(frame[f"{field}_consistency"], errors="raise").lt(consistency)
        fields[field] = {
            "no_observed_evidence": int((~observed).sum()),
            "low_confidence_total": int(low.sum()),
            "low_confidence_without_observed_evidence": int((low & ~observed).sum()),
            "low_confidence_with_observed_evidence": int((low & observed).sum()),
            "below_consistency_threshold": int(inconsistent.sum()),
        }
    flags = Counter(flag for values in frame.attribute_consistency_flags for flag in values)
    return {"scope": "All committed canonical records, using configured actual gate thresholds; confidence refers to persisted canonical scores.",
            "canonical_count": len(frame), "confidence_threshold": confidence,
            "consistency_threshold": consistency, "fields": fields,
            "flag_frequencies": dict(sorted(flags.items())),
            "source_disagreement_flag_frequencies": {key: count for key, count in sorted(flags.items())
                                                    if "sources_disagree" in key or "description_conflict" in key or "inconsistency" in key}}


def wiring_inventory(endpoints: dict) -> dict:
    """Configured inventory plus explicit code-reading findings; no veto inference."""
    from core.attribute_conflicts import CRITICAL_NAME_BY_CENSUS_KEY
    from core.attribute_universe import NON_YIELD_KINDS, attribute_registry
    from core.columns import raw_of

    cfg = data_cfg()
    consumers = {
        "sku_name_eng": "extract_all; canonical text; engine original-column reparse",
        "attribute": "extract_all; all registered dimension parsers; engine original-column reparse",
        "description_short_eng": "extract_all; engine original-column reparse",
        "sku_url": "extract_all URL volume/pack/variant evidence; identity-link scope",
        "image_url": "extract_all image-filename evidence (not image pixels)",
        "breadcrumbs_eng": "extract_all category-path variant evidence",
        "category": "extract_all category evidence",
        "brand": "canonical mode_brand and within-brand candidate grouping",
        "gtin": "GTIN trust, identity grouping, review policy, endpoint lookup",
        "sku_id": "reviewed identity/field correction scope and listing provenance",
        "retailer": "listing provenance; not a three_way_gate comparison",
        "country": "listing provenance; not a three_way_gate comparison",
        "sku_last_price": "captured commercial context; not read as gate attribute evidence",
    }
    columns = []
    for key, spec in cfg.column_evidence.items():
        column = spec.column
        columns.append({"configured_key": key, "raw_column": raw_of(column), "canonical_column": column,
                        "persisted_in_source_rows": spec.capture, "capture_reason": spec.reason,
                        "gate_input_path": consumers.get(column, "Unmapped code-reading inventory entry; inspect wiring"),
                        "sampled_rows_with_column": sum(column in row for endpoint in endpoints.values() for row in endpoint["source_rows"])})
    veto = set(training_cfg().rand_matching.targeted_veto_gates.veto_dimensions)
    dimensions = []
    for key, spec in attribute_registry().items():
        critical = CRITICAL_NAME_BY_CENSUS_KEY.get(key)
        if spec.kind in NON_YIELD_KINDS:
            role = "intentionally_ignored_constant_no_yield"
        elif critical in {"volume", "pack", "package_type", "pack_material"}:
            role = "mapped_critical_dimension_with_separate_numeric_or_packaging_checks"
        elif critical in veto:
            role = "eligible_categorical_gate_conflict_when_engine_reached"
        else:
            role = "engine_diagnostic_only_for_three_way_gate"
        dimensions.append({"key": key, "kind": spec.kind, "parser": spec.parser,
                           "critical_dimension": critical, "gate_role": role,
                           "persisted_channel": "volume_set" if key == "volume" else "canonical universe_evidence plus structured critical sets where applicable",
                           "registry_note": spec.note})
    ledger = [entry for endpoint in endpoints.values() for entry in endpoint["persisted_evidence_ledger"]]
    return {
        "capture_persist_gate_use": {"scope": "Code-reading inventory with column/registry/config vocabulary; row/ledger counts cover unique sampled endpoints only. Registry gate roles describe the engine conflict-key dispatch. A raw cell may also influence separately extracted critical sets; diagnostic-only does not prove the source text is unused elsewhere.",
                                     "source_columns": columns, "registered_dimensions": dimensions,
                                     "sampled_endpoint_count": len(endpoints),
                                     "sampled_ledger_field_counts": dict(sorted(Counter(entry.get("field", "unknown") for entry in ledger).items())),
                                     "sampled_ledger_entries_without_confidence": sum("confidence" not in entry for entry in ledger)},
        "wiring_findings": [
            {"finding": "Persisted evidence_ledger is diagnostic provenance; three_way_gate consumes extracted sets/confidence and source_rows, not the ledger directly.", "source": "src/pipeline.py:three_way_gate and generate_canonical"},
            {"finding": "source_rows carries original listing columns but no per-claim confidence; evidence_ledger supplies confidence for numeric volume/pack claims while other fields often lack numeric confidence.", "source": "src/pipeline.py:extract_all; src/core/columns.py:source_row_pairs"},
            {"finding": "Only configured critical dimensions can trigger the categorical veto. Other populated registry dimensions are evaluated diagnostically, not promoted automatically into identity blockers.", "source": "src/pipeline.py:three_way_gate categorical_conflicts"},
            {"finding": "Low volume confidence, low pack confidence, ambiguous volume, low consistency, and one-sided packaging level are explicit review pathways. They must not be scored as extraction mistakes solely because they are fallback.", "source": "src/pipeline.py:three_way_gate"},
            {"finding": "Confidence resolution after an early low-confidence return cannot occur through the later engine original-column reparse. The independent engine report therefore does not prove that the actual gate used its rescue.", "source": "src/pipeline.py:three_way_gate; src/core/attribute_decision.py:_fallback_reparse"},
        ],
        "low_confidence_investigation_options": [
            "Separate truly missing volume from present low-confidence volume by volume_status and raw spans in live extraction; count both by retailer/source column.",
            "Compare agreeing title/attribute/URL/image claims against numeric confidence fusion and canonical averaging; flag disagreement rather than treating corroboration as label truth.",
            "Inspect one-sided pack evidence before relaxing low-pack thresholds; a default single unit is not a declared pack count.",
            "Measure same-GTIN cross-listing consistency and semantic-family rescues independently from the actual fired gate; review conflicting listing variants before trusting a union.",
            "Evaluate any proposed low-confidence rescue on reviewed positive/negative labels, stratified by fired reason, preserving missing evidence as unknown.",
        ],
    }


def markdown_report(report: dict) -> str:
    lines = ["# Gate logic evidence review", "", report["scope"], "", report["status_definition"], "",
             f"Population: {report['population_pairs']:,} pairs. Reviewed: {report['sampled_pairs']:,} samples.", "",
             report["dashboard_limitation"], "", "## Configured stage coverage", "",
             "| Stage | Committed pairs | Coverage |", "|---|---:|---|"]
    lines.extend(f"| {stage['stage']} | {stage['committed_population']:,} | {stage['coverage']} |"
                 for stage in report["configured_stages"])
    lines += ["", "## Full committed decision census", "", "| Decision | Reason | Pairs |", "|---|---|---:|"]
    lines.extend(f"| {row['decision']} | {row['reason']} | {row['pairs']:,} |" for row in report["population_decision_reasons"])
    lines += ["", "## Canonical confidence and source disagreements", "", "```json",
              json.dumps(report["canonical_evidence_quality"], indent=2), "```"]
    inventory = report["capture_persist_gate_use"]
    lines += ["", "## Capture → persistence → gate-use inventory", "", inventory["scope"], "",
              "| Raw column | Canonical column | Captured | Consumer |", "|---|---|---|---|"]
    lines.extend(f"| {column['raw_column']} | {column['canonical_column']} | {column['persisted_in_source_rows']} | {column['gate_input_path']} |"
                 for column in inventory["source_columns"])
    lines += ["", "| Registered dimension | Kind | Gate role |", "|---|---|---|"]
    lines.extend(f"| {dimension['key']} | {dimension['kind']} | {dimension['gate_role']} |"
                 for dimension in inventory["registered_dimensions"])
    lines += ["", "## Static wiring findings", ""]
    lines.extend(f"- {finding['finding']} ({finding['source']})" for finding in report["wiring_findings"])
    lines += ["", "## Low-confidence investigation options", ""]
    lines.extend(f"- {option}" for option in report["low_confidence_investigation_options"])
    for sample in report["samples"]:
        current = sample["current_gate"]
        lines += ["", f"## {sample['gtin1']} ↔ {sample['gtin2']}", "",
                  f"Status: **{sample['status']}**. Similarity: {sample['similarity']:.6f}.", "",
                  f"Committed: {sample['committed_gate']['decision']} — {sample['committed_gate']['reason']}", "",
                  f"Current actual gate: {current['decision']} — {current['reason']}" if current else "Current gate unavailable: canonical endpoint missing."]
        if current:
            lines += ["", f"Actual fired stage: `{sample['actual_fired_stage']}`.", "",
                      "Independent attribute diagnostics (may include branches bypassed by the actual gate): "
                      + ", ".join(sample["independent_attribute_diagnostics"]["conflicts"]) + "."]
        if sample["inspection_flags"]:
            lines += ["", "Inspection flags: " + ", ".join(sample["inspection_flags"])]
        for side in ("left", "right"):
            if side not in sample:
                continue
            endpoint = sample[side]
            lines += ["", f"### {side}: {endpoint['gtin']}", "", endpoint["canonical"], "",
                      "```json", json.dumps({"sets": endpoint["extracted_sets"],
                                             "confidence": endpoint["confidence_and_consistency"]}, indent=2), "```", "",
                      f"All {endpoint['source_listing_count']} persisted source listings follow; live extraction evaluated {endpoint['live_extraction_listing_count']}.", ""]
            for row in endpoint["source_rows"]:
                lines += [f"- Listing {row.get('sku_id', '')} ({row.get('retailer', '')}): {row.get('sku_name_eng', '')}",
                          f"  Attributes: {row.get('attribute', '')}"]
            lines += ["", "```json", json.dumps({"persisted_evidence_ledger": endpoint["persisted_evidence_ledger"],
                                                   "live_listing_extractions": endpoint["live_listing_extractions"]}, indent=2), "```"]
    lines += ["", "## Source fingerprints", "", "```json", json.dumps(report.get("source_fingerprints", {}), indent=2), "```", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-reason", type=int, default=2)
    parser.add_argument("--listing-limit", type=int, default=3)
    parser.add_argument("--output", type=Path, default=RESULTS / "gate_logic_eval" / "evidence.json")
    args = parser.parse_args(argv)
    if args.samples_per_reason < 1 or args.listing_limit < 1:
        parser.error("sample and listing counts must be positive")
    gates = pd.read_csv(F["gate_results"], dtype=str, keep_default_na=False)
    report = build_report(gates, canonical_records_from_csv(),
                          samples_per_reason=args.samples_per_reason, listing_limit=args.listing_limit)
    paths = {"raw_dataset": F["dataset"], "canonical_records": F["canonical_records"], "gate_results": F["gate_results"],
             "pipeline_code": TRAIN_ROOT / "src/pipeline.py", "attribute_engine_code": TRAIN_ROOT / "src/core/attribute_decision.py",
             "regex_text_code": TRAIN_ROOT / "src/core/text.py", "critical_attributes_code": TRAIN_ROOT / "src/core/critical_attributes.py",
             "title_attribute_regex_code": TRAIN_ROOT / "src/ner/ner_product_attributes.py", "unit_canonicalization_code": TRAIN_ROOT / "src/core/unit_canonicalization.py",
             "training_config": TRAIN_ROOT / "config/training.yaml", "data_config": TRAIN_ROOT / "config/paths.yaml"}
    report["source_fingerprints"] = {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(markdown_report(report))
    print(f"Reviewed {report['sampled_pairs']} of {report['population_pairs']} pairs: {report['sample_status_counts']}")
    print(f"Evidence: {args.output}\nReadable report: {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
