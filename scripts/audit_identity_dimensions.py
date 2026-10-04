#!/usr/bin/env python3
"""Measure every registered raw identity dimension on the real catalog.

Read-only over datasets. Same-GTIN cross-retailer pairs measure feed disagreement;
same-retailer/title different-GTIN pairs measure hard product distinctions.
These populations are NOT unbiased samples of all product pairs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations, islice
import json
from pathlib import Path

import pandas as pd

from core.audit_guard import (
    attribute_dimension_guard_specs,
    guard_dimensions,
    self_comparison_parse_gaps,
)
from core.common import F, RESULTS
from core.gtin import normalize_and_validate_gtin
from core.product_dimensions import dimension_policy, row_dimensions, evaluate_dimensions
from core.text import normalize_retailer, normalized_attribute_text
from core.progress import tracked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=F["dataset_deduped"])
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "identity_dimensions")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pairs-per-group", type=int, default=20)
    args = parser.parse_args()
    if args.pairs_per_group < 1 or (args.limit is not None and args.limit < 1):
        parser.error("limits must be positive")
    frame = pd.read_csv(args.dataset, dtype=str, keep_default_na=False, nrows=args.limit)
    if not {"sku_id", "gtin", "attribute", "retailer", "sku_name_eng"} <= set(frame.columns):
        raise ValueError("catalog missing required identity columns")
    if frame.sku_id.duplicated().any():
        raise ValueError("duplicate listing IDs")
    policy = dimension_policy()
    records = frame.to_dict("records")
    evidence = [row_dimensions(row) for row in tracked(records, "identity dimensions", len(records))]
    coverage, values, unknown = Counter(), defaultdict(set), Counter()
    for record in evidence:
        unknown.update(record.unclassified_keys)
        for key, members in record.attributes.items():
            coverage[key] += 1
            values[key].update(members)
    facts = normalize_and_validate_gtin(frame.gtin)
    positive_groups, negative_groups = defaultdict(list), defaultdict(list)
    for i, (row, valid, gtin) in enumerate(zip(records, facts.gtin_structurally_valid, facts.gtin_clean)):
        if valid:
            positive_groups[str(gtin)].append(i)
            negative_groups[(normalize_retailer(row["retailer"]), normalized_attribute_text(row["sku_name_eng"]))].append(i)
    counts, examples, population = defaultdict(Counter), [], Counter()

    def collect(label: str, i: int, j: int) -> None:
        population[label] += 1
        evaluation = evaluate_dimensions(evidence[i], evidence[j])
        for name, result in evaluation.items():
            counts[(label, name)][result["status"]] += 1
            if result["review"] and sum(e["population"] == label and e["dimension"] == name for e in examples) < 3:
                examples.append({"population": label, "dimension": name,
                    "sku_id1": records[i]["sku_id"], "sku_id2": records[j]["sku_id"],
                    "title1": records[i]["sku_name_eng"], "title2": records[j]["sku_name_eng"], **result})

    for members in tracked(list(positive_groups.values()), "same-GTIN evidence", len(positive_groups)):
        candidates = ((i, j) for i, j in combinations(members, 2)
                      if normalize_retailer(records[i]["retailer"]) != normalize_retailer(records[j]["retailer"]))
        for i, j in islice(candidates, args.pairs_per_group):
            collect("same_gtin", i, j)
    for (retailer, title), members in tracked(list(negative_groups.items()), "different-GTIN title groups", len(negative_groups)):
        if not title or not retailer:
            continue
        candidates = ((i, j) for i, j in combinations(members, 2) if facts.gtin_clean.iat[i] != facts.gtin_clean.iat[j])
        for i, j in islice(candidates, args.pairs_per_group):
            collect("different_gtin_same_title", i, j)
    table = []
    for name in sorted(policy.attributes.keys() | coverage.keys()):
        row = {"dimension": name, "populated_rows": coverage[name], "coverage": coverage[name] / len(frame),
               "distinct_values": len(values[name]), "classified": name in policy.attributes}
        for label in ("same_gtin", "different_gtin_same_title"):
            stats = counts[(label, name)]
            observed = sum(stats[s] for s in ("equal", "overlap", "different", "unparsed"))
            row.update({f"{label}_{s}": stats[s] for s in ("equal", "overlap", "different", "unparsed", "unknown")})
            row[f"{label}_both_observed"] = observed
            row[f"{label}_difference_rate"] = stats["different"] / observed if observed else None
        table.append(row)
    # Fail-closed guards, one per registered dimension (all 37), built
    # through the shared spec helper (core.audit_guard). For each: its
    # observed value vocabulary must occur in the source attribute text, a
    # row compared with itself must never report that dimension "different"
    # or "unparsed", and its coverage must not be exactly 0% or 100% (a
    # degenerate dimension is reported unmeasured, not silently published).
    def _dimension_status(a, b, name: str) -> str:
        return evaluate_dimensions(a, b)[name]["status"]

    guard_specs = attribute_dimension_guard_specs(
        policy.attributes,
        values=values,
        source_texts=list(frame["attribute"]),
        populated=coverage,
        total=len(frame),
        evaluate_status=_dimension_status,
        evidence_samples=evidence,
    )
    guard_results = guard_dimensions(guard_specs, label="identity-dimensions")
    unmeasured = [r.name for r in guard_results if r.unmeasured]
    # Parser gaps surfaced by the self-comparison: a dimension whose own value
    # cannot be parsed back (e.g. Caffeine "200+ mg") reports "unparsed"
    # against itself. Reported, never silently absorbed.
    parse_gaps = self_comparison_parse_gaps(
        policy.attributes,
        evaluate_status=_dimension_status,
        evidence_samples=evidence,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(args.output_dir / "dimension_coverage.csv", index=False)
    report = {"rows": len(frame), "registered_dimensions": len(policy.attributes),
              "observed_dimensions": len(coverage), "unclassified_keys": dict(unknown),
              "column_roles": {name: policy.columns.get(name, "unclassified") for name in frame.columns},
              "malformed_attribute_rows": sum(bool(e.malformed_parts) for e in evidence),
              "pair_populations": dict(population), "pairs_per_group_cap": args.pairs_per_group,
              "dimension_evaluation": table, "review_examples": examples,
              "dimension_guards": [
                  {"dimension": r.name, "status": "unmeasured" if r.unmeasured else "passed",
                   "detail": r.detail}
                  for r in guard_results
              ],
              "guarded_dimensions": len(guard_results), "unmeasured_dimensions": unmeasured,
              "self_comparison_parse_gaps": parse_gaps,
              "decision_policy": "raw differences require review; equality is not identity authority"}
    (args.output_dir / "identity_dimensions.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[identity-audit] rows={len(frame):,} dimensions={len(coverage)} unknown={dict(unknown)} "
          f"pairs={dict(population)} guards={len(guard_results)} unmeasured={unmeasured} "
          f"parse_gaps={parse_gaps} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
