"""Offline checkpoint integrity checks and current-gate replay for all audits."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "src"))
from training.gate_replay import canonical_records_from_csv
from pipeline import three_way_gate


def verify(sample_path, checkpoint_path, records):
    sample = json.loads(sample_path.read_text())
    rows = [json.loads(line) for line in checkpoint_path.read_text().splitlines()]
    key = lambda x: (x["gtin1"], x["gtin2"])
    staged = {key(x): x for x in sample}
    completed = {key(x): x for x in rows}
    errors = []
    if len(staged) != len(sample): errors.append("duplicate staged ordered pairs")
    if len(completed) != len(rows): errors.append("duplicate checkpoint ordered pairs")
    if staged.keys() != completed.keys(): errors.append("checkpoint and sample pair sets differ")
    pairs = defaultdict(list)
    for row in rows:
        k = key(row)
        if k not in staged:
            errors.append(f"unstaged pair {k}")
            continue
        if any(row.get(field) != value for field, value in staged[k].items()):
            errors.append(f"staged metadata differs: {k}")
        score = row.get("noul")
        if row.get("status") != "ok" or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            errors.append(f"invalid response {k}")
        if any(g not in records for g in k): errors.append(f"missing record {k}")
        pairs[tuple(sorted(k))].append(row)
    replay = []
    for pair, orders in pairs.items():
        if len(orders) != 2 or {x["copy"] for x in orders} != {"a_order", "b_swapped"} or key(orders[0]) != tuple(reversed(key(orders[1]))):
            errors.append(f"incorrect doubling {pair}")
            continue
        a, b = pair
        forward = three_way_gate(records[a], records[b])
        reverse = three_way_gate(records[b], records[a])
        replay.append({"gtin1": a, "gtin2": b, "stratum": orders[0]["stratum"],
                       "input_scope": orders[0].get("input_scope", "first_source_listing"),
                       "scores": [x["noul"] for x in orders],
                       "mean_score": sum(x["noul"] for x in orders)/2,
                       "order_gap": abs(orders[0]["noul"]-orders[1]["noul"]),
                       "current_gate": forward, "reverse_gate": reverse})
    strata = {}
    for stratum in sorted({x["stratum"] for x in rows}):
        scores = [x["noul"] for x in rows if x["stratum"] == stratum]
        strata[stratum] = {"calls": len(scores), "mean": sum(scores)/len(scores),
                           "below_0_2": sum(x < .2 for x in scores), "above_0_8": sum(x > .8 for x in scores)}
    cohorts = {}
    for scope in sorted({x['input_scope'] for x in replay}):
        scope_pairs = [x for x in replay if x['input_scope'] == scope]
        cohorts[scope] = {}
        for decision in ('proceed','hard_no','fallback'):
            subset = [x for x in scope_pairs if x['current_gate']['decision'] == decision]
            cohorts[scope][decision] = {'pairs':len(subset),
                'mean_score':sum(x['mean_score'] for x in subset)/len(subset) if subset else None,
                'both_below_0_2':sum(max(x['scores']) < .2 for x in subset),
                'both_above_0_8':sum(min(x['scores']) > .8 for x in subset)}
    return {"sample": sample_path.name, "input_cohorts":cohorts, "checkpoint": checkpoint_path.name,
            "staged_calls": len(sample), "completed_calls": len(rows), "unique_pairs": len(pairs),
            "integrity_errors": errors, "strata": strata,
            "max_order_gap": max(x["order_gap"] for x in replay),
            "mean_order_gap": sum(x["order_gap"] for x in replay)/len(replay),
            "gate_asymmetries": [x for x in replay if x["current_gate"]["decision"] != x["reverse_gate"]["decision"]],
            "reason_order_differences": sum(x["current_gate"]["reason"] != x["reverse_gate"]["reason"] for x in replay),
            "current_routes": dict(Counter(x["current_gate"]["decision"] for x in replay)),
            "low_score_proceeds": [x for x in replay if x["current_gate"]["decision"] == "proceed" and max(x["scores"]) < .2],
            "high_score_rejections": [x for x in replay if x["current_gate"]["decision"] == "hard_no" and min(x["scores"]) > .8],
            "pairs": replay}


def main():
    records = canonical_records_from_csv()
    reports = [verify(ROOT / "jev" / f"sample_doubled{suffix}.json",
                      ROOT / "jev" / f"audit_results{suffix}.jsonl", records) for suffix in ("", "_2", "_3", "_4")]
    out = ROOT / "jev" / "verification_results.json"
    out.write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        print(json.dumps({k:v for k,v in report.items() if k not in {"pairs", "low_score_proceeds", "high_score_rejections", "gate_asymmetries"}}, indent=2))
        print("low-score proceeds:",len(report["low_score_proceeds"]),"high-score rejections:",len(report["high_score_rejections"]),"gate asymmetries:",len(report["gate_asymmetries"]))
    print("Saved", out)
    return int(any(x["integrity_errors"] or x["gate_asymmetries"] for x in reports))

if __name__ == "__main__":
    raise SystemExit(main())
