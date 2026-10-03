JEV sample ledger — updated 2026-10-02

| Round | Status | Sample | Checkpoint | Pairs / calls |
| --- | --- | --- | --- | --- |
| 1 | Tested | sample_doubled.json | audit_results.jsonl | 80 / 160 |
| 2 | Tested | sample_doubled_2.json | audit_results_2.jsonl | 55 / 110 |
| 3 | Tested | sample_doubled_3.json | audit_results_3.jsonl (not created) | 500 / 1000 |

Round 1: proceed split by mode-flavor evidence, fallback, and negative
similarity strata. Selection seed 29. Round 2: post-fix proceed survivors,
new negatives, new fallbacks, and retained negatives; its generation seed
was not recorded. Both rounds tested each pair in both directions.

Round 3: seed 43; 84 pairs in each positive stratum and 83 in each negative/uncertain stratum, across high/lower
similarity stratum. High similarity is >= 0.8. Polarity means the current
gate's decision, not an independent truth label. Retained the original 120 reserved pairs. Replayed 2,496 candidates
and selected for diverse critical and full-universe attribute states.
Coverage includes volume, pack, package type, flavor, carbonation,
sweetener, pulp, and pack material. sample_3_summary.json records source
hashes, candidate pool sizes, allocations, and attribute coverage.

Round 3 excludes all 135 pairs staged/tested in rounds 1–2. All 635 pairs
across these three rounds are reserved: exclude unordered pairs, so
swapping GTIN order cannot reintroduce one. sample_ledger.json records
explicit unordered pairs and sample SHA-256 hashes for each round.
Future samplers must read this ledger or all sample_doubled*.json files.
After a run, update status and completed-call counts in this ledger.

Reproduce round 3: `.venv/bin/python jev/stage_sample_3.py --pairs 500`.
Run it when ready: `.venv/bin/python jev/run_audit.py --staging jev/sample_doubled_3.json --out jev/audit_results_3.jsonl`.
Round 3 completed all 1,000 calls via OpenRouter (typesafe/jev-1.13).
The initial sandbox DNS errors are preserved separately in audit_errors_3.jsonl;
audit_results_3.jsonl contains the successful checkpoint. audit_run_3.json
records the adapter, model, question, completion time, and artifact hashes.

Round 4: 500 fresh pairs, 1,000 completed OpenRouter calls. Excludes all
635 prior pairs. Stratified split seed 44 assigns 250 pairs to merged gate
inputs and 250 to all original source listings. Both halves have 84 proceed,
83 negative, and 83 fallback pairs, with balanced similarity bands. Gate
inputs omit source_rows and do not include gate decisions. Original inputs
retain all source fields and full descriptions. Independent cohorts do not
establish the causal effect of the input representation.

Round 5: deliberate controlled repeat of round-4 pair 810036262576 /
810036266772 (Bones Coffee S'morey Time versus High Voltage). One unique
pair, both input representations, both orders: four completed OpenRouter
calls. This repetition is intentional and must not be counted as four
unique pairs. sample_control_5.json and audit_results_5.jsonl preserve it.

Rounds 4–5 save frozen input_states files, request SHA-256 hashes, raw
responses, model/adapter metadata, and timestamps. Reproduce verification
with `.venv/bin/python jev/verify_frozen_runs.py`. The ledger now records
1,135 distinct reserved pairs; round 5 repeats one of those intentionally.

Round 6: paired comparison of 100 fresh pairs, excluding all 1,135 prior
reserved pairs. Selection seed 46; 34 proceed, 34 rejected, 32 fallback.
The same pairs are judged with merged processed gate evidence and all
original listings, each in both orders: 400 completed OpenRouter calls.
Each request contains both products using one evidence format, not the gate
verdict or pair-level attribute agreement summary. input_states_6.json
freezes both formats; audit_results_6.jsonl saves responses and request
hashes; paired_comparison_6.json saves matched results. All hashes and scores
verified. The ledger now reserves 1,235 distinct pairs, plus the intentional
round-5 repeat. Future sampling must exclude the ledger's unordered pairs.

Round 7: staged offline; 900 fresh training pairs (720 stored positives,
180 stored negatives), plus 100 swapped original-input checks: 1,000
planned calls. Current gate replay and similarity stratification separate
stale training labels from current decisions. Frozen full original listings
and source hashes are saved; no live calls yet. See NEXT_RUN.md.

Round 7 completed: all 1,000 calls succeeded; request hashes and raw
response scores verified. Ledger status is tested. See RESULTS_7.md
and report_7.json for results and weighted fresh-population estimates.

Round 8 completed: final 1,000 calls, 900 fresh pairs plus 100 swaps. Primary cohorts: 640 current positives, 160 negative controls and 100 lost-positive diagnostics. All supplied source fields were verified against the official source loader, with no description truncation. Request hashes and saved raw scores verified. A preliminary unexecuted staging revision is retained under rebuild_8/staging_revision_0/. See rebuild_8/RESULTS_REVIEW.md; no further JEV calls are scheduled.

Round 9: user explicitly requested 50 old pairs again after the round-8 closure. Exactly 50 prior round-8 pairs were repeated once each: 20 low-score positives, 10 uncertain positives, 10 high-score positives and 10 negatives, seed 950. All 50 calls succeeded. All request hashes equal their prior request hashes: original listings only were sent to JEV; saved scores and gates remained local sample metadata. Forty-eight bands were unchanged, two same-to-uncertain changes; no score changed by 0.2 or more. This is a diagnostic stability check, not a fresh population estimate. See report_9.json. No additional calls are queued.
