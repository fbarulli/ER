Next JEV audit — rebuilt labeled-pair training population

Compared current artifacts against round 6 (e6d0b46 and its saved summary):
- gate_results.csv and canonical_records.csv match round-6 SHA-256 hashes.
- labeled_pairs.csv matches the file at the round-6 commit byte for byte.
- No tracked gate, extraction, or pair-threshold changes since round 6.
- The regenerated labeled-pairs manifest is timestamped 21:13 on October 2;
  regeneration did not change the resulting CSV. Its recorded git SHA differs
  from the current checkout, so source hashes are the reproducibility anchor.

The actual training population is 15,837 pairs: 8,071 gate-derived positives
(proceed, similarity >= 0.50) and 7,766 hard negatives (hard_no, similarity
>= 0.80). Fallback and lower-similarity pairs are excluded by training policy.
The labeled_pairs.py module docstring still says positives start at 0.80;
the configuration and build manifest specify 0.50.

Round 6 covered 100 pairs using both evidence formats and both directions
(400 calls). Original evidence scored 30/34 approvals below 0.2 in both
orders. Ten pairs changed score buckets between input formats. Original
input had an order gap >= 0.2 for 2/100 pairs (maximum 0.38). These are
findings from a stratified diagnostic sample, not population error rates.

Efficient next run: 1,000 calls = 900 fresh unique pairs plus 100 reversed
order checks. Allocate 80% of unique pairs to training positives and 20%
to hard negatives. Within each label, spread random selection over current
gate decision and similarity bands (<0.8, 0.8–0.9, >=0.9), redistributing
capacity from small cells. This concentrates judgment on the approvals
that need investigation while retaining a negative control cohort.

Use all original listings with full fields/descriptions as JEV input.
Do not repeat the processed-input experiment across unchanged artifacts.
Freeze inputs, replay the current gate for all fresh eligible pairs, record
source hashes and gate transitions, and exclude all earlier staged/tested
unordered pairs, including unsuccessful calls and deliberate controls.
Training labels and gate reasons remain audit metadata, outside the request
state. Retain the existing model and exact-product question for comparison.

Reproduce staging in a NEW unused round:

    .venv/bin/python jev/rebuild_sample.py --calls 1000 --order-checks 100 --round 7

Run staged round 7 when ready:

    .venv/bin/python jev/run_audit.py --adapter openrouter --workers 6 --staging jev/sample_doubled_7.json --states jev/input_states_7.json --out jev/audit_results_7.jsonl

Report primary results on a_order only, by training label, current gate,
and similarity band. Report the swapped subset separately for order gaps
and verdict flips. Report primary stratum inclusion probabilities and use
population weights for estimates over the fresh eligible training subset.
Do not pool repeated calls as independent pairs. Do not call agreement
with gate-derived labels ground-truth precision or recall. No conclusions
about excluded fallbacks or previously audited pairs follow from this run.

Full current-gate replay over the 15,189 fresh eligible pairs found:

| Stored gate | Current gate | Pairs |
| --- | --- | ---: |
| proceed | proceed | 3,275 |
| proceed | hard_no | 3,185 |
| proceed | fallback | 1,149 |
| hard_no | hard_no | 7,554 |
| hard_no | proceed | 1 |
| hard_no | fallback | 25 |

The committed census and training labels are stale relative to current
gate behavior, despite matching round-6 artifacts. Sampling therefore
separates gate transitions rather than assuming stored positives remain
approved. Round 7 stages 720 stored positives and 180 stored negatives,
plus 100 extra order checks. All 1,235 prior reserved pairs are excluded;
648 of them occurred in this training population. This audit does not
rebuild or relabel the training CSV.
