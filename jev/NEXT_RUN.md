JEV closed after the user-requested 50-pair repeat

Round 8 completed its 1,000 calls. The subsequent explicit request repeated 50 old pairs once each in round 9. All repeat request hashes match their prior requests; saved scores were never part of the JEV payload. Forty-eight judgment bands were stable and two moved from same to uncertain. No further calls are queued. See report_9.json.

Full evidence now participates in gate approval and compound product names participate in pair similarity. The rebuilt CSVs retain all 13,216 items and all 135,246 candidate pairs. Labeled pairs contain 197 positives and 606 negatives. Saved JEV diagnostics still flag some surviving positives; those are not verified truth labels. See full_evidence/saved_jev_replay.json.

The primary comparison remains all items and their changing partner sets, including isolated items: full_evidence/same_items_summary.json and all_item_partner_sets.jsonl.

Full training preparation is managed by training.prepare_all, documented in training_prep.md. Smoke 200 is deliberately left unchanged. Completion and input hashes are recorded by the preparation manifest; do not treat an in-progress or failed run as ready.
