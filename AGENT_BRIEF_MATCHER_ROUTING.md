You are the MATCHER-ROUTING agent. You own ONE worktree and a bounded set of files. Verify whether a routing regression is real, decide the correct semantics, fix it, and prove the fix on live data. Work autonomously.

# Your worktree (yours alone)
cd /home/opc/ONE/ER-verify-346f401 && git worktree add -b fix/346-routing /home/opc/ONE/ER-346-routing 2cfd772
Then soft-link shared data:
  ln -sfn /home/opc/ONE/EuromonitoR/dataset.csv /home/opc/ONE/ER-346-routing/dataset.csv
  ln -sfn /home/opc/ONE/EuromonitoR/dataset_model_input.csv /home/opc/ONE/ER-346-routing/dataset_model_input.csv
  ln -sfn /home/opc/ONE/EuromonitoR/.env /home/opc/ONE/ER-346-routing/.env
Run Python: cd /home/opc/ONE/ER-346-routing && PYTHONPATH=src /home/opc/ONE/EuromonitoR/.venv/bin/python ...
NEVER run GPU work or model fine-tuning/training. CPU-only is expected and sufficient.

# Data available inside your worktree
dataset.csv (soft-linked 54MB), artifacts/data/dataset_deduped.csv, results/canonical_records.csv, results/gate_results.csv, config/training.yaml, config/paths.yaml.
NOTE: running the real pipeline REWRITES results/*.csv — run `git checkout -- results/` afterwards. Never leave results/ dirty.

# OWNERSHIP — you may only edit these files
  src/training/rand_matching.py   (OWNER)
  config/training.yaml            (ONLY the `calibration.rand_matching.targeted_veto_gates` block, if a setting is genuinely the right lever)
  tests/test_targeted_ann_gates.py (you may UPDATE the cases this change affects, and add new ones)
OFF-LIMITS (other agents own these; editing them collides): src/pipeline.py, src/core/tracing.py,
src/core/schemas.py, src/core/hard_negatives.py, src/core/critical_attributes.py,
src/core/attribute_conflicts.py, src/core/ranking_metrics.py, src/training/training.py,
src/training/train.py, src/training/sample_balanced_pairs.py, src/training/folds.py,
src/training/generate_training_report.py, all OTHER tests/. If you need a change in an off-limits file,
REPORT it precisely instead of making it.

# THE SUSPECTED REGRESSION (I measured this on the base commit — reproduce it yourself first)
In `targeted_veto_gate` (src/training/rand_matching.py), the `missing` list is built by looping over
`CRITICAL_ATTRIBUTE_DIMENSIONS` — all SEVEN dimensions (volume, pack, package_type, flavor, carbonation,
sweetener, pulp) — and a non-empty `missing` routes the candidate to `missing_pack_or_volume_route` (human_review)
INSTEAD of auto_merge. The setting's own name says "missing_pack_or_volume".
MEASURED on the live artifacts, over the 1,592 candidate pairs the training gate labelled `proceed`:
  human_review 1,591 (99.94%)   auto_merge 1 (0.06%)
  missing-dimension census: pulp_a 1,565 / pulp_b 1,558 / package_type_a 1,086 / package_type_b 1,068 /
  sweetener_a 969 / sweetener_b 913 / flavor_a 566 / flavor_b 519 / carbonation_a 170 / carbonation_b 190
For contrast, the PRE-commit behaviour was reported as 1,592/1,592 auto_merge-eligible.
Note `pulp_set` is populated on only ~2.3% of canonical records, so a requirement that every dimension be
explicit is close to unsatisfiable — which makes `auto_merge` effectively dead.

# WHAT TO ESTABLISH (do not assume the regression framing is right)
1. Reproduce the routing distribution and the missing-dimension census above with your own script. Show it.
2. Determine whether this is a BUG or INTENDED caution. Read the surrounding doctrine carefully: the training
   gate (`pipeline.three_way_gate`) treats absence as UNKNOWN and does not fabricate agreement; explicit
   conflicts are hard blocks. Meanwhile the calibration gate's job is to decide auto_merge vs human_review.
   Argue from the code and the repo's own doctrine (the training gate's comments, the calibrator's comments,
   the Rand-Index work in PRESENT.md / FINDINGS.md if present) which behaviour is CORRECT, and state clearly
   what "missing" should mean: should an UNKNOWN dimension defer an otherwise-clean pair, or only an unknown
   dimension that is DECISIVE for the decision?
3. Quantify the CONSEQUENCE either way. Trace what actually consumes `targeted_gate_route`: find every reader
   (grep `targeted_gate_route`, `accepted`, `human_review`, `auto_merge`), and determine the concrete effect on
   the Rand-Index calibration and the submission lane — e.g. how many pairs can no longer auto-merge, and whether
   the calibration lane starves as a result. Report numbers, not adjectives.

# THE FIX (only if you conclude it is a bug)
4. Implement the minimal, correctly-scoped fix. The likely shape: `missing` must not blanket-defer on dimensions
   whose absence the decision does not depend on. Keep the REPORTING of missing dimensions intact (it is
   load-bearing audit data — `targeted_missing_attributes` and `..._count` are asserted by existing tests), and
   keep explicit conflicts a hard reject. Do NOT weaken the veto: a real conflict must still reject, and a pair
   with genuinely insufficient decisive evidence must still be able to defer.
5. If you conclude it is NOT a bug, do not change behaviour — instead add a regression test that PINS the
   intended 99.94% deferral with a comment explaining why it is correct, and report the evidence that convinced
   you. Refuting the hypothesis is a valid and valuable outcome; do not manufacture a fix.
6. If the right lever is a config setting rather than code, prefer the config change and say so explicitly.

# HARD REQUIREMENTS
- Every claim proven by an executed command with observed output, on LIVE artifacts (1,592 proceed pairs).
- Whatever you change, the full suite must stay green: PYTHONPATH=src python -m pytest tests/ -q
  (base is 72 passed). `tests/test_targeted_ann_gates.py` currently asserts the 7-dimension missing behaviour
  (`targeted_missing_attributes == "pack_b"` and an `auto_merge` case) — if your fix legitimately changes those
  expectations, UPDATE those tests and justify each change; do not delete coverage.
- Do not edit any off-limits file. If the correct fix REQUIRES an off-limits change, report the exact file, line,
  and patch you would apply.
- Commit your work in your worktree with a clear message.

# DELIVERABLE
Verdict (BUG or INTENDED) with the evidence; the reproduced routing table; the consequence analysis with consumers
named; your fix (or the pinning test); the before/after routing distribution on live data; files changed; test
count; and any off-limits recommendation. State what you EXECUTED vs READ. Concise, evidence-first, no padding.