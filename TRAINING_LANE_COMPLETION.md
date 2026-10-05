# TRAINING_LANE_COMPLETION.md — completion contract for the training-module-fixes lane

Adopted 2026-10-05 in `/tmp/opencode/er-train-review` (branch `training-module-fixes`
@ 4359736, clean, fully merged into main). Intent reconstructed from:
main `HANDOFF.md` §0/§0.1/§9, the branch's two commits (977316e "module
efficiency + attestation fail-closed + sampler pins", 4359736 "resume-state
audit script"), `TODO.md`, and the attestation/negative-supply/resume
sources in `src/training/`.

## State at pickup

The two lane commits are already merged into `main` (merge a6ebbe6) and the
pinned suites are as follows at pickup:

- `tests/test_prepare_all.py` + `tests/test_data_gate_attestation.py`: 18 passed.
- `tests/test_training_module_fixes.py` + `tests/test_training_module_telemetry.py`
  + `tests/test_training_handoff.py`: 35 passed (30 new pins from the batch).
- `tests/test_suite_training_controls.py`: 2 of 24 FAIL
  (`test_text_worker_passes_suite_controls[False]`/`[True]`).

## Root cause of the remaining gap (verified before any commit)

The 2 failures pre-date this branch and were left by 55578b0
("refactor(preparation): own CSV-to-bundle work in one process"), which
rewrote the text-worker adapter in `src/model_tracks/worker.py` from a full
`load_prepared_bundle` (verified load, ~19s "barrier straggler") to a
typed sidecar-header read:
`PreparedBundleManifest.model_validate_json(bundle.json sidecar)`.

The colocated test stub was partially updated (1e18b36): it stubs
`PreparedBundleManifest` as a SimpleNamespace with `model_validate_json`,
but the fixture writes NO sidecar file — the worker's real
`bundle_path.with_suffix(...).read_text()` raises FileNotFoundError
before the stub is ever consulted. Confirmed pre-existing by running the
same test at 55578b0, 1e18b36, and this branch @ 977316e — ALL fail both
parametrizations. The production code is correct: the sidecar read is
55578b0's intended behavior (the worker must NOT unpickle the full
bundle just for a header, PIN.md contract; the worker later emits
`input_validation configured` from the typed header only). The stale
stub is the defect: the batching commit 1e18b36 was supposed to keep the
colocated test aligned (its message says "test_suite_training_controls:
stub PreparedBundleManifest (sidecar)" — the stub was done, but the
sidecar file was never added to the fixture).

## Completion contract (in-scope here, what "done" means)

1. **Fix the sidecar-header adapter test** so the pinned suite passes:
   the test's fake must write the `bundle.json` sidecar header and pass it
   through the REAL `PreparedBundleManifest` (pydantic) rather than
   monkeypatching the class with a lambda — that keeps the pin honest
   (the worker really reads the sidecar; the stub only supplies bytes).
   No production-code edit: the sidecar read is 55578b0's intended
   behavior (worker must NOT unpickle the full bundle for a header,
   PIPE.md contract) and it is exercised here; the stale stub is the
   defect.
2. **do NOT** implement the diet-gate re-derivation (TODO "Resolve the
   current MNRL diet gate … neg_aug_frac / pos-neg ratio") — owner call,
   explicitly out of lane scope.
3. **do NOT** flip `negative_supply.mode` default (`gate` — TODO owner
   ruling 2026-10-03): discriminator/eval evidence required; the mode
   default stays config-owned and untouched.
4. **do NOT** change `colab` bundle-lane/`resume_from` semantics (D2
   resume decision is owner-pending; `scripts/audit_resume_state.py` +
   its telemetry pin remain the read-only audit evidence).
5. **do NOT** touch `/home/opc/ONE/ER` or other worktrees. All work in
   `/tmp/opencode/er-train-review`, committed granularly on
   `training-module-fixes`, branch stays local.
6. **do NOT** start full builds; code-only correction + pinned fixtures
   only.

## Contract-honoring done-ness

- [x] Pinned suite green: test_prepare_all + test_data_gate_attestation +
      test_suite_training_controls + the batch's own test files + any new
      test files, run as one command.
- [x] Test sidecar-header contract: fixture writes the sidecar; the
      failing 2 params pass; the OTHER 22 params of the file stay green.
- [x] No production-file edits; new code (the sidecar fixture helper) sits
      in the test module only; config untouched (single-writer rule;
      nothing needed for a test-only correction).
- [x] Verifier (fresh subagent, no self-approval) reports PASS or all
      findings fixed/justified.

## Open questions for the owner (out of this lane)

- Diet-gate re-derivation (TIERs in TODO) — data-policy decision.
- Negative-supply mode flip — awaits discriminator + stratified eval.
- D2 resume semantics change beyond the audit script — decision pending.
