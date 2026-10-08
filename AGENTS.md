# ER — build rules (the engineering contract)

Non-negotiable for every change. If you cannot follow a rule, say so and why — never silently
skip it.

## 1. Structure & design (code shape)

- **Everything behavior-bearing lives in a class.** No loose top-level functions as the
  public API. Module-level code is thin: constants + classes + a tiny entry point.
- **Single responsibility.** One function/method does exactly one thing. No god-functions;
  split anything large.
- **Factory paradigm.** Construction lives in a factory (a class with `build` / `from_config`
  / `from_...`) — never scatter object assembly across call sites.
- **Clear separation of concerns.** Orchestration, business logic, and I/O live in separate
  layers. No feature-specific logic leaking into shared paths; no implementation details
  leaking through an API.
- **Files must stay ≤ 1000 lines.** If a change would push a file past 1k, decompose first
  (extract helpers/modules). This is a hard limit, not a preference.
- **Canonical layer, no duplicates.** Before creating a helper, SEARCH the codebase. If it
  already exists (even in the wrong file), move it to the correct canonical file and reuse
  it. Never introduce a near-duplicate; never re-implement a canonical utility.
- **Prefer the higher-level abstraction over low-level ad-hoc code.** Model the domain with
  types/classes; do not drop to raw primitives (dicts/strings/tuples/ints) where a modeled
  type, helper, or existing abstraction exists.
- **Deletion beats addition.** No wrapper, indirection, option, or "for later" code that does
  not buy clarity. If an abstraction is just a pass-through, delete it.

## 2. Types & config (boundaries)

- **Type hints on everything** — every parameter and return value.
- **Pydantic** for data shapes and every external boundary (config, files, network, CLI
  args, env). Parse/validate at the boundary; trust internal types afterward.
- **SSOT.** Every value is declared exactly once; every consumer references that one source.
  No second registry, no duplicated default, no re-spelled literal.
- **No hardcoded vars.** Magic numbers/strings/paths/flags go in config (YAML/env) and are
  read from there — never literals buried in code.
- **Make illegal states unrepresentable** (enums/models over free strings/booleans).

## 3. Errors, logging, observability

- **Full tracebacks.** Every caught, handled, or reported error records the FULL traceback
  (`traceback.format_exc()` / `logger.exception(...)` / `exc_info=True`). Never truncate to a
  one-liner. Never swallow an error silently.
- **Log timing when time is a concern.** If something is slow or could be, measure and record
  it (timing/log line) — do not guess or leave it silent.
- **Fail loud on real errors; fail-soft only with a recorded reason.** An empty/no-op outcome
  must never masquerade as success.

## 4. Process (how to work)

- **Rewrite every non-trivial function at least twice.** First pass makes it work; second
  pass simplifies it (delete complexity, tighten). Do not ship the first draft of anything
  with a branch, a loop, a parser, money, or security.
- **Investigate blast radius BEFORE changing anything.** List every place the change reaches:
  callers, tests, fixtures, config, exports, contracts. State what it could break for users.
  That set is the scope; extra features are not.
- **Bug workflow:** ISOLATE → REPRODUCE (failing test first) → REPORT → FIX. **One commit per
  problem.** Root-cause once in the shared code — never a guard at the symptom.
- **Don't stop at the single fix.** A bug is a *class*, not an instance: after the reported
  one, grep the neighborhood for the same nature of error — same module/file, sibling callers
  of the same helper, the same pattern in adjacent code — and fix the whole class in the same
  pass, with a test that pins each.
- **Encode a repeated mistake as structure** — a class, a lint, a runtime check, or a pinned
  test — not as more prose. Pinning a regression with a test counts.
- **Verify all claims.** Prove behavior against the real artifact: run it, read the actual
  value, inspect the diff, run the test. Never rely on "it compiles", a self-report, or a
  proxy. Vet every number: find what limits it and rule out that it measured something else.
- **Keep it green.** Run lint/typecheck/tests; a new branch/loop/parser/money/security path
  leaves one small test behind. Trivial changes need none.
- **Sequence as small verifiable units**, each ending in a checkable state; order delivery so
  the sequence proves itself.

## 5. Writing code (laziness ladder)

Take the first rung that fully works:

1. Does it need to exist? Skip features/options nobody asked for (name them in one line).
2. Already in this codebase (helper/component/pattern)? Use it the way the code does.
3. Standard library / platform feature? Use it.
4. Installed dependency? Use it; never add a dependency for a few lines.
5. Can it be one clear line? One line.
6. Otherwise: the minimum code that works.

- **Smallest complete change**, in the codebase's existing structure, layers, and conventions.
- Comment only the *why* the code cannot show, one line. A known limit gets
  `shortcut: <the limit>, <when to upgrade>`.
- **Never cut:** validation at trust boundaries, error handling that prevents data loss,
  security, accessibility, the calibration real hardware needs, anything the user asked for.
- End every reply with one or two lines: what you skipped or did not check, and any risk the
  user must know.

## 6. Review bar (structural simplification)

When you make or review a change, hunt **"code judo"**: restructure so whole branches, helpers,
modes, layers, or conditionals *disappear* — do not just rearrange the same complexity. Refuse:
a file crossing 1k lines, ad-hoc conditionals bolted onto busy flows, feature checks scattered
through shared code, thin wrappers/casts/optionality churn, and logic living in the wrong layer.
Approval requires: **no structural regression, and no visible, untaken path to a dramatic
simplification.**
