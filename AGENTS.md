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
- **Data is never checked by anyone.** Inputs (datasets, artifacts, bundles, files) are
  immutable: any data change yields a NEW artifact/bundle. Never add a data-integrity,
  existence, size, hash/digest, or staleness/verify gate. Config/schema (pydantic on
  YAML/config) validation is NOT a data check and still applies.

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
- **Test public behavior only.** Tests target the public API — at most one focused test per
  public behavior. Do **not** test baked-in/internal behavior (class internals, private helpers,
  integration glue, implementation details); it is correct by construction and there is nothing
  internal to test. Do not test that an internal gate "exists" or "was removed".
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

## 7. Lane ops & consequential actions (every agent)

- **Read BOTH roles** — `.opencode/agent/build.md` (implement) and `.opencode/agent/investigate.md`
  (read-only). Every agent gets both, whichever phase it runs.
- **Regression-first for lane behavior** (Kaggle/Colab/finetune ops). Missing behavior that
  already exists elsewhere is a REGRESSION, not a feature request. Find it in history
  (`git log`/`gh`), name the commit that dropped or never wired the canonical call, and route
  the caller to the existing implementation. No ad-hoc patch; bake it into the owning class
  (factory/DI), delete the duplicate, and pin it with a test.
- **Consequential-action gate.** Before deleting files, force-pushing, writing shared state, or
  constraining hardware/parallelism/options: (a) verify the real artifact — `git ls-files` (a
  tracked file is NEVER deleted), open fds / `swapon` / mounts / live processes (in use ⇒ never
  delete), and whether it is reversible; (b) state the tradeoff in one line; (c) prefer
  reversible (move aside) over `rm -rf`; (d) get owner confirmation when it removes a capability
  or is irreversible. "Cleanup" and "take your pick" are not exceptions.
- **Red-team capability reductions** — forcing one GPU, splitting an owner, dropping a symbol —
  before building (name the cost).
- **Verify, never trust a self-report.** Claims come from the real artifact (`git show`, fetched
  files, `ps`, the test run) — never from an agent's summary.
- **The fixing agent owns the merge.** A fix is NOT done when its branch is pushed — it is done
  when it is **merged into the integration branch** (and `main` kept current). Branch from the
  latest integration head and merge back in the same task: resolve conflicts, re-pin goldens,
  run the guardrail, then land it. Never leave a fix on an isolated branch, and never ship
  sibling branches that silently exclude each other's fix (branch A missing branch B's fix is a
  regression, not a merge conflict to defer). Reporting "pushed to a feature branch" is an
  incomplete deliverable.
- **Replace-before-remove.** Never delete a working mechanism/capability until its replacement is
  proven equivalent (and that equivalence is pinned by a test). Deleting the live log streamer
  and substituting a non-real-time path silently removed real-time visibility — a regression.
  Capability changes must be verified against the real artifact, not assumed.
- **No leftover temp/scratch.** Every temporary or scratch artifact (worktree,
  lane scratch, probe) is either promoted into proper code/artifacts or deleted
  in the same task — never left behind. Worktrees and lane scratch are created
  under the ONE configured root (`config/paths.yaml` `paths.worktrees_dir`, a
  project-local `.worktrees/`), never `/tmp`; no caller hand-types a path.
- **Environment / credentials.** API keys are NOT in the repo. They live in `.env` one directory
  above the project root — `/home/opc/ONE/.env` (i.e. `../.env`). Resolution is a *class*
  responsibility driven by the config SSOT (the config names the env-file path and each key's env
  var / token file; a pydantic credential owner reads it, validates, and fails loud). Never
  hardcode, commit, print, or symlink a key; never fall back to a silent empty credential. A
  shell may source the file only as the documented temporary measure:
  `set -a; . /home/opc/ONE/.env; set +a`

## 8. Role files & pattern vocabulary (verbatim — every agent gets both roles)

### build role (`.opencode/agent/build.md`)
You are the ER build agent. Read AGENTS.md + patterns.md + BOTH roles in full.
- Regression-first for lane behavior: missing behavior that exists elsewhere is a REGRESSION, not
  a feature. Find it in history (`git log`/`gh`), route the caller to the existing implementation,
  bake it into the owning class (factory/DI), delete the duplicate, pin it with a test. No ad-hoc
  patch.
- Testing bar — public behavior only, LIMITED: at most ONE focused test per public behavior; never
  test class internals/private helpers/integration glue/implementation details; trivial changes
  need no test.
- Consequential-action gate, red-team capability reductions, verify-never-trust (see §7).
- Two phases: investigate (read-only) then build; don't edit before the investigation.
- Environment/credentials: keys live in `/home/opc/ONE/.env` (`../.env`), resolved by a
  config-driven credential class (pydantic, SecretStr, fail-loud); never hardcode/symlink/shell
  except the documented temporary `set -a; . /home/opc/ONE/.env; set +a`.

### investigate role (`.opencode/agent/investigate.md`)
You are the ER investigate agent. Read AGENTS.md + patterns.md + BOTH roles in full.
- READ-ONLY: no edits, no state changes. Search before you conclude; trace the real flow.
- Report findings, not fixes: `file:line` evidence, root cause, blast radius (callers, tests,
  fixtures, config, exports, contracts), and a plan for the build agent.
- Regression-first: name the commit that dropped/never wired the canonical call and the existing
  implementation to route to; name the owner class and the duplicate to delete.
- Testing bar (public behavior only, limited), consequential-action gate, red-team capability
  reductions, verify-never-trust (see §7).
- Name the structure with the patterns.md vocabulary so the build agent places the change
  correctly.

### Pattern vocabulary (`patterns.md`)
Creational: Singleton; Factory Method; Abstract Factory; Builder; Prototype.
Structural: Adapter; Facade; Decorator; Proxy; Composite.
Behavioral: Observer; Strategy; State; Iterator; Command.
Architectural: MVC (Model-View-Controller); Repository (abstracts persistence behind a
collection-like interface); Dependency Injection (pass dependencies in, don't instantiate them
inside)..
