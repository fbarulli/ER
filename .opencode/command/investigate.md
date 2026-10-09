---
description: Investigate read-only under the ER rules (runs the investigate agent).
agent: investigate
---

Investigate the following read-only. Report `file:line` findings, root cause, blast radius, and
a plan for the build agent (do not edit).

Mandatory injection: `AGENTS.md` and `patterns.md` in the project root, and BOTH role files
`.opencode/agent/build.md` and `.opencode/agent/investigate.md`.

Standing rules:
- Regression-first for lane behavior (Kaggle/Colab/finetune): missing behavior that exists
  elsewhere is a regression — find it in history (`git log`/`gh`) and name the existing
  implementation to route to. No new ad-hoc patch; name the owner class and the duplicate to
  delete.
- Testing bar: public behavior only, LIMITED — propose at most ONE focused test per public
  behavior; never test class internals, private helpers, integration glue, or implementation
  details; trivial changes need no test.
- Credentials: API keys live in `.env` one directory above the project root
  (`/home/opc/ONE/.env`, i.e. `../.env`); never print or read values.
- Flag consequential actions (delete, force-push, shared-state write, constraining
  hardware/parallelism): require verifying the artifact (`git ls-files` — tracked files are
  NEVER deleted; open fds / `swapon` / mounts / live processes ⇒ never; reversible?), stating
  the tradeoff, preferring reversible over `rm -rf`, and owner confirmation when a capability
  is removed.
- Capability reductions (force one GPU, split an owner, drop a symbol) need a red-teamed
  opposing argument first.
- Verify every claim against the real artifact — never trust a summary or self-report.

$ARGUMENTS
