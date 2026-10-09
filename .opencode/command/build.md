---
description: Build a change under the ER engineering contract (runs the build agent).
agent: build
---

Implement the following under the ER contract.

Mandatory injection (read before acting): `AGENTS.md` and `patterns.md` in the project root,
and BOTH role files `.opencode/agent/build.md` and `.opencode/agent/investigate.md`.

Standing rules:
- Regression-first for lane behavior (Kaggle/Colab/finetune): missing behavior that exists
  elsewhere is a regression — find it in history (`git log`/`gh`), route the caller to the
  existing implementation, bake it into the owning class (factory/DI), delete the duplicate.
  No ad-hoc patch.
- Testing bar: public behavior only, LIMITED — at most ONE focused test per public behavior;
  never test class internals, private helpers, integration glue, or implementation details;
  trivial changes need no test.
- Credentials: API keys live in `.env` one directory above the project root
  (`/home/opc/ONE/.env`, i.e. `../.env`); never symlink, never print; fail loud when absent.
- Consequential actions (delete, force-push, shared-state write, constraining
  hardware/parallelism): verify the artifact (`git ls-files` — tracked files are NEVER
  deleted; open fds / `swapon` / mounts / live processes ⇒ never; reversible?), state the
  tradeoff, prefer reversible over `rm -rf`, and get owner confirmation when a capability is
  removed. "Cleanup"/"take your pick" are not exceptions.
- Capability reductions (force one GPU, split an owner, drop a symbol) need a red-teamed
  opposing argument before implementation.
- Verify every claim against the real artifact — never trust a summary or self-report.

$ARGUMENTS
