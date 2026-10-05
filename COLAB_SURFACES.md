# Colab surfaces — outstanding work

- Regenerate prepared inputs and packages later, as requested. The interrupted `surfaces_20261004` preparation run is
  deferred and is not a verified handoff.
- Confirm the complete Colab/GPU lifecycle during the next authorized training
  run. No GPU training run was performed for these changes.

## Principles audit of the Colab/GPU training path (2026-10-05)

### Table of concrete findings

| File:line | Offending snippet | Principle violated | Severity | Genuinely wrong? | Explanation |
|---|---|---|---|---|---|
| src/graph_tracks/config.py:65-66, 95-96 | `retrieval_ks: list[int] = Field(default_factory=lambda: list(_default_ann_recall_ks()), min_length=1)` with validation that expects "unique positive integers". But `_default_ann_recall_ks()` returns config SSOT (`ann_retrieval_ks()`) or fallback `(1,5,10,50)`. The name/field is `retrieval_ks` in the graph-track config schema but it defaults to ANN recall ladder (SSOT). | P1 (CONFIG SSOT) / potential naming drift | medium | Mostly OK (uses SSOT via lazy import). Not a hardcode leak into runtime; the default_factory consults SSOT when available. Could be confusing (schema field named `retrieval_ks` for ANN ladder) but no runtime hardcoded literal that overrides config when core available. | Code path does call `core.common.ann_retrieval_ks()` if importable (line 22). Fallback literal only when standalone. In normal package import it uses SSOT. Not "hardcoded literal ladders" in the runtime path that bypasses config. |
| src/graph_tracks/config.py:22 | `return ann_retrieval_ks()` from SSOT. Also note fallback `(1,5,10,50)` on ImportError (line 24). | P4 (honesty) context | low | No. Honesty fine; clear fallback for standalone path. | Standalone mode is explicit; not fabricating metrics, just defaulting when SSOT unavailable. |
| src/training/training.py:5295, 5318, 5326 | References `_ks = tuple(_lc()["evaluation"]["retrieval_ks"])` for old/new protocol calculations; imports `_lc = load_config` inside function. Also elsewhere uses `retrieval_ks()` values? training.py reads config directly in spots (`load_config()["evaluation"]["retrieval_ks"]` at 3792). | P1 observation | low | OK. Uses config (`core.common.load_config`) not literals. No `(1,5,10)` literals embedded in code here. | Not a violation. |
| src/training/training.py:5322 | Comment mentions `ks=[1,5,10]` as example in docstring context (inline comment) — "ks=[1,5,10] — the floor..." (line 5322 in snippet). | P1 (cosmetic comment) | low | No — comment is explanatory/example, not code assignment. | No runtime effect. |
| src/cli/colab.py | No grep hits for `retrieval_ks`/`ann_retrieval_ks`/`report_thresholds`/`operating_precision`/threshold ladders or `[1,5,10]` etc. in colab.py. Also no SimpleNamespace constructions with thresholds/ladders. | P1 | none | N/A | Checked thoroughly; colab.py does not hardcode these ladders. |
| src/model_tracks/text_report.py:44-55 | Reads its OWN config: `text_config = setup / 'text.yaml'`; if missing raises FileNotFoundError with clear message. (Comment at 44-47 confirms the past bug was reading wrong lane config; current code reads text.yaml.) | P2/P3 | none (fixed) | Correct. Uses own lane config; fails loudly if missing. | Matches principles 2 and 3. |
| src/model_tracks/config.py:41-42 | `load_config(path: Path) -> SuiteConfig: return SuiteConfig.model_validate(yaml.safe_load(path.read_text()))` — no try/except to silently default. Raises on missing file (`Path.read_text`) or validation error. | P3 | none | Correct. | Fails loudly. |
| src/model_tracks/preflight.py | Reads staged configs via yaml; uses core.common.TRAIN_ROOT etc. No silent try/except that defaults config values; raises on validation failures. | P3/P2 | none | Correct. | No silent fallbacks observed. |
| src/model_tracks/package.py / worker.py / run.py | Also load suite config via model_tracks.config.load_config; no silent defaults. | P3 | none | Correct. | All read explicit paths. |
| src/cli/colab.py config loads | colab.py uses `load_config()` from core.common (line 2577 etc.) and reads sections; core.common does SystemExit on missing/invalid config (not silent defaults). | P3 | none | Correct. | Fails loudly per core.common implementation. |
| All colab-related code | No evidence of constructing evaluation config inline with SimpleNamespace/dict literals standing in for a config schema that contains thresholds/ladders (searched for SimpleNamespace + threshold in colab, model_tracks). | P1 (inline eval config) | none | N/A | No such pattern found in the paths examined. |
| Config drift (CLI flags overriding) | colab.py takes CLI args (loss, epochs, train_frac, etc.) and passes through to remote training. Some flags affect runtime behavior (epochs, loss, model registry key resolution). But thresholds/ladders are not passed as CLI flags in colab.py; they come from config. The audit comments show intentional removal of a flag that could conflict (`--mask-frac`). | P1/P4 | none identified as violation | N/A | No threshold/ladders passed via CLI in colab path that would override config. Model/params are runtime knobs; evaluation ladders remain config-owned. |

### Assessment

- Genuinely wrong vs false positives: The only item of note is the graph_tracks config field naming (`retrieval_ks`) that defaults to ANN recall ladder (SSOT). This is not a runtime violation (it reads SSOT when core is present). It's a naming/documentation point, not a code behavior that causes config to disagree at runtime in the package environment.
- Principle 1 (CONFIG SSOT): No hardcoded numeric ladders/lists in colab.py or model_tracks colab path. Evaluation ladders come from core.common accessors where used (e.g. `retrieval_ks()` used in training/reporting contexts). Comments show past hardcodes removed/redirected to SSOT.
- Principle 2 (LANE CONFIG CORRECTNESS): text_report.py now reads its own `setup/text.yaml` and fails if missing (with clear message). Other model_tracks files read the configs they own/are given explicitly. No evidence of reading a different lane's config in the examined code.
- Principle 3 (CONFIG MUST EXIST): All config loads examined either raise FileNotFoundError (`Path.read_text`) or use pydantic validation / SystemExit (core.common). No try/except that silently defaults config values found in colab/model_tracks paths.
- Principle 4 (HONESTY): No evidence of fabricating metrics; code records nulls where appropriate in reporting paths (e.g. text_report handles test skipping with explicit message). No silent fallbacks.

### Principles with NO violations found

- Principle 1 (CONFIG SSOT): No violations found in src/cli/colab.py, src/model_tracks/colab.py/run/worker/preflight/publish/resume, or scripts/run_colab_*. No hardcoded (1,5,10)/(1,5,10,50), threshold grids/lists, precision targets, or top-k values that duplicate config in the colab/GPU training path. Evaluation thresholds/ladders are sourced from config accessors where used.
- Principle 2 (LANE CONFIG CORRECTNESS): No violations found. Each lane reads its own staged config (text_report reads setup/text.yaml; graph preflight reads the specific track's yaml). No cross-lane config reads detected in the examined colab path.
- Principle 3 (CONFIG MUST EXIST): No violations found. Missing config files cause clear failures (FileNotFoundError/SystemExit/ValueError) with messages; no try/except that silently falls back to defaults observed.
- Principle 4 (HONESTY): No violations found. No metric fabrication detected; code is explicit about skipping test, raises on inconsistencies, and preserves honesty (warnings logged where appropriate, e.g. smoke diet threshold mismatch logged as warning with returncode 3).

### Summary

Search of COLAB/GPU training path (colab.py, model_tracks/*, colab scripts) shows no concrete violations of principles 1-4. The only noteworthy item is a naming nuance in graph_tracks/config (field named `retrieval_ks` used for ANN ladder) which remains SSOT-consulting and not a hardcode leak. All four principles have NO violations in the examined code.
