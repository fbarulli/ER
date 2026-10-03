# TODO (updated 2026-10-03, branch main)

## Current training dataflow checkpoint — 2026-10-03

This section supersedes the earlier wave status below. Required sequence:
fixed preparation on local CPU → baseline encoding, training, inference and
ablations in one Colab GPU session → verified result collection and shutdown →
local metrics, ANN indexes, gate/JEV comparisons and dashboard reports.

Implemented and checked locally:

- [x] Checkpoint-native token policy shared across embeddings, text training and
  hybrid; native prompt/formatting binding, zero truncation, overflow rejection,
  strict token dtypes and complete row-count validation.
- [x] Population metadata excluded from model text inputs; contrastive feature
  column repaired; configured loss-specific sampler preserves all presentations.
- [x] Fixed training tokens and deterministic objective/split/calibration/sampler
  plans prepared locally; GPU entrypoint consumes frozen plans. Live ANN refresh
  follows config; unsupported noncontrastive live refresh fails explicitly.
- [x] Full local graph preparation: 27,820 listings, 28 configured query batches,
  train-only vocabulary/support; float32 features and int64 indices validated.
- [x] Portable hybrid input provenance, saved GPU inference for local reports,
  strict vector dtype and checkpoint/composition validation.
- [x] Prepared ablation templates bind selected weights on GPU; saved-result
  reporting runs locally without reopening GPU. Thresholds must match the saved
  baseline calibration and selected checkpoint.
- [x] Ablation temporary-path and interrupted-publication defects repaired;
  GPU shutdown precedes local retrieval reporting; current encoding logs retained.
- [x] Suite inputs use the existing Git publisher and cloned tar.gz transport,
  including recovery inputs; no direct suite input upload to Colab.
- [x] Repeated composition/config copying reduced with output-parity checks;
  shared token and graph preparation helpers avoid parallel implementations.
- [x] Active lineage reconstructed for 11,569 graph pairs only after exact
  source/catalog/split/pair agreement. Reconstruction is explicitly identified.

Remaining items:

- [ ] Generate final production CPU text-export requests, pending-baseline
  token archives and all attribute-ablation templates against the final source
  and input-manifest hashes; verify the complete prepared package and save its
  immutable tar.gz through GitHub. Code exists; these production artifacts have
  not yet all been generated.
- [ ] Resolve the current MNRL diet gate using evidence from the actual loss and
  batch construction, without silently relaxing it: current full bundle has
  neg_aug_frac 0.2592 versus minimum 0.3000 and surviving pos/neg ratio 2.9821
  versus maximum 1.5000. This still blocks the training suite.
- [ ] Run the end-to-end Colab GPU smoke: generate the missing production
  baseline cache, train, export inference and ablations in the same session,
  verify/persist results, close the session, then complete local analysis.
  The new GPU workflow has not been executed end to end.
- [ ] Verify real selected-checkpoint/full-catalog ANN ablation results and
  dashboard restoration from a clean clone of the saved artifacts.
- [ ] Complete graph baseline/candidate vector reuse in ablations; selected text
  vector reuse is implemented, but graph reuse remains a performance gap.
- [ ] Extend live ANN refresh to noncontrastive objectives if required; currently
  only contrastive supports it and unsupported requests fail explicitly.
- [ ] Populate difficulty, masking and generated-data axes from genuine source
  lineage where available. Current graph source labels contain only GTINs and
  the label; missing axes stay unknown, and graph augmentation is explicitly
  not applicable. Do not invent historical evidence.
- [ ] Evaluate shared graph support storage between training checkpoints;
  self-contained support remains duplicated for deployment compatibility.
- [ ] Verify large input/recovery transports against GitHub's file-size limit
  and provide a DVC fallback where needed; oversized Git transports currently
  fail explicitly.

- [ ] Negative-supply lane (owner ruling 2026-10-03): the gate has left the
      decision path in DESIGN — `src/training/negative_supply.py` emits
      real-partner-first + minted negatives, and the trainer consumes them
      behind `training.negative_supply.mode` (default `gate`, config-owned).
      Do NOT flip the default on "the code is in": flip only after the
      real-vs-minted discriminator (`scripts/negative_supply_discriminator.py`)
      clears AND the stratified eval shows model-alone >= the gate on
      real-pair recall and false-merge rate, split by gate decision and by
      differing attribute. The gate stays the SHADOW baseline until then.
      This flag measures nothing on its own — the open questions below do not
      go away: (a) anchors-with-real-partner count / coverage share; (b) the
      entity-level definition (`gtin` vs `sku`); (c) the flavor/material
      audit. Carry them here so the default is flipped on evidence, not on
      code being present. Minted partners are TRAINING-ONLY (`neg_minted`);
      the gate-derived miners (targeted-attribute, cross-brand) are dropped in
      lane mode and the lane's real partners are their replacement.

Audit evidence: [text/ANN](results/audits/text_training.md),
[GNN/hybrid](results/audits/graph_hybrid.md), and
[exports/post-training](results/audits/exports_posttraining.md).

## Earlier wave notes (historical; current status above takes precedence)

## OPEN GAPS — 2026-10-03 (ablation/embedding wave follow-ups)

| Gap | What's left |
|---|---|
| **Prepared training inputs** | DONE 2026-10-03: rebuilt `data/track_setup` post-rename
  (`graph_tracks.setup`): `listing_splits.csv`/`listing_pairs.csv`/
  `prepared/pairs.csv`/`prepared/report_attributes.json` were stale
  pre-rename copies (manifest already pinned the post-rename hashes);
  `text_prepared.pkl.gz` was a pre-rename-columns copy (refreshed
  from the post-rename `data/prepared/full/worker_1_baseline.pkl.gz`;
  bundle-embeds-current-CSV + graph-manifest hash checks all pass).
  `gnn_only` preflight now PASSES; `hybrid` fails only on the absent
  production text cache (separate gap below). Both graph tracks train
  + postprocess end-to-end on the 100-sample CPU smoke
  (`results/graph_tracks/{gnn_only,hybrid}__smoke_*_20261003/`,
  dev-Youden thresholds, retrieval + attribute-separation reports,
  report manifests with checkpoint/listings/pairs sha256). Stale
  tree backed up at `data/track_setup.pre_rename_stale`. BLOCKER for
  the suite supervisor path: the diet gate fails on every current
  bundle — see the diet blocker below. |
| **37-attribute evaluation** | Implemented and unit-tested, but no complete Colab ablation run has finished yet. |
| **Real trained checkpoints** | Verify prepared-token and graph-tensor inference against the existing text, GNN, and hybrid inference paths. |
| **Slice coverage** | Confirm real inputs populate difficulty × masking × generated-data lineage. Missing fields currently remain unknown. |
| **Full retrieval evaluation** | Current ablation ranks use sampled endpoints; full-catalog ANN effects remain unverified. |
| **Zero truncation across every entry point** | CLOSED 2026-10-03: guard consolidated into the shared loader `load_local_sentence_transformer` (`core/common.py:886-890`) — one adoption point now covers reranking (`rerank.py:68` bi-encoder, both `bi.encode` sites), train.py mask-effect scoring (`train.py:1550`), `core/nlp.py` encoder, `predict_items.py` production inference (`predict_items.py:128`), uniformity, rand_matching, zero_shot_sims, ann_refresh, and all 11 eval scripts. Remaining direct `SentenceTransformer` loads verified safe by construction: `encode_prepared_embeddings.py`/`ablation.py` feed pre-tokenized batches (sha256 + `tokenization_policy` validated) through `model(features)` — no tokenization at encode; `run_colab_embeddings.py`/`ablation_inputs.py` use `prepare_token_batches` (raises on overflow); `text_cache.py` keeps its explicit guard; `hpo_metrics.py:611` receives the already-guarded training model (idempotent). Verified: 31 passed (encoding_inputs/prepare_embeddings/encode_prepared_embeddings/attribute_ablation) + 83 passed (rerank/predict/nlp/training/smoke subsets); functional CPU check: loader-loaded `minilm_l6` reports `_zero_truncation_enabled`, short text encodes (1,384), 802-token text rejected with `zero truncation required`. |
| **Threshold attribution** | CLOSED 2026-10-03: `frozen_threshold` (`src/model_tracks/ablation.py`) now extracts the identity the source report attests — `model` (track) and `checkpoint` columns from CSV summaries, sibling/top-level keys from JSON manifests — with fail-closed unanimity checks, and `report()` verifies the attested track equals the request's track, the attested checkpoint name equals the ablated checkpoint's name, and (when locatable under the report's run root or via the manifest's absolute path) the checkpoint's sha256 identity equals `request['sources'][checkpoint]`. Provenance now records `track` + `checkpoint` alongside path/sha256. 2 new unit tests (CSV binding incl. track/name/identity mismatch rejection; manifest absolute-path binding). |
| **Publication** | CLOSED 2026-10-03: all implementation changes consolidated, committed (`01a8368`) and pushed to `origin/main`; smoke artifacts already on GitHub. |

The production embedding cache is also still absent—the smoke intentionally produced a separate cache. **BLOCKER (found 2026-10-03, pre-existing, not caused by the rename):** the diet gate (`scripts/diet_manifest.py`, wired into `model_tracks/preflight.py:64-74` unconditionally — smokes are exempt from stale-hash checks but NOT from the diet contract) fails on EVERY current bundle, so `model_tracks.run` (and any Colab suite launch) cannot start, not even a 100-sample smoke.

**Diet-gate evidence (surfaced 2026-10-03, from gate history and the realized bundle diet):**

- **JEV's role (corrected 2026-10-03):** JEV is an EXTERNAL REFERENCE only — a second opinion on our gating scores and pair-matching ability. Method: take a sample seen by JEV, compare its performance with another pair or the same pair, and extrapolate to trace back our gating performance. JEV is NOT a limiting factor and is NOT the basis for diet thresholds. (For the record, the JEV audit samples themselves are positive-skewed — rounds 7/8/9 = 720/180, 640/188, 40/10 pos:neg — but that composition is not a diet constraint.)
- **The only limiting factors are the loss function and batch creation.** The diet contract exists to keep the MNRL training diet healthy: enough augmented negative views so the loss does not overfit base hard negatives, and a bounded pos/neg view ratio so batches are not swamped by positive triples. The re-derivation must be based on what MNRL + batch creation actually require — not on any external reference population.
- **Current full bundle** (`data/prepared/full/worker_1_baseline.pkl.gz`, profile=baseline, re-measured 2026-10-03): positive views 56,162 (base 24,523 + declaration_dropout 7,283 + random 19,488 + swap_values 4,868); hard-negative audit rows 10,768 (base 7,937 + counterfactual 2,426 + random 243 + swap_values 162); MNRL survival excludes **23,589 dead masked-positive copies** (no source negative in the training pool) → surviving pos_views 32,573; `neg_aug_views` 2,831 / presentations 10,923 = **0.2592 < 0.30 FAIL**; `pos_neg_view_ratio` 32,573 / 10,923 = **2.9821 > 1.50 FAIL**. Recorded smoke ratios: smoke_200 2.6939, 100-sample smoke 2.5179 — every bundle fails the ratio ceiling.
- **Gate history (why the floors are miscalibrated):** both thresholds landed in `018c6e1` (2026-09-27) with no recorded calibration basis tied to the loss function; **no `DIET PASS` exists anywhere in the repo** — the 1.50 ceiling has never been satisfied by any bundle. `ead6966` (honest MNRL accounting) dropped the then-shipped bundle's `neg_aug_frac` **0.306 → 0.2122** (2,707/8,828 never-trained swap copies excluded; 13,124/17,098 masked positives dead). `e2e4cf5` (TIER 1(a) counterpart positives, 100% swap coverage: 2,216 replayed + 493 direct) projected the real bundle **0.2119 → 0.3058**, clearing 0.30 at the time. `09e2ce3` removed the phantom dynamic-mask +30% view projection (in-place replacement adds no views), making `neg_aug_frac` stricter still. The current bundle was rebuilt with a different augmentation mix (only 162 swap copies vs 2,707 then; counterfactual 2,426), landing at 0.2592.
- **Root cause (loss/batch framing):** (1) the **1.50 ratio ceiling has no recorded basis in what MNRL + batch creation tolerate** — the realized diet is 2.98:1 surviving pos views per neg view (5.14:1 before survival accounting), so either the loss genuinely needs a tighter ratio (and the augmentation/pair mix must change to deliver it) or the ceiling was mis-set. (2) the **0.30 neg-aug floor sits 4.1pp above the realized augmentation** (2,831/10,923 = 25.92%: counterfactual 2,426 + random 243 + swap-with-counterpart 162) — either 26% is a healthy MNRL diet (floor mis-set) or the augmentation fractions (`hard_negative_swap_frac` / `swap_agreed_frac` / counterfactual) must rise to deliver 30%.
- **Owner call needed (TIER 2 floor re-derivation):** derive both thresholds from the loss function and batch-creation requirements — what pos/neg view ratio MNRL tolerates per batch, and what augmented-negative fraction keeps the negative diet varied — then either set the floors to the realized healthy values (`ratio` ≥3.0, `neg_aug_frac` ~0.25) or change the augmentation/pair mix to meet the current floors. Then re-run the smoke through the supervisor.

**Still separate or incomplete:**

- Baseline embeddings and ablations have separate launchers.
- Ablations require an explicit preparation/evaluation step; they aren't automatically scheduled after training.
- ~~Reranking and other inference entry points haven't all adopted the shared zero-truncation guard.~~ CLOSED 2026-10-03 — guard consolidated into `load_local_sentence_transformer` (see gap table).
- ~~Threshold selection isn't strictly bound to the exact checkpoint/track.~~ CLOSED 2026-10-03 — attested track/checkpoint extracted from the source report and bound to the request (see gap table).
- Historical artifacts have incomplete lineage and mixed naming conventions.
- Sampled retrieval comparisons aren't integrated with full-catalog ANN evaluation.
- ~~Implementation changes remain uncommitted.~~ CLOSED 2026-10-03 — committed `01a8368`, pushed to `origin/main`.

Additional gaps found in the 2026-10-03 dataflow audit (fix-first):

- [x] `missing_axes` only flagged absent slice columns; a present-but-empty
  column (read as `''` with `dtype=str`) silently became a
  single-stratum grouping axis instead of being reported missing.
  CLOSED 2026-10-03: `sample_pairs` excludes all-empty object
  columns from the stratification axes (degenerate single stratum),
  `prepare` normalizes all-empty axes to `None` in the chosen
  records, and `missing_axes` treats `''` as missing
  (`src/model_tracks/ablation.py`). New test
  `test_prepare_reports_empty_slice_column_as_missing`.
- [x] Ablation directory checkpoints are not shipped in the
  input tar; the launcher validated local hashes but never
  checked the path exists in the remote Git clone before
  provisioning T4. CLOSED 2026-10-03:
  `scripts/run_colab_ablation.py` now pre-flights every
  directory source with `git ls-tree -r --name-only HEAD --
  <path>` (cwd TRAIN_ROOT) and fails fast with an actionable
  error when the branch tree does not carry it — the
  publication push ships this branch, so a committed
  directory is guaranteed present in the clone (the ablation
  bootstrap has no `dvc pull`). 2 new tests (untracked dir
  rejected; committed dir passes pre-flight into the launch
  sequence).
- [x] `scripts/run_colab_ablation.py` ran `report()` twice
  (save=False validation + save=True) and
  `frozen_threshold` five times per run. CLOSED
  2026-10-03: the launcher now computes the report
  exactly once (validation-only on the hash-checked
  download) and persists that validated output via the
  new `save_report` helper (`src/model_tracks/ablation.py`,
  extracted from `report`'s save branch); frozen_threshold
  runs 3x (1 pre-flight + 2 inside the single report,
  the second being the intentional tamper check). New
  test `test_launcher_reports_once_and_persists_validated_output`
  asserts report is called once with save=False and the
  validated dict is persisted, never recomputed.
- [x] gpu_only fold-metrics CSV: deferred rows
  lack auc/pr_auc/f1/etc. (NaN columns); per-row
  `deferred_local` markers carried the reason but
  there was no run-level statement. CLOSED
  2026-10-03: the results pointer (suite manifest)
  now carries a run-level `metrics_status`
  (`deferred_local` / `partial_deferred_local` /
  `available` / None) plus `n_folds_metrics_deferred`
  (`src/training/train.py` `results_pointer`,
  extracted as a pure helper). Also fixed a latent
  crash the audit surfaced: deferred rows have
  `status:'ok'` but no metric columns, and the
  pointer's `mean_auc`/`mean_pr_auc` used bare
  `r["auc"]` — a gpu_only run raised KeyError at
  pointer-write time; both now guard with
  `all("auc" in r ...)` like the calibration means.
  New test `test_results_pointer_states_deferred_metrics`.
- [x] `local_complete` interrupted-report rename was a
  fixed 5-name allowlist (was a `text__*` glob); any
  future `text__*` artifact would silently mix
  attempts. CLOSED 2026-10-03: the rename now globs
  every `text__*` path in the track output
  (`src/model_tracks/local_complete.py`) — checkpoints
  live under `_checkpoints/**` and are not matched.
  New test `test_interrupted_text_report_preserves_future_artifacts`
  (an artifact outside the old allowlist is preserved
  with an `interrupted-` prefix on retry, never mixed).
- [x] Staged `scripts/run_colab_embeddings.py` still
  had `r['product_id']` (KeyError on
  `--prepared-request` resume); the working-tree fix
  to `sku_id` must be staged too. CLOSED
  2026-10-03: staged the working-tree versions of
  `scripts/run_colab_embeddings.py` AND
  `tests/test_prepare_embeddings.py` (same
  divergence — staged catalog header was
  `product_id`, working tree `sku_id`); index now
  matches the tested working tree (0 `product_id`
  occurrences in either staged file; 22 tests pass
  against the staged content).
- [x] Baseline ablation rows carried `current_attribute_evidence`
  `{None: None}` (JSON `"null"` key). CLOSED 2026-10-03:
  attribute-less variants now carry the pair's full evidence
  map (nothing ablated → nothing to extract); variant rows
  keep the single ablated-attribute entry
  (`src/model_tracks/ablation.py` `report`). New test
  `test_report_evidence_omits_null_key_for_attribute_less_variants`.

## Continued extraction/gating investigation — 2026-10-02

Two agents use separate ownership: primary owns shared measurement/sugar
parsing and this ledger; additional agent owns pipeline numeric gate/pack
parsing. Existing generated-data edits and prepared-worker deletions preserved.

- [x] Correct added-sugar semantics: fingerprinted scan of 71,623 source rows
  screens 262 “No/Zero Sugar Added” titles; title-only `no_sugar` and negated
  sugar go 262 -> 0, `no_added_sugar` goes 0 -> 262. These are extraction
  assertions, not source conflicts, canonical movement, or match accuracy.
- [x] Numeric pack/volume vetoes obey configured dimensions in both early
  `pack_gate` and later overlap branches. Flavor-only policy previously gives
  hard_no for 355/1000 ml or pack 1/6; corrected examples proceed.
- [x] Reject nonfinite/out-of-range reliability at gate consumption: NaN
  confidence/consistency previously proceeds, now falls back. Definite
  configured categorical contradictions still veto. CSV loader continues
  accepting NaN; review happens when the gate consumes it.
- [x] Fix decimal/thousands pack boundaries, price-as-count artifacts, nested
  container multipliers, and Unicode multiplication (additional agent).
  Reproduced: SKU674502574/674559022 `38.304 bottles` -> 304 at .9;
  `$0.05 Bottle` adds false count 5 (including SKU674512045). SKU87399796
  `6 330 ml (Total1980ml)` and SKU90558864 `3 200 ml (Total600ml)` have
  unknown pack 1/0. Synthetic `0.5 bottle` -> 5 and
  `2 x 12 bottles x 330 ml` -> 2. Repair explicit arithmetic with provenance.
  During full title review, a proposed `Pack - N` extension misreads
  SKU955609156/961351592 “Pack - 12 Fl. Oz.” as count 12. Distinguish
  volume-bearing number from count; “Pack - 4-12 Fl. Oz.” can support 4.
  Pack provenance must also retain the original Unicode × span unchanged.
  Further full-scan counterexample: SKU153429386/989087387 “Cortas Combo
  Pack - 1) ... & 2) ... Total 2 Bottles” changes 2/.9 -> 1/.75 because
  item enumeration is mistaken for compact pack quantity. Preserve count 2.
  Further real missing evidence: SKU111529819 “Three Pack 16oz Bottles” ->
  unknown; SKU863131687 “LT.1.5 X 6BT” -> unknown count and no title volume;
  SKU476889523 “12 Oz ... Total of 72 Oz” -> unknown count, total wrongly
  package volume; SKU1014485680 “6 Sticks per Box (Pack - 12)” -> 12
  without inner/outer distinction (72 sticks). Preserve hierarchy and derive
  count only from explicit compatible total/per-unit arithmetic.
- [x] Fix mixed/improper fraction volumes without breaking count/size `24 / 2oz`;
  distinguish per-serving/ingredient quantity, total volume, and package size.
- [x] Diagnose model-input golden mismatch for SKU1041744561: golden has
  `volume_ml_8` for dry drink mix; HEAD already omits it. Baseline reproduction
  confirmed independently of this session's code changes. Do not restore a
  false liquid volume merely to satisfy stale golden bytes.
  Full 855-row golden comparison exposes 13 SKU-text movements. Review
  pack regressions before accepting: SKU508628351/483090740 `Pack of 10,`
  -> unknown; SKU689989863 `24 x 8 fl ounce`, SKU54290452 `4 x 1, 5l`,
  SKU955789718 `Pack - 6`, SKU7842215 `6x20 organic cl` lose count.
  Distinguish pre-existing HEAD failures from new boundary changes. Any golden
  correction must retain old bytes and an independently justified expectation.
  Later broad check adds SKU114282359 explicit “Six Pack” -> 6 instead of
  unknown. The older unit test treating “12 cases” as 12 consumer units is
  stale against HEAD's outer-count policy; pin unknown unit count and retained
  outer count 12 instead. Eight golden corrections preserve historical bytes.
- [ ] Full fingerprinted HEAD/working extraction comparison, fresh affected
  canonical aggregation and gate replay; keep extraction and gate populations
  separate and do not claim independent accuracy from rule comparisons.
- [x] Preserve negated ingredient evidence across listings: `generate_canonical`
  unions positive sweetener types but drops negative sets. Two listings of
  one GTIN declaring “No sucralose” and `Sweetener: sucralose` therefore lose
  the cross-listing conflict. Union negatives and flag intersections.
- [x] Recover explicit separated numeric URL quantities before noise filtering:
  `drink-330-ml` loses 330 and `drink-24-pack` loses 24. Protect only spans
  recognized by shared measurement/pack grammar; preserve ID/hash controls
  and keep copied title/URL/image surfaces in one confidence source group.
- [x] Synthetic declared-field parser boundaries: `Volume: 1, 5 l` and
  `Volume: 1 / 2 l` incorrectly give 1 ml at .9; `Volume: 1.5 dl`
  gives 2 ml because the optional unit regex omits dl and accepts a unitless
  prefix. No spaced-decimal, fraction, or decimal pack attribute examples
  matched in the full source scan. Reject fractional count prefixes and use
  shared volume grammar within the declared field; do not claim prevalence.
- [x] Gate review loses description contradictions: canonical “no sugar” title
  + “Contains sugar” description flags `description_conflict:sweetener`, yet
  self-pair proceeds. Likewise “with pulp” + “No pulp” flags but proceeds.
  Review configured conflicted dimensions while preserving definite unrelated
  categorical vetoes. Config comment says sweetener excluded while live
  `veto_dimensions` includes it; correct stale rationale, retain runtime policy.
- [x] Date usage audit: no timestamp column or date attribute keys in the
  71,623-row export. Initial numeric calendar screen finds 7 title/10
  description rows; wording screen finds 20 title/779 description/14 category
  path rows (overlapping, lexical counts; dot dates and “Best BY” not yet in
  screen). Examples: SKU918764187 `Best BY: 11-13-2024`, SKU53164338
  `best before 2018-12-31`, and SKU87983382 `best before 26.3.2024`.
  Preserve date spans and explicit expiry/manufacture roles as review context;
  ambiguous date ordering stays ambiguous. Expiry belongs to stock/batch,
  so a difference alone must not become a product-identity veto. There is no
  collection timestamp to establish how current these listings are.
  Extend review evidence to explicit shelf-life durations and expiry references
  such as “Best Before (See Base)” without inventing a calendar date; preserve
  two-digit-year expiry text as ambiguous-century evidence.
  Date full-scan counterexample: descriptions saying “not Best Before /
  Expiration UK is DD/MM/YYYY\n8-12 DAYS DELIVERY” incorrectly attach
  delivery ranges (and decimal oz/lbs on the next line) to expiry. Require
  same-clause cues, exclude measurement/delivery suffixes, and retain negated
  date-format guidance as guidance instead of a positive expiry assertion.
- [x] Registry routing bug: `VETO_CENSUS_KEY_BY_DIMENSION` maps numeric
  `pack` to categorical `pack type`, while `count per unit` is separate.
  `_universe_value` consequently substitutes pack counts into `pack type`
  and loses actual package-format evidence. Route count and format to their
  respective keys, keeping direct gate policy and missing-count semantics.
  The sweetener eligibility ledger also incorrectly describes current policy
  as excluded; distinguish historical loss measurement from runtime config.
  Correct routing exposes a targeted-gate disagreement on frozen pair
  688267001253/688267001574: missing curated package type borrows raw
  `aerosol` and hard rejects despite `targeted_package_type_conflict=0`.
  Keep raw format comparison in full evidence; permit package-format veto
  only with two-sided curated format evidence, matching direct gate policy.
- [x] Explicit sweetening conflicts not consumed: `Sweetener: unsweetened,
  cane sugar` emits `unsweetened_with_declared_sweetener` yet self-pair
  proceeds. Route explicit contradictory sweetening states to sweetener
  review, preserving unrelated veto precedence. `no_added_sugar_with_cane_sugar`
  is a separate claim/ingredient uncertainty; review its semantics and
  population before changing hard-negative policy.
- [x] Synthetic zero-quantity safety: `extract_all('0 ml bottle; total 600 ml','')`
  now divides by zero in derived-pack arithmetic; `Volume: 0` raises instead
  of remaining unknown. Invalid/nonpositive measurements must not become
  package evidence or crash extraction. Not a measured corpus prevalence.
- [x] Removed-feature rationale error: registry notes call sports ingredients
  3.6% and coffee type 2.6% “below” the 2.5% veto-band floor. Correct the
  arithmetic and distinguish within-GTIN feed disagreement from separability
  between different products. The band is advisory; it cannot establish that
  a feature lacks signal or that enabling a hard veto is identity-safe.
- [x] Surface newly extracted date roles in the gate review listing cards,
  including ambiguous dates and shelf-life duration. Raw source strings alone
  do not expose the parsed interpretation; keep stock context separate from
  the gate's deciding clause and do not infer current expiration without a
  collection timestamp.
- [ ] Fresh registry census reproduces 71,623 rows / 26,214 valid-GTIN rows /
  30,182 same-GTIN pairs, but flavour distinct sets are 338 versus pin 333
  (+1.50%, outside 1%). Keep the pin unchanged; diagnose baseline/source
  provenance. Current diagnostic census is separate from the production pin.
  HEAD's unchanged flavour parser also reproduces 338 on the same source;
  this drift predates the current fixes, so the pin remains unchanged.
- [x] Original-column clarification routing error: critical extraction returns
  `flavor`/`carbonation`, but stage 7 expects `flavour`/`carbonization` and
  fails to route their claims. It also inserts sugar-status claims into the
  sweetener-ingredient registry, conflating different semantics. Correct
  aliases and use declared ingredient readers for the ingredient channel.
  Fixed SSOT aliases and ingredient-only readers; source contradictions route
  to review. Direct evidence: URL vanilla/orange now vetoes flavor; sugar
  versus no-added-sugar stays compatible; No Stevia plus declared Stevia
  routes to review. Tests deferred by user.
- [x] Missing-flavor supporting evidence gap: capture-correct replay finds
  86 passing pairs with two-sided made-from sets where neither is a subset.
  Include configured ingredient disagreement as review evidence when flavor
  is missing; preserve hard-veto policy. Evidence includes cabbage/tomato
  juice and Peapod vegetable versus peach/mango juice.


- [x] Sweetener absence scope error: `no artificial sweeteners` was emitted
  as `no_sweeteners`; 3,713 source-field occurrences in the current export.
  Preserve artificial-only and added-only scopes separately. Natural sugar
  or stevia presence must not contradict absence of artificial sweeteners.

- [x] Explicit sweetener ingredient capture gap: ingredient reader only
  recognizes cane/brown/raw sugar and stevia, dropping `made with sugar`
  and other explicit ingredient phrases (105 title fields, 462 description
  fields screened). Extend the shared explicit phrase grammar and retain
  negation; include product-bearing URL/image sources in the consumer.

- [x] Ingredient decision channel still consumes registry sweetening states
  and unmapped declarations as positive ingredients. Direct source proof:
  `Sweetener: unsweetened` versus `Sweetener: stevia` invents an ingredient
  hard veto. Preserve raw registry diagnostics but filter ingredient decisions
  to recognized ingredients, keeping absence/negation in their own channels.
  Fixed both canonical resolver and original-source clarification; raw
  diagnostics persist. Direct proof now proceeds without invented ingredient
  conflict. Explicit source negation contradictions still review.

- [x] Ingredient absence suffixes are dropped: `aspartame-free` and other
  ingredient-specific `X free` phrases produced no negated ingredient. Source
  screen finds 11 title fields and 67 description fields, including SKU
  208935320 `Aspartame- Free`. Capture named ingredient absence separately
  from general sugar-status claims, and retain contradiction review.

- [x] Named-month dates and BBD expiry shorthand are dropped: SKU718539463
  `BBD: april 2024` produced no date evidence; now expiry month 2024-04. Named-date source screen finds
  one title and 145 description fields (including brand history). Capture
  calendar precision and original spans; classify BBD/BBE as expiry while
  historical dates without a stock cue remain unspecified calendar context.

- [x] Partial expiry date misclassified: SKU784178483 `exp. 20 / 09`
  was invalid under a month/year-only interpretation. Preserve possible
  day/month components without inventing a year; short dates such as 05/12
  retain both component-order and century ambiguity.

- [x] 2026-10-02 - `no_added_sugar_with_cane_sugar` now consumed by gate
  review (three_way_gate uncertainty routing; src/pipeline.py). FRUISS
  935465903 / RISE 49918733 style rows — one listing declaring both cane
  sugar and no added sugar — emit the flag and route the sweetener
  dimension to fallback/review, never hard_no; self-gate stays a plain
  proceed for clean siblings. Tests: tests/test_gate_categorical_source_
  review.py + extraction-level tests/test_added_sugar_semantics.py.
  Still open: measuring the same-source vs cross-listing absence-evidence
  screens (103 / 370 pairs) before changing further gate behavior.

- [x] 2026-10-02 - Multiplier pack unit boundary fixed
  (src/pipeline.py extract_pack_evidence): the x-multiplier/nested tails no
  longer consume any following letter as a unit; the tail must be a
  recognized measurement unit or container word, with one descriptive word
  tolerated before a measurement ('6x20 organic cl', 'LT.1.5 X 6BT' kept;
  '12x1 mineralwasser'/'12x1 pet'/'12x1 pet bottles' no longer fire).
  SKUs 935970386/935979247/955514786 spans repaired without changing their
  selected scalars (title corroboration unchanged). Tests:
  tests/test_url_separated_quantities.py (gated multipliers + whole-unit
  spans), all pre-existing pack pins (tests/test_pack_quantity_
  boundaries.py) pass unchanged.
- [x] 2026-10-02 - Boundary sweep across the quantity extractors (probe
  outcome): (1) x-multiplier/nested tails also swallowed the '.' of URL
  extension separators — '24x330ml.html' carried the raw span '24x330ml.'
  (66 sku rows); the span now dies at the unit's word boundary. (2) The
  volume unit grammar ended on (?![a-z]), so a non-ASCII glued word was
  consumed — '124 LÜ' SKU 404195454 read the 'Ü' into the unit; the
  lookahead is now any-unicode-letter. (3) Full-corpus OLD-vs-NEW
  extract_pack_from_title diff: ZERO changed pack readings — both fixes
  are span-level and strictly additive. (4) Glued-code junk from image
  CDN naming (a1l1/wO4L/GOzLa6L) was screened: shadowed by title/attr in
  826/880 rows and decodes to 0-the-rest when it surfaces (dead to the
  value>0 guard) — no behavior change shipped. Tests: tests/test_url_
  separated_quantities.py (extension separator), tests/test_unit_
  canonicalization.py (cased-word tail).

- [x] 2026-10-02 - Stage-7 sweetener claim lift wired + measured
  (core/attribute_decision.py + three_way_gate uncertainty routing). Claims
  only in the capture ("zero sugar"/"no sugar"/"diet" wording, no
  ingredient tokens) now lift the sweetener axis: claim-vs-claim resolves
  via the family predicate (compatible phrasings = MATCH/original_columns);
  apparent clashes and claim-vs-ingredient cross-channel comparisons
  downgrade to review ('claim_conflict' == source_conflict treatment).
  Replay census over the committed 135,246-pair universe: 549 marginal
  pairs, ALL toward review (proceed->fallback 188 labeled true pairs;
  no new hard_no, no new merges beyond the baseline 5 new_merge_risk vs
  18 baseline). Lexical claim comparison REJECTED by measurement (it
  split 133 true-positive GTIN pairs). Tests: tests/test_column_ssot.py
  (44 pass incl. both planted stage-7 cases).

- [ ] Sweetener absence evidence remains diagnostic across some comparisons:
  screen finds 103 passing pairs involving canonical absence plus ingredient
  evidence, and 370 with absence versus positive evidence across endpoints.
  These counts are screens; distinguish same-source contradictions, natural
  sugar, scoped artificial-only absence, and cross-listing disagreement before
  changing gate behavior.

- [x] 2026-10-02 - URL decimal pack notation dropped as noise (scoped +
  fixed). `_has_unit_suffix` accepted integer heads/parts only, so
  `6x1.5l`/`4x0.25l`/`12x50.7oz`/`12x0.33l` shapes failed the pack-notation
  branch and died under the no-vowel rule. Measured: 26 sku_url rows change
  extracted URL pack 1 -> real count; 24/26 corroborated by the title pack,
  2 rows (324800000 `6x33.8oz`, 766328750 `6x17.5cl`) gain their only pack
  count from the URL. No media-dimension regression (220x1280 stays dead).
  Fixed in core/url_evidence.py; tests: tests/test_url_separated_
  quantities.py (decimal shapes + retrievability). Remaining adjacent
  DEFERRED: the 'lt' unit alias is absent from config/paths.yaml
  url_evidence.units, so '5x15lt'/'9x3lt'/'4x1lt' (8 url rows) still drop;
  adding it changes 729 skus' bare-'lt' token handling — needs its own
  cabinet measure before a config edit.

Fix-first checkpoint: full 71,623-row extraction has zero errors; existing
extraction fields change on 16,942 rows and volume/pack assignment on 602.
The new date evidence field is additive on all rows, so that alone is not an
improvement count. Fresh canonical/gate replay is being corrected to include
exact production source-row captures. Earlier broad checks passed 574 tests
and exposed two stale expectations; source-reviewed corrections are retained
locally. User ruling: push fixes now and defer further test work.
Evidence: `results/regex_logic_eval/added_sugar_before.json` and
`added_sugar_after.json`; repeat with `scripts/audit_added_sugar.py`.

## Current work — measured gate/regex repairs (2026-10-02)

Evidence: `GATE_REGEX_REVIEW_20261002.md`,
`results/gate_logic_eval/evidence.json`, and `results/regex_logic_eval/`.
These counts are observations and rule counterfactuals, not independently
labeled accuracy estimates. Existing generated-data edits are preserved.

- [x] Audit all 13 source-column paths and 37 registered attribute dimensions;
  distinguish original evidence, extracted claims, diagnostic metrics, and
  actual deciding gate branches.
- [x] Freeze previously mishandled source listings and old outputs in
  `tests/fixtures/gate_regex_regressions.json` for before/after tracking.
- [x] Fix blocking-audit pack parser's undefined configured bounds.
- [x] Remove unused duplicate critical evaluation from the gate.
- [ ] Repair fractions/decimal whitespace, nutrition denominators, prepared
  yield/net-weight roles, nested pack quantities, and outer/inner counts.
- [ ] Capture explicitly negated sweetener ingredients and expose source
  contradictions instead of retaining an unqualified positive claim.
- [ ] Correct confidence fusion across all contradicting readers; prevent
  copied title/URL/image surfaces from acting as independent corroboration.
- [ ] Move extraction bounds/confidence policy into validated YAML; stop
  exempting arbitrary per-unit outliers solely because pack count exceeds one.
- [ ] Make early categorical vetoes obey configured dimensions, preserve
  trusted contradictions before review, and treat missing pack as unknown.
- [ ] Add repeatable before/after extraction/gate measurement, tracked sample
  assertions, source fingerprints, and regression/control tests.
- [ ] Validate candidate decision movement and freshly extracted affected
  canonical evidence separately; frozen-canonical replay cannot measure regex
  improvements. Do not repin census counts to conceal unexplained drift.

Measured starting points: 25,739 one-sided-pack rejected pairs (1,040 with
low-confidence known pack); removing that blocker alone moves 7,875 to review,
including 3,948 with other configured structured contradictions. There are
1,236 low-confidence volume canonicals (875 missing, 361 observed), 9,921
low-confidence pack canonicals (9,861 missing, 60 observed), and 1,252 volume
source-disagreement flags. The confidence scan found 49 ignored weaker
contradicting readers among 31,509 selected numeric URL/image rows. Preserve
these denominators when reporting improvement.

Follow-up after validated code: rebuild derived datasets/bundles using the
existing stage manifests, review census/label movement, and rerun independent
validation before training. Training-run calibration and independently reviewed
accuracy estimates remain distinct from parser regression fixes.

## Solo error audit checkpoint — 2026-10-02 (OPEN)

Candidate repairs and measurement tooling are checkpointed, not a claim that
all errors are fixed. Full refreshed extraction/gate replay and independent
accuracy validation remain pending. Latest solo audit reproduced:

- [ ] **Claim semantics:** 261 source titles contain “No Sugar Added”; the
  parser classifies this as `no_sugar` and negates the sugar ingredient.
  No positive/negative ingredient intersections occurred in those 261 rows;
  do not report them as 261 source-conflict flags. “Zero Sugar Added” also
  incorrectly retains `no_sugar`. Distinguish added sugar from total sugar.
- [ ] **Quantity boundaries:** IDs `674502574` and `674559022` describe
  19 pallets × 84 cases = 1.596 cases / 38.304 bottles. Extraction selects
  304 bottles at confidence 0.9, without a conflict flag; expected 38,304.
  `$0.05 Bottle` also generates bogus count 5 (including ID `674512045`).
  Seven source titles matched the decimal-container screening pattern;
  that screening count is not a count of confirmed erroneous winners.
- [ ] **Missing quantity signal / total roles:** `87399796` has 6 × 330 ml,
  total 1,980 ml; `90558864` has 3 × 200 ml, total 600 ml. Both return pack
  1 with confidence 0 and no pack evidence. Totals are marked package volume
  instead of total volume. Nine titles matched the total-volume screen.
- [ ] **Configured veto bypass:** with only `flavor` enabled in
  `veto_dimensions`, differing volume or pack still causes `hard_no`.
  Numeric veto branches must honor the same configured dimension policy.
- [ ] **Nonfinite confidence:** synthetic gate inputs containing NaN volume
  confidence, pack confidence, or volume consistency can return `proceed`.
  CSV canonical parsing accepts float NaN; reject or review invalid evidence.
- [ ] **Synthetic parser edge regressions:** `1 1/2 l` becomes 500 ml;
  `3/2 l` becomes 2,000 ml; `0.5 bottle` becomes count 5;
  `2 x 12 bottles x 330 ml` becomes count 2. No corresponding mixed-fraction
  or nested-container title matches were found in the current source scan.
- [ ] **Synthetic measurement role gaps:** “330 ml per serving; bottle 1 l”
  selects 330 ml; “contains 100 ml juice in 330 ml bottle” selects 100 ml.
  These are reproduced edge cases, not measured corpus error prevalence.

Previous focused verification passed 82 gate tests and 141 earlier checks.
After the latest changes, all 90 focused extraction/gate/measurement tests
passed; Python compilation and git diff whitespace checks passed. Broad model-input checks
previously showed a failure and need diagnosis before rebuild/training.
Keep generated dataset edits and deleted prepared workers out of this commit.

## SESSION LEDGER — GTIN + attribute capture + veto (2026-09-30, this branch)
### Landed (all measured; suite 840 passed / 2 skipped; selftest 279 oracles green)
### Open (owner calls / next training cycle)
- [ ] TIER 2 — re-derive diet floor on the fresh bundle's numbers.
- [ ] TIER 3 — calibration fracs (counterfactual/hard-negative; sweetener
  slice first) — needs training runs; sweetener exclusion precedent (12:1)
  stays in the ledger.
- [ ] +116 clause-level attribution could move beyond commits-level if a
  consumer asks (per-clause manifest diff between worktree states).


## PRIORITY ORDER (owner ruling 2026-09-29)
- **P1 — FINALIZE before training (data alignment etc.)**: bundle rebuild
  (Questions #1), MNRL telemetry port (reference-contract CSVs), swap-copy
  diet accounting (owner decision), masked-positive minting survival,
  coverage-analysis remainder, 4a39fdc verification remainder, flavor-twin
  policy decision, smoke-sampler code committed.
- **P2 — remaining open gaps** (parser/extraction, augmentation, training/
  eval, process) in "Open gaps".
- **DEAD LAST — graph/linkage additions** (record linkage, GNN/RQ-VAE
  semantic IDs — SKIPPED by owner verdict, graph-construction tiers, queued
  measurement). Do not start until P1/P2 closed.
- **P0 — VALIDATION REBUILD (owner ruling 2026-09-29, supersedes the 3k/5k
  lanes)**: emit ONE final validation CSV from a single merged component graph.
  Blocks any TIER 3 decision. Scoped below.

## P0 — VALIDATION REBUILD (owner ruling 2026-09-29: one final validation CSV) — COMPLETE 2026-09-30/10-01
Status: single merged component graph + `folds.derive_holdout` SSOT entry point;
`data/final_validation.csv` (6,351 = 565 pos / 5,786 neg, folds 2+3);
`results/training/validation_fold_map.csv` (14,946 entities); leak guards
raise-before-write; 9 leak regression tests; regen byte-identical after the
2026-09-30 wiring waves; selftest 279 oracles green. 3k/5k lanes: keys deleted
(paths.yaml + DataFilesSpec), producers deleted, final_inference retargeted to
the scored-pair population (schema-first); diet_manifest verified lane-free.
Residual open decision (NOT validation): scored-half thinness + slice-flag
set semantics — see Open gaps.

**Owner ruling.** Do not keep the 3k/5k lanes. Add everything back to training
and keep ONE final validation CSV. Fix the gates so they are measured on a
population the decision did not touch.

### The blocker that was found: 74.7% of current validation is contaminated
- `row_bc` (gtins) and `data/labeled_pairs.csv` (gtins) looked like disjoint
  namespaces: **intersection 0/5,428**. The training-side component split was
  structurally blind to the validation set, so neither side protected the other.
- Root cause of the apparent disjointness is missing normalization, not two
  identifier worlds. `norm(s) = strip non-digits, zfill(14)` gives:
  - validation gtins that ARE `row_bc` gtins: **5,428/5,428 (100%)**
  - positive pairs with both sides resolvable: **1,414/1,414 (100%)**
  - `row_bc` matching `dataset.gtin` (normalized): 14,901/14,921
  - raw `dataset.gtin` lengths are messy (7..14 digits), which is why the
    unnormalized join returned zero.
- Measured contamination of the CURRENT protocol (train = gtin folds 0+1):
  - POSITIVES: both gtins in train **23.3%** / one in train **51.4%** /
    clean **25.3%** -> only 358/1,414 are clean.
  - NEGATIVES: both **24.2%** / one **49.2%** / clean **26.6%**.
  - So the test-quarter P@R95 measures memorization as well as generalization.

### The fix: single merged component graph (feasible, verified)
- Union-find over the normalized entity namespace, with edges from BOTH
  `bundle['pos']` base positive pairs AND `labeled_pairs` positive pairs.
- Merged graph is safe to split (no giant component):
  - entities 14,921 -> **components 14,080**
  - **largest component = 15 entities (0.1%)**; nothing >= 50 entities
  - validation positives in components >= 50 entities: **0**
- Guarantee: because validation positive pairs contribute edges, both gtins of
  a positive always land in the SAME fold. Verified: **0 straddles in 1,414/1,414**
  positives. A test-fold positive therefore has neither side in train -> full
  leak 0% by construction.
- Negatives CAN straddle folds (mined negatives are not identity links). This
  matches current behaviour; treat it as a known, documented property, not a
  regression. Do not assert no-straddle on negatives.

### Resulting single validation CSV (folds 2+3, seed 1337)
- **758 positives + 3,873 negatives** (vs 1,414/7,722 total today).
- Training cost is effectively zero: training base positive pairs retained
  (folds 0+1) = **10,682 vs 10,692 today = -10 pairs (-0.09%)**.
- Per-field measuring power, old test quarter (~354 pos) -> new validation:

  | field | valPairs | distinct | singletons | largest bucket | old test | gain |
  |---|---|---|---|---|---|---|
  | carbonation | 654 | 3 | 0 | 362 | ~177 | 3.7x |
  | pack | 758 | 30 | 7 | 358 | ~142 | 5.3x |
  | volume | 758 | 39 | 9 | 170 | ~84 | 9.0x |
  | package_type | 267 | 7 | 3 | 166 | ~55 | 4.9x |
  | **sweetener** | **165** | 8 | 2 | 75 | ~38 | **4.3x** |
  | **flavor** | **390** | 34 real values | 4 | 42 | ~11 | **35x** |
  | pulp | 2 | 1 | 0 | 2 | ~1 | n/a |

- Fold balance is already adequate and needs no stratification to fix:
  `component_folds` gives gtins 3730/3730/3730/3730 and base positive pairs
  5299/5453/5249/5372 against an ideal of 5343 (within +/-4%). Stratified
  assignment is therefore OPTIONAL here, not the fix — the leak and the
  row_bc/labeled-pairs gtin join are the real defects. Revisit only if a slice is still thin.

### Gate realignment required (the "align our gates" item)
- `build_field_slice.py` buckets by TWIN (1 bucket per field, ~34% each from
  2,044 twins) while `labeled_pairs` slices by CANONICAL VALUE (163 flavors).
  These are different notions of a bucket, yet the checkpoint contract compares
  a twin-bucket P@R95 floor of 0.500 against field-value P@R95 as if equivalent.
  **Reconcile to one definition before either number is a decision gate.**
- **Flavor gates — CORRECTED 2026-09-29 (an earlier entry here was WRONG).**
  An earlier note claimed "107/163 flavor values are singletons, drop per-flavor."
  That counted FUSED PAIR COMBINATIONS (`{orange,lemon}` treated as one unit),
  not flavor values, and manufactured fake sparsity. The truth:
    - `flavor_set` has **34 real values**; true singletons = **4**.
    - Per-value coverage in the 758-pair validation: ginger 83, fruit 68,
      apple 59, coffee 45, lemon 40, aloe 30, peach 19, strawberry 19,
      **orange 18**, tonic 18, berry 13, coconut 11.
    - **6 of 31 values have n >= 30** -> per-flavor IS supportable for the top 6.
  Correct policy: **gate on the top-6 flavor aggregate; treat the tail
  (peach/orange/tonic and below) as INFORMATIONAL ONLY** — no floor. Orange at
  n=18 is a usable coarse signal but far below sweetener's largest bucket (75),
  so it must not carry a standalone floor.
- **Drop pulp as a gate entirely**: 2 pairs in validation. Unmeasurable at any
  budget that does not also break the component constraint. Root cause is
  POPULATION SCARCITY, not split or parsing weakness — do not "fix" an extractor:
    - `pulp_set` populated in only **302/13,250 (2.3%)** canonical records vs
      flavor 74.2% / carbonation 81.8% / sweetener 43.9% (~30x rarer).
    - 140/5,428 (2.6%) validation gtins; positives are the SAME product so
      "both sides flagged" reduces to "is this a pulp product" -> only
      **7/1,414 (0.5%)** of verified positives are pulp -> 2 in the val half.
    - Provenance is SOURCE ATTRIBUTES, not canonical text (hence the two-way
      disagreement: 301 flagged records whose text never says "pulp", and 25
      texts that do but are unflagged). Unioning the attribute signal recovers
      only 26 more gtins (+9%); 302/324 = 93% already captured. No real gap.
  Pulp becomes a gate only if the verified positive population grows for that
  category — better allocation cannot fix a 0.5% category.
- Per-slice gates that ARE supportable after this rebuild: volume, pack,
  carbonation, package_type, sweetener (aggregate per value, 8 values, largest
  bucket 75, only 2 singletons).

### The 16,345 dead masked-positive anchors — NOT validation (owner asked)
- Confirmed they must not enter the validation CSV. Reasons in order of force:
  1. **Distribution shift** — they are masked text (20-30% `[MASK]` tokens);
     P@R95 on them measures dropout robustness, not clean-text matching.
  2. **Circularity** — they are the OUTPUT of the augmentation knobs being
     tuned (`frac`, swap fields, twin composition). TODO's own doctrine: the
     deciding metric must be measured on a population the decision did NOT touch.
  3. **Not independent** — they derive from the same source anchors as training.
- Correct use: a **dedicated augmentation-invariance probe**, kept strictly
  separate from P@R95, answering "did augmentation damage the model?". The
  harness already exists in `build_field_slice.py` (twin buckets + the
  twin-bucket floor >= 0.500 in the checkpoint contract); it currently reads
  only `target_mode == "counterfactual"` and could gain a masked-anchor bucket.
- "We don't need 8k, add them all back": there is no 8k to add back. The 3k/5k
  files DO NOT EXIST (`training_data/` is absent), so nothing is currently
  reserved from them. The 7,722 figure is validation NEGATIVES, not reserved
  positives. The 16,345 dead anchors are a training-side issue the gate already
  excludes.

### Implementation checklist (LANDED 2026-09-30 — measured, not estimated)
**All targets below were re-measured, because the census drifted (1,414 -> 1,223
positives) and every number in this section was written against the old census.
Re-deriving them was the first task; a stale target is a wrong target.**

- [ ] **STILL OPEN — the scored halves are thin, and this is a real decision.**
      Withholding straddling negatives is the honest choice (one side is a
      trained-on entity), but it costs 4,741 of 5,781 negatives:
      **DEV 315 pos / 585 neg, TEST 249 pos / 455 neg.** A Youden fit and a
      P@R95 on 249 test positives is thin. The alternative is to assign each
      negative by the fold of its TRAIN-side entity so all 5,781 survive
      whole — but that changes what a negative measures and needs an owner call.
- [ ] **STILL OPEN — slice flags are set-valued and per side; gates must not
      compare `v1 == v2`.** Measured disagreement across the 564 positives:
      flavor 366, package_type 173, sweetener 155, carbonation 32, pack 31,
      volume 9. Same product, two feeds, different extracted sets. Treat each
      side as a bag of values. Flavor has 52 real values (10 with n>=30, 8
      singletons) after splitting the stored list-literal — the 129 "distinct"
      in the manifest are FUSED combinations, exactly the trap noted above.
- (closed 2026-10-01 — informational: `normalize_gtin` kept for CROSS-NAMESPACE
      edge resolution only; measured to contribute nothing to the split itself.)
- (closed 2026-10-01 — `scripts/diet_manifest.py` verified lane-free: bundle-agnostic, zero 3k/5k references; gate wired into the rebuild path.)

## P1 — FINALIZE before training (data alignment etc.)
- [ ] Port per-population coverage to the MNRL fold path so the next run
      publishes the reference contract's tracking CSVs (usage + type
      coverage per fold; populations from triples twin/masked/base).
      Reference: publication_manifest (DVC 20260913T123559565190Z) lists
      datapoint_usage_fold0.csv, datapoint_type_coverage_fold0.csv,
      pair_backprop_fold0.csv, loss_backprop_fold0.csv,
      masking_per_epoch_fold0.csv, mask_hard_negative_visibility.csv,
      train_rows_fold0.csv — the MNRL lane produces NONE of these
      (coverage writer is contrastive-only): telemetry regression to close
      before training.
- [ ] Verification of 4a39fdc remainder: claims 1-7 verified (5 PASS,
      1 partial, 1 FAIL). **2026-09-29 audit DISPUTED both remaining items.**
      (a) 49->30 ambiguous_volume arithmetic is WRONG: shipped code moved
          49 -> **79** raw rows (the 30 is 79-49, misread). Unit is also
          wrong — `_flag_census` (data_prep.py:136-157) counts per CANONICAL
          RECORD; real effect is **3 records (was 2)**: 8722200964525
          de-flags, 5021554989646 + 7311676670255 newly flag. 45 of the 49
          pre-fix hits have a MISSING gtin and are dropped by the guard
          (pipeline.py:1722-1738) before any canonical exists. THIRD
          inconsistent number lives at data_prep.py:59 ("49 -> 69") —
          not reproducible under any of 8 predicate variants. Replace all
          three with the measured 3 records.
      (b) claim 8 `flip_validity_audit.py` self-review: TWO real defects.
          - `_value_strings` rule R3 (centilitre, ml//10) emits a BARE
            integer the matcher accepts unitless: 671/817 of its hits (82%)
            land on unrelated numbers (pct100, caffeine 15 25, 3x). R2 is
            dead code (subsumed by R4). So volume/pack figures are
            CONTAMINATED: twin volume contradicted 0.5459 -> **0.4521**;
            swap_positive 0.5361 -> 0.4689; swap_negative 0.5802 -> 0.4987.
            pack is contaminated the OTHER way (new-side hits push rows
            into `both`): twin pack 0.4118 -> 0.4314.
            **`TODO.md`'s "volume 0.55" cites the contaminated figure.**
          - `_classify` (lines 122-123) probes only the FIRST token of a
            multi-token value: 13.1% of rows affected, so the metric can
            only UNDER-detect. sweetener_type disagrees on 74% of them.
          - All NON-numeric fields are bit-identical under every policy,
            so **the 717/717 flavor figure is TRUSTWORTHY** and the closed
            flavor verdict (P1) stands on it.
          - `global` section is a statistic the code never computes — see
            the withdrawn TIER 1(c).
      VERIFIED CLEAN: the frac-1.00/1.622 sweep found NO load-bearing site
      assuming either. `diet_manifest.py` is frac-agnostic (recomputes from
      the bundle's own arrays, reproduced 0.2119/1.0553 exactly);
      `prepared_bundle.py:155,193-199` interpolates live counts;
      `colab.py:2826-2839` `_tree_digest()` hashes config/* into the bundle
      cache key so the frac change invalidates the cache; no test pins 1.622
      or any frac. 2 cosmetic defects: diet_manifest.py:206-224 computes
      `pos_neg_ratio` and `effective_ratio` from the IDENTICAL expression
      yet labels the first "bundle-only (informational)" — it is not, and it
      is the value the gate consumes; diet_manifest.py:120-121,226-231
      print ACTIVE-config masking values beside bundle numbers that enter
      no arithmetic.
- [ ] Checkpoint eval contract execution: after each epoch/checkpoint,
      build_field_slice.py + minimal_flip_slice.py on the live bundle;
      P@R95 0.355 -> ~0.500 by epoch 3; twin-bucket floor >= 0.500;
      twin margin must lift off ~0.008. (Harness ready; blocked on the
      bundle rebuild + first training run.)

## AUGMENTATION TIERED ACTIONS (added 2026-09-29; HPO dropped — not in use)
Decision doctrine: sort every augmentation change by whether it needs a
TRAINING RUN to adjudicate. A row that is minted, audited, diet-counted and
then silently dropped is a LEAK, not a tradeoff — leaks are fixed, not
measured. Metric decisions apply only to rows that actually train, and the
deciding metric must be measured on a population the decision did NOT touch
(held-out P@R95), never on the augmented rows being questioned.


- [ ] **TIER 1(c) — WITHDRAWN 2026-09-29: the premise was false.**
      The "volume realized 1.428x
          the 0.35 cap (1.549x swap_positive)" figures came from
          `flip_validity_audit.py`'s `global` section, whose denominator is
          WRONG on two counts: (i) it divides by realized audit rows
          (8,790) while the caps bind against `_swap_pick_total`
          (train.py:1008-1011) = 10,415, inflating every share 1.185x; and
          (ii) the FIELD cap is enforced PER LANE (fresh Counter per
          function, masking.py:629,:827 — only `shared_value_counts` threads
          across lanes), so a "global" field share is a statistic the code
          never computes. Recomputed on the code's own arithmetic: field cap
          peaks **0.600x**, value cap **0.875x** — COMPLIANT, zero breaches.
          The soft fallback at masking.py:680-682 did fire but never pushed
          a lane past the cap. Residue is a soft/hard SEMANTIC mismatch
          (config wording vs behavior) needing a decision, NOT an
          enforcement fix. Fix the audit's denominator/reporting separately.
      (d) Prune dead knobs `swap_agreed_frac` / `hard_negative_swap_frac`
          (gone in 7967ccc/54f889f) from any surviving sidecar, so drift
          detection is not reading a stale schema generation.
      (e) Tighten bundle drift from WARNING to hard failure
          (prepared_bundle.py:283-299) — the current bundle loaded cleanly
          while carrying two deleted knobs. **SEQUENCING: land AFTER the
          rebuild.** Hardening drift makes the current bundle UNLOADABLE,
          which strands every open audit finding measured on it (the 717/717
          flavor figure, the volume contamination numbers, diet 0.2119).
          Rebuild first, then harden. Wire `scripts/diet_manifest.py` into
          `_build_local_training_bundles` (colab.py:2914-3007) at the same
          time — it is the only diet gate and today nothing calls it, so a
          rebuild completes "successfully" while the gate still exits 2.
- [ ] **TIER 2 — re-derive the diet floor AFTER Tier 1** (do not tune the
      data to hit a stale number): `diet_min_neg_aug_frac: 0.30` was
      calibrated against a denominator the gate now correctly rejects.
      Measured: honest MNRL = 0.2119; counting the 2,709 never-trained
      swap copies = 0.3058 (would have passed by a hair). `ead6966` made
      the denominator honest and dropped it under the floor. A floor has to
      be re-derived when its denominator's definition changes. Sequence:
      Tier 1 changes the numerator by construction, so re-derive once,
      after, not before.
- [ ] **TIER 3 — genuine calibration; requires training runs; decided on
      held-out P@R95 only**:
      (a) `counterfactual_frac: 0.10` and `hard_negative_frac: 0.30` —
          unswept, and the entire counterfactual hypothesis rests on the
          former. (Config-only route for hard_negative_frac alone is
          ~0.33, but that is a band-aid: prefer Tier 1(a), which makes the
          rows real rather than merely numerous.)
      (b) SWEETENER is the weak slice and gets the first run: margin
          **0.0017, 62.4% ranked correct**, 35.8% flip-SUPPORTED, n=109.
          Flavor second (0.0113/89.3%). `swap_max_donor_overlap: 0.95` has
          never fired (0/490) — an untested net, not a tuned value.
          CAVEAT: the 0.0017/62.4% margins come from
          `minimal_flip_slice.py`, which has NOT been self-reviewed (only
          `flip_validity_audit.py` was). The 35.8% flip-supported rate is
          from flip_validity_audit and is on a non-numeric field, so it is
          clean. Review minimal_flip_slice before treating the margin
          ranking as settled.

## Checkpoint eval contract (twin training curve)
- After each epoch/checkpoint: `build_field_slice.py --model <ckpt>` +
  `minimal_flip_slice.py --model <ckpt>` on the live bundle.
- Evaluation target: overall P@R95 0.355 -> ~0.500 by epoch 3;
  verify improvement on trained checkpoints.
- Invariance floor: per-bucket twin P@R95 must hold >= 0.500; overall up +
  twin down = over-smoothing -> raise counterfactual_frac or guarantee
  twin triples per batch (seeded stratified sampler, component-safe).
- Twin margin mean must lift off ~0.008; if ~= 0 post-training, reopen
  field markers as an ablation.

## Open gaps (P2 — remaining) — STATUS AUDIT 2026-10-01
### NOT SOLVED — needs work (4)
- [ ] Positive/negative coverage EXPANSION (the datagen action wave): 1,414 positives / 1,414 pack-blocker negatives is the floor, not the target — evidence exists in the census + veto audit; expansion executes after the running rebuild lands.
- [ ] 4a39fdc incorrect numbers: 3 inconsistent figures to replace with the measured "3 records" statement (being measured by the decision-briefs agent now).
- [ ] MNRL telemetry port: fold-census CSVs the reference contract expects (code-only; queued next wave).
- [ ] Duplicate-code hunt: remaining islands (NER dead cluster, hpo_persistence PG machinery, STOPWORDS residues) — LOW priority.
### DECIDED IN CODE (owner posture change: the system decides from evidence, 2026-10-01) — (4)
- [x] Thin scored halves: DECIDED `split.negative_fold_policy: "train_side"` (policy B) —
      evidence/measured at tests::test_scored_half_decisions + the DECISION block in
      src/training/build_final_validation.py: with policy B every negative scores by
      train-side entity fold: DEV 592 -> 1,087 negs (+83.6%), TEST 466 -> 957 (+105.4%),
      min_test_negatives=5 thin-cell share DEV 67.6% -> 64.9% / TEST 70.2% -> 67.2%,
      no trained-on endpoint enters the scored half under either policy. Policy A
      ("withhold_straddle") stays reachable via the same config key. Artifact regen:
      emitters support config; `PYTHONPATH=src .venv/bin/python -m src.training.build_final_validation`
      is the next regen command for the rebuild chain.
- [x] Slice-flag gate semantics: DECIDED `evaluation.slice_agreement: "set_bag"` (bag
      equality) — implemented at the count sites via count_slice_disagreements()
      (src/training/build_final_validation.py, imported by the consumer gate);
      measured on data/final_validation.csv ALL disagree counts byte-identical
      (volume 13, pack 48, package_type 153, sweetener 127, flavor 363,
      carbonation 38 — attribution recorded in write_manifest's comment;
      legacy "scalar" reachable via the same config key). Pinned in
      tests/test_scored_half_decisions.py.
- [x->SPEC] TIER 3 calibration fracs: FORMALIZED as documented-but-disabled
      `calibration_sweep:` block in config/training.yaml + CalibrationSweepSpec
      (src/core/schemas.py) — fracs sweep declared only when a training-run lane
      exists (fail-loud schema: enabled=false refuses non-null fractions; slice
      order pinned sweetener first), acceptance = the twin-bucket floor contract
      (per-bucket twin P@R95 >= 0.500; overall up + twin down = over-smoothing =
      reject; twin margin must lift off ~0.008). NOT run here (no training).
- [x->VERIFIED] Veto-eligibility: END-TO-END CONFIG-SOURCED — runtime veto consumes
      rand_matching.targeted_veto_gates.veto_dimensions only
      (src/training/rand_matching.py:320,422,450), schema allow-list gates what may
      be configured (src/core/schemas.py TargetedVetoGatesSpec._veto_dimensions_are_critical),
      and veto_eligibility_ledger (src/core/attribute_conflicts.py) reports the same
      config surface + the exact config delta per dimension; sweetener stays
      audit-column-only; no orphan/hard-coded veto logic found. (The veto-admission
      rule itself is being landed by the running veto-admission wave.)
### NOT SOLVED — process debts, do not block training (2)
- [ ] Metrics unversioned (results/ gitignored — today's evidence lives locally only).
- [ ] 41MB prepared bundles in git while DVC disabled (bloat policy).

## DEAD LAST — recent additions (2026-09-29; do NOT start until P1/P2 done)

### Standing rules for all new code (owner)
- SSOT config loading for EVERYTHING, paths included (RESULTS root,
  files./layouts. bindings; honors EUROMONITOR_RESULTS_DIR).
- graphify blast radius before/after every change (rebuild when HEAD moves).
- Every new code path EXECUTED (synthetic -> smoke -> real-data sandboxed).
- Limit test writing: guard tests only where behavior could silently regress.

### 1. Barcode-less record linkage — OPERATIONAL, residuals documented
- src/core/record_linkage.py + scripts/build_gtin_less_linkage.py;
  guard tests tests/test_record_linkage.py (5). Suite 561+2 green.
- Design (final): finalized-title comparison (SSOT build_sku_texts,
  title-side — attributes blanked, mirroring the payload title_only
  variant), brand blocks, cross-retailer-only, exact OR IDF-weighted
  Jaccard >= 0.7 on pack-stripped tokens, AVERAGE-LINKAGE agglomeration
  over DISTINCT titles (chain-drift guard), per-item uniqueness (corpus
  token-IDF) as measured feature, low-coherence cluster flag (margin 0.05).
- Full-run census (2026-09-29): 38,159 eligible; 3,867 multirow clusters;
  11,150 rows in them; 2,498 exact + 8,511 fuzzy links; 406 merges refused
  by the internal-mean guard; 500 low-coherence clusters flagged; ~2m5s
  (single finalized-text build, deterministic).
- Defect trajectory: union-find flavor merge (Obsesso 61-row cluster)
  -> fixed by block-IDF weighting + average-linkage over distinct titles.
  Verified clean on real data: Obsesso (Black/Mocha/Caramel/Latte
  separated), International Delight, Clearly Canadian, Aspire (variety
  packs reduced to same-flavor pairs).
- Residuals (documented, review-queue): Wandering Bear bl-020834-style
  mixes (cross-flavor sims 0.71-0.78 — too close to threshold for any
  global value; short titles, weak block IDF); Aspire multi-flavor variety
  listings need flavor-SET extraction (future work). 500 low-coherence
  clusters flagged for review. Linkage is candidate generation — the
  gate/verifier keeps the final say.
- GTIN-14 indicator-digit folding: MEASURED NEGATIVE — 46 GTIN-14 rows,
  0 fold to checksum-valid GTIN-13s, 0 match existing units; EAN-8 padding
  collisions 0; UPC-12<->13 padding collisions 0. NOT built (no yield).
- Retailer alias normalization: DONE (core.text.normalize_retailer SSOT,
  wired into blocking.py build_pairs/build_true_pairs, common.py
  kfold_barcodes, record_linkage). 4 alias groups merged (Voila/Voilà
  1,533 rows; publix/Publix 938; El Corte Inglés/Ingles 695;
  Shop Apotheke/shop-apotheke 184); 7 fake multi-retailer eval groups
  removed; blocking ground truth + k-folds now alias-safe.
- Brand variants (within-GTIN measurement, decision gate): 62/62 groups
  survive case/accent folding — variants are SEMANTIC (a shoc/
  adrenaline shoc/accelerator rebrand; olvi/kevytolo parent-co;
  fitaid/lifeaid sister brands), not typos. DECISION: config-owned
  brand alias map (vocabulary.json, like FLAVOR_ALIASES), NOT edit-distance
  fuzzy. TO DO: seed vocabulary.json brand_aliases from the measured
  62-group list (~30 distinct pairs) + wire into brand blocking/veto with
  veto-asymmetry doctrine.
- Uniqueness score: implemented as measured feature (corpus token-IDF
  mean over finalized text; CLI emits sku_id,cluster_id,uniqueness).
  Signal direction confirmed: linked rows 4.53 vs singletons 5.77 (fuzzy
  matches concentrate on generic listings). NOT gated on it yet.

### Consolidation + duplicate hunt (owner directive 2026-09-29)
- [ ] Hunt remaining duplicate code (grep for parallel loops, second
      normalizations, re-implemented helpers). Known suspects: duplicated
      regexes (TODO B10), brand/fold helpers (core.attribute_conflicts
      vs critical_attributes both define NFKD accent-fold — consolidate
      into core.text), STOPWORDS islands.

### GNN + RQ-VAE Semantic IDs: SKIPPED (owner verdict 2026-09-29)
Wrong architecture for a 61k-row matching pipeline (no generative
consumer, no scale pressure, quantization hides the exact distinctions the
vetoes need — mocha vs latte). Revisit only if: catalog ~100x, generative
ranker adopted, or real-time constraints appear. Cheap substitute if a
semantic bucket feature is ever needed: category_macros + linkage cluster
id (deterministic, auditable).

## Completed log (one-liners; verbose root causes in git history)
- 230fb13 record-linkage lane + finalized TODO priority order
- 9c10c71 flag census persisted in data_prep manifest (9c10c71)
- 3695c90 B8 triplet lane graceful skip (kept optional)
- 78fa96d MNRL subset monitoring + twin warmup (item 7, EXP-03)
- ead6966 honest MNRL diet accounting (swap negatives excluded, masked-
  positive survival reported)
- 5c012d1/4e8db93 pre-existing test failures fixed (title-wins, golden)
- 7a6af8b A3 ann_finetuned attribution; f401601 fix
- 06920ce B6 _optuna_mlflow_cb deleted + wiring guard
- 54f889f hard_negative_swap_frac dead knob removed
- 051c906 stale swap-mode comment/fixture fixed
- 65b270a/23f9115 F1 <source>+aug normalized; 5c012d1/4e8db93 test+golden
  fixes; 033c4ab reviewed source fixes
- Diet gate: frac 1.00->0.80, swap_agreed deleted; bundle rebuild pending
- Smoke 128 stratified regeneration; easy-negative replace=False;
  dynamic-mask diet projection removed
- 2026-10-01 - [x] GTIN integrity: longest-digit-run extraction (0 cells changed on this corpus — census regression byte-identical; the "3,715 invalid" now closes exactly as 3,646×11 + 50×10 + 22×7); `gtin_equivalent()` sibling equality (UP...
- 2026-10-01 - [x] model_input dead-def consolidation + description alias (`description_short_eng` OR `description`): deduped lane regains description evidence — 52,856/63,079 rows (was 0); +330 carbonation / +545 sweetener / +67 pulp fille...
- 2026-10-01 - [x] brand_aliases seeded in vocabulary.json (8 entries, 6 families: shoc chain, hi/hiball, fitaid/lifeaid, olvi/kevytolo, biotech, dg/ting) + 27 false-veto dissolutions, 22 declined groups filed; alias-aware fold wired at ran...
- 2026-10-01 - [x] AttributeUniverse (src/core/attribute_universe.py): all 37 raw keys registered (FieldSpec kinds/parser/conflict), census (rows/distinct sets/same-GTIN conflict rates; verify_census self-check), datagen_budget() (donor/vet...
- 2026-10-01 - [x] Capture wiring (additive, byte-prefix contract): pack material title∪attributes (canonical material sets 774 → 7,433 populated), juice content bands (27 canonical bands), [FIELD_PACK_MATERIAL] / [FIELD_JUICE_CONTENT_BAND]...
- 2026-10-01 - [x] Merge re-run + attribution: 71,623 → 63,079 (+116 vs last refresh — ATTRIBUTED to commits 0452692..2d3ac4b (GLN quarantine/collapse repairs), NOT to wave-1 fixes (verified byte-identical dedupe at HEAD-clean via worktree...
- 2026-10-01 - [x] labeled_pairs + gate pins: fallback 42,039 → 41,748 (−291 pairs now resolvable on evidence; proceed 1,239); pins updated together (common.py + selftest.py) with attribution cascade (proceed 1,737→1,506→1,395→1,239 narrati...
- 2026-10-01 - [x] Diet gate wired INTO the rebuild path (`_run_diet_gate`, colab.py:2886/3031): rebuild refuses to ship on gate failure; cached gate-failing bundles are never reused. Fresh worker_1: neg_aug_frac 0.3136 OK, ratio 1.1664 OK...
- 2026-10-01 - [x] TIER 1(e) drift hardened: `prepared_bundle_drift_strict` (default now TRUE post-rebuild; env override honored); stale bundle hard-fails, fresh loads; audit reproducibility kept via env=0.
- 2026-10-01 - [x] `pack_material` ADDED to veto_dimensions (owner go 2026-09-30). Evidence: hard_no 92,259 / both-populated 67,899 / disjoint 32,641 (48%). Identity-safe by construction (canonical-level union; verified byte-identical train...
- 2026-10-01 - [x] 3k/5k lane retirement: paths.yaml + DataFilesSpec keys deleted; run_ann_full_data.py + sample_deduped_dataset.py deleted; STOPPED live consumers: sample_balanced_pairs.py (miner/gate test pins), colab.py training_csv (own...
- 2026-10-01 - [x] Dashboard: /gate route (original-columns samples, per-dimension mismatch evidence w/ deciding-clause gloss, full strings in details), /datagen (session ledger + census auto-render) and /graphs track pages; all six routes...
- 2026-10-01 - [x] BLOCKER FIRST — single source of truth for split derivation. DONE. `folds.derive_holdout` is the entry point and now builds the GRAPH internally, so no caller can bypass the leak fix. `holdout_split` / `partition_componen...
- 2026-10-01 - [x] `normalize_gtin()` added as the single entity key — and MEASURED TO BE THE WRONG FIX. The "0/5,428 intersection / missing normalization" root cause above is incorrect. Measured on the live data: the deduped set holds 14,9...
- 2026-10-01 - [x] Validation positive edges folded in behind the single entry point. `folds.merged_component_graph` = training positives UNION labeled positives. Negatives are never unioned (a similarity claim is not an identity claim). Me...
- 2026-10-01 - [x] Leak closed, and the leak guarantee is a hard stop, not a log line. 0/1,223 positives straddle a fold; 0 test-fold positives have either side in train. `build_final_validation` raises BEFORE writing if either holds. Repro...
- 2026-10-01 - [x] ONE validation CSV emitted: `data/final_validation.csv`, 6,345 rows = 564 positives / 5,781 negatives (folds 2+3), 23 columns — `gtin1, gtin2, gtin1_norm, gtin2_norm, true_label, fold, fold_2, component_id, component_id_2...
- 2026-10-01 - [x] Complete split accounting emitted: `results/training/validation_fold_map.csv`, 14,981 entities -> fold + component (7,508 train / 3,742 dev / 3,731 test). Required, not optional: the validation CSV holds only the scored h...
- 2026-10-01 - [x] `evaluate_models.py` retargeted at the artifacts. It built its OWN graph from the labeled census ALONE with `component_split_k=2` — the leak site. It now reads the fold map; accounting closes 8,889/8,889; 0 scored pairs t...
- 2026-10-01 - [x] Leak regression tests in `tests/test_validation_leak.py` (9 tests). Includes `test_merged_graph_is_not_vacuous`, which asserts the bare graph DOES straddle before asserting the merged one does not — a first draft named on...
- 2026-10-01 - [x] RETIRED — the 3k/5k lanes (2026-09-30). paths.yaml + DataFilesSpec keys deleted; `run_ann_full_data.py`, `sample_deduped_dataset.py`, and its lane test deleted. Deliberately left in place (live consumers, evidence filed):...
- 2026-10-01 - [x] DONE (earlier this P0) — single SSOT split entry point. `folds.derive_holdout` builds the graph internally; `holdout_split`/`partition_component_pairs` banned outside `folds.py` and enforced by the selftest guard.
- 2026-10-01 - [x] DONE — `normalize_gtin()` scope fixed wrong: kept as CROSS-NAMESPACE bridge only (see the measured reversal at the top of this P0 — zfill on `row_bc` silently empties 8,559/14,981 filters; the real fix was the merged edge...
- 2026-10-01 - [x] DONE — validation positive edges folded in behind the SSOT (24,420 training + 1,223 validation over 14,981 entities; 0 straddles).
- 2026-10-01 - [x] DONE — ONE validation CSV emitted (`data/final_validation.csv`, folds 2+3, slice flags for volume/pack/package_type/sweetener/flavor/ carbonation; pulp excluded 2.3% population).
- 2026-10-01 - [x] DONE — leak regression tests (9 tests; straddle + both-sides-in-train hard stops inside `build_final_validation`).
- 2026-10-01 - [x] DONE — flavor reopen trigger restated on aggregate/twin buckets (top-6 aggregate gate; tail informational-only; policy CLOSED 2026-09-29).
- 2026-10-01 - [x] Bundle rebuild — LANDED 2026-09-30 with the gate wired in. The 0.2119 clause-1 FAIL was the STALE bundle (frac=1.00, pre-TIER-1(a) mint). Fresh worker_1 (built on data/dataset_deduped.csv, since the retired train_minus_50...
- 2026-10-01 - [x] Swap-copy diet accounting for MNRL: RESOLVED IN CODE (ead6966 excludes swap negatives from neg_aug_views when loss=mnrl) — fresh bundle diet numbers confirm the accounting (neg_aug 0.3136 on the mint-inclusive denominator...
- 2026-10-01 - [x] Masked-positive minting survival — stale figure refreshed on the fresh bundle: DEAD masked positives 16,345 → 14,832 @ frac 0.80 (quantized census high 9,660 / low 9,828). Diet REPORTS real survival. Remaining owner decis...
- 2026-10-01 - [x] Coverage analysis remainder — SUPERSEDED by results/attribute_ universe_census.json + the AttributeUniverse datagen_budget() class (measured per-key headroom, donor/veto/eval/channel classes). The agent-2 handoff items (a...
- 2026-10-01 - [x] Flavor-twin policy: CLOSED — accept flavor twins, no allowlist (owner verdict 2026-09-29, evidence-based). The "717/717 prose-contradicted" figure does NOT argue for exclusion: - the metric measures whether the old flavor...
- 2026-10-01 - [x] TIER 0 — unblock attribution (DONE 2026-09-29): `mnrl_monitoring. enabled: true` (config/training.yaml). `twin_loss_warmup` deliberately left OFF — it changes loss WEIGHTING, so it is a TIER 3 calibration decision, not ob...
- 2026-10-01 - [x] TIER 1(a) — counterpart positives for swapped negatives (DONE 2026-09-29): all 2,709 real `swap_values` hard-negative copies now reach a gradient; measured on the real bundle, 100% coverage. Mechanism: an anchor-only tran...
- 2026-10-01 - [x] TIER 1(b) — WITHDRAWN 2026-09-29: redundant, and unsafe to "fix". The 16,345/21,373 dead masked-positive measurement is REAL, but the diet gate ALREADY accounts for it: `diet_manifest.effective_pos_views` subtracts them,...
- 2026-10-01 - [x] Swap-copy accounting for MNRL (diet): 2,709 swap copies diet-counted as augmented views but 0% MNRL-train — ead6966 implemented the exclusion for MNRL. Diet side closed; the underlying 2,709 inert rows are now TIER 1(a)....
- 2026-10-01 - [x] T1.5 shipped (21705f9): same retailer + same checksum-invalid gtin + same product -> collapse. 108 groups collapsed, 11 escalated.
- 2026-10-01 - [x] T3 IDENTITY LOSS FIXED (data deletion, not noise). T3 keyed its collapse on (retailer, title) ALONE. T2 explicitly DEFERS rows whose trusted gtins disagree, and T3 then merged them anyway: 692 groups, 1,778 rows, 1,223...
- 2026-10-01 - [x] Verified after the fix: deduped 61,414 -> 62,963 rows; trusted gtins present 12,986 -> 13,250; canonical products orphaned 264 -> 0; 0 trusted gtins lost their last row (new hard invariant gate); closure 71,623 == 6...
- 2026-10-01 - [x] New hard gate: the dedupe now REFUSES to finish if any trusted gtin present in the input is absent from the output. This failure mode is silent and unrecoverable downstream, so it must never be a report.
- 2026-10-01 - [x] `core/sku_identity.py` (new SSOT): one descriptor bundle with every field a SET, so a descriptor restated across title/attribute/category collapses to one token and cannot move a comparison. `identity_conflict` is the...
- 2026-10-01 - [x] price / url / image_url removed from identity: `price` is a seller attribute, not a product description. T2 no longer keys on it, and representative choice uses DESCRIPTOR completeness (`descriptor_completeness`) instead...
- 2026-10-01 - [x] `results/training/dedupe_conflicts.csv` (new): the durable review queue. Anything the descriptor bundle cannot settle is ESCALATED, not guessed — 68 proven splits + 11 unresolved across 3,504 malformed-gtin groups. "Th...
- 2026-10-01 - [x] Fixed a silently DEAD dimension: `package_material` used the regex `pack\s*material`, which compiles to `pack\s*material\s*:` and never matches the real corpus key `Pack Material Type:` — it read empty across all 35,571 n...
- 2026-10-01 - [x] 17 new regression tests in `tests/test_dedupe_identity.py` pinning the identity partition, the T1.5 verdict table, completeness independence from price/urls, and absence-is-not-contradiction. Full suite: 586 passed, 2 ski...
- 2026-10-01 - [x] Consolidated the per-row model composition loop (sku_info -> model_input_info -> build_sku_text) into ONE source: core.model_input.build_sku_texts(frame, structured_enabled=...) -> (texts, infos). Refactored call sites: p...
