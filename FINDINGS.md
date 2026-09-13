Commit	Verdict
9bb0138 gate teardown on publication	Confirmed working + 2 new risks (below). finalize_local_training_run runs inside main(); VM kept on any failure. Hardenings replace silent skips with raises.
9bb7129 ANN band config-driven	Good — explicit band_mode (fixed/adaptive_quantile/intersection), schema-validated (schemas.py:477), no silent fallback.
944d85d activate fine-tuned ANN refresh	G1 fully resolved — transformers 4.53.2 passes model= to callbacks, and the callback now stores self.model at construction and raises if None (training.py:1129-1142).
3f08933 independent deferred miners	ANN + attribute-conflict mined from one fine-tuned encode; slots allocated by target share with score fill; per-source W&B telemetry. Attribute-conflict is now actually mined at refresh (was never), resolving the initial-fold config gap partially — but initial fold remains gate-only by design.
ab7234d loss weight wiring	structured_feature_weight computed per fold and threaded into loss/evaluator/tail scoring — removes the import-time constant divergence.
8e19bfe missing volume → unknown	Fixes the phantom volume-conflict bug (attribute_conflicts.py:41-56). Partial: pack still fabricates {1} via or 1.
f8f36f5 uniformity + less neg masking	New uniformity.py (unrelated-pair collapse audit) wired into local reports/report.json + masking.hard_negative_frac: 0.25 (label-0 masking 100%→25%; positives unchanged).
New / regression findings
- N1 (MED): publish_local_wandb_artifacts raises if no WANDB_API_KEY/wandb_run_id (colab.py:1117-1138). But WandbCtx is disabled whenever the key is absent → run_id is null in status (training.py:802) → the whole finalization chain aborts and the VM is kept alive indefinitely (quota burn). Previously it skipped. A non-W&B run can no longer complete/teardown.
- N2 (MED): uniformity.enabled: true by default, and select_unrelated_pairs raises RuntimeError when <100 brand/category/token-disjoint pairs exist (uniformity.py:64-67) — small or heavily-overlapping catalogs strand finalization (→ N1-style VM hang).
- N3 (LOW-MED): H1 residual — if a refresh yields fewer candidates than slot capacity, unfilled slots keep gate text but are still counted ann_finetuned (ann_sources.get(pair_id, "ann_finetuned"), training.py:1819).
- N4 (LOW): attribute_conflicts.py:46 pack sentinel {1} still fabricated for missing pack.
- N5 (pos): per-pointer dvc pull (dvc_store.py:120-124) resolves the earlier argv-growth concern.
- N6 (MED, perf): refresh still encodes the full payload (incl. ~37k masked copies that no miner reads) and re-reads canonical_records.csv per refresh — unchanged.
- N7 (LOW): select_unrelated_pairs is O(n²); fees per-checkpoint CPU encode ×2; uniformity.py:168 dpi=150 literal.














Corrected standing finding (goes in Corrections)
- G1 REFUTED: installed transformers is 4.53.2; callback_handler.call_event passes model=self.model into every callback event (trainer_callback.py:554-563), so FineTunedAnnRefreshCallback does fire (the earlier 6.0.1-based finding was wrong).
A. Confirmed bugs (HIGH)
1. hpo.py:252-253 _grid_folds swaps data[2]/data[4] (structured_features/country as row_bc/pos) → --split cv --grid crashes at folds.py:72 (ambiguous truth value). Fix: data[3]/data[5].
2. rerank.py:250,265-266 country[b] out-of-bounds (canonical endpoints ≥ len(df)) → crash after A/B prints; panel-7 CSV never written. Fix: pad country like train.py:788-811.
3. training.py:1346-1370+:1810-1823 gate slots not actually replaced are labeled ann_finetuned (ann_sources.get(pair_id, "ann_finetuned") default) → datapoint_usage/coverage lies; can spuriously raise RuntimeError; with strict check at 1764-1768. Fix: attribute only when replacement realized.
4. training.py:1729-1759 _write_datapoint_usage RuntimeError on any population hitting 0 presentations — brittle. Fix: missing status + warning.
B. Confirmed (MED) bugs + dead code
 5. masking.py:165 augment_hard_negatives zero call sites → mask_hard_negatives:true is a no-op; mask_hard_negative_visibility.csv never written; n_masked_hard_negatives always 0.
 6. training.py:4027 _optuna_mlflow_cb never registered (only _optuna_tracking_cb).
 7. Initial-fold ANN + attribute-conflict negatives impossible (emb0 always empty, train.py:709,722-744) — desgined but undocumented; telemetry shows not_reached/unavailable.
 8. Triplet lane dead-by-construction (training.py:2397-2405 → "no triples built") — --loss triplet can never run.
 9. HPO lanes tune a gate-only dataset (use_hp=False, no refresh flags).
10. Dead: text.py:263 attributes_keys+REs 284-298; common.py:540 has_barcode; manifest.py:419 verify_manifests(+manifest_stages schema); gtin.py 12 len; NER island (ner.py/colab_ner.py/config_loader.py, +N-2 config-contract crash, N-3 success-on-dead-worker, N-4 HF token never forwarded); colab dead downloader cluster (_download_remote_manifests, _verify_manifest_downloads, download_results, download_checkpoints); hpo_persistence.write_pointer_registry/aggregate_pointer_registries; hpo_fencing.TrialLeaseStore/hpo_champions.ChampionStore (PG machinery inert); pipeline.py:77 STOPWORDS, :584/699 brand_tokens, duplicated regexes :913/:1026; _write_checkpoint_manifest trainer_control param.
C. Data drops / transparency (see agent inventory)
- H1 gate-vs-ann_finetuned misattribution (=A3); H2 persisted text dumps ≠ ingested tensor (1841-1851 fresh masks vs pre-mask dump); H3 mined-capacity truncation uncounted (1254-1310); H4 attr-conflict 0 at start unexplained.
- M1 manifest._csv_rows newline/comma-count drift (quoted fields); M2 pipeline/gate has no closure manifest (n_neg_dropped single number swallows reasons, 1663-1690); M3 ANN keep-mask + max_per_canonical/brand cap drops uncounted; M4 volume_verified unresolved/C-dup drops silent (51-59); M5 train_frac drop not logged; M6 payload_pairs.csv run-tag-less last-wins + non-atomic write_visibility_log.
- L1-L5 low items; predict_items SKU_ID dtype loss + NaN→"nan" collision with UNMATCHED_nan; canonical_records.csv read twice with differing NaN semantics.
D. Silent errors / unexpected behavior
mlflow/wandb drop non-finite silently; fuse_torch/fuse_numpy missing shape+NaN guards; attribute_conflicts.sku_attribute_info sentinel {0.0}/{1} → phantom conflicts (45-48); blocking.eval_blocking NaN-key recall understated; common.py:452 env truthiness trap; colab.py:1044-1050 blanket except swallows mask-effect failure; :479-483 re-wraps Ctrl-C; B-1 HPO env divergence (no RUN_ID/WANDB_DIR/PROFILE); B-2 checkpoint-download skip silently kills mask-effect; B-3 false "persisted" for sims/stop; B-4 sample runs lack latest_results.json; generate_training_report._json_list [] on corruption + :428 0.55 fallback; composition_plot ignores masking.enabled; pipeline.py:1339/1350 all-NaN brand AttributeError; cli per-file/per-poll subprocess + byte-wise read(1) + pointer re-mirror.
E. Hardcoded → config
- HIGH: dpi=150 at training.py:3114,3706/train.py:1061/colab.py:1033 (use plot_dpi()).
- MED: predict_items batch_size=128(vs 64), default=0.55, random_state=42, top_k=1; train.py:665 quarters 4; hard_negatives.py:27-28 default band; :429 re-validated set; colab.py:1725 model registry literal; :649/1502 profile vocab; training.py:343 *4; structured_features.py:96 *5 block dim (no config knob); RNG offsets 2069/2173/2330/2388/3221; :3393 recall 0.90; :2804/2890 population vocab; evaluate_models:362 cap 3 + literal PNG names; generate_training_report:414 ks.
- LOW: colab stage-timeout literals (60s-24h), pipeline n-gram caps + vocab wordlists → config/vocabulary.json.
F. Training-loop optimizations (by wall-time impact)
1. Dev evaluator double-encodes dev every eval + eval_ds loss eval → ~4 encodes/eval (~15-25% wall). Reuse one encode.
2. Tow-ers run as two separate batches (separate_first=True) — merge 2N (validate).
3. ANN refresh encodes unmined masked-copy block (~37k), re-reads canonical_records.csv/config per epoch, blocks main thread — encode payload[:len(df)+n_canon], cache, offload.
4. num_workers=0, per-sample set_transform churn → pure transform + workers.
5. Post-train tail re-encodes dev/train already encoded — reuse last refresh embeddings.
6. Per-batch .tolist() telemetry in live loss graph → vectorize, gate on flag.
7. load_config() uncached on hot paths → lru_cache.
8. hard_negatives.py:354-406 python append/cap loops → vectorize; :44.. per-call CSV re-read; blocking.py Counter.
9. cli: single tar download, long-lived stream, chunked reads.
G. Coverage gaps
manifest_stages unwired; predict_items no column validation/entry checks; hpo_grid.csv overwritten per run; rerank CV fold-0 only; NER island.

# ═══════════════════════════════════════════════════════════════════════════
# 2026-09-13 — rand_matching.py audit + prior-findings re-verification
# ═══════════════════════════════════════════════════════════════════════════
STANDING STANCE (owner, this session): FAIL LOUDLY. Every degradation that
today is a SILENT fallback/skip/coercion must raise or persist an explicit
warning before a run may claim success. The "Fail-loud mapping" under each
item is the intended behavior change.

NOTE: the file was edited mid-session (rev mtime 09-13 08:46, 44,645 B); the
original audit read an earlier revision. Items marked "already fixed" were
confirmed gone in the 09-13 revision and cross-checked by a verification
agent; the rest are open against the CURRENT checkout.

## rand_matching.py — open items (re-read 2026-09-13)

| ID | Issue | Location |
|----|-------|----------|
| RM1 | `_threshold_at_recall` silently returns `np.min(scores)` when no threshold reaches `target_recall`; the run then reports precision_at_recall as if valid | rand_matching.py:699-715 (fallback at 715) |
| RM2 | `_fold_sensitivity` silently `continue`s on NaN alternative thresholds (fold with no positives) → threshold_comparison rows silently vanish | rand_matching.py:848-851 |
| RM3 | `_fit_fold_threshold` ties break toward the LARGEST threshold → a degenerate all-UNMATCHED plateau (Rand=1.0 from singletons) can win with no warning | rand_matching.py:820 |
| RM4 | `_candidate_labels` inner-merges candidates×truth → SKUs whose true item is unretrievable (not in top_k AND no trusted gtin) are invisible to recall/Youden fitting and never counted as recall failures. Population context: 34,636/61,529 deduped SKUs (~56%) are gtin-blind | rand_matching.py:718-732 |
| RM5 | `choose_assignments` re-annotates + re-sorts the WHOLE candidate frame per threshold (~19 thresholds × folds × 2 sweeps) though only `score_pass`/`accepted` depend on the threshold | rand_matching.py:813, 852, 863-864 |
| RM6 | Same file, divergent NaN keying: `load_canonical_map` reads canonical_records.csv with default NaN semantics ("nan" key) vs `record_map` keep_default_na=False ("" key). Dormant (0 blank gtin rows today) but a latent corpus/pollution mismatch | pipeline.py:920 vs rand_matching.py:122-138 |
| RM7 | `_normalise_skus` accepts blank SKU_ID — `astype(str)` has no blank guard, only duplicates raise | rand_matching.py:234-244 |
| RM8 | `UNMATCHED_` prefix hardcoded in 4 places; `fit_threshold_at_90pct_recall` column name hardcoded while the LOOKUP is a target_recall f-string → mislabeled column if target_recall ≠ 0.90 | rand_matching.py:485, 614, 982, 1175; :929-931 |
| RM9 | `GATE_COLUMNS` constant is never consumed | rand_matching.py:91-96 |

Fail-loud mapping (owner stance):
- RM1: raise/flag instead of `np.min`; persist the observed recall at the chosen threshold.
- RM2: write explicit n=0/unavailable rows instead of `continue`; raise when a fold fails to fit.
- RM3: detect all-UNMATCHED-dominated plateaus and surface them before selecting.
- RM4: persist per-fold counts of unretrievable/gtin-blind SKUs into the diagnostics.
- RM6: align `load_canonical_map` to `keep_default_na=False` + add a blank-gtin guard.
- RM7: raise on blank SKU_ID (match `_load_labeled_input`).
- RM8: make the unmatched prefix config-driven and rename the fit column via the f-string.
- RM9: use the constant in `_annotate_candidates` or delete it.
- RM5: performance — compute the sort/tie-break once, filter per threshold.

## Already fixed in the 09-13 revision (original audit, now closed)
- barcode/gtin asymmetry GONE — all four GTIN/field reads now use
  `row_metadata_text(row, "barcode", "gtin")` (alias used only when the column
  is ABSENT, NaN→"") symmetrically: rand_matching.py:282, 361, 780, 1071,
  common.py:62-68.
- `record_map.get(candidate_gtin, {})` empty-dict fallback GONE — direct index
  + duplicate-gtin raise (127-128) + missing-record raise (133-139) in
  `__init__`; `_candidate_row` indexes `self.record_map[candidate_gtin]` (284).
- Double attribute parse GONE — `score_candidates` calls `_text_and_info`
  once and threads `texts` into `_encode_skus` (351-352); no re-parse inside
  (246-262).

## Prior FINDINGS re-verification (agent round, 2026-09-13)
RESOLVED since last log: N1 (wandb skip path, colab.py:1133-1228), N2
(uniformity no-raise + insufficient_pairs status, uniformity.py:62-127), N4 +
D3 (pack/volume sentinels gone, attribute_conflicts.py:49-75), N5 (per-pointer
dvc pull), A2 (rerank country padding + panel-7 CSV, rerank.py:250-316), B10
gtin 12-len branch live (gtin.py:59-60), B10 brand_tokens, H4 (attr-conflict
0 explained and loud, train.py:709-745), E colab model registry (resolve_model),
E generate_training_report 0.55 fallback now raises.

PARTIAL: N6 (masked-tail encode persists; canonical_records re-read gone), B5
(dynamic masking fixed the no-op; n_masked_hard_negatives still 0), B9 (TPE no
longer gate-only; no refresh flags anywhere), M2 (negative_resolution_manifest
loud, no closure manifest/stage), M4 (cross-country/disagreement loud; 51-59
unresolvable-SKU drop still silent), D2 (fuse_numpy length guard only — no
NaN/shape; fuse_torch none), D6 (mask-effect traceback now printed), D9
(checkpoint skips now printed), D10 (sims "persisted" still unconditional;
stop early-returns), D12 (_json_list [] on corruption still unlogged; 0.55
raises), E dpi (plot_dpi exists, 5 literals remain: training.py:3401,4014,
train.py:1065, colab.py:1077, uniformity.py:190), E hard_negatives band
defaults, E report ks, E colab timeouts, E pipeline ngram caps, F9 (HPO
single-tar only).

STILL OPEN — HIGH: A1 (_grid_folds data[2]/data[4] swap → --split cv --grid
crash, hpo.py:252-253); A4 (datapoint_usage RuntimeError on 0 presentations,
training.py:2001-2031); A3/H1/N3 (gate slots > replacement mislabeled
ann_finetuned, training.py:2082-2086).
STILL OPEN — C/transparency: H2 (dumps ≠ tensor), H3 (capacity truncation
uncounted), M1 (manifest quote-drift), M3 (cap/keep drops uncounted), M5
(train_frac drop unlogged), M6 (payload_pairs last-wins + non-atomic
write_visibility_log), L1 predict_items SKU_ID dtype, L2 UNMATCHED_nan
collision, L3 canonical_records NaN-semantics divergence (= RM6).
STILL OPEN — D: mlflow/wandb non-finite silent drop; blocking.eval_blocking
NaN-key; common.py:478 env truthiness trap; colab Ctrl-C re-wrap:479-483; HPO
env divergence (no RUN_ID/WANDB_DIR/PROFILE, colab.py:1726-1732); sample runs
lack latest_results.json; composition_plot ignores masking.enabled:31;
pipeline.py:1364 all-NaN brand AttributeError; cli read(1)/per-file subprocess.
STILL OPEN — E: predict_items 128/0.55/42/1 literals; train.py:669 quarters 4;
colab.py:649/1585 profile vocab; training.py:347 *4; training.py:3694 recall
0.90 (NOT wired to rand_matching.target_recall); KNOWN_DATAPOINT_POPULATIONS
(training.py:101-110); structured_features.py:100 *5; RNG offsets; evaluate_models
cap 3 + PNG names.
STILL OPEN — F: F1-F8 all confirmed intact (dev re-encode, separate towers,
ANN encode of masked tail, num_workers=0, post-train tail re-encode, .tolist()
telemetry, uncached load_config, hard_negatives loops).
STILL OPEN — G: manifest_stages unwired; predict_items no column validation;
hpo_grid.csv overwritten per run; rerank fold-0 only; NER training/colab island
(ner_product_attributes IS live via pipeline.py:48).
STILL OPEN — B: B6 (_optuna_mlflow_cb unregistered), B7 (initial-fold deferred
— now documented), B8 (triplet lane dead), B10 dead-code cluster (text.py
attrs+REs, common.has_barcode, manifest.verify_manifests, colab downloader
cluster, hpo_persistence registry fns, TrialLeaseStore/ChampionStore, pipeline
STOPWORDS, _CANON_KEEP_DIGIT≡_NUTRIENT_RE duplicate, _write_checkpoint_manifest
trainer_control param).
STILL OPEN — N7 (uniformity O(n²) + dual CPU encode + dpi=150).

rand_matching RM1-RM9 above: all confirmed open in the CURRENT checkout.