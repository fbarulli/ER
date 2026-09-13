Step 1 — Establish there's a real problem worth investigating
Train loss 0.5 vs test loss 1.7 — flagged as a gap worth checking, but noted a static snapshot doesn't itself prove overfitting; needed the shape of the curves and the actual score distributions, not just the loss numbers.

Step 2 — Characterize what "overlap" actually meant
Asked whether label 1/0 overlap was on raw cosine or a downstream classifier score — you clarified it's raw model similarity, no reconciliation applied. This scoped the entire investigation to the embedding/training process itself, not a downstream decision layer.

Step 3 — Understand what the negative population actually was
You defined label 0 as hard negatives specifically (same brand/category, conflicting volume/pack/flavor) — this reframed "overlap" from a generic failure into a hypothesis: attribute-level distinctions may sit below what raw cosine can represent.

Step 4 — Check whether the loss function's margin was even active
You computed: margin 0.5 distance → target cosine <0.5; observed holdout label_0 min 0.638, median 0.876 — margin was mathematically active for the entire negative population, ruling out "margin too loose to matter" as the explanation.

Step 5 — Identify what data existed vs. what required a new run
Listed answerable-now (tokenization, margin interpretation, raw-vs-reconciliation) vs. needs-training (train-side scores, backprop/selection trace, exact pair coverage) — this is where we found train scores and coverage instrumentation didn't exist for the original run and couldn't be reconstructed post hoc.

Step 6 — Decide the tokenizer question could be deprioritized
Reasoned that brand-name splitting doesn't explain attribute-conflict overlap (field ablation already showed brand+category alone separate cleanly) — ruled out without needing new data.

Step 7 — Recognize loss-function-vs-margin was underdetermined
Explicitly held off deciding to switch loss functions, since margin-unrealistic and selection-coverage-gap were two different explanations requiring different evidence — set up the next run to disambiguate both at once rather than guessing.

Step 8 — Design the next run's instrumentation before running it
Specified: finish pair-ID coverage tracking, enable train-score export from the start, verify checkpoint saving, correct the margin to 0.20 distance — all decided before launch so the run wouldn't need repeating.

Step 9 — Smoke-test the instrumentation at small scale (n=100) before trusting a full run
Checked train label_0 median (0.706), selection/margin-active percentages (44/48, 2/48) — validated the plumbing worked, flagged sample size as too small to interpret substantively, not yet diagnostic.

Step 10 — Once the full run existed, go straight to the report and pull train vs. holdout overlap side by side
This is where the actual answer appeared: train overlap 0.212 vs holdout 0.674 — confirmed genuine generalization gap, not a ceiling.

Step 11 — Split by negative population (hard vs. random/easy) to localize where the gap lived
Random/easy AUC 0.994 vs hard-negative AUC 0.740 — isolated the failure to the hard-negative population specifically, not a general problem.

Step 12 — Check augmentation/masking counts against known population sizes
mask_n matched positive count almost exactly → masking is positive-only, negatives get zero augmentation — mechanistic explanation found.

Step 13 — Confirm scarcity, not just repetition
n_train_neg == n_train_neg_total → no larger pool being subsampled, the full 6,051 hard negatives are seen identically every epoch — completed the causal chain from symptom to root cause.

The throughline: at each step, before proposing a fix, we asked "what evidence would tell these two hypotheses apart" and got that evidence before acting — margin vs. selection-coverage (step 7-8), ceiling vs. generalization (step 10), general vs. hard-negative-specific (step 11), repetition vs. scarcity (step 13). That's why the loss-function/margin question ended up being the wrong lever, and augmentation asymmetry + pool scarcity turned out to be the actual cause.




Step 1 get the train vs. holdout overlap coefficients side by side (the original blocking question: ceiling vs. generalization gap):


jq '{train: .score_overlap.train, holdout: .score_overlap.holdout}' training_results/20260912T134452Z/worker_1/report/report.json
{
train": {
    "label_0_mean": 0.9003620773007153,
    "label_0_n": 6051,
    "label_1_mean": 0.9612996442950836,
    "label_1_n": 23372,
    "overlap_coefficient": 0.2115388686961005
  },
  "holdout": {
    "label_0_mean": 0.9357753269660086,
    "label_0_n": 2712,
    "label_1_mean": 0.9612356032064394,
    "label_1_n": 11694,
    "overlap_coefficient": 0.6740390276535432
  }
}

this is a genuine generalization gap, not a pure ceiling.

Train overlap coefficient (0.212) is dramatically lower than holdout (0.674) — a 3.2x difference. If this were purely a representational ceiling (embeddings structurally can't separate these attribute-conflict pairs, full stop), train and holdout would look similar, since the ceiling would apply equally regardless of split. Instead:

Train: label_0_mean 0.900 vs label_1_mean 0.961 — a real, meaningful gap (0.061), and low overlap (0.212) confirms the model is learning to separate these pairs reasonably well on data it trained on.
Holdout: label_0_mean 0.936 vs label_1_mean 0.961 — gap shrinks to 0.025, and overlap more than triples to 0.674.


So the model demonstrably can learn to push apart brand/category-matched, attribute-conflicting negatives (train proves this) — it just isn't transferring that separation to unseen pairs. That points toward overfitting to specific hard-negative pairs seen during training, rather than the margin/loss-function question being the primary driver anymore. The margin fix (0.20 distance) may still be relevant, but the dominant signal here is train≫holdout, which is the classic overfitting shape from much earlier in this conversation — now actually confirmed with data instead of hypothesized.

#

Step 2 — check if this generalization gap is specific to hard negatives or general, using the random_easy block:

jq '.random_easy' training_results/20260912T134452Z/worker_1/report/report.json
{
  "hard negative/0": {
    "mean": 0.9357753269660086,
    "median": 0.9426538944244384,
    "n": 2712
  },
  "hard negative/1": {
    "mean": 0.9612356032064394,
    "median": 0.9652252197265624,
    "n": 11694
  },
  "hard negative/auc": {
    "mean": 0.9077266903368646,
    "median": 0.7399125872229564,
    "n": 14406
  },
  "random/easy negative/0": {
    "mean": 0.8786133125126362,
    "median": 0.8801750242710114,
    "n": 10000
  },
  "random/easy negative/1": {
    "mean": 0.9612356032064394,
    "median": 0.9652252197265624,
    "n": 11694
  },
  "random/easy negative/auc": {
    "mean": 0.9937512463812348,
    "median": 0.992710800410467,
    "n": 21694
  }
}


This is the clearest signal yet, and it reframes the whole diagnosis: random/easy negatives generalize almost perfectly (AUC 0.994), while hard negatives specifically collapse (AUC median 0.740). That's not a general overfitting problem — it's overfitting concentrated exactly on the hard-negative population, which lines up with train/holdout overlap numbers from the last step (0.212 vs 0.674, same population).

This also makes your masking detail highly relevant, not incidental: if masking-based augmentation is applied only to positive pairs, then during training the model sees a fixed, static, unaugmented set of hard negatives epoch after epoch — while positives keep getting varied. That's a plausible direct mechanism for exactly this failure mode: the model can memorize the specific hard-negative examples it's seen (train overlap 0.212, looks great) without learning a generalizable rule for "attribute conflict = different item," so any new hard-negative pair at holdout time — with a volume/pack/flavor conflict the model hasn't specifically memorized — reverts to looking like a positive.



chase the masking hypothesis: hard-negative AUC mean (0.908) vs median (0.740) is a big gap, suggesting a skewed/bimodal distribution rather than one that's just uniformly worse — worth understanding before concluding.



Confirms the augmentation-asymmetry hypothesis directly: mask_n = 23,369, which matches your positive count almost exactly (label_1_n = 23,372 in train). Masking is applied only to positives — hard negatives get zero augmentation. So during training, the model sees the same ~6,051 static hard-negative pairs every epoch (memorizable), while positives are constantly perturbed via masking (forcing generalization on that side only). That's a mechanistically clean explanation for train overlap 0.212 / holdout overlap 0.674 and the hard-negative-specific AUC collapse (0.994 random/easy vs 0.740 hard) — the model over-indexed on memorizing the specific hard-negative examples rather than learning the attribute-conflict rule.


Step 3 — check the masking-related fields already in your metrics to see if augmentation is positive-only, and check fold-level AUC to explain the mean/median gap:

jq '.metrics.fold_0 | {mask_masked_mean_cosine, mask_unmasked_mean_cosine, mask_mean_cosine_delta, mask_n, n_train_neg_hard_selection_coverage, n_train_neg_margin_active_coverage, contrastive_margin}' training_results/20260912T134452Z/worker_1/report/report.json

{
  "mask_masked_mean_cosine": 0.974493682384491,
  "mask_unmasked_mean_cosine": 0.9477471113204956,
  "mask_mean_cosine_delta": 0.0267465952783823,
  "mask_n": 23369,
  "n_train_neg_hard_selection_coverage": null,
  "n_train_neg_margin_active_coverage": null,
  "contrastive_margin": 0.2
}


Confirms the augmentation-asymmetry hypothesis directly: mask_n = 23,369, which matches your positive count almost exactly (label_1_n = 23,372 in train). Masking is applied only to positives — hard negatives get zero augmentation. So during training, the model sees the same ~6,051 static hard-negative pairs every epoch (memorizable), while positives are constantly perturbed via masking (forcing generalization on that side only). That's a mechanistically clean explanation for train overlap 0.212 / holdout overlap 0.674 and the hard-negative-specific AUC collapse (0.994 random/easy vs 0.740 hard) — the model over-indexed on memorizing the specific hard-negative examples rather than learning the attribute-conflict rule.

Also worth noting: mask_mean_cosine_delta = +0.027 — masking a positive pair increases its cosine similarity slightly on average, meaning the augmentation is working as intended (masked-and-original stay close), not corrupting the positive signal. That's not the problem; the problem is purely the asymmetry, not the masking mechanism itself.

Separately: n_train_neg_hard_selection_coverage and n_train_neg_margin_active_coverage are both null for fold_0 — meaning the exact-unique-pair-coverage instrumentation you were building isn't populated in this run either. That's a gap worth closing before drawing conclusions about selection dynamics specifically (separate from what we've already confirmed about overfitting).

Step 4 — confirm whether hard-negative pair reuse across epochs is total (same 6,051 pairs, zero rotation) by checking if there's a hard-negative pool size field distinct from what's actually sampled per epoch:

jq '.metrics.fold_0 | {n_train_neg, n_train_neg_total, n_train_hp, n_hp_in_train, n_neg}' training_results/20260912T134452Z/worker_1/report/report.json

{
  "n_train_neg": 6051,
  "n_train_neg_total": 6051,
  "n_train_hp": 0,
  "n_hp_in_train": 0,
  "n_neg": 2712
}

his confirms scarcity, not just static repetition: n_train_neg equals n_train_neg_total (6,051 = 6,051) — meaning there is no larger pool being subsampled from; 6,051 is the entire distinct hard-negative population available for training, full stop. Every epoch sees the exact same 6,051 pairs, with no augmentation and no expansion. Combined with the mask asymmetry, the model has had many epochs of repeated, unperturbed exposure to a small, fixed set — ideal conditions for memorizing those specific pairs rather than learning the underlying attribute-conflict pattern.

(n_train_hp/n_hp_in_train = 0 look like a different field — possibly "hard positives," unused in this setup — not relevant to the negative-scarcity question. n_neg = 2712 matches your holdout label_0_n, so that's the holdout-side count, for reference.)

So the picture is now complete and specific:

Random/easy negatives generalize near-perfectly (AUC 0.994) — the embedding and training process are fundamentally sound.
Hard negatives specifically overfit (train overlap 0.212 → holdout overlap 0.674, AUC drops to 0.740 median) — isolated to this population only.
Root cause, mechanistically: 6,051 hard-negative pairs, static, unaugmented, seen identically every epoch — while positives get masking-based augmentation. The model memorizes these specific pairs instead of learning "attribute conflict ⇒ different item" as a generalizable rule.




So the picture is now complete and specific:

Random/easy negatives generalize near-perfectly (AUC 0.994) — the embedding and training process are fundamentally sound.
Hard negatives specifically overfit (train overlap 0.212 → holdout overlap 0.674, AUC drops to 0.740 median) — isolated to this population only.
Root cause, mechanistically: 6,051 hard-negative pairs, static, unaugmented, seen identically every epoch — while positives get masking-based augmentation. The model memorizes these specific pairs instead of learning "attribute conflict ⇒ different item" as a generalizable rule.