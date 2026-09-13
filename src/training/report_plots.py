"""07_report: report plots — score distribution, PR curve, threshold sweep,
error breakdown by attribute, FN characterization + country slice, field-ablation
bars, and the data-scaling curve.

Reads the zero-shot bi-encoder embeddings (cached) for the first five; the
field-ablation and data-scaling plots read 07c_field_ablation.csv and
07d_data_scaling.csv when they exist.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact
from sklearn.metrics import average_precision_score, precision_recall_curve

from core.blocking import build_pairs
from core.common import (
    DATA_DIR,
    SEED,
    F,
    artifact,
    category_macros,
    embedding_model_keys,
    ensure_parent,
    load_config,
    load_dataset_deduped,
    pair_similarity,
    plot_dpi,
    rand_matching_cfg,
    resolve_model,
    recall_column_suffix,
    runtime,
    trace_artifact,
    training_cfg,
)
from core.nlp import encode_corpus
from core.text import extract_volume_ml

# 05-03/06-3: the 07c/07d recall-tied column name and its labels follow the
# config SSOT (rand_matching.target_recall) — the same derivation the producer
# (training.py) uses — so a retune cannot leave this consumer reading the old
# fixed-suffix column or printing a stale recall target.
_RECALL_TARGET = float(rand_matching_cfg()["target_recall"])
_RECALL_KEY = recall_column_suffix(_RECALL_TARGET)
_RECALL_LABEL = f"{_RECALL_TARGET:.0%}"
_RECALL_PREC_COL = f"precision_at_{_RECALL_KEY}_recall"
_RECALL_TP_COL = f"tp_at_{_RECALL_KEY}_recall"
_RECALL_FP_COL = f"fp_at_{_RECALL_KEY}_recall"
_RECALL_THR_COL = f"threshold_at_{_RECALL_KEY}_recall"

# MACRO_MAP moved to config (SSOT): config/paths.yaml category_macros —
# read via lib.common.category_macros(), never a module-level copy.
MACRO_MAP = category_macros()

_cfg = load_config()
# Only config-declared bi-encoder keys are valid inputs to encode_corpus. Model
# paths are resolved when the report actually runs so importing this module
# remains possible on a checkout whose Git-shipped model bundle is absent.
MODEL_KEYS = embedding_model_keys()
CACHE = str(DATA_DIR / "embeddings_cache")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="subset of config model keys (default: all; e.g. --models minilm_l6)",
    )
    args = ap.parse_args()
    selected_keys = (
        tuple(key for key in MODEL_KEYS if key in args.models)
        if args.models
        else MODEL_KEYS
    )
    unknown = sorted(set(args.models or ()) - set(MODEL_KEYS))
    if unknown:
        raise SystemExit(
            f"unknown --models entries: {unknown} (have {list(MODEL_KEYS)})"
        )
    models = {key: resolve_model(key) for key in selected_keys}
    print(f"report models: {list(models)}", flush=True)

    df = load_dataset_deduped()
    payload = (
        df["title"].fillna("")
        + " | "
        + df["brand"].fillna("")
        + " | "
        + df["category"].fillna("")
    ).tolist()
    # eval pair caps from the SSOT pairs block (config/training.yaml) — were
    # hardcoded 4 / 10_000 inline
    _pairs_cfg = _cfg["pairs"]
    pos, neg = build_pairs(
        df, SEED, int(_pairs_cfg["max_pos_per_group"]), int(_pairs_cfg["n_neg"])
    )

    shared = _shared_frames(df, pos)
    for mkey, mid in models.items():
        print(f"\n=== model: {mkey} ({mid}) ===", flush=True)
        emb, _ = encode_corpus(
            mid,
            payload,
            batch_size=runtime("batch_size_embed"),
            max_seq_length=runtime("max_seq_length"),
            cache_dir=CACHE,
        )
        _per_model_block(df, mkey, emb, pos, neg, shared)
    # ---- 5. field-ablation bars (if CSV exists) ----
    # AUDIT FIX (round 2 F05, round 3): 07-series CSV names read via the
    # F map (config/paths.yaml files:) — a rename through config now reaches
    # this consumer instead of silently desynchronizing it.
    fab = F["field_ablation"]
    if fab.exists():
        fdf = pd.read_csv(fab).set_index("variant")
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        fdf[["average_precision", _RECALL_PREC_COL]].plot(
            kind="bar", ax=ax, rot=0
        )
        ax.set_ylabel("score")
        ax.set_xlabel("payload variant")
        ax.set_title(
            f"Field ablation — AP and precision@{_RECALL_LABEL}recall per variant"
        )
        _plot = artifact("report_plot", {"kind": "field_ablation"})
        fig.savefig(_plot, dpi=plot_dpi())
        trace_artifact("report_plot", _plot)
        plt.close(fig)
        print(f"field ablation plot written (n={len(fdf)})", flush=True)
        print("field ablation TP/FP/threshold (audit):", flush=True)
        for variant, row in fdf.iterrows():
            print(
                f"  {variant:<22} TP@{_RECALL_LABEL}R={row[_RECALL_TP_COL]:.0f} "
                f"FP@{_RECALL_LABEL}R={row[_RECALL_FP_COL]:.0f} "
                f"thr={row[_RECALL_THR_COL]:.4f}",
                flush=True,
            )
    else:
        print("field ablation CSV not ready yet", flush=True)

    # ---- 6. data-scaling curve (if CSV exists) ----
    dsc = F["data_scaling"]  # SSOT name (audit round 2 F05)
    if dsc.exists():
        ddf = pd.read_csv(dsc).sort_values("n_triples")
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        ax.plot(
            ddf["n_triples"],
            ddf["average_precision"],
            marker="o",
            color="#4C72B0",
            label="AP",
        )
        ax.plot(
            ddf["n_triples"],
            ddf[_RECALL_PREC_COL],
            marker="s",
            color="#C44E52",
            label=f"precision@{_RECALL_LABEL}recall",
        )
        ax.set_xscale("log")
        ax.set_xlabel("n_triples (log)")
        ax.set_ylabel("score")
        ax.set_title("Data-scaling curve — hard-band performance vs train size")
        ax.legend()
        _plot = artifact("report_plot", {"kind": "data_scaling"})
        fig.savefig(_plot, dpi=plot_dpi())
        trace_artifact("report_plot", _plot)
        plt.close(fig)
        print(f"data-scaling plot written (n={len(ddf)})", flush=True)
        print("data-scaling TP/FP/threshold (audit):", flush=True)
        for _, row in ddf.iterrows():
            print(
                f"  n_triples={int(row['n_triples']):<6} TP@{_RECALL_LABEL}R={row[_RECALL_TP_COL]:.0f} "
                f"FP@{_RECALL_LABEL}R={row[_RECALL_FP_COL]:.0f} "
                f"thr={row[_RECALL_THR_COL]:.4f}",
                flush=True,
            )
    else:
        print("data-scaling CSV not ready yet", flush=True)

    # ---- 7. four-population score distribution (fine-tuned, if CSV exists) ----
    # 01h's "cross-country proxy" (silver, brand+category) is NOT the same as
    # 07b's "cross-country positives" (true same-barcode ground truth); keep the
    # two population names distinct so the artifacts are not confused.
    four = F["four_pop_scores"]  # SSOT name (audit round 2 F05)
    if four.exists():
        fdf = pd.read_csv(four)
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        bins = np.linspace(0, 1, 51)
        colors = {
            "in_country_pos": "#4C72B0",
            "cross_country_pos": "#55A868",
            "hard_neg": "#C44E52",
            "random_neg": "#BBBBBB",
        }
        for name, grp in fdf.groupby("population"):
            ax.hist(
                grp["cosine"],
                bins=bins,
                alpha=0.55,
                color=colors.get(name, "#999"),
                label=f"{name} (n={len(grp):,})",
            )
        # AUDIT FIX (round 2 F04, round 3): the operating threshold is
        # split.fixed_threshold (the SSOT) — the 0.55 was re-declared
        # inline, so a config change would leave the plot's line stale.
        ax.axvline(
            training_cfg().split.fixed_threshold,
            color="k",
            ls="--",
            lw=1,
            label=f"operating threshold {training_cfg().split.fixed_threshold}",
        )
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("pairs")
        ax.set_title("Fine-tuned four-population score distribution (07b)")
        ax.legend(fontsize=7)
        _plot = artifact("report_plot", {"kind": "four_pop_dist"})
        fig.savefig(_plot, dpi=plot_dpi())
        trace_artifact("report_plot", _plot)
        plt.close(fig)
        print("four-population distribution plot written", flush=True)
    else:
        print("four-population scores CSV not ready yet", flush=True)

    print("report plots done", flush=True)


def _shared_frames(df, pos):
    """attribute arrays the error/FN analysis needs for every model."""
    brand = df["brand"].fillna("").astype(str).to_numpy()
    cat = df["category"].fillna("").astype(str).to_numpy()
    macro = df["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "?")).to_numpy()
    vol = df["title"].fillna("").map(extract_volume_ml).map(lambda t: t[0]).to_numpy()
    title_len = df["title"].fillna("").astype(str).str.len().to_numpy()
    country = df["country"].fillna("").astype(str).to_numpy()
    bc_arr = df["barcode"].fillna("").astype(str).to_numpy()
    return brand, cat, macro, vol, title_len, country, bc_arr


def _per_model_block(df, mkey, emb, pos, neg, shared) -> None:
    tag = mkey  # file suffix per model
    brand, cat, macro, vol, title_len, country, bc_arr = shared
    # AUDIT FIX (round 2 F19, round 3): pair_similarity (lib.common, the
    # SSOT module) is the canonical row-pair cosine — was a duplicate
    # import of lib.nlp._cosine, the same function re-implemented.
    pos_s = pair_similarity(emb, pos)
    neg_s = pair_similarity(emb, neg)
    y = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))]
    scores = np.r_[pos_s, neg_s]
    thr = float(np.quantile(neg_s, 0.95))

    # ---- 1. score distribution ----
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bins = np.linspace(0, 1, 61)
    ax.hist(
        neg_s,
        bins=bins,
        alpha=0.6,
        color="#C44E52",
        label=f"negative (n={len(neg_s):,})",
    )
    ax.hist(
        pos_s,
        bins=bins,
        alpha=0.6,
        color="#4C72B0",
        label=f"positive (n={len(pos_s):,})",
    )
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"thr@5%FPR={thr:.3f}")
    ax.set_xlabel("cosine")
    ax.set_ylabel("pairs")
    ax.legend()
    ax.set_title(
        f"Score distribution — zero-shot {tag} "
        f"(n: pos={len(pos_s):,} / neg={len(neg_s):,})"
    )
    _plot = artifact("report_plot", {"kind": f"score_dist_{tag}"})
    fig.savefig(_plot, dpi=plot_dpi())
    trace_artifact("report_plot", _plot)
    plt.close(fig)

    # ---- 2. precision-recall curve ----
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.plot(recall, precision, color="#4C72B0", lw=2, label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color="#4C72B0")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Precision-Recall — {tag} (n: pos={len(pos_s):,} / neg={len(neg_s):,})"
    )
    ax.legend(loc="lower left")
    _plot = artifact("report_plot", {"kind": f"pr_curve_{tag}"})
    fig.savefig(_plot, dpi=plot_dpi())
    trace_artifact("report_plot", _plot)
    plt.close(fig)

    # ---- 3. threshold sweep: precision / recall / F1 ----
    ths = np.linspace(0, 1, 201)
    tp = np.array([(pos_s >= t).sum() for t in ths], dtype=float)
    fp = np.array([(neg_s >= t).sum() for t in ths], dtype=float)
    fn = len(pos_s) - tp
    # guard: at the top of the sweep tp==fp==0, so precision/F1 are 0/0 -> NaN.
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        r = tp / (tp + fn)
        f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(ths, p, color="#4C72B0", label="precision")
    ax.plot(ths, r, color="#55A868", label="recall")
    ax.plot(ths, f1, color="#C44E52", label="F1")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"thr@5%FPR={thr:.3f}")
    ax.set_xlabel("cosine threshold")
    ax.set_ylabel("rate")
    ax.set_title(f"Precision / recall / F1 vs threshold — {tag}")
    ax.legend()
    _plot = artifact("report_plot", {"kind": f"threshold_sweep_{tag}"})
    fig.savefig(_plot, dpi=plot_dpi())
    trace_artifact("report_plot", _plot)
    plt.close(fig)

    # ---- 4. error breakdown by attribute ----
    brand = df["brand"].fillna("").astype(str).to_numpy()
    cat = df["category"].fillna("").astype(str).to_numpy()
    macro = df["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "?")).to_numpy()
    vol = df["title"].fillna("").map(extract_volume_ml).map(lambda t: t[0]).to_numpy()
    title_len = df["title"].fillna("").astype(str).str.len().to_numpy()
    country = df["country"].fillna("").astype(str).to_numpy()

    def shares(pairs, mask):
        a, b = pairs[mask][:, 0], pairs[mask][:, 1]
        va, vb = vol[a], vol[b]
        both = ~pd.isna(va) & ~pd.isna(vb)
        return {
            "same brand": (brand[a] == brand[b]).mean(),
            "same category": (cat[a] == cat[b]).mean(),
            "same macro": (macro[a] == macro[b]).mean(),
            "same volume": (both & (va == vb)).sum() / max(both.sum(), 1),
        }

    fn = pos_s < thr
    fp = neg_s > thr
    data = {
        "FN (missed matches)": shares(pos, fn),
        "FP (false matches)": shares(neg, fp),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    edf = pd.DataFrame(data).T[
        ["same brand", "same category", "same macro", "same volume"]
    ]
    edf.plot(kind="bar", ax=ax, rot=0)
    # actual counts: the % bars need the n behind them (share * n errors)
    n_fn, n_fp = int(fn.sum()), int(fp.sum())
    for col_i, c in enumerate(edf.columns):
        for row_i, (lbl, r) in enumerate(edf.iterrows()):
            n_err = n_fn if lbl.startswith("FN") else n_fp
            n_val = round(r[c] * n_err)
            ax.text(
                col_i + (row_i - 0.5) * 0.38,
                r[c] + 0.01,
                f"{n_val:,}",
                ha="center",
                fontsize=7,
            )
    ax.set_ylabel("share of errors")
    ax.set_xlabel("")
    ax.set_title(
        f"Error breakdown by attribute — {tag} (n: FN={n_fn:,} / FP={n_fp:,})"
    )
    _plot = artifact("report_plot", {"kind": f"error_breakdown_{tag}"})
    fig.savefig(_plot, dpi=plot_dpi())
    trace_artifact("report_plot", _plot)
    plt.close(fig)

    # ---- 4b. FN characterization (missed true matches) + country slice ----
    # Same zero-shot pos/neg/emb as above: characterize WHAT the false negatives
    # look like, and slice the positive/FN side by in-country vs cross-country.
    def bucket_masks(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        return {
            "missing brand": (brand[a] == "") | (brand[b] == ""),
            "short title": (title_len[a] < 20) | (title_len[b] < 20),
            "cross-country": (country[a] != "")
            & (country[b] != "")
            & (country[a] != country[b]),
        }

    pos_buckets = bucket_masks(pos)
    n_fn = int(fn.sum())
    rows: list[dict] = []
    fn_shares: list[float] = []
    tp_shares: list[float] = []
    print("FN characterization (missed true matches, cosine < thr):", flush=True)
    for name, mask in pos_buckets.items():
        fn_mask = mask & fn
        fn_share = float(fn_mask.sum() / n_fn) if n_fn else float("nan")
        tp_share = float(mask.mean())
        fn_shares.append(fn_share)
        tp_shares.append(tp_share)
        rows.append(
            {
                "bucket/slice": f"FN: {name}",
                "n": int(fn_mask.sum()),
                "share": round(fn_share, 4),
                "median_cosine": round(float(np.median(pos_s[fn_mask])), 4)
                if fn_mask.any()
                else float("nan"),
            }
        )
        print(
            f"  FN  {name:<14} {fn_share:6.1%}  (TP baseline {tp_share:6.1%})",
            flush=True,
        )
    for name, mask in pos_buckets.items():
        rows.append(
            {
                "bucket/slice": f"TP: {name}",
                "n": int(mask.sum()),
                "share": round(float(mask.mean()), 4),
                "median_cosine": round(float(np.median(pos_s[mask])), 4)
                if mask.any()
                else float("nan"),
            }
        )

    # Country slice of the positive/FN side only: cross-barcode negatives are not
    # country-defined (a negative is two DIFFERENT products), so there is no
    # in-country vs cross-country split for the FP side — state that explicitly.
    a, b = pos[:, 0], pos[:, 1]
    ca, cb = country[a], country[b]
    both_country = (ca != "") & (cb != "")
    slices = {
        "pos: in-country": both_country & (ca == cb),
        "pos: cross-country": both_country & (ca != cb),
    }
    # Entity-level n: pairs inside one barcode group are correlated, so the
    # stable unit is the DISTINCT barcode (product), not the pair count. This is
    # the same few-hundred-group cross-country population as the earlier census
    # (01g/07e: 349 multi-country barcode groups in the raw export; 292 in the
    # deduped frame this step consumes).
    bc_arr = df["barcode"].fillna("").astype(str).to_numpy()

    def n_groups(mask):
        return int(np.unique(bc_arr[pos[mask][:, 0]]).size)

    def _exact_ci(k, n):
        """Clopper-Pearson 95% interval (exact binomial); NaN when n == 0."""
        if n:
            ci = binomtest(k, n).proportion_ci()
            return ci.low, ci.high
        return float("nan"), float("nan")

    bcs = df["barcode"].fillna("").astype(str)
    known = df[bcs.str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    n_cross_groups_full = int((multi.groupby("barcode")["country"].nunique() > 1).sum())
    print(
        "country slice (positive/FN side only; cross-barcode negatives carry no country label):",
        flush=True,
    )
    slice_fn_rates: dict[str, float] = {}
    entity: dict[str, dict] = {}
    for label, mask in slices.items():
        n = int(mask.sum())
        fn_rate = float((pos_s[mask] < thr).mean()) if n else float("nan")
        med = float(np.median(pos_s[mask])) if n else float("nan")
        g = n_groups(mask)
        fg = n_groups(mask & fn)
        erate = fg / g if g else float("nan")
        elo, ehi = _exact_ci(fg, g)
        slice_fn_rates[label] = fn_rate
        entity[label] = {
            "groups": g,
            "fn_groups": fg,
            "rate": erate,
            "lo": elo,
            "hi": ehi,
        }
        rows.append(
            {
                "bucket/slice": label,
                "n": n,
                "groups": g,
                "share": round(fn_rate, 4),
                "entity_fn_rate": round(erate, 6),
                "ci_low": round(elo, 6),
                "ci_high": round(ehi, 6),
                "median_cosine": round(med, 4),
            }
        )
        print(
            f"  {label:<18} n={n:<6} groups={g:<5} pair FN rate={fn_rate:6.1%}  median pos cosine={med:.4f}",
            flush=True,
        )
    print(
        f"FN rate: in-country {slice_fn_rates['pos: in-country']:.1%} vs "
        f"cross-country {slice_fn_rates['pos: cross-country']:.1%}",
        flush=True,
    )
    cc = entity["pos: cross-country"]
    ic = entity["pos: in-country"]
    cc_fn_pairs = int((slices["pos: cross-country"] & fn).sum())
    if cc["groups"] and ic["groups"]:
        _, fisher_p = fisher_exact(
            [
                [cc["fn_groups"], cc["groups"] - cc["fn_groups"]],
                [ic["fn_groups"], ic["groups"] - ic["fn_groups"]],
            ],
            alternative="two-sided",
        )
    else:
        fisher_p = float("nan")
    ratio = cc["rate"] / ic["rate"] if ic["rate"] else float("nan")
    print(
        f"  n: cross-country FN = {cc_fn_pairs} pairs on {cc['fn_groups']} barcode groups "
        f"(sampled {cc['groups']} / {n_cross_groups_full} deduped multi-country groups); "
        f"in-country FN = {ic['fn_groups']} groups of {ic['groups']:,}.",
        flush=True,
    )
    print(
        f"  entity-level FN rate (Clopper-Pearson 95% CI): cross-country {cc['rate']:.1%} "
        f"[{cc['lo']:.1%}–{cc['hi']:.1%}, {cc['fn_groups']}/{cc['groups']} groups] vs in-country "
        f"{ic['rate']:.2%} [{ic['lo']:.2%}–{ic['hi']:.2%}, {ic['fn_groups']}/{ic['groups']:,} groups] — "
        f"Fisher's exact p={fisher_p:.3f} (nominally higher), but the interval spans "
        f"{cc['lo']:.1%}–{cc['hi']:.1%} from n={cc['fn_groups']} FN groups, so the gap is fragile, not a precise "
        f"{ratio:.0f}x.",
        flush=True,
    )

    fndf = pd.DataFrame(rows)[
        [
            "bucket/slice",
            "n",
            "groups",
            "share",
            "entity_fn_rate",
            "ci_low",
            "ci_high",
            "median_cosine",
        ]
    ]
    fndf["groups"] = fndf["groups"].astype("Int64")
    _csv = ensure_parent(artifact("report_csv", {"kind": f"fn_analysis_{tag}"}))
    fndf.to_csv(_csv, index=False)
    trace_artifact("report_csv", _csv)

    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    x = np.arange(len(fn_shares))
    w = 0.38
    ax.bar(x - w / 2, fn_shares, w, color="#C44E52", label="FN (missed matches)")
    ax.bar(
        x + w / 2, tp_shares, w, color="#4C72B0", label="TP baseline (all positives)"
    )
    # actual counts behind the shares: FN n and TP n per bucket
    _pb = bucket_masks(pos)
    for _i, (_name, _mask) in enumerate(_pb.items()):
        _n_fn = int((_mask & fn).sum())
        _n_tp = int(_mask.sum())
        ax.text(_i - w / 2, fn_shares[_i] + 0.005, f"{_n_fn:,}", ha="center", fontsize=7)
        ax.text(_i + w / 2, tp_shares[_i] + 0.005, f"{_n_tp:,}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(pos_buckets), rotation=0)
    ax.set_ylabel("share")
    ax.set_xlabel("")
    ax.set_title(
        f"FN characterization vs positive baseline — {tag} "
        f"(n: FN={int(fn.sum()):,} / TP={len(pos):,})"
    )
    ax.legend()
    _plot = artifact("report_plot", {"kind": f"fn_breakdown_{tag}"})
    fig.savefig(_plot, dpi=plot_dpi())
    trace_artifact("report_plot", _plot)
    plt.close(fig)


if __name__ == "__main__":
    main()
