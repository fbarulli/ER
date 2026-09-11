"""selftest.py — pinned oracles for every layer the lane touched.

Run:  python src/euromonitor/training/selftest.py
Exit 0 = all oracles pass. Any FAIL prints the exact expectation and the
actual value. Known-good entries are PUBLISHED GTINs (GS1 worked examples,
Wikipedia EAN-13/8, UPC-A) — never "I computed this so it must be right":
each checksum entry was cross-checked against an INDEPENDENT spec-derived
implementation during development (left-indexed weight rule).

Covers:
  1. GS1 checksum        known-valid / known-invalid GTIN-8/12/13/14
  2. barcode_validity   empty/None/placeholder/garbage -> False (no trust)
  3. normalize_text     NaN guard, case, x-multiply sign, punctuation
  4. soft-stop strip    strip-words die, keep-words (variant signals) live
  5. number tokens      reference verdicts: strip vs keep_brand/nutrient
  6. eval pairs         build_pairs: invalid-GTIN groups excluded from pos
  7. mining             mine_hard_negatives: invalid barcodes not certified
  8. folds              component_folds: no barcode straddles a boundary
  9. _precision_at_recall  hand-computed 10-pair oracle + sklearn bound
  10. _append_csv       replace-by-key idempotency
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


import numpy as np
import pandas as pd

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def oracle_gtin() -> None:
    from euromonitor.core.gtin import barcode_validity, is_valid_gtin_checksum

    valid = {
        "4006381333931": "GS1 worked example (EAN-13)",
        "5901234123457": "Wikipedia EAN-13",
        "96385074": "EAN-8 (computed)",
        "12345670": "Wikipedia EAN-8",
        "012345678905": "Wikipedia UPC-A",
        "12345678901231": "GTIN-14 (computed)",
        "5012345678900": "EAN-13 (computed)",
    }
    invalid = {
        "4006381333930": "GS1 INVALID example (check digit off)",
        "5901234123458": "off-by-one check digit",
        "12345671": "EAN-8 bad check",
        "012345678904": "UPC-A bad check",
        "": "empty",
        "12345": "too short",
        "12345678901234x": "not all digits",
    }
    for g, why in valid.items():
        check(
            f"checksum VALID {g} ({why})",
            is_valid_gtin_checksum(g) is True,
            f"got {is_valid_gtin_checksum(g)}",
        )
    for g, why in invalid.items():
        check(
            f"checksum INVALID {g!r} ({why})",
            is_valid_gtin_checksum(g) is False,
            f"got {is_valid_gtin_checksum(g)}",
        )
    v = barcode_validity(
        pd.Series(["4006381333931", "4006381333930", "", None, "0000000000000"])
    )
    check(
        "barcode_validity vector",
        list(v) == [True, False, False, False, False],
        f"got {list(v)}",
    )
    check(
        "placeholder all-zeros rejected",
        bool(barcode_validity(pd.Series(["0000000000000"])).iloc[0]) is False,
    )


def oracle_cleaning() -> None:
    from euromonitor.pipeline import MODEL_PAYLOAD_SOFT_STOP, normalize_text, strip_schema_words

    check("NaN -> empty", normalize_text(float("nan")) == "")
    check("None -> empty", normalize_text(None) == "")
    check("lowercase", normalize_text("Cola 500ml") == "cola 500ml")
    check("multiply sign -> x", normalize_text("Juice ×6") == "juice x6")
    check(
        "punctuation stripped",
        normalize_text("Coca-Cola, 2L!") == "coca cola 2l",
    )
    s = strip_schema_words("cola type carbonization volume 500ml pack")
    check(
        "soft-stop strip kills schema+format words",
        "type" not in s and "carbonization" not in s and "volume" not in s,
        f"got {s!r}",
    )
    for keep in ("still", "sugar", "sweetener", "concentrate", "powder"):
        check(
            f"variant signal {keep!r} survives",
            keep in strip_schema_words(f"cola {keep} type"),
            f"got {strip_schema_words(f'cola {keep} type')!r}",
        )
    check(
        "concentrate NOT in soft-stop set (owner ruling)",
        "concentrate" not in MODEL_PAYLOAD_SOFT_STOP,
    )


def oracle_number_reference() -> None:
    from euromonitor.core.common import DATA_DIR, F

    ref = pd.read_csv(DATA_DIR / F["number_reference"], dtype={"token": str})
    v = ref.set_index("token")["verdict"]
    for tok, want in [
        ("b12", "keep_nutrient"),
        ("1724", "keep_brand"),
        ("473", "strip"),
        ("15", "strip"),
    ]:
        if tok in v.index:
            check(f"reference verdict {tok!r} == {want}", v[tok] == want, f"got {v.get(tok)}")
        else:
            check(f"reference verdict {tok!r} == {want}", False, "token missing")
    check("reference row count 1,745", len(ref) == 1745, f"got {len(ref)}")


def oracle_eval_pairs() -> None:
    """build_pairs must exclude checksum-invalid barcode groups from the
    positive population and never certify negatives on them."""
    from euromonitor.core.blocking import build_pairs

    # synthetic 8-row corpus: a valid multi-retailer group (1 pos pair), an
    # INVALID-checksum multi-retailer group (must yield NOTHING), and a
    # valid cross-barcode negative
    df = pd.DataFrame(
        {
            "barcode": [
                "4006381333931",  # valid, retailer A
                "4006381333931",  # valid, retailer B -> 1 positive
                "4006381333930",  # INVALID check digit
                "4006381333930",  # INVALID -> must not pair
                "5901234123457",  # valid
                "5012345678900",  # valid
            ],
            "retailer": ["A", "B", "A", "B", "A", "B"],
            "title": [
                "same product one",
                "same product variant uno",   # distinct text, same barcode
                "different title x",
                "different title y",
                "distinct product p",
                "distinct product q",
            ],
        }
    )
    pos, neg = build_pairs(df, seed=42, max_pos_per_group=10, n_neg=10)
    bad = {"4006381333930"}
    pos_uses_bad = any(
        df["barcode"].iloc[i] in bad or df["barcode"].iloc[j] in bad
        for i, j in pos
    )
    check("eval positives: invalid-GTIN group excluded", not pos_uses_bad and len(pos) == 1,
          f"pos={pos.tolist()}")
    neg_uses_bad = any(
        df["barcode"].iloc[i] in bad or df["barcode"].iloc[j] in bad
        for i, j in neg
    )
    check("eval negatives: invalid-GTIN rows excluded", not neg_uses_bad,
          f"neg={neg.tolist()}")


def oracle_mining() -> None:
    from euromonitor.core.hard_negatives import mine_hard_negatives

    df = pd.DataFrame(
        {
            "barcode": ["4006381333931", "4006381333930", "5901234123457"],
            "brand": ["brandx", "brandy", "brandz"],
            "category": ["beverages", "beverages", "beverages"],
            "title": ["alpha beta gamma", "alpha beta gamma", "alpha beta gamma"],
        }
    )
    emb = np.array([[1.0, 0.0], [0.9999, 0.01], [0.0, 1.0]])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    pairs, _cos = mine_hard_negatives(df, emb, seed=0, n_target=10)
    bad = {"4006381333930"}
    check(
        "mining never certifies an invalid barcode",
        all(df["barcode"].iloc[a] not in bad and df["barcode"].iloc[b] not in bad
            for a, b in pairs),
        f"pairs={pairs.tolist()}",
    )


def oracle_folds() -> None:
    from euromonitor.training.folds import component_folds

    # pos edges over ROW indices: (0,1),(1,2) link barcodes A-B-C into ONE
    # component; (3,4) links D-E into a second. AUDIT 2026-09-09: the old
    # checks were tautologies — `x in "ABC"` is a substring test that is
    # always true for single-char barcodes (so check 1 passed regardless),
    # and check 3's disjunction `("D" in f0) != ("E" in f0)` passed even
    # when D and E were split across folds, which is exactly the
    # component-split bug this oracle exists to catch.
    pos = np.array([[0, 1], [1, 2], [3, 4]])
    row_bc = np.array(["A", "B", "C", "D", "E"])
    folds = component_folds(pos, row_bc, k=2, seed=0)
    f0, f1 = set(folds[0]), set(folds[1])
    comp_abc = {"A", "B", "C"}
    comp_de = {"D", "E"}
    check(
        "component A,B,C never split across folds",
        comp_abc <= f0 or comp_abc <= f1,
        f"f0={f0} f1={f1}",
    )
    check(
        "component D,E never split across folds",
        comp_de <= f0 or comp_de <= f1,
        f"f0={f0} f1={f1}",
    )
    # every barcode is in exactly one fold (no leak, no drop)
    check(
        "every barcode in exactly one fold",
        f0 | f1 == {"A", "B", "C", "D", "E"} and not (f0 & f1),
        f"f0={f0} f1={f1}",
    )


def oracle_holdout_integrity() -> None:
    """REAL-data holdout discipline: the 50/25/25 component split must be
    pairwise disjoint AND every positive pair must live entirely inside
    one side (no pair, hence no product, straddles a boundary)."""
    import euromonitor.pipeline
    from euromonitor.core.common import SEED, load_dataset_deduped
    from euromonitor.training.folds import component_folds

    df = load_dataset_deduped()
    d = euromonitor.pipeline.build_training_data(df)
    pos, row_bc = d["pos"], d["row_bc"]

    quarters = component_folds(pos, row_bc, 4, SEED)
    train_bc, dev_bc, test_bc = quarters[0] | quarters[1], quarters[2], quarters[3]

    check(
        "train/dev/test barcode sets pairwise disjoint",
        not (train_bc & dev_bc) and not (train_bc & test_bc)
        and not (dev_bc & test_bc),
        f"leaks: tr∩dev={len(train_bc & dev_bc)} "
        f"tr∩test={len(train_bc & test_bc)} dev∩test={len(dev_bc & test_bc)}",
    )
    straddle = sum(
        1
        for a, b in pos
        if (row_bc[a] in train_bc) + (row_bc[b] in train_bc) == 1
        or (row_bc[a] in dev_bc) + (row_bc[b] in dev_bc) == 1
        or (row_bc[a] in test_bc) + (row_bc[b] in test_bc) == 1
    )
    check(
        "no positive pair straddles any boundary",
        straddle == 0,
        f"{straddle} straddling pairs",
    )
    # measured split sizes (25/25/25 by construction; train = q0+q1 = 50%)
    n_bc = len(set(row_bc.tolist()))
    check(
        "split sizes 50/25/25 (±2pp) over barcodes",
        abs(len(train_bc) / n_bc - 0.50) < 0.02
        and abs(len(dev_bc) / n_bc - 0.25) < 0.02
        and abs(len(test_bc) / n_bc - 0.25) < 0.02,
        f"got {len(train_bc)}/{len(dev_bc)}/{len(test_bc)} of {n_bc}",
    )
    print(
        f"    [info] real split: train {len(train_bc):,} / dev {len(dev_bc):,} "
        f"/ test {len(test_bc):,} barcodes; positives {len(pos):,}",
    )


def oracle_precision_at_recall() -> None:
    from sklearn.metrics import precision_recall_curve

    from euromonitor.training.training import _precision_at_recall

    y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
    s = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3])
    p, r, thr = _precision_at_recall(y, s, 0.75)
    check(
        "P@75R hand-computed (0.75, 0.75, 0.75)",
        abs(p - 0.75) < 1e-9 and abs(r - 0.75) < 1e-9 and abs(thr - 0.75) < 1e-9,
        f"got {(p, r, thr)}",
    )
    p, r, thr = _precision_at_recall(y, s, 1.0)
    check(
        "P@100R hand-computed (4/7, 1.0, 0.6)",
        abs(p - 4 / 7) < 1e-9 and r == 1.0 and abs(thr - 0.6) < 1e-9,
        f"got {(p, r, thr)}",
    )
    rng = np.random.default_rng(0)
    y2 = rng.integers(0, 2, 500)
    s2 = rng.random(500)
    prec, rec, _ = precision_recall_curve(y2, s2)
    mask = rec[:-1] >= 0.9
    sk = prec[:-1][mask].max() if mask.any() else float("nan")
    p2, _r2, _ = _precision_at_recall(y2, s2, 0.9)
    check(
        "P@90R <= sklearn max-at-recall (boundary point)",
        p2 <= sk + 1e-9,
        f"ours {p2:.4f} vs sklearn {sk:.4f}",
    )


def oracle_youden_discipline() -> None:
    """Holdout discipline: Youden threshold picked on DEV must be applied
    verbatim to TEST — hand-computable case where the dev-optimal and
    test-optimal thresholds DIFFER (a test-fitted threshold would flatter
    the accuracy)."""
    from euromonitor.training.training import _youden_thr

    # dev: pos at 0.9/0.6, neg at 0.55/0.2 -> dev Youden picks thr in
    # (0.55, 0.6] ... argmax over sorted-desc scores picks the SMALLEST
    # score with maximal J: candidates 0.9,0.6 (both TP, no FP yet) ->
    # J at 0.6 = 1.0 - 0.0 (thr 0.6: TP=2, FP=0) -> thr = 0.6
    dev_s = np.array([0.9, 0.6, 0.55, 0.2])
    dev_y = np.array([1, 1, 0, 0])
    thr = _youden_thr(dev_s, dev_y)
    check(
        "dev Youden hand-computed (0.6)",
        abs(thr - 0.6) < 1e-12,
        f"got {thr}",
    )
    # test: the SAME model scores pos at 0.7 (weaker) and negs at 0.65, 0.62
    # (in band). A test-fitted Youden picks 0.7 -> acc 1.0 (flattering);
    # the honest dev-picked 0.6 lets both negatives in -> acc 1/3. The
    # GAP between the two is exactly the leak the discipline removes.
    test_s = np.array([0.7, 0.65, 0.62])
    test_y = np.array([1, 0, 0])
    pred = (test_s >= thr).astype(int)
    acc_honest = float((pred == test_y).mean())
    thr_test_fit = _youden_thr(test_s, test_y)
    pred_fit = (test_s >= thr_test_fit).astype(int)
    acc_fitted = float((pred_fit == test_y).mean())
    check(
        "dev-picked thr applied verbatim (acc 1/3, not test-fitted 1.0)",
        abs(acc_honest - 1 / 3) < 1e-12 and abs(acc_fitted - 1.0) < 1e-12,
        f"honest {acc_honest:.3f} vs test-fitted {acc_fitted:.3f} "
        f"(thr {thr} vs {thr_test_fit})",
    )


def oracle_append_csv() -> None:
    from euromonitor.training.train import _append_csv

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.csv"
        _append_csv(p, [{"variant": "full", "ap": 0.9}], "variant")
        _append_csv(p, [{"variant": "title_only", "ap": 0.8}], "variant")
        _append_csv(p, [{"variant": "full", "ap": 0.95}], "variant")
        df = pd.read_csv(p)
        check(
            "_append_csv replace-by-key",
            len(df) == 2
            and df[df.variant == "full"].ap.iloc[0] == 0.95,
            f"got {df.to_dict('records')}",
        )


def oracle_phrase_variants() -> None:
    """Diet/pulp phrase variations — one concept, many retail phrasings, all
    mapping to the SAME canonical keep-token (owner ruling 2026-09-07).
    Hand-derived expectations, run through the REAL extractor."""
    from euromonitor.pipeline import NgramIDF, extract_discriminative_ngrams

    rows_by_gtin = {
        "1": [("Cola Classic 330ml", "type carbonated")],
        "2": [("Cola Zero Sugar 330ml", "type carbonated")],
        "3": [("Lemonade Sugar-Free 500ml", "type still")],
        "4": [("Pineapple Juice 1L", "type juice")],
    }
    gidf = NgramIDF(rows_by_gtin)

    def keeps(title: str) -> set[str]:
        from euromonitor.pipeline import PHRASE_VARIANTS

        sel = extract_discriminative_ngrams(
            [title], ["type carbonated"], ["brand"], gidf, None, top_k=5
        )
        # only the phrase-variation family (pulp/carbonated/zero are plain
        # unigram keep-tokens, not what this oracle pins)
        return {s for s in sel if s in PHRASE_VARIANTS}

    cases = {
        # sugar-free family -> sugar_free
        "Sugar Free Cola": "sugar_free",
        "Sugar-Free Cola": "sugar_free",
        "sugarfree Cola": "sugar_free",
        "sugarless Cola": "sugar_free",
        "free sugar Cola": "sugar_free",
        "free of sugar Cola": "sugar_free",
        "without sugar Cola": "sugar_free",
        "zero sugar Cola": "sugar_free+no_sugar",
        "no sugar Cola": "sugar_free+no_sugar",
        "no added sugar Cola": "sugar_free+no_sugar",
        "without added sugar Cola": "sugar_free+no_sugar",
        "with added sugar Cola": "added_sugar",  # positive claim, NOT sugar-free
        # pulp family (previously dead keep-tokens: with/no stopworded away)
        "Cola with pulp": "with_pulp",
        "cola no pulp": "no_pulp",
        "with extra pulp Cola": "with_pulp",
        "without pulp Cola": "no_pulp",
        # controls: no diet/pulp claim -> no diet/pulp token
        "Classic Cola": "",
        "Cola with extra pulp": "with_pulp",  # unigram 'pulp' alone is not a claim
        "Pulp free Cola 330ml": "no_pulp",  # pulp free = the NO-pulp claim
        "free of pulp Cola": "no_pulp",
    }
    for title, expect in cases.items():
        got = keeps(title)
        want = {t for t in expect.split("+") if t}
        check(
            f"phrase variant {title!r} -> {expect or 'no token'}",
            got == want,
            f"got {sorted(got)}",
        )


def oracle_word_once() -> None:
    """Token-once + concept-once discipline (owner directives 2026-09-08):
    every canonical carries each WORD at most once AND each CONCEPT once
    (sparkling==carbonated, minerals==mineral — SSOT stopwords.json
    CONCEPT_FOLDS). Repeated 'water' x5 was measured in 1,489 canonicals;
    sparkling+carbonated co-occurrence in 836."""
    import json as _json

    from euromonitor.core.common import RESULTS, F

    folds = dict(
        _json.loads(
            (Path(__file__).resolve().parents[1] / "core" / F["stopwords"]).read_text()
        )["CONCEPT_FOLDS"]
    )
    c = pd.read_csv(RESULTS / F["canonical_records"], keep_default_na=False)

    def words_of(txt: str) -> list[str]:
        return [
            folds.get(w, w)
            for t in txt.split()
            for w in t.split("_")
        ]

    bad = [
        txt
        for txt in c["canonical"].astype(str)
        if len(set(words_of(txt))) != len(words_of(txt))
    ]
    check(
        "canonicals have no repeated words or concepts (folded)",
        not bad,
        f"{len(bad)} violating canonicals, e.g. {bad[:2]}",
    )
    co = sum(
        1
        for txt in c["canonical"].astype(str)
        if "sparkling" in txt and "carbonated" in txt
    )
    check(
        "no canonical holds both sparkling and carbonated",
        co == 0,
        f"{co} co-occurrences",
    )
    empty = (c["canonical"].astype(str).str.strip() == "").sum()
    check(
        "no empty canonicals after dedup discipline",
        empty == 0,
        f"{empty} empty canonicals",
    )


def oracle_pinned_counts() -> None:
    """Pinned real-data counts — drift here means a pipeline change
    altered the committed-data contract (update alongside any
    intentional drift, e.g. GTIN enforcement)."""
    from euromonitor.core.common import RESULTS, F

    try:
        g = pd.read_csv(RESULTS / F["gate_results"], keep_default_na=False)
        c = pd.read_csv(RESULTS / F["canonical_records"], keep_default_na=False)
        lp = pd.read_csv(RESULTS / F["labeled_pairs"])
        check("canonicals == 13,250", len(c) == 13250, f"got {len(c)}")
        check("gate pairs == 135,769", len(g) == 135769, f"got {len(g)}")
        dec = g.gate_decision.value_counts().to_dict()
        check(
            "gate decisions hard_no=92,985 proceed=29,019 fallback=13,765",
            # AUDIT 2026-09-08: the ExtractedAttributes pack_qty >= 1
            # contract caught pack_set holding an impossible 0 ("pack 0.5
            # l" / "0% sugar" title forms) on 3 canonicals; the zero-guard
            # fixed 26 gate decisions. This is what current code+data
            # reproducibly yields.
            dec == {"hard_no": 92985, "proceed": 29019, "fallback": 13765},
            f"got {dec}",
        )
        check(
            "labeled pairs == 19,918 (7,330 pos / 12,588 hard-neg)",
            # same audit as above: +10 hard-negs vs the 2026-09-07 census
            # (pack_set [0] -> [1] lets those pairs compare packs honestly)
            len(lp) == 19918 and (lp.true_label == 1).sum() == 7330,
            f"got {len(lp)} rows, {(lp.true_label == 1).sum()} pos",
        )
    except FileNotFoundError as e:
        check("pinned counts (CSVs present)", False, str(e))


def oracle_config_split() -> None:
    """Split-SSOT (2026-09-08; EDA removed 2026-09-10): the monolithic
    00_config.yaml is now TWO files, each validated by its pydantic model
    at load. Pins the split layout + the merge discipline (the domain file
    overlays the root view) + the EDA-key migration (the five TRAIN-consumed
    EDA keys now live in src/euromonitor/training/training.yaml)."""
    from euromonitor.core.common import (
        CONFIG_PATH,
        TRAINING_CONFIG_PATH,
        data_cfg,
        load_config,
        resolve_model,
        training_cfg,
    )

    check("root config exists", CONFIG_PATH.exists(), str(CONFIG_PATH))
    check("src/euromonitor/training/training.yaml exists", TRAINING_CONFIG_PATH.exists())
    # EDA dir deleted 2026-09-10 (owner directive: training-only lane) —
    # pin that it stays deleted
    check("EDA dir removed", not (CONFIG_PATH.parent / "EDA").exists())

    # typed singletons load + validate
    check("DataConfig validates", data_cfg().seed == 42)
    check("TrainingConfig validates", training_cfg().training.loss == "contrastive")

    # merged view: domain knobs overlay the root data contract
    m = load_config()
    check(
        "merged view carries root + domain keys",
        all(k in m for k in ("paths", "files", "seed", "models", "training", "pairs", "bands", "mining", "masking", "split", "plots", "audit", "hpo", "rerank", "sweep")),
    )
    # stopwords moved to lib/ (pipe_stopwords.json + sklearn_stopwords.json)
    check(
        "files.stopwords points at the core package word list",
        data_cfg().files.stopwords == "pipe_stopwords.json",
    )
    from pathlib import Path as _P

    check(
        "core/pipe_stopwords.json exists",
        (CONFIG_PATH.parent / "src/euromonitor/core/pipe_stopwords.json").exists(),
    )
    check(
        "core/sklearn_stopwords.json exists",
        (CONFIG_PATH.parent / "src/euromonitor/core/sklearn_stopwords.json").exists(),
    )
    # model registry resolution: shared helper, no hardcoded hub ids
    check(
        "resolve_model('multilingual_l12') -> hub id",
        resolve_model("multilingual_l12").endswith("paraphrase-multilingual-MiniLM-L12-v2"),
        resolve_model("multilingual_l12"),
    )
    # split shares sum to 1 (pydantic-enforced; pin the values)
    sp = training_cfg().split
    check(
        "split 50/25/25 sums to 1",
        abs(sp.train_fraction + sp.dev_fraction + sp.test_fraction - 1.0) < 1e-9,
    )


def oracle_no_fallback_ssot() -> None:
    """NO-FALLBACK SSOT (audit round, owner Q27): every knob that used to
    be an inline literal in a script now lives in a config file and the
    accessor raises (not defaults) when it is missing. Pins the new
    config blocks + the derived-value discipline so a regression can't
    silently reintroduce a literal."""
    import subprocess
    import sys as _sys

    from euromonitor.core.common import (
        hpo_cfg,
        plot_dpi,
        rerank_cfg,
        runtime,
        sweep_cfg,
        training_cfg,
    )

    # ── hpo: block — the sweep spaces are config data, not module literals
    h = hpo_cfg()
    check(
        "hpo.grid is the 11-config second07 sweep",
        len(h["grid"]) == 11 and {r["epochs"] for r in h["grid"]} == {1, 2, 4},
        f"got {len(h['grid'])} rows",
    )
    check(
        "hpo.quick is the 3-config smoke subset",
        len(h["quick"]) == 3,
        f"got {len(h['quick'])}",
    )
    check(
        "hpo.tpe_space ranges ordered + numeric",
        all(lo < hi for lo, hi in h["tpe_space"].values()),
        str(h["tpe_space"]),
    )
    check("hpo.n_trials >= 1", h["n_trials"] >= 1)
    # src/euromonitor/training/hpo.py reads the SAME lists (no divergent module-level copy)
    import euromonitor.training.hpo as _hpo_mod

    check(
        "src/euromonitor/training/hpo.GRID == config hpo.grid",
        _hpo_mod.GRID == h["grid"] and _hpo_mod.QUICK == h["quick"],
    )
    # src/euromonitor/training/training.HPO_SPACE == config tpe_space
    import euromonitor.training.training as _tr_mod

    check(
        "src/euromonitor/training/training.HPO_SPACE == config hpo.tpe_space",
        _tr_mod.HPO_SPACE == {k: tuple(v) for k, v in h["tpe_space"].items()},
    )

    # ── rerank: the quantitative A/B decision rule
    r = rerank_cfg()
    check(
        "rerank decision margins in [0,1]",
        0.0 <= r["min_delta_pr_auc"] <= 1.0 and 0.0 <= r["min_delta_f1"] <= 1.0,
        str(r),
    )

    # ── sweep: run_all's ablation axes
    s = sweep_cfg()
    check(
        "sweep payload/frac axes non-empty, fracs in (0,1)",
        len(s["payload_variants"]) >= 1
        and all(0.0 < f < 1.0 for f in s["train_fracs"]),
        str(s),
    )
    check("sweep smoke/sweep sample >= 1", s["smoke_sample"] >= 1 and s["sweep_sample"] >= 1)
    check(
        "sweep.rerank_model names a cross-encoder",
        "cross-encoder" in s["rerank_model"],
        s["rerank_model"],
    )

    # ── new training knobs (formerly inline): layer_decay, save_total_limit,
    #    rerank_max_length — all present + in range
    t = training_cfg().training
    check("training.layer_decay in (0,1]", 0.0 < t.layer_decay <= 1.0)
    check("training.save_total_limit >= 1", t.save_total_limit >= 1)
    check("training.rerank_max_length >= 8", t.rerank_max_length >= 8)
    check(
        "runtime('layer_decay') == pydantic value",
        float(runtime("layer_decay")) == float(t.layer_decay),
    )

    # ── migrated EDA keys (2026-09-10): pairs caps + strip-audit sample +
    #    plot dpi now live in src/euromonitor/training/training.yaml
    t2 = training_cfg()
    check("pairs.max_pos_per_group >= 1", t2.pairs.max_pos_per_group >= 1)
    check("pairs.n_neg >= 0", t2.pairs.n_neg >= 0)
    check("pairs.neg_oversample >= 1", t2.pairs.neg_oversample >= 1)
    check("audit.strip_audit_sample >= 1", t2.audit.strip_audit_sample >= 1)

    # ── plot dpi SSOT: one accessor, used everywhere
    check("plot_dpi() == training plots.dpi", plot_dpi() == t2.plots.dpi)

    # ── NO-FALLBACK discipline: a MISSING key raises (never a default).
    # runtime() on a key absent from the YAML must KeyError.
    try:
        runtime("definitely_not_a_knob_q27")
        check("runtime(missing key) raises", False, "returned without error")
    except KeyError:
        check("runtime(missing key) raises", True)

    # ── literal-drift scan: the old inline literals must not reappear in
    #    the modules we moved them out of (regression pin for this audit)
    import re as _re

    import euromonitor.core.common as _lc

    root = _lc.TRAIN_ROOT

    def _code_only(text: str) -> str:
        """Strip comments and DOCSTRINGS so the drift scan matches
        EXECUTABLE literals — not the audit notes that document their
        removal, nor the docstrings that merely describe bands in prose
        (e.g. masking.py's "U(0.05,0.15)" historical note). Docstring
        removal is required by the widened band scan below: prose quotes
        band-shaped decimals that are NOT code.

        LINE-NUMBER PRESERVING: every removed line/docstring line is
        replaced by an EMPTY line of the same count, so the widened scan
        can report the drift's position in the REAL file.
        """
        import re as _dre

        def _blank_docstring(m: _dre.Match) -> str:
            return '""' + "\n" * m.group(0).count("\n")

        text = _dre.sub(r'("""|\'\'\')[\s\S]*?\1', _blank_docstring, text)
        out = []
        for ln in text.splitlines():
            s = ln.lstrip()
            if s.startswith("#"):
                out.append("")  # keep the line, drop the comment
                continue
            # trailing full-line comments (crude but sufficient: a '#'
            # outside a string literal after code — only strip when the
            # line has no quote before the '#')
            if "#" in ln:
                h = ln.index("#")
                if '"' not in ln[:h] and "'" not in ln[:h]:
                    ln = ln[:h]
            out.append(ln)
        return "\n".join(out)

    banned = {
        # AUDIT round 2 F06 (round 3): the grid lane's inline optimizer
        # knobs must not come back
        "src/euromonitor/training/hpo.py": [
            r"GRID\s*=\s*\[",
            r"QUICK\s*=\s*\[",
            r'"weight_decay":\s*0\.01',
            r'"lr_scheduler":\s*"linear"',
        ],
        # the literal DICT BODY is the ban target; the config-derived
        # comprehension ({k: (lo, hi) for ...}) is the sanctioned form
        "src/euromonitor/training/training.py": [r'HPO_SPACE\s*=\s*\{\s*\n\s*"epochs":'],
        "src/euromonitor/training/rerank.py": [r"max_length\s*=\s*512", r"d_pr\s*>\s*0\.005"],
        "src/euromonitor/training/train.py": [
            r'"warmup_ratio":\s*0\.05',
            r'"weight_decay":\s*0\.01',
            r"batch_size=128",
        ],
        "run_all.py": [r'"title_only",\s*\)', r'"0\.25",\s*"0\.50",\s*"0\.75"'],
        # AUDIT round 2 F03 (round 3): the two dpi=150 literals that
        # regressed in evaluate_models must not come back
        "src/euromonitor/training/evaluate_models.py": [r"dpi\s*=\s*150"],
    }
    for rel, pats in banned.items():
        text = _code_only((root / rel).read_text(encoding="utf-8"))
        for pat in pats:
            check(
                f"no inline literal {pat!r} in {rel}",
                _re.search(pat, text) is None,
            )

    # ── WIDENED band-drift scan (this round): the banned-literal pins above
    #    only guard the EXACT literals past audits removed. This scan is
    #    SHAPE-based — it catches NEW config-drift, not just old
    #    regressions: any inline numeric-literal band/threshold shape in
    #    the tracked src/euromonitor/training/*.py + lib/*.py + euromonitor.pipeline.py + run_all.py +
    #    colab_backend.py executable lines (docstrings/comments stripped).
    #
    #    Patterns (bands/thresholds ONLY — deliberately narrow):
    #      pair    — a (lo, hi) decimal pair inside () or []: the cosine/
    #                jaccard band shape, e.g. (0.35, 0.90) / [0.50, 0.75]
    #      declist — a [] list of 2+ bare decimals: the band-EDGE shape,
    #                e.g. [0.25, 0.50, 0.75]
    #      band3x  — a 0.3x decimal in comma/list context (the task's
    #                example drift shape: "0.31," cosine-band edges)
    #
    #    EXEMPTIONS (line-context, listed so nobody "fixes" a false
    #    positive by deleting the scan): plot cosmetics (set_ylim/set_xlim/
    #    figsize/fontsize/alpha/lw/linewidth/zorder/ha/va/color/ls/dpi and
    #    the axhline/axvline/hlines/vlines reference-line family), histogram
    #    grid constants (np.linspace / np.arange / bins=), quantile cuts
    #    (np.quantile), and float ARITHMETIC factors (a decimal directly
    #    after an operator — 29.5735 * x, (row_i - 0.5) * 0.38 — never a
    #    band). NOT numeric but structurally exempt: integer seeds/dpis/
    #    versions/batch sizes never match the decimal-only patterns by
    #    construction.
    _wide_files = sorted(
        str(p.relative_to(root))
        for p in [
            *(root / "src/euromonitor/training").glob("*.py"),
            *(root / "src/euromonitor/core").glob("*.py"),
            root / "src/euromonitor/pipeline.py",
            root / "run_all.py",
            root / "colab_backend.py",
        ]
        if p.name not in ("selftest.py", "__init__.py")
    )
    _wide_pats = {
        "pair": _re.compile(
            r"[\(\[]\s*[01]?\.\d+\s*,\s*[01]?\.\d+\s*[\)\]]"
        ),
        "declist": _re.compile(
            r"\[\s*[01]?\.\d+\s*(?:,\s*[01]?\.\d+\s*)+\]"
        ),
        "band3x": _re.compile(r"0\.3[0-9]?\s*[,\]]"),
    }
    _wide_exempt = _re.compile(
        r"set_ylim|set_xlim|ylim|xlim|figsize|fontsize|alpha|linewidth"
        r"|\blw\b|zorder|axhline|axvline|hlines|vlines|linspace|arange"
        r"|bins|quantile|dpi|color|marker|\bls\b|ha=|va=|\*|\+|/|-"
    )
    for rel in _wide_files:
        for _ln_no, _ln in enumerate(
            _code_only((root / rel).read_text(encoding="utf-8")).splitlines(), 1
        ):
            if _wide_exempt.search(_ln):
                continue  # plot cosmetics / grid constants / arithmetic
            for _tag, _p in _wide_pats.items():
                if _p.search(_ln):
                    print(
                        f"  [DRIFT] {_tag} shape at {rel}:{_ln_no}: "
                        f"{_ln.strip()[:100]}"
                    )
                    check(
                        f"widened scan: no inline band literal in {rel}",
                        False,
                        f"{_tag} shape at line {_ln_no}: {_ln.strip()[:100]}",
                    )
    check(
        "widened band-drift scan ran (files covered)",
        len(_wide_files) >= 29,
        f"{len(_wide_files)} files",
    )

    # ── deterministic-derivation pin: the mask extent midpoint is derived
    #    from the config band, not hardcoded
    msk = training_cfg().masking
    _mid = (float(msk.mask_lo) + float(msk.mask_hi)) / 2.0
    check(
        "mask extent midpoint derived from band",
        0.0 < _mid < 1.0,
        f"midpoint {_mid}",
    )
    # the fixed-threshold metric names follow split.fixed_threshold
    thr = training_cfg().split.fixed_threshold
    check(
        f"fixed-thr metric name key == {thr:g}",
        f"f1_at_{thr:g}".endswith(f"{thr:g}"),
    )

    # ── run_all consumes sweep_cfg (subprocess --help smoke, no full run)
    try:
        res = subprocess.run(
            [_sys.executable, "-c", "import run_all"],  # importable = SSOT loads
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        check("run_all imports (sweep_cfg live)", res.returncode == 0, res.stderr[-200:])
    except subprocess.TimeoutExpired:
        check("run_all imports (sweep_cfg live)", False, "timeout")


def oracle_round3_pins() -> None:
    """AUDIT ROUND 2 -> ROUND 3 fix pins (F01-F21 remediation): the gate
    thresholds match the config block; dead keys/symbols stay deleted; the
    SSOT-derived defaults (colab train-frac) stay derived; blocking-audit
    knobs read the audit: block. A regression here means a round-3 fix
    was silently reverted."""
    from euromonitor.pipeline import three_way_gate
    from euromonitor.core.common import F, sweep_cfg, training_cfg

    # ── F01 pin: the gate's decision table == the config values, and
    #    three_way_gate with no explicit args uses exactly them
    g = training_cfg().gate
    check(
        "gate thresholds live in config (0.05/0.85/0.3)",
        (g.vol_tolerance, g.raw_conf_threshold, g.consistency_fallback_threshold)
        == (0.05, 0.85, 0.3),
        f"got {g.vol_tolerance}/{g.raw_conf_threshold}/"
        f"{g.consistency_fallback_threshold}",
    )
    _attrs = {
        "volume_confidence": 0.9,
        "pack_confidence": 0.9,
        "volume_set": {500.0},
        "pack_set": {1},
        "volume_consistency": 1.0,
        "pack_consistency": 1.0,
        "mode_flavor": "",
    }
    _same = dict(_attrs, volume_set={505.0})  # inside ±5% tolerance
    _no_vol = dict(_attrs, volume_set={5000.0})  # outside tolerance
    check(
        "three_way_gate defaults == config gate (proceed case)",
        three_way_gate(_attrs, _same)["decision"] == "proceed",
        str(three_way_gate(_attrs, _same)),
    )
    check(
        "three_way_gate defaults == config gate (hard_no case)",
        three_way_gate(_attrs, _no_vol)["decision"] == "hard_no",
        str(three_way_gate(_attrs, _no_vol)),
    )
    _low_conf = dict(_attrs, volume_confidence=0.5)
    check(
        "three_way_gate defaults == config gate (fallback case)",
        three_way_gate(_attrs, _low_conf)["decision"] == "fallback",
        str(three_way_gate(_attrs, _low_conf)),
    )

    # ── F02 pin: the dead files.hpo_tpe_best key stays deleted
    check(
        "files.hpo_tpe_best ABSENT from config (dead key removed)",
        "hpo_tpe_best" not in F,
        str(sorted(k for k in F if "hpo" in k)),
    )

    # ── F21 pin: bands.mining_band stays deleted; only the live bands exist
    from euromonitor.core.common import _CFG  # the merged view

    check(
        "bands.mining_band ABSENT (dead key removed)",
        "mining_band" not in _CFG.get("bands", {}),
        str(sorted(_CFG.get("bands", {}))),
    )
    check(
        "bands keys == {eval_mining, rerank_band}",
        set(_CFG.get("bands", {})) == {"eval_mining", "rerank_band"},
    )

    # ── F07 pin: colab's train-frac default == sweep.train_fracs[0]
    import euromonitor.cli.colab as _cb

    check(
        "colab _TRAIN_FRAC_DEFAULT == sweep.train_fracs[0]",
        _cb._TRAIN_FRAC_DEFAULT == float(sweep_cfg()["train_fracs"][0]),
        f"got {_cb._TRAIN_FRAC_DEFAULT}",
    )

    # ── F18 pin: the deleted dead symbols stay deleted (import surface)
    import euromonitor.core.text as _lt

    for sym in (
        "get_measurement_type",
        "CATEGORY_MEASUREMENT_TYPE",
        "DEFAULT_MEASUREMENT_TYPE",
        "is_pack_multiple",
        "DRY_MIX_HINTS",
        "SUSPECT_ROUND",
        "FLAVOR_VOCAB",
        "FLAVOR_RE",
        "_LITER_UNITS",
        "MULTIPACK_RE",
    ):
        check(
            f"lib.text.{sym} ABSENT (dead symbol removed)",
            not hasattr(_lt, sym),
        )
    import euromonitor.core.common as _lc2

    check(
        "lib.common.multi_retailer_mask ABSENT (dead symbol removed)",
        not hasattr(_lc2, "multi_retailer_mask"),
    )

    # ── F18 pin: blocking_audit knobs read the audit: block, same values
    a = training_cfg().audit
    check(
        "blocking_audit knobs == audit: block values",
        (a.blocking_budget, a.blocking_min_recall) == (5_000_000, 0.95),
        f"got {a.blocking_budget}/{a.blocking_min_recall}",
    )
    import euromonitor.training.blocking_audit as _ba

    check(
        "src/euromonitor/training/blocking_audit.BUDGET == audit.blocking_budget",
        _ba.BUDGET == a.blocking_budget,
        f"got {_ba.BUDGET}",
    )
    check(
        "src/euromonitor/training/blocking_audit.MIN_RECALL == audit.blocking_min_recall",
        _ba.MIN_RECALL == a.blocking_min_recall,
        f"got {_ba.MIN_RECALL}",
    )

    # ── strip-audit ladder bands pin (SSOT move, this round): the band
    #    edges moved from an inline list in src/euromonitor/training/strip_audit.py into
    #    audit.strip_ladder_bands — SAME VALUES, new home, so the pin
    #    proves neither the values nor the move drifted
    from euromonitor.core.common import strip_ladder_bands

    check(
        "strip ladder bands == (0.0,0.2)...(0.8,1.01) from audit: block",
        strip_ladder_bands()
        == [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)],
        f"got {strip_ladder_bands()}",
    )
    check(
        "strip ladder bands contiguous ascending (schema-validated)",
        all(
            float(lo) < float(hi)
            for lo, hi in strip_ladder_bands()
        )
        and all(
            a.strip_ladder_bands[i].hi == a.strip_ladder_bands[i + 1].lo
            for i in range(len(a.strip_ladder_bands) - 1)
        ),
        f"got {a.strip_ladder_bands}",
    )

    # ── MACRO_MAP pin (SSOT move, this round): the category->macro
    #    taxonomy is DOMAIN DATA moved from lib/text.py into 00_config.yaml
    #    category_macros: — pin the exact 24-category shape + the module
    #    copy stays deleted
    from euromonitor.core.common import category_macros, load_dataset_deduped

    _mm = category_macros()
    check(
        "category_macros: 24 categories -> 6 macro buckets",
        len(_mm) == 24 and set(_mm.values()) == {
            "JUICE", "CARBONATES", "ENERGY_SPORTS", "WATER", "TEA_COFFEE",
            "CONCENTRATES",
        },
        f"got {len(_mm)} cats -> {sorted(set(_mm.values()))}",
    )
    check(
        "category_macros covers every strict category in the dataset",
        all(
            c in _mm
            for c in sorted(
                load_dataset_deduped()["category"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )
        ),
        "some dataset category has no macro bucket",
    )
    import euromonitor.core.text as _lt

    check(
        "lib.text.MACRO_MAP stays deleted (config is the SSOT)",
        not hasattr(_lt, "MACRO_MAP"),
    )

    # ── F19 pin: lib.nlp._cosine IS lib.common.pair_similarity (one impl)
    import euromonitor.core.common as _lc3
    import euromonitor.core.nlp as _ln

    check(
        "lib.nlp._cosine is lib.common.pair_similarity (one implementation)",
        _ln._cosine is _lc3.pair_similarity,
    )

    # ── F13 pin: the second04 manifest name is SSOT-side
    check(
        "files.second04_pairs_positive declared in the files map",
        F.get("second04_pairs_positive") == "second04_pairs_positive.csv",
        str(F.get("second04_pairs_positive")),
    )


def oracle_schemas() -> None:
    """Boundary contracts (lib.schemas): reject the exact shape breaks the
    audits found in this lane's history — out-of-range indices, unlocked
    payload/row_bc, out-of-bounds confidences, non-disjoint folds, invalid
    gate decisions/verdicts."""
    import numpy as np

    from euromonitor.core.schemas import (
        CanonicalRecord,
        DataTuple,
        ExtractedAttributes,
        FoldSets,
        GateResult,
        MaskingResult,
        TrainingData,
        check_verdict_map,
    )

    def _attrs(**over):
        base = {
            "flavor": "orange", "type": "water", "volume_ml": 500.0,
            "volume_confidence": 0.9, "volume_raw": "500ml",
            "volume_status": "metric_volume", "pack_qty": 1,
            "pack_confidence": 0.9,
        }
        base.update(over)
        return ExtractedAttributes(**base)

    check("ExtractedAttributes accepts valid row", _attrs().pack_qty == 1)
    for bad_over, why in (
        ({"volume_confidence": 1.7}, "volume_confidence > 1"),
        ({"pack_confidence": -0.1}, "pack_confidence < 0"),
        ({"pack_qty": 0}, "pack_qty 0 (zero-pack bug)"),
        ({"volume_ml": -5.0}, "negative volume"),
    ):
        try:
            _attrs(**bad_over)
            check(f"ExtractedAttributes rejects {why}", False)
        except Exception:
            check(f"ExtractedAttributes rejects {why}", True)

    check("GateResult accepts proceed", GateResult(decision="proceed", reason="ok").decision == "proceed")
    try:
        GateResult(decision="maybe", reason="??")
        check("GateResult rejects unknown decision", False)
    except Exception:
        check("GateResult rejects unknown decision", True)

    def _canon(**over):
        base = {
            "gtin": "4006381333931", "canonical": "brand orange water",
            "mode_brand": "b", "mode_flavor": "orange", "mode_type": "water",
            "salient_ngrams": ["x"], "dropped_redundant_ngrams": [],
            "volume_set": {500.0}, "pack_set": {1},
            "volume_confidence": 0.9, "pack_confidence": 0.9,
            "volume_consistency": 1.0, "pack_consistency": 1.0, "n_titles": 2,
        }
        base.update(over)
        return CanonicalRecord(**base)

    check("CanonicalRecord accepts valid", _canon().gtin == "4006381333931")
    for bad_over, why in (
        ({"volume_set": {0.0}}, "volume 0 in set"),
        ({"pack_set": {0}}, "pack 0 in set"),
        ({"volume_consistency": 1.2}, "consistency > 1"),
        ({"n_titles": 0}, "n_titles 0"),
    ):
        try:
            _canon(**bad_over)
            check(f"CanonicalRecord rejects {why}", False)
        except Exception:
            check(f"CanonicalRecord rejects {why}", True)

    payload = ["a", "b", "c"]
    row_bc = np.array(["1", "2", "3"])
    ok_pos = np.array([[0, 2], [1, 2]])
    good = TrainingData(
        payload=payload, row_bc=row_bc, pos=ok_pos,
        neg=np.empty((0, 2), dtype=int), gtin_to_row={"1": 0}, stats={
            "n_rows": 3, "n_sku_with_canonical": 1, "n_pos_empty_dropped": 0,
            "n_empty_sku_texts": 0, "n_empty_canon_texts": 0, "n_canonicals": 1,
            "n_pos_gate_rows": 1, "n_neg_gate_rows": 0, "n_neg_resolved": 0,
            "n_neg_dropped": 0,
        },
    )
    check("TrainingData accepts aligned bundle", good.payload == payload)
    try:
        TrainingData(
            payload=payload, row_bc=np.array(["1", "2"]), pos=ok_pos,
            neg=np.empty((0, 2), dtype=int), gtin_to_row={}, stats=good.stats.model_dump(),
        )
        check("TrainingData rejects unlocked row_bc", False)
    except Exception:
        check("TrainingData rejects unlocked row_bc", True)
    try:
        TrainingData(
            payload=payload, row_bc=row_bc,
            pos=np.array([[0, 3]]),  # index 3 out of range
            neg=np.empty((0, 2), dtype=int), gtin_to_row={}, stats=good.stats.model_dump(),
        )
        check("TrainingData rejects out-of-range pos", False)
    except Exception:
        check("TrainingData rejects out-of-range pos", True)

    mr = MaskingResult(
        pos=np.array([[0, 1]]), payload=["x y z", "w"], row_bc=np.array(["1", "1"]),
        n_added=1,
        audit=[{
            "anchor_payload_idx": 0, "copy_payload_idx": 2, "pair_payload_idx": 1,
            "barcode": "1", "realized_extent": 0.2, "anchor_text": "x y z",
            "masked_text": "` y z",
        }],
    )
    check("MaskingResult accepts aligned mask", mr.n_added == 1)
    try:
        MaskingResult(
            pos=np.array([[0, 1]]), payload=["x", "y"], row_bc=np.array(["1", "2"]),
            n_added=1, audit=[],  # n_added != audit rows
        )
        check("MaskingResult rejects n_added/audit mismatch", False)
    except Exception:
        check("MaskingResult rejects n_added/audit mismatch", True)

    ok_dt = DataTuple(
        n_df=2, payload=["a", "b", "c"], row_bc=np.array(["1", "2", "3"]),
        country=np.array(["X", "Y", "X"]), pos=np.array([[0, 2]]),
        hp_pairs=np.empty((0, 2), dtype=int), emb0=np.zeros((3, 4)),
    )
    check("DataTuple accepts aligned 7-tuple", ok_dt.n_df == 2)
    for over, why in (
        ({"country": np.array(["X"])}, "country shorter than payload"),
        ({"emb0": np.zeros((2, 4))}, "emb0 rows != payload"),
        ({"pos": np.array([[0, 9]])}, "pos out of range"),
    ):
        kw = {
            "n_df": 2, "payload": ["a", "b", "c"],
            "row_bc": np.array(["1", "2", "3"]),
            "country": np.array(["X", "Y", "X"]), "pos": np.array([[0, 2]]),
            "hp_pairs": np.empty((0, 2), dtype=int), "emb0": np.zeros((3, 4)),
        }
        kw.update(over)
        try:
            DataTuple(**kw)
            check(f"DataTuple rejects {why}", False)
        except Exception:
            check(f"DataTuple rejects {why}", True)

    check(
        "FoldSets accepts disjoint folds",
        FoldSets(folds=[{"a"}, {"b"}]).folds[0] == {"a"},
    )
    try:
        FoldSets(folds=[{"a", "b"}, {"b", "c"}])
        check("FoldSets rejects overlapping folds", False)
    except Exception:
        check("FoldSets rejects overlapping folds", True)

    check(
        "check_verdict_map accepts vocabulary",
        check_verdict_map({"b12": "keep_nutrient", "473": "strip"})["b12"] == "keep_nutrient",
    )
    try:
        check_verdict_map({"x": "keep"})
        check("check_verdict_map rejects bad verdict", False)
    except Exception:
        check("check_verdict_map rejects bad verdict", True)


def oracle_zero_pack_guard() -> None:
    """The pack_qty >= 1 contract (found live by the schema on first run):
    'pack 0.5 l' / '0% ... pack' title forms must NOT produce pack_qty=0 —
    they fall through to the next rule or the default single."""
    import euromonitor.pipeline

    cases = [
        ("DIA cola drink 0% pack 12 cans 33 cl", 12),  # real pack 12
        ("bar-le- Duc water pack 0.5 l", 1),            # 0.5l is a VOLUME
        ("ACTIPH alkaline water pH 9.0 bottle 600 ml", 1),
        ("cola pack 0, 5l", 1),
    ]
    for title, want in cases:
        got = euromonitor.pipeline.extract_all(title, "")["pack_qty"]
        check(
            f"zero-guard {title[:36]!r} -> pack {want}",
            got == want,
            f"got {got}",
        )
    # regression: the ordinary forms still parse
    for title, want in (
        ("24 x 330ml", 24), ("Pack of 6", 6), ("12 cans", 12),
        ("pack23", 23), ("sugar free cola", 1),
    ):
        got = euromonitor.pipeline.extract_pack_from_title(title)[0]
        check(f"pack regression {title!r} == {want}", got == want, f"got {got}")


def main() -> None:
    print("== 1. GS1 checksum ==")
    oracle_gtin()
    print("== 2. cleaning / soft-stop ==")
    oracle_cleaning()
    print("== 3. number-token reference ==")
    oracle_number_reference()
    print("== 4. eval pairs (blocking) ==")
    oracle_eval_pairs()
    print("== 5. hard-negative mining ==")
    oracle_mining()
    print("== 6. component folds ==")
    oracle_folds()
    print("== 6b. real-data holdout integrity (50/25/25) ==")
    oracle_holdout_integrity()
    print("== 7. precision-at-recall ==")
    oracle_precision_at_recall()
    print("== 7b. Youden holdout discipline ==")
    oracle_youden_discipline()
    print("== 8. _append_csv ==")
    oracle_append_csv()
    print("== 10. diet/pulp phrase variations ==")
    oracle_phrase_variants()
    print("== 11. word-once canonical discipline ==")
    oracle_word_once()
    print("== 12. config split (SSOT files) ==")
    oracle_config_split()
    print("== 12b. no-fallback SSOT (inline literals -> config) ==")
    oracle_no_fallback_ssot()
    print("== 12c. round-3 fix pins (audit F01-F21 remediation) ==")
    oracle_round3_pins()
    print("== 13. pydantic boundary schemas ==")
    oracle_schemas()
    print("== 14. zero-pack guard ==")
    oracle_zero_pack_guard()
    print("== 9. pinned real-data counts ==")
    oracle_pinned_counts()
    print()
    if FAILED:
        print(f"SELFTEST FAILED: {len(FAILED)} oracle(s):")
        for f in FAILED:
            print(f"  - {f}")
        raise SystemExit(1)
    print("SELFTEST PASSED — all oracles green")


if __name__ == "__main__":
    main()
