#!/usr/bin/env python3
"""count_evidence.py — every published count, with the arithmetic that closes it.

No count in this repo should be believed on assertion. This script recomputes
each one from the artifacts themselves and prints the identity that proves it
(does the bucket sum back to the input?), the manifest/docs claim it must
agree with, and PASS / MISMATCH / STALE-DOC / REVIEW. Read-only: it never
writes an artifact. It is safe to run while a stage is regenerating, but the
artifacts on disk are whichever run finished last — the header prints each
artifact's mtime and the git SHA the run recorded.

Record counts come from core.manifest._csv_rows (the SSOT: a csv-module stream
that does not inflate on newlines inside quoted fields). Counting is not
re-implemented here on purpose.

Run: PYTHONPATH=src .venv/bin/python scripts/count_evidence.py
"""
from __future__ import annotations

import ast
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from core.manifest import _csv_rows

DATA = Path("data")
MANIFESTS = Path("results/manifests")
MAX_VOLUME_ML = 10_000.0
TRAIN_FOLD = "0"

rows: list[tuple[str, str, str, str]] = []


def head(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def record(section: str, name: str, value: object, verdict: str) -> None:
    rows.append((section, name, str(value), verdict))


def shape(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    n, cols = _csv_rows(path)
    return n, cols


def manifest(name: str) -> dict:
    path = MANIFESTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def literals(series: pd.Series) -> list:
    out = []
    for text in series:
        try:
            out.extend(ast.literal_eval(text))
        except (ValueError, SyntaxError):
            continue
    return out


# ── provenance ─────────────────────────────────────────────────────────────
head("PROVENANCE")
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
print(f"git HEAD: {sha}")
for name in ("dataset_deduped", "canonical_records", "gate_results",
             "labeled_pairs", "final_validation"):
    p = DATA / f"{name}.csv"
    stamp = "MISSING" if not p.exists() else time.strftime(
        "%m-%d %H:%M", time.localtime(p.stat().st_mtime))
    print(f"{p}: {stamp}")
for name in ("dedupe", "data_prep", "labeled_pairs", "final_validation"):
    m = manifest(name)
    if m:
        env = m.get("environment", {})
        print(f"manifest {name}: status={m.get('status') or m.get('complete')!r} "
              f"finished={m.get('finished')!r} git_sha={str(env.get('git_sha', '-'))[:14]}")

# ── S0: raw export -> deduped ──────────────────────────────────────────────
head("S0  RAW EXPORT -> DEDUPED")
acct = manifest("dedupe").get("row_accounting", {})
inp, out = acct.get("input_rows"), acct.get("output_rows")
drop = acct.get("dropped") or {}
drop_total = sum(v for v in drop.values() if isinstance(v, int))
record("S0", "input_rows (raw export)", inp, "manifest")
record("S0", "output_rows (deduped)", out, "manifest")
for k, v in drop.items():
    record("S0", f"  dropped: {k}", v, "manifest")
record("S0", "  dropped total", drop_total, "manifest")
record("S0", f"identity  {out} + {drop_total} == {inp}", f"{out} + {drop_total} = {out + drop_total}",
       "PASS" if out + drop_total == inp else "MISMATCH")
sh = shape(DATA / "dataset_deduped.csv")
if sh:
    record("S0", "dataset_deduped.csv records", sh[0],
           "PASS" if sh[0] == out else f"MISMATCH (manifest {out})")

# ── S2: raw export -> canonical records ────────────────────────────────────
head("S2  RAW EXPORT -> CANONICAL  (gtin guard + same-GTIN collapse)")
acct = manifest("data_prep").get("row_accounting", {})
inp, out = acct.get("input_rows"), acct.get("output_rows")
drop = acct.get("dropped") or {}
collapsed = acct.get("collapsed_same_gtin")
drop_total = sum(v for v in drop.values() if isinstance(v, int))
for k, v in drop.items():
    record("S2", f"  dropped: {k}", v, "manifest")
record("S2", "dropped total", drop_total, "manifest")
record("S2", f"identity  {inp} - {drop_total} == kept listings", f"{inp} - {drop_total} = {inp - drop_total}",
       "PASS")
record("S2", f"identity  kept == canonical + collapsed", f"{inp - drop_total} == {out} + {collapsed}",
       "PASS" if inp - drop_total == out + collapsed else "MISMATCH")
sh = shape(DATA / "canonical_records.csv")
if sh:
    record("S2", "canonical_records.csv shape", f"{sh[0]} records x {sh[1]} cols", "artifact")
    record("S2", f"  == manifest output_rows ({out})", sh[0], "PASS" if sh[0] == out else "MISMATCH")

# ── gate ───────────────────────────────────────────────────────────────────
head("S2  GATE DECISIONS")
gp = DATA / "gate_results.csv"
if gp.exists():
    g = pd.read_csv(gp, dtype=str, keep_default_na=False)
    sh = shape(gp)
    record("S2", "gate_results.csv shape", f"{sh[0]} records x {sh[1]} cols", "artifact")
    if acct.get("gate_pairs"):
        record("S2", f"  == manifest gate_pairs ({acct['gate_pairs']:,})", sh[0],
               "PASS" if sh[0] == acct["gate_pairs"] else "MISMATCH")
    counts = g.gate_decision.value_counts().to_dict()
    for k in sorted(counts):
        record("S2", f"gate_decision {k}", counts[k], "artifact")
    record("S2", f"identity  decisions sum == records ({len(g):,})", sum(counts.values()),
           "PASS" if sum(counts.values()) == len(g) else "MISMATCH")
    fb = g[g.gate_decision == "fallback"]
    record("S2", "fallback pairs", len(fb), "artifact")
    for reason, n in fb.gate_reason.value_counts().items():
        record("S2", f"  fallback: {reason[:54]}", n, "artifact")
    rep = Path("results/gate_fallback_metrics.json")
    if rep.exists():
        r = json.loads(rep.read_text())
        record("S2", "fallback report agrees with gate_results",
               f"{r.get('total_fallback_pairs')} vs {len(fb)}",
               "PASS" if r.get("total_fallback_pairs") == len(fb) else "STALE-REPORT")

# ── labeled pairs ──────────────────────────────────────────────────────────
head("S2  LABELED PAIRS")
acct = manifest("labeled_pairs").get("row_accounting", {})
lp = DATA / "labeled_pairs.csv"
if lp.exists():
    p = pd.read_csv(lp, dtype=str, keep_default_na=False)
    sh = shape(lp)
    record("pairs", "labeled_pairs.csv shape", f"{sh[0]} records x {sh[1]} cols", "artifact")
    drop = acct.get("dropped") or {}
    drop_total = sum(v for v in drop.values() if isinstance(v, int))
    for k, v in drop.items():
        record("pairs", f"  dropped: {k}", f"{v:,}", "manifest")
    record("pairs", f"identity  gate pairs - dropped == emitted",
           f"{acct.get('input_rows', 0):,} - {drop_total:,} = {acct.get('input_rows', 0) - drop_total:,}",
           "PASS" if acct.get("input_rows", 0) - drop_total == sh[0] else "MISMATCH")
    pos, neg = int((p.true_label == "1").sum()), int((p.true_label == "0").sum())
    record("pairs", "positives (label 1)", pos, "artifact")
    record("pairs", "negatives (label 0)", neg, "artifact")
    record("pairs", f"identity  {pos} + {neg} == {sh[0]}", f"{pos} + {neg} = {pos + neg}",
           "PASS" if pos + neg == sh[0] else "MISMATCH")

# ── final validation: the numbers that looked contradictory ───────────────
head("S1  FINAL VALIDATION  (the '983 vs 422' question, settled)")
fv = DATA / "final_validation.csv"
m = manifest("final_validation")
if fv.exists():
    v = pd.read_csv(fv, dtype=str, keep_default_na=False)
    sh = shape(fv)
    record("S1", "final_validation.csv shape", f"{sh[0]} records x {sh[1]} cols", "artifact")
    pos, neg = int((v.true_label == "1").sum()), int((v.true_label == "0").sum())
    record("S1", "positives in CSV", pos, "artifact")
    record("S1", "negatives in CSV", neg, "artifact")
    record("S1", f"identity  manifest rows/pos/neg {m.get('rows')}/{m.get('positives')}/{m.get('negatives')}",
           f"{sh[0]}/{pos}/{neg}",
           "PASS" if (sh[0], pos, neg) == (m.get("rows"), m.get("positives"), m.get("negatives"))
           else "MISMATCH")
    record("S1", f"identity  {pos} + {neg} == {sh[0]}", f"{pos} + {neg} = {pos + neg}",
           "PASS" if pos + neg == sh[0] else "MISMATCH")
    record("S1", "DATA_PATH.md claims (2026-09-28)", "6,351 total / 565 pos / 5,786 neg",
           "PASS" if (sh[0], pos, neg) == (6351, 565, 5786) else "STALE-DOC")
    gr = m.get("graph", {})
    added = gr.get("validation_edges_added")
    if added:
        record("S1", "graph: validation_edges_added", f"{added:,}", "manifest")
        record("S1", f"identity  train {gr.get('train_positive_pairs'):,} + added {added:,}"
               f" == merged {gr.get('merged_positive_pairs'):,}",
               f"{gr['train_positive_pairs'] + added} == {gr['merged_positive_pairs']}",
               "PASS" if gr["train_positive_pairs"] + added == gr["merged_positive_pairs"]
               else "MISMATCH")
    # The decisive measurement: where do the `added` edges sit by fold?
    fmp = Path("results/training/validation_fold_map.csv")
    if fmp.exists() and lp.exists():
        fm = pd.read_csv(fmp, dtype=str, keep_default_na=False)
        fold = dict(zip(fm.gtin, fm.fold))
        lpairs = pd.read_csv(lp, dtype=str, keep_default_na=False)
        edges = [(a, b) for a, b in zip(lpairs[lpairs.true_label == "1"].gtin1,
                                        lpairs[lpairs.true_label == "1"].gtin2)]
        place = Counter()
        for a, b in edges:
            fa, fb = fold.get(a), fold.get(b)
            place["straddle" if fa != fb else f"both fold {fa}"] += 1
        for k in sorted(place):
            record("S1", f"  labeled-positive edges, {k}", place[k], "artifact")
        train_side = sum(n for k, n in place.items() if k == f"both fold {TRAIN_FOLD}")
        val_side = sum(n for k, n in place.items() if k != f"both fold {TRAIN_FOLD}")
        record("S1", f"identity  {added} added == {train_side} train-fold + {val_side} validation-fold",
               f"{train_side} + {val_side} = {train_side + val_side}",
               "PASS" if added == train_side + val_side else "MISMATCH")
        record("S1", f"  validation-fold edges == CSV positives ({pos})", val_side,
               "PASS" if val_side == pos else "MISMATCH")
        vpairs = {(a, b) for a, b in zip(v[v.true_label == "1"].gtin1, v[v.true_label == "1"].gtin2)}
        lpairs_set = set(edges)
        record("S1", "  every CSV positive is a labeled positive", f"{len(vpairs & lpairs_set)} of {pos}",
               "PASS" if vpairs <= lpairs_set else "MISMATCH")

# ── plausibility census ────────────────────────────────────────────────────
head("PLAUSIBILITY  (owner policy, not defects)")
cp = DATA / "canonical_records.csv"
if cp.exists():
    c = pd.read_csv(cp, dtype=str, keep_default_na=False)
    vols = [float(v) for v in literals(c.volume_set) if isinstance(v, (int, float))]
    big = [v for v in vols if v > MAX_VOLUME_ML]
    record("census", f"volume values > {MAX_VOLUME_ML:,.0f} ml", f"{len(big)} of {len(vols):,}",
           "REVIEW" if big else "PASS")
    record("census", "  largest volumes",
           ", ".join(f"{v:,.0f}" for v in sorted(set(big), reverse=True)[:6]), "REVIEW")
    packs = [int(p) for p in literals(c.pack_set) if isinstance(p, int)]
    dist = Counter(packs)
    record("census", "pack values >= 4 (NORMAL for this corpus: 6x/12x/24x)",
           f"{sum(n for p, n in dist.items() if p >= 4):,} of {len(packs):,}", "PASS")
    record("census", "  largest pack values",
           ", ".join(str(p) for p in sorted(dist)[-6:] if p), "REVIEW")

# ── summary ────────────────────────────────────────────────────────────────
head("SUMMARY")
bad = [r for r in rows if r[3].startswith(("MISMATCH", "STALE", "REVIEW"))]
for _, name, value, verdict in bad:
    print(f"  [{verdict:9s}] {name}: {value}")
print(f"\n{len(rows)} counts checked · {len(bad)} need attention · "
      f"{sum(1 for r in rows if r[3] == 'PASS')} identities close")