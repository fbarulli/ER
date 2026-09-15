#!/usr/bin/env python3
"""Explain holdout false positives/negatives with unmatched text tokens.

The pair CSV is the scored holdout population.  For every incorrect decision
at the supplied threshold this script joins source/target text, reports words
present on only one side, and counts those words by error type.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "of", "on",
    "or", "the", "to", "with",
}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")


def tokens(*values: object) -> set[str]:
    text = " ".join(str(v or "") for v in values).lower()
    return {t for t in TOKEN_RE.findall(text) if t not in STOPWORDS and len(t) > 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--canonical-records", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    data = pd.read_csv(args.data, dtype=str, keep_default_na=False).fillna("")
    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_source = "--threshold"
    elif args.metrics is not None:
        metrics = pd.read_csv(args.metrics)
        ok = metrics[metrics["status"].eq("ok")] if "status" in metrics else metrics
        if ok.empty or "youden_thr" not in ok:
            raise ValueError("metrics file has no usable youden_thr")
        threshold = float(ok.iloc[0]["youden_thr"])
        threshold_source = f"{args.metrics}:youden_thr"
    else:
        raise ValueError("provide --threshold or --metrics")

    required = {"label", "score", "sku_id_a", "sku_id_b"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs missing columns: {sorted(missing)}")

    columns = [c for c in ("product_id", "title", "attributes", "description", "brand", "category") if c in data]
    records = {
        str(row["product_id"]): row.to_dict()
        for _, row in data[columns].iterrows()
    }
    if args.canonical_records:
        canon = pd.read_csv(args.canonical_records, dtype=str, keep_default_na=False).fillna("")
        for _, row in canon.iterrows():
            gtin = str(row.get("gtin", ""))
            if gtin:
                records[f"canon#{gtin}"] = row.to_dict()

    def text_for(identifier: object) -> str:
        row = records.get(str(identifier), {})
        return " ".join(str(row.get(c, "")) for c in ("title", "canonical", "attributes", "description", "brand", "category"))

    pairs["score"] = pd.to_numeric(pairs["score"])
    pairs["label"] = pd.to_numeric(pairs["label"]).astype(int)
    pairs["predicted"] = (pairs["score"] >= threshold).astype(int)
    incorrect = pairs[pairs["label"] != pairs["predicted"]].copy()
    rows: list[dict[str, object]] = []
    missed = Counter()
    unmatched = Counter()
    for _, row in incorrect.iterrows():
        source_id, target_id = str(row["sku_id_a"]), str(row["sku_id_b"])
        source_text, target_text = text_for(source_id), text_for(target_id)
        source_words, target_words = tokens(source_text), tokens(target_text)
        source_only, target_only = sorted(source_words - target_words), sorted(target_words - source_words)
        error_type = "false_positive" if row["predicted"] else "false_negative"
        for word in target_only:
            missed[(error_type, word)] += 1
        for word in source_only:
            unmatched[(error_type, word)] += 1
        rows.append({
            **row.to_dict(),
            "error_type": error_type,
            "source_text": source_text,
            "target_text": target_text,
            "source_only_words": " ".join(source_only),
            "target_only_words": " ".join(target_only),
            "shared_words": " ".join(sorted(source_words & target_words)),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "incorrect_predictions_with_word_diffs.csv", index=False)
    word_rows = [
        {"error_type": error, "word": word, "side": side, "count": count}
        for side, counter in (("target_only", missed), ("source_only", unmatched))
        for (error, word), count in counter.items()
    ]
    words = pd.DataFrame(word_rows, columns=["error_type", "word", "side", "count"])
    if not words.empty:
        words = words.sort_values(["error_type", "side", "count", "word"], ascending=[True, True, False, True])
    words.to_csv(args.out_dir / "unmatched_word_summary.csv", index=False)
    summary = pd.DataFrame([{
        "threshold": threshold,
        "threshold_source": threshold_source,
        "scored_rows": len(pairs),
        "incorrect_rows": len(incorrect),
        "false_positives": int(((incorrect["predicted"] == 1) & (incorrect["label"] == 0)).sum()),
        "false_negatives": int(((incorrect["predicted"] == 0) & (incorrect["label"] == 1)).sum()),
    }])
    summary.to_csv(args.out_dir / "incorrect_prediction_summary.csv", index=False)
    print(f"[done] threshold={threshold:g}; incorrect={len(incorrect):,}; output={args.out_dir}")


if __name__ == "__main__":
    main()
