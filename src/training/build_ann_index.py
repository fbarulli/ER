"""Build/query the complete persisted catalog HNSW index."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core.ann_config import load_ann_config
from training.rand_matching import RandMatcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--query-input", type=Path)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    cfg = load_ann_config()
    matcher = RandMatcher(
        args.checkpoint.resolve(),
        batch_size=cfg.embedding.encode_batch_size,
        top_k=cfg.index.top_k,
        ann_index_dir=args.index_dir,
        rebuild_ann_index=args.rebuild,
    )
    if args.query_input is not None:
        frame = pd.read_csv(args.query_input, dtype=str).head(args.query_limit)
        candidates = matcher.score_candidates(frame)
        print(
            f"HNSW query smoke rows={len(frame):,} "
            f"candidates={len(candidates):,} top_k={matcher.top_k}"
        )


if __name__ == "__main__":
    main()
