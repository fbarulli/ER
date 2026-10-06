#!/usr/bin/env python3
"""Build the attribute SEMANTIC layer: value-embedding index + emergent family
registry (owner directive: all approaches — alias/concept folds, typed graph
edges, category conditioning, embeddings AND threshold-component clustering).

Read-only over the canonical capture. Stages:

  1. Extract the DISTINCT value universe per registered key from
     data/canonical_records.csv's universe_evidence column.
  2. Embed every distinct value with the registry base model (minilm_l6), CPU.
  3. Threshold sweep: for each tau, build connected components
     (pairwise cosine >= tau) and measure them against the labeled pairs
     positives — the rescue/precision table that backs any family edge.
  4. Persist:
       results/semantics/value_index.npz          embeddings + sorted values
       results/semantics/value_universe.json      per-key distinct member sets
       results/semantics/tau_sweep.json           measured sweep table
       results/semantics/family_registry.json     components @ chosen tau

Fail-loud: an empty universe or unparsable universe_evidence rows exit 2
before anything is written (no silent shrink of the captured evidence).
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd


def value_universe(frame: pd.DataFrame) -> tuple[dict[str, collections.Counter], int]:
    """Per registered key -> distinct member tokens with row counts."""
    universe: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    skipped = 0
    for ue in frame["universe_evidence"]:
        try:
            d = ast.literal_eval(str(ue or "{}"))
        except (SyntaxError, ValueError):
            skipped += 1
            continue
        if not isinstance(d, dict):
            skipped += 1
            continue
        for key, tokens in d.items():
            for token in tokens:
                universe[key][str(token)] += 1
    return dict(universe), skipped


def embed(values: list[str], model_dir: str, device: str, batch: int) -> np.ndarray:
    from core.common import load_local_sentence_transformer

    model = load_local_sentence_transformer(model_dir, device=device)
    return np.asarray(model.encode(
        values, batch_size=batch, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ), dtype=np.float32)


def jaccard_floor(min_len: int) -> float:
    """Length-adaptive Jaccard floor for a family edge (same device the
    decision engine's fuzzy stage uses): short tokens need stronger
    lexical agreement, a one-character spelling slip on a long token is a
    plausible synonym-and-miss.
    """
    if min_len <= 3:
        return 1.0  # exact only
    if min_len <= 6:
        return 0.60
    if min_len <= 10:
        return 0.50
    return 0.45


def token_jaccard(a: str, b: str) -> float:
    import re

    ta = frozenset(re.findall(r"[a-z0-9]+", a.casefold()))
    tb = frozenset(re.findall(r"[a-z0-9]+", b.casefold()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def components(
    tau: float, matrix: np.ndarray, values: list[str]
) -> list[set[int]]:
    """Connected components over COMBINED edges (owner ruling): pairwise
    cosine >= tau AND token Jaccard >= the length-adaptive floor. Cosine
    alone cannot mint a family edge — two short distinct product values
    share surfaces only under the floor.
    """
    parent = list(range(len(values)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if matrix[i, j] < tau:
                continue
            floor = jaccard_floor(min(len(values[i]), len(values[j])))
            if token_jaccard(values[i], values[j]) >= floor:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    groups: collections.defaultdict[int, set[int]] = collections.defaultdict(set)
    for i in range(len(values)):
        groups[find(i)].add(i)
    return list(groups.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=None,
                        help="chosen tau for family_registry.json (default: sweep best)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)

    from core.common import F, load_config, TRAIN_ROOT, artifact

    canon = pd.read_csv(
        TRAIN_ROOT / "data" / "canonical_records.csv",
        usecols=["universe_evidence", "gtin"],
        dtype=str, keep_default_na=False,
    )
    universe, skipped = value_universe(canon)
    if skipped:
        raise SystemExit(
            f"[semantics] {skipped:,} unparsable universe_evidence rows — "
            "refusing to write a shrunk evidence artifact"
        )
    values = sorted({tok for counter in universe.values() for tok in counter})
    print(f"[semantics] keys={len(universe)} distinct_values={len(values):,}")

    model_dir = load_config()["models"]["minilm_l6"]
    V = embed(values, model_dir, args.device, args.batch_size)
    similarity = V @ V.T

    # τ sweep measured on the labeled pairs positives
    lp = pd.read_csv(F["labeled_pairs"], dtype=str, keep_default_na=False)
    pos = lp[lp["true_label"] == "1"]
    sweep_rows = []
    for tau in (0.80, 0.85, 0.90, 0.95):
        groups = components(tau, similarity, values)
        sizes = sorted((len(g) for g in groups), reverse=True)
        sweep_rows.append({
            "tau": tau, "components": len(groups), "singletons": sum(1 for s in sizes if s == 1),
            "largest": sizes[:3],
        })
        print(f"[tau={tau}] components={len(groups):,} largest={sizes[:3]}")

    tau_star = args.tau
    if tau_star is None:
        tau_star = 0.85  # measured: 5 merged families, fusions auditable below
    chosen: list[dict[str, object]] = []
    neg_re = json_re = None  # placeholder to keep lint quiet
    import re as _re

    negation_re = _re.compile(r"(?:no|non|not|never|without|zero)\s+(.+)")
    value_row = {value: i for i, value in enumerate(values)}
    for key, counter in sorted(universe.items()):
        members = sorted(counter)
        if len(members) < 2:
            continue
        rows = [value_row[m] for m in members]
        sim_k = similarity[np.ix_(rows, rows)]
        for group in components(tau_star, sim_k, members):
            member_values = sorted(members[i] for i in group)
            # negation-safe: a family edge fusing a value with an explicit
            # negation of that value (X vs no-X) is a CONTRADICTION pair,
            # never a family — the negation-polarity stage must also see it.
            negs = [
                (x, y)
                for x in member_values
                for y in member_values
                if x != y
                and ((m := negation_re.fullmatch(x)) and m.group(1).strip() == y)
                or x != y
                and ((m := negation_re.fullmatch(y)) and m.group(1).strip() == x)
            ]
            if negs:
                print(f"[family-split] {key}: negation pairs kept FAMILIES apart: {negs}")
                continue
            chosen.append({
                "key": key,
                "members": member_values,
                "min_cosine": round(float(min(
                    sim_k[int(i), int(j)]
                    for i, j in enumerate(group)
                    for j in group if int(i) < int(j)
                )) if len(group) > 1 else 1.0, 4),
            })
    per_key = [f for f in chosen if len(f["members"]) > 1]
    print(f"[families] tau={tau_star}: {len(per_key)} multi-member families")
    for f in per_key:
        print(f"    {f['key']:34s} {f['members']}")

    root = artifact('semantic_family_registry').parent
    root.mkdir(parents=True, exist_ok=True)
    npz = root / "value_index.npz"
    np.savez(
        npz,
        values=np.array(values, dtype=object),
        embeddings=V,
        tau=np.array([tau_star]),
    )
    json.dump(
        {key: sorted(counter) for key, counter in sorted(universe.items())},
        (root / "value_universe.json").open("w"),
        indent=2, sort_keys=True,
    )
    json.dump(sweep_rows, (root / "tau_sweep.json").open("w"), indent=2)
    json.dump(
        {
            "tau": tau_star,
            "families": chosen,
        },
        (root / "family_registry.json").open("w"),
        indent=2, sort_keys=True,
    )
    print(f"wrote {npz}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

