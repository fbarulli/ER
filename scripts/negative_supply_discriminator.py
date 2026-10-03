"""Real-vs-minted discriminator: the pre-scale check before 50k minted rows.

Owner order: blocker -> count anchors with real partners -> mint the
remainder -> DISCRIMINATOR -> train. The discriminator trains logistic
regression on cheap text-geometry features (no model embeddings yet: that
is the embeddings stage landing behind the same blocker interface) over
two arms:

  real arm    : base_negative + real_partner rows (both real labels 0)
  minted arm  : minted_partner rows

If the arms separate easily (AUC near 1.0), the generator has a
detectable signature and the mint rules get fixed BEFORE minting 50k
rows. Loud verdict, not a silent number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_NOTE = "features are universe-geometry only; mint provenance is NOT a feature"


def _features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    anchor = frame["anchor_text"].fillna("").astype(str)
    partner = frame["partner_text"].fillna("").astype(str)
    for column in ("anchor_gtin", "partner_gtin", "score"):
        if column not in frame.columns:
            frame = frame.assign(**{column: "" if column != "score" else 0.0})
    sets_a = anchor.map(lambda cell: set(cell.split()))
    sets_p = partner.map(lambda cell: set(cell.split()))
    jaccard = [
        len(x & y) / max(len(x | y), 1)
        for x, y in zip(sets_a, sets_p)
    ]
    subset_share = [
        len(x & y) / max(min(len(x), len(y)), 1)
        for x, y in zip(sets_a, sets_p)
    ]
    a_len = anchor.str.split().map(len).to_numpy()
    p_len = partner.str.split().map(len).to_numpy()
    cols = np.column_stack([
        jaccard,
        subset_share,
        np.abs(a_len.astype(float) - p_len.astype(float)) / np.maximum(a_len, 1),
        frame["score"].fillna(0.0).astype(float).to_numpy(),
    ])
    names = [
        "token_jaccard",
        "subset_share",
        "token_count_gap",
        "blocker_score",
    ]
    return cols, names


def discriminate(pairs: pd.DataFrame) -> dict:
    """Fit and read the separation; the verdict is loud by construction."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_score

    real = pairs[
        pairs.population.isin(["base_negative", "real_partner"])
    ].copy()
    minted = pairs[pairs.population == "minted_partner"].copy()
    arm = pd.concat([real, minted], ignore_index=True)
    for column in ("anchor_gtin", "partner_gtin"):
        if column not in arm.columns:
            arm[column] = ""
    if arm["population"].nunique() < 2 or len(minted) < 10 or len(real) < 10:
        return {
            "verdict": "insufficient",
            "real_rows": int(len(real)), "minted_rows": int(len(minted)),
            "reason": "need >= 10 rows on each arm before the weapon-scale check",
        }
    X, names = _features(arm)
    y = (arm["population"] == "minted_partner").astype(int).to_numpy()
    groups = []
    for position, row in enumerate(arm.itertuples(index=False)):
        anchor = str(getattr(row, "anchor_gtin", "") or "").strip()
        groups.append(anchor if anchor else f"__mint__{position}")
    groups = np.asarray(groups)
    model = LogisticRegression(max_iter=2000)
    splitter = GroupKFold(n_splits=min(5, np.unique(groups).shape[0]))
    auc = float(cross_val_score(
        model, X, y, cv=splitter, scoring="roc_auc", groups=groups
    ).mean())
    model.fit(X, y)
    coefficients = dict(sorted(zip(names, model.coef_[0].round(4)), key=lambda kv: -abs(kv[1])))
    verdict = "SEPARABLE" if auc >= 0.90 else (
        "borderline" if auc >= 0.75 else "not-separable"
    )
    return {
        "verdict": verdict,
        "auc_gtin_grouped_cv": round(auc, 4),
        "feature_coefficients": coefficients,
        "real_rows": int(len(real)),
        "minted_rows": int(len(minted)),
        "note": REPO_NOTE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_csv", type=Path, help="emitted pairs.csv")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs_csv)
    report = discriminate(pairs)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report.get("verdict") == "SEPARABLE":
        raise SystemExit(
            "real-vs-minted discriminator: SEPARABLE — fix the generator "
            "before generating 50k minted rows"
        )


if __name__ == "__main__":
    main()
