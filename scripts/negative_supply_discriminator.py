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

Class map (one owner per responsibility):
  - ArmGeometry     — the universe-geometry feature columns
  - RealMintedArms  — arm selection + the grouped-anchor fold key
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.run_log import RunLogger

_LOG = RunLogger(__name__)

REPO_NOTE = "features are universe-geometry only; mint provenance is NOT a feature"

_REAL_POPULATIONS = ("base_negative", "real_partner")
_MINTED_POPULATION = "minted_partner"


class ArmGeometry:
    """The universe-geometry features over one arm frame.

    Four cheap columns: token jaccard, subset share, the token-count gap
    and the blocker cosine. Mint provenance is deliberately NOT a feature
    (REPO_NOTE): the discriminator must not see the answer key.
    """

    @staticmethod
    def columns(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """The feature matrix plus its pinned column names."""
        anchor = frame["anchor_text"].fillna("").astype(str)
        partner = frame["partner_text"].fillna("").astype(str)
        for column in ("anchor_gtin", "partner_gtin", "score"):
            if column not in frame.columns:
                frame = frame.assign(**{column: "" if column != "score" else 0.0})
        sets_a = anchor.map(lambda cell: set(cell.split()))
        sets_p = partner.map(lambda cell: set(cell.split()))
        jaccard: list[float] = []
        subset_share: list[float] = []
        for x, y in _LOG.progress(
            zip(sets_a, sets_p), desc="arm_features", unit="row",
            total=len(anchor),
        ):
            overlap = len(x & y)
            jaccard.append(overlap / max(len(x | y), 1))
            subset_share.append(overlap / max(min(len(x), len(y)), 1))
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


def _features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """The universe-geometry feature matrix (see :class:`ArmGeometry`)."""
    return ArmGeometry.columns(frame)


class RealMintedArms:
    """The two comparison arms and the grouped-anchor fold key."""

    @staticmethod
    def split(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """(real, minted, stacked arm) frames, real first."""
        real = pairs[pairs.population.isin(_REAL_POPULATIONS)].copy()
        minted = pairs[pairs.population == _MINTED_POPULATION].copy()
        arm = pd.concat([real, minted], ignore_index=True)
        for column in ("anchor_gtin", "partner_gtin"):
            if column not in arm.columns:
                arm[column] = ""
        return real, minted, arm

    @staticmethod
    def anchor_groups(arm: pd.DataFrame) -> np.ndarray:
        """The fold key: the anchor gtin, or a per-row group when missing."""
        groups: list[str] = []
        rows = list(arm.itertuples(index=False))
        for position, row in _LOG.progress(
            enumerate(rows), desc="anchor_groups", unit="row", total=len(rows),
        ):
            value = getattr(row, "anchor_gtin", "")
            anchor = "" if pd.isna(value) else str(value).strip()
            groups.append(anchor if anchor else f"__mint__{position}")
        return np.asarray(groups)


def discriminate(pairs: pd.DataFrame, spec=None) -> dict:
    """Fit and read the separation; the verdict is loud by construction.

    Thresholds ride the negative-supply lane's validated spec
    (DiscriminatorSpec via EUROMONITOR_NEGATIVE_SUPPLY_SPEC JSON) — the same
    config document that drives mining, so tuning the run-fail gates is a
    config edit, not a script edit.
    """
    from training.negative_supply import DiscriminatorSpec

    spec = spec or DiscriminatorSpec()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

    real, minted, arm = RealMintedArms.split(pairs)
    if arm["population"].nunique() < 2 or len(minted) < spec.min_arm_rows or len(real) < spec.min_arm_rows:
        return {
            "verdict": "insufficient",
            "real_rows": int(len(real)), "minted_rows": int(len(minted)),
            "reason": (
                f"need >= {spec.min_arm_rows} rows on each arm before the "
                "weapon-scale check"
            ),
        }
    X, names = _features(arm)
    y = (arm["population"] == _MINTED_POPULATION).astype(int).to_numpy()
    groups = RealMintedArms.anchor_groups(arm)
    model = LogisticRegression(max_iter=spec.max_iter)
    group_count = np.unique(groups).shape[0]
    if group_count < 2:
        return {
            "verdict": "insufficient",
            "real_rows": int(len(real)), "minted_rows": int(len(minted)),
            "reason": "need at least two independent anchor groups",
        }
    splitter = StratifiedGroupKFold(n_splits=min(spec.cv_folds, group_count))
    splits = list(splitter.split(X, y, groups))
    if any(np.unique(y[index]).size < 2 for split in splits for index in split):
        return {
            "verdict": "insufficient",
            "real_rows": int(len(real)), "minted_rows": int(len(minted)),
            "reason": "grouped CV requires both arms in every training and validation fold",
        }
    scores = cross_val_score(model, X, y, cv=splits, scoring="roc_auc", error_score="raise")
    if not np.isfinite(scores).all():
        raise ValueError("discriminator grouped CV produced non-finite AUC scores")
    auc = float(scores.mean())
    model.fit(X, y)
    coefficients = dict(sorted(zip(names, model.coef_[0].round(4)), key=lambda kv: -abs(kv[1])))
    verdict = "SEPARABLE" if auc >= spec.separable_auc else (
        "borderline" if auc >= spec.borderline_auc else "not-separable"
    )
    return {
        "verdict": verdict,
        "auc_gtin_grouped_cv": round(auc, 4),
        "feature_coefficients": coefficients,
        "real_rows": int(len(real)),
        "minted_rows": int(len(minted)),
        "thresholds": {
            "separable_auc": spec.separable_auc,
            "borderline_auc": spec.borderline_auc,
            "min_arm_rows": spec.min_arm_rows,
            "cv_folds": spec.cv_folds,
        },
        "note": REPO_NOTE,
    }


def main() -> None:
    RunLogger.configure_console()
    from training.negative_supply import DiscriminatorSpec, load_spec

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_csv", type=Path, help="emitted pairs.csv")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    pairs = pd.read_csv(
        args.pairs_csv, dtype={"anchor_gtin": str, "partner_gtin": str},
        keep_default_na=False,
    )
    # The lane spec owns the thresholds (env EUROMONITOR_NEGATIVE_SUPPLY_SPEC
    # JSON overrides module defaults, exactly like the mining stage).
    supply_spec = load_spec()
    spec = supply_spec.discriminator or DiscriminatorSpec()
    report = discriminate(pairs, spec=spec)
    print(json.dumps(report, indent=2, sort_keys=True))
    _LOG.info(f"[discriminator] verdict={report.get('verdict')}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report.get("verdict") == "SEPARABLE":
        raise SystemExit(
            "real-vs-minted discriminator: SEPARABLE — fix the generator "
            f"before minting up to {supply_spec.mint.max_minted:,} partner rows"
        )


if __name__ == "__main__":
    main()
