"""folds.py — connected-component folds over the positive-pair graph
The problem with random splitting
If you just randomly assign each pair to training or testing, the same product (same gtin) might end up in both sets (data leakage).

The problem with splitting by gtin
If you split by gtin, those two gtins could land in different folds, and the pair gets broken — you lose training/testing examples.

The connected‑component solution
Imagine drawing a line (edge) between two gtins every time they appear together in a positive pair. Some gtins link directly, and through a chain of links they
form a cluster (called a connected component). For example:

A is paired with B
B is paired with C
So A, B, C are all in one connected component.

Now, instead of splitting individual pairs or gtins, we split these clusters. All gtins in one cluster go into the same fold. That means:
Every positive pair stays entirely inside one fold → no pairs are lost.
No gtin appears in more than one fold → no leakage.
Any gtin that never appears in a positive pair becomes its own little cluster of one.

Why this is good
Fair testing: the model cannot cheat by seeing the same product in training and testing.
No wasted data: every positive pair remains available for training or evaluation.
Leakage prevention: any information that could leak travels along those edges, and we split exactly along those edges, so nothing crosses.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.disjoint_sets import DisjointSet
from core.schemas import CalibrationPartition, FoldSets

_GTIN_DIGITS = frozenset("0123456789")


def normalize_gtin(raw: object) -> str:
    """THE canonical entity key for CROSS-NAMESPACE joins.

    P0 measured the old validation protocol as 73.7% contaminated, and
    ``data/labeled_pairs.csv`` (gtins) looked like a namespace disjoint from
    the training graph's gtins. This function is the one place the two are
    reconciled, used by ``merged_component_graph`` for EDGE RESOLUTION only.

    Rule: drop a float ``.0`` artifact, keep ASCII digits only, then
    left-zero-pad to 14.

    ========  ==================  ==================
    raw       digits              normalized
    ========  ==================  ==================
    ``4006381333931``    13           ``04006381333931``
    ``04006381333931``   14           ``04006381333931``
    ``4006381333931.0``  13 (after)   ``04006381333931``
    ``GTIN:4006...``     13           ``04006381333931``
    ========  ==================  ==================

    The 13/14 pair is the one that matters: a GTIN-14 is an EAN-13 with a
    leading zero, so the same product legitimately appears at both lengths and
    zfill(14) is what collapses them. zfill is idempotent, so repeated
    normalization is stable.

    The float artifact is handled BEFORE the digit scrub, and the order is not
    cosmetic: ``"4006381333931.0"`` scrubbed naively leaves
    ``40063813339310`` -- a THIRTEEN-digit string that pads to
    ``040063813339310``, a completely different key. Scrub-then-strip would
    fabricate a phantom entity for every float-round-tripped gtin, which is
    the very bug this function exists to prevent.

    Anything with no digits is returned as an explicit empty string, never
    coerced to ``"00000000000000"`` -- collapsing every junk key into one
    shared value would fabricate a giant false component and hand it a fold.

    SCOPE, deliberately narrow. This function must NOT be applied to a
    ``row_bc`` array that is used as the node key set of a split that
    downstream code filters against with raw strings: it changes 8,559 of
    14,981 gtins, and a normalized key never equals its raw spelling, so
    those filters would return empty with no error raised. And it buys nothing
    on this dataset for identity purposes -- the deduped data holds 14,981 raw
    gtins and 14,981 distinct normalized keys, i.e. zero duplicate
    spellings -- while retaining the ability to merge two genuinely different
    malformed codes. Use it to join two namespaces. Do not use it to relabel
    one.
    """
    text = str(raw).strip()
    # Drop a float round-trip artifact (".0") BEFORE the digit scrub, else
    # "4006381333931.0" scrubs to 40063813339310 and pads to a DIFFERENT
    # gtin. See the table above.
    if text.endswith(".0"):
        text = text[:-2]
    elif text.endswith("."):
        text = text[:-1]
    digits = "".join(ch for ch in text if ch in _GTIN_DIGITS)
    return digits.zfill(14) if digits else ""


def merged_positive_graph(
    train_pos: np.ndarray,
    train_row_bc: np.ndarray,
    extra_pos: np.ndarray,
    extra_row_bc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Union two positive-pair graphs into ONE component graph.

    This is the P0 leak fix. Leakage travels along positive edges, so any
    gtin reachable from a training gtin via a positive pair is the same
    entity as far as the split is concerned. ``evaluate_models.py`` was
    building its graph from the validation universe ALONE, so validation
    gtins that chain onto a training gtin (or onto each other) were
    invisible to it and the two sides derived DIFFERENT components from the
    same data — measured 23.3% of validation positives with BOTH endpoints in
    the training fold.

    Both inputs are normalized first and the edge sets are unioned, so
    ``row_bc`` is disjoint (every gtin has exactly one index) and
    ``pos`` indexes into it consistently. GTINs present in only one input
    become their own component, which is correct: an unlinked gtin cannot
    leak through a positive edge that does not exist.

    Returns ``(pos, row_bc, stats)``. ``stats`` is for the transparency
    contract — a silent union would make a leak fix indistinguishable from a
    bug that accidentally merged distinct products.
    """
    norm = lambda bc: normalize_gtin(bc)  # noqa: E731
    edges: dict[tuple[str, str], None] = {}
    seen: set[str] = set()
    for pos_arr, row_arr in ((train_pos, train_row_bc), (extra_pos, extra_row_bc)):
        if row_arr is None:
            continue
        if pos_arr is not None and len(pos_arr):
            for a, b in pos_arr:
                ka, kb = norm(row_arr[int(a)]), norm(row_arr[int(b)])
                if not ka or not kb or ka == kb:
                    continue
                edges[tuple(sorted((ka, kb)))] = None
        seen.update(norm(b) for b in row_arr if norm(b))

    # ``seen`` is the union over BOTH inputs. It has to be accumulated
    # separately from the loop above: a gtin that appears in the extra
    # universe but in none of its positive edges (a validation-only negative
    # endpoint) is still a node, and it must be a node in the SAME index
    # space as the training nodes or the two halves cannot be compared.
    universe = sorted(seen | {k for e in edges for k in e})
    idx = {bc: i for i, bc in enumerate(universe)}
    merged = np.array(
        [[idx[a], idx[b]] for a, b in edges], dtype=np.int64
    ).reshape(-1, 2)
    stats = {
        "train_entities": len({norm(b) for b in train_row_bc if norm(b)}),
        "extra_entities": len({norm(b) for b in extra_row_bc if norm(b)}),
        "merged_entities": len(universe),
        "train_edges": int(len(train_pos)) if train_pos is not None else 0,
        "extra_edges": int(len(extra_pos)) if extra_pos is not None else 0,
        "unique_positive_edges": len(edges),
    }
    return merged, np.array(universe, dtype=object), stats


def load_labeled_pairs(labeled_pairs_csv: str | Path | None = None) -> pd.DataFrame:
    """Read the labeled pair census (gtin1, gtin2, true_label) ONCE, cached.

    The split entry point folds these positive edges into the component graph
    on every call, so the read is cached on (path, mtime, size): a retrain
    that regenerates the census in place must see the new rows, and a
    long-running HPO sweep that calls the entry point dozens of times must not
    re-parse the file each time.
    """
    from core.common import F

    path = Path(labeled_pairs_csv if labeled_pairs_csv is not None else F["labeled_pairs"])
    if not path.exists():
        raise FileNotFoundError(
            f"labeled pair census not found at {path}. The holdout split folds "
            "validation positive edges into the component graph; without the "
            "census it would silently derive the LEAKY split this function "
            "exists to prevent. Regenerate it with "
            "`PYTHONPATH=src python -m src.training.labeled_pairs`."
        )
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _LABELED_PAIRS_CACHE.get("key")
    if cached == key:
        return _LABELED_PAIRS_CACHE["frame"]
    frame = pd.read_csv(
        path, dtype={"gtin1": str, "gtin2": str}, keep_default_na=False
    )
    _LABELED_PAIRS_CACHE.update(key=key, frame=frame)
    return frame


_LABELED_PAIRS_CACHE: dict[str, Any] = {}


def labeled_positive_edges(
    row_bc: np.ndarray,
    labeled_pairs: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, int]]:
    """Resolve labeled POSITIVE pairs to index edges over ``row_bc``.

    These are the edges the P0 leak fix folds into the training graph. Every
    lookup goes through :func:`normalize_gtin` on BOTH sides, which is the
    whole point: ``labeled_pairs`` spells a GTIN as ``4006381333931`` while
    ``row_bc`` may carry ``004006381333931`` or ``4006381333931.0``, and an
    unnormalized join returns zero matches -- which is how the split ended up
    structurally blind to validation in the first place.

    Only ``true_label == 1`` pairs contribute edges. A negative pair is a
    mined *similarity* relation, not an identity claim, so unioning it would
    merge two products the census says are different -- the mirror image of
    the leak. Negatives are still allowed to STRADDLE folds for that reason.

    Both endpoints must resolve to a row in ``row_bc``. An unresolved positive
    is a census/gtin disagreement, so it is counted and reported rather
    than dropped in silence: a silent partial merge is indistinguishable from
    a leak that came back.
    """
    first_row: dict[str, int] = {}
    for i, bc in enumerate(row_bc):
        key = normalize_gtin(bc)
        if key and key not in first_row:
            first_row[key] = i

    label_col = "true_label" if "true_label" in labeled_pairs.columns else None
    if label_col is None:
        raise ValueError(
            "labeled pair census has no `true_label` column; got "
            f"{list(labeled_pairs.columns)}"
        )

    edges: list[tuple[int, int]] = []
    unresolved = self_edge = 0
    for g1, g2, label in zip(
        labeled_pairs["gtin1"], labeled_pairs["gtin2"], labeled_pairs[label_col]
    ):
        if int(label) != 1:
            continue
        a = first_row.get(normalize_gtin(g1))
        b = first_row.get(normalize_gtin(g2))
        if a is None or b is None:
            unresolved += 1
        elif a == b:
            self_edge += 1
        else:
            edges.append((a, b))

    stats = {
        "labeled_positives": int((labeled_pairs[label_col] == 1).sum()),
        "edges_added": len(edges),
        "endpoints_unresolved": unresolved,
        "self_edges_skipped": self_edge,
        "row_entities": len(first_row),
    }
    return np.array(edges, dtype=np.int64).reshape(-1, 2), stats


def merged_component_graph(
    pos: np.ndarray,
    row_bc: np.ndarray,
    *,
    labeled_pairs_csv: str | Path | None = None,
    labeled_pairs: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """THE component graph: training positives UNION validation positives.

    The P0 leak is fixed here, in one function, because a caller that built
    its own graph from the bare training pairs would derive a split that
    leaks -- and a leak is not a crash, so nothing surfaces it. That is
    exactly what ``evaluate_models.py`` did: it built a validation-only graph
    and disagreed with training. Measured on the current census, the old
    protocol contaminates 73.7% of labeled pairs (24.0% of positives had BOTH
    endpoints in train, 49.7% had one), so the P@R95 it reported measured
    memorization as much as generalization.

    Labeled POSITIVE pairs are unioned in as edges. A validation positive with
    one side in train and one side in test is the leak; once the pair IS an
    edge, both endpoints land in the same component and therefore the same
    fold, so a test-fold positive has neither side in train by construction.
    Negatives are not identity claims and are deliberately NOT unioned.

    ``row_bc`` IS RETURNED UNCHANGED, and that is load-bearing. The labeled
    census and ``row_bc`` are two namespaces; :func:`normalize_gtin` bridges
    them for EDGE RESOLUTION ONLY. Normalizing the returned array would be a
    silent, total breakage: every downstream filter matches raw ``row_bc``
    against the returned sets, and 8,559 of 14,981 gtins are 13-digit, so
    a normalized key (`02000000944753`) never matches its raw spelling
    (`2000000944753`) and every pair built from it is quietly dropped --
    with no error, only a smaller training set. Note also that normalizing the
    graph is not merely unnecessary here, it is mildly harmful: the deduped
    data holds 14,981 raw gtins and 14,981 distinct normalized keys (no
    duplicate spellings at all), so normalization changes nothing while being
    able to collapse two genuinely different malformed codes into one
    component. TODO.md's "0/5,428 intersection / missing normalization" root
    cause is therefore wrong: the join resolves 8,809/8,809 pairs with OR
    without it. The defect is the absent graph edges alone.

    Returns ``(pos, row_bc, stats)``.
    """
    from core.identity_policy import review_mask
    if review_mask(pd.Series(row_bc)).any():
        raise ValueError("split graph contains quarantined identity groups; rebuild eligible inputs")
    census = labeled_pairs if labeled_pairs is not None else load_labeled_pairs(
        labeled_pairs_csv
    )
    held_edges = review_mask(census["gtin1"]) | review_mask(census["gtin2"])
    census = census.loc[~held_edges]
    extra, edge_stats = labeled_positive_edges(row_bc, census)
    edge_stats["identity_review_pairs_excluded"] = int(held_edges.sum())
    merged = np.vstack([pos, extra]) if (extra.size and len(pos)) else (
        extra if extra.size else pos
    )
    stats = {
        **edge_stats,
        "train_positive_pairs": int(len(pos)) if pos is not None else 0,
        "merged_positive_pairs": int(len(merged)),
    }
    return merged, row_bc, stats


def component_folds(
    pos: np.ndarray, row_bc: np.ndarray, k: int, seed: int
) -> list[set[str]]:
    """K folds over CONNECTED COMPONENTS of the positive-pair graph.

    Pipeline positives link two DIFFERENT gtins, so gtin-level folds
    straddle pairs (one endpoint per side) and pairs_in_set silently drops
    them — measured 7,489 positives → ~1,500 straddling per fold boundary,
    test sets shrinking to ~130 pairs. Union-find over the pair edges groups
    transitively-linked gtins into components; components are shuffled
    (seeded) and dealt to k folds. Every positive pair sits inside ONE
    component → inside ONE fold: no straddle, no silent loss, no leak
    (leakage travels exactly along the edges we split on).

    Unlinked gtins become singleton components — still fold members so
    their mined negatives split group-aware.

    BOUNDARY CONTRACT (lib.schemas.FoldSets): the returned folds are
    pairwise DISJOINT — a gtin in two folds would put one product in
    train and test at once. Validated on return.
    """
    ds = DisjointSet()

    # every gtin in the dataset is a node (singletons included)
    for bc in row_bc:
        if bc:
            ds.add(bc)
    # union along positive pairs
    for a, b in pos:
        bca, bcb = str(row_bc[a]), str(row_bc[b])
        if bca and bcb:
            ds.union(bca, bcb)

    # components canonically ordered by sorted members — deterministic
    # component identity, independent of union order
    comp_list = ds.components()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(comp_list))
    folds: list[set[str]] = [set() for _ in range(k)]
    for i, comp_idx in enumerate(order):
        folds[i % k] |= comp_list[comp_idx]
    return FoldSets(folds=folds).folds


def component_ids(
    pos: np.ndarray, row_bc: np.ndarray
) -> dict[str, int]:
    """Stable component id per gtin over the positive-pair graph.

    Same union-find and same node set as :func:`component_folds`, but returns
    the id instead of the fold. Needed because ``component_folds`` answers
    "which fold" and cannot answer "are these two gtins the same product" —
    which is the question a leak regression test actually asks.

    Ids are assigned by sorted member list, so they are deterministic for a
    given (pos, row_bc) and stable across runs. The single largest component
    is returned as id 0 by construction only when it sorts first; no code
    should depend on a particular id's magnitude, only on ids being equal
    for gtins that are linked and different for gtins that are not.
    """
    ds = DisjointSet()

    for bc in row_bc:
        if bc:
            ds.add(bc)
    for a, b in pos:
        bca, bcb = str(row_bc[a]), str(row_bc[b])
        if bca and bcb:
            ds.union(bca, bcb)

    ordered = ds.components()
    return {bc: i for i, members in enumerate(ordered) for bc in members}


def holdout_split(
    pos: np.ndarray,
    row_bc: np.ndarray,
    *,
    n_folds: int,
    seed: int,
    dev_fraction: float,
    test_fraction: float,
) -> tuple[set[str], set[str], set[str]]:
    """The SINGLE derivation of the holdout split: (train, dev, test) gtins.

    Roles come from ``n_folds``, not from hardcoded indices: ``test =
    quarters[-1]``, ``dev = quarters[-2]``, ``train = the remaining
    quarters``.  ``component_folds`` deals the components round-robin over
    ``n_folds`` groups, so each quarter is ``1.0 / n_folds`` of the graph and
    the realized shares are fixed by the arity.  A configured
    ``dev_fraction``/``test_fraction`` that does not match that share is a
    mis-configured 50/25/25 contract (the schema pins
    ``holdout_component_folds`` at 4 = 0.25/0.25); it raises here instead of
    silently producing 60/20/20 under a "50/25/25" label.
    """
    if n_folds < 3:
        raise ValueError(
            "holdout split needs at least 3 component folds so train, dev, "
            f"and test are all represented, got {n_folds}"
        )
    quarter = 1.0 / n_folds
    if abs(dev_fraction - quarter) > 1e-9 or abs(test_fraction - quarter) > 1e-9:
        raise ValueError(
            f"holdout split contract violated: n_folds={n_folds} deals quarters "
            f"of {quarter:.4f} each, but the split declares dev_fraction="
            f"{dev_fraction} and test_fraction={test_fraction} "
            f"(the 50/25/25 contract requires n_folds=4)"
        )
    quarters = component_folds(pos, row_bc, n_folds, seed)
    train = set().union(*quarters[:-2])
    if not train:
        raise ValueError(
            "holdout split produced an empty train gtin set; "
            f"n_folds={n_folds} has insufficient component coverage"
        )
    return train, quarters[-2], quarters[-1]


def derive_holdout(
    pos: np.ndarray,
    row_bc: np.ndarray,
    split_cfg: Mapping[str, Any],
    *,
    seed: int,
    labeled_pairs_csv: str | Path | None = None,
    labeled_pairs: pd.DataFrame | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """THE single entry point for the holdout split. Every caller goes here.

    This exists to kill a class of silent divergence: ``holdout_split`` was
    invoked from 5 independent call sites (``train.py``, ``train_prepared.py``,
    ``sid_phase0_report.py``, ``sid_hybrid_eval.py``, ``sid_graph_eval.py``),
    each re-threading ``holdout_component_folds`` / ``dev_fraction`` /
    ``test_fraction`` out of config by hand and passing them positionally. A
    site that misspelled or mis-threaded one of those knobs would derive a
    DIFFERENT split from its siblings and nothing would fail — the split is
    never stored in the bundle (it is re-derived at every call), so divergence
    is invisible in the artifacts. Routing every caller through one function
    makes the contract checkable once, here.

    The graph is built HERE, not by the caller, and that is the P0 blocker
    resolved. P0 changes the component GRAPH (it unions in validation
    positive edges), so the graph had to move behind this one door: a caller
    that built its own graph from the bare training pairs would derive a
    split that leaks -- and being a leak rather than a crash, nothing would
    surface it. Leaving the choice to callers is what let
    ``evaluate_models.py`` build a validation-only graph and disagree with
    training in the first place.

    The returned gtin sets are keys of the caller's own ``row_bc``, NOT
    normalized ones -- callers filter payload rows against them directly (see
    ``core.hard_negatives.pairs_in_set``), so normalizing here would silently
    empty those filters for the 8,559 13-digit gtins.
    """
    merged_pos, graph_bc, stats = merged_component_graph(
        pos,
        row_bc,
        labeled_pairs_csv=labeled_pairs_csv,
        labeled_pairs=labeled_pairs,
    )
    print(
        f"[split] merged component graph: {stats['merged_positive_pairs']:,} positive "
        f"pairs ({stats['train_positive_pairs']:,} training + "
        f"{stats['edges_added']:,} validation) over {stats['row_entities']:,} "
        f"entities; {stats['endpoints_unresolved']:,} labeled positive "
        f"endpoints unresolved",
        flush=True,
    )
    return holdout_split(
        merged_pos,
        graph_bc,
        n_folds=int(split_cfg["holdout_component_folds"]),
        seed=seed,
        dev_fraction=float(split_cfg["dev_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
    )


SPLIT_ROLES: tuple[str, ...] = (
    "train",
    "calibration_fit",
    "calibration_reserved",
    "test",
)
"""The four populations the training-side split actually produces.

NOT two. ``config/training.yaml`` describes train/dev/test, but DEV is then
carved in half by ``partition_component_pairs`` into a GTIN-disjoint
calibration fit/reserved pair (``calibration_dev_fraction``). So the realized
populations are train / calibration_fit / calibration_reserved / test.

The fifth population — ``external_eval`` (the ``labeled_pairs`` set scored by
``evaluate_models.py``) — is deliberately NOT in this tuple: it lives in a
different identifier namespace and derives its own folds. It belongs in the
emitted validation CSV, not in the training split contract.
"""

EXTERNAL_EVAL_ROLE = "external_eval"
ALL_ROLES: tuple[str, ...] = SPLIT_ROLES + (EXTERNAL_EVAL_ROLE,)


def derive_calibration_carve(
    dev_pos: np.ndarray,
    dev_neg: np.ndarray,
    row_bc: np.ndarray,
    split_cfg: Mapping[str, Any],
    *,
    seed: int,
    fold_index: int,
    ensure_different_gtin: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """THE single entry point for the calibration fit/reserved carve.

    Same rationale as ``derive_holdout``: the fraction, the seed offset, and
    the per-fold seed arithmetic were inlined at the one call site in
    ``training.py``, where ``seed + fold_i + calibration_seed_offset`` is a
    second, independent seed entering the fold machinery. Left inline it is a
    silent-divergence site exactly like the config threading that
    ``derive_holdout`` just eliminated — a fold that forgot the ``+17`` offset
    would carve a different calibration set and nothing downstream would say
    so, because the carve is not persisted anywhere.

    The carve is PER FOLD, so the same physical pair can be calibration-fit in
    one fold and test in another. That is why the emitted validation CSV needs
    a ``fold_id`` beside ``role``: under ``--split cv`` a single ``role``
    column cannot express a per-fold assignment.
    """
    fraction = float(split_cfg["calibration_dev_fraction"])
    if not 0.0 < fraction <= 0.5:
        raise ValueError(
            "calibration_dev_fraction must be in (0, 0.5]: it is a share of DEV "
            f"reserved for threshold fitting, and reserving more than half "
            f"leaves calibration_reserved too small to validate the threshold "
            f"it validates. Got {fraction}"
        )
    offset = int(split_cfg["calibration_seed_offset"])
    return partition_component_pairs(
        dev_pos,
        dev_neg,
        row_bc,
        fraction,
        calibration_seed(seed, fold_index, offset),
        ensure_different_gtin=ensure_different_gtin,
    )


def calibration_seed(base_seed: int, fold_index: int, offset: int) -> int:
    """The per-fold calibration carve seed: ``base + fold_index + offset``.

    Named rather than inlined so the arithmetic that used to live at the single
    call site in ``training.py`` is directly testable. Getting it wrong is
    silent: a fold that dropped the ``offset`` would carve a different
    calibration set and nothing downstream would report it.
    """
    return base_seed + fold_index + offset


def partition_component_pairs(
    positive_pairs: np.ndarray,
    negative_pairs: np.ndarray,
    row_bc: np.ndarray,
    fraction: float,
    seed: int,
    *,
    ensure_different_gtin: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition pair pools by positive-pair components without leakage.

    The returned tuple is ``(positive_fit, positive_reserved,
    negative_fit, negative_reserved)`` — see ``CalibrationPartition``, which
    validates the populations and the boundary contract on construction.  The
    reservation is made by whole components AND a pair is reserved only when
    BOTH of its endpoints are reserved: a positive sits inside one component
    (so it is unaffected), but a negative links two DIFFERENT components, and
    a left-endpoint-only rule deals the two mirrored orientations of the same
    product pair to opposite sides of the calibration boundary — the fit half
    keeps ``(rep(g1), canon(g2))`` for early stopping while the calibration
    half keeps ``(rep(g2), canon(g1))``.

    If ``ensure_different_gtin`` is true, one reserved positive component-fold
    is selected from the different-GTIN stratum when that stratum exists. The
    pair remains whole because reservation is still by complete component.
    """
    if len(positive_pairs) == 0:
        raise ValueError("cannot reserve calibration data without DEV positives")
    if not 0.0 < fraction <= 0.5:
        raise ValueError(
            "calibration fraction must be in (0, 0.5]: the component grid "
            "reserves a whole number of folds and can never reserve a "
            f"majority of DEV, got {fraction}"
        )
    pair_rows = np.unique(positive_pairs.ravel())
    local_index = {int(row): position for position, row in enumerate(pair_rows)}
    local_pairs = np.asarray(
        [
            [local_index[int(left)], local_index[int(right)]]
            for left, right in positive_pairs
        ],
        dtype=int,
    )
    # the identity a gtin is matched by is the STRIPPED one on both the
    # component side and the mask side (a padded gtin used to build a
    # component it could never be reserved by)
    local_gtins = np.asarray(
        [str(row_bc[int(row)]).strip() for row in pair_rows], dtype=str
    )
    n_folds = max(2, int(np.ceil(1.0 / fraction)))
    component_groups = component_folds(local_pairs, local_gtins, n_folds, seed)
    n_reserved = min(max(1, int(round(n_folds * fraction))), n_folds - 1)
    selected_group_indices = list(range(n_reserved))
    if ensure_different_gtin:
        def positive_statuses(groups: tuple[int, ...]) -> set[str]:
            selected = set().union(*(component_groups[index] for index in groups))
            return {
                (
                    "both_equal"
                    if str(row_bc[int(left)]).strip()
                    == str(row_bc[int(right)]).strip()
                    else "different"
                )
                for left, right in positive_pairs
                if str(row_bc[int(left)]).strip() in selected
                and str(row_bc[int(right)]).strip() in selected
            }

        def internal_negative_count(groups: tuple[int, ...]) -> int:
            selected = set().union(*(component_groups[index] for index in groups))
            return sum(
                str(row_bc[int(left)]).strip() in selected
                and str(row_bc[int(right)]).strip() in selected
                for left, right in negative_pairs
            )

        feasible = [
            group_indices
            for group_indices in combinations(range(n_folds), n_reserved)
            if {"both_equal", "different"} <= positive_statuses(group_indices)
            and internal_negative_count(group_indices) > 0
        ]
        if not feasible:
            all_statuses = positive_statuses(tuple(range(n_folds)))
            per_group = [
                {
                    "group": group_index,
                    "both_equal": int("both_equal" in positive_statuses((group_index,))),
                    "different": int("different" in positive_statuses((group_index,))),
                    "internal_negatives": internal_negative_count((group_index,)),
                }
                for group_index in range(n_folds)
            ]
            raise RuntimeError(
                "no component-safe calibration reservation can retain both_equal, "
                "different, and a negative population: "
                f"n_folds={n_folds}, n_reserved={n_reserved}, "
                f"available_statuses={sorted(all_statuses)}, groups={per_group}"
            )
        # component_folds is seeded and ``combinations`` is lexicographic, so
        # first feasible is an explicit deterministic reservation rule.
        selected_group_indices = list(feasible[0])
    selected_gtins = set().union(
        *(component_groups[index] for index in selected_group_indices)
    )

    def mask_inside(pairs: np.ndarray, selected: set[str]) -> np.ndarray:
        return np.asarray(
            [
                str(row_bc[int(pair[0])]).strip() in selected
                and str(row_bc[int(pair[1])]).strip() in selected
                for pair in pairs
            ],
            dtype=bool,
        )

    all_positive_gtins = {
        str(row_bc[int(row)]).strip() for row in positive_pairs.ravel()
    }
    fit_gtins = all_positive_gtins - selected_gtins
    positive_mask = mask_inside(positive_pairs, selected_gtins)
    negative_reserved_mask = mask_inside(negative_pairs, selected_gtins)
    negative_fit_mask = mask_inside(negative_pairs, fit_gtins)
    crossing_negative_count = int(
        len(negative_pairs)
        - negative_reserved_mask.sum()
        - negative_fit_mask.sum()
    )
    if crossing_negative_count:
        print(
            "[calibration-partition] excluded "
            f"{crossing_negative_count:,} cross-boundary negatives to keep "
            "fit/reserved positive identities disjoint",
            flush=True,
        )
    partition = CalibrationPartition(
        positive_fit=positive_pairs[~positive_mask],
        positive_reserved=positive_pairs[positive_mask],
        negative_fit=negative_pairs[negative_fit_mask],
        negative_reserved=negative_pairs[negative_reserved_mask],
        row_bc=row_bc,
        n_positive_pairs=len(positive_pairs),
        n_negative_pairs=len(negative_pairs),
        n_negative_pairs_excluded=crossing_negative_count,
    )
    # checks that can actually fail: the two "population" guards that used to
    # sit here compared pairs[~m] + pairs[m] against len(pairs) for one and the
    # same boolean mask, i.e. they held identically for ANY mask.
    if len(partition.positive_reserved) == 0:
        raise RuntimeError(
            "calibration reservation is empty: no positive-pair component was "
            "reserved — the component identity is broken (check row_bc)"
        )
    if len(negative_pairs) and len(partition.negative_reserved) == 0:
        raise RuntimeError(
            "calibration reservation kept no negative pair: no negative pair "
            "has both endpoints inside a reserved component"
        )
    if ensure_different_gtin:
        reserved_status_counts = {
            "both_equal": sum(
                str(row_bc[int(left)]).strip() == str(row_bc[int(right)]).strip()
                for left, right in partition.positive_reserved
            ),
            "different": sum(
                str(row_bc[int(left)]).strip() != str(row_bc[int(right)]).strip()
                for left, right in partition.positive_reserved
            ),
        }
        if any(count == 0 for count in reserved_status_counts.values()):
            raise RuntimeError(
                "required GTIN calibration stratum absent from reserved "
                f"component-fold: counts={reserved_status_counts}"
            )
        print(
            "[calibration-partition] reserved positive strata="
            f"{reserved_status_counts}; fit_pos={len(partition.positive_fit):,}; "
            f"reserved_neg={len(partition.negative_reserved):,}; "
            f"fit_neg={len(partition.negative_fit):,}",
            flush=True,
        )
    return partition.pools()
# (trailing HARDNEG_SIM_THRESHOLD removed — dead constant, no readers; the
# threshold lives in config/training.yaml pairs.hardneg_sim_threshold)
