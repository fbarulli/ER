"""Leak regression tests for the P0 validation rebuild.

The P0 leak was not a bug anyone could see: `evaluate_models.py` built a
validation-only component graph while training built its own, the two sides
derived DIFFERENT components from the same data, and 73.7% of the old
validation population turned out to be contaminated (24.0% of positives had
BOTH endpoints in train). Nothing raised, because a leak is not a crash.

So these tests pin the two things that make it detectable next time:

  * a synthetic graph where the leak is REAL, proving the invariant is not
    vacuous -- if the fix regressed, the same assertion below goes red rather
    than passing because the fixture happens to be clean;
  * the emitted `data/final_validation.csv` artifact, so a re-derivation that
    quietly reintroduces contamination fails the suite.

Kept deliberately cheap: no `build_training_data` call. The full derivation
takes ~5 minutes and is exercised by `selftest`, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.common import F
from training.folds import (
    component_folds,
    component_ids,
    holdout_split,
    labeled_positive_edges,
    merged_component_graph,
    normalize_gtin,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """A graph where validation positives DO straddle until the fix lands.

    400 gtins. The training graph links them in chains, so folds are
    balanced. The census then claims `bc00000` and `bc00399` are the same
    product -- two gtins that land in different folds absent the merged
    edge, i.e. exactly the leak.
    """
    row_bc = np.array([f"bc{i:05d}" for i in range(400)], dtype=object)
    train_pos = np.array(
        [[i, i + 1] for i in range(0, 398, 2)], dtype=np.int64
    )
    census = pd.DataFrame(
        {
            "gtin1": ["bc00000", "bc00399"],
            "gtin2": ["bc00399", "bc00001"],
            "true_label": [1, 1],
        }
    )
    return train_pos, row_bc, census


SPLIT = {"holdout_component_folds": 4, "dev_fraction": 0.25, "test_fraction": 0.25}


def test_merged_graph_adds_validation_edges() -> None:
    train_pos, row_bc, census = _fixture()
    merged, out_bc, stats = merged_component_graph(
        train_pos, row_bc, labeled_pairs=census
    )
    assert stats["edges_added"] == 2, "validation positives must become edges"
    assert len(merged) == len(train_pos) + 2
    # row_bc MUST come back untouched: downstream filters match raw strings.
    assert list(out_bc) == list(row_bc)


def test_merged_graph_is_not_vacuous() -> None:
    """The census leaks WITHOUT the fix. Guards a passing-but-empty test.

    Asserted as a COUNT over many pairs rather than one hand-picked pair: an
    earlier version of this fixture named a single pair that turned out to land
    in the same fold anyway, so the test passed while proving nothing.
    """
    row_bc = np.array([f"bc{i:05d}" for i in range(400)], dtype=object)
    # chain-linked training graph, so components are pairs and folds balanced
    train_pos = np.array([[i, i + 1] for i in range(0, 398, 2)], dtype=np.int64)
    # claim 40 cross-fold pairs are the same product
    pairs = [(f"bc{i:05d}", f"bc{399 - i:05d}") for i in range(0, 40)]
    census = pd.DataFrame(
        {
            "gtin1": [a for a, _ in pairs],
            "gtin2": [b for _, b in pairs],
            "true_label": [1] * len(pairs),
        }
    )

    def straddles(pos_arr: np.ndarray, graph_bc: np.ndarray) -> int:
        groups = component_folds(pos_arr, graph_bc, 4, 42)
        fold_of = {bc: k for k, g in enumerate(groups) for bc in g}
        return sum(
            1 for a, b in pairs if fold_of[a] != fold_of[b]
        )

    bare = straddles(train_pos, row_bc)
    assert bare > 0, (
        f"fixture no longer reproduces the leak ({bare} straddles) -- this "
        "test is now vacuous"
    )
    merged, graph_bc, stats = merged_component_graph(
        train_pos, row_bc, labeled_pairs=census
    )
    assert stats["edges_added"] == len(pairs)
    assert straddles(merged, graph_bc) == 0


def test_positive_endpoints_share_a_component() -> None:
    train_pos, row_bc, census = _fixture()
    merged, graph_bc, _ = merged_component_graph(
        train_pos, row_bc, labeled_pairs=census
    )
    comp = component_ids(merged, graph_bc)
    assert comp["bc00000"] == comp["bc00399"]


def test_negatives_are_never_unioned() -> None:
    """A negative is a similarity claim, not an identity claim."""
    row_bc = np.array([f"bc{i:05d}" for i in range(4)], dtype=object)
    census = pd.DataFrame(
        {"gtin1": ["bc00000"], "gtin2": ["bc00001"], "true_label": [0]}
    )
    edges, stats = labeled_positive_edges(row_bc, census)
    assert len(edges) == 0, "label-0 pair must not become a graph edge"
    assert stats["labeled_positives"] == 0


def test_normalize_gtin_bridges_spellings_without_inventing_collision() -> None:
    assert normalize_gtin("2000000944753") == normalize_gtin("02000000944753")
    assert normalize_gtin("2000000944753.0") == normalize_gtin("2000000944753")
    assert normalize_gtin("") == "" and normalize_gtin(None) == ""
    # no-digits must NOT collapse into one shared key
    assert normalize_gtin("n/a") == ""


# ── the emitted artifact ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def validation() -> pd.DataFrame:
    import os

    path = F["final_validation"]
    if not os.path.exists(path):
        pytest.skip(f"{path} not built; run -m src.training.build_final_validation")
    return pd.read_csv(path)


def test_validation_artifact_has_no_positive_leak(validation: pd.DataFrame) -> None:
    pos = validation[validation.true_label == 1]
    assert len(pos) > 0, "validation population has no positives at all"
    assert not pos.straddles_fold.any(), (
        f"{int(pos.straddles_fold.sum())} positives straddle a fold"
    )
    assert not pos.endpoint_in_train.any(), (
        f"{int(pos.endpoint_in_train.sum())} positives have an endpoint in train"
    )


def test_validation_artifact_positive_sides_agree(validation: pd.DataFrame) -> None:
    pos = validation[validation.true_label == 1]
    assert (pos.fold == pos.fold_2).all()
    assert (pos.component_id == pos.component_id_2).all()


def test_validation_artifact_only_holds_out_folds(validation: pd.DataFrame) -> None:
    """Validation is folds 2+3; a train-fold row would be scored on training data."""
    assert validation.fold.max() <= 3
    assert validation.fold.min() >= 0


def test_validation_artifact_slices_frozen(validation: pd.DataFrame) -> None:
    """Both sides of every positive carry a value, for every gate field.

    Deliberately asserts POPULATION, not value equality. 9 of the 564
    positives have a different `volume_set` on the two endpoints
    (`[518.0]` vs `[479.0, 518.0]`) -- same product, one side's text
    mentions an extra volume. That is legitimate extractor variance, and
    asserting equality here would fail on the data rather than on a defect.
    """
    fields = ("volume", "pack", "sweetener", "flavor", "package_type",
              "carbonation")
    for field in fields:
        assert f"v1_{field}" in validation.columns
        assert f"v2_{field}" in validation.columns
    pos = validation[validation.true_label == 1]
    for field in fields:
        assert (pos[f"v1_{field}"] != "").all(), f"{field}: v1 unpopulated"
        assert (pos[f"v2_{field}"] != "").all(), f"{field}: v2 unpopulated"


def test_negative_free_census_keeps_the_evidence_contract_shape() -> None:
    """A census with no negatives must still produce the full evidence shape.

    The 2026-10-08 shape fix (policy -> half -> census) stops
    ``_scored_contract`` raising KeyError on a negative-free census; this pins
    that branch, which has no positives-only sibling elsewhere.
    """
    from training.build_final_validation import SLICE_FIELDS, negative_policy_evidence

    columns: dict[str, list] = {
        "true_label": [1, 1],
        "fold": [2, 3],
        "fold_2": [2, 3],
    }
    for name, _col in SLICE_FIELDS:
        columns[f"v1_{name}"] = ["", ""]
        columns[f"v2_{name}"] = ["", ""]
    frame = pd.DataFrame(columns)

    evidence = negative_policy_evidence(frame, min_test_negatives=1, n_folds=4)
    zero_cells = {name: 0 for name, _col in SLICE_FIELDS}
    for policy in ("withhold_straddle", "train_side"):
        assert evidence[policy]["scored_dev_negatives"] == 0
        assert evidence[policy]["scored_test_negatives"] == 0
        assert evidence[policy]["scored_negatives_with_trained_on_endpoint"] == 0
        for half in ("dev", "test"):
            assert evidence[policy]["populated_cells"][half] == zero_cells
            assert evidence[policy]["thin_cells"][half] == zero_cells


# ── the consumer must not fork the split a second time ──────────────────────

def test_evaluate_models_dev_test_selectors_are_the_p0_quarters() -> None:
    """evaluate_models selects the P0 fold roles, never the legacy 2-fold knobs.

    The scored half must name the same quarters ``build_final_validation``
    wrote, or it silently contains gtins the model trained on -- the P0 leak's
    shape, which raised nothing. So this pins the DERIVATION, not a comment:
    ``split.holdout_component_folds`` (schema-pinned Literal[4]) gives
    train=0 / dev=n-2 / test=n-1, and evaluate_models' own module-level
    expressions must evaluate to exactly those roles.

    ``evaluation.component_split_k/dev_fold/test_fold`` are the LEGACY 2-fold
    protocol (committed 2/0/1) and are deliberately NOT the selectors; a
    consumer that read them by subscript would score a trained-on quarter, so
    no such subscript may reappear in the file.
    """
    import ast
    from pathlib import Path

    from core.common import training_cfg
    from training.build_final_validation import _resolve_quarter_folds

    cfg = training_cfg()
    n_folds = int(cfg.split.holdout_component_folds)
    roles = _resolve_quarter_folds({"train"}, {"dev"}, {"test"}, n_folds)
    assert (roles["train"], roles["dev"], roles["test"]) == (0, 2, 3)
    assert (n_folds - 2, n_folds - 1) == (roles["dev"], roles["test"])
    # the legacy knobs still describe the retired 2-fold protocol
    assert (cfg.evaluation.dev_fold, cfg.evaluation.test_fold) == (0, 1)

    source = Path(__file__).parents[1] / "src" / "training" / "evaluate_models.py"
    tree = ast.parse(source.read_text())
    derived: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            derived[node.targets[0].id] = ast.unparse(node.value)
    env: dict = {"int": int, "_split": cfg.split}
    for name in ("_N_FOLDS", "_DEV_FOLD", "_TEST_FOLD"):
        exec(f"{name} = {derived[name]}", env)  # noqa: S102 - pinned source
    assert (env["_DEV_FOLD"], env["_TEST_FOLD"]) == (roles["dev"], roles["test"])
    assert env["_N_FOLDS"] == n_folds

    legacy = {"component_split_k", "dev_fold", "test_fold"}
    re_inlined = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value in legacy
    ]
    assert not re_inlined, (
        "evaluate_models subscripted a legacy evaluation fold knob: "
        f"{[ast.unparse(n) for n in re_inlined]}"
    )
