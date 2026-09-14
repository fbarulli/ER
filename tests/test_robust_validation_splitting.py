import pandas as pd
import pytest

from training.robust_validation import _strict_fold_assignment


def _pairs(components: int, negatives_per_component: int) -> pd.DataFrame:
    rows = []
    for component in range(components):
        left, right = f"sku-{component}-a", f"sku-{component}-b"
        brand = f"brand-{component % 2}"
        category = f"category-{component % 3}"
        common = {
            "sku_id_a": left,
            "sku_id_b": right,
            "brand_a": brand,
            "brand_b": brand,
            "category_a": category,
            "category_b": category,
            "score": 0.9,
        }
        rows.append({**common, "label": 1})
        rows.extend(
            {**common, "label": 0, "score": 0.1} for _ in range(negatives_per_component)
        )
    return pd.DataFrame(rows)


def _assignment(frame: pd.DataFrame, key_to_fold: dict[str, int]) -> list[int | None]:
    values = []
    for _, row in frame.iterrows():
        folds = {
            key_to_fold[f"sku_id:{row[column]}"] for column in ("sku_id_a", "sku_id_b")
        }
        values.append(next(iter(folds)) if len(folds) == 1 else None)
    return values


def test_strict_group_split_keeps_components_and_negative_support() -> None:
    frame = _pairs(components=10, negatives_per_component=5)
    key_to_fold, _, _ = _strict_fold_assignment(
        frame, 5, 42, min_test_negatives=5, max_attempts=20
    )
    assignment = _assignment(frame, key_to_fold)
    for fold in range(5):
        test = frame.loc[[value == fold for value in assignment]]
        assert int((test["label"] == 0).sum()) >= 5
        assert int((test["label"] == 1).sum()) >= 1
    # Both endpoints of every positive pair stay in exactly one fold.
    for _, row in frame.loc[frame["label"].eq(1)].iterrows():
        assert (
            key_to_fold[f"sku_id:{row.sku_id_a}"]
            == key_to_fold[f"sku_id:{row.sku_id_b}"]
        )


def test_strict_group_split_rejects_insufficient_negative_population() -> None:
    with pytest.raises(ValueError, match="negative test support"):
        _strict_fold_assignment(
            _pairs(components=5, negatives_per_component=4),
            5,
            42,
            min_test_negatives=5,
            max_attempts=1,
        )
