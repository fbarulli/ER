import numpy as np
import pytest

from core.common import training_cfg
from core.structured_features import vector
from training.masking import extend_augmented_features
from training.prepared_bundle import _validate_augmented_features


def test_interleaved_symmetric_copies_follow_explicit_payload_indices():
    features = np.arange(40, dtype=np.float32).reshape(4, 10)
    payload = ["base"] * 4 + ["flavor_lime"] * 4
    audit = [
        dict(anchor_payload_idx=2, pair_payload_idx=3, copy_payload_idx=6,
             copy_pair_payload_idx=7, target_mode="swap_values", fields_hit=["flavor"]),
        dict(anchor_payload_idx=0, pair_payload_idx=1, copy_payload_idx=4,
             copy_pair_payload_idx=5, target_mode="swap_values", fields_hit=["flavor"]),
    ]
    actual = extend_augmented_features(features, payload, audit)
    np.testing.assert_array_equal(actual[4:], features)
    _validate_augmented_features(payload, actual, audit)
    corrupted = actual.copy()
    corrupted[[5, 6]] = corrupted[[6, 5]]
    with pytest.raises(ValueError, match="augmentation features"):
        _validate_augmented_features(payload, corrupted, audit)


@pytest.mark.parametrize("field,token", [("volume", "volume_ml_330"), ("pack", "pack_qty_6")])
def test_numeric_twins_update_changed_field_and_preserve_other_field(field, token):
    cfg = training_cfg().training.structured_features
    kwargs = dict(volume_scale_ml=cfg.volume_scale_ml, pack_scale=cfg.pack_scale,
                  max_set_size=cfg.max_set_size)
    original = {"volume": {500}, "pack": {2}}
    features = np.array([vector(original, **kwargs)], dtype=np.float32)
    changed = dict(original)
    changed[field] = {330 if field == "volume" else 6}
    audit = [dict(anchor_payload_idx=0, copy_payload_idx=1,
                  target_mode="counterfactual", fields_hit=[field])]
    actual = extend_augmented_features(features, ["original", token], audit)
    np.testing.assert_allclose(actual[1], vector(changed, **kwargs))
    np.testing.assert_array_equal(actual[0], features[0])


def test_masks_preserve_features_and_missing_lineage_fails():
    features = np.ones((1, 10), dtype=np.float32)
    audit = [dict(anchor_payload_idx=0, copy_payload_idx=1, target_mode="random")]
    result = extend_augmented_features(features, ["volume_ml_500", "[MASK]"], audit)
    np.testing.assert_array_equal(result[1], features[0])
    with pytest.raises(ValueError, match="cover the payload suffix"):
        extend_augmented_features(features, ["base", "copy", "untracked"], audit)
