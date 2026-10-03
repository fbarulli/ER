import pickle

import pandas as pd

from training.prepared_bundle import _portable_dataframe


def test_prepared_dataframe_preserves_values_without_string_dtype_metadata():
    source = pd.DataFrame({'sku_name_eng': pd.array(['a', None], dtype='string'), 'count': [1, 2]})
    source.columns = pd.Index(source.columns, dtype='string')
    source.index = pd.Index(['first', 'second'], dtype='string')
    restored = pickle.loads(pickle.dumps(_portable_dataframe(source)))
    assert all(dtype == object for dtype in restored.dtypes)
    assert restored.columns.dtype == object
    assert restored.index.dtype == object
    pd.testing.assert_frame_equal(restored, source.astype(object).rename_axis(None), check_dtype=False, check_index_type=False, check_column_type=False)
    assert isinstance(source['sku_name_eng'].dtype, pd.StringDtype)


def test_native_canonical_layout_rejects_projected_reference_block(monkeypatch):
    import numpy as np
    import pytest
    import pipeline
    from training.prepared_bundle import canonical_payload_rows

    monkeypatch.setattr(pipeline, 'load_canonical_map', lambda: {'a': 'A', 'b': 'B'})
    assert canonical_payload_rows(1, ['source', 'A', 'B', 'copy'], np.array(['a', 'a', 'b', 'a'])).tolist() == [1, 2]
    with pytest.raises(ValueError, match='exceeds the payload'):
        canonical_payload_rows(1, ['source', 'A'], np.array(['a', 'a']))
    with pytest.raises(ValueError, match='canonical map'):
        canonical_payload_rows(1, ['source', 'A', 'copy'], np.array(['a', 'a', 'a']))


def test_calibration_negatives_exclude_copies_but_keep_real_canonical_targets():
    import numpy as np
    from training.training import _evaluation_negative_mask

    pairs = np.array([[0, 5], [7, 5], [1, 8], [5, 0], [1, 2]])
    before = pairs.copy()
    mask = _evaluation_negative_mask(pairs, 3, {7, 8})
    assert mask.tolist() == [True, False, False, False, True]
    np.testing.assert_array_equal(pairs, before)
