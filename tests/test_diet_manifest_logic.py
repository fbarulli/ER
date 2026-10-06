import pytest
from scripts.diet_manifest import mnrl_presentation_counts

def test_mnrl_presentation_counts_classic():
    # Classic bundle: audit rows use (anchor, pair, copy) that match triples
    data = {
        "hard_negative_mask_audit": [
            {"population": "hard_negative", "target_mode": "counterfactual",
             "anchor_payload_idx": 10, "pair_payload_idx": 20, "copy_payload_idx": 100},
            {"population": "hard_negative", "target_mode": "random",
             "anchor_payload_idx": 11, "pair_payload_idx": 21, "copy_payload_idx": 101},
        ],
        "training_plan": {"identity": {"loss": "mnrl"}, "inputs": {"folds": [
            {"fold_i": 0, "objective": {
                "triples": [
                    [10, 20, 100],  # twin: (source, positive, copy)
                    [101, 30, 21],  # hard-negative copy as ANCHOR: (copy, source_positive, pair)
                    [12, 40, 50],   # base negative
                ],
                # Parallel per-triple rows, exactly as training.py builds them.
                "dataset": {"anchor": [10,101,12], "positive":[20,30,40],
                            "negative":[100,21,50]}
            }}
        ]}}
    }
    # No "augmentation_coverage" key -> classic mode
    results = mnrl_presentation_counts(data)
    assert results[0]["negative_augmented"] == 2

def test_mnrl_presentation_counts_balanced():
    # Balanced bundle: audit row anchor/pair indices are in a different space,
    # so only the copy index matches the triple.
    data = {
        "augmentation_coverage": {}, # marker for balanced lane
        "hard_negative_mask_audit": [
            {"population": "hard_negative", "target_mode": "counterfactual",
             "anchor_payload_idx": 999, "pair_payload_idx": 888, "copy_payload_idx": 100},
        ],
        "training_plan": {"identity": {"loss": "mnrl"}, "inputs": {"folds": [
            {"fold_i": 0, "objective": {
                "triples": [
                    [10, 20, 100],  # matches twin by copy only
                    [11, 30, 50],   # base negative
                ],
                "dataset": {"anchor": [1, 2], "positive": [1, 2], "negative": [1, 2]}
            }}
        ]}}
    }
    results = mnrl_presentation_counts(data)
    # Without the fix, this would be 0. With the fix, it is 1.
    assert results[0]["negative_augmented"] == 1
