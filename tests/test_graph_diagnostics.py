import pandas as pd

from core.graph_diagnostics import candidate_graph_diagnostics


def test_different_gtin_edges_are_visible_in_graph_diagnostics():
    frame = pd.DataFrame(
        [
            {
                "SKU_ID": "sku-1",
                "candidate_gtin": "1111111111111",
                "score": 0.80,
                "gtin_status": "different",
                "exact_gtin": 0,
                "rule_ok": 1,
            }
        ]
    )
    result = candidate_graph_diagnostics(frame, threshold=0.60)
    assert result["diagnostic_edge_count"] == 1
    assert result["diagnostic_component_count"] == 1
    assert result["plausible_group_count"] == 1
