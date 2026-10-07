"""Colab retention publishing: local HPO report generation + snapshots.

Split from cli/colab.py (capability module, phase 1). The retention lane
itself lives in model_tracks.run_retention (publish/replace/smoke) — this
module owns cli.colab's publishing surface. Collaborators owned by cli.colab
are resolved late through ``cli.colab`` so monkeypatch surfaces are unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _colab():
    """The RUNNING cli.colab module (never a second import copy)."""
    return sys.modules["__colab_runtime_self__"]


def publish_local_hpo_results(run_id: str, persistence: str) -> None:
    """Generate HPO reports and persist snapshots before VM teardown."""
    from training.generate_training_report import generate_report

    generation = _colab().TRAINING_RESULTS / "hpo_runs" / run_id
    if not generation.is_dir():
        raise FileNotFoundError(f"local HPO archive missing: {generation}")
    for model_dir in sorted((generation / "models").iterdir()):
        if not model_dir.is_dir():
            continue
        metrics = sorted(model_dir.glob("*_holdout_*_fold_metrics.csv"))
        pairs = sorted(model_dir.glob("*_fold*_pairs.csv"))
        if metrics and pairs:
            pointer = model_dir / _colab().F["results_pointer"].name
            pointer_data = json.loads(pointer.read_text(encoding="utf-8")) if pointer.is_file() else {}
            report_tag = str(pointer_data.get("run_tag") or model_dir.name)
            generate_report(
                metrics[-1], pairs, model_dir / f"report_{report_tag}",
                sorted(model_dir.glob("*_fold*_train_scores.csv")),
                sorted(model_dir.glob("*_fold*_random_easy_scores.csv")),
            )
            print(_colab()._stamp(), f"[report-local] HPO {model_dir.name}: report generated", flush=True)
    print(_colab()._stamp(), "[report-local] HPO results retained locally", flush=True)
