"""Dead-code guard for Optuna callback wiring (audit B6).

Defect: ``_optuna_mlflow_cb`` (training.py) was defined but never registered
at any ``study.optimize`` call site. It was superseded by
``_optuna_tracking_cb`` (same commit, superset telemetry: MLflow metric with
the correct objective name + wandb params) but never deleted, so it sat as
dead code whose only contribution had it been registered would have been a
misleadingly-named ``trial_N_auc`` metric on a non-AUC objective
(discriminative-LR calibration Rand).

The guard: every ``_optuna_*_cb`` helper defined in training.py must be
referenced at a ``callbacks=[...]`` registration. A defined-but-unregistered
callback fails this test, so the dead-code decision (register-or-delete)
cannot silently regress.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PY = REPO_ROOT / "src" / "training" / "training.py"

CALLBACK_DEF = re.compile(r"^def (_optuna_\w*)\(", re.MULTILINE)


class OptunaCallbackRegistrationTest(unittest.TestCase):
    def test_every_optuna_callback_is_registered(self):
        source = TRAINING_PY.read_text(encoding="utf-8")
        defined = CALLBACK_DEF.findall(source)
        self.assertTrue(defined, "expected at least one _optuna_*_cb definition")
        unregistered = [
            name
            for name in defined
            if len(re.findall(rf"\b{re.escape(name)}\b", source)) < 2
        ]
        self.assertEqual(
            unregistered,
            [],
            "defined but never registered at study.optimize (dead code): "
            + ", ".join(unregistered),
        )


if __name__ == "__main__":
    unittest.main()
