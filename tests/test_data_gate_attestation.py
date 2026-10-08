"""The pre-training data gate's trust contract.

The suite supervisor runs every data test once, then hands its three workers an
attestation over the suite configuration and the sizes of every declared input.
A worker that matches that attestation may skip re-verifying those immutable
inputs; anything else must still run the tests itself.

These tests pin the boundaries that make that safe:

* no attestation, or one that does not match, means every check stays live;
* ER_DATA_GATE_ENFORCE=1 ignores trust completely;
* there is no pending/not-yet-produced input state at all (owner directive
  2026-10-08): a declared input that is absent fails the gate loudly;
* without the suite configuration a worker cannot recompute the total at all,
  so it must verify rather than trust.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_tracks import data_gate


CONFIG = Path("config/model_tracks.yaml")
TOKEN_A = 4194304
TOKEN_B = 8388608


class TrustContract(unittest.TestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for key in (
            data_gate.ATTESTATION_ENV,
            data_gate.FORCE_ENV,
            data_gate.CONFIG_ENV,
        ):
            os.environ.pop(key, None)
        self._cache = mock.patch.object(data_gate, "_enforced", new=None)
        self._cache.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(self._cache.stop)

    def _trust(self, token: int = TOKEN_A, *, attested: int | None = None, **env: str) -> bool:
        if attested is not None:
            os.environ[data_gate.ATTESTATION_ENV] = str(attested)
        os.environ.update(env)
        with mock.patch.object(data_gate, "attestation", return_value=token):
            return data_gate.enforced(CONFIG)

    def test_no_attestation_forces_verification(self) -> None:
        self.assertTrue(self._trust())

    def test_matching_attestation_lets_the_worker_skip(self) -> None:
        self.assertFalse(self._trust(TOKEN_A, attested=TOKEN_A))

    def test_stale_attestation_forces_verification(self) -> None:
        # The total the worker recomputes moved: the sizes or the configuration
        # changed after the gate ran, so the worker must test the data itself.
        self.assertTrue(self._trust(TOKEN_B, attested=TOKEN_A))

    def test_unreadable_attestation_forces_verification(self) -> None:
        # A corrupt value is not proof of anything: the gate stays fully live
        # instead of crashing the worker on a malformed environment value.
        os.environ[data_gate.ATTESTATION_ENV] = "not-a-size"
        with mock.patch.object(data_gate, "attestation", return_value=TOKEN_A):
            self.assertTrue(data_gate.enforced(CONFIG))

    def test_force_env_overrides_a_matching_attestation(self) -> None:
        self.assertTrue(
            self._trust(
                TOKEN_A,
                attested=TOKEN_A,
                **{data_gate.FORCE_ENV: "1"},
            )
        )

    def test_trusted_is_false_when_enforcement_stays_on(self) -> None:
        with mock.patch.object(data_gate, "attestation", return_value=TOKEN_A):
            self.assertFalse(data_gate.trusted(CONFIG, "text bundle"))

    def test_attestation_is_a_structural_total_not_a_hex_token(self) -> None:
        """The gate's proof is a byte-size total, never a fixed-width digest.

        Regression: the hash removal left ``attestation`` returning an int while
        its annotation, the pydantic result field and the ``enforced`` comparison
        still assumed a lowercase-hex string, so ``validate``/``enforced`` died on
        ``int.encode()`` before the gate could ever pass.
        """

        class Config:
            def model_dump(self, mode: str = "json"):
                return {"setup_dir": "setup", "text_bundle": "bundle"}

        with mock.patch.object(data_gate, "load_config", return_value=Config()), \
                mock.patch.object(data_gate, "input_sizes",
                                  return_value={"text_bundle": 4096, "setup_manifest": 128}):
            token = data_gate.attestation(CONFIG)
        self.assertIsInstance(token, int)
        self.assertFalse(isinstance(token, bool))
        self.assertGreater(token, 0)
        # The result model accepts the total as an integer (it used to demand a
        # 64-char hex string, which the int failed to validate against).
        result = data_gate.DataGateResult(suite={}, tracks={}, attestation=token)
        self.assertEqual(result.attestation, token)

    def test_no_pending_input_state_exists(self) -> None:
        """A declared input is required: the pending tolerance is gone.

        Owner directive 2026-10-08 removed every "not-yet-fresh" allowance, so
        the gate has no GPU-pending marker/env and no keyword that substitutes a
        sentinel size for an absent text cache.
        """
        self.assertFalse(hasattr(data_gate, "GPU_PENDING"))
        self.assertFalse(hasattr(data_gate, "GPU_PENDING_ENV"))
        self.assertFalse(hasattr(data_gate, "_gpu_pending"))
        import inspect

        for function in (data_gate.input_sizes, data_gate.attestation,
                         data_gate.enforced, data_gate.trusted, data_gate.validate):
            self.assertNotIn("allow_gpu_pending", inspect.signature(function).parameters)

    def test_config_absent_cannot_trust(self) -> None:
        # A worker that cannot recompute the total has no proof at all.
        self.assertIsNone(data_gate.suite_config())
        self.assertFalse(data_gate._owner_trusted("text bundle"))

    def test_config_is_read_from_the_environment(self) -> None:
        os.environ[data_gate.CONFIG_ENV] = str(CONFIG.resolve())
        self.assertIsNotNone(data_gate.suite_config())


if __name__ == "__main__":
    unittest.main()
