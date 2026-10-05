"""The pre-training data gate's trust contract.

The suite supervisor runs every data test once, then hands its three workers an
attestation over the suite configuration and the bytes of every declared input.
A worker that matches that attestation may skip re-verifying those immutable
inputs; anything else must still run the tests itself.

These tests pin the boundaries that make that safe:

* no attestation, or one that does not match, means every check stays live;
* ER_DATA_GATE_ENFORCE=1 ignores trust completely;
* the exported GPU-pending marker is what lets a legitimately unbuilt GPU-only
  text cache digest consistently between supervisor and workers;
* without the suite configuration a worker cannot recompute the digest at all,
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
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


class TrustContract(unittest.TestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for key in (
            data_gate.ATTESTATION_ENV,
            data_gate.GPU_PENDING_ENV,
            data_gate.FORCE_ENV,
            data_gate.CONFIG_ENV,
        ):
            os.environ.pop(key, None)
        self._cache = mock.patch.object(data_gate, "_enforced", new=None)
        self._cache.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(self._cache.stop)

    def _trust(self, digest: str = DIGEST_A, **env: str) -> bool:
        os.environ.update(env)
        with mock.patch.object(data_gate, "attestation", return_value=digest):
            return data_gate.enforced(CONFIG)

    def test_no_attestation_forces_verification(self) -> None:
        self.assertTrue(self._trust())

    def test_matching_attestation_lets_the_worker_skip(self) -> None:
        self.assertFalse(self._trust(DIGEST_A, **{data_gate.ATTESTATION_ENV: DIGEST_A}))

    def test_stale_attestation_forces_verification(self) -> None:
        # The digest the worker recomputes moved: the bytes or the configuration
        # changed after the gate ran, so the worker must test the data itself.
        self.assertTrue(self._trust(DIGEST_B, **{data_gate.ATTESTATION_ENV: DIGEST_A}))

    def test_force_env_overrides_a_matching_attestation(self) -> None:
        self.assertTrue(
            self._trust(
                DIGEST_A,
                **{
                    data_gate.ATTESTATION_ENV: DIGEST_A,
                    data_gate.FORCE_ENV: "1",
                },
            )
        )

    def test_trusted_is_false_when_enforcement_stays_on(self) -> None:
        with mock.patch.object(data_gate, "attestation", return_value=DIGEST_A):
            self.assertFalse(data_gate.trusted(CONFIG, "text bundle"))

    def test_gpu_pending_env_reaches_the_digest(self) -> None:
        os.environ[data_gate.ATTESTATION_ENV] = DIGEST_A
        # The supervisor exported the pending marker for a text cache that its own
        # baseline export has not written yet. The worker must be allowed to
        # compute the same digest, or an honest worker would distrust the gate.
        with mock.patch.object(
            data_gate, "attestation", return_value=DIGEST_A
        ) as attestation:
            data_gate.enforced(CONFIG)
        self.assertFalse(attestation.call_args.kwargs["allow_gpu_pending"])

        with mock.patch.object(
            data_gate, "attestation", return_value=DIGEST_A
        ) as attestation:
            os.environ[data_gate.GPU_PENDING_ENV] = "1"
            data_gate._enforced = None
            data_gate.enforced(CONFIG)
        self.assertTrue(attestation.call_args.kwargs["allow_gpu_pending"])

    def test_config_absent_cannot_trust(self) -> None:
        # A worker that cannot recompute the digest has no proof at all.
        self.assertIsNone(data_gate.suite_config())
        self.assertFalse(data_gate._owner_trusted("text bundle"))

    def test_config_is_read_from_the_environment(self) -> None:
        os.environ[data_gate.CONFIG_ENV] = str(CONFIG.resolve())
        self.assertIsNotNone(data_gate.suite_config())


if __name__ == "__main__":
    unittest.main()
