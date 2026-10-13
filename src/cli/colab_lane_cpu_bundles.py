"""CPU lane training-bundle stage (phase owner + the VM entry point).

Split phase of cli/colab_lane.py (the kaggle_lane.py owner-class pattern).  The
CPU bundle lane builds the prepared TRAINING bundles the GPU lane trains from —
on the Colab CPU VM, never on the operator's machine — by running
``training.train --prepare-bundle`` once per declared
``(worker, masking profile, model)`` under ``config colab.prepared_bundles``.

It is a STAGE of the lane's existing prepare run, not a second spawn path:
:mod:`cli.colab_lane_cpu_provision` appends :meth:`bundle_stage_command` to the
same detached prepare process that runs ``training.prepare_all``, so the stage
inherits the log file, the status file, the ``timeout`` budget and the
transcript poll that lane already owns.  The bundles land in the declared
``data/prepared/<dir>`` tree, which ``DELIVERY_PREPARED_DIRS`` already ships as
one directory — the delivery needs no change.

What this stage does NOT do: worker 1's bundle.  ``training.prepare_all``'s
``full_bundle`` stage writes it, syncs the text bundle from it and verifies the
handoff against it (``ColabPreparedBundlesSpec.stage_entries`` is the one
statement of that division).  ``--mask-effect`` is left at its default because
the stage exits before the post-training audit can run, and ``--payload`` /
``--collapse-guardrail-profile`` are left at the defaults ``training.train``
reads from the same config, so the stage spells no value the trainer owns.

Run on the VM as ``python -u -m cli.colab_lane_cpu_bundles``; :func:`main` is
that tiny entry point.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from core.common import TRAIN_ROOT, training_cfg
from core.schemas import ColabPreparedBundleSpec
from training.prepare_all_trace import timed

#: ``scripts/diet_manifest.py``'s own pass code. The gate is the same one
#: ``model_tracks.preflight`` runs during the CPU lane's ``suite_inputs``
#: stage, invoked here as the tracked script it is (its exit codes are its own
#: vocabulary, not a config value).
DIET_GATE_PASS_RC = 0


class ColabCPULaneTrainingBundles:
    """The CPU lane's training-bundle stage: the declared bundles, on the VM."""

    def bundle_stage_command(self) -> list[str]:
        """The command that runs THIS stage inside the lane's prepare process.

        ``-m`` on the module under the VM's checkout ``src/``, so the stage is
        the code the repository ships rather than text a launcher interpolated
        into a script.
        """
        return [sys.executable, "-u", "-m", __name__]

    @timed
    def run(self, root: Path | None = None) -> None:
        """Build every declared bundle this stage owns, in declaration order."""
        checkout = Path(root) if root is not None else Path(TRAIN_ROOT)
        entries = training_cfg().colab.prepared_bundles.stage_entries()
        print(
            f"[bundle-stage] building {len(entries)} declared training bundle(s) "
            f"in {checkout}",
            flush=True,
        )
        for entry in entries:
            self._build(checkout, entry)

    @timed
    def _build(self, checkout: Path, entry: ColabPreparedBundleSpec) -> None:
        """One bundle: build it, then refuse to continue unless it passes the diet gate."""
        bundle = checkout / training_cfg().colab.prepared_bundles.path_of(entry)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[bundle-stage] worker={entry.worker} profile={entry.masking_profile} "
            f"model={entry.model} -> {bundle}",
            flush=True,
        )
        # check=True: a failed build raises here and the stage dies, so the
        # lane's status file carries a non-zero rc and the poll refuses the
        # whole delivery. The child's own full traceback is in the prepare log.
        subprocess.run(self.bundle_command(checkout, entry), cwd=checkout,
                       env=self._stage_env(), check=True)
        self._require_diet_pass(checkout, bundle)

    @staticmethod
    def bundle_command(checkout: Path, entry: ColabPreparedBundleSpec) -> list[str]:
        """The ONE ``training.train --prepare-bundle`` invocation for one entry.

        The VM-side command is this stage's contract with ``training.train``:
        the declared entry's model, masking profile and output path, plus the
        source CSV and the two switches that keep a bundle build from writing
        reports. Everything else ``training.train`` offers (payload variant,
        collapse-guardrail profile, mask effect) is left at the default it reads
        from the same config, so the stage spells no value the trainer owns.
        """
        return [
            sys.executable, "-u", "-m", "training.train",
            "--model", entry.model,
            "--dataset", str(checkout / training_cfg().colab.training_dataset_csv),
            "--masking-profile", entry.masking_profile,
            "--prepare-bundle", str(checkout / training_cfg().colab.prepared_bundles.path_of(entry)),
            "--no-mask-effect",
            "--no-plot",
        ]

    @staticmethod
    def _stage_env() -> dict[str, str]:
        """The prepare process's environment (its PYTHONPATH already resolves ``src``)."""
        return {**os.environ, "PYTHONUNBUFFERED": "1"}

    @timed
    def _require_diet_pass(self, checkout: Path, bundle: Path) -> None:
        """Fail loud unless this bundle passes the bundle diet contract.

        The same gate ``model_tracks.preflight`` runs on the text bundle during
        the CPU lane's ``suite_inputs`` stage, executed as the tracked script it
        is. The gate's own output streams into the prepare log; this raises with
        the failing rc so the lane's status file is non-zero and the delivery is
        refused rather than shipping inputs that violate the diet contract.
        """
        gate = checkout / "scripts" / "diet_manifest.py"
        result = subprocess.run(
            [sys.executable, str(gate), str(bundle)], cwd=checkout,
            env=self._stage_env(), check=False)
        if result.returncode != DIET_GATE_PASS_RC:
            raise RuntimeError(
                f"[bundle-stage] bundle {bundle} FAILED the diet gate (rc="
                f"{result.returncode}); the CPU lane refuses to deliver training "
                "inputs that violate config/training.yaml's diet contract"
            )
        print(f"[bundle-stage] diet gate passed for {bundle}", flush=True)


def main() -> None:
    """The VM entry point: build this checkout's declared training bundles."""
    ColabCPULaneTrainingBundles().run()