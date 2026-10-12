"""Sources over the contents-API request ceiling ship as reassembled parts.

Public behavior pinned: ``_upload_with_retries`` never hands the CLI an upload
whose payload exceeds ``ColabSpec.remote_upload_part_bytes`` — it uploads
numbered parts beside the target and assembles them VM-side in order, then
leaves no local part files behind. The fake CLI below reproduces the proven
rule: a request whose local payload is over the cap answers 500 (the
2026-10-12 full-launch failure), parts within it succeed.

No VM: the CLI call and the exec stream are mocked on the legacy surface.
"""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cli import colab


class UploadPartsTest(unittest.TestCase):
    def test_oversized_source_uploads_as_parts_and_assembles_on_the_vm(self):
        cap = 40
        uploaded: list[tuple[str, int]] = []
        execs: list[str] = []

        def exec_stream(_session, script, timeout, **kwargs):
            if "mkdir(parents=True" in script:
                return
            execs.append(script)

        def fake_colab(*args, **kwargs):
            command, _flag, _session, local, remote = args[:5]
            assert command == "upload"
            size = Path(local).stat().st_size
            if size > cap:
                raise subprocess.CalledProcessError(1, ["colab", *args])
            uploaded.append((remote, size))

        target = "/content/EuromonitoR/results/model_tracks/inputs/big.tar.zst"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "big.tar.zst"
            source.write_bytes(b"x" * 100)
            with mock.patch.object(colab, "run_colab_exec_stream", exec_stream), \
                 mock.patch.object(colab, "colab", fake_colab), \
                 mock.patch.object(colab._COLAB, "remote_upload_part_bytes", cap):
                colab._upload_with_retries(source, target, timeout=60)

            self.assertEqual(
                uploaded,
                [(f"{target}.part00{n}", size)
                 for n, size in enumerate((40, 40, 20), start=1)],
            )
            assemble = execs[-1]
            for n in (1, 2, 3):
                self.assertIn(f"{target}.part00{n}", assemble)
            self.assertIn("os.replace", assemble)
            self.assertIn(repr(target), assemble)
            self.assertEqual(list(source.parent.glob("big.tar.zst.part*")), [])


if __name__ == "__main__":
    unittest.main()
