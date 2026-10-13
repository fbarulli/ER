"""Uploading into a VM directory the checkout does not have yet.

The Colab contents API answers 500 (never 404) when an upload's parent
directory is missing, so the transport must establish the target directory
itself. The fake CLI below reproduces exactly that rule.
"""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cli import colab


class UploadRemoteDirectoryTest(unittest.TestCase):
    def test_upload_creates_the_remote_directory_it_targets(self):
        created: set[str] = {"/content"}
        uploaded: list[str] = []

        def exec_stream(_session, script, timeout, **kwargs):  # noqa: ARG001
            if "mkdir(parents=True, exist_ok=True)" not in script:
                return
            target = script.split("pathlib.Path(", 1)[1].split(")", 1)[0]
            created.add(target.strip("'\""))

        def fake_colab(*args, **kwargs):  # noqa: ARG001
            command, _flag, _session, _source, remote = args[:5]
            assert command == "upload"
            if str(Path(remote).parent) not in created:
                # The API's real answer for a missing parent, verbatim.
                raise subprocess.CalledProcessError(1, ["colab", *args])
            uploaded.append(remote)

        target = "/content/EuromonitoR/results/model_tracks/inputs/1234.tar.zst"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "1234.tar.zst"
            source.write_bytes(b"payload")
            with mock.patch.object(colab, "run_colab_exec_stream", exec_stream), \
                 mock.patch.object(colab, "colab", fake_colab):
                colab._upload_with_retries(source, target, timeout=60)

        self.assertEqual(uploaded, [target])
        self.assertIn("/content/EuromonitoR/results/model_tracks/inputs", created)


if __name__ == "__main__":
    unittest.main()
