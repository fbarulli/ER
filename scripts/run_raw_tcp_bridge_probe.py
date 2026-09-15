#!/usr/bin/env python3
"""Run the gated raw-TCP Colab bridge probe without printing secrets."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Paths come from the shared contract (core.common -> config/paths.yaml), never
# from a __file__/__parents__ offset: a magic parent count breaks silently when
# the script moves. The repo's src/ is put on the path from the same contract.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from core.common import TRAIN_ROOT  # noqa: E402

SESSION = "euromonitor-hpo-cpu-test"
REMOTE_SCRIPT = "/tmp/colab_tailscale_userspace.sh"


def _env() -> dict[str, str]:
    values = {}
    for line in (TRAIN_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value.strip().strip('"').strip("'")
    if not values.get("TAILSCALE_AUTH_KEY"):
        raise RuntimeError("TAILSCALE_AUTH_KEY is missing")
    return values


def _redacted(output: str, secrets: set[str]) -> str:
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            output = output.replace(secret, "<redacted>")
    return output


def main() -> int:
    values = _env()
    secrets = {values["TAILSCALE_AUTH_KEY"]}
    subprocess.run(
        ["colab", "upload", "-s", SESSION,
         str(TRAIN_ROOT / "scripts/colab_tailscale_userspace.sh"), REMOTE_SCRIPT],
        check=True,
    )
    bridge_cell = (
        "import os, subprocess\n"
        f"env={{**os.environ, 'TAILSCALE_AUTH_KEY': {values['TAILSCALE_AUTH_KEY']!r}, "
        "'TAILSCALE_TARGET_HOST': '100.91.130.10', "
        "'TAILSCALE_TARGET_PORT': '10000', 'TAILSCALE_LOCAL_PORT': '10000'}\n"
        f"r=subprocess.run(['bash','{REMOTE_SCRIPT}'], env=env, capture_output=True, text=True)\n"
        "print(r.stdout, end=''); print(r.stderr, end=''); raise SystemExit(r.returncode)\n"
    )
    first = subprocess.run(
        ["colab", "exec", "-s", SESSION, "--timeout", "120"],
        input=bridge_cell, text=True, capture_output=True,
    )
    print(_redacted(first.stdout + first.stderr, secrets), end="")
    first_output = first.stdout + first.stderr
    if first.returncode != 0 or "[tailscale] bridge ready:" not in first_output:
        raise SystemExit("bridge setup failed; raw TCP send was not attempted")
    send_cell = (
        "import socket\n"
        "with socket.create_connection(('127.0.0.1', 10000), 10) as s:\n"
        "    s.sendall(b'hello\\n')\n"
        "print('[raw-tcp] client sent hello')\n"
    )
    second = subprocess.run(
        ["colab", "exec", "-s", SESSION, "--timeout", "60"],
        input=send_cell, text=True, capture_output=True,
    )
    print(_redacted(second.stdout + second.stderr, secrets), end="")
    return second.returncode


if __name__ == "__main__":
    raise SystemExit(main())
