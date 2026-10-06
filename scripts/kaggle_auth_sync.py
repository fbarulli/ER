"""Sync Kaggle CLI 2.x OAuth credentials into the files kagglehub reads.

kaggle 2.2.4 (Kaggle/kaggle-cli) stores OAuth tokens in ~/.kaggle/credentials.json
with a refresh token, while kagglehub only reads KAGGLE_API_TOKEN or a plain-text
~/.kaggle/access_token. This bridge refreshes via the CLI and writes that file.

Usage:
    python scripts/kaggle_auth_sync.py          # sync and validate
    python scripts/kaggle_auth_sync.py --quiet  # no output
"""

import subprocess
import sys
from pathlib import Path

CREDENTIALS = Path.home() / ".kaggle" / "credentials.json"
ACCESS_TOKEN = Path.home() / ".kaggle" / "access_token"


def main() -> int:
    quiet = "--quiet" in sys.argv
    if not CREDENTIALS.exists():
        print(
            "No OAuth credentials at ~/.kaggle/credentials.json. "
            "Run `kaggle auth login` once (browser flow).",
            file=sys.stderr,
        )
        return 1
    token = subprocess.run(
        ["kaggle", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not token:
        print("kaggle auth print-access-token returned an empty token.", file=sys.stderr)
        return 1
    ACCESS_TOKEN.write_text(token)
    ACCESS_TOKEN.chmod(0o600)

    import kagglehub

    who = kagglehub.whoami(verbose=False)
    if not quiet:
        print(f"Kaggle auth synced; authenticated as {who['username']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
