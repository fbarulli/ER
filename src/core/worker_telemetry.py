"""Shared live-status contract for remote workers."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


def write_worker_live_status(
    *,
    target: Path,
    event: str,
    step: int,
    max_steps: int,
    epoch: float,
    wandb_run_id: str | None = None,
    wandb_url: str | None = None,
    **values: object,
) -> None:
    """Atomically publish the common launcher heartbeat payload."""
    payload = {
        "updated_at": time.time(),
        "event": event,
        "step": int(step),
        "max_steps": int(max_steps),
        "epoch": float(epoch),
        "wandb_run_id": wandb_run_id,
        "wandb_url": wandb_url,
        **{key: value for key, value in values.items() if value is not None},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    ) as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)
