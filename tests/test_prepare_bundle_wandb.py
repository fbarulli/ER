"""``training.train --prepare-bundle`` must not require W&B.

Bundle preparation materializes the suite's training inputs and runs no
experiment, so WANDB_API_KEY must never be a precondition of that lane. Real
training/HPO lanes keep the always-on enforcement.
"""
from __future__ import annotations

import sys

import pytest

from training import train as train_mod


def test_prepare_bundle_entry_runs_without_wandb_key(monkeypatch, tmp_path):
    from core.wandb_ctx import WandbCtx

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(train_mod, "training_trace", lambda *args, **kwargs: None)
    seen: list[object] = []
    monkeypatch.setattr(train_mod, "_main_inner", lambda wandb: seen.append(wandb))

    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "--prepare-bundle", str(tmp_path / "bundle.pkl.gz")],
    )
    train_mod.main()  # must not raise with no WANDB_API_KEY
    assert len(seen) == 1
    assert isinstance(seen[0], WandbCtx)

    # the real training lane still fails loud without a key
    monkeypatch.setattr(sys, "argv", ["train.py"])
    with pytest.raises(RuntimeError, match="WANDB_API_KEY absent"):
        train_mod.main()
    assert len(seen) == 1
