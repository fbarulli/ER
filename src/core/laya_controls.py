"""src/core/laya_controls.py — pure, GPU-free logic for the laya controls.

The fine-tune kernel runs from ATTACHED datasets (it never clones the repo), so
it cannot import this package: ``cli.laya_lane`` bakes these exact function
sources into the staged perf patch via ``inspect.getsource``. That keeps ONE
source of truth — the functions below are unit-tested here with plain Python
(no CUDA) and executed byte-for-byte inside the remote kernel.

Nothing here knows about torch at import time; ``build_lr_scheduler`` imports it
lazily so this module stays importable on a box without a GPU.
"""

from __future__ import annotations


def parse_control(raw, defaults):
    """Merge a baked control block over ``defaults`` (``None`` keeps default).

    The staged kernel bakes the FULL block (every key already present), so this
    is a defensive parse: an absent/partial block never raises, it just falls
    back to the workspace defaults.
    """
    control = dict(defaults or {})
    if isinstance(raw, dict):
        for key, value in raw.items():
            if value is not None:
                control[key] = value
    return control


def metric_is_better(value, best, min_delta, lower_is_better):
    """Whether ``value`` improves on ``best`` by at least ``min_delta``."""
    if best is None:
        return True
    if lower_is_better:
        return value < best - min_delta
    return value > best + min_delta


def early_stop_step(value, best, bad_epochs, patience, min_delta,
                    lower_is_better):
    """One patience step for early stopping.

    Returns ``(best, bad_epochs, improved, stop)``. An improvement resets the
    counter; otherwise the counter increments and triggers ``stop`` once it is
    >= ``patience``. ``patience=0`` stops on the first non-improving epoch.
    """
    if metric_is_better(value, best, min_delta, lower_is_better):
        return value, 0, True, False
    bad = int(bad_epochs) + 1
    return best, bad, False, bad >= int(patience)


def effective_warmup_steps(warmup_steps, warmup_frac, total_updates):
    """Explicit ``warmup_steps`` wins over ``warmup_frac`` of ``total_updates``."""
    total = max(1, int(total_updates))
    if warmup_steps and int(warmup_steps) > 0:
        return max(0, min(int(warmup_steps), total - 1))
    if warmup_frac and float(warmup_frac) > 0.0:
        return max(0, min(int(round(float(warmup_frac) * total)), total - 1))
    return 0


def build_lr_scheduler(optimizer, kind, total_updates, min_lr, warmup,
                       plateau_mode="max"):
    """Build the per-update scheduler from the YAML-driven control block.

    ``warmup == 0`` and ``kind == "cosine"`` reproduces the landed
    ``CosineAnnealingLR(optimizer, T_max=updates, eta_min=min_lr)`` exactly.
    ``plateau`` is stepped per EPOCH by the caller (on the dev metric), never
    per update.
    """
    import torch

    total = max(1, int(total_updates))
    kind = str(kind or "cosine").lower()
    warmup = int(warmup or 0)
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=plateau_mode, factor=0.5, patience=2,
            min_lr=min_lr)
    if kind == "onecycle":
        base = [float(group.get("lr", min_lr)) for group in optimizer.param_groups]
        pct = (float(warmup) / float(total)) if warmup > 0 else 0.3
        pct = min(max(pct, 1e-8), 0.9)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=base, total_steps=total, pct_start=pct)
    main_steps = max(1, total - warmup) if warmup > 0 else total
    if kind == "constant":
        main = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    elif kind == "linear":
        def _decay(step):
            if main_steps <= 1:
                return 1.0
            return max(0.0, 1.0 - float(step) / float(main_steps - 1))
        main = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay)
    else:  # cosine — the landed default
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_steps, eta_min=min_lr)
    if warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0,
            total_iters=warmup)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, main], milestones=[warmup])
    return main


def flatten_epoch_metrics(prefix, mapping):
    """``{"a": 1}`` -> ``{"prefix/a": 1}`` dropping ``None`` values."""
    flat = {}
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            if value is None:
                continue
            flat[str(prefix) + "/" + str(key)] = value
    return flat


def derive_abstain_coverage(records, confidence_threshold):
    """Best-effort ``(abstain_rate, coverage)`` from calibration records.

    ``laya.train.evaluate_records`` does not emit coverage/abstention keys, so
    the kernel derives them from the records it already has: a record abstains
    when its top-1 calibrated-by-temperature=1 probability falls below the
    YAML-configured confidence threshold. Returns ``(None, None)`` when the
    probability cannot be computed (e.g. an empty records list).
    """
    try:
        import numpy as np
    except Exception:
        return None, None
    total = abstained = 0
    for record in records or ():
        try:
            _qtype, logits, _target, k = record
            z = np.asarray(logits, dtype=float)[:int(k)]
            z = z - z.max()
            p = np.exp(z)
            p = p / p.sum()
            conf = float(p.max())
        except Exception:
            continue
        total += 1
        if conf < float(confidence_threshold):
            abstained += 1
    if total == 0:
        return None, None
    rate = abstained / float(total)
    return rate, 1.0 - rate
