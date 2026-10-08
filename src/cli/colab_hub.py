"""cli/colab_hub.py — the ONE home for the RUNNING cli.colab identity.

Split phases of ``cli.colab`` (colab_launch/colab_transport/colab_runtime/
colab_bundle_prewarm/colab_validation_upload/colab_self_watch/colab_retention,
and the lane's ``colab_lane_contracts``) never hold a second copy of the
launcher module.  Whichever identity executes ``colab.py`` — the ``cli.colab``
import or ``__main__`` under ``python -m cli.colab`` — registers itself in
``sys.modules["__colab_runtime_self__"]`` at import, and every phase re-reads
that registration at CALL time through ``hub()``.

Ownership rule (consolidated 2026-10-08): this module is the single resolver and
the single lazy step-timing shim.  A phase that needs ``colab.SESSION``,
``colab.TRAINING_RESULTS``, ``colab._stamp`` or the ``_timed_colab`` step
decorator imports them from here; it never re-spells
``sys.modules["__colab_runtime_self__"]`` itself (the old
``_hub``/``_colab``/``_colab_hub`` trio, 9 bodies, 2 semantics).

Semantics (one, get-or-import): an already-registered running identity always
wins, so the legacy ``from cli import colab`` monkeypatch surface keeps driving
every phase; a phase imported before the launcher falls back to importing
``cli.colab`` — whose first import registers the identity — instead of raising
``KeyError``.
"""
from __future__ import annotations

import functools
import sys
from typing import Any


def hub() -> Any:
    """The RUNNING cli.colab module (never a second import copy)."""
    registered = sys.modules.get("__colab_runtime_self__")
    if registered is not None:
        return registered
    import cli.colab as surface

    return surface


def timed_colab(kind: str):
    """Lazy step-timing shim: cli.colab owns ``_timed_colab`` at call time."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            return hub()._timed_colab(kind)(function)(*args, **kwargs)
        return wrapped
    return decorate
