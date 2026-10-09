"""Baseline warm-start seed for the laya HPO lane (host-side, recipe SSOT).

The staged kernel enqueues ONE warm-start trial built from the SAME
``FinetuneSpec`` recipe the trainer reads, so the current best-known fine-tune
runs as one worker PROCESS concurrently with the TPE sweep. The seed is never a
second recipe registry: it is projected from ``LayaRecipeFactory`` and filtered
to the search-space ``dials``, so it cannot drift from the spec.

Two ``FinetuneSpec`` defaults are not directly representable as an Optuna
sample, and each has an in-space equivalent:

* an unset schedule end (``w_sph_end`` / ``w_rps_end`` /
  ``contrastive_margin_end`` is ``None``) means "hold at the start value"
  (``core.laya_controls.LossWeightSchedule._interp``), so the paired start
  value (name minus ``_end``) is used — identical training;
* ``base_batch`` defaults to ``0`` ("scaling off"), below its positive log
  floor; a zero below a positive floor is the disabled sentinel and is clamped
  to that floor (unused while ``lr_scaling`` is the baseline ``"none"``).

Any OTHER out-of-range default is real recipe/space drift and fails loud at
staging rather than silently sampling a value the recipe never had.
"""
from __future__ import annotations

from typing import Any

from cli.laya_recipe import LayaRecipeFactory


class BaselineSeedFactory:
    """Project one ``FinetuneSpec`` recipe onto the search-space dials (SSOT)."""

    END_SUFFIX = "_end"

    def __init__(self, spec: Any):
        self._recipe = LayaRecipeFactory(spec)

    def recipe_values(self) -> dict[str, Any]:
        """The trainer recipe: the config channel overlaid by the control one."""
        values = self._recipe.finetune_config()
        values.update(self._recipe.finetune_control())
        return values

    def seed(self, space: dict[str, Any]) -> dict[str, Any]:
        """One valid in-space dial value per declared dial (an Optuna sample)."""
        values = self.recipe_values()
        return {name: self._dial_value(name, dial, values)
                for name, dial in space["dials"].items()}

    def apply(self, space: dict[str, Any]) -> dict[str, Any]:
        """Append the baseline seed to ``warm_start.enqueue`` when enabled."""
        if not self._enabled(space):
            return space
        options = space.setdefault("options", {})
        warm = options.get("warm_start") or {}
        options["warm_start"] = warm
        enqueue = list(warm.get("enqueue") or [])
        seed = self.seed(space)
        if seed not in enqueue:
            enqueue.append(seed)
        warm["enqueue"] = enqueue
        return space

    @staticmethod
    def _enabled(space: dict[str, Any]) -> bool:
        return bool(((space.get("options") or {}).get("warm_start") or {})
                    .get("baseline", False))

    def _dial_value(self, name: str, dial: dict[str, Any],
                    values: dict[str, Any]) -> Any:
        value = values.get(name)
        if value is None and name.endswith(self.END_SUFFIX):
            value = values.get(name[: -len(self.END_SUFFIX)])
        if dial["type"] == "categorical":
            if value not in dial["choices"]:
                raise ValueError(
                    f"baseline value {value!r} for dial {name!r} is not one of "
                    f"{dial['choices']}; the FinetuneSpec default and the "
                    "search space have drifted")
            return value
        if value is None:
            raise ValueError(
                f"baseline dial {name!r} has no FinetuneSpec value to seed")
        lo, hi = dial["lo"], dial["hi"]
        if value < lo or value > hi:
            # A zero below a positive floor is the "disabled" sentinel
            # (``base_batch=0``); use the floor. Anything else is drift.
            if value == 0 and lo > 0:
                value = lo
            else:
                raise ValueError(
                    f"baseline value {value!r} for dial {name!r} is outside "
                    f"[{lo}, {hi}]; the FinetuneSpec default and the search "
                    "space have drifted")
        return int(value) if dial["type"] == "int" else float(value)
