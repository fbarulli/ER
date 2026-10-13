"""src/core/smart_split.py — the ONE stratified, power-consistent split owner.

Owner directive (2026-10-09): the split that both training checkouts share
(``laya`` and the ER ``tracks`` lane consume the SAME source rows) is the
"smart split": a deterministic, STRATIFIED ``train`` / ``dev`` / ``validation``
carve, sized power-consistently so the rarest meaningful subgroup still clears
the declared minimum detectable effect.

This module is that ONE owner. It has three jobs, no more:

* :attr:`SmartSplit.roles` — the declared role map (``train`` fits, ``select``
  is ``dev``, ``validate`` is ``test``); both lanes read it, so the split
  contract is never forked.
* :meth:`SmartSplit.allocate_stratified` — the single stratified allocator
  (per-stratum largest-remainder at the declared ratios) the laya corpus carve
  uses, and :meth:`SmartSplit.deal_uniform` — the single component round-robin
  deal the tracks holdout uses. One class, two grains, no second implementation.
* :meth:`SmartSplit.report` — the power-consistent sizing/MDE, delegated to the
  canonical ``core.sample_plan.SamplePlan`` (targets from ``config/sampling.yaml``).

The declaration is ``config/smart_split.yaml`` (roles, ratios, strata, seed);
no value is re-spelled in code. ``SmartSplitSpec`` validates at the boundary.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from core.sample_plan import SamplePlan, SamplePlanReport, SliceCensus

#: The declared smart-split document, beside the other config documents.
CONFIG_NAME = "smart_split.yaml"

#: The three roles the split contract names (the physical split name per role).
_TRAIN = "train"
_SELECT = "select"
_VALIDATE = "validate"


class SmartSplitRoles:
    """The code-level default role names (the config may re-state them).

    Declared once so no module spells ``"dev"`` / ``"test"`` as the SELECT /
    VALIDATE role; ``SmartSplitSpec`` defaults to these and validates any
    override.
    """

    TRAIN = "train"
    SELECT = "dev"
    VALIDATE = "test"
    ROLES: dict[str, str] = {_TRAIN: TRAIN, _SELECT: SELECT, _VALIDATE: VALIDATE}


class SmartSplitSpec(BaseModel):
    """The validated smart-split contract (``config/smart_split.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    roles: dict[str, str] = Field(
        default_factory=lambda: dict(SmartSplitRoles.ROLES))
    #: role/physical-split name -> fraction of the population (sums to 1).
    ratios: dict[str, float]
    #: The subgroup keys the stratified carve balances across every split.
    strata: tuple[str, ...] = Field(min_length=1)
    seed: int
    #: Named purpose-built carves under the SAME owner (e.g. the HPO lane's
    #: 50/50 ``laya_hpo_5050``). Additive: the default spec is unchanged when
    #: no profile is declared, so both existing consumers stay byte-identical.
    profiles: dict[str, SmartSplitProfile] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _contract_is_complete(self) -> "SmartSplitSpec":
        if set(self.roles) != {_TRAIN, _SELECT, _VALIDATE}:
            raise ValueError(
                f"smart_split.roles must declare exactly "
                f"{sorted({_TRAIN, _SELECT, _VALIDATE})}, got {sorted(self.roles)}")
        physical = list(self.roles.values())
        if len(set(physical)) != len(physical):
            raise ValueError(
                f"smart_split.roles maps two roles to one split: {self.roles}")
        if set(self.ratios) != set(physical):
            raise ValueError(
                f"smart_split.ratios keys {sorted(self.ratios)} must equal the "
                f"physical split names {sorted(physical)}")
        _ratios_are_fractions(self.ratios, "smart_split.ratios")
        return self

    @property
    def splits(self) -> tuple[str, ...]:
        """The physical split names in role order (train, select, validate)."""
        return (self.roles[_TRAIN], self.roles[_SELECT], self.roles[_VALIDATE])


def smart_split_config_path():
    """The declared smart-split document (``config/smart_split.yaml``)."""
    from core.common import TRAIN_ROOT

    return TRAIN_ROOT / "config" / CONFIG_NAME


def smart_split_spec() -> SmartSplitSpec:
    """Read + validate the smart-split declaration through the ONE home."""
    from core.common import load_validated_yaml

    return load_validated_yaml(
        smart_split_config_path(), SmartSplitSpec,
        label="Smart-split declaration")


def smart_split_profile(name: str) -> SmartSplitProfile:
    """Resolve ONE named carve profile from the smart-split declaration.

    Fails loud (KeyError) on an undeclared profile, so a lane can never
    silently fall back to the default three-role ratios: the profile IS the
    carve contract (e.g. the HPO lane's 50/50 ``laya_hpo_5050``).
    """
    profiles = smart_split_spec().profiles
    if name not in profiles:
        raise KeyError(
            f"smart_split profile {name!r} is not declared; known profiles: "
            f"{sorted(profiles)} (config/smart_split.yaml `profiles:`)")
    return profiles[name]


def largest_remainder(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of ``total`` at ``ratios`` (sums to total)."""
    raw = {key: total * ratios[key] for key in ratios}
    allocation = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(allocation.values())
    for key in sorted(ratios, key=lambda k: (-(raw[k] - allocation[k]), k)):
        if remainder <= 0:
            break
        allocation[key] += 1
        remainder -= 1
    return allocation


def _ratios_are_fractions(ratios: Mapping[str, float], where: str) -> None:
    """The ONE ratio contract: named, positive, and summing to 1.0."""
    if not ratios:
        raise ValueError(f"{where} must declare at least one split ratio")
    if any(value <= 0.0 for value in ratios.values()):
        raise ValueError(f"{where} must be positive: {dict(ratios)}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"{where} must sum to 1.0, got {total!r}: {dict(ratios)}")


class SmartSplitProfile(BaseModel):
    """A NAMED carve profile (``config/smart_split.yaml`` ``profiles:``).

    The default :class:`SmartSplitSpec` names the three-role train/select/
    validate carve both training checkouts share. A profile is a lighter,
    purpose-built carve under the SAME owner: its own ``ratios`` (any number of
    physical splits, e.g. the HPO lane's 50/50 ``train``/``dev`` with no test
    role), the ``strata`` it balances and the ``seed`` that makes it
    reproducible. Declared once in config; no consumer re-spells a ratio.
    """

    model_config = ConfigDict(extra="forbid")

    #: physical split name -> fraction of the population (sums to 1).
    ratios: dict[str, float]
    #: The subgroup keys the stratified carve balances across every split.
    strata: tuple[str, ...] = Field(min_length=1)
    seed: int

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> "SmartSplitProfile":
        _ratios_are_fractions(self.ratios, "smart_split profile ratios")
        return self

    @property
    def splits(self) -> tuple[str, ...]:
        """The physical split names, in declared ratio order."""
        return tuple(self.ratios)


class SmartSplit:
    """The ONE smart-split owner: role map + stratified/component allocation + sizing.

    Construction reads the declared config (``from_config``); the targets come
    from the canonical ``SamplePlan`` so the carve and the measurement share one
    MDE contract.
    """

    def __init__(self, spec: SmartSplitSpec, plan: "SamplePlan") -> None:
        self._spec = spec
        self._plan = plan

    @classmethod
    def from_config(cls) -> "SmartSplit":
        from core.sample_plan import SamplePlan

        return cls(smart_split_spec(), SamplePlan.from_config())

    @property
    def spec(self) -> SmartSplitSpec:
        return self._spec

    @property
    def roles(self) -> dict[str, str]:
        """The declared role map (``train``/``select``/``validate``)."""
        return dict(self._spec.roles)

    @property
    def splits(self) -> tuple[str, ...]:
        return self._spec.splits

    @property
    def strata(self) -> tuple[str, ...]:
        return self._spec.strata

    def allocate_stratified(self, items: list, stratum_of: Callable[[dict], str],
                            weight_of: Callable[[dict], int] | None = None
                            ) -> dict[str, list]:
        """Deterministically carve ``items`` into the physical splits, STRATIFIED.

        Each ``stratum_of(item)`` group is allocated independently at the
        declared ratios (largest remainder), so a rare stratum is represented
        in every split instead of being concentrated by one global shuffle.
        The seed is per-stratum, so the carve is reproducible byte-for-byte.

        ``weight_of`` carries unevenly sized items (the component-grain carve,
        where one item is a whole identity component and its weight is the
        listing count it brings along): the per-stratum largest remainder then
        allocates WEIGHT, and the seeded order fills each split greedily up to
        its declared weight share (within one item's weight at the seam).
        Omitted, every item weighs one and the carve is by item count exactly
        as before.
        """
        ratios = self._spec.ratios
        by_stratum: dict[str, list] = defaultdict(list)
        for item in items:
            by_stratum[stratum_of(item)].append(item)
        out: dict[str, list] = {name: [] for name in ratios}
        for stratum in sorted(by_stratum):
            members = list(by_stratum[stratum])
            random.Random(f"{self._spec.seed}:{stratum}").shuffle(members)
            if weight_of is None:
                allocation = largest_remainder(len(members), ratios)
                cursor = 0
                for name in ratios:
                    out[name].extend(members[cursor:cursor + allocation[name]])
                    cursor += allocation[name]
                continue
            weights = [weight_of(member) for member in members]
            targets = largest_remainder(sum(weights), ratios)
            cursor = 0
            for name in ratios:
                start, filled = cursor, 0
                while cursor < len(members) and filled < targets[name]:
                    filled += weights[cursor]
                    cursor += 1
                out[name].extend(members[start:cursor])
            out[list(ratios)[-1]].extend(members[cursor:])
        return out

    @staticmethod
    def allocate_components(items: list, component_of: Callable[[dict], str],
                            stratum_of: Callable[[dict], str],
                            ratios: Mapping[str, float], seed: int
                            ) -> dict[str, list]:
        """Carve ``items`` at the COMPONENT grain, STRATIFIED (zero leakage).

        The component-aware sibling of :meth:`allocate_stratified`: a component
        (``component_of(item)`` — an atomic entity such as the union-find root
        of a record's GTIN endpoints) is dealt WHOLE, so it never straddles two
        splits. That is the zero-leakage guarantee — every item sharing an
        entity lands in the SAME split, so no product/pair/endpoint crosses the
        carve. Whole components are then dealt greedily, largest first, each to
        the split most behind its per-stratum largest-remainder ``ratios``
        target, so the ``stratum_of`` proportions are preserved in every split
        instead of being concentrated by one global shuffle.

        Deterministic: components are seeded-shuffled then stably sorted
        largest-first, and the greedy target is a pure function of ``ratios``,
        so a rerun reproduces every byte.
        """
        _ratios_are_fractions(ratios, "allocate_components ratios")
        names = list(ratios)
        by_component: dict[str, list] = defaultdict(list)
        for item in items:
            by_component[str(component_of(item))].append(item)
        # Per-component stratum census + the population stratum totals, so the
        # per-split target is the largest-remainder share of each stratum.
        component_strata: dict[str, Counter] = {}
        stratum_totals: Counter = Counter()
        for component, members in by_component.items():
            census = Counter(stratum_of(member) for member in members)
            component_strata[component] = census
            stratum_totals.update(census)
        target = {name: {stratum: ratios[name] * count
                         for stratum, count in stratum_totals.items()}
                  for name in names}
        filled: dict[str, Counter] = {name: Counter() for name in names}
        order = list(by_component)
        random.Random(seed).shuffle(order)
        order.sort(key=lambda component: -len(by_component[component]))
        out: dict[str, list] = {name: [] for name in names}
        for component in order:
            census = component_strata[component]
            # The split most behind target over THIS component's strata; ties
            # break on the declared ratio order so the deal is reproducible.
            chosen = min(
                names,
                key=lambda name: (sum(filled[name][s] - target[name][s]
                                      for s in census), names.index(name)))
            out[chosen].extend(by_component[component])
            filled[chosen].update(census)
        return out

    @staticmethod
    def deal_uniform(components: Sequence[set[str]], k: int, seed: int
                     ) -> list[set[str]]:
        """Deal components round-robin over ``k`` seeded folds (the tracks grain).

        The single component-fold deal shared with the tracks holdout: a
        seeded permutation of the component list, dealt ``i % k``, so every
        positive pair stays inside ONE component and therefore ONE fold.
        """
        order = np.random.default_rng(seed).permutation(len(components))
        folds: list[set[str]] = [set() for _ in range(k)]
        for index, component_index in enumerate(order):
            folds[index % k] |= components[component_index]
        return folds

    def report(self, censuses: "Sequence[SliceCensus]", *,
               labeled_census: int | None = None,
               current_validation: int | None = None,
               component_folds: int | None = None) -> "SamplePlanReport":
        """The power-consistent sizing/MDE for the measured carrier censuses."""
        return self._plan.plan(
            censuses, labeled_census=labeled_census,
            current_validation=current_validation,
            component_folds=component_folds)

    def mde_paired(self, n: int) -> float:
        """The achievable paired MDE at ``n`` (the declared targets)."""
        return self._plan.mde_paired(n)

    def per_subgroup_n(self) -> int:
        """The declared per-subgroup floor (targets via ``SamplePlan``)."""
        targets = self._plan.spec.targets
        return max(
            self._plan.required_paired_n(targets.target_effect),
            self._plan.required_proportion_n(targets.worst_case_proportion),
        )


__all__ = [
    "CONFIG_NAME",
    "largest_remainder",
    "SmartSplit",
    "SmartSplitProfile",
    "SmartSplitRoles",
    "SmartSplitSpec",
    "smart_split_config_path",
    "smart_split_profile",
    "smart_split_spec",
]
