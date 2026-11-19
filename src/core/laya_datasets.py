"""src/core/laya_datasets.py — the laya lane's dataset SSOT (two registries).

Owner directive (2026-10-09): the laya lane uses EXACTLY TWO training corpora —
the carved full corpus and the smoke subset — and nothing else. The registry is
split along that seam:

* :class:`LayaCorpora` — the ONLY corpora the lane selects from: ``FULL`` (the
  carved full corpus the ``dev``/``test`` split is built from) and ``SMOKE``
  (the tiny deterministic subset). :meth:`LayaCorpora.select` is the lane's
  selection SSOT and fails loud on anything but ``full`` / ``smoke`` — there is
  no ``3k`` / ``50pct`` / ``10k`` cohort.
* :class:`LayaTransports` — the base-model checkpoint and the decision / eval /
  output datasets that travel around a run; they are never a training corpus.

Every slug is declared exactly once; config models (``LayaSpec``, the shared
``KaggleSpec``) reference these registries rather than spelling a literal. The
tracks-lane CPU bundle slug no longer lives here: it is owned by
``config/training.yaml`` ``kaggle.bundle_dataset_slug`` (its own layer).
"""
from __future__ import annotations

from typing import Iterator

from pydantic import BaseModel, ConfigDict


class LayaDataset(BaseModel):
    """One dataset the ER/laya project reads or publishes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    slug: str
    role: str
    exists: bool
    note: str = ""

    @property
    def owner(self) -> str:
        return self.slug.rpartition("/")[0]

    @property
    def name(self) -> str:
        return self.slug.rpartition("/")[2]


class _DatasetRegistry:
    """Static classmethod surface shared by the two laya registries.

    Members are stable class attributes; ``all()`` and friends iterate them
    deterministically. A registry is a namespace, never instantiated.
    """

    _MEMBERS: tuple[LayaDataset, ...] = ()

    def __init__(self) -> None:
        raise TypeError(
            f"{type(self).__name__} is a static registry; use its classmethods")

    @classmethod
    def all(cls) -> tuple[LayaDataset, ...]:
        return cls._MEMBERS

    @classmethod
    def __iter__(cls) -> Iterator[LayaDataset]:
        return iter(cls._MEMBERS)

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        return tuple(member.key for member in cls._MEMBERS)

    @classmethod
    def slugs(cls) -> tuple[str, ...]:
        return tuple(member.slug for member in cls._MEMBERS)

    @classmethod
    def get(cls, key: str) -> LayaDataset:
        for member in cls._MEMBERS:
            if member.key == key:
                return member
        raise KeyError(
            f"unknown {cls.__name__} entry {key!r}; known: {sorted(cls.keys())}")

    @classmethod
    def by_slug(cls, slug: str) -> LayaDataset:
        for member in cls._MEMBERS:
            if member.slug == slug:
                return member
        raise KeyError(
            f"{slug!r} is not a registered {cls.__name__} dataset; known: "
            f"{sorted(cls.slugs())}")

    @classmethod
    def existing(cls) -> tuple[LayaDataset, ...]:
        return tuple(member for member in cls._MEMBERS if member.exists)

    @classmethod
    def missing(cls) -> tuple[LayaDataset, ...]:
        return tuple(member for member in cls._MEMBERS if not member.exists)


class LayaCorpora(_DatasetRegistry):
    """The lane's ONLY training corpora — the selection SSOT.

    Exactly two members: the carved FULL corpus (production, the one the
    ``dev``/``test`` carve is built from) and the SMOKE subset. The kind
    vocabulary is ``("full", "smoke")``; anything else is unrepresentable
    through :meth:`select`.
    """

    FULL = LayaDataset(
        key="full", slug="fbarulli/er-laya-train", role="input", exists=True,
        note="the carved full corpus (data/laya/{train,dev,test}.jsonl + "
             "receipt; scripts/laya_build_dataset.py). The production "
             "fine-tune corpus; dev selects, test validates")
    SMOKE = LayaDataset(
        key="smoke", slug="fbarulli/er-laya-train-smoke", role="input",
        exists=False,
        note="the tiny deterministic smoke subset of the full corpus "
             "(cli.laya_smoke.FinetuneSmokeCorpus); published on demand")

    _MEMBERS = (FULL, SMOKE)
    #: The ONLY corpus kinds the lane's dataset/kind selection accepts.
    KINDS: tuple[str, ...] = ("full", "smoke")

    @classmethod
    def select(cls, kind: str) -> LayaDataset:
        """Resolve one corpus kind; fail loud on anything but full/smoke."""
        if kind not in cls.KINDS:
            raise ValueError(
                f"unknown laya corpus kind {kind!r}; the lane selects exactly "
                f"{cls.KINDS} (owner directive: the carved full corpus + the "
                "smoke corpus, and nothing else)")
        return cls.get(kind)


class LayaTransports(_DatasetRegistry):
    """The base-model + decision/eval/output datasets around a run.

    These are NOT training corpora: the base checkpoint the fine-tune starts
    from, the typed decision payloads, the holdout/checkpoint transports and
    the decision export target.
    """

    BASE = LayaDataset(
        key="base", slug="fbarulli/er-laya-base", role="input", exists=True,
        note="shipped convaiinnovations/laya checkpoint archive; attached by "
             "the fine-tune kernel")
    REQUESTS = LayaDataset(
        key="requests", slug="fbarulli/er-laya-requests", role="input",
        exists=True,
        note="typed decision payloads ({decision}.csv) attached by the decision "
             "kernels")
    DECISIONS = LayaDataset(
        key="decisions", slug="fbarulli/er-laya-decisions", role="output",
        exists=False,
        note="typed-decision export target (LayaSpec.export_dataset_slug); not "
             "published yet")
    HOLDOUT = LayaDataset(
        key="holdout", slug="fbarulli/er-laya-holdout", role="input",
        exists=False,
        note="component-disjoint holdout JSONL attached by the holdout-eval "
             "kernel; not published yet")
    FINETUNE_CKPT = LayaDataset(
        key="finetune_ckpt", slug="fbarulli/er-laya-finetune-ckpt",
        role="output", exists=False,
        note="fine-tuned checkpoint transport; never published. The checkpoint "
             "only ever existed inside the fine-tune kernel's /kaggle/working "
             "output (the JOB 1 regression)")

    _MEMBERS = (BASE, REQUESTS, DECISIONS, HOLDOUT, FINETUNE_CKPT)


__all__ = ["LayaCorpora", "LayaDataset", "LayaTransports", "_DatasetRegistry"]
