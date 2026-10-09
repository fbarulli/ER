"""src/core/laya_datasets.py — the ONE dataset registry for the laya lane.

Every dataset slug, role and existence flag the laya staging/publish/download
surfaces touch is declared here exactly once. Config models (``LayaSpec``) and
shared transport models (``KaggleSpec.bundle_dataset_slug``) take their
defaults from this registry rather than spelling a literal, so a slug is
renamed in one place and every path follows.

``exists`` records the observed Kaggle state at 2026-10-08 (``kaggle datasets
list --mine``): the five live production datasets are marked present; the
input/checkpoint datasets the eval-only kernels reference were never published
— the exact gap that stranded the fine-tune checkpoint inside a kernel output
(see the JOB 1 regression). A missing dataset is data, not an error at import;
a download of it fails loud through :class:`cli.kaggle_download.DatasetDownloader`.
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


class LayaDatasets:
    """Class-based SSOT registry of every ER/laya dataset.

    Members are stable class attributes (``LayaDatasets.BASE``); ``all()`` and
    friends iterate them deterministically for staging, publishing and
    verification. No consumer may spell a dataset slug literal: read it from a
    member (``LayaDatasets.BASE.slug``).
    """

    # ── live production datasets (observed present 2026-10-08) ────────────
    BASE = LayaDataset(
        key="base", slug="fbarulli/er-laya-base", role="input", exists=True,
        note="shipped convaiinnovations/laya checkpoint archive; attached by "
             "the fine-tune kernel")
    CORPUS = LayaDataset(
        key="corpus", slug="fbarulli/er-laya-train", role="input", exists=True,
        note="data/laya/{train,dev,test}.jsonl + receipt; attached by the "
             "fine-tune and eval-only kernels. Roles (LayaSplitRoles): train = "
             "fit, dev = HPO select/early-stop, test = held-out validate")
    REQUESTS = LayaDataset(
        key="requests", slug="fbarulli/er-laya-requests", role="input",
        exists=True,
        note="typed decision payloads ({decision}.csv) attached by the decision "
             "kernels")
    BUNDLE = LayaDataset(
        key="bundle", slug="fbarulli/er-10k-bundle", role="input", exists=True,
        note="verified CPU-prepared bundle the track-train kernel attaches")
    REVIEWS = LayaDataset(
        key="reviews", slug="fbarulli/reviews", role="input", exists=True,
        note="source review corpus dataset")

    # ── referenced-but-never-published datasets ───────────────────────────
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

    _MEMBERS = (
        BASE, CORPUS, REQUESTS, BUNDLE, REVIEWS, DECISIONS, HOLDOUT,
        FINETUNE_CKPT,
    )

    def __init__(self) -> None:  # registry is a namespace, never instantiated
        raise TypeError("LayaDatasets is a static registry; use its classmethods")

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
            f"unknown laya dataset {key!r}; known: {sorted(cls.keys())}")

    @classmethod
    def by_slug(cls, slug: str) -> LayaDataset:
        for member in cls._MEMBERS:
            if member.slug == slug:
                return member
        raise KeyError(
            f"{slug!r} is not a registered laya dataset; known: "
            f"{sorted(cls.slugs())}")

    @classmethod
    def existing(cls) -> tuple[LayaDataset, ...]:
        return tuple(member for member in cls._MEMBERS if member.exists)

    @classmethod
    def missing(cls) -> tuple[LayaDataset, ...]:
        return tuple(member for member in cls._MEMBERS if not member.exists)
