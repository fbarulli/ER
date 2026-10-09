"""core/pair_identity.py — the ONE owner of a pair's stable identity.

Every pair-bearing artifact (``gate_results.csv``, ``labeled_pairs.csv``,
``final_validation.csv`` and the per-pair prediction dumps) has to name the
same pair the same way, or the pairs cannot be joined across the pipeline and a
single sample's error cannot be traced end to end. ``(a, b)`` and ``(b, a)``
are the SAME pair — the gate, the miner and the split already treat them so —
which means a direction-dependent key (``f"{gtin1}|{gtin2}"``, spelled by hand
in several places) silently splits one pair into two ids.

:class:`PairIdentity` is the one place that key is computed. It normalizes both
endpoints through the existing ``core.gtin`` normalizer, orders the two
spellings, and joins them. Normalization is what makes a UPC-12 and its
zero-prefixed GTIN-13 sibling the same endpoint; the raw spelling is the
fallback for a non-conforming endpoint (retailer-export noise), so two distinct
endpoints can never collapse to the same id. No hashing: the repo's identity is
structural and the key stays readable.
"""
from __future__ import annotations

import pandas as pd

from core.gtin import normalize_gtin_value


class PairIdentity:
    """Direction-independent identity for one GTIN pair: ``of(a, b) == of(b, a)``."""

    #: Joins the two ordered endpoint keys. ``|`` matches the pair spelling the
    #: trace already uses, so the key reads the same everywhere it lands.
    SEPARATOR = "|"

    @staticmethod
    def endpoint_key(raw: object) -> str:
        """The canonical spelling of one endpoint, raw fallback for junk.

        ``normalize_gtin_value`` yields ``None`` for an endpoint that is not a
        plausible checksummed GTIN (measured: never on the current gate /
        labeled / validation frames, but the raw exports carry such cells). A
        ``None`` must not become the shared key of every malformed endpoint, so
        the stripped raw spelling is the fallback.
        """
        key, _valid = normalize_gtin_value(raw)
        return key if key else str(raw).strip()

    @classmethod
    def of(cls, gtin1: object, gtin2: object) -> str:
        """The pair id of two endpoints, invariant under endpoint swap."""
        lo, hi = sorted((cls.endpoint_key(gtin1), cls.endpoint_key(gtin2)))
        return f"{lo}{cls.SEPARATOR}{hi}"

    @classmethod
    def column(cls, gtin1: pd.Series, gtin2: pd.Series) -> pd.Series:
        """Vectorized :meth:`of` over two aligned endpoint series.

        ``sort``-free and NaN-safe: an element-wise compare picks the ordered
        pair without a Python loop over the frame.
        """
        key1 = gtin1.map(cls.endpoint_key)
        key2 = gtin2.map(cls.endpoint_key)
        first = key1 <= key2
        lo = key1.where(first, key2)
        hi = key2.where(first, key1)
        return lo + cls.SEPARATOR + hi
