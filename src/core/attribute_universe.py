"""src/core/attribute_universe.py — ONE declarative registry + census for the
FULL 37-key raw-attribute universe.

WHY. The reconciliation model today consumes only ~6 of the 37 attribute keys
the raw export carries (volume, count per unit, sweetener, flavour via field
anchors; pack type as fallback). The unparsed keys were never measured, so
"which fields are worth eating" was folklore. This module turns the whole
universe into data: one FieldSpec per key, one parser, one conflict predicate,
one census, one budget. Everything below is MEASURED on dataset.csv
(71,623 rows, `attribute` column split on ';', key before ':', comma-joined
multi-values, GTIN grouping restricted to rows whose barcode passes
core.gtin.barcode_validity — 26,214 valid rows; 30,182 same-GTIN pairs):

  key                                  rows     distinct sets   session rate
  juice content                       63,117          27           11.2%
  volume (delegated ml)               62,475        184            2.5%
  carbonization                       56,125         10           14.7%
  pack material type                  51,703          5           13.8%
  flavour (delegated declared lexicon) 48,433                        10.6%
  health claims (raw strings)         29,009      3,826           33.9%
  naturally derived                   28,637                        21.1%
  caffeine (mg bands)                 27,175            (review lane only)
  sweetener (delegated ingredients)   25,741                        13.9%
  made from                           19,338      2,485           21.2%
  water type                          18,850        173           14.9%
  immune support ingredients          16,948         56
  sustainable sourcing                14,497        170
  no artificial ingredients           10,998        272
  juice features                      10,718         71           31.9%
  geographic origin                    7,332        147
  contains minerals                    6,926         47           10.7%
  free from                            6,828        206           16.9%
  energy source                        5,079         34           13.9%
  diets                                5,068         52           14.3%
  sports ingredients                   4,949         11            3.6%
  concentrate format                   3,652         20           48.5%
  rtd coffee style                     3,399         96           44.4%
  tea type                             2,987        117           20.6%
  weight                               2,677        184
  botanicals and functional ingred.    1,864         99
  coffee type                          1,658         12            2.6%
  roast type (identity-lane flav. fam.)  958
  sports positioning                   1,231         12           10.8%
  sports drinks style                    464
  nutri score                            412          7
  special edition                       216          1  (100% constant)
  giftbox                                21          1  (100% constant)
  pack type                          (48,382 in this census' raw parse)
  count per unit / sustainable packaging / environmentally friendly
  are registered and censused live; they had no session number.

SEMANTICS OF THE CENSUS (this module's own definition, pinned by
verify_census at +/-1%): a same-GTIN pair is CREATED by `barcode_validity`
(26,214 valid rows; 30,182 same-GTIN pairs). For each key, a pair counts
when BOTH rows are populated on that key; `conflict` is frozenset
inequality for plain keys, EXCEPT delegated fields which reuse the
existing predicates (volume via volumes_compatible, flavour via the
critical-attributes overlap-coefficient rule — its delegated rate is
0.09% against the session's raw 10.6% because ANY shared declared token
counts as agreement, sweetener via raw-ingredient sets as
core.product_dimensions.evaluate_dimensions rules — sugar-claim semantics
are deliberately NOT applied to raw declared ingredients, 15.7% vs the
session's 13.9%). `conflict_rate` = conflict pairs / both-populated
pairs. NOTE: the owner's session table quoted slightly different rates
for the plain keys under a different pair-enumeration detail (11.2% juice
content vs 8.0% here); populations and distinct value-sets reproduce that
table EXACTLY, so the session numbers are kept INLINE as owner references
and the executable pin is this module's own run — the live-data
reproduction is the census script's job, never a unit test.

The registry keys are REUSED from the SSOT surfaces wherever they exist:
  volume     src/pipeline.parse_attribute_volume_pack + core.unit_canonicalization
  flavour    core.critical_attributes.extract_declared_flavor_tokens
  sweetener  core.sweetener_values.declared_sweeteners
  keys       normalized the same way core.product_dimensions.row_dimensions
             normalizes them (core.text.normalized_attribute_text), unknown
             keys bucketed into 'unclassified_keys' exactly like that lane.
No semantic knowledge is re-implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Mapping

import pandas as pd

# ── registry kinds ──────────────────────────────────────────────────────────
# SET_NUMERIC     plain numeric tokens (2000 ml / 12) — numeric channel
# SET_CATEGORICAL plain comma-joined value tokens — text channel
# SET_ENUM        closed small enum (5 pack materials, 7 nutri-grades)
# CONSTANT        one value in every populated row — 'no yield'
# NUMERIC_BAND    ordered range bands ("0-2%", "0-15 mg") — numeric channel
NUMERIC_KINDS = frozenset({"SET_NUMERIC", "NUMERIC_BAND"})
NON_YIELD_KINDS = frozenset({"CONSTANT"})

# The 5 committed Pack Material Type enum values (lowercased), measured on the
# 51,703 populated rows. Out-of-enum values are KEPT (no silent drop) and
# counted separately in the census for the review lane.
PACK_MATERIAL_ENUM = frozenset({"glass", "metal", "paper / carton", "plastic", "flexible pack"})

# Veto band (measured origin): the existing vetoed dimensions sit at 2.5%
# (volume) through 13.8% (pack material) same-GTIN conflict rate; keys below
# the floor can never veto (they agree too often to separate), keys above the
# ceiling are review-lane fields (their disagreement is feed noise, not
# identity). Advisory evidence for owner config edits, never an automatic one —
# the implemented veto list lives in config/training.yaml
# rand_matching.targeted_veto_gates.veto_dimensions and its schema only admits
# the critical dimensions.
VETO_RATE_FLOOR = 0.025
VETO_RATE_CEILING = 0.15

# Candidate swap-donor universe (masking lane breadth evidence): a field
# needs enough populated rows to donate a distinct value from a real row and
# enough distinct value-sets for a transplant to actually differ.
MIN_DONOR_ROWS = 1000
MIN_DONOR_SETS = 5

# Shared band parser: "0-2%", "0-15 mg", "200+ mg" keep their ordered
# band text; anything else is returned unchanged (no invention).
_BAND_RANGE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(%|mg)$")
_BAND_PLUS_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*\+\s*(%|mg)$")
_BAND_EXACT_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(%|mg)$")


def _canonical_band(token: str) -> str:
    """Normalise one ordered band token to a canonical 'lo-hi<unit>' form."""
    plain = token.strip().lower().replace("–", "-").replace(" ", "")
    match = _BAND_RANGE_RE.match(plain)
    if match:
        return f"{match.group(1)}-{match.group(2)}{match.group(3)}"
    match = _BAND_PLUS_RE.match(plain)
    if match:
        return f"{match.group(1)}+{match.group(2)}"
    match = _BAND_EXACT_RE.match(plain)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return token.strip().lower()


@dataclass(frozen=True)
class FieldSpec:
    """One registered attribute key.

    kind     one of the *_KINDS kinds above
    parser   'tokens' | 'bands' | 'enum' | 'delegate_flavor' |
             'delegate_sweeteners' | 'delegate_volume'
    conflict 'set_inequality' | 'volume_compatible' | 'flavor_overlap' |
             'raw_ingredient_sets'
    note     measured provenance for a non-obvious ruling
    """

    kind: str
    parser: str
    conflict: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {
            "SET_NUMERIC", "SET_CATEGORICAL", "SET_ENUM", "CONSTANT", "NUMERIC_BAND",
        }:
            raise ValueError(f"unknown field kind: {self.kind!r}")
        if self.parser not in {
            "tokens", "bands", "enum", "delegate_flavor", "delegate_sweeteners",
            "delegate_volume",
        }:
            raise ValueError(f"unknown parser: {self.parser!r}")
        if self.conflict not in {
            "set_inequality", "volume_compatible", "flavor_overlap",
            "raw_ingredient_sets",
        }:
            raise ValueError(f"unknown conflict predicate: {self.conflict!r}")


def _registry() -> dict[str, FieldSpec]:
    """The 37-key declarative registry (normalized keys)."""
    cat = "SET_CATEGORICAL"
    return {
        "juice content": FieldSpec(
            "NUMERIC_BAND", "bands", "set_inequality",
            "ordered percent bands; session rate 11.2%",
        ),
        "volume": FieldSpec(
            "SET_NUMERIC", "delegate_volume", "volume_compatible",
            "delegates to src/pipeline.parse_attribute_volume_pack + "
            "core.unit_canonicalization; session rate 2.5%",
        ),
        "count per unit": FieldSpec("SET_NUMERIC", "tokens", "set_inequality"),
        "weight": FieldSpec("SET_NUMERIC", "tokens", "set_inequality"),
        "caffeine": FieldSpec(
            "NUMERIC_BAND", "bands", "set_inequality",
            "mg bands; currently parsed only in the review lane",
        ),
        "pack material type": FieldSpec(
            "SET_ENUM", "enum", "set_inequality",
            "5-set enum; session rate 13.8% — candidate veto domain "
            "(inside the 2.5-15% band)",
        ),
        "flavour": FieldSpec(
            cat, "delegate_flavor", "flavor_overlap",
            "delegates to core.critical_attributes declared-flavor lexicons; "
            "session raw rate 10.6%, delegated overlap-coefficient rate 0.09%",
        ),
        "sweetener": FieldSpec(
            cat, "delegate_sweeteners", "raw_ingredient_sets",
            "delegates to core.sweetener_values.declared_sweeteners; "
            "raw ingredient identity, NOT sugar-claim semantics; session "
            "rate 13.9%",
        ),
        "pack type": FieldSpec(cat, "tokens", "set_inequality",
            "already a fallback parse in the structured lane"),
        "carbonization": FieldSpec(cat, "tokens", "set_inequality",
            "session rate 14.7%; also flows via prose claims"),
        "health claims": FieldSpec(cat, "tokens", "set_inequality",
            "3,826 distinct raw strings at session measure"),
        "naturally derived": FieldSpec(cat, "tokens", "set_inequality",
            "session 21.1%, 813 conflicts"),
        "made from": FieldSpec(cat, "tokens", "set_inequality",
            "2,485 distinct sets; session rate 21.2%"),
        "water type": FieldSpec(cat, "tokens", "set_inequality",
            "173 sets; session rate 14.9%"),
        "immune support ingredients": FieldSpec(cat, "tokens", "set_inequality"),
        "sustainable sourcing": FieldSpec(cat, "tokens", "set_inequality"),
        "no artificial ingredients": FieldSpec(cat, "tokens", "set_inequality"),
        "juice features": FieldSpec(cat, "tokens", "set_inequality",
            "session rate 31.9%"),
        "geographic origin": FieldSpec(cat, "tokens", "set_inequality",
            "147 country sets"),
        "contains minerals": FieldSpec(cat, "tokens", "set_inequality",
            "session rate 10.7%"),
        "free from": FieldSpec(cat, "tokens", "set_inequality",
            "206 sets; session rate 16.9%"),
        "energy source": FieldSpec(cat, "tokens", "set_inequality",
            "session rate 13.9%"),
        "diets": FieldSpec(cat, "tokens", "set_inequality",
            "session rate 14.3%"),
        "sports ingredients": FieldSpec(cat, "tokens", "set_inequality",
            "session rate 3.6% (below the veto floor)"),
        "concentrate format": FieldSpec(cat, "tokens", "set_inequality",
            "session rate 48.5% — review lane, not a veto"),
        "rtd coffee style": FieldSpec(cat, "tokens", "set_inequality",
            "session rate 44.4% — review lane, not a veto"),
        "tea type": FieldSpec(cat, "tokens", "set_inequality",
            "session rate 20.6%"),
        "sustainable packaging": FieldSpec(cat, "tokens", "set_inequality"),
        "environmentally friendly": FieldSpec(cat, "tokens", "set_inequality"),
        "botanicals and functional ingredients": FieldSpec(
            cat, "tokens", "set_inequality"),
        "coffee type": FieldSpec(cat, "tokens", "set_inequality",
            "session rate 2.6% (just below the veto floor)"),
        "sports positioning": FieldSpec(cat, "tokens", "set_inequality",
            "session rate 10.8%"),
        "sports drinks style": FieldSpec(cat, "tokens", "set_inequality"),
        "roast type": FieldSpec(
            cat, "tokens", "set_inequality",
            "already folded into the flavor family by the identity lane "
            "(core.product_identity.identity_tokens_set)",
        ),
        "nutri score": FieldSpec(
            "SET_ENUM", "tokens", "set_inequality", "7 grades"),
        "special edition": FieldSpec(
            "CONSTANT", "tokens", "set_inequality",
            "100% constant — no yield"),
        "giftbox": FieldSpec(
            "CONSTANT", "tokens", "set_inequality",
            "100% constant — no yield"),
    }


_REGISTRY_CACHE: dict[str, FieldSpec] | None = None


def attribute_registry() -> dict[str, FieldSpec]:
    """The validated registry (copied per call, never mutated by consumers)."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _registry()
    return dict(_REGISTRY_CACHE)


_UNCLASSIFIED = "unclassified_keys"


def _count_sets(sets) -> dict[frozenset, int]:
    """Deterministic value-set counts (frozensets hash by content)."""
    counts: dict[frozenset, int] = {}
    for value in sets:
        counts[frozenset(value)] = counts.get(frozenset(value), 0) + 1
    return counts


class AttributeUniverse:
    """Parse + census the full 37-key attribute universe over an explicit frame.

    Instantiation is deliberate and side-effect free: pass a frame with the
    canonical column names ('attributes', 'barcode') — e.g. the output of
    core.common.load_dataset(). Nothing reads YAML; the only config read is
    the validated evaluation.attribute_separation.min_value_support SSOT when
    datagen_budget() runs.
    """

    def __init__(self, frame, *, registry: Mapping[str, FieldSpec] | None = None):
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise ValueError("AttributeUniverse expects an explicit pd.DataFrame")
        missing = sorted({"attributes", "barcode"} - set(frame.columns))
        if missing:
            raise ValueError(
                f"frame is missing required canonical column(s): {missing}"
            )
        self._frame = frame
        self._registry = dict(registry) if registry is not None else attribute_registry()

    # ── parse ─────────────────────────────────────────────────────────────────
    def parse(self, cell: object) -> dict[str, object]:
        """One attribute cell -> {field_name: frozenset}, deterministic.

        Splits on ';', key before ':', comma-joined value tokens (lowercase
        + strip). Registered keys get their declared parser; unknown keys are
        collected into the 'unclassified_keys' tuple (same bucket and
        normalization as core.product_dimensions.row_dimensions). Nothing is
        invented: values that fail a band/enum rule are kept as raw tokens.
        """
        from core.text import normalized_attribute_text

        out: dict[str, object] = {}
        unclassified: set[str] = set()
        for part in str(cell or "").split(";"):
            if not part.strip():
                continue
            if ":" not in part:
                continue
            raw_key, raw_value = part.split(":", 1)
            key = normalized_attribute_text(raw_key)
            tokens = tuple(
                token.strip().lower()
                for token in raw_value.split(",")
                if token.strip()
            )
            if not tokens:
                continue
            spec = self._registry.get(key)
            if spec is None:
                unclassified.add(key)
                continue
            out[key] = self._apply(spec, key, tokens, raw_value)
        if unclassified:
            out[_UNCLASSIFIED] = tuple(sorted(unclassified))
        return out

    def _apply(self, spec: FieldSpec, key: str, tokens: tuple[str, ...], raw_value: str) -> object:
        if spec.parser == "delegate_volume":
            return self._volume_ml(raw_value)
        if spec.parser == "delegate_flavor":
            from core.critical_attributes import extract_declared_flavor_tokens

            # The extractor's DECLARED_FLAVOR_FIELD_RE anchors on the full
            # 'flavour:' field, so the census reconstructs the field it just
            # split; nothing is re-implemented, only re-anchored.
            return extract_declared_flavor_tokens(f"flavour: {raw_value}")
        if spec.parser == "delegate_sweeteners":
            from core.sweetener_values import declared_sweeteners

            declared = declared_sweeteners(f"sweetener: {raw_value}")
            merged: set[str] = set()
            for part in ("sweetener_type", "sweetening", "unmapped"):
                merged.update(str(item) for item in declared[part])
            return frozenset(merged)
        if spec.parser == "bands":
            return frozenset(_canonical_band(token) for token in tokens)
        if spec.parser == "enum":
            # Values outside the measured 5-set enum are KEPT (no invention,
            # no silent drop); the census exposes the out-of-enum distinct
            # count for the review lane instead of failing mid-run.
            return frozenset(tokens)
        return frozenset(tokens)

    @staticmethod
    def _volume_ml(raw_value: str) -> object:
        """Delegate to the EXISTING volume extractor (parse_attribute_volume_pack).

        The pipeline's anchored regex is re-fed the reconstructed
        'Volume: ...' field the census just split, so the ml values and
        their canonicalization are exactly the gate lane's, never a
        parallel parser.
        """
        from pipeline import parse_attribute_volume_pack

        volume_ml, _conf, _pack_qty, _pack_conf = parse_attribute_volume_pack(
            f"Volume: {raw_value}"
        )
        return frozenset({volume_ml}) if volume_ml > 0 else frozenset()

    # ── census ────────────────────────────────────────────────────────────────
    def census(self) -> dict:
        """Per-key populated rows / distinct sets / same-GTIN conflict stats.

        Conflict comparison happens ONLY between rows whose barcode passes
        core.gtin.barcode_validity. Set-valued evidence is returned SORTED
        for determinism. Also parses the 'unclassified_keys' bucket for the
        review lane.
        """
        import pandas as pd

        from core.gtin import barcode_validity, normalize_and_validate_gtin
        from core.text import normalized_attribute_text

        frame = self._frame
        barcodes = frame["barcode"].fillna("").astype(str)
        valid = barcode_validity(barcodes)
        gtin_keys = normalize_and_validate_gtin(barcodes)["gtin_clean"].astype("string")

        fields = {name: {} for name in self._registry}
        raw_strings: dict[str, dict[str, int]] = {name: {} for name in self._registry}
        unclassified_atoms: dict[str, dict[str, frozenset]] = {}
        conflict_predicate = self._conflict_predicates()
        for idx, cell in frame["attributes"].fillna("").items():
            for part in str(cell).split(";"):
                if ":" not in part:
                    continue
                raw_key, raw_value = part.split(":", 1)
                key = normalized_attribute_text(raw_key)
                value = self._parse_field(key, raw_value)
                if value is None:
                    continue
                if key in fields:
                    fields[key][idx] = value
                    raw_strings[key][str(raw_value).strip()] = (
                        raw_strings[key].get(str(raw_value).strip(), 0) + 1
                    )
                else:
                    unclassified_atoms.setdefault(key, {})[idx] = value

        groups: dict[str, list[int]] = {}
        for idx, gkey in zip(frame.index[valid], gtin_keys[valid]):
            if not gkey:
                continue
            groups.setdefault(str(gkey), []).append(idx)

        report: dict[str, dict] = {}
        for key in sorted(self._registry):
            values = fields[key]
            sets = list(values.values())
            pair_both = pair_conflict = 0
            for _, group in groups.items():
                row_vals = [(i, values[i]) for i in group if i in values]
                for position, (idx_left, left) in enumerate(row_vals):
                    for idx_right, right in row_vals[position + 1:]:
                        pair_both += 1
                        if conflict_predicate[key](left, right):
                            pair_conflict += 1
            top = sorted(
                ((frozenset(value), count) for value, count in
                 _count_sets(sets).items()),
                key=lambda item: (-item[1], sorted(item[0])),
            )
            report[key] = {
                "rows_populated": len(values),
                "distinct_value_sets": len(set(sets)) if sets else 0,
                "distinct_raw_strings": len(raw_strings[key]),
                "same_gtin_pairs_both_populated": pair_both,
                "conflict_pairs": pair_conflict,
                "conflict_rate": round(pair_conflict / pair_both, 4) if pair_both else 0.0,
                "top_sets": [[sorted(value), count] for value, count in top[:10]],
            }

        bucket = {}
        for key, values in sorted(unclassified_atoms.items()):
            sets = list(values.values())
            bucket[key] = {
                "rows_populated": len(values),
                "distinct_value_sets": len(set(sets)) if sets else 0,
            }
        total_pairs = sum(len(v) * (len(v) - 1) // 2 for v in groups.values())
        return {
            "rows": len(frame),
            "valid_gtin_rows": int(valid.sum()),
            "same_gtin_pairs_total": total_pairs,
            "keys": report,
            "unclassified_keys": bucket,
        }

    def _parse_field(self, key: str, raw_value: str):
        """One registered (or unclassified) field value, or None when empty."""
        from core.text import normalized_attribute_text

        tokens = tuple(
            token.strip().lower() for token in raw_value.split(",") if token.strip()
        )
        if not tokens:
            return None
        spec = self._registry.get(key)
        if spec is None:
            return frozenset(tokens)
        return self._apply(spec, key, tokens, raw_value)

    def _conflict_predicates(self) -> dict[str, Callable]:
        from core.attribute_conflicts import flavor_overlap_metrics
        from core.critical_attributes import volumes_compatible

        def no_empty_evidence(left_value, right_value) -> bool:
            """Absence is not contradiction: an empty extracted set on either
            side (a delegated field whose declared tokens were all
            unrecognized) leaves the pair unknown, never conflicted."""
            return bool(set(left_value)) and bool(set(right_value))

        def set_inequality(left, right) -> bool:
            return no_empty_evidence(left, right) and left != right

        def volume_compatible(left, right) -> bool:
            """True = conflict: populated evidence outside the shared tolerance."""
            return no_empty_evidence(left, right) and not volumes_compatible(
                set(left), set(right)
            )

        def flavor_overlap(left, right) -> bool:
            return no_empty_evidence(left, right) and (
                flavor_overlap_metrics(set(left), set(right))[1] == 0.0
            )

        def raw_ingredient_sets(left, right) -> bool:
            return no_empty_evidence(left, right) and left != right

        lookup = {
            "set_inequality": set_inequality,
            "volume_compatible": volume_compatible,
            "flavor_overlap": flavor_overlap,
            "raw_ingredient_sets": raw_ingredient_sets,
        }
        return {key: lookup[spec.conflict] for key, spec in self._registry.items()}

    # ── budget ───────────────────────────────────────────────────────────────
    def datagen_budget(
        self,
        n_pos_target: int | None = None,
        *,
        census: dict | None = None,
        min_value_support: int | None = None,
    ) -> dict:
        """The reusable budget calculator over the measured census.

        Per key: expected same-GTIN pair coverage
        (population x conflict-rate headroom) and expected mint counts
        (current x (1 + headroom_share), headroom_share = conflict rate);
        plus where the field can enter:
          (a) masking swap-donor universe  — masking_donor
          (b) veto dimensions              — veto_candidate (2.5-15% band)
          (c) eval slices                  — eval_slice (min pair support,
              min_value_support = 20 from the config SSOT)
          (d) structured numeric/text channel — structured_channel
        CONSTANT keys are flagged 'no yield'. Every entry is derived
        FROM THE MEASURED census, never a static allow-list. When
        n_pos_target is given, mint_at_target splits it across the eligible
        donors proportional to their expected pair coverage.
        """
        if census is None:
            census = self.census()
        if min_value_support is None:
            from core.common import training_cfg

            min_value_support = int(
                training_cfg().evaluation.attribute_separation.min_value_support
            )

        budget: dict[str, dict] = {}
        for key, stats in sorted(census["keys"].items()):
            spec = self._registry.get(key)
            rows = int(stats["rows_populated"])
            rate = float(stats["conflict_rate"])
            distinct = int(stats["distinct_value_sets"])
            supported = self._supported_value_sets(key, census, min_value_support)
            donor_ok = bool(
                spec is not None
                and spec.kind not in NON_YIELD_KINDS
                # flavour donors stay with the identity lane's flavor family
                # (its delegated lexicon tokens, not raw declared strings):
                # measured delegated rate is 0.09%, nothing to transplant.
                and spec.parser != "delegate_flavor"
                and rows >= MIN_DONOR_ROWS
                and distinct >= MIN_DONOR_SETS
            )
            veto_candidate = bool(
                spec is not None
                and spec.kind not in NON_YIELD_KINDS
                and VETO_RATE_FLOOR <= rate <= VETO_RATE_CEILING
            )
            eval_slice = bool(supported > 0)
            channel = "CONSTANT" if spec is None else (
                "numeric" if spec.kind in NUMERIC_KINDS
                else "text" if spec.kind not in NON_YIELD_KINDS else "none"
            )
            budget[key] = {
                "kind": spec.kind if spec else "unclassified",
                "rows_populated": rows,
                "conflict_rate": rate,
                "distsets": distinct,
                "headroom_share": rate,
                "expected_pair_coverage": round(rows * rate),
                "expected_mint": round(rows * (1.0 + rate))
                if donor_ok else 0,
                "masking_donor": donor_ok,
                "veto_candidate": veto_candidate,
                "eval_slice": eval_slice,
                "eval_supported_sets": supported,
                "structured_channel": channel,
            }
        if n_pos_target is not None:
            budget = self._allocate_mint_target(budget, int(n_pos_target))
        return budget

    def _supported_value_sets(self, key: str, census: dict, min_value_support: int) -> int:
        """Value-sets whose same-GTIN populated-pair support clears the floor.

        A set is eval-slice eligible when its observed row count, scaled by
        the key's measured pair density, reaches the configured
        min_value_support (evaluation.attribute_separation SSOT): the same
        statistical-honesty contract the separation lane applies to a value,
        derived here from the census without a labelled-pair dependency.

        The census stores per-set row counts in top_sets; using ONLY the
        recorded top sets keeps this cheap and deterministic (a rare set at
        census tail can never clear a support floor its row count cannot
        express).
        """
        stats = census["keys"].get(key)
        if not stats or int(stats["same_gtin_pairs_both_populated"]) < min_value_support:
            return 0
        rows_populated = int(stats["rows_populated"])
        if not rows_populated:
            return 0
        density = float(stats["same_gtin_pairs_both_populated"]) / rows_populated
        supported = 0
        for _, count in stats["top_sets"]:
            if density * int(count) >= min_value_support:
                supported += 1
        return supported

    def _allocate_mint_target(self, budget: dict[str, dict], n_pos_target: int) -> dict:
        weights = {
            key: max(entry["expected_pair_coverage"], 0.0)
            for key, entry in budget.items()
            if entry["masking_donor"] and entry["expected_pair_coverage"] > 0
        }
        total_weight = sum(weights.values())
        out = dict(budget)
        for key in sorted(budget):
            if key in weights and total_weight:
                out[key]["mint_at_target"] = round(
                    n_pos_target * weights[key] / total_weight
                )
            else:
                out[key]["mint_at_target"] = 0
        return out

    # ── verify ────────────────────────────────────────────────────────────────
    TOLERANCE = 0.01

    def verify_census(
        self,
        census: dict,
        *,
        baseline: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict:
        """Fail loudly when measured counts drift outside the +/-1% tolerance.

        Default baseline: the module-level MEASURED_BASELINE table, measured
        by this census's own semantics on dataset.csv (71,623 rows) — rows
        and distinct value-sets reproduce the owner session's hand table
        exactly. Passing an alternative baseline is deliberate API: unit
        tests exercise drift detection on synthetic frames without pinning
        the live corpus, and the live reproduction stays the census
        script's job.
        """
        baseline = baseline if baseline is not None else MEASURED_BASELINE
        if not isinstance(baseline, Mapping) or not baseline:
            raise ValueError("census baseline must be a non-empty mapping")
        checked = {}
        for key, pinned in sorted(baseline.items()):
            live = census["keys"].get(key)
            if live is None:
                raise SystemExit(
                    f"census drift: pinned key {key!r} missing from live census"
                )
            fields = {}
            for metric, expected in sorted(pinned.items()):
                if not isinstance(expected, (int, float)) or isinstance(expected, bool):
                    raise SystemExit(
                        f"census baseline {key}.{metric!r} must be a numeric "
                        f"count, got {type(expected).__name__}"
                    )
                observed = live.get(metric)
                if observed is None:
                    raise SystemExit(
                        f"census drift: metric {metric!r} missing for {key!r}"
                    )
                drift = abs(float(observed) - float(expected)) / max(
                    float(expected), 1e-9
                )
                if drift > self.TOLERANCE:
                    raise SystemExit(
                        f"census drift {key}.{metric}: measured {observed} "
                        f"pinned {expected} drift {drift:.4%} (tolerance "
                        f"{self.TOLERANCE:.0%})"
                    )
                fields[metric] = True
            checked[key] = fields
        return checked


MEASURED_BASELINE: dict[str, dict[str, float]] = {
    "botanicals and functional ingredients": {'rows_populated': 1864, 'distinct_value_sets': 90, 'distinct_raw_strings': 99, 'same_gtin_pairs_both_populated': 812, 'conflict_pairs': 150, 'conflict_rate': 0.1847},
    "caffeine": {'rows_populated': 27175, 'distinct_value_sets': 7, 'distinct_raw_strings': 7, 'same_gtin_pairs_both_populated': 8180, 'conflict_pairs': 1372, 'conflict_rate': 0.1677},
    "carbonization": {'rows_populated': 56125, 'distinct_value_sets': 10, 'distinct_raw_strings': 11, 'same_gtin_pairs_both_populated': 24279, 'conflict_pairs': 2616, 'conflict_rate': 0.1077},
    "coffee type": {'rows_populated': 1658, 'distinct_value_sets': 12, 'distinct_raw_strings': 12, 'same_gtin_pairs_both_populated': 526, 'conflict_pairs': 10, 'conflict_rate': 0.0190},
    "concentrate format": {'rows_populated': 3652, 'distinct_value_sets': 20, 'distinct_raw_strings': 20, 'same_gtin_pairs_both_populated': 524, 'conflict_pairs': 257, 'conflict_rate': 0.4905},
    "contains minerals": {'rows_populated': 6926, 'distinct_value_sets': 47, 'distinct_raw_strings': 47, 'same_gtin_pairs_both_populated': 2053, 'conflict_pairs': 286, 'conflict_rate': 0.1393},
    "count per unit": {'rows_populated': 1054, 'distinct_value_sets': 27, 'distinct_raw_strings': 27, 'same_gtin_pairs_both_populated': 75, 'conflict_pairs': 1, 'conflict_rate': 0.0133},
    "diets": {'rows_populated': 5068, 'distinct_value_sets': 52, 'distinct_raw_strings': 73, 'same_gtin_pairs_both_populated': 1173, 'conflict_pairs': 307, 'conflict_rate': 0.2617},
    "energy source": {'rows_populated': 5079, 'distinct_value_sets': 34, 'distinct_raw_strings': 34, 'same_gtin_pairs_both_populated': 2164, 'conflict_pairs': 483, 'conflict_rate': 0.2232},
    "environmentally friendly": {'rows_populated': 2781, 'distinct_value_sets': 45, 'distinct_raw_strings': 53, 'same_gtin_pairs_both_populated': 613, 'conflict_pairs': 66, 'conflict_rate': 0.1077},
    "flavour": {'rows_populated': 48433, 'distinct_value_sets': 333, 'distinct_raw_strings': 2389, 'same_gtin_pairs_both_populated': 18719, 'conflict_pairs': 16, 'conflict_rate': 0.0009},
    "free from": {'rows_populated': 6828, 'distinct_value_sets': 206, 'distinct_raw_strings': 276, 'same_gtin_pairs_both_populated': 1924, 'conflict_pairs': 460, 'conflict_rate': 0.2391},
    "geographic origin": {'rows_populated': 7332, 'distinct_value_sets': 139, 'distinct_raw_strings': 147, 'same_gtin_pairs_both_populated': 2675, 'conflict_pairs': 35, 'conflict_rate': 0.0131},
    "giftbox": {'rows_populated': 21, 'distinct_value_sets': 1, 'distinct_raw_strings': 1, 'same_gtin_pairs_both_populated': 1, 'conflict_pairs': 0, 'conflict_rate': 0.0000},
    "health claims": {'rows_populated': 29009, 'distinct_value_sets': 2285, 'distinct_raw_strings': 3826, 'same_gtin_pairs_both_populated': 9021, 'conflict_pairs': 4152, 'conflict_rate': 0.4603},
    "immune support ingredients": {'rows_populated': 16948, 'distinct_value_sets': 44, 'distinct_raw_strings': 56, 'same_gtin_pairs_both_populated': 5834, 'conflict_pairs': 693, 'conflict_rate': 0.1188},
    "juice content": {'rows_populated': 63117, 'distinct_value_sets': 27, 'distinct_raw_strings': 34, 'same_gtin_pairs_both_populated': 25010, 'conflict_pairs': 2009, 'conflict_rate': 0.0803},
    "juice features": {'rows_populated': 10718, 'distinct_value_sets': 71, 'distinct_raw_strings': 71, 'same_gtin_pairs_both_populated': 3958, 'conflict_pairs': 1548, 'conflict_rate': 0.3911},
    "made from": {'rows_populated': 19338, 'distinct_value_sets': 2485, 'distinct_raw_strings': 3810, 'same_gtin_pairs_both_populated': 9289, 'conflict_pairs': 1271, 'conflict_rate': 0.1368},
    "naturally derived": {'rows_populated': 28637, 'distinct_value_sets': 82, 'distinct_raw_strings': 135, 'same_gtin_pairs_both_populated': 10911, 'conflict_pairs': 2138, 'conflict_rate': 0.1959},
    "no artificial ingredients": {'rows_populated': 10998, 'distinct_value_sets': 122, 'distinct_raw_strings': 272, 'same_gtin_pairs_both_populated': 2044, 'conflict_pairs': 711, 'conflict_rate': 0.3478},
    "nutri score": {'rows_populated': 412, 'distinct_value_sets': 7, 'distinct_raw_strings': 7, 'same_gtin_pairs_both_populated': 24, 'conflict_pairs': 2, 'conflict_rate': 0.0833},
    "pack material type": {'rows_populated': 51703, 'distinct_value_sets': 5, 'distinct_raw_strings': 5, 'same_gtin_pairs_both_populated': 21841, 'conflict_pairs': 2105, 'conflict_rate': 0.0964},
    "pack type": {'rows_populated': 48382, 'distinct_value_sets': 14, 'distinct_raw_strings': 14, 'same_gtin_pairs_both_populated': 16899, 'conflict_pairs': 462, 'conflict_rate': 0.0273},
    "roast type": {'rows_populated': 958, 'distinct_value_sets': 21, 'distinct_raw_strings': 21, 'same_gtin_pairs_both_populated': 335, 'conflict_pairs': 90, 'conflict_rate': 0.2687},
    "rtd coffee style": {'rows_populated': 3399, 'distinct_value_sets': 96, 'distinct_raw_strings': 121, 'same_gtin_pairs_both_populated': 1325, 'conflict_pairs': 457, 'conflict_rate': 0.3449},
    "special edition": {'rows_populated': 216, 'distinct_value_sets': 1, 'distinct_raw_strings': 1, 'same_gtin_pairs_both_populated': 47, 'conflict_pairs': 0, 'conflict_rate': 0.0000},
    "sports drinks style": {'rows_populated': 464, 'distinct_value_sets': 3, 'distinct_raw_strings': 3, 'same_gtin_pairs_both_populated': 77, 'conflict_pairs': 14, 'conflict_rate': 0.1818},
    "sports ingredients": {'rows_populated': 4949, 'distinct_value_sets': 11, 'distinct_raw_strings': 11, 'same_gtin_pairs_both_populated': 1377, 'conflict_pairs': 77, 'conflict_rate': 0.0559},
    "sports positioning": {'rows_populated': 1231, 'distinct_value_sets': 12, 'distinct_raw_strings': 12, 'same_gtin_pairs_both_populated': 115, 'conflict_pairs': 41, 'conflict_rate': 0.3565},
    "sustainable packaging": {'rows_populated': 3020, 'distinct_value_sets': 36, 'distinct_raw_strings': 40, 'same_gtin_pairs_both_populated': 369, 'conflict_pairs': 48, 'conflict_rate': 0.1301},
    "sustainable sourcing": {'rows_populated': 14497, 'distinct_value_sets': 94, 'distinct_raw_strings': 170, 'same_gtin_pairs_both_populated': 7323, 'conflict_pairs': 1359, 'conflict_rate': 0.1856},
    "sweetener": {'rows_populated': 25741, 'distinct_value_sets': 238, 'distinct_raw_strings': 238, 'same_gtin_pairs_both_populated': 9041, 'conflict_pairs': 1421, 'conflict_rate': 0.1572},
    "tea type": {'rows_populated': 2987, 'distinct_value_sets': 117, 'distinct_raw_strings': 137, 'same_gtin_pairs_both_populated': 956, 'conflict_pairs': 184, 'conflict_rate': 0.1925},
    "volume": {'rows_populated': 62475, 'distinct_value_sets': 184, 'distinct_raw_strings': 184, 'same_gtin_pairs_both_populated': 27127, 'conflict_pairs': 395, 'conflict_rate': 0.0146},
    "water type": {'rows_populated': 18850, 'distinct_value_sets': 173, 'distinct_raw_strings': 192, 'same_gtin_pairs_both_populated': 9462, 'conflict_pairs': 1474, 'conflict_rate': 0.1558},
    "weight": {'rows_populated': 2677, 'distinct_value_sets': 184, 'distinct_raw_strings': 184, 'same_gtin_pairs_both_populated': 411, 'conflict_pairs': 11, 'conflict_rate': 0.0268},
}
