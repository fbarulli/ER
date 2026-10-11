"""scripts/laya_corpus_growth.py — fold the pipeline's minted data into cases.

Owner order "laya is overfitting" (2026-10-08): the corpus folds the masking
and augmentation the pipeline already mints ON TOP of the base sources. Both
are OPT-IN so the hermetic builder tests are unchanged.

`GrowthFoldIngestor` turns the prepared text bundle's `mask_audit` /
`hard_negative_mask_audit` and the ground-truth `labeled_pairs.csv` into
corpus records, reading every token off the payload (never invented). A
counterfactual that flips a field the six-field state cannot carry collapses
to two identical sides and is SKIPPED and COUNTED.
"""
from __future__ import annotations

from pathlib import Path

from scripts.laya_corpus_cases import CorpusCaseRenderer
from scripts.laya_corpus_composer import compose_side, compose_state
from scripts.laya_corpus_io import CsvSource, PreparedBundle
from scripts.laya_corpus_rules import PackageStateRules, PairLabelRules

# Cleaned payload token prefix -> standardized attribute key. This is the
# inverse vocabulary mapping the six identity slice fields need: the prepared
# bundle's structured tail spells a field as `volume_ml_500` / `flavor_apple`,
# the reused `compose_side` reads `Volume: 500` / `Flavour: apple`.
# Longest/most-specific prefixes first so `sweetener_type_` never falls
# through to `sweetener_`.
_PAYLOAD_ATTR_PREFIXES: tuple[tuple[str, str], ...] = (
    ("sweetener_diet_", "Sweetener"),
    ("sweetener_type_", "Sweetener"),
    ("sweetening_", "Sweetening"),
    ("volume_ml_", "Volume"),
    ("pack_qty_", "Pack Size"),
    ("package_type_", "Pack Type"),
    ("flavor_", "Flavour"),
    ("carbonation_", "Carbonization"),
    ("sweetener_", "Sweetener"),
)
_PAYLOAD_ATTR_ORDER = (
    "Volume", "Pack Type", "Flavour", "Carbonization", "Sweetener",
)


class GrowthFoldIngestor:
    """Folds the prepared bundle + labeled pairs into mask/augmentation cases."""

    @staticmethod
    def _attr_from_payload(text: str) -> str:
        """Cleaned payload text -> a standardized `Key: value` attribute string.

        Only the structured tail is mapped (the vocabulary `compose_side`
        parses); prose is dropped exactly as the existing pair cases drop it.
        Values keep their payload spelling (underscores are token separators
        except volume/pack, where `_` is the decimal point). No token is
        invented: every value is read off the payload.
        """
        values: dict[str, list[str]] = {}
        for token in str(text).split():
            for prefix, key in _PAYLOAD_ATTR_PREFIXES:
                if token.startswith(prefix):
                    value = token[len(prefix):]
                    if prefix in ("volume_ml_", "pack_qty_"):
                        value = value.replace("_", ".")
                    if value:
                        values.setdefault(key, []).append(value)
                    break
        return "; ".join(
            f"{key}: {', '.join(values[key])}"
            for key in _PAYLOAD_ATTR_ORDER if key in values)

    @staticmethod
    def _side_from_payload(text: str) -> dict[str, str]:
        """One cleaned payload text -> the six-field side dict `compose_side`
        emits (reused, never re-implemented: the render lives in the metrics
        pairs composer)."""
        return compose_side(GrowthFoldIngestor._attr_from_payload(text))

    @staticmethod
    def ingest(
        *, bundle_path: Path | None, labeled_pairs_path: Path | None,
        by_gtin: dict[str, dict], existing_pair_states: set[str],
        questions: dict,
    ) -> tuple[list[dict], list[dict], dict, dict]:
        """Fold the pipeline's minted masking/augmentation into corpus records.

        Returns `(mask_records, aug_records, census, origins)`. `origins` maps
        each augmentation PAIR state to its source ("bundle" for the
        pipeline-minted counterfactual/twin/swap negatives, "labeled_pairs" for
        the ground-truth labeled pairs), so the rebalance knob can thin ONLY the
        minted population. Deterministic: inputs are walked in file/pickle
        order, membership sets are never iterated, and the only RNG (split
        assignment) lives in the caller.
        """
        census: dict = {
            "bundle": str(bundle_path) if bundle_path else None,
            "labeled_pairs": (str(labeled_pairs_path)
                              if labeled_pairs_path else None),
            "bundle_mask_audit_total": 0,
            "bundle_hard_negative_audit_total": 0,
            "mask_positive_audits": 0,
            "mask_cases": 0,
            "mask_package_state_true": 0,
            "mask_package_state_false": 0,
            "mask_skipped_duplicate": 0,
            "aug_pair_audits": 0,
            "aug_pair_cases": 0,
            "aug_pair_positive": 0,
            "aug_pair_negative": 0,
            "aug_pair_counterfactual": 0,
            "aug_skipped_duplicate": 0,
            "aug_skipped_unrepresentable": 0,
            "labeled_pairs_rows": 0,
            "labeled_pairs_added": 0,
            "labeled_pairs_missing_gtin": 0,
            "labeled_pairs_skipped_duplicate": 0,
        }
        mask_records: list[dict] = []
        aug_records: list[dict] = []
        origins: dict[str, str] = {}
        mask_seen: set[str] = set()
        aug_seen: set[str] = set(existing_pair_states)

        if bundle_path is not None and Path(bundle_path).is_file():
            bundle = PreparedBundle.load(bundle_path)
            frame = bundle["df"]
            payload = bundle["payload"]
            mask_audits = list(bundle.get("mask_audit", []))
            hard_negatives = list(bundle.get("hard_negative_mask_audit", []))
            census["bundle_mask_audit_total"] = len(mask_audits)
            census["bundle_hard_negative_audit_total"] = len(hard_negatives)

            for audit in [*mask_audits, *hard_negatives]:
                population = audit.get("population")
                target_mode = audit.get("target_mode")
                if population == "positive" and target_mode != "swap_values":
                    # a masked-positive STATE variant
                    census["mask_positive_audits"] += 1
                    state = audit.get("masked_text")
                    if not state or state in mask_seen:
                        census["mask_skipped_duplicate"] += 1
                        continue
                    anchor = int(audit["anchor_payload_idx"])
                    attribute = str(frame.iloc[anchor]["attribute"])
                    label = "true" if PackageStateRules.package_state(
                        attribute) else "false"
                    expected = {"package_state": label}
                    if "evidence_sufficient" in questions:
                        side = compose_side(attribute)
                        expected["evidence_sufficient"] = (
                            "true" if PairLabelRules._has_evidence(side)
                            else "false")
                    mask_records.append(
                        CorpusCaseRenderer._record(
                            state, questions, expected,
                            **CorpusCaseRenderer._single_meta(attribute)))
                    mask_seen.add(state)
                    census[f"mask_package_state_{label}"] += 1
                    continue
                # a side-by-side augmentation PAIR (counterfactual/twin/minted)
                census["aug_pair_audits"] += 1
                label = "true" if population == "swap_counterpart" else "false"
                is_counterfactual = "true" if target_mode == "counterfactual" \
                    else "false"
                copy_text = (audit.get("masked_text")
                             or payload[int(audit["copy_payload_idx"])])
                pair_text = payload[int(audit["pair_payload_idx"])]
                side_one = GrowthFoldIngestor._side_from_payload(copy_text)
                side_two = GrowthFoldIngestor._side_from_payload(pair_text)
                if side_one == side_two or not (
                        PairLabelRules._has_evidence(side_one)
                        or PairLabelRules._has_evidence(side_two)):
                    census["aug_skipped_unrepresentable"] += 1
                    continue
                state = compose_state(side_one, side_two)
                if state in aug_seen:
                    census["aug_skipped_duplicate"] += 1
                    continue
                expected = CorpusCaseRenderer._pair_expected(
                    questions, side_one, side_two,
                    attr_one=GrowthFoldIngestor._attr_from_payload(copy_text),
                    attr_two=GrowthFoldIngestor._attr_from_payload(pair_text),
                    identity=label, counterfactual=is_counterfactual)
                if label == "true" and "identity_claim" not in expected:
                    # the fixture schema labelled no identity_claim: keep the
                    # historical positive/negative census meaningful anyway.
                    pass
                aug_records.append(
                    CorpusCaseRenderer._record(
                        state, questions, expected,
                        **CorpusCaseRenderer._pair_meta(side_one, side_two)))
                origins[state] = "bundle"
                aug_seen.add(state)
                census[f"aug_pair_{'positive' if label == 'true' else 'negative'}"] += 1
                if is_counterfactual == "true":
                    census["aug_pair_counterfactual"] = (
                        census.get("aug_pair_counterfactual", 0) + 1)

        if labeled_pairs_path is not None and Path(labeled_pairs_path).is_file():
            header, rows = CsvSource.read(labeled_pairs_path)
            if header != ["gtin1", "gtin2", "true_label"]:
                raise RuntimeError(f"labeled_pairs header drifted: {header}")
            census["labeled_pairs_rows"] = len(rows)
            for row in rows:
                one = by_gtin.get((row["gtin1"] or "").strip())
                two = by_gtin.get((row["gtin2"] or "").strip())
                if one is None or two is None:
                    census["labeled_pairs_missing_gtin"] += 1
                    continue
                label = "true" if int(row["true_label"]) == 1 else "false"
                side_one = compose_side(one["attribute"])
                side_two = compose_side(two["attribute"])
                state = compose_state(side_one, side_two)
                if state in aug_seen:
                    census["labeled_pairs_skipped_duplicate"] += 1
                    continue
                expected = CorpusCaseRenderer._pair_expected(
                    questions, side_one, side_two,
                    attr_one=one["attribute"], attr_two=two["attribute"],
                    identity=label,
                    brand_one=one.get("brand"), brand_two=two.get("brand"),
                    counterfactual="false")
                aug_records.append(
                    CorpusCaseRenderer._record(
                        state, questions, expected,
                        **CorpusCaseRenderer._pair_meta(side_one, side_two)))
                origins[state] = "labeled_pairs"
                aug_seen.add(state)
                census["labeled_pairs_added"] += 1
                census[f"aug_pair_{'positive' if label == 'true' else 'negative'}"] += 1

        census["mask_cases"] = len(mask_records)
        census["aug_pair_cases"] = len(aug_records)
        return mask_records, aug_records, census, origins
