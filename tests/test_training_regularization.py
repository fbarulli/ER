from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from core.common import masking_cfg, training_cfg
from core.schemas import MaskAuditEntry
from training import training
from training.masking import (
    augment_counterfactual_twins,
    augment_value_swaps,
    field_of,
)


class TrainingRegularizationTests(unittest.TestCase):
    def test_regularization_config_is_conservative_and_validated(self) -> None:
        cfg = training_cfg().training
        self.assertAlmostEqual(cfg.weight_decay, 0.01)
        self.assertAlmostEqual(cfg.projection_dropout, 0.10)
        self.assertAlmostEqual(cfg.label_smoothing, 0.05)
        self.assertTrue(cfg.random_easy_negatives.enabled)
        self.assertAlmostEqual(cfg.random_easy_negatives.ratio_to_hard, 1.0)

    def test_projection_dropout_is_idempotent_and_eval_safe(self) -> None:
        model = nn.Sequential()
        self.assertTrue(training._configure_projection_dropout(model, 0.10))
        self.assertFalse(training._configure_projection_dropout(model, 0.20))
        self.assertEqual(len(model), 1)
        model.eval()
        features = {"sentence_embedding": torch.ones(2, 4)}
        self.assertTrue(
            torch.equal(model(features)["sentence_embedding"], torch.ones(2, 4))
        )

    def test_zero_smoothing_preserves_online_contrastive_arithmetic(self) -> None:
        positives = torch.tensor([0.10, 0.30])
        negatives = torch.tensor([0.05, 0.30])
        positive_loss, negative_loss, hinge = training._smoothed_contrastive_losses(
            positives, negatives, margin=0.20, label_smoothing=0.0
        )
        self.assertAlmostEqual(positive_loss.item(), float((positives**2).sum()))
        self.assertAlmostEqual(negative_loss.item(), float((hinge**2).sum()))

    def test_random_easy_mixing_is_deterministic_ratio_exact_and_split_safe(
        self,
    ) -> None:
        candidates = np.asarray([[2, 3], [3, 4]], dtype=int)
        hard = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=int)
        sources = np.asarray(["gate", "gate", "attribute_conflict"], dtype=object)
        row_bc = np.asarray(["a", "b", "c", "d", "e"], dtype=object)
        kwargs = {
            "df": pd.DataFrame({"gtin": row_bc}),
            "row_bc": row_bc,
            "train_gtins": set(row_bc),
            "seed": 17,
            "enabled": True,
            "ratio_to_hard": 1.0,
            "candidate_pool_size": 10,
        }
        with patch.object(
            training,
            "_split_safe_random_negative_pairs",
            return_value=candidates.copy(),
        ):
            first = training._mix_random_easy_training_negatives(
                hard, sources, **kwargs
            )
            second = training._mix_random_easy_training_negatives(
                hard, sources, **kwargs
            )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[0][: len(hard)], hard)
        self.assertEqual(first[2], 2)
        self.assertEqual(len(first[0]), 6)
        self.assertEqual(list(first[1]).count("random_easy"), len(hard))
        self.assertTrue(training.pairs_in_set(first[0], row_bc, set(row_bc)).all())


class ValueSwapAugmentationTests(unittest.TestCase):
    """Static donor value transplants (coconut -> lime), decided pre-training.

    Positives are rewritten on BOTH sides from an agreeing donor pair, so a
    match stays a match. Hard negatives are rewritten on the anchor side
    only and stay label 0 by canonical identity. Donor tokens always come
    from the corpus payload — nothing is invented, nothing is dynamic.
    """

    def _payload(self) -> list[str]:
        # Structured tokens are the swap surface (the attribute channel the
        # gate, miners, vetoes, and vector all read). Prose keeps the
        # original word as background noise — the same standing the
        # unmasked prose around [MASK] tokens already has. Counterpart
        # prose differs from anchor prose, as real sku/canonical pairs do;
        # byte-identical sides would make symmetric copies degenerate.
        return [
            "cola coconut water volume_ml_500 flavor_coconut",
            "cola coconut aqua volume_ml_500 flavor_coconut",
            "cola lime water volume_ml_500 flavor_lime",
            "cola lime aqua volume_ml_500 flavor_lime",
        ]

    def _row_bc(self) -> np.ndarray:
        return np.asarray(["g1", "canon#g1", "g2", "canon#g2"], dtype=object)

    def test_positive_swap_is_symmetric_and_agrees(self) -> None:
        pairs = np.asarray([[0, 1], [2, 3]], dtype=int)
        out, payload, row_bc, n_added, audit = augment_value_swaps(
            pairs, self._payload(), self._row_bc(),
            frac=1.0, seed=7, population="positive", symmetric=True,
        )
        self.assertEqual(n_added, 2)
        self.assertEqual(len(payload), 8)
        for row in audit:
            entry = MaskAuditEntry.model_validate(row)
            self.assertEqual(entry.target_mode, "swap_values")
            self.assertEqual(entry.fields_hit, ["flavor"])
            self.assertIsNotNone(entry.copy_pair_payload_idx)
            self.assertIsNotNone(entry.donor_anchor_payload_idx)
            a_text = payload[entry.copy_payload_idx]
            b_text = payload[entry.copy_pair_payload_idx]
            a_flavor = [t for t in a_text.split() if field_of(t) == "flavor"]
            b_flavor = [t for t in b_text.split() if field_of(t) == "flavor"]
            # both sides carry the donor value, and they agree with each other
            self.assertEqual(a_flavor, b_flavor)
            self.assertTrue(a_flavor)
            # no invented tokens: every transplanted token exists in the corpus
            corpus = set(" ".join(self._payload()).split())
            self.assertTrue(set(a_flavor) <= corpus)
            # the anchor copy actually changed
            self.assertNotEqual(a_text, payload[entry.anchor_payload_idx])
            # #1 field-level proof: every token OUTSIDE the swapped group is
            # byte-identical to the source side — 90% shared tokens never move
            def _strip(text: str, field: str) -> list[str]:
                return [t for t in text.split() if field_of(t) != field]

            field = entry.fields_hit[0]
            self.assertEqual(_strip(a_text, field), _strip(entry.anchor_text, field))
            self.assertEqual(
                _strip(b_text, field), _strip(payload[entry.pair_payload_idx], field)
            )

    def test_negative_swap_touches_only_the_anchor(self) -> None:
        pairs = np.asarray([[0, 1], [2, 3]], dtype=int)
        before = self._payload()
        out, payload, row_bc, n_added, audit = augment_value_swaps(
            pairs, list(before), self._row_bc(),
            frac=1.0, seed=11, population="hard_negative", symmetric=False,
        )
        self.assertEqual(n_added, 2)
        self.assertEqual(len(payload), 6)
        for row in audit:
            entry = MaskAuditEntry.model_validate(row)
            self.assertEqual(entry.target_mode, "swap_values")
            self.assertIsNone(entry.copy_pair_payload_idx)
            # counterpart side is untouched: still the original pair index
            # with its original text
            self.assertIn(entry.pair_payload_idx, (1, 3))
            self.assertEqual(payload[entry.pair_payload_idx], before[entry.pair_payload_idx])
            self.assertNotEqual(payload[entry.copy_payload_idx], entry.anchor_text)
            self.assertEqual(row_bc[entry.copy_payload_idx], row_bc[entry.anchor_payload_idx])

    def test_swap_is_deterministic_for_a_seed(self) -> None:
        pairs = np.asarray([[0, 1], [2, 3]], dtype=int)
        first = augment_value_swaps(
            pairs, self._payload(), self._row_bc(),
            frac=0.5, seed=42, population="positive", symmetric=True,
        )
        second = augment_value_swaps(
            pairs, self._payload(), self._row_bc(),
            frac=0.5, seed=42, population="positive", symmetric=True,
        )
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[3], second[3])
        self.assertEqual(first[4], second[4])

    def test_no_eligible_donor_emits_nothing(self) -> None:
        pairs = np.asarray([[0, 1]], dtype=int)
        payload = [
            "cola coconut water volume_ml_500 flavor_coconut",
            "cola coconut water volume_ml_500 flavor_coconut",
        ]
        out, new_payload, row_bc, n_added, audit = augment_value_swaps(
            pairs, payload, np.asarray(["g1", "g1"], dtype=object),
            frac=1.0, seed=3, population="positive", symmetric=True,
        )
        self.assertEqual(n_added, 0)
        self.assertEqual(audit, [])
        np.testing.assert_array_equal(out, pairs)

    def test_value_swap_fracs_are_config_owned(self) -> None:
        cfg = masking_cfg()
        self.assertAlmostEqual(float(cfg["swap_value_frac"]), 0.20)
        self.assertAlmostEqual(float(cfg["hard_negative_swap_value_frac"]), 0.20)
        self.assertAlmostEqual(float(cfg["counterfactual_frac"]), 0.10)

    def test_donor_sharing_a_gtin_is_refused(self) -> None:
        # The only donor with a different value is a duplicate record of the
        # same entity (same gtin): no transplant may happen.
        pairs = np.asarray([[0, 1], [2, 3]], dtype=int)
        out, _, _, n_added, audit = augment_value_swaps(
            pairs, self._payload(), np.asarray(["g1", "g1", "g1", "g1"], dtype=object),
            frac=1.0, seed=7, population="positive", symmetric=True,
        )
        self.assertEqual(n_added, 0)
        self.assertEqual(audit, [])
        np.testing.assert_array_equal(out, pairs)

    def test_swap_that_erases_every_difference_is_skipped(self) -> None:
        # Anchor and counterpart differ ONLY in flavor; the donor offers the
        # counterpart's value, so the copy would be byte-identical to the
        # pair side — an impossible label-0 row. It must not be emitted.
        pairs = np.asarray([[0, 1], [2, 3]], dtype=int)
        payload = [
            "cola flavor_coconut volume_ml_500",
            "cola flavor_lime volume_ml_500",
            "cola flavor_lime volume_ml_500",
            "cola flavor_lime volume_ml_500",
        ]
        row_bc = np.asarray(["g1", "g1", "g2", "g2"], dtype=object)
        _, new_payload, _, _, audit = augment_value_swaps(
            pairs, payload, row_bc,
            frac=1.0, seed=7, population="hard_negative", symmetric=False,
        )
        self.assertFalse(
            [row for row in audit if row["anchor_payload_idx"] == 0],
            "anchor 0's only transplant reproduces its counterpart exactly",
        )
        for row in audit:
            MaskAuditEntry.model_validate(row)
            self.assertNotEqual(
                new_payload[row["copy_payload_idx"]],
                payload[row["pair_payload_idx"]],
            )


class CounterfactualTwinTests(unittest.TestCase):
    """Minimal-flip negatives: one agreed field broken, labeled 0.

    (A1', A2) differs from a verified match in exactly one load-bearing
    attribute, so it cannot be the same product. Both the original match
    and its twin train together.
    """

    def _payload(self) -> list[str]:
        return [
            "cola coconut water volume_ml_500 flavor_coconut",
            "cola coconut water volume_ml_500 flavor_coconut",
            "cola lime water volume_ml_500 flavor_lime",
            "cola lime water volume_ml_500 flavor_lime",
        ]

    def _row_bc(self) -> np.ndarray:
        return np.asarray(["g1", "g1", "g2", "g2"], dtype=object)

    def test_twin_breaks_exactly_one_agreed_field(self) -> None:
        pairs = np.asarray([[0, 1], [2, 3]], dtype=int)
        out, payload, row_bc, n_added, audit = augment_counterfactual_twins(
            pairs, self._payload(), self._row_bc(), frac=1.0, seed=7,
        )
        self.assertEqual(n_added, 2)
        for row in audit:
            entry = MaskAuditEntry.model_validate(row)
            self.assertEqual(entry.target_mode, "counterfactual")
            self.assertEqual(entry.population, "hard_negative")
            twin = payload[entry.copy_payload_idx]
            other = payload[entry.pair_payload_idx]
            # the twin differs from the pair side in the flipped group only
            field = entry.fields_hit[0]
            self.assertNotEqual(
                [t for t in twin.split() if field_of(t) == field],
                [t for t in other.split() if field_of(t) == field],
            )
            self.assertEqual(
                [t for t in twin.split() if field_of(t) != field],
                [t for t in other.split() if field_of(t) != field],
            )
            # ...and the flipped field was agreed before the flip
            before = row["anchor_text"]
            self.assertEqual(
                {t.lower() for t in before.split() if field_of(t) == field},
                {t.lower() for t in other.split() if field_of(t) == field},
            )

    def test_disagreed_fields_never_flip(self) -> None:
        # Anchor and counterpart already disagree on flavor; volume agrees
        # everywhere, so no minimal flip exists.
        pairs = np.asarray([[0, 1]], dtype=int)
        payload = [
            "cola flavor_coconut volume_ml_500",
            "cola flavor_lime volume_ml_500",
        ]
        out, _, _, n_added, audit = augment_counterfactual_twins(
            pairs, payload, np.asarray(["g1", "g1"], dtype=object),
            frac=1.0, seed=7,
        )
        self.assertEqual(n_added, 0)
        self.assertEqual(audit, [])

    def test_twin_trains_as_explicit_negative_of_its_source(self) -> None:
        # Regression: twins first shipped without this branch and were
        # silently dropped from MNRL triples (eval/diet only, no gradient).
        from training.training import _build_mnrl_training_triples

        train_pos = np.asarray([[0, 1]], dtype=int)
        train_neg = np.asarray([[2, 3], [4, 1]], dtype=int)
        hard_audit = [{
            "anchor_payload_idx": 0, "copy_payload_idx": 4,
            "pair_payload_idx": 1, "gtin": "g1",
            "realized_extent": 0.1, "configured_mask_lo": None,
            "configured_mask_hi": None, "mask_prob": None,
            "anchor_text": "a", "masked_text": "a-prime",
            "population": "hard_negative", "target_mode": "counterfactual",
            "fields_hit": ["flavor"], "donor_anchor_payload_idx": 2,
            "donor_pair_payload_idx": 3, "copy_pair_payload_idx": None,
        }]
        triples = _build_mnrl_training_triples(
            train_pos, train_neg, mask_audit=[],
            hard_negative_mask_audit=hard_audit,
        )
        self.assertIn((0, 1, 4), triples)

    def test_twin_is_deterministic_for_a_seed(self) -> None:
        pairs = np.asarray([[0, 1], [2, 3]], dtype=int)
        first = augment_counterfactual_twins(
            pairs, self._payload(), self._row_bc(), frac=0.5, seed=42,
        )
        second = augment_counterfactual_twins(
            pairs, self._payload(), self._row_bc(), frac=0.5, seed=42,
        )
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[3], second[3])
        self.assertEqual(first[4], second[4])

    def test_field_share_cap_breaks_volume_dominance(self) -> None:
        # Eligibility is what skews real pools (volume is parseable almost
        # everywhere; flavor only sometimes). The cap bounds any one field
        # to its share WITHOUT losing yield: capped picks redirect to an
        # eligible alternative instead of being dropped.
        flavors = ["coconut"] * 3 + ["lime"] * 3
        payload = [
            f"cola water volume_ml_{100 * i} flavor_{flavors[i]}"
            for i in range(6)
        ] + [
            f"cola aqua volume_ml_{100 * i} flavor_{flavors[i]}"
            for i in range(6)
        ]
        pairs = np.asarray([[i, i + 6] for i in range(6)], dtype=int)
        row_bc = np.asarray(
            [f"g{i}" if i % 2 == 0 else f"canon#g{i - 1}" for i in range(12)],
            dtype=object,
        )
        kwargs = {"pairs": pairs, "payload": payload, "row_bc": row_bc,
                  "frac": 1.0, "seed": 5}
        uncapped = augment_counterfactual_twins(**kwargs)[4]
        capped = augment_counterfactual_twins(
            **kwargs, max_field_share=0.2,
        )[4]
        from collections import Counter

        uncapped_fields = Counter(
            r["fields_hit"][0] for r in uncapped if r["fields_hit"]
        )
        capped_fields = Counter(
            r["fields_hit"][0] for r in capped if r["fields_hit"]
        )
        # no yield lost to the cap: every pick still minted a twin
        self.assertEqual(len(capped), len(uncapped))
        self.assertEqual(
            (uncapped_fields["volume"], uncapped_fields["flavor"]), (3, 3),
        )
        self.assertLessEqual(capped_fields["volume"], 2)
        self.assertGreaterEqual(capped_fields["flavor"], 4)

    def test_value_share_cap_stops_a_donor_footprint(self) -> None:
        # One lime donor pair, five coconut anchors: uncapped, the lime
        # value floods every twin; capped at 50% of picks, its absolute
        # reuse is bounded no matter how narrow the pool is.
        payload = (
            ["cola coconut water volume_ml_500 flavor_coconut"] * 2
            + ["cola lime water volume_ml_500 flavor_lime"] * 2
        ) + ["cola coconut aqua volume_ml_500 flavor_coconut"] * 8
        pairs = np.asarray(
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], dtype=int
        )
        row_bc = np.asarray([f"g{i}" for i in range(12)], dtype=object)
        kwargs = {"pairs": pairs, "payload": payload, "row_bc": row_bc,
                  "frac": 1.0, "seed": 9}
        uncapped = augment_counterfactual_twins(**kwargs)[4]
        capped = augment_counterfactual_twins(
            **kwargs, max_value_share=0.5,
        )[4]

        def lime_uses(audits: list[dict]) -> int:
            return sum(
                1 for r in audits
                if "flavor_lime" in payload[r["donor_anchor_payload_idx"]]
            )

        self.assertLess(lime_uses(capped), lime_uses(uncapped))
        self.assertLessEqual(lime_uses(capped), 3)
        for row in capped:
            from core.schemas import MaskAuditEntry

            MaskAuditEntry.model_validate(row)

    def test_value_cap_binds_across_lanes_when_counters_are_shared(self) -> None:
        # Per-call caps let three lanes triple the footprint; one shared
        # counter with a shared budget holds the bundle-global line.
        from collections import Counter

        payload = (
            ["cola coconut water volume_ml_500 flavor_coconut"] * 2
            + ["cola lime water volume_ml_500 flavor_lime"] * 2
        ) + ["cola coconut aqua volume_ml_500 flavor_coconut"] * 8
        pairs = np.asarray(
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], dtype=int
        )
        row_bc = np.asarray([f"g{i}" for i in range(12)], dtype=object)
        shared: Counter = Counter()
        total_lime = 0
        for seed in (21, 22, 23):
            audits = augment_counterfactual_twins(
                pairs, payload, row_bc, frac=1.0, seed=seed,
                max_value_share=0.5, shared_value_counts=shared,
                cap_base=6,
            )[4]
            total_lime += sum(
                1 for r in audits
                if "flavor_lime" in payload[r["donor_anchor_payload_idx"]]
            )
        self.assertLessEqual(total_lime, 3)

    def test_near_identical_donor_is_refused(self) -> None:
        # Donor differs in exactly one token out of 40+: probable
        # same-entity relist -> refuse at 0.95, allow uncapped.
        filler = " ".join(f"w{i}" for i in range(38))
        payload = [
            f"cola {filler} volume_ml_500 flavor_coconut",
            f"cola {filler} volume_ml_500 flavor_coconut",
            f"cola {filler} volume_ml_500 flavor_lime",
            f"cola {filler} volume_ml_500 flavor_lime",
        ]
        pairs = np.asarray([[0, 1], [2, 3]], dtype=int)
        row_bc = np.asarray(["g0", "g1", "g2", "g3"], dtype=object)
        refused = augment_counterfactual_twins(
            pairs, payload, row_bc, frac=1.0, seed=7,
            max_donor_overlap=0.95,
        )[4]
        allowed = augment_counterfactual_twins(
            pairs, payload, row_bc, frac=1.0, seed=7,
        )[4]
        self.assertEqual(refused, [])
        self.assertGreater(len(allowed), 0)

    def test_entity_key_normalizes_gtin_length_variants(self) -> None:
        from training.masking import normalize_entity_key

        self.assertEqual(
            normalize_entity_key("012345678905", "row:0"),
            normalize_entity_key("12345678905", "row:1"),
        )
        self.assertEqual(normalize_entity_key("", "row:7"), "row:7")
        self.assertNotEqual(
            normalize_entity_key("", "row:7"), normalize_entity_key("", "row:8"),
        )


class BundleProvenanceTests(unittest.TestCase):
    """Bundles pin the diet-relevant config; drift warns loudly on load."""

    def _write(self, path, *, pos=None, train_neg=None) -> None:
        import pandas as pd

        from training.prepared_bundle import write_prepared_bundle

        return write_prepared_bundle(
            path,
            df=pd.DataFrame({"sku_id": ["p1", "p2"]}),
            payload=["cola water", "cola aqua", "cola canon"],
            structured_features=np.zeros((3, 4), dtype=np.float32),
            row_bc=np.asarray(["g1", "g1", "g1"], dtype=object),
            country=np.asarray(["US", "US", "US"], dtype=object),
            pos=(
                np.asarray([[0, 2]], dtype=int)
                if pos is None
                else np.asarray(pos, dtype=int)
            ),
            hp_pairs=np.empty((0, 2), dtype=int),
            emb0=np.zeros((3, 4), dtype=np.float32),
            neg=np.empty((0, 2), dtype=int),
            train_neg=(
                np.empty((0, 2), dtype=int)
                if train_neg is None
                else np.asarray(train_neg, dtype=int)
            ),
            neg_sources=np.empty(0, dtype=object),
            train_neg_sources=np.asarray(
                ["gate"] * (0 if train_neg is None else len(train_neg)),
                dtype=object,
            ),
            mask_audit=[],
            hard_negative_mask_audit=[],
            labeled_pairs_csv=b"x",
            canonical_records_csv=b"y",
            gate_results_csv=b"z",
            payload_variant="full",
            masking_profile="baseline",
        )

    def test_roundtrip_records_active_diet_config(self) -> None:
        import tempfile
        from pathlib import Path

        from core.common import load_config, masking_cfg
        from training.prepared_bundle import load_prepared_bundle

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.pkl.gz"
            manifest = self._write(path)
            self.assertEqual(manifest.masking_config, masking_cfg("baseline"))
            self.assertEqual(
                manifest.easy_config,
                dict(load_config()["training"]["random_easy_negatives"]),
            )
            reloaded, _ = load_prepared_bundle(path)
            self.assertEqual(reloaded.masking_config, manifest.masking_config)
            self.assertAlmostEqual(manifest.ratio_to_hard, 1.0)
            self.assertGreater(manifest.static_view_ratio, 0.0)
            self.assertGreater(manifest.effective_train_ratio, 0.0)
            self.assertIn("not guaranteed", manifest.ratio_contract_note)

    def test_disabled_easy_negatives_do_not_change_effective_view_ratio(self) -> None:
        import copy
        import tempfile
        from pathlib import Path
        from unittest import mock

        from core.common import load_config

        config = copy.deepcopy(load_config())
        config["training"]["random_easy_negatives"]["enabled"] = False
        config["training"]["random_easy_negatives"]["ratio_to_hard"] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disabled-easy.pkl.gz"
            with mock.patch("core.common.load_config", return_value=config):
                manifest = self._write(
                    path,
                    pos=[[0, 2]],
                    train_neg=[[0, 1], [1, 2]],
                )

        self.assertEqual(manifest.easy_config["enabled"], False)
        self.assertEqual(manifest.static_view_ratio, 0.5)
        self.assertEqual(manifest.effective_train_ratio, 0.5)
        self.assertIn("disabled", manifest.ratio_contract_note)

    def test_loader_rejects_mismatched_augmentation_features_with_valid_hash(self) -> None:
        import gzip
        import hashlib
        import json
        import pickle
        import tempfile
        from pathlib import Path

        from training.prepared_bundle import load_prepared_bundle

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-lineage.pkl.gz"
            self._write(path)
            with gzip.open(path, "rb") as handle:
                data = pickle.load(handle)
            data["mask_audit"] = [{
                "anchor_payload_idx": 0,
                "copy_payload_idx": 2,
                "target_mode": "random",
                "fields_hit": [],
            }]
            data["structured_features"][-1] = np.ones(4, dtype=np.float32)
            with gzip.open(path, "wb", compresslevel=6) as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            manifest_path = path.with_suffix(path.suffix + ".json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "augmentation features disagree"):
                load_prepared_bundle(path)

    def test_drifted_config_warns_on_load(self) -> None:
        import copy
        import io
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from unittest import mock

        import core.common as core_common
        from training.prepared_bundle import load_prepared_bundle

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.pkl.gz"
            self._write(path)
            real_config = core_common.load_config()
            drifted = copy.deepcopy(real_config)
            drifted["training"]["random_easy_negatives"]["ratio_to_hard"] = 9.99
            # Lenient mode: the drift must stay a loud warning for any bundle
            # the audits still need to read (prepared_bundle_drift_strict off).
            drifted["prepared_bundle_drift_strict"] = False
            with mock.patch.object(
                core_common, "load_config", return_value=drifted
            ):
                captured = io.StringIO()
                with redirect_stdout(captured):
                    _, _ = load_prepared_bundle(path)
        self.assertIn("bundle-drift", captured.getvalue())
        self.assertIn("random_easy_negatives", captured.getvalue())
    """Transitive-closure entity IDs: pairs + shared gtins, one cluster."""

    def test_closure_links_pairs_and_gtin_groups(self) -> None:
        import pandas as pd

        from training.masking import build_entity_cluster_map

        truth = pd.DataFrame({
            "anchor_id": ["r0", "r2"],
            "pair_id": ["r1", "r3"],
        })
        pool = pd.DataFrame({
            "record_id": ["r0", "r1", "r2", "r3", "r4"],
            "gtin": ["A", "A", "B", "C", ""],
        })
        # r0-r1 share gtin A AND a truth pair; r2-r3 share only a truth
        # pair; r3's gtin C is a singleton; r4's gtin is empty.
        # Extra link: r1-r2 truth pair merges everything except r4.
        truth = pd.concat(
            [truth, pd.DataFrame({"anchor_id": ["r1"], "pair_id": ["r2"]})],
            ignore_index=True,
        )
        clusters = build_entity_cluster_map(truth, pool)
        self.assertEqual(
            {clusters["r0"], clusters["r1"], clusters["r2"], clusters["r3"]},
            {clusters["r0"]},
        )
        self.assertNotIn("r4", clusters)
        self.assertTrue(clusters["r0"].startswith("CLUSTER_"))

    def test_cluster_circuit_breaker_trips_on_giant(self) -> None:
        from training.masking import check_cluster_sizes

        star = {f"r{i}": "CLUSTER_000000" for i in range(20)}
        with self.assertRaises(ValueError) as caught:
            check_cluster_sizes(
                star, max_component_size=15, max_giant_ratio=0.05
            )
        self.assertIn("CLUSTER_000000", str(caught.exception))
        healthy = {f"s{i}": f"CLUSTER_{i:06d}" for i in range(38)}
        healthy.update({"a": "CLUSTER_999999", "b": "CLUSTER_999999"})
        ok = check_cluster_sizes(
            healthy, max_component_size=15, max_giant_ratio=0.05,
        )
        self.assertEqual(ok["max_size"], 2)

    def test_cluster_ids_are_deterministic(self) -> None:
        import pandas as pd

        from training.masking import build_entity_cluster_map

        truth = pd.DataFrame({
            "anchor_id": ["b", "c"],
            "pair_id": ["a", "d"],
        })
        pool = pd.DataFrame({
            "record_id": ["a", "b", "z"],
            "gtin": ["X", "X", "Y"],
        })
        first = build_entity_cluster_map(truth, pool)
        second = build_entity_cluster_map(truth, pool)
        self.assertEqual(first, second)

    def test_same_entity_different_gtins_refuses_donation(self) -> None:
        # Donor pair carries other gtins but a truth pair links it into
        # the anchor's cluster: multi-hop duplicate, no transplant allowed.
        import pandas as pd

        from training.masking import build_entity_cluster_map
        from training.masking import augment_value_swaps

        payload = [
            "cola coconut water volume_ml_500 flavor_coconut",
            "cola coconut aqua volume_ml_500 flavor_coconut",
            "cola lime water volume_ml_500 flavor_lime",
            "cola lime aqua volume_ml_500 flavor_lime",
        ]
        pairs = np.asarray([[0, 1], [2, 3]], dtype=int)
        records = ["r0", "r1", "r2", "r3"]
        truth = pd.DataFrame({
            "anchor_id": ["r0", "r2", "r1"],
            "pair_id": ["r1", "r3", "r2"],
        })
        pool = pd.DataFrame({
            "record_id": records,
            "gtin": ["A", "A", "B", "B"],
        })
        clusters = build_entity_cluster_map(truth, pool)
        self.assertEqual(len(set(clusters.values())), 1)
        entity_keys = [clusters[r] for r in records]
        out, _, _, n_added, audit = augment_value_swaps(
            pairs, payload,
            np.asarray(["A", "A", "B", "B"], dtype=object),
            frac=1.0, seed=7, population="positive", symmetric=True,
            entity_keys=entity_keys,
        )
        self.assertEqual(n_added, 0)
        self.assertEqual(audit, [])
        np.testing.assert_array_equal(out, pairs)


if __name__ == "__main__":
    unittest.main()
