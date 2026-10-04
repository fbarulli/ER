"""Tracked training objectives and per-population loss accounting.

The orchestration module retains the loss factory; these implementations only
consume explicit objective settings and do not load runtime configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class ContrastiveTelemetryBatch:
    """Detached bounded telemetry; no optimizer-facing graph is retained."""
    payload: Tensor
    pair_ids: Tensor | None
    label_count: int
    negative_count: int
    selected_count: int
    counts: tuple[int, int, int, int]
    maximum_pending_batches: ClassVar[int] = 16


def _smoothed_contrastive_losses(
    positive_pairs,
    negative_pairs,
    *,
    margin: float,
    label_smoothing: float,
):
    """Return positive/negative OnlineContrastiveLoss terms with smoothing."""
    import torch.nn.functional as F

    smoothing = float(label_smoothing)
    if not 0.0 <= smoothing < 0.5:
        raise ValueError("contrastive label smoothing must be in [0, 0.5)")
    positive_hinge = F.relu(float(margin) - positive_pairs)
    negative_hinge = F.relu(float(margin) - negative_pairs)
    positive_loss = (
        (1.0 - smoothing) * positive_pairs.pow(2)
        + smoothing * positive_hinge.pow(2)
    ).sum()
    negative_loss = (
        (1.0 - smoothing) * negative_hinge.pow(2)
        + smoothing * negative_pairs.pow(2)
    ).sum()
    return positive_loss, negative_loss, negative_hinge


def _tracking_contrastive_loss(
    model,
    *,
    margin: float,
    structured_feature_weight: float,
    uniformity_weight: float,
    uniformity_temperature: float,
    uniformity_min_batch_size: int,
    label_smoothing: float,
):
    """Return OnlineContrastiveLoss with selection/backprop telemetry.

    The implementation preserves the installed loss's hard-pair selection
    and arithmetic. It only accumulates detached counters and loss-component
    values during gradient-enabled forwards; ProgressCallback drains them at
    Trainer logging steps.
    """
    import torch
    import torch.nn.functional as F
    from sentence_transformers.sentence_transformer import losses

    class _TrackedOnlineContrastiveLoss(losses.OnlineContrastiveLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tracking_totals: dict[str, float] = {}
            self._tracking_batches = 0
            self._pending_telemetry: list[ContrastiveTelemetryBatch] = []
            self._batch_pair_ids = None
            self._batch_structured_features = None
            self._total_negative_pairs = 0
            self._seen_hard_negative_ids: set[int] = set()
            self._seen_margin_active_negative_ids: set[int] = set()
            self._negative_present_counts: dict[int, int] = {}
            self._negative_selected_counts: dict[int, int] = {}
            self._negative_backprop_counts: dict[int, int] = {}
            self._per_epoch_counts: dict[int, dict[int, dict[str, int]]] = {}
            self._pair_lineage: list[dict] = []
            self._current_epoch = 0

        def _uniformity_penalty(self, embeddings):
            if uniformity_weight <= 0:
                return embeddings[0].sum() * 0.0
            vectors = torch.cat(embeddings, dim=0)
            if len(vectors) < uniformity_min_batch_size:
                return vectors.sum() * 0.0
            vectors = F.normalize(vectors, p=2, dim=1)
            distances = torch.pdist(vectors, p=2).pow(2)
            if not len(distances):
                return vectors.sum() * 0.0
            return torch.logsumexp(
                -uniformity_temperature * distances,
                dim=0,
            ) - torch.log(
                torch.as_tensor(
                    len(distances),
                    dtype=distances.dtype,
                    device=distances.device,
                )
            )

        def set_batch_pair_ids(self, pair_ids) -> None:
            self._batch_pair_ids = pair_ids.detach()

        def set_batch_structured_features(self, features) -> None:
            self._batch_structured_features = features.detach()

        def set_total_negative_pairs(self, count: int) -> None:
            self._total_negative_pairs = int(count)

        def set_pair_lineage(self, pair_lineage: list[dict]) -> None:
            self._pair_lineage = pair_lineage

        def set_epoch(self, epoch: int) -> None:
            self.flush_tracking()
            self._current_epoch = int(epoch)

        def compute_loss_from_embeddings(self, embeddings, labels):
            if not self._checked_labels:
                self._checked_labels = True
                if labels.ne(0).logical_and(labels.ne(1)).any().item():
                    import warnings

                    warnings.warn(
                        "OnlineContrastiveLoss expects binary labels (0 or 1). "
                        "Pairs with any other label are ignored, since they "
                        "match neither the positive nor the negative set.",
                        UserWarning,
                        stacklevel=4,
                    )

            structured = self._batch_structured_features
            self._batch_structured_features = None
            if structured is not None and structured_feature_weight > 0:
                from core.structured_features import fuse_torch

                pair_features = structured.to(device=embeddings[0].device)
                embeddings = [
                    fuse_torch(
                        embedding,
                        pair_features[:, side, :],
                        structured_feature_weight,
                    )
                    for side, embedding in enumerate(embeddings)
                ]
            distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
            negs = distance_matrix[labels == 0]
            poss = distance_matrix[labels == 1]
            batch_pair_ids = self._batch_pair_ids
            self._batch_pair_ids = None

            # This is the installed sentence-transformers selection rule.
            negative_selection = negs < (
                poss.max() if len(poss) > 1 else negs.mean()
            )
            negative_pairs = negs[
                negative_selection
            ]
            positive_selection = poss > (
                negs.min() if len(negs) > 1 else poss.mean()
            )
            positive_pairs = poss[
                positive_selection
            ]
            # Binary label smoothing mixes a small amount of the opposite
            # class objective into each selected pair. At smoothing=0 this
            # is exactly the installed OnlineContrastiveLoss arithmetic.
            smoothing = float(label_smoothing)
            positive_loss, negative_loss, negative_hinge = (
                _smoothed_contrastive_losses(
                    positive_pairs,
                    negative_pairs,
                    margin=self.margin,
                    label_smoothing=smoothing,
                )
            )
            uniformity_loss = self._uniformity_penalty(embeddings)
            anti_collapse_loss = uniformity_weight * uniformity_loss
            loss_value = (
                positive_loss
                + negative_loss
                + anti_collapse_loss
            )

            # Evaluator forwards are no-grad; only optimizer-facing forwards
            # belong to the backprop attribution window.
            if torch.is_grad_enabled():
                scalars = torch.stack([positive_loss.detach(), negative_loss.detach(),
                                       uniformity_loss.detach(), anti_collapse_loss.detach()])
                payload = torch.cat([labels.detach().to(scalars.dtype),
                                     negative_selection.detach().to(scalars.dtype),
                                     (negative_hinge > 0).detach().to(scalars.dtype), scalars])
                self._pending_telemetry.append(ContrastiveTelemetryBatch(
                    payload, batch_pair_ids, len(labels), len(negs), len(negative_pairs),
                    (len(positive_pairs), len(negative_pairs), len(poss), len(negs))))
                if len(self._pending_telemetry) >= ContrastiveTelemetryBatch.maximum_pending_batches:
                    self.flush_tracking()

            return loss_value

        def flush_tracking(self) -> None:
            if not self._pending_telemetry:
                return
            batches = self._pending_telemetry
            payloads = torch.cat([batch.payload for batch in batches]).cpu()
            ids = [batch.pair_ids for batch in batches if batch.pair_ids is not None]
            pair_ids = torch.cat(ids).cpu() if ids else None
            payload_offset = id_offset = 0
            for batch in batches:
                width = batch.label_count + batch.negative_count + batch.selected_count + 4
                payload = payloads[payload_offset:payload_offset + width]
                payload_offset += width
                labels_cpu = payload[:batch.label_count]
                negative_selection = payload[batch.label_count:batch.label_count + batch.negative_count].bool()
                margin_active = payload[batch.label_count + batch.negative_count:-4].bool()
                if batch.pair_ids is not None:
                    batch_pair_ids = pair_ids[id_offset:id_offset + batch.label_count]
                    id_offset += batch.label_count
                    negative_ids = batch_pair_ids[labels_cpu == 0]
                    selected_negative_ids = negative_ids[
                        negative_selection
                    ]
                    margin_active_negative_ids = selected_negative_ids[
                        margin_active
                    ]
                    backprop_negative_ids = (
                        selected_negative_ids
                        if float(label_smoothing) > 0
                        else margin_active_negative_ids
                    )
                    for value in negative_ids.tolist():
                        key = int(value)
                        self._negative_present_counts[key] = (
                            self._negative_present_counts.get(key, 0) + 1
                        )
                        self._per_epoch_counts.setdefault(self._current_epoch, {}).setdefault(
                            key, {"present_count": 0, "hard_selected_count": 0, "backprop_count": 0}
                        )["present_count"] += 1
                    for value in selected_negative_ids.tolist():
                        key = int(value)
                        self._negative_selected_counts[key] = (
                            self._negative_selected_counts.get(key, 0) + 1
                        )
                        self._per_epoch_counts.setdefault(self._current_epoch, {}).setdefault(
                            key, {"present_count": 0, "hard_selected_count": 0, "backprop_count": 0}
                        )["hard_selected_count"] += 1
                    for value in backprop_negative_ids.tolist():
                        key = int(value)
                        self._negative_backprop_counts[key] = (
                            self._negative_backprop_counts.get(key, 0) + 1
                        )
                        self._per_epoch_counts.setdefault(self._current_epoch, {}).setdefault(
                            key, {"present_count": 0, "hard_selected_count": 0, "backprop_count": 0}
                        )["backprop_count"] += 1
                    self._seen_hard_negative_ids.update(
                        int(value) for value in selected_negative_ids.tolist()
                    )
                    self._seen_margin_active_negative_ids.update(
                        int(value) for value in margin_active_negative_ids.tolist()
                    )
                    source_sets = {
                        "present": negative_ids,
                        "selected": selected_negative_ids,
                        "backprop": backprop_negative_ids,
                    }
                    for event, ids in source_sets.items():
                        for pair_id in ids.tolist():
                            source = str(
                                self._pair_lineage[int(pair_id)].get(
                                    "population", "unknown"
                                )
                            )
                            metric = f"negative_source_{source}_{event}_count"
                            self._tracking_totals[metric] = (
                                self._tracking_totals.get(metric, 0.0) + 1.0
                            )
                positive, negative, all_positive, all_negative = batch.counts
                values = dict(zip(('positive_loss', 'negative_loss', 'uniformity_loss',
                                   'anti_collapse_loss'), payload[-4:].tolist()))
                values.update(hard_positive_count=float(positive), hard_negative_count=float(negative),
                              margin_active_negative_count=float(margin_active.sum()),
                              all_positive_count=float(all_positive), all_negative_count=float(all_negative))
                for key, value in values.items():
                    self._tracking_totals[key] = self._tracking_totals.get(key, 0.0) + value
                self._tracking_batches += 1
            self._pending_telemetry = []

        def pop_tracking_stats(self) -> dict[str, float]:
            self.flush_tracking()
            batches = self._tracking_batches
            totals = self._tracking_totals
            self._tracking_totals = {}
            self._tracking_batches = 0
            if not batches:
                return {}
            result = {
                "hard_positive_count": totals.get("hard_positive_count", 0.0),
                "hard_negative_count": totals.get("hard_negative_count", 0.0),
                "margin_active_negative_count": totals.get(
                    "margin_active_negative_count", 0.0
                ),
                "all_positive_count": totals.get("all_positive_count", 0.0),
                "all_negative_count": totals.get("all_negative_count", 0.0),
                "positive_loss": totals.get("positive_loss", 0.0),
                "negative_loss": totals.get("negative_loss", 0.0),
                "uniformity_loss": totals.get("uniformity_loss", 0.0),
                "anti_collapse_loss": totals.get("anti_collapse_loss", 0.0),
                "tracking_batches": float(batches),
            }
            result.update(
                {
                    key: value
                    for key, value in totals.items()
                    if key.startswith("negative_source_")
                }
            )
            result["margin_active_negative_fraction"] = (
                result["margin_active_negative_count"]
                / result["hard_negative_count"]
                if result["hard_negative_count"]
                else 0.0
            )
            total_loss = result["positive_loss"] + result["negative_loss"]
            result["negative_loss_fraction"] = (
                result["negative_loss"] / total_loss if total_loss else 0.0
            )
            return result

        def coverage_stats(self) -> dict[str, float]:
            self.flush_tracking()
            selected = len(self._seen_hard_negative_ids)
            active = len(self._seen_margin_active_negative_ids)
            total = self._total_negative_pairs
            return {
                "contrastive_margin": float(self.margin),
                "label_smoothing": float(label_smoothing),
                "negative_cosine_target": float(1.0 - self.margin),
                "n_train_neg_total": float(total),
                "n_train_neg_hard_selected_unique": float(selected),
                "n_train_neg_margin_active_unique": float(active),
                "train_neg_hard_selection_coverage": selected / total if total else 0.0,
                "train_neg_margin_active_coverage": active / total if total else 0.0,
                "n_train_neg_present_unique": float(len(self._negative_present_counts)),
                "n_train_neg_backprop_unique": float(len(self._negative_backprop_counts)),
                "train_neg_backprop_events": float(sum(self._negative_backprop_counts.values())),
            }

        def pair_usage_rows(self) -> list[dict]:
            """Return cumulative per-pair usage and gradient attribution."""
            self.flush_tracking()
            ids = set(self._negative_present_counts)
            ids.update(self._negative_selected_counts)
            ids.update(self._negative_backprop_counts)
            rows = []
            for pair_id in sorted(ids):
                lineage = (
                    self._pair_lineage[pair_id]
                    if pair_id < len(self._pair_lineage)
                    else {}
                )
                rows.append(
                    {
                        "pair_id": pair_id,
                        "present_count": self._negative_present_counts.get(pair_id, 0),
                        "hard_selected_count": self._negative_selected_counts.get(pair_id, 0),
                        "backprop_count": self._negative_backprop_counts.get(pair_id, 0),
                        **lineage,
                    }
                )
            return rows

        def pair_usage_rows_by_epoch(self) -> list[dict]:
            """Return per-epoch pair presentation/selection/backprop counts."""
            self.flush_tracking()
            rows = []
            for epoch in sorted(self._per_epoch_counts):
                for pair_id in sorted(self._per_epoch_counts[epoch]):
                    lineage = (
                        self._pair_lineage[pair_id]
                        if pair_id < len(self._pair_lineage)
                        else {}
                    )
                    rows.append(
                        {
                            "epoch": epoch,
                            "pair_id": pair_id,
                            **self._per_epoch_counts[epoch][pair_id],
                            **lineage,
                        }
                    )
            return rows

    return _TrackedOnlineContrastiveLoss(model, margin=margin)


def _tracking_mnrl_loss(
    model,
    *,
    monitoring_enabled: bool,
    warmup_enabled: bool,
    warmup_epochs: int,
    twin_weight: float,
):
    """Return MultipleNegativesRankingLoss with per-population telemetry.

    Reimplements the installed sentence-transformers 6.0.1
    ``MultipleNegativesRankingLoss.compute_loss_from_embeddings`` so each
    row's loss ``-(positive_score - log_z)`` can be attributed to a training
    population (base/masked/twin) and, when the twin warmup is enabled, so the
    twin rows can be down-weighted during the first ``warmup_epochs``.

    When neither monitoring nor warmup is enabled ``_make_loss`` returns the
    installed loss directly (this class is never constructed), and even if it
    were, ``compute_loss_from_embeddings`` falls through to the installed
    arithmetic — the bit-identical no-op guarantee.
    """
    from sentence_transformers.sentence_transformer import losses
    from sentence_transformers.util import all_gather_with_grad
    import torch

    class _TrackedMultipleNegativesRankingLoss(
        losses.MultipleNegativesRankingLoss
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._monitoring_enabled = bool(monitoring_enabled)
            self._warmup_enabled = bool(warmup_enabled)
            self._warmup_epochs = int(warmup_epochs)
            self._twin_weight = float(twin_weight)
            self._current_epoch = 0
            self._batch_pair_ids = None
            self._triple_populations: list[str] = []
            self._subset_totals: dict[int, dict[str, dict[str, float]]] = {}
            self._population_names = ['unknown']
            self._population_lookup = torch.empty(0, dtype=torch.long)
            self._population_device_lookup = {}
            self._pending_subset = None

        def set_epoch(self, epoch: int) -> None:
            if int(epoch) != self._current_epoch:
                self._flush_subset_totals()
            self._current_epoch = int(epoch)

        def set_batch_pair_ids(self, pair_ids) -> None:
            self._batch_pair_ids = pair_ids.detach()

        def set_triple_populations(self, populations) -> None:
            self._flush_subset_totals()
            self._triple_populations = list(populations)
            self._population_names = sorted(set(map(str, populations)) | {'unknown'})
            codes = {name: index for index, name in enumerate(self._population_names)}
            self._population_lookup = torch.tensor([codes[str(name)] for name in populations], dtype=torch.long)
            self._population_device_lookup = {}

        def _batch_population_codes(self, pair_ids, device):
            pair_ids = pair_ids.to(device=device, dtype=torch.long)
            lookup = self._population_device_lookup.get(device)
            if lookup is None:
                lookup = self._population_lookup.to(device)
                self._population_device_lookup[device] = lookup
            unknown = self._population_names.index('unknown')
            if not len(self._triple_populations):
                return torch.full_like(pair_ids, unknown)
            valid = pair_ids < len(self._triple_populations)
            safe_ids = torch.where(valid, pair_ids, 0)
            return torch.where(valid, lookup[safe_ids], unknown)

        def _flush_subset_totals(self):
            if self._pending_subset is None:
                return
            # One small host transfer per epoch/report, never one per batch.
            sums, counts = self._pending_subset.cpu().tolist()
            for population, total, count in zip(self._population_names, sums, counts):
                if not count:
                    continue
                stats = self._subset_totals.setdefault(self._current_epoch, {}).setdefault(
                    population, {'loss_sum': 0.0, 'count': 0.0})
                stats['loss_sum'] += total
                stats['count'] += count
            self._pending_subset = None

        def _twin_weight_for_epoch(self) -> float:
            if not self._warmup_enabled:
                return 1.0
            if self._warmup_epochs <= 1:
                return self._twin_weight
            progress = min(
                1.0, (self._current_epoch - 1) / (self._warmup_epochs - 1)
            )
            return self._twin_weight + (1.0 - self._twin_weight) * progress

        def compute_loss_from_embeddings(self, embeddings, labels):
            if not (self._monitoring_enabled or self._warmup_enabled):
                return super().compute_loss_from_embeddings(embeddings, labels)

            import torch

            if len(embeddings) < 2:
                raise ValueError(
                    f"Expected at least 2 embeddings, got {len(embeddings)}"
                )

            queries = embeddings[0]
            docs = embeddings[1:]
            batch_size = queries.size(0)
            offset = 0
            if self.gather_across_devices:
                queries = all_gather_with_grad(queries)
                docs = [all_gather_with_grad(doc) for doc in docs]
                offset = (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0) * batch_size

            world_batch_size = queries.size(0)
            docs_all = torch.cat(docs, dim=0)
            docs_pos = docs[0]
            local_indices = torch.arange(
                offset, offset + batch_size, device=queries.device
            )
            row_indices = torch.arange(batch_size, device=queries.device)
            local_queries = queries[local_indices]
            local_docs = docs_pos[local_indices]

            sim_matrices = {}
            sim_matrices["query_to_doc"] = self.similarity_fct(
                local_queries, docs_all
            )
            if "query_to_query" in self.directions:
                sim_matrices["query_to_query"] = self.similarity_fct(
                    local_queries, queries
                )
                sim_matrices["query_to_query"][
                    row_indices, local_indices
                ] = -torch.inf
            if "doc_to_query" in self.directions:
                sim_matrices["doc_to_query"] = self.similarity_fct(
                    queries, local_docs
                ).T
            if "doc_to_doc" in self.directions:
                sim_matrices["doc_to_doc"] = self.similarity_fct(
                    docs_all, local_docs
                ).T
                same_query_doc_mask = torch.eye(
                    world_batch_size, device=queries.device
                )[local_indices]
                same_query_doc_mask = same_query_doc_mask.repeat(
                    1, len(docs)
                ).bool()
                sim_matrices["doc_to_doc"].masked_fill_(
                    same_query_doc_mask, -torch.inf
                )

            penalties = {}
            if (
                self.hardness_mode
                in ("in_batch_negatives", "hard_negatives", "all_negatives")
                and self.hardness_strength > 0.0
            ):
                penalty = (
                    self.hardness_strength
                    * sim_matrices["query_to_doc"].detach()
                )
                own_doc_mask = torch.eye(
                    world_batch_size,
                    device=queries.device,
                    dtype=torch.bool,
                )[local_indices]
                own_doc_mask = own_doc_mask.repeat(1, len(docs))
                if self.hardness_mode == "hard_negatives":
                    penalty_exclusion_mask = ~own_doc_mask
                    penalty_exclusion_mask[:, :world_batch_size] = True
                elif self.hardness_mode == "in_batch_negatives":
                    penalty_exclusion_mask = own_doc_mask
                else:
                    penalty_exclusion_mask = own_doc_mask
                    penalty_exclusion_mask[:, world_batch_size:] = False
                penalty[penalty_exclusion_mask] = 0.0
                penalties["query_to_doc"] = penalty

            for key in sim_matrices:
                sim_matrices[key] = sim_matrices[key] * self.scale
            for key, pen in penalties.items():
                sim_matrices[key] = sim_matrices[key] + pen

            positive_scores = sim_matrices["query_to_doc"][
                row_indices, local_indices
            ]
            if self.partition_mode == "joint":
                scores = torch.cat(list(sim_matrices.values()), dim=1)
                log_z = torch.logsumexp(scores, dim=1)
            else:
                log_z = 0.0
                for sim_matrix in sim_matrices.values():
                    log_z += torch.logsumexp(sim_matrix, dim=1)
                log_z /= len(sim_matrices)

            row_losses = -(positive_scores - log_z)

            batch_pair_ids = self._batch_pair_ids
            self._batch_pair_ids = None
            population_codes = (self._batch_population_codes(batch_pair_ids, row_losses.device)
                                if batch_pair_ids is not None else None)
            if batch_pair_ids is not None and self._monitoring_enabled:
                if self._pending_subset is None:
                    self._pending_subset = torch.zeros((2, len(self._population_names)),
                                                       dtype=torch.float64, device=row_losses.device)
                values = row_losses.detach().to(dtype=torch.float64)
                self._pending_subset[0].scatter_add_(0, population_codes, values)
                self._pending_subset[1].scatter_add_(0, population_codes, torch.ones_like(values))

            if self._warmup_enabled:
                if batch_pair_ids is not None:
                    weight = self._twin_weight_for_epoch()
                    twin_code = (self._population_names.index('twin')
                                 if 'twin' in self._population_names else -1)
                    weights = torch.where(population_codes == twin_code,
                                          row_losses.new_full(row_losses.shape, weight),
                                          torch.ones_like(row_losses))
                    return (row_losses * weights).mean()
                return row_losses.mean()

            return row_losses.mean()

        def mnrl_subset_rows_by_epoch(self) -> list[dict]:
            self._flush_subset_totals()
            rows = []
            for epoch in sorted(self._subset_totals):
                for population in sorted(self._subset_totals[epoch]):
                    stats = self._subset_totals[epoch][population]
                    count = stats["count"]
                    rows.append(
                        {
                            "epoch": epoch,
                            "population": population,
                            "mean_loss": (
                                stats["loss_sum"] / count if count else 0.0
                            ),
                            "triple_count": int(count),
                        }
                    )
            return rows

    return _TrackedMultipleNegativesRankingLoss(model)
