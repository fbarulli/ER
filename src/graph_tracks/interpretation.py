"""Descriptive comparison of graph-quality experiments on one fixed population."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class GraphQualityMetrics(BaseModel):
    """Reported metrics; absent measurements remain absent, never zero."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    pr_auc: float | None = Field(default=None, description="Pair ranking")
    recall_at_precision: float | None = Field(
        default=None, description="Pair recall at the agreed precision")
    known_positive_recall: float | None = Field(
        default=None, description="Retrieval recall at the agreed K")
    bcubed_precision: float | None = Field(
        default=None, description="Cluster purity around each listing")
    bcubed_recall: float | None = Field(
        default=None, description="Cluster completeness around each listing")
    brier: float | None = Field(
        default=None, description="Probability error",
        json_schema_extra={"lower_is_better": True})
    overmerge_rate: float | None = Field(
        default=None, description="Fraction of predicted clusters mixing true identities",
        json_schema_extra={"lower_is_better": True})


class GraphComparisonPolicy(BaseModel):
    """Absolute effect-size tolerance, not a statistical significance test."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    minimum_change: float = Field(ge=0)


class Change(StrEnum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    STABLE = "within_tolerance"
    UNAVAILABLE = "not_reported"


class MetricComparison(BaseModel):
    baseline: float | None
    candidate: float | None
    delta: float | None
    change: Change
    meaning: str

    @classmethod
    def build(
        cls, baseline: float | None, candidate: float | None, *,
        lower_is_better: bool, minimum_change: float, meaning: str,
    ) -> MetricComparison:
        if baseline is None or candidate is None:
            return cls(baseline=baseline, candidate=candidate, delta=None,
                       change=Change.UNAVAILABLE, meaning=meaning)
        delta = candidate - baseline
        gain = -delta if lower_is_better else delta
        change = Change.STABLE
        if gain > minimum_change:
            change = Change.IMPROVED
        elif gain < -minimum_change:
            change = Change.REGRESSED
        return cls(baseline=baseline, candidate=candidate, delta=delta,
                   change=change, meaning=meaning)


class GraphInterpretation(BaseModel):
    metrics: dict[str, MetricComparison]
    notes: list[str]


class GraphQualityInterpreter:
    """Compare point estimates without selecting a model or changing artifacts."""

    @classmethod
    def compare(
        cls, baseline: GraphQualityMetrics, candidate: GraphQualityMetrics,
        policy: GraphComparisonPolicy,
    ) -> GraphInterpretation:
        definitions = GraphQualityMetrics.model_json_schema()["properties"]
        metrics = {
            name: MetricComparison.build(
                getattr(baseline, name), getattr(candidate, name),
                lower_is_better=definition.get("lower_is_better", False),
                minimum_change=policy.minimum_change, meaning=definition["description"],
            )
            for name, definition in definitions.items()
        }
        return GraphInterpretation(metrics=metrics, notes=cls._notes(metrics))

    @staticmethod
    def _notes(metrics: dict[str, MetricComparison]) -> list[str]:
        changes = {name: metric.change for name, metric in metrics.items()}
        notes = [
            ("Descriptive deltas only: use matched populations, precision targets, "
             "retrieval K, candidate budgets, and cluster policies. Compare paired "
             "component-bootstrap intervals and repeated seeds before selecting a model.")
        ]
        if changes["recall_at_precision"] == Change.IMPROVED:
            notes.append("Pair recall improved at the precision target; check the "
                         "dev-selected threshold on held-out data before deployment.")
            if changes["known_positive_recall"] != Change.IMPROVED:
                notes.append("Better pair decisions do not establish better retrieval embeddings.")
        if (changes["pr_auc"] == Change.IMPROVED
                and changes["recall_at_precision"] == Change.REGRESSED):
            notes.append("Overall ranking improved but the required operating point worsened.")
        if changes["known_positive_recall"] == Change.REGRESSED:
            notes.append("Fewer known matches are retrieved: investigate neighborhoods "
                         "and embedding supervision before attributing this to the decoder.")
        if (changes["bcubed_precision"] == Change.REGRESSED
                or changes["overmerge_rate"] == Change.REGRESSED):
            notes.append("Cluster mixing worsened: inspect false bridge edges, "
                         "high-degree attribute hubs, and the merge threshold.")
        if changes["bcubed_recall"] == Change.REGRESSED:
            notes.append("Cluster completeness worsened: inspect fragmented identities "
                         "and matches lost during neighbor selection.")
        if changes["brier"] == Change.REGRESSED:
            notes.append("Probability error increased; inspect reliability by score "
                         "range before reusing the previous threshold.")
        if Change.UNAVAILABLE in changes.values():
            notes.append("Unreported metrics provide no evidence of improvement or regression.")
        return notes
