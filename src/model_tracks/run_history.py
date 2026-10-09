"""Run-history layer — trace and compare every generated run for regression analysis.

Two run sources share one facts model:
- bundle runs: results/training_prep/<id>/ folders carrying manifest.json
  (the offline preparation pipeline's per-run record);
- training runs: training_results/<id>/ recognized by the retention layer's
  track-marker logic (model_tracks.run_retention._looks_like_run — reused,
  never duplicated).

Facts are read fail-loud: every file that IS read must parse, with the
precise path in the error. Optional per-run artifacts (handoff.json,
timing_offenders.log, labeled_pairs.log, ...) are None/absent when the run
never wrote them — recorded absence, never a guessed value.

Comparison is regression-focused and scale-aware: stage seconds are shown
alongside seconds_per_thousand_rows_normalized (using each side's own
dataset_rows when both are known), output counts become per-1000-rows rates,
and the regressions list names every stage whose per-thousand-rows time
GREW, every output count whose per-1000-rows rate FELL, and every contract
that switched pass->fail. Runs of different overall status are marked
'incomplete_pair' — a half-finished run is not a regression witness.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.timing import Timing, emit_timing

_RUN_ID_PATTERN = re.compile(r"^(\d{8}T\d{6})(\d{6})$")
_DATASET_ROWS_PATTERNS = (
    re.compile(r"sku_to_rep\.csv \(([\d,]+) rows\)"),
    re.compile(r"^([\d,]+) rows -> ", re.MULTILINE),
)
_LABELED_PATTERN = re.compile(
    r"labeled_pairs\.csv: ([\d,]+) rows \(([\d,]+) pos / ([\d,]+) hard-neg\)")
_OFFENDER_PATTERN = re.compile(
    r"^\s*\d+\.\s+(.+?)\s{2,}([\d.]+)s\s+([\d.]+)%$")
_OFFENDER_TOP = 5
# Stages shorter than this are noise-prone (a 10 ms stage can double its
# per-1k rate from rounding alone); growth is only a regression from here up.
_STAGE_REGRESSION_FLOOR_SECONDS = 1.0


def _preparation():
    """The config-owned preparation run contract (name/path SSOT)."""
    from core.common import training_cfg
    return training_cfg().preparation


class StageFacts(BaseModel):
    """One preparation stage as the run's manifest recorded it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    seconds: float | None = None
    status: str | None = None
    returncode: int | None = None
    started_at: str | None = None
    finished_at: str | None = None


class OffenderFacts(BaseModel):
    """One timed surface from the run's worst-offender report."""

    model_config = ConfigDict(extra="forbid")

    label: str
    seconds: float = Field(ge=0.0)


class CensusFacts(BaseModel):
    """Gate-census pair counts (gate_census)."""

    model_config = ConfigDict(extra="forbid")

    total_pairs: int | None = None
    hard_no: int | None = None
    proceed: int | None = None
    fallback: int | None = None


class LabeledPairCounts(BaseModel):
    """Labeled-pair output counts from the run's labeled_pairs.log."""

    model_config = ConfigDict(extra="forbid")

    kept: int
    pos: int
    hard_neg: int


class HandoffSummary(BaseModel):
    """The run's handoff.json: consumer-boundary readiness evidence."""

    model_config = ConfigDict(extra="forbid")

    status: str
    metered_inputs: int = Field(ge=0)
    loss_batch_attested: bool
    loss_batch: dict[str, Any] | None = None


class BundleRatios(BaseModel):
    """Bundle manifest ratios recorded by the run (bundle header)."""

    model_config = ConfigDict(extra="forbid")

    static_view_ratio: float | None = None
    effective_train_ratio: float | None = None
    n_pos: int | None = None
    n_neg: int | None = None


class RunFacts(BaseModel):
    """Everything the trace layer can state about one run, with sources."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["bundle", "training"]
    run_id: str
    dir: str
    created: str | None = None
    status: str | None = None
    dataset_rows: int | None = None
    stages: list[StageFacts] = Field(default_factory=list)
    census: CensusFacts | None = None
    labeled: LabeledPairCounts | None = None
    minted_rows: int | None = None
    bundle_ratios: BundleRatios | None = None
    offenders_top: list[OffenderFacts] = Field(default_factory=list)
    offenders_skipped: int = 0
    handoff: HandoffSummary | None = None

    @property
    def total_stage_seconds(self) -> float:
        return round(sum(stage.seconds or 0.0 for stage in self.stages), 3)

    def stage_seconds_per_1k(self, stage: str) -> float | None:
        """Seconds per 1000 dataset rows for one stage (None when unscalable)."""
        seconds = next((s.seconds for s in self.stages if s.name == stage), None)
        if seconds is None or self.dataset_rows in (None, 0):
            return None
        return seconds * 1000.0 / self.dataset_rows

    def output_per_1k(self, count: int | None) -> float | None:
        if count is None or self.dataset_rows in (None, 0):
            return None
        return count * 1000.0 / self.dataset_rows


class StageCompare(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    a_seconds: float | None
    b_seconds: float | None
    a_per_1k: float | None
    b_per_1k: float | None
    per_1k_ratio: float | None = None
    a_status: str | None = None
    b_status: str | None = None
    regression: bool | None = None


class CensusCompare(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    a: int | None
    b: int | None
    a_per_1k: float | None
    b_per_1k: float | None
    per_1k_ratio: float | None = None


class OutputCompare(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    a: int | None
    b: int | None
    a_per_1k: float | None
    b_per_1k: float | None
    per_1k_ratio: float | None = None
    regression: bool | None = None


class ContractCompare(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    a: str
    b: str
    switched_pass_to_fail: bool = False


class RegressionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["stage_time", "output_rate", "contract"]
    detail: str


class ComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a_id: str
    b_id: str
    pair_kind: Literal["complete_pair", "incomplete_pair"]
    dataset_rows_a: int | None
    dataset_rows_b: int | None
    scale_normalized: bool
    stages: list[StageCompare]
    census: list[CensusCompare]
    outputs: list[OutputCompare]
    contracts: list[ContractCompare]
    worst_offender_a: OffenderFacts | None
    worst_offender_b: OffenderFacts | None
    worst_offender_drift: bool
    offenders_top: list[tuple[str, float | None, float | None]] = Field(
        default_factory=list,
        description="Merged top offender labels with each side's seconds",
    )
    regressions: list[RegressionItem] = Field(default_factory=list)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except ValueError as error:
        raise ValueError(f"run-history: unreadable JSON {path}: {error}") from error


def _parse_rows(text: str) -> int:
    return int(text.replace(",", ""))


def _created_from_id(run_id: str) -> str | None:
    match = _RUN_ID_PATTERN.match(run_id)
    if not match:
        return None
    moment = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc, microsecond=int(match.group(2)))
    return moment.isoformat()


def _stage_facts(manifest: dict, run_dir: Path) -> list[StageFacts]:
    """Stage metrics from the manifest; timings.json backs older manifests."""
    metrics = manifest.get("stage_metrics")
    if isinstance(metrics, dict) and metrics:
        return [
            StageFacts(
                name=name,
                seconds=(entry or {}).get("seconds"),
                status=(entry or {}).get("status"),
                returncode=(entry or {}).get("returncode"),
                started_at=(entry or {}).get("started_at"),
                finished_at=(entry or {}).get("finished_at"),
            )
            for name, entry in sorted(metrics.items())
        ]
    seconds = manifest.get("stage_seconds")
    if isinstance(seconds, dict) and seconds:
        return [StageFacts(name=name, seconds=value)
                for name, value in sorted(seconds.items())]
    timings_path = run_dir / _preparation().timings_file
    if timings_path.exists():
        stages = _read_json(timings_path).get("stages", {})
        return [
            StageFacts(
                name=name,
                seconds=(entry or {}).get("seconds"),
                status=(entry or {}).get("status"),
                returncode=(entry or {}).get("returncode"),
            )
            for name, entry in sorted(stages.items())
        ]
    return []


def _census_facts(manifest: dict, run_dir: Path) -> CensusFacts | None:
    raw = manifest.get("gate_census")
    if not isinstance(raw, dict):
        census_path = run_dir / _preparation().gate_census_file
        if not census_path.exists():
            return None
        raw = _read_json(census_path)
    return CensusFacts(
        total_pairs=raw.get("total_pairs"),
        hard_no=raw.get("hard_no"),
        proceed=raw.get("proceed"),
        fallback=raw.get("fallback"),
    )


def _labeled_facts(run_dir: Path) -> LabeledPairCounts | None:
    log_path = run_dir / "labeled_pairs.log"
    if not log_path.exists():
        return None
    match = _LABELED_PATTERN.search(log_path.read_text())
    if not match:
        raise ValueError(
            f"run-history: {log_path} carries no labeled_pairs.csv summary line")
    kept, pos, hard_neg = (_parse_rows(group) for group in match.groups())
    return LabeledPairCounts(kept=kept, pos=pos, hard_neg=hard_neg)


def _minted_rows(run_dir: Path) -> int | None:
    discriminator_path = run_dir / _preparation().discriminator_file
    if discriminator_path.exists():
        value = _read_json(discriminator_path).get("minted_rows")
        return int(value) if value is not None else None
    supply_path = run_dir / "negative_supply.log"
    if not supply_path.exists():
        return None
    match = re.search(r'"minted_partner":\s*(\d+)', supply_path.read_text())
    return int(match.group(1)) if match else None


def _dataset_rows(run_dir: Path) -> int | None:
    """The run's own dataset-row audit pin, from its dedupe.log.

    The run records the dataset it started from ("wrote ... sku_to_rep.csv
    (71,623 rows)"). No run-side record, no number: None, never guessed.
    """
    log_path = run_dir / "dedupe.log"
    if not log_path.exists():
        return None
    text = log_path.read_text()
    for pattern in _DATASET_ROWS_PATTERNS:
        match = pattern.search(text)
        if match:
            return _parse_rows(match.group(1))
    return None


def _offenders(run_dir: Path) -> tuple[list[OffenderFacts], int]:
    """Top offenders from timing_offenders.log; timings.json when absent.

    Returns (offenders, skipped) where skipped counts unparseable lines that
    the report log carried but this reader could not decode — recorded, not
    silently dropped.
    """
    log_path = run_dir / _preparation().offender_report
    if log_path.exists():
        offenders = []
        skipped = 0
        for line in log_path.read_text().splitlines():
            match = _OFFENDER_PATTERN.match(line)
            if match:
                offenders.append(OffenderFacts(
                    label=match.group(1), seconds=float(match.group(2))))
            elif line and not line.startswith("#"):
                skipped += 1
        if offenders:
            return offenders[:_OFFENDER_TOP], skipped
        if skipped:
            # The run's own offender report exists but carries nothing
            # decodable: recorded absence, never a guessed value.
            return [], skipped
    timings_path = run_dir / _preparation().timings_file
    if timings_path.exists():
        stages = _read_json(timings_path).get("stages", {})
        ranked = sorted(
            (OffenderFacts(label=f"stage/{name}",
                           seconds=float((entry or {}).get("seconds") or 0.0))
             for name, entry in stages.items()),
            key=lambda offender: offender.seconds, reverse=True)
        return ranked[:_OFFENDER_TOP], 0
    return [], 0


def _handoff(run_dir: Path) -> HandoffSummary | None:
    handoff_path = run_dir / _preparation().handoff_file
    if not handoff_path.exists():
        return None
    report = _read_json(handoff_path)
    attestation = report.get("loss_batch_correctness")
    return HandoffSummary(
        # A handoff that never recorded a status is not 'pass' and not a
        # guessed value: recorded absence under a precise sentinel name.
        status=str(report.get("status") or "missing_status"),
        metered_inputs=len(report.get("inputs") or []),
        loss_batch_attested=attestation is not None,
        loss_batch=attestation,
    )


def _bundle_ratios(manifest: dict) -> BundleRatios | None:
    bundle = manifest.get("bundle")
    if not isinstance(bundle, dict):
        return None
    return BundleRatios(
        static_view_ratio=bundle.get("static_view_ratio"),
        effective_train_ratio=bundle.get("effective_train_ratio"),
        n_pos=bundle.get("n_pos"),
        n_neg=bundle.get("n_neg"),
    )


def bundle_facts(run_dir: Path) -> RunFacts:
    """Facts for one bundle run (results/training_prep/<id>)."""
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"run-history: bundle run {run_dir.name} has no manifest.json "
            f"(looked for {manifest_path})")
    with Timing("run_history.facts").section(f"bundle:{run_dir.name}"):
        manifest = _read_json(manifest_path)
        offenders, offenders_skipped = _offenders(run_dir)
        facts = RunFacts(
            source="bundle",
            run_id=run_dir.name,
            dir=str(run_dir),
            created=_created_from_id(run_dir.name),
            status=manifest.get("status"),
            dataset_rows=_dataset_rows(run_dir),
            stages=_stage_facts(manifest, run_dir),
            census=_census_facts(manifest, run_dir),
            labeled=_labeled_facts(run_dir),
            minted_rows=_minted_rows(run_dir),
            bundle_ratios=_bundle_ratios(manifest),
            offenders_top=offenders,
            offenders_skipped=offenders_skipped,
            handoff=_handoff(run_dir),
        )
    emit_timing(f"[timing] run_history facts bundle={run_dir.name} "
                f"stages={len(facts.stages)} rows={facts.dataset_rows}")
    return facts


def training_run_facts(run_dir: Path) -> RunFacts:
    """Facts for one training run (training_results/<id>, marker-verified)."""
    from model_tracks.run_retention import _looks_like_run

    run_dir = Path(run_dir).resolve()
    if not _looks_like_run(run_dir):
        raise FileNotFoundError(
            f"run-history: {run_dir} is not a completed training run "
            f"(no track-subdir run markers)")
    with Timing("run_history.facts").section(f"training:{run_dir.name}"):
        facts = RunFacts(
            source="training",
            run_id=run_dir.name,
            dir=str(run_dir),
            created=_created_from_id(run_dir.name),
            status=None,
        )
    emit_timing(f"[timing] run_history facts training={run_dir.name}")
    return facts


def _default_prep_roots() -> list[Path]:
    from core.common import TRAIN_ROOT

    return [TRAIN_ROOT / "results" / _preparation().run_dir_base]


def _default_training_roots() -> list[Path]:
    from core.common import TRAIN_ROOT

    return [TRAIN_ROOT / "training_results"]


def list_bundles(prep_roots: Iterable[Path] | None = None) -> list[Path]:
    """Bundle-run directories carrying manifest.json, oldest first."""
    roots = list(prep_roots) if prep_roots is not None else _default_prep_roots()
    timing = Timing("run_history.list")
    runs: list[Path] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"run-history: bundle root missing: {root}")
        with timing.section(f"list:{root}"):
            runs.extend(sorted(
                entry for entry in root.iterdir()
                if entry.is_dir() and (entry / "manifest.json").exists()))
    emit_timing(f"[timing] run_history list bundles={len(runs)} roots={len(roots)}")
    return runs


def list_training_runs(training_roots: Iterable[Path] | None = None) -> list[Path]:
    """Completed training runs via the retention layer's marker logic."""
    from model_tracks.run_retention import _looks_like_run

    roots = list(training_roots) if training_roots is not None else _default_training_roots()
    timing = Timing("run_history.list")
    runs: list[Path] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"run-history: training-results root missing: {root}")
        with timing.section(f"list:{root}"):
            runs.extend(sorted(
                entry for entry in root.iterdir()
                if entry.is_dir() and _looks_like_run(entry)))
    emit_timing(f"[timing] run_history list training_runs={len(runs)} roots={len(roots)}")
    return runs


def facts(run_id_or_dir: str, *, prep_roots: Iterable[Path] | None = None,
          training_roots: Iterable[Path] | None = None) -> RunFacts:
    """Facts for a run id or directory, resolved against the known roots.

    A directory path is used directly (bundle facts when manifest.json is
    present, training-run facts when retention markers say so). An id is
    searched in every bundle root, then every training root; unknown ids
    fail loudly with every path that was tried.
    """
    candidate = Path(run_id_or_dir)
    if candidate.is_dir():
        if (candidate / "manifest.json").exists():
            return bundle_facts(candidate)
        return training_run_facts(candidate)
    bundle_dirs = {entry.name: entry for entry in list_bundles(prep_roots)}
    if run_id_or_dir in bundle_dirs:
        return bundle_facts(bundle_dirs[run_id_or_dir])
    training_dirs = {entry.name: entry for entry in list_training_runs(training_roots)}
    if run_id_or_dir in training_dirs:
        return training_run_facts(training_dirs[run_id_or_dir])
    searched = ([str(entry) for entry in sorted(bundle_dirs.values())]
                + [str(entry) for entry in sorted(training_dirs.values())])
    raise FileNotFoundError(
        f"run-history: unknown run id {run_id_or_dir!r}; "
        f"known runs: {searched or 'none'}")


def _per_1k_ratio(a: float | None, b: float | None) -> float | None:
    if a in (None, 0) or b is None:
        return None
    return b / a


def compare(a: RunFacts, b: RunFacts) -> ComparisonReport:
    """Regression-focused, scale-aware comparison of two runs.

    A pair with ANY training-run side is treated with incomplete_pair
    semantics for regressions: training runs publish no stage/census/output
    facts to this layer, so a stage-time or output-rate claim would be a
    guess. The renderable facts (stages shown, worst offender) stay.
    """
    with Timing("run_history.compare").section(f"{a.run_id}->{b.run_id}"):
        pair_kind: Literal["complete_pair", "incomplete_pair"] = (
            "complete_pair" if a.status == b.status else "incomplete_pair")
        if "training" in (a.source, b.source) and pair_kind == "complete_pair":
            pair_kind = "incomplete_pair"
        scale_normalized = (a.dataset_rows is not None and b.dataset_rows is not None
                            and a.dataset_rows > 0 and b.dataset_rows > 0)

        stage_names = sorted({s.name for s in a.stages} | {s.name for s in b.stages})
        stage_rows: list[StageCompare] = []
        for name in stage_names:
            a_stage = next((s for s in a.stages if s.name == name), None)
            b_stage = next((s for s in b.stages if s.name == name), None)
            a_per_1k = a.stage_seconds_per_1k(name)
            b_per_1k = b.stage_seconds_per_1k(name)
            ratio = _per_1k_ratio(a_per_1k, b_per_1k)
            regression = (None if pair_kind == "incomplete_pair" or ratio is None
                          else (b_stage is not None and b_stage.seconds is not None
                                and b_stage.seconds >= _STAGE_REGRESSION_FLOOR_SECONDS
                                and ratio > 1.0))
            stage_rows.append(StageCompare(
                stage=name,
                a_seconds=a_stage.seconds if a_stage else None,
                b_seconds=b_stage.seconds if b_stage else None,
                a_per_1k=a_per_1k, b_per_1k=b_per_1k, per_1k_ratio=ratio,
                a_status=a_stage.status if a_stage else None,
                b_status=b_stage.status if b_stage else None,
                regression=regression,
            ))

        census_keys = ("total_pairs", "hard_no", "proceed", "fallback")
        census_rows: list[CensusCompare] = []
        if a.census and b.census:
            for key in census_keys:
                a_per_1k = a.output_per_1k(getattr(a.census, key))
                b_per_1k = b.output_per_1k(getattr(b.census, key))
                census_rows.append(CensusCompare(
                    key=key, a=getattr(a.census, key), b=getattr(b.census, key),
                    a_per_1k=a_per_1k, b_per_1k=b_per_1k,
                    per_1k_ratio=_per_1k_ratio(a_per_1k, b_per_1k),
                ))

        def _outputs(run: RunFacts) -> dict[str, int | None]:
            return {
                "labeled_pos": run.labeled.pos if run.labeled else None,
                "labeled_hard_neg": run.labeled.hard_neg if run.labeled else None,
                "labeled_kept": run.labeled.kept if run.labeled else None,
                "minted": run.minted_rows,
            }

        a_outputs, b_outputs = _outputs(a), _outputs(b)
        output_rows: list[OutputCompare] = []
        for name in sorted(set(a_outputs) | set(b_outputs)):
            a_count, b_count = a_outputs.get(name), b_outputs.get(name)
            a_per_1k = a.output_per_1k(a_count)
            b_per_1k = b.output_per_1k(b_count)
            ratio = _per_1k_ratio(a_per_1k, b_per_1k)
            regression = (None if pair_kind == "incomplete_pair" or ratio is None
                          else ratio < 1.0)
            output_rows.append(OutputCompare(
                name=name, a=a_count, b=b_count,
                a_per_1k=a_per_1k, b_per_1k=b_per_1k, per_1k_ratio=ratio,
                regression=regression,
            ))

        contracts: list[ContractCompare] = []
        for row in stage_rows:
            if row.a_status is not None and row.b_status is not None \
                    and row.a_status != row.b_status:
                contracts.append(ContractCompare(
                    name=f"stage/{row.stage}", a=row.a_status, b=row.b_status,
                    switched_pass_to_fail=(
                        row.a_status in ("complete", "pass")
                        and row.b_status not in ("complete", "pass", None)),
                ))
        handoff_a = a.handoff.status if a.handoff else None
        handoff_b = b.handoff.status if b.handoff else None
        if handoff_a != handoff_b:
            contracts.append(ContractCompare(
                name="handoff", a=handoff_a or "absent",
                b=handoff_b or "absent",
                switched_pass_to_fail=(handoff_a == "pass" and handoff_b != "pass"),
            ))

        worst_a = a.offenders_top[0] if a.offenders_top else None
        worst_b = b.offenders_top[0] if b.offenders_top else None
        offender_labels = [o.label for o in a.offenders_top]
        offender_labels += [o.label for o in b.offenders_top
                            if o.label not in offender_labels]
        a_seconds = {o.label: o.seconds for o in a.offenders_top}
        b_seconds = {o.label: o.seconds for o in b.offenders_top}
        offender_rows = [(label, a_seconds.get(label), b_seconds.get(label))
                         for label in offender_labels[:_OFFENDER_TOP]]

        regressions: list[RegressionItem] = []
        if pair_kind == "complete_pair":
            regressions.extend(
                RegressionItem(kind="stage_time", detail=(
                    f"{row.stage}: {row.a_per_1k:.3f} -> {row.b_per_1k:.3f} "
                    f"s/1k rows (x{row.per_1k_ratio:.2f}); raw "
                    f"{row.a_seconds}s -> {row.b_seconds}s"))
                for row in stage_rows if row.regression)
            regressions.extend(
                RegressionItem(kind="output_rate", detail=(
                    f"{row.name}: per-1k output rate "
                    f"{row.a_per_1k:.3f} -> {row.b_per_1k:.3f} "
                    f"(x{row.per_1k_ratio:.2f}); raw {row.a} -> {row.b}"))
                for row in output_rows if row.regression)
            regressions.extend(
                RegressionItem(kind="contract", detail=(
                    f"{row.name}: {row.a} -> {row.b}"))
                for row in contracts if row.switched_pass_to_fail)
    emit_timing(
        f"[timing] run_history compare {a.run_id} -> {b.run_id} "
        f"pair={pair_kind} regressions={len(regressions)}")
    return ComparisonReport(
        a_id=a.run_id, b_id=b.run_id, pair_kind=pair_kind,
        dataset_rows_a=a.dataset_rows, dataset_rows_b=b.dataset_rows,
        scale_normalized=scale_normalized,
        stages=stage_rows, census=census_rows, outputs=output_rows,
        contracts=contracts,
        worst_offender_a=worst_a, worst_offender_b=worst_b,
        worst_offender_drift=(worst_a is not None and worst_b is not None
                              and worst_a.label != worst_b.label),
        offenders_top=offender_rows,
        regressions=regressions,
    )


def _fmt(value: float | int | None, spec: str = ",.0f") -> str:
    return "—" if value is None else format(value, spec)


def _print_facts_line(run: RunFacts) -> None:
    labeled = (f"{run.labeled.kept:,} ({run.labeled.pos:,} pos / "
               f"{run.labeled.hard_neg:,} neg)" if run.labeled else "—")
    worst = run.offenders_top[0] if run.offenders_top else None
    if run.handoff is None:
        handoff = "absent"
    else:
        handoff = (f"present ({run.handoff.status}, "
                   f"{run.handoff.metered_inputs} metered inputs, "
                   f"loss/batch attested: {run.handoff.loss_batch_attested})")
    print(
        f"{run.run_id} [{run.source}] status={run.status or '—'} "
        f"created={run.created or '—'} dataset_rows={_fmt(run.dataset_rows)} "
        f"stages={len(run.stages)} total_stage_seconds="
        f"{_fmt(run.total_stage_seconds, ',.1f')} "
        f"census={_fmt(run.census.total_pairs if run.census else None)} "
        f"labeled={labeled} minted={_fmt(run.minted_rows)} "
        f"handoff={handoff} "
        f"worst_offender="
        f"{f'{worst.label} ({worst.seconds}s)' if worst else '—'}"
        f"{f' +{run.offenders_skipped} unparseable offender lines' if run.offenders_skipped else ''}\n"
        f"  dir: {run.dir}")


def _print_compare(report: ComparisonReport) -> None:
    print(f"compare {report.a_id} -> {report.b_id}  pair={report.pair_kind} "
          f"scale_normalized={'yes' if report.scale_normalized else 'no'} "
          f"(dataset_rows {_fmt(report.dataset_rows_a)} vs "
          f"{_fmt(report.dataset_rows_b)})")
    print("stages: raw seconds | seconds per 1000 dataset rows "
          "(REGRESSION when b's per-1k time grew)")
    for row in report.stages:
        if row.regression:
            flag = "  REGRESSION"
        elif row.regression is False and row.per_1k_ratio is not None \
                and row.per_1k_ratio < 1.0:
            flag = "  improved"
        else:
            flag = ""
        print(f"  {row.stage:<24} {_fmt(row.a_seconds, ',.1f')} -> "
              f"{_fmt(row.b_seconds, ',.1f')} s | "
              f"{_fmt(row.a_per_1k, '.3f')} -> {_fmt(row.b_per_1k, '.3f')} s/1k"
              f"{f' (x{row.per_1k_ratio:.2f})' if row.per_1k_ratio is not None else ''}"
              f"{flag}")
    if report.census:
        print("census (gate pairs, per-1000-rows):")
        for row in report.census:
            print(f"  {row.key:<12} {_fmt(row.a)} -> {_fmt(row.b)} | "
                  f"{_fmt(row.a_per_1k, '.3f')} -> {_fmt(row.b_per_1k, '.3f')} per-1k"
                  f"{f' (x{row.per_1k_ratio:.2f})' if row.per_1k_ratio is not None else ''}")
    if report.outputs:
        print("outputs per-1000-rows (REGRESSION when the rate fell):")
        for row in report.outputs:
            if row.regression:
                flag = "  REGRESSION (rate fell)"
            elif row.regression is False and row.per_1k_ratio is not None \
                    and row.per_1k_ratio > 1.0:
                flag = "  improved"
            else:
                flag = ""
            print(f"  {row.name:<16} {_fmt(row.a)} -> {_fmt(row.b)} | "
                  f"{_fmt(row.a_per_1k, '.3f')} -> {_fmt(row.b_per_1k, '.3f')} per-1k"
                  f"{f' (x{row.per_1k_ratio:.2f})' if row.per_1k_ratio is not None else ''}"
                  f"{flag}")
    print(f"worst offender: {report.a_id} "
          f"{report.worst_offender_a.label if report.worst_offender_a else '—'} "
          f"({_fmt(report.worst_offender_a.seconds, ',.1f') if report.worst_offender_a else '—'}s)"
          f" -> {report.b_id} "
          f"{report.worst_offender_b.label if report.worst_offender_b else '—'} "
          f"({_fmt(report.worst_offender_b.seconds, ',.1f') if report.worst_offender_b else '—'}s)"
          f"{' — DRIFTED' if report.worst_offender_drift else ''}")
    if report.offenders_top:
        print("offenders top-5 (label | a s | b s):")
        for label, a_seconds, b_seconds in report.offenders_top:
            print(f"  {label:<52} {_fmt(a_seconds, ',.1f')} | {_fmt(b_seconds, ',.1f')}")
    if report.pair_kind == "incomplete_pair":
        print("incomplete_pair: runs of different status are NOT compared as "
              "regressions — finish or re-run one side first")
    if report.regressions:
        print(f"regressions ({len(report.regressions)}):")
        for item in report.regressions:
            print(f"  [{item.kind}] {item.detail}")
    else:
        print("regressions (0): none")


def main() -> None:
    """CLI: list | facts <run_id> | compare <idA> <idB>."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "facts", "compare"])
    parser.add_argument("run_id", nargs="?", default=None)
    parser.add_argument("run_id_b", nargs="?", default=None)
    parser.add_argument("--prep-root", action="append", default=None,
                        help="results/training_prep root (repeatable)")
    parser.add_argument("--training-root", action="append", default=None,
                        help="training_results root (repeatable)")
    arguments = parser.parse_args()
    prep_roots = [Path(p) for p in arguments.prep_root] if arguments.prep_root else None
    training_roots = ([Path(p) for p in arguments.training_root]
                      if arguments.training_root else None)
    if arguments.command == "list":
        bundles = list_bundles(prep_roots)
        training_runs = list_training_runs(training_roots)
        print(f"bundle runs ({len(bundles)}):")
        for entry in bundles:
            print(f"  {entry.name}  {entry}")
        print(f"training runs ({len(training_runs)}):")
        for entry in training_runs:
            print(f"  {entry.name}  {entry}")
    elif arguments.command == "facts":
        if arguments.run_id is None:
            raise SystemExit("facts requires a run id (see `list`)")
        _print_facts_line(facts(arguments.run_id, prep_roots=prep_roots,
                                training_roots=training_roots))
    elif arguments.command == "compare":
        if arguments.run_id is None or arguments.run_id_b is None:
            raise SystemExit("compare requires two run ids (see `list`)")
        report = compare(
            facts(arguments.run_id, prep_roots=prep_roots,
                  training_roots=training_roots),
            facts(arguments.run_id_b, prep_roots=prep_roots,
                  training_roots=training_roots))
        _print_compare(report)


if __name__ == "__main__":
    main()
