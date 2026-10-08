"""Replay ONLY the pair gate over the existing candidate universe.

Rebuilding data prep from the raw export (~71k source rows) to iterate on
one gate dimension wastes extraction/canonical work that has not changed.
The gate's inputs are frozen artifacts that already serialize everything
three_way_gate reads: data/canonical_records.csv (record-level evidence —
exactly what run_within_brand_pipeline feeds the gate) and
data/gate_results.csv (the committed candidate pair universe).

Method: rebuild the per-GTIN canonical record dict from the committed CSV
(the same dict shape gtin_to_canon carries during the real run), re-run
three_way_gate — the REAL gate, unchanged code, training_cfg() SSOT
defaults — over each committed pair in parallel chunks with per-chunk
progress, and diff the new decisions against the committed census.

PER-GATE VISIBILITY. three_way_gate's stage checks are inline, so a
replay that re-implemented "stage X only" would silently drift from the
real gate the moment either copy changed. Instead every moved pair keeps
its fired-stage verdict (FiredGate.replay old+new reasons → dimension),
so per-dimension traffic comes out of the diff rather than from a
parallel re-implementation: filter with --fired <stage>.

Usage:
  PYTHONPATH=src python -m training.gate_replay [--fired categorical_mismatch] [--workers 4]
  (--fired choices are the config gate.reasons keys)

Output:
  Fidelity (no drift): summary, exit 0.
  Drift: results/gate_replay_diff.csv — every moved pair with old+new
  decision/reason, the census per-sample degraded map at
  results/gate_census_drift.json, exit 1.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from collections import Counter

import pandas as pd

from core.common import F, RESULTS
from core.common import gate_census_drift_report, training_cfg
from core.schemas import CANONICAL_RECORDS_COLUMNS

_STR_COLUMNS = frozenset(
    {
        "canonical", "mode_brand", "mode_flavor", "mode_type",
        "salient_ngrams", "dropped_redundant_ngrams",
    }
)
_SET_COLUMNS = frozenset(
    name
    for name in CANONICAL_RECORDS_COLUMNS
    if name.endswith("_set") or name.endswith("_flags")
)
_FLOAT_COLUMNS = frozenset(
    name
    for name in CANONICAL_RECORDS_COLUMNS
    if name.endswith("_confidence") or name.endswith("_consistency")
)
_VERBATIM_COLUMNS = tuple(
    name
    for name in CANONICAL_RECORDS_COLUMNS
    if name not in (_STR_COLUMNS | _SET_COLUMNS | _FLOAT_COLUMNS)
    and name != "gtin"
)


def fired_stage(reason: str) -> str:
    """Map a gate reason string to its dimension/stage name — from config.

    The stage vocabulary IS the config rand... gate.reasons keys; the
    longest match wins (the categorical prefix also prefixes nothing else).
    """
    reasons = training_cfg().gate.reasons.model_dump()
    best_stage, best_value = "unknown", 0
    for stage, literal in reasons.items():
        value = str(literal)
        if value and reason.startswith(value) and len(value) > best_value:
            best_stage, best_value = stage, len(value)
    return best_stage


def _literal(value) -> list:
    text = str(value).strip()
    if text in ("", "[]", "{}", "set()", "frozenset()"):
        return []
    return ast.literal_eval(text)


_CANON_CACHE: dict[str, dict] | None = None
_DIFF_COLUMNS = ["gtin1", "gtin2", "old_decision", "old_reason", "new_decision", "new_reason", "similarity"]


def _bounded_chunks(pool, chunks, *, workers):
    """Keep at most two chunks per worker queued, including completed results."""
    chunks = iter(chunks)
    pending = set()
    for _ in range(2 * workers):
        chunk = next(chunks, None)
        if chunk is None:
            break
        pending.add(pool.submit(_evaluate_chunk, chunk))
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()
            chunk = next(chunks, None)
            if chunk is not None:
                pending.add(pool.submit(_evaluate_chunk, chunk))


def canonical_records_from_csv() -> dict[str, dict]:
    """Rebuild the gate-time canonical record dict from the committed CSV.

    Module-level memo: each worker process builds it once, not per pair.
    """
    global _CANON_CACHE
    if _CANON_CACHE is not None:
        return _CANON_CACHE
    can = pd.read_csv(F["canonical_records"], dtype=str, keep_default_na=False)
    records: dict[str, dict] = {}
    for row in can.itertuples(index=False):
        rec: dict = {"gtin": str(row.gtin)}
        for column in _STR_COLUMNS:
            rec[column] = getattr(row, column)
        for column in _SET_COLUMNS:
            try:
                rec[column] = set(_literal(getattr(row, column)))
            except Exception:
                rec[column] = set()
        for column in _FLOAT_COLUMNS:
            try:
                rec[column] = float(getattr(row, column) or 0.0)
            except (TypeError, ValueError):
                rec[column] = 0.0
        for column in _VERBATIM_COLUMNS:
            rec[column] = str(getattr(row, column))
        try:
            rec["n_titles"] = int(row.n_titles or 0)
        except (TypeError, ValueError):
            rec["n_titles"] = 0
        records[str(row.gtin)] = rec
    _CANON_CACHE = records
    return records


def _evaluate_chunk(chunk: list[tuple[str, str]]) -> list[dict]:
    from pipeline import three_way_gate

    canon = canonical_records_from_csv()
    out: list[dict] = []
    for gtin1, gtin2 in chunk:
        new = three_way_gate(canon[gtin1], canon[gtin2])
        out.append(
            {
                "gtin1": gtin1,
                "gtin2": gtin2,
                "new_decision": new["decision"],
                "new_reason": new["reason"],
            }
        )
    return out


def replay(*, fired: str = "", workers: int = 0):
    if workers <= 0:
        import multiprocessing

        workers = min(4, max(1, multiprocessing.cpu_count() - 1))
    gate_path = RESULTS / F["gate_results"]
    gr = pd.read_csv(gate_path, dtype=str, keep_default_na=False)
    if gr.empty:
        return 0, 0, pd.DataFrame(columns=_DIFF_COLUMNS)
    if gr.duplicated(["gtin1", "gtin2"]).any():
        raise ValueError("committed candidate universe contains duplicate pairs")
    workers = min(workers, len(gr))
    pairs = list(zip(gr.gtin1.tolist(), gr.gtin2.tolist()))
    canonical_records_from_csv()

    chunk_size = max(64, min(2_000, len(pairs) // workers))
    chunk_count = (len(pairs) + chunk_size - 1) // chunk_size
    chunks = (pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size))
    from tqdm import tqdm

    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(
            _bounded_chunks(pool, chunks, workers=workers), total=chunk_count, desc="replay chunks"
        ):
            frames.append(pd.DataFrame(result))
    replayed = pd.concat(frames, ignore_index=True)

    merged = gr.merge(
        replayed,
        on=["gtin1", "gtin2"],
        how="left",
        suffixes=("", "_replayed"),
        validate="one_to_one",
    )
    if merged.new_reason.isna().any():
        raise RuntimeError(
            f"{int(merged.new_reason.isna().sum())} committed pairs missing a "
            "replayed verdict — pair universe and canonical coverage diverged"
        )
    changed = merged.gate_decision.ne(merged.new_decision) | merged.gate_reason.ne(
        merged.new_reason
    )
    same = int((~changed).sum())
    total = len(merged)
    moved = merged.loc[changed].copy()
    moved = moved.rename(
        columns={"gate_decision": "old_decision", "gate_reason": "old_reason"}
    )[
        ["gtin1", "gtin2", "old_decision", "old_reason", "new_decision", "new_reason", "similarity"]
    ]
    if fired:
        keep = moved.new_reason.map(fired_stage).eq(fired) | moved.old_reason.map(
            fired_stage
        ).eq(fired)
        print(
            f"[replay] fired filter {fired!r}: {int(keep.sum()):,} of "
            f"{total - same:,} moved pairs involve that gate"
        )
    # Return the complete diff: filtered display must never change fidelity
    # or the global census written by main.
    return same, total, moved


def _recount(gr: pd.DataFrame, moved: pd.DataFrame) -> dict[str, int]:
    new_decisions = _current_gate(gr, moved).gate_decision
    return {
        "hard_no": int((new_decisions == "hard_no").sum()),
        "proceed": int((new_decisions == "proceed").sum()),
        "fallback": int((new_decisions == "fallback").sum()),
    }


def _current_gate(gr: pd.DataFrame, moved: pd.DataFrame) -> pd.DataFrame:
    """Apply every replayed outcome while preserving the candidate universe."""
    current = gr.copy()
    if moved.empty:
        return current
    updates = moved.set_index(["gtin1", "gtin2"])
    keys = pd.MultiIndex.from_frame(current[["gtin1", "gtin2"]])
    for destination, source in (("gate_decision", "new_decision"), ("gate_reason", "new_reason")):
        values = updates[source].reindex(keys)
        mask = values.notna().to_numpy()
        current.loc[mask, destination] = values.to_numpy()[mask]
    return current


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fired", default="", help="show only moved pairs whose old or new reason fired this gate")
    ap.add_argument("--workers", type=int, default=0, help="parallel workers (0 = up to 4 CPUs)")
    args = ap.parse_args()
    same, total, moved = replay(fired=args.fired, workers=args.workers)
    print(f"[replay] {same:,}/{total:,} byte-identical decisions")
    if same == total:
        print("[replay] FIDELITY PASS — replayed census matches the committed one exactly")
        return 0
    displayed = moved
    if args.fired:
        displayed = moved.loc[moved.new_reason.map(fired_stage).eq(args.fired) | moved.old_reason.map(fired_stage).eq(args.fired)]
    matrix = displayed.groupby(["old_decision", "new_decision"]).size()
    print("[replay] decision moves:")
    for (old, new), count in matrix.sort_values(ascending=False).items():
        print(f"    {str(old):>9} -> {str(new):<9} {count:,}")
    stage_moves = Counter(
        (fired_stage(r.old_reason), fired_stage(r.new_reason))
        for r in displayed.itertuples(index=False)
    )
    print("[replay] which gate moved each pair (top 8):")
    for (old_stage, new_stage), count in stage_moves.most_common(8):
        print(f"    {old_stage:>40} -> {new_stage:<40} {count:,}")
    diff_path = RESULTS / "gate_replay_diff.csv"
    moved.to_csv(diff_path, index=False)
    print(f"[replay] full per-pair diff: {diff_path.as_posix()}")
    gr = pd.read_csv(RESULTS / F["gate_results"], dtype=str, keep_default_na=False)
    measured = {"total_pairs": total, **_recount(gr, moved)}
    report = gate_census_drift_report(measured=measured, current_gate=_current_gate(gr, moved))
    print(
        "[replay] per-sample degraded map: results/gate_census_drift.json — "
        f"degraded {report['degraded']}, survived {report['survived']:,}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
