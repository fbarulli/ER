"""Bare-bones FastAPI dashboard + per-experiment pages across experiment series.

The UI is multi-series: any folder under the configured experiment root that contains a
least one numbered ``NN_*.py`` step script is a series, addressed by its id
(the folder path relative to that root) via the ``?focus=<series id>`` query
parameter on every page. The default is the first discovered series unless
``BROADWAY_DEFAULT_EXPERIMENT_SERIES`` selects one.

To add a series: drop a folder with numbered ``NN_*.py`` scripts under the
configured experiment root; its results live under that root's ``results`` directory and
its dashboard becomes reachable at ``/?focus=<series id>``.

Each step page renders a bare-bones HTML pipeline graph (upstream -> this
step -> downstream consumers of the datasets it writes), the step's result
artifacts, and a form for a human observation plus a required verdict,
persisted to ``artifacts/experiments/observations/<stem>.json``. No
templates, no external assets. The series view has one small inline script
for drag-and-drop step reordering: a reorder renumbers the involved scripts
(window-scoped) so filename order stays the source of truth.
"""

import ast
import html
import json
import logging
import mimetypes
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, quote

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

EXPERIMENTS_ROOT = Path(os.environ.get("BROADWAY_EXPERIMENTS_ROOT", "experiments"))
DEFAULT_SERIES = os.environ.get("BROADWAY_DEFAULT_EXPERIMENT_SERIES", "")
OBSERVATIONS_DIR = Path(os.environ.get("BROADWAY_OBSERVATIONS_DIR", "artifacts/experiments/observations"))

_PARQUET_CONSTANTS = ("CLEAN_PARQUET", "FULL_PARQUET", "RATECODE1_PARQUET")
_EXTERNAL_RE = re.compile(r"read_training_sample|load_metered|load_working")
_NUMBER_PREFIX = re.compile(r"^\d+[a-z]?:\s*")
_STEM_PREFIX = re.compile(r"^\d+")
_NN_PREFIX = re.compile(r"^\d")
_KEY_RE = re.compile(r"^(\d+)([a-z]*)_(.*)$")
_SLUG_RE = re.compile(r"[a-z0-9_]+")
_VERDICTS = ("supported", "refuted", "inconclusive", "partial")
_UPSTREAM_LABEL = "upstream (project data / working dataset)"

app = FastAPI()


def configure_from_env() -> None:
    """Re-read dashboard env and mount static surfaces (idempotent).

    Import-time globals freeze the first-import env, but the project launcher
    sets ``BROADWAY_*`` after importing this module to build canvas specs.
    Call this after the env is final so ``/results`` + ``/diagrams`` serve
    the configured directories even when the module was imported early.
    """
    global EXPERIMENTS_ROOT, DEFAULT_SERIES, OBSERVATIONS_DIR, DIAGRAMS_DIR
    EXPERIMENTS_ROOT = Path(os.environ.get("BROADWAY_EXPERIMENTS_ROOT", "experiments"))
    DEFAULT_SERIES = os.environ.get("BROADWAY_DEFAULT_EXPERIMENT_SERIES", "")
    OBSERVATIONS_DIR = Path(
        os.environ.get("BROADWAY_OBSERVATIONS_DIR", "artifacts/experiments/observations")
    )
    DIAGRAMS_DIR = Path(os.environ.get("BROADWAY_DIAGRAMS_DIR", "diagrams"))
    mounted = {getattr(route, "path", "") for route in app.routes}
    # Results are experiment-scratch and may be absent (e.g. a platform-only
    # checkout) — only mount the static surface when the directory exists.
    if (EXPERIMENTS_ROOT / "results").is_dir() and "/results" not in mounted:
        app.mount(
            "/results", StaticFiles(directory=EXPERIMENTS_ROOT / "results"), name="results"
        )
    if DIAGRAMS_DIR.is_dir() and "/diagrams" not in mounted:
        app.mount("/diagrams", StaticFiles(directory=DIAGRAMS_DIR), name="diagrams")


# Results are experiment-scratch and may be absent (e.g. a platform-only
# checkout) — only mount the static surface when the directory exists.
if (EXPERIMENTS_ROOT / "results").is_dir():
    app.mount("/results", StaticFiles(directory=EXPERIMENTS_ROOT / "results"), name="results")

DIAGRAMS_DIR = Path(os.environ.get("BROADWAY_DIAGRAMS_DIR", "diagrams"))
_TLDRAW_VERSION = "5.4.2"
_DIAGRAM_RE = re.compile(r"[A-Za-z0-9_-]+")
if DIAGRAMS_DIR.is_dir():
    app.mount("/diagrams", StaticFiles(directory=DIAGRAMS_DIR), name="diagrams")


@dataclass
class ScriptProfile:
    """How one step script relates to the series' shared parquet constants."""

    produced: set[str] = field(default_factory=set)
    consumed: set[str] = field(default_factory=set)
    external: bool = False


def _series_dir(focus: str) -> Path:
    """Return the folder for a series id (``experiments/<focus>``)."""
    return EXPERIMENTS_ROOT / focus


def _results_dir(focus: str) -> Path:
    """Return the configured result folder for a series id."""
    return EXPERIMENTS_ROOT / "results" / focus


def _h(path: str, focus: str) -> str:
    """Append ``?focus=<quoted series id>`` to an internal path (``&`` if it has a query)."""
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}focus={quote(focus, safe='')}"


def list_series() -> list[str]:
    """Return series ids: folders under ``experiments/`` with a numbered ``NN_*.py``, sorted."""
    found: set[str] = set()
    for script in EXPERIMENTS_ROOT.rglob("*.py"):
        if script.name.startswith("_") or not _NN_PREFIX.match(script.name):
            continue
        found.add(script.relative_to(EXPERIMENTS_ROOT).parent.as_posix())
    return sorted(found)


def _resolve_focus(focus: str) -> str:
    """Use the configured focus or the first discovered series."""
    series = list_series()
    return focus or (series[0] if series else "")


def series_scripts(focus: str) -> list[Path]:
    """Return the focused series' step scripts (``*.py``, excluding ``_*.py``), sorted."""
    return sorted(p for p in _series_dir(focus).glob("*.py") if not p.name.startswith("_"))


def script_profile(script: Path) -> ScriptProfile:
    """Classify a script: parquet constants it produces/consumes + external source."""
    text = script.read_text(encoding="utf-8")
    lines = text.splitlines()
    produced = {
        const
        for const in _PARQUET_CONSTANTS
        if any(const in line and "to_parquet" in line for line in lines)
    }
    consumed = {const for const in _PARQUET_CONSTANTS if const in text} - produced
    return ScriptProfile(
        produced=produced, consumed=consumed, external=bool(_EXTERNAL_RE.search(text))
    )


def series_profiles(focus: str) -> dict[str, ScriptProfile]:
    """Return ``{stem: profile}`` for every step script in the focused series."""
    return {script.stem: script_profile(script) for script in series_scripts(focus)}


def census_rows(focus: str) -> list[dict[str, str | int]]:
    """Build one dashboard row per focused-series script (experiment | question | status | artifacts)."""
    rows: list[dict[str, str | int]] = []
    for script in series_scripts(focus):
        stem = script.stem
        results = _result_files(stem, focus)
        rows.append(
            {
                "experiment": stem,
                "question": _question_from_docstring(script),
                "status": "ran" if results else "no outputs",
                "artifacts": len(results),
            }
        )
    return rows


def load_observations(stem: str) -> dict[str, str]:
    """Load saved observations for ``stem``, or an empty dict."""
    path = OBSERVATIONS_DIR / f"{stem}.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _question_from_docstring(script: Path) -> str:
    """Return the docstring's first line with its ``NN:`` prefix stripped."""
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return ""
    doc = ast.get_docstring(tree, clean=False)
    if not doc:
        return ""
    first_line = doc.strip().splitlines()[0]
    return _NUMBER_PREFIX.sub("", first_line, count=1).strip()


def _result_files(stem: str, focus: str) -> list[Path]:
    """Return result files for ``stem`` in the focused series, sorted."""
    results_dir = _results_dir(focus)
    if not results_dir.is_dir():
        return []
    return sorted(results_dir.glob(f"{stem}*"))


def _has_results(stem: str, focus: str) -> bool:
    """Whether any result files exist for ``stem`` in the focused series."""
    return any(_results_dir(focus).glob(f"{stem}.*"))


def _box_class(stem: str, focus: str) -> str:
    """CSS class for a node box: green when the step ran, gray otherwise."""
    return "box green" if _has_results(stem, focus) else "box gray"


def _producers_of(constant: str, profiles: dict[str, ScriptProfile]) -> list[str]:
    """Return stems of scripts that produce ``constant``, sorted."""
    return sorted(stem for stem, profile in profiles.items() if constant in profile.produced)


def _consumers_of(constant: str, stem: str, profiles: dict[str, ScriptProfile]) -> list[str]:
    """Return stems of other scripts that consume ``constant``, sorted."""
    return sorted(
        other for other, profile in profiles.items() if other != stem and constant in profile.consumed
    )


def _graph_html(stem: str, profiles: dict[str, ScriptProfile], focus: str) -> str:
    """Render the bare-bones pipeline graph as escaped HTML boxes + arrows."""
    profile = profiles[stem]
    nodes: list[tuple[str, str]] = []
    if profile.external:
        nodes.append((_UPSTREAM_LABEL, "box gray"))
    else:
        for constant in sorted(profile.consumed):
            for producer in _producers_of(constant, profiles):
                nodes.append((producer, _box_class(producer, focus)))
    nodes.append((stem, "box strong " + _box_class(stem, focus)))
    for constant in sorted(profile.produced):
        for consumer in _consumers_of(constant, stem, profiles):
            nodes.append((consumer, _box_class(consumer, focus)))
    boxes = " <span class=\"arrow\">→</span> ".join(
        f'<span class="{cls}"><a href="{html.escape(_node_href(label, focus))}">{html.escape(label)}</a></span>'
        for label, cls in nodes
    )
    return f'<div class="graph">{boxes}</div>'


def _node_href(label: str, focus: str) -> str:
    """Href for a graph node: the focused dashboard for upstream, the step page otherwise."""
    if label == _UPSTREAM_LABEL:
        return _h("/", focus)
    return _h(f"/experiments/{label}", focus)


def _render_table(rows: list[dict[str, str | int]], focus: str) -> str:
    """Render the dashboard table body: linked experiment, question, status, artifacts."""
    cells = []
    for row in rows:
        stem = html.escape(str(row["experiment"]))
        href = html.escape(_h(f"/experiments/{stem}", focus))
        cells.append(
            "<tr>"
            f'<td><a href="{href}">{stem}</a></td>'
            f"<td>{html.escape(str(row['question']))}</td>"
            f"<td>{html.escape(str(row['status']))}</td>"
            f"<td>{row['artifacts']}</td>"
            "</tr>"
        )
    return "\n".join(cells)


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
_TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".log"}
_MAX_PREVIEW_ROWS = 200
_MAX_TEXT_PREVIEW = 100_000


def _artifact_url(path: Path, focus: str) -> str:
    """Return the static URL for one result artifact."""
    return f"/results/{quote(focus, safe='/')}/{quote(path.name)}"


def _render_csv_preview(path: Path) -> str:
    """Render a bounded, escaped CSV preview suitable for large result files."""
    try:
        frame = pd.read_csv(path, nrows=_MAX_PREVIEW_ROWS)
    except (ValueError, OSError, UnicodeDecodeError):
        return "<p>Could not preview this CSV.</p>"
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for values in frame.itertuples(index=False, name=None):
        cells = "".join(
            f"<td>{'' if pd.isna(value) else html.escape(str(value))}</td>" for value in values
        )
        rows.append(f"<tr>{cells}</tr>")
    truncated = (
        f'<p class="preview-note">Showing the first {_MAX_PREVIEW_ROWS} rows.</p>'
        if len(frame) == _MAX_PREVIEW_ROWS
        else ""
    )
    return (
        '<div class="table-scroll"><table class="evidence">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody>"
        f"</table></div>{truncated}"
    )


def _render_artifact_preview(path: Path, focus: str) -> str:
    """Render an inline preview for supported result artifacts."""
    suffix = path.suffix.lower()
    url = html.escape(_artifact_url(path, focus))
    name = html.escape(path.name)
    if suffix in _IMAGE_SUFFIXES:
        return f'<figure><img src="{url}" alt="{name}"><figcaption>{name}</figcaption></figure>'
    if suffix == ".html":
        return f'<iframe title="{name}" src="{url}" sandbox="allow-same-origin allow-popups" style="width:100%;height:1050px;border:0"></iframe>'
    if suffix == ".csv":
        return _render_csv_preview(path)
    if suffix in _TEXT_SUFFIXES:
        try:
            content = path.read_text(encoding="utf-8")[:_MAX_TEXT_PREVIEW]
        except (OSError, UnicodeDecodeError):
            return "<p>Could not preview this text artifact.</p>"
        return f'<pre class="text-preview">{html.escape(content)}</pre>'
    mime, _ = mimetypes.guess_type(path.name)
    return f'<p>No inline preview for {html.escape(mime or suffix or "this file type")}.</p>'


def _render_artifacts(stem: str, focus: str, selected: str = "") -> str:
    """Render a selectable, notebook-like result artifact viewer."""
    files = _result_files(stem, focus)
    if not files:
        return "<p>No artifacts yet.</p>"
    selected_path = next((path for path in files if path.name == selected), None)
    if selected_path is None:
        selected_path = next(
            (path for path in files if path.suffix.lower() in _IMAGE_SUFFIXES), files[0]
        )
    options = "".join(
        f'<option value="{html.escape(path.name)}"'
        f'{" selected" if path == selected_path else ""}>{html.escape(path.name)}</option>'
        for path in files
    )
    download = html.escape(_artifact_url(selected_path, focus))
    picker = (
        '<form class="artifact-picker" method="get">'
        f'<input type="hidden" name="focus" value="{html.escape(focus)}">'
        '<label for="artifact">output</label> '
        f'<select id="artifact" name="artifact" onchange="this.form.submit()">{options}</select> '
        '<noscript><button type="submit">show</button></noscript>'
        f'<a href="{download}" download>download</a>'
        "</form>"
    )
    preview = _render_artifact_preview(selected_path, focus)
    return f'{picker}<div class="artifact-preview">{preview}</div>'


def _render_evidence(stem: str, focus: str) -> str:
    """Render each matching ``<stem>*.csv`` evidence file as an escaped HTML table, or ""."""
    results_dir = _results_dir(focus)
    if not results_dir.is_dir():
        return ""
    tables = []
    for csv_path in sorted(results_dir.glob(f"{stem}*.csv")):
        try:
            desc = pd.read_csv(csv_path, index_col=0)
        except (ValueError, OSError):
            continue
        header = "".join(f"<th>{html.escape(str(col))}</th>" for col in desc.columns)
        body = []
        for label, row in desc.iterrows():
            cells = []
            for value in row:
                if isinstance(value, str):
                    cells.append(f"<td>{html.escape(value)}</td>")
                elif pd.isna(value):
                    cells.append("<td></td>")
                elif float(value).is_integer():
                    cells.append(f"<td>{float(value):,.0f}</td>")
                else:
                    cells.append(f"<td>{value:,.2f}</td>")
            body.append(f"<tr><th>{html.escape(str(label))}</th>{''.join(cells)}</tr>")
        tables.append(
            '<table class="evidence">'
            f"<thead><tr><th>stat</th>{header}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody>"
            "</table>"
        )
    return "\n".join(tables)


def _verdict_options(saved: str) -> str:
    """Render the verdict select options, preselected to ``saved``."""
    return "".join(
        f'<option value="{verdict}"{" selected" if verdict == saved else ""}>{verdict}</option>'
        for verdict in _VERDICTS
    )


def _render_observations_form(
    stem: str, observations: dict[str, str], focus: str, next_url: str = ""
) -> str:
    """Render saved observations (if any) above the observation form."""
    saved_block = ""
    if observations:
        verdict = html.escape(str(observations.get("verdict", "")))
        text = html.escape(str(observations.get("observations", "")))
        updated = html.escape(str(observations.get("updated_at", "")))
        saved_block = (
            '<div class="saved">'
            f"<p><strong>verdict:</strong> {verdict}</p>"
            f"<p><strong>observations:</strong> {text}</p>"
            f"<p><strong>updated:</strong> {updated}</p>"
            "</div>"
        )
    selected = str(observations.get("verdict", _VERDICTS[0]))
    prefill = html.escape(str(observations.get("observations", "")))
    hidden = (
        f'<input type="hidden" name="next" value="{html.escape(next_url)}">' if next_url else ""
    )
    return (
        saved_block
        + f'<form method="post" action="{html.escape(_h(f"/experiments/{stem}/observations", focus))}">'
        + hidden
        + '<p><label for="verdict">verdict</label> '
        + f'<select name="verdict" id="verdict" required>{_verdict_options(selected)}</select></p>'
        + '<p><label for="observations">observations</label></p>'
        + f'<p><textarea name="observations" id="observations" rows="6" cols="60">{prefill}</textarea></p>'
        + '<p><button type="submit">save observation</button></p>'
        + "</form>"
    )


def _prev_next_hrefs(stems: list[str], current: str, href: str) -> tuple[str, str]:
    """Return (prev, next) step hrefs around ``current`` in sorted ``stems``."""
    idx = stems.index(current)
    prev = f"{href}{stems[idx - 1]}" if idx > 0 else ""
    nxt = f"{href}{stems[idx + 1]}" if idx < len(stems) - 1 else ""
    return prev, nxt


def _series_strip(
    stems: list[str],
    current: str,
    href_for: Callable[[str], str],
    prev_href: str,
    next_href: str,
) -> str:
    """Render the series strip: prev/next plus one mini-link per stem (current highlighted)."""
    items = []
    for stem in stems:
        cls = "strip-link" + (" current" if stem == current else "")
        items.append(
            f'<a class="{cls}" href="{html.escape(href_for(stem))}">{html.escape(stem)}</a>'
        )
    prev = (
        f'<a class="strip-nav" href="{html.escape(prev_href)}">« prev</a>'
        if prev_href
        else '<span class="strip-nav strip-dim">« prev</span>'
    )
    nxt = (
        f'<a class="strip-nav" href="{html.escape(next_href)}">next »</a>'
        if next_href
        else '<span class="strip-nav strip-dim">next »</span>'
    )
    return f'<nav class="strip">{prev}{"".join(items)}{nxt}</nav>'


def _next_prefix(stem: str, stems: list[str]) -> str:
    """Pick the filename prefix for a step inserted after ``stem``."""
    match = _STEM_PREFIX.match(stem)
    base = int(match.group(0)) if match else 0
    candidate = f"{base + 1:02d}"
    if not any(s.startswith(candidate) for s in stems):
        return candidate
    letter = "a"
    while any(s.startswith(f"{base:02d}{letter}") for s in stems):
        letter = chr(ord(letter) + 1)
    return f"{base:02d}{letter}"


def _scaffold_text(prefix: str, question: str) -> str:
    """Return the scaffolded step-script text for an inserted step."""
    safe = question.replace('"""', "'''")
    return (
        f'"""{prefix}: {safe}"""\n'
        "\n"
        "from pathlib import Path\n"
        "\n"
        "import pandas as pd\n"
        "\n"
        "from _common import RESULTS, load_metered\n"
        "\n"
        'OUT = RESULTS / f"{Path(__file__).stem}.png"\n'
        "\n"
        "\n"
        "def main() -> None:\n"
        "    RESULTS.mkdir(parents=True, exist_ok=True)\n"
        "    df = load_metered()\n"
        '    print(f"rows: {len(df)}")\n'
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


def _stem_key(stem: str) -> tuple[int, str, str] | None:
    """Split a step stem into (number, letter, name); None if it lacks a ``NN[l]_`` prefix."""
    match = _KEY_RE.match(stem)
    if not match:
        return None
    return int(match.group(1)), match.group(2), match.group(3)


def _reorder_moves(renamed: dict[str, str], focus: str) -> list[tuple[Path, Path]]:
    """Collect (old, new) rename pairs: reordered scripts plus their observation records."""
    moves: list[tuple[Path, Path]] = []
    for old, new in renamed.items():
        if old == new:
            continue
        moves.append((_series_dir(focus) / f"{old}.py", _series_dir(focus) / f"{new}.py"))
        old_obs = OBSERVATIONS_DIR / f"{old}.json"
        if old_obs.is_file():
            moves.append((old_obs, OBSERVATIONS_DIR / f"{new}.json"))
    return moves


def _renumber_stems(order: list[str], focus: str) -> list[str]:
    """Renumber ``order``'s scripts window-scoped: sorted keys assigned in position order."""
    candidates = [_stem_key(stem) for stem in order]
    if any(candidate is None for candidate in candidates):
        raise ValueError("order contains an invalid step stem")
    parsed = [candidate for candidate in candidates if candidate is not None]
    keys = sorted({(n, letter) for n, letter, _ in parsed})
    renamed = {
        stem: f"{keys[i][0]:02d}{keys[i][1]}_{parsed[i][2]}" for i, stem in enumerate(order)
    }
    moves = _reorder_moves(renamed, focus)
    temps = [(src, src.with_name(f".{src.stem}.tmp")) for src, _ in moves]
    for src, tmp in temps:
        src.rename(tmp)
    for (_, dst), (_, tmp) in zip(moves, temps):
        tmp.rename(dst)
    return [renamed[stem] for stem in order]


def _series_selector(series: list[str], focus: str) -> str:
    """Render a compact dropdown that navigates to one experiment series."""
    options = []
    for sid in series:
        options.append(
            f'<option value="{html.escape(sid)}"'
            f'{" selected" if sid == focus else ""}>{html.escape(sid)}</option>'
        )
    return (
        '<form class="selector" method="get" action="/">'
        '<label for="project-experiment">project experiment</label> '
        f'<select id="project-experiment" name="focus" onchange="this.form.submit()">'
        f'{"".join(options)}</select>'
        '<noscript><button type="submit">open</button></noscript>'
        "</form>"
    )


def _render_dashboard_page(
    rows: list[dict[str, str | int]], focus: str, series: list[str]
) -> str:
    """Render the full dashboard HTML page with minimal inline styling."""
    table_rows = _render_table(rows, focus)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Broadway experiments</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  .selector {{ margin: 0.75rem 0; font-size: 0.9rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }}
  th {{ background: #f0f0f0; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>Broadway experiments</h1>
<p><a href="/canvas">maps</a> — annotated canvases (tldraw) over diagrams/ + results</p>
{_series_selector(series, focus)}
<table>
<thead>
<tr><th>experiment</th><th>question</th><th>status</th><th>artifacts</th></tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>
"""


def _render_experiment_page(
    stem: str,
    question: str,
    stems: list[str],
    profiles: dict[str, ScriptProfile],
    focus: str,
    series: list[str],
    artifact: str = "",
) -> str:
    """Render the full per-experiment HTML page (strip, graph, artifacts, observation form)."""
    prev_href, next_href = _prev_next_hrefs(stems, stem, "/experiments/")
    prev_href = _h(prev_href, focus) if prev_href else ""
    next_href = _h(next_href, focus) if next_href else ""
    strip = _series_strip(
        stems, stem, lambda s: _h(f"/experiments/{s}", focus), prev_href, next_href
    )
    nav = (
        "<p>"
        f'<a href="{html.escape(_h(f"/series?from={stem}", focus))}">show with next</a>'
        f' | <a href="{html.escape(_h(f"/experiments/{stem}/new", focus))}">＋ new step after this</a>'
        "</p>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(stem)} — Broadway experiments</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  .selector {{ margin: 0.75rem 0; font-size: 0.9rem; }}
  .strip {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin: 1rem 0; font-size: 0.85rem; }}
  .strip a, .strip span {{ padding: 0.15rem 0.4rem; border: 1px solid #ccc; border-radius: 3px; text-decoration: none; color: #222; }}
  .strip a.current {{ background: #d9f2d9; border-color: #2a7; }}
  .strip .strip-dim {{ color: #999; border-style: dashed; }}
  .graph {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem; margin: 1rem 0; }}
  .box {{ border: 1px solid #999; border-radius: 4px; padding: 0.4rem 0.8rem; font-size: 0.85rem; }}
  .box a {{ text-decoration: none; color: inherit; }}
  .box.green {{ background: #d9f2d9; }}
  .box.gray {{ background: #ececec; }}
  .box.strong {{ font-weight: bold; border-width: 2px; }}
  .arrow {{ color: #666; }}
  img {{ max-width: 100%; }}
  .artifact-picker {{ display: flex; gap: 0.6rem; align-items: center; margin: 0.75rem 0; }}
  .artifact-preview {{ border: 1px solid #ddd; border-radius: 6px; padding: 1rem; background: #fafafa; }}
  .artifact-preview figure {{ margin: 0; }}
  .artifact-preview figcaption, .preview-note {{ color: #666; font-size: 0.85rem; margin-top: 0.5rem; }}
  .table-scroll {{ max-height: 38rem; overflow: auto; }}
  .text-preview {{ max-height: 38rem; overflow: auto; white-space: pre-wrap; }}
  .evidence {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  .evidence th, .evidence td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }}
  .evidence th {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>{html.escape(stem)}</h1>
<p>{html.escape(question)}</p>
{_series_selector(series, focus)}
{strip}
{nav}
{_graph_html(stem, profiles, focus)}
<h2>artifacts</h2>
{_render_artifacts(stem, focus, artifact)}
<h2>observations</h2>
{_render_observations_form(stem, load_observations(stem), focus)}
</body>
</html>
"""


def _render_series_cards(
    stems: list[str], profiles: dict[str, ScriptProfile], back_url: str, focus: str
) -> str:
    """Render one card per step: header link, question, figure, observation form."""
    cards = []
    for stem in stems:
        script = _script_for(stem, focus)
        question = _question_from_docstring(script) if script else ""
        png = next((f for f in _result_files(stem, focus) if f.suffix == ".png"), None)
        figure = (
            f'<p><img src="/results/{html.escape(focus)}/{html.escape(png.name)}" alt="{html.escape(png.name)}"></p>'
            if png
            else "<p>No figure yet.</p>"
        )
        cards.append(
            '<section class="card" draggable="true"'
            f' data-stem="{html.escape(stem)}">'
            f'<h2><a href="{html.escape(_h(f"/experiments/{stem}", focus))}">{html.escape(stem)}</a></h2>'
            f"<p>{html.escape(question)}</p>"
            f"{figure}"
            f"{_render_observations_form(stem, load_observations(stem), focus, back_url)}"
            "</section>"
        )
    return "\n".join(cards)


_SERIES_REORDER_JS = """
(() => {
  const container = document.querySelector(".cards");
  if (!container) return;
  const focus = container.dataset.focus;
  const endZone = container.querySelector(".drop-zone");
  const stems = () =>
    Array.from(container.querySelectorAll(".card")).map((card) => card.dataset.stem);
  const clearOver = () => {
    for (const el of container.querySelectorAll(".drag-over")) el.classList.remove("drag-over");
  };
  let dragged = null;

  container.addEventListener("dragstart", (event) => {
    const card = event.target.closest(".card");
    if (!card) return;
    dragged = card;
    card.classList.add("dragging");
    event.dataTransfer.effectAllowed = "move";
  });

  container.addEventListener("dragend", () => {
    if (dragged) dragged.classList.remove("dragging");
    clearOver();
    dragged = null;
  });

  container.addEventListener("dragover", (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    const target = event.target.closest(".card, .drop-zone");
    clearOver();
    if (target && target !== dragged) target.classList.add("drag-over");
  });

  container.addEventListener("drop", async (event) => {
    event.preventDefault();
    const target = event.target.closest(".card, .drop-zone");
    if (!dragged || !target || target === dragged) return;
    if (target.classList.contains("card")) {
      container.insertBefore(dragged, target);
    } else {
      container.insertBefore(dragged, endZone);
    }
    const order = stems();
    if (dragged) dragged.classList.remove("dragging");
    clearOver();
    dragged = null;
    const response = await fetch("/series/reorder?focus=" + encodeURIComponent(focus), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order }),
    });
    if (!response.ok) {
      alert(await response.text());
      return;
    }
    const reordered = (await response.json()).reordered;
    location.href = "/series?from=" + encodeURIComponent(reordered[0])
      + "&to=" + encodeURIComponent(reordered[reordered.length - 1])
      + "&focus=" + encodeURIComponent(focus);
  });
})();
"""


def _render_series_page(
    from_stem: str,
    to_stem: str,
    stems: list[str],
    profiles: dict[str, ScriptProfile],
    focus: str,
    series: list[str],
) -> str:
    """Render the multi-step series view: strip plus one card per step in range."""
    from_idx = stems.index(from_stem)
    to_idx = stems.index(to_stem)
    prev_href = ""
    if from_idx > 0:
        prev_href = _h(f"/series?from={stems[from_idx - 1]}&to={stems[to_idx - 1]}", focus)
    next_href = ""
    if to_idx < len(stems) - 1:
        next_href = _h(f"/series?from={stems[from_idx + 1]}&to={stems[to_idx + 1]}", focus)
    strip = _series_strip(
        stems, from_stem, lambda s: _h(f"/series?from={s}", focus), prev_href, next_href
    )
    back_url = _h(f"/series?from={from_stem}&to={to_stem}", focus)
    cards = _render_series_cards(stems[from_idx : to_idx + 1], profiles, back_url, focus)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>series — Broadway experiments</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  .selector {{ margin: 0.75rem 0; font-size: 0.9rem; }}
  .strip {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin: 1rem 0; font-size: 0.85rem; }}
  .strip a, .strip span {{ padding: 0.15rem 0.4rem; border: 1px solid #ccc; border-radius: 3px; text-decoration: none; color: #222; }}
  .strip a.current {{ background: #d9f2d9; border-color: #2a7; }}
  .strip .strip-dim {{ color: #999; border-style: dashed; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }}
  .card {{ border: 1px solid #ccc; border-radius: 6px; padding: 0.8rem 1rem; flex: 1 1 22rem; }}
  .card img {{ max-width: 100%; }}
  .card[draggable="true"] {{ cursor: grab; }}
  .card.dragging {{ opacity: 0.5; }}
  .card.drag-over {{ outline: 2px dashed #2a7; }}
  .drop-zone {{ flex: 1 1 100%; min-height: 2rem; border: 2px dashed #ccc; border-radius: 6px; }}
  .drop-zone.drag-over {{ border-color: #2a7; background: #f0f8f0; }}
</style>
</head>
<body>
<h1>series</h1>
{_series_selector(series, focus)}
{strip}
<div class="cards" data-focus="{html.escape(focus)}">
{cards}
<div class="drop-zone"></div>
</div>
<script>
{_SERIES_REORDER_JS}
</script>
</body>
</html>
"""


def _render_new_step_page(stem: str, focus: str, series: list[str]) -> str:
    """Render the insert-a-step form page."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>new step after {html.escape(stem)} — Broadway experiments</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  .selector {{ margin: 0.75rem 0; font-size: 0.9rem; }}
  label {{ display: block; margin-top: 0.6rem; }}
  input, textarea {{ margin-top: 0.2rem; }}
</style>
</head>
<body>
<h1>new step after {html.escape(stem)}</h1>
{_series_selector(series, focus)}
<form method="post" action="{html.escape(_h(f"/experiments/{stem}/new", focus))}">
<p><label for="name">name (slug: [a-z0-9_]+)</label>
<input name="name" id="name" required pattern="[a-z0-9_]+" size="40"></p>
<p><label for="question">question</label>
<textarea name="question" id="question" rows="3" cols="60" required></textarea></p>
<p><button type="submit">create step</button></p>
</form>
</body>
</html>
"""


def _script_for(name: str, focus: str) -> Path | None:
    """Return the focused-series script with stem ``name``, or None if unknown."""
    if not name or name.startswith("_"):
        return None
    candidate = _series_dir(focus) / f"{name}.py"
    return candidate if candidate.is_file() else None


@app.get("/", response_model=None)
def index(focus: str = Query(default=DEFAULT_SERIES)) -> HTMLResponse | PlainTextResponse:
    """Serve the dashboard for the focused series."""
    focus = _resolve_focus(focus)
    if focus not in list_series():
        return PlainTextResponse("unknown series", status_code=404)
    return HTMLResponse(_render_dashboard_page(census_rows(focus), focus, list_series()))


@app.get("/experiments/{name}", response_model=None)
def experiment_page(
    name: str,
    focus: str = Query(default=DEFAULT_SERIES),
    artifact: str = Query(default=""),
) -> HTMLResponse | PlainTextResponse:
    """Serve the per-experiment page with strip, graph, artifacts, and observation form."""
    focus = _resolve_focus(focus)
    if focus not in list_series():
        return PlainTextResponse("unknown series", status_code=404)
    script = _script_for(name, focus)
    if script is None:
        return PlainTextResponse("unknown experiment", status_code=404)
    stems = [s.stem for s in series_scripts(focus)]
    return HTMLResponse(
        _render_experiment_page(
            name,
            _question_from_docstring(script),
            stems,
            series_profiles(focus),
            focus,
            list_series(),
            artifact,
        )
    )


@app.get("/series", response_model=None)
def series_page(
    from_: str = Query(default="", alias="from"),
    to: str = Query(default=""),
    focus: str = Query(default=DEFAULT_SERIES),
) -> HTMLResponse | PlainTextResponse:
    """Serve the multi-step series view: one card per step from ``from_`` to ``to``."""
    focus = _resolve_focus(focus)
    if focus not in list_series():
        return PlainTextResponse("unknown series", status_code=404)
    stems = [script.stem for script in series_scripts(focus)]
    if not stems:
        return PlainTextResponse("no experiments", status_code=404)
    from_stem = from_ or stems[0]
    if from_stem not in stems or (to and to not in stems):
        return PlainTextResponse("unknown experiment", status_code=404)
    from_idx = stems.index(from_stem)
    to_idx = stems.index(to) if to else min(from_idx + 1, len(stems) - 1)
    if to_idx < from_idx:
        from_idx, to_idx = to_idx, from_idx
    return HTMLResponse(
        _render_series_page(
            stems[from_idx], stems[to_idx], stems, series_profiles(focus), focus, list_series()
        )
    )


@app.post("/series/reorder", response_model=None)
async def reorder_series(
    request: Request, focus: str = Query(default=DEFAULT_SERIES)
) -> dict[str, list[str]] | PlainTextResponse:
    """Renumber the given stems (window-scoped) so filename order matches the new order."""
    focus = _resolve_focus(focus)
    if focus not in list_series():
        return PlainTextResponse("unknown series", status_code=404)
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return PlainTextResponse("invalid order", status_code=422)
    order = body.get("order") if isinstance(body, dict) else None
    if (
        not isinstance(order, list)
        or not order
        or not all(isinstance(stem, str) for stem in order)
        or len(order) != len(set(order))
    ):
        return PlainTextResponse("invalid order", status_code=422)
    stems = [script.stem for script in series_scripts(focus)]
    if not all(stem in stems and _stem_key(stem) is not None for stem in order):
        return PlainTextResponse("invalid order", status_code=422)
    reordered = _renumber_stems(order, focus)
    logger.info("reordered %d stems in %s", len(order), focus)
    return {"reordered": reordered}


@app.get("/experiments/{name}/new", response_model=None)
def new_step_page(
    name: str, focus: str = Query(default=DEFAULT_SERIES)
) -> HTMLResponse | PlainTextResponse:
    """Serve the form to insert a new step after ``name`` in the focused series."""
    focus = _resolve_focus(focus)
    if focus not in list_series():
        return PlainTextResponse("unknown series", status_code=404)
    if _script_for(name, focus) is None:
        return PlainTextResponse("unknown experiment", status_code=404)
    return HTMLResponse(_render_new_step_page(name, focus, list_series()))


@app.post("/experiments/{name}/new", response_model=None)
async def create_step(
    name: str, request: Request, focus: str = Query(default=DEFAULT_SERIES)
) -> RedirectResponse | PlainTextResponse:
    """Validate and scaffold a new step script after ``name``; 303 to its page."""
    focus = _resolve_focus(focus)
    if focus not in list_series():
        return PlainTextResponse("unknown series", status_code=404)
    if _script_for(name, focus) is None:
        return PlainTextResponse("unknown experiment", status_code=404)
    fields = parse_qs((await request.body()).decode("utf-8"))
    new_name = fields.get("name", [""])[0]
    question = fields.get("question", [""])[0]
    if not _SLUG_RE.fullmatch(new_name) or not question.strip():
        return PlainTextResponse("invalid name or question", status_code=422)
    stems = [script.stem for script in series_scripts(focus)]
    prefix = _next_prefix(name, stems)
    stem = f"{prefix}_{new_name}"
    (_series_dir(focus) / f"{stem}.py").write_text(
        _scaffold_text(prefix, question), encoding="utf-8"
    )
    logger.info("created step script %s after %s in %s", stem, name, focus)
    return RedirectResponse(url=_h(f"/experiments/{stem}", focus), status_code=303)


@app.post("/experiments/{name}/observations", response_model=None)
async def save_observations(
    name: str, request: Request, focus: str = Query(default=DEFAULT_SERIES)
) -> RedirectResponse | PlainTextResponse:
    """Validate and persist an observation + verdict for a focused-series script."""
    focus = _resolve_focus(focus)
    if focus not in list_series():
        return PlainTextResponse("unknown series", status_code=404)
    if _script_for(name, focus) is None:
        return PlainTextResponse("unknown experiment", status_code=404)
    fields = parse_qs((await request.body()).decode("utf-8"))
    verdict = fields.get("verdict", [""])[0]
    if verdict not in _VERDICTS:
        return PlainTextResponse("invalid verdict", status_code=422)
    record = {
        "experiment": name,
        "verdict": verdict,
        "observations": fields.get("observations", [""])[0],
        "updated_at": datetime.now(UTC).isoformat(),
    }
    OBSERVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    (OBSERVATIONS_DIR / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("saved observations for %s (verdict=%s)", name, verdict)
    next_url = fields.get("next", [""])[0]
    if not (next_url.startswith("/") and not next_url.startswith("//")):
        next_url = ""
    return RedirectResponse(url=next_url or _h(f"/experiments/{name}", focus), status_code=303)


# --------------------------------------------------------------------------- #
# Canvas maps (tldraw over diagrams/ + results)
# --------------------------------------------------------------------------- #
# The one big canvas, hosted: diagrams/*.tldr render as an editable infinite
# canvas; annotations save back to the same files (versioned). Plots stay on
# the experiment pages (/results static); maps link to them. The ONLY page
# allowed external assets (tldraw via esm.sh, pinned) — everything else in
# this app remains dependency-free HTML.


def list_diagrams() -> list[str]:
    """Sorted .tldr stems under DIAGRAMS_DIR (empty when absent)."""
    if not DIAGRAMS_DIR.is_dir():
        return []
    return sorted(p.stem for p in DIAGRAMS_DIR.glob("*.tldr") if p.is_file())


def _diagram_file(name: str) -> Path | None:
    """Resolve a diagram file, or None (unknown name, traversal, missing)."""
    if not name or not _DIAGRAM_RE.fullmatch(name):
        return None
    candidate = (DIAGRAMS_DIR / f"{name}.tldr").resolve()
    if DIAGRAMS_DIR.resolve() not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


def _render_canvas_index(names: list[str]) -> str:
    items = "".join(
        f'<li><a href="/canvas/{html.escape(n)}">{html.escape(n)}</a></li>' for n in names
    ) or "<li>No diagrams yet.</li>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Maps</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
</style>
</head>
<body>
<h1>Maps</h1>
<p><a href="/">experiments</a> — annotated canvases over diagrams/ (tldraw, editable, saves back)</p>
<ul>{items}</ul>
</body>
</html>
"""


def _render_canvas_page(name: str) -> str:
    version = _TLDRAW_VERSION
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Map — {html.escape(name)}</title>
<link rel="stylesheet" href="https://esm.sh/tldraw@{version}/tldraw.css">
<style>
  html, body {{ margin: 0; height: 100%; font-family: system-ui, sans-serif; }}
  #bar {{ padding: 0.4rem 0.8rem; background: #f0f0f0; font-size: 0.9rem; }}
  #bar a {{ margin-right: 1rem; }}
  #canvas {{ position: fixed; inset: 48px 0 0 0; }}
</style>
<script type="importmap">
{{
  "imports": {{
    "react": "https://esm.sh/react@18.3.1",
    "react/jsx-runtime": "https://esm.sh/react@18.3.1/jsx-runtime",
    "react-dom": "https://esm.sh/react-dom@18.3.1",
    "react-dom/client": "https://esm.sh/react-dom@18.3.1/client",
    "tldraw": "https://esm.sh/tldraw@{version}?external=react,react-dom"
  }}
}}
</script>
</head>
<body>
<div id="bar"><a href="/canvas">maps</a><span id="status">loading…</span>
<button id="save" type="button">save annotations</button></div>
<div id="canvas"></div>
<script type="module">
import React from 'react';
import {{ createRoot }} from 'react-dom/client';
import {{ Tldraw, createTLStore, loadSnapshot, getSnapshot }} from 'tldraw';

const NAME = {json.dumps(name)};
const status = (msg) => {{ document.getElementById('status').textContent = msg; }};

async function main() {{
  const store = createTLStore();
  try {{
    const res = await fetch('/diagrams/' + encodeURIComponent(NAME) + '.tldr');
    if (!res.ok) throw new Error('diagram not found');
    const snap = await res.json();
    loadSnapshot(store, {{ store: snap.records ?? snap.store, schema: snap.schema }});
  }} catch (err) {{
    status('failed to load: ' + err.message);
    return;
  }}
  function App() {{
    return React.createElement(Tldraw, {{
      store,
      onMount: (editor) => {{
        window.__editor = editor;
        status('ready — annotate freely, then save');
        document.getElementById('save').onclick = async () => {{
          status('saving…');
          const snap = getSnapshot(editor.store);
          const res = await fetch('/canvas/' + encodeURIComponent(NAME) + '/save', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ document: snap.document ?? snap.store }}),
          }});
          status(res.ok ? 'saved' : 'save failed: ' + res.status);
        }};
      }},
    }});
  }}
  createRoot(document.getElementById('canvas')).render(React.createElement(App));
  status('ready');
}}
main();
</script>
</body>
</html>
"""


@app.get("/canvas", response_model=None)
def canvas_index() -> HTMLResponse:
    """List available canvases."""
    return HTMLResponse(_render_canvas_index(list_diagrams()))


@app.get("/canvas/{name}", response_model=None)
def canvas_page(name: str) -> HTMLResponse | PlainTextResponse:
    """Serve the editable tldraw canvas for a diagram."""
    if _diagram_file(name) is None:
        return PlainTextResponse("unknown diagram", status_code=404)
    return HTMLResponse(_render_canvas_page(name))


@app.post("/canvas/{name}/save", response_model=None)
async def canvas_save(name: str, request: Request) -> HTMLResponse | PlainTextResponse:
    """Persist canvas annotations back to the .tldr source file."""
    target = _diagram_file(name)
    if target is None:
        return PlainTextResponse("unknown diagram", status_code=404)
    try:
        payload = await request.json()
    except ValueError:
        return PlainTextResponse("invalid snapshot json", status_code=422)
    document = payload.get("document") if isinstance(payload, dict) else None
    if not isinstance(document, dict) or not document:
        return PlainTextResponse("snapshot carries no document records", status_code=422)
    try:
        current = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return PlainTextResponse("stored diagram unreadable", status_code=422)
    if not isinstance(current, dict):
        return PlainTextResponse("stored diagram malformed", status_code=422)
    current["records"] = document
    target.write_text(json.dumps(current, indent=1), encoding="utf-8")
    logger.info("saved canvas annotations for %s", name)
    return HTMLResponse("<p>saved</p>")
