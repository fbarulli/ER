"""Side-by-side model comparison for a completed suite run.

Reads only what the run already published (scored pairs, per-track manifests,
the exhaustive ablation reports and the graph vectors), derives the comparison
metrics, renders the comparison plots, and serves one page that puts every
model next to every other model.

Nothing here trains, provisions or refits: it is a read-only analysis surface
over frozen artifacts, and every number it shows carries the population it was
measured on so a thin population cannot read as a strong result.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np

from core.common import artifact
from model_tracks.package import package_member

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

TRACKS = ('baseline', 'text', 'gnn_only', 'cascade')
# The cascade is a combinator, not a fitted encoder: it owns no ablation report,
# no vectors and no ablation cells, so it stays out of the fitted-model metrics.
FITTED = ('text', 'gnn_only')
LABEL = {'baseline': 'baseline (frozen MiniLM)', 'text': 'A text',
         'gnn_only': 'B gnn_only', 'cascade': 'C cascade'}
COLOR = {'baseline': '#9ca3af', 'text': '#2563eb',
         'gnn_only': '#16a34a', 'cascade': '#9333ea'}
# Track-owned artifact suffixes; the paths themselves come from paths.yaml.
SUMMARY_SUFFIX = '__model_evaluation_summary.csv'
SCORED_SUFFIX = '__scored_pairs.csv'
RETRIEVAL_SUFFIX = '__retrieval_summary.csv'
SLICE_SUFFIX = '__slice_metrics.csv'
MANIFEST_SUFFIXES = ('__completion_manifest.json', '__report_manifest.json')
VECTOR_SUFFIX = '__vectors.npz'
ARCHIVE_FORMAT = 'tar.zst'


# ── path resolution (SSOT: config/paths.yaml layouts) ───────────────────────
def run_dir(run_tag: str) -> Path:
    return artifact('suite_outputs', {'run_tag': run_tag})


def local_inputs(run_tag: str) -> Path:
    return artifact('suite_output_local_inputs', {'run_tag': run_tag})


def ablation_report(run_tag: str, track: str) -> Path:
    return artifact('track_ablation_report', {'run_tag': run_tag, 'track': track})


def plot_path(run_tag: str, name: str) -> Path:
    return artifact('model_comparison_plot', {'run_tag': run_tag, 'name': name})


def input_archive(run_tag: str) -> Path:
    return artifact('suite_output_input_archive',
                    {'run_tag': run_tag, 'fmt': ARCHIVE_FORMAT})


def training_archive(run_tag: str) -> Path:
    return artifact('suite_output_training_archive',
                    {'run_tag': run_tag, 'fmt': ARCHIVE_FORMAT})


def packaged_setup(run_tag: str) -> Path:
    """The restored shared setup root (a package member under local_inputs)."""
    return local_inputs(run_tag) / package_member('suite_package_shared')


# ── artifact discovery ──────────────────────────────────────────────────────
def _first(path: Path, *suffixes: str) -> Path | None:
    if not path.is_dir():
        return None
    for suffix in suffixes:
        hits = sorted(p for p in path.rglob(f'*{suffix}') if '__inputs' not in p.name)
        if hits:
            return hits[0]
    return None


def discover_runs() -> list[str]:
    base = artifact('suite_outputs', {'run_tag': '*'}).parent
    if not base.is_dir():
        return []
    tags = []
    for p in base.iterdir():
        if p.is_dir() and ablation_report(p.name, 'text').is_file():
            tags.append(p.name)
    return sorted(tags, reverse=True)


# ── metrics ─────────────────────────────────────────────────────────────────
def _ablation_cells(doc: dict) -> dict[tuple, dict]:
    """(sku1, sku2, population) -> row; baseline_score is constant per cell."""
    cells: dict[tuple, dict] = {}
    for r in doc.get('rows', ()):
        cells.setdefault((r.get('sku_id1'), r.get('sku_id2'), r.get('population')), r)
    return cells


def _auc(pos, neg):
    if not pos or not neg:
        return None
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


def _pop_table(cells_by_model):
    """Per-population AUC against the real-positive reference, per model."""
    pops = sorted({cell[2] for cell in cells_by_model[FITTED[0]]})
    out = {}
    for pop in pops:
        row = {}
        for model, cells in cells_by_model.items():
            pos = [c['baseline_score'] for k, c in cells.items()
                   if k[2] in ('real', 'positive') and str(c.get('label')) == '1'
                   and c.get('baseline_score') is not None]
            neg = [c['baseline_score'] for k, c in cells.items()
                   if k[2] == pop and str(c.get('label')) == '0'
                   and c.get('baseline_score') is not None]
            row[model] = {'auc': _auc(pos, neg), 'n_neg': len(neg), 'n_ref': len(pos)}
        out[pop] = row
    return out


def _geometry(X, gtin_labels):
    """Cluster structure WITHOUT single-linkage chaining.

    Thresholded connected components manufacture one giant blob (single linkage
    chains any two points bridged by a single borderline edge), so structure is
    measured on a mutual-kNN graph with greedy-modularity communities and scored
    against the identity labels with ARI. NMI is deliberately not the headline:
    on a catalog of mostly-distinct GTINs both labellings are near-unique, so NMI
    rewards that trivial agreement (it reads ~0.78 while ARI reads ~0.05).
    """
    import networkx as nx
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    n = X.shape[0]
    Xn = X / np.clip(np.linalg.norm(X, axis=1)[:, None], 1e-12, None)
    S = Xn @ Xn.T
    np.fill_diagonal(S, np.nan)
    fin = S[~np.isnan(S)]
    out = {'n': int(n), 'dim': int(X.shape[1]),
           'cos_mean': float(fin.mean()), 'cos_p50': float(np.nanpercentile(S, 50)),
           'cos_p90': float(np.nanpercentile(S, 90)),
           'cos_p99': float(np.nanpercentile(S, 99))}
    for k in (5, 10, 20):
        if n <= k + 1:
            continue
        Sk = S.copy()
        np.fill_diagonal(Sk, -np.inf)
        idx = np.argpartition(-Sk, k, axis=1)[:, :k]
        knn = [set(int(j) for j in idx[i]) for i in range(n)]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i, js in enumerate(knn):
            for j in js:
                if i in knn[j]:
                    G.add_edge(i, j)
        comps = list(nx.community.greedy_modularity_communities(G)) \
            if G.number_of_edges() else []
        sizes = sorted((len(c) for c in comps), reverse=True)
        rec = {'communities': len(sizes) or 1, 'largest': sizes[0] if sizes else n,
               'mutual_knn_components': sorted(
                   (len(c) for c in nx.connected_components(G)), reverse=True)[:3]}
        if len(sizes) > 1:
            lab = {}
            for ci, c in enumerate(comps):
                for v in c:
                    lab[v] = ci
            vec = [lab[i] for i in range(n)]
            if gtin_labels and len(set(gtin_labels)) > 1:
                rec['ari_gtin'] = float(adjusted_rand_score(gtin_labels, vec))
                rec['nmi_gtin'] = float(normalized_mutual_info_score(gtin_labels, vec))
        out[f'k{k}'] = rec
    return out


def _attribute_effect(doc: dict) -> dict[str, dict]:
    """Mean |score delta| and mean |embedding cosine delta| per attribute."""
    acc: dict[str, dict] = {}
    for r in doc.get('rows', ()):
        a = r.get('attribute')
        if a is None:
            continue
        b = acc.setdefault(a, {'n': 0, 'delta': 0.0, 'cos': 0.0, 'changed': 0})
        b['n'] += 1
        b['delta'] += abs(r.get('score_delta') or 0.0)
        v = r.get('embedding_cosine_delta') or []
        b['cos'] += max((abs(x) for x in v), default=0.0)
        b['changed'] += 1 if any(r.get('endpoint_input_changed') or []) else 0
    for b in acc.values():
        n = b['n'] or 1
        b['mean_abs_delta'] = b['delta'] / n
        b['mean_abs_cos'] = b['cos'] / n
        b['changed_pct'] = 100 * b['changed'] / n
        b['dead'] = b['mean_abs_delta'] < 1e-9
    return acc


def _manifest(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _summary_rows(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    with path.open(newline='') as fh:
        return list(csv.DictReader(fh))


def _scored(path: Path | None) -> list[dict]:
    return _summary_rows(path)


def _pair_truth(run_tag: str, models: dict) -> tuple[list, dict, dict]:
    """Ground truth per scored pair, recovered from GTIN + the gate decision."""
    setup = packaged_setup(run_tag) / 'eligible_catalog.csv'
    catalog = {}
    if setup.is_file():
        with setup.open(newline='') as fh:
            for row in csv.DictReader(fh):
                catalog[row.get('sku_id')] = row
    gate_path = None
    for candidate in sorted(run_dir(run_tag).rglob('gate_results.csv')):
        gate_path = candidate
        break
    gate = {}
    if gate_path is not None:
        with gate_path.open(newline='') as fh:
            for row in csv.DictReader(fh):
                gate[(row.get('gtin1'), row.get('gtin2'))] = row
    keys = sorted({k for m in models if m in FITTED for k in
                   ((r['sku_id1'], r['sku_id2']) for r in models[m]['scored'])})
    truth = {}
    for a, b in keys:
        g1 = catalog.get(a, {}).get('gtin')
        g2 = catalog.get(b, {}).get('gtin')
        g = gate.get((g1, g2)) or gate.get((g2, g1)) or {}
        label = next((int(r['true_label']) for m in models if m in FITTED
                      for r in models[m]['scored']
                      if (r['sku_id1'], r['sku_id2']) == (a, b)), None)
        truth[(a, b)] = {
            'label': label, 'gtin1': g1, 'gtin2': g2,
            'same_gtin': bool(g1 and g1 == g2),
            'gate': g.get('gate_decision'), 'reason': g.get('gate_reason'),
            'brand': catalog.get(a, {}).get('brand'),
            'title_a': catalog.get(a, {}).get('sku_name_eng'),
            'title_b': catalog.get(b, {}).get('sku_name_eng'),
        }
    return keys, truth, catalog


def _score_of(model: str, pair, models: dict):
    for r in models[model]['scored']:
        if (r['sku_id1'], r['sku_id2']) == pair:
            try:
                return float(r['score'])
            except (TypeError, ValueError):
                return None
    return None


def analyse(run_tag: str) -> dict | None:
    dest = run_dir(run_tag)
    if not dest.is_dir():
        return None
    models: dict[str, dict] = {}
    cells_by_model: dict[str, dict] = {}
    for model in TRACKS:
        base = dest / model
        doc = _manifest(ablation_report(run_tag, model))
        man = _manifest(_first(base, *MANIFEST_SUFFIXES))
        summary = _summary_rows(_first(base, SUMMARY_SUFFIX))
        models[model] = {
            'label': LABEL[model], 'color': COLOR[model],
            'threshold': doc.get('threshold') or man.get('threshold'),
            'threshold_source': man.get('threshold_source'),
            'manifest': man,
            'summary': summary,
            'ablation': {'present': bool(doc), 'schema': doc.get('schema'),
                         'coverage': doc.get('coverage', {}).get('mode'),
                         'catalog': doc.get('retrieval_catalog_count'),
                         'rows': len(doc.get('rows', ()))},
            'scored': _scored(_first(base, SCORED_SUFFIX)),
            'retrieval': _summary_rows(_first(base, RETRIEVAL_SUFFIX)),
            'slices': _summary_rows(_first(base, SLICE_SUFFIX)),
        }
        if doc:
            cells_by_model[model] = _ablation_cells(doc)

    if not cells_by_model:
        return None

    # discrimination per population
    pops = _pop_table(cells_by_model) if all(m in cells_by_model for m in FITTED) else {}

    # real-pair discrimination (the honest generalization number)
    real_auc = {}
    for model, cells in cells_by_model.items():
        pos = [c['baseline_score'] for k, c in cells.items()
               if k[2] == 'real' and str(c.get('label')) == '1' and c.get('baseline_score') is not None]
        neg = [c['baseline_score'] for k, c in cells.items()
               if k[2] == 'real' and str(c.get('label')) == '0' and c.get('baseline_score') is not None]
        real_auc[model] = {'auc': _auc(pos, neg), 'n_pos': len(pos), 'n_neg': len(neg)}

    # calibration: where the fitted threshold sits vs the positive distribution
    calib = {}
    for model, doc_cells in cells_by_model.items():
        thr = models[model]['threshold']
        pos = sorted(c['baseline_score'] for k, c in doc_cells.items()
                     if k[2] in ('real', 'real_bundle') and str(c.get('label')) == '1'
                     and c.get('baseline_score') is not None)
        neg = sorted(c['baseline_score'] for k, c in doc_cells.items()
                     if k[2] in ('real', 'real_bundle') and str(c.get('label')) == '0'
                     and c.get('baseline_score') is not None)
        if thr is None or not pos or not neg:
            continue
        above = sum(1 for x in pos if x >= thr)
        tp = above
        fp = sum(1 for x in neg if x >= thr)
        fn = len(pos) - tp
        calib[model] = {
            'threshold': thr,
            'pos_p05': pos[max(0, int(0.05 * len(pos)))], 'pos_median': pos[len(pos) // 2],
            'neg_median': neg[len(neg) // 2], 'neg_p95': neg[min(int(0.95 * len(neg)), len(neg) - 1)],
            'pos_above': above, 'n_pos': len(pos), 'n_neg': len(neg),
            'precision': tp / (tp + fp) if (tp + fp) else 0.0,
            'recall': tp / len(pos),
        }

    # attribute effect per model
    attrs = {}
    all_attrs = set()
    per_model_attr = {}
    for model in TRACKS:
        doc = _manifest(ablation_report(run_tag, model))
        if not doc:
            continue
        per_model_attr[model] = _attribute_effect(doc)
        all_attrs |= set(per_model_attr[model])
    for a in all_attrs:
        attrs[a] = {m: per_model_attr.get(m, {}).get(a) for m in TRACKS}

    # embedding geometry + cluster structure (what the encoders actually organise)
    catalog = {}
    cat_path = packaged_setup(run_tag) / 'eligible_catalog.csv'
    if cat_path.is_file():
        with cat_path.open(newline='') as fh:
            for row in csv.DictReader(fh):
                catalog[row.get('sku_id')] = row.get('gtin')
    geom = {}
    for model in FITTED:
        vec = _first(dest / model, VECTOR_SUFFIX)
        if vec is None:
            continue
        try:
            z = np.load(vec, allow_pickle=True)
            X = np.asarray(z['embeddings'], dtype=np.float64)
            ids = [str(v) for v in z['ids']] if 'ids' in z else None
        except (OSError, KeyError, ValueError):
            continue
        if ids is None:
            continue
        gtin = [catalog.get(i) or f'__{i}' for i in ids]
        geom[model] = _geometry(X, gtin)

    keys, truth, catalog = _pair_truth(run_tag, models)
    scores = {m: {k: _score_of(m, k, models) for k in keys} for m in models}
    # Positive-vs-negative error budget: does the negative cloud sit ABOVE the
    # positive one (inverted) or merely overlap it (noisy)?
    budget = {}
    for m in models:
        pos = sorted(v for k, v in scores[m].items()
                     if v is not None and truth[k]['label'] == 1)
        neg = sorted(v for k, v in scores[m].items()
                     if v is not None and truth[k]['label'] == 0)
        if not pos or not neg:
            continue
        pmed, nmed = pos[len(pos) // 2], neg[len(neg) // 2]
        budget[m] = {
            'pos_median': pmed, 'neg_median': nmed, 'separation': pmed - nmed,
            'pos_below_neg_median': sum(1 for x in pos if x < nmed),
            'neg_above_pos_median': sum(1 for x in neg if x > pmed),
            'n_pos': len(pos), 'n_neg': len(neg),
        }
    # Expected calibration error: the score treated as P(match).
    ece = {}
    for m in models:
        pts = sorted((v, truth[k]['label']) for k, v in scores[m].items()
                     if v is not None and truth[k]['label'] is not None)
        n = len(pts)
        if n < 4:
            continue
        acc = 0.0
        bins = []
        for b in range(4):
            chunk = pts[b * n // 4:(b + 1) * n // 4]
            if not chunk:
                continue
            ms = sum(x for x, _ in chunk) / len(chunk)
            pr = sum(l for _, l in chunk) / len(chunk)
            acc += (len(chunk) / n) * abs(pr - ms)
            bins.append({'lo': b * 25, 'hi': b * 25 + 25, 'n': len(chunk),
                         'mean_score': ms, 'positive_rate': pr})
        ece[m] = {'ece': acc, 'bins': bins}

    return {'run_tag': run_tag, 'models': models, 'real_auc': real_auc,
            'populations': pops, 'calibration': calib, 'attributes': attrs,
            'geometry': geom, 'pairs': keys, 'truth': truth, 'scores': scores,
            'budget': budget, 'ece': ece, 'catalog': catalog,
            'population_counts': dict(Counter(
                k[2] for k in cells_by_model[FITTED[0]]))}


# ── plots ───────────────────────────────────────────────────────────────────
def _plots(run_tag: str, data: dict) -> dict[str, str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = plot_path(run_tag, 'x').parent
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    models = [m for m in TRACKS if m in data['real_auc']]

    # 1. discrimination, real held-out-ish pairs
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vals = [data['real_auc'][m]['auc'] or 0 for m in models]
    ax.bar([LABEL[m] for m in models], vals,
           color=[COLOR[m] for m in models])
    ax.axhline(0.5, ls='--', c='#ef4444', lw=1.2, label='chance (0.50)')
    for i, m in enumerate(models):
        n = data['real_auc'][m]['n_pos'] + data['real_auc'][m]['n_neg']
        ax.text(i, vals[i] + .02, f'n={n}', ha='center', fontsize=8, color='#374151')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('AUC on real pairs')
    ax.set_title('Discrimination on real catalog pairs')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout(); fig.savefig(out / 'discrimination_real.png', dpi=130); plt.close(fig)
    written['discrimination_real'] = 'discrimination_real.png'

    # 2. per-population discrimination (synthetic negatives)
    pops = [p for p in data['populations'] if p in ('twin', 'hard_negative', 'base', 'masked', 'positive')]
    if pops:
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        w = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            ys = []
            for p in pops:
                a = data['populations'][p].get(m, {}).get('auc')
                ys.append(a if a is not None else np.nan)
            ax.bar(np.arange(len(pops)) + i * w, ys, width=w, label=LABEL[m], color=COLOR[m])
        ax.axhline(0.5, ls='--', c='#ef4444', lw=1.2)
        ax.set_xticks(np.arange(len(pops)) + 0.4 - w / 2)
        ax.set_xticklabels([f'{p}\n(n={data["populations"][p][models[0]]["n_neg"]})' for p in pops],
                           fontsize=8)
        ax.set_ylabel('AUC vs real positives')
        ax.set_title('Synthetic-negative families (0.5 = chance, 1.0 = fully rejected)')
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / 'population_discrimination.png', dpi=130); plt.close(fig)
        written['population_discrimination'] = 'population_discrimination.png'

    # 3. calibration: threshold vs score distribution
    if data['calibration']:
        fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 3.6), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, models):
            c = data['calibration'].get(m)
            if not c:
                ax.set_visible(False); continue
            ax.axvspan(c['neg_p95'], c['pos_median'], color='#fde68a', alpha=.5,
                       label='neg p95 → pos median')
            ax.axvline(c['threshold'], color='#dc2626', lw=1.6, label=f"threshold {c['threshold']:.3f}")
            ax.set_title(f'{LABEL[m]}\n{c["pos_above"]}/{c["n_pos"]} positives above', fontsize=9)
            ax.set_xlabel('score'); ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5, loc='upper left')
        axes[0].set_ylabel('threshold placement')
        fig.suptitle('Calibration: is the fitted threshold inside the positive band?', fontsize=10)
        fig.tight_layout(); fig.savefig(out / 'calibration_thresholds.png', dpi=130); plt.close(fig)
        written['calibration_thresholds'] = 'calibration_thresholds.png'

    # 4. attribute effect heat-style bar (top attributes by mean effect)
    if data['attributes']:
        ranked = sorted(
            (a for a in data['attributes'] if any(v and not v['dead'] for v in data['attributes'][a].values())),
            key=lambda a: -max((v['mean_abs_delta'] for v in data['attributes'][a].values()
                                if v), default=0))[:14]
        if ranked:
            fig, ax = plt.subplots(figsize=(8.6, 5.0))
            h = 0.8 / max(1, len(models))
            for i, m in enumerate(models):
                ys = [(data['attributes'][a].get(m) or {}).get('mean_abs_delta') or 0 for a in ranked]
                ax.barh(np.arange(len(ranked)) + i * h, ys, height=h, label=LABEL[m], color=COLOR[m])
            ax.set_yticks(np.arange(len(ranked)) + 0.4 - h / 2)
            ax.set_yticklabels(ranked, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('mean |score delta| when the attribute is removed')
            ax.set_title('Which attributes the models actually use')
            ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(out / 'attribute_effect.png', dpi=130); plt.close(fig)
            written['attribute_effect'] = 'attribute_effect.png'

    # 5. embedding geometry: does the space recover identity clusters?
    if data['geometry']:
        ks = [k for k in ('k5', 'k10', 'k20')
              if k in next(iter(data['geometry'].values()))]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        w = 0.8 / max(1, len(data['geometry']))
        for i, (m, g) in enumerate(data['geometry'].items()):
            axes[0].bar(np.arange(len(ks)) + i * w,
                        [100 * g[k]['largest'] / g['n'] for k in ks],
                        width=w, label=LABEL[m], color=COLOR[m])
            axes[0].text(np.arange(len(ks))[i] + i * w,
                         100 * g[ks[0]]['largest'] / g['n'] + 1.5,
                         f"{g[ks[0]]['communities']} comms", ha='center', fontsize=6.5)
            aris = [g[k].get('ari_gtin') for k in ks]
            axes[1].bar(np.arange(len(ks)) + i * w,
                        [a if a is not None else 0 for a in aris],
                        width=w, label=LABEL[m], color=COLOR[m])
        axes[0].set_xticks(np.arange(len(ks)) + 0.4 - w / 2)
        axes[0].set_xticklabels([f'mutual-kNN {k[1:]}' for k in ks], fontsize=8)
        axes[0].set_ylabel('% of catalog in the largest community')
        axes[0].set_title('Community size (kNN + modularity)')
        axes[1].set_xticks(np.arange(len(ks)) + 0.4 - w / 2)
        axes[1].set_xticklabels([f'mutual-kNN {k[1:]}' for k in ks], fontsize=8)
        axes[1].set_ylabel('ARI vs GTIN identity')
        axes[1].set_title('Do the communities recover product identity?')
        axes[1].axhline(0, c='#ef4444', lw=1)
        for ax in axes:
            ax.legend(fontsize=7)
        fig.suptitle('Embedding geometry: clusters exist but are not identity clusters',
                     fontsize=10)
        fig.tight_layout(); fig.savefig(out / 'graph_clusters.png', dpi=130); plt.close(fig)
        written['graph_clusters'] = 'graph_clusters.png'
    return written


# ── page ────────────────────────────────────────────────────────────────────
def _table(headers, rows, cls=''):
    head = ''.join(f'<th style="text-align:left;padding:.4rem .6rem;border-bottom:2px solid #d1d5db">{escape(str(h))}</th>' for h in headers)
    body = []
    for r in rows:
        cells = []
        for v in r:
            style = 'padding:.35rem .6rem;border-bottom:1px solid #f3f4f6'
            if isinstance(v, tuple):
                v, style = v
            cells.append(f'<td style="{style}">{v}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return (f'<table class="{cls}" style="border-collapse:collapse;margin:.8rem 0;font-size:.86rem">'
            f'<thead><tr>{head}</tr></thead><tbody>' + ''.join(body) + '</tbody></table>')


def _fmt(v, nd=4, dash='—'):
    if v is None:
        return dash
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return escape(str(v))


@router.get('/compare', response_class=HTMLResponse)
def compare(run: str | None = None):
    tags = discover_runs()
    if not tags:
        return HTMLResponse('<h1>No completed suite run with a post-training ablation receipt</h1>')
    run_tag = run if run in tags else tags[0]
    data = analyse(run_tag)
    if data is None:
        return HTMLResponse(f'<h1>Run {escape(run_tag)} has no ablation reports to compare</h1>')
    plots = _plots(run_tag, data)
    models = [m for m in TRACKS if m in data['real_auc']]

    sel = ''.join(f'<option value="{escape(t)}"{" selected" if t == run_tag else ""}>{escape(t)}</option>'
                  for t in tags)
    parts = ['<h1>Model comparison</h1>',
             f'<form method="get"><label>run </label><select name="run">{sel}</select> '
             '<button type="submit">show</button></form>',
             f'<p style="color:#4b5563;font-size:.9rem">Every metric below is measured on the '
             f'population named in its own column. A thin population is reported as thin.</p>']

    # headline table
    rows = []
    for m in models:
        c = data['calibration'].get(m, {})
        ra = data['real_auc'][m]
        abl = data['models'][m]['ablation']
        rows.append([
            f'<b>{escape(LABEL[m])}</b>', escape(m),
            _fmt(ra['auc']), f"{ra['n_pos']}+{ra['n_neg']}",
            _fmt(c.get('threshold'), 4), escape(str(data['models'][m]['threshold_source'])),
            f"{c.get('pos_above', '—')}/{c.get('n_pos', '—')}",
            _fmt(c.get('precision'), 3), _fmt(c.get('recall'), 3),
            f"{abl['rows']:,}" if abl['present'] else '—',
        ])
    parts.append('<h2>Side by side</h2>' + _table(
        ['model', 'key', 'AUC real pairs', 'n pos+neg', 'threshold', 'fitted by',
         'pos above thr', 'precision', 'recall', 'ablation rows'], rows))

    for key, title in (('discrimination_real', 'Discrimination on real pairs'),
                       ('population_discrimination', 'Synthetic-negative families'),
                       ('calibration_thresholds', 'Calibration placement'),
                       ('attribute_effect', 'Attribute usage'),
                       ('graph_clusters', 'Embedding geometry and cluster structure')):
        if key in plots:
            parts.append(f'<h2>{escape(title)}</h2><img src="/compare/plot/{escape(run_tag)}/'
                         f'{escape(plots[key])}" style="max-width:100%;border:1px solid #e5e7eb;'
                         'border-radius:6px">')

    # population table
    if data['populations']:
        prows = []
        for pop, per in data['populations'].items():
            n = per[models[0]]['n_neg']
            prows.append([escape(str(pop)), f'{n}',
                          *[_fmt(per[m]['auc']) for m in models]])
        parts.append('<h2>Per-population AUC (against real positives)</h2>' + _table(
            ['population', 'n negatives', *[LABEL[m] for m in models]], prows))

    # attribute table
    if data['attributes']:
        arows = []
        for a, per in sorted(data['attributes'].items(),
                             key=lambda kv: -max((v['mean_abs_delta'] for v in kv[1].values() if v), default=0)):
            cells = []
            for m in models:
                v = per.get(m)
                if not v:
                    cells.append('—')
                elif v['dead']:
                    cells.append(('<span style="color:#9ca3af">dead</span>', ''))
                else:
                    cells.append(f'{v["mean_abs_delta"]:.4f} <span style="color:#6b7280">'
                                 f'({v["changed_pct"]:.0f}% touched)</span>')
            arows.append([escape(str(a)), *cells])
        parts.append('<h2>Attribute effect (mean |score delta| when removed)</h2>' + _table(
            ['attribute', *[LABEL[m] for m in models]], arows))

    # geometry table
    if data['geometry']:
        ks = [k for k in ('k5', 'k10', 'k20')
              if k in next(iter(data['geometry'].values()))]
        grows = []
        for m, g in data['geometry'].items():
            grows.append([escape(LABEL[m]), f'{g["n"]}×{g["dim"]}',
                          _fmt(g['cos_mean'], 3), _fmt(g['cos_p50'], 3),
                          _fmt(g['cos_p90'], 3), _fmt(g['cos_p99'], 3),
                          *[(f'{g[k]["communities"]} comms / largest {g[k]["largest"]}'
                             + (f' / ARI {g[k]["ari_gtin"]:+.3f}' if 'ari_gtin' in g[k] else ''))
                            for k in ks]])
        parts.append('<h2>Embedding geometry and cluster structure</h2>' + _table(
            ['track', 'catalog×dim', 'cos mean', 'cos p50', 'cos p90', 'cos p99',
             *[f'mutual-kNN {k[1:]}' for k in ks]], grows))

    # ground truth + every model's score, side by side
    if data['pairs'] and models:
        prows = []
        for k in data['pairs']:
            t = data['truth'][k]
            verdict = []
            for m in models:
                s = data['scores'][m].get(k)
                verdict.append(_fmt(s))
            prows.append([
                f'{escape(str(k[0]))}<br><span style="color:#6b7280;font-size:.78em">'
                f'{escape(str(t["title_a"] or "")[:60])}</span>',
                f'{escape(str(k[1]))}<br><span style="color:#6b7280;font-size:.78em">'
                f'{escape(str(t["title_b"] or "")[:60])}</span>',
                escape(str(t['label'])),
                ('<b style="color:#16a34a">same</b>' if t['same_gtin']
                 else '<span style="color:#dc2626">differ</span>'),
                escape(str(t['gate'] or '—')),
                escape(str(t['reason'] or '')[:40]),
                *verdict])
        same_n = sum(1 for k in data['pairs'] if data['truth'][k]['same_gtin'])
        pos_n = sum(1 for k in data['pairs'] if data['truth'][k]['label'] == 1)
        parts.append(
            f'<h2>Every scored pair, with ground truth</h2>'
            f'<p style="font-size:.86rem;color:#4b5563">'
            f'{len(data["pairs"])} pairs; {pos_n} positive, '
            f'{len(data["pairs"]) - pos_n} negative; {same_n} share a GTIN. '
            f'A positive with a matching GTIN means the label is correct, so a '
            f'miss here is a model failure and not label noise.</p>'
            + _table(['listing A', 'listing B', 'label', 'GTIN', 'gate', 'gate reason',
                      *[LABEL[m] for m in models]], prows))

    # positive-vs-negative error budget
    if data['budget']:
        brows = [[escape(LABEL[m]),
                  _fmt(b['pos_median']), _fmt(b['neg_median']),
                  (f'<b style="color:{"#dc2626" if b["separation"] < 0 else "#16a34a"}">'
                   f'{b["separation"]:+.4f}</b>'),
                  f'{b["pos_below_neg_median"]}/{b["n_pos"]}',
                  f'{b["neg_above_pos_median"]}/{b["n_neg"]}']
                 for m, b in data['budget'].items()]
        parts.append(
            '<h2>Positives vs negatives — which side fails</h2>'
            '<p style="font-size:.86rem;color:#4b5563">A negative '
            '<b>separation</b> means the negative cloud sits above the positive '
            'one: the errors are over-scored negatives dragging positives down, '
            'not merely missed positives.</p>'
            + _table(['model', 'median positive', 'median negative', 'separation',
                      'positives below median negative', 'negatives above median positive'],
                     brows))

    # calibration bins
    if data['ece']:
        crows = []
        for m, block in data['ece'].items():
            for b in block['bins']:
                crows.append([escape(LABEL[m]), f'{b["lo"]}-{b["hi"]}', b['n'],
                              _fmt(b['mean_score']), _fmt(b['positive_rate'], 3)])
            crows.append([escape(LABEL[m]), '<b>ECE</b>', len(data['pairs']),
                          '', f'<b>{block["ece"]:.4f}</b>'])
        parts.append('<h2>Calibration — score bin vs empirical positive rate</h2>'
                     + _table(['model', 'score bin', 'n', 'mean score',
                               'positive rate'], crows))

    # confidence intervals from the manifests
    crows = []
    for m in models:
        ci = (data['models'][m]['manifest'].get('confidence_intervals') or {})
        for split, block in ci.items():
            for metric, v in (block.get('metrics') or {}).items():
                crows.append([escape(LABEL[m]), escape(split), escape(metric),
                              f'{block.get("rows")}', _fmt(v.get('point')),
                              _fmt(v.get('low')), _fmt(v.get('high'))])
    if crows:
        parts.append('<h2>Bootstrap confidence intervals (pair resampling)</h2>' + _table(
            ['model', 'split', 'metric', 'n', 'point', 'low', 'high'], crows))
    return HTMLResponse('\n'.join(parts))


@router.get('/compare/plot/{run_tag}/{name}')
def compare_plot(run_tag: str, name: str):
    from fastapi import HTTPException
    from fastapi.responses import Response
    if '/' in name or '..' in name or name not in _plots(run_tag, analyse(run_tag) or {}):
        if not plot_path(run_tag, name).is_file():
            raise HTTPException(404, 'plot not found')
    path = plot_path(run_tag, name)
    if not path.is_file():
        raise HTTPException(404, 'plot not found')
    return Response(path.read_bytes(), media_type='image/png')