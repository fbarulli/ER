"""Result-archive member contract: selected-only checkpoints, no profiling.

The result archive ships what reports/publication consume; resume-only state
(optimizer/scheduler, non-selected epoch checkpoints) ships only in the recovery
archive (owner directive 2026-10-08). One predicate keeps the archive walk and
the per-track inventory in agreement.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.bundle import Bundle
from model_tracks.resume import selected_checkpoint_dirs


def test_profiling_data_is_dropped() -> None:
    assert not Bundle.is_result_member('text/profiles/r-t/fold0/training_trace.json')
    assert not Bundle.is_result_member('hybrid/profiles/x/fold0/training_trace.json')
    assert not Bundle.is_result_member('resource_profile/gpu.csv')


def test_resume_only_filenames_dropped_even_when_selected() -> None:
    selected = frozenset({'text/_checkpoints/m/r-t_f0/checkpoint-281'})
    for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin', 'scaler.pt'):
        assert not Bundle.is_result_member(
            f'text/_checkpoints/m/r-t_f0/checkpoint-281/{name}',
            selected_checkpoints=selected)


def test_only_selected_checkpoint_kept() -> None:
    selected = frozenset({'text/_checkpoints/m/r-t_f0/checkpoint-281'})
    assert Bundle.is_result_member(
        'text/_checkpoints/m/r-t_f0/checkpoint-281/model.safetensors',
        selected_checkpoints=selected)
    assert Bundle.is_result_member(
        'text/_checkpoints/m/r-t_f0/checkpoint-281/trainer_state.json',
        selected_checkpoints=selected)
    assert not Bundle.is_result_member(
        'text/_checkpoints/m/r-t_f0/checkpoint-562/model.safetensors',
        selected_checkpoints=selected)


def test_no_selection_drops_all_checkpoints() -> None:
    # Fail-closed: without a resolved selection no checkpoint member ships, so a
    # broken selection can never silently ship every epoch's weights.
    assert not Bundle.is_result_member('text/_checkpoints/m/r-t_f0/checkpoint-281/model.safetensors')


def test_ordinary_members_kept() -> None:
    assert Bundle.is_result_member('text/text__vectors.npz')
    assert Bundle.is_result_member('suite_events.jsonl')
    assert Bundle.is_result_member('baseline/shared_minilm__embeddings.npz')
    assert not Bundle.is_result_member('.env')
    assert not Bundle.is_result_member('wandb/run-1/files/config.yaml')


def test_selected_checkpoint_dirs_text_and_graph(tmp_path: Path) -> None:
    text_ck = tmp_path / 'text/_checkpoints/m/r-text_f0/checkpoint-281'
    text_ck.mkdir(parents=True)
    (text_ck / 'trainer_state.json').write_text(json.dumps({
        'best_model_checkpoint': 'checkpoint-281', 'best_metric': 0.9, 'global_step': 1124}))
    (tmp_path / 'text/_checkpoints/m/r-text_f0/checkpoint-562').mkdir()

    graph_ck = tmp_path / 'hybrid/hybrid__t/_checkpoints/hybrid/t_f0/checkpoint-3'
    graph_ck.mkdir(parents=True)
    (graph_ck / 'hybrid__graph_model.pt').write_bytes(b'w')
    (tmp_path / 'hybrid/hybrid__best_checkpoint.json').write_text(json.dumps({
        'path': str(graph_ck / 'hybrid__graph_model.pt'), 'metric': 1.0}))

    selected = selected_checkpoint_dirs(tmp_path)
    assert 'text/_checkpoints/m/r-text_f0/checkpoint-281' in selected
    assert 'text/_checkpoints/m/r-text_f0/checkpoint-562' not in selected
    assert 'hybrid/hybrid__t/_checkpoints/hybrid/t_f0/checkpoint-3' in selected
