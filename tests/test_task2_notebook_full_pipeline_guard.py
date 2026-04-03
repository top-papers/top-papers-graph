from __future__ import annotations

from pathlib import Path

import nbformat


def _load_notebook_source() -> str:
    nb_path = Path(__file__).resolve().parents[1] / 'notebooks' / 'task2_temporal_graph_validation_colab.ipynb'
    nb = nbformat.read(nb_path, as_version=4)
    return "\n\n".join(cell.source for cell in nb.cells)


def test_notebook_import_cell_uses_full_task2_module_without_stub() -> None:
    notebook_source = _load_notebook_source()

    assert 'import scireason.task2_validation as _task2_validation_module' in notebook_source
    assert "TASK2_NOTEBOOK_PIPELINE_MODE = 'full'" in notebook_source
    assert 'TASK2_NOTEBOOK_PIPELINE_SOURCE' in notebook_source
    assert 'prepare_task2_validation_bundle as _prepare_task2_validation_bundle' not in notebook_source
    assert 'def build_task2_validation_bundle(*args, **kwargs):' not in notebook_source


def test_notebook_run_button_guard_rejects_fallback_mode() -> None:
    notebook_source = _load_notebook_source()

    assert "globals().get('TASK2_NOTEBOOK_PIPELINE_MODE') != 'full'" in notebook_source
    assert 'scireason.task2_validation без fallback-ветки' in notebook_source
    assert 'bundle = build_task2_validation_bundle(' in notebook_source
