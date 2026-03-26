from __future__ import annotations

from pathlib import Path

import nbformat


def _load_notebook_source() -> tuple[str, str]:
    nb_path = Path(__file__).resolve().parents[1] / 'notebooks' / 'task2_temporal_graph_validation_colab.ipynb'
    nb = nbformat.read(nb_path, as_version=4)
    return nb.cells[2].source, nb.cells[4].source


def test_notebook_import_cell_uses_full_task2_module_without_stub() -> None:
    cell2, _ = _load_notebook_source()

    assert 'import scireason.task2_validation as _task2_validation_module' in cell2
    assert "TASK2_NOTEBOOK_PIPELINE_MODE = 'full'" in cell2
    assert 'TASK2_NOTEBOOK_PIPELINE_SOURCE' in cell2
    assert 'prepare_task2_validation_bundle as _prepare_task2_validation_bundle' not in cell2
    assert 'def build_task2_validation_bundle(*args, **kwargs):' not in cell2


def test_notebook_run_button_guard_rejects_fallback_mode() -> None:
    _, cell4 = _load_notebook_source()

    assert "globals().get('TASK2_NOTEBOOK_PIPELINE_MODE') != 'full'" in cell4
    assert 'scireason.task2_validation без fallback-ветки' in cell4
    assert 'bundle = build_task2_validation_bundle(' in cell4
