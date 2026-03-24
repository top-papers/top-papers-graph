from __future__ import annotations

from pathlib import Path

import nbformat


def test_notebook_bootstrap_can_extract_local_repo_zip_before_git_clone() -> None:
    nb_path = Path(__file__).resolve().parents[1] / 'notebooks' / 'task2_temporal_graph_validation_colab.ipynb'
    nb = nbformat.read(nb_path, as_version=4)
    cell2 = nb.cells[2].source

    assert 'import json, os, sys, tempfile, subprocess, zipfile' in cell2
    assert 'def iter_repo_archives():' in cell2
    assert 'def extract_repo_archive(archive_path: Path) -> Path:' in cell2
    assert "top-papers-graph*.zip" in cell2
    assert 'zipfile.ZipFile(archive_path)' in cell2
    assert 'Не удалось найти локальный репозиторий и не получилось использовать локальный ZIP-архив' in cell2
