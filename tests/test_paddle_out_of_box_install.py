from pathlib import Path

import nbformat


def test_pyproject_uses_paddle_doc_parser_extra() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'paddleocr[doc-parser]>=3.0.0' in text


def test_notebook_bootstraps_paddle_doc_parser_stack() -> None:
    nb = nbformat.read("notebooks/task2_temporal_graph_validation_colab.ipynb", as_version=4)
    cell2 = nb.cells[2].source
    assert 'PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK' in cell2
    assert 'ensure_paddle_stack' in cell2
    assert 'paddleocr[doc-parser]>=3.0.0' in cell2
    assert 'langchain-community>=0.3' in cell2
    assert 'langchain-text-splitters>=0.3' in cell2
    assert 'paddlepaddle==3.2.0' in cell2
    assert 'ensure_paddle_stack()' in cell2
