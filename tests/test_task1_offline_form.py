from pathlib import Path

import nbformat

from scireason.task1_offline_form import build_task1_offline_form


def test_build_task1_offline_form_writes_embedded_html(tmp_path: Path) -> None:
    output = tmp_path / 'task1_offline.html'
    build_task1_offline_form(
        output,
        initial_state={
            'topic': 'Battery degradation',
            'expert': {'last_name': 'Иванов', 'first_name': 'Иван', 'patronymic': '-'},
        },
        domain_configs=[
            {
                'domain_id': 'science',
                'title': 'Science',
                'wikidata_qid': 'Q336',
                'required_conditions': ['temperature'],
                'source_path': 'embedded',
            }
        ],
    )
    html = output.read_text(encoding='utf-8')
    assert 'Task 1 — автономная форма Reasoning Trajectories' in html
    assert 'Скачать YAML' in html
    assert 'temperature' in html
    assert 'Battery degradation' in html
    assert 'task1-offline-form-v1' in html


def test_task1_notebook_contains_offline_download_button() -> None:
    notebook_path = Path('notebooks/task1_reasoning_trajectories_form_colab.ipynb')
    nb = nbformat.read(notebook_path, as_version=4)
    source = '\n'.join(cell.source for cell in nb.cells if cell.cell_type == 'code')
    assert '⬇️ Скачать автономную форму' in source
    assert 'def _download_offline_form' in source
    assert 'build_task1_offline_form' in source
