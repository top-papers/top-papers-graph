# Task 2 validation bundle

This repository snapshot includes:

- direct notebook support via `pip install -e ".[task2_notebook]"`
- alias extras: `.[multimodal]`, `.[notebook_viz]`, `.[notebook]`, `.[colab]`
- compatibility module `scireason.task2_validation` for legacy notebooks
- CLI aliases `top-papers-graph prepare-task2-validation` and `top-papers-graph task2-bundle`
- temporal review schema v3 (`start_date`, `end_date`, `valid_from`, `valid_to`)
- backward compatibility with legacy `time_interval`
- notebook `notebooks/task2_temporal_graph_validation_colab.ipynb` that works without patching repository files from inside the notebook

Recommended entrypoint for experts:

```bash
pip install -e ".[task2_notebook]"
# or: pip install -e ".[temporal,multimodal,notebook_viz]"
```

Then open the notebook and run it top-to-bottom.

## Что нового в рабочем ноутбуке

- Вкладки для просмотра **эталонного** и **автоматически сгенерированного** графов.
- Внутри каждого графа доступны отдельные окна:
  - **Assertions** — таблица утверждений/рёбер с поиском.
  - **Визуализация** — граф в notebook.
- Раздел **Валидация эксперта** позволяет:
  - выставлять verdict по каждому ребру;
  - добавлять комментарии;
  - корректировать временные параметры (`start_date`, `end_date`, `valid_from`, `valid_to`, `time_source`);
  - сохранять готовые `CSV`, `JSON` и `ZIP` с результатами валидации.
