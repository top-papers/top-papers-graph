# Patch report

Исправления в этом архиве:

- `src/scireason/temporal/schemas.py`: добавлена мягкая нормализация `subject`, `predicate`, `object`, `evidence_quote`, `start/end`, `ts_start/ts_end` до Pydantic-валидации.
- `src/scireason/temporal/temporal_triplet_extractor.py`: невалидные не-словарные ответы LLM теперь игнорируются, а пустые триплеты фильтруются перед возвратом.
- `tests/test_task2_pipeline_regressions.py`: добавлен регрессионный тест на нестроковые поля триплета.
- `task2_temporal_graph_validation.ipynb` и `notebooks/task2_temporal_graph_validation_colab.ipynb`: обновлены на версию с отдельной исполняемой ячейкой запуска и compatibility patch для схем Task 2.
