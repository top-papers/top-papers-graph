# Исправления по логам Kaggle

Исправлены проблемы, выявленные по `errors.txt`:

1. `pip` вызывал конфликтные обновления системных пакетов Kaggle.
   - Notebook теперь ставит только отсутствующие пакеты.
   - Добавлен `pip install -e <repo> --no-deps`, чтобы пакет `scireason` был импортируем без лишних апгрейдов окружения.

2. `gdown --id` выдавал FutureWarning.
   - Notebook теперь использует `gdown --fuzzy` и строит Google Drive URL из file id.

3. `gdown` мог скачать HTML вместо ZIP.
   - Добавлена проверка `zipfile.is_zipfile(...)` и понятная ошибка с preview содержимого.

4. `scireason.mm.vlm_worker` не находился в дочернем процессе.
   - В `src/scireason/mm/vlm.py` дочерний worker теперь запускается с корректным `PYTHONPATH` и `cwd`, включающими `repo_root` и `repo_root/src`.
   - Runner и notebook тоже экспортируют `PYTHONPATH` перед запуском.

5. Убран `SyntaxWarning` из `src/scireason/llm.py` для строки с JSON escape-последовательностями.
