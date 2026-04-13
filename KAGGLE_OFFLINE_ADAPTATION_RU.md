# Kaggle offline adaptation for Task 3 notebook

Сделано для сценария Kaggle **Save Version** без интернета.

Что изменено:
- добавлена верхняя кодовая ячейка с `TASK3_DUAL_SETUP` для автономного запуска без ручных кликов по widgets;
- ранний и основной конфиг теперь по умолчанию используют `out_dir` внутри `/kaggle/working`;
- быстрый валидатор разрешает сценарий `processed_path` без обязательного `query`;
- notebook ищет репозиторий и архивы также в `/kaggle/input` и `/kaggle/working`;
- в Kaggle/offline режиме отключён обязательный `git clone`, вместо него ожидается локальный архив/датасет или `TPG_REPO_DIR`;
- editable install переведён на безопасный offline fallback `-e . --no-deps`;
- после успешного запуска создаются `task3_dual_run_manifest.json` и дополнительный zip с артефактами.

Рекомендуемый сценарий на Kaggle:
1. Прикрепить датасет с репозиторием/архивом и входными файлами.
2. Отредактировать верхнюю ячейку `TASK3_DUAL_SETUP`.
3. Нажать **Save Version**.
4. Забрать результаты из Output и `/kaggle/working`.
