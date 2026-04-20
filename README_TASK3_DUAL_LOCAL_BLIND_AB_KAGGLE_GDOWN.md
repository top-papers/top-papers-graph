# Task 3 dual-local blind A/B for Kaggle Save Version + gdown

Этот вариант предназначен для Kaggle notebook в режиме **Save Version / Run All**.

Что изменено:

- orchestration вынесен в Python-скрипт `scripts/kaggle/run_task3_dual_local_blind_ab.py`;
- notebook только:
  - находит и распаковывает архив репозитория,
  - ставит зависимости,
  - скачивает ZIP с экспертным YAML через `gdown`,
  - извлекает YAML,
  - собирает CLI-аргументы,
  - запускает `.py`-скрипт;
- все артефакты сохраняются в `/kaggle/working/...` и попадают во вкладку **Output** после `Save Version`.

Основные выходные файлы:

- `variant_alpha/...`
- `variant_beta/...`
- `task3_dual_run_manifest.json`
- `*_kaggle_outputs.zip`
- blind offline HTML
- owner-only key
- expert ZIP


## Рекомендация для VLM A/B

Для сравнения baseline VLM vs SFT/DPO VLM используйте не случайный набор статей, а curated **hard subset**.
Лучше подключать заранее подготовленный `processed_dir` и повышать `top_pairs` минимум до 16.
Подробный туториал: `docs/TASK3_AB_HARD_SUBSET_TUTORIAL_RU.md`.
Шаблоны для эксперта: `data/experts/mm_ab_reviews/`.
