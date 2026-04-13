# KAGGLE REPO ACQUISITION + QWEN MODEL CONFIG UPDATE

Что изменено:
- notebook сначала ищет локальный репозиторий в `/kaggle/input`, `/kaggle/working`, текущей папке и `/mnt/data`;
- если локальный репозиторий не найден, пробует `git clone` из `repo_git_url`, но только если DNS резолвит хост GitHub;
- если `github.com` не резолвится, notebook не вызывает `git clone` и выдаёт понятную инструкцию прикрепить repo как Kaggle dataset;
- в `CFG` добавлены значения по умолчанию для:
  - Model α: `Qwen/Qwen2.5-VL-7B-Instruct`
  - Model β: `Qwen/Qwen3-VL-8B-Instruct`
- в `CFG` добавлены ссылки на model cards Hugging Face для обеих моделей.

Практический эффект:
- исчезает бессмысленная ошибка `fatal: unable to access ... Could not resolve host: github.com` в среде без доступа к GitHub;
- notebook продолжает работать через локальный архив/датасет репозитория, что безопаснее для Kaggle Save Version.
