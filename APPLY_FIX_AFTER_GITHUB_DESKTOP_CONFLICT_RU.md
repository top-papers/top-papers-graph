# Как применить исправленный архив после merge conflict в GitHub Desktop

GitHub Desktop показывает `Resolve conflicts before Merge`, когда репозиторий находится в незавершённом merge-состоянии. Если после замены файлов окно осталось, значит Git всё ещё видит конфликтные маркеры в рабочей копии или сам merge не был завершён.

## Безопасный порядок действий

1. В GitHub Desktop нажмите **Abort merge**.
2. Закройте GitHub Desktop.
3. Распакуйте этот архив в отдельную папку.
4. Скопируйте содержимое папки `top-papers-graph-main` поверх содержимого вашей локальной папки репозитория `top-papers-graph` с заменой файлов.
5. Откройте терминал в локальной папке репозитория и выполните:

```bash
git status
grep -nR -E "<<<<<<<|=======|>>>>>>>" experiments/vlm_finetuning/scripts/train_vlm_grpo.py experiments/vlm_finetuning/scripts/train_vlm_sft.py
python -S -m py_compile experiments/vlm_finetuning/scripts/train_vlm_grpo.py experiments/vlm_finetuning/scripts/train_vlm_sft.py
git add experiments/vlm_finetuning/scripts/train_vlm_grpo.py experiments/vlm_finetuning/scripts/train_vlm_sft.py top_papers_graph_scidatapipe_hf_colab_from_csv_only_fixed_gdown_scope_fixed.ipynb
git commit -m "Fix VLM SFT and GRPO dataset formats"
git pull --rebase origin main
git push origin main
```

Если `grep` ничего не выводит, conflict markers в двух train-скриптах удалены.

## Что исправлено

- `train_vlm_sft.py`: поддержка `chat` / `chat.messages`, нормализация в `messages`, вынос embedded image paths в `images`.
- `train_vlm_grpo.py`: поддержка `prompt_chat` / `prompt_chat.messages`, нормализация в `prompt`, вынос embedded image paths в `images`.
- Блокнот: `MAX_MULTIMODAL_RECORDS_PER_SAMPLE = 8` вместо `0`, чтобы настройка не выглядела как отключение multimodal-записей.
