# Фикс падения full-run SFT на TRL model card / ASCII locale

## Симптом

Полный DataSphere запуск `hf-full-managed` дошёл до SFT step 60, выполнил eval и упал при checkpoint save:

```text
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 ...
```

Traceback показывает, что ошибка возникла не в forward/backward и не из-за GPU/OOM, а в необязательной генерации TRL model card:

```text
SFTTrainer._save_checkpoint -> create_model_card -> huggingface_hub.ModelCard.from_template -> Path(...).read_text()
```

В DataSphere процесс мог стартовать с ASCII locale, поэтому `huggingface_hub` читал UTF-8 template как ASCII.

## Исправления

1. В `run_hf_top_papers_sft_grpo_full.sh` принудительно задаётся UTF-8 окружение до запуска Python:
   - `LANG=C.UTF-8`
   - `LC_ALL=C.UTF-8`
   - `PYTHONUTF8=1`
   - `PYTHONIOENCODING=utf-8`

2. В `train_vlm_sft.py` и `train_vlm_grpo.py` добавлен guard `disable_trl_model_card_creation(...)`.
   По умолчанию он отключает необязательный side effect `trainer.create_model_card`, потому что pipeline уже пишет собственные UTF-8 артефакты:
   - `run_config.json`
   - `planned_run_config.json`
   - `final_summary.json`
   - `budget_plan.json`

   Отключение можно снять через:

   ```bash
   DISABLE_TRL_MODEL_CARD=0
   ```

3. В full/smoke job YAML добавлены env-переменные для UTF-8 locale и отключения TRL model card.

4. Упаковщик артефактов теперь создаёт placeholder-файлы для declared outputs, которые ещё не были произведены при раннем падении:
   - `outputs/*_grpo_lora.tar.gz`
   - `hf_upload_summary.json`
   - `hf_upload_manifest.json`

   Это не скрывает ошибку job — exit code сохраняется, но DataSphere больше не добавляет шум вида `Error while processing file` для отсутствующих declared outputs.

## Проверка

```bash
python -m compileall -q experiments/vlm_finetuning/scripts experiments/vlm_finetuning/datasphere
bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh experiments/vlm_finetuning/datasphere/bin/*.sh
pytest -q tests/test_datasphere_job_configs.py tests/test_vlm_training_format_normalization.py
```

Результат:

```text
35 passed
```
