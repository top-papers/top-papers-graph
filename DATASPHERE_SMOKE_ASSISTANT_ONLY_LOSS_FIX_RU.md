# Исправление smoke run: assistant_only_loss для VLM SFT

## Симптом

Новый smoke run проходит CUDA/GPU preflight, скачивает ограниченную smoke-подвыборку, успешно нормализует и валидирует Qwen3-VL messages, загружает модель и LoRA, но падает при создании `SFTTrainer`:

```text
ValueError: Assistant-only loss is not yet supported for vision-language models. Please set `assistant_only_loss=False` in the `SFTConfig`.
```

## Причина

`--assistant-only-loss` был включён в DataSphere wrapper для `train_vlm_sft.py`. В актуальной версии TRL эта опция не поддерживается для vision-language model training и приводит к hard fail до начала обучения.

## Исправления

- Из `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh` удалён флаг `--assistant-only-loss` для VLM SFT.
- В `experiments/vlm_finetuning/scripts/train_vlm_sft.py` добавлен защитный guard: если пользователь всё же передал `--assistant-only-loss` при `actual_mode == "vlm"`, скрипт печатает предупреждение и автоматически выставляет `assistant_only_loss=False` перед созданием `SFTConfig`.
- Добавлены regression tests, чтобы wrapper снова не начал передавать этот флаг для VLM.

## Ожидаемый следующий результат

Следующий smoke run должен пройти создание `SFTTrainer` и начать реальные SFT training steps.
