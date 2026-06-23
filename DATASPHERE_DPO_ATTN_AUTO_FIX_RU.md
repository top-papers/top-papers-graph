# DataSphere DPO attention auto fix

## Причина падения

Последний прогон дошел до `train_vlm_dpo.py`, после чего оба DDP-rank процесса упали на загрузке `Qwen3VLForConditionalGeneration.from_pretrained(...)` с ошибкой:

```text
ValueError: Specified `attn_implementation="auto"` is not supported.
```

SFT и GRPO entrypoints уже резолвили `auto` в один из поддерживаемых backend-ов (`flash_attention_2`, если установлен `flash_attn`, иначе `sdpa`). DPO entrypoint передавал строку `auto` напрямую в Transformers, поэтому full pipeline ломался именно на DPO-стадии.

## Исправление

- В `experiments/vlm_finetuning/scripts/train_vlm_dpo.py` добавлены:
  - `import importlib.util`;
  - `_flash_attn_available()`;
  - `resolve_attn_implementation()`;
  - fallback `flash_attention_2 -> sdpa`, если загрузка модели с FlashAttention падает.
- DPO model loader теперь передает в `from_pretrained()` только поддерживаемые backend значения, а не пользовательский sentinel `auto`.
- Добавлен regression test `tests/test_vlm_dpo_attention_resolution.py`, который проверяет, что DPO не форвардит `auto` в model loader.

## Проверки

```bash
python -m py_compile tests/test_vlm_dpo_attention_resolution.py experiments/vlm_finetuning/scripts/train_vlm_dpo.py
python -m pytest -q tests/test_vlm_dpo_attention_resolution.py \
  tests/test_vlm_training_format_normalization.py \
  tests/test_scireason_alignment_dataset_v2.py \
  tests/test_audit_full_data_usage.py \
  tests/test_datasphere_job_configs.py
```

Результат targeted suite: `67 passed`. Полный `pytest -q` запускался отдельно, но был остановлен по таймауту окружения после первых успешно стартовавших тестов; падения тестов в этом запуске не зафиксировано.
