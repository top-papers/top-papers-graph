# DataSphere DPO: исправление сериализации run_config.json с pathlib.Path

## Что произошло

Новый DataSphere-прогон прошел дальше предыдущих падений:

1. датасет экспортировался и прошел quality gates;
2. базовая модель `Qwen/Qwen3-VL-8B-Instruct` была скачана в HF cache;
3. SFT-стадия успешно завершила обучение и финальное сохранение адаптера;
4. DPO-стадия стартовала, подготовила mixed VLM/DPO датасет и дошла до записи `run_config.json`.

После этого оба DDP-rank процесса упали на одной и той же строке:

```text
TypeError: Object of type PosixPath is not JSON serializable
```

Причина: `train_vlm_dpo.py` строил `run_config = vars(args).copy()`, где `argparse` содержит несколько параметров типа `pathlib.Path` (`--train-file`, `--eval-file`, `--output-dir`, `--sft-adapter-path`). Затем DPO-скрипт вызывал `json.dumps(run_config, ensure_ascii=False, indent=2)` без `default=str`. В SFT/GRPO этот guard уже был, а в DPO — нет.

## Исправление

Файл: `experiments/vlm_finetuning/scripts/train_vlm_dpo.py`

- Запись `run_config.json` теперь использует `json.dumps(..., default=str)`.
- `--dry-run` печатает тот же JSON-safe config через `default=str`.
- Поведение выровнено с `train_vlm_sft.py` и `train_vlm_grpo.py`.

## Регрессионный тест

Файл: `tests/test_vlm_dpo_attention_resolution.py`

Добавлен тест `test_dpo_run_config_json_handles_path_arguments`, который проверяет, что DPO run config с `Path`-аргументами сериализуется в JSON без `TypeError`.

## Проверки

```bash
python3 -m py_compile experiments/vlm_finetuning/scripts/train_vlm_dpo.py tests/test_vlm_dpo_attention_resolution.py
python3 -m pytest -q tests/test_vlm_dpo_attention_resolution.py \
  tests/test_vlm_training_format_normalization.py \
  tests/test_scireason_alignment_dataset_v2.py \
  tests/test_audit_full_data_usage.py \
  tests/test_datasphere_job_configs.py
```

Результат:

```text
70 passed
```

## Ожидаемый эффект

Следующий DataSphere-прогон не должен падать при записи DPO `run_config.json` из-за `PosixPath`. Если далее появится новая ошибка, она уже будет находиться после этой точки — в фактическом DPO train/eval/save этапе.
