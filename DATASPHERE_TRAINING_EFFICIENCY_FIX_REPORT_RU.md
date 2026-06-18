# Исправления DataSphere VLM SFT/GRPO обучения

Дата: 2026-06-18

## Что было найдено по логам

1. **Критическая ошибка GRPO на сохранении чекпоинта**: `NameError: name 'is_main_process' is not defined` в `train_vlm_grpo.py` внутри отключения опционального TRL model card. Из-за этого DataSphere job завершался с кодом 1 после уже выполненной части обучения.
2. **Лишний DDP overhead**: обе стадии запускались с `--ddp-find-unused-parameters`, а PyTorch DDP предупреждал, что unused-параметров не найдено и включенный флаг добавляет лишний обход autograd-графа.
3. **Слабый GRPO-сигнал**: в GRPO логе встречались шаги с `loss: 0.0`, `grad_norm: 0.0`, `reward_std: 0.0`, `frac_reward_zero_std: 1.0`. Это означает, что внутри групп генераций не было различий по reward, поэтому GRPO не получал полезного advantage-сигнала.
4. **SFT стадия в целом работала**: SFT дошла до конца 480 шагов, eval loss снижался примерно с 1.51 до 1.23, поэтому основной блокирующий дефект был в GRPO.

## Внесенные изменения

### `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`

- Добавлена функция `is_main_process()`, чтобы `disable_trl_model_card_creation()` больше не падал на checkpoint save.
- `ddp_find_unused_parameters` теперь по умолчанию `False`; включается только явным CLI-флагом.
- Для `assertion_review_rl` графовая reward-компонента больше не штрафует отсутствие реконструированного графа, если задача является review/verdict, а не reconstruction.
- `reward_expert_override_match()` теперь различает:
  - точный вердикт: `1.0`,
  - валидный, но неверный вердикт: `-0.75`,
  - отсутствующий/невалидный вердикт: `-1.0`.
  Это дает более плотный сигнал и снижает вероятность полностью одинакового reward по группе.

### `experiments/vlm_finetuning/scripts/train_vlm_sft.py`

- `ddp_find_unused_parameters` также переведен в быстрый default `False` с явным escape hatch через `--ddp-find-unused-parameters`.

### `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`

- Удалена принудительная передача `--ddp-find-unused-parameters`.
- Добавлены переменные:
  - `SFT_DDP_FIND_UNUSED_PARAMETERS`
  - `GRPO_DDP_FIND_UNUSED_PARAMETERS`
- Если переменная равна `1/true/yes/on`, wrapper передает `--ddp-find-unused-parameters`; если равна `0/false/no/off`, передает `--no-ddp-find-unused-parameters`.

### DataSphere job configs

- В full/smoke конфиги добавлены `SFT_DDP_FIND_UNUSED_PARAMETERS: 0` и `GRPO_DDP_FIND_UNUSED_PARAMETERS: 0`.
- В full GRPO увеличено `GRPO_NUM_GENERATIONS` с 2 до 4 для лучшей внутригрупповой reward-дисперсии.
- В full GRPO уменьшено `MAX_GRPO_STEPS` с 160 до 120, чтобы компенсировать более дорогие группы генераций.
- В full GRPO поднята exploration temperature до `1.0`.
- В smoke SFT выставлен `SFT_DATALOADER_NUM_WORKERS: 0`, чтобы smoke-тесты были ближе к стабильному full-run поведению и меньше зависели от multiprocessing overhead.

### Тесты

Обновлены regression-тесты:

- DDP unused-parameter detection теперь проверяется как выключенный по умолчанию и включаемый явно.
- Добавлен тест на наличие `is_main_process()` в GRPO.
- Добавлены тесты на нейтральный graph reward для review-задач и на более плотный verdict reward.
- Добавлен тест, что DataSphere wrapper делает DDP unused detection конфигурируемым, а не принудительным.

## Проверка

Выполнено:

```bash
python -m py_compile experiments/vlm_finetuning/scripts/train_vlm_sft.py experiments/vlm_finetuning/scripts/train_vlm_grpo.py
python -m pytest tests/test_vlm_training_format_normalization.py tests/test_datasphere_job_configs.py -q
```

Результат целевых тестов: **44 passed**.

Дополнительно был запущен полный `pytest -q`; он дошел до значительной части suite без видимых failures, но был остановлен лимитом времени окружения, поэтому как полную проверку его не учитываю.

## Как запускать после патча

Обычный рекомендуемый запуск full config:

```bash
datasphere project job execute \
  -p <PROJECT_ID> \
  -c experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml
```

Если в будущем появится реальная DDP ошибка про unused parameters, можно временно включить диагностику/совместимость:

```bash
SFT_DDP_FIND_UNUSED_PARAMETERS=1 GRPO_DDP_FIND_UNUSED_PARAMETERS=1 \
  bash experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh
```

По умолчанию это выключено, потому что в текущих логах DDP сам сообщает, что unused-параметров не нашел.
