# smolagents: поддержка агентного режима (код-агент) с локальными HF и g4f

В проекте есть **два** варианта «код‑агента» (агента, который решает задачу написанием Python‑кода):

1) **Встроенный** (по умолчанию) — `HYP_AGENT_BACKEND=internal`
   - максимально простой и прозрачный (см. `src/scireason/agentic/code_agent.py`)
   - работает полностью **оффлайн** с `LLM_PROVIDER=mock`

2) **Hugging Face smolagents** — `HYP_AGENT_BACKEND=smolagents`
   - использует официальный `smolagents.CodeAgent`
   - поддерживает разные бэкенды моделей, включая **локальные** модели через Transformers и внешние API

## Установка

### Минимально
```bash
pip install -e ".[agents]"
```

### Локальные модели Hugging Face (TransformersModel)
```bash
pip install -e ".[agents_hf]"
# или напрямую
pip install "smolagents[transformers]"
```

### smolagents + g4f
```bash
pip install -e ".[agents,g4f]"
```

## Быстрый запуск (demo)

```bash
# 1) включаем smolagents-агента
export HYP_AGENT_BACKEND=smolagents

# 2) выбираем модельный бэкенд для smolagents
# Вариант A (рекомендуется для совместимости): использовать общий роутер проекта
export SMOL_MODEL_BACKEND=scireason

# Вариант B: прямой g4f
# export SMOL_MODEL_BACKEND=g4f
# export SMOL_G4F_MODEL=auto

# Вариант C: локальная HF модель
# export SMOL_MODEL_BACKEND=transformers
# export SMOL_MODEL_ID=HuggingFaceTB/SmolLM-135M-Instruct

top-papers-graph demo-run --edge-mode cooccurrence
```

## Параметры окружения

- `HYP_AGENT_BACKEND=internal|smolagents`
- `SMOL_MODEL_BACKEND=scireason|g4f|transformers`
- `SMOL_MODEL_ID=<hf_model_id_or_path>` (для `transformers`)
- `SMOL_MAX_NEW_TOKENS=768`
- `SMOL_DEVICE_MAP=auto` (если нужен, для `transformers`)
- `SMOL_TORCH_DTYPE=float16|bfloat16` (если нужен)
- `SMOL_G4F_MODEL=auto|<model_name>`
- `SMOL_EXECUTOR=local|docker` (для sandboxes; `docker` требует Docker)
- `SMOL_PRINT_STEPS=1` (для подробного лога шагов агента)

## Как это встроено в пайплайн

smolagents включается **только** в части генерации кандидатов‑гипотез:

- `src/scireason/agents/graph_candidate_agent.py` → `agent_generate_candidates()`
  - backend `internal`: наш минимальный агент
  - backend `smolagents`: `smolagents.CodeAgent` + инструменты из `src/scireason/smolagents_integration/tools.py`

Дальше эти кандидаты подаются в общий ранкер/писатель гипотез.

## Запуск без `.env`: CLI-флаги

Чтобы студентам **не нужно было** настраивать `.env`, добавлены CLI‑переопределения:

- `--smol-model-backend scireason|transformers|g4f`
- `--smol-model-id <hf_model_id_or_local_path>` (для `transformers`)

Пример (smolagents + оффлайн mock‑LLM):

```bash
top-papers-graph demo-run \
  --agent-backend smolagents \
  --llm-provider mock \
  --smol-model-backend scireason
```

Пример (smolagents + локальная HF модель через Transformers):

```bash
pip install -e ".[agents_hf]"

top-papers-graph demo-run \
  --agent-backend smolagents \
  --llm-provider mock \
  --smol-model-backend transformers \
  --smol-model-id HuggingFaceTB/SmolLM2-1.7B-Instruct
```
