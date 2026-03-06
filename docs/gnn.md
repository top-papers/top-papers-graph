# Temporal link prediction and optional static GNN baseline

Репозиторий теперь различает два режима поиска недостающих связей:

1. **TGNN / TGN-style temporal prediction** — основной режим
2. **Static GNN (PyTorch Geometric / GraphSAGE)** — optional baseline для ablation studies

## Основной режим: TGNN / TGN-style

По умолчанию проект предпочитает event-stream temporal prediction.

### Что происходит

Во время `generate_candidates(...)`:

1. Из temporal KG строится chronological event stream.
2. Считаются recency-aware node memories.
3. Вычисляются temporal common neighbors и pair recurrence.
4. Формируются top-k кандидаты `kind = tgnn_missing_link`.

### Почему это лучше для temporal KG

Static GNN видит только агрегированный граф. TGNN/TGN-style predictor использует порядок
временных событий, поэтому лучше подходит для задач вида:
- emerging relations
- future edge prediction
- temporal hypothesis discovery

### Knobs

```bash
HYP_TGNN_ENABLED=1
HYP_TGNN_RECENT_WINDOW_YEARS=3
HYP_TGNN_HALF_LIFE_YEARS=2.0
HYP_TGNN_MIN_CANDIDATE_SCORE=0.05
```

### CLI

```bash
top-papers-graph train-tgn --temporal-kg-json runs/<run_id>/temporal_kg.json
```

Или экспортировать event layer из Neo4j:

```bash
top-papers-graph export-temporal-events --out runs/temporal_events.json
```

## Optional static GNN baseline (PyTorch Geometric)

Static GraphSAGE link prediction сохранён как baseline.

### Install

```bash
pip install -e ".[gnn]"
```

Для лучшей производительности можно также установить PyG extension wheels.

### Enable baseline

```bash
HYP_GNN_ENABLED=1
```

Дополнительные параметры:

```bash
HYP_GNN_EPOCHS=80
HYP_GNN_HIDDEN_DIM=64
HYP_GNN_LR=0.01
HYP_GNN_NODE_CAP=300
```

### Что делает baseline

1. Строит NetworkX граф из temporal KG.
2. Берёт индуцированный подграф top-degree узлов.
3. Обучает небольшой GraphSAGE encoder + dot-product decoder.
4. Возвращает top-k кандидатов `kind = gnn_missing_link`.

Если PyG не установлен, пайплайн не падает: просто остаётся temporal predictor + эвристики.
