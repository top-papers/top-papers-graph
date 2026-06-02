# validation/

Отчёт валидации последней сборки. Создаётся `scripts/build.py`.

## Файл

| файл | что внутри |
|---|---|
| `VALIDATION_LOG.txt` | статус каждого кейса (READY/NOT), длина `creator_prompt`, найденные проблемы (`PROB:…`) и распределение страт. |

## Как читать
- `total=N` — сколько кейсов в наборе; `all_ready=True` — все проходят правило готовности.
- Строка кейса `NOT … PROB:required:…` показывает, каких полей не хватает (правило как
  в форме: непусты `paper_title`, `creator_prompt`, `creator_rationale`, `review_focus`≥1
  и хотя бы одно `match.*`).
- Блок `--- распределение страт ---` — счётчики и доли `multimodal_hard` /
  `temporal_hard` / `easy_control` (информативно, без жёсткой цели).

## Воспроизведение
`python3 scripts/build.py` (из корня репозитория).
