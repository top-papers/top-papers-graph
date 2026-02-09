# Paper (Agents4Science) — сборка

## Важно
Конференция требует официальный LaTeX шаблон и AI Contribution Disclosure checklist.
Шаблон скачиваем скриптом ниже.

## 1) Скачать официальный шаблон
Из папки `paper/agents4science` выполните:

```bash
bash fetch_template.sh
```

Скрипт скачает архив и распакует его в `paper/agents4science/template/`.

## 2) Собрать PDF
```bash
make paper
```

или вручную:
```bash
cd paper/agents4science
latexmk -pdf -interaction=nonstopmode main.tex
```

## 3) Структура
- `main.tex` — основной файл статьи (заготовка)
- `sections/` — секции
- `statements/` — disclosure/reproducibility/responsible AI
- `refs.bib` — библиография
- `notes/` — протоколы решений/заметки команды
