# Sources / API connectors

Проект **top-papers-graph** умеет искать метаданные публикаций через несколько крупнейших открытых API.

> ⚠️ Важно: многие API рекомендуют указывать корректный `User-Agent` и email ("polite pool").
> Укажите `CONTACT_EMAIL` (и при желании `USER_AGENT`) в `.env`.

## CLI

Единая команда:

```bash
top-papers-graph fetch "graph rag" --source openalex --limit 10 --out data/papers/openalex.json
```

Поддерживаемые источники:

- `arxiv` — arXiv Atom API
- `openalex` — OpenAlex Works API
- `s2` — Semantic Scholar Graph API
- `crossref` — Crossref REST API (часто удобно для подбора DOI)
- `pubmed` — NCBI PubMed E-utilities (ESearch + ESummary; опционально EFetch для абстрактов)
- `europepmc` — Europe PMC (агрегатор PubMed + PMC + preprints и др.)
- `biorxiv` / `medrxiv` — bioRxiv/medRxiv API (details)

## OpenAlex

> Начиная с 2026-02-13, OpenAlex требует API key для нормальной работы (бесплатный ключ даёт больше дневных кредитов).

Переменные окружения:
- `OPENALEX_API_KEY`
- `OPENALEX_MAILTO` (опционально; иначе берётся из `CONTACT_EMAIL`)

## PubMed

```bash
top-papers-graph fetch "graph neural network survey" --source pubmed --limit 20 --with-abstract false
```

Переменные окружения:
- `NCBI_API_KEY` (опционально)
- `NCBI_TOOL` (по умолчанию `top-papers-graph`)
- `NCBI_EMAIL` (если пусто — берётся из `CONTACT_EMAIL`)

## bioRxiv / medRxiv

По DOI:

```bash
top-papers-graph fetch "10.1101/2020.09.09.20191205" --source medrxiv --out data/papers/medrxiv.json
```

По интервалу:

```bash
top-papers-graph fetch "2025-03-21/2025-03-28" --source biorxiv --cursor 0 --category cell_biology
```
