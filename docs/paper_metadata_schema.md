# Paper Metadata Schema

Единый формат метаданных статьи (`PaperMetadata`) нужен, чтобы:

- объединять данные из разных источников (Crossref, OpenAlex, Semantic Scholar, PubMed, Europe PMC, bioRxiv/medRxiv, arXiv)
- строить граф цитирований / знаний с минимальным количеством условностей
- стандартизировать идентификаторы (DOI/PMID/arXiv/OpenAlex/S2) и ссылки

## Модель

Файл: `src/scireason/papers/schema.py`

Ключевые идеи:

- `id` — канонический идентификатор с префиксом: `doi:... | pmid:... | arxiv:... | openalex:... | s2:...`
- `ids` (`ExternalIds`) — все внешние идентификаторы, которые удалось извлечь
- `raw` — исходная запись источника (для отладки/обогащения)

## Нормализация источников

Файл: `src/scireason/papers/normalize.py`

Каждая функция `normalize_<source>(raw_record)` возвращает `PaperMetadata`.

## Resolver

Файл: `src/scireason/papers/resolver.py`

`resolve_ids()` делает best-effort сопоставление между идентификаторами:

- DOI ⇄ PMID/PMCID (NCBI idconv, затем Europe PMC)
- DOI ⇄ OpenAlexID (OpenAlex /works endpoint)
- arXivID → DOI (arXiv Atom API)

## FastAPI

Файл: `src/scireason/api/app.py`

Использует `PaperMetadata` как response_model, чтобы наружный API был строго типизирован.
