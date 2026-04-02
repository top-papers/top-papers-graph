from __future__ import annotations

from scireason.ingest.acquire import candidate_pdf_urls
from scireason.papers.schema import ExternalIds, PaperMetadata, PaperSource


def test_wikipedia_entries_do_not_generate_pdf_candidates() -> None:
    meta = PaperMetadata(
        id='https://en.wikipedia.org/wiki/Word2vec',
        source=PaperSource.unknown,
        title='Word2vec',
        year=2013,
        url='https://en.wikipedia.org/wiki/Word2vec',
        pdf_url=None,
        ids=ExternalIds(),
    )
    assert candidate_pdf_urls(meta) == []
