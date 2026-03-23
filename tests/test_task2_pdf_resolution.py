from __future__ import annotations

import nbformat

from scireason.ingest.acquire import candidate_pdf_urls
from scireason.papers.schema import PaperMetadata, PaperSource
from scireason.pipeline import task2_validation as pipeline


def test_candidate_pdf_urls_derives_acl_pdf_from_landing_page() -> None:
    paper = PaperMetadata(
        id="https://aclanthology.org/2020.emnlp-main.541/",
        source=PaperSource.unknown,
        title="Recurrent Event Network",
        url="https://aclanthology.org/2020.emnlp-main.541/",
    )

    urls = candidate_pdf_urls(paper)

    assert "https://aclanthology.org/2020.emnlp-main.541.pdf" in urls


def test_resolve_papers_from_trajectory_uses_pdf_hint_from_sources_when_offline() -> None:
    doc = {
        "papers": [
            {
                "id": "https://aclanthology.org/2024.acl-long.8/",
                "title": "A Unified Temporal Knowledge Graph Reasoning Model Towards Interpolation and Extrapolation",
                "year": 2024,
            }
        ],
        "steps": [
            {
                "step_id": 1,
                "sources": [
                    {
                        "type": "image",
                        "source": "https://aclanthology.org/2024.acl-long.8.pdf",
                        "locator": "Figure 2",
                    }
                ],
            }
        ],
    }

    resolved = pipeline.resolve_papers_from_trajectory(doc, enable_remote_lookup=False)

    assert len(resolved) == 1
    assert resolved[0].url == "https://aclanthology.org/2024.acl-long.8/"
    assert resolved[0].pdf_url == "https://aclanthology.org/2024.acl-long.8.pdf"


def test_notebook_enables_remote_lookup_in_bundle_kwargs() -> None:
    nb = nbformat.read("notebooks/task2_temporal_graph_validation_colab.ipynb", as_version=4)
    cell4 = nb.cells[4].source

    assert "remote_lookup = W.Checkbox(value=True" in cell4
    assert "'enable_remote_lookup': bool(remote_lookup.value)" in cell4


def test_candidate_pdf_urls_includes_doi_resolver_and_plos_printable_candidates() -> None:
    paper = PaperMetadata(
        id="doi:10.1371/journal.ppat.1013519",
        source=PaperSource.unknown,
        title="EntV-derived peptides",
    )

    urls = candidate_pdf_urls(paper)

    assert "https://doi.org/10.1371/journal.ppat.1013519" in urls
    assert any("journals.plos.org/plospathogens/article/file" in u for u in urls)


def test_resolve_papers_from_trajectory_adds_step_source_papers_and_acquire_hints() -> None:
    doc = {
        "papers": [
            {
                "id": "doi:10.1371/journal.ppat.1013519",
                "title": "The antifungal mechanism of EntV-derived peptides",
                "year": 2025,
            }
        ],
        "steps": [
            {
                "step_id": 1,
                "sources": [
                    {"type": "text", "source": "doi:10.1073/pnas.1620432114"},
                    {"type": "text", "source": "doi:10.1371/journal.ppat.1013519"},
                ],
            }
        ],
    }

    resolved = pipeline.resolve_papers_from_trajectory(doc, enable_remote_lookup=False)

    ids = {paper.id for paper in resolved}
    assert "doi:10.1371/journal.ppat.1013519" in ids
    assert "doi:10.1073/pnas.1620432114" in ids

    target = next(paper for paper in resolved if paper.id == "doi:10.1371/journal.ppat.1013519")
    assert "doi:10.1371/journal.ppat.1013519" in (target.raw or {}).get("acquire_hints", [])
