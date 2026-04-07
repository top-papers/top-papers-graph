from __future__ import annotations

import asyncio
import json
from pathlib import Path

from scireason.domain import load_domain_config
from scireason.pipeline.task2_validation import _flatten_automatic_graph
from scireason.temporal.schemas import TemporalTriplet, TimeInterval
from scireason.temporal.temporal_kg_builder import (
    EdgeStats,
    PaperEvidenceUnit,
    PaperRecord,
    TemporalKnowledgeGraph,
    build_temporal_kg,
    load_papers_from_processed,
)


def test_build_temporal_kg_passes_explicit_llm_override_to_triplet_extractor(monkeypatch) -> None:
    captured: list[tuple[str | None, str | None]] = []

    def _fake_extract(*, domain, chunk_text, paper_year, llm_provider=None, llm_model=None, **kwargs):
        captured.append((llm_provider, llm_model))
        return [
            TemporalTriplet(
                subject="graph neural networks",
                predicate="improves",
                object="link prediction",
                confidence=0.9,
                polarity="supports",
                evidence_quote="Graph neural networks improve link prediction.",
                time=TimeInterval(start="2024-05", end="2024-05", granularity="month"),
            )
        ]

    monkeypatch.setattr("scireason.temporal.temporal_kg_builder.extract_temporal_triplets", _fake_extract)
    monkeypatch.setattr("scireason.temporal.temporal_kg_builder.settings.g4f_async_enabled", False)

    kg = build_temporal_kg(
        [PaperRecord(paper_id="p1", title="Demo", year=2024, text="Graph neural networks improve link prediction.")],
        domain=load_domain_config("science"),
        query="demo",
        edge_mode="auto",
        llm_provider="g4f",
        llm_model="deepseek-r1",
    )

    assert captured == [("g4f", "deepseek-r1")]
    assert kg.edges
    assert kg.edges[0].time_intervals[0]["start"] == "2024-05"



def test_auto_mode_fallback_is_localized_per_evidence_unit(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_extract(*, domain, chunk_text, paper_year, **kwargs):
        calls.append(chunk_text)
        if "first unit" in chunk_text:
            raise RuntimeError("synthetic llm failure")
        return [
            TemporalTriplet(
                subject="temporal graph model",
                predicate="improves",
                object="forecast accuracy",
                confidence=0.88,
                polarity="supports",
                evidence_quote="second unit says the temporal graph model improves forecast accuracy in 2024-06-15.",
                time=TimeInterval(start="2024-06-15", end="2024-06-15", granularity="day"),
            )
        ]

    monkeypatch.setattr("scireason.temporal.temporal_kg_builder.extract_temporal_triplets", _fake_extract)

    paper = PaperRecord(
        paper_id="p1",
        title="Demo",
        year=2024,
        text="fallback demo",
        evidence_units=[
            PaperEvidenceUnit(unit_id="u1", text="first unit broken path", source_kind="text_chunk"),
            PaperEvidenceUnit(unit_id="u2", text="second unit working path", source_kind="text_chunk"),
        ],
    )
    kg = build_temporal_kg([paper], domain=load_domain_config("science"), query="demo", edge_mode="auto")

    assert len(calls) == 2
    assert kg.meta["edge_mode"] == "llm_triplets"
    assert kg.meta["localized_fallbacks"] >= 1
    assert kg.meta["llm_failures"] == 1
    assert any(edge.predicate == "improves" for edge in kg.edges)



def test_load_papers_from_processed_includes_multimodal_units_and_kg_uses_them(tmp_path: Path, monkeypatch) -> None:
    paper_dir = tmp_path / "paper-1"
    (paper_dir / "mm").mkdir(parents=True)
    (paper_dir / "meta.json").write_text(json.dumps({"id": "paper-1", "title": "Demo", "year": 2024}), encoding="utf-8")
    (paper_dir / "chunks.jsonl").write_text(json.dumps({"chunk_id": "c1", "text": "Plain text without the target relation."}) + "\n", encoding="utf-8")
    (paper_dir / "mm" / "pages.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "paper-1",
                "page": 3,
                "text": "Figure text.",
                "image_path": "/tmp/page_003.png",
                "vlm_caption": "multimodal clue: catalyst A reduces latency in 2024-05-11",
                "tables_md": "",
                "equations_md": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def _fake_extract(*, chunk_text, **kwargs):
        if "multimodal clue" not in chunk_text:
            return []
        return [
            TemporalTriplet(
                subject="catalyst A",
                predicate="reduces",
                object="latency",
                confidence=0.91,
                polarity="supports",
                evidence_quote="multimodal clue: catalyst A reduces latency in 2024-05-11",
                time=TimeInterval(start="2024-05-11", end="2024-05-11", granularity="day"),
            )
        ]

    monkeypatch.setattr("scireason.temporal.temporal_kg_builder.extract_temporal_triplets", _fake_extract)

    papers = load_papers_from_processed(tmp_path)
    assert len(papers) == 1
    assert any(unit.source_kind == "multimodal_page" for unit in papers[0].evidence_units)
    assert "multimodal clue" in papers[0].multimodal_text

    kg = build_temporal_kg(papers, domain=load_domain_config("science"), query="demo", edge_mode="auto")

    assert any(edge.source == "catalyst a" and edge.target == "latency" for edge in kg.edges)
    edge = next(edge for edge in kg.edges if edge.source == "catalyst a" and edge.target == "latency")
    assert edge.evidence_quotes[0]["source_kind"] == "multimodal_page"
    assert edge.time_intervals[0]["start"] == "2024-05-11"



def test_flatten_automatic_graph_preserves_precise_time_intervals() -> None:
    kg = TemporalKnowledgeGraph(
        edges=[
            EdgeStats(
                source="catalyst a",
                target="latency",
                predicate="reduces",
                papers={"paper-1"},
                evidence_quotes=[
                    {
                        "paper_id": "paper-1",
                        "quote": "catalyst A reduces latency on 2024-05-11",
                        "page": 3,
                        "source_kind": "multimodal_page",
                    }
                ],
                time_intervals=[
                    {
                        "start": "2024-05-11",
                        "end": "2024-05-11",
                        "granularity": "day",
                        "source": "extracted",
                        "paper_id": "paper-1",
                        "unit_id": "mm-page-00003",
                    }
                ],
                yearly_count={2024: 1},
            )
        ]
    )

    rows = _flatten_automatic_graph(kg)

    assert rows[0]["start_date"] == "2024-05-11"
    assert rows[0]["end_date"] == "2024-05-11"
    assert rows[0]["time_source"] == "triplet_extractor"
    assert rows[0]["time_candidates"][0]["granularity"] == "day"
    assert rows[0]["evidence"]["page"] == 3
    assert rows[0]["evidence"]["source_kind"] == "multimodal_page"


def test_time_interval_accepts_interval_and_range_aliases() -> None:
    direct = TimeInterval(start="2020", end="2022", granularity="interval")
    alias = TimeInterval.model_validate({"start": "2020", "end": "2022", "granularity": "range"})

    assert direct.granularity == "interval"
    assert alias.granularity == "interval"


def test_extract_temporal_triplets_normalizes_interval_granularity(monkeypatch) -> None:
    from scireason.temporal import temporal_triplet_extractor as extractor

    monkeypatch.setattr(extractor.settings, "llm_provider", "g4f")
    monkeypatch.setattr(extractor, "temporary_llm_selection", lambda **kwargs: __import__("contextlib").nullcontext())
    monkeypatch.setattr(
        extractor,
        "chat_json",
        lambda **kwargs: [
            {
                "subject": "model",
                "predicate": "improves",
                "object": "accuracy",
                "confidence": 0.9,
                "polarity": "supports",
                "time": {"start": "2020", "end": "2022", "granularity": "range"},
            }
        ],
    )

    rows = extractor.extract_temporal_triplets(domain="science", chunk_text="demo text", paper_year=2022)

    assert rows[0].time is not None
    assert rows[0].time.granularity == "interval"
    assert rows[0].time.start == "2020"
    assert rows[0].time.end == "2022"



def test_extract_temporal_triplets_coerces_non_string_triplet_fields_and_skips_empty_rows(monkeypatch) -> None:
    from scireason.temporal import temporal_triplet_extractor as extractor

    monkeypatch.setattr(extractor.settings, "llm_provider", "g4f")
    monkeypatch.setattr(extractor, "temporary_llm_selection", lambda **kwargs: __import__("contextlib").nullcontext())
    monkeypatch.setattr(
        extractor,
        "chat_json",
        lambda **kwargs: [
            {
                "subject": {"name": "temporal graph network"},
                "predicate": b"improves",
                "object": ["forecast accuracy", "latency stability"],
                "confidence": 0.9,
                "polarity": "supports",
                "evidence_quote": ["improves", "forecast accuracy"],
                "time": {"start": 2020, "end": 2022, "granularity": "range"},
            },
            {
                "subject": None,
                "predicate": None,
                "object": None,
                "confidence": 0.4,
                "polarity": "unknown",
            },
            "unexpected raw row",
        ],
    )

    rows = extractor.extract_temporal_triplets(domain="science", chunk_text="demo text", paper_year=2022)

    assert len(rows) == 1
    assert rows[0].subject == "temporal graph network"
    assert rows[0].predicate == "improves"
    assert rows[0].object == "forecast accuracy; latency stability"
    assert rows[0].evidence_quote == "improves; forecast accuracy"
    assert rows[0].time is not None
    assert rows[0].time.start == "2020"
    assert rows[0].time.end == "2022"
    assert rows[0].time.granularity == "interval"


def test_build_temporal_kg_keeps_trying_async_g4f_before_fallback(monkeypatch) -> None:
    calls: list[str] = []

    async def _fake_extract_async(*, chunk_text, **kwargs):
        calls.append(chunk_text)
        raise TimeoutError("synthetic timeout")

    monkeypatch.setattr("scireason.temporal.temporal_kg_builder.extract_temporal_triplets_async", _fake_extract_async)
    monkeypatch.setattr("scireason.temporal.temporal_kg_builder.settings.g4f_async_enabled", True)
    monkeypatch.setattr("scireason.temporal.temporal_kg_builder.settings.g4f_async_max_concurrency", 2)

    paper = PaperRecord(
        paper_id="p-timeout",
        title="Demo",
        year=2024,
        text="timeout demo",
        evidence_units=[
            PaperEvidenceUnit(unit_id="u1", text="graph signal improves forecast accuracy in production", source_kind="text_chunk"),
            PaperEvidenceUnit(unit_id="u2", text="temporal model improves latency stability for deployment", source_kind="text_chunk"),
        ],
    )

    kg = build_temporal_kg([paper], domain=load_domain_config("science"), query="demo", edge_mode="auto", llm_provider="g4f")

    assert calls == [
        "graph signal improves forecast accuracy in production",
        "temporal model improves latency stability for deployment",
    ]
    assert kg.meta["g4f_async_batch"] is True
    assert kg.meta["llm_disabled_after_timeout"] is False
    assert kg.meta["llm_failures"] == 2
    assert kg.meta["localized_fallbacks"] >= 2
    assert kg.meta["heuristic_fallbacks"] >= 2
    assert any(edge.predicate == "improves" for edge in kg.edges)
    assert not any(edge.predicate == "cooccurs_with" for edge in kg.edges)


def test_build_temporal_kg_async_g4f_respects_semaphore_limit(monkeypatch) -> None:
    active = 0
    max_active = 0

    async def _fake_extract_async(*, chunk_text, semaphore=None, **kwargs):
        nonlocal active, max_active
        assert semaphore is not None
        async with semaphore:
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
        return [
            TemporalTriplet(
                subject=chunk_text.split()[0],
                predicate="improves",
                object="accuracy",
                confidence=0.9,
                polarity="supports",
                evidence_quote=chunk_text,
                time=TimeInterval(start="2024", end="2024", granularity="year"),
            )
        ]

    monkeypatch.setattr("scireason.temporal.temporal_kg_builder.extract_temporal_triplets_async", _fake_extract_async)
    monkeypatch.setattr("scireason.temporal.temporal_kg_builder.settings.g4f_async_enabled", True)
    monkeypatch.setattr("scireason.temporal.temporal_kg_builder.settings.g4f_async_max_concurrency", 2)

    paper = PaperRecord(
        paper_id="p-async",
        title="Demo Async",
        year=2024,
        text="demo",
        evidence_units=[
            PaperEvidenceUnit(unit_id="u1", text="alpha improves forecast accuracy", source_kind="text_chunk"),
            PaperEvidenceUnit(unit_id="u2", text="beta improves forecast accuracy", source_kind="text_chunk"),
            PaperEvidenceUnit(unit_id="u3", text="gamma improves forecast accuracy", source_kind="text_chunk"),
        ],
    )

    kg = build_temporal_kg([paper], domain=load_domain_config("science"), query="demo", edge_mode="auto", llm_provider="g4f")

    assert kg.meta["g4f_async_batch"] is True
    assert kg.meta["g4f_async_max_concurrency"] == 2
    assert max_active <= 2
    assert any(edge.predicate == "improves" for edge in kg.edges)


def test_json_best_effort_repairs_invalid_escape_sequences() -> None:
    from scireason.llm import _json_loads_best_effort

    payload = '[{"subject":"dust crystal","predicate":"leads_to","object":"melting",' \
              '"evidence_quote":"The derivation uses \\alpha and \\mathrm{O} in the text.","confidence":0.8}]'
    parsed = _json_loads_best_effort(payload)

    assert isinstance(parsed, list)
    assert parsed[0]["evidence_quote"] == r"The derivation uses \alpha and \mathrm{O} in the text."


def test_extract_temporal_triplets_prefers_directional_local_fallback() -> None:
    from scireason.temporal.temporal_triplet_extractor import extract_temporal_triplets_localized_fallback

    rows = extract_temporal_triplets_localized_fallback(
        "Increasing plasma power leads to lattice melting after defect proliferation in 2021.",
        paper_year=2021,
    )

    assert rows
    assert rows[0].predicate in {"leads_to", "causes", "results_in"}
    assert rows[0].time is not None
    assert rows[0].time.start == "2021"
