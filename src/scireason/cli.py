from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import settings
from .domain import load_domain_config

from .connectors.arxiv import search as arxiv_search
from .connectors.openalex import search_works as openalex_search
from .connectors.semantic_scholar import search_papers as s2_search

from .connectors.crossref import search_works as crossref_search
from .connectors.pubmed import search as pubmed_search
from .connectors.europe_pmc import search as europepmc_search
from .connectors.biorxiv import (
    details_by_doi as biorxiv_details_by_doi,
    details_by_interval as biorxiv_details_by_interval,
    normalize_record as biorxiv_normalize_record,
)

from .ingest.pipeline import ingest_pdf
from .ingest.mm_pipeline import ingest_pdf_multimodal
from .ingest.one_click import one_click_ingest_arxiv, normalize_arxiv_id

from .graph.build_kg import build_from_paper_dir
from .graph.build_tg_mmkg import build_temporal_and_multimodal
from .graph.graphrag_query import retrieve_context
from .graph.review_applier import compile_overrides
from .graph.temporal_neo4j_store import Neo4jTemporalStore

from .temporal.schemas import TemporalTriplet, TimeInterval

from .agents.debate_graph import run_debate
from .agents.hypothesis_tester import load_hypothesis_from_json, test_hypothesis

from .pipeline.e2e import run_pipeline
from .pipeline.demo import run_demo_pipeline


app = typer.Typer(help="top-papers-graph CLI (ex SciReason)", add_completion=False)
console = Console()

def _user_agent() -> str:
    """Build a polite User-Agent for external APIs.

    Many scholarly APIs (Crossref/OpenAlex/arXiv/NCBI) recommend identifying your client
    and providing a contact email.
    """
    if settings.user_agent:
        return settings.user_agent
    if settings.contact_email:
        return f"top-papers-graph (mailto:{settings.contact_email})"
    return "top-papers-graph"



def _normalize_meta(meta: Dict[str, Any], *, fallback_id: str, source: str = "") -> Dict[str, Any]:
    """Normalize metadata to the minimal fields used across the repo."""
    m = dict(meta)

    # id
    if not m.get("id"):
        # prefer DOI if present, else fallback
        if m.get("doi"):
            m["id"] = f"doi:{m['doi']}"
        else:
            m["id"] = fallback_id

    # title
    m.setdefault("title", "")

    # year
    if not m.get("year"):
        published = str(m.get("published") or "")
        if len(published) >= 4 and published[:4].isdigit():
            m["year"] = int(published[:4])

    # source / url
    if source:
        m.setdefault("source", source)
    m.setdefault("url", m.get("id") if isinstance(m.get("id"), str) and m["id"].startswith("http") else "")

    return m


@app.command()
def doctor() -> None:
    """Проверка окружения/настроек."""
    domain_cfg = load_domain_config()

    t = Table(title="top-papers-graph doctor")
    t.add_column("Key")
    t.add_column("Value")

    t.add_row("Domain", f"{domain_cfg.domain_id} — {domain_cfg.title}")
    t.add_row("LLM provider", settings.llm_provider)
    t.add_row("LLM model", settings.llm_model)
    t.add_row("Embed provider", settings.embed_provider)
    t.add_row("Embed model", settings.embed_model)
    t.add_row("Neo4j", settings.neo4j_uri)
    t.add_row("Qdrant", settings.qdrant_url)
    t.add_row("GROBID", settings.grobid_url)
    t.add_row("CONTACT_EMAIL", str(settings.contact_email or ""))
    t.add_row("USER_AGENT", _user_agent())
    t.add_row("NCBI_EMAIL", str((settings.ncbi_email or settings.contact_email) or ""))
    t.add_row("VLM backend", settings.vlm_backend)
    t.add_row("MM embed backend", settings.mm_embed_backend)

    console.print(t)
    console.print("[green]Если сервисы подняты через docker compose — вы готовы.[/green]")

    # g4f sanity: list working models from g4f/models.py (no network call)
    try:
        import g4f  # type: ignore
        from g4f import models as gm  # type: ignore

        models_list = []
        try:
            Model = getattr(gm, "Model", None)
            if Model is not None and hasattr(Model, "__all__"):
                cand = Model.__all__()  # type: ignore
                if isinstance(cand, (list, tuple)):
                    models_list = list(cand)
        except Exception:
            models_list = []

        if not models_list:
            models_list = list(getattr(gm, "_all_models", []) or [])

        console.print(f"g4f: {getattr(g4f, '__version__', 'unknown')} | models (working): {len(models_list)}")
        if models_list:
            console.print("g4f sample models: " + ", ".join(models_list[:10]))
    except Exception as e:
        console.print(f"g4f: not available ({e})")



@app.command()
def fetch(
    query: str,
    source: str = typer.Option(
        "arxiv",
        help="arxiv|openalex|s2|crossref|pubmed|europepmc|biorxiv|medrxiv",
    ),
    limit: int = typer.Option(10, help="Сколько результатов вернуть (где поддерживается)"),
    out: Path = typer.Option(Path("data/papers/search.json"), help="Куда сохранить JSON"),
    with_abstract: bool = typer.Option(False, help="PubMed: подтянуть абстракт (EFetch)."),
    cursor: int = typer.Option(0, help="biorxiv/medrxiv: cursor для пагинации (по 100 записей)."),
    category: Optional[str] = typer.Option(None, help="biorxiv/medrxiv: фильтр subject category."),
    normalize: bool = typer.Option(False, help="Нормализовать к единому PaperMetadata schema (Pydantic)"),
) -> None:
    """Поиск статей (метаданные) в одном из источников.

    Источники:
    - arxiv: arXiv Atom API (пример query: "all:graph rag" или "cat:cs.AI AND all:retrieval")
    - openalex: OpenAlex works search
    - s2: Semantic Scholar Graph API
    - crossref: Crossref works search (часто полезно для кандидатов DOI)
    - pubmed: NCBI PubMed E-utilities (ESearch + ESummary; опц. EFetch для абстрактов)
    - europepmc: Europe PMC (агрегатор PubMed + PMC + preprints и др.)
    - biorxiv/medrxiv: bioRxiv details API (query = DOI "10.1101/..." или interval "YYYY-MM-DD/YYYY-MM-DD" / "Nd" / "N")
    """
    src = source.lower().strip()
    ua = _user_agent()

    # Unified normalization layer: returns PaperMetadata[] for all sources.
    if normalize:
        from scireason.papers import PaperSource as _PS, search_papers as _search

        srcs = None
        if src not in ("all", "*"):
            parts = [p.strip() for p in src.split(",") if p.strip()]
            srcs = []
            for p in parts:
                # map legacy shorthand
                if p in ("s2", "semanticscholar"):
                    p = "semantic_scholar"
                if p in ("europepmc", "europe_pmc"):
                    p = "europe_pmc"
                try:
                    srcs.append(_PS(p))
                except Exception:
                    continue

        papers = _search(query, limit=limit, sources=srcs, with_abstracts=with_abstract)
        # PaperMetadata contains `published_date: date` → use Pydantic JSON mode
        # so standard `json.dumps(...)` does not fail.
        data = [p.model_dump(mode="json") for p in papers]

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]Saved (normalized):[/green] {out}")
        return

    # Legacy raw mode (source-specific outputs)
    crossref_mailto = settings.crossref_mailto or settings.contact_email
    openalex_mailto = settings.openalex_mailto or settings.contact_email
    ncbi_email = settings.ncbi_email or settings.contact_email

    if src == "arxiv":
        data = arxiv_search(query=query, max_results=limit, user_agent=ua)
    elif src == "openalex":
        data = openalex_search(query=query, per_page=limit, mailto=openalex_mailto, api_key=settings.openalex_api_key, user_agent=ua)
    elif src in ("s2", "semanticscholar", "semantic_scholar"):
        data = s2_search(query=query, limit=limit, api_key=settings.s2_api_key)
    elif src == "crossref":
        data = crossref_search(query=query, rows=limit, mailto=crossref_mailto, user_agent=ua)
    elif src == "pubmed":
        data = pubmed_search(
            query,
            retmax=limit,
            api_key=settings.ncbi_api_key,
            tool=settings.ncbi_tool,
            email=ncbi_email,
            with_abstract=with_abstract,
        )
    elif src in ("europepmc", "europe_pmc"):
        data = europe_pmc_search(query=query, page_size=limit, user_agent=ua)
    elif src in ("biorxiv", "medrxiv"):
        recs = biorxiv_search_details(
            query=query,
            server=src,
            cursor=cursor,
            category=category,
            user_agent=ua,
        )
        data = recs
    else:
        raise typer.BadParameter(
            "source must be one of: arxiv, openalex, s2, crossref, pubmed, europepmc, biorxiv, medrxiv"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Saved:[/green] {out}")
@app.command()
def parse(

    pdf: Path = typer.Option(..., help="Путь к PDF"),
    meta: Path = typer.Option(..., help="Путь к meta.json"),
    out_dir: Path = typer.Option(Path("data/processed/papers"), help="Корневая папка для paper_dir"),
) -> None:
    """Парсинг PDF через GROBID и сохранение чанков."""
    meta_obj = json.loads(meta.read_text(encoding="utf-8"))
    # best-effort normalize
    meta_obj = _normalize_meta(meta_obj, fallback_id=pdf.stem, source=str(meta_obj.get("source") or ""))
    paper_dir = ingest_pdf(pdf_path=pdf, meta=meta_obj, out_dir=out_dir)
    console.print(f"[green]Paper stored:[/green] {paper_dir}")


@app.command("parse-mm")
def parse_mm(
    pdf: Path = typer.Option(..., help="Путь к PDF"),
    meta: Path = typer.Option(..., help="Путь к meta.json"),
    out_dir: Path = typer.Option(Path("data/processed/papers"), help="Корневая папка для paper_dir"),
    vlm: bool = typer.Option(True, help="Включить VLM подписи/таблицы/формулы"),
) -> None:
    """Парсинг PDF + мультимодальность (страницы/картинки)."""
    meta_obj = json.loads(meta.read_text(encoding="utf-8"))
    meta_obj = _normalize_meta(meta_obj, fallback_id=pdf.stem, source=str(meta_obj.get("source") or ""))
    paper_dir = ingest_pdf_multimodal(pdf_path=pdf, meta=meta_obj, out_dir=out_dir, run_vlm=vlm)
    console.print(f"[green]Paper stored (mm):[/green] {paper_dir}")


@app.command("ingest-arxiv")
def ingest_arxiv(
    arxiv_id: str = typer.Argument(..., help="arXiv id или URL (например 2401.01234 или https://arxiv.org/abs/2401.01234)"),
    raw_dir: Path = typer.Option(Path("data/raw/papers"), help="Куда скачать PDF"),
    meta_dir: Path = typer.Option(Path("data/raw/metadata"), help="Куда сохранить metadata JSON"),
    processed_dir: Path = typer.Option(Path("data/processed/papers"), help="Куда сохранить обработанные paper_dir"),
    multimodal: bool = typer.Option(True, help="Использовать мультимодальный ingest (страницы/таблицы/формулы)."),
    build_graph: bool = typer.Option(True, help="После ingest собрать temporal+mm граф (TG-MMKG)."),
    collection_text: Optional[str] = typer.Option(None, help="Qdrant коллекция (текст). По умолчанию из domain config."),
    collection_mm: Optional[str] = typer.Option(None, help="Qdrant коллекция (multimodal). По умолчанию <text>_mm."),
) -> None:
    """Ingestion “в один клик”: скачать arXiv PDF + metadata resolver + ingest pipeline."""
    arxiv_norm = normalize_arxiv_id(arxiv_id)
    pdf_path, meta_path = one_click_ingest_arxiv(arxiv_id=arxiv_norm, raw_dir=raw_dir, meta_dir=meta_dir)

    console.print(f"[green]Downloaded:[/green] {pdf_path}")
    console.print(f"[green]Metadata:[/green] {meta_path}")

    meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_obj = _normalize_meta(
        meta_obj,
        fallback_id=f"arxiv:{arxiv_norm}",
        source="arxiv",
    )
    # make id stable and compact
    meta_obj["id"] = meta_obj.get("doi") and f"doi:{meta_obj['doi']}" or f"arxiv:{arxiv_norm}"

    if multimodal:
        paper_dir = ingest_pdf_multimodal(pdf_path=pdf_path, meta=meta_obj, out_dir=processed_dir, run_vlm=True)
    else:
        paper_dir = ingest_pdf(pdf_path=pdf_path, meta=meta_obj, out_dir=processed_dir)

    console.print(f"[green]Ingested:[/green] {paper_dir}")

    if build_graph:
        domain_cfg = load_domain_config()
        ct = collection_text or (domain_cfg.kg.get("collection") if domain_cfg.kg else None) or "demo"
        cm = collection_mm
        if cm is None:
            cm = f"{ct}_mm"
        build_temporal_and_multimodal(
            paper_dir=paper_dir,
            collection_text=ct,
            collection_mm=cm if multimodal else None,
            domain=domain_cfg.title,
        )
        console.print("[green]TG-MMKG built.[/green]")


@app.command("build-tg-mmkg")
def build_tg_mmkg(
    paper_dir: Path = typer.Option(..., help="Папка paper_dir (meta.json + chunks.jsonl + optional mm/)"),
    collection_text: str = typer.Option("demo", help="Qdrant коллекция (текст)"),
    collection_mm: Optional[str] = typer.Option(None, help="Qdrant коллекция (multimodal). None => skip mm index."),
    domain: str = typer.Option("Science", help="Доменные подсказки для LLM при извлечении утверждений"),
    max_chunks_for_triplets: int = typer.Option(16, help="Сколько чанков использовать для извлечения темпоральных триплетов"),
) -> None:
    """Строит Temporal KG + (опционально) Multimodal индекс для одной статьи."""
    build_temporal_and_multimodal(
        paper_dir=paper_dir,
        collection_text=collection_text,
        collection_mm=collection_mm,
        domain=domain,
        max_chunks_for_triplets=max_chunks_for_triplets,
    )
    console.print("[green]Done.[/green]")


@app.command("build-corpus")
def build_corpus(
    papers_dir: Path = typer.Option(Path("data/processed/papers"), help="Корневая папка с множеством paper_dir"),
    collection_text: Optional[str] = typer.Option(None, help="Qdrant коллекция (текст). Default: domain config."),
    collection_mm: Optional[str] = typer.Option(None, help="Qdrant коллекция (multimodal). Default: <text>_mm."),
    domain: Optional[str] = typer.Option(None, help="Domain hint. Default: domain config title."),
    max_papers: int = typer.Option(0, help="Если >0 — ограничить число обработанных статей"),
    max_chunks_for_triplets: int = typer.Option(16, help="Сколько чанков на статью использовать для извлечения триплетов"),
) -> None:
    """Build TG-MMKG for *all* processed papers in a directory.

    This is the recommended entry point for batch ingestion once you have a folder of `paper_dir`.
    """
    domain_cfg = load_domain_config()
    ct = collection_text or (domain_cfg.kg.get("collection") if domain_cfg.kg else None) or "demo"
    cm = collection_mm
    if cm is None:
        cm = f"{ct}_mm"
    dom = domain or domain_cfg.title

    paper_dirs = sorted([p for p in papers_dir.iterdir() if p.is_dir() and (p / "meta.json").exists()])
    if max_papers and max_papers > 0:
        paper_dirs = paper_dirs[:max_papers]

    if not paper_dirs:
        console.print(f"[yellow]No paper_dir found in {papers_dir}[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Building corpus:[/cyan] papers={len(paper_dirs)} text_collection={ct} mm_collection={cm}")

    for i, pd in enumerate(paper_dirs, start=1):
        console.print(f"[dim]({i}/{len(paper_dirs)})[/dim] {pd}")
        try:
            pages_path = pd / "mm" / "pages.jsonl"
            has_mm = pages_path.exists()
            build_temporal_and_multimodal(
                paper_dir=pd,
                collection_text=ct,
                collection_mm=(cm if has_mm else None),
                domain=dom,
                max_chunks_for_triplets=max_chunks_for_triplets,
            )
        except Exception as e:
            console.print(f"[yellow]Skip {pd.name}: {e}[/yellow]")

    console.print("[green]Corpus build finished.[/green]")


@app.command("build-kg")
def build_kg(
    paper_dir: Path = typer.Option(..., help="Папка paper_dir (meta.json + chunks.jsonl)"),
    collection: str = typer.Option("demo", help="Qdrant коллекция (текст)"),
    domain: str = typer.Option("Science", help="Доменные подсказки для LLM"),
) -> None:
    """Строит обычный KG в Neo4j и эмбеддинги в Qdrant из paper_dir."""
    build_from_paper_dir(paper_dir=paper_dir, collection=collection, domain=domain)
    console.print("[green]Done.[/green]")


@app.command()
def debate(
    query: str,
    collection: str = typer.Option("demo", help="Qdrant коллекция (текст)"),
    domain: str = typer.Option("Science", help="Domain hint for agents"),
    k: int = typer.Option(8, help="Сколько документов достать в контекст"),
    max_rounds: int = typer.Option(3, help="Сколько раундов дебатов"),
    allow_empty_context: bool = typer.Option(
        False,
        help=(
            "Разрешить запуск дебатов без найденного контекста (например, если коллекция ещё не создана). "
            "По умолчанию команда подскажет, как собрать коллекцию, и завершится с ошибкой."
        ),
    ),
) -> None:
    """GraphRAG: достать контекст + дебаты агентов -> гипотеза."""
    try:
        ctx = retrieve_context(collection=collection, query=query, limit=k)
    except Exception as e:
        console.print(
            "[red]Failed to retrieve context from Qdrant.[/red] "
            "Make sure Qdrant is running and the collection is built (parse + build-kg)."
        )
        console.print(f"[dim]{e}[/dim]")
        if not allow_empty_context:
            raise typer.Exit(code=1)
        ctx = []

    if not ctx and not allow_empty_context:
        console.print(
            "[yellow]No context chunks were found.[/yellow] "
            "Run `top-papers-graph parse ...` and `top-papers-graph build-kg ...` first, "
            "or pass --allow-empty-context to proceed without retrieval."
        )
        raise typer.Exit(code=1)
    context_text = "\n\n".join(
        [f"[{c['payload'].get('paper_id')}] {c['payload'].get('text')}" for c in ctx]
    )
    res = run_debate(domain=domain, context=context_text, max_rounds=max_rounds)
    console.print(res.model_dump_json(indent=2, ensure_ascii=False))


@app.command("test-hypothesis")
def test_hypothesis_cmd(
    hypothesis_json: Path = typer.Option(..., help="Путь к JSON (HypothesisDraft или DebateResult из команды debate)"),
    collection: str = typer.Option("demo", help="Qdrant коллекция (текст)"),
    domain: Optional[str] = typer.Option(None, help="Domain hint. Default: domain config title."),
    k: int = typer.Option(12, help="Сколько чанков извлечь для проверки"),
) -> None:
    """Проверить (verification) гипотезу на основе литературы и темпорального KG."""
    domain_cfg = load_domain_config()
    dom = domain or domain_cfg.title

    hyp = load_hypothesis_from_json(hypothesis_json)
    result = test_hypothesis(domain=dom, hypothesis=hyp, collection_text=collection, k=k)
    console.print(result.model_dump_json(indent=2, ensure_ascii=False))


@app.command("refresh-feedback")
def refresh_feedback(
    graph_reviews_dir: Path = typer.Option(Path("data/experts/graph_reviews"), help="Папка с graph_reviews (JSON)."),
    out_path: Path = typer.Option(Path("data/derived/expert_overrides.jsonl"), help="Куда сохранить overrides (JSONL)."),
) -> None:
    """Graph reviews → overrides (для мгновенного эффекта на reward/retriever)."""
    stats = compile_overrides(graph_reviews_dir, out_path)
    console.print(f"[green]Compiled overrides:[/green] {out_path}")
    console.print(
        f"accepted={stats.accepted} rejected={stats.rejected} needs_fix={stats.needs_fix} added={stats.added}"
    )


@app.command("apply-graph-reviews")
def apply_graph_reviews(
    graph_reviews_dir: Path = typer.Option(Path("data/experts/graph_reviews"), help="Папка с graph_reviews (JSON)."),
    overrides_path: Path = typer.Option(Path("data/derived/expert_overrides.jsonl"), help="Куда сохранить overrides (JSONL)."),
    to_neo4j: bool = typer.Option(False, help="Проставить вердикты/веса на Assertion-нодах в Neo4j."),
) -> None:
    """Собрать overrides и (опционально) применить к Neo4j."""
    stats = compile_overrides(graph_reviews_dir, overrides_path)
    console.print(f"[green]Compiled overrides:[/green] {overrides_path}")
    console.print(
        f"accepted={stats.accepted} rejected={stats.rejected} needs_fix={stats.needs_fix} added={stats.added}"
    )

    if not to_neo4j:
        return

    try:
        tneo = Neo4jTemporalStore()
        tneo.ensure_schema()

        count = 0
        for line in overrides_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            tneo.apply_expert_override(
                subj=str(rec.get("subject")),
                pred=str(rec.get("predicate")),
                obj=str(rec.get("object")),
                verdict=str(rec.get("verdict")),
                weight=float(rec.get("weight", 0.0)),
                time_interval=str(rec.get("time_interval", "unknown")),
            )
            count += 1
        tneo.close()
        console.print(f"[green]Applied to Neo4j:[/green] {count} overrides")
    except Exception as e:
        console.print(f"[red]Neo4j apply failed:[/red] {e}")


@app.command("apply-temporal-corrections")
def apply_temporal_corrections(
    corrections_dir: Path = typer.Option(
        Path("data/experts/temporal_corrections"), help="Папка с temporal_corrections (JSON)."
    ),
    dry_run: bool = typer.Option(False, help="Не писать в Neo4j, только показать план изменений."),
) -> None:
    """Применить temporal_corrections к Neo4j Temporal KG.

    Т.к. assertion_id включает время, исправление времени реализовано как:
    old_assertion -[:REPLACED_BY]-> new_assertion.
    """

    paths = sorted(corrections_dir.glob("**/*.json"))
    if not paths:
        console.print(f"[yellow]No JSON files found in {corrections_dir}[/yellow]")
        return

    tneo = Neo4jTemporalStore()
    tneo.ensure_schema()

    total = 0
    created = 0
    replaced = 0

    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        reviewer_id = str(doc.get("reviewer_id", ""))
        for corr in doc.get("corrections", []):
            total += 1
            old_id = str(corr.get("assertion_id", "")).strip()
            if not old_id:
                continue

            details = tneo.get_assertion_details(old_id)
            if not details:
                console.print(f"[yellow]Skip:[/yellow] cannot find assertion {old_id}")
                continue

            try:
                corrected_time = TimeInterval.model_validate(corr.get("corrected_time"))
            except Exception as e:
                console.print(f"[yellow]Skip:[/yellow] invalid corrected_time for {old_id} ({e})")
                continue

            t = TemporalTriplet(
                subject=str(details.get("subject")),
                predicate=str(details.get("predicate")),
                object=str(details.get("object")),
                confidence=float(details.get("confidence") or 0.5),
                polarity=str(details.get("polarity") or "unknown"),
                time=corrected_time,
                evidence_quote=str(corr.get("evidence_quote") or details.get("evidence_quote") or "").strip() or None,
            )

            paper_id = str(details.get("paper_id"))
            if dry_run:
                console.print(
                    f"[cyan]DRY RUN[/cyan] {old_id} -> time {corrected_time.start}-{corrected_time.end} ({corrected_time.granularity})"
                )
                continue

            new_id = tneo.upsert_assertion(paper_id=paper_id, t=t, evidence_quote=t.evidence_quote)
            created += 1
            tneo.link_replacement(
                old_id,
                new_id,
                rationale=str(corr.get("rationale", "")).strip(),
                reviewer_id=reviewer_id,
            )
            replaced += 1

    tneo.close()
    console.print(
        f"[green]Temporal corrections processed[/green]: total={total}, new_assertions={created}, replaced_links={replaced}"
    )


@app.command("pybamm-fastcharge")
def pybamm_fastcharge(
    profile: str = typer.Option("baseline_cc", help="baseline_cc|proposed_two_stage|..."),
    profiles_dir: Path | None = typer.Option(
        None,
        "--profiles-dir",
        envvar="CHARGING_PROFILES_DIR",
        help="Папка с YAML профилями зарядки (можно задать через CHARGING_PROFILES_DIR)",
    ),
    out_dir: Path = typer.Option(Path("results/pybamm/run"), help="Куда сохранить результаты"),
) -> None:
    """Запуск симуляции PyBaMM для профиля зарядки (пример: battery_fastcharge)."""
    from .experiments.pybamm_fastcharge import run_simulation

    out = run_simulation(profile_name=profile, out_dir=out_dir, config_dir=profiles_dir)
    console.print(f"[green]Saved metrics:[/green] {out}")


@app.command("import-top-papers")
def import_top_papers(
    inp: Path = typer.Option(..., help="JSON файл, который выдаёт top-papers-bot"),
    out_dir: Path = typer.Option(Path("configs/top_papers_meta"), help="Куда сохранить meta-файлы"),
) -> None:
    """Импорт JSON из top-papers-bot в meta-файлы SciReason."""
    from .integrations.top_papers_import import export_meta_files

    files = export_meta_files(inp, out_dir)
    console.print(f"[green]Generated meta files:[/green] {len(files)} → {out_dir}")


@app.command("run")
def run_cmd(
    query: str = typer.Option(..., help="Пользовательский запрос (topic/query)."),
    domain_id: str = typer.Option(
        None,  # type: ignore[arg-type]
        help="ID домена (configs/domains/<id>.yaml). По умолчанию берётся из .env (DOMAIN_ID) или science.",
    ),
    sources: str = typer.Option(
        "all",
        help="Источники через запятую: all|openalex,semantic_scholar,crossref,arxiv,pubmed,europe_pmc,biorxiv.",
    ),
    search_limit: int = typer.Option(50, help="Сколько результатов запросить у источников."),
    top_papers: int = typer.Option(20, help="Сколько лучших статей взять в пайплайн."),
    out_dir: Path = typer.Option(Path("runs"), help="Куда сохранить артефакты запуска."),
    multimodal: bool = typer.Option(False, help="Извлекать страницы/картинки (MM) при наличии зависимостей."),
    no_llm_hypotheses: bool = typer.Option(False, help="Не использовать LLM для переформулировки гипотез."),

    # --- LLM overrides (CLI > env/config defaults) ---
    llm: Optional[str] = typer.Option(
        None,
        "--llm",
        help=(
            "Переопределить LLM одним флагом. Форматы: 'g4f:deepseek-r1', 'g4f:gpt-4o-mini', "
            "'local:llama3.2' (Ollama), 'ollama:llama3.2', или 'openai/gpt-4o-mini' (LiteLLM)."
        ),
    ),
    g4f_model: Optional[str] = typer.Option(
        None,
        "--g4f-model",
        help="Явно использовать g4f с указанной моделью (например deepseek-r1).",
    ),
    local_model: Optional[str] = typer.Option(
        None,
        "--local-model",
        help="Явно использовать локальную Ollama модель (например llama3.2).",
    ),
    llm_provider: Optional[str] = typer.Option(
        None,
        "--llm-provider",
        help="Явно задать провайдера (g4f|ollama|openai|anthropic|...).",
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        "--llm-model",
        help="Явно задать имя модели провайдера.",
    ),
    smol_model_backend: Optional[str] = typer.Option(
        None,
        "--smol-model-backend",
        help="smolagents model backend (scireason|transformers|g4f). Overrides SMOL_MODEL_BACKEND.",
    ),
    smol_model_id: Optional[str] = typer.Option(
        None,
        "--smol-model-id",
        help="HF model id/path for smolagents TransformersModel. Overrides SMOL_MODEL_ID.",
    ),
) -> None:
    """Полностью автоматический пайплайн: query → papers → temporal KG → hypotheses."""
    # ---- Apply LLM overrides ----
    def _apply_llm_overrides() -> None:
        # 1) single-flag format
        if llm:
            raw = llm.strip()

            # Accept provider/model as "provider:model" or "provider/model"
            if ":" in raw:
                prov, model = raw.split(":", 1)
            elif "/" in raw:
                prov, model = raw.split("/", 1)
            else:
                # No separator -> assume g4f model
                prov, model = "g4f", raw

            prov = prov.strip().lower()
            model = model.strip()

            if prov in {"local", "ollama"}:
                settings.llm_provider = "ollama"
                settings.llm_model = model
            elif prov == "g4f":
                settings.llm_provider = "g4f"
                settings.llm_model = model
            else:
                # LiteLLM-style provider/model
                settings.llm_provider = prov
                settings.llm_model = model
            return

        # 2) convenience flags
        if local_model:
            settings.llm_provider = "ollama"
            settings.llm_model = local_model.strip()
            return

        if g4f_model:
            settings.llm_provider = "g4f"
            settings.llm_model = g4f_model.strip()
            return

        # 3) explicit provider/model flags
        if llm_provider:
            settings.llm_provider = llm_provider.strip()
        if llm_model:
            settings.llm_model = llm_model.strip()

<<<<<<< HEAD
    # Apply overrides (CLI > env/config defaults)
    _apply_llm_overrides()
=======

>>>>>>> 4b764f1ef18eb8f9b4f2392395b6de0d99c57e12

    # smolagents model overrides (CLI > env)
    if smol_model_backend:
        settings.smol_model_backend = smol_model_backend.strip()
    if smol_model_id:
        settings.smol_model_id = smol_model_id.strip()

    console.print(
        f"[bold]LLM:[/bold] {settings.llm_provider}/{settings.llm_model}  |  "
        f"[bold]Embeddings:[/bold] {getattr(settings, 'embed_provider', 'hash')}"
    )

    did = domain_id or settings.domain_id or "science"
    src_list = None
    if sources.strip().lower() != "all":
        src_list = [s.strip() for s in sources.split(",") if s.strip()]

    run_path = run_pipeline(
        query=query,
        domain_id=did,
        sources=src_list,
        search_limit=search_limit,
        top_papers=top_papers,
        run_dir=out_dir,
        include_multimodal=multimodal,
        use_llm_for_hypotheses=not no_llm_hypotheses,
    )
    console.print(f"[bold green]Artifacts saved:[/bold green] {run_path}")


@app.command("demo-run")
def demo_run_cmd(
    query: str = typer.Option("temporal knowledge graph hypothesis", help="Demo query (offline)."),
    edge_mode: str = typer.Option("cooccurrence", help="cooccurrence|llm_triplets"),
    out_dir: Path = typer.Option(Path("runs"), help="Where to write demo artifacts."),
    domain_id: str = typer.Option(None, help="Domain config id (defaults to env DOMAIN_ID or science)."),
    no_llm_hypotheses: bool = typer.Option(False, help="Disable LLM rewriting for hypotheses."),
    gnn: bool = typer.Option(False, help="Enable optional GNN link prediction (requires '.[gnn]')."),
    agent_backend: Optional[str] = typer.Option(
        None,
        help="Override HYP_AGENT_BACKEND for this run (internal|smolagents).",
    ),
    llm_provider: Optional[str] = typer.Option(None, help="Override LLM_PROVIDER for this run (e.g. mock)."),
    llm_model: Optional[str] = typer.Option(None, help="Override LLM_MODEL for this run."),
smol_model_backend: Optional[str] = typer.Option(
    None,
    "--smol-model-backend",
    help="smolagents model backend (scireason|transformers|g4f). Overrides SMOL_MODEL_BACKEND.",
),
smol_model_id: Optional[str] = typer.Option(
    None,
    "--smol-model-id",
    help="HF model id/path for smolagents TransformersModel. Overrides SMOL_MODEL_ID.",
),
) -> None:
    """Offline demo pipeline: build temporal KG + hypotheses from a tiny built-in corpus.

    This command is used for smoke tests and for the first classroom run without network/services.
    """

    if llm_provider:
        settings.llm_provider = llm_provider.strip()
    if llm_model:
        settings.llm_model = llm_model.strip()

    if gnn:
        settings.hyp_gnn_enabled = True

    if agent_backend:
        settings.hyp_agent_backend = agent_backend.strip()

    did = domain_id or settings.domain_id or "science"
    run_path = run_demo_pipeline(
        query=query,
        domain_id=did,
        edge_mode=edge_mode,
        out_dir=out_dir,
        use_llm_for_hypotheses=not no_llm_hypotheses,
    )
    console.print(f"[bold green]Demo artifacts saved:[/bold green] {run_path}")


@app.command("smoke-all")
def smoke_all(
    out_dir: Path = typer.Option(Path("runs"), help="Where to write artifacts."),
    include_g4f: bool = typer.Option(
        False,
        help="Also run smoke with LLM_PROVIDER=g4f (requires '.[g4f]' and internet; can be unstable).",
    ),
    smol_model_backend: Optional[str] = typer.Option(
        None,
        "--smol-model-backend",
        help="smolagents model backend for smolagents runs (scireason|transformers|g4f).",
    ),
    smol_model_id: Optional[str] = typer.Option(
        None,
        "--smol-model-id",
        help="HF model id/path for smolagents TransformersModel.",
    ),
) -> None:
    """Run an offline smoke matrix for key pipeline branches."""

    # Prefer deterministic offline mode by default.
    llm_providers = ["mock"]
    if include_g4f:
        llm_providers.append("g4f")

    # Try both agent backends if smolagents is available.
    import importlib.util

    agent_backends = ["internal"]
    if importlib.util.find_spec("smolagents") is not None:
        agent_backends.append("smolagents")

    # smolagents model overrides (CLI > env)
    if smol_model_backend:
        settings.smol_model_backend = smol_model_backend.strip()
    if smol_model_id:
        settings.smol_model_id = smol_model_id.strip()

    combos = [
        ("cooccurrence", True, False),
        ("cooccurrence", False, False),
        ("llm_triplets", True, False),
        ("llm_triplets", False, False),
        # Optional GNN branch (best-effort; will fall back if PyG isn't installed)
        ("cooccurrence", True, True),
        ("cooccurrence", False, True),
    ]
    for llm_provider in llm_providers:
        settings.llm_provider = llm_provider
        settings.llm_model = "mock" if llm_provider == "mock" else (settings.llm_model or "auto")

        for agent_backend in agent_backends:
            settings.hyp_agent_backend = agent_backend
            for edge_mode, no_llm, gnn in combos:
                settings.hyp_gnn_enabled = bool(gnn)
                console.print(
                    f"[cyan]Smoke[/cyan] llm_provider={llm_provider} agent_backend={agent_backend} edge_mode={edge_mode} no_llm_hypotheses={no_llm} gnn={gnn}"
                )
                rp = run_demo_pipeline(
                    query="demo smoke",
                    domain_id=settings.domain_id or "science",
                    edge_mode=edge_mode,
                    out_dir=out_dir,
                    use_llm_for_hypotheses=not no_llm,
                )
                # Ensure key artifacts exist
                for f in ["paper_records.json", "temporal_kg.json", "hypotheses.json"]:
                    p = rp / f
                    if not p.exists():
                        raise RuntimeError(f"Smoke failed: missing {p}")
    console.print("[bold green]Smoke-all: OK[/bold green]")


if __name__ == "__main__":
    app()
