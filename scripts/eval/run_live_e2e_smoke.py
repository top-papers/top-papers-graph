#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from scireason.config import settings
from scireason.ingest.mm_pipeline import ingest_pdf_multimodal_auto
from scireason.graph.build_tg_mmkg import build_temporal_and_multimodal
from scireason.temporal.temporal_kg_builder import load_papers_from_processed, build_temporal_kg
from scireason.domain import load_domain_config
from scireason.hypotheses.temporal_graph_hypotheses import generate_hypotheses
from scireason.graph.qdrant_store import QdrantStore
from scireason.graph.memgraph_store import MemgraphTemporalStore


def make_figure(path: Path, title: str, labels: List[str]) -> None:
    img = Image.new('RGB', (960, 540), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 920, 500), outline='black', width=3)
    draw.text((60, 60), title, fill='black')
    x0 = 120
    for idx, label in enumerate(labels):
        x = x0 + idx * 220
        h = 140 + idx * 60
        draw.rectangle((x, 420 - h, x + 80, 420), outline='black', width=2)
        draw.text((x, 430), label, fill='black')
    img.save(path)


def make_pdf(pdf_path: Path, title: str, year: int, sentences: List[str], table_rows: List[List[str]], fig_title: str) -> None:
    styles = getSampleStyleSheet()
    fig_path = pdf_path.with_suffix('.png')
    make_figure(fig_path, fig_title, [row[0] for row in table_rows[1:4]])
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    story = [Paragraph(title, styles['Title']), Spacer(1, 12)]
    for sent in sentences:
        story.append(Paragraph(sent, styles['BodyText']))
        story.append(Spacer(1, 8))
    tbl = Table(table_rows)
    tbl.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ]))
    story.append(Spacer(1, 12))
    story.append(tbl)
    story.append(Spacer(1, 12))
    story.append(RLImage(str(fig_path), width=420, height=236))
    doc.build(story)


def query_qdrant_count(collection: str) -> int:
    store = QdrantStore(url=settings.qdrant_url)
    try:
        info = store._client.get_collection(collection)
    except Exception:
        return 0
    return int(getattr(info, 'vectors_count', None) or getattr(info, 'points_count', 0) or 0)


def query_memgraph_counts() -> Dict[str, int]:
    mg = MemgraphTemporalStore()
    try:
        counts: Dict[str, int] = {}
        with mg._driver.session() as s:
            queries = {
                'papers': 'MATCH (n:Paper) RETURN count(n) AS c',
                'chunks': 'MATCH (n:Chunk) RETURN count(n) AS c',
                'assertions': 'MATCH (n:Assertion) RETURN count(n) AS c',
                'events': 'MATCH (n:Event) RETURN count(n) AS c',
            }
            for key, q in queries.items():
                rec = s.run(q).single()
                counts[key] = int(rec['c']) if rec else 0
        return counts
    finally:
        mg.close()


def main() -> None:
    out_root = Path(os.environ.get('E2E_SMOKE_OUT', 'runs/live_service_smoke')).resolve()
    raw_dir = out_root / 'raw_pdfs'
    processed_dir = out_root / 'processed_papers'
    out_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    settings.ocr_backend = 'pymupdf'
    settings.graph_backend = 'memgraph'
    settings.vlm_backend = 'none'
    settings.mm_embed_backend = 'hash'
    settings.hyp_agent_enabled = False
    settings.llm_provider = 'mock'  # extractor will use rule-based fallback instead of mock triplets

    papers = [
        {
            'id': 'demo:graphene-fastcharge-2022',
            'title': 'Graphene coating improves fast-charging stability in lithium cells',
            'year': 2022,
            'country_id': 'Q183',
            'country_label': 'Germany',
            'city_id': 'Q64',
            'city_label': 'Berlin',
            'science_branch_ids': ['Q11372'],
            'science_branch_labels': ['physical chemistry'],
            'sentences': [
                'Graphene coating improves fast charging stability in lithium cells in 2022.',
                'Graphene coating reduces lithium plating under 3C charging.',
                'Pulse charging is associated with lower impedance growth.',
            ],
            'table': [['Factor', 'Effect'], ['Graphene coating', 'improves stability'], ['Fast charging', 'increases stress'], ['Pulse charging', 'reduces plating']],
            'fig_title': 'Fast charging factors 2022',
        },
        {
            'id': 'demo:pulse-protocol-2023',
            'title': 'Pulse charging increases cycle life during high-rate charging',
            'year': 2023,
            'country_id': 'Q30',
            'country_label': 'United States of America',
            'city_id': 'Q60',
            'city_label': 'New York City',
            'science_branch_ids': ['Q11410'],
            'science_branch_labels': ['engineering'],
            'sentences': [
                'Pulse charging increases cycle life during high-rate charging in 2023.',
                'Pulse charging improves thermal stability in graphite anodes.',
                'High-rate charging causes lithium plating when cooling is insufficient.',
            ],
            'table': [['Factor', 'Effect'], ['Pulse charging', 'increases cycle life'], ['High-rate charging', 'causes plating'], ['Cooling', 'improves stability']],
            'fig_title': 'Pulse charging factors 2023',
        },
        {
            'id': 'demo:cooling-monitoring-2024',
            'title': 'Cooling control reduces plating and improves diagnostic accuracy',
            'year': 2024,
            'country_id': 'Q145',
            'country_label': 'United Kingdom',
            'city_id': 'Q84',
            'city_label': 'London',
            'science_branch_ids': ['Q11190'],
            'science_branch_labels': ['medicine'],
            'sentences': [
                'Cooling control reduces plating during extreme fast charging in 2024.',
                'Impedance monitoring improves diagnostic accuracy for thermal stability.',
                'Graphene coating is associated with cooling control in integrated protocols.',
            ],
            'table': [['Factor', 'Effect'], ['Cooling control', 'reduces plating'], ['Impedance monitoring', 'improves diagnostics'], ['Integrated protocols', 'enhance safety']],
            'fig_title': 'Cooling and diagnostics 2024',
        },
    ]

    for paper in papers:
        pdf_path = raw_dir / f"{paper['id'].split(':')[-1]}.pdf"
        make_pdf(pdf_path, paper['title'], paper['year'], paper['sentences'], paper['table'], paper['fig_title'])
        meta = {
            'id': paper['id'],
            'title': paper['title'],
            'year': paper['year'],
            'source': 'live_smoke',
            'url': '',
            'country_id': paper['country_id'],
            'country_label': paper['country_label'],
            'city_id': paper['city_id'],
            'city_label': paper['city_label'],
            'science_branch_ids': paper['science_branch_ids'],
            'science_branch_labels': paper['science_branch_labels'],
        }
        paper_dir = ingest_pdf_multimodal_auto(pdf_path=pdf_path, meta=meta, out_dir=processed_dir, run_vlm=False)
        build_temporal_and_multimodal(
            paper_dir=paper_dir,
            collection_text='live_smoke_text',
            collection_mm='live_smoke_mm',
            domain='Science',
            max_chunks_for_triplets=12,
        )

    paper_records = load_papers_from_processed(processed_dir)
    kg = build_temporal_kg(paper_records, domain=load_domain_config('science'), query='fast charging stability', edge_mode='auto')
    kg_path = out_root / 'temporal_kg.json'
    kg.dump_json(kg_path)
    hyps = generate_hypotheses(kg, papers=paper_records, domain='Science', query='fast charging stability', top_k=6, use_llm=False)
    hyps_path = out_root / 'hypotheses.json'
    hyps_path.write_text(json.dumps([h.model_dump(mode='json') for h in hyps], ensure_ascii=False, indent=2), encoding='utf-8')

    report = {
        'processed_papers': len(paper_records),
        'temporal_kg_nodes': len(kg.nodes),
        'temporal_kg_edges': len(kg.edges),
        'hypotheses': len(hyps),
        'qdrant': {
            'text_collection': 'live_smoke_text',
            'text_points': query_qdrant_count('live_smoke_text'),
            'mm_collection': 'live_smoke_mm',
            'mm_points': query_qdrant_count('live_smoke_mm'),
        },
        'memgraph': query_memgraph_counts(),
        'artifacts': {
            'temporal_kg': str(kg_path),
            'hypotheses': str(hyps_path),
            'processed_dir': str(processed_dir),
        },
    }
    report_path = out_root / 'live_e2e_report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
