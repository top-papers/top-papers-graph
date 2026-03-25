from __future__ import annotations

import json
from pathlib import Path

from scireason.contracts import ChunkRecord
from scireason.ingest.store import save_paper


def test_save_paper_preserves_structured_chunk_records(tmp_path: Path) -> None:
    out_dir = tmp_path / 'papers'
    meta = {'id': 'paper1', 'title': 'Demo'}
    chunks = [
        ChunkRecord(
            chunk_id='paper1:0',
            paper_id='paper1',
            text='hello world',
            page=1,
            modality='table',
            source_backend='paddleocr_ppstructurev3',
            table_md='|a|b|',
            metadata={'lang': 'ru'},
        )
    ]
    paper_dir = save_paper(out_dir, meta, chunks)
    saved = [json.loads(line) for line in (paper_dir / 'chunks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(saved) == 1
    assert saved[0]['chunk_id'] == 'paper1:0'
    assert saved[0]['modality'] == 'table'
    assert saved[0]['table_md'] == '|a|b|'
    assert saved[0]['paper_id'] == 'paper1'


def test_save_paper_uses_single_safe_directory_for_doi_ids(tmp_path: Path) -> None:
    out_dir = tmp_path / "papers"
    meta = {"id": "doi:10.1070/pu1997v040n11abeh000308", "title": "Demo DOI"}

    paper_dir = save_paper(out_dir, meta, ["chunk"])

    assert paper_dir.name == "doi_10.1070_pu1997v040n11abeh000308"
    assert (paper_dir / "meta.json").exists()
    assert (paper_dir / "chunks.jsonl").exists()
    # No nested DOI path fragments should be created under processed_papers/.
    assert not (out_dir / "doi:10.1070").exists()
