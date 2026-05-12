from __future__ import annotations

import json
from pathlib import Path

from scireason.task3_hf_benchmark import BuildConfig, adapt_creator_prompt_for_model, build_dataset


def test_adapt_creator_prompt_for_single_vlm() -> None:
    prompt = "Сравните, какой вариант лучше извлекает из рисунка 3 значения угла"
    assert adapt_creator_prompt_for_model(prompt).startswith("Извлеките из рисунка 3")


def test_build_task3_hf_benchmark_from_manifest(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "task3_ab_case_manifest_v1",
        "experiment_meta": {"topic": "demo", "submission_id": "demo__123", "creator_id": "Expert"},
        "cases": [
            {
                "case_id": "CASE-001",
                "enabled": True,
                "primary_endpoint": True,
                "stratum": "multimodal_hard",
                "paper_title": "Demo paper",
                "paper_id": "arxiv:2401.00001",
                "year": "2024",
                "evidence_kind": "figure",
                "page_hint": "Fig. 1",
                "creator_prompt": "Сравните, какой вариант лучше извлекает значение из Fig. 1",
                "creator_rationale": "diagnostic rationale",
                "review_focus": ["visual_fact"],
                "expected_error_modes": ["missed_visual_fact"],
            }
        ],
    }
    inp = tmp_path / "input"
    inp.mkdir()
    (inp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "dataset"
    stats = build_dataset(BuildConfig(input_paths=(inp,), output_dir=out, render_pdf_pages=False))
    assert stats["cases_written"] == 1
    rows = [json.loads(line) for line in (out / "data" / "task3_vlm_generation.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["messages"][1]["content"][0]["type"] == "text"
    assert rows[0]["images"] == []
    assert "diagnostic rationale" not in rows[0]["messages"][1]["content"][0]["text"]
