#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ExportStats:
    trajectory_records: int = 0
    graph_repair_records: int = 0
    temporal_fix_records: int = 0
    mm_records: int = 0


def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def choose_split(key: str, val_ratio: float = 0.15) -> str:
    h = int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) % 10_000
    return 'eval' if h < int(10_000 * val_ratio) else 'train'


def source_block(sources: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        head_parts = [str(s.get('type') or '').strip(), str(s.get('source') or '').strip()]
        locator = str(s.get('locator') or s.get('page') or '').strip()
        if locator:
            head_parts.append(locator)
        head = ' | '.join(p for p in head_parts if p)
        snippet = str(s.get('snippet_or_summary') or '').strip()
        if snippet:
            lines.append(f'- {head}\n  snippet: {snippet}')
        else:
            lines.append(f'- {head}')
    return '\n'.join(lines) if lines else '-'


def maybe_image_path(source: Dict[str, Any], base_dir: Path) -> Optional[str]:
    for key in ('image_path', 'image', 'local_image', 'file'):
        value = source.get(key)
        if not value:
            continue
        p = Path(str(value))
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if p.exists() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
            return str(p)
    return None


def build_messages(system: str, user: str, assistant: str) -> List[Dict[str, str]]:
    return [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
        {'role': 'assistant', 'content': assistant},
    ]


def trajectory_records(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    doc = read_yaml(path)
    topic = str(doc.get('topic') or '').strip()
    domain = str(doc.get('domain') or 'science').strip()
    submission_id = str(doc.get('submission_id') or path.stem)
    expert = doc.get('expert') or {}
    expert_key = str(expert.get('latin_slug') or expert.get('full_name') or submission_id)
    papers = ensure_list(doc.get('papers'))
    paper_lines = [f"- {p.get('id','?')} ({p.get('year','?')}): {p.get('title','')}" for p in papers if isinstance(p, dict)]
    paper_block = '\n'.join(paper_lines) if paper_lines else '-'

    for idx, step in enumerate(ensure_list(doc.get('steps')), start=1):
        if not isinstance(step, dict):
            continue
        sources = ensure_list(step.get('sources'))
        conditions = step.get('conditions') or {}
        claim = str(step.get('claim') or '').strip()
        inference = str(step.get('inference') or '').strip()
        next_question = str(step.get('next_question') or '').strip()
        if not claim or not inference:
            continue

        user = (
            f"Topic: {topic}\n"
            f"Domain: {domain}\n"
            f"Submission: {submission_id}\n\n"
            f"Papers:\n{paper_block}\n\n"
            f"Claim to analyze:\n{claim}\n\n"
            f"Sources:\n{source_block(sources)}\n\n"
            f"Conditions:\n{json.dumps(conditions, ensure_ascii=False, indent=2)}\n\n"
            "Return a compact grounded scientific step with fields: inference, next_question, and extracted_assertions."
        )
        extracted = []
        for sidx, s in enumerate(sources, start=1):
            if not isinstance(s, dict):
                continue
            extracted.append({
                'assertion_id': f'{submission_id}:step{idx}:src{sidx}',
                'subject': claim,
                'predicate': 'supported_by',
                'object': str(s.get('snippet_or_summary') or s.get('source') or '').strip(),
                'evidence': {
                    'source': s.get('source'),
                    'locator': s.get('locator'),
                    'snippet_or_summary': s.get('snippet_or_summary'),
                },
                'time': {'granularity': 'unknown', 'start': 'unknown', 'end': 'unknown'},
            })
        assistant = json.dumps(
            {
                'inference': inference,
                'next_question': next_question,
                'extracted_assertions': extracted,
            },
            ensure_ascii=False,
            indent=2,
        )

        image_path = None
        for src in sources:
            if isinstance(src, dict):
                image_path = maybe_image_path(src, path.parent)
                if image_path:
                    break

        rec = {
            'id': f'trajectory:{submission_id}:{idx}',
            'task_family': 'trajectory_reasoning',
            'domain': domain,
            'topic': topic,
            'expert_key': expert_key,
            'source_file': str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            'messages': build_messages(
                'You are a careful scientific extraction assistant. Be concise, evidence-grounded and schema-faithful.',
                user,
                assistant,
            ),
            'metadata': {
                'submission_id': submission_id,
                'step_id': step.get('step_id', idx),
                'has_image_source': bool(image_path),
            },
        }
        if image_path:
            rec['image'] = image_path
        yield expert_key, rec


def graph_repair_records(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    doc = read_json(path)
    domain = str(doc.get('domain') or 'science')
    expert_key = str(doc.get('expert_key') or path.stem)
    assertions = ensure_list(doc.get('assertions'))
    for idx, a in enumerate(assertions, start=1):
        if not isinstance(a, dict):
            continue
        corrections = {
            'subject': a.get('corrected_subject'),
            'predicate': a.get('corrected_predicate'),
            'object': a.get('corrected_object'),
            'start_date': a.get('corrected_start_date') or a.get('corrected_time_interval') or a.get('start_date'),
            'end_date': a.get('corrected_end_date') or a.get('end_date'),
        }
        has_correction = any(v not in (None, '', 'unknown') for v in corrections.values())
        if not has_correction:
            continue
        ev = a.get('evidence') or {}
        user = (
            f"Domain: {domain}\n"
            f"Original assertion:\n{json.dumps({k: a.get(k) for k in ['subject', 'predicate', 'object', 'start_date', 'end_date']}, ensure_ascii=False)}\n\n"
            f"Evidence:\n{json.dumps(ev, ensure_ascii=False)}\n\n"
            f"Review verdict: {a.get('verdict')}\n"
            f"Review rationale: {a.get('rationale')}\n\n"
            "Produce the corrected assertion JSON only."
        )
        assistant = json.dumps(
            {
                'subject': corrections['subject'] or a.get('subject'),
                'predicate': corrections['predicate'] or a.get('predicate'),
                'object': corrections['object'] or a.get('object'),
                'start_date': corrections['start_date'] or a.get('start_date', 'unknown'),
                'end_date': corrections['end_date'] or a.get('end_date', 'unknown'),
                'evidence': ev,
            },
            ensure_ascii=False,
            indent=2,
        )
        yield expert_key, {
            'id': f'graph_repair:{path.stem}:{idx}',
            'task_family': 'graph_repair',
            'domain': domain,
            'expert_key': expert_key,
            'source_file': str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            'messages': build_messages(
                'You fix scientific graph assertions using expert feedback. Output only valid JSON.',
                user,
                assistant,
            ),
            'metadata': {'review_index': idx},
        }


def temporal_fix_records(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    doc = read_json(path)
    expert_key = str(doc.get('expert_key') or path.stem)
    for idx, c in enumerate(ensure_list(doc.get('corrections')), start=1):
        if not isinstance(c, dict):
            continue
        corrected = c.get('corrected_time') or c.get('corrected_interval')
        if not corrected:
            continue
        original = c.get('original_time') or c.get('predicted_time') or {'granularity': 'unknown', 'start': 'unknown', 'end': 'unknown'}
        user = (
            f"Assertion id: {c.get('assertion_id')}\n"
            f"Original temporal assignment:\n{json.dumps(original, ensure_ascii=False)}\n\n"
            f"Review rationale: {c.get('rationale','')}\n\n"
            "Return the corrected temporal object as JSON only."
        )
        yield expert_key, {
            'id': f'temporal_fix:{path.stem}:{idx}',
            'task_family': 'temporal_fix',
            'domain': str(doc.get('domain') or 'science'),
            'expert_key': expert_key,
            'source_file': str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            'messages': build_messages(
                'You correct temporal scopes for scientific assertions. Output only valid JSON.',
                user,
                json.dumps(corrected, ensure_ascii=False, indent=2),
            ),
            'metadata': {'assertion_id': c.get('assertion_id')},
        }


def mm_review_records(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    doc = read_json(path)
    expert_key = str(doc.get('expert_key') or path.stem)
    for idx, item in enumerate(ensure_list(doc.get('items')), start=1):
        if not isinstance(item, dict):
            continue
        corrected = item.get('corrected_caption') or item.get('vlm_caption')
        original = item.get('original_caption') or item.get('auto_caption')
        if not corrected:
            continue
        rec = {
            'id': f'mm_review:{path.stem}:{idx}',
            'task_family': 'mm_caption_fix',
            'domain': str(doc.get('domain') or 'science'),
            'expert_key': expert_key,
            'source_file': str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            'messages': build_messages(
                'You revise multimodal descriptions of scientific figures using expert feedback.',
                (
                    f"Page: {item.get('page')}\n"
                    f"Original caption: {original or 'unknown'}\n"
                    f"Verdict: {item.get('verdict')}\n"
                    f"Rationale: {item.get('rationale','')}\n\n"
                    "Return the corrected multimodal description in one paragraph."
                ),
                str(corrected),
            ),
            'metadata': {'page': item.get('page')},
        }
        image_path = item.get('image') or item.get('image_path')
        if image_path:
            p = Path(str(image_path))
            if not p.is_absolute():
                p = (path.parent / p).resolve()
            if p.exists():
                rec['image'] = str(p)
        yield expert_key, rec


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    return count


def iter_paths(paths: List[Path], patterns: Tuple[str, ...]) -> Iterable[Path]:
    for base in paths:
        if not base.exists():
            continue
        if base.is_file():
            yield base
            continue
        seen = set()
        for pattern in patterns:
            for p in sorted(base.glob(pattern)):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, default=REPO_ROOT)
    ap.add_argument('--trajectory-dir', type=Path, action='append', default=[])
    ap.add_argument('--graph-review-dir', type=Path, action='append', default=[])
    ap.add_argument('--temporal-correction-dir', type=Path, action='append', default=[])
    ap.add_argument('--mm-review-dir', type=Path, action='append', default=[])
    ap.add_argument('--output-train', type=Path, default=Path('data/derived/training/vlm_sft_train.jsonl'))
    ap.add_argument('--output-eval', type=Path, default=Path('data/derived/training/vlm_sft_eval.jsonl'))
    ap.add_argument('--output-all', type=Path, default=Path('data/derived/training/vlm_sft.jsonl'))
    ap.add_argument('--output-smoke', type=Path, default=Path('data/derived/training/vlm_sft_smoke.jsonl'))
    ap.add_argument('--output-summary', type=Path, default=Path('reports/build_summary.txt'))
    ap.add_argument('--smoke-size', type=int, default=256)
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    traj_dirs = args.trajectory_dir or [repo_root / 'data' / 'experts' / 'trajectories', repo_root / 'examples' / 'task2_validation_inputs', repo_root / 'examples' / 'uploaded_expert_artifacts']
    graph_dirs = args.graph_review_dir or [repo_root / 'data' / 'experts' / 'graph_reviews']
    temporal_dirs = args.temporal_correction_dir or [repo_root / 'data' / 'experts' / 'temporal_corrections']
    mm_dirs = args.mm_review_dir or [repo_root / 'data' / 'experts' / 'mm_reviews']

    stats = ExportStats()
    all_rows: List[Tuple[str, Dict[str, Any]]] = []

    for p in iter_paths(traj_dirs, ('**/*.yaml', '**/*.yml')):
        batch = list(trajectory_records(p))
        stats.trajectory_records += len(batch)
        all_rows.extend(batch)
    for p in iter_paths(graph_dirs, ('**/*.json',)):
        batch = list(graph_repair_records(p))
        stats.graph_repair_records += len(batch)
        all_rows.extend(batch)
    for p in iter_paths(temporal_dirs, ('**/*.json',)):
        batch = list(temporal_fix_records(p))
        stats.temporal_fix_records += len(batch)
        all_rows.extend(batch)
    for p in iter_paths(mm_dirs, ('**/*.json',)):
        batch = list(mm_review_records(p))
        stats.mm_records += len(batch)
        all_rows.extend(batch)

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for expert_key, row in all_rows:
        split = choose_split(expert_key, val_ratio=args.val_ratio)
        (eval_rows if split == 'eval' else train_rows).append(row)

    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed + 1).shuffle(eval_rows)
    if not eval_rows and len(train_rows) > 4:
        eval_rows.append(train_rows.pop())

    smoke_rows = train_rows[: min(args.smoke_size, len(train_rows))]
    all_only = [row for _, row in all_rows]

    output_train = repo_root / args.output_train if not args.output_train.is_absolute() else args.output_train
    output_eval = repo_root / args.output_eval if not args.output_eval.is_absolute() else args.output_eval
    output_all = repo_root / args.output_all if not args.output_all.is_absolute() else args.output_all
    output_smoke = repo_root / args.output_smoke if not args.output_smoke.is_absolute() else args.output_smoke
    output_summary = repo_root / args.output_summary if not args.output_summary.is_absolute() else args.output_summary

    write_jsonl(output_train, train_rows)
    write_jsonl(output_eval, eval_rows)
    write_jsonl(output_all, all_only)
    write_jsonl(output_smoke, smoke_rows)

    summary = {
        'trajectory_records': stats.trajectory_records,
        'graph_repair_records': stats.graph_repair_records,
        'temporal_fix_records': stats.temporal_fix_records,
        'mm_records': stats.mm_records,
        'total_records': len(all_only),
        'train_records': len(train_rows),
        'eval_records': len(eval_rows),
        'smoke_records': len(smoke_rows),
        'task_family_counts': dict(Counter(row.get('task_family') for row in all_only)),
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
