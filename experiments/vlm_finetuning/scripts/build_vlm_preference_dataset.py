#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def ensure_list(v: Any) -> List[Any]:
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


def choose_split(key: str, val_ratio: float) -> str:
    h = int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) % 10_000
    return 'eval' if h < int(10_000 * val_ratio) else 'train'


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    return count


def graph_pairs(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    doc = read_json(path)
    domain = str(doc.get('domain') or 'science')
    expert_key = str(doc.get('expert_key') or path.stem)
    for idx, a in enumerate(ensure_list(doc.get('assertions')), start=1):
        if not isinstance(a, dict):
            continue
        has_any = any(a.get(k) not in (None, '') for k in ('corrected_subject', 'corrected_predicate', 'corrected_object', 'corrected_time_interval', 'corrected_start_date', 'corrected_end_date'))
        if not has_any:
            continue
        ev = a.get('evidence') or {}
        yield expert_key, {
            'id': f'graph_pref:{path.stem}:{idx}',
            'task_family': 'graph_repair_preference',
            'domain': domain,
            'expert_key': expert_key,
            'prompt': (
                f"You are correcting a scientific temporal-graph assertion.\n"
                f"Domain: {domain}\n"
                f"Evidence: {json.dumps(ev, ensure_ascii=False)}\n"
                f"Review rationale: {a.get('rationale','')}\n"
                f"Original assertion: {json.dumps({k: a.get(k) for k in ['subject', 'predicate', 'object', 'start_date', 'end_date']}, ensure_ascii=False)}\n"
                "Return only the best corrected JSON assertion."
            ),
            'chosen': json.dumps({
                'subject': a.get('corrected_subject') or a.get('subject'),
                'predicate': a.get('corrected_predicate') or a.get('predicate'),
                'object': a.get('corrected_object') or a.get('object'),
                'start_date': a.get('corrected_start_date') or a.get('corrected_time_interval') or a.get('start_date', 'unknown'),
                'end_date': a.get('corrected_end_date') or a.get('end_date', 'unknown'),
                'evidence': ev,
            }, ensure_ascii=False, indent=2),
            'rejected': json.dumps({
                'subject': a.get('subject'),
                'predicate': a.get('predicate'),
                'object': a.get('object'),
                'start_date': a.get('start_date', 'unknown'),
                'end_date': a.get('end_date', 'unknown'),
                'evidence': ev,
            }, ensure_ascii=False, indent=2),
            'metadata': {'source_file': str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)},
        }


def temporal_pairs(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    doc = read_json(path)
    expert_key = str(doc.get('expert_key') or path.stem)
    for idx, c in enumerate(ensure_list(doc.get('corrections')), start=1):
        if not isinstance(c, dict):
            continue
        corrected = c.get('corrected_time') or c.get('corrected_interval')
        original = c.get('original_time') or c.get('predicted_time')
        if not corrected or not original:
            continue
        yield expert_key, {
            'id': f'temporal_pref:{path.stem}:{idx}',
            'task_family': 'temporal_preference',
            'domain': str(doc.get('domain') or 'science'),
            'expert_key': expert_key,
            'prompt': (
                f"You are correcting temporal scope for a scientific assertion.\n"
                f"Assertion id: {c.get('assertion_id')}\n"
                f"Rationale: {c.get('rationale','')}\n"
                f"Original time: {json.dumps(original, ensure_ascii=False)}\n"
                "Return only the corrected temporal JSON."
            ),
            'chosen': json.dumps(corrected, ensure_ascii=False, indent=2),
            'rejected': json.dumps(original, ensure_ascii=False, indent=2),
            'metadata': {'source_file': str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)},
        }


def mm_pairs(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    doc = read_json(path)
    expert_key = str(doc.get('expert_key') or path.stem)
    for idx, item in enumerate(ensure_list(doc.get('items')), start=1):
        if not isinstance(item, dict):
            continue
        original = item.get('original_caption') or item.get('auto_caption')
        corrected = item.get('corrected_caption') or item.get('vlm_caption')
        if not original or not corrected or str(original).strip() == str(corrected).strip():
            continue
        rec = {
            'id': f'mm_pref:{path.stem}:{idx}',
            'task_family': 'mm_caption_preference',
            'domain': str(doc.get('domain') or 'science'),
            'expert_key': expert_key,
            'prompt': (
                f"You are revising a multimodal description of a scientific page or figure.\n"
                f"Page: {item.get('page')}\n"
                f"Verdict: {item.get('verdict')}\n"
                f"Rationale: {item.get('rationale','')}\n"
                "Return the better one-paragraph grounded description."
            ),
            'chosen': str(corrected),
            'rejected': str(original),
            'metadata': {'source_file': str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)},
        }
        image_path = item.get('image') or item.get('image_path')
        if image_path:
            p = Path(str(image_path))
            if not p.is_absolute():
                p = (path.parent / p).resolve()
            if p.exists():
                rec['image'] = str(p)
        yield expert_key, rec


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
    ap.add_argument('--graph-review-dir', type=Path, action='append', default=[])
    ap.add_argument('--temporal-correction-dir', type=Path, action='append', default=[])
    ap.add_argument('--mm-review-dir', type=Path, action='append', default=[])
    ap.add_argument('--output-train', type=Path, default=Path('data/derived/training/vlm_dpo_train.jsonl'))
    ap.add_argument('--output-eval', type=Path, default=Path('data/derived/training/vlm_dpo_eval.jsonl'))
    ap.add_argument('--output-all', type=Path, default=Path('data/derived/training/vlm_dpo.jsonl'))
    ap.add_argument('--output-summary', type=Path, default=Path('reports/dpo_build_summary.txt'))
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    graph_dirs = args.graph_review_dir or [repo_root / 'data' / 'experts' / 'graph_reviews']
    temporal_dirs = args.temporal_correction_dir or [repo_root / 'data' / 'experts' / 'temporal_corrections']
    mm_dirs = args.mm_review_dir or [repo_root / 'data' / 'experts' / 'mm_reviews']

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for p in iter_paths(graph_dirs, ('**/*.json',)):
        rows.extend(graph_pairs(p))
    for p in iter_paths(temporal_dirs, ('**/*.json',)):
        rows.extend(temporal_pairs(p))
    for p in iter_paths(mm_dirs, ('**/*.json',)):
        rows.extend(mm_pairs(p))

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for expert_key, row in rows:
        split = choose_split(expert_key, args.val_ratio)
        (eval_rows if split == 'eval' else train_rows).append(row)

    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed + 1).shuffle(eval_rows)
    if not eval_rows and len(train_rows) > 4:
        eval_rows.append(train_rows.pop())

    output_train = repo_root / args.output_train if not args.output_train.is_absolute() else args.output_train
    output_eval = repo_root / args.output_eval if not args.output_eval.is_absolute() else args.output_eval
    output_all = repo_root / args.output_all if not args.output_all.is_absolute() else args.output_all
    output_summary = repo_root / args.output_summary if not args.output_summary.is_absolute() else args.output_summary

    write_jsonl(output_train, train_rows)
    write_jsonl(output_eval, eval_rows)
    write_jsonl(output_all, [r for _, r in rows])
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps({'total_pairs': len(rows), 'train_pairs': len(train_rows), 'eval_pairs': len(eval_rows)}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(output_summary.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
