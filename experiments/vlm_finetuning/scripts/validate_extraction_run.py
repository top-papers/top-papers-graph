#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm(v: Any) -> str:
    if v is None:
        return ''
    return ' '.join(str(v).strip().lower().split())


def triple_key(x: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        norm(x.get('subject')),
        norm(x.get('predicate')),
        norm(x.get('object')),
        norm(x.get('start_date') or (x.get('time') or {}).get('start')),
        norm(x.get('end_date') or (x.get('time') or {}).get('end')),
    )


def flatten_assertions(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(row.get('assertions'), list):
        return [x for x in row['assertions'] if isinstance(x, dict)]
    if isinstance(row.get('extracted_assertions'), list):
        return [x for x in row['extracted_assertions'] if isinstance(x, dict)]
    if isinstance(row.get('prediction'), dict):
        return flatten_assertions(row['prediction'])
    return []


def compute_metrics(pred: List[Dict[str, Any]], gold: List[Dict[str, Any]]) -> Dict[str, float]:
    pred_keys = {triple_key(x) for row in pred for x in flatten_assertions(row)}
    gold_keys = {triple_key(x) for row in gold for x in flatten_assertions(row)}
    tp = len(pred_keys & gold_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'pred_triplets': len(pred_keys), 'gold_triplets': len(gold_keys), 'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--gold', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    metrics = compute_metrics(read_jsonl(args.predictions), read_jsonl(args.gold))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.out_dir / 'summary.md').write_text(
        '\n'.join([
            '# Validation summary',
            f"- pred_triplets: {metrics['pred_triplets']}",
            f"- gold_triplets: {metrics['gold_triplets']}",
            f"- precision: {metrics['precision']:.4f}",
            f"- recall: {metrics['recall']:.4f}",
            f"- f1: {metrics['f1']:.4f}",
        ]),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
