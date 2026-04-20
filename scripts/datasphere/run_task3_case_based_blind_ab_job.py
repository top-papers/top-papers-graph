from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = REPO_ROOT / 'scripts' / 'task3' / 'run_task3_case_based_blind_ab.py'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='DataSphere Jobs entrypoint for Task 3 case-based blind A/B.')
    parser.add_argument('--trajectory', required=True)
    parser.add_argument('--case-manifest', required=True)
    parser.add_argument('--processed-dir', required=True)
    parser.add_argument('--out-dir', default='job_outputs/task3_case_based_blind_ab')
    parser.add_argument('--query', default='')
    parser.add_argument('--model-a-vlm-model-id', default='')
    parser.add_argument('--model-b-vlm-model-id', default='')
    parser.add_argument('--model-a-local-text-model', default=None)
    parser.add_argument('--model-b-local-text-model', default=None)
    parser.add_argument('--run-vlm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--include-multimodal', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hf-offline', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--top-hypotheses', type=int, default=16)
    parser.add_argument('--candidate-top-k', type=int, default=24)
    parser.add_argument('--top-papers', type=int, default=12)
    parser.add_argument('--edge-mode', default='auto')
    parser.add_argument('--link-prediction-backend', default='auto')
    return parser


def main() -> int:
    args, unknown = build_parser().parse_known_args()
    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(BASE_SCRIPT),
        '--trajectory',
        args.trajectory,
        '--case-manifest',
        args.case_manifest,
        '--processed-dir',
        args.processed_dir,
        '--out-dir',
        str(out_dir),
        '--query',
        args.query,
        '--top-hypotheses',
        str(args.top_hypotheses),
        '--candidate-top-k',
        str(args.candidate_top_k),
        '--top-papers',
        str(args.top_papers),
        '--edge-mode',
        args.edge_mode,
        '--link-prediction-backend',
        args.link_prediction_backend,
    ]
    if args.model_a_vlm_model_id:
        cmd.extend(['--model-a-vlm-model-id', args.model_a_vlm_model_id])
    if args.model_b_vlm_model_id:
        cmd.extend(['--model-b-vlm-model-id', args.model_b_vlm_model_id])
    if args.model_a_local_text_model:
        cmd.extend(['--model-a-local-text-model', args.model_a_local_text_model])
    if args.model_b_local_text_model:
        cmd.extend(['--model-b-local-text-model', args.model_b_local_text_model])
    cmd.append('--run-vlm' if args.run_vlm else '--no-run-vlm')
    cmd.append('--include-multimodal' if args.include_multimodal else '--no-include-multimodal')
    cmd.append('--hf-offline' if args.hf_offline else '--no-hf-offline')
    cmd.extend(unknown)
    print('Executing:', ' '.join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=os.environ.copy())


if __name__ == '__main__':
    raise SystemExit(main())
