from __future__ import annotations

from pathlib import Path

from PIL import Image

from scireason.mm import vlm
from scireason.mm import vlm_worker


def test_worker_error_does_not_fallback_inprocess_by_default(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / 'page.png'
    Image.new('RGB', (8, 8), color='white').save(image_path)

    monkeypatch.setenv('SCIREASON_LOCAL_VLM_MODE', 'worker')

    def _boom(**kwargs):
        raise RuntimeError('worker failed')

    monkeypatch.setattr(vlm, '_describe_image_qwen_worker', _boom)
    called = {'inproc': 0}

    def _inproc(**kwargs):
        called['inproc'] += 1
        return vlm.VLMResult(caption='inproc-ok')

    monkeypatch.setattr(vlm, '_describe_image_qwen_inprocess', _inproc)

    try:
        vlm._describe_image_qwen(
            image_path=image_path,
            prompt='demo',
            model_id='Qwen/Qwen2.5-VL-3B-Instruct',
            max_new_tokens=16,
        )
    except RuntimeError as exc:
        assert 'worker failed' in str(exc)
    else:
        raise AssertionError('expected worker error to propagate in default worker mode')

    assert called['inproc'] == 0


def test_worker_cpu_fallback_on_nvrtc_runtime(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / 'page.png'
    Image.new('RGB', (8, 8), color='white').save(image_path)

    monkeypatch.setattr(vlm_worker, '_FORCE_CPU_MODEL_IDS', set())
    monkeypatch.setattr(vlm_worker, '_preferred_device_mode', lambda model_id: 'cuda')

    calls: list[str] = []

    def _fake_run_once(*, image_path, prompt, model_id, max_new_tokens, device_mode):
        calls.append(device_mode)
        if device_mode == 'cuda':
            raise RuntimeError('nvrtc: error: failed to open libnvrtc-builtins.so.13.0')
        return {'caption': 'cpu-ok', 'extracted_tables_md': None, 'extracted_equations_md': None}

    monkeypatch.setattr(vlm_worker, '_run_once', _fake_run_once)
    monkeypatch.setattr(vlm_worker, '_clear_model_cache', lambda: None)

    result = vlm_worker._describe(
        image_path=image_path,
        prompt='demo',
        model_id='Qwen/Qwen2.5-VL-3B-Instruct',
        max_new_tokens=8,
    )

    assert result['caption'] == 'cpu-ok'
    assert calls == ['cuda', 'cpu']
    assert 'Qwen/Qwen2.5-VL-3B-Instruct' in vlm_worker._FORCE_CPU_MODEL_IDS
