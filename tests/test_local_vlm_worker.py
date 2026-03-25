from __future__ import annotations

from pathlib import Path

from PIL import Image

from scireason.mm import vlm


def test_describe_image_prefers_worker_path(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / 'page.png'
    Image.new('RGB', (8, 8), color='white').save(image_path)

    monkeypatch.setenv('SCIREASON_LOCAL_VLM_MODE', 'worker')
    monkeypatch.setattr(vlm, '_describe_image_qwen_worker', lambda **kwargs: vlm.VLMResult(caption='worker-ok'))
    monkeypatch.setattr(vlm, '_describe_image_qwen_inprocess', lambda **kwargs: vlm.VLMResult(caption='inproc-ok'))

    res = vlm._describe_image_qwen(
        image_path=image_path,
        prompt='demo',
        model_id='Qwen/Qwen2.5-VL-3B-Instruct',
        max_new_tokens=16,
    )
    assert res.caption == 'worker-ok'


def test_worker_failure_falls_back_to_inprocess_when_hybrid_mode_enabled(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / 'page.png'
    Image.new('RGB', (8, 8), color='white').save(image_path)

    monkeypatch.setenv('SCIREASON_LOCAL_VLM_MODE', 'hybrid')

    def _boom(**kwargs):
        raise RuntimeError('worker failed')

    monkeypatch.setattr(vlm, '_describe_image_qwen_worker', _boom)
    monkeypatch.setattr(vlm, '_local_stack_available', lambda model_id=None: True)
    monkeypatch.setattr(vlm, '_describe_image_qwen_inprocess', lambda **kwargs: vlm.VLMResult(caption='inproc-ok'))

    res = vlm._describe_image_qwen(
        image_path=image_path,
        prompt='demo',
        model_id='Qwen/Qwen2.5-VL-3B-Instruct',
        max_new_tokens=16,
    )
    assert res.caption == 'inproc-ok'
