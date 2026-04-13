from __future__ import annotations

from pathlib import Path

from PIL import Image

from scireason.mm import vlm


def test_describe_image_uses_settings_max_new_tokens(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    monkeypatch.setattr(vlm.settings, "vlm_backend", "qwen2_vl")
    monkeypatch.setattr(vlm.settings, "vlm_model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
    monkeypatch.setattr(vlm.settings, "vlm_max_new_tokens", 123)

    captured = {}

    def _fake_qwen(**kwargs):
        captured.update(kwargs)
        return vlm.VLMResult(caption="ok")

    monkeypatch.setattr(vlm, "_describe_image_qwen", _fake_qwen)
    monkeypatch.setattr(vlm, "_resolve_backend", lambda backend, model_id=None: "qwen2_vl")

    res = vlm.describe_image(image_path=image_path, prompt="demo", backend="qwen2_vl", model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    assert res.caption == "ok"
    assert captured["max_new_tokens"] == 123
