from __future__ import annotations

from pathlib import Path

from PIL import Image

from scireason.mm import vlm


def test_describe_image_falls_back_to_empty_caption_when_g4f_missing(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)

    monkeypatch.setattr(vlm, "_has_g4f", lambda: False)
    monkeypatch.setattr(vlm, "_has_local_vlm_stack", lambda: False)

    res = vlm.describe_image(image_path=image_path, prompt="demo", backend="g4f", model_id="auto")
    assert res.caption == ""
    assert res.extracted_tables_md is None


def test_describe_image_uses_local_qwen_when_g4f_requested_but_missing(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)

    monkeypatch.setattr(vlm, "_has_g4f", lambda: False)
    monkeypatch.setattr(vlm, "_has_local_vlm_stack", lambda: True)

    called = {}

    def _fake_qwen(*, image_path: Path, prompt: str, model_id: str, max_new_tokens: int):
        called["model_id"] = model_id
        return vlm.VLMResult(caption="ok")

    monkeypatch.setattr(vlm, "_describe_image_qwen", _fake_qwen)

    res = vlm.describe_image(image_path=image_path, prompt="demo", backend="g4f", model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    assert res.caption == "ok"
    assert called["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_describe_image_disables_g4f_after_missing_auth(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)

    monkeypatch.setattr(vlm, "_has_g4f", lambda: True)
    monkeypatch.setattr(vlm, "_has_local_vlm_stack", lambda: False)
    monkeypatch.setattr(vlm, "_G4F_AUTH_DISABLED", False)

    def _boom(**kwargs):
        raise RuntimeError('openai_style=MissingAuthError: Add a "api_key"; images_arg=MissingAuthError: API key is required for Puter.js API.')

    monkeypatch.setattr(vlm, "_describe_image_g4f", _boom)

    res = vlm.describe_image(image_path=image_path, prompt="demo", backend="g4f", model_id="auto")
    assert res.caption == ""
    assert vlm._G4F_AUTH_DISABLED is True
