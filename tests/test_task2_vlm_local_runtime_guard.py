from __future__ import annotations

from pathlib import Path

from PIL import Image

from scireason.mm import vlm


def test_local_vlm_failure_is_disabled_for_remaining_pages(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)

    monkeypatch.setattr(vlm, "_LOCAL_VLM_DISABLED", False)
    monkeypatch.setattr(vlm, "_LOCAL_VLM_DISABLE_REASON", "")
    monkeypatch.setattr(vlm, "_has_local_vlm_stack", lambda model_id=None: True)

    calls = {"count": 0}

    def _boom(**kwargs):
        calls["count"] += 1
        raise RuntimeError("Для VLM backend нужна зависимость 'transformers/torch'.")

    monkeypatch.setattr(vlm, "_describe_image_qwen", _boom)

    res1 = vlm.describe_image(
        image_path=image_path,
        prompt="demo",
        backend="qwen2_vl",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    assert res1.caption == ""
    assert calls["count"] == 1
    assert vlm._LOCAL_VLM_DISABLED is True

    res2 = vlm.describe_image(
        image_path=image_path,
        prompt="demo",
        backend="qwen2_vl",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    assert res2.caption == ""
    assert calls["count"] == 1
