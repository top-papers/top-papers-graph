from __future__ import annotations

from scireason.mm import vlm


def test_temporary_vlm_selection_resets_disable_flags() -> None:
    vlm._G4F_AUTH_DISABLED = True
    vlm._G4F_TIMEOUT_DISABLED = True
    vlm._LOCAL_VLM_DISABLED = True
    vlm._LOCAL_VLM_DISABLE_REASON = 'boom'

    with vlm.temporary_vlm_selection(vlm_backend='qwen2_vl', vlm_model_id='Qwen/Qwen2.5-VL-3B-Instruct'):
        assert vlm._G4F_AUTH_DISABLED is False
        assert vlm._G4F_TIMEOUT_DISABLED is False
        assert vlm._LOCAL_VLM_DISABLED is False
        assert vlm._LOCAL_VLM_DISABLE_REASON == ''
        vlm._LOCAL_VLM_DISABLED = True
        vlm._LOCAL_VLM_DISABLE_REASON = 'again'

    assert vlm._G4F_AUTH_DISABLED is False
    assert vlm._G4F_TIMEOUT_DISABLED is False
    assert vlm._LOCAL_VLM_DISABLED is False
    assert vlm._LOCAL_VLM_DISABLE_REASON == ''
