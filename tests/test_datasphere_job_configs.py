from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASPHERE_DIR = REPO_ROOT / "experiments" / "vlm_finetuning" / "datasphere"


def test_datasphere_job_configs_do_not_mix_root_path_and_local_paths() -> None:
    for path in sorted((DATASPHERE_DIR / "job_configs").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        python_env = cfg.get("env", {}).get("python", {})
        assert "local-paths" in python_env, path
        assert not ({"root-path", "root-paths"} & set(python_env)), path


def test_datasphere_job_configs_have_explicit_ssd_working_storage() -> None:
    for path in sorted((DATASPHERE_DIR / "job_configs").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        storage = cfg.get("working-storage")
        assert isinstance(storage, dict), path
        assert storage.get("type") == "SSD", path
        assert str(storage.get("size", "")).endswith("Gb"), path


def test_datasphere_common_sh_resolves_repository_root() -> None:
    script = DATASPHERE_DIR / "bin" / "common.sh"
    cmd = ["bash", "-lc", f"source {script.as_posix()} >/dev/null 2>&1; pwd"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, stdout=subprocess.PIPE)
    assert Path(result.stdout.strip()) == REPO_ROOT


def test_qwen3vl_full_job_uses_export_and_budget_guard() -> None:
    path = DATASPHERE_DIR / "job_configs" / "hf_top_papers_sft_grpo_full_g2_2.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    vars_list = cfg.get("env", {}).get("vars", [])
    env = {next(iter(item)): next(iter(item.values())) for item in vars_list if isinstance(item, dict) and item}

    assert cfg.get("cloud-instance-types") == ["g2.2"]
    assert env["HF_DATASET_SOURCE_MODE"] == "export"
    assert env["HF_DATASET_EXPORT_SUBDIR"] == "exports/colab-run-001"
    assert float(env["BUDGET_RUB"]) == 100000
    assert float(env["BUDGET_RESERVE_RUB"]) >= 5000
    assert float(env["G2_2_RUB_PER_HOUR"]) == 1085.76
    timeout_sum = sum(float(env[key]) for key in ["DATA_TIMEOUT_HOURS", "SFT_TIMEOUT_HOURS", "GRPO_TIMEOUT_HOURS", "HF_UPLOAD_TIMEOUT_HOURS"])
    assert timeout_sum * float(env["G2_2_RUB_PER_HOUR"]) + float(env["BUDGET_RESERVE_RUB"]) <= float(env["BUDGET_RUB"])


def test_vlm_sft_wrapper_does_not_pass_assistant_only_loss() -> None:
    script = (DATASPHERE_DIR / "bin" / "run_hf_top_papers_sft_grpo_full.sh").read_text(encoding="utf-8")
    assert "--assistant-only-loss" not in script




def test_vlm_wrapper_passes_ddp_find_unused_parameters() -> None:
    script = (DATASPHERE_DIR / "bin" / "run_hf_top_papers_sft_grpo_full.sh").read_text(encoding="utf-8")
    assert script.count("--ddp-find-unused-parameters") >= 2


def test_datasphere_requirements_pin_trl_below_future_17_surface() -> None:
    text = (DATASPHERE_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "trl>=1.4.0,<1.7" in text

def test_smoke_config_uses_valid_grpo_generation_count() -> None:
    path = DATASPHERE_DIR / "job_configs" / "hf_top_papers_sft_grpo_smoke_g2_2.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    vars_list = cfg.get("env", {}).get("vars", [])
    env = {next(iter(item)): next(iter(item.values())) for item in vars_list if isinstance(item, dict) and item}

    assert int(env["GRPO_NUM_GENERATIONS"]) >= 2
    assert int(env["GRPO_NUM_GENERATIONS_EVAL"]) >= 2


def test_run_full_pipeline_does_not_parse_auth_subject_id_as_job_id() -> None:
    import importlib.util

    path = DATASPHERE_DIR / "run_full_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_full_pipeline_for_job_id_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    sample = """
    You are going to be authenticated via subject-id 'ajesipqgkc5efi6eo4am'.
    2026-06-16 12:54:04,689 - [INFO] - created job `bt184khn14ilcq93juc6`
    """
    assert mod.parse_job_id(sample) == "bt184khn14ilcq93juc6"
    assert mod.parse_job_id("subject-id 'ajesipqgkc5efi6eo4am'") is None

