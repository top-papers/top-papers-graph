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
