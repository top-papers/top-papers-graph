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




def test_vlm_wrapper_keeps_ddp_find_unused_configurable_not_forced() -> None:
    script = (DATASPHERE_DIR / "bin" / "run_hf_top_papers_sft_grpo_full.sh").read_text(encoding="utf-8")
    assert "SFT_DDP_FIND_UNUSED_PARAMETERS" in script
    assert "GRPO_DDP_FIND_UNUSED_PARAMETERS" in script
    assert 'append_optional_bool_flag SFT_DDP_FIND_UNUSED_PARAMETERS --ddp-find-unused-parameters --no-ddp-find-unused-parameters' in script
    assert 'append_optional_bool_flag GRPO_DDP_FIND_UNUSED_PARAMETERS --ddp-find-unused-parameters --no-ddp-find-unused-parameters' in script


def test_full_config_uses_safe_ddp_unused_detection_and_larger_grpo_groups() -> None:
    path = DATASPHERE_DIR / "job_configs" / "hf_top_papers_sft_grpo_full_g2_2.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    vars_list = cfg.get("env", {}).get("vars", [])
    env = {next(iter(item)): next(iter(item.values())) for item in vars_list if isinstance(item, dict) and item}

    assert str(env["SFT_DDP_FIND_UNUSED_PARAMETERS"]) == "1"
    assert str(env["GRPO_DDP_FIND_UNUSED_PARAMETERS"]) == "1"
    assert int(env["GRPO_NUM_GENERATIONS"]) >= 4
    assert int(env["MAX_GRPO_STEPS"]) <= 120


def test_datasphere_requirements_are_datasphere_cli_parseable() -> None:
    from packaging.requirements import Requirement

    lines = (DATASPHERE_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert lines, "requirements.txt must not be empty"
    for line in lines:
        assert line.strip() == line, line
        assert line and not line.startswith("#") and not line.startswith("--"), line
        Requirement(line)


def test_datasphere_requirements_pin_runtime_surface_for_env_resolution() -> None:
    text = (DATASPHERE_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "datasets>=4.7.0,<5" in text
    assert "huggingface_hub>=0.34.0,<1.0" in text
    assert "transformers>=4.57.0,<4.58" in text
    assert "trl>=1.4.0,<1.7" in text
    assert "peft>=0.17.0,<0.20" in text
    assert "bitsandbytes==0.48.1" in text

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



def test_smoke_config_avoids_all_masked_grpo_completions() -> None:
    path = DATASPHERE_DIR / "job_configs" / "hf_top_papers_sft_grpo_smoke_g2_2.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    vars_list = cfg.get("env", {}).get("vars", [])
    env = {next(iter(item)): next(iter(item.values())) for item in vars_list if isinstance(item, dict) and item}

    assert int(env["GRPO_MAX_COMPLETION_LENGTH"]) >= 128
    assert str(env["GRPO_MASK_TRUNCATED_COMPLETIONS"]).lower() in {"0", "false", "no", "off"}


def test_vlm_wrapper_makes_grpo_truncation_mask_configurable() -> None:
    script = (DATASPHERE_DIR / "bin" / "run_hf_top_papers_sft_grpo_full.sh").read_text(encoding="utf-8")
    assert "GRPO_MASK_TRUNCATED_COMPLETIONS" in script
    assert "GRPO_MASK_ARGS+=(--mask-truncated-completions)" in script
    assert script.count("--mask-truncated-completions") == 1


def test_full_config_avoids_all_masked_grpo_completions() -> None:
    path = DATASPHERE_DIR / "job_configs" / "hf_top_papers_sft_grpo_full_g2_2.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    vars_list = cfg.get("env", {}).get("vars", [])
    env = {next(iter(item)): next(iter(item.values())) for item in vars_list if isinstance(item, dict) and item}

    assert int(env["GRPO_MAX_COMPLETION_LENGTH"]) >= 512
    assert str(env["GRPO_MASK_TRUNCATED_COMPLETIONS"]).lower() in {"0", "false", "no", "off"}


def test_full_tutorial_does_not_recommend_grpo_32_token_masking_combo() -> None:
    text = (DATASPHERE_DIR / "TUTORIAL_FULL_EXPERIMENT_RU.md").read_text(encoding="utf-8")
    assert "GRPO_MAX_COMPLETION_LENGTH=32" not in text
    assert "GRPO_MASK_TRUNCATED_COMPLETIONS=0" in text


def test_datasphere_vlm_jobs_force_utf8_locale_and_disable_trl_model_card() -> None:
    for name in ["hf_top_papers_sft_grpo_full_g2_2.yaml", "hf_top_papers_sft_grpo_smoke_g2_2.yaml"]:
        path = DATASPHERE_DIR / "job_configs" / name
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        vars_list = cfg.get("env", {}).get("vars", [])
        env = {next(iter(item)): next(iter(item.values())) for item in vars_list if isinstance(item, dict) and item}
        assert env["LANG"] == "C.UTF-8"
        assert env["LC_ALL"] == "C.UTF-8"
        assert str(env["PYTHONUTF8"]) == "1"
        assert str(env["PYTHONIOENCODING"]).lower() == "utf-8"
        assert str(env["DISABLE_TRL_MODEL_CARD"]) == "1"


def test_v2_config_prefetches_model_and_disables_training_hub_calls() -> None:
    path = DATASPHERE_DIR / "job_configs" / "hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    vars_list = cfg.get("env", {}).get("vars", [])
    env = {next(iter(item)): next(iter(item.values())) for item in vars_list if isinstance(item, dict) and item}

    assert str(env["PREFETCH_BASE_MODEL"]) == "1"
    assert str(env["ENABLE_TRAINING_HF_OFFLINE"]) == "1"
    assert int(env["MAX_IMAGES_PER_EXAMPLE_SFT"]) == 0
    assert int(env["MAX_IMAGES_PER_EXAMPLE_GRPO"]) == 0
    assert int(env["SFT_MAX_TEXT_CHARS"]) == 0


def test_vlm_trainers_disable_optional_trl_model_card_creation_by_default() -> None:
    for rel in [
        "experiments/vlm_finetuning/scripts/train_vlm_sft.py",
        "experiments/vlm_finetuning/scripts/train_vlm_grpo.py",
    ]:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "def disable_trl_model_card_creation" in text
        assert "DISABLE_TRL_MODEL_CARD", rel
        assert "trainer.create_model_card" in text


def test_vlm_pipeline_creates_placeholder_declared_outputs_after_early_failure() -> None:
    script = (DATASPHERE_DIR / "bin" / "run_hf_top_papers_sft_grpo_full.sh").read_text(encoding="utf-8")
    assert "This placeholder archive was created because" in script
    assert "hf_upload_summary.json" in script
    assert "hf_upload_manifest.json" in script
    assert "not_run_or_incomplete" in script


def test_full_sft_config_has_full_data_and_ddp_straggler_settings() -> None:
    path = DATASPHERE_DIR / "job_configs" / "hf_top_papers_sft_grpo_full_g2_2.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    vars_list = cfg.get("env", {}).get("vars", [])
    env = {next(iter(item)): next(iter(item.values())) for item in vars_list if isinstance(item, dict) and item}

    assert int(env["SFT_MAX_TEXT_CHARS"]) == 0
    assert int(env["MAX_IMAGES_PER_EXAMPLE_SFT"]) == 0
    assert int(env["MAX_IMAGES_PER_EXAMPLE_GRPO"]) == 0
    assert int(env["MAX_SFT_STEPS"]) == -1
    assert int(env["SFT_DDP_TIMEOUT_SECONDS"]) >= 7200
    assert int(env["SFT_DATALOADER_NUM_WORKERS"]) <= 1


def test_sft_wrapper_passes_ddp_straggler_guard_args() -> None:
    script = (DATASPHERE_DIR / "bin" / "run_hf_top_papers_sft_grpo_full.sh").read_text(encoding="utf-8")
    assert "--max-text-chars" in script
    assert "SFT_MAX_TEXT_CHARS" in script
    assert "--ddp-timeout-seconds" in script
    assert "SFT_DDP_TIMEOUT_SECONDS" in script


def test_sft_trainer_config_accepts_ddp_timeout() -> None:
    text = (REPO_ROOT / "experiments/vlm_finetuning/scripts/train_vlm_sft.py").read_text(encoding="utf-8")
    assert "--ddp-timeout-seconds" in text
    assert "'ddp_timeout': args.ddp_timeout_seconds" in text
    assert "filter_dataset_by_text_chars" in text

def test_v2_and_legacy_full_configs_enable_complete_sft_dpo_grpo_by_default() -> None:
    for name in ["hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml", "hf_top_papers_sft_grpo_full_g2_2.yaml"]:
        path = DATASPHERE_DIR / "job_configs" / name
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        vars_list = cfg.get("env", {}).get("vars", [])
        env = {next(iter(item)): next(iter(item.values())) for item in vars_list if isinstance(item, dict) and item}
        assert str(env.get("ENABLE_GRPO_POLISH")) == "1", name
        assert "DPO_EPOCHS" in env, name
        assert "GRPO_MAX_STEPS" in env or name == "hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml", name
        outputs = "\n".join(map(str, cfg.get("outputs", [])))
        assert "dpo_lora.tar.gz" in outputs, name
        assert "grpo_lora.tar.gz" in outputs, name


def test_v2_wrapper_runs_grpo_after_dpo_by_default() -> None:
    script = (DATASPHERE_DIR / "bin" / "run_hf_top_papers_sft_dpo_grpo_v2.sh").read_text(encoding="utf-8")
    assert "train_vlm_dpo.py" in script
    assert "train_vlm_grpo.py" in script
    assert "--sft-adapter-path \"$DPO_DIR\"" in script
    assert 'if [ "${ENABLE_GRPO_POLISH:-1}" = "1" ]; then' in script
    assert "train_vlm_dpo.py" in script[: script.index("train_vlm_grpo.py")]
