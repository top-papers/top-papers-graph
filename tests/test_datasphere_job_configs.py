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
