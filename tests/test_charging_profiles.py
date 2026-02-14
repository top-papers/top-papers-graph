from __future__ import annotations

from pathlib import Path

from scireason.experiments.pybamm_fastcharge import load_literature_profiles


def test_load_literature_profiles_env_dir(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    profiles_dir = repo_root / "examples" / "battery_fastcharge" / "configs" / "charging_profiles"
    monkeypatch.setenv("CHARGING_PROFILES_DIR", str(profiles_dir))

    # Passing a non-existing config_dir should still work thanks to CHARGING_PROFILES_DIR.
    lit = load_literature_profiles(config_dir=repo_root / "does_not_exist")
    assert "an2019_g9_2.0-1.5-0.9C_0-80soc" in lit
