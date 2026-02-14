from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional

import json
import time

import numpy as np
import yaml

try:
    import pybamm
except Exception:  # pragma: no cover
    pybamm = None  # type: ignore


def _require_pybamm() -> None:
    if pybamm is None:  # pragma: no cover
        raise RuntimeError("PyBaMM is not installed. Install extras: pip install -e '.[battery]'")


@dataclass(frozen=True)
class LitProfileMeta:
    """Metadata for a literature-based profile."""
    name: str
    source: str
    chemistry: str
    temperature_C: float
    soc_from: float
    soc_to: float


def _soc_delta_to_seconds(delta_soc: float, c_rate: float) -> float:
    """Idealized mapping from SOC window to time under constant C-rate.

    Assumption: 1C means charging full nominal capacity in 1 hour with ~100% coulombic efficiency.
    This is the same approximation used in many lab protocols when stages are separated 'by time'
    for a given SOC window.
    """
    if c_rate <= 0:
        raise ValueError("c_rate must be > 0")
    return float(delta_soc / c_rate * 3600.0)


def resolve_profiles_dir(config_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve a directory that contains charging profile YAML files.

    Precedence:
    1) explicit ``config_dir`` argument (if it exists)
    2) ``$CHARGING_PROFILES_DIR`` env var (if it exists)
    3) ``configs/charging_profiles`` (legacy, relative to CWD)
    4) ``examples/battery_fastcharge/configs/charging_profiles`` (repo example)
    """

    candidates: List[Path] = []
    if config_dir is not None:
        candidates.append(config_dir)

    env_dir = os.getenv("CHARGING_PROFILES_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(Path("configs/charging_profiles"))

    try:
        repo_root = Path(__file__).resolve().parents[3]
        candidates.append(repo_root / "examples" / "battery_fastcharge" / "configs" / "charging_profiles")
    except Exception:
        pass

    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def load_literature_profiles(config_dir: Optional[Path] = None) -> Dict[str, dict]:
    """Load YAML profile definitions (time-staged CC profiles).

    By default we look in:
    1) $CHARGING_PROFILES_DIR (if set)
    2) configs/charging_profiles (legacy)
    3) examples/battery_fastcharge/configs/charging_profiles (repo example)

    This keeps the core repository topic-agnostic while still shipping a battery fast-charge example.
    """
    profiles: Dict[str, dict] = {}

    chosen = resolve_profiles_dir(config_dir)
    if chosen is None:
        return profiles
    # NOTE: we must iterate `chosen` (the first existing candidate directory).
    # Previously we accidentally iterated `config_dir`, which broke $CHARGING_PROFILES_DIR.
    for p in sorted(chosen.glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        key = p.stem
        data["_file"] = str(p)
        profiles[key] = data
    return profiles


def default_profiles() -> Dict[str, List[str]]:
    """A small benchmark suite: one standard CCCV baseline + several literature MS-CC profiles.

    Baseline (CCCV) is expressed with voltage/current terminations to avoid brittle timing.
    MS-CC profiles are time-based (derived from SOC window lengths); see configs/charging_profiles/*.yaml.
    """
    return {
        # Friendly alias used in docs/quickstart.
        # (Matches `baseline_cc_1p5C_32min`.)
        "baseline_cc": [
            "Charge at 1.5C for 32 minutes",
            "Rest for 10 minutes",
        ],
        # Standard CCCV baseline (typical Li-ion cell procedure).
        # PyBaMM supports 'until <V>' and 'until C/<n>' termination in Experiment strings.
        "baseline_cccv_1C_4p2V": [
            "Charge at 1C until 4.2V",
            "Hold at 4.2V until C/20",
            "Rest for 10 minutes",
        ],
        # Faster naive baseline (often degrades / plating risk in practice; included for comparison).
        "baseline_cc_1p5C_32min": [
            "Charge at 1.5C for 32 minutes",
            "Rest for 10 minutes",
        ],
    }


def build_steps_from_lit_yaml(yaml_profile: dict) -> List[str]:
    """Convert a literature YAML profile into PyBaMM Experiment strings."""
    stages = yaml_profile["stages"]
    steps: List[str] = []
    for s in stages:
        if s.get("mode", "CC") != "CC":
            raise ValueError("Only CC stages are supported in YAML profiles for now.")
        soc_from = float(s["soc_from"])
        soc_to = float(s["soc_to"])
        c_rate = float(s["c_rate"])
        duration_s = _soc_delta_to_seconds(soc_to - soc_from, c_rate)
        steps.append(f"Charge at {c_rate}C for {duration_s:.0f} seconds")
    # Optional cool-down rest
    for r in yaml_profile.get("rests", []):
        if r.get("after") == "charge":
            steps.append(f"Rest for {float(r['duration_min']):g} minutes")
    return steps


def list_available_profiles(config_dir: Optional[Path] = None) -> Dict[str, str]:
    """Return mapping profile_key -> human name."""
    out: Dict[str, str] = {}
    out.update({k: k for k in default_profiles().keys()})
    for key, data in load_literature_profiles(config_dir).items():
        out[key] = data.get("name", key)
    return out


def run_simulation(profile_name: str, out_dir: Path, config_dir: Optional[Path] = None) -> Path:
    """Run one simulation and save a compact metrics JSON."""
    _require_pybamm()
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = default_profiles()
    lit = load_literature_profiles(config_dir)

    if profile_name in profiles:
        steps = profiles[profile_name]
        meta: Optional[LitProfileMeta] = None
    elif profile_name in lit:
        steps = build_steps_from_lit_yaml(lit[profile_name])
        meta = LitProfileMeta(
            name=lit[profile_name].get("name", profile_name),
            source=lit[profile_name].get("source", ""),
            chemistry=lit[profile_name].get("chemistry", ""),
            temperature_C=float(lit[profile_name].get("temperature_C", 25)),
            soc_from=float(lit[profile_name].get("soc_window", {}).get("from", 0.0)),
            soc_to=float(lit[profile_name].get("soc_window", {}).get("to", 0.8)),
        )
    else:
        avail = list(list_available_profiles(config_dir).keys())
        resolved_dir = resolve_profiles_dir(config_dir)
        hint = f" Profiles dir: {resolved_dir}" if resolved_dir is not None else " Profiles dir: (not found)"
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {avail}.{hint}")

    experiment = pybamm.Experiment(steps)

    # DFN + a standard parameter set is a reasonable default for a computational fast-charge proxy study.
    model = pybamm.lithium_ion.DFN()
    param = pybamm.ParameterValues("Chen2020")

    # Make the initial SOC explicit so that the first CC step isn't skipped as infeasible.
    # (Some parameter sets default to a high SOC.)
    try:
        param.update({"Initial State of Charge": 0.0})
    except Exception:
        pass
    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)

    t0 = time.time()
    sol = sim.solve()
    elapsed_s = time.time() - t0

    def get(name: str):
        try:
            return sol[name].entries
        except KeyError:
            return None

    t_s = sol["Time [s]"].entries
    voltage = get("Terminal voltage [V]")

    metrics = {
        "profile": profile_name,
        "elapsed_s": float(elapsed_s),
        "steps": steps,
        "t_s": t_s.tolist(),
        "voltage_V": (voltage if voltage is not None else np.array([])).tolist(),
    }
    if meta is not None:
        metrics["literature_meta"] = {
            "name": meta.name,
            "source": meta.source,
            "chemistry": meta.chemistry,
            "temperature_C": meta.temperature_C,
            "soc_from": meta.soc_from,
            "soc_to": meta.soc_to,
        }

    # Optional proxy outputs (may not exist in all configs)
    overpot = get("X-averaged reaction overpotential [V]")
    if overpot is not None:
        metrics["overpotential_V"] = overpot.tolist()

    out_json = out_dir / f"{profile_name}_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json
