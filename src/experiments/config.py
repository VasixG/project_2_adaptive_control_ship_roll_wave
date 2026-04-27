from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.types import Array
from disturbances.wave import WaveDisturbanceParameters
from simulation.simulator import SimulationConfig
from systems.ship_roll import ShipRollParameters

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


@dataclass(frozen=True)
class ExperimentConfig:
    system: ShipRollParameters
    wave: WaveDisturbanceParameters
    simulation: SimulationConfig
    K: Array
    a_loc: float
    eta_margin: float
    r_theta: float
    gamma_eta_factor: float
    gamma_factor: float
    outer_g_min: float
    x0: Array


def load_experiment_config(path: Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    system = raw.get("system", {})
    controller = raw.get("controller", {})
    wave = raw.get("wave", {})
    simulation = raw.get("simulation", {})

    return ExperimentConfig(
        system=ShipRollParameters(
            c_damp=float(system.get("c_damp", 0.85)),
            k_restoring=float(system.get("k_restoring", 2.20)),
        ),
        wave=WaveDisturbanceParameters(
            beam=float(wave.get("beam", 2.8)),
            mu_wave=float(wave.get("mu_wave", 1.0)),
            n_wave=int(wave.get("n_wave", 30)),
            seed=int(wave.get("seed", 12)),
            amp_min=float(wave.get("amp_min", 0.10)),
            amp_span=float(wave.get("amp_span", 0.08)),
            omega_min=float(wave.get("omega_min", 0.35)),
            omega_max=float(wave.get("omega_max", 3.50)),
            wave_number_scale=float(wave.get("wave_number_scale", 0.65)),
            theta_star_radius_fraction=float(wave.get("theta_star_radius_fraction", 0.85)),
        ),
        simulation=SimulationConfig(
            t0=float(simulation.get("t0", 0.0)),
            tf=float(simulation.get("tf", 45.0)),
            dt=float(simulation.get("dt", 0.005)),
            state_max_norm=simulation.get("state_max_norm", 50.0),
        ),
        K=np.asarray(controller.get("K", [-3.8, -2.4]), dtype=float),
        a_loc=float(controller.get("a_loc", 0.20)),
        eta_margin=float(controller.get("eta_margin", 0.06)),
        r_theta=float(controller.get("R_theta", 0.65)),
        gamma_eta_factor=float(controller.get("gamma_eta_factor", 1.05)),
        gamma_factor=float(controller.get("gamma_factor", 1.25)),
        outer_g_min=float(controller.get("outer_g_min", 1e-6)),
        x0=np.asarray(simulation.get("x0", [0.95, 0.0]), dtype=float),
    )

