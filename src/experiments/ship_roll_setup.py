from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controllers.ship_roll import (
    AdaptiveLocalOnlyController,
    ControllerParameters,
    HybridAdaptiveController,
    HybridNominalController,
    LocalLinearBaselineController,
    ShipRollControlComponents,
    ShipRollController,
)
from disturbances.wave import WaveMatchedDisturbance
from experiments.config import ExperimentConfig, load_experiment_config
from lyapunov.quadratic import (
    QuadraticLyapunovData,
    QuadraticLyapunovFunction,
    build_local_lyapunov_data,
)
from simulation.simulator import ClosedLoopSimulator, SimulationResult
from systems.ship_roll import ShipRollSystem


@dataclass(frozen=True)
class AdaptiveDesignLevels:
    a_loc: float
    eta_margin: float
    a_bar: float
    c_theta: float
    gamma_eta: float
    gamma: float
    r_gamma: float


@dataclass(frozen=True)
class ShipRollExperiment:
    config: ExperimentConfig
    system: ShipRollSystem
    disturbance: WaveMatchedDisturbance
    lyapunov_data: QuadraticLyapunovData
    lyapunov: QuadraticLyapunovFunction
    levels: AdaptiveDesignLevels
    components: ShipRollControlComponents
    controllers: dict[str, ShipRollController]
    simulator: ClosedLoopSimulator


def build_design_levels(config: ExperimentConfig) -> AdaptiveDesignLevels:
    a_bar = config.a_loc - config.eta_margin
    if a_bar <= 0.0:
        raise ValueError("a_bar must be positive.")

    c_theta = 2.0 * config.r_theta**2
    gamma_eta = config.gamma_eta_factor * c_theta / a_bar
    gamma = config.gamma_factor * gamma_eta
    r_gamma = a_bar - c_theta / gamma
    if r_gamma <= 0.0:
        raise ValueError("r_gamma must be positive.")

    return AdaptiveDesignLevels(
        a_loc=config.a_loc,
        eta_margin=config.eta_margin,
        a_bar=a_bar,
        c_theta=c_theta,
        gamma_eta=gamma_eta,
        gamma=gamma,
        r_gamma=r_gamma,
    )


def build_experiment(config: ExperimentConfig | None = None) -> ShipRollExperiment:
    if config is None:
        config = load_experiment_config()

    system = ShipRollSystem(config.system)
    disturbance = WaveMatchedDisturbance(config.wave, config.r_theta)
    lyapunov_data = build_local_lyapunov_data(system, config.K)
    lyapunov = QuadraticLyapunovFunction(lyapunov_data.P)
    levels = build_design_levels(config)

    controller_params = ControllerParameters(
        K=lyapunov_data.K,
        r_gamma=levels.r_gamma,
        gamma=levels.gamma,
        r_theta=config.r_theta,
        outer_g_min=config.outer_g_min,
    )
    components = ShipRollControlComponents(
        system=system,
        lyapunov=lyapunov,
        disturbance=disturbance,
        params=controller_params,
    )
    controllers: dict[str, ShipRollController] = {
        "hybrid_adaptive": HybridAdaptiveController(components),
        "local_linear": LocalLinearBaselineController(components),
        "adaptive_local_only": AdaptiveLocalOnlyController(components),
        "hybrid_nominal": HybridNominalController(components),
    }
    simulator = ClosedLoopSimulator(
        system=system,
        lyapunov=lyapunov,
        disturbance=disturbance,
        config=config.simulation,
        gamma=levels.gamma,
        r_theta=config.r_theta,
    )

    return ShipRollExperiment(
        config=config,
        system=system,
        disturbance=disturbance,
        lyapunov_data=lyapunov_data,
        lyapunov=lyapunov,
        levels=levels,
        components=components,
        controllers=controllers,
        simulator=simulator,
    )


def run_controller(experiment: ShipRollExperiment, controller_name: str) -> SimulationResult:
    return experiment.simulator.simulate(
        experiment.controllers[controller_name],
        x0=experiment.config.x0,
    )


def run_all_controllers(experiment: ShipRollExperiment) -> dict[str, SimulationResult]:
    return {
        name: run_controller(experiment, name)
        for name in [
            "hybrid_adaptive",
            "local_linear",
            "adaptive_local_only",
            "hybrid_nominal",
        ]
    }


def local_decay_diagnostic(experiment: ShipRollExperiment, samples_radius: int = 160, samples_angle: int = 360) -> dict[str, float]:
    P = experiment.lyapunov.P
    values, vectors = np.linalg.eigh(P)
    transform = vectors @ np.diag(1.0 / np.sqrt(values))
    max_ratio = -np.inf
    nonnegative_count = 0

    for radius in np.linspace(1e-5, np.sqrt(experiment.levels.a_loc), samples_radius):
        for angle in np.linspace(0.0, 2.0 * np.pi, samples_angle):
            unit = np.array([np.cos(angle), np.sin(angle)], dtype=float)
            x = transform @ (radius * unit)
            w = experiment.lyapunov.value(x)
            u = float((experiment.lyapunov_data.K @ x)[0])
            closed_loop = experiment.system.dynamics_with_disturbance(x, u, 0.0)
            dot_w = float(experiment.lyapunov.grad(x) @ closed_loop)
            ratio = dot_w / w
            max_ratio = max(max_ratio, ratio)
            if dot_w >= 0.0:
                nonnegative_count += 1

    return {
        "max_dot_w_over_w": float(max_ratio),
        "estimated_c_loc_lower_bound": float(-max_ratio),
        "nonnegative_samples": float(nonnegative_count),
    }
