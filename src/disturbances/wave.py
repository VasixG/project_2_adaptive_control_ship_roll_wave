from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.types import Array, as_float_array


@dataclass(frozen=True)
class WaveDisturbanceParameters:
    beam: float = 2.8
    mu_wave: float = 1.0
    n_wave: int = 30
    seed: int = 12
    amp_min: float = 0.10
    amp_span: float = 0.08
    omega_min: float = 0.35
    omega_max: float = 3.50
    wave_number_scale: float = 0.65
    theta_star_radius_fraction: float = 0.85


class WaveMatchedDisturbance:
    def __init__(self, params: WaveDisturbanceParameters, r_theta: float):
        self.params = params
        self.r_theta = float(r_theta)

        rng = np.random.default_rng(params.seed)
        self.wave_amps = params.amp_min + params.amp_span * rng.random(params.n_wave)
        self.omegas = np.linspace(params.omega_min, params.omega_max, params.n_wave)
        self.wave_phases = 2.0 * np.pi * rng.random(params.n_wave)
        self.wave_numbers = params.wave_number_scale * self.omegas
        self.theta_star = self._build_theta_star()

        norm_theta_star = np.linalg.norm(self.theta_star)
        max_norm = params.theta_star_radius_fraction * self.r_theta
        if norm_theta_star > max_norm:
            scale = max_norm / norm_theta_star
            self.wave_amps *= scale
            self.theta_star *= scale

    @property
    def n_param(self) -> int:
        return 2 * self.params.n_wave

    def _build_theta_star(self) -> Array:
        theta_star = np.zeros(self.n_param, dtype=float)
        for i, (amp, number, phase) in enumerate(
            zip(self.wave_amps, self.wave_numbers, self.wave_phases)
        ):
            harmonic_gain = 2.0 * self.params.mu_wave * amp * np.sin(
                number * self.params.beam / 2.0
            )
            theta_star[2 * i] = harmonic_gain * np.sin(phase)
            theta_star[2 * i + 1] = harmonic_gain * np.cos(phase)
        return theta_star

    def phi(self, t: float) -> Array:
        values: list[float] = []
        for omega in self.omegas:
            values.append(np.sin(omega * t))
            values.append(np.cos(omega * t))
        return np.asarray(values, dtype=float)

    def eta(self, y: float, t: float) -> float:
        total = 0.0
        for amp, number, omega, phase in zip(
            self.wave_amps, self.wave_numbers, self.omegas, self.wave_phases
        ):
            total += amp * np.sin(number * y - omega * t + phase)
        return float(total)

    def true_moment_from_wave(self, t: float) -> float:
        eta_right = self.eta(+self.params.beam / 2.0, t)
        eta_left = self.eta(-self.params.beam / 2.0, t)
        return float(self.params.mu_wave * (eta_right - eta_left))

    def true_moment(self, t: float) -> float:
        return float(self.phi(t) @ self.theta_star)

    def estimate(self, theta_hat: Array, t: float) -> float:
        theta_hat = as_float_array(theta_hat)
        return float(self.phi(t) @ theta_hat)

