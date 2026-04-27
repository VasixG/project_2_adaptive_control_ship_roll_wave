from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptation.projection import parameter_energy, project_to_ball
from controllers.ship_roll import ShipRollController
from core.types import Array, as_float_array
from disturbances.wave import WaveMatchedDisturbance
from lyapunov.quadratic import QuadraticLyapunovFunction
from simulation.rk4 import rk4_step
from systems.ship_roll import ShipRollSystem


@dataclass(frozen=True)
class SimulationConfig:
    t0: float = 0.0
    tf: float = 45.0
    dt: float = 0.005
    state_max_norm: float | None = 50.0


@dataclass(frozen=True)
class SimulationResult:
    controller_name: str
    controller_display_name: str
    t: Array
    x: Array
    theta_hat: Array
    u: Array
    W: Array
    V_gamma: Array
    parameter_energy: Array
    d_true: Array
    d_wave: Array
    d_hat: Array
    disturbance_error: Array
    theta_hat_norm: Array
    theta_tilde_norm: Array
    input_gradient: Array
    mode: list[str]
    alive: bool
    switch_time: float | None
    switch_index: int | None


class ClosedLoopSimulator:
    def __init__(
        self,
        system: ShipRollSystem,
        lyapunov: QuadraticLyapunovFunction,
        disturbance: WaveMatchedDisturbance,
        config: SimulationConfig,
        gamma: float,
        r_theta: float,
    ):
        self.system = system
        self.lyapunov = lyapunov
        self.disturbance = disturbance
        self.config = config
        self.gamma = float(gamma)
        self.r_theta = float(r_theta)
        self.B = system.G(np.zeros(system.state_dim, dtype=float))

    def simulate(
        self,
        controller: ShipRollController,
        x0: Array,
        theta_hat0: Array | None = None,
    ) -> SimulationResult:
        x0 = as_float_array(x0)
        if theta_hat0 is None:
            theta_hat0 = np.zeros(self.disturbance.n_param, dtype=float)
        theta_hat0 = project_to_ball(theta_hat0, self.r_theta)

        t0 = self.config.t0
        tf = self.config.tf
        dt = self.config.dt
        n_steps = int(np.floor((tf - t0) / dt)) + 1
        t = np.linspace(t0, t0 + dt * (n_steps - 1), n_steps)

        z = np.zeros((n_steps, 2 + self.disturbance.n_param), dtype=float)
        z[0, :2] = x0
        z[0, 2:] = theta_hat0

        u_hist = np.full(n_steps, np.nan, dtype=float)
        W_hist = np.full(n_steps, np.nan, dtype=float)
        V_hist = np.full(n_steps, np.nan, dtype=float)
        param_energy_hist = np.full(n_steps, np.nan, dtype=float)
        d_true_hist = np.full(n_steps, np.nan, dtype=float)
        d_wave_hist = np.full(n_steps, np.nan, dtype=float)
        d_hat_hist = np.full(n_steps, np.nan, dtype=float)
        error_hist = np.full(n_steps, np.nan, dtype=float)
        theta_hat_norm_hist = np.full(n_steps, np.nan, dtype=float)
        theta_tilde_norm_hist = np.full(n_steps, np.nan, dtype=float)
        input_gradient_hist = np.full(n_steps, np.nan, dtype=float)
        mode_hist: list[str] = []

        mode = controller.initial_mode(x0, self.lyapunov)
        switched = mode == "local"
        switch_time = 0.0 if switched else None
        switch_index = 0 if switched else None
        alive = True

        def rhs_factory(mode_value: str):
            def rhs(t_now: float, z_now: Array) -> Array:
                x_now = z_now[:2]
                theta_hat_now = z_now[2:]
                u_now, theta_dot_now = controller.control_and_adaptation(
                    x_now,
                    theta_hat_now,
                    t_now,
                    mode_value,
                )
                d_now = self.disturbance.true_moment(t_now)
                x_dot_now = self.system.dynamics_with_disturbance(x_now, u_now, d_now)
                return np.concatenate([x_dot_now, theta_dot_now]).astype(float)

            return rhs

        for k in range(n_steps):
            xk = z[k, :2]
            theta_hat_k = z[k, 2:]
            tk = t[k]

            if not np.all(np.isfinite(z[k])):
                alive = False
                break

            if self.config.state_max_norm is not None:
                if np.linalg.norm(xk) > self.config.state_max_norm:
                    alive = False
                    break

            mode, switched_now = controller.next_mode(xk, self.lyapunov, mode, switched)
            if switched_now and not switched:
                switch_time = tk
                switch_index = k
            switched = switched or switched_now
            mode_hist.append(mode)

            uk, _ = controller.control_and_adaptation(xk, theta_hat_k, tk, mode)
            u_hist[k] = uk
            W_hist[k] = self.lyapunov.value(xk)
            param_energy_hist[k] = parameter_energy(theta_hat_k, self.disturbance.theta_star, self.gamma)
            V_hist[k] = W_hist[k] + param_energy_hist[k]
            d_true_hist[k] = self.disturbance.true_moment(tk)
            d_wave_hist[k] = self.disturbance.true_moment_from_wave(tk)
            d_hat_hist[k] = self.disturbance.estimate(theta_hat_k, tk)
            error_hist[k] = d_true_hist[k] - d_hat_hist[k]
            theta_hat_norm_hist[k] = np.linalg.norm(theta_hat_k)
            theta_tilde_norm_hist[k] = np.linalg.norm(theta_hat_k - self.disturbance.theta_star)
            input_gradient_hist[k] = self.lyapunov.input_gradient(xk, self.B)

            if k < n_steps - 1:
                z[k + 1] = rk4_step(rhs_factory(mode), tk, z[k], dt)
                z[k + 1, 2:] = project_to_ball(z[k + 1, 2:], self.r_theta)

        if len(mode_hist) < n_steps:
            mode_hist.extend(["terminated"] * (n_steps - len(mode_hist)))

        return SimulationResult(
            controller_name=controller.name,
            controller_display_name=controller.display_name,
            t=t,
            x=z[:, :2],
            theta_hat=z[:, 2:],
            u=u_hist,
            W=W_hist,
            V_gamma=V_hist,
            parameter_energy=param_energy_hist,
            d_true=d_true_hist,
            d_wave=d_wave_hist,
            d_hat=d_hat_hist,
            disturbance_error=error_hist,
            theta_hat_norm=theta_hat_norm_hist,
            theta_tilde_norm=theta_tilde_norm_hist,
            input_gradient=input_gradient_hist,
            mode=mode_hist,
            alive=alive,
            switch_time=switch_time,
            switch_index=switch_index,
        )

