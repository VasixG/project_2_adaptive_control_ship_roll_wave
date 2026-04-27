from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptation.projection import tangent_cone_projection
from core.types import Array, as_float_array
from disturbances.wave import WaveMatchedDisturbance
from lyapunov.quadratic import QuadraticLyapunovFunction
from systems.ship_roll import ShipRollSystem

MODE_OUTER = "outer"
MODE_LOCAL = "local"
MODE_LOCAL_ONLY = "local-only"
MODE_LINEAR = "linear"


@dataclass(frozen=True)
class ControllerParameters:
    K: Array
    r_gamma: float
    gamma: float
    r_theta: float
    outer_g_min: float = 1e-6


class ShipRollController:
    name = "controller"
    display_name = "controller"

    def initial_mode(self, x: Array, lyapunov: QuadraticLyapunovFunction) -> str:
        _ = x
        _ = lyapunov
        return MODE_LOCAL

    def next_mode(
        self,
        x: Array,
        lyapunov: QuadraticLyapunovFunction,
        current_mode: str,
        switched: bool,
    ) -> tuple[str, bool]:
        _ = x
        _ = lyapunov
        _ = current_mode
        return MODE_LOCAL, switched

    def control_and_adaptation(self, x: Array, theta_hat: Array, t: float, mode: str) -> tuple[float, Array]:
        raise NotImplementedError


class ShipRollControlComponents:
    def __init__(
        self,
        system: ShipRollSystem,
        lyapunov: QuadraticLyapunovFunction,
        disturbance: WaveMatchedDisturbance,
        params: ControllerParameters,
    ):
        self.system = system
        self.lyapunov = lyapunov
        self.disturbance = disturbance
        self.params = params
        self.B = system.G(np.zeros(system.state_dim, dtype=float))
        self.K = np.asarray(params.K, dtype=float).reshape(1, 2)

    def local_linear_u(self, x: Array) -> float:
        x = as_float_array(x)
        return float((self.K @ x)[0])

    def adaptation_unprojected(self, x: Array, t: float) -> Array:
        g = self.lyapunov.input_gradient(x, self.B)
        return self.params.gamma * g * self.disturbance.phi(t)

    def adaptation_projected(self, x: Array, theta_hat: Array, t: float) -> Array:
        raw = self.adaptation_unprojected(x, t)
        return tangent_cone_projection(theta_hat, raw, self.params.r_theta)

    def local_adaptive_u(self, x: Array, theta_hat: Array, t: float) -> float:
        return self.local_linear_u(x) - self.disturbance.estimate(theta_hat, t)

    def outer_nominal_u(self, x: Array) -> float:
        x = as_float_array(x)
        grad = self.lyapunov.grad(x)
        g = self.lyapunov.input_gradient(x, self.B)
        if abs(g) < self.params.outer_g_min:
            g = self.params.outer_g_min * np.sign(g if g != 0.0 else 1.0)
        numerator = self.lyapunov.value(x) + float(grad @ self.system.f(x))
        return float(-numerator / g)

    def outer_adaptive_u(self, x: Array, theta_hat: Array, t: float) -> float:
        return self.outer_nominal_u(x) - self.disturbance.estimate(theta_hat, t)


class HybridAdaptiveController(ShipRollController):
    name = "hybrid_adaptive"
    display_name = "Hybrid adaptive"

    def __init__(self, components: ShipRollControlComponents):
        self.c = components

    def initial_mode(self, x: Array, lyapunov: QuadraticLyapunovFunction) -> str:
        return MODE_LOCAL if lyapunov.value(x) <= self.c.params.r_gamma else MODE_OUTER

    def next_mode(
        self,
        x: Array,
        lyapunov: QuadraticLyapunovFunction,
        current_mode: str,
        switched: bool,
    ) -> tuple[str, bool]:
        if switched or current_mode == MODE_LOCAL:
            return MODE_LOCAL, True
        if lyapunov.value(x) <= self.c.params.r_gamma:
            return MODE_LOCAL, True
        return MODE_OUTER, False

    def control_and_adaptation(self, x: Array, theta_hat: Array, t: float, mode: str) -> tuple[float, Array]:
        if mode == MODE_OUTER:
            return self.c.outer_adaptive_u(x, theta_hat, t), self.c.adaptation_projected(x, theta_hat, t)
        return self.c.local_adaptive_u(x, theta_hat, t), self.c.adaptation_projected(x, theta_hat, t)


class AdaptiveLocalOnlyController(ShipRollController):
    name = "adaptive_local_only"
    display_name = "Adaptive local only"

    def __init__(self, components: ShipRollControlComponents):
        self.c = components

    def initial_mode(self, x: Array, lyapunov: QuadraticLyapunovFunction) -> str:
        _ = x
        _ = lyapunov
        return MODE_LOCAL_ONLY

    def next_mode(
        self,
        x: Array,
        lyapunov: QuadraticLyapunovFunction,
        current_mode: str,
        switched: bool,
    ) -> tuple[str, bool]:
        _ = x
        _ = lyapunov
        _ = current_mode
        return MODE_LOCAL_ONLY, switched

    def control_and_adaptation(self, x: Array, theta_hat: Array, t: float, mode: str) -> tuple[float, Array]:
        _ = mode
        return self.c.local_adaptive_u(x, theta_hat, t), self.c.adaptation_projected(x, theta_hat, t)


class LocalLinearBaselineController(ShipRollController):
    name = "local_linear"
    display_name = "Local linear, no adaptation"

    def __init__(self, components: ShipRollControlComponents):
        self.c = components

    def initial_mode(self, x: Array, lyapunov: QuadraticLyapunovFunction) -> str:
        _ = x
        _ = lyapunov
        return MODE_LINEAR

    def next_mode(
        self,
        x: Array,
        lyapunov: QuadraticLyapunovFunction,
        current_mode: str,
        switched: bool,
    ) -> tuple[str, bool]:
        _ = x
        _ = lyapunov
        _ = current_mode
        return MODE_LINEAR, switched

    def control_and_adaptation(self, x: Array, theta_hat: Array, t: float, mode: str) -> tuple[float, Array]:
        _ = theta_hat
        _ = t
        _ = mode
        return self.c.local_linear_u(x), np.zeros(self.c.disturbance.n_param, dtype=float)


class HybridNominalController(ShipRollController):
    name = "hybrid_nominal"
    display_name = "Hybrid without adaptation"

    def __init__(self, components: ShipRollControlComponents):
        self.c = components

    def initial_mode(self, x: Array, lyapunov: QuadraticLyapunovFunction) -> str:
        return MODE_LOCAL if lyapunov.value(x) <= self.c.params.r_gamma else MODE_OUTER

    def next_mode(
        self,
        x: Array,
        lyapunov: QuadraticLyapunovFunction,
        current_mode: str,
        switched: bool,
    ) -> tuple[str, bool]:
        if switched or current_mode == MODE_LOCAL:
            return MODE_LOCAL, True
        if lyapunov.value(x) <= self.c.params.r_gamma:
            return MODE_LOCAL, True
        return MODE_OUTER, False

    def control_and_adaptation(self, x: Array, theta_hat: Array, t: float, mode: str) -> tuple[float, Array]:
        _ = theta_hat
        _ = t
        if mode == MODE_OUTER:
            return self.c.outer_nominal_u(x), np.zeros(self.c.disturbance.n_param, dtype=float)
        return self.c.local_linear_u(x), np.zeros(self.c.disturbance.n_param, dtype=float)
