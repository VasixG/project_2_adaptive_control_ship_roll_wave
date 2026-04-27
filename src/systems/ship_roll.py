from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.interfaces import ControlAffineSystem
from core.types import Array, as_float_array


@dataclass(frozen=True)
class ShipRollParameters:
    c_damp: float = 0.85
    k_restoring: float = 2.20


class ShipRollSystem(ControlAffineSystem):
    def __init__(self, params: ShipRollParameters):
        self.params = params

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def control_dim(self) -> int:
        return 1

    def f(self, x: Array) -> Array:
        x = as_float_array(x)
        if x.shape[0] != 2:
            raise ValueError("ShipRollSystem expects state x = [theta, p].")

        theta, p = x
        c = self.params.c_damp
        k = self.params.k_restoring
        return np.array([p, -c * p - k * np.sin(theta)], dtype=float)

    def G(self, x: Array) -> Array:
        _ = x
        return np.array([[0.0], [1.0]], dtype=float)

    def B_vector(self) -> Array:
        return self.G(np.zeros(2, dtype=float))[:, 0]

    def linearization(self) -> tuple[Array, Array]:
        c = self.params.c_damp
        k = self.params.k_restoring
        A = np.array([[0.0, 1.0], [-k, -c]], dtype=float)
        B = np.array([[0.0], [1.0]], dtype=float)
        return A, B

    def dynamics_with_disturbance(self, x: Array, u: float, d: float) -> Array:
        return self.f(x) + self.B_vector() * (float(u) + float(d))

