from __future__ import annotations

from collections.abc import Callable

from core.types import Array


def rk4_step(rhs: Callable[[float, Array], Array], t: float, z: Array, dt: float) -> Array:
    k1 = rhs(t, z)
    k2 = rhs(t + 0.5 * dt, z + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, z + 0.5 * dt * k2)
    k4 = rhs(t + dt, z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

