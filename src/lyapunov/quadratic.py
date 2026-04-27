from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigvals, solve_continuous_lyapunov

from core.interfaces import ControlAffineSystem
from core.types import Array, as_float_array


@dataclass(frozen=True)
class QuadraticLyapunovData:
    K: Array
    P: Array
    A: Array
    B: Array
    Acl: Array
    eig_Acl: Array
    eig_P: Array


def build_local_lyapunov_data(system: ControlAffineSystem, K: Array) -> QuadraticLyapunovData:
    A, B = system.linearization()
    K = np.asarray(K, dtype=float).reshape(system.control_dim, system.state_dim)
    Acl = A + B @ K
    eig_Acl = eigvals(Acl)

    if np.max(np.real(eig_Acl)) >= 0.0:
        raise ValueError("A + B K is not Hurwitz.")

    P = solve_continuous_lyapunov(Acl.T, -np.eye(system.state_dim))
    P = 0.5 * (P + P.T)
    eig_P = np.linalg.eigvalsh(P)

    if np.min(eig_P) <= 0.0:
        raise ValueError("Computed P is not positive definite.")

    return QuadraticLyapunovData(
        K=K,
        P=P,
        A=A,
        B=B,
        Acl=Acl,
        eig_Acl=eig_Acl,
        eig_P=eig_P,
    )


class QuadraticLyapunovFunction:
    def __init__(self, P: Array):
        self.P = np.asarray(P, dtype=float)
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("P must be a square matrix.")
        if np.min(np.linalg.eigvalsh(self.P)) <= 0.0:
            raise ValueError("P must be positive definite.")

    def value(self, x: Array) -> float:
        x = as_float_array(x)
        return float(x.T @ self.P @ x)

    def grad(self, x: Array) -> Array:
        x = as_float_array(x)
        return 2.0 * (self.P @ x)

    def input_gradient(self, x: Array, B: Array) -> float:
        return float(self.grad(x) @ B[:, 0])

