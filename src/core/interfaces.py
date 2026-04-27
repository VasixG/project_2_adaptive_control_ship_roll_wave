from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Array


class ControlAffineSystem(ABC):
    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def control_dim(self) -> int:
        pass

    @abstractmethod
    def f(self, x: Array) -> Array:
        pass

    @abstractmethod
    def G(self, x: Array) -> Array:
        pass

    @abstractmethod
    def linearization(self) -> tuple[Array, Array]:
        pass

    def dynamics(self, x: Array, u: Array) -> Array:
        return self.f(x) + self.G(x) @ u

