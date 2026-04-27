from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def as_float_array(x: Array | list[float] | tuple[float, ...]) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Vectors must be one-dimensional.")
    return arr

