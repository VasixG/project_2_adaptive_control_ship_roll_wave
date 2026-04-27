from __future__ import annotations

import numpy as np

from core.types import Array, as_float_array


def project_to_ball(theta: Array, radius: float) -> Array:
    theta = as_float_array(theta)
    norm_theta = np.linalg.norm(theta)
    if norm_theta <= radius:
        return theta.copy()
    return radius * theta / norm_theta


def tangent_cone_projection(theta: Array, theta_dot: Array, radius: float) -> Array:
    theta = as_float_array(theta)
    theta_dot = as_float_array(theta_dot)
    norm_theta = np.linalg.norm(theta)

    if norm_theta < radius or norm_theta <= 1e-12:
        return theta_dot

    outward = theta / norm_theta
    outward_component = float(theta_dot @ outward)
    if outward_component <= 0.0:
        return theta_dot
    return theta_dot - outward_component * outward


def parameter_energy(theta_hat: Array, theta_star: Array, gamma: float) -> float:
    theta_tilde = as_float_array(theta_hat) - as_float_array(theta_star)
    return 0.5 * float(theta_tilde @ theta_tilde) / gamma

