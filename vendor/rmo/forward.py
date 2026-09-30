from __future__ import annotations

import math


LIGHT_SPEED_KM_S = 299792.458


def uniform_slab_observer(
    electron_density: float,
    los_depth: float,
    response_G: float,
    los_velocity_km_s: float,
    line_rest_A: float,
    line_sigma_A: float,
) -> dict[str, float]:
    if min(electron_density, los_depth, response_G, line_rest_A, line_sigma_A) < 0.0:
        raise ValueError("INVALID_STATE")
    intensity = electron_density**2 * response_G * los_depth
    centroid = line_rest_A * (1.0 + los_velocity_km_s / LIGHT_SPEED_KM_S)
    return {
        "optically_thin_intensity": intensity,
        "centroid_A": centroid,
        "line_sigma_A": line_sigma_A,
    }


def standardized_residual(observed: float, predicted: float, sigma: float) -> float:
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("COVARIANCE_INVALID")
    return (observed - predicted) / sigma
