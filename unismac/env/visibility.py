"""Visibility rules: sight circle (upstream SMAC) and conic FOV (SMACv2 `conic_fov`).

Distances use `np.hypot`, which may differ from upstream `math.hypot` by 1 ULP in float64 (~0.6%
of random float32 coordinate pairs), identical once cast to float32; thresholds can differ only if
a distance lands within 1 ULP of the boundary.
"""

from __future__ import annotations

import numpy as np

__all__ = ['FOV_EPS', 'circle_visible', 'cone_visible', 'distance']

FOV_EPS = 1e-7  # Upstream EPS in is_position_in_cone


def distance(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    """Compute `hypot(bx - ax, by - ay)` with NumPy broadcasting."""
    return np.hypot(bx - ax, by - ay)


def circle_visible(dist: np.ndarray, target_health: np.ndarray, sight_range: float) -> np.ndarray:
    """Upstream circle visibility: `dist < sight_range and health > 0`."""
    return (dist < sight_range) & (target_health > 0)


def cone_visible(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    fov_x: np.ndarray,
    fov_y: np.ndarray,
    unit_range: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Element-wise port of SMACv2 `is_position_in_cone` (preserving its x-clamping sign quirk).

    Checks `dist <= range` and `|arctan(dy / dx) - arctan(fov_y / fov_x')| < angle / 2`, where
    `dx` is clamped away from zero by +/-EPS using its own sign and `fov_x'` is clamped using the
    sign of `dx` (as in upstream). Does not check whether the target is alive.
    """
    dist = distance(ax, ay, bx, by)
    dx = bx - ax
    pos = dx > 0
    dx = np.where(pos, np.maximum(dx, FOV_EPS), np.minimum(dx, -FOV_EPS))
    obj = np.arctan((by - ay) / dx)
    fx = np.where(pos, np.maximum(fov_x, FOV_EPS), np.minimum(fov_x, -FOV_EPS))
    fov_angle = np.arctan(fov_y / fx)
    return (dist <= unit_range) & (np.abs(obj - fov_angle) < angle / 2)
