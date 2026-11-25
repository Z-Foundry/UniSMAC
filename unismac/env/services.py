"""Numerical services for semantic components: shared hot loops written as small array helpers.

* `pairwise(ax, ay, bx, by)`: pairwise relative offsets and distances (`[B, A]` vs `[B, N]`);
* `grid_lookup(grid, xi, yi, outside)`: integer grid lookup with `outside` fallback;
* `ordered_sum(values, mask)`: index-ordered masked sum (matching sequential Python floats).

This module provides the NumPy reference implementation; the bitwise-identical Rust counterpart
lives in `zgame-smac` (`_zgame.env`). Semantic components receive a `Services` bundle (`PYTHON`
for `accel='python'`, Rust for `accel='services'`; see `unismac.env.accel`) while keeping
component logic in Python.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np

__all__ = ['PYTHON', 'Services', 'grid_lookup', 'ordered_sum', 'pairwise']


@cache
def _others(n: int) -> np.ndarray:
    """Return `int64[n, n-1]` where row `i` lists all indices other than `i` in ascending order."""
    idx = [[j for j in range(n) if j != i] for i in range(n)]
    return np.array(idx, np.int64).reshape(n, max(n - 1, 0))


def pairwise(
    ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray, *, exclude_self: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(dx, dy, dist)` (`float64[B, A, N]`) from observers `(ax, ay)` (`float64[B, A]`)
    to targets `(bx, by)` (`float64[B, N]`): `dx = bx - ax`, `dy = by - ay`,
    `dist = np.hypot(dx, dy)`.

    When `exclude_self=True`, targets are the observers themselves (`N = A`) and self entries are
    dropped, yielding `[B, A, A-1]` in ascending index order. Rust counterpart: zgame-smac
    `env::services::pairwise`.
    """
    if exclude_self:
        others = _others(ax.shape[1])
        tx, ty = bx[:, others], by[:, others]
    else:
        tx, ty = bx[:, None, :], by[:, None, :]
    dx = tx - ax[:, :, None]
    dy = ty - ay[:, :, None]
    return dx, dy, np.hypot(dx, dy)


def grid_lookup(grid: np.ndarray, xi: np.ndarray, yi: np.ndarray, outside: bool | float):
    """Return `np.where(in_bounds, grid[xi, yi], outside)` for `[X, Y]` grid and `int64` indices.

    Returns bool when `grid` and `outside` are bool, otherwise float64. Rust counterpart:
    zgame-smac `env::services::grid_lookup`.
    """
    gx, gy = grid.shape
    inb = (xi >= 0) & (xi < gx) & (yi >= 0) & (yi < gy)
    xc = np.clip(xi, 0, gx - 1)
    yc = np.clip(yi, 0, gy - 1)
    return np.where(inb, grid[xc, yc], outside)


def ordered_sum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return `float64[B]`: sequential index-order sum of `values` (`[B, N]`) where `mask` is True.

    Avoids `np.sum` pairwise summation, which can differ by a few ULPs from sequential addition.
    Rust counterpart: zgame-smac `env::services::ordered_sum`.
    """
    total = np.zeros(values.shape[0])
    for j in range(values.shape[1]):
        total = np.where(mask[:, j], total + values[:, j], total)
    return total


@dataclass(frozen=True)
class Services:
    """Bundle of numerical service functions (Python or Rust), matching module signatures."""

    name: str
    pairwise: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
    grid_lookup: Callable[..., np.ndarray]
    ordered_sum: Callable[[np.ndarray, np.ndarray], np.ndarray]


PYTHON = Services('python', pairwise, grid_lookup, ordered_sum)
