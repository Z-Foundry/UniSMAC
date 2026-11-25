"""Pure terrain-grid helpers: cell indexing, A* pathfinding, line-of-sight, and projection.

Grid layout matches `MapInfo`: `grid[iy, ix]` covers world coordinates `[ix, ix+1) x [iy, iy+1)`,
where True marks ground-passable cells. Cells are tuples `(ix, iy)`. Passability checks unit
centers only without shrinking by unit radius.

All functions here are pure (no `Sim` state), so pathfinding queries can be cached by
`(map, start_cell, goal_cell)`. Geometry uses Python float (float64) for cross-backend parity.
"""

from __future__ import annotations

import heapq
import math
from typing import TypeAlias

import numpy as np

SQRT2 = math.sqrt(2.0)
# Margin kept inside cell bounds when clamping (power of two, exact in float32).
CELL_EPS = 1.0 / 256.0

Cell: TypeAlias = tuple[int, int]

# 8-connected neighbors: (dx, dy, cost). Expansion order is fixed by the heap's total order.
_NEIGHBORS: tuple[tuple[int, int, float], ...] = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, SQRT2),
    (1, -1, SQRT2),
    (-1, 1, SQRT2),
    (-1, -1, SQRT2),
)


def cell_of(grid: np.ndarray, x: float, y: float) -> Cell:
    """Return the grid cell containing `(x, y)`, clamped to grid bounds."""
    h, w = grid.shape
    return (min(max(math.floor(x), 0), w - 1), min(max(math.floor(y), 0), h - 1))


def passable(grid: np.ndarray, ix: int, iy: int) -> bool:
    """Return True if `(ix, iy)` lies inside the grid and is passable."""
    h, w = grid.shape
    return 0 <= ix < w and 0 <= iy < h and bool(grid[iy, ix])


def cell_center(cell: Cell) -> tuple[float, float]:
    return cell[0] + 0.5, cell[1] + 0.5


def clamp_into_cell(x: float, y: float, cell: Cell) -> tuple[float, float]:
    """Clamp `(x, y)` inside `cell` with `CELL_EPS` margin from its edges."""
    ix, iy = cell
    return (
        min(max(x, ix + CELL_EPS), ix + 1 - CELL_EPS),
        min(max(y, iy + CELL_EPS), iy + 1 - CELL_EPS),
    )


def _octile(ax: int, ay: int, bx: int, by: int) -> float:
    dx, dy = abs(ax - bx), abs(ay - by)
    return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)


def find_path(grid: np.ndarray, start: Cell, goal: Cell) -> tuple[Cell, ...]:
    """8-connected, no-corner-cutting A* returning cells from `start` to destination (inclusive).

    * Diagonal steps require both adjacent orthogonal cells to be passable (no corner cutting);
      costs are 1 (orthogonal) and sqrt(2) (diagonal), with octile heuristic.
    * `start` need not be passable (units may be pushed onto edges), while `goal` must be.
    * If `goal` is unreachable (impassable or disconnected), falls back to the cell in `start`'s
      connected component closest to `goal` by Euclidean center distance, breaking ties by lower
      path cost then `(iy, ix)`.
    * Heap pops in total order `(f, h, iy, ix)` for deterministic cross-backend parity.
    """
    gx, gy = goal
    goal_ok = passable(grid, gx, gy)
    cost: dict[Cell, float] = {start: 0.0}
    parent: dict[Cell, Cell] = {}
    closed: set[Cell] = set()
    h0 = _octile(*start, gx, gy)
    heap: list[tuple[float, float, int, int]] = [(h0, h0, start[1], start[0])]
    best: tuple[int, float, int, int] | None = None  # (dist^2, cost, iy, ix)
    end = start
    while heap:
        _, _, y, x = heapq.heappop(heap)
        cell = (x, y)
        if cell in closed:
            continue
        closed.add(cell)
        g = cost[cell]
        if goal_ok and cell == goal:
            end = cell
            break
        key = ((x - gx) ** 2 + (y - gy) ** 2, g, y, x)
        if best is None or key < best:
            best, end = key, cell
        for dx, dy, step in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not passable(grid, nx, ny) or (nx, ny) in closed:
                continue
            if dx and dy and not (passable(grid, nx, y) and passable(grid, x, ny)):
                continue  # no corner cutting
            ng = g + step
            if ng < cost.get((nx, ny), math.inf):
                cost[(nx, ny)] = ng
                parent[(nx, ny)] = cell
                h = _octile(nx, ny, gx, gy)
                heapq.heappush(heap, (ng + h, h, ny, nx))
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return tuple(reversed(path))


def line_of_sight(grid: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> bool:
    """Return True if all cells traversed by `(x0, y0) -> (x1, y1)` (excluding start) are passable.

    Uses Amanatides-Woo grid traversal; exact grid-corner crossings require both adjacent
    orthogonal cells to be passable, matching A*'s no-corner-cutting rule. Checks unit center
    only without considering unit radius.
    """
    ix, iy = math.floor(x0), math.floor(y0)
    ex, ey = math.floor(x1), math.floor(y1)
    dx, dy = x1 - x0, y1 - y0
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    tdx = abs(1.0 / dx) if dx != 0 else math.inf
    tdy = abs(1.0 / dy) if dy != 0 else math.inf
    tmx = ((ix + 1 - x0) if dx > 0 else (x0 - ix)) * tdx if dx != 0 else math.inf
    tmy = ((iy + 1 - y0) if dy > 0 else (y0 - iy)) * tdy if dy != 0 else math.inf
    budget = abs(ex - ix) + abs(ey - iy)  # step cap against floating-point drift
    while (ix, iy) != (ex, ey) and budget > 0:
        if tmx < tmy:
            ix, tmx, budget = ix + sx, tmx + tdx, budget - 1
        elif tmy < tmx:
            iy, tmy, budget = iy + sy, tmy + tdy, budget - 1
        else:  # exact grid-corner crossing
            if not (passable(grid, ix + sx, iy) and passable(grid, ix, iy + sy)):
                return False
            ix, iy, budget = ix + sx, iy + sy, budget - 2
            tmx, tmy = tmx + tdx, tmy + tdy
        if not passable(grid, ix, iy):
            return False
    return (ix, iy) == (ex, ey) or passable(grid, ex, ey)


def project_to_passable(grid: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Return `(x, y)` unchanged if its cell is passable; otherwise project to the nearest
    passable cell (minimum point-to-rect distance, ties broken by `(iy, ix)`) and clamp inside.
    Returns `(x, y)` unchanged if the entire grid is impassable.
    """
    if passable(grid, *cell_of(grid, x, y)):
        return x, y
    iy, ix = np.nonzero(grid)
    if ix.size == 0:
        return x, y
    ddx = np.maximum(np.maximum(ix - x, x - (ix + 1)), 0.0)
    ddy = np.maximum(np.maximum(iy - y, y - (iy + 1)), 0.0)
    k = int(np.argmin(ddx * ddx + ddy * ddy))  # np.nonzero is row-major (iy, ix), first min wins
    return clamp_into_cell(x, y, (int(ix[k]), int(iy[k])))


__all__ = [
    'CELL_EPS',
    'Cell',
    'cell_center',
    'cell_of',
    'clamp_into_cell',
    'find_path',
    'line_of_sight',
    'passable',
    'project_to_passable',
]
