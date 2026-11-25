"""Terrain grids: upstream SMAC pathing / height grids and map dimensions, indexed as `[x, y]`.

`MapInfo` stores grids as `grid[iy, ix]` (row = y, unflipped; see `unismac.assets`). Upstream
indexing:

* Pathing (1 bpp): `reshape(map_x, map_y/8)`, unpack, transpose -> `pathing[x, y] = grid[y, x]`
  (valid only on square maps; upstream misaligns or goes out of bounds on non-square maps, while
  here we always use `grid[y, x]`).
* Height: `flip(transpose(reshape(map_x, map_y)), 1) / 255`, equal to `grid[H-1-y, x] / 255` on
  square maps (y-flipped; enabled by `quirks.terrain_height_flip`, square maps only). When
  disabled, uses `grid[y, x] / 255`.

All 23 classic SMAC maps are 32x32 with playable area `(2, 2, 30, 30)`.

Lookups use `services.grid_lookup` (returning the given out-of-bounds value), dispatched via `ops`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...contract import ContractError, MapInfo
from . import services

__all__ = ['TerrainGrids']

# Upstream get_surrounding_points offsets in units of move_amount; height appends self (0, 0).
_SURROUND = ((0, 2), (0, -2), (2, 0), (-2, 0), (1, 1), (-1, -1), (1, -1), (-1, 1))
# Movement directions for can_move: North, South, East, West.
_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))


@dataclass(frozen=True)
class TerrainGrids:
    map_id: str
    map_x: int
    map_y: int
    pathing: np.ndarray  # bool[map_x, map_y], indexed [x, y]
    height: np.ndarray  # float64[map_x, map_y], raw / 255
    max_distance_x: int  # Playable width
    max_distance_y: int
    ops: services.Services = field(default=services.PYTHON, compare=False, repr=False)

    @classmethod
    def from_map_info(
        cls, info: MapInfo, *, height_flip: bool, ops: services.Services = services.PYTHON
    ) -> TerrainGrids:
        grid_h = info.terrain_height.astype(np.float64)
        if height_flip:
            if info.width != info.height:
                raise ContractError(
                    f'{info.map_id} is not a square map; '
                    'quirks.terrain_height_flip only supports square maps'
                )
            height = grid_h[::-1, :].T / 255
        else:
            height = grid_h.T / 255
        x0, y0, x1, y1 = info.playable
        return cls(
            map_id=info.map_id,
            map_x=int(info.width),
            map_y=int(info.height),
            pathing=np.ascontiguousarray(info.pathing.T),
            height=np.ascontiguousarray(height),
            max_distance_x=int(x1 - x0),
            max_distance_y=int(y1 - y0),
            ops=ops,
        )

    def can_move(self, x: np.ndarray, y: np.ndarray, move_amount: float) -> np.ndarray:
        """Upstream `can_move`: `bool[..., 4]` (North, South, East, West) for `float64` x, y."""
        m = move_amount / 2
        out = []
        for dx, dy in _DIRS:
            xi = np.trunc(x + dx * m).astype(np.int64)
            yi = np.trunc(y + dy * m).astype(np.int64)
            out.append(self.ops.grid_lookup(self.pathing, xi, yi, False))
        return np.stack(out, axis=-1)

    def _surround(self, grid: np.ndarray, x, y, move_amount: float, with_self: bool):
        ma = int(move_amount)
        xi = np.trunc(x).astype(np.int64)
        yi = np.trunc(y).astype(np.int64)
        offsets = _SURROUND + (((0, 0),) if with_self else ())
        vals = [self.ops.grid_lookup(grid, xi + ox * ma, yi + oy * ma, 1.0) for ox, oy in offsets]
        return np.stack(vals, axis=-1)

    def surrounding_pathing(self, x: np.ndarray, y: np.ndarray, move_amount: float) -> np.ndarray:
        """Upstream `get_surrounding_pathing`: `float64[..., 8]`, out-of-bounds filled with 1."""
        return self._surround(self.pathing, x, y, move_amount, False)

    def surrounding_height(self, x: np.ndarray, y: np.ndarray, move_amount: float) -> np.ndarray:
        """Upstream `get_surrounding_height`: `float64[..., 9]` (last is self), OOB = 1."""
        return self._surround(self.height, x, y, move_amount, True)
