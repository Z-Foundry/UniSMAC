"""Stage 9 visibility: writes `world.visible[b, u, p]` (and `vision.true_sight` in `'sc2'` mode)."""

from __future__ import annotations

import numpy as np

from unismac.contract import MapInfo

from ..sim import Sim, cell_index

# Height byte delta between adjacent cliff levels (e.g., low ground 191 vs high ground 207).
LAYER_STEP = 16.0
_LEVELS: dict[tuple[str, str], np.ndarray] = {}

# `vision_levels='interp'` ramp-vision model: sample terrain_height at `x + INTERP_DX` via linear
# interpolation between cell centers along x. An observer >= OBSERVER_FRAC layers above the lower
# level counts as upper level, while a target must reach >= TARGET_FRAC layers to count as upper
# level; unit `u` sees `t` iff `t`'s level does not exceed `u`'s level.
INTERP_DX = 0.6
OBSERVER_FRAC = 4.0 / LAYER_STEP
TARGET_FRAC = 8.5 / LAYER_STEP


def full(sim: Sim) -> None:
    """No fog of war: alive units are visible to all players; dead/empty slots are invisible."""
    w = sim.world
    w.visible[:] = w.alive[:, :, None]


def levels(info: MapInfo) -> np.ndarray:
    """Return `int64[H, W]` cliff height layers. The lowest passable cell defines layer 0, with
    one layer per `LAYER_STEP`; cells at or above the midpoint between layers round up. Impassable
    border cells lower than layer 0 clamp to layer 0.
    """
    key = (info.map_id, info.sha256)
    lv = _LEVELS.get(key)
    if lv is None:
        h = info.terrain_height.astype(np.float64)
        base = h[info.pathing].min() if info.pathing.any() else h.min()
        lv = np.floor((h - base) / LAYER_STEP + 0.5).astype(np.int64)
        lv = _LEVELS[key] = np.maximum(lv, 0)
    return lv


def unit_levels(sim: Sim) -> np.ndarray:
    """Return `int64[B, U]` terrain height layer of each unit's cell (unused for dead/empty)."""
    w = sim.world
    out = np.zeros(w.x.shape, np.int64)
    maps: dict[str, tuple[MapInfo, list[int]]] = {}
    for b, info in enumerate(sim.env_map):
        if info is not None:
            maps.setdefault(info.map_id, (info, []))[1].append(b)
    for info, bs in maps.values():
        iy, ix = cell_index(info, w.x[bs], w.y[bs])
        out[bs] = levels(info)[iy, ix]
    return out


def _interp_height(info: MapInfo, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample terrain height at `(x + INTERP_DX, y)`: row `floor(y)`, linear interp in x."""
    h = info.terrain_height.astype(np.float64)
    with np.errstate(invalid='ignore'):
        iy = np.clip(np.nan_to_num(np.floor(y), nan=0.0), 0, info.height - 1).astype(np.int64)
        fx = np.nan_to_num(x + INTERP_DX - 0.5, nan=0.0)
    i0 = np.clip(np.floor(fx), 0, info.width - 1).astype(np.int64)
    i1 = np.minimum(i0 + 1, info.width - 1)
    t = np.clip(fx - i0, 0.0, 1.0)
    return h[iy, i0] * (1.0 - t) + h[iy, i1] * t


def interp_levels(sim: Sim) -> tuple[np.ndarray, np.ndarray]:
    """Return `(observer_levels, target_levels)` as `int64[B, U]` for `vision_levels='interp'`."""
    w = sim.world
    obs = np.zeros(w.x.shape, np.int64)
    tgt = np.zeros(w.x.shape, np.int64)
    for b, info in enumerate(sim.env_map):
        if info is None:
            continue
        hm = info.terrain_height.astype(np.float64)
        base = hm[info.pathing].min() if info.pathing.any() else hm.min()
        rel = (_interp_height(info, w.x[b], w.y[b]) - base) / LAYER_STEP
        obs[b] = np.maximum(np.floor(rel + 1.0 - OBSERVER_FRAC), 0)
        tgt[b] = np.maximum(np.floor(rel + 1.0 - TARGET_FRAC), 0)
    return obs, tgt


def sc2(sim: Sim) -> None:
    """SC2 fog-of-war visibility model (sets capability `fog=True`).

    * True sight: alive unit `u` of player `p` sees alive unit `t` if center distance <= `u`'s
      sight radius AND (`u` or `t` is flying, OR `t`'s height level <= `u`'s height level).
      Own units are always visible. When `vision_levels='interp'`, observer and target layers
      are computed separately via `interp_levels`.
    * Reveal: when an attack hits, the attacker is revealed to the victim's owner for
      `reveal_loops` game loops (see `on_hit`).
    * `world.visible` is the union of true sight and reveal; `vision.true_sight` stores true
      sight only (used by targeting).
    """
    w, v = sim.world, sim.vision
    alive = w.alive
    if sim.config.vision_levels == 'interp':
        lv_obs, lv_tgt = interp_levels(sim)
    else:
        lv_obs = lv_tgt = unit_levels(sim)
    with np.errstate(invalid='ignore'):
        dx = w.x[:, None, :] - w.x[:, :, None]
        dy = w.y[:, None, :] - w.y[:, :, None]
        see = np.sqrt(dx * dx + dy * dy) <= sim.params.sight[:, :, None]
    see &= alive[:, :, None] & alive[:, None, :]
    fly = w.is_flying
    see &= fly[:, :, None] | fly[:, None, :] | (lv_tgt[:, None, :] <= lv_obs[:, :, None])
    for p in range(w.n_players):
        mine = alive & (w.owner == p + 1)
        v.true_sight[:, :, p] = (see & mine[:, :, None]).any(axis=1) | mine
    revealed = v.reveal_to >= w.game_loop[:, None, None]
    w.visible[:] = (v.true_sight | revealed) & alive[:, :, None]
    if sim.config.lost_vision == 'sc2':
        from . import targeting  # avoid circular import: targeting does not depend on visibility

        targeting.after_vision(sim)


def on_hit(sim: Sim, b: int, attacker: int, victim: int) -> None:
    """Reveal `attacker` to `victim`'s owner through loop `current + reveal_loops - 1`
    (visible for `reveal_loops` loops starting on the hit loop) under `visibility='sc2'`.
    """
    if sim.config.visibility != 'sc2':
        return
    o = int(sim.world.owner[b, victim])
    if o > 0:
        loop = int(sim.world.game_loop[b])
        sim.vision.reveal_to[b, attacker, o - 1] = loop + sim.config.reveal_loops - 1


IMPLS = {'full': full, 'sc2': sc2}
