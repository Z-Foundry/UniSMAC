"""Batched `[B, U, U]` pairwise geometry for vectorized subsystems.

Results are bit-identical to pairwise helpers such as `sim.edge_distance`: the same float32
operations run in the same order (NumPy elementwise arithmetic and sqrt are correctly rounded,
matching scalar evaluation). Dead or empty slots have NaN coordinates and produce NaN entries
that callers mask out with alive masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import WEAPON_AIR, WEAPON_GROUND, Sim, range_radii


@dataclass
class Pairs:
    """Entry `[b, u, t]` holds the pairwise quantity from unit `u` to unit `t`."""

    center: np.ndarray  # float32 center-to-center distance
    edge: np.ndarray  # float32 edge-to-edge distance for range checks (same as sim.range_edge)
    enemy: np.ndarray  # bool, True when t is an enemy of u (same as sim.is_enemy)
    weapon: np.ndarray  # int64 weapon index of u against t, or -1 if none (same as sim.weapon_for)
    reach: np.ndarray  # float32 weapon range of u against t (0 if none; use with weapon)


def pairs(sim: Sim) -> Pairs:
    w, p = sim.world, sim.params
    with np.errstate(invalid='ignore'):
        dx = w.x[:, None, :] - w.x[:, :, None]
        dy = w.y[:, None, :] - w.y[:, :, None]
        center = np.sqrt(dx * dx + dy * dy)
        rr = range_radii(sim)  # defaults to w.radius
        edge = center - rr[:, :, None] - rr[:, None, :]
    owner_t = w.owner[:, None, :]
    enemy = w.alive[:, None, :] & (owner_t != 0) & (owner_t != w.owner[:, :, None])
    widx = np.where(w.is_flying, WEAPON_AIR, WEAPON_GROUND)  # [B, T]
    widx = np.broadcast_to(widx[:, None, :], edge.shape)
    has = np.take_along_axis(p.has_weapon, widx, axis=2)
    reach = np.take_along_axis(p.weapon_range, widx, axis=2)
    weapon = np.where(has, widx, -1)
    return Pairs(center=center, edge=edge, enemy=enemy, weapon=weapon, reach=reach)


def in_range(pr: Pairs, extra: float = 0.0) -> np.ndarray:
    """`[b, u, t]`: True if u has a weapon against t within range + extra
    (matches `targeting.in_weapon_range`).
    """
    with np.errstate(invalid='ignore'):
        return (pr.weapon >= 0) & (pr.edge <= pr.reach + np.float32(extra))


def nearest(mask: np.ndarray, dist: np.ndarray, no: int) -> np.ndarray:
    """`[b, u]`: slot t with minimum `dist` where `mask` is True (lowest slot on tie), else `no`."""
    d = np.where(mask, dist, np.float32(np.inf))
    best = np.argmin(d, axis=2)
    found = np.take_along_axis(d, best[:, :, None], axis=2)[:, :, 0] < np.inf
    return np.where(found, best, no)


def gather(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """`[b, u]`: gather `a[b, u, idx[b, u]]`; negative `idx` wraps as in Python indexing."""
    return np.take_along_axis(a, (idx % a.shape[2])[:, :, None], axis=2)[:, :, 0]


__all__ = ['Pairs', 'gather', 'in_range', 'nearest', 'pairs']
