"""SMAC-Hard ability action segment (`use_ability=True`), ported from `available_action_ability.py`.

* From each agent's available abilities returned by `AbilityQuery.query_abilities` (native IDs),
  pick the first entry present in `ABILITY_DICT`. Upstream takes the first in engine order; here
  the list is sorted ascending so we pick the smallest. Each unit type has at most one matching
  ability in the table (Marine 380, Marauder 253, Medivac 2116, Stalker 1442, VoidRay 2393,
  Hellion 1998, Reaper 2588, Queen 1664, SiegeTank 388, SiegeTankSieged 390), so both agree.
* Ability segment length equals target padding (`max(n_enemies, n_agents, 9)`), with availability
  determined by `perform`:
  - `self`: slot 0 in the segment is available;
  - `ally` (Transfusion 1664): allied agents other than self with `health != health_max` and
    distance < 7 (indexed by agent ID). Upstream also matches dead agents (health 0, coordinates
    retained at last alive position), preserved here; selecting a dead target issues no command
    (the backend contract disallows targeting dead slots);
  - `point` (KD8Charge 2588, Blink 1442): 9 candidate points at `r = ceil(range / 2)` (self, N,
    NE, E, SE, S, SW, W, NW; offsets match upstream, including south using `r / 2`), available
    when `0 <= x < map_x` and `0 <= y < map_y`.
* Dispatch: `self` takes no target, `ally` targets that agent, `point` targets the k-th point,
  using the second element of `ABILITY_DICT` as ability ID (generic ID 3675 for Stim, 3687 for
  Blink).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ['ABILITY_DICT', 'ability_avail', 'choose_ability', 'point_target']

# Native ability ID -> (perform, issued ability ID, range).
ABILITY_DICT: dict[int, tuple[str, int, int]] = {
    380: ('self', 3675, 0),  # Effect_Stim_Marine
    1664: ('ally', 1664, 7),  # Effect_Transfusion
    2393: ('self', 2393, 0),  # Effect_VoidRayPrismaticAlignment
    1998: ('self', 1998, 0),  # Morph_Hellbat
    1978: ('self', 1978, 0),  # Morph_Hellion
    2588: ('point', 2588, 5),  # Effect_KD8Charge
    1442: ('point', 3687, 8),  # Effect_Blink_Stalker
    388: ('self', 388, 0),  # Morph_SiegeMode
    253: ('self', 3675, 0),  # Effect_Stim_Marauder
    2116: ('self', 2116, 0),  # Effect_MedivacIgniteAfterburners
    390: ('self', 390, 0),  # Morph_Unsiege
}


def choose_ability(abilities: list[int]) -> int:
    """Return the first native ability ID in `abilities` found in `ABILITY_DICT`, or 0."""
    for a in abilities:
        if a in ABILITY_DICT:
            return a
    return 0


def _point_target(x: float, y: float, k: int, r: float) -> tuple[float, float]:
    # Exact upstream arithmetic (x + r/2, etc.) to preserve floating-point parity.
    if k == 0:
        return x, y
    if k == 1:
        return x, y + r
    if k == 2:
        return x + r / 2, y + r / 2
    if k == 3:
        return x + r, y
    if k == 4:
        return x + r / 2, y - r / 2
    if k == 5:
        return x, y - r / 2
    if k == 6:
        return x - r / 2, y - r / 2
    if k == 7:
        return x - r, y
    return x - r / 2, y + r / 2


def point_target(ability: int, x: float, y: float, k: int) -> tuple[float, float]:
    """Return the k-th target point (world coordinates) for a `point` ability."""
    r = math.ceil(ABILITY_DICT[ability][2] / 2)
    return _point_target(x, y, k, r)


def ability_avail(
    choice: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    health: np.ndarray,
    health_max: np.ndarray,
    alive: np.ndarray,
    n_pad: int,
    map_size: tuple[int, int],
) -> np.ndarray:
    """Return `int64[B, A, n_pad]` ability availability mask.

    `choice` is `int[B, A]` from `choose_ability`; remaining arrays are from the ally view
    (dead agents retain their last alive coordinates and have health 0). Dead-agent masking is
    handled by the caller.
    """
    b_size, n_agents = choice.shape
    out = np.zeros((b_size, n_agents, n_pad), np.int64)
    map_x, map_y = map_size
    for b in range(b_size):
        for a in range(n_agents):
            ab = int(choice[b, a])
            if not ab or not alive[b, a]:
                continue
            perform, _, rng = ABILITY_DICT[ab]
            if perform == 'self':
                out[b, a, 0] = 1
            elif perform == 'ally':
                ax, ay = float(x[b, a]), float(y[b, a])
                for u in range(n_agents):
                    if u == a:
                        continue
                    if float(health[b, u]) != float(health_max[b, u]) and (
                        math.hypot(float(x[b, u]) - ax, float(y[b, u]) - ay) < 7
                    ):
                        out[b, a, u] = 1
            elif perform == 'point':
                r = math.ceil(rng / 2)
                for k in range(9):
                    px, py = _point_target(float(x[b, a]), float(y[b, a]), k, r)
                    out[b, a, k] = int(0 <= px < map_x and 0 <= py < map_y)
    return out
