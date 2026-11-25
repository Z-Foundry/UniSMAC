"""Unit view (`UnitView`): extracts `[B, A]` ally and `[B, E]` enemy snapshots from `WorldState`.

* Agent indexing: `ally_order='type_xy'` stably sorts by `(type_code, x, y)` (matching upstream
  SMAC sorting by native `unit_type`, `pos.x`, `pos.y`), while `'slot'` keeps backend slot order.
  Enemies default to `'slot'` (matching upstream SMAC obs order for `map_trigger`), or sort by
  `(native unit_type, x, y)` when `enemy_order='type_xy'` (SMAC-Hard). Indices are fixed at reset
  for the duration of each episode.
* Death rule: allies are dead when `alive=False`; enemies under `enemy_source='visible'` also
  require visibility by the owning player (matching upstream "absent from obs => health 0"). Dead
  units report health 0 while retaining their last alive coordinates (matching upstream `agents` /
  `enemies` dicts); other fields come from the backend.
* Numeric fields are widened to `float64` (preserving exact float32 values) to match upstream
  Python float calculations.
* `owners`: `(ally_owner, enemy_owner)` player IDs, defaulting to `(1, 2)`. In two-sided mode,
  Player 2's view uses `(2, 1)` (`unismac.env.dual`) and checks the ally owner's visibility column.

`UnitView.assign`, `UnitView.frame`, `copy_rows`, and `advance` belong to the system layer: this
module is the Python reference, while Rust implementations live in `zgame-smac` (`_zgame.env`)
for `accel='services'` and above; `advance` writes active rows directly in-kernel under
`accel='rust'` with `ZgameBackend` (see `unismac.env.accel`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace

import numpy as np

from ..contract import Backend, Command, ContractError, WorldState
from .config import UnitsConfig

__all__ = ['Frame', 'Side', 'UnitView', 'advance', 'copy_rows']


@dataclass(frozen=True)
class Side:
    """Snapshot of one side's units, shape `[B, N]`."""

    slot: np.ndarray  # int64, corresponding WorldState slot index
    alive: np.ndarray  # bool, per death rule
    unit_type: np.ndarray  # int32, standard UnitTypeId
    variant: np.ndarray  # int8
    x: np.ndarray  # float64
    y: np.ndarray
    health: np.ndarray  # float64; 0 when dead
    health_max: np.ndarray
    shield: np.ndarray
    shield_max: np.ndarray
    energy: np.ndarray
    cooldown: np.ndarray  # weapon_cooldown in loops; NaN for non-controlled units


@dataclass(frozen=True)
class Frame:
    allies: Side
    enemies: Side


def copy_rows(dst: WorldState, src: WorldState, rows: np.ndarray) -> None:
    """Copy `rows` of all fields from `src` into `dst`. Rust: zgame-smac `env::rows::copy_rows`."""
    for f in fields(WorldState):
        getattr(dst, f.name)[rows] = getattr(src, f.name)[rows]


def advance(
    backend: Backend, command: Command, n_loops: int, world: WorldState, rows: np.ndarray
) -> None:
    """Step the backend by `n_loops` and copy active `rows` into `world` (`copy_rows`)."""
    copy_rows(world, backend.step(command, n_loops), rows)


def _gather(world: WorldState, slots: np.ndarray, alive: np.ndarray) -> Side:
    rows = np.arange(world.batch_size)[:, None]

    def g(name: str) -> np.ndarray:
        return getattr(world, name)[rows, slots]

    f64 = np.float64
    return Side(
        slot=slots,
        alive=alive,
        unit_type=g('unit_type'),
        variant=g('variant'),
        x=g('x').astype(f64),
        y=g('y').astype(f64),
        health=np.where(alive, g('health').astype(f64), 0.0),
        health_max=g('health_max').astype(f64),
        shield=g('shield').astype(f64),
        shield_max=g('shield_max').astype(f64),
        energy=g('energy').astype(f64),
        cooldown=g('weapon_cooldown').astype(f64),
    )


class UnitView:
    def __init__(
        self,
        n_agents: int,
        n_enemies: int,
        config: UnitsConfig,
        owners: tuple[int, int] = (1, 2),
    ) -> None:
        self.n_agents = n_agents
        self.n_enemies = n_enemies
        self.config = config
        self.owners = owners
        self.ally_slots: np.ndarray | None = None  # int64 [B, A]
        self.enemy_slots: np.ndarray | None = None  # int64 [B, E]
        # Last alive coordinates float64 [2, B, N] for ally/enemy; cleared on env reset.
        self._last: dict[str, np.ndarray] = {}
        self._fresh: np.ndarray | None = None  # bool [B], True after reset before first frame

    def assign(
        self, world: WorldState, env_ids: Sequence[int], ally_code: Callable[[int], int]
    ) -> None:
        """Assign agent and enemy indices for `env_ids` after reset.
        Rust counterpart: zgame-smac `env::units::assign`."""
        b_size = world.batch_size
        if self.ally_slots is None or self.ally_slots.shape[0] != b_size:
            self.ally_slots = np.zeros((b_size, self.n_agents), np.int64)
            self.enemy_slots = np.zeros((b_size, self.n_enemies), np.int64)
        assert self.enemy_slots is not None
        if self._fresh is None or self._fresh.shape[0] != b_size:
            self._fresh = np.ones(b_size, bool)
        self._fresh[list(env_ids)] = True
        own, foe = self.owners
        for b in env_ids:
            live = world.alive[b]
            allies = [s for s in range(world.n_slots) if live[s] and world.owner[b, s] == own]
            enemies = [s for s in range(world.n_slots) if live[s] and world.owner[b, s] == foe]
            if (len(allies), len(enemies)) != (self.n_agents, self.n_enemies):
                raise ContractError(
                    f'env {b} has {len(allies)} ally and {len(enemies)} enemy units after reset; '
                    f'scenario requires {self.n_agents} / {self.n_enemies}'
                )
            if self.config.ally_order == 'type_xy':
                allies.sort(
                    key=lambda s: (
                        ally_code(int(world.unit_type[b, s])),
                        float(world.x[b, s]),
                        float(world.y[b, s]),
                    )
                )
            if self.config.enemy_order == 'type_xy':
                enemies.sort(
                    key=lambda s: (
                        int(world.unit_type[b, s]),
                        float(world.x[b, s]),
                        float(world.y[b, s]),
                    )
                )
            self.ally_slots[b] = allies
            self.enemy_slots[b] = enemies

    def frame(self, world: WorldState) -> Frame:
        """Build the current frame (`_gather` + `_hold`). Rust: zgame-smac `env::units::side`."""
        if self.ally_slots is None or self.enemy_slots is None:
            raise RuntimeError('UnitView is not assigned yet (call reset() first)')
        rows = np.arange(world.batch_size)[:, None]
        a_alive = world.alive[rows, self.ally_slots]
        e_alive = world.alive[rows, self.enemy_slots]
        if self.config.enemy_source == 'visible':
            e_alive = e_alive & world.visible[rows, self.enemy_slots, self.owners[0] - 1]
        allies = self._hold('ally', _gather(world, self.ally_slots, a_alive))
        enemies = self._hold('enemy', _gather(world, self.enemy_slots, e_alive))
        assert self._fresh is not None
        self._fresh[:] = False
        return Frame(allies=allies, enemies=enemies)

    def _hold(self, key: str, side: Side) -> Side:
        """Preserve dead units' coordinates at their last alive positions."""
        assert self._fresh is not None
        prev = self._last.get(key)
        x, y = side.x, side.y
        if prev is not None and prev.shape == (2, *x.shape):
            keep = ~side.alive & ~self._fresh[:, None]
            x = np.where(keep, prev[0], x)
            y = np.where(keep, prev[1], y)
        self._last[key] = np.stack([x, y])
        return replace(side, x=x, y=y)
