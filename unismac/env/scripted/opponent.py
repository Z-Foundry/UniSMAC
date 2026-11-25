"""`ScriptedOpponent`: SMAC-Hard scripted opponent (Player 2 as an `AgentPlayer` driven by
per-step decision-tree scripts).

Correspondence with upstream `StarCraft2Env` (SMAC-Hard):

* **Script selection**: upstream draws `random.choice(SCRIPT_DICT[map_name])` on each reset; here
  we draw `integers(len(pool))` from `stream(seed, 'opponent', env_id, episode, 0)` (duplicates
  in `pool` act as weights; see `VecEnv.rng_stream` for per-env seeds). Envs specified via
  `VecEnv.reset(opponents=...)` skip sampling and use the given script (must be in `pool`),
  without affecting other envs or subsequent episodes. Script-internal `random.choice` calls use
  `self.random.choice`, drawing sequentially from `stream(seed, 'opponent', env_id, episode, t+1)`
  at episode step `t`.
* **Unit dicts**: script `agents` are Player 2 units (env enemy indices; matches upstream
  `self.enemies` order when `enemy_order='type_xy'`), and `enemies` are Player 1 units (env ally
  indices). Fields use backend ground truth. Mirrors upstream `update_units`: alive units are
  replaced with fresh objects each step, while newly dead units reuse the prior object with
  `health` set to 0 (visible through cached script references).
* **iteration**: episode step count prior to executing this step.
* **visible_matrix**: `blue` = alive Player 1 units visible to Player 2; `red` = alive Player 2
  units visible to Player 1 (`WorldState.visible`).
* **agent_ability**: queried from `AbilityQuery` only when `script.needs_abilities` is True,
  producing one entry per alive Player 2 unit (empty list if backend lacks ability queries).
* **Commands**: `RawAction` maps by ability ID (16 -> `MOVE`, 23 -> `ATTACK_MOVE` / `ATTACK_UNIT`,
  4 -> `STOP`, others -> `ABILITY`). Multiple actions for the same unit keep the last one.
  Deviation: commands from dead actors or targeting dead/missing units are dropped (contract
  forbids them). `script_abilities='drop'` also drops `ABILITY` commands (`smac-hard-portable` on
  pyref/zgame).
* **Script exceptions**: upstream crash points (e.g. `min([])` / `center([])` when only Medivacs
  remain, or `nearest_n_units(...)[0]` when enemies are wiped out) are faithfully preserved in the
  scripts. Deviation: exceptions are caught instead of crashing training; Player 2 issues no
  commands that step, `errors[env_id]` increments, and a `RuntimeWarning` is emitted once per
  `script:exception_type`.

Public read-only attributes: `pool` (scenario script pool, matching `script_pool`), `names`
(`env_id -> active script name`), `errors` (`env_id -> cumulative exception step count`); exposed
per-env via `VecEnv.opponent_names()` and `VecEnv.opponent_errors()`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ...contract import AgentPlayer, Command, CommandKind, Player, WorldState
from ..config import OpponentConfig
from ..opponents import Opponent
from .api import AbilityEntry, AbilityItem, RawAction, ScriptUnit
from .base import POOLS, SCRIPTS, Script

if TYPE_CHECKING:
    from ..core import VecEnv

__all__ = ['ScriptedOpponent', 'StreamChoice', 'script_pool']


def script_pool(config: OpponentConfig, scenario: str) -> tuple[str, ...]:
    """Return the script pool for `scenario`: `config.scripts` if provided, else the upstream
    `SCRIPT_DICT` entry (`POOLS`); duplicates act as sampling weights.

    Raises `ValueError` if the scenario has no script pool or contains unknown script names.
    """
    pool = config.scripts if config.scripts is not None else POOLS.get(scenario)
    if not pool:
        raise ValueError(
            f'Scenario {scenario} has no SMAC-Hard script (not in SCRIPT_DICT) '
            'and cannot use a scripted opponent'
        )
    unknown = [n for n in pool if n not in SCRIPTS]
    if unknown:
        raise ValueError(f'unknown script {unknown}; available: {sorted(SCRIPTS)}')
    return tuple(pool)


class StreamChoice:
    """Drop-in replacement for `random.choice`, drawing `integers(len(seq))` from a `Generator`."""

    def __init__(self) -> None:
        self.rng: np.random.Generator | None = None

    def choice(self, seq: Sequence[Any]) -> Any:
        if self.rng is None:
            raise RuntimeError('StreamChoice RNG stream not set')
        if not len(seq):
            raise IndexError('Cannot choose from an empty sequence')
        return seq[int(self.rng.integers(len(seq)))]


def _unit(world: WorldState, b: int, s: int) -> ScriptUnit:
    return ScriptUnit(
        tag=s + 1,
        unit_type=int(world.unit_type[b, s]),
        owner=int(world.owner[b, s]),
        x=float(world.x[b, s]),
        y=float(world.y[b, s]),
        health=float(world.health[b, s]),
        health_max=float(world.health_max[b, s]),
        shield=float(world.shield[b, s]),
        shield_max=float(world.shield_max[b, s]),
        energy=float(world.energy[b, s]),
        weapon_cooldown=float(world.weapon_cooldown[b, s]),
    )


class ScriptedOpponent(Opponent):
    def __init__(self, config: OpponentConfig) -> None:
        self.config = config
        self.env: VecEnv | None = None
        self.pool: tuple[str, ...] = ()
        self.scripts: dict[int, Script] = {}
        self.names: dict[int, str] = {}  # env_id -> current episode script name
        self._choice: dict[int, StreamChoice] = {}
        # env_id -> (Player 2 units dict, Player 1 units dict), index -> ScriptUnit
        self._units: dict[int, tuple[dict[int, ScriptUnit], dict[int, ScriptUnit]]] = {}
        self.errors: dict[int, int] = {}  # env_id -> cumulative script exception step count
        self._warned: set[str] = set()

    def player(self, race: str) -> Player:
        return AgentPlayer(race)  # type: ignore[arg-type]

    def bind(self, env: VecEnv) -> None:
        self.env = env
        self.pool = script_pool(self.config, env.spec.name)

    def _require_env(self) -> VecEnv:
        if self.env is None:
            raise RuntimeError('ScriptedOpponent is not bound to an environment')
        return self.env

    def _stream(self, b: int, frame: int) -> np.random.Generator:
        return self._require_env().rng_stream('opponent', b, frame)

    def check_names(self, names: Sequence[str | None]) -> None:
        for name in names:
            if name is None:
                continue
            if name not in SCRIPTS:
                raise ValueError(f'unknown script {name!r}; available: {sorted(SCRIPTS)}')
            if name not in self.pool:
                scenario = self.env.spec.name if self.env is not None else '?'
                raise ValueError(
                    f'Script {name!r} is not in scenario {scenario} '
                    f'script pool {sorted(set(self.pool))}'
                )

    def on_reset(
        self,
        world: WorldState,
        env_ids: Sequence[int],
        names: Sequence[str | None] | None = None,
    ) -> None:
        env = self._require_env()
        for k, b in enumerate(env_ids):
            name = None if names is None else names[k]
            if name is None:
                name = self.pool[int(self._stream(b, 0).integers(len(self.pool)))]
            script = SCRIPTS[name](env.spec.name)
            choice = StreamChoice()
            script.random = choice
            self.scripts[b] = script
            self.names[b] = name
            self._choice[b] = choice
            self._units.pop(b, None)
            self._sync(world, b)

    def _slots(self, b: int) -> tuple[np.ndarray, np.ndarray]:
        view = self._require_env().view
        assert view.ally_slots is not None and view.enemy_slots is not None
        return view.enemy_slots[b], view.ally_slots[b]

    def _sync(self, world: WorldState, b: int) -> None:
        """Mirror upstream `update_units`: fresh objects for alive units; dead units keep their
        previous object with `health = 0.0`."""
        mine, theirs = self._slots(b)
        old = self._units.get(b)
        new: list[dict[int, ScriptUnit]] = []
        for k, slots in enumerate((mine, theirs)):
            prev = old[k] if old is not None else None
            cur: dict[int, ScriptUnit] = {}
            for i, s in enumerate(slots):
                s = int(s)
                if world.alive[b, s]:
                    cur[i] = _unit(world, b, s)
                else:
                    unit = prev[i] if prev is not None else _unit(world, b, s)
                    unit.health = 0.0
                    cur[i] = unit
            new.append(cur)
        self._units[b] = (new[0], new[1])

    def _visible(self, world: WorldState, b: int) -> dict[str, list[int]]:
        mine, theirs = self._slots(b)
        blue = [
            int(s) + 1
            for s in theirs
            if world.alive[b, s] and world.health[b, s] > 0 and world.visible[b, s, 1]
        ]
        red = [
            int(s) + 1
            for s in mine
            if world.alive[b, s] and world.health[b, s] > 0 and world.visible[b, s, 0]
        ]
        return {'red': red, 'blue': blue}

    def _abilities(self, world: WorldState, b: int) -> list[AbilityEntry]:
        env = self._require_env()
        query = env.query_abilities()
        mine, _ = self._slots(b)
        out = []
        for s in mine:
            s = int(s)
            if world.alive[b, s]:
                ids = query[b][s] if query is not None else []
                out.append(AbilityEntry(s + 1, tuple(AbilityItem(int(a)) for a in ids)))
        return out

    def issue(self, world: WorldState, command: Command, active: np.ndarray) -> None:
        env = self._require_env()
        for b in np.flatnonzero(active):
            b = int(b)
            script = self.scripts[b]
            self._sync(world, b)
            mine, theirs = self._units[b]
            t = int(env.episode_steps[b])
            self._choice[b].rng = self._stream(b, t + 1)
            abilities = self._abilities(world, b) if script.needs_abilities else []
            try:
                actions = script.script(mine, theirs, abilities, self._visible(world, b), t)
            except Exception as exc:  # noqa: BLE001 - upstream script crash points; see module doc
                self.errors[b] = self.errors.get(b, 0) + 1
                key = f'{self.names[b]}:{type(exc).__name__}'
                if key not in self._warned:
                    self._warned.add(key)
                    name = self.names[b]
                    warnings.warn(
                        f'SMAC-Hard script {name} raised {exc!r} at step {t}; '
                        'Player 2 issues no commands this step',
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue
            drop = self.config.script_abilities == 'drop'
            for act in actions or ():  # Some upstream scripts use bare return (None)
                if act is not None:
                    self._apply(world, command, b, act, drop_abilities=drop)

    @staticmethod
    def _apply(
        world: WorldState, command: Command, b: int, act: RawAction, *, drop_abilities: bool
    ) -> None:
        slot = int(act.unit_tag) - 1
        if not (0 <= slot < world.n_slots and world.alive[b, slot] and world.owner[b, slot] == 2):
            return
        target = None
        if act.target_tag is not None:
            target = int(act.target_tag) - 1
            if not (0 <= target < world.n_slots and world.alive[b, target]):
                return
        xy = None
        if act.target_pos is not None:
            xy = (
                float(np.float32(act.target_pos[0])),
                float(np.float32(act.target_pos[1])),
            )
        ability = int(act.ability_id)
        basic = (ability in (16, 23) and xy is not None) or (ability == 23 and target is not None)
        basic = basic or (ability == 4 and target is None and xy is None)
        if drop_abilities and not basic:
            return
        # Clear any prior command on this slot so the last action wins.
        command.set(b, slot, CommandKind.NOOP)
        if ability == 16 and xy is not None:
            command.set(b, slot, CommandKind.MOVE, target_xy=xy)
        elif ability == 23 and xy is not None:
            command.set(b, slot, CommandKind.ATTACK_MOVE, target_xy=xy)
        elif ability == 23 and target is not None:
            command.set(b, slot, CommandKind.ATTACK_UNIT, target_slot=target)
        elif ability == 4 and target is None and xy is None:
            command.set(b, slot, CommandKind.STOP)
        else:
            kw: dict[str, Any] = {'ability': ability}
            if target is not None:
                kw['target_slot'] = target
            if xy is not None:
                kw['target_xy'] = xy
            command.set(b, slot, CommandKind.ABILITY, **kw)
