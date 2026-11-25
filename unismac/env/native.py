"""Rust-accelerated system-layer functions and numerical services (`accel='services'`).

Each function mirrors its pure-Python reference counterpart in name and signature,
dispatching array operations to ``zgame._zgame.env``. Inputs outside the native
fast path raise ``_zgame.env.Fallback`` without mutating outputs and transparently
fall back to the Python reference implementation.

``fallbacks`` tracks fallback counts by function name for diagnostics and tests.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import numpy as np

from ..backends.zgame.backend import _module
from ..contract import Backend, Command, WorldState
from . import catalog as catalog_mod
from . import reward as reward_mod
from . import services
from . import settle as settle_mod
from . import units as units_mod
from .abilities import ABILITY_DICT
from .actions import ActionSpace, FovState
from .catalog import AllyTables, EnemyTables, UnitCatalog
from .config import RewardConfig
from .rng import Streams
from .units import Frame, Side, UnitView

if TYPE_CHECKING:
    from .core import VecEnv

__all__ = ['SERVICES', 'SERVICES_VERSION', 'NativeKernels', 'fallbacks']

SERVICES_VERSION = 1

_env: Any = getattr(_module(), 'env', None)
if _env is None:
    raise ImportError(
        "The installed 'zgame' extension does not expose the 'env' submodule. "
        'Please install a compatible version of `zfoundry-zgame`.'
    )
_got_services = getattr(_env, 'SERVICES_VERSION', None)
if _got_services != SERVICES_VERSION:
    raise ImportError(
        f'Incompatible zgame env services version: expected {SERVICES_VERSION!r}, '
        f'got {_got_services!r}. '
        'Please install compatible versions of `unismac` and `zfoundry-zgame`.'
    )
Fallback: type[Exception] = _env.Fallback
fallbacks: Counter[str] = Counter()

_WORLD_FIELDS = tuple(f.name for f in fields(WorldState))
# Ability table: (native_id, perform, issued_id, ceil(range / 2)), matching abilities.point_target
_ABILITY_TABLE = [
    (ab, perform, code, math.ceil(rng / 2)) for ab, (perform, code, rng) in ABILITY_DICT.items()
]


# ---------------------------------------------------------------- Numerical services


def pairwise(ax, ay, bx, by, *, exclude_self: bool = False):
    """Rust counterpart of `services.pairwise`."""
    try:
        return _env.pairwise(ax, ay, bx, by, exclude_self)
    except Fallback:
        fallbacks['pairwise'] += 1
        return services.pairwise(ax, ay, bx, by, exclude_self=exclude_self)


def grid_lookup(grid, xi, yi, outside):
    """Rust counterpart of `services.grid_lookup`."""
    try:
        return _env.grid_lookup(grid, xi, yi, outside)
    except Fallback:
        fallbacks['grid_lookup'] += 1
        return services.grid_lookup(grid, xi, yi, outside)


def ordered_sum(values, mask):
    """Rust counterpart of `services.ordered_sum`."""
    try:
        return _env.ordered_sum(values, mask)
    except Fallback:
        fallbacks['ordered_sum'] += 1
        return services.ordered_sum(values, mask)


SERVICES = services.Services('rust', pairwise, grid_lookup, ordered_sum)


# ---------------------------------------------------------------- System layer


def to_command(
    space: ActionSpace,
    actions: np.ndarray,
    active: np.ndarray,
    avail: np.ndarray,
    frame: Frame,
    tables: AllyTables,
    n_slots: int,
    *,
    fov: FovState | None = None,
    attack_prob: np.ndarray | None = None,
    attack_rng: Streams | None = None,
    ability_choice: np.ndarray | None = None,
    out: Command | None = None,
) -> Command:
    """Rust counterpart of `ActionSpace.to_command`."""

    def reference() -> Command:
        fallbacks['to_command'] += 1
        return ActionSpace.to_command(
            space,
            actions,
            active,
            avail,
            frame,
            tables,
            n_slots,
            fov=fov,
            attack_prob=attack_prob,
            attack_rng=attack_rng,
            ability_choice=ability_choice,
            out=out,
        )

    prob, keys = None, None
    if attack_prob is not None and attack_rng is not None:
        if not isinstance(attack_rng, Streams):
            return reference()
        prob, keys = attack_prob, attack_rng.table()
    al = frame.allies
    reach = None
    if space.config.range_check and not space.config.mask_by_range:
        reach = space.target_ok(frame, tables, fov, True)
    cmd = Command.noop(actions.shape[0], n_slots) if out is None else out
    try:
        _env.to_command(
            cmd.kind,
            cmd.target_slot,
            cmd.target_x,
            cmd.target_y,
            cmd.ability,
            actions,
            active,
            avail,
            al.slot,
            al.x,
            al.y,
            al.health,
            frame.enemies.slot,
            space.heals(tables),
            space.n_no_attack,
            space.n_targets,
            float(space.config.move_amount),
            reach=reach,
            fov_dirs=None if fov is None else fov.dirs,
            fov_canonical=None if fov is None else fov.canonical,
            attack_prob=prob,
            attack_keys=keys,
            ability_choice=ability_choice,
            ability_table=None if ability_choice is None else _ABILITY_TABLE,
        )
    except Fallback:
        return reference()
    return cmd


def copy_rows(dst: WorldState, src: WorldState, rows: np.ndarray) -> None:
    """Rust counterpart of `units.copy_rows`."""
    idx = [int(r) for r in rows]
    try:
        if any(r < 0 for r in idx):
            raise Fallback('negative row index')
        _env.copy_rows(
            [getattr(dst, n) for n in _WORLD_FIELDS], [getattr(src, n) for n in _WORLD_FIELDS], idx
        )
    except Fallback:
        fallbacks['copy_rows'] += 1
        units_mod.copy_rows(dst, src, rows)


def advance(
    backend: Backend, command: Command, n_loops: int, world: WorldState, rows: np.ndarray
) -> None:
    """Step the backend and copy active rows using Rust `copy_rows`."""
    copy_rows(world, backend.step(command, n_loops), rows)


def frame(view: UnitView, world: WorldState) -> Frame:
    """Rust counterpart of `UnitView.frame`."""
    if view.ally_slots is None or view.enemy_slots is None or view._fresh is None:
        fallbacks['frame'] += 1
        return UnitView.frame(view, world)
    arrays = (
        world.alive,
        world.visible,
        world.unit_type,
        world.variant,
        world.x,
        world.y,
        world.health,
        world.health_max,
        world.shield,
        world.shield_max,
        world.energy,
        world.weapon_cooldown,
    )
    visible = view.config.enemy_source == 'visible'
    if visible and view.owners[0] != 1:  # Rust kernel checks Player 1 visibility only
        fallbacks['frame'] += 1
        return UnitView.frame(view, world)
    try:
        a = _env.unit_side(arrays, view.ally_slots, False, view._fresh, view._last.get('ally'))
        e = _env.unit_side(arrays, view.enemy_slots, visible, view._fresh, view._last.get('enemy'))
    except Fallback:
        fallbacks['frame'] += 1
        return UnitView.frame(view, world)
    view._last['ally'], view._last['enemy'] = a[-1], e[-1]
    view._fresh[:] = False
    return Frame(allies=Side(view.ally_slots, *a[:-1]), enemies=Side(view.enemy_slots, *e[:-1]))


def settle(
    env: VecEnv, active: np.ndarray, end: np.ndarray, cur: Frame, reward: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Rust counterpart of `settle.settle`."""
    rc = env.config.reward
    try:
        return _env.settle(
            active,
            end,
            env.episode_steps,
            cur.allies.health,
            cur.enemies.health,
            env.max_rewards,
            reward,
            env.battles_game,
            env.battles_won,
            env.timeouts,
            env.episode_count,
            env.done,
            int(env.episode_limit),
            bool(env.config.termination.continuing_episode),
            bool(rc.sparse),
            float(rc.win),
            float(rc.defeat),
            bool(rc.scale),
            float(rc.scale_rate),
        )
    except Fallback:
        fallbacks['settle'] += 1
        return settle_mod.settle(env, active, end, cur, reward)


def settle_enemy(
    active: np.ndarray,
    end: np.ndarray,
    reward: np.ndarray,
    max_rewards: np.ndarray,
    config: RewardConfig,
    infos: list[dict[str, Any]],
) -> np.ndarray:
    """Rust counterpart of `settle.settle_enemy`."""
    try:
        return _env.settle_enemy(
            active,
            end,
            reward,
            max_rewards,
            infos,
            bool(config.sparse),
            float(config.win),
            float(config.defeat),
            bool(config.scale),
            float(config.scale_rate),
        )
    except Fallback:
        fallbacks['settle_enemy'] += 1
        return settle_mod.settle_enemy(active, end, reward, max_rewards, config, infos)


def fill_max_rewards(
    out: np.ndarray,
    env_ids: Sequence[int],
    n_enemies: int,
    config: RewardConfig,
    health_max: np.ndarray,
    shield_max: np.ndarray,
) -> None:
    """Rust counterpart of `reward.fill_max_rewards`."""
    base = n_enemies * config.death_value + config.win
    try:
        _env.fill_max_rewards(out, [int(b) for b in env_ids], float(base), health_max, shield_max)
    except Fallback:
        fallbacks['fill_max_rewards'] += 1
        reward_mod.fill_max_rewards(out, env_ids, n_enemies, config, health_max, shield_max)


def _one(dtype: type) -> Callable[[int], np.ndarray]:
    return lambda v: np.full((1, 1), v, dtype)


_I32, _I8 = _one(np.int32), _one(np.int8)


class NativeKernels:
    """Stateful system-layer kernels with per-type memoization (one instance per `VecEnv`):
    sort type codes and unit constant tables."""

    def __init__(self) -> None:
        self._codes: dict[Callable[[int], int], dict[int, int]] = {}
        self._ally: dict[UnitCatalog, dict[tuple[int, int], tuple]] = {}
        self._enemy: dict[UnitCatalog, dict[tuple[int, int], tuple]] = {}

    def assign(
        self, view: UnitView, world: WorldState, env_ids: Sequence[int], ally_code: Callable
    ) -> None:
        """Rust counterpart of `UnitView.assign`."""
        b_size = world.batch_size
        if view.ally_slots is None or view.ally_slots.shape[0] != b_size:
            view.ally_slots = np.zeros((b_size, view.n_agents), np.int64)
            view.enemy_slots = np.zeros((b_size, view.n_enemies), np.int64)
        if view._fresh is None or view._fresh.shape[0] != b_size:
            view._fresh = np.ones(b_size, bool)
        ids = [int(b) for b in env_ids]
        view._fresh[ids] = True
        cfg = view.config
        codes = self._codes.setdefault(ally_code, {})
        while True:
            try:
                missing = _env.assign_units(
                    world.alive,
                    world.owner,
                    world.unit_type,
                    world.x,
                    world.y,
                    ids,
                    view.ally_slots,
                    view.enemy_slots,
                    cfg.ally_order == 'type_xy',
                    cfg.enemy_order == 'type_xy',
                    list(codes),
                    list(codes.values()),
                    ally_owner=view.owners[0],
                    enemy_owner=view.owners[1],
                )
            except Fallback:
                break
            if missing is None:
                return
            try:  # Populate missing type code via reference (errors fall back to reference)
                codes[missing] = int(ally_code(missing))
            except Exception:
                break
        fallbacks['assign'] += 1
        UnitView.assign(view, world, env_ids, ally_code)

    def ally_tables(
        self,
        catalogs: Sequence[UnitCatalog],
        unit_type: np.ndarray,
        variant: np.ndarray,
        shield_max: np.ndarray,
    ) -> AllyTables:
        """Rust counterpart of `catalog.ally_tables_by_env`."""
        first = catalogs[0]
        if all(c is first for c in catalogs):
            uniq, row = [first], np.zeros(len(catalogs), np.int64)
        else:
            index: dict[int, int] = {}
            uniq = []
            for c in catalogs:
                if index.setdefault(id(c), len(uniq)) == len(uniq):
                    uniq.append(c)
            row = np.array([index[id(c)] for c in catalogs], np.int64)
        static = first.standard and first.shield_static
        if any(
            (c.standard, c.shield_static) != (first.standard, first.shield_static) for c in uniq
        ):
            fallbacks['ally_tables'] += 1
            return catalog_mod.ally_tables_by_env(catalogs, unit_type, variant, shield_max)

        def entry(cat: UnitCatalog, t: int, v: int) -> tuple:
            tab = cat.ally_tables(_I32(t), _I8(v), np.zeros((1, 1)))
            return tuple(
                f(getattr(tab, n)[0, 0])
                for f, n in (
                    (int, 'type_id'),
                    (float, 'cooldown_max'),
                    (float, 'shield_max'),
                    (bool, 'is_medivac'),
                    (float, 'shoot_range'),
                    (float, 'sight_range'),
                )
            )

        out = self._tables(
            self._ally,
            uniq,
            row,
            unit_type,
            variant,
            shield_max if static else None,
            first.shield_bits_ally,
            entry,
        )
        if out is None:
            fallbacks['ally_tables'] += 1
            return catalog_mod.ally_tables_by_env(catalogs, unit_type, variant, shield_max)
        return AllyTables(*out)

    def enemy_tables(
        self,
        catalog: UnitCatalog,
        unit_type: np.ndarray,
        variant: np.ndarray,
        shield_max: np.ndarray | None = None,
    ) -> EnemyTables:
        """Rust counterpart of `UnitCatalog.enemy_tables`."""
        if catalog.shield_static and shield_max is None:
            fallbacks['enemy_tables'] += 1
            return UnitCatalog.enemy_tables(catalog, unit_type, variant, shield_max)

        def entry(cat: UnitCatalog, t: int, v: int) -> tuple:
            tab = cat.enemy_tables(_I32(t), _I8(v), np.zeros((1, 1)))
            tid, sh = int(tab.type_id[0, 0]), float(tab.shield_max[0, 0])
            return (tid, 0.0, sh, bool(tab.is_medivac[0, 0]), 0.0, 0.0)

        row = np.zeros(unit_type.shape[0], np.int64)
        out = self._tables(
            self._enemy,
            [catalog],
            row,
            unit_type,
            variant,
            shield_max if catalog.shield_static else None,
            catalog.shield_bits_enemy,
            entry,
        )
        if out is None:
            fallbacks['enemy_tables'] += 1
            return UnitCatalog.enemy_tables(catalog, unit_type, variant, shield_max)
        type_id, _, shield, medivac, _, _ = out
        return EnemyTables(type_id=type_id, shield_max=shield, is_medivac=medivac)

    @staticmethod
    def _tables(
        memo: dict[UnitCatalog, dict[tuple[int, int], tuple]],
        uniq: list[UnitCatalog],
        row: np.ndarray,
        unit_type: np.ndarray,
        variant: np.ndarray,
        shield_in: np.ndarray | None,
        shield_bits: int,
        entry: Callable[[UnitCatalog, int, int], tuple],
    ) -> tuple | None:
        """Expand unit tables by `(catalog, unit_type, variant)`, lazily populating missing keys
        via `entry`. Returns six arrays or None to trigger reference fallback."""
        books = [memo.setdefault(c, {}) for c in uniq]
        table = [(k, t, v, *e) for k, book in enumerate(books) for (t, v), e in book.items()]
        while True:
            try:
                r = _env.unit_tables(table, row, unit_type, variant, shield_in, bool(shield_bits))
            except Fallback:
                return None
            if len(r) == 6:
                return r
            k, t, v = (int(x) for x in r)
            try:
                e = entry(uniq[k], t, v)
            except Exception:
                return None
            books[k][(t, v)] = e
            table.append((k, t, v, *e))
