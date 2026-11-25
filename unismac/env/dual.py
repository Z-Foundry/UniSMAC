"""Two-sided mode (`opponent.kind='external'`): Player 2 is also an `AgentPlayer`, controlled
per step by caller-supplied external actions.

* Player 2's perspective components and per-episode state live in `Seat`, following the same rules
  as Player 1: unit view (allies = Player 2, `UnitView(owners=(2, 1))`), unit catalog (fitted on
  Player 2's initial units with swapped races), action space, obs/state layouts, reward and death
  tracking, `max_reward`, spotter (stream `obs_shuffle_p2`), and `last_action`.
* Under `smac` type encoding, if Player 2 has unit types absent from `map_type`'s role table (e.g.
  Zealots on asymmetric `3s_vs_5z`), `seat_catalog` replaces Player 2's role table with its own
  sorted standard unit type IDs.
* Termination and settlement use `termination.game_end_dual` and `settle.settle_enemy`.
* Unsupported configs raise `ValueError` via `check_config`; backend player symmetry is validated
  by `check_backend`.
* Mirror spawning (`backend.spawn_mirror='auto'`, explicit spawns, mirrorable map): Player 2
  spawns mirror Player 1 (`mirror_spawns`), and facing is mirrored by the backend
  (`Scenario.mirror`). See `mirror_axis` for mirrorability criteria.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..contract import CapabilityError, Command, MapInfo, Spawn, UnitOverride, WorldState
from .actions import ActionSpace
from .catalog import AllyTables, EnemyTables, UnitCatalog
from .features import FeatureContext, ObsLayout, StateLayout
from .reward import BattleReward
from .spotter import Spotter
from .units import Frame, UnitView

if TYPE_CHECKING:
    from .config import EnvConfig
    from .core import VecEnv

__all__ = [
    'OBS_STREAM',
    'Seat',
    'check_backend',
    'check_config',
    'check_options',
    'mirror_axis',
    'mirror_spawns',
    'seat_catalog',
    'symmetric_options',
]

# Player 2 RNG stream for spotter and obs ordering (Player 1 uses obs_shuffle)
OBS_STREAM = 'obs_shuffle_p2'


def check_config(config: EnvConfig) -> None:
    """Validate config for two-sided mode, raising `ValueError` listing unsupported options."""
    bad = []
    if config.units.enemy_source != 'all':
        bad.append(
            "units.enemy_source must be 'all' ('visible' requires two-sided visibility, "
            'not implemented)'
        )
    if config.visibility.model != 'circle':
        bad.append("visibility.model must be 'circle' (conic FOV state exists only for Player 1)")
    if config.capability.attack is not None or config.capability.enemy_mask is not None:
        bad.append('capability.attack / enemy_mask must be None (sampled only for Player 1)')
    if config.actions.ability:
        bad.append('actions.ability must be False (ability queries only cover Player 1 slots)')
    if bad:
        raise ValueError(
            'Unsupported in two-sided mode (opponent.kind=external): ' + '; '.join(bad)
        )


def symmetric_options(options: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Adapt `backend.pyref_options` for two-sided mode: expands `passive_from_map=True` to
    symmetric passive (`auto_acquire_owners=()`, `retreat_owners=(1, 2)`,
    `passive_from_map=False`)."""
    out = dict(options)
    if out.get('passive_from_map'):
        out.update(passive_from_map=False, auto_acquire_owners=(), retreat_owners=(1, 2))
    return out


def _both_or_neither(owners: Any) -> bool:
    return {int(o) for o in owners} & {1, 2} in ({1, 2}, set())


def check_options(options: Mapping[str, Any], map_id: str) -> None:
    """Verify symmetry of backend player options (non-default pyref / zgame fields).

    Raises `ValueError` unless `auto_acquire_owners` is None or covers both/neither player,
    `retreat_owners` covers both/neither player, `hit_slow_owner1_only` is False, `charge` owners
    for `map_id` cover both/neither player, and `passive_from_map` is False (`make_vec_env`
    expands it via `symmetric_options`).
    """
    from ..assets import load_charge_owners

    bad = []
    acq = options.get('auto_acquire_owners')
    if acq is not None and not _both_or_neither(acq):
        bad.append(f'auto_acquire_owners={tuple(acq)}')
    if not _both_or_neither(options.get('retreat_owners', ())):
        bad.append(f'retreat_owners={tuple(options["retreat_owners"])}')
    if options.get('hit_slow_owner1_only', False):
        bad.append('hit_slow_owner1_only=True')
    if options.get('charge', False) and not _both_or_neither(load_charge_owners(map_id)):
        bad.append(f'charge ({map_id} charge owners: {load_charge_owners(map_id)})')
    if options.get('passive_from_map', False):
        bad.append(
            'passive_from_map=True (convert to symmetric passive when passing a backend instance)'
        )
    if bad:
        raise ValueError(
            'Two-sided mode requires symmetric player options; asymmetric fields: '
            f'{", ".join(bad)}. Use a symmetric profile such as `sc2-symmetric`.'
        )


def _options(config: Any) -> dict[str, Any]:
    """Return non-default fields of a backend config (dataclass instance such as `PyrefConfig` or
    mapping)."""
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        default = type(config)()
        return {
            f.name: getattr(config, f.name)
            for f in dataclasses.fields(config)
            if getattr(config, f.name) != getattr(default, f.name)
        }
    return {}


def check_backend(backend: Any, map_id: str) -> None:
    """Check whether `backend` supports two-sided mode.

    SC2 raises `CapabilityError` (map triggers apply asymmetric settings such as Passive only to
    Player 1); pyref / zgame must pass `check_options`.
    """
    name = backend.capabilities.name
    if name == 'sc2':
        raise CapabilityError(
            'SC2 backend does not support two-sided mode (opponent.kind=external): '
            'map triggers apply asymmetric settings such as Passive only to Player 1'
        )
    check_options(_options(getattr(backend, 'config', None)), map_id)


# Mirror tolerance (tiles): midpoint offset from playable center and perpendicular centroid diff
MIRROR_TOL = 0.5


def _grid_symmetric(info: MapInfo, k: int) -> bool:
    """Check whether the playable grid is symmetric along axis `k` (0 = x): cell `i` matches
    `lo + hi - 1 - i` in pathability, and in terrain height wherever both cells are pathable
    (unpathable terrain height does not affect ground units)."""
    x0, y0, x1, y1 = (int(v) for v in info.playable)
    path = info.pathing[y0:y1, x0:x1]
    height = info.terrain_height[y0:y1, x0:x1]
    flip = (slice(None), slice(None, None, -1)) if k == 0 else (slice(None, None, -1), slice(None))
    p2, h2 = path[flip], height[flip]
    both = path & p2
    return bool((path == p2).all() and (height[both] == h2[both]).all())


def mirror_axis(
    spawns: Iterable[Spawn], info: MapInfo, overrides: Iterable[UnitOverride] = ()
) -> Literal['x', 'y'] | None:
    """Return mirror axis (`'x'` vertical midline, `'y'` horizontal midline) if mirrorable, else
    None.

    Requires matching `(unit_type, variant)` multisets and `unit_overrides` across owners 1 and 2,
    team centroids aligned primarily along one axis with midpoint within `MIRROR_TOL` of playable
    center and perpendicular separation within `MIRROR_TOL`, and symmetric playable terrain
    (`_grid_symmetric`).
    """
    spawns = tuple(spawns)
    side = {o: [s for s in spawns if s.owner == o] for o in (1, 2)}
    if not side[1] or sorted((s.unit_type, s.variant) for s in side[1]) != sorted(
        (s.unit_type, s.variant) for s in side[2]
    ):
        return None
    ov = {
        o: sorted(
            (u.unit_type, u.variant, u.health_max, u.shield_max) for u in overrides if u.owner == o
        )
        for o in (1, 2)
    }
    if ov[1] != ov[2]:
        return None
    c1, c2 = (np.mean([[s.x, s.y] for s in side[o]], axis=0) for o in (1, 2))
    d = np.abs(c2 - c1)
    k = 0 if d[0] >= d[1] else 1
    lo, hi = float(info.playable[k]), float(info.playable[k + 2])
    if abs((c1[k] + c2[k]) / 2 - (lo + hi) / 2) > MIRROR_TOL or d[1 - k] > MIRROR_TOL:
        return None
    if not _grid_symmetric(info, k):
        return None
    return 'x' if k == 0 else 'y'


def mirror_spawns(
    spawns: Iterable[Spawn], axis: Literal['x', 'y'], playable: tuple[float, ...]
) -> tuple[Spawn, ...]:
    """Return mirrored spawns: Player 1 unchanged first, followed by Player 2 mirrored across the
    playable midline (`lo + hi - coord` in float64)."""
    p1 = [s for s in spawns if s.owner == 1]
    x0, y0, x1, y1 = (float(v) for v in playable)
    if axis == 'x':
        p2 = [replace(s, owner=2, x=x0 + x1 - s.x) for s in p1]
    else:
        p2 = [replace(s, owner=2, y=y0 + y1 - s.y) for s in p1]
    return (*p1, *p2)


def seat_catalog(catalog: UnitCatalog, unit_types: np.ndarray) -> UnitCatalog:
    """Fit Player 2's `UnitCatalog` on its initial units; under non-`*_gen` `smac` encoding,
    replaces `roles` with Player 2's sorted standard unit type IDs if any are absent."""
    types = sorted({int(t) for t in np.asarray(unit_types).ravel()})
    smac = not catalog.standard and not catalog.generated
    if smac and any(t not in catalog.roles for t in types):
        catalog = replace(catalog, roles=tuple(types))
    return catalog.fit(unit_types)


class Seat:
    """Player 2's perspective components and per-episode state (`VecEnv._p2`).
    Allies = Player 2, enemies = Player 1."""

    def __init__(self, env: VecEnv) -> None:
        config, spec = env.config, env.spec
        ops = env._k.services
        self.env = env
        self.n_agents = env.n_enemies
        self.n_enemies = env.n_agents
        cc = config.catalog
        self.catalog = UnitCatalog.create(
            spec.map_type,
            spec.unit_type_bits,
            str(config.players.enemy_race),
            str(config.players.agent_race),
            ranges=cc.ranges,
            shoot_range=cc.shoot_range,
            sight_range=cc.sight_range,
            min_attack_range=cc.min_attack_range,
            ally_type_override=config.quirks.ally_type_override,
            type_id=cc.type_id,
            shield_max=cc.shield_max,
            role_offset=config.quirks.role_offset,
        )
        self._catalog0 = self.catalog
        self._row_catalogs: dict[int, UnitCatalog] = {}
        heal = config.actions.heal == 'mmm_rule' and self.catalog.heal_possible
        try:
            self.actions = ActionSpace(
                self.n_agents,
                self.n_enemies,
                config.actions,
                heal_enabled=heal,
                heal_map=self.catalog.heal_possible,
                ops=ops,
            )
        except ValueError as exc:
            raise ValueError(f'Two-sided mode: Player 2 action space is invalid ({exc})') from None
        self.n_actions = self.actions.n_actions
        self.obs_layout = ObsLayout.build(
            config, self.catalog, self.n_agents, self.n_enemies, self.n_actions
        )
        self.state_layout = StateLayout.build(
            config, self.catalog, self.obs_layout, self.n_agents, self.n_enemies, self.n_actions
        )
        b = env.batch_size
        self.view = UnitView(self.n_agents, self.n_enemies, config.units, owners=(2, 1))
        self.rewarder = BattleReward(b, self.n_agents, self.n_enemies, config.reward, ops)
        self.max_rewards = np.full(b, np.nan)
        vis = config.visibility
        self.spotter = (
            Spotter(b, self.n_agents, self.n_enemies, vis.prob_obs_enemy)
            if vis.spotter_gate
            else None
        )
        self.last_action = np.zeros((b, self.n_agents, self.n_actions))
        self._frame: Frame | None = None
        self._next: Frame | None = None  # Post-step unit view during step()
        self._ally_tables: AllyTables | None = None
        self._enemy_tables: EnemyTables | None = None
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------ Per-episode state
    def forget(self, b: int) -> None:
        """Reset `max_reward` and per-env catalog for env `b` when its per-env seed changes."""
        self.max_rewards[b] = np.nan
        self._row_catalogs.pop(b, None)

    def reset(self, world: WorldState, ids: list[int], episodes: dict[int, int]) -> None:
        """Mirror Player 1's reset bookkeeping for Player 2: catalog fit, slot assignment, frame
        and tables, `max_reward`, spotter, death tracking, and `last_action`."""
        env, k = self.env, self.env._k
        if self.catalog.min_code is None:
            rows = world.alive[ids[0]] & (world.owner[ids[0]] == 2)
            self.catalog = seat_catalog(self.catalog, world.unit_type[ids[0], rows])
        if self._catalog0.min_code is None:
            for b in ids:
                if env._env_seeds[b] is not None and (
                    episodes[b] == 0 or b not in self._row_catalogs
                ):
                    rows = world.alive[b] & (world.owner[b] == 2)
                    self._row_catalogs[b] = seat_catalog(self._catalog0, world.unit_type[b, rows])
        k.assign(self.view, world, ids, self.catalog.ally_code)
        frame = k.frame(self.view, world)
        self._ally_tables = self._ally_tables_for(frame)
        en = frame.enemies
        self._enemy_tables = k.enemy_tables(self.catalog, en.unit_type, en.variant, en.shield_max)
        k.fill_max_rewards(
            self.max_rewards, ids, self.n_enemies, env.config.reward, en.health_max, en.shield_max
        )
        if self.spotter is not None:
            self.spotter.reset(ids)
        self.rewarder.reset(ids)
        self.last_action[ids] = 0.0
        self._frame = frame
        self._cache = {}

    def _ally_tables_for(self, frame: Frame) -> AllyTables:
        a = frame.allies
        catalogs = [self._row_catalogs.get(b, self.catalog) for b in range(self.env.batch_size)]
        return self.env._k.ally_tables(catalogs, a.unit_type, a.variant, a.shield_max)

    def require(self) -> tuple[Frame, AllyTables, EnemyTables]:
        if self._frame is None or self._ally_tables is None or self._enemy_tables is None:
            raise RuntimeError('Must call reset() first')
        return self._frame, self._ally_tables, self._enemy_tables

    # ------------------------------------------------------------ Player 2 step lifecycle
    def parse(self, enemy_actions: Any) -> np.ndarray:
        """Convert `enemy_actions` to `int64[B, A2]`; required in two-sided mode."""
        if enemy_actions is None:
            raise ValueError(
                'Two-sided mode (opponent.kind=external) requires enemy_actions in step()'
            )
        acts = np.asarray(enemy_actions, dtype=np.int64)
        want = (self.env.batch_size, self.n_agents)
        if acts.size != want[0] * want[1]:
            raise ValueError(
                f'enemy_actions shape {acts.shape} does not match expected {want} '
                '(Player 2 agent count)'
            )
        return acts.reshape(want)

    def issue(self, acts: np.ndarray, active: np.ndarray, n_slots: int, cmd: Command) -> None:
        """Translate Player 2 actions into Player 1's `Command` (`to_command(out=cmd)`), prefixing
        validation errors with `enemy_actions: `, and record `last_action`."""
        frame, a_tab, _ = self.require()
        try:
            self.env._k.to_command(
                self.actions,
                acts,
                active,
                self.get_avail_actions(),
                frame,
                a_tab,
                n_slots,
                out=cmd,
            )
        except ValueError as exc:
            raise ValueError(f'enemy_actions: {exc}') from None
        self.last_action[active] = np.eye(self.n_actions)[acts[active]]

    def game_end(self, world: WorldState, cur: Frame, ally: AllyTables) -> np.ndarray:
        """Build Player 2's post-step frame and evaluate two-sided termination with Player 1's."""
        k = self.env._k
        self._next = k.frame(self.view, world)
        _, a_tab, _ = self.require()
        return k.game_end_dual(
            cur, ally, self._next, a_tab, medivac_rules=self.env.catalog.heal_possible
        )

    def settle(self, active: np.ndarray, end: np.ndarray, infos: list[dict[str, Any]]) -> None:
        """Update Player 2 spotter, rewards, and settlement after Player 1 settles (adding
        `battle_lost` and `reward_enemy` to `infos`)."""
        k = self.env._k
        prev, cur = self.require()[0], self._next
        assert cur is not None
        if self.spotter is not None:
            dead = (cur.allies.health == 0, cur.enemies.health == 0)
            k.spotter_after_step(self.spotter, *dead, np.flatnonzero(active))
        reward = k.reward(self.rewarder, prev, cur, active)
        k.settle_enemy(active, end, reward, self.max_rewards, self.env.config.reward, infos)
        self._frame, self._next = cur, None
        self._cache = {}

    # ------------------------------------------------------------ Observations
    def get_avail_actions(self) -> np.ndarray:
        if 'avail' not in self._cache:
            frame, a_tab, _ = self.require()
            env = self.env
            self._cache['avail'] = env._k.avail(
                self.actions,
                frame,
                a_tab,
                env.terrain,
                None,
                ability_choice=None,
                map_size=env.map_size,
            )
        return self._cache['avail']

    def context(self) -> FeatureContext:
        frame, a_tab, e_tab = self.require()
        env = self.env
        if env.config.actions.mask_by_range:
            true_avail = self.get_avail_actions()
        else:
            if 'true_avail' not in self._cache:
                self._cache['true_avail'] = env._k.avail(
                    self.actions,
                    frame,
                    a_tab,
                    env.terrain,
                    None,
                    mask_by_range=True,
                    ability_choice=None,
                    map_size=env.map_size,
                )
            true_avail = self._cache['true_avail']
        return FeatureContext(
            config=env.config,
            catalog=self.catalog,
            terrain=env.terrain,
            frame=frame,
            ally_tables=a_tab,
            enemy_tables=e_tab,
            avail=self.get_avail_actions(),
            last_action=self.last_action,
            episode_steps=env.episode_steps,
            episode_limit=env.episode_limit,
            true_target=true_avail[..., self.actions.n_no_attack :],
            ops=env._k.services,
        )

    def _obs_rng(self, b: int) -> np.random.Generator:
        return self.env.rng_stream(OBS_STREAM, b, int(self.env.episode_steps[b]))

    def get_obs(self) -> np.ndarray:
        if 'obs' not in self._cache:
            env = self.env
            ctx = self.context()
            if self.spotter is not None:
                streams = (
                    env.rng_streams(OBS_STREAM, env.episode_steps)
                    if env.reset_count.all()
                    else self._obs_rng
                )
                gate = env._k.observe_spotter(
                    ctx, self.spotter, streams, env.config.quirks.obs_shuffle
                )
                ctx = replace(ctx, gate=gate)
            self._cache['obs'] = env._k.encode_obs(ctx, self.obs_layout)
        return self._cache['obs']

    def get_state(self) -> np.ndarray:
        if 'state' not in self._cache:
            obs = self.get_obs() if self.state_layout.obs_instead is not None else None
            self._cache['state'] = self.env._k.encode_state(self.context(), self.state_layout, obs)
        return self._cache['state']
