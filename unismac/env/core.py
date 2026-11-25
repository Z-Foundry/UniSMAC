"""`VecEnv` (batched API) and `UnismacEnv` (upstream `MultiAgentEnv` API, `batch_size=1`).

Shape conventions: `B` = batch size, `A` = `n_agents`, `D` = obs dim, `S` = state dim,
`K` = `n_actions`.

Envs in a batch terminate independently: a finished env stays frozen (the backend still steps the
batch, while finished rows preserve their terminal snapshot and `step` returns `reward=0`,
`terminated=True`, `info={}`) until reset via `reset(env_ids=[...])`.

All randomness (SMACv2 procedural generation, `stochastic_attack`, spotter and obs order, scripted
opponents) comes from counter-based Philox streams (`unismac.env.rng`) keyed by
`(seed, stream, env_id)` with counter `(episode, frame)`: envs do not interfere with each other
and episodes are reproducible under a fixed seed. `seed=None` draws system entropy once.

Per-env seeds (`seeds` at construction or `reset(..., seeds=...)`): env `b` uses stream key
`(seeds[b], stream, 0)`, backend seed `seeds[b]`, and a per-env unit catalog fitted on its own
episode 0, making env `b` of a `B = K` batch bitwise-identical to a `B = 1` env with
`seed=seeds[b]`. Without `seeds`, env `b` uses `(seed, stream, b)` and backend seed `seed + b`.

Acceleration tiers (`accel`, see `unismac.env.accel`): system- and semantic-layer kernels dispatch
through `self._k` (Python reference or Rust) and semantic components receive `self._k.services`;
all tiers produce bitwise-identical results.

Two-sided mode (`opponent.kind='external'`, see `unismac.env.dual`): Player 2 is also controlled
by the caller via `step(actions, enemy_actions)`; `side=2` observations, states, masks, and
layouts use Player 2's perspective (`self._p2`) with the same rules as Player 1. `side=1` and
single-sided mode are unchanged.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Literal

import numpy as np

from ..assets import load_unit_overrides
from ..contract import (
    AbilityQuery,
    AgentPlayer,
    Backend,
    CapabilityError,
    ContractError,
    MapInfo,
    Scenario,
    Spawn,
    Variant,
    WorldState,
)
from . import accel as accel_mod
from . import rng as rng_mod
from .abilities import choose_ability
from .actions import ActionSpace, FovState
from .catalog import MEDIVAC, AllyTables, EnemyTables, UnitCatalog
from .config import EnvConfig
from .dual import Seat, check_backend, check_config, mirror_axis, mirror_spawns
from .features import FeatureContext, ObsLayout, StateLayout
from .generators import DEFAULT_ALLY_POS, DEFAULT_ENEMY_POS, UNIT_NAMES, ScenarioGenerator
from .opponents import make_opponent
from .reward import BattleReward
from .spotter import Spotter
from .terrain import TerrainGrids
from .units import Frame, UnitView

__all__ = ['UnismacEnv', 'VecEnv', 'resolve_map_ids']


def resolve_map_ids(config: EnvConfig, spec: Any, backend_name: str) -> str:
    """Resolve the effective `map_id` for `backend_name`."""
    tail = spec.map_id.split('/', 1)[1]
    directory = config.maps.dir
    if backend_name == 'sc2' and config.maps.sc2_dir is not None:
        directory = config.maps.sc2_dir
    return spec.map_id if directory is None else f'{directory}/{tail}'


def _per_env(values: Sequence[Any], n: int, what: str) -> list[Any]:
    out = list(values)
    if len(out) != n:
        raise ValueError(f'{what} has {len(out)} items, expected {n}')
    return out


class VecEnv:
    """Batched multi-agent environment.

    * `reset(env_ids=None, *, opponents=None, seeds=None) -> (obs float32[B, A, D],
      state float32[B, S])`
    * `step(actions int[B, A]) -> (reward float64[B], terminated bool[B], infos list[dict])`
    * `get_obs() -> float32[B, A, D]`, `get_state() -> float32[B, S]`,
      `get_avail_actions() -> int64[B, A, K]`
    * Read-only queries: `alive() -> bool[B, A]`, `game_loop() -> int64[B]`, `opponent_names()`,
      `opponent_errors()`; `close_backend()` closes the backend while preserving env state.
    * Two-sided mode (`opponent.kind='external'`): `step(actions, enemy_actions int[B, A2])`,
      adding `reward_enemy` (Player 2 reward) and `battle_lost` (Player 1 lost) to info;
      `get_obs(side=2)`, `get_state(side=2)`, `get_avail_actions(side=2)`, `alive(side=2)`, and
      `frame_for(2)` return Player 2's perspective, with dimensions from `obs_layout_for(side)`,
      `state_layout_for(side)`, `n_agents_for(side)`, and `n_actions_for(side)`. `side=1`
      (default) matches single-sided mode.

    `get_obs_size()` / `get_state_size()` return upstream-reported sizes (off by 1 when
    `quirks.timestep_pad` is enabled; see `unismac.env.features`), while `obs_dim` / `state_dim`
    are actual feature lengths.

    `seed` is batch-wide; `seeds` is per-env (length `batch_size`, mutually exclusive with `seed`).
    `accel` requests a tier (`'auto'`, `'python'`, `'services'`, `'rust'`); `self.accel` is the
    resolved tier.
    """

    def __init__(
        self,
        config: EnvConfig,
        spec: Any,
        backend: Backend,
        map_info: Any,
        *,
        seed: int | None = None,
        seeds: Sequence[int] | None = None,
        spawn_record: Any = None,
        accel: str = 'auto',
    ) -> None:
        self.accel = accel_mod.resolve(accel, config, spec)
        self._k = accel_mod.kernels(self.accel)
        ops = self._k.services
        self.config = config
        self.spec = spec
        self.backend = backend
        self.batch_size = int(backend.batch_size)
        if seed is not None and seeds is not None:
            raise ValueError(
                'Specify at most one of seed or seeds (seed is batch-wide; seeds is per-env)'
            )
        self.seed = seed
        self._rng_seed = rng_mod.entropy_seed() if seed is None else int(seed)
        # Per-env seeds; None entries fall back to batch seed (stream key (seed, stream, env_id)).
        self._env_seeds: list[int | None] = [None] * self.batch_size
        if seeds is not None:
            per = _per_env(seeds, self.batch_size, f'seeds (batch_size={self.batch_size})')
            self._env_seeds = [operator.index(s) for s in per]
        # Cached stream keys by stream name; invalidated when per-env seeds change.
        self._stream_keys: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        cap = config.capability
        if cap.health is not None:
            raise CapabilityError(
                'capability.health (stochastic_health) requires the backend to kill units '
                'mid-step, which is not yet supported by the contract'
            )
        self.procedural = cap.team_gen is not None
        if bool(getattr(spec, 'procedural', False)) and not self.procedural:
            raise ValueError(
                f'{spec.name} is a procedural scenario and requires capability.team_gen'
            )
        if self.procedural:
            self.n_agents = int(cap.n_units if cap.n_units is not None else spec.n_agents)
            self.n_enemies = int(cap.n_enemies if cap.n_enemies is not None else spec.n_enemies)
        else:
            self.n_agents = int(spec.n_agents)
            self.n_enemies = int(spec.n_enemies)
            for k, v in (('n_units', cap.n_units), ('n_enemies', cap.n_enemies)):
                want = self.n_agents if k == 'n_units' else self.n_enemies
                if v is not None and int(v) != want:
                    raise ValueError(
                        f'capability.{k}={v} does not match scenario {spec.name} ({want})'
                    )
            if cap.start_positions is not None:
                raise ValueError(
                    'capability.start_positions requires a procedural scenario (team_gen)'
                )
        self.generator = (
            ScenarioGenerator(cap, self.n_agents, self.n_enemies) if cap.enabled else None
        )
        limit = config.termination.episode_limit
        self.episode_limit = int(spec.limit if limit is None else limit)
        players = config.players
        cc = config.catalog
        self.catalog = UnitCatalog.create(
            spec.map_type,
            spec.unit_type_bits,
            str(players.agent_race),
            str(players.enemy_race),
            ranges=cc.ranges,
            shoot_range=cc.shoot_range,
            sight_range=cc.sight_range,
            min_attack_range=cc.min_attack_range,
            ally_type_override=config.quirks.ally_type_override,
            type_id=cc.type_id,
            shield_max=cc.shield_max,
            role_offset=config.quirks.role_offset,
        )
        # min_code (upstream episode 0 init_units): batch shares self.catalog fitted on first
        # reset; per-env seeded envs fit their own episode 0 catalog (_row_catalogs) to match B=1.
        self._catalog0 = self.catalog
        self._row_catalogs: dict[int, UnitCatalog] = {}
        heal = config.actions.heal == 'mmm_rule' and self.catalog.heal_possible
        vis = config.visibility
        n_fov = int(vis.n_fov) if vis.model == 'cone' else 0
        if vis.model == 'cone' and n_fov < 1:
            raise ValueError('visibility.model=cone requires n_fov >= 1')
        self.actions = ActionSpace(
            self.n_agents,
            self.n_enemies,
            config.actions,
            heal_enabled=heal,
            heal_map=self.catalog.heal_possible,
            n_fov=n_fov,
            ops=ops,
        )
        self.n_actions = self.actions.n_actions
        o = config.obs
        if (o.pathing_grid or o.terrain_height) and float(config.actions.move_amount) != int(
            config.actions.move_amount
        ):
            raise ValueError('obs.pathing_grid / terrain_height requires an integer move_amount')
        self.obs_layout = ObsLayout.build(
            config, self.catalog, self.n_agents, self.n_enemies, self.n_actions
        )
        self.state_layout = StateLayout.build(
            config, self.catalog, self.obs_layout, self.n_agents, self.n_enemies, self.n_actions
        )
        self.terrain = TerrainGrids.from_map_info(
            map_info, height_flip=config.quirks.terrain_height_flip, ops=ops
        )
        self.opponent = make_opponent(config.opponent)
        # Two-sided mode (opponent.kind=external): validate config and backend symmetry upfront.
        self.dual = config.opponent.kind == 'external'
        if self.dual:
            check_config(config)
            check_backend(backend, map_info.map_id)
        self.map_id = map_info.map_id
        self._playable = tuple(float(v) for v in map_info.playable)
        # Map bounds for ability point targets (upstream map_info.map_size).
        self.map_size = (int(map_info.width), int(map_info.height))
        self._players = (
            AgentPlayer(config.players.agent_race),  # type: ignore[arg-type]
            self.opponent.player(str(config.players.enemy_race)),
        )
        # Map-specific unit stat overrides applied to every episode's Scenario.
        self.unit_overrides = (
            load_unit_overrides(map_info.map_id) if config.backend.unit_overrides == 'asset' else ()
        )
        # Mirror spawn axis: set only in two-sided mode with spawn_mirror='auto', explicit spawns,
        # and a mirrorable map (see dual.mirror_axis).
        self.mirror: Literal['x', 'y'] | None = None
        self.scenario: Scenario | None = (
            None if self.procedural else self._build_scenario(map_info, spawn_record)
        )
        if self.procedural and config.backend.spawn_mode != 'explicit':
            raise ValueError('Procedural scenarios require backend.spawn_mode=explicit')
        self._check_capabilities(heal)

        self.view = UnitView(self.n_agents, self.n_enemies, config.units)
        b = self.batch_size
        self.rewarder = BattleReward(b, self.n_agents, self.n_enemies, config.reward, ops)
        # Per-env max_reward computed on first reset (upstream episode 0 init_units); NaN before.
        self.max_rewards = np.full(b, np.nan)
        # Per-env reset count (= RNG episode index + 1); reset to 0 when per-env seed changes.
        self.reset_count = np.zeros(b, np.int64)
        # Most recent Scenario passed to backend per env_id (for inspecting procedural episodes).
        self.last_scenarios: dict[int, Scenario] = {}
        self.fov = FovState(b, self.n_agents, n_fov) if n_fov else None
        self.spotter = (
            Spotter(b, self.n_agents, self.n_enemies, vis.prob_obs_enemy)
            if vis.spotter_gate
            else None
        )
        self.attack_prob = np.ones((b, self.n_agents)) if cap.attack is not None else None
        self.enemy_mask = (
            np.zeros((b, self.n_agents, self.n_enemies), np.int64)
            if cap.enemy_mask is not None
            else None
        )
        self.episode_steps = np.zeros(b, np.int64)
        self.episode_count = np.zeros(b, np.int64)
        self.last_action = np.zeros((b, self.n_agents, self.n_actions))
        self.done = np.ones(b, bool)
        self.battles_won = np.zeros(b, np.int64)
        self.battles_game = np.zeros(b, np.int64)
        self.timeouts = np.zeros(b, np.int64)
        self._world: WorldState | None = None
        self._frame: Frame | None = None
        self._ally_tables: AllyTables | None = None
        self._enemy_tables: EnemyTables | None = None
        self._cache: dict[str, Any] = {}
        # Player 2 perspective components and per-episode state (two-sided mode only).
        self._p2: Seat | None = Seat(self) if self.dual else None
        self.opponent.bind(self)

    # ------------------------------------------------------------ Construction
    def _build_scenario(self, map_info: MapInfo, record: Any) -> Scenario:
        cfg = self.config
        map_id = map_info.map_id
        players = self._players
        n_slots = self.n_agents + self.n_enemies
        if cfg.backend.spawn_mode == 'map_trigger':
            return Scenario(
                map_id,
                players=players,
                spawn_mode='map_trigger',
                n_slots=n_slots,
                unit_overrides=self.unit_overrides,
            )
        if record is None:
            raise ValueError(
                'spawn_mode=explicit requires a spawn record (unismac.scenarios.spawn_record)'
            )
        spawns = tuple(record.spawns)
        if cfg.backend.spawn_variant == 'std':
            spawns = tuple(replace(s, variant=int(Variant.STD)) for s in spawns)
        elif cfg.backend.spawn_variant == 'smac':
            spawns = tuple(
                replace(
                    s,
                    variant=int(Variant.RL)
                    if (s.owner == 1 and s.unit_type == MEDIVAC)
                    else int(Variant.STD),
                )
                for s in spawns
            )
        counts = (sum(s.owner == 1 for s in spawns), sum(s.owner == 2 for s in spawns))
        if counts != (self.n_agents, self.n_enemies):
            raise ValueError(f'Spawn record unit counts {counts} do not match scenario')
        spawns = tuple(sorted(spawns, key=lambda s: s.owner))  # Slots: allies first
        if cfg.backend.spawn_clamp:
            x0, y0, x1, y1 = self._playable
            spawns = tuple(
                replace(s, x=min(max(s.x, x0), x1), y=min(max(s.y, y0), y1)) for s in spawns
            )
        spawns = tuple(Spawn(**vars(s)) for s in spawns)
        if self.dual and cfg.backend.spawn_mirror == 'auto':
            self.mirror = mirror_axis(spawns, map_info, self.unit_overrides)
        if self.mirror is not None:
            spawns = mirror_spawns(spawns, self.mirror, map_info.playable)
        return Scenario(
            map_id,
            players=players,
            spawns=spawns,
            unit_overrides=self.unit_overrides,
            mirror=self.mirror,
        )

    def _generated_scenario(self, draw: Any) -> Scenario:
        """Build one procedural episode's `Scenario`: allies first, then enemies in team order
        (matching upstream `DebugCreateUnit` order). Ally variant: `recorded` -> `*_RL`, `std` ->
        standard, `smac` -> `Variant.RL` only for Player 1 Medivacs and `Variant.STD` otherwise."""
        sv = self.config.backend.spawn_variant
        spawns: list[Spawn] = []
        for owner, team, pos, default in (
            (1, draw.ally_team, draw.ally_pos, DEFAULT_ALLY_POS),
            (2, draw.enemy_team, draw.enemy_pos, DEFAULT_ENEMY_POS),
        ):
            for i, name in enumerate(team):
                x, y = default if pos is None else (float(pos[i, 0]), float(pos[i, 1]))
                if self.config.backend.spawn_clamp:
                    x0, y0, x1, y1 = self._playable
                    x, y = min(max(x, x0), x1), min(max(y, y0), y1)
                ut = UNIT_NAMES[name]
                var = (
                    Variant.RL
                    if owner == 1 and (sv == 'recorded' or (sv == 'smac' and ut == MEDIVAC))
                    else Variant.STD
                )
                spawns.append(Spawn(owner, ut, x, y, int(var)))
        return Scenario(
            self.map_id,
            players=self._players,
            spawns=tuple(spawns),
            unit_overrides=self.unit_overrides,
        )

    def _check_capabilities(self, heal: bool) -> None:
        caps = self.backend.capabilities
        needs: dict[str, object] = {}
        if self.config.backend.spawn_mode == 'map_trigger' and not self.procedural:
            needs['map_trigger'] = True
        if self.config.opponent.kind == 'engine_ai':
            needs['engine_ai'] = True
        if heal:
            needs['abilities'] = True
        if self.unit_overrides:
            needs['unit_overrides'] = True
        if self.mirror is not None:
            needs['mirror_spawn'] = True
        caps.require(**needs)
        if self.config.actions.ability and not (
            caps.ability_query and isinstance(self.backend, AbilityQuery)
        ):
            raise CapabilityError(
                f'Backend {caps.name} does not support ability queries (AbilityQuery), so the '
                'ability segment is unavailable; set use_ability=False (actions.ability=False)'
            )
        if caps.max_batch is not None and self.batch_size > caps.max_batch:
            raise CapabilityError(
                f'Backend {caps.name} supports at most batch_size={caps.max_batch}, '
                f'got {self.batch_size}'
            )

    # ------------------------------------------------------------ Dimensions
    @property
    def obs_dim(self) -> int:
        return self.obs_layout.size

    @property
    def state_dim(self) -> int:
        return self.state_layout.size

    def get_obs_size(self) -> int:
        return self.obs_layout.reported_size

    def get_state_size(self) -> int:
        return self.state_layout.reported_size

    def get_total_actions(self) -> int:
        return self.n_actions

    def get_env_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            'state_shape': self.get_state_size(),
            'obs_shape': self.get_obs_size(),
        }
        if getattr(self.spec, 'benchmark', None) == 'smacv2':
            info['cap_shape'] = self.get_cap_size()
        info.update(
            {
                'n_actions': self.get_total_actions(),
                'n_agents': self.n_agents,
                'episode_limit': self.episode_limit,
                'agent_features': list(self.state_layout.ally),
                'enemy_features': list(self.state_layout.enemy),
            }
        )
        return info

    def get_cap_size(self) -> int:
        """SMACv2 `get_cap_size`: hit probability + type bits."""
        return int(self.attack_prob is not None) + self.catalog.unit_type_bits

    def get_capabilities(self) -> np.ndarray:
        """Return `float32[B, A * cap_size]` (SMACv2 `get_capabilities` flattened across agents)."""
        _, _, a_tab, _ = self._require()
        cols = []
        if self.attack_prob is not None:
            cols.append(self.attack_prob[..., None])
        bits = self.catalog.unit_type_bits
        if bits:
            cols.append((a_tab.type_id[..., None] == np.arange(bits)).astype(np.float64))
        if not cols:
            return np.zeros((self.batch_size, 0), np.float32)
        cap = np.concatenate(cols, axis=-1)
        return cap.reshape(self.batch_size, -1).astype(np.float32)

    @property
    def max_reward(self) -> float | None:
        """`max_reward` of env 0 (upstream `max_reward` when `batch_size=1`); None before reset."""
        v = self.max_rewards[0]
        return None if np.isnan(v) else float(v)

    def _require(self) -> tuple[WorldState, Frame, AllyTables, EnemyTables]:
        if self._world is None or self._frame is None:
            raise RuntimeError('Must call reset() first')
        assert self._ally_tables is not None and self._enemy_tables is not None
        return self._world, self._frame, self._ally_tables, self._enemy_tables

    # ------------------------------------------------------------ Seeds & RNG streams
    @property
    def env_seeds(self) -> tuple[int | None, ...]:
        """Per-env seeds (`seeds` at construction or updated on `reset`); None uses batch `seed`."""
        return tuple(self._env_seeds)

    def rng_stream(self, name: str, env_id: int, frame: int = 0) -> np.random.Generator:
        """Return a fresh `Generator` for stream `name` of env `env_id`'s current episode at
        `frame` (`unismac.env.rng.stream`).

        Per-env seeded envs use `(env_seed, name, 0, episode, frame)` (matching a `B=1` env);
        other envs use `(batch_seed, name, env_id, episode, frame)`.
        """
        s = self._env_seeds[env_id]
        seed, slot = (self._rng_seed, env_id) if s is None else (s, 0)
        return rng_mod.stream(seed, name, slot, int(self.reset_count[env_id]) - 1, frame)

    def rng_streams(self, name: str, frames: np.ndarray) -> rng_mod.Streams:
        """Return batched `Streams` for `name`: `streams(b)` matches
        `rng_stream(name, b, frames[b])` (unreset envs get episode 0 and should not be drawn from).
        Keys and counters are packed as arrays for Rust kernels."""
        if name not in self._stream_keys:
            keys = [
                rng_mod.stream_key(self._rng_seed, name, b)
                if s is None
                else rng_mod.stream_key(s, name, 0)
                for b, s in enumerate(self._env_seeds)
            ]
            k0, k1 = (np.array(k, np.uint64) for k in zip(*keys, strict=True))
            self._stream_keys[name] = (k0, k1)
        k0, k1 = self._stream_keys[name]
        episode = np.maximum(self.reset_count - 1, 0).astype(np.uint64)
        return rng_mod.Streams(k0, k1, np.asarray(frames).astype(np.uint64), episode)

    def _backend_seed(self, b: int) -> int | None:
        s = self._env_seeds[b]
        if s is not None:
            return s
        return None if self.seed is None else self.seed + b

    def _ally_tables_for(self, frame: Frame) -> AllyTables:
        """Build ally tables, using per-env catalogs where per-env `min_code` is active."""
        a = frame.allies
        catalogs = [self._row_catalogs.get(b, self.catalog) for b in range(self.batch_size)]
        return self._k.ally_tables(catalogs, a.unit_type, a.variant, a.shield_max)

    # ------------------------------------------------------------ Main loop
    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        *,
        opponents: Sequence[str | None] | None = None,
        seeds: Sequence[int | None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset `env_ids` (all envs by default) and return batch `(obs, state)`.

        * `opponents`: script names aligned 1-to-1 with `env_ids` (or all envs), overriding the
          `opponent` stream draw (`None` draws normally). Valid only for `scripted` opponents;
          names must belong to the scenario's script pool (`unismac.env.scripted.script_pool`),
          else raises `ValueError`. Does not affect other envs or later episodes.
        * `seeds`: new per-env seeds aligned 1-to-1 with `env_ids`. Integer entries switch that
          env to the new per-env seed and reset its episode counter to 0 (matching a fresh `B=1`
          env); `None` entries keep their current seed.
        """
        b_size = self.batch_size
        ids = list(range(b_size)) if env_ids is None else [int(i) for i in env_ids]
        if len(set(ids)) != len(ids) or not all(0 <= i < b_size for i in ids):
            raise ContractError(f'env_ids={ids} has duplicates or values outside [0, {b_size})')
        names: list[str | None] | None = None
        if opponents is not None:
            names = _per_env(opponents, len(ids), 'opponents (must match env_ids 1-to-1)')
            self.opponent.check_names(names)
            if all(n is None for n in names):
                names = None
        if seeds is not None:
            new = _per_env(seeds, len(ids), 'seeds (must match env_ids 1-to-1)')
            new = [None if s is None else operator.index(s) for s in new]
            for b, s in zip(ids, new, strict=True):
                if s is not None:
                    # Match a freshly created env: reset episode counter, max_reward, and catalog.
                    self._env_seeds[b] = s
                    self._stream_keys = {}
                    self.reset_count[b] = 0
                    self.max_rewards[b] = np.nan
                    self._row_catalogs.pop(b, None)
                    if self._p2 is not None:
                        self._p2.forget(b)
        backend_seeds = [self._backend_seed(b) for b in ids]
        episodes: dict[int, int] = {}
        for b in ids:
            episodes[b] = int(self.reset_count[b])
            self.reset_count[b] += 1
        draws: list[Any] = [None] * len(ids)
        if self.generator is not None:
            gens = self.rng_streams('generators', np.zeros(b_size, np.int64))
            draws = self._k.draw(self.generator, gens, ids)
        scenarios: list[Scenario] = []
        for b, draw in zip(ids, draws, strict=True):
            if draw is not None:
                if self.attack_prob is not None:
                    assert draw.attack_prob is not None
                    self.attack_prob[b] = draw.attack_prob
                if self.enemy_mask is not None:
                    assert draw.enemy_mask is not None
                    self.enemy_mask[b] = draw.enemy_mask
            if self.procedural:
                scenarios.append(self._generated_scenario(draw))
            else:
                assert self.scenario is not None
                scenarios.append(self.scenario)
        self.last_scenarios = dict(zip(ids, scenarios, strict=True))
        fresh = self.backend.reset(
            scenarios,
            env_ids=None if env_ids is None else ids,
            seeds=backend_seeds,
        )
        if self._world is None or env_ids is None:
            self._world = fresh
        else:
            self._k.copy_rows(self._world, fresh, np.asarray(ids))
        world = self._world
        if self.catalog.min_code is None:
            rows = world.alive[ids[0]] & (world.owner[ids[0]] == 1)
            self.catalog = self.catalog.fit(world.unit_type[ids[0], rows])
        if self._catalog0.min_code is None:
            for b in ids:
                if self._env_seeds[b] is not None and (
                    episodes[b] == 0 or b not in self._row_catalogs
                ):
                    rows = world.alive[b] & (world.owner[b] == 1)
                    self._row_catalogs[b] = self._catalog0.fit(world.unit_type[b, rows])
        self._k.assign(self.view, world, ids, self.catalog.ally_code)
        frame = self._k.frame(self.view, world)
        self._ally_tables = self._ally_tables_for(frame)
        en = frame.enemies
        self._enemy_tables = self._k.enemy_tables(
            self.catalog, en.unit_type, en.variant, en.shield_max
        )
        self._k.fill_max_rewards(
            self.max_rewards, ids, self.n_enemies, self.config.reward, en.health_max, en.shield_max
        )
        if self.fov is not None:
            self.fov.reset(ids)
        if self.spotter is not None:
            self.spotter.reset(ids)
        self.rewarder.reset(ids)
        self.episode_steps[ids] = 0
        self.last_action[ids] = 0.0
        self.done[ids] = False
        if self._p2 is not None:
            self._p2.reset(world, ids, episodes)
        self._k.opponent_reset(self.opponent, world, ids, names)
        self._frame = frame
        self._cache = {}
        return self.get_obs(), self.get_state()

    def step(
        self, actions: Any, enemy_actions: Any = None
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Advance one environment step. `enemy_actions` (`int[B, A2]`, Player 2 actions) is
        required in two-sided mode and forbidden otherwise."""
        world, frame, a_tab, e_tab = self._require()
        acts = np.asarray(actions, dtype=np.int64).reshape(self.batch_size, self.n_agents)
        p2 = self._p2
        if p2 is None and enemy_actions is not None:
            raise ValueError(
                'enemy_actions is only valid in two-sided mode (opponent.kind=external)'
            )
        acts2 = None if p2 is None else p2.parse(enemy_actions)
        active = ~self.done
        attack_rng = None
        if self.attack_prob is not None:
            attack_rng = self.rng_streams('attack', self.episode_steps)
        cmd = self._k.to_command(
            self.actions,
            acts,
            active,
            self.get_avail_actions(),
            frame,
            a_tab,
            world.n_slots,
            fov=self.fov,
            attack_prob=self.attack_prob,
            attack_rng=attack_rng,
            ability_choice=self.ability_choice(),
        )
        if p2 is not None and acts2 is not None:
            p2.issue(acts2, active, world.n_slots, cmd)
        self._k.opponent_issue(self.opponent, world, cmd, active)
        eye = np.eye(self.n_actions)
        self.last_action[active] = eye[acts[active]]
        if active.any():
            rows = np.flatnonzero(active)
            self._k.advance(self.backend, cmd, self.config.backend.step_mul, world, rows)
        self.episode_steps[active] += 1
        cur = self._k.frame(self.view, world)
        if p2 is None:
            end = self._k.game_end(
                cur, a_tab, e_tab, self.config.termination, medivac_rules=self.catalog.heal_possible
            )
        else:
            end = p2.game_end(world, cur, a_tab)
        if self.spotter is not None:
            self._k.spotter_after_step(
                self.spotter,
                cur.allies.health == 0,
                cur.enemies.health == 0,
                np.flatnonzero(active),
            )
        reward = self._k.reward(self.rewarder, frame, cur, active)
        reward, terminated, infos = self._k.settle(self, active, end, cur, reward)
        if p2 is not None:
            p2.settle(active, end, infos)
        self._frame = cur
        self._cache = {}
        return reward, terminated, infos

    # ------------------------------------------------------------ Observations
    def _context(self) -> FeatureContext:
        _, frame, a_tab, e_tab = self._require()
        if self.config.actions.mask_by_range:
            true_avail = self.get_avail_actions()
        else:
            if 'true_avail' not in self._cache:
                self._cache['true_avail'] = self._k.avail(
                    self.actions,
                    frame,
                    a_tab,
                    self.terrain,
                    self.fov,
                    mask_by_range=True,
                    ability_choice=self.ability_choice(),
                    map_size=self.map_size,
                )
            true_avail = self._cache['true_avail']
        return FeatureContext(
            config=self.config,
            catalog=self.catalog,
            terrain=self.terrain,
            frame=frame,
            ally_tables=a_tab,
            enemy_tables=e_tab,
            avail=self.get_avail_actions(),
            last_action=self.last_action,
            episode_steps=self.episode_steps,
            episode_limit=self.episode_limit,
            true_target=true_avail[..., self.actions.n_no_attack :],
            fov=None if self.fov is None else self.fov.dirs.copy(),
            fov_angle=0.0 if self.fov is None else self.fov.angle,
            attack_prob=self.attack_prob,
            enemy_mask=self.enemy_mask,
            ops=self._k.services,
        )

    def _side2(self, side: int) -> Seat:
        """Return Player 2's `Seat` when `side == 2` in two-sided mode, else raise `ValueError`."""
        if side == 2 and self._p2 is not None:
            return self._p2
        if side == 2:
            raise ValueError('side=2 is only valid in two-sided mode (opponent.kind=external)')
        raise ValueError(f'side must be 1 or 2, got {side!r}')

    def n_agents_for(self, side: int) -> int:
        """Return agent count for `side` (`side=2` is Player 2's unit count in two-sided mode)."""
        return self.n_agents if side == 1 else self._side2(side).n_agents

    def n_actions_for(self, side: int) -> int:
        """Return action count for `side` (sides may differ in target count, e.g. `5m_vs_6m`)."""
        return self.n_actions if side == 1 else self._side2(side).n_actions

    def obs_layout_for(self, side: int) -> ObsLayout:
        return self.obs_layout if side == 1 else self._side2(side).obs_layout

    def state_layout_for(self, side: int) -> StateLayout:
        return self.state_layout if side == 1 else self._side2(side).state_layout

    def get_avail_actions(self, side: int = 1) -> np.ndarray:
        """Return `int64[B, A, K]`; `side=2` returns Player 2's `[B, A2, K2]` (two-sided mode)."""
        if side != 1:
            return self._side2(side).get_avail_actions()
        if 'avail' not in self._cache:
            _, frame, a_tab, _ = self._require()
            self._cache['avail'] = self._k.avail(
                self.actions,
                frame,
                a_tab,
                self.terrain,
                self.fov,
                ability_choice=self.ability_choice(),
                map_size=self.map_size,
            )
        return self._cache['avail']

    def query_abilities(self) -> list[list[list[int]]] | None:
        """Return available native ability IDs (ascending) per env and slot for the current frame,
        or None if unsupported by the backend. Cached once per frame (matching upstream)."""
        if 'abilities' not in self._cache:
            backend = self.backend
            ok = self.backend.capabilities.ability_query and isinstance(backend, AbilityQuery)
            self._cache['abilities'] = backend.query_abilities() if ok else None  # type: ignore[union-attr]
        return self._cache['abilities']

    def ability_choice(self) -> np.ndarray | None:
        """Return `int64[B, A]` selected ability per agent (`choose_ability`), or None if the
        ability segment is disabled."""
        if not self.actions.n_ability:
            return None
        if 'ability_choice' not in self._cache:
            _, frame, _, _ = self._require()
            abilities = self.query_abilities()
            assert abilities is not None
            slots = frame.allies.slot
            out = np.zeros(slots.shape, np.int64)
            for b in range(slots.shape[0]):
                for a in range(slots.shape[1]):
                    out[b, a] = choose_ability(abilities[b][int(slots[b, a])])
            self._cache['ability_choice'] = out
        return self._cache['ability_choice']

    def _obs_rng(self, b: int) -> np.random.Generator:
        return self.rng_stream('obs_shuffle', b, int(self.episode_steps[b]))

    def get_obs(self, side: int = 1) -> np.ndarray:
        """Return `float32[B, A, obs_dim]`; `side=2` returns Player 2's `[B, A2, obs_dim2]`
        (two-sided mode).

        When spotter gating is enabled, updates gates per env in agent order (shuffled when
        `quirks.obs_shuffle` is on) via `features.observe_spotter` using the frame's `obs_shuffle`
        stream (`obs_shuffle_p2` for Player 2); cached once per frame.
        """
        if side != 1:
            return self._side2(side).get_obs()
        if 'obs' not in self._cache:
            ctx = self._context()
            if self.spotter is not None:
                # If any env has not been reset yet, fetch per-env via rng_stream so it raises.
                streams = (
                    self.rng_streams('obs_shuffle', self.episode_steps)
                    if self.reset_count.all()
                    else self._obs_rng
                )
                gate = self._k.observe_spotter(
                    ctx, self.spotter, streams, self.config.quirks.obs_shuffle
                )
                ctx = replace(ctx, gate=gate)
            self._cache['obs'] = self._k.encode_obs(ctx, self.obs_layout)
        return self._cache['obs']

    def get_state(self, side: int = 1) -> np.ndarray:
        """Return `float32[B, state_dim]`; `side=2` returns Player 2's view (two-sided mode)."""
        if side != 1:
            return self._side2(side).get_state()
        if 'state' not in self._cache:
            obs = self.get_obs() if self.state_layout.obs_instead is not None else None
            self._cache['state'] = self._k.encode_state(self._context(), self.state_layout, obs)
        return self._cache['state']

    @property
    def world(self) -> WorldState:
        """Current batched `WorldState` (read-only view for debugging)."""
        return self._require()[0]

    @property
    def frame(self) -> Frame:
        """Current `Frame` (allies / enemies sorted by index, `[B, N]`; read-only, for heuristics
        and evaluation)."""
        return self._require()[1]

    def frame_for(self, side: int) -> Frame:
        """Return `Frame` from `side`'s perspective (`side=2`: allies = Player 2, two-sided)."""
        return self._require()[1] if side == 1 else self._side2(side).require()[0]

    @property
    def healers(self) -> np.ndarray:
        """Return `bool[B, A]` indicating agents whose target segment heals allies (Medivacs)."""
        return self.actions.heals(self._require()[2])

    def alive(self, side: int = 1) -> np.ndarray:
        """Return `bool[B, A]` (fresh array) of alive agents (`frame.allies.health > 0`, matching
        obs, rewards, and `dead_allies`); `side=2` is Player 2's (two-sided mode)."""
        return self.frame_for(side).allies.health > 0

    def game_loop(self) -> np.ndarray:
        """Return `int64[B]` (fresh array) of current game loops (`world.game_loop`)."""
        return self._require()[0].game_loop.copy()

    def opponent_names(self) -> list[str | None]:
        """Return current episode script name per env (`opponent.names` for scripted opponents);
        None for other opponents (including two-sided mode) or unreset envs."""
        names: dict[int, str] = getattr(self.opponent, 'names', None) or {}
        return [names.get(b) for b in range(self.batch_size)]

    def opponent_errors(self) -> np.ndarray:
        """Return `int64[B]` (fresh array) of cumulative steps where a scripted opponent raised an
        exception (`opponent.errors`); 0 for other opponents."""
        errors: dict[int, int] = getattr(self.opponent, 'errors', None) or {}
        return np.array([errors.get(b, 0) for b in range(self.batch_size)], np.int64)

    def close_backend(self) -> None:
        """Close the backend while retaining environment state (config, RNG episode counters,
        per-env seeds, and stats).

        All envs are marked done (`step` returns 0, True, `{}`); the next full `reset()` restarts
        the backend (new SC2 session or rebuilt pyref/zgame batch; partial reset is invalid because
        other slots would be empty). `alive()` and `game_loop()` retain the last pre-close frame.
        """
        self.backend.close()
        self.done[:] = True

    def get_stats(self) -> dict[str, Any]:
        won, games = int(self.battles_won.sum()), int(self.battles_game.sum())
        return {
            'battles_won': won,
            'battles_game': games,
            'battles_draw': int(self.timeouts.sum()),
            'win_rate': won / games if games else 0.0,
            'timeouts': int(self.timeouts.sum()),
            'restarts': 0,
        }

    def close(self) -> None:
        self.backend.close()


class UnismacEnv:
    """Upstream `MultiAgentEnv` wrapper (`batch_size=1`), matching SMAC `StarCraft2Env`:

    * `reset() -> (obs: list[float32[D]], state: float32[S])`
    * `step(actions) -> (reward: float, terminated: bool, info: dict)`
    * `get_obs() -> list[float32[D]]`, `get_avail_actions() -> list[list[int]]`, etc.

    Call `reset()` before stepping again after termination (stepping while done returns
    `0, True, {}`).
    """

    def __init__(self, vec: VecEnv) -> None:
        if vec.batch_size != 1:
            raise ValueError('UnismacEnv only wraps a VecEnv with batch_size=1')
        self.vec = vec
        self.n_agents = vec.n_agents
        self.n_enemies = vec.n_enemies
        self.episode_limit = vec.episode_limit
        self.n_actions = vec.n_actions
        self.config = vec.config

    def reset(self) -> tuple[list[np.ndarray], np.ndarray]:
        self.vec.reset()
        return self.get_obs(), self.get_state()

    def step(self, actions: Any, enemy_actions: Any = None) -> tuple[float, bool, dict[str, Any]]:
        """Step the environment; `enemy_actions` provides Player 2 actions in two-sided mode."""
        acts = np.array([int(a) for a in actions], dtype=np.int64)[None]
        if enemy_actions is None:
            reward, terminated, infos = self.vec.step(acts)
        else:
            acts2 = np.array([int(a) for a in enemy_actions], dtype=np.int64)[None]
            reward, terminated, infos = self.vec.step(acts, acts2)
        return float(reward[0]), bool(terminated[0]), infos[0]

    def get_obs(self, side: int = 1) -> list[np.ndarray]:
        return list(self.vec.get_obs(side)[0])

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        return self.vec.get_obs()[0, agent_id]

    def get_state(self, side: int = 1) -> np.ndarray:
        return self.vec.get_state(side)[0]

    def get_avail_actions(self, side: int = 1) -> list[list[int]]:
        return self.vec.get_avail_actions(side)[0].tolist()

    def get_avail_agent_actions(self, agent_id: int) -> list[int]:
        return self.vec.get_avail_actions()[0, agent_id].tolist()

    def get_obs_size(self) -> int:
        return self.vec.get_obs_size()

    def get_state_size(self) -> int:
        return self.vec.get_state_size()

    def get_total_actions(self) -> int:
        return self.vec.get_total_actions()

    def get_env_info(self) -> dict[str, Any]:
        return self.vec.get_env_info()

    def get_cap_size(self) -> int:
        return self.vec.get_cap_size()

    def get_capabilities(self) -> np.ndarray:
        return self.vec.get_capabilities()[0]

    @property
    def max_reward(self) -> float | None:
        return self.vec.max_reward

    def get_stats(self) -> dict[str, Any]:
        return self.vec.get_stats()

    def seed(self) -> int | None:
        return self.vec.seed

    def close(self) -> None:
        self.vec.close()

    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError('Rendering is not supported in unismac')

    def save_replay(self) -> None:
        raise NotImplementedError('Saving replays is not supported in unismac')
