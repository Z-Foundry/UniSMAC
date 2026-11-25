"""Observation and global state feature blocks with batched `[B, A]` encoding.

Observation (per agent) = `Move` + `Enemy` x E + `Ally` x (A-1) + `Own` (+ `Timestep`).
State = `AllyState` x A + `EnemyState` x E (+ `LastAction` x A) (+ `Timestep`), or concatenated
observations when `state.obs_instead_of_state` is enabled. Block `names()` defines dimensions.

Element-wise parity with upstream SMAC / SMACv2: computed in `float64` (matching upstream Python
floats) and cast to `float32` at the end.

* `quirks.timestep_pad`: when `obs.timestep_number` is True, appends a constant-zero feature to
  `Own` and reports `get_obs_size()` without the `Timestep` block (off by 1 from actual length).
* SMACv2 blocks: `Enemy` `available` uses range-masked target availability unaffected by
  `action_mask` (upstream `get_true_avail_agent_actions`); `enemy_mask` zeroes health, shield, and
  type; `Ally` includes `attack_probability` (zeroed when `capability.attack.observe` is False)
  and zeroes type bits when `team_gen.observe` is False; `Own` orders health, shield,
  `attack_probability`, `own_pos`, FOV, and type.
* Spotter gating (`visibility.spotter_gate`) passes precomputed `Spotter` gates; when
  `quirks.enemy_block_reset` is enabled, a 0 gate zeroes all enemies up to and including that index.
* Global state ally columns order health, energy/cooldown, relative x/y, shield,
  `attack_probability`, FOV, and type; `agent_features` names are generated directly from the
  layout so they match the array order.

Pairwise offsets and distances dispatch through `FeatureContext.ops` (`services.pairwise`). Under
`accel='rust'`, full obs/state encoding and circle-mode spotter updates run in Rust (`zgame-smac`
`env::features`, `env::spotter`; conic FOV stays in Python).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import EnvConfig
from . import rng as rng_mod
from . import services
from .actions import N_ACTIONS_MOVE, N_ACTIONS_NO_ATTACK
from .catalog import AllyTables, EnemyTables, UnitCatalog
from .spotter import Spotter
from .terrain import TerrainGrids
from .units import Frame
from .visibility import cone_visible

__all__ = [
    'FeatureContext',
    'ObsLayout',
    'StateLayout',
    'encode_obs',
    'encode_state',
    'enemy_raw_visible',
    'observe_spotter',
]

_DIRS = ('north', 'south', 'east', 'west')


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return num / den


@dataclass(frozen=True)
class ObsLayout:
    """Feature names for each observation block."""

    move: tuple[str, ...]
    enemy: tuple[str, ...]  # Per enemy
    ally: tuple[str, ...]  # Per ally
    own: tuple[str, ...]
    timestep: tuple[str, ...]
    n_enemies: int
    n_allies: int

    @classmethod
    def build(
        cls, config: EnvConfig, catalog: UnitCatalog, n_agents: int, n_enemies: int, n_actions: int
    ) -> ObsLayout:
        o = config.obs
        cap = config.capability
        own_health = o.own_health or o.all_health
        types = tuple(f'type_{i}' for i in range(catalog.unit_type_bits))
        attack = ('attack_probability',) if cap.attack is not None else ()
        move = tuple(f'move_{d}' for d in _DIRS)
        if o.pathing_grid:
            move += tuple(f'pathing_{i}' for i in range(8))
        if o.terrain_height:
            move += tuple(f'height_{i}' for i in range(9))
        health: tuple[str, ...] = ()
        if o.all_health:
            health = ('health',) + (('shield',) if catalog.shield_bits_enemy else ())
        enemy = ('available', 'distance', 'rel_x', 'rel_y') + health + types
        a_health: tuple[str, ...] = ()
        if o.all_health:
            a_health = ('health',) + (('shield',) if catalog.shield_bits_ally else ())
        last = tuple(f'last_action_{i}' for i in range(n_actions)) if o.last_action else ()
        ally = ('visible', 'distance', 'rel_x', 'rel_y') + a_health + attack + types + last
        own: tuple[str, ...] = ()
        if own_health:
            own = ('health',) + (('shield',) if catalog.shield_bits_ally else ())
        own += attack
        if o.own_pos:
            own += ('pos_x', 'pos_y')
        if config.visibility.model == 'cone':
            own += ('fov_x', 'fov_y')
        own += types
        if o.timestep_number and config.quirks.timestep_pad:
            own += ('timestep_pad',)
        timestep = ('timestep',) if o.timestep_number else ()
        return cls(move, enemy, ally, own, timestep, n_enemies, n_agents - 1)

    @property
    def reported_size(self) -> int:
        """`get_obs_size()`: excludes `Timestep` when `quirks.timestep_pad` is active."""
        size = self.size
        if self.timestep and 'timestep_pad' in self.own:
            size -= len(self.timestep)
        return size

    @property
    def size(self) -> int:
        """Actual observation vector length."""
        return (
            len(self.move)
            + self.n_enemies * len(self.enemy)
            + self.n_allies * len(self.ally)
            + len(self.own)
            + len(self.timestep)
        )

    def names(self) -> list[str]:
        out = list(self.move)
        out += [f'enemy_{e}.{n}' for e in range(self.n_enemies) for n in self.enemy]
        out += [f'ally_{a}.{n}' for a in range(self.n_allies) for n in self.ally]
        out += [f'own.{n}' for n in self.own] + list(self.timestep)
        return out


@dataclass(frozen=True)
class StateLayout:
    ally: tuple[str, ...]
    enemy: tuple[str, ...]
    n_agents: int
    n_enemies: int
    n_actions: int
    last_action: bool
    timestep: bool
    obs_instead: ObsLayout | None  # ObsLayout when obs_instead_of_state is enabled

    @classmethod
    def build(
        cls,
        config: EnvConfig,
        catalog: UnitCatalog,
        obs: ObsLayout,
        n_agents: int,
        n_enemies: int,
        n_actions: int,
    ) -> StateLayout:
        types = tuple(f'type_{i}' for i in range(catalog.unit_type_bits))
        ally = ('health', 'energy/cooldown', 'rel_x', 'rel_y')
        ally += ('shield',) if catalog.shield_bits_ally else ()
        ally += ('attack_probability',) if config.capability.attack is not None else ()
        ally += ('fov_x', 'fov_y') if config.visibility.model == 'cone' else ()
        ally += types
        enemy = ('health', 'rel_x', 'rel_y') + (('shield',) if catalog.shield_bits_enemy else ())
        enemy += types
        s = config.state
        return cls(
            ally,
            enemy,
            n_agents,
            n_enemies,
            n_actions,
            s.last_action,
            s.timestep_number,
            obs if s.obs_instead_of_state else None,
        )

    @property
    def reported_size(self) -> int:
        if self.obs_instead is not None:
            return self.obs_instead.reported_size * self.n_agents
        return self.size

    @property
    def size(self) -> int:
        if self.obs_instead is not None:
            return self.obs_instead.size * self.n_agents
        size = self.n_agents * len(self.ally) + self.n_enemies * len(self.enemy)
        size += self.n_agents * self.n_actions if self.last_action else 0
        return size + int(self.timestep)


@dataclass(frozen=True)
class FeatureContext:
    """All inputs required to encode one frame."""

    config: EnvConfig
    catalog: UnitCatalog
    terrain: TerrainGrids
    frame: Frame
    ally_tables: AllyTables
    enemy_tables: EnemyTables
    avail: np.ndarray  # int64[B, A, n_actions]
    last_action: np.ndarray  # float64[B, A, n_actions], one-hot
    episode_steps: np.ndarray  # int64[B]
    episode_limit: int
    true_target: np.ndarray | None = None  # [B, A, n_targets] range-masked; None uses avail
    fov: np.ndarray | None = None  # float64[B, A, 2]; facing vectors in cone mode
    fov_angle: float = 0.0
    gate: np.ndarray | None = None  # bool[B, A, E]; spotter gate
    attack_prob: np.ndarray | None = None  # float64[B, A]
    enemy_mask: np.ndarray | None = None  # int64[B, A, E]; 1 = masked
    ops: services.Services = services.PYTHON  # Numerical services (Python or Rust)


def _one_hot(type_id: np.ndarray, bits: int) -> np.ndarray:
    return (type_id[..., None] == np.arange(bits)).astype(np.float64)


def _raw_visible(ctx: FeatureContext, tx, ty, d, sight) -> np.ndarray:
    """Circle (`dist < sight`) or conic FOV (`dist <= sight` and in sector), ignoring target alive
    status. `tx`, `ty` broadcast to `[B, A, N]` and are used only in cone mode."""
    if ctx.config.visibility.model == 'cone':
        assert ctx.fov is not None
        al = ctx.frame.allies
        mx, my = al.x[:, :, None], al.y[:, :, None]
        fx, fy = ctx.fov[..., 0][:, :, None], ctx.fov[..., 1][:, :, None]
        return cone_visible(mx, my, tx, ty, fx, fy, sight, ctx.fov_angle)
    return d < sight


def enemy_raw_visible(ctx: FeatureContext) -> np.ndarray:
    """Return `bool[B, A, E]`: enemy within agent sight range / cone (input to spotter)."""
    al, en = ctx.frame.allies, ctx.frame.enemies
    _, _, d = ctx.ops.pairwise(al.x, al.y, en.x, en.y)
    sight = ctx.ally_tables.sight_range[:, :, None]
    return _raw_visible(ctx, en.x[:, None, :], en.y[:, None, :], d, sight)


def observe_spotter(
    ctx: FeatureContext,
    spotter: Spotter,
    streams: Callable[[int], np.random.Generator],
    shuffle: bool,
) -> np.ndarray:
    """Update `spotter` per env and return a copy of the visibility gate `bool[B, A, E]`
    (pass via `replace(ctx, gate=...)` before calling `encode_obs`).

    Env `b` uses `streams(b)` (the frame's `obs_shuffle` stream): when `shuffle` is True, agent
    evaluation order is shuffled first and `Spotter.observe` draws from the same stream.
    """
    raw = enemy_raw_visible(ctx)
    alive = ctx.frame.allies.health > 0
    for b in range(raw.shape[0]):
        gen = streams(b)
        order = list(range(raw.shape[1]))
        if shuffle:
            rng_mod.shuffle(gen, order)
        spotter.observe(b, order, raw[b], alive[b], gen)
    return spotter.gate.copy()


def encode_obs(ctx: FeatureContext, layout: ObsLayout) -> np.ndarray:
    """Encode observations as `float32[B, A, layout.size]`."""
    cfg, cat = ctx.config, ctx.catalog
    cap = cfg.capability
    full = cfg.visibility.fully_observable
    al, en = ctx.frame.allies, ctx.frame.enemies
    at, et = ctx.ally_tables, ctx.enemy_tables
    sight = at.sight_range[:, :, None]  # [B, A, 1]
    b_size, n_agents = al.x.shape
    n_enemies = en.x.shape[1]
    me_alive = al.health > 0  # [B, A]
    bits = cat.unit_type_bits

    # Move
    move = [ctx.avail[..., 2 : 2 + N_ACTIONS_MOVE].astype(np.float64)]
    if cfg.obs.pathing_grid:
        move.append(ctx.terrain.surrounding_pathing(al.x, al.y, cfg.actions.move_amount))
    if cfg.obs.terrain_height:
        move.append(ctx.terrain.surrounding_height(al.x, al.y, cfg.actions.move_amount))
    move_f = np.concatenate(move, axis=-1) * me_alive[..., None]

    # Enemy [B, A, E, F]
    rel_x, rel_y, d = ctx.ops.pairwise(al.x, al.y, en.x, en.y)
    e_alive = (en.health > 0)[:, None, :]
    raw = _raw_visible(ctx, en.x[:, None, :], en.y[:, None, :], d, sight)
    gate = np.ones(d.shape, bool) if ctx.gate is None else ctx.gate
    vis = ((raw & gate & e_alive) | (e_alive & full)) & me_alive[..., None]
    if ctx.true_target is None:
        n_fov = cfg.visibility.n_fov if cfg.visibility.model == 'cone' else 0
        target = ctx.avail[..., N_ACTIONS_NO_ATTACK + n_fov :]
    else:
        target = ctx.true_target
    cols = [target[..., :n_enemies].astype(np.float64), d / sight, rel_x / sight, rel_y / sight]
    shape_e = d.shape
    show = np.ones(shape_e) if ctx.enemy_mask is None else (ctx.enemy_mask == 0).astype(np.float64)
    if cfg.obs.all_health:
        cols.append(
            np.broadcast_to(_safe_div(en.health, en.health_max)[:, None, :], shape_e) * show
        )
        if cat.shield_bits_enemy:
            sh = _safe_div(en.shield, et.shield_max)[:, None, :]
            cols.append(np.broadcast_to(sh, shape_e) * show)
    enemy = np.stack(cols, axis=-1)
    if bits:
        oh = np.broadcast_to(_one_hot(et.type_id, bits)[:, None, :, :], (*shape_e, bits))
        enemy = np.concatenate([enemy, oh * show[..., None]], axis=-1)
    enemy = np.where(vis[..., None], enemy, 0.0)
    if ctx.gate is not None:
        bad = ~gate
        if cfg.quirks.enemy_block_reset:
            # Zero all enemies up to and including the last one with gate == 0.
            idx = np.arange(n_enemies)
            last = np.where(bad.any(-1), n_enemies - 1 - np.argmax(bad[..., ::-1], -1), -1)
            bad = idx[None, None, :] <= last[..., None]
        enemy = np.where(bad[..., None], 0.0, enemy)

    # Ally [B, A, A-1, F]
    others = np.array([[j for j in range(n_agents) if j != i] for i in range(n_agents)], np.int64)
    others = others.reshape(n_agents, max(n_agents - 1, 0))
    o_health = al.health[:, others]
    rel_xa, rel_ya, d_a = ctx.ops.pairwise(al.x, al.y, al.x, al.y, exclude_self=True)
    o_alive = o_health > 0
    raw_a = _raw_visible(ctx, al.x[:, others], al.y[:, others], d_a, sight)
    vis_a = ((raw_a & o_alive) | (o_alive & full)) & me_alive[..., None]
    shape_a = d_a.shape
    cols = [np.ones(shape_a), d_a / sight, rel_xa / sight, rel_ya / sight]
    if cfg.obs.all_health:
        cols.append(_safe_div(o_health, al.health_max[:, others]))
        if cat.shield_bits_ally:
            cols.append(_safe_div(al.shield[:, others], at.shield_max[:, others]))
    if cap.attack is not None:
        assert ctx.attack_prob is not None
        cols.append(ctx.attack_prob[:, others] * float(cap.attack.observe))
    ally = np.stack(cols, axis=-1)
    if bits:
        show_t = cap.team_gen is None or cap.team_gen.observe
        ally = np.concatenate([ally, _one_hot(at.type_id[:, others], bits) * show_t], axis=-1)
    if cfg.obs.last_action:
        ally = np.concatenate([ally, ctx.last_action[:, others]], axis=-1)
    ally = np.where(vis_a[..., None], ally, 0.0)

    # Own [B, A, F]
    own_cols = []
    if cfg.obs.own_health or cfg.obs.all_health:
        own_cols.append(_safe_div(al.health, al.health_max)[..., None])
        if cat.shield_bits_ally:
            own_cols.append(_safe_div(al.shield, at.shield_max)[..., None])
    if cap.attack is not None:
        assert ctx.attack_prob is not None
        own_cols.append(ctx.attack_prob[..., None])
    if cfg.obs.own_pos:
        own_cols += [(al.x / ctx.terrain.map_x)[..., None], (al.y / ctx.terrain.map_y)[..., None]]
    if cfg.visibility.model == 'cone':
        assert ctx.fov is not None
        own_cols.append(ctx.fov)
    if bits:
        own_cols.append(_one_hot(at.type_id, bits))
    if 'timestep_pad' in layout.own:
        own_cols.append(np.zeros((b_size, n_agents, 1)))
    own = np.concatenate(own_cols, axis=-1) if own_cols else np.zeros((b_size, n_agents, 0))
    own = np.where(me_alive[..., None], own, 0.0)

    parts = [
        move_f,
        enemy.reshape(b_size, n_agents, -1),
        ally.reshape(b_size, n_agents, -1),
        own,
    ]
    if layout.timestep:
        ts = (ctx.episode_steps / ctx.episode_limit)[:, None, None]
        parts.append(np.broadcast_to(ts, (b_size, n_agents, 1)))
    out = np.concatenate(parts, axis=-1).astype(np.float32)
    assert out.shape[-1] == layout.size, (out.shape, layout.size)
    return out


def encode_state(ctx: FeatureContext, layout: StateLayout, obs: np.ndarray | None) -> np.ndarray:
    """Encode global state as `float32[B, layout.size]`; requires `obs` from `encode_obs` when
    `obs_instead_of_state` is enabled."""
    if layout.obs_instead is not None:
        assert obs is not None
        return obs.reshape(obs.shape[0], -1).astype(np.float32)
    cfg, cat = ctx.config, ctx.catalog
    al, en = ctx.frame.allies, ctx.frame.enemies
    at, et = ctx.ally_tables, ctx.enemy_tables
    t = ctx.terrain
    bits = cat.unit_type_bits
    b_size = al.x.shape[0]
    cx, cy = t.map_x / 2, t.map_y / 2

    a_alive = al.health > 0
    cool = np.where(
        at.is_medivac & cat.heal_possible,
        _safe_div(al.energy, at.cooldown_max),
        _safe_div(al.cooldown, at.cooldown_max),
    )
    cols = [
        _safe_div(al.health, al.health_max),
        cool,
        (al.x - cx) / t.max_distance_x,
        (al.y - cy) / t.max_distance_y,
    ]
    if cat.shield_bits_ally:
        cols.append(_safe_div(al.shield, at.shield_max))
    if cfg.capability.attack is not None:
        assert ctx.attack_prob is not None
        cols.append(ctx.attack_prob)
    ally = np.stack(cols, axis=-1)
    if cfg.visibility.model == 'cone':
        assert ctx.fov is not None
        ally = np.concatenate([ally, ctx.fov], axis=-1)
    if bits:
        ally = np.concatenate([ally, _one_hot(at.type_id, bits)], axis=-1)
    ally = np.where(a_alive[..., None], ally, 0.0)

    e_alive = en.health > 0
    cols = [
        _safe_div(en.health, en.health_max),
        (en.x - cx) / t.max_distance_x,
        (en.y - cy) / t.max_distance_y,
    ]
    if cat.shield_bits_enemy:
        cols.append(_safe_div(en.shield, et.shield_max))
    enemy = np.stack(cols, axis=-1)
    if bits:
        enemy = np.concatenate([enemy, _one_hot(et.type_id, bits)], axis=-1)
    enemy = np.where(e_alive[..., None], enemy, 0.0)

    parts = [ally.reshape(b_size, -1), enemy.reshape(b_size, -1)]
    if layout.last_action:
        parts.append(ctx.last_action.reshape(b_size, -1))
    if layout.timestep:
        parts.append((ctx.episode_steps / ctx.episode_limit)[:, None])
    out = np.concatenate(parts, axis=-1).astype(np.float32)
    assert out.shape[-1] == layout.size, (out.shape, layout.size)
    return out
