"""Rust-accelerated semantic-layer functions (`accel='rust'`; see `unismac.env.accel`).

Each function matches its Python reference counterpart in name and signature,
dispatching batched observation/state encoding, action masking, rewards,
termination, procedural generation, spotter gating, and scripted opponents to
``zgame._zgame.env``. Custom subclasses or unsupported inputs transparently fall
back to the Python reference implementation and increment `native.fallbacks`.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

import numpy as np

from ..backends.zgame.backend import ZgameBackend
from ..contract import Backend, Command, WorldState
from . import accel as accel_mod
from . import classic
from . import features as features_mod
from . import generators as generators_mod
from . import scripted as scripted_pkg
from . import termination as termination_mod
from .actions import N_ACTIONS_NO_ATTACK, ActionSpace, FovState
from .catalog import AllyTables, EnemyTables
from .config import TerminationConfig
from .features import FeatureContext, ObsLayout, StateLayout
from .generators import EpisodeDraw, ScenarioGenerator
from .native import Fallback, _env, copy_rows, fallbacks
from .opponents import OpeningOrders
from .reward import BattleReward
from .rng import Streams
from .scripted.base import SCRIPTS
from .scripted.opponent import ScriptedOpponent
from .spotter import Spotter
from .terrain import TerrainGrids
from .units import Frame

__all__ = ['RustKernels']

# Built-in scripts (defined in unismac.env.scripted) that also have a Rust implementation:
# name -> class. Rust scripted opponent is used only when every pool entry maps to its built-in.
_PREFIX = scripted_pkg.__name__ + '.'
_RUST_SCRIPTS = {
    n: c
    for n, c in classic.BUILTIN_SCRIPTS.items()
    if c.__module__.startswith(_PREFIX) and n in set(getattr(_env, 'script_names', list)())
}


@cache
def _warning_site() -> tuple[str, int]:
    """Locate the reference warning site: the `opponent.issue(...)` call inside
    `accel.opponent_issue` (`stacklevel=2`)."""
    lines, start = inspect.getsourcelines(accel_mod.opponent_issue)
    i = next(i for i, line in enumerate(lines) if line.strip().startswith('opponent.issue('))
    return accel_mod.opponent_issue.__code__.co_filename, start + i


def _script_warning(message: str) -> None:
    """Emit a warning matching the reference in message, category, site, and dedup registry
    (`accel.__warningregistry__`)."""
    filename, lineno = _warning_site()
    g = vars(accel_mod)
    warnings.warn_explicit(
        message,
        RuntimeWarning,
        filename,
        lineno,
        module=accel_mod.__name__,
        registry=g.setdefault('__warningregistry__', {}),
        module_globals=g,
    )


def advance(
    backend: Backend, command: Command, n_loops: int, world: WorldState, rows: np.ndarray
) -> None:
    """`units.advance`: write active rows directly in-kernel for `ZgameBackend`, else copy rows."""
    if type(backend) is ZgameBackend:
        backend.step_rows(command, n_loops, world, rows)
    else:
        copy_rows(world, backend.step(command, n_loops), rows)


# ---------------------------------------------------------------- Action masks


def avail(
    space: ActionSpace,
    frame: Frame,
    tables: AllyTables,
    terrain: TerrainGrids,
    fov: FovState | None = None,
    *,
    mask_by_range: bool | None = None,
    ability_choice: np.ndarray | None = None,
    map_size: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Rust counterpart of `ActionSpace.avail` (without FOV turn or ability segments)."""
    kw: dict[str, Any] = dict(
        mask_by_range=mask_by_range, ability_choice=ability_choice, map_size=map_size
    )
    if type(space) is not ActionSpace or space.n_fov or space.n_ability or fov is not None:
        fallbacks['avail'] += 1
        return space.avail(frame, tables, terrain, fov, **kw)
    by_range = space.config.mask_by_range if mask_by_range is None else mask_by_range
    try:
        return _env.avail(
            frame,
            tables,
            terrain.pathing,
            space.n_targets,
            space.n_no_attack,
            space.n_actions,
            float(space.config.move_amount),
            bool(space.heal_enabled),
            bool(by_range),
        )
    except Fallback:
        fallbacks['avail'] += 1
        return space.avail(frame, tables, terrain, fov, **kw)


# ---------------------------------------------------------------- Spotter, reward, end, opponent


def observe_spotter(
    ctx: FeatureContext,
    spotter: Spotter,
    streams: Callable[[int], np.random.Generator],
    shuffle: bool,
) -> np.ndarray:
    """Rust counterpart of `features.observe_spotter` (circle visibility; `rng.Streams` only)."""
    if (
        type(spotter) is not Spotter
        or not isinstance(streams, Streams)
        or ctx.config.visibility.model == 'cone'
    ):
        fallbacks['observe_spotter'] += 1
        return features_mod.observe_spotter(ctx, spotter, streams, shuffle)
    try:
        _env.spotter_observe(
            spotter, ctx.frame, ctx.ally_tables.sight_range, streams.table(), bool(shuffle)
        )
    except Fallback:
        fallbacks['observe_spotter'] += 1
        return features_mod.observe_spotter(ctx, spotter, streams, shuffle)
    return spotter.gate.copy()


def spotter_after_step(
    spotter: Spotter, ally_dead: np.ndarray, enemy_dead: np.ndarray, rows: np.ndarray
) -> None:
    """Rust counterpart of `Spotter.after_step`."""
    if type(spotter) is Spotter:
        try:
            _env.spotter_after_step(spotter, ally_dead, enemy_dead, np.asarray(rows))
            return
        except Fallback:
            pass
    fallbacks['spotter_after_step'] += 1
    spotter.after_step(ally_dead, enemy_dead, rows)


def reward(rewarder: BattleReward, prev: Frame, cur: Frame, active: np.ndarray) -> np.ndarray:
    """Rust counterpart of `BattleReward.step`."""
    if type(rewarder) is BattleReward:
        cfg = rewarder.config
        try:
            return _env.battle_reward(
                prev,
                cur,
                active,
                rewarder.dead_ally,
                rewarder.dead_enemy,
                bool(cfg.sparse),
                bool(cfg.only_positive),
                cfg.only_positive_mode == 'abs',
                float(cfg.death_value),
                float(cfg.negative_scale),
            )
        except Fallback:
            pass
    fallbacks['reward'] += 1
    return rewarder.step(prev, cur, active)


def game_end(
    frame: Frame,
    ally: AllyTables,
    enemy: EnemyTables,
    config: TerminationConfig,
    medivac_rules: bool,
) -> np.ndarray:
    """Rust counterpart of `termination.game_end`."""
    try:
        return _env.game_end(frame, ally, enemy, bool(medivac_rules), config.enemy_medivac_rule)
    except Fallback:
        fallbacks['game_end'] += 1
        return termination_mod.game_end(frame, ally, enemy, config, medivac_rules=medivac_rules)


def game_end_dual(
    frame: Frame, ally: AllyTables, frame2: Frame, ally2: AllyTables, medivac_rules: bool
) -> np.ndarray:
    """Rust counterpart of `termination.game_end_dual`."""
    try:
        return _env.game_end_dual(frame, ally, frame2, ally2, bool(medivac_rules))
    except Fallback:
        fallbacks['game_end_dual'] += 1
        return termination_mod.game_end_dual(frame, ally, frame2, ally2, medivac_rules)


def opponent_reset(
    opponent: Any, world: WorldState, env_ids: Sequence[int], names: Sequence[str | None] | None
) -> None:
    """Rust counterpart of `Opponent.on_reset` (`OpeningOrders`; others delegate to Python)."""
    if type(opponent) is OpeningOrders and names is None:
        b_size = world.batch_size
        if opponent.target is None or opponent.target.shape[0] != b_size:  # Match reference
            opponent.target = np.zeros((b_size, 2), np.float32)
            opponent.pending = np.zeros(b_size, bool)
        try:
            _env.opening_reset(world, [int(b) for b in env_ids], opponent.target, opponent.pending)
            return
        except Fallback:
            fallbacks['opponent_reset'] += 1
    accel_mod.opponent_reset(opponent, world, env_ids, names)


def opponent_issue(opponent: Any, world: WorldState, command: Command, active: np.ndarray) -> None:
    """Rust counterpart of `Opponent.issue` (`OpeningOrders`; others delegate to Python)."""
    if type(opponent) is OpeningOrders:
        try:
            _env.opening_issue(world, opponent.target, opponent.pending, active, command)
            return
        except Fallback:
            fallbacks['opponent_issue'] += 1
    opponent.issue(world, command, active)


# ---------------------------------------------------------------- Memoized: draw, obs, state


def _hit(memo: tuple[Any, ...] | None, key: tuple[Any, ...], n_ident: int) -> bool:
    """Check cache hit: first `n_ident` key items match by identity, remainder by equality."""
    if memo is None:
        return False
    old = memo[0]
    if any(a is not b for a, b in zip(old[:n_ident], key[:n_ident], strict=True)):
        return False
    return old[n_ident:] == key[n_ident:]


def _remember(table: dict[int, tuple[Any, ...]], key: int, memo: tuple[Any, ...]) -> None:
    """Cache entry by layout ID, clearing if the table grows beyond 8 layouts."""
    if len(table) >= 8:
        table.clear()
    table[key] = memo


class RustKernels:
    """Stateful semantic-layer Rust kernels (one instance per `VecEnv`): procedural generation
    specs, obs/state layout specs, and Rust scripted opponent state."""

    def __init__(self) -> None:
        self._gen: tuple[Any, ...] | None = None
        # Memoized obs/state specs keyed by layout ID (two-sided mode has one layout per side).
        self._obs: dict[int, tuple[Any, ...]] = {}
        self._state: dict[int, tuple[Any, ...]] = {}
        # (opponent, Rust _env.ScriptedOpponent state; None means fall back to Python reference).
        self._script: tuple[Any, Any] | None = None

    # ------------------------------------------------ Opponent

    def _scripted(self, opponent: Any, first: bool) -> Any:
        """Return Rust scripted opponent state, or None to use the Python reference.

        Ownership is decided on the opponent's first reset: exact `ScriptedOpponent` type, bound,
        no existing Python `scripts` state, and every pool name has a Rust implementation whose
        registry class is still the built-in. Ownership stays fixed thereafter.
        """
        if self._script is not None and self._script[0] is opponent:
            return self._script[1]
        if not first:
            return None
        rs = None
        env = opponent.env
        if (
            env is not None
            and not opponent.scripts
            and all(_RUST_SCRIPTS.get(n, ...) is SCRIPTS.get(n) for n in opponent.pool)
        ):
            try:
                rs = _env.ScriptedOpponent(str(env.spec.name), [str(n) for n in opponent.pool])
            except Fallback:
                rs = None
        self._script = (opponent, rs)
        return rs

    def opponent_reset(
        self,
        opponent: Any,
        world: WorldState,
        env_ids: Sequence[int],
        names: Sequence[str | None] | None,
    ) -> None:
        """Rust counterpart of `Opponent.on_reset` (`OpeningOrders`, `ScriptedOpponent`)."""
        if type(opponent) is not ScriptedOpponent:
            opponent_reset(opponent, world, env_ids, names)
            return
        first = self._script is None or self._script[0] is not opponent
        rs = self._scripted(opponent, first=True)
        if rs is None:  # Custom or non-Rust pool: delegate without incrementing fallbacks
            accel_mod.opponent_reset(opponent, world, env_ids, names)
            return
        env = opponent._require_env()
        ids = [int(b) for b in env_ids]
        table = env.rng_streams('opponent', np.zeros(world.batch_size, np.int64)).table()
        view = env.view
        try:
            chosen = rs.reset(
                world,
                view.enemy_slots,
                view.ally_slots,
                ids,
                [None] * len(ids) if names is None else list(names),
                table,
            )
        except Fallback as e:
            if not first:
                raise RuntimeError(f'Rust scripted opponent cannot continue: {e}') from e
            self._script = (opponent, None)
            fallbacks['opponent_reset'] += 1
            accel_mod.opponent_reset(opponent, world, env_ids, names)
            return
        for b, name in zip(ids, chosen, strict=True):
            opponent.names[b] = name

    def opponent_issue(
        self, opponent: Any, world: WorldState, command: Command, active: np.ndarray
    ) -> None:
        """Rust counterpart of `Opponent.issue` (`OpeningOrders`, `ScriptedOpponent`).

        Matches reference error counting (`opponent.errors`) and once-per-`script:exc_type`
        `RuntimeWarning`.
        """
        rs = self._scripted(opponent, first=False) if type(opponent) is ScriptedOpponent else None
        if rs is None:
            opponent_issue(opponent, world, command, active)
            return
        env = opponent._require_env()
        view = env.view
        steps = env.episode_steps
        table = env.rng_streams('opponent', steps + 1).table() if rs.uses_choice(active) else None
        abilities = env.query_abilities() if rs.needs_abilities(active) else None
        try:
            errors = rs.issue(
                world,
                view.enemy_slots,
                view.ally_slots,
                steps,
                active,
                table,
                abilities,
                opponent.config.script_abilities == 'drop',
                command,
            )
        except Fallback as e:
            raise RuntimeError(f'Rust scripted opponent cannot continue: {e}') from e
        for b, t, kind, text in errors:
            opponent.errors[b] = opponent.errors.get(b, 0) + 1
            key = f'{opponent.names[b]}:{kind}'
            if key not in opponent._warned:
                opponent._warned.add(key)
                name = opponent.names[b]
                _script_warning(
                    f'SMAC-Hard script {name} raised {text} at step {t}; '
                    'Player 2 issues no commands this step'
                )

    # ------------------------------------------------ Procedural generation

    def _gen_spec(self, gen: ScenarioGenerator) -> dict[str, Any]:
        if self._gen is not None and self._gen[0] is gen and self._gen[1] is gen.config:
            return self._gen[2]
        c = gen.config
        tg, sp = c.team_gen, c.start_positions
        spec: dict[str, Any] = {
            'n_units': int(gen.n_units),
            'n_enemies': int(gen.n_enemies),
            'team_weights': None,
            'start_kind': None,
            'attack': None,
            'enemy_mask': None,
        }
        if tg is not None:
            spec['team_weights'] = [float(w) for w in np.asarray(tg.weights, np.float64)]
            spec['team_exception'] = [u in tg.exception_unit_types for u in tg.unit_types]
        if sp is not None:
            spec.update(
                start_kind=str(sp.dist_type),
                start_p=float(sp.p),
                map_x=float(sp.map_x),
                map_y=float(sp.map_y),
            )
        if c.attack is not None:
            spec['attack'] = (float(c.attack.lower_bound[0]), float(c.attack.upper_bound[0]))
        if c.enemy_mask is not None:
            spec['enemy_mask'] = float(c.enemy_mask.mask_probability)
        self._gen = (gen, c, spec)
        return spec

    def draw(
        self,
        generator: ScenarioGenerator,
        streams: Callable[[int], np.random.Generator],
        env_ids: Sequence[int],
    ) -> list[EpisodeDraw]:
        """Rust counterpart of `generators.draw_batch`."""
        if (
            type(generator) is not ScenarioGenerator
            or not isinstance(streams, Streams)
            or generator.config.health is not None
        ):
            fallbacks['draw'] += 1
            return generators_mod.draw_batch(generator, streams, env_ids)
        spec = self._gen_spec(generator)
        try:
            rows = _env.draw(spec, streams.table(), [int(b) for b in env_ids])
        except Fallback:
            fallbacks['draw'] += 1
            return generators_mod.draw_batch(generator, streams, env_ids)
        names = generator.config.team_gen.unit_types if generator.config.team_gen else ()
        out = []
        for ally, enemy, ally_pos, enemy_pos, attack, mask in rows:
            out.append(
                EpisodeDraw(
                    None if ally is None else tuple(str(names[i]) for i in ally),
                    None if enemy is None else tuple(str(names[i]) for i in enemy),
                    ally_pos,
                    enemy_pos,
                    attack,
                    mask,
                )
            )
        return out

    # ------------------------------------------------ Observations

    def _obs_spec(self, ctx: FeatureContext, layout: ObsLayout) -> Any:
        key = (layout, ctx.config, ctx.catalog, ctx.terrain, ctx.avail.shape[-1], ctx.episode_limit)
        memo = self._obs.get(id(layout))
        if _hit(memo, key, 4):
            return memo[1]  # type: ignore[index]
        cfg, cat, t = ctx.config, ctx.catalog, ctx.terrain
        o, cap = cfg.obs, cfg.capability
        grids = bool(o.pathing_grid or o.terrain_height)
        spec = _env.ObsSpec(
            {
                'n_agents': layout.n_allies + 1,
                'n_enemies': layout.n_enemies,
                'n_actions': int(ctx.avail.shape[-1]),
                'bits': int(cat.unit_type_bits),
                'shield_ally': bool(cat.shield_bits_ally),
                'shield_enemy': bool(cat.shield_bits_enemy),
                'all_health': bool(o.all_health),
                'own_health': bool(o.own_health or o.all_health),
                'pathing_grid': bool(o.pathing_grid),
                'terrain_height': bool(o.terrain_height),
                'move_int': int(cfg.actions.move_amount) if grids else 0,
                'last_action': bool(o.last_action),
                'attack': None if cap.attack is None else float(cap.attack.observe),
                'show_types': bool(cap.team_gen is None or cap.team_gen.observe),
                'own_pos': bool(o.own_pos),
                'timestep_pad': 'timestep_pad' in layout.own,
                'timestep': bool(layout.timestep),
                'fully_observable': bool(cfg.visibility.fully_observable),
                'enemy_block_reset': bool(cfg.quirks.enemy_block_reset),
                'map_x': float(t.map_x),
                'map_y': float(t.map_y),
                'episode_limit': float(ctx.episode_limit),
                'size': int(layout.size),
            }
        )
        _remember(self._obs, id(layout), (key, spec))
        return spec

    def encode_obs(self, ctx: FeatureContext, layout: ObsLayout) -> np.ndarray:
        """Rust counterpart of `features.encode_obs` (circle visibility)."""
        cfg = ctx.config
        if cfg.visibility.model == 'cone' or ctx.fov is not None or type(layout) is not ObsLayout:
            fallbacks['encode_obs'] += 1
            return features_mod.encode_obs(ctx, layout)
        try:
            spec = self._obs_spec(ctx, layout)
            target = (
                ctx.avail[..., N_ACTIONS_NO_ATTACK:] if ctx.true_target is None else ctx.true_target
            )
            o = cfg.obs
            t = ctx.terrain
            return _env.encode_obs(
                spec,
                ctx.frame,
                ctx.ally_tables,
                ctx.enemy_tables,
                ctx.avail,
                target,
                ctx.last_action,
                ctx.episode_steps,
                gate=ctx.gate,
                attack_prob=None if cfg.capability.attack is None else ctx.attack_prob,
                enemy_mask=ctx.enemy_mask,
                pathing=t.pathing if o.pathing_grid or o.terrain_height else None,
                height=t.height if o.pathing_grid or o.terrain_height else None,
            )
        except Fallback:
            fallbacks['encode_obs'] += 1
            return features_mod.encode_obs(ctx, layout)

    # ------------------------------------------------ Global state

    def _state_spec(self, ctx: FeatureContext, layout: StateLayout) -> Any:
        key = (layout, ctx.config, ctx.catalog, ctx.terrain, ctx.episode_limit)
        memo = self._state.get(id(layout))
        if _hit(memo, key, 4):
            return memo[1]  # type: ignore[index]
        cfg, cat, t = ctx.config, ctx.catalog, ctx.terrain
        spec = _env.StateSpec(
            {
                'n_agents': int(layout.n_agents),
                'n_enemies': int(layout.n_enemies),
                'n_actions': int(layout.n_actions),
                'bits': int(cat.unit_type_bits),
                'shield_ally': bool(cat.shield_bits_ally),
                'shield_enemy': bool(cat.shield_bits_enemy),
                'attack': cfg.capability.attack is not None,
                'heal_possible': bool(cat.heal_possible),
                'last_action': bool(layout.last_action),
                'timestep': bool(layout.timestep),
                'centre': (t.map_x / 2, t.map_y / 2),
                'max_distance': (float(t.max_distance_x), float(t.max_distance_y)),
                'episode_limit': float(ctx.episode_limit),
                'size': int(layout.size),
            }
        )
        _remember(self._state, id(layout), (key, spec))
        return spec

    def encode_state(
        self, ctx: FeatureContext, layout: StateLayout, obs: np.ndarray | None
    ) -> np.ndarray:
        """Rust counterpart of `features.encode_state` (`obs_instead_of_state` delegates to
        Python since it only reshapes `obs`)."""
        cfg = ctx.config
        if (
            layout.obs_instead is not None
            or cfg.visibility.model == 'cone'
            or type(layout) is not StateLayout
        ):
            if layout.obs_instead is None:
                fallbacks['encode_state'] += 1
            return features_mod.encode_state(ctx, layout, obs)
        try:
            spec = self._state_spec(ctx, layout)
            return _env.encode_state(
                spec,
                ctx.frame,
                ctx.ally_tables,
                ctx.enemy_tables,
                ctx.last_action,
                ctx.episode_steps,
                attack_prob=None if cfg.capability.attack is None else ctx.attack_prob,
            )
        except Fallback:
            fallbacks['encode_state'] += 1
            return features_mod.encode_state(ctx, layout, obs)
