"""Backend-agnostic opening-order opponent matching the 23 standard SMAC maps.

Provides `NativeSMACOpponent` (`opponent.kind='native_smac'`) and
`native_smac_overrides(...)` for reproducing observed StarCraft II built-in
VeryHard opponent behavior across all 23 classic SMAC scenarios on both `pyref`
and `zgame` backends:

* Fixed-target maps (20 maps): computes the Q12 (1/4096 tile) centroid of
  player-2 units and issues formation-offset `ATTACK_MOVE` orders toward the
  map's attack anchor (`(9.0, 16.0)`, `(14.5998, 9.49)`, `(16.0, 8.0)`, or
  `(4.47, 4.3298)`).
* `2m_vs_1z`: samples an opening attack point uniformly along the segment
  `((13.0, 12.0), (19.0, 12.0))` via `VecEnv.rng_stream('smac_trigger', b, 0)`.
* `2c_vs_64zg`: assigns explicit round-robin `ATTACK_UNIT` orders across
  player-1 targets (`j = i % len(allies)`).
* `2s_vs_1sc`: issues no opening order (stationary defender auto-engages).
* `MMM` / `MMM2` Medivac escort behavior: moves with the formation waypoint,
  switches to healing (`ability=386`) when an injured biological ground ally is
  within 6.0 edge-to-edge distance and energy >= 5 (ordered by lowest life
  fraction, nearest distance, then slot), and resumes advancing once healing
  finishes.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..assets import load_static_data
from ..contract import (
    NO_SLOT,
    AgentPlayer,
    Command,
    CommandKind,
    Player,
    StaticData,
    Variant,
    WorldState,
)
from .opponents import Opponent

if TYPE_CHECKING:
    from .core import VecEnv

__all__ = [
    'MULTI_LEVEL_SMAC_MAPS',
    'SMAC_TRIGGER_SPECS',
    'NativeSMACOpponent',
    'SMACTriggerSpec',
    'native_smac_overrides',
]

Q12 = 4096
MEDIVAC_TYPE = 54
COLOSSUS_TYPE = 4
BANELING_TYPE = 9
HEAL_ABILITY = 386
HEAL_AUTOCAST_RANGE = 6.0
HEAL_AUTOCAST_MIN_ENERGY = 5.0
MEDIVAC_ARRIVAL_TOL = 0.25

TriggerKind = Literal['point', 'random_segment', 'round_robin_unit_group', 'none']


@dataclass(frozen=True)
class SMACTriggerSpec:
    """Opening-order specification for a classic SMAC scenario."""

    map_class: str
    kind: TriggerKind
    target_xy: tuple[float, float] | None = None
    segment: tuple[tuple[float, float], tuple[float, float]] | None = None
    formation_offset: bool = True


_STANDARD_A_MAPS: tuple[str, ...] = (
    '3m',
    '8m',
    '25m',
    '5m_vs_6m',
    '8m_vs_9m',
    '10m_vs_11m',
    '27m_vs_30m',
    '2s3z',
    '3s5z',
    '3s5z_vs_3s6z',
    '3s_vs_3z',
    '3s_vs_4z',
    '3s_vs_5z',
    '1c3s5z',
)

SMAC_TRIGGER_SPECS: dict[str, SMACTriggerSpec] = {
    **{m: SMACTriggerSpec('A', 'point', target_xy=(9.0, 16.0)) for m in _STANDARD_A_MAPS},
    'MMM': SMACTriggerSpec('A+', 'point', target_xy=(9.0, 16.0)),
    'MMM2': SMACTriggerSpec('A+', 'point', target_xy=(9.0, 16.0)),
    '6h_vs_8z': SMACTriggerSpec('B', 'point', target_xy=(14.5998, 9.49)),
    'bane_vs_bane': SMACTriggerSpec('C', 'point', target_xy=(16.0, 8.0)),
    'corridor': SMACTriggerSpec('D', 'point', target_xy=(4.47, 4.3298)),
    'so_many_baneling': SMACTriggerSpec('D', 'point', target_xy=(4.47, 4.3298)),
    '2m_vs_1z': SMACTriggerSpec(
        'E',
        'random_segment',
        segment=((13.0, 12.0), (19.0, 12.0)),
    ),
    '2c_vs_64zg': SMACTriggerSpec('F', 'round_robin_unit_group', formation_offset=False),
    '2s_vs_1sc': SMACTriggerSpec('G', 'none', formation_offset=False),
}

# Classic SMAC maps whose playable area contains multiple terrain elevation levels.
MULTI_LEVEL_SMAC_MAPS: frozenset[str] = frozenset({'2c_vs_64zg'})


@dataclass(frozen=True)
class _UnitCombatMeta:
    """Lightweight read-only unit combat metadata pre-extracted from `StaticData`."""

    has_weapon: bool
    has_ground_weapon: bool
    has_air_weapon: bool
    ground_range: float
    air_range: float
    ground_damage: float
    air_damage: float
    biological: bool
    splash: bool
    speed: float


@functools.cache
def _unit_meta_table() -> dict[tuple[int, int], _UnitCombatMeta]:
    static: StaticData = load_static_data()
    out: dict[tuple[int, int], _UnitCombatMeta] = {}
    for (ut, var), spec in static.units.items():
        g_range, a_range = 0.0, 0.0
        g_dmg, a_dmg = 0.0, 0.0
        has_g, has_a = False, False
        for w in spec.weapons:
            dmg = float(w.damage) * w.attacks
            if w.target in ('ground', 'any'):
                has_g = True
                g_range = max(g_range, float(w.range))
                g_dmg = max(g_dmg, dmg)
            if w.target in ('air', 'any'):
                has_a = True
                a_range = max(a_range, float(w.range))
                a_dmg = max(a_dmg, dmg)
        if 'suicide_damage' in spec.measured:
            has_g = True
            g_range = max(g_range, float(spec.measured.get('suicide_trigger_range', 0.41)))
            g_dmg = max(g_dmg, float(spec.measured['suicide_damage']))
        has_splash = (
            'splash_radial_radius_1' in spec.measured
            or 'splash_perp_half_length' in spec.measured
            or 'splash_forward_length' in spec.measured
            or ut in (COLOSSUS_TYPE, BANELING_TYPE)
        )
        out[(ut, var)] = _UnitCombatMeta(
            has_weapon=has_g or has_a,
            has_ground_weapon=has_g,
            has_air_weapon=has_a,
            ground_range=g_range,
            air_range=a_range,
            ground_damage=g_dmg,
            air_damage=a_dmg,
            biological='biological' in spec.attributes,
            splash=has_splash,
            speed=float(spec.movement_speed),
        )
    return out


_FALLBACK_META = _UnitCombatMeta(
    has_weapon=True,
    has_ground_weapon=True,
    has_air_weapon=False,
    ground_range=5.0,
    air_range=0.0,
    ground_damage=6.0,
    air_damage=0.0,
    biological=False,
    splash=False,
    speed=2.25,
)


def _lookup_meta(unit_type: int, variant: int) -> _UnitCombatMeta:
    table = _unit_meta_table()
    meta = table.get((unit_type, variant))
    if meta is not None:
        return meta
    return table.get((unit_type, int(Variant.STD)), _FALLBACK_META)


def native_smac_overrides(
    scenario_name: str | None = None,
    *,
    opponent: Literal['native_smac'] = 'native_smac',
    single_client: bool = True,
    smac_spawn_variant: bool = True,
    rl_no_retreat: bool = True,
    terrain_elevation: Literal['auto', 'projectile_3d', 'full', 'off'] | bool = 'auto',
    enemy_source: Literal['all', 'visible'] | None = None,
) -> dict[str, Any]:
    """Build config overrides matching single-client classic SMAC (`SMAC_Maps`) on `smac-portable`.

    Combine with `unismac.profiles.merge_overrides(profile_overrides('sc2-native')[1], ...)`.

    Args:
        scenario_name: Scenario name (e.g. `'2c_vs_64zg'`, `'MMM'`) used when
            `terrain_elevation='auto'`.
        opponent: Opponent kind (only `'native_smac'` is supported).
        single_client: If True, use zero command delay (`immediate`, delay 0, move lag 0).
        smac_spawn_variant: If True, set `backend.spawn_variant='smac'` so player-1
            `Medivac_RL` disables autocast healing while player-2 `Medivac` keeps it.
        rl_no_retreat: If True, disable idle retreat on hit (`retreat_owners=()`,
            `passive_from_map=False`) matching classic `SMAC_Maps` player-1 units.
        terrain_elevation: Multi-level terrain elevation handling:
            - `'auto'`: enable `projectile_3d=True` on multi-level maps (`2c_vs_64zg`)
              while keeping full visibility.
            - `'projectile_3d'`: explicitly enable `projectile_3d=True`.
            - `'full'` or `True`: enable true visibility, lost-vision tracking, and
              3D projectile travel (`visibility='sc2'`, `lost_vision='sc2'`, `projectile_3d=True`).
            - `'off'` or `False`: leave elevation options unchanged.
        enemy_source: Optional override for `units.enemy_source` (`'all'` or `'visible'`).
    """
    if opponent != 'native_smac':
        raise ValueError(f"Unsupported opponent={opponent!r}; expected 'native_smac'")
    over: dict[str, Any] = {'opponent.kind': opponent}
    if smac_spawn_variant:
        over['backend.spawn_variant'] = 'smac'
    if single_client:
        over['backend.pyref_commands'] = 'immediate'
        over['backend.pyref_command_delay'] = 0
        over['backend.pyref_move_lag'] = 0
    pyref_opts: dict[str, Any] = {}
    if rl_no_retreat:
        pyref_opts['retreat_owners'] = ()
        pyref_opts['passive_from_map'] = False
    if terrain_elevation is True or terrain_elevation == 'full':
        pyref_opts['visibility'] = 'sc2'
        pyref_opts['lost_vision'] = 'sc2'
        pyref_opts['projectile_3d'] = True
    elif terrain_elevation == 'projectile_3d' or (
        terrain_elevation == 'auto' and scenario_name in MULTI_LEVEL_SMAC_MAPS
    ):
        pyref_opts['projectile_3d'] = True
    elif terrain_elevation not in ('auto', 'off', False):
        raise ValueError(
            f'Unknown terrain_elevation={terrain_elevation!r}; '
            "expected 'auto', 'projectile_3d', 'full', 'off', or bool"
        )
    if pyref_opts:
        over['backend.pyref_options'] = pyref_opts
    if enemy_source is not None:
        over['units.enemy_source'] = enemy_source
    return over


@dataclass(frozen=True)
class _OpeningOrder:
    slot: int
    kind: CommandKind
    target_slot: int = NO_SLOT
    target_xy: tuple[float, float] | None = None


class NativeSMACOpponent(Opponent):
    """Opening-order opponent reproducing all 23 classic SMAC maps and Medivac escort healing."""

    def __init__(self) -> None:
        self.env: VecEnv | None = None
        self.spec: SMACTriggerSpec | None = None
        self._can_heal: bool = False
        self._playable: tuple[float, float, float, float] = (0.0, 0.0, 32.0, 32.0)
        self._pending: dict[int, bool] = {}
        self.opening_targets: dict[int, tuple[float, float] | None] = {}
        self._opening_orders: dict[int, list[_OpeningOrder]] = {}
        # Per-env escort waypoint and arrival status for unarmed support units (Medivac).
        self._medivac_wp: dict[int, dict[int, tuple[float, float]]] = {}
        self._medivac_arrived: dict[int, dict[int, bool]] = {}

    def player(self, race: str) -> Player:
        return AgentPlayer(race)  # type: ignore[arg-type]

    def bind(self, env: VecEnv) -> None:
        self.env = env
        name = str(env.spec.name)
        self.spec = SMAC_TRIGGER_SPECS.get(
            name, SMACTriggerSpec('CUSTOM', 'point', target_xy=None, formation_offset=True)
        )
        self._can_heal = env.backend.capabilities.abilities
        x0, y0, x1, y1 = env._playable
        self._playable = (float(x0), float(y0), float(x1), float(y1))

    def _require_env(self) -> VecEnv:
        if self.env is None or self.spec is None:
            raise RuntimeError(f'{type(self).__name__} is not bound to an environment')
        return self.env

    def opening_orders(
        self, env_id: int
    ) -> tuple[tuple[int, CommandKind, int, tuple[float, float] | None], ...]:
        """Return precomputed opening orders from the latest reset for `env_id`."""
        return tuple(
            (o.slot, o.kind, o.target_slot, o.target_xy)
            for o in self._opening_orders.get(env_id, ())
        )

    def on_reset(
        self,
        world: WorldState,
        env_ids: Sequence[int],
        names: Sequence[str | None] | None = None,
    ) -> None:
        env = self._require_env()
        assert self.spec is not None
        spec = self.spec
        x0, y0, x1, y1 = self._playable
        min_x_q12, min_y_q12 = round(x0 * Q12), round(y0 * Q12)
        max_x_q12, max_y_q12 = round(x1 * Q12), round(y1 * Q12)

        for b in env_ids:
            self._pending[b] = True
            self._medivac_wp[b] = {}
            self._medivac_arrived[b] = {}
            enemies = np.flatnonzero(world.alive[b] & (world.owner[b] == 2)).tolist()
            allies = np.flatnonzero(world.alive[b] & (world.owner[b] == 1)).tolist()

            if spec.kind == 'none' or not enemies:
                self.opening_targets[b] = None
                self._opening_orders[b] = []
                continue

            if spec.kind == 'round_robin_unit_group':
                self.opening_targets[b] = None
                orders: list[_OpeningOrder] = []
                if allies:
                    m = len(allies)
                    for i, s in enumerate(enemies):
                        orders.append(
                            _OpeningOrder(s, CommandKind.ATTACK_UNIT, target_slot=allies[i % m])
                        )
                self._opening_orders[b] = orders
                continue

            if spec.kind == 'random_segment':
                assert spec.segment is not None
                (sx0, sy0), (sx1, sy1) = spec.segment
                t = float(env.rng_stream('smac_trigger', b, 0).uniform(0.0, 1.0))
                tx = sx0 + (sx1 - sx0) * t
                ty = sy0 + (sy1 - sy0) * t
            elif spec.target_xy is not None:
                tx, ty = spec.target_xy
            elif allies:
                tx = float(np.mean(world.x[b, allies]))
                ty = float(np.mean(world.y[b, allies]))
            else:
                tx, ty = (x0 + x1) * 0.5, (y0 + y1) * 0.5

            tx_q12 = round(tx * Q12)
            ty_q12 = round(ty * Q12)
            self.opening_targets[b] = (tx_q12 / Q12, ty_q12 / Q12)

            # Fixed-point Q12 centroid and parallel formation offset.
            n = len(enemies)
            xs_q12 = [round(float(world.x[b, s]) * Q12) for s in enemies]
            ys_q12 = [round(float(world.y[b, s]) * Q12) for s in enemies]
            cx_q12 = sum(xs_q12) // n
            cy_q12 = sum(ys_q12) // n

            orders = []
            for s, ux_q12, uy_q12 in zip(enemies, xs_q12, ys_q12, strict=True):
                if spec.formation_offset:
                    ox_q12 = tx_q12 - cx_q12 + ux_q12
                    oy_q12 = ty_q12 - cy_q12 + uy_q12
                else:
                    ox_q12, oy_q12 = tx_q12, ty_q12
                ox_q12 = max(min_x_q12, min(max_x_q12, ox_q12))
                oy_q12 = max(min_y_q12, min(max_y_q12, oy_q12))
                ox = float(np.float32(ox_q12 / Q12))
                oy = float(np.float32(oy_q12 / Q12))
                meta = _lookup_meta(int(world.unit_type[b, s]), int(world.variant[b, s]))
                if meta.has_weapon:
                    orders.append(_OpeningOrder(s, CommandKind.ATTACK_MOVE, target_xy=(ox, oy)))
                else:
                    # Unarmed support units (e.g. Medivac) advance with escort-heal waypoints.
                    self._medivac_wp[b][s] = (ox, oy)
                    self._medivac_arrived[b][s] = False
                    orders.append(_OpeningOrder(s, CommandKind.MOVE, target_xy=(ox, oy)))
            self._opening_orders[b] = orders

    def _find_heal_target(self, world: WorldState, b: int, medivac_slot: int) -> int:
        """Select an injured biological ground ally within 6.0 tiles by HP ratio and distance."""
        if not self._can_heal:
            return NO_SLOT
        if float(world.energy[b, medivac_slot]) < HEAL_AUTOCAST_MIN_ENERGY:
            return NO_SLOT
        mx = float(world.x[b, medivac_slot])
        my = float(world.y[b, medivac_slot])
        mr = float(world.radius[b, medivac_slot])
        best_slot = NO_SLOT
        best_key: tuple[float, float, int] = (math.inf, math.inf, world.n_slots)
        for t in np.flatnonzero(world.alive[b] & (world.owner[b] == 2)).tolist():
            if t == medivac_slot or bool(world.is_flying[b, t]):
                continue
            hp = float(world.health[b, t])
            hp_max = float(world.health_max[b, t])
            if hp >= hp_max or hp_max <= 0.0:
                continue
            meta = _lookup_meta(int(world.unit_type[b, t]), int(world.variant[b, t]))
            if not meta.biological:
                continue
            edge = (
                math.hypot(float(world.x[b, t]) - mx, float(world.y[b, t]) - my)
                - mr
                - float(world.radius[b, t])
            )
            if edge <= HEAL_AUTOCAST_RANGE:
                key = (hp / hp_max, edge, t)
                if key < best_key:
                    best_key = key
                    best_slot = t
        return best_slot

    def _has_engaged_bio_ally(self, world: WorldState, b: int, medivac_slot: int) -> bool:
        """Check whether an engaged biological ground ally is within 6.0 tiles."""
        mx = float(world.x[b, medivac_slot])
        my = float(world.y[b, medivac_slot])
        mr = float(world.radius[b, medivac_slot])
        for t in np.flatnonzero(world.alive[b] & (world.owner[b] == 2)).tolist():
            if t == medivac_slot or bool(world.is_flying[b, t]):
                continue
            if int(world.engaged_target_slot[b, t]) == NO_SLOT:
                continue
            meta = _lookup_meta(int(world.unit_type[b, t]), int(world.variant[b, t]))
            if not meta.biological:
                continue
            edge = (
                math.hypot(float(world.x[b, t]) - mx, float(world.y[b, t]) - my)
                - mr
                - float(world.radius[b, t])
            )
            if edge <= HEAL_AUTOCAST_RANGE:
                return True
        return False

    def _step_medivac_escort(self, world: WorldState, command: Command, b: int) -> None:
        """Keep Medivac escort moving, switch to healing injured allies, and resume waypoint."""
        wps = self._medivac_wp.get(b)
        if not wps:
            return
        arrived_map = self._medivac_arrived[b]
        for u, (wx, wy) in wps.items():
            if not (world.alive[b, u] and world.owner[b, u] == 2):
                continue
            if not arrived_map.get(u, False):
                if (
                    math.hypot(float(world.x[b, u]) - wx, float(world.y[b, u]) - wy)
                    <= MEDIVAC_ARRIVAL_TOL
                ):
                    arrived_map[u] = True
            ord_kind = int(world.order_kind[b, u])
            ord_abil = int(world.order_ability[b, u])
            if ord_kind == CommandKind.ABILITY and ord_abil == HEAL_ABILITY:
                continue
            if int(world.unit_type[b, u]) == MEDIVAC_TYPE:
                heal_t = self._find_heal_target(world, b, u)
                if heal_t != NO_SLOT:
                    command.set(b, u, CommandKind.ABILITY, target_slot=heal_t, ability=HEAL_ABILITY)
                    continue
                if self._can_heal and self._has_engaged_bio_ally(world, b, u):
                    if ord_kind == CommandKind.MOVE:
                        command.set(b, u, CommandKind.STOP)
                    continue
            if not arrived_map.get(u, False):
                if ord_kind != CommandKind.MOVE and int(command.kind[b, u]) != CommandKind.MOVE:
                    command.set(b, u, CommandKind.MOVE, target_xy=(wx, wy))

    def issue(self, world: WorldState, command: Command, active: np.ndarray) -> None:
        self._require_env()
        for b in np.flatnonzero(active).tolist():
            if self._pending.get(b, False):
                for o in self._opening_orders.get(b, ()):
                    if not (world.alive[b, o.slot] and world.owner[b, o.slot] == 2):
                        continue
                    if o.kind == CommandKind.ATTACK_UNIT:
                        if 0 <= o.target_slot < world.n_slots and world.alive[b, o.target_slot]:
                            command.set(
                                b, o.slot, CommandKind.ATTACK_UNIT, target_slot=o.target_slot
                            )
                    elif o.kind in (CommandKind.ATTACK_MOVE, CommandKind.MOVE):
                        assert o.target_xy is not None
                        command.set(b, o.slot, o.kind, target_xy=o.target_xy)
                self._pending[b] = False
            self._step_medivac_escort(world, command, b)
