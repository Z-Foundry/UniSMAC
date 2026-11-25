"""Stage 3b abilities: Medivac Heal (`MedivacHeal`, ability 386) and energy regeneration
(`abilities='heal'`).

Constants match SC2 4.10 empirical measurements (1/4096 fixed-point values; standard Medivac
and Medivac_RL share identical healing dynamics).

**Channeling**: starting one loop after the heal order takes effect, the Medivac begins
channeling when edge distance `<= HEAL_RANGE`. The first channeling loop applies two ticks
(`2 * HEAL_PER_TICK`), followed by one tick per loop, consuming `HEAL_ENERGY_PER_TICK` each
(scaled proportionally on the final partial tick). Each tick requires energy `>= HEAL_MIN_ENERGY`.
If distance exceeds `HEAL_LEASH`, channeling breaks and the unit closes back to `HEAL_RANGE`
(restarting with two ticks).

**Movement**: flies toward the target while out of range (pathing reads `buf.in_range`). Units
with finite deceleration brake early when `edge_distance - speed <= HEAL_RANGE`, take one
extra deceleration step on the loop channeling starts (2x decel total that loop), and coast
to a stop while channeling.

**Completion**:

* Ends `HEAL_FULL_LOOPS - 1` loops after the target reaches full health (returning to `NOOP`,
  or `HOLD` if autocast from `HOLD`).
* Ends when energy drops below `HEAL_MIN_ENERGY` (the loop after the last tick).
* Cleared by damage cleanup if the target dies.

**Autocast** (`heal_autocast`; enabled on standard Medivacs, not `Medivac_RL`): applies only
to `Variant.STD` Medivacs. When idle (`NOOP`) or on `HOLD` with energy `>= AUTOCAST_MIN_ENERGY`,
scans every `AUTOCAST_PERIOD` loops (`game_loop % 8 == AUTOCAST_PHASE`; shifted by
`combat.scan_shift` under `scan_phase='per_unit'`, or sharing targeting phase
`(game_loop - scan_shift) % 8 == acquire_start_loop % 8` under `'tag_order'` / `'tag_rotate'`).
Idle scans wounded allies within `AUTOCAST_RANGE` and flies to them; `HOLD` scans only within
`HEAL_RANGE` and returns to `HOLD` afterward. Picks lowest health (or lowest health fraction
when `heal_pick='fraction'`), breaking ties by nearest distance then lowest slot. Autocast
orders start channeling two loops later (one loop later than explicit commands). MOVE units do
not autocast, except when `heal_on_arrival` triggers `autocast_on_arrival` on the arrival loop
regardless of phase. When `heal_facing` is enabled, autocast starts immediately if facing error
`<= HEAL_ARC`, otherwise turns `facing` toward the target each loop until residual angle
`<= HEAL_ARC`.

**Energy**: all alive units with `energy_max > 0` regenerate `ENERGY_REGEN` per loop (0.5625
per game second), clamped at `energy_max`, including while channeling. Active only when
`abilities='heal'` (`abilities='none'` leaves energy unchanged).

**Order validation**: target must be a living, non-self, allied ground biological unit below
max health, and caster energy must be `>= HEAL_MIN_ENERGY`. Invalid commands are silently
ignored (preserving the existing order), matching SC2.

Known differences and assumptions:

* `AUTOCAST_PHASE` is aligned to standard explicit-spawn sessions.
* Whether full-health completion rescans immediately or waits for the next scan period is
  modeled as waiting for the next period.
* Threshold constants (`AUTOCAST_MIN_ENERGY=5`, `AUTOCAST_RANGE=6`, `HEAL_MIN_ENERGY=1`) are
  set within observed empirical brackets.
* Target death clears the heal order on the death loop (SC2 clears debug-kill orders 2 loops
  later).
"""

from __future__ import annotations

import math

import numpy as np

from unismac.contract import NO_SLOT, CommandKind, Variant

from ..sim import Sim, edge_distance, f32

HEAL_ABILITY = 386  # MedivacHeal (cast_range 4, autocastable)
HEAL_CASTERS = frozenset({54})  # Medivac (standard and _RL)
HEAL_RANGE = 4.0  # edge distance to start channeling
HEAL_LEASH = 5.0  # max edge distance while channeling (breaks above 5.0)
HEAL_PER_TICK = 2302 / 4096  # ~0.562 HP/loop (~8.99 HP/game-sec)
HEAL_ENERGY_PER_TICK = 759 / 4096  # ~0.185 energy/tick (~3.03 HP per energy)
HEAL_MIN_ENERGY = 1.0
HEAL_FULL_LOOPS = 3  # finish order on the 3rd loop at full health
HEAL_START_DELAY = 1  # loops after order activation before channeling can start
ENERGY_REGEN = 144 / 4096  # ~0.0352 energy/loop
AUTOCAST_MIN_ENERGY = 5.0
AUTOCAST_RANGE = 6.0
AUTOCAST_PERIOD = 8
AUTOCAST_PHASE = 5
# heal_facing: maximum residual facing error (rad) allowed to start channeling heal
# (empirically bracketed in [0.121, 0.176); matches unit attack_arc = 0.17).
HEAL_ARC = 0.17


def can_heal(sim: Sim, b: int, u: int, t: int) -> bool:
    """Return True if unit `u` can currently issue a heal order on target `t`."""
    w = sim.world
    return bool(
        int(w.unit_type[b, u]) in HEAL_CASTERS
        and w.energy[b, u] >= f32(HEAL_MIN_ENERGY)
        and 0 <= t < sim.n_slots
        and t != u
        and w.alive[b, t]
        and w.owner[b, t] == w.owner[b, u]
        and sim.params.biological[b, t]
        and not w.is_flying[b, t]
        and w.health[b, t] < w.health_max[b, t]
    )


def start_order(sim: Sim, b: int, u: int, t: int, active_loop: int, hold_after: bool) -> None:
    """Write a heal order; channeling may begin at `game_loop = active_loop + HEAL_START_DELAY`."""
    w, a = sim.world, sim.ability
    w.order_kind[b, u] = CommandKind.ABILITY
    w.order_ability[b, u] = HEAL_ABILITY
    w.order_target_slot[b, u] = t
    w.order_x[b, u] = w.order_y[b, u] = np.nan
    a.channel[b, u] = False
    a.ready_loop[b, u] = active_loop + HEAL_START_DELAY
    a.full_loops[b, u] = 0
    a.hold_after[b, u] = hold_after
    a.face[b, u] = False


def reset_state(sim: Sim, b: int, u: int) -> None:
    a = sim.ability
    a.channel[b, u] = False
    a.ready_loop[b, u] = -1
    a.full_loops[b, u] = 0
    a.hold_after[b, u] = False
    a.face[b, u] = False


def _finish(sim: Sim, b: int, u: int) -> None:
    w = sim.world
    w.order_kind[b, u] = CommandKind.HOLD if sim.ability.hold_after[b, u] else CommandKind.NOOP
    w.order_ability[b, u] = 0
    w.order_target_slot[b, u] = NO_SLOT
    w.order_x[b, u] = w.order_y[b, u] = np.nan
    reset_state(sim, b, u)


def _tick(sim: Sim, b: int, u: int, t: int) -> bool:
    """Apply one heal tick; return False without healing if low energy or target is full."""
    w = sim.world
    if w.energy[b, u] < f32(HEAL_MIN_ENERGY):
        return False
    gap = float(w.health_max[b, t] - w.health[b, t])
    if gap <= 0:
        return False
    heal = min(HEAL_PER_TICK, gap)
    w.health[b, t] = f32(w.health[b, t] + f32(heal))
    w.energy[b, u] = f32(w.energy[b, u] - f32(HEAL_ENERGY_PER_TICK * heal / HEAL_PER_TICK))
    return True


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _angle_to(sim: Sim, b: int, u: int, t: int) -> float:
    w = sim.world
    return math.atan2(float(w.y[b, t] - w.y[b, u]), float(w.x[b, t] - w.x[b, u]))


def _turned(sim: Sim, b: int, u: int, t: int) -> bool:
    """`heal_facing`: aim heading toward `t` and return whether residual facing error
    after this loop's turn is `<= HEAL_ARC` (same logic as `attack._face`).
    """
    angle = _angle_to(sim, b, u, t)
    sim.motion.heading[b, u] = f32(angle)
    left = abs(_wrap(angle - float(sim.world.facing[b, u]))) - float(sim.params.facing_rate[b, u])
    return left <= HEAL_ARC


def _channel(sim: Sim, b: int, u: int, loop: int) -> None:
    """Advance one heal order by one loop and set `buf.in_range` (holds pathing when True)."""
    w, a, buf = sim.world, sim.ability, sim.buf
    t = int(w.order_target_slot[b, u])
    if w.energy[b, u] < f32(HEAL_MIN_ENERGY):
        _finish(sim, b, u)
        return
    if w.health[b, t] >= w.health_max[b, t]:
        a.full_loops[b, u] += 1
        if a.full_loops[b, u] >= HEAL_FULL_LOOPS:
            _finish(sim, b, u)
            return
    else:
        a.full_loops[b, u] = 0
    edge = float(edge_distance(w, b, u, t))
    decel = float(sim.params.decel[b, u])
    if a.channel[b, u]:
        if edge > HEAL_LEASH:
            a.channel[b, u] = False  # leash broken; close back in
        else:
            _tick(sim, b, u, t)
    elif (
        loop >= a.ready_loop[b, u]
        and edge <= HEAL_RANGE
        and (not a.face[b, u] or _turned(sim, b, u, t))
    ):
        a.channel[b, u] = True
        _tick(sim, b, u, t)
        _tick(sim, b, u, t)
        if np.isfinite(decel):
            m = sim.motion
            m.speed[b, u] = f32(max(float(m.speed[b, u]) - decel, 0.0))
    if a.channel[b, u]:
        buf.in_range[b, u] = True
        return
    lookahead = float(sim.motion.speed[b, u]) if np.isfinite(decel) else 0.0
    buf.in_range[b, u] = edge - lookahead <= HEAL_RANGE


def _autocast(sim: Sim, b: int, u: int) -> None:
    w = sim.world
    kind = int(w.order_kind[b, u])
    if kind not in (CommandKind.NOOP, CommandKind.HOLD):
        return
    if w.energy[b, u] < f32(AUTOCAST_MIN_ENERGY):
        return
    reach = HEAL_RANGE if kind == CommandKind.HOLD else AUTOCAST_RANGE
    fraction = sim.config.heal_pick == 'fraction'
    best, key = NO_SLOT, (np.inf, np.inf)
    for t in range(sim.n_slots):
        if not can_heal(sim, b, u, t):
            continue
        edge = float(edge_distance(w, b, u, t))
        if edge > reach:
            continue
        life = float(w.health[b, t])
        k = (life / float(w.health_max[b, t]) if fraction else life, edge)
        if k < key:
            best, key = t, k
    if best != NO_SLOT:
        loop = int(w.game_loop[b])
        start_order(sim, b, u, best, loop + 1, hold_after=kind == CommandKind.HOLD)
        edge = float(edge_distance(w, b, u, best))
        decel = float(sim.params.decel[b, u])
        lookahead = float(sim.motion.speed[b, u]) if np.isfinite(decel) else 0.0
        sim.buf.in_range[b, u] = edge - lookahead <= HEAL_RANGE
        if sim.config.heal_facing:
            _face_start(sim, b, u, best, loop, edge)


def _face_start(sim: Sim, b: int, u: int, t: int, loop: int, edge: float) -> None:
    """`heal_facing`: gate autocast channeling by facing from the next loop, or start
    immediately (two ticks, matching `_channel`) if already in range and facing the target.
    """
    a = sim.ability
    a.face[b, u] = True
    a.ready_loop[b, u] = loop + 1
    if edge > HEAL_RANGE:
        return
    if abs(_wrap(_angle_to(sim, b, u, t) - float(sim.world.facing[b, u]))) > HEAL_ARC:
        return
    a.channel[b, u] = True
    _tick(sim, b, u, t)
    _tick(sim, b, u, t)
    decel = float(sim.params.decel[b, u])
    if np.isfinite(decel):
        m = sim.motion
        m.speed[b, u] = f32(max(float(m.speed[b, u]) - decel, 0.0))
    sim.buf.in_range[b, u] = True


def autocast_on_arrival(sim: Sim, arrived: np.ndarray) -> None:
    """`heal_on_arrival`: run an immediate autocast scan (ignoring phase) for standard Medivacs
    that moved and arrived at a MOVE destination this loop (called from kinematics).
    """
    cfg = sim.config
    if cfg.abilities != 'heal' or not cfg.heal_autocast:
        return
    w = sim.world
    caster = np.isin(w.unit_type, list(HEAL_CASTERS)) & (w.variant == Variant.STD)
    for b, u in zip(*np.nonzero(arrived & caster & w.alive), strict=True):
        _autocast(sim, int(b), int(u))


def none(sim: Sim) -> None:
    """Abilities disabled (default): energy remains unchanged."""


def heal(sim: Sim) -> None:
    """Medivac heal, autocast, and energy regeneration (see module docstring)."""
    w = sim.world
    autocast = sim.config.heal_autocast
    # Visit only units that may channel or autocast (row-major order): other units are no-ops
    # and cannot become candidates mid-loop. Under scan_phase='per_unit', each unit shifts by
    # combat.scan_shift (default 0); under 'tag_order'/'tag_rotate', autocast shares targeting's
    # periodic scan beat.
    shift = sim.combat.scan_shift
    if sim.config.native_scan_cadence:
        phase = 0
    else:
        phase = AUTOCAST_PHASE
        if sim.config.scan_phase in ('tag_order', 'tag_rotate'):
            phase = sim.config.acquire_start_loop % AUTOCAST_PERIOD
    scan_m = ((w.game_loop[:, None] - shift) % AUTOCAST_PERIOD == phase) & autocast
    chan = (w.order_kind == CommandKind.ABILITY) & (w.order_ability == HEAL_ABILITY)
    caster = np.isin(w.unit_type, list(HEAL_CASTERS)) & (w.variant == Variant.STD)
    cand = w.alive & (chan | (caster & scan_m))
    for b, u in zip(*np.nonzero(cand), strict=True):
        b, u = int(b), int(u)
        loop = int(w.game_loop[b])
        scan = bool(scan_m[b, u])
        if (
            int(w.order_kind[b, u]) == CommandKind.ABILITY
            and int(w.order_ability[b, u]) == HEAL_ABILITY
        ):
            _channel(sim, b, u, loop)
        if scan and int(w.unit_type[b, u]) in HEAL_CASTERS and int(w.variant[b, u]) == Variant.STD:
            _autocast(sim, b, u)
    # Energy regen: each env updates only its own row, so batching after channeling is identical.
    regen = w.alive & (w.energy_max > 0)
    w.energy[regen] = np.minimum(w.energy[regen] + f32(ENERGY_REGEN), w.energy_max[regen])


IMPLS = {'none': none, 'heal': heal}
