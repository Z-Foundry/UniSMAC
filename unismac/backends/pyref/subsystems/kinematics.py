"""Stage 6 kinematics: integrates desired velocity into positions, rotates facing, and detects
MOVE / ATTACK_MOVE arrival.

Distinguishes two angles: `motion.heading` is internal movement direction, while `world.facing`
is observed orientation. `facing` turns toward `heading` by at most `facing_rate` rad/loop
(default `inf`, i.e., equal to `heading`), affecting observation without constraining ground
displacement (ground units change movement direction instantaneously while `facing` turns at
~1.09 rad/loop).

Optional behaviors active in both implementations: `move_lag=1` advances newly commanded units
by their previous-loop velocity (`_lagged`); `damage='sc2'` hit slow scales desired velocity by
`slow_factor` (`_apply_slow`); `slow_arrival` caps the final step to a waypoint by slowed speed
(`_slow_last_step`).

`config.charge` (Zealot Charge) also runs here: checks charge start after movement
(`_charge_start`), scales desired velocity by `charge_speed / speed` (`_charge_speed`), and
checks charge termination before movement (`_charge_end`).

`config.heal_on_arrival` invokes `abilities.autocast_on_arrival` for units that moved and
completed a MOVE order on arrival this loop.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from unismac.contract import NO_SLOT, CommandKind

from ..sim import Sim, f32
from . import abilities


def _load_atan2f() -> Callable[[float, float], float] | None:
    """Load platform C single-precision `atan2f` (falls back to `np.arctan2` if unavailable)."""
    candidates: list[str | None] = ['libm.so.6']
    found = ctypes.util.find_library('m')
    if found and found not in candidates:
        candidates.append(found)
    candidates.extend(('libSystem.B.dylib', 'ucrtbase.dll', None))
    for lib_name in candidates:
        try:
            lib = ctypes.CDLL(lib_name)
            fn: Any = lib.atan2f
        except (OSError, AttributeError):
            continue
        fn.argtypes = (ctypes.c_float, ctypes.c_float)
        fn.restype = ctypes.c_float
        return fn
    return None


_ATAN2F: Callable[[float, float], float] | None = _load_atan2f()


def _atan2f(vy: float | np.floating[Any], vx: float | np.floating[Any]) -> np.float32:
    """Single-precision `atan2f`: prefers platform libm, falls back to `np.arctan2`."""
    if _ATAN2F is not None:
        return np.float32(_ATAN2F(float(vy), float(vx)))
    return np.float32(np.arctan2(np.float32(vy), np.float32(vx)))


def _atan2f_array(vy: np.ndarray, vx: np.ndarray) -> np.ndarray:
    """Elementwise single-precision `atan2f` over 1D float32 arrays, returning float32."""
    if _ATAN2F is None:
        return np.arctan2(vy, vx).astype(np.float32, copy=False)
    fn = _ATAN2F
    return np.fromiter(
        (fn(y, x) for y, x in zip(vy.tolist(), vx.tolist(), strict=True)),
        dtype=np.float32,
        count=len(vx),
    )


# heading_blend: snap instantaneously when angle error is near 180 deg (SC2 Medivac U-turn).
REVERSE_THRESHOLD = math.pi - 1e-3

# rl_turn='curve': empirical Medivac_RL per-loop turn angle (deg) vs remaining angle error (deg),
# interpolated linearly between endpoints (0, 0) and (180, 180).
RL_TURN_TABLE = (
    (0.0, 0.0), (0.68, 0.49), (0.8, 0.55), (0.85, 0.46), (0.99, 0.57), (1.03, 0.59),
    (1.04, 0.67), (1.53, 0.87), (1.8, 1.02), (1.97, 1.15), (2.11, 1.14), (2.21, 1.21),
    (2.42, 1.4), (3.29, 1.8), (3.9, 2.15), (4.11, 2.19), (4.34, 2.29), (4.6, 2.45),
    (5.12, 2.76), (6.81, 3.61), (8.12, 4.32), (8.51, 4.5), (8.86, 4.62), (9.33, 4.84),
    (10.55, 5.56), (13.88, 7.25), (16.36, 8.46), (17.33, 9.04), (18.0, 9.35), (18.85, 9.74),
    (21.51, 11.23), (25.99, 12.46), (28.51, 12.56), (29.45, 12.53), (30.02, 12.41),
    (31.05, 12.64), (33.61, 12.61), (38.15, 12.8), (40.68, 12.85), (41.71, 12.94),
    (42.26, 12.87), (43.33, 12.97), (50.59, 13.32), (53.37, 13.6), (54.36, 13.56),
    (54.98, 13.56), (56.13, 13.69), (63.85, 14.34), (66.72, 14.45), (67.94, 14.67),
    (68.46, 14.48), (69.75, 14.68), (78.38, 15.77), (81.5, 16.03), (82.93, 16.23),
    (83.41, 16.09), (85.04, 16.49), (99.16, 19.0), (100.9, 19.29), (103.55, 19.77),
    (129.92, 30.32), (146.33, 48.52), (157.46, 80.43), (168.7, 135.86), (180.0, 180.0),
)  # fmt: skip
_RL_ERR = np.radians([e for e, _ in RL_TURN_TABLE])
_RL_TURN = np.radians([t for _, t in RL_TURN_TABLE])


def rl_turn_step(diff: float) -> float:
    """`rl_turn='curve'`: signed per-loop turn angle (rad) for remaining error `diff`,
    clamped so it never overshoots the target angle.
    """
    a = abs(diff)
    return math.copysign(min(float(np.interp(a, _RL_ERR, _RL_TURN)), a), diff)


def _check_arrival(sim: Sim, b: int, u: int) -> bool:
    """Clear MOVE / ATTACK_MOVE order when within `arrival_radius` of `buf.dest_*`.

    Checked after integration and before collision resolution so units pushed aside still count
    as arrived. With default `arrival_radius=0`, the final step snaps directly onto the waypoint.
    When `arrival_clears_waypoint` is set, also clears the waypoint so collision treats the unit
    as idle.

    Known difference: SC2 clears `orders` one loop after stopping, whereas here `order_kind`
    becomes `NOOP` on the arrival loop (positions are unaffected).
    """
    w = sim.world
    if int(w.order_kind[b, u]) not in (CommandKind.MOVE, CommandKind.ATTACK_MOVE):
        return False
    if (
        sim.config.native_stop
        and not w.is_flying[b, u]
        and (
            int(w.order_kind[b, u]) == CommandKind.MOVE
            or (
                int(w.order_kind[b, u]) == CommandKind.ATTACK_MOVE
                and int(w.engaged_target_slot[b, u]) == NO_SLOT
            )
        )
    ):
        return False
    dx = sim.buf.dest_x[b, u] - w.x[b, u]
    dy = sim.buf.dest_y[b, u] - w.y[b, u]
    if not f32(np.sqrt(dx * dx + dy * dy)) <= f32(sim.config.arrival_radius):
        return False
    w.order_kind[b, u] = CommandKind.NOOP
    w.order_target_slot[b, u] = NO_SLOT
    w.order_x[b, u] = w.order_y[b, u] = np.nan
    if sim.config.arrival_clears_waypoint:
        sim.buf.waypoint_x[b, u] = sim.buf.waypoint_y[b, u] = np.nan
    return True


def _wrap(a: float) -> float:
    """Wrap angle into `[-pi, pi)`."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _turn_facing(sim: Sim, b: int, u: int, target: float | None = None) -> None:
    """Rotate `facing` toward `target` (default `heading`) by at most `facing_rate` rad/loop."""
    w, m = sim.world, sim.motion
    goal = float(m.heading[b, u]) if target is None else target
    rate = float(sim.params.facing_rate[b, u])
    if math.isinf(rate):
        w.facing[b, u] = f32(goal)
        return
    diff = _wrap(goal - float(w.facing[b, u]))
    w.facing[b, u] = f32(_wrap(float(w.facing[b, u]) + max(-rate, min(rate, diff))))


def _apply_slow(sim: Sim, b: int, u: int) -> None:
    """`damage='sc2'` hit slow: multiply desired velocity by `slow_factor` during `[from, to]`."""
    c = sim.combat
    loop = int(sim.world.game_loop[b])
    if c.slow_from[b, u] <= loop <= c.slow_to[b, u]:
        sim.buf.vel_x[b, u] *= c.slow_factor[b, u]
        sim.buf.vel_y[b, u] *= c.slow_factor[b, u]


def _lagged(sim: Sim, b: int, u: int) -> bool:
    """`move_lag=1` on a command's activation loop: integrate previous-loop velocity
    (`motion.lag_v*` saved by commands) while rotating `facing` toward the new desired direction
    (or `heading` when stationary, which attack aims at the target). Matches lockstep SC2 where
    facing updates on the order loop while displacement lags by one loop. Returns True if handled.

    Approximation: continues the old velocity for one loop without re-checking old arrival.
    """
    m, w, buf = sim.motion, sim.world, sim.buf
    if m.lag_loop[b, u] != w.game_loop[b]:
        return False
    nvx, nvy = float(buf.vel_x[b, u]), float(buf.vel_y[b, u])
    target = math.atan2(nvy, nvx) if nvx != 0 or nvy != 0 else float(m.heading[b, u])
    vx, vy = m.lag_vx[b, u], m.lag_vy[b, u]
    if vx != 0 or vy != 0:
        w.x[b, u] += vx
        w.y[b, u] += vy
        m.heading[b, u] = _atan2f(vy, vx)
    m.speed[b, u] = np.hypot(vx, vy)
    _turn_facing(sim, b, u, target)
    return True


def _move_constant(sim: Sim, b: int, u: int) -> None:
    """Integrate desired velocity and set heading from velocity. Snaps directly onto waypoint
    when velocity equals the exact remaining offset, avoiding float error for `arrival_radius=0`.
    """
    w, buf, m = sim.world, sim.buf, sim.motion
    vx, vy = buf.vel_x[b, u], buf.vel_y[b, u]
    if vx != 0 or vy != 0:
        wx, wy = buf.waypoint_x[b, u], buf.waypoint_y[b, u]
        if vx == wx - w.x[b, u] and vy == wy - w.y[b, u]:
            w.x[b, u], w.y[b, u] = wx, wy
        else:
            w.x[b, u] += vx
            w.y[b, u] += vy
        m.heading[b, u] = _atan2f(vy, vx)
    m.speed[b, u] = np.hypot(vx, vy)
    _turn_facing(sim, b, u)


# ---- Batched versions: bit-identical per element to the scalar functions above
# (units only read/write their own kinematics state, so order does not matter).


def _apply_slow_all(sim: Sim, sel: np.ndarray) -> None:
    c, buf = sim.combat, sim.buf
    loop = sim.world.game_loop[:, None]
    hit = sel & (c.slow_from <= loop) & (loop <= c.slow_to)
    if hit.any() and sim.config.slow_arrival:
        hit = _slow_last_step(sim, hit)
    if hit.any():
        buf.vel_x[hit] *= c.slow_factor[hit]
        buf.vel_y[hit] *= c.slow_factor[hit]


def _slow_last_step(sim: Sim, hit: np.ndarray) -> np.ndarray:
    """`slow_arrival`: when a slowed unit's desired velocity is already the exact final step to
    its waypoint (length `d`, already clamped to `min(speed, d)` by steering), cap it by slowed
    speed `cap = slow_factor * speed`: keep unchanged if `d <= cap` so `_move_constant_all`
    snaps onto the waypoint, otherwise normalize to length `cap`. Returns remaining slowed units
    that still scale linearly by `slow_factor`.
    """
    w, c, buf = sim.world, sim.combat, sim.buf
    vx, vy = buf.vel_x, buf.vel_y
    with np.errstate(invalid='ignore'):
        last = hit & ((vx != 0) | (vy != 0))
        last &= (vx == buf.waypoint_x - w.x) & (vy == buf.waypoint_y - w.y)
    if not last.any():
        return hit
    d = np.hypot(vx[last], vy[last])
    cap = (c.slow_factor[last] * sim.params.speed[last]).astype(np.float32)
    k = np.where(d <= cap, np.float32(1), cap / np.maximum(d, np.float32(1e-12)))
    vx[last] = (vx[last] * k).astype(np.float32)
    vy[last] = (vy[last] * k).astype(np.float32)
    return hit & ~last


def _charge_goal(sim: Sim) -> np.ndarray:
    """Return charge target: order target for ATTACK_UNIT, engaged target for ATTACK_MOVE/idle."""
    w = sim.world
    kind = w.order_kind
    auto = (kind == CommandKind.ATTACK_MOVE) | (kind == CommandKind.NOOP)
    t = np.where(kind == CommandKind.ATTACK_UNIT, w.order_target_slot, NO_SLOT)
    return np.where(auto, w.engaged_target_slot, t).astype(np.int64)


def _charge_end(sim: Sim) -> None:
    """Pre-move: end Charge on contact (`buf.in_range`), target change/death, or duration expiry.
    If the target was stationary (`speed == 0`) on contact, arm the first-hit bonus damage.
    """
    w, c, p = sim.world, sim.combat, sim.params
    on = c.charge_target != NO_SLOT
    if not on.any():
        return
    n = w.x.shape[1]
    ct = c.charge_target.astype(np.int64)
    idx = ct % n
    goal = _charge_goal(sim)
    alive_t = np.take_along_axis(w.alive, idx, axis=1)
    still_t = np.take_along_axis(sim.motion.speed, idx, axis=1) == 0
    since = w.game_loop[:, None] - c.charge_from
    contact = on & w.alive & (goal == ct) & alive_t & sim.buf.in_range
    end = on & (~w.alive | (goal != ct) | ~alive_t | contact | (since > p.charge_max_loops))
    bonus = contact & still_t & (p.charge_bonus > 0)
    c.charge_bonus_target[bonus] = c.charge_target[bonus]
    c.charge_target[end] = NO_SLOT


def _charge_speed(sim: Sim, sel: np.ndarray) -> None:
    """Scale charging units' desired velocity by `charge_speed / speed` without overshooting."""
    c, p, buf, w = sim.combat, sim.params, sim.buf, sim.world
    on = sel & (c.charge_target != NO_SLOT) & (p.speed > 0)
    if not on.any():
        return
    k = (p.charge_speed[on] / p.speed[on]).astype(np.float32)
    vx, vy = (buf.vel_x[on] * k).astype(np.float32), (buf.vel_y[on] * k).astype(np.float32)
    dx = (buf.waypoint_x[on] - w.x[on]).astype(np.float32)
    dy = (buf.waypoint_y[on] - w.y[on]).astype(np.float32)
    with np.errstate(invalid='ignore'):
        snap = np.hypot(dx, dy) <= np.hypot(vx, vy)  # False when waypoint is NaN
    buf.vel_x[on] = np.where(snap, dx, vx)
    buf.vel_y[on] = np.where(snap, dy, vy)


def _charge_start(sim: Sim) -> None:
    """Post-move: start Charge when off cooldown, target is an alive out-of-range enemy, and
    center distance is below threshold (`charge_range_small` if target radius `<= small`, else
    `charge_range_large`).
    """
    w, c, p = sim.world, sim.combat, sim.params
    ok = w.alive & ~np.isnan(p.charge_speed) & (c.charge_target == NO_SLOT)
    if not ok.any():
        return
    loop = w.game_loop[:, None]
    ok &= (c.charge_from < 0) | (loop - c.charge_from >= p.charge_cooldown)
    goal = _charge_goal(sim)
    ok &= (goal != NO_SLOT) & ~sim.buf.in_range
    if sim.config.follow_lock or sim.config.slop_on_arrival:
        ok &= ~sim.buf.follow_hold  # units held by follow lock are not chasing, so do not charge
    if not ok.any():
        return
    idx = goal % w.x.shape[1]
    tx, ty = np.take_along_axis(w.x, idx, axis=1), np.take_along_axis(w.y, idx, axis=1)
    tr = np.take_along_axis(w.radius, idx, axis=1)
    t_alive = np.take_along_axis(w.alive, idx, axis=1)
    t_owner = np.take_along_axis(w.owner, idx, axis=1)
    gate = np.where(tr <= p.charge_small_radius, p.charge_range_small, p.charge_range_large)
    with np.errstate(invalid='ignore'):
        center = np.hypot(tx - w.x, ty - w.y)
    ok &= t_alive & (t_owner != 0) & (t_owner != w.owner) & (center < gate)
    c.charge_from[ok] = np.broadcast_to(loop, ok.shape)[ok]
    c.charge_target[ok] = goal[ok]
    c.charge_bonus_target[ok] = NO_SLOT


def _slow_or_charge(sim: Sim, alive: np.ndarray, lag: np.ndarray) -> None:
    """Apply hit slow and Charge speed boost. Active Charge ignores hit slow (e.g., Concussive
    Shells). Lagged units move at prior velocity without Charge boost.
    """
    if not sim.config.charge:
        _apply_slow_all(sim, alive)
        return
    _charge_end(sim)
    _apply_slow_all(sim, alive & (sim.combat.charge_target == NO_SLOT))
    _charge_speed(sim, alive & ~lag)


def _lag_mask(sim: Sim, sel: np.ndarray) -> np.ndarray:
    return sel & (sim.motion.lag_loop == sim.world.game_loop[:, None])


def _wrap_all(a: np.ndarray) -> np.ndarray:
    return np.remainder(a + math.pi, 2.0 * math.pi) - math.pi


def _turn_facing_all(sim: Sim, sel: np.ndarray) -> None:
    """Batched `_turn_facing` toward `motion.heading`."""
    w, m = sim.world, sim.motion
    goal = m.heading[sel].astype(np.float64)
    rate = sim.params.facing_rate[sel].astype(np.float64)
    face = w.facing[sel].astype(np.float64)
    inf = np.isinf(rate)
    diff = _wrap_all(goal - face)
    # max(-rate, min(rate, diff)): matches Python min/max NaN semantics (returns first arg)
    lo = np.where(diff < rate, diff, rate)
    lo = np.where(np.isnan(diff), rate, lo)
    step = np.where(-rate < lo, lo, -rate)
    w.facing[sel] = np.where(inf, goal, _wrap_all(face + step)).astype(np.float32)


def _move_constant_all(sim: Sim, sel: np.ndarray) -> None:
    w, buf, m = sim.world, sim.buf, sim.motion
    vx, vy = buf.vel_x[sel], buf.vel_y[sel]
    x, y = w.x[sel], w.y[sel]
    wx, wy = buf.waypoint_x[sel], buf.waypoint_y[sel]
    move = (vx != 0) | (vy != 0)
    snap = move & (vx == wx - x) & (vy == wy - y)
    w.x[sel] = np.where(snap, wx, np.where(move, x + vx, x))
    w.y[sel] = np.where(snap, wy, np.where(move, y + vy, y))
    if move.any():
        h = m.heading[sel]
        h[move] = _atan2f_array(vy[move], vx[move])
        m.heading[sel] = h
    m.speed[sel] = np.hypot(vx, vy)
    _turn_facing_all(sim, sel)


def _check_arrival_all(sim: Sim, sel: np.ndarray) -> None:
    """Batched `_check_arrival`; zeros speed for arriving units."""
    w, buf = sim.world, sim.buf
    kind = w.order_kind
    sel = sel & ((kind == CommandKind.MOVE) | (kind == CommandKind.ATTACK_MOVE))
    if sim.config.native_stop:
        native_pt = ~w.is_flying & (
            (kind == CommandKind.MOVE)
            | ((kind == CommandKind.ATTACK_MOVE) & (w.engaged_target_slot == NO_SLOT))
        )
        sel = sel & ~native_pt
    with np.errstate(invalid='ignore'):
        dx = buf.dest_x - w.x
        dy = buf.dest_y - w.y
        hit = sel & (np.sqrt(dx * dx + dy * dy) <= f32(sim.config.arrival_radius))
    if hit.any():
        w.order_kind[hit] = CommandKind.NOOP
        w.order_target_slot[hit] = NO_SLOT
        w.order_x[hit] = np.nan
        w.order_y[hit] = np.nan
        sim.motion.speed[hit] = 0
        if sim.config.arrival_clears_waypoint:
            buf.waypoint_x[hit] = np.nan
            buf.waypoint_y[hit] = np.nan


def _lagged_each(sim: Sim, lag: np.ndarray) -> None:
    for b, u in zip(*np.nonzero(lag), strict=True):
        _lagged(sim, int(b), int(u))


def _arrival_snapshot(sim: Sim) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """`heal_on_arrival`: snapshot MOVE units and positions before integration."""
    if not sim.config.heal_on_arrival:
        return None
    w = sim.world
    return w.order_kind == CommandKind.MOVE, w.x.copy(), w.y.copy()


def _arrival_autocast(sim: Sim, snap: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> None:
    """`heal_on_arrival`: run autocast for units that moved and arrived at a MOVE dest."""
    if snap is None:
        return
    w = sim.world
    moving, x0, y0 = snap
    arrived = moving & (w.order_kind == CommandKind.NOOP) & ((w.x != x0) | (w.y != y0))
    if arrived.any():
        abilities.autocast_on_arrival(sim, arrived)


def _start_positions(sim: Sim) -> None:
    """Save pre-movement positions (`buf.move_x0`, `buf.move_y0`) for collision when
    `mover_speed_cap` or `collision='native'` is active.
    """
    if sim.config.mover_speed_cap or sim.config.collision == 'native':
        sim.buf.move_x0[:] = sim.world.x
        sim.buf.move_y0[:] = sim.world.y


def constant_speed(sim: Sim) -> None:
    """Constant-speed integration: actual velocity equals desired velocity with instant turning."""
    alive = sim.world.alive
    _start_positions(sim)
    arrival = _arrival_snapshot(sim)
    lag = _lag_mask(sim, alive)
    _slow_or_charge(sim, alive, lag)
    _lagged_each(sim, lag)
    _move_constant_all(sim, alive & ~lag)
    _check_arrival_all(sim, alive)
    _arrival_autocast(sim, arrival)
    if sim.config.charge:
        _charge_start(sim)


def stop_distance(speed: float, decel: float) -> float:
    """Discrete stopping distance when decelerating by `decel` per loop:
    `sum_{k=1..n} (speed - k*decel)` with `n = floor(speed / decel)` (~1.27417 for full-speed
    Medivac, matching SC2 STOP coast distance).
    """
    n = math.floor(speed / decel)
    return n * speed - decel * n * (n + 1) / 2


def _steer_heading(
    heading: float, target: float, turn: float, blend: float, curve: bool = False
) -> float:
    """Advance `heading` toward `target` for one loop. Uses exponential `blend` when finite,
    capped by `turn_rate` (or empirical `rl_turn_step` when `curve=True`), or snaps on near-180
    reversals.
    """
    diff = _wrap(target - heading)
    if not math.isnan(blend):
        if blend >= 0.999 or abs(diff) >= REVERSE_THRESHOLD:
            return target
        if math.isinf(turn):
            return _wrap(heading + blend * diff)
        if curve:
            return _wrap(heading + rl_turn_step(diff))
        return _wrap(heading + math.copysign(min(blend * abs(diff), turn), diff))
    if math.isinf(turn):
        return target
    return _wrap(heading + max(-turn, min(turn, diff)))


def accel(sim: Sim) -> None:
    """Acceleration and turning dynamics (parameters from `UnitTypeSpec.measured`).

    Per game loop:

    1. Turn `heading` toward desired velocity (unchanged when desired speed is 0):
       * With `heading_blend` beta, shrink error exponentially by beta per loop (snapping when
         beta >= 0.999 or near 180-deg reversal); when `turn_rate` is also set, cap each step
         by `turn_rate` (or `rl_turn_step` under `rl_turn='curve'`).
       * Otherwise turn at most `turn_rate` rad/loop and scale desired speed by
         `max(cos(residual), 0)`; snaps instantly if both are unset.
    2. Speed: if an active destination (`buf.dest_*`) is within `stop_distance(cur, decel) +
       decel`, brake by `decel` down to `min(accel, remain)` (re-evaluated each loop without
       latching so units never stop short); otherwise accelerate/decelerate toward desired speed.
    3. Advance along `heading`, snapping onto the waypoint when within one step so arrival lands
       cleanly on the destination.
    4. Rotate `facing` toward `heading` (`facing_rate`) and check arrival (zeroing speed).

    Units with default accel/decel/turn_rate/heading_blend degrade bit-identically to
    `constant_speed`.
    """
    w, p, buf, m = sim.world, sim.params, sim.buf, sim.motion
    alive = w.alive
    _start_positions(sim)
    arrival = _arrival_snapshot(sim)
    lag = _lag_mask(sim, alive)
    _slow_or_charge(sim, alive, lag)
    _lagged_each(sim, lag)
    const = np.isinf(p.accel) & np.isinf(p.decel) & np.isinf(p.turn_rate)
    const &= np.isnan(p.heading_blend)
    _move_constant_all(sim, alive & ~lag & const)  # default params: degenerate to constant_speed
    _check_arrival_all(sim, alive & (lag | const))
    curve = sim.config.rl_turn == 'curve'
    for b, u in zip(*np.nonzero(alive & ~lag & ~const), strict=True):
        b, u = int(b), int(u)
        a, d = float(p.accel[b, u]), float(p.decel[b, u])
        turn, blend = float(p.turn_rate[b, u]), float(p.heading_blend[b, u])
        vx, vy = float(buf.vel_x[b, u]), float(buf.vel_y[b, u])
        want = math.hypot(vx, vy)
        heading = float(m.heading[b, u])
        if want > 0:
            target = math.atan2(vy, vx)
            heading = _steer_heading(heading, target, turn, blend, curve)
            if math.isnan(blend):
                want *= max(math.cos(_wrap(target - heading)), 0.0)
        x, y = float(w.x[b, u]), float(w.y[b, u])
        cur = float(m.speed[b, u])
        remain = math.inf
        if not np.isnan(buf.dest_x[b, u]) and not math.isinf(d):
            remain = math.hypot(float(buf.dest_x[b, u]) - x, float(buf.dest_y[b, u]) - y)
        if remain <= stop_distance(cur, d) + d:
            # Brake with floor min(accel, remain) so the unit never stops short of dest.
            speed = max(cur - d, min(a, remain))
        else:
            speed = min(want, cur + a) if want >= cur else max(want, cur - d)
        step, snap = speed, False
        if not np.isnan(buf.waypoint_x[b, u]):
            wx, wy = float(buf.waypoint_x[b, u]), float(buf.waypoint_y[b, u])
            to_wp = math.hypot(wx - x, wy - y)
            if 0 < to_wp <= step:
                step, snap, speed = to_wp, True, to_wp
        m.heading[b, u] = heading
        if snap:
            w.x[b, u], w.y[b, u] = buf.waypoint_x[b, u], buf.waypoint_y[b, u]
        elif step > 0:
            w.x[b, u] = f32(x + math.cos(heading) * step)
            w.y[b, u] = f32(y + math.sin(heading) * step)
        m.speed[b, u] = speed
        _turn_facing(sim, b, u)
        if _check_arrival(sim, b, u):
            m.speed[b, u] = 0
    _arrival_autocast(sim, arrival)
    if sim.config.charge:
        _charge_start(sim)


IMPLS = {'constant_speed': constant_speed, 'accel': accel}
