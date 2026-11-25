"""Fixed-point Q12 combat, target scoring, assist-on-hit, and loop-order primitives.

All coordinates, radii, squared distances, strict weapon ranges, and target
scores use Q12 fixed-point integers (`round(v * 4096)`) or scalar Python `math`
to preserve bitwise determinism regardless of SIMD instruction sets.
"""

from __future__ import annotations

import math

import numpy as np

from unismac.contract import NO_SLOT, CommandKind

from .. import rng
from ..sim import Sim, f32, is_enemy, weapon_for

Q12_ONE = 4096
LEASH_RESET_RADIUS_Q12 = 5 * Q12_ONE  # 5.0 tiles = 20480 Q12
LEASH_RESET_RADIUS_SQ = LEASH_RESET_RADIUS_Q12 * LEASH_RESET_RADIUS_Q12
ASSIST_TTL_LOOPS = 32
THREAT_TTL_LOOPS = 48
ATTACK_TARGET_PRIORITY = 20
THREAT_BIT = 0x80000000
HEAL_ABILITY_ID = 386


def q12_scalar(val: float) -> int:
    """Convert float scalar to Q12 integer using IEEE round-half-to-even."""
    return int(round(float(val) * Q12_ONE))


def unit_xy_q12(sim: Sim, b: int, u: int) -> tuple[int, int]:
    """Return unit `u`'s Q12 coordinates `(x_q, y_q)`."""
    w = sim.world
    return q12_scalar(float(w.x[b, u])), q12_scalar(float(w.y[b, u]))


def body_radius_q12(sim: Sim, b: int, u: int) -> int:
    """Return unit body radius in Q12 units (`world.radius`)."""
    return q12_scalar(float(sim.world.radius[b, u]))


def range_radius_q12(sim: Sim, b: int, u: int) -> int:
    """Return range/acquisition radius in Q12 (`observed` or `measured`)."""
    if sim.config.range_radius == 'measured':
        return q12_scalar(float(sim.params.range_radius[b, u]))
    return q12_scalar(float(sim.world.radius[b, u]))


def d2_raw(ax_q: int, ay_q: int, bx_q: int, by_q: int) -> int:
    """Q24 center-to-center squared distance `dx^2 + dy^2`."""
    dx = bx_q - ax_q
    dy = by_q - ay_q
    return dx * dx + dy * dy


def d2q(ax_q: int, ay_q: int, bx_q: int, by_q: int) -> int:
    """Q12 center-to-center squared distance `((dx^2 >> 12) + (dy^2 >> 12))`."""
    dx = bx_q - ax_q
    dy = by_q - ay_q
    return ((dx * dx) >> 12) + ((dy * dy) >> 12)


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def is_strict_range_q12(sim: Sim, b: int, u: int, t: int, wi: int | None = None) -> bool:
    """Strict Q12 weapon range check (without `follow_slop`)."""
    if wi is None:
        wi = weapon_for(sim, b, u, t)
    if wi < 0:
        return False
    ax_q, ay_q = unit_xy_q12(sim, b, u)
    tx_q, ty_q = unit_xy_q12(sim, b, t)
    ru_q = range_radius_q12(sim, b, u)
    rt_q = range_radius_q12(sim, b, t)
    reach_q = q12_scalar(float(sim.params.weapon_range[b, u, wi]))
    lim_q = ru_q + reach_q + rt_q
    if lim_q < 0:
        return False
    return d2_raw(ax_q, ay_q, tx_q, ty_q) <= lim_q * lim_q


def is_in_acquire_q12(sim: Sim, b: int, u: int, t: int, wi: int) -> bool:
    """Q12 target acquisition range check (`edge <= max(weapon_range + acquire_margin,
    acquire_min_range)`).
    """
    cfg = sim.config
    ax_q, ay_q = unit_xy_q12(sim, b, u)
    tx_q, ty_q = unit_xy_q12(sim, b, t)
    ru_q = range_radius_q12(sim, b, u)
    rt_q = range_radius_q12(sim, b, t)
    reach = float(sim.params.weapon_range[b, u, wi])
    acq_q = max(q12_scalar(reach + cfg.acquire_margin), q12_scalar(cfg.acquire_min_range))
    lim_q = ru_q + acq_q + rt_q
    if lim_q < 0:
        return False
    return d2_raw(ax_q, ay_q, tx_q, ty_q) <= lim_q * lim_q


def is_in_sight_q12(sim: Sim, b: int, u: int, t: int) -> bool:
    """Q12 sight radius check (`center_dist <= sight`)."""
    ax_q, ay_q = unit_xy_q12(sim, b, u)
    tx_q, ty_q = unit_xy_q12(sim, b, t)
    sight_q = q12_scalar(float(sim.params.sight[b, u]))
    return d2_raw(ax_q, ay_q, tx_q, ty_q) <= sight_q * sight_q


def is_seen_by(sim: Sim, b: int, viewer: int, target: int) -> bool:
    """Check true vision when `visibility='sc2'`; always True otherwise."""
    if sim.config.visibility != 'sc2':
        return True
    owner = int(sim.world.owner[b, viewer])
    if owner <= 0:
        return False
    return bool(sim.vision.true_sight[b, target, owner - 1])


def weapon_busy(sim: Sim, b: int, u: int) -> bool:
    """Return True if unit `u`'s weapon is in an uninterruptible windup or backswing window."""
    w, c = sim.world, sim.combat
    if int(c.windup_left[b, u]) >= 0:
        wt = int(c.windup_target[b, u])
        if wt != NO_SLOT and w.alive[b, wt] and w.health[b, wt] > 0:
            return True
    return bool(sim.config.native_backswing and int(c.backswing_left[b, u]) > 0)


def _facing_ok_for_sticky(sim: Sim, b: int, u: int, t: int) -> bool:
    """Fixed-facing units require the target within `attack_arc` for sticky bonus;
    turning/turret units pass.
    """
    p, w = sim.params, sim.world
    if float(p.facing_rate[b, u]) > 0.0 or not math.isnan(float(p.turret_rate[b, u])):
        return True
    arc = float(p.attack_arc[b, u])
    if math.isinf(arc):
        return True
    ang = math.atan2(float(w.y[b, t]) - float(w.y[b, u]), float(w.x[b, t]) - float(w.x[b, u]))
    return abs(_wrap_angle(ang - float(w.facing[b, u]))) <= arc


def target_armed(sim: Sim, b: int, t: int) -> bool:
    """Return True if target `t` counts as an armed threat (sets bit 31 in score)."""
    if bool(sim.params.has_weapon[b, t].any()):
        return True
    if sim.config.heal_counts_armed:
        w = sim.world
        return (
            int(w.order_kind[b, t]) == CommandKind.ABILITY
            and int(w.order_ability[b, t]) == HEAL_ABILITY_ID
        )
    return False


def base_score(sim: Sim, b: int, t: int) -> int:
    """Base threat priority score: `armed_bit | (20 << 8) | 1`."""
    return (THREAT_BIT if target_armed(sim, b, t) else 0) | (ATTACK_TARGET_PRIORITY << 8) | 1


def score_target_q12(sim: Sim, b: int, u: int, t: int, cur: int, wi: int | None = None) -> int:
    """Target priority score: strict-range current target `| 0x6`, other in-range `| 0x4`, else
    `+0`.
    """
    s = base_score(sim, b, t)
    if is_strict_range_q12(sim, b, u, t, wi):
        if t == cur and _facing_ok_for_sticky(sim, b, u, t):
            s |= 0x6
        else:
            s |= 0x4
    return s


def compare_key(
    sim: Sim, b: int, u: int, t: int, cur: int, wi: int | None = None
) -> tuple[int, int, int]:
    """Total-order target comparison key (smaller is preferred): `(-score, d2q, spawn_rank)`."""
    ux_q, uy_q = unit_xy_q12(sim, b, u)
    tx_q, ty_q = unit_xy_q12(sim, b, t)
    s = score_target_q12(sim, b, u, t, cur, wi)
    return (-s, d2q(ux_q, uy_q, tx_q, ty_q), int(sim.combat.spawn_rank[b, t]))


def defer_key(sim: Sim, b: int, u: int, t: int) -> tuple[int, int, int]:
    """Deferred-target comparison key: `(-base_score, d2q, spawn_rank)`."""
    ux_q, uy_q = unit_xy_q12(sim, b, u)
    tx_q, ty_q = unit_xy_q12(sim, b, t)
    return (-base_score(sim, b, t), d2q(ux_q, uy_q, tx_q, ty_q), int(sim.combat.spawn_rank[b, t]))


def store_defer_target(sim: Sim, b: int, u: int, cand: int) -> None:
    """Store the higher-priority candidate in `combat.defer_target[b, u]` while weapon is busy."""
    w, c = sim.world, sim.combat
    if cand == NO_SLOT or not w.alive[b, cand] or w.health[b, cand] <= 0:
        return
    if bool(c.engaged_transient[b, u]):
        c.defer_abort[b, u] = True
    old = int(c.defer_target[b, u])
    if old == cand:
        return
    if old == NO_SLOT or not w.alive[b, old] or w.health[b, old] <= 0:
        c.defer_target[b, u] = cand
        return
    # Replace when new candidate has better or equal priority and non-greater distance.
    ux_q, uy_q = unit_xy_q12(sim, b, u)
    nx_q, ny_q = unit_xy_q12(sim, b, cand)
    ox_q, oy_q = unit_xy_q12(sim, b, old)
    new_pr = (-base_score(sim, b, cand), d2q(ux_q, uy_q, nx_q, ny_q))
    old_pr = (-base_score(sim, b, old), d2q(ux_q, uy_q, ox_q, oy_q))
    if new_pr <= old_pr:
        c.defer_target[b, u] = cand


def tick_native_timers(sim: Sim) -> None:
    """Decrement assist-dedup, threat-memory, and backswing timers for living units before Pass
    1.
    """
    cfg, w, c = sim.config, sim.world, sim.combat
    alive = w.alive
    if cfg.native_assist_on_hit:
        a3 = alive[:, :, None]
        dec_assist = a3 & (c.assist_dedup_ttl > 0)
        if dec_assist.any():
            c.assist_dedup_ttl[dec_assist] -= 1
            expired = (
                alive[:, :, None] & (c.assist_dedup_ttl == 0) & (c.assist_dedup_slot != NO_SLOT)
            )
            c.assist_dedup_slot[expired] = NO_SLOT
        dec_th = a3 & (c.threat_ttl > 0)
        if dec_th.any():
            c.threat_ttl[dec_th] -= 1
            expired_t = alive[:, :, None] & (c.threat_ttl == 0) & (c.threat_slot != NO_SLOT)
            c.threat_slot[expired_t] = NO_SLOT
    if cfg.native_backswing:
        dec_bs = alive & (c.backswing_left > 0)
        if dec_bs.any():
            c.backswing_left[dec_bs] -= 1


def assist_throttle_record(sim: Sim, b: int, victim: int, attacker: int) -> bool:
    """Two-slot 32-loop assist deduplication table per victim (fills slot 1 before slot 0)."""
    c = sim.combat
    s0 = int(c.assist_dedup_slot[b, victim, 0])
    s1 = int(c.assist_dedup_slot[b, victim, 1])
    if s0 == attacker or s1 == attacker:
        return False
    if s1 == NO_SLOT:
        c.assist_dedup_slot[b, victim, 1] = attacker
        c.assist_dedup_ttl[b, victim, 1] = ASSIST_TTL_LOOPS
        return True
    if s0 == NO_SLOT:
        c.assist_dedup_slot[b, victim, 0] = attacker
        c.assist_dedup_ttl[b, victim, 0] = ASSIST_TTL_LOOPS
        return True
    return False


def add_threat_memory(sim: Sim, b: int, u: int, attacker: int) -> None:
    """Record or refresh an attacker in unit `u`'s 4-slot 48-loop threat memory."""
    c = sim.combat
    slots = c.threat_slot[b, u]
    ttls = c.threat_ttl[b, u]
    for k in range(4):
        if int(slots[k]) == attacker:
            ttls[k] = THREAT_TTL_LOOPS
            return
    for k in range(4):
        if int(slots[k]) == NO_SLOT:
            slots[k] = attacker
            ttls[k] = THREAT_TTL_LOOPS
            return
    min_k = min(range(4), key=lambda k: int(ttls[k]))
    slots[min_k] = attacker
    ttls[min_k] = THREAT_TTL_LOOPS


def _is_assist_responder_eligible(sim: Sim, b: int, h: int) -> bool:
    """Return True if ally `h` is in an auto-acquire-eligible order (idle, attack-move, or
    near-leash return).
    """
    cfg, w, c = sim.config, sim.world, sim.combat
    kind = int(w.order_kind[b, h])
    if kind == CommandKind.NOOP:
        return cfg.auto_acquire
    if kind == CommandKind.ATTACK_MOVE:
        return True
    if kind == CommandKind.MOVE and bool(c.leash_return[b, h]) and cfg.auto_acquire:
        hx_q, hy_q = unit_xy_q12(sim, b, h)
        ax_q = q12_scalar(float(w.order_x[b, h]))
        ay_q = q12_scalar(float(w.order_y[b, h]))
        return d2_raw(hx_q, hy_q, ax_q, ay_q) <= LEASH_RESET_RADIUS_SQ
    return False


def _set_transient_engaged(
    sim: Sim, b: int, u: int, target: int, *, record_assist: bool = False
) -> None:
    """Assign a transient engagement target and record leash coordinates if starting from
    idle/return.
    """
    w, c = sim.world, sim.combat
    kind = int(w.order_kind[b, u])
    if kind == CommandKind.MOVE and bool(c.leash_return[b, u]):
        if np.isnan(c.leash_x[b, u]):
            c.leash_x[b, u] = w.order_x[b, u]
            c.leash_y[b, u] = w.order_y[b, u]
        w.order_kind[b, u] = CommandKind.NOOP
        w.order_x[b, u] = w.order_y[b, u] = np.float32(np.nan)
        c.leash_return[b, u] = False
    elif kind == CommandKind.NOOP and sim.config.auto_acquire and np.isnan(c.leash_x[b, u]):
        c.leash_x[b, u] = w.x[b, u]
        c.leash_y[b, u] = w.y[b, u]
    w.engaged_target_slot[b, u] = target
    c.engaged_transient[b, u] = True
    c.scan_loop[b, u] = -1
    c.defer_abort[b, u] = False
    if record_assist:
        c.assist_target[b, u] = target
        c.assist_loop[b, u] = int(w.game_loop[b])
    else:
        c.assist_target[b, u] = NO_SLOT
        c.assist_loop[b, u] = -1


def assist_on_hit_q12(sim: Sim, b: int, attacker: int, victim: int, lethal: bool) -> None:
    """Broadcast assist-on-hit alert to nearby eligible allies using Q12 edge distance and
    target scoring.
    """
    cfg, w, c = sim.config, sim.world, sim.combat
    owner = int(w.owner[b, victim])
    if cfg.auto_acquire_owners is not None and owner not in cfg.auto_acquire_owners:
        return
    if not (w.alive[b, attacker] and w.health[b, attacker] > 0):
        return
    if not is_seen_by(sim, b, victim, attacker):
        return
    # Non-lethal hits obey 2-slot 32-loop deduplication; lethal hits always broadcast.
    passed = assist_throttle_record(sim, b, victim, attacker)
    if not lethal and not passed:
        return

    vx_q, vy_q = unit_xy_q12(sim, b, victim)
    rv_q = body_radius_q12(sim, b, victim)
    assist_q = q12_scalar(float(cfg.hit_assist_range))

    nearby: list[tuple[int, int]] = []
    for h in range(sim.n_slots):
        if h == victim or not w.alive[b, h] or w.health[b, h] <= 0:
            continue
        if int(w.owner[b, h]) != owner:
            continue
        hx_q, hy_q = unit_xy_q12(sim, b, h)
        rh_q = body_radius_q12(sim, b, h)
        rad_q = rv_q + assist_q + rh_q
        d2 = d2_raw(vx_q, vy_q, hx_q, hy_q)
        if d2 <= rad_q * rad_q:
            nearby.append((d2, h))
    nearby.sort()

    for _, h in nearby:
        if weapon_for(sim, b, h, attacker) < 0:
            continue
        if not _is_assist_responder_eligible(sim, b, h):
            continue
        add_threat_memory(sim, b, h, attacker)
        cur = int(w.engaged_target_slot[b, h])
        if cur == attacker:
            continue
        busy = weapon_busy(sim, b, h)
        if cur == NO_SLOT:
            if busy:
                store_defer_target(sim, b, h, attacker)
            else:
                _set_transient_engaged(sim, b, h, attacker, record_assist=True)
            continue
        if not w.alive[b, cur] or w.health[b, cur] <= 0:
            store_defer_target(sim, b, h, attacker)
            continue
        if compare_key(sim, b, h, attacker, cur) < compare_key(sim, b, h, cur, cur):
            if busy:
                store_defer_target(sim, b, h, attacker)
            else:
                same_loop_assist = int(c.assist_loop[b, h]) == int(w.game_loop[b]) and cur == int(
                    c.assist_target[b, h]
                )
                _set_transient_engaged(sim, b, h, attacker, record_assist=same_loop_assist)
                c.defer_target[b, h] = NO_SLOT


def phase_offset(sim: Sim, b: int, u: int) -> int:
    """Return current game loop offset relative to unit `u`'s scan phase."""
    cfg = sim.config
    start = 0 if cfg.native_scan_cadence else cfg.acquire_start_loop
    return int(sim.world.game_loop[b]) - int(sim.combat.scan_shift[b, u]) - start


def _valid_threat_candidates(sim: Sim, b: int, u: int) -> list[int]:
    """Collect valid living enemy slots currently stored in unit `u`'s 4-slot threat memory."""
    if not sim.config.native_assist_on_hit:
        return []
    w, c = sim.world, sim.combat
    out: list[int] = []
    for k in range(4):
        t = int(c.threat_slot[b, u, k])
        ttl = int(c.threat_ttl[b, u, k])
        if t == NO_SLOT or ttl <= 0 or t == u:
            continue
        if not w.alive[b, t] or w.health[b, t] <= 0 or not is_enemy(w, b, u, t):
            continue
        if weapon_for(sim, b, u, t) < 0:
            continue
        if not is_seen_by(sim, b, u, t):
            continue
        if t not in out:
            out.append(t)
    return out


def pick_best_q12(
    sim: Sim,
    b: int,
    u: int,
    *,
    cur: int = NO_SLOT,
    hold_only: bool = False,
    allow_spatial: bool = True,
    ignore_lock: bool = False,
) -> int:
    """Select the highest-priority target from the Q12 candidate pool, or return `NO_SLOT`.

    * `hold_only=True`: restrict candidates to enemies within strict weapon range (HOLD).
    * `allow_spatial=False`: skip spatial scan (odd-beat 16-loop throttle), considering only
      `cur` and `threat_slot`.
    * Includes active 4-slot threat memory when `native_assist_on_hit=True`.
    """
    cfg, w, c = sim.config, sim.world, sim.combat
    lock_mode = cfg.retarget_lock
    locked_tgt = int(c.rt_lock[b, u])
    if not ignore_lock and lock_mode == 'new' and cur != NO_SLOT and cur == locked_tgt:
        return cur

    cands: list[tuple[int, int]] = []  # (slot, wi)
    seen_slots: set[int] = set()

    def _add(t: int) -> None:
        if t in seen_slots or t == u or t < 0 or t >= sim.n_slots:
            return
        if not w.alive[b, t] or w.health[b, t] <= 0 or not is_enemy(w, b, u, t):
            return
        if not ignore_lock and lock_mode == 'previous' and t == locked_tgt and t != cur:
            return
        wi = weapon_for(sim, b, u, t)
        if wi < 0:
            return
        seen_slots.add(t)
        cands.append((t, wi))

    if cur != NO_SLOT and w.alive[b, cur] and w.health[b, cur] > 0:
        if is_seen_by(sim, b, u, cur):
            if not hold_only or is_strict_range_q12(sim, b, u, cur):
                _add(cur)
        elif (
            not hold_only
            and cfg.lost_vision == 'sc2'
            and int(w.order_kind[b, u]) == CommandKind.NOOP
        ):
            _add(cur)

    if not hold_only:
        for t in _valid_threat_candidates(sim, b, u):
            _add(t)

    if allow_spatial:
        for t in range(sim.n_slots):
            if t in seen_slots or t == u:
                continue
            if not w.alive[b, t] or w.health[b, t] <= 0 or not is_enemy(w, b, u, t):
                continue
            if not is_seen_by(sim, b, u, t):
                continue
            wi = weapon_for(sim, b, u, t)
            if wi < 0:
                continue
            if hold_only:
                if not is_strict_range_q12(sim, b, u, t, wi):
                    continue
            elif not is_in_acquire_q12(sim, b, u, t, wi):
                continue
            _add(t)

    if not cands:
        return NO_SLOT
    best_t, _ = min(cands, key=lambda tw: compare_key(sim, b, u, tw[0], cur, tw[1]))
    return best_t


def should_skip_spatial_16loop(
    sim: Sim, b: int, u: int, cur: int, *, forced: bool, periodic: bool
) -> bool:
    """16-loop spatial scan throttle: skip spatial scan on odd 8-loop beats when `cur` is in
    strict range/arc.
    """
    if not sim.config.native_retarget or forced or not periodic:
        return False
    if cur == NO_SLOT:
        return False
    w = sim.world
    if not w.alive[b, cur] or w.health[b, cur] <= 0:
        return False
    if not is_enemy(w, b, u, cur) or not is_seen_by(sim, b, u, cur):
        return False
    if not is_strict_range_q12(sim, b, u, cur) or not _facing_ok_for_sticky(sim, b, u, cur):
        return False
    return (phase_offset(sim, b, u) & 0xF) != 0


def pop_dead_order(sim: Sim, b: int, u: int) -> tuple[bool, bool]:
    """Clear a dead target's order/engagement state and return `(was_amove, had_attack_unit)`."""
    from . import abilities

    cfg, w, c = sim.config, sim.world, sim.combat
    kind = int(w.order_kind[b, u])
    was_amove = kind == CommandKind.ATTACK_MOVE
    was_au = kind == CommandKind.ATTACK_UNIT
    if was_au:
        hold = bool(sim.ability.hold_after[b, u])
        if cfg.kill_no_retreat:
            c.kill_idle[b, u] = not hold
        w.order_kind[b, u] = CommandKind.HOLD if hold else CommandKind.NOOP
        w.order_ability[b, u] = 0
        w.order_target_slot[b, u] = NO_SLOT
        abilities.reset_state(sim, b, u)
    w.engaged_target_slot[b, u] = NO_SLOT
    c.engaged_transient[b, u] = False
    c.defer_abort[b, u] = False
    c.follow_tol[b, u] = False
    c.tol_target[b, u] = NO_SLOT
    c.rt_due[b, u] = -1
    c.rt_lock[b, u] = NO_SLOT
    c.assist_target[b, u] = NO_SLOT
    c.assist_loop[b, u] = -1
    c.scan_loop[b, u] = -1
    return was_amove, was_au


def reconcile_pop_rescan(sim: Sim, b: int, u: int, *, allow_scan: bool) -> int:
    """Perform an immediate rescan upon order pop and reconcile with any `defer_target`."""
    w, c = sim.world, sim.combat
    rem = int(c.defer_target[b, u])
    c.defer_target[b, u] = NO_SLOT
    c.defer_abort[b, u] = False
    if rem != NO_SLOT:
        if (
            not w.alive[b, rem]
            or w.health[b, rem] <= 0
            or not is_enemy(w, b, u, rem)
            or weapon_for(sim, b, u, rem) < 0
        ):
            rem = NO_SLOT

    fresh = NO_SLOT
    if allow_scan:
        if (
            sim.config.native_retarget
            or sim.config.native_assist_on_hit
            or sim.config.native_backswing
        ):
            fresh = pick_best_q12(sim, b, u, cur=NO_SLOT, hold_only=False, allow_spatial=True)

    chosen = NO_SLOT
    if fresh != NO_SLOT and rem != NO_SLOT:
        ux_q, uy_q = unit_xy_q12(sim, b, u)
        fx_q, fy_q = unit_xy_q12(sim, b, fresh)
        rx_q, ry_q = unit_xy_q12(sim, b, rem)
        f_pr = (-base_score(sim, b, fresh), d2q(ux_q, uy_q, fx_q, fy_q))
        r_pr = (-base_score(sim, b, rem), d2q(ux_q, uy_q, rx_q, ry_q))
        chosen = fresh if f_pr <= r_pr else rem
    elif fresh != NO_SLOT:
        chosen = fresh
    elif rem != NO_SLOT:
        chosen = rem

    if chosen != NO_SLOT:
        _set_transient_engaged(sim, b, u, chosen)
    else:
        c.assist_target[b, u] = NO_SLOT
        c.assist_loop[b, u] = -1
    return chosen


def shuffle_unit_order(sim: Sim) -> None:
    """End-of-loop unit-order update: promote each living unit to front with probability 0.5 via
    `'update_order'` RNG.
    """
    w, c = sim.world, sim.combat
    for b in range(sim.batch_size):
        seed, loop = int(sim.seed[b]), int(w.game_loop[b])
        prev = [int(u) for u in c.unit_order[b].tolist() if 0 <= int(u) < sim.n_slots]
        if len(prev) < sim.n_slots:
            seen = set(prev)
            prev.extend(u for u in range(sim.n_slots) if u not in seen)
        alive_order = [u for u in prev if w.alive[b, u]]
        dead_slots = [u for u in prev if not w.alive[b, u]]
        head_seq = list(alive_order)
        for u in alive_order:
            if rng.uniform(seed, 'update_order', loop, u) < 0.5:
                head_seq.remove(u)
                head_seq.insert(0, u)
        new_order = head_seq + dead_slots
        c.unit_order[b, : sim.n_slots] = np.asarray(new_order, dtype=np.int32)


def order_before(sim: Sim, b: int, k: int, u: int) -> bool:
    """Return True if unit `k` precedes unit `u` in `combat.unit_order[b]`."""
    order = sim.combat.unit_order[b].tolist()
    pk = pu = -1
    for idx, slot in enumerate(order):
        s = int(slot)
        if s == k:
            pk = idx
        if s == u:
            pu = idx
    if pk >= 0 and pu >= 0:
        return pk < pu
    return k < u


def apply_leash_return(sim: Sim, b: int, u: int) -> None:
    """Issue a return-to-leash MOVE order for an idle unit and mark `leash_return=True`."""
    from .commands import _apply

    c = sim.combat
    lx, ly = c.leash_x[b, u], c.leash_y[b, u]
    _apply(sim, b, u, int(CommandKind.MOVE), NO_SLOT, lx, ly)
    c.leash_return[b, u] = True


def apply_retreat_flee(sim: Sim, b: int, victim: int, attacker: int) -> None:
    """Compute scalar retreat target away from attacker for SIMD-independent bitwise parity."""
    from .commands import _apply

    w, cfg = sim.world, sim.config
    dx = float(w.x[b, victim]) - float(w.x[b, attacker])
    dy = float(w.y[b, victim]) - float(w.y[b, attacker])
    n = math.hypot(dx, dy)
    if n > 0.0:
        d = float(cfg.retreat_distance) / n
        tx = f32(float(w.x[b, victim]) + dx * d)
        ty = f32(float(w.y[b, victim]) + dy * d)
        _apply(sim, b, victim, int(CommandKind.MOVE), NO_SLOT, tx, ty)
