"""Stage 3 attack timing: advances weapon cooldowns and writes `buf.fire_target`."""

from __future__ import annotations

import math

import numpy as np

from unismac.contract import NO_SLOT

from .. import rng
from ..sim import WEAPON_AIR, WEAPON_GROUND, Sim, edge_distance, f32, weapon_for


def instant(sim: Sim) -> None:
    """Fire immediately once cooldown hits 0 in range; damage resolves in the same loop
    (no windup, no projectile travel).

    Cooldown decrements by 1 each loop (clamped at 0); firing adds weapon period (in loops).
    Sub-loop remainders are discarded, so shots occur every `ceil(cooldown * 16)` loops
    (e.g., Marine: `ceil(13.77) = 14`). Dead units do not tick cooldown.
    """
    w, p, buf = sim.world, sim.params, sim.buf
    buf.fire_target.fill(NO_SLOT)
    alive = w.alive
    cd = w.weapon_cooldown
    # max(cd - 1, 0): matches Python max() tie/NaN behavior (returns first arg)
    dec = cd - f32(1)
    cd[alive] = np.where(f32(0) > dec, f32(0), dec)[alive]
    t = w.engaged_target_slot
    fire = alive & (t != NO_SLOT) & buf.in_range & ~(cd > 0)
    if not fire.any():
        return
    b, u = np.nonzero(fire)
    tt = t[b, u].astype(np.int64)
    wi = np.where(w.is_flying[b, tt], WEAPON_AIR, WEAPON_GROUND)
    wi = np.where(p.has_weapon[b, u, wi], wi, -1)
    buf.fire_target[b, u] = tt
    sim.combat.last_fire_loop[b, u] = w.game_loop[b]
    cd[b, u] += p.weapon_period[b, u, wi]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _face(sim: Sim, b: int, u: int, t: int) -> bool:
    """Aim an in-range attacker toward `t` (stationary units rotate `facing` in kinematics)
    and return whether the residual angle after this loop's turn is `<= attack_arc`.

    Units with `turret_rate` (e.g., Hellion, Phoenix) keep body heading/facing fixed and
    instead turn `combat.turret` (initialized to spawn facing) by at most `turret_rate`.
    """
    w, p = sim.world, sim.params
    angle = math.atan2(float(w.y[b, t] - w.y[b, u]), float(w.x[b, t] - w.x[b, u]))
    rate = float(p.turret_rate[b, u])
    if not math.isnan(rate):
        c = sim.combat
        diff = _wrap(angle - float(c.turret[b, u]))
        c.turret[b, u] = f32(_wrap(float(c.turret[b, u]) + max(-rate, min(rate, diff))))
        return abs(diff) - rate <= float(p.attack_arc[b, u])
    sim.motion.heading[b, u] = f32(angle)
    arc = float(p.attack_arc[b, u])
    if math.isinf(arc):
        return True
    left = abs(_wrap(angle - float(w.facing[b, u]))) - float(p.facing_rate[b, u])
    return left <= arc


def _face_many(sim: Sim, b: np.ndarray, u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Batched `_face` over `(b, u, t)` returning a bool ready-to-fire mask. Each unit writes
    only its own heading/turret, so order does not matter. Uses scalar `math.atan2` (differs
    from `np.arctan2` by 1 ulp on ~7% of inputs) and float64 mod/min/max for bit-parity.
    """
    w, p = sim.world, sim.params
    if not b.size:
        return np.zeros(0, np.bool_)
    f64 = np.float64
    dy = (w.y[b, t] - w.y[b, u]).astype(f64).tolist()
    dx = (w.x[b, t] - w.x[b, u]).astype(f64).tolist()
    angle = np.array([math.atan2(y, x) for y, x in zip(dy, dx, strict=True)], f64)
    two_pi = 2.0 * math.pi

    def wrap(a: np.ndarray) -> np.ndarray:
        return np.mod(a + math.pi, two_pi) - math.pi

    out = np.zeros(b.size, np.bool_)
    rate = p.turret_rate[b, u].astype(f64)
    arc = p.attack_arc[b, u].astype(f64)
    tur = ~np.isnan(rate)
    if tur.any():
        tb, tu, r = b[tur], u[tur], rate[tur]
        cur = sim.combat.turret[tb, tu].astype(f64)
        diff = wrap(angle[tur] - cur)
        m = np.where(diff < r, diff, r)  # min(rate, diff)
        cl = np.where(m > -r, m, -r)  # max(-rate, ...)
        sim.combat.turret[tb, tu] = wrap(cur + cl).astype(np.float32)
        out[tur] = np.abs(diff) - r <= arc[tur]
    rest = ~tur
    if rest.any():
        rb, ru, ang = b[rest], u[rest], angle[rest]
        sim.motion.heading[rb, ru] = ang.astype(np.float32)
        a = arc[rest]
        left = np.abs(wrap(ang - w.facing[rb, ru].astype(f64))) - p.facing_rate[rb, ru].astype(f64)
        out[rest] = np.isinf(a) | (left <= a)
    return out


def sc2(sim: Sim) -> None:
    """SC2 combat attack timing (parameters loaded from `assets/measured.json`).

    Per game loop:

    1. Decrement internal `combat.cooldown` by 1 when `> 0` (continuous; may go negative).
    2. Units in windup: cancel windup (without refunding cooldown) if engaged target changed or
       left range; otherwise decrement `windup_left` by 1.
    3. Units not in windup with engaged target in range: aim toward target (kinematics turns
       `facing` at `facing_rate`). If `cooldown <= 0` and residual angle `<= attack_arc`, start
       attack: add `cooldown * 16 + U(cooldown_jitter_min, cooldown_jitter_max)` (preserving
       negative remainder; jitter drawn from counter-based RNG `'cooldown'` stream keyed by
       `(seed, game_loop, slot)`), and set `windup_left = attack_windup`.
    4. Units with `windup_left == 0` fire this loop (write `buf.fire_target`; damage resolves
       in the damage subsystem).
    5. Clamp negative cooldown to 0 if no attack started (negative remainder only carries over
       when starting an attack in the same loop).

    Special rules: targets closer than `min_range` (e.g., Siege Tank = 2) cannot start windup
    and cancel active windup (pathing still treats them as in range so the unit stays put and
    keeps its order); turret turning is handled by `_face`; suicide units attacking structures
    treat edge distance `> suicide_range_structure` as out of range (keep closing in).

    `follow_lock`: "in range" above also includes range slop (`buf.fire_slop`; see
    `targeting._follow_lock`), and firing records `combat.last_fire_target` (also recorded when
    `chase_lock_target` or `pin_after_fire` is enabled).

    `attack_commit`: step 2 cancels windup only if the target changed or died, refunding the
    cooldown added at windup start (`combat.windup_cost`). Kept windups whose target moved out
    of range set `buf.windup_hold` (pathing holds position). Prior `last_fire_loop` and
    `last_fire_target` are saved into `buf.fire_prev_*` so `damage.sc2` can restore them if a
    shot is aborted. `attack_on_arrival`: when starting an attack with cooldown already `<= 0`
    at the end of the previous loop, moving last loop (`motion.speed > 0`), same engaged target
    (`combat.arrive_target`), and `attack_windup >= 1`, both windup and cooldown are reduced by
    1 (allowing `windup == 0` to fire on the arrival loop).

    Observed `weapon_cooldown` reads 0 during windup and `max(combat.cooldown, 0)` otherwise.
    Continuous fire shows `prev - 1 + period + j - windup`, while the first shot from idle
    shows `period + j - windup` with `j ~ U(-1, 2)`, making mean shot interval roughly
    `cooldown * 16 + 0.5` loops.

    With default measured parameters (no windup, facing check, or jitter), the first shot
    matches `instant`, and subsequent intervals alternate `floor`/`ceil(cooldown * 16)` because
    fractional remainders carry over.
    """
    w, p, buf, c = sim.world, sim.params, sim.buf, sim.combat
    # Units only read/write their own state; vectorized except per-unit attack start (RNG).
    buf.fire_target.fill(NO_SLOT)
    alive = w.alive
    commit, arrive = sim.config.attack_commit, sim.config.attack_on_arrival
    ready = c.cooldown <= 0  # cooldown already 0 at previous loop end (for attack_on_arrival)
    dec = alive & (c.cooldown > 0)
    c.cooldown[dec] = c.cooldown[dec] - f32(1)
    t = w.engaged_target_slot
    ok = (t != NO_SLOT) & buf.in_range
    follow = sim.config.follow_lock or sim.config.slop_on_arrival
    if follow:  # range slop also counts as in-range (targeting._follow_lock)
        ok |= (t != NO_SLOT) & buf.fire_slop
    if sim.config.native_backswing:
        has_t = t != NO_SLOT
        tgt_alive = np.zeros_like(alive)
        if has_t.any():
            tb, tu = np.nonzero(has_t)
            tgt_alive[tb, tu] = w.alive[tb, t[tb, tu]]
        ok &= tgt_alive
    else:
        tgt_alive = t != NO_SLOT
    # Suicide unit vs structure: edge > suicide_range_structure is out of range (keep closing).
    for b, u in zip(*np.nonzero(alive & ok & p.suicide), strict=True):
        b, u = int(b), int(u)
        tt = int(t[b, u])
        if p.structure[b, tt] and edge_distance(w, b, u, tt) > p.suicide_range_structure[b, u]:
            ok[b, u] = False
            buf.in_range[b, u] = False
    near = alive & ok & (p.min_range > 0)
    for b, u in zip(*np.nonzero(near), strict=True):
        b, u = int(b), int(u)
        if edge_distance(w, b, u, int(t[b, u])) < p.min_range[b, u]:
            ok[b, u] = False
    winding = alive & (c.windup_left >= 0)
    if commit:  # cancel windup only when target changed or died, refunding cooldown
        keep = winding & (t != NO_SLOT) & (t == c.windup_target)
        if sim.config.native_backswing:
            keep &= tgt_alive
        buf.windup_hold[:] = keep & ~ok
    else:
        keep = winding & ok & (t == c.windup_target)
    c.windup_left[keep] -= 1
    drop = winding & ~keep
    if commit:
        c.cooldown[drop] -= c.windup_cost[drop]
    c.windup_left[drop], c.windup_target[drop] = -1, NO_SLOT
    cb, cu = np.nonzero(alive & ~winding & ok)
    faced = _face_many(sim, cb, cu, t[cb, cu].astype(np.int64))
    go = faced & (c.cooldown[cb, cu] <= 0)
    # Process attack starts per unit (keyed counter RNG is order-independent).
    for b, u in zip(cb[go].tolist(), cu[go].tolist(), strict=True):
        tt = int(t[b, u])
        loop, seed = int(w.game_loop[b]), int(sim.seed[b])
        lo, hi = float(p.cooldown_jitter_min[b, u]), float(p.cooldown_jitter_max[b, u])
        jitter = lo + (hi - lo) * rng.uniform(seed, 'cooldown', loop, u) if hi > lo else lo
        period = float(p.weapon_period[b, u, weapon_for(sim, b, u, tt)])
        before = float(c.cooldown[b, u])
        cd, windup = before + period + jitter, int(p.attack_windup[b, u])
        chased = arrive and c.arrive_target[b, u] == tt and sim.motion.speed[b, u] > 0
        if chased and windup > 0 and ready[b, u]:
            cd, windup = cd - 1.0, windup - 1  # equivalent to starting attack on previous loop
        c.cooldown[b, u] = f32(cd)
        if commit:
            c.windup_cost[b, u] = f32(float(c.cooldown[b, u]) - before)
        c.windup_left[b, u], c.windup_target[b, u] = windup, tt
    fire = alive & (c.windup_left == 0)
    if commit:
        buf.fire_prev_loop[:] = c.last_fire_loop
        buf.fire_prev_target[:] = c.last_fire_target
    if sim.config.native_backswing:
        buf.fire_prev_backswing[:] = c.backswing_left
        if fire.any():
            bs = np.where(p.follow_lock > 0, p.follow_lock - 1, 0)
            c.backswing_left[fire] = bs[fire].astype(np.int32)
    buf.fire_target[fire] = c.windup_target[fire]
    c.last_fire_loop[fire] = np.broadcast_to(w.game_loop[:, None], fire.shape)[fire]
    if follow or sim.config.chase_lock_target or sim.config.pin_after_fire:
        c.last_fire_target[fire] = c.windup_target[fire]
    c.windup_left[fire], c.windup_target[fire] = -1, NO_SLOT
    c.cooldown[alive & (c.cooldown < 0)] = 0
    if arrive:
        c.arrive_target[:] = t
    shown = np.where(c.windup_left > 0, f32(0), c.cooldown)
    w.weapon_cooldown[alive] = shown[alive]


IMPLS = {'instant': instant, 'sc2': sc2}
