"""Stage 8 damage resolution and cleanup: resolves fired attacks and processes unit deaths."""

from __future__ import annotations

import math

import numpy as np

from unismac.contract import NO_SLOT, CommandKind

from ..sim import WEAPON_AIR, WEAPON_GROUND, Sim, cell_index, f32, height_z, weapon_for
from . import native_combat, targeting, visibility
from .abilities import reset_state


def apply_hit(
    shield: np.float32,
    health: np.float32,
    damage: np.float32 | float,
    armor: np.float32 | float,
    min_damage: np.float32 | float,
) -> tuple[np.float32, np.float32]:
    """Resolve one weapon strike: subtract from shield first, then apply armor to any remaining
    health damage (floored at `min_damage`; shield armor is treated as 0). Health may drop below
    zero and is zeroed during death cleanup.
    """
    shield, health, damage = f32(shield), f32(health), f32(damage)
    armor, min_damage = f32(armor), f32(min_damage)
    absorbed = min(shield, damage)
    shield = f32(shield - absorbed)
    rest = f32(damage - absorbed)
    if rest > 0:
        health = f32(health - max(rest - armor, min_damage))
    return shield, health


def standard(sim: Sim) -> None:
    """Resolve fired shots in slot order, then clean up dead units.

    Shots in the same loop were decided from pre-damage state, allowing mutual kills; slot order
    only sets accumulation order on shared targets. Attribute bonuses add to base damage before
    applying each multi-attack hit.
    """
    p = sim.params
    min_damage = f32(sim.config.min_damage)
    for b, u in zip(*np.nonzero(sim.buf.fire_target != NO_SLOT), strict=True):  # row-major order
        b, u = int(b), int(u)
        t = int(sim.buf.fire_target[b, u])
        wi = weapon_for(sim, b, u, t)
        _hit(sim, b, u, t, wi, int(p.weapon_attacks[b, u, wi]), min_damage)
        visibility.on_hit(sim, b, u, t)
        targeting.on_hit(sim, b, u, t)
    _cleanup_all(sim)


def _cleanup_all(sim: Sim) -> None:
    """Run `_cleanup` on environments with at least one newly dying unit."""
    w = sim.world
    dying = (w.alive & (w.health <= 0)).any(axis=1)
    for b in np.flatnonzero(dying).tolist():
        _cleanup(sim, b)


def _weapon_damage(sim: Sim, b: int, u: int, t: int, wi: int) -> np.float32:
    """Single-strike damage of unit `u`'s weapon `wi` against `t`, including attribute bonuses."""
    p = sim.params
    damage = p.weapon_damage[b, u, wi]
    for a in range(p.weapon_bonus.shape[3]):
        if p.attr_mask[b, t] >> a & 1:
            damage = f32(damage + p.weapon_bonus[b, u, wi, a])
    return damage


def _hit(sim: Sim, b: int, u: int, t: int, wi: int, n: int, min_damage: np.float32) -> None:
    """Apply `n` hits of unit `u`'s weapon `wi` onto target `t`."""
    w, p = sim.world, sim.params
    damage = _weapon_damage(sim, b, u, t, wi)
    shield, health = w.shield[b, t], w.health[b, t]
    for _ in range(n):
        shield, health = apply_hit(shield, health, damage, p.armor[b, t], min_damage)
    w.shield[b, t], w.health[b, t] = shield, health


def _barrier_hit(
    sim: Sim, b: int, t: int, damage: np.float32, armor: np.float32, n: int, loop: int
) -> None:
    """Resolve `n` hits on a Barrier unit (e.g., Immortal): the triggering hit deals full damage
    and activates Barrier through `loop + barrier_duration` (inclusive); subsequent hits absorb
    from `barrier_left` before shield/armor until depleted or expired. Cooldown `barrier_cooldown`
    counts from the trigger loop (0 means once per life).
    """
    w, p, c = sim.world, sim.params, sim.combat
    min_damage = f32(sim.config.min_damage)
    shield, health = w.shield[b, t], w.health[b, t]
    for _ in range(n):
        d = f32(damage)
        if c.barrier_until[b, t] >= loop and c.barrier_left[b, t] > 0:
            absorbed = min(c.barrier_left[b, t], d)
            c.barrier_left[b, t] = f32(c.barrier_left[b, t] - absorbed)
            d = f32(d - absorbed)
        if d > 0:
            shield, health = apply_hit(shield, health, d, armor, min_damage)
            last, cd = int(c.barrier_from[b, t]), int(p.barrier_cooldown[b, t])
            if last < 0 or (cd > 0 and loop - last >= cd):
                c.barrier_from[b, t] = loop
                c.barrier_left[b, t] = p.barrier_absorb[b, t]
                c.barrier_until[b, t] = loop + int(p.barrier_duration[b, t])
    w.shield[b, t], w.health[b, t] = shield, health


def _strike(
    sim: Sim,
    b: int,
    t: int,
    damage: np.float32,
    armor: np.float32,
    n: int,
    loop: int,
) -> None:
    """Apply `n` hits of `damage` (including bonuses/splash) in `damage='sc2'`, honoring Barrier."""
    if sim.params.barrier_absorb[b, t] > 0:
        _barrier_hit(sim, b, t, damage, armor, n, loop)
        return
    w = sim.world
    shield, health = w.shield[b, t], w.health[b, t]
    min_damage = f32(sim.config.min_damage)
    for _ in range(n):
        shield, health = apply_hit(shield, health, damage, armor, min_damage)
    w.shield[b, t], w.health[b, t] = shield, health


# `damage='sc2'`: hit slow starts `SLOW_ONSET` loops after impact and lasts `hit_slow_loops`.
SLOW_ONSET = 2


def _sc2_hit(
    sim: Sim, b: int, u: int, t: int, wi: int, n: int, loop: int, proj: bool = False
) -> None:
    """Resolve one SC2 strike: deal damage, record hit loop, and apply/refresh hit slow.
    `proj` marks projectile impact (used when recording killer under `kill_rescan_order`).
    """
    if not sim.world.alive[b, t] or sim.world.health[b, t] <= 0:
        return
    if sim.params.barrier_absorb[b, t] > 0:
        dmg = _weapon_damage(sim, b, u, t, wi)
        _barrier_hit(sim, b, t, dmg, sim.params.armor[b, t], n, loop)
    else:
        _hit(sim, b, u, t, wi, n, f32(sim.config.min_damage))
    if sim.config.kill_rescan_order and sim.world.health[b, t] <= 0:
        sim.buf.kill_by[b, t], sim.buf.kill_proj[b, t] = u, proj
    if sim.combat.charge_bonus_target[b, u] != NO_SLOT:
        _charge_bonus(sim, b, u, t, loop)
    visibility.on_hit(sim, b, u, t)
    targeting.on_hit(sim, b, u, t)
    c, p = sim.combat, sim.params
    c.last_hit_loop[b, t] = loop
    loops = int(p.hit_slow_loops[b, u])
    if sim.config.hit_slow_owner1_only and sim.world.owner[b, u] != 1:
        loops = 0
    if loops > 0:
        if not c.slow_from[b, t] <= loop + 1 <= c.slow_to[b, t]:
            c.slow_from[b, t] = loop + SLOW_ONSET
        c.slow_to[b, t] = loop + SLOW_ONSET + loops - 1
        c.slow_factor[b, t] = p.hit_slow_factor[b, u]


def _charge_bonus(sim: Sim, b: int, u: int, t: int, loop: int) -> None:
    """Apply unarmored Charge impact bonus on the first strike against the charged stationary
    target; discarded if hitting another target.
    """
    c = sim.combat
    if int(c.charge_bonus_target[b, u]) == t and sim.world.health[b, t] > 0:
        _strike(sim, b, t, f32(sim.params.charge_bonus[b, u]), f32(0), 1, loop)
    c.charge_bonus_target[b, u] = NO_SLOT


def _regen(sim: Sim, b: int, loop: int) -> None:
    """Regenerate shields (after `shield_regen_delay` loops since last hit; immediately if never
    hit) and health (`health_regen_rate` every loop), both clamped to max.
    """
    w, p, c = sim.world, sim.params, sim.combat
    for u in range(sim.n_slots):
        if not w.alive[b, u]:
            continue
        rate = p.shield_regen_rate[b, u]
        last = int(c.last_hit_loop[b, u])
        if rate > 0 and w.shield[b, u] < w.shield_max[b, u]:
            if last < 0 or loop - last >= int(p.shield_regen_delay[b, u]):
                w.shield[b, u] = min(f32(w.shield[b, u] + rate), w.shield_max[b, u])
        rate = p.health_regen_rate[b, u]
        if rate > 0 and w.health[b, u] < w.health_max[b, u]:
            w.health[b, u] = min(f32(w.health[b, u] + rate), w.health_max[b, u])


def _regen_all(sim: Sim) -> None:
    """Batched `_regen` (each unit updates itself; matches Python `min` tie semantics)."""
    w, p, c = sim.world, sim.params, sim.combat
    alive = w.alive & (w.health > 0) if sim.config.native_loop_order else w.alive
    loop = w.game_loop[:, None]
    last = c.last_hit_loop.astype(np.int64)
    rate = p.shield_regen_rate
    sh = alive & (rate > 0) & (w.shield < w.shield_max)
    sh &= (last < 0) | (loop - last >= p.shield_regen_delay.astype(np.int64))
    if sh.any():
        v = w.shield[sh] + rate[sh]
        mx = w.shield_max[sh]
        w.shield[sh] = np.where(mx < v, mx, v)
    rate = p.health_regen_rate
    hp = alive & (rate > 0) & (w.health < w.health_max)
    if hp.any():
        v = w.health[hp] + rate[hp]
        mx = w.health_max[hp]
        w.health[hp] = np.where(mx < v, mx, v)


def _z(sim: Sim, b: int, u: int) -> float:
    """`projectile_3d`: world terrain elevation of unit `u`'s cell (via `sim.height_z`)."""
    info = sim.env_map[b]
    assert info is not None
    iy, ix = cell_index(info, sim.world.x[b, u : u + 1], sim.world.y[b, u : u + 1])
    return float(height_z(float(info.terrain_height[int(iy[0]), int(ix[0])])))


def _distance_to_edge(sim: Sim, b: int, u: int, t: int) -> tuple[float, float, float, float]:
    """Return `(dx, dy, dz, edge_dist)` from projectile to target center (`dz != 0` only under
    `projectile_3d`; target radius is 0 when ground weapon has `projectile_to_center`).
    """
    w, c = sim.world, sim.combat
    rt = float(w.radius[b, t])
    if int(c.proj_weapon[b, u]) != WEAPON_AIR and sim.params.projectile_to_center[b, u]:
        rt = 0.0
    dx, dy = float(w.x[b, t] - c.proj_x[b, u]), float(w.y[b, t] - c.proj_y[b, u])
    if not sim.config.projectile_3d:
        return dx, dy, 0.0, math.hypot(dx, dy) - rt
    dz = _z(sim, b, t) - float(c.proj_z[b, u])
    return dx, dy, dz, math.hypot(dx, dy, dz) - rt


def _fly(sim: Sim, b: int, u: int, arrive: float, loop: int) -> None:
    """Advance projectile toward target center; start `projectile_delay` countdown once within
    `arrive` of target edge, and resolve on countdown completion (3D when `projectile_3d` is on).
    """
    c = sim.combat
    t = int(c.proj_target[b, u])
    speed, delay, _ = _proj(sim, b, u, int(c.proj_weapon[b, u]))
    if c.proj_wait[b, u] < 0:
        dx, dy, dz, edge = _distance_to_edge(sim, b, u, t)
        if edge <= arrive + 1e-6:
            c.proj_wait[b, u] = delay
        else:
            if sim.config.projectile_3d:
                g = math.hypot(dx, dy, dz)
                c.proj_z[b, u] = f32(float(c.proj_z[b, u]) + dz / g * speed)
            else:
                g = math.hypot(dx, dy)
            c.proj_x[b, u] = f32(float(c.proj_x[b, u]) + dx / g * speed)
            c.proj_y[b, u] = f32(float(c.proj_y[b, u]) + dy / g * speed)
            return
    else:
        c.proj_wait[b, u] -= 1
    if c.proj_wait[b, u] <= 0:
        _land(sim, b, u, loop)


def _proj(sim: Sim, b: int, u: int, wi: int) -> tuple[float, int, float]:
    """Return `(speed, delay, offset)` for weapon `wi` (air reads `projectile_*_air`)."""
    p = sim.params
    if wi == WEAPON_AIR:
        return (
            float(p.projectile_speed_air[b, u]),
            int(p.projectile_delay_air[b, u]),
            float(p.projectile_offset_air[b, u]),
        )
    return (
        float(p.projectile_speed[b, u]),
        int(p.projectile_delay[b, u]),
        float(p.projectile_offset[b, u]),
    )


def _instant(sim: Sim, b: int, u: int, t: int, wi: int) -> bool:
    """Return True if ground weapon resolves instantaneously (`edge <= projectile_instant_edge`)."""
    lim = float(sim.params.projectile_instant_edge[b, u])
    if wi == WEAPON_AIR or math.isnan(lim):
        return False
    w = sim.world
    gap = math.hypot(float(w.x[b, t] - w.x[b, u]), float(w.y[b, t] - w.y[b, u]))
    return gap - float(w.radius[b, u]) - float(w.radius[b, t]) <= lim + 1e-6


def _land(sim: Sim, b: int, u: int, loop: int) -> None:
    c = sim.combat
    t = int(c.proj_target[b, u])
    wi, n = int(c.proj_weapon[b, u]), int(c.proj_hits[b, u])
    origin = (float(c.proj_x[b, u]), float(c.proj_y[b, u]))
    c.proj_target[b, u], c.proj_wait[b, u] = NO_SLOT, -1
    if _has_splash(sim, b, u) and sim.world.alive[b, t] and sim.world.health[b, t] > 0:
        _splash(sim, b, u, t, wi, n, loop, origin)
    spacing = int(sim.params.attack_spacing[b, u])
    if spacing > 0 and n > 1:
        # Multi-hit projectile (e.g., Queen Talons): first hit resolves on landing, remaining
        # hits follow at `attack_spacing` intervals with the same follow-chain cancellation rules.
        if c.follow_target[b, u] != NO_SLOT:
            while c.follow_target[b, u] != NO_SLOT:
                _follow(sim, b, u, loop)
        _sc2_hit(sim, b, u, t, wi, 1, loop, proj=True)
        c.follow_target[b, u], c.follow_weapon[b, u] = t, wi
        c.follow_left[b, u], c.follow_loop[b, u] = n - 1, loop + spacing
        return
    _sc2_hit(sim, b, u, t, wi, n, loop, proj=True)


# ---------------------------------------------------------------- Splash & suicide burst
# Splash evaluates at primary-target impact time using live positions, hitting alive same-domain
# units excluding attacker and primary target (respecting `splash_friendly`). Each victim takes
# `(single_damage_with_bonus * fraction)` resolved per hit against its own shield and armor.
# Colossus beams resolve all hits on lined-up victims on the primary impact loop.


def _has_splash(sim: Sim, b: int, u: int) -> bool:
    p = sim.params
    return bool(
        not math.isnan(float(p.splash_radius[b, u, 0]))
        or not math.isnan(float(p.splash_perp[b, u, 0]))
        or not math.isnan(float(p.splash_forward[b, u, 0]))
    )


def _ring(sim: Sim, b: int, u: int, dist: float) -> float:
    """Radial splash: return fraction for the smallest ring containing `dist`, or 0 if outside."""
    p = sim.params
    for k in range(p.splash_radius.shape[2]):
        r = float(p.splash_radius[b, u, k])
        if math.isnan(r):
            break
        if dist <= r + 1e-6:
            return float(p.splash_fraction[b, u, k])
    return 0.0


def _splash(
    sim: Sim,
    b: int,
    u: int,
    t: int,
    wi: int,
    n: int,
    loop: int,
    origin: tuple[float, float],
    center: tuple[float, float] | None = None,
) -> None:
    """Apply splash when `u`'s weapon `wi` hits `t`. `origin` defines the attack axis (attacker
    position if alive, else projectile position); `center` overrides radial splash center.
    """
    w, p = sim.world, sim.params
    tx, ty = float(w.x[b, t]), float(w.y[b, t])
    if w.alive[b, u]:
        origin = (float(w.x[b, u]), float(w.y[b, u]))
    cx, cy = center if center is not None else (tx, ty)
    ux, uy = tx - origin[0], ty - origin[1]
    norm = math.hypot(ux, uy)
    ux, uy = (ux / norm, uy / norm) if norm > 0 else (1.0, 0.0)
    perp, fwd = p.splash_perp[b, u], p.splash_forward[b, u]
    flying = bool(w.is_flying[b, t])
    for v in range(sim.n_slots):
        if v in (u, t) or not w.alive[b, v] or w.health[b, v] <= 0:
            continue
        if bool(w.is_flying[b, v]) != flying:
            continue
        own = w.owner[b, v] == w.owner[b, u]
        if own and not p.splash_friendly[b, u]:
            continue
        rv = float(w.radius[b, v])
        vx, vy = float(w.x[b, v]), float(w.y[b, v])
        hits, frac = 0, 0.0
        if not math.isnan(float(p.splash_radius[b, u, 0])):
            frac = _ring(sim, b, u, math.hypot(vx - cx, vy - cy) - rv)
            hits = n if frac > 0 else 0
        if not math.isnan(float(perp[0])):
            along = (vx - tx) * ux + (vy - ty) * uy
            side = -(vx - tx) * uy + (vy - ty) * ux
            if abs(side) - rv <= float(perp[0]) and abs(along) - rv <= float(perp[1]):
                frac, hits = 1.0, n
        if not math.isnan(float(fwd[0])):
            ax, ay = origin
            along = (vx - ax) * ux + (vy - ay) * uy
            side = -(vx - ax) * uy + (vy - ay) * ux
            ru = float(w.radius[b, u])
            if along > 0 and along - ru - rv <= float(fwd[0]) and abs(side) - rv <= float(fwd[1]):
                frac, hits = 1.0, n
        if hits <= 0 or frac <= 0:
            continue
        dmg = _weapon_damage(sim, b, u, v, wi)
        _strike(sim, b, v, f32(dmg * f32(frac)), p.armor[b, v], hits, loop)
        sim.combat.last_hit_loop[b, v] = loop
        if not own:
            visibility.on_hit(sim, b, u, v)
            targeting.on_hit(sim, b, u, v)


def _burst(sim: Sim, b: int, u: int, t: int, loop: int) -> None:
    """Suicide burst (Baneling): deal damage + bonus to primary target (unarmored
    `suicide_damage_structure` vs structures), apply radial splash, then kill the attacker.
    Detonation center advances one full-speed step toward the target (pathing already stopped
    the unit in range, so `p.speed` is used rather than `motion.speed`).
    """
    w, p = sim.world, sim.params
    wi = WEAPON_GROUND
    was_alive = bool(w.alive[b, t] and w.health[b, t] > 0)
    ux, uy = float(w.x[b, t] - w.x[b, u]), float(w.y[b, t] - w.y[b, u])
    norm = math.hypot(ux, uy)
    step = float(p.speed[b, u]) / norm if norm > 0 else 0.0
    center = (float(w.x[b, u]) + ux * step, float(w.y[b, u]) + uy * step)
    if p.structure[b, t]:
        _strike(sim, b, t, f32(p.suicide_damage_structure[b, u]), f32(0), 1, loop)
    else:
        _strike(sim, b, t, _weapon_damage(sim, b, u, t, wi), p.armor[b, t], 1, loop)
    sim.combat.last_hit_loop[b, t] = loop
    visibility.on_hit(sim, b, u, t)
    if was_alive or not sim.config.native_assist_on_hit:
        targeting.on_hit(sim, b, u, t)
    _splash(sim, b, u, t, wi, 1, loop, center, center=center)
    w.health[b, u] = 0


def _follow(sim: Sim, b: int, u: int, loop: int) -> None:
    """Resolve the next follow-up strike of a multi-attack weapon."""
    c, p = sim.combat, sim.params
    t, wi = int(c.follow_target[b, u]), int(c.follow_weapon[b, u])
    proj = (
        False
        if sim.config.native_loop_order
        else (sim.config.kill_rescan_order and not math.isnan(_proj(sim, b, u, wi)[0]))
    )
    _sc2_hit(sim, b, u, t, wi, 1, loop, proj=proj)
    c.follow_left[b, u] -= 1
    if c.follow_left[b, u] > 0:
        c.follow_loop[b, u] = loop + int(p.attack_spacing[b, u])
    else:
        c.follow_target[b, u] = NO_SLOT


def _fire(sim: Sim, b: int, u: int, t: int, loop: int) -> None:
    w, p, c = sim.world, sim.params, sim.combat
    if p.suicide[b, u]:
        _burst(sim, b, u, t, loop)
        return
    wi = weapon_for(sim, b, u, t)
    n = int(p.weapon_attacks[b, u, wi])
    speed, _, offset = _proj(sim, b, u, wi)
    if not math.isnan(speed) and _instant(sim, b, u, t, wi):
        speed = math.nan  # edge <= projectile_instant_edge: resolve instantaneously
    if not math.isnan(speed):
        if c.proj_target[b, u] != NO_SLOT:  # flush previous in-flight projectile immediately
            _land(sim, b, u, loop)
        dx, dy = float(w.x[b, t] - w.x[b, u]), float(w.y[b, t] - w.y[b, u])
        if sim.config.projectile_3d:
            z0 = _z(sim, b, u)
            dz = _z(sim, b, t) - z0
            g = math.hypot(dx, dy, dz)
            off = offset / g if g > 0 else 0.0
            c.proj_z[b, u] = f32(z0 + dz * off)
        else:
            g = math.hypot(dx, dy)
            off = offset / g if g > 0 else 0.0
        c.proj_x[b, u] = f32(float(w.x[b, u]) + dx * off)
        c.proj_y[b, u] = f32(float(w.y[b, u]) + dy * off)
        c.proj_target[b, u], c.proj_wait[b, u] = t, -1
        c.proj_hits[b, u], c.proj_weapon[b, u] = n, wi
        _fly(sim, b, u, 0.0, loop)
        return
    if _has_splash(sim, b, u):
        _splash(sim, b, u, t, wi, n, loop, (float(w.x[b, u]), float(w.y[b, u])))
    spacing = int(p.attack_spacing[b, u])
    if spacing <= 0 or n <= 1:
        _sc2_hit(sim, b, u, t, wi, n, loop)
        return
    if c.follow_target[b, u] != NO_SLOT:  # flush remaining strikes from prior volley
        while c.follow_target[b, u] != NO_SLOT:
            _follow(sim, b, u, loop)
    _sc2_hit(sim, b, u, t, wi, 1, loop)
    c.follow_target[b, u], c.follow_weapon[b, u] = t, wi
    c.follow_left[b, u], c.follow_loop[b, u] = n - 1, loop + spacing


def sc2(sim: Sim) -> None:
    """SC2 damage timing (parameters loaded from `assets/measured.json`).

    Per game loop:

    1. Regeneration (`_regen`): shields recover `shield_regen_rate` per loop after
       `shield_regen_delay` loops without hits (e.g., Zealot 160 / 0.125); health recovers
       `health_regen_rate` every loop (e.g., Zergling 70/4096).
    2. In-flight projectiles: track target center at `projectile_speed`, arrive once within
       `speed` of target edge, and resolve all hits after `projectile_delay` loops (discarded
       if target died; continues flying if attacker died).
    3. Due follow-up strikes (`attack_spacing`, e.g., Zealot second strike +4 loops); canceled
       if attacker or target died.
    4. Shots fired this loop: projectile weapons spawn at `projectile_offset` toward target
       (arriving immediately if already inside target radius); instant weapons resolve on the
       same loop (first strike immediately when `attack_spacing > 0`). Ground weapons also
       support `projectile_to_center` and `projectile_instant_edge`.
    5. Hit hooks: record `last_hit_loop` and apply/refresh hit slow (`hit_slow_loops`,
       `hit_slow_factor`, gated by `hit_slow_owner1_only`).
    6. Death cleanup (same as `standard`).

    Same-loop shots remain simultaneous, allowing mutual kills; default measured keys match
    `standard` damage. Under `attack_commit`, shots whose target already dropped to `<= 0` HP
    earlier in step 4 are aborted (`_void`). When `hit_assist_cap > 0`, `targeting.assist_flush`
    runs before death cleanup.
    """
    w, c = sim.world, sim.combat
    commit = sim.config.attack_commit
    if sim.config.kill_rescan_order:
        sim.buf.kill_by[:] = NO_SLOT
        sim.buf.kill_proj[:] = False
    if sim.config.native_loop_order:
        pre_proj = {
            b: np.flatnonzero(c.proj_target[b] != NO_SLOT).tolist() for b in range(sim.batch_size)
        }
        skip_p2: set[tuple[int, int]] = set()
        active_p1 = (c.follow_target != NO_SLOT) | (sim.buf.fire_target != NO_SLOT)
        for b in np.flatnonzero(active_p1.any(axis=1)).tolist():
            loop = int(w.game_loop[b])
            order = [int(u) for u in c.unit_order[b].tolist() if 0 <= int(u) < sim.n_slots]
            if len(order) < sim.n_slots:
                seen_u = set(order)
                order.extend(u for u in range(sim.n_slots) if u not in seen_u)
            for u in order:
                if not w.alive[b, u] or w.health[b, u] <= 0:
                    if c.follow_target[b, u] != NO_SLOT:
                        c.follow_target[b, u] = NO_SLOT
                    ft = int(sim.buf.fire_target[b, u])
                    if ft != NO_SLOT:
                        if commit:
                            _void(sim, b, u)
                        else:
                            sim.buf.fire_target[b, u] = NO_SLOT
                    continue
                ft_f = int(c.follow_target[b, u])
                if ft_f != NO_SLOT:
                    if not w.alive[b, ft_f] or w.health[b, ft_f] <= 0:
                        c.follow_target[b, u] = NO_SLOT
                    elif loop >= c.follow_loop[b, u]:
                        _follow(sim, b, u, loop)
                t = int(sim.buf.fire_target[b, u])
                if t != NO_SLOT:
                    if commit and (not w.alive[b, t] or w.health[b, t] <= 0):
                        _void(sim, b, u)
                    elif w.alive[b, t]:
                        if u in pre_proj[b] and not sim.params.suicide[b, u]:
                            wi_cur = weapon_for(sim, b, u, t)
                            if (
                                wi_cur >= 0
                                and not math.isnan(_proj(sim, b, u, wi_cur)[0])
                                and not _instant(sim, b, u, t, wi_cur)
                            ):
                                # _fire lands P_old and spawns P_new; skip advancing again in pass 2
                                skip_p2.add((b, u))
                        _fire(sim, b, u, t, loop)
        _regen_all(sim)
        for b, plist in pre_proj.items():
            if not plist:
                continue
            loop = int(w.game_loop[b])
            for u in plist:
                if (b, u) in skip_p2:
                    continue
                t = int(c.proj_target[b, u])
                if t == NO_SLOT:
                    continue
                if not w.alive[b, t] or w.health[b, t] <= 0:
                    c.proj_target[b, u], c.proj_wait[b, u] = NO_SLOT, -1
                    continue
                _fly(sim, b, u, _proj(sim, b, u, int(c.proj_weapon[b, u]))[0], loop)
    else:
        _regen_all(sim)
        # Projectiles, follow-ups, and shots mutate target health: process in slot order per env.
        busy = (c.proj_target != NO_SLOT) | (c.follow_target != NO_SLOT)
        busy |= sim.buf.fire_target != NO_SLOT
        for b in np.flatnonzero(busy.any(axis=1)).tolist():
            loop = int(w.game_loop[b])
            for u in np.flatnonzero(c.proj_target[b] != NO_SLOT).tolist():
                t = int(c.proj_target[b, u])
                if t == NO_SLOT:
                    continue
                if not w.alive[b, t]:
                    c.proj_target[b, u], c.proj_wait[b, u] = NO_SLOT, -1
                    continue
                _fly(sim, b, u, _proj(sim, b, u, int(c.proj_weapon[b, u]))[0], loop)
            for u in np.flatnonzero(c.follow_target[b] != NO_SLOT).tolist():
                t = int(c.follow_target[b, u])
                if t == NO_SLOT:
                    continue
                if not w.alive[b, u] or not w.alive[b, t]:
                    c.follow_target[b, u] = NO_SLOT
                elif loop >= c.follow_loop[b, u]:
                    _follow(sim, b, u, loop)
            for u in np.flatnonzero(sim.buf.fire_target[b] != NO_SLOT).tolist():
                t = int(sim.buf.fire_target[b, u])
                if t == NO_SLOT:
                    continue
                if commit and (not w.alive[b, t] or w.health[b, t] <= 0):
                    _void(sim, b, u)
                elif w.alive[b, t]:
                    _fire(sim, b, u, t, loop)
    if sim.config.hit_assist_cap > 0 and not sim.config.native_assist_on_hit:
        targeting.assist_flush(sim)  # cap assist triggers per victim to earliest hits in order
    _cleanup_all(sim)
    dead = ~w.alive
    c.windup_left[dead] = -1
    c.windup_target[dead] = NO_SLOT


def _void(sim: Sim, b: int, u: int) -> None:
    """`attack_commit`: abort this shot, refunding windup cooldown (floored at 0), clearing
    `buf.fire_target`, and restoring `last_fire_loop` / `last_fire_target` (and backswing).
    """
    c, buf = sim.combat, sim.buf
    buf.fire_target[b, u] = NO_SLOT
    c.cooldown[b, u] = max(c.cooldown[b, u] - c.windup_cost[b, u], f32(0))
    sim.world.weapon_cooldown[b, u] = c.cooldown[b, u]
    c.last_fire_loop[b, u] = buf.fire_prev_loop[b, u]
    c.last_fire_target[b, u] = buf.fire_prev_target[b, u]
    if sim.config.native_backswing:
        c.backswing_left[b, u] = buf.fire_prev_backswing[b, u]


def _cleanup(sim: Sim, b: int) -> None:
    """Kill units with `health <= 0`; clear orders and engagements targeting dead units
    (recording `combat.kill_idle` for ATTACK_UNIT becoming idle under `kill_no_retreat`).

    Overkill damage is discarded (multiple units may strike a dying target in the same loop).
    """
    w, c = sim.world, sim.combat
    dead = [u for u in range(sim.n_slots) if w.alive[b, u] and w.health[b, u] <= 0]
    for u in dead:
        w.alive[b, u] = False
        w.health[b, u] = 0
        w.order_kind[b, u] = CommandKind.NOOP
        w.order_ability[b, u] = 0
        w.order_target_slot[b, u] = NO_SLOT
        w.order_x[b, u] = w.order_y[b, u] = np.nan
        w.engaged_target_slot[b, u] = NO_SLOT
        c.engaged_transient[b, u] = False
        c.leash_return[b, u] = False
        c.defer_target[b, u] = NO_SLOT
        c.defer_abort[b, u] = False
        c.assist_target[b, u] = NO_SLOT
        c.assist_loop[b, u] = -1
        c.backswing_left[b, u] = 0
        c.assist_dedup_slot[b, u] = NO_SLOT
        c.assist_dedup_ttl[b, u] = 0
        c.threat_slot[b, u] = NO_SLOT
        c.threat_ttl[b, u] = 0
    use_native_any = (
        sim.config.native_backswing or sim.config.native_retarget or sim.config.native_assist_on_hit
    )
    for u in range(sim.n_slots):
        if not w.alive[b, u]:
            continue
        eng, tgt = int(w.engaged_target_slot[b, u]), int(w.order_target_slot[b, u])
        rem = NO_SLOT
        if use_native_any:
            rem = int(c.defer_target[b, u])
            if rem in dead:
                c.defer_target[b, u] = NO_SLOT
                rem = NO_SLOT
        if sim.config.native_backswing:
            kind = int(w.order_kind[b, u])
            if kind == CommandKind.ABILITY:
                if tgt in dead:
                    hold = bool(sim.ability.hold_after[b, u])
                    w.order_kind[b, u] = CommandKind.HOLD if hold else CommandKind.NOOP
                    w.order_ability[b, u] = 0
                    w.order_target_slot[b, u] = NO_SLOT
                    reset_state(sim, b, u)
                if eng in dead:
                    w.engaged_target_slot[b, u] = NO_SLOT
                    c.engaged_transient[b, u] = False
                    c.defer_abort[b, u] = False
            elif eng in dead or (kind == CommandKind.ATTACK_UNIT and tgt in dead):
                if int(c.backswing_left[b, u]) > 0:
                    continue  # backswing active: retain dead target, defer order pop / on_kill
                dead_t = eng if eng in dead else tgt
                has_rem = rem != NO_SLOT and bool(w.alive[b, rem])
                was_amove, _ = native_combat.pop_dead_order(sim, b, u)
                targeting.on_kill(sim, b, u, dead_t, was_amove=was_amove, has_rem=has_rem)
            continue
        if eng in dead or tgt in dead:
            targeting.on_kill(sim, b, u, eng if eng in dead else tgt)
        if int(w.order_target_slot[b, u]) in dead:
            # Medivacs autocasting from HOLD return to HOLD (see subsystems/abilities.py).
            hold = bool(sim.ability.hold_after[b, u])
            if sim.config.kill_no_retreat and int(w.order_kind[b, u]) == CommandKind.ATTACK_UNIT:
                sim.combat.kill_idle[b, u] = not hold
            w.order_kind[b, u] = CommandKind.HOLD if hold else CommandKind.NOOP
            w.order_ability[b, u] = 0
            w.order_target_slot[b, u] = NO_SLOT
            reset_state(sim, b, u)
        if int(w.engaged_target_slot[b, u]) in dead:
            w.engaged_target_slot[b, u] = NO_SLOT
            if use_native_any:
                c.engaged_transient[b, u] = False
                c.defer_abort[b, u] = False


IMPLS = {'standard': standard, 'sc2': sc2}

__all__ = ['IMPLS', 'SLOW_ONSET', 'apply_hit', 'sc2', 'standard']
