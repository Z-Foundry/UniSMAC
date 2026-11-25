"""Stage 2 targeting: writes `world.engaged_target_slot` and `buf.in_range`."""

from __future__ import annotations

import numpy as np

from unismac.contract import NO_SLOT, CommandKind

from .. import batch, rng
from ..sim import Sim, f32, is_enemy, range_edge, weapon_for
from . import abilities, native_combat


def in_weapon_range(sim: Sim, b: int, u: int, t: int, extra: float = 0.0) -> bool:
    """Return True if unit `u` can attack `t` and edge distance `<= weapon_range + extra`."""
    w = weapon_for(sim, b, u, t)
    if w < 0:
        return False
    reach = sim.params.weapon_range[b, u, w] + np.float32(extra)
    return bool(range_edge(sim, b, u, t) <= reach)


def nearest_enemy_in_range(sim: Sim, b: int, u: int, extra: float = 0.0) -> int:
    """Return closest enemy by edge distance within `weapon_range + extra` (lowest slot on tie)."""
    best, best_d = NO_SLOT, np.float32(np.inf)
    for t in range(sim.n_slots):
        if t == u or not is_enemy(sim.world, b, u, t) or not in_weapon_range(sim, b, u, t, extra):
            continue
        d = range_edge(sim, b, u, t)
        if d < best_d:
            best, best_d = t, d
    return best


def _in_sight(sim: Sim, b: int, u: int, t: int) -> bool:
    """Return True if `t` is an attackable enemy within `u`'s sight radius (center distance)."""
    w = sim.world
    if not is_enemy(w, b, u, t) or weapon_for(sim, b, u, t) < 0:
        return False
    dx, dy = w.x[b, t] - w.x[b, u], w.y[b, t] - w.y[b, u]
    return bool(np.sqrt(dx * dx + dy * dy) <= sim.params.sight[b, u])


def nearest_attacker(sim: Sim, b: int, u: int) -> int:
    """Return closest attackable enemy in sight that fired on `u` last loop, or `NO_SLOT`.

    Reads `buf.fire_target` before the attack stage overwrites it for the current loop.
    """
    best, best_d = NO_SLOT, np.float32(np.inf)
    for t in range(sim.n_slots):
        if int(sim.buf.fire_target[b, t]) != u or not _in_sight(sim, b, u, t):
            continue
        d = range_edge(sim, b, u, t)
        if d < best_d:
            best, best_d = t, d
    return best


def _attack_move_target(sim: Sim, b: int, u: int) -> int:
    """Keep current target if still an in-range enemy; otherwise pick nearest enemy in range."""
    w = sim.world
    cur = int(w.engaged_target_slot[b, u])
    if cur != NO_SLOT and is_enemy(w, b, u, cur) and in_weapon_range(sim, b, u, cur):
        return cur
    return nearest_enemy_in_range(sim, b, u)


def _idle_target(sim: Sim, b: int, u: int) -> int:
    """Target selection for idle units under `auto_acquire`:

    1. Keep current engaged target while alive and in sight (chases until out of sight).
    2. Otherwise pick nearest enemy within `weapon_range + acquire_margin`.
    3. Otherwise retaliate against nearest visible enemy that fired on `u` last loop.
    """
    cur = int(sim.world.engaged_target_slot[b, u])
    if cur != NO_SLOT and _in_sight(sim, b, u, cur):
        return cur
    t = nearest_enemy_in_range(sim, b, u, sim.config.acquire_margin)
    return t if t != NO_SLOT else nearest_attacker(sim, b, u)


def _acquires(sim: Sim, b: int, u: int) -> bool:
    """Return True if idle unit `(b, u)` may auto-acquire on this loop."""
    cfg = sim.config
    if int(sim.world.game_loop[b]) < cfg.acquire_start_loop:
        return False
    return cfg.auto_acquire_owners is None or int(sim.world.owner[b, u]) in cfg.auto_acquire_owners


def basic(sim: Sim) -> None:
    """Basic targeting implementation.

    * `ATTACK_UNIT`: engages the ordered target.
    * `ATTACK_MOVE`: keeps current in-range enemy, else nearest enemy in range.
    * `HOLD`: same as `ATTACK_MOVE` when `auto_acquire` is enabled, else none.
    * `NOOP`: `_idle_target` when `auto_acquire` is enabled, else none.
    * `MOVE`: does not engage or retaliate.

    Vectorized via `batch.pairs`, bit-identical to the scalar helpers above.
    """
    w, cfg = sim.world, sim.config
    auto = cfg.auto_acquire
    pr = batch.pairs(sim)
    not_self = ~np.eye(sim.n_slots, dtype=np.bool_)[None]
    hit0 = batch.in_range(pr)
    cand = not_self & pr.enemy & hit0
    kind = w.order_kind.astype(np.int64)
    cur = w.engaged_target_slot.astype(np.int64)
    has_cur = cur != NO_SLOT
    # ATTACK_MOVE / HOLD: keep in-range current target, else pick nearest in range.
    keep = has_cur & batch.gather(pr.enemy & hit0, cur)
    amove = np.where(keep, cur, batch.nearest(cand, pr.edge, NO_SLOT))
    target = np.full(kind.shape, NO_SLOT, np.int64)
    target = np.where(kind == CommandKind.ATTACK_UNIT, w.order_target_slot, target)
    use_amove = kind == CommandKind.ATTACK_MOVE
    if auto:
        use_amove |= kind == CommandKind.HOLD
    target = np.where(use_amove, amove, target)
    if auto:
        idle = (kind == CommandKind.NOOP) & (w.game_loop[:, None] >= cfg.acquire_start_loop)
        if cfg.auto_acquire_owners is not None:
            idle &= np.isin(w.owner, cfg.auto_acquire_owners)
        if idle.any():
            with np.errstate(invalid='ignore'):
                sight = pr.enemy & (pr.weapon >= 0) & (pr.center <= sim.params.sight[:, :, None])
            near = batch.nearest(
                not_self & pr.enemy & batch.in_range(pr, cfg.acquire_margin), pr.edge, NO_SLOT
            )
            fired = sim.buf.fire_target[:, None, :] == np.arange(sim.n_slots)[None, :, None]
            back = batch.nearest(fired & sight, pr.edge, NO_SLOT)
            pick = np.where(near != NO_SLOT, near, back)
            pick = np.where(has_cur & batch.gather(sight, cur), cur, pick)
            target = np.where(idle, pick, target)
    target = np.where(w.alive, target, NO_SLOT)
    w.engaged_target_slot[:] = target
    sim.buf.in_range[:] = (target != NO_SLOT) & batch.gather(hit0, target)


def sc2(sim: Sim) -> None:
    """SC2-style target acquisition, retargeting, and assist-on-hit subsystem.

    * Acquire range: `max(weapon_range + acquire_margin, acquire_min_range)` (edge-to-edge).
    * Priority: armed enemies first (including channeling healers under `heal_counts_armed`),
      then nearest edge distance (or center distance for idle/ATTACK_MOVE scans under
      `acquire_center`), breaking ties by lowest slot.
    * Scan cadence: idle (`NOOP`) and `ATTACK_MOVE` scan every `scan_period` loops (offset by
      `acquire_start_loop` and per-unit `combat.scan_shift`), plus immediate forced scans on
      order activation (`STOP`/`HOLD`/`ATTACK_UNIT`/`ATTACK_MOVE`) and `kill_rescan_delay` loops
      after a kill (`combat.scan_loop`).
    * Chase & retarget: keep engaged targets while attackable and visible; under `retarget`,
      switch on scan beats if out of weapon range and a closer candidate exists (`_retarget`).
    * `ATTACK_UNIT` engages its order target; `HOLD` scans in-range targets every loop; `MOVE`
      never engages.
    * Leash: idle units record `combat.leash_*` on engagement and MOVE back when a later scan
      finds no target.
    * Retaliation and retreat triggers run in `on_hit` during damage resolution.
    * Fog (`visibility='sc2'`): only considers enemies in `vision.true_sight` (fire reveal does
      not count). `ATTACK_UNIT` downgrades to `ATTACK_MOVE` when target leaves true sight and
      restores on re-sighting (`_fog_orders` / `after_vision`).

    When `auto_acquire_owners` is set, non-listed owners never auto-acquire on idle,
    `ATTACK_MOVE`, or `HOLD`.
    """
    w, cfg, c = sim.world, sim.config, sim.combat
    if cfg.native_assist_on_hit or cfg.native_backswing:
        native_combat.tick_native_timers(sim)
    hold_dead = np.zeros(w.alive.shape, np.bool_)
    promoted_this_loop = np.zeros(w.alive.shape, np.bool_)
    amove_no_rescan = np.zeros(w.alive.shape, np.bool_)
    if cfg.native_backswing or cfg.native_retarget or cfg.native_assist_on_hit:
        for b, u in zip(*np.nonzero(w.alive), strict=True):
            b, u = int(b), int(u)
            k_u = int(w.order_kind[b, u])
            eng_u = int(w.engaged_target_slot[b, u])
            au_t = int(w.order_target_slot[b, u]) if k_u == CommandKind.ATTACK_UNIT else NO_SLOT
            dead_tgt = (eng_u != NO_SLOT and not w.alive[b, eng_u]) or (
                au_t != NO_SLOT and not w.alive[b, au_t]
            )
            owners_u = (
                cfg.auto_acquire_owners is None or int(w.owner[b, u]) in cfg.auto_acquire_owners
            )
            if cfg.native_backswing and dead_tgt:
                if int(c.backswing_left[b, u]) > 0:
                    hold_dead[b, u] = True
                    continue
                rem = int(c.defer_target[b, u])
                has_rem = rem != NO_SLOT and bool(w.alive[b, rem] and w.health[b, rem] > 0)
                was_amove, _ = native_combat.pop_dead_order(sim, b, u)
                idle_acq = (
                    int(w.order_kind[b, u]) == CommandKind.NOOP and cfg.auto_acquire and owners_u
                )
                if was_amove and not has_rem:
                    amove_no_rescan[b, u] = True
                elif has_rem or idle_acq:
                    chosen = native_combat.reconcile_pop_rescan(
                        sim, b, u, allow_scan=idle_acq or (was_amove and owners_u)
                    )
                    promoted_this_loop[b, u] = True
                    if (
                        chosen == NO_SLOT
                        and int(w.order_kind[b, u]) == CommandKind.NOOP
                        and cfg.auto_acquire
                        and not np.isnan(c.leash_x[b, u])
                    ):
                        native_combat.apply_leash_return(sim, b, u)
                continue
            if dead_tgt:
                continue
            rem_b = int(c.defer_target[b, u])
            has_rem_b = rem_b != NO_SLOT and bool(w.alive[b, rem_b] and w.health[b, rem_b] > 0)
            abort_b = bool(c.defer_abort[b, u])
            if (rem_b != NO_SLOT or abort_b) and not native_combat.weapon_busy(sim, b, u):
                c.defer_abort[b, u] = False
                if not has_rem_b and not abort_b:
                    c.defer_target[b, u] = NO_SLOT
                    continue
                if k_u in (CommandKind.NOOP, CommandKind.ATTACK_MOVE):
                    if (abort_b or has_rem_b) and eng_u != NO_SLOT:
                        w.engaged_target_slot[b, u] = NO_SLOT
                        c.engaged_transient[b, u] = False
                        c.follow_tol[b, u] = False
                        c.tol_target[b, u] = NO_SLOT
                        c.rt_due[b, u] = -1
                        c.rt_lock[b, u] = NO_SLOT
                        c.assist_target[b, u] = NO_SLOT
                        c.assist_loop[b, u] = -1
                        c.scan_loop[b, u] = -1
                    if k_u == CommandKind.ATTACK_MOVE and not has_rem_b:
                        c.defer_target[b, u] = NO_SLOT
                        amove_no_rescan[b, u] = True
                    else:
                        allow = owners_u and (k_u == CommandKind.ATTACK_MOVE or cfg.auto_acquire)
                        chosen = native_combat.reconcile_pop_rescan(sim, b, u, allow_scan=allow)
                        promoted_this_loop[b, u] = True
                        if (
                            chosen == NO_SLOT
                            and k_u == CommandKind.NOOP
                            and cfg.auto_acquire
                            and not np.isnan(c.leash_x[b, u])
                        ):
                            native_combat.apply_leash_return(sim, b, u)
                else:
                    c.defer_target[b, u] = NO_SLOT

    pr = batch.pairs(sim)
    not_self = ~np.eye(sim.n_slots, dtype=np.bool_)[None]
    hit0 = batch.in_range(pr)
    foe = not_self & pr.enemy & (pr.weapon >= 0)
    lv_sc2 = cfg.lost_vision == 'sc2'
    foe_any = foe
    if cfg.visibility == 'sc2':
        seen = _seen(sim)
        foe_any = foe.copy()
        foe &= seen
        if not lv_sc2:
            _fog_orders(sim, seen)  # `lost_vision='sc2'` runs after visibility (`after_vision`)
    kind = w.order_kind.astype(np.int64)
    loop = w.game_loop[:, None]
    cur = w.engaged_target_slot.astype(np.int64)
    has_cur = cur != NO_SLOT
    with np.errstate(invalid='ignore'):
        acq = np.maximum(pr.reach + f32(cfg.acquire_margin), f32(cfg.acquire_min_range))
        in_acq = foe & (pr.edge <= acq)
        sight = foe & (pr.center <= sim.params.sight[:, :, None])
    armed_u = sim.params.has_weapon.any(axis=2)
    if cfg.heal_counts_armed:
        armed_u = armed_u | (
            (w.order_kind == CommandKind.ABILITY) & (w.order_ability == abilities.HEAL_ABILITY)
        )
    armed = armed_u[:, None, :]

    def pick(mask: np.ndarray, dist: np.ndarray = pr.edge) -> np.ndarray:
        first = batch.nearest(mask & armed, dist, NO_SLOT)
        return np.where(first != NO_SLOT, first, batch.nearest(mask, dist, NO_SLOT))

    owners_ok = np.ones(kind.shape, np.bool_)
    if cfg.auto_acquire_owners is not None:
        owners_ok = np.isin(w.owner, cfg.auto_acquire_owners)
    forced = (c.scan_loop >= 0) & (loop >= c.scan_loop)
    pending = (c.scan_loop >= 0) & ~forced
    if cfg.native_scan_cadence:
        periodic = (loop - c.scan_shift) % cfg.scan_period == 0
    elif cfg.scan_phase in ('per_unit', 'tag_order', 'tag_rotate'):
        # Per-unit phase `acquire_start_loop + scan_shift` with no initial loop gate.
        periodic = (loop - cfg.acquire_start_loop - c.scan_shift) % cfg.scan_period == 0
    else:
        periodic = (loop >= cfg.acquire_start_loop) & (
            (loop - cfg.acquire_start_loop) % cfg.scan_period == 0
        )
    scan = (forced | (periodic & ~pending)) & owners_ok
    # `scan_after_move`: periodic new acquires run at loop end (`late_scan`); only forced runs here.
    acquire = forced & owners_ok if cfg.scan_after_move else scan
    keep = has_cur & batch.gather(sight, cur) & owners_ok
    if lv_sc2:
        # Under `lost_vision='sc2'`, keep unseen idle targets here; `after_vision` handles 2-loop
        # loss.
        keep |= has_cur & (kind == CommandKind.NOOP) & batch.gather(foe_any, cur) & owners_ok
    keep &= ~hold_dead
    use_native_scoring = cfg.native_retarget
    use_native_assist_scan = cfg.native_assist_on_hit
    use_native_bs = cfg.native_backswing
    if use_native_scoring:
        scan_pick = np.full(kind.shape, NO_SLOT, np.int64)
        need_acq = (
            w.alive
            & acquire
            & ~has_cur
            & ~hold_dead
            & ~promoted_this_loop
            & ~amove_no_rescan
            & ((kind == CommandKind.ATTACK_MOVE) | ((kind == CommandKind.NOOP) & cfg.auto_acquire))
        )
        for b, u in zip(*np.nonzero(need_acq), strict=True):
            scan_pick[int(b), int(u)] = native_combat.pick_best_q12(
                sim, int(b), int(u), cur=NO_SLOT, hold_only=False, allow_spatial=True
            )
    else:
        scan_pick = pick(in_acq, pr.center) if cfg.acquire_center else pick(in_acq)
        if use_native_assist_scan:
            need_th = (
                w.alive
                & acquire
                & ~has_cur
                & ~hold_dead
                & ~promoted_this_loop
                & ~amove_no_rescan
                & (c.threat_ttl > 0).any(axis=2)
                & (
                    (kind == CommandKind.ATTACK_MOVE)
                    | ((kind == CommandKind.NOOP) & cfg.auto_acquire)
                )
            )
            for b, u in zip(*np.nonzero(need_th), strict=True):
                scan_pick[int(b), int(u)] = native_combat.pick_best_q12(
                    sim, int(b), int(u), cur=NO_SLOT, hold_only=False, allow_spatial=True
                )
    roam = np.where(
        keep, cur, np.where(acquire & ~promoted_this_loop & ~amove_no_rescan, scan_pick, NO_SLOT)
    )
    if use_native_bs and hold_dead.any():
        # During backswing dead-target hold, store best scan candidate into `defer_target`.
        hd_scan = (
            hold_dead
            & scan
            & owners_ok
            & ((kind == CommandKind.ATTACK_MOVE) | ((kind == CommandKind.NOOP) & cfg.auto_acquire))
        )
        for b, u in zip(*np.nonzero(hd_scan), strict=True):
            b, u = int(b), int(u)
            if cfg.scan_after_move and not forced[b, u]:
                sim.buf.late_retarget[b, u] = True
            else:
                cand = native_combat.pick_best_q12(
                    sim, b, u, cur=NO_SLOT, hold_only=False, allow_spatial=True
                )
                if cand != NO_SLOT:
                    native_combat.store_defer_target(sim, b, u, cand)
    if cfg.retarget:
        if use_native_scoring:
            rt_scan = scan & ~promoted_this_loop
            if cfg.retarget_after_move:
                late = scan & ~forced & keep & ~promoted_this_loop
                sim.buf.late_retarget[:] = sim.buf.late_retarget | late
                rt_scan = rt_scan & ~late
            rt_eval = (
                w.alive
                & rt_scan
                & keep
                & (
                    (kind == CommandKind.ATTACK_MOVE)
                    | ((kind == CommandKind.NOOP) & cfg.auto_acquire)
                )
            )
            for b, u in zip(*np.nonzero(rt_eval), strict=True):
                b, u = int(b), int(u)
                cur_u = int(cur[b, u])
                skip = native_combat.should_skip_spatial_16loop(
                    sim, b, u, cur_u, forced=bool(forced[b, u]), periodic=bool(periodic[b, u])
                )
                alt = native_combat.pick_best_q12(
                    sim, b, u, cur=cur_u, hold_only=False, allow_spatial=not skip
                )
                if alt != NO_SLOT and alt != cur_u:
                    if native_combat.weapon_busy(sim, b, u):
                        native_combat.store_defer_target(sim, b, u, alt)
                    else:
                        if cfg.retarget_lock != 'none':
                            c.rt_lock[b, u] = cur_u if cfg.retarget_lock == 'previous' else alt
                        roam[b, u] = alt
                        c.defer_target[b, u] = NO_SLOT
                        c.defer_abort[b, u] = False
                        c.assist_target[b, u] = NO_SLOT
                        c.assist_loop[b, u] = -1
        else:
            rt_scan = scan
            if cfg.retarget_after_move:
                # Defer non-busy, non-due periodic retarget checks to `late_scan` (post-move).
                busy, _ = _rt_busy(sim)
                due = (c.rt_due >= 0) & (loop >= c.rt_due)
                late = scan & ~forced & keep & ~busy & ~due
                sim.buf.late_retarget[:] = late
                rt_scan = scan & ~late
            roam = _retarget(sim, pr, roam, keep, rt_scan, hit0, cur, in_acq, pick, armed_u)

    target = np.full(kind.shape, NO_SLOT, np.int64)
    target = np.where(kind == CommandKind.ATTACK_UNIT, w.order_target_slot, target)
    target = np.where(kind == CommandKind.ATTACK_MOVE, roam, target)
    if use_native_scoring:
        hold = np.full(kind.shape, NO_SLOT, np.int64)
        for b, u in zip(
            *np.nonzero(w.alive & (kind == CommandKind.HOLD) & owners_ok & ~hold_dead), strict=True
        ):
            b, u = int(b), int(u)
            cur_u = int(cur[b, u]) if has_cur[b, u] else NO_SLOT
            hold[b, u] = native_combat.pick_best_q12(
                sim, b, u, cur=cur_u, hold_only=True, allow_spatial=True, ignore_lock=True
            )
    else:
        hold_keep = has_cur & batch.gather(foe & hit0, cur)
        hold = np.where(hold_keep, cur, pick(foe & hit0))
    target = np.where((kind == CommandKind.HOLD) & owners_ok, hold, target)
    idle = kind == CommandKind.NOOP
    if cfg.auto_acquire:
        target = np.where(idle, roam, target)
    if promoted_this_loop.any():
        target = np.where(promoted_this_loop, cur, target)
    if hold_dead.any():
        hd_tgt = np.where(kind == CommandKind.ATTACK_UNIT, w.order_target_slot, cur)
        target = np.where(hold_dead, hd_tgt, target)
    target = np.where(w.alive, target, NO_SLOT)
    roaming = idle | (kind == CommandKind.ATTACK_MOVE)
    c.scan_loop[forced & roaming & ~hold_dead] = -1
    if use_native_scoring or use_native_assist_scan or use_native_bs:
        tr = w.alive & ~hold_dead & roaming & (target != NO_SLOT)
        c.engaged_transient[:] = np.where(hold_dead, c.engaged_transient, tr)
    back = np.zeros(kind.shape, np.bool_)
    if cfg.auto_acquire:
        start = w.alive & ~hold_dead & idle & (target != NO_SLOT) & ~has_cur & np.isnan(c.leash_x)
        c.leash_x[start], c.leash_y[start] = w.x[start], w.y[start]
        back = (
            w.alive
            & ~hold_dead
            & ~promoted_this_loop
            & idle
            & acquire
            & (target == NO_SLOT)
            & ~np.isnan(c.leash_x)
        )
    w.engaged_target_slot[:] = target
    if use_native_scoring:
        sim.buf.in_range[:] = False
        for b, u in zip(*np.nonzero(w.alive & (target != NO_SLOT) & ~hold_dead), strict=True):
            b, u = int(b), int(u)
            sim.buf.in_range[b, u] = native_combat.is_strict_range_q12(sim, b, u, int(target[b, u]))
    else:
        sim.buf.in_range[:] = (target != NO_SLOT) & batch.gather(hit0, target)
    if cfg.follow_lock or cfg.slop_on_arrival:
        _follow_lock(sim, pr, target)
    if hold_dead.any():
        sim.buf.in_range[hold_dead] = True
        sim.buf.follow_hold[hold_dead] = False
        sim.buf.fire_slop[hold_dead] = False
    if back.any():
        for b, u in zip(*np.nonzero(back), strict=True):
            native_combat.apply_leash_return(sim, int(b), int(u))


def late_scan(sim: Sim) -> None:
    """`scan_after_move`: run periodic idle and ATTACK_MOVE target acquisition after movement,
    collision, damage, and visibility (invoked by `pipeline.run_loop` before incrementing
    `game_loop`).

    Uses the same acquire range, armed priority, true sight, and leash return rules as `sc2`.
    Under `hit_assist_pick='nearest'`, units that acquired via assist-on-hit this loop also scan
    and keep their assist target if no better enemy is in acquire range.
    """
    cfg = sim.config
    w, c = sim.world, sim.combat
    kind = w.order_kind
    loop = w.game_loop[:, None]
    roaming = kind == CommandKind.ATTACK_MOVE
    idle = kind == CommandKind.NOOP
    if cfg.auto_acquire:
        roaming = roaming | idle
    if cfg.auto_acquire_owners is not None:
        roaming &= np.isin(w.owner, cfg.auto_acquire_owners)
    if cfg.native_scan_cadence:
        periodic = (loop - c.scan_shift) % cfg.scan_period == 0
    elif cfg.scan_phase in ('per_unit', 'tag_order', 'tag_rotate'):
        periodic = (loop - cfg.acquire_start_loop - c.scan_shift) % cfg.scan_period == 0
    else:
        periodic = (loop >= cfg.acquire_start_loop) & (
            (loop - cfg.acquire_start_loop) % cfg.scan_period == 0
        )
    use_native_late = cfg.native_assist_on_hit or cfg.native_retarget or cfg.native_backswing
    pending = (c.scan_loop >= 0) & (loop < c.scan_loop)
    base = w.alive & roaming & periodic & ~pending
    need = base & (w.engaged_target_slot == NO_SLOT)
    fresh = np.zeros_like(need)
    if cfg.hit_assist_pick == 'nearest' or use_native_late:
        # Unit woke via same-loop assist: still scan and upgrade if a candidate is found.
        fresh = base & (w.engaged_target_slot == c.assist_target) & (c.assist_loop == loop)
        fresh &= w.engaged_target_slot != NO_SLOT
        if use_native_late and fresh.any():
            for b_i, u_i in zip(*np.nonzero(fresh), strict=True):
                if native_combat.weapon_busy(sim, int(b_i), int(u_i)):
                    fresh[b_i, u_i] = False
    if cfg.retarget_after_move or (cfg.native_backswing and sim.buf.late_retarget.any()):
        # Deferred retarget / backswing-hold scans flagged at start of loop.
        tgt = w.engaged_target_slot.astype(np.int64)
        valid_tgt = (tgt >= 0) & (tgt < sim.n_slots)
        alive_t = np.zeros(tgt.shape, np.bool_)
        if valid_tgt.any():
            vb, vu = np.nonzero(valid_tgt)
            alive_t[vb, vu] = w.alive[vb, tgt[vb, vu]]
        if cfg.native_backswing:
            hd_late = sim.buf.late_retarget & w.alive & ~pending & valid_tgt & ~alive_t
            if hd_late.any():
                for b, u in zip(*np.nonzero(hd_late), strict=True):
                    cand = native_combat.pick_best_q12(
                        sim, int(b), int(u), cur=NO_SLOT, hold_only=False, allow_spatial=True
                    )
                    if cand != NO_SLOT:
                        native_combat.store_defer_target(sim, int(b), int(u), cand)
        rt = sim.buf.late_retarget & w.alive & ~pending & (tgt != NO_SLOT) & alive_t & ~fresh
        if not cfg.native_retarget:
            rt &= c.last_fire_loop != loop
        sim.buf.late_retarget[:] = False
        if rt.any():
            _late_retarget(sim, rt)
    if fresh.any():
        _late_pick(sim, fresh)
    if not need.any():
        return
    got = _late_pick(sim, need)
    if not cfg.auto_acquire:
        return
    back = need & ~got & (kind == CommandKind.NOOP) & ~np.isnan(c.leash_x)
    if back.any():
        for b, u in zip(*np.nonzero(back), strict=True):
            native_combat.apply_leash_return(sim, int(b), int(u))


def kill_scan(sim: Sim) -> None:
    """`kill_rescan_delay=0` (or `native_backswing`): rescan idle and ATTACK_MOVE units whose
    target died this loop using end-of-loop positions; clears `scan_loop` when a target is picked.
    """
    cfg = sim.config
    w, c = sim.world, sim.combat
    kind = w.order_kind
    roaming = kind == CommandKind.ATTACK_MOVE
    if cfg.auto_acquire:
        roaming = roaming | (kind == CommandKind.NOOP)
    if cfg.auto_acquire_owners is not None:
        roaming &= np.isin(w.owner, cfg.auto_acquire_owners)
    now = c.scan_loop == w.game_loop[:, None]
    if cfg.native_backswing:
        # Include ATTACK_MOVE units holding `defer_target` even if filtered by `owners_ok`.
        need = (
            w.alive
            & now
            & (w.engaged_target_slot == NO_SLOT)
            & (roaming | ((kind == CommandKind.ATTACK_MOVE) & (c.defer_target != NO_SLOT)))
        )
        if not need.any():
            return
        for b, u in zip(*np.nonzero(need), strict=True):
            b, u = int(b), int(u)
            owners_u = (
                cfg.auto_acquire_owners is None or int(w.owner[b, u]) in cfg.auto_acquire_owners
            )
            allow = (int(kind[b, u]) == CommandKind.NOOP and cfg.auto_acquire and owners_u) or (
                int(kind[b, u]) == CommandKind.ATTACK_MOVE and owners_u
            )
            chosen = native_combat.reconcile_pop_rescan(sim, b, u, allow_scan=allow)
            if (
                chosen != NO_SLOT
                or int(kind[b, u]) != CommandKind.NOOP
                or np.isnan(c.leash_x[b, u])
            ):
                c.scan_loop[b, u] = -1
        return
    need = w.alive & roaming & now & (w.engaged_target_slot == NO_SLOT)
    if not need.any():
        return
    got = _late_pick(sim, need)
    c.scan_loop[got] = -1


def _late_pick(sim: Sim, need: np.ndarray) -> np.ndarray:
    """Acquire targets for `need` using current end-of-loop positions (recording leash for idle
    units); returns bool mask of units that acquired a target.
    """
    cfg = sim.config
    w, c = sim.world, sim.combat
    if cfg.native_retarget or cfg.native_assist_on_hit or cfg.native_backswing:
        got = np.zeros_like(need)
        for b, u in zip(*np.nonzero(need), strict=True):
            b, u = int(b), int(u)
            pick_t = native_combat.pick_best_q12(
                sim, b, u, cur=NO_SLOT, hold_only=False, allow_spatial=True
            )
            if pick_t != NO_SLOT:
                w.engaged_target_slot[b, u] = pick_t
                c.engaged_transient[b, u] = True
                c.defer_abort[b, u] = False
                c.assist_target[b, u] = NO_SLOT
                c.assist_loop[b, u] = -1
                if (
                    cfg.auto_acquire
                    and int(w.order_kind[b, u]) == CommandKind.NOOP
                    and np.isnan(c.leash_x[b, u])
                ):
                    c.leash_x[b, u], c.leash_y[b, u] = w.x[b, u], w.y[b, u]
                got[b, u] = True
        return got
    idle = w.order_kind == CommandKind.NOOP
    pr = batch.pairs(sim)
    not_self = ~np.eye(sim.n_slots, dtype=np.bool_)[None]
    foe = not_self & pr.enemy & (pr.weapon >= 0)
    if cfg.visibility == 'sc2':
        foe &= _seen(sim)
    with np.errstate(invalid='ignore'):
        acq = np.maximum(pr.reach + f32(cfg.acquire_margin), f32(cfg.acquire_min_range))
        in_acq = foe & (pr.edge <= acq)
    armed_u = sim.params.has_weapon.any(axis=2)
    if cfg.heal_counts_armed:
        armed_u = armed_u | (
            (w.order_kind == CommandKind.ABILITY) & (w.order_ability == abilities.HEAL_ABILITY)
        )
    dist = pr.center if cfg.acquire_center else pr.edge
    first = batch.nearest(in_acq & armed_u[:, None, :], dist, NO_SLOT)
    pick = np.where(first != NO_SLOT, first, batch.nearest(in_acq, dist, NO_SLOT))
    got = need & (pick != NO_SLOT)
    w.engaged_target_slot[got] = pick[got]
    if cfg.auto_acquire:
        start = got & idle & np.isnan(c.leash_x)
        c.leash_x[start], c.leash_y[start] = w.x[start], w.y[start]
    return got


def _retarget(sim: Sim, pr, roam, keep, scan, hit0, cur, in_acq, pick, armed_u):
    """Evaluate `retarget` on scan/due beats for units keeping an out-of-range target.

    Switches to a closer in-range candidate (plus `retarget_ready` / `retarget_acquire` sets)
    via `pick`, respecting `retarget_lock` (`'new'`, `'previous'`, `'none'`),
    `retarget_armed_only`, busy-window deferral (`retarget_busy` / `retarget_busy_lock`),
    `retarget_defer` (`'scan'` / `'switch'`), and `retarget_defer_memory` (`combat.rt_alt`).
    """
    c, w, cfg = sim.combat, sim.world, sim.config
    lock = cfg.retarget_lock
    loop = w.game_loop[:, None]
    busy, nb = _rt_busy(sim)
    curi = np.where(cur != NO_SLOT, cur, 0)
    switch_only = cfg.retarget_defer == 'switch'
    memory = cfg.retarget_defer_memory
    mem = c.rt_alt.copy() if memory else c.rt_alt
    if switch_only:
        # Newly deferred `rt_due` values are in the future, so evaluating due first matches.
        due = (c.rt_due >= 0) & (loop >= c.rt_due)
        c.rt_due[due | ~keep] = -1
        if memory:
            c.rt_alt[due | ~keep] = NO_SLOT
        out = keep & (scan | due) & ~batch.gather(hit0, curi)
    else:
        defer = keep & scan & busy
        c.rt_due[defer] = (c.last_fire_loop + nb)[defer]
        due = (c.rt_due >= 0) & (loop >= c.rt_due)
        c.rt_due[due | ~keep] = -1
        out = keep & ((scan & ~busy) | due) & ~batch.gather(hit0, curi)
    if lock == 'new':
        out &= cur != c.rt_lock
    if not out.any():
        return roam
    alt = _rt_pick(sim, pr, curi, hit0, in_acq, pick, armed_u)
    switch = out & (alt != NO_SLOT)
    if switch_only:
        defer = switch & scan & busy & ~due
        c.rt_due[defer] = (c.last_fire_loop + nb)[defer]
        switch &= ~defer
        if memory:
            # At expiry with no candidate and current target out of acquire range: fall back to
            # remembered `rt_alt` (or drop engagement if `rt_alt` died).
            c.rt_alt[defer] = alt[defer]
            fall = out & due & (alt == NO_SLOT) & (mem != NO_SLOT) & ~batch.gather(in_acq, curi)
            live = np.take_along_axis(w.alive, np.where(mem != NO_SLOT, mem, 0), axis=1)
            alt = np.where(fall, np.where(live, mem, NO_SLOT), alt)
            switch |= fall
    if lock != 'none':
        moved = switch & (alt != NO_SLOT)
        c.rt_lock[moved] = (cur if lock == 'previous' else alt)[moved]
    return np.where(switch, alt, roam)


def _rt_busy(sim: Sim) -> tuple[np.ndarray, int | np.ndarray]:
    """Return `(busy_mask, busy_loops)` for retarget deferral after firing."""
    c, cfg = sim.combat, sim.config
    loop = sim.world.game_loop[:, None]
    nb: int | np.ndarray = cfg.retarget_busy
    if cfg.retarget_busy_lock:
        fl = sim.params.follow_lock
        nb = np.where(fl >= 0, fl - 1, cfg.retarget_busy)
    busy = (c.last_fire_loop >= 0) & (loop - c.last_fire_loop < nb)
    return busy, nb


def _rt_pick(sim: Sim, pr, curi, hit0, in_acq, pick, armed_u) -> np.ndarray:
    """Select best retarget candidate (shared by `_retarget` and `_late_retarget`), or `NO_SLOT`."""
    c, w, cfg = sim.combat, sim.world, sim.config
    with np.errstate(invalid='ignore'):
        cur_edge = batch.gather(pr.edge, curi)[:, :, None]
        closer = in_acq & hit0 & (pr.edge < cur_edge)
        if cfg.retarget_ready:
            ready = ~(w.weapon_cooldown > 0)
            margin = np.float32(cfg.retarget_ready_margin)
            closer |= ready[:, :, None] & in_acq & (pr.edge < cur_edge - margin)
        if cfg.retarget_acquire:
            cur_center = batch.gather(pr.center, curi)[:, :, None]
            closer |= in_acq & (pr.center < cur_center)
    if cfg.retarget_lock == 'previous':
        closer &= c.rt_lock[:, :, None] != np.arange(closer.shape[2])
    if cfg.retarget_armed_only:
        cur_armed = np.take_along_axis(armed_u, curi, axis=1)
        closer &= armed_u[:, None, :] | ~cur_armed[:, :, None]
    return pick(closer)


def _late_retarget(sim: Sim, mask: np.ndarray) -> None:
    """`retarget_after_move`: run deferred periodic retarget evaluation for units in `mask`
    using post-move positions (idle under `auto_acquire` and `ATTACK_MOVE` only).
    """
    cfg = sim.config
    w, c = sim.world, sim.combat
    if cfg.native_retarget:
        for b, u in zip(*np.nonzero(mask), strict=True):
            b, u = int(b), int(u)
            kind_u = int(w.order_kind[b, u])
            if not (
                kind_u == CommandKind.ATTACK_MOVE
                or (kind_u == CommandKind.NOOP and cfg.auto_acquire)
            ):
                continue
            cur_u = int(w.engaged_target_slot[b, u])
            if cur_u == NO_SLOT or not w.alive[b, cur_u]:
                continue
            skip = native_combat.should_skip_spatial_16loop(
                sim, b, u, cur_u, forced=False, periodic=True
            )
            alt = native_combat.pick_best_q12(
                sim, b, u, cur=cur_u, hold_only=False, allow_spatial=not skip
            )
            if alt != NO_SLOT and alt != cur_u:
                if native_combat.weapon_busy(sim, b, u):
                    native_combat.store_defer_target(sim, b, u, alt)
                else:
                    if cfg.retarget_lock != 'none':
                        c.rt_lock[b, u] = cur_u if cfg.retarget_lock == 'previous' else alt
                    w.engaged_target_slot[b, u] = alt
                    c.engaged_transient[b, u] = True
                    c.defer_target[b, u] = NO_SLOT
                    c.defer_abort[b, u] = False
                    c.assist_target[b, u] = NO_SLOT
                    c.assist_loop[b, u] = -1
        return
    pr = batch.pairs(sim)
    not_self = ~np.eye(sim.n_slots, dtype=np.bool_)[None]
    hit0 = batch.in_range(pr)
    foe = not_self & pr.enemy & (pr.weapon >= 0)
    if cfg.visibility == 'sc2':
        foe &= _seen(sim)
    with np.errstate(invalid='ignore'):
        acq = np.maximum(pr.reach + f32(cfg.acquire_margin), f32(cfg.acquire_min_range))
        in_acq = foe & (pr.edge <= acq)
    armed_u = sim.params.has_weapon.any(axis=2)
    if cfg.heal_counts_armed:
        armed_u = armed_u | (
            (w.order_kind == CommandKind.ABILITY) & (w.order_ability == abilities.HEAL_ABILITY)
        )
    armed = armed_u[:, None, :]

    def pick(m: np.ndarray, dist: np.ndarray = pr.edge) -> np.ndarray:
        first = batch.nearest(m & armed, dist, NO_SLOT)
        return np.where(first != NO_SLOT, first, batch.nearest(m, dist, NO_SLOT))

    cur = w.engaged_target_slot.astype(np.int64)
    curi = np.where(cur != NO_SLOT, cur, 0)
    out = mask & (cur != NO_SLOT) & ~batch.gather(hit0, curi)
    if cfg.retarget_lock == 'new':
        out &= cur != c.rt_lock
    if not out.any():
        return
    alt = _rt_pick(sim, pr, curi, hit0, in_acq, pick, armed_u)
    switch = out & (alt != NO_SLOT)
    if cfg.retarget_lock != 'none':
        c.rt_lock[switch] = (cur if cfg.retarget_lock == 'previous' else alt)[switch]
    roaming = w.order_kind == CommandKind.ATTACK_MOVE
    if cfg.auto_acquire:
        roaming = roaming | (w.order_kind == CommandKind.NOOP)
    switch &= roaming
    w.engaged_target_slot[switch] = alt[switch]


_FOLLOW_KINDS = (CommandKind.ATTACK_UNIT, CommandKind.ATTACK_MOVE, CommandKind.NOOP)


def _follow_lock(sim: Sim, pr: batch.Pairs, target: np.ndarray) -> None:
    """`follow_lock` and `slop_on_arrival`: write `buf.follow_hold`, `buf.fire_slop`, and
    `combat.follow_tol` without altering strict `buf.in_range`.

    Applies to alive units with `measured.json` `follow_lock` `L >= 0` on `ATTACK_UNIT`,
    `ATTACK_MOVE`, or `NOOP` orders engaging `last_fire_target` (`since = game_loop - last_fire`):

    1. `since < L` (locked): hold position; allow firing only in strict range (no slop).
    2. `since == L` (lock expiry): enter range slop if in strict range, else chase this loop.
    3. `since > L`: while in range slop (`range + follow_slop`), hold and allow firing; leave slop
       and chase once outside (re-entering slop upon returning to strict range).

    Lock-breaking commands (`combat.follow_cmd > last_fire_loop`) shift the expiry to
    `follow_cmd + move_lag`, whereas same-target `ATTACK_UNIT` and same-point `ATTACK_MOVE`
    reissues preserve the lock.

    With `slop_on_arrival`, non-`last_fire_target` engagements also enter range slop upon
    reaching strict range (tracked per target via `combat.tol_target`) without a post-fire lock.
    """
    w, c, buf = sim.world, sim.combat, sim.buf
    lock = sim.params.follow_lock
    kind = w.order_kind
    loop = w.game_loop[:, None]
    since = loop - c.last_fire_loop
    # Lock-breaking command after fire: shift virtual expiry to `e = follow_cmd + move_lag`.
    broke = c.follow_cmd > c.last_fire_loop
    if broke.any():
        end = c.follow_cmd + sim.config.move_lag
        since = np.where(broke, loop - end + lock, since)
    follow_kind = np.isin(kind, [int(k) for k in _FOLLOW_KINDS])
    base = w.alive & (lock >= 0) & (target != NO_SLOT) & follow_kind
    on = base & (c.last_fire_loop >= 0) & (target == c.last_fire_target)
    strict = buf.in_range
    ti = np.where(target != NO_SLOT, target, 0)
    with np.errstate(invalid='ignore'):
        reach = batch.gather(pr.reach, ti) + f32(sim.config.follow_slop)
        slop = (batch.gather(pr.weapon, ti) >= 0) & (batch.gather(pr.edge, ti) <= reach)
    locked = on & (since < lock)
    if not sim.config.follow_lock:  # `slop_on_arrival` only: leave last_fire_target untouched
        locked[:] = False
        on[:] = False
        base &= c.last_fire_target != target
    if sim.config.slop_on_arrival:
        # Slop is bound to `tol_target`; lock-breaking command on this loop resets to strict range.
        prev = c.follow_tol & (c.tol_target == target)
        tol = np.where(since == lock, strict, (prev & slop) | strict) & on & ~locked
        arr = base & ~on
        prev &= c.follow_cmd != loop
        tol |= arr & ((prev & slop) | strict)
        c.tol_target[:] = np.where(tol, target, NO_SLOT)
    else:
        # At expiry (`since == lock`), require strict range (avoid carrying stale pre-fire tol).
        tol = np.where(since == lock, strict, (c.follow_tol & slop) | strict)
        tol &= on & ~locked
    c.follow_tol[:] = tol
    buf.fire_slop[:] = tol & ~strict
    buf.follow_hold[:] = (locked | tol) & ~strict


def _seen(sim: Sim) -> np.ndarray:
    """Return `bool[B, U, T]`: whether `u`'s owner has true sight on `t` (`visibility='sc2'`)."""
    w = sim.world
    owner = np.clip(w.owner.astype(np.int64) - 1, 0, w.n_players - 1)  # unused for empty slots
    ts = np.transpose(sim.vision.true_sight, (0, 2, 1))  # [B, P, T]
    return np.take_along_axis(ts, owner[:, :, None], axis=1)


def _fog_orders(sim: Sim, seen: np.ndarray) -> None:
    """Downgrade `ATTACK_UNIT` to `ATTACK_MOVE` when target leaves true sight, and restore
    `ATTACK_UNIT` if the remembered target re-enters true sight.
    """
    w, c = sim.world, sim.combat
    kind = w.order_kind
    tgt = w.order_target_slot.astype(np.int64)
    lost = w.alive & (kind == CommandKind.ATTACK_UNIT) & (tgt != NO_SLOT)
    lost &= ~batch.gather(seen, tgt)
    old = c.lost_target.astype(np.int64)
    has_old = (old != NO_SLOT) & (kind == CommandKind.ATTACK_MOVE) & w.alive
    has_old &= batch.gather(w.alive[:, None, :] & seen, old)
    c.lost_target[kind != CommandKind.ATTACK_MOVE] = NO_SLOT
    for b, u in zip(*np.nonzero(has_old), strict=True):
        b, u = int(b), int(u)
        w.order_kind[b, u] = CommandKind.ATTACK_UNIT
        w.order_target_slot[b, u] = c.lost_target[b, u]
        w.order_x[b, u] = w.order_y[b, u] = np.nan
        c.lost_target[b, u] = NO_SLOT
    for b, u in zip(*np.nonzero(lost), strict=True):
        b, u = int(b), int(u)
        t = int(tgt[b, u])
        x0, y0, x1, y1 = sim.playable[b]
        w.order_kind[b, u] = CommandKind.ATTACK_MOVE
        w.order_target_slot[b, u] = NO_SLOT
        w.order_x[b, u] = np.clip(w.x[b, t], x0, x1)
        w.order_y[b, u] = np.clip(w.y[b, t], y0, y1)
        w.engaged_target_slot[b, u] = NO_SLOT
        c.lost_target[b, u] = t


def after_vision(sim: Sim) -> None:
    """`lost_vision='sc2'`: handle engaged-target vision loss at the end of `visibility.sc2`.

    * Record target position on the first unseen loop (`combat.unseen_*`) for `ATTACK_UNIT` and
      idle engagements; downgrade to `ATTACK_MOVE` toward that position on the second consecutive
      unseen loop.
    * Restore original `ATTACK_UNIT` (or idle engagement with leash intact) immediately upon
      regaining true sight.
    * Give up after `lost_vision_giveup` loops without re-sighting or engaging another target:
      idle units return to leash via MOVE, `ATTACK_UNIT` units STOP.
    """
    w, c, cfg = sim.world, sim.combat, sim.config
    kind = w.order_kind
    seen = _seen(sim)
    au = w.alive & (kind == CommandKind.ATTACK_UNIT) & (w.order_target_slot != NO_SLOT)
    idle = w.alive & (kind == CommandKind.NOOP) & (w.engaged_target_slot != NO_SLOT)
    track = au | idle
    tgt = np.where(au, w.order_target_slot, w.engaged_target_slot).astype(np.int64)
    tgt = np.where(track, tgt, 0)
    t_alive = np.take_along_axis(w.alive, tgt, axis=1)
    t_seen = batch.gather(seen, tgt)
    c.unseen_x[~track | ~t_alive | t_seen] = np.nan
    c.unseen_y[~track | ~t_alive | t_seen] = np.nan
    c.lost_target[kind != CommandKind.ATTACK_MOVE] = NO_SLOT
    gone = track & t_alive & ~t_seen
    for b, u in zip(*np.nonzero(gone), strict=True):
        b, u = int(b), int(u)
        t = int(tgt[b, u])
        if np.isnan(c.unseen_x[b, u]):
            c.unseen_x[b, u], c.unseen_y[b, u] = w.x[b, t], w.y[b, t]
            continue
        x0, y0, x1, y1 = sim.playable[b]
        c.lost_idle[b, u] = bool(idle[b, u])
        w.order_kind[b, u] = CommandKind.ATTACK_MOVE
        w.order_target_slot[b, u] = NO_SLOT
        w.order_x[b, u] = np.clip(c.unseen_x[b, u], x0, x1)
        w.order_y[b, u] = np.clip(c.unseen_y[b, u], y0, y1)
        w.engaged_target_slot[b, u] = NO_SLOT
        c.lost_target[b, u] = t
        c.lost_loop[b, u] = int(w.game_loop[b])
        c.unseen_x[b, u] = c.unseen_y[b, u] = np.nan
    lost = w.alive & (kind == CommandKind.ATTACK_MOVE) & (c.lost_target != NO_SLOT) & ~gone
    old = np.where(lost, c.lost_target, 0).astype(np.int64)
    back = lost & batch.gather(w.alive[:, None, :] & seen, old)
    giveup = lost & ~back & np.take_along_axis(w.alive, old, axis=1)
    giveup &= w.engaged_target_slot == NO_SLOT
    giveup &= w.game_loop[:, None] - c.lost_loop >= cfg.lost_vision_giveup
    for b, u in zip(*np.nonzero(back), strict=True):
        b, u = int(b), int(u)
        t = int(c.lost_target[b, u])
        if c.lost_idle[b, u]:
            w.order_kind[b, u] = CommandKind.NOOP
            w.order_target_slot[b, u] = NO_SLOT
        else:
            w.order_kind[b, u] = CommandKind.ATTACK_UNIT
            w.order_target_slot[b, u] = t
        w.order_x[b, u] = w.order_y[b, u] = np.nan
        w.engaged_target_slot[b, u] = t
        c.lost_target[b, u] = NO_SLOT
        c.lost_loop[b, u] = -1
    if giveup.any():
        from .commands import _apply

        for b, u in zip(*np.nonzero(giveup), strict=True):
            b, u = int(b), int(u)
            lx, ly = c.leash_x[b, u], c.leash_y[b, u]
            if c.lost_idle[b, u] and not np.isnan(lx):
                _apply(sim, b, u, int(CommandKind.MOVE), NO_SLOT, lx, ly)
            else:
                _apply(sim, b, u, int(CommandKind.STOP), NO_SLOT, np.float32(np.nan),
                       np.float32(np.nan))  # fmt: skip
            w.engaged_target_slot[b, u] = NO_SLOT


# ---------------------------------------------------------------- `targeting='sc2'` event hooks
# Called by other subsystems; no-ops when `targeting != 'sc2'`.

_SCAN_ON_COMMAND = (
    CommandKind.STOP,
    CommandKind.HOLD,
    CommandKind.ATTACK_UNIT,
    CommandKind.ATTACK_MOVE,
)


def on_command(sim: Sim, b: int, u: int, kind: int) -> None:
    """Command activation hook (`commands._apply`): reset leash/retarget/vision state and
    schedule an immediate scan on this loop for `_SCAN_ON_COMMAND` kinds.
    """
    if sim.config.targeting != 'sc2':
        return
    c = sim.combat
    c.leash_x[b, u] = c.leash_y[b, u] = np.nan
    c.lost_target[b, u] = NO_SLOT
    c.lost_loop[b, u] = -1
    c.unseen_x[b, u] = c.unseen_y[b, u] = np.nan
    c.rt_due[b, u], c.rt_lock[b, u] = -1, NO_SLOT
    c.engaged_transient[b, u] = False
    c.leash_return[b, u] = False
    c.defer_target[b, u] = NO_SLOT
    c.defer_abort[b, u] = False
    c.backswing_left[b, u] = 0
    if sim.config.kill_no_retreat:
        c.kill_idle[b, u] = False
    if (
        sim.config.hit_assist_pick != 'first'
        or sim.config.native_assist_on_hit
        or sim.config.native_retarget
        or sim.config.native_backswing
    ):
        c.assist_target[b, u], c.assist_loop[b, u] = NO_SLOT, -1
    c.scan_loop[b, u] = int(sim.world.game_loop[b]) if kind in _SCAN_ON_COMMAND else -1


def on_hit(sim: Sim, b: int, attacker: int, victim: int) -> None:
    """Damage-resolution hit hook for `targeting='sc2'`.

    For surviving unengaged victims:
    * Idle units belonging to `retreat_owners` flee by `retreat_distance` away from `attacker`
      (suppressed during post-kill idle window under `kill_no_retreat`).
    * Otherwise, eligible idle or `ATTACK_MOVE` units with a weapon (and true sight under
      `visibility='sc2'`) immediately engage `attacker` regardless of acquire range (skipped
      during post-kill scan wait when `hit_keeps_kill_wait` is enabled).

    Also triggers nearby ally assist (`hit_assist`, deferred to `assist_flush` if
    `hit_assist_cap > 0`, or routed through `native_combat.assist_on_hit_q12` when
    `native_assist_on_hit` is enabled).
    """
    cfg = sim.config
    if cfg.targeting != 'sc2':
        return
    w, c = sim.world, sim.combat
    use_native_assist = cfg.native_assist_on_hit
    use_native_any = use_native_assist or cfg.native_backswing or cfg.native_retarget
    if use_native_assist:
        if not (w.alive[b, victim] and w.alive[b, attacker] and w.health[b, attacker] > 0):
            return
        lethal = bool(w.health[b, victim] <= 0)
        if cfg.hit_assist:
            native_combat.assist_on_hit_q12(sim, b, attacker, victim, lethal=lethal)
        if lethal:
            return
    else:
        if not (w.alive[b, victim] and w.health[b, victim] > 0 and w.alive[b, attacker]):
            return
        if cfg.hit_assist:
            if cfg.hit_assist_cap > 0:
                sim.buf.assist_hits[b, victim, attacker] = True
            else:
                _assist(sim, b, attacker, victim)
    kind = int(w.order_kind[b, victim])
    owner = int(w.owner[b, victim])
    cur_v = int(w.engaged_target_slot[b, victim])
    if cur_v != NO_SLOT:
        if (
            use_native_any
            and cur_v != attacker
            and kind in (CommandKind.NOOP, CommandKind.ATTACK_MOVE)
        ):
            if kind == CommandKind.NOOP and not cfg.auto_acquire:
                return
            if cfg.auto_acquire_owners is not None and owner not in cfg.auto_acquire_owners:
                return
            if weapon_for(sim, b, victim, attacker) < 0:
                return
            if cfg.visibility == 'sc2' and not sim.vision.true_sight[b, attacker, owner - 1]:
                return
            if use_native_assist:
                native_combat.add_threat_memory(sim, b, victim, attacker)
            if not w.alive[b, cur_v] or w.health[b, cur_v] <= 0:
                native_combat.store_defer_target(sim, b, victim, attacker)
            elif native_combat.compare_key(
                sim, b, victim, attacker, cur_v
            ) < native_combat.compare_key(sim, b, victim, cur_v, cur_v):
                if native_combat.weapon_busy(sim, b, victim):
                    native_combat.store_defer_target(sim, b, victim, attacker)
                else:
                    same_loop_assist = int(c.assist_loop[b, victim]) == int(
                        w.game_loop[b]
                    ) and cur_v == int(c.assist_target[b, victim])
                    native_combat._set_transient_engaged(
                        sim, b, victim, attacker, record_assist=same_loop_assist
                    )
                    c.defer_target[b, victim] = NO_SLOT
        return
    if kind == CommandKind.NOOP and owner in cfg.retreat_owners:
        if cfg.kill_no_retreat and c.kill_idle[b, victim] and _kill_lock_holds(sim, b, victim):
            return
        if use_native_any:
            native_combat.apply_retreat_flee(sim, b, victim, attacker)
            return
        dx = float(w.x[b, victim]) - float(w.x[b, attacker])
        dy = float(w.y[b, victim]) - float(w.y[b, attacker])
        n = float(np.hypot(dx, dy))
        if n > 0:
            from .commands import _apply

            d = cfg.retreat_distance / n
            tx, ty = f32(float(w.x[b, victim]) + dx * d), f32(float(w.y[b, victim]) + dy * d)
            _apply(sim, b, victim, int(CommandKind.MOVE), NO_SLOT, tx, ty)
        return
    if kind == CommandKind.NOOP and not cfg.auto_acquire:
        return
    if not (
        kind in (CommandKind.NOOP, CommandKind.ATTACK_MOVE)
        or (
            use_native_assist
            and kind == CommandKind.MOVE
            and bool(c.leash_return[b, victim])
            and cfg.auto_acquire
            and native_combat.d2_raw(
                *native_combat.unit_xy_q12(sim, b, victim),
                native_combat.q12_scalar(float(w.order_x[b, victim])),
                native_combat.q12_scalar(float(w.order_y[b, victim])),
            )
            <= native_combat.LEASH_RESET_RADIUS_SQ
        )
    ):
        return
    if cfg.auto_acquire_owners is not None and owner not in cfg.auto_acquire_owners:
        return
    if weapon_for(sim, b, victim, attacker) < 0:
        return
    if cfg.visibility == 'sc2' and not sim.vision.true_sight[b, attacker, owner - 1]:
        return  # fire reveal does not count: require true sight to retaliate
    sl = int(c.scan_loop[b, victim])
    if (
        not cfg.native_backswing
        and cfg.hit_keeps_kill_wait
        and 0 <= sl
        and int(w.game_loop[b]) < sl
    ):
        return  # do not cut post-kill scan wait short on hit
    if use_native_assist:
        native_combat.add_threat_memory(sim, b, victim, attacker)
    if use_native_any:
        if native_combat.weapon_busy(sim, b, victim):
            native_combat.store_defer_target(sim, b, victim, attacker)
        else:
            native_combat._set_transient_engaged(sim, b, victim, attacker, record_assist=True)
        return
    w.engaged_target_slot[b, victim] = attacker
    c.scan_loop[b, victim] = -1
    if kind == CommandKind.NOOP and np.isnan(c.leash_x[b, victim]):
        c.leash_x[b, victim], c.leash_y[b, victim] = w.x[b, victim], w.y[b, victim]


def _assist(sim: Sim, b: int, attacker: int, victim: int) -> None:
    """`hit_assist`: wake unengaged idle (`auto_acquire`) and `ATTACK_MOVE` allies within edge
    distance `<= hit_assist_range` of surviving `victim` (not waiting for a post-kill scan, with
    weapon and true sight) to engage `attacker` on this loop (recording leash for idle allies).

    Under `hit_assist_pick='nearest'`, allies already engaging a prior assist target that have
    not yet fired switch to `attacker` if closer by center distance (breaking same-loop ties by
    `hit_assist_tie`).
    """
    cfg = sim.config
    w, c = sim.world, sim.combat
    owner = int(w.owner[b, victim])
    if cfg.auto_acquire_owners is not None and owner not in cfg.auto_acquire_owners:
        return
    if cfg.visibility == 'sc2' and not sim.vision.true_sight[b, attacker, owner - 1]:
        return
    reach = f32(cfg.hit_assist_range)
    loop = int(w.game_loop[b])
    nearest = cfg.hit_assist_pick == 'nearest'
    for h in range(sim.n_slots):
        if h == victim or not w.alive[b, h] or int(w.owner[b, h]) != owner:
            continue
        kind = int(w.order_kind[b, h])
        if not (kind == CommandKind.ATTACK_MOVE or (kind == CommandKind.NOOP and cfg.auto_acquire)):
            continue
        if weapon_for(sim, b, h, attacker) < 0:
            continue
        cur = int(w.engaged_target_slot[b, h])
        if cur != NO_SLOT and not (nearest and _assist_switch(sim, b, h, cur, attacker)):
            continue
        if 0 <= int(c.scan_loop[b, h]) and loop < int(c.scan_loop[b, h]):
            continue  # skip allies waiting in post-kill rescan delay
        if range_edge(sim, b, h, victim) > reach:
            continue
        w.engaged_target_slot[b, h] = attacker
        c.scan_loop[b, h] = -1
        if nearest:
            c.assist_target[b, h], c.assist_loop[b, h] = attacker, loop
        if kind == CommandKind.NOOP and np.isnan(c.leash_x[b, h]):
            c.leash_x[b, h], c.leash_y[b, h] = w.x[b, h], w.y[b, h]


def _assist_switch(sim: Sim, b: int, h: int, cur: int, attacker: int) -> bool:
    """`hit_assist_pick='nearest'`: return True if assist-acquired, unfired ally `h` should
    switch from `cur` to `attacker` (strictly closer center distance, or same-loop equal distance
    breaking by `spawn_order`).
    """
    w, c = sim.world, sim.combat
    if cur == attacker or cur != int(c.assist_target[b, h]):
        return False
    if int(c.last_fire_loop[b, h]) > int(c.assist_loop[b, h]):
        return False
    x, y = w.x[b, h], w.y[b, h]
    d_new = f32(np.hypot(w.x[b, attacker] - x, w.y[b, attacker] - y))
    d_cur = f32(np.hypot(w.x[b, cur] - x, w.y[b, cur] - y))
    if d_new < d_cur:
        return True
    return bool(
        d_new == d_cur
        and sim.config.hit_assist_tie == 'spawn_order'
        and int(c.assist_loop[b, h]) == int(w.game_loop[b])
        and spawn_before(sim, b, attacker, cur)
    )


def spawn_before(sim: Sim, b: int, k: int, u: int) -> bool:
    """`hit_assist_tie='spawn_order'`: return True if `k` precedes `u` in the episode's random
    spawn permutation (`scan_phase` RNG stream at loop 0, lowest slot on tie).
    """
    seed = int(sim.seed[b])
    return (rng.uniform(seed, 'scan_phase', 0, k), k) < (rng.uniform(seed, 'scan_phase', 0, u), u)


def assist_flush(sim: Sim) -> None:
    """`hit_assist_cap > 0` (called by `damage.sc2` after hits resolve, before death cleanup):
    for each victim in `buf.assist_hits`, retain only the first `hit_assist_cap` attackers in
    per-loop update order (`update_order` RNG stream), invoke `_assist` in attacker update order,
    and clear the buffer.
    """
    hits = sim.buf.assist_hits
    cap = sim.config.hit_assist_cap
    for b in np.flatnonzero(hits.any(axis=(1, 2))).tolist():
        seed, loop = int(sim.seed[b]), int(sim.world.game_loop[b])
        key = {
            u: (rng.uniform(seed, 'update_order', loop, u), u)
            for u in np.flatnonzero(hits[b].any(axis=0)).tolist()
        }
        kept: list[tuple[int, int]] = []
        for v in np.flatnonzero(hits[b].any(axis=1)).tolist():
            att = sorted(np.flatnonzero(hits[b, v]).tolist(), key=key.__getitem__)
            kept += [(a, v) for a in att[:cap]]
        kept.sort(key=lambda av: (key[av[0]], av[1]))
        for a, v in kept:
            _assist(sim, b, a, v)
        hits[b] = False


def updates_before(sim: Sim, b: int, k: int, u: int) -> bool:
    """`kill_rescan_order`: return True if `k` updates before `u` on this loop (`update_order`
    counter RNG stream, or `combat.unit_order` when `native_loop_order` is enabled).
    """
    if sim.config.native_loop_order:
        return native_combat.order_before(sim, b, k, u)
    seed, loop = int(sim.seed[b]), int(sim.world.game_loop[b])
    return rng.uniform(seed, 'update_order', loop, k) < rng.uniform(seed, 'update_order', loop, u)


def _kill_lock_holds(sim: Sim, b: int, u: int) -> bool:
    """Return True if `kill_no_retreat` still suppresses retreat for `(b, u)` on this loop."""
    if sim.config.kill_no_retreat_end == 'command':
        return True
    lock, last = int(sim.params.follow_lock[b, u]), int(sim.combat.last_fire_loop[b, u])
    return lock >= 0 and last >= 0 and int(sim.world.game_loop[b]) < last + lock


def on_kill(
    sim: Sim,
    b: int,
    u: int,
    t: int = NO_SLOT,
    *,
    was_amove: bool = False,
    has_rem: bool = False,
) -> None:
    """Death-cleanup hook when `u`'s engaged or ordered target `t` dies: schedules a rescan
    `kill_rescan_delay` loops later (suppressing periodic scans in the meantime).

    Under `kill_rescan_order` (`kill_rescan_delay=0`), instant kills by a unit `k` updated before
    `u` rescan on the same loop, while later or projectile kills defer to the next loop.
    Under `native_backswing`, schedules an immediate rescan once backswing reaches 0 (for
    `ATTACK_MOVE` without `defer_target`, returns to `ATTACK_MOVE` until the next scan beat).
    """
    cfg = sim.config
    if cfg.targeting != 'sc2':
        return
    c = sim.combat
    if cfg.native_backswing:
        if was_amove and not has_rem:
            return
        due = int(sim.world.game_loop[b])
        if cfg.kill_rescan_order and t != NO_SLOT:
            k = int(sim.buf.kill_by[b, t])
            if cfg.native_loop_order:
                if sim.buf.kill_proj[b, t] or (
                    k not in (NO_SLOT, u) and not updates_before(sim, b, k, u)
                ):
                    due += 1
            elif k not in (NO_SLOT, u) and (
                sim.buf.kill_proj[b, t] or not updates_before(sim, b, k, u)
            ):
                due += 1
        c.scan_loop[b, u] = due
        return
    due = int(sim.world.game_loop[b]) + cfg.kill_rescan_delay
    if cfg.kill_rescan_order and t != NO_SLOT:
        k = int(sim.buf.kill_by[b, t])
        if cfg.native_loop_order:
            if sim.buf.kill_proj[b, t] or (
                k not in (NO_SLOT, u) and not updates_before(sim, b, k, u)
            ):
                due += 1
        elif k not in (NO_SLOT, u) and (
            sim.buf.kill_proj[b, t] or not updates_before(sim, b, k, u)
        ):
            due += 1
    if cfg.kill_rescan_lock:
        lock, last = int(sim.params.follow_lock[b, u]), int(c.last_fire_loop[b, u])
        if lock >= 0 and last >= 0:
            due = max(due, last + lock - 1)
    c.scan_loop[b, u] = due


IMPLS = {'basic': basic, 'sc2': sc2}

__all__ = [
    'IMPLS',
    'after_vision',
    'assist_flush',
    'basic',
    'in_weapon_range',
    'kill_scan',
    'late_scan',
    'nearest_attacker',
    'nearest_enemy_in_range',
    'on_command',
    'on_hit',
    'on_kill',
    'sc2',
    'spawn_before',
    'updates_before',
]
