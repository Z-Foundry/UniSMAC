"""Stage 1 command handling: applies `sim.pending` commands to unit orders.

Commands are validated by the backend at the start of `step()` (`check_command`); this stage
only activates them.
"""

from __future__ import annotations

import numpy as np

from unismac.contract import NO_SLOT, CapabilityError, Command, CommandKind, ContractError

from ..sim import WEAPON_AIR, WEAPON_GROUND, Sim, f32, range_radii, weapon_for
from . import abilities, targeting


def check_command(sim: Sim, command: Command) -> None:
    """Validate `command` without mutating state. Raises `CapabilityError` for unsupported
    capabilities and `ContractError` for invalid targets, owners, or dead slots.
    """
    command.validate(sim.batch_size, sim.n_slots)
    ability = command.kind == CommandKind.ABILITY
    if np.any(ability):
        if sim.config.abilities != 'heal':
            raise CapabilityError("pyref only supports ABILITY commands when abilities='heal'")
        bad = sorted({int(a) for a in command.ability[ability]} - {abilities.HEAL_ABILITY})
        if bad:
            raise CapabilityError(
                f'pyref does not support ability {bad} (only {abilities.HEAL_ABILITY} is supported)'
            )
    w = sim.world
    if _commands_ok(sim, command):
        return
    for b, u in zip(*np.nonzero(command.kind != CommandKind.NOOP), strict=True):
        b, u = int(b), int(u)
        kind = CommandKind(int(command.kind[b, u]))
        if not w.alive[b, u]:
            raise ContractError(f'env {b} slot {u} is not alive; cannot issue {kind.name}')
        if not sim.params.agent[b, u]:
            raise ContractError(
                f'env {b} slot {u} does not belong to an AgentPlayer; cannot issue {kind.name}'
            )
        if kind in (CommandKind.ATTACK_UNIT, CommandKind.ABILITY):
            t = int(command.target_slot[b, u])
            if not 0 <= t < sim.n_slots or t == u:
                raise ContractError(f'env {b} slot {u} has invalid {kind.name} target slot {t}')
        elif kind in (CommandKind.MOVE, CommandKind.ATTACK_MOVE):
            if not (np.isfinite(command.target_x[b, u]) and np.isfinite(command.target_y[b, u])):
                raise ContractError(f'env {b} slot {u} has non-finite {kind.name} target point')


def _commands_ok(sim: Sim, command: Command) -> bool:
    """Fast batched pre-check for `check_command`; falls back to per-unit error reporting."""
    w = sim.world
    kind = command.kind
    act = kind != CommandKind.NOOP
    if not act.any():
        return True
    known = np.isin(kind, [int(k) for k in CommandKind])
    bad = act & (~known | ~w.alive | ~sim.params.agent)
    by_slot = (kind == CommandKind.ATTACK_UNIT) | (kind == CommandKind.ABILITY)
    t = command.target_slot.astype(np.int64)
    self_slot = np.arange(sim.n_slots)[None, :]
    bad |= by_slot & ((t < 0) | (t >= sim.n_slots) | (t == self_slot))
    by_point = (kind == CommandKind.MOVE) | (kind == CommandKind.ATTACK_MOVE)
    bad |= by_point & ~(np.isfinite(command.target_x) & np.isfinite(command.target_y))
    return not bad.any()


def _apply_all(sim: Sim, cmd: Command, b: int) -> None:
    """Apply commands for environment `b`; units that died while queued are skipped."""
    w = sim.world
    for u in np.flatnonzero((cmd.kind[b] != CommandKind.NOOP) & w.alive[b]).tolist():
        kind = int(cmd.kind[b, u])
        _apply(
            sim,
            b,
            u,
            kind,
            int(cmd.target_slot[b, u]),
            cmd.target_x[b, u],
            cmd.target_y[b, u],
            int(cmd.ability[b, u]),
        )


def _apply_rows(sim: Sim, cmd: Command, rows: np.ndarray) -> None:
    """Batched `_apply_all` across `rows`. Each command only mutates its own unit (target alive
    and weapon lookups are read-only), so results are bit-identical to per-unit application;
    `ABILITY` (heal) commands still run per unit.
    """
    w = sim.world
    act = rows[:, None] & (cmd.kind != CommandKind.NOOP) & w.alive
    if not act.any():
        return
    b, u = np.nonzero(act)
    kind = cmd.kind[b, u].astype(np.int64)
    abil = kind == CommandKind.ABILITY
    for i in np.flatnonzero(abil).tolist():
        bb, uu = int(b[i]), int(u[i])
        _apply(sim, bb, uu, int(kind[i]), int(cmd.target_slot[bb, uu]), cmd.target_x[bb, uu],
               cmd.target_y[bb, uu], int(cmd.ability[bb, uu]))  # fmt: skip
    target = cmd.target_slot[b, u].astype(np.int64)
    tx, ty = cmd.target_x[b, u], cmd.target_y[b, u]
    skip = abil.copy()
    au = kind == CommandKind.ATTACK_UNIT
    dead = au & ~w.alive[b, target % sim.n_slots]
    if sim.config.dead_target_move and dead.any():
        tt = target % sim.n_slots
        conv = dead & np.isfinite(w.x[b, tt]) & np.isfinite(w.y[b, tt])
        kind = np.where(conv, int(CommandKind.MOVE), kind)
        tx, ty = np.where(conv, w.x[b, tt], tx), np.where(conv, w.y[b, tt], ty)
        au = au & ~conv
        dead = dead & ~conv
    # Reissued ATTACK_UNIT on the same target (checked before overwriting order state).
    same_au = (
        au & (w.order_kind[b, u] == CommandKind.ATTACK_UNIT) & (w.order_target_slot[b, u] == target)
    )
    skip |= dead
    if sim.config.reissue_unit:
        skip |= same_au  # skip entirely without resetting state (see config.reissue_unit)
    if sim.config.amove_needs_weapon:
        skip |= (kind == CommandKind.ATTACK_MOVE) & ~sim.params.has_weapon[b, u].any(axis=1)
    if sim.config.attack_unit_needs_weapon:
        tt = target % sim.n_slots
        wi = np.where(w.is_flying[b, tt], WEAPON_AIR, WEAPON_GROUND)
        skip |= au & ~sim.params.has_weapon[b, u, wi]
    if sim.config.stop_keeps_heal:
        skip |= (
            (kind == CommandKind.STOP)
            & (w.order_kind[b, u] == CommandKind.ABILITY)
            & (w.order_ability[b, u] == abilities.HEAL_ABILITY)
        )
    if sim.config.reissue and sim.config.targeting == 'sc2':
        skip |= _reissue(sim, cmd, b, u, kind)
    if sim.config.native_move_reissue:
        skip |= _same_move(sim, b, u, kind, tx, ty)
    keep = ~skip
    b, u, kind, target, au = b[keep], u[keep], kind[keep], target[keep], au[keep]
    tx, ty = tx[keep], ty[keep]
    if not b.size:
        return
    if sim.config.follow_lock or sim.config.slop_on_arrival:
        brk = ~(same_au[keep] | _same_amove(sim, cmd, b, u, kind))
        sim.combat.follow_cmd[b[brk], u[brk]] = w.game_loop[b[brk]]
    a = sim.ability
    a.channel[b, u] = False
    a.ready_loop[b, u] = -1
    a.full_loops[b, u] = 0
    a.hold_after[b, u] = False
    stop = kind == CommandKind.STOP
    hold = kind == CommandKind.HOLD
    point = ~(stop | hold | au)
    new_kind = np.where(stop, int(CommandKind.NOOP), np.where(hold, int(CommandKind.HOLD), kind))
    w.order_kind[b, u] = new_kind
    w.order_target_slot[b, u] = np.where(au, target, NO_SLOT)
    ox, oy = _clamp_points(sim, b, u, tx, ty)
    w.order_x[b, u] = np.where(point, ox, np.float32(np.nan))
    w.order_y[b, u] = np.where(point, oy, np.float32(np.nan))
    w.order_ability[b, u] = 0
    w.engaged_target_slot[b, u] = NO_SLOT
    m = sim.motion
    m.home_x[b, u] = m.home_y[b, u] = np.nan
    m.cmd_loop[b, u] = w.game_loop[b]
    m.crowd_goal_x[b, u] = m.crowd_goal_y[b, u] = np.nan
    if sim.config.crowd_cover:
        m.crowd_cover_due[b, u] = False
    if sim.config.steering == 'native' or sim.config.collision == 'native':
        _init_native_rows(sim, b, u, stop | hold, au, point, ox, oy)
    if sim.config.move_lag:
        m.lag_loop[b, u] = w.game_loop[b]
        spd = m.speed[b, u].astype(np.float64)
        hd = m.heading[b, u].astype(np.float64)
        pos = spd > 0
        m.lag_vx[b, u] = np.where(pos, (spd * np.cos(hd)).astype(np.float32), np.float32(0))
        m.lag_vy[b, u] = np.where(pos, (spd * np.sin(hd)).astype(np.float32), np.float32(0))
    if sim.config.targeting == 'sc2':
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
        if (
            sim.config.native_assist_on_hit
            or sim.config.native_retarget
            or sim.config.native_backswing
        ):
            c.assist_target[b, u] = NO_SLOT
            c.assist_loop[b, u] = -1
        if sim.config.kill_no_retreat:
            c.kill_idle[b, u] = False
        scan = np.isin(kind, [int(k) for k in targeting._SCAN_ON_COMMAND])
        c.scan_loop[b, u] = np.where(scan, w.game_loop[b], -1)


def _q12_arr(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        return np.where(np.isfinite(a), np.rint(np.asarray(a, np.float64) * 4096.0), 0).astype(
            np.int64
        )


def _clamp_points(
    sim: Sim, b: np.ndarray, u: np.ndarray, tx: np.ndarray, ty: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Clamp point targets; under `native_wall_clearance`, pick playable or full half-open
    bounds by unit position.
    """
    if not sim.config.native_wall_clearance:
        pl = sim.playable[b]
        return np.clip(tx, pl[:, 0], pl[:, 2]), np.clip(ty, pl[:, 1], pl[:, 3])
    w = sim.world
    pl_q = _q12_arr(sim.playable[b])
    px_q, py_q = _q12_arr(w.x[b, u]), _q12_arr(w.y[b, u])
    tx_q, ty_q = _q12_arr(tx), _q12_arr(ty)
    in_pl = (px_q >= pl_q[:, 0]) & (px_q < pl_q[:, 2]) & (py_q >= pl_q[:, 1]) & (py_q < pl_q[:, 3])
    lo_x = np.where(in_pl, pl_q[:, 0], 0)
    lo_y = np.where(in_pl, pl_q[:, 1], 0)
    hi_x = np.where(in_pl, pl_q[:, 2] - 1, pl_q[:, 2] - 1)
    hi_y = np.where(in_pl, pl_q[:, 3] - 1, pl_q[:, 3] - 1)
    if (~in_pl).any():
        for i in np.flatnonzero(~in_pl).tolist():
            info = sim.env_map[int(b[i])]
            if info is not None:
                hi_x[i] = (info.width << 12) - 1
                hi_y[i] = (info.height << 12) - 1
    ox = (np.clip(tx_q, lo_x, hi_x) / 4096.0).astype(np.float32)
    oy = (np.clip(ty_q, lo_y, hi_y) / 4096.0).astype(np.float32)
    return ox, oy


def _same_move(
    sim: Sim, b: np.ndarray, u: np.ndarray, kind: np.ndarray, tx: np.ndarray, ty: np.ndarray
) -> np.ndarray:
    """`native_move_reissue`: skip zero-tolerance (Q12) same-point MOVE reissue while preserving
    order progress.
    """
    w = sim.world
    mv = (kind == CommandKind.MOVE) & (w.order_kind[b, u] == CommandKind.MOVE)
    if not mv.any():
        return mv
    ox, oy = _clamp_points(sim, b, u, tx, ty)
    return (
        mv
        & (_q12_arr(ox) == _q12_arr(w.order_x[b, u]))
        & (_q12_arr(oy) == _q12_arr(w.order_y[b, u]))
    )


def _init_native_rows(
    sim: Sim,
    b: np.ndarray,
    u: np.ndarray,
    stop_hold: np.ndarray,
    au: np.ndarray,
    point: np.ndarray,
    ox: np.ndarray,
    oy: np.ndarray,
) -> None:
    """Initialize fixed-point movement state (`native_state`, `B/k/f/F`, `G`) when a command
    takes effect.
    """
    w, m = sim.world, sim.motion
    if stop_hold.any():
        bs, us = b[stop_hold], u[stop_hold]
        m.native_state[bs, us] = 0
        m.native_has_g[bs, us] = False
        m.native_air_cd[bs, us] = 0
    if au.any():
        ba, ua = b[au], u[au]
        m.native_state[ba, ua] = 8
        m.native_has_g[ba, ua] = False
        m.native_air_cd[ba, ua] = 0
    if point.any():
        bp, up = b[point], u[point]
        px_q, py_q = _q12_arr(w.x[bp, up]), _q12_arr(w.y[bp, up])
        gx_q, gy_q = _q12_arr(ox[point]), _q12_arr(oy[point])
        m.native_gx[bp, up] = gx_q
        m.native_gy[bp, up] = gy_q
        m.native_has_g[bp, up] = True
        ne = (gx_q != px_q) | (gy_q != py_q)
        if ne.any():
            bn, un = bp[ne], up[ne]
            m.native_state[bn, un] = 8
            m.native_bx[bn, un] = px_q[ne]
            m.native_by[bn, un] = py_q[ne]
            m.native_k[bn, un] = 0
            m.native_f[bn, un] = True
            m.native_tol[bn, un] = 0
            m.native_air_cd[bn, un] = 0


def _same_amove(
    sim: Sim, cmd: Command, b: np.ndarray, u: np.ndarray, kind: np.ndarray
) -> np.ndarray:
    """Return mask of units already executing ATTACK_MOVE to the same clamped point."""
    w = sim.world
    ox, oy = _clamp_points(sim, b, u, cmd.target_x[b, u], cmd.target_y[b, u])
    return (
        (kind == CommandKind.ATTACK_MOVE)
        & (w.order_kind[b, u] == CommandKind.ATTACK_MOVE)
        & (w.order_x[b, u] == ox)
        & (w.order_y[b, u] == oy)
    )


def _reissue(sim: Sim, cmd: Command, b: np.ndarray, u: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """`config.reissue`: detect repeated same-point ATTACK_MOVE commands (returning a skip mask)
    and update in place: retain engaged targets that remain attackable within acquire range,
    otherwise clear them (triggering an immediate scan only if `reissue_rescan` is set).
    """
    w, p, cfg = sim.world, sim.params, sim.config
    same = _same_amove(sim, cmd, b, u, kind)
    t = w.engaged_target_slot[b, u].astype(np.int64)
    eng = same & (t != NO_SLOT)
    if eng.any():
        bb, uu, tt = b[eng], u[eng], t[eng]
        wi = np.where(w.is_flying[bb, tt], WEAPON_AIR, WEAPON_GROUND)
        ok = p.has_weapon[bb, uu, wi] & w.alive[bb, tt]
        dx, dy = w.x[bb, tt] - w.x[bb, uu], w.y[bb, tt] - w.y[bb, uu]
        rr = range_radii(sim)
        edge = np.sqrt(dx * dx + dy * dy) - rr[bb, uu] - rr[bb, tt]
        reach = p.weapon_range[bb, uu, wi] + f32(cfg.acquire_margin)
        acq = np.maximum(reach, f32(cfg.acquire_min_range))
        drop = ~(ok & (edge <= acq))
        if drop.any():
            db, du = bb[drop], uu[drop]
            w.engaged_target_slot[db, du] = NO_SLOT
            c = sim.combat
            c.rt_due[db, du], c.rt_lock[db, du] = -1, NO_SLOT
            if cfg.reissue_rescan:  # immediately rescan cleared units this loop
                c.scan_loop[db, du] = w.game_loop[db]
    if cfg.reissue_ends_wait:  # immediately rescan units waiting after a kill
        c = sim.combat
        loop = w.game_loop[b]
        sl = c.scan_loop[b, u]
        wait = same & (t == NO_SLOT) & (sl >= 0) & (loop < sl)
        if wait.any():
            c.scan_loop[b[wait], u[wait]] = loop[wait]
    return same


def _clear(cmd: Command, b: int | slice) -> None:
    cmd.kind[b] = CommandKind.NOOP
    cmd.target_slot[b] = NO_SLOT
    cmd.target_x[b] = np.nan
    cmd.target_y[b] = np.nan
    cmd.ability[b] = 0


def immediate(sim: Sim) -> None:
    """Apply commands on the loop received, producing movement within the same step."""
    _apply_rows(sim, sim.pending, np.ones(sim.batch_size, np.bool_))
    _clear(sim.pending, slice(None))


def delayed(sim: Sim) -> None:
    """Delay command activation by `k = config.command_delay` loops.

    If `step(cmd, n)` begins at `game_loop = L`, `cmd` activates in the command stage of loop
    `L + k` and produces movement on that loop (observed after `k + 1` loops). `k = 0` is
    bit-identical to `immediate`.

    For lockstep two-player SC2 alignment (where orders/facing/attack lag 1 loop and movement
    lags 2 loops), `command_delay=1` with `move_lag=1` aligns both orders and displacement
    (holding the activation loop's displacement at the prior loop's velocity; see kinematics).

    Commands issued during the delay queue per loop and activate in order (later commands
    overwrite earlier ones). Dead units are skipped at activation, and ATTACK_UNIT on a dead
    target is ignored (or converted to MOVE under `dead_target_move`).
    """
    n = len(sim.queue)
    k = sim.config.command_delay
    p = sim.pending
    # Each env accesses only its own row; grouping by game_loop is identical to per-env loops.
    loops = sim.world.game_loop
    mask = p.kind != CommandKind.NOOP
    for loop in np.unique(loops).tolist():
        rows = loops == loop
        due = sim.queue[(loop + k) % n]
        m = mask & rows[:, None]
        if m.any():
            for name in ('kind', 'target_slot', 'target_x', 'target_y', 'ability'):
                getattr(due, name)[m] = getattr(p, name)[m]
        now = sim.queue[loop % n]
        _apply_rows(sim, now, rows)
        _clear(now, rows)
    _clear(p, slice(None))


def _apply(
    sim: Sim,
    b: int,
    u: int,
    kind: int,
    target: int,
    tx: np.float32,
    ty: np.float32,
    ability: int = 0,
) -> None:
    w = sim.world
    if kind == CommandKind.ATTACK_UNIT and not w.alive[b, target]:
        tt = target % sim.n_slots
        if not (
            sim.config.dead_target_move and np.isfinite(w.x[b, tt]) and np.isfinite(w.y[b, tt])
        ):
            return  # dead or empty target slot: ignore command, matching SC2
        # dead_target_move: convert to MOVE toward the target's death position
        kind, target, tx, ty = int(CommandKind.MOVE), NO_SLOT, w.x[b, tt], w.y[b, tt]
    if (
        kind == CommandKind.ATTACK_UNIT
        and sim.config.attack_unit_needs_weapon
        and weapon_for(sim, b, u, target) < 0
    ):
        return  # attack_unit_needs_weapon: ignore if unit lacks weapon for target
    same_au = (
        kind == CommandKind.ATTACK_UNIT
        and int(w.order_kind[b, u]) == CommandKind.ATTACK_UNIT
        and int(w.order_target_slot[b, u]) == target
    )
    if same_au and sim.config.reissue_unit:
        return  # reissue_unit: preserve state when reissuing ATTACK_UNIT on same target
    keeps_follow = same_au
    if kind == CommandKind.ATTACK_MOVE and (sim.config.follow_lock or sim.config.slop_on_arrival):
        ox_f, oy_f = _clamp_points(
            sim,
            np.array([b]),
            np.array([u]),
            np.array([tx], np.float32),
            np.array([ty], np.float32),
        )
        keeps_follow = (
            int(w.order_kind[b, u]) == CommandKind.ATTACK_MOVE
            and w.order_x[b, u] == ox_f[0]
            and w.order_y[b, u] == oy_f[0]
        )
    if (
        kind == CommandKind.ATTACK_MOVE
        and sim.config.amove_needs_weapon
        and not sim.params.has_weapon[b, u].any()
    ):
        return  # amove_needs_weapon: unarmed units ignore ATTACK_MOVE
    if kind == CommandKind.ABILITY:
        # check_command guarantees heal ability; ignore invalid target/energy, matching SC2.
        if ability == abilities.HEAL_ABILITY and abilities.can_heal(sim, b, u, target):
            abilities.start_order(sim, b, u, target, int(w.game_loop[b]), hold_after=False)
            w.engaged_target_slot[b, u] = NO_SLOT
            sim.combat.engaged_transient[b, u] = False
            sim.combat.leash_return[b, u] = False
            sim.combat.defer_target[b, u] = NO_SLOT
            sim.combat.backswing_left[b, u] = 0
            if sim.config.kill_no_retreat:
                sim.combat.kill_idle[b, u] = False
            sim.motion.home_x[b, u] = sim.motion.home_y[b, u] = np.nan
            sim.motion.cmd_loop[b, u] = sim.world.game_loop[b]
            sim.motion.crowd_goal_x[b, u] = sim.motion.crowd_goal_y[b, u] = np.nan
            if sim.config.crowd_cover:
                sim.motion.crowd_cover_due[b, u] = False
            if sim.config.steering == 'native' or sim.config.collision == 'native':
                sim.motion.native_state[b, u] = 8
                sim.motion.native_air_cd[b, u] = 0
            _mark_lag(sim, b, u)
            _follow_break(sim, b, u)
        return
    if (
        kind == CommandKind.STOP
        and sim.config.stop_keeps_heal
        and int(w.order_kind[b, u]) == CommandKind.ABILITY
        and int(w.order_ability[b, u]) == abilities.HEAL_ABILITY
    ):
        return  # stop_keeps_heal: keep channeling heal
    if kind == CommandKind.MOVE and sim.config.native_move_reissue:
        b_arr, u_arr = np.array([b]), np.array([u])
        if _same_move(
            sim,
            b_arr,
            u_arr,
            np.array([kind]),
            np.array([tx], np.float32),
            np.array([ty], np.float32),
        )[0]:
            return
    abilities.reset_state(sim, b, u)
    if kind in (CommandKind.STOP, CommandKind.HOLD):
        w.order_kind[b, u] = CommandKind.NOOP if kind == CommandKind.STOP else CommandKind.HOLD
        w.order_target_slot[b, u] = NO_SLOT
        w.order_x[b, u] = w.order_y[b, u] = np.nan
        ox_s = oy_s = f32(np.nan)
    elif kind == CommandKind.ATTACK_UNIT:
        w.order_kind[b, u] = kind
        w.order_target_slot[b, u] = target
        w.order_x[b, u] = w.order_y[b, u] = np.nan
        ox_s = oy_s = f32(np.nan)
    else:  # MOVE / ATTACK_MOVE: clamp target point into playable area
        ox_a, oy_a = _clamp_points(
            sim,
            np.array([b]),
            np.array([u]),
            np.array([tx], np.float32),
            np.array([ty], np.float32),
        )
        ox_s, oy_s = f32(ox_a[0]), f32(oy_a[0])
        w.order_kind[b, u] = kind
        w.order_target_slot[b, u] = NO_SLOT
        w.order_x[b, u] = ox_s
        w.order_y[b, u] = oy_s
    w.order_ability[b, u] = 0
    w.engaged_target_slot[b, u] = NO_SLOT  # new command drops engagement; targeting re-evaluates
    sim.motion.home_x[b, u] = sim.motion.home_y[b, u] = np.nan  # cancel idle return-home
    sim.motion.cmd_loop[b, u] = sim.world.game_loop[b]
    # Drop previous MOVE crowd goal (re-recorded by collision._crowd_arrival on MOVE).
    sim.motion.crowd_goal_x[b, u] = sim.motion.crowd_goal_y[b, u] = np.nan
    if sim.config.crowd_cover:
        sim.motion.crowd_cover_due[b, u] = False
    if sim.config.steering == 'native' or sim.config.collision == 'native':
        _init_native_rows(
            sim,
            np.array([b]),
            np.array([u]),
            np.array([kind in (CommandKind.STOP, CommandKind.HOLD)]),
            np.array([kind == CommandKind.ATTACK_UNIT]),
            np.array([kind in (CommandKind.MOVE, CommandKind.ATTACK_MOVE)]),
            np.array([ox_s], np.float32),
            np.array([oy_s], np.float32),
        )
    _mark_lag(sim, b, u)
    if not keeps_follow:
        _follow_break(sim, b, u)
    targeting.on_command(sim, b, u, kind)


def _follow_break(sim: Sim, b: int, u: int) -> None:
    """`follow_lock`: record a command activating this loop that breaks follow lock
    (`combat.follow_cmd`; see `targeting._follow_lock`). Same-target `ATTACK_UNIT` and
    same-point `ATTACK_MOVE` reissues do not break the lock.
    """
    if sim.config.follow_lock or sim.config.slop_on_arrival:
        sim.combat.follow_cmd[b, u] = int(sim.world.game_loop[b])


def _mark_lag(sim: Sim, b: int, u: int) -> None:
    """`move_lag=1`: mark command activation loop so kinematics holds prior velocity this loop."""
    if not sim.config.move_lag:
        return
    m = sim.motion
    m.lag_loop[b, u] = int(sim.world.game_loop[b])
    spd = float(m.speed[b, u])
    m.lag_vx[b, u] = f32(spd * np.cos(float(m.heading[b, u]))) if spd > 0 else 0
    m.lag_vy[b, u] = f32(spd * np.sin(float(m.heading[b, u]))) if spd > 0 else 0


IMPLS = {'immediate': immediate, 'delayed': delayed}
