"""Stage 7 collision resolution: resolves unit-unit and unit-terrain overlaps."""

from __future__ import annotations

import math

import numpy as np

from unismac.contract import NO_SLOT, CommandKind

from .. import navmesh, terrain
from ..sim import Sim, f32
from . import pathing


def none(sim: Sim) -> None:
    """No collision resolution; units may overlap freely (`collision='none'`)."""


def hard(sim: Sim) -> None:
    """Hard-disc collision (`collision='hard'`), applied to alive ground units only.

    Contact distance = sum of radii - `contact_slack`. Each loop runs `collision_iterations`
    iterations of:

    1. Ordered Gauss-Seidel pairwise pass over `(u, t)` (`u < t`): push overlapping discs apart
       along their center vector (subsequent pairs read updated positions). Equal movers/idles
       split 50/50 (or per `mover_push_share` for two movers); mover-vs-idle puts
       `idle_push_share` on the idle unit. Coincident centers push `u` toward `-x`, `t` toward `+x`.
    2. Project ground units whose centers landed in impassable cells back onto the nearest
       passable cell (`terrain.project_to_passable`).

    Finite iteration counts may leave small overlaps in crowds, but terrain projection runs last
    so unit centers never rest in impassable cells.

    Options:
    * `pin_engaged` / `pin_after_fire` / `pin_lag` / `pin_pair_share`: immovable stationary
      HOLD/engaged units (`pinned_mask`) push 100% of overlap onto movable units without recording
      an idle push; two pinned units split 50/50 or by `'younger'` streak start (`pin_since`).
    * `push_clearance`: after cell projection, push units closer than navmesh `rho` back to `rho`
      (`_clear`). `arrival_clears_waypoint` makes arriving units count as idle.
    * `mover_push_share`: `'speed'` (`v^p` weighted), `'approach'` (positive closing speed),
      `'approach_same_half'` / `'approach_lone_half'` (50/50 for equal-speed or lone equal-speed
      pairs), or `'axis'` (independent per-axis closing split weighted by speed power).
    * `mover_speed_cap`: clamp total per-loop displacement of moving units to
      `max(params.speed, kinematics_step)` (`_speed_cap`).
    * `crowd_arrival` / `crowd_cover`: clear MOVE orders when crowded around a shared goal or
      when a contacting ally covers the destination.
    * `idle_return` (`'legacy'` / `'diameter'`): issue a return-home MOVE order for displaced
      never-commanded idle units once thresholds and delay are met.

    Implementation: bit-identical to sequential `_sweep_exact`, batching disjoint pairs into
    dependency rounds (`_sweep`) and growing the candidate set (`_sweep_grow`) if cumulative
    displacements approach `SWEEP_MARGIN`.
    """
    w, cfg = sim.world, sim.config
    consts = _Consts(cfg)
    ground_m = w.alive & ~w.is_flying
    moving_m = ~np.isnan(sim.buf.waypoint_x)
    pin_now = _pinned_now(sim) if cfg.pin_engaged else None
    if pin_now is not None:
        consts.pin = pin_now & sim.motion.pinned_prev if cfg.pin_lag else pin_now
        if cfg.pin_pair_share == 'younger':
            m = sim.motion
            start = np.where(m.pin_since >= 0, m.pin_since, w.game_loop[:, None])
            m.pin_since[:] = np.where(pin_now, start, -1)
            consts.since = m.pin_since
    if cfg.mover_push_share == 'speed':
        consts.mover = mover_weight(sim)
    elif cfg.mover_push_share in ('approach', 'approach_same_half', 'approach_lone_half'):
        consts.approach = _Approach(sim)
    elif cfg.mover_push_share == 'axis':
        consts.axis = _Axis(sim)
    pre = (w.x.copy(), w.y.copy()) if cfg.mover_speed_cap else None
    for _ in range(cfg.collision_iterations):
        _sweep(sim, ground_m, moving_m, consts)
        _project(sim, ground_m)
    if pre is not None:
        _speed_cap(sim, ground_m & moving_m, *pre)
    after_contact(sim, ground_m, moving_m, pin_now)


def after_contact(
    sim: Sim, ground_m: np.ndarray, moving_m: np.ndarray, pin_now: np.ndarray | None
) -> None:
    """Shared post-contact hooks (used by both `hard` and `native.contact`): records `pin_lag`
    state, evaluates `crowd_arrival` and `crowd_cover`, and issues `idle_return` MOVE orders.
    """
    w, cfg = sim.world, sim.config
    if cfg.pin_lag and pin_now is not None:
        sim.motion.pinned_prev[:] = pin_now
    if cfg.crowd_arrival and not cfg.native_stop:
        # Envs without ground MOVE units simply clear crowd counters.
        movers = ground_m & (w.order_kind == CommandKind.MOVE)
        _crowd_arrival_many(sim, np.flatnonzero(movers.any(axis=1)), ground_m)
        idle_env = ~movers.any(axis=1)[:, None] & ground_m
        sim.motion.crowd_dist[idle_env] = np.nan
        sim.motion.crowd_blocked[idle_env] = 0
    if cfg.crowd_cover and not cfg.native_stop:
        _crowd_cover(sim, ground_m)
    if cfg.idle_return:
        m = sim.motion
        if cfg.idle_return_rule == 'diameter':
            homed = ground_m & ~np.isnan(m.home_x) & (m.pushed_loop >= 0)
            away = np.hypot(w.x - m.home_x, w.y - m.home_y)
            m.pushed_max[homed] = np.maximum(m.pushed_max, away)[homed]
        maybe = ground_m & ~moving_m & ~np.isnan(m.home_x) & (m.pushed_loop >= 0)
        maybe &= (w.order_kind == CommandKind.NOOP) & (m.pushed_loop != w.game_loop[:, None])
        for b, u in zip(*np.nonzero(maybe), strict=True):
            _maybe_return(sim, int(b), int(u), int(w.game_loop[b]))


# Candidate-pair margin (tiles), float guard, and scalar fast-path threshold.
SWEEP_MARGIN = 0.5
SWEEP_EPS = 1e-3
SWEEP_SMALL = 48
# Extra margin (tiles) added when growing the candidate set on fallback rescan.
GROW_MARGIN = 0.25
# `approach_same_half`: relative desired-speed difference threshold for equal-speed 50/50 split
# (distinct ground unit types differ by >= 10% at full speed, so 2% separates distinct speeds).
SAME_SPEED_RTOL = 0.02
# `approach_lone_half`: overlap > -CHAIN_EPS counts two ground units as touching in a chain.
CHAIN_EPS = 0.01


class _Consts:
    def __init__(self, cfg) -> None:
        self.slack = f32(cfg.contact_slack)
        self.share_idle = f32(cfg.idle_push_share)
        self.share_mover = f32(1) - self.share_idle
        self.half = f32(0.5)
        self.pin: np.ndarray | None = None  # [B, U] pinned mask (`pin_engaged`), None if off
        # [B, U] start loop of continuous pinned streak (`pin_pair_share='younger'`), else None
        self.since: np.ndarray | None = None
        # [B, U] float64 mover-vs-mover weight v^p (`mover_push_share='speed'`), else None
        self.mover: np.ndarray | None = None
        # Desired velocity & cap for `approach` / `approach_*_half`, else None
        self.approach: _Approach | None = None
        # Desired velocity, speed weight & cap for `axis`, else None
        self.axis: _Axis | None = None


class _Approach:
    """State for `mover_push_share='approach'`, `'approach_same_half'`, and `'approach_lone_half'`:
    float64 desired velocity, share cap, optional speed magnitude (`same_half`/`lone_half`), and
    `odd` mask (`lone_half`: touching any alive ground unit with a different desired speed).
    """

    def __init__(self, sim: Sim) -> None:
        self.vx = sim.buf.vel_x.astype(np.float64)
        self.vy = sim.buf.vel_y.astype(np.float64)
        self.cap = float(sim.config.mover_push_cap)
        share = sim.config.mover_push_share
        half = share in ('approach_same_half', 'approach_lone_half')
        self.speed = np.hypot(self.vx, self.vy) if half else None
        self.odd: np.ndarray | None = None
        if share == 'approach_lone_half':
            assert self.speed is not None
            touch = _touching(sim, sim.world.alive & ~sim.world.is_flying)
            touch &= ~_same_speed(self.speed[:, :, None], self.speed[:, None, :])
            self.odd = touch.any(axis=2)


def _same_speed(a, c):
    """Return True if speeds `a` and `c` match within `SAME_SPEED_RTOL * max(a, c)` (float64)."""
    return np.abs(a - c) <= SAME_SPEED_RTOL * np.maximum(a, c)


def _touching(sim: Sim, ground_m: np.ndarray) -> np.ndarray:
    """Return `[B, U, U]` mask of distinct alive ground pairs with overlap `> -CHAIN_EPS`."""
    w = sim.world
    x, y, r = (a.astype(np.float64) for a in (w.x, w.y, w.radius))
    with np.errstate(invalid='ignore'):
        d = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        ov = r[:, :, None] + r[:, None, :] - float(sim.config.contact_slack) - d
        out = ov > -CHAIN_EPS
    n = w.x.shape[1]
    return out & ground_m[:, :, None] & ground_m[:, None, :] & ~np.eye(n, dtype=np.bool_)


def _approach_split(ap: _Approach, b, u, t, nx, ny):
    """Compute push shares `(u, t)` by positive closing velocity along contact normal `n` (`u->t`),
    clamped to `[1 - cap, cap]` (50/50 if neither approaches). Overrides to 50/50 for equal-speed
    pairs (`approach_same_half`) or lone equal-speed pairs (`approach_lone_half`). Evaluated in
    float64 arithmetic and cast to float32 for scalar/array bit-parity.
    """
    nx = np.asarray(nx, np.float64)
    ny = np.asarray(ny, np.float64)
    au = np.maximum(ap.vx[b, u] * nx + ap.vy[b, u] * ny, 0.0)
    at = np.maximum(-(ap.vx[b, t] * nx + ap.vy[b, t] * ny), 0.0)
    tot = au + at
    pos = tot > 0
    su = np.where(pos, au / np.where(pos, tot, 1.0), 0.5)
    su = np.minimum(np.maximum(su, 1.0 - ap.cap), ap.cap)
    if ap.speed is not None:
        same = _same_speed(ap.speed[b, u], ap.speed[b, t])
        if ap.odd is not None:
            same = same & ~ap.odd[b, u] & ~ap.odd[b, t]
        su = np.where(same, 0.5, su)
    return su.astype(np.float32), (1.0 - su).astype(np.float32)


class _Axis:
    """State for `mover_push_share='axis'`: float64 desired velocity, speed weight
    `|buf.vel|^mover_push_power`, and per-axis share cap `mover_push_cap`.
    """

    def __init__(self, sim: Sim) -> None:
        self.vx = sim.buf.vel_x.astype(np.float64)
        self.vy = sim.buf.vel_y.astype(np.float64)
        self.w = np.hypot(self.vx, self.vy) ** float(sim.config.mover_push_power)
        self.cap = float(sim.config.mover_push_cap)


def _axis_part(cap: float, vu, vt, n, wu, wt):
    """Per-axis share for `u`: split positive closing components along `sign(n)`, clamp to
    `[1 - cap, cap]`, and reweight by `s*w_u / (s*w_u + (1 - s)*w_t)` when denominator > 0.
    """
    sg = np.where(n >= 0, 1.0, -1.0)
    au = np.maximum(sg * vu, 0.0)
    at = np.maximum(-sg * vt, 0.0)
    tot = au + at
    pos = tot > 0
    s = np.where(pos, au / np.where(pos, tot, 1.0), 0.5)
    s = np.minimum(np.maximum(s, 1.0 - cap), cap)
    num = s * wu
    den = num + (1.0 - s) * wt
    ok = den > 0
    return np.where(ok, num / np.where(ok, den, 1.0), s)


def _axis_split(ax: _Axis, b, u, t, nx, ny):
    """Compute world-axis push shares `(u, t)` (`mover_push_share='axis'`): evaluate `_axis_part`
    along x and y, combine by `(nx^2 * sx + ny^2 * sy) / (nx^2 + ny^2)`, and cast to float32.
    Invariant under 90-deg rotations, mirrors, and slot swaps.
    """
    nx = np.asarray(nx, np.float64)
    ny = np.asarray(ny, np.float64)
    wu, wt = ax.w[b, u], ax.w[b, t]
    sx = _axis_part(ax.cap, ax.vx[b, u], ax.vx[b, t], nx, wu, wt)
    sy = _axis_part(ax.cap, ax.vy[b, u], ax.vy[b, t], ny, wu, wt)
    qx, qy = nx * nx, ny * ny
    su = (qx * sx + qy * sy) / (qx + qy)
    return su.astype(np.float32), (1.0 - su).astype(np.float32)


def mover_weight(sim: Sim) -> np.ndarray:
    """Compute `v^mover_push_power` weight for `mover_push_share='speed'` (float64)."""
    v = np.hypot(sim.buf.vel_x, sim.buf.vel_y).astype(np.float64)
    return v ** float(sim.config.mover_push_power)


def _mover_split(wu, wt):
    """Compute `w_u / (w_u + w_t)` and `w_t / (w_u + w_t)` in float64 -> float32 (50/50 if 0)."""
    tot = wu + wt
    pos = tot > 0
    safe = np.where(pos, tot, 1.0)
    su = np.where(pos, wu / safe, 0.5).astype(np.float32)
    st = np.where(pos, wt / safe, 0.5).astype(np.float32)
    return su, st


def pinned_mask(sim: Sim) -> np.ndarray:
    """Return `[B, U]` mask of pinned (`pin_engaged`) units: alive stationary ground units on
    HOLD or with an engaged target (requiring prior fire on that target under `pin_after_fire`,
    and previous-loop pinned status under `pin_lag`).
    """
    now = _pinned_now(sim)
    return now & sim.motion.pinned_prev if sim.config.pin_lag else now


def _pinned_now(sim: Sim) -> np.ndarray:
    """Current-loop pinned predicate (ignoring `pin_lag`)."""
    w = sim.world
    still = w.alive & ~w.is_flying & np.isnan(sim.buf.waypoint_x)
    engaged = w.engaged_target_slot != NO_SLOT
    if sim.config.pin_after_fire:
        engaged &= sim.combat.last_fire_target == w.engaged_target_slot
    return still & ((w.order_kind == CommandKind.HOLD) | engaged)


def _push(sim: Sim, b: int, u: int, t: int, mu: bool, mt: bool, k: _Consts):
    """Resolve one overlapping pair `(u, t)` in sequential sweep; return `(idle, mover, du, dt)`
    or None if separated (`idle = mover = -1` when both moving or both idle).
    """
    w = sim.world
    dx = w.x[b, t] - w.x[b, u]
    dy = w.y[b, t] - w.y[b, u]
    dist = f32(np.sqrt(dx * dx + dy * dy))
    overlap = f32(w.radius[b, u] + w.radius[b, t] - k.slack - dist)
    if overlap <= 0:
        return None
    if dist > 0:
        nx, nyy = dx / dist, dy / dist
    else:
        nx, nyy = f32(1), f32(0)
    idle = mover = -1
    if k.pin is not None and k.pin[b, u] != k.pin[b, t]:
        pu, pt = (f32(0), overlap) if k.pin[b, u] else (overlap, f32(0))
    elif (
        k.since is not None
        and k.pin is not None
        and k.pin[b, u]
        and (k.since[b, u] != k.since[b, t])
    ):  # both pinned with different streak starts: younger takes full overlap
        pu, pt = (overlap, f32(0)) if k.since[b, u] > k.since[b, t] else (f32(0), overlap)
    elif mu and mt and k.mover is not None:
        su, st = _mover_split(k.mover[b, u], k.mover[b, t])
        pu, pt = overlap * su[()], overlap * st[()]
    elif mu and mt and k.approach is not None:
        su, st = _approach_split(k.approach, b, u, t, nx, nyy)
        pu, pt = overlap * su[()], overlap * st[()]
    elif mu and mt and k.axis is not None:
        su, st = _axis_split(k.axis, b, u, t, nx, nyy)
        pu, pt = overlap * su[()], overlap * st[()]
    elif mu == mt:
        pu = pt = overlap * k.half
    elif mu:
        pu, pt = overlap * k.share_mover, overlap * k.share_idle
        idle, mover = t, u
    else:
        pu, pt = overlap * k.share_idle, overlap * k.share_mover
        idle, mover = u, t
    w.x[b, u] -= nx * pu
    w.y[b, u] -= nyy * pu
    w.x[b, t] += nx * pt
    w.y[b, t] += nyy * pt
    du = abs(float(nx * pu)) + abs(float(nyy * pu))
    dt = abs(float(nx * pt)) + abs(float(nyy * pt))
    return idle, mover, du, dt


def _sweep_exact(sim: Sim, b: int, ground: list[int], moving: np.ndarray, k: _Consts) -> None:
    """Reference sequential Gauss-Seidel sweep over `u < t` for one env and one iteration."""
    loop = int(sim.world.game_loop[b])
    for i, u in enumerate(ground):
        for t in ground[i + 1 :]:
            r = _push(sim, b, u, t, bool(moving[u]), bool(moving[t]), k)
            if r is not None and r[0] >= 0:
                _mark_pushed(sim, b, r[0], r[1], loop)


def _sweep(sim: Sim, ground_m: np.ndarray, moving_m: np.ndarray, k: _Consts) -> None:
    """Batched sweep bit-identical to per-env `_sweep_exact`.

    Pairs starting with `overlap <= -SWEEP_MARGIN` cannot collide unless cumulative displacements
    exceed the margin; candidate pairs run in dependency-preserving parallel rounds.
    """
    w = sim.world
    bsz, n = w.x.shape
    x0, y0 = w.x.copy(), w.y.copy()
    with np.errstate(invalid='ignore'):
        dx = w.x[:, None, :] - w.x[:, :, None]
        dy = w.y[:, None, :] - w.y[:, :, None]
        dist = np.sqrt(dx * dx + dy * dy)
        ov0 = w.radius[:, :, None] + w.radius[:, None, :] - k.slack - dist
    gpair = ground_m[:, :, None] & ground_m[:, None, :] & np.triu(np.ones((n, n), np.bool_), 1)
    with np.errstate(invalid='ignore'):
        cand = gpair & (ov0 > -SWEEP_MARGIN)
    pb, pu, pt = np.nonzero(cand)
    disp = np.zeros((bsz, n))
    pushes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if len(pb) <= SWEEP_SMALL:
        pl_b, pl_i, pl_m = [], [], []
        for b, u, t in zip(pb.tolist(), pu.tolist(), pt.tolist(), strict=True):
            r = _push(sim, b, u, t, bool(moving_m[b, u]), bool(moving_m[b, t]), k)
            if r is None:
                continue
            disp[b, u] += r[2] + SWEEP_EPS / 16
            disp[b, t] += r[3] + SWEEP_EPS / 16
            if r[0] >= 0:
                pl_b.append(b), pl_i.append(r[0]), pl_m.append(r[1])
        if pl_b:
            pushes.append((np.array(pl_b), np.array(pl_i), np.array(pl_m)))
    else:
        _rounds(sim, pb, pu, pt, moving_m, k, disp, pushes)
    # Check whether any skipped pair could have collided given cumulative displacements.
    fallback: list[int] = []
    grow = cand.copy()
    top = np.sort(disp, axis=1)[:, -2:].sum(axis=1) if n >= 2 else np.zeros(bsz)
    for b in np.flatnonzero(top >= SWEEP_MARGIN - SWEEP_EPS).tolist():
        with np.errstate(invalid='ignore'):
            slack = -ov0[b].astype(np.float64) - disp[b][:, None] - disp[b][None, :]
        if (gpair[b] & ~cand[b] & (slack <= SWEEP_EPS)).any():
            fallback.append(b)
            # Grow candidate superset with GROW_MARGIN buffer for fallback rescan.
            grow[b] |= gpair[b] & (slack <= SWEEP_EPS + GROW_MARGIN)
    for pb_, pi_, pm_ in pushes:
        keep = ~np.isin(pb_, fallback)
        _mark_pushed_many(sim, pb_[keep], pi_[keep], pm_[keep])
    if fallback:
        _sweep_grow(sim, np.array(fallback), grow, gpair, ov0, x0, y0, moving_m, k)


def _sweep_grow(sim, rows, cand, gpair, ov0, x0, y0, moving_m, k: _Consts) -> None:
    """Rescan environments whose non-candidate slack dropped below `SWEEP_EPS`, monotonically
    expanding `cand` from `(x0, y0)` until every skipped pair is certified non-colliding.
    Bit-identical to `_sweep_exact` while preserving batched round execution.
    """
    w = sim.world
    bsz, n = w.x.shape
    c = cand.copy()
    while rows.size:
        w.x[rows], w.y[rows] = x0[rows], y0[rows]
        sb, pu, pt = np.nonzero(c[rows])
        disp = np.zeros((bsz, n))
        pushes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        _rounds(sim, rows[sb], pu, pt, moving_m, k, disp, pushes)
        d = disp[rows]
        with np.errstate(invalid='ignore'):
            slack = -ov0[rows].astype(np.float64) - d[:, :, None] - d[:, None, :]
        more = gpair[rows] & ~c[rows] & (slack <= SWEEP_EPS)
        bad = more.any(axis=(1, 2))
        redo = rows[bad]
        for pb_, pi_, pm_ in pushes:
            keep = ~np.isin(pb_, redo)
            _mark_pushed_many(sim, pb_[keep], pi_[keep], pm_[keep])
        c[redo] |= more[bad]
        rows = redo


def _schedule(pb: np.ndarray, pu: np.ndarray, pt: np.ndarray, n: int) -> np.ndarray:
    """Assign each lexicographically ordered pair `(b, u, t)` to `1 + max` prior round of `u, t`.

    Pairs with equal `u + t` within an env are vertex-disjoint (`u < t`) and depend only on
    smaller `u + t`, so grouping by `u + t` allows vectorized recurrence.
    """
    gu, gt = pb * n + pu, pb * n + pt
    last = np.full(int(pb.max(initial=0) + 1) * n, -1, np.int64)
    rnd = np.empty(len(pb), np.int64)
    s = pu + pt
    order = np.argsort(s, kind='stable')
    for g in np.split(order, np.flatnonzero(np.diff(s[order])) + 1):
        r = np.maximum(last[gu[g]], last[gt[g]]) + 1
        rnd[g] = r
        last[gu[g]] = r
        last[gt[g]] = r
    return rnd


def _rounds(sim, pb, pu, pt, moving_m, k: _Consts, disp, pushes) -> None:
    """Resolve candidate pairs in vertex-disjoint parallel rounds."""
    w = sim.world
    rnd = _schedule(pb, pu, pt, w.x.shape[1])
    order = np.argsort(rnd, kind='stable')
    cuts = np.flatnonzero(np.diff(rnd[order])) + 1
    for e in np.split(order, cuts):
        b, u, t = pb[e], pu[e], pt[e]
        dx = w.x[b, t] - w.x[b, u]
        dy = w.y[b, t] - w.y[b, u]
        dist = np.sqrt(dx * dx + dy * dy)
        ov = w.radius[b, u] + w.radius[b, t] - k.slack - dist
        hit = ov > 0
        if not hit.any():
            continue
        b, u, t, dx, dy, dist, ov = (a[hit] for a in (b, u, t, dx, dy, dist, ov))
        pos = dist > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            nx = np.where(pos, dx / dist, f32(1))
            ny = np.where(pos, dy / dist, f32(0))
        mu, mt = moving_m[b, u], moving_m[b, t]
        same = mu == mt
        ph = ov * k.half
        p_u = np.where(same, ph, np.where(mu, ov * k.share_mover, ov * k.share_idle))
        p_t = np.where(same, ph, np.where(mu, ov * k.share_idle, ov * k.share_mover))
        if k.mover is not None:
            both = mu & mt
            if both.any():
                su, st = _mover_split(k.mover[b, u], k.mover[b, t])
                p_u = np.where(both, ov * su, p_u)
                p_t = np.where(both, ov * st, p_t)
        elif k.approach is not None:
            both = mu & mt
            if both.any():
                su, st = _approach_split(k.approach, b, u, t, nx, ny)
                p_u = np.where(both, ov * su, p_u)
                p_t = np.where(both, ov * st, p_t)
        elif k.axis is not None:
            both = mu & mt
            if both.any():
                su, st = _axis_split(k.axis, b, u, t, nx, ny)
                p_u = np.where(both, ov * su, p_u)
                p_t = np.where(both, ov * st, p_t)
        one_pin = None
        if k.pin is not None:
            qu, qt = k.pin[b, u], k.pin[b, t]
            one_pin = qu != qt
            p_u = np.where(one_pin, np.where(qu, f32(0), ov), p_u)
            p_t = np.where(one_pin, np.where(qt, f32(0), ov), p_t)
            if k.since is not None:  # pin_pair_share='younger': younger pinned unit yields all
                both = qu & qt
                yu = both & (k.since[b, u] > k.since[b, t])
                yt = both & (k.since[b, t] > k.since[b, u])
                p_u = np.where(yu, ov, np.where(yt, f32(0), p_u))
                p_t = np.where(yt, ov, np.where(yu, f32(0), p_t))
        ux, uy, tx, ty = nx * p_u, ny * p_u, nx * p_t, ny * p_t
        w.x[b, u] -= ux
        w.y[b, u] -= uy
        w.x[b, t] += tx
        w.y[b, t] += ty
        disp[b, u] += np.abs(ux.astype(np.float64)) + np.abs(uy) + SWEEP_EPS / 16
        disp[b, t] += np.abs(tx.astype(np.float64)) + np.abs(ty) + SWEEP_EPS / 16
        a = mu & ~mt  # u moving, t idle
        c = ~mu & mt
        if one_pin is not None:  # pinned pairs do not record idle push
            a &= ~one_pin
            c &= ~one_pin
        if a.any() or c.any():
            pushes.append(
                (
                    np.concatenate([b[a], b[c]]),
                    np.concatenate([t[a], u[c]]),
                    np.concatenate([u[a], t[c]]),
                )
            )


def _speed_cap(sim: Sim, sel: np.ndarray, px: np.ndarray, py: np.ndarray) -> None:
    """`mover_speed_cap`: cap total per-loop displacement of moving ground units in `sel`
    (relative to pre-kinematics `buf.move_x0/y0`) to `max(params.speed, kinematics_step)`
    (using post-kinematics `px/py`), re-projecting terrain for any shortened unit.
    """
    w, buf = sim.world, sim.buf
    x0, y0 = buf.move_x0, buf.move_y0
    with np.errstate(invalid='ignore'):
        dx, dy = w.x - x0, w.y - y0
        d = np.hypot(dx, dy)
        kin = np.hypot(px - x0, py - y0)
        cap = np.maximum(sim.params.speed, kin).astype(np.float32)
        over = sel & (d > cap)
    if not over.any():
        return
    k = (cap[over] / d[over]).astype(np.float32)
    w.x[over] = (x0[over] + dx[over] * k).astype(np.float32)
    w.y[over] = (y0[over] + dy[over] * k).astype(np.float32)
    _project(sim, over)


def _project(sim: Sim, ground_m: np.ndarray) -> None:
    """Project ground units in impassable cells back onto passable terrain (plus navmesh
    clearance push-out via `_clear` when `push_clearance` is enabled).
    """
    _project_cells(sim, ground_m)
    if sim.config.push_clearance:
        _clear(sim, ground_m)


# `push_clearance`: max edge push-out iterations (handles concave corners) and float32 guard.
CLEAR_ITERS = 4
CLEAR_EPS = 1e-4


def _clear(sim: Sim, ground_m: np.ndarray) -> None:
    """`push_clearance`: push ground units closer than `rho` to navmesh obstacle edges out to
    `rho + CLEAR_EPS` along the nearest-point normal (up to `CLEAR_ITERS` iterations in float64).
    """
    w = sim.world
    maps: dict[str, tuple[object, list[int]]] = {}
    for b in np.flatnonzero(ground_m.any(axis=1)).tolist():
        info = sim.env_map[b]
        assert info is not None
        maps.setdefault(info.map_id, (info, []))[1].append(b)
    near: list[tuple[int, int, navmesh.NavMesh]] = []
    for info, bs in maps.values():  # pre-filter by (map, radius) for clearance < rho
        sub = np.zeros_like(ground_m)
        sub[bs] = ground_m[bs]
        for r in np.unique(w.radius[sub]).tolist():
            mesh = pathing._mesh(sim, info, navmesh.path_radius(float(r)))  # noqa: SLF001
            if mesh.edges.shape[0] == 0:
                continue
            bi, ui = np.nonzero(sub & (w.radius == r))
            d = navmesh._point_seg_dist(  # noqa: SLF001
                w.x[bi, ui].astype(np.float64), w.y[bi, ui].astype(np.float64), mesh.edges
            ).min(axis=1)
            hit = d < mesh.rho
            near.extend((int(b), int(u), mesh) for b, u in zip(bi[hit], ui[hit], strict=True))
    for b, u, mesh in near:  # per-unit independent updates
        e = mesh.edges
        x, y = float(w.x[b, u]), float(w.y[b, u])
        moved = False
        for _ in range(CLEAR_ITERS):
            px, py, dist = _nearest_on_edges(x, y, e)
            if dist >= mesh.rho:
                break
            if dist > 0:
                nx, ny = (x - px) / dist, (y - py) / dist
            else:
                dists = navmesh._point_seg_dist(  # noqa: SLF001
                    np.array([x]), np.array([y]), e
                )[0]
                k = int(np.argmin(dists))
                ex, ey = e[k, 2] - e[k, 0], e[k, 3] - e[k, 1]
                n = math.hypot(ex, ey)
                nx, ny = ey / n, -ex / n
            step = mesh.rho + CLEAR_EPS - dist
            x, y = x + nx * step, y + ny * step
            moved = True
        if moved:
            w.x[b, u], w.y[b, u] = f32(x), f32(y)


def _nearest_on_edges(x: float, y: float, e: np.ndarray) -> tuple[float, float, float]:
    """Return `(qx, qy, dist)` of the closest point on segments `e[E, 4]` to `(x, y)`."""
    x0, y0, x1, y1 = e[:, 0], e[:, 1], e[:, 2], e[:, 3]
    dx, dy = x1 - x0, y1 - y0
    ll = dx * dx + dy * dy
    t = np.where(ll > 0, ((x - x0) * dx + (y - y0) * dy) / np.where(ll > 0, ll, 1), 0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = x0 + t * dx, y0 + t * dy
    d = np.hypot(x - qx, y - qy)
    k = int(np.argmin(d))
    return float(qx[k]), float(qy[k]), float(d[k])


def _project_cells(sim: Sim, ground_m: np.ndarray) -> None:
    """Cell-level terrain projection (default behavior of `_project`)."""
    w = sim.world
    groups: dict[int, tuple[np.ndarray, list[int]]] = {}
    for b in range(sim.batch_size):
        info = sim.env_map[b]
        assert info is not None
        groups.setdefault(id(info.pathing), (info.pathing, []))[1].append(b)
    for grid, bs in groups.values():
        h, wd = grid.shape
        idx = np.array(bs)
        xs, ys = w.x[idx], w.y[idx]
        g = ground_m[idx]
        with np.errstate(invalid='ignore'):
            ix = np.clip(np.floor(np.where(g, xs, 0)), 0, wd - 1).astype(np.int64)
            iy = np.clip(np.floor(np.where(g, ys, 0)), 0, h - 1).astype(np.int64)
        bad = g & ~grid[iy, ix]
        for i, u in zip(*np.nonzero(bad), strict=True):
            b = bs[int(i)]
            x, y = terrain.project_to_passable(grid, float(w.x[b, u]), float(w.y[b, u]))
            w.x[b, u], w.y[b, u] = x, y


def _crowd_arrival(sim: Sim, b: int, ground: list[int]) -> None:
    """Crowd-stop arrival: clear MOVE orders for ground units crowded around a shared goal.

    A "same-goal ally" is a ground unit of the same owner in contact (`center_dist < radii_sum -
    contact_slack + 0.02`) whose last MOVE goal is within `crowd_goal_tol`:

    1. Chain stop (`crowd_chain`): clear immediately on contact with an already-stopped ally.
    2. Blocked count: increment when forward progress `< crowd_progress * speed` while touching
       a same-goal ally (else reset to 0); clear once reaching `crowd_loops`.

    Retains `motion.crowd_goal_*` after stopping so later arrivals can chain-stop.
    """
    w, m, cfg = sim.world, sim.motion, sim.config
    reach = [float(w.radius[b, u]) for u in range(sim.n_slots)]
    contact_eps = 0.02 - float(cfg.contact_slack)
    tol = float(cfg.crowd_goal_tol)
    movers = [u for u in ground if int(w.order_kind[b, u]) == CommandKind.MOVE]
    for u in movers:  # record current MOVE goals first
        m.crowd_goal_x[b, u], m.crowd_goal_y[b, u] = w.order_x[b, u], w.order_y[b, u]
    clear: list[int] = []
    for u in movers:
        gx, gy = float(w.order_x[b, u]), float(w.order_y[b, u])
        ux, uy = float(w.x[b, u]), float(w.y[b, u])
        same, stopped = False, False
        for t in ground:
            if t == u or w.owner[b, t] != w.owner[b, u] or np.isnan(m.crowd_goal_x[b, t]):
                continue
            if math.hypot(float(m.crowd_goal_x[b, t]) - gx, float(m.crowd_goal_y[b, t]) - gy) > tol:
                continue
            d = math.hypot(float(w.x[b, t]) - ux, float(w.y[b, t]) - uy)
            if d >= reach[u] + reach[t] + contact_eps:
                continue
            same = True
            stopped = stopped or int(w.order_kind[b, t]) == CommandKind.NOOP
        dist = math.hypot(gx - ux, gy - uy)
        prev = float(m.crowd_dist[b, u])
        m.crowd_dist[b, u] = dist
        step = math.inf if math.isnan(prev) else prev - dist  # progress toward goal this loop
        speed = float(sim.params.speed[b, u])
        if stopped and cfg.crowd_chain:
            clear.append(u)
            continue
        slow = step < cfg.crowd_progress * speed
        m.crowd_blocked[b, u] = m.crowd_blocked[b, u] + 1 if same and slow else 0
        if m.crowd_blocked[b, u] >= cfg.crowd_loops:
            clear.append(u)
    for u in clear:  # clear simultaneously so result is independent of slot order
        w.order_kind[b, u] = CommandKind.NOOP
        w.order_target_slot[b, u] = NO_SLOT
        w.order_x[b, u] = w.order_y[b, u] = np.nan
        m.crowd_dist[b, u] = np.nan
        m.crowd_blocked[b, u] = 0
    for u in ground:  # non-MOVE units do not accumulate blocked counts
        if int(w.order_kind[b, u]) != CommandKind.MOVE:
            m.crowd_dist[b, u] = np.nan
            m.crowd_blocked[b, u] = 0


CROWD_MARGIN = 1e-9  # batched pre-filter margin (`np.hypot` vs `math.hypot` ulp tolerance)


def _crowd_arrival_many(sim: Sim, envs: np.ndarray, ground_m: np.ndarray) -> None:
    """Batched `_crowd_arrival` over `envs`, bit-identical to per-env calls (pre-filters pairs
    with `np.hypot` + `CROWD_MARGIN` and verifies candidates with `math.hypot`).
    """
    if envs.size == 0:
        return
    w, m, cfg = sim.world, sim.motion, sim.config
    g = ground_m[envs]
    kind = w.order_kind[envs]
    mv = g & (kind == CommandKind.MOVE)
    for dst, src in ((m.crowd_goal_x, w.order_x), (m.crowd_goal_y, w.order_y)):
        dst[envs] = np.where(mv, src[envs], dst[envs])
    goal_x, goal_y = m.crowd_goal_x[envs], m.crowd_goal_y[envs]
    ox, oy = w.order_x[envs], w.order_y[envs]
    x, y, r = w.x[envs], w.y[envs], w.radius[envs]
    contact_eps = 0.02 - float(cfg.contact_slack)
    tol = float(cfg.crowd_goal_tol)
    n = g.shape[1]
    cand = mv[:, :, None] & g[:, None, :] & ~np.eye(n, dtype=np.bool_)[None]
    cand &= w.owner[envs][:, :, None] == w.owner[envs][:, None, :]
    cand &= ~np.isnan(goal_x)[:, None, :]
    f64 = np.float64
    with np.errstate(invalid='ignore'):
        gd = np.hypot(
            goal_x.astype(f64)[:, None, :] - ox.astype(f64)[:, :, None],
            goal_y.astype(f64)[:, None, :] - oy.astype(f64)[:, :, None],
        )
        cand &= gd <= tol + CROWD_MARGIN
        cd = np.hypot(
            x.astype(f64)[:, None, :] - x.astype(f64)[:, :, None],
            y.astype(f64)[:, None, :] - y.astype(f64)[:, :, None],
        )
        rr = r.astype(f64)
        cand &= cd < rr[:, :, None] + rr[:, None, :] + contact_eps + CROWD_MARGIN
    # Exact scalar check on pre-filtered candidates (matching `_crowd_arrival`).
    ci, cu, ct = np.nonzero(cand)
    ok = [
        math.hypot(gxt - gxu, gyt - gyu) <= tol
        and math.hypot(xt - xu, yt - yu) < ru + rt + contact_eps
        for gxt, gyt, gxu, gyu, xt, yt, xu, yu, ru, rt in zip(
            goal_x[ci, ct].tolist(), goal_y[ci, ct].tolist(),
            ox[ci, cu].tolist(), oy[ci, cu].tolist(),
            x[ci, ct].tolist(), y[ci, ct].tolist(),
            x[ci, cu].tolist(), y[ci, cu].tolist(),
            r[ci, cu].tolist(), r[ci, ct].tolist(),
            strict=True,
        )
    ]  # fmt: skip
    hit = np.asarray(ok, dtype=np.bool_)
    same = np.zeros(g.shape, np.bool_)
    stopped = np.zeros(g.shape, np.bool_)
    same[ci[hit], cu[hit]] = True
    noop = kind[ci, ct] == CommandKind.NOOP
    stopped[ci[hit & noop], cu[hit & noop]] = True
    # Per-mover progress, blocked counter, and order clearing:
    mi, mu = np.nonzero(mv)
    dist = np.array(
        [
            math.hypot(gx - ux, gy - uy)
            for gx, gy, ux, uy in zip(
                ox[mi, mu].tolist(), oy[mi, mu].tolist(), x[mi, mu].tolist(), y[mi, mu].tolist(),
                strict=True,
            )
        ],
        dtype=f64,
    )  # fmt: skip
    eb = envs[mi]
    prev = m.crowd_dist[eb, mu].astype(f64)
    m.crowd_dist[eb, mu] = dist
    step = np.where(np.isnan(prev), np.inf, prev - dist)
    speed = sim.params.speed[eb, mu].astype(f64)
    chain = stopped[mi, mu] & bool(cfg.crowd_chain)
    slow = step < cfg.crowd_progress * speed
    blocked = m.crowd_blocked[eb, mu]
    new = np.where(same[mi, mu] & slow, blocked + 1, 0).astype(blocked.dtype)
    m.crowd_blocked[eb[~chain], mu[~chain]] = new[~chain]
    clear = chain | (~chain & (new >= cfg.crowd_loops))
    cb, cu2 = eb[clear], mu[clear]
    w.order_kind[cb, cu2] = CommandKind.NOOP
    w.order_target_slot[cb, cu2] = NO_SLOT
    w.order_x[cb, cu2] = np.nan
    w.order_y[cb, cu2] = np.nan
    m.crowd_dist[cb, cu2] = np.nan
    m.crowd_blocked[cb, cu2] = 0
    # Clear counters for non-MOVE ground units (evaluated after clears, matching `_crowd_arrival`).
    still = g & (w.order_kind[envs] != CommandKind.MOVE)
    si, su = np.nonzero(still)
    m.crowd_dist[envs[si], su] = np.nan
    m.crowd_blocked[envs[si], su] = 0


# `crowd_cover`: moving allies with heading cosine >= this count as parallel (not blocking).
COVER_PARALLEL_COS = 0.9


def _crowd_cover(sim: Sim, ground_m: np.ndarray) -> None:
    """`crowd_cover`: clear MOVE orders when an ally covers the destination.

    1. Clear MOVE orders marked `motion.crowd_cover_due` on the previous loop (preserving
       `crowd_goal_*`).
    2. For remaining ground MOVE units `u`, set `crowd_cover_due` for next loop if an allied
       ground unit `t != u` contacts `u` (`dist < radii_sum - contact_slack + 0.02`), covers `u`'s
       goal (`dist(t, goal) < radii_sum - contact_slack`), and is either stationary non-NOOP
       (HOLD / attacking in place) or moving with heading cosine `< COVER_PARALLEL_COS`.
       Idle (`NOOP`, no waypoint) allies are pushed aside and do not trigger `crowd_cover`.

    New commands clear `crowd_cover_due` in `commands`. Evaluated in float64, order-independent.
    """
    w, m = sim.world, sim.motion
    mv = ground_m & (w.order_kind == CommandKind.MOVE)
    due = m.crowd_cover_due & mv
    if due.any():
        cb, cu = np.nonzero(due)
        w.order_kind[cb, cu] = CommandKind.NOOP
        w.order_target_slot[cb, cu] = NO_SLOT
        w.order_x[cb, cu] = np.nan
        w.order_y[cb, cu] = np.nan
        m.crowd_dist[cb, cu] = np.nan
        m.crowd_blocked[cb, cu] = 0
    m.crowd_cover_due[:] = False
    mv &= ~due
    if not mv.any():
        return
    f64 = np.float64
    b, u = np.nonzero(mv)
    k = np.arange(b.size)
    x, y, r = w.x[b].astype(f64), w.y[b].astype(f64), w.radius[b].astype(f64)  # [k, U]
    ux, uy = x[k, u][:, None], y[k, u][:, None]
    gx, gy = w.order_x[b, u].astype(f64)[:, None], w.order_y[b, u].astype(f64)[:, None]
    reach = r[k, u][:, None] + r - float(sim.config.contact_slack)
    wx, wy = sim.buf.waypoint_x[b].astype(f64), sim.buf.waypoint_y[b].astype(f64)
    moving = ~np.isnan(wx)
    with np.errstate(invalid='ignore', divide='ignore'):
        touch = np.hypot(x - ux, y - uy) < reach + 0.02
        cover = np.hypot(x - gx, y - gy) < reach
        dux, duy = gx - ux, gy - uy
        tdx, tdy = wx - x, wy - y
        cos = (dux * tdx + duy * tdy) / (np.hypot(dux, duy) * np.hypot(tdx, tdy))
    still = ~moving & (w.order_kind[b] != CommandKind.NOOP)
    kind_ok = still | (moving & (cos < COVER_PARALLEL_COS))
    ok = ground_m[b] & (w.owner[b] == w.owner[b, u][:, None]) & touch & cover & kind_ok
    ok[k, u] = False
    hit = ok.any(axis=1)
    m.crowd_cover_due[b[hit], u[hit]] = True


def _mark_pushed_many(sim: Sim, b: np.ndarray, idle: np.ndarray, mover: np.ndarray) -> None:
    """Batched `_mark_pushed`: records max pushing mover radius on the current loop."""
    if b.size == 0:
        return
    m, w = sim.motion, sim.world
    loop = w.game_loop[b]
    stale = m.pushed_loop[b, idle] != loop
    m.pusher_radius[b[stale], idle[stale]] = 0
    m.pushed_loop[b, idle] = loop
    np.maximum.at(m.pusher_radius, (b, idle), w.radius[b, mover])


def _mark_pushed(sim: Sim, b: int, idle: int, mover: int, loop: int) -> None:
    m = sim.motion
    if m.pushed_loop[b, idle] != loop:
        m.pusher_radius[b, idle] = 0
    m.pushed_loop[b, idle] = loop
    m.pusher_radius[b, idle] = max(m.pusher_radius[b, idle], sim.world.radius[b, mover])


def _maybe_return(sim: Sim, b: int, u: int, loop: int) -> None:
    """Issue a return-to-home MOVE order for a displaced idle unit when criteria are met."""
    w, m, cfg = sim.world, sim.motion, sim.config
    if np.isnan(m.home_x[b, u]) or m.pushed_loop[b, u] < 0:
        return
    if int(w.order_kind[b, u]) != CommandKind.NOOP or m.pushed_loop[b, u] == loop:
        return
    away = float(np.hypot(w.x[b, u] - m.home_x[b, u], w.y[b, u] - m.home_y[b, u]))
    if away <= cfg.return_threshold:
        return
    waited = loop - m.pushed_loop[b, u] >= cfg.return_delay
    if cfg.idle_return_rule == 'diameter':
        if m.pushed_max[b, u] <= cfg.return_ratio * 2 * w.radius[b, u]:
            return
        if away <= cfg.return_far and not waited:
            return
    else:
        bigger = m.pusher_radius[b, u] > w.radius[b, u]
        if away <= cfg.return_far and not (bigger and waited):
            return
    w.order_kind[b, u] = CommandKind.MOVE
    w.order_target_slot[b, u] = NO_SLOT
    w.order_x[b, u], w.order_y[b, u] = m.home_x[b, u], m.home_y[b, u]
    w.order_ability[b, u] = 0
    m.pushed_loop[b, u] = -1
    m.pusher_radius[b, u] = 0
    m.pushed_max[b, u] = 0


IMPLS = {'none': none, 'hard': hard}

__all__ = [
    'CHAIN_EPS',
    'CLEAR_EPS',
    'CLEAR_ITERS',
    'COVER_PARALLEL_COS',
    'CROWD_MARGIN',
    'GROW_MARGIN',
    'IMPLS',
    'SAME_SPEED_RTOL',
    'SWEEP_EPS',
    'SWEEP_MARGIN',
    'SWEEP_SMALL',
    'after_contact',
    'hard',
    'mover_weight',
    'none',
    'pinned_mask',
]
