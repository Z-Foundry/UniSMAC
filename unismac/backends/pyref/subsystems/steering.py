"""Stage 5 local steering: converts waypoints into desired velocities `buf.vel_*` (tiles/loop)."""

from __future__ import annotations

import math

import numpy as np

from unismac.contract import NO_SLOT, CommandKind

from ..sim import Sim, f32
from .collision import pinned_mask


def _seek(sim: Sim, b: int, u: int) -> tuple[np.float32, np.float32]:
    """Steer straight toward the waypoint at max speed without overshooting; `(0, 0)` if none."""
    w, buf = sim.world, sim.buf
    if np.isnan(buf.waypoint_x[b, u]):
        return f32(0), f32(0)
    dx = buf.waypoint_x[b, u] - w.x[b, u]
    dy = buf.waypoint_y[b, u] - w.y[b, u]
    dist = f32(np.sqrt(dx * dx + dy * dy))
    if dist <= 0:
        return f32(0), f32(0)
    if dist <= sim.params.speed[b, u]:
        return dx, dy  # exact final step onto waypoint for zero-radius arrival snap
    step = sim.params.speed[b, u]
    return dx / dist * step, dy / dist * step


def _seek_all(sim: Sim) -> tuple[np.ndarray, np.ndarray]:
    """Batched `_seek`, bit-identical elementwise (same float32 operation order)."""
    w, buf = sim.world, sim.buf
    speed = sim.params.speed
    with np.errstate(invalid='ignore', divide='ignore'):
        dx = buf.waypoint_x - w.x
        dy = buf.waypoint_y - w.y
        dist = np.sqrt(dx * dx + dy * dy)
        far_x, far_y = dx / dist * speed, dy / dist * speed
        zero = np.isnan(buf.waypoint_x) | (dist <= 0)
        near = dist <= speed
    vx = np.where(zero, f32(0), np.where(near, dx, far_x)).astype(np.float32)
    vy = np.where(zero, f32(0), np.where(near, dy, far_y)).astype(np.float32)
    return vx, vy


def none(sim: Sim) -> None:
    """No local avoidance: seek directly toward the waypoint at max speed."""
    vx, vy = _seek_all(sim)
    sim.buf.vel_x[:] = vx
    sim.buf.vel_y[:] = vy


def separation(sim: Sim) -> None:
    """Boids-style separation: adds neighbor repulsion onto seek velocity and clamps to max speed.

    * Neighbors: alive same-layer (ground/air) units with edge distance `< separation_margin`.
    * Repulsion pushes away from each neighbor by `separation_gain * (margin - gap) * speed`
      (coincident centers push along `+/-x` by slot order).
    * Stationary units (no waypoint) receive no repulsion; overlaps resolve in collision.
    * Reads start-of-loop positions only (independent of slot order).
    """
    w, buf = sim.world, sim.buf
    gain = f32(sim.config.separation_gain)
    margin = f32(sim.config.separation_margin)
    for b in range(sim.batch_size):
        for u in range(sim.n_slots):
            vx, vy = _seek(sim, b, u)
            buf.vel_x[b, u], buf.vel_y[b, u] = vx, vy
            if not w.alive[b, u] or np.isnan(buf.waypoint_x[b, u]):
                continue
            speed = sim.params.speed[b, u]
            px, py = f32(0), f32(0)
            for t in range(sim.n_slots):
                if t == u or not w.alive[b, t] or w.is_flying[b, t] != w.is_flying[b, u]:
                    continue
                dx = w.x[b, u] - w.x[b, t]
                dy = w.y[b, u] - w.y[b, t]
                dist = f32(np.sqrt(dx * dx + dy * dy))
                gap = f32(dist - w.radius[b, u] - w.radius[b, t])
                if gap >= margin:
                    continue
                push = gain * (margin - gap) * speed
                if dist > 0:
                    px, py = f32(px + dx / dist * push), f32(py + dy / dist * push)
                else:
                    px = f32(px + (push if u < t else -push))
            if px == 0 and py == 0:
                continue
            vx, vy = f32(vx + px), f32(vy + py)
            mag = f32(np.sqrt(vx * vx + vy * vy))
            if mag > speed:
                vx, vy = vx / mag * speed, vy / mag * speed
            buf.vel_x[b, u], buf.vel_y[b, u] = vx, vy


TIE_EPS = 0.01  # lateral offset (tiles) below which a neighbor counts as head-on/aligned
# Heading cosine > PARALLEL_COS is parallel, < -PARALLEL_COS is head-on, else crossing (~25 deg).
PARALLEL_COS = 0.9


def sc2_avoid(sim: Sim) -> None:
    """SC2-style pairwise ground local avoidance applied to moving ground units.

    Reads only start-of-loop state (positions and previous speed/heading), independent of slot
    order. For each moving unit `u`, classifies ahead ground neighbors `t` within `avoid_range`
    using lateral offset `lat = cross(d, t - u)` (positive = left):

    * Idle `t` (no waypoint): let collision push it aside; nudge rightward by `idle_nudge` if
      dead ahead (`|lat| < TIE_EPS`) to break symmetry.
    * Parallel dead-ahead `t` (`cos > PARALLEL_COS`, `|lat| < TIE_EPS`): match leader speed plus
      remaining gap when `edge < follow_gap` instead of overtaking.
    * Crossing `t` (`-PARALLEL_COS <= cos <= avoid_cross_cos`): do not avoid before contact; once
      `edge < avoid_margin`, slide clockwise (rightward) around `t`.
    * Otherwise (offset parallel or head-on): predict closest approach within `avoid_horizon`; if
      predicted clearance `< contact + avoid_margin`, steer around `t` away from its side (right
      on ties; flipped by `avoid_side_check` when a parallel ally occupies the preferred side).
      Optional filters: `avoid_closing_only`, `avoid_parallel_faster_only`.

    With `pin_engaged`, stationary pinned units (`collision.pinned_mask`, excluding `u`'s own
    target) act as zero-velocity obstacles in the closest-approach branch. Only the earliest
    imminent neighbor drives avoidance.

    Optional post-modifiers: `avoid_share='speed'` (scale turn angle by `_share_gain` using
    `avoid_share_power_headon` / `avoid_share_power_cross`), `contact_orbit` (`_orbit`), and
    `pin_sidestep` (`_sidestep`).
    """
    buf = sim.buf
    seek_x, seek_y = _seek_all(sim)
    out_x, out_y = seek_x.copy(), seek_y.copy()
    near = _avoid_candidates(sim, seek_x, seek_y)
    if near.any():
        for b, u in _avoid_batch(sim, seek_x, seek_y, near, out_x, out_y):
            ts = np.flatnonzero(near[b, u]).tolist()
            out_x[b, u], out_y[b, u] = _avoid_unit(sim, b, u, seek_x[b, u], seek_y[b, u], ts)
    if sim.config.contact_orbit:
        _orbit(sim, seek_x, seek_y, out_x, out_y)
    if sim.config.pin_sidestep:
        _sidestep(sim, seek_x, seek_y, out_x, out_y)
    buf.vel_x[:] = out_x
    buf.vel_y[:] = out_y


# `pin_sidestep`: clearance margin (tiles) and dead-ahead lateral tie threshold (tiles).
SIDESTEP_RANGE = 0.35
SIDESTEP_TIE = 0.05
# `pin_sidestep_drift`: minimum previous-loop lateral drift (tiles) to hold sidestep side.
SIDESTEP_DRIFT_TIE = 0.01
# `pin_sidestep_lat_scope='marauder'`: Marauder unit_type ID (shared across variants).
MARAUDER = 51


def _sidestep(
    sim: Sim, seek_x: np.ndarray, seek_y: np.ndarray, out_x: np.ndarray, out_y: np.ndarray
) -> None:
    """`pin_sidestep`: steer moving ground MOVE units laterally around pinned blockers directly
    ahead.

    Qualifying blockers `t` are pinned (`collision.pinned_mask`), non-self, non-target units
    within `contact + SIDESTEP_RANGE` ahead inside `pin_sidestep_cone` degrees (and lateral
    distance `< pin_sidestep_lat` if set, scoped by `pin_sidestep_lat_scope`). Selects the lowest
    slot blocker (or closest center under `pin_sidestep_tangent`) and replaces desired velocity
    in-place with a perpendicular (or blocker-tangent under `pin_sidestep_tangent`) vector of
    length `seek_speed * pin_sidestep_speed`. Side prefers away from `t` if `|lat| > SIDESTEP_TIE`,
    otherwise matches `out` (right on tie), overridden by previous lateral drift when
    `pin_sidestep_drift` is set. Evaluated in float64 and cast to float32.
    """
    w, cfg = sim.world, sim.config
    moved = None  # `pin_sidestep_drift`: previous-loop displacement (NaN on loop 0)
    if cfg.pin_sidestep_drift:
        m = sim.motion
        moved = ((w.x - m.drift_px).astype(np.float64), (w.y - m.drift_py).astype(np.float64))
        m.drift_px[:] = w.x
        m.drift_py[:] = w.y
    move = (seek_x != 0) | (seek_y != 0)
    move &= w.alive & ~w.is_flying & (w.order_kind == CommandKind.MOVE)
    if not move.any():
        return
    pin = pinned_mask(sim)
    bi, ui = np.nonzero(move & pin.any(axis=1, keepdims=True))
    if bi.size == 0:
        return
    f64 = np.float64
    spd = np.hypot(seek_x[bi, ui], seek_y[bi, ui]).astype(f64)
    dx = seek_x[bi, ui].astype(f64) / spd
    dy = seek_y[bi, ui].astype(f64) / spd
    rx = (w.x[bi] - w.x[bi, ui][:, None]).astype(f64)  # [k, U]: t - u
    ry = (w.y[bi] - w.y[bi, ui][:, None]).astype(f64)
    dist = np.hypot(rx, ry)
    contact = (w.radius[bi, ui][:, None] + w.radius[bi]).astype(f64) - cfg.contact_slack
    ahead = rx * dx[:, None] + ry * dy[:, None]
    slots = np.arange(w.x.shape[1])[None, :]
    ok = pin[bi] & (slots != ui[:, None])
    ok &= slots != w.engaged_target_slot[bi, ui][:, None]
    ok &= slots != w.order_target_slot[bi, ui][:, None]
    ok &= (dist > 0) & (dist <= contact + SIDESTEP_RANGE) & (ahead > 0)
    cos_cone = float(np.cos(np.radians(cfg.pin_sidestep_cone)))
    with np.errstate(invalid='ignore', divide='ignore'):
        ok &= ahead / dist >= cos_cone
    if cfg.pin_sidestep_lat is not None:  # filter out blockers too far laterally
        near = np.abs(dx[:, None] * ry - dy[:, None] * rx) < f64(cfg.pin_sidestep_lat)
        if cfg.pin_sidestep_lat_scope == 'marauder':  # skip filter if neither unit is a Marauder
            pair = (w.unit_type[bi] == MARAUDER) | (w.unit_type[bi, ui] == MARAUDER)[:, None]
            near |= ~pair
        ok &= near
    hit = ok.any(axis=1)
    if not hit.any():
        return
    k = np.flatnonzero(hit)
    if cfg.pin_sidestep_tangent:  # closest center distance (lowest slot on tie)
        t = np.argmin(np.where(ok[k], dist[k], np.inf), axis=1)
    else:
        t = np.argmax(ok[k], axis=1)  # lowest slot blocker
    bi, ui, spd, dx, dy = bi[k], ui[k], spd[k], dx[k], dy[k]
    lat = dx * ry[k, t] - dy * rx[k, t]  # > 0 means t is left of d
    left = dx * out_y[bi, ui].astype(f64) - dy * out_x[bi, ui].astype(f64) > 1e-9
    left = np.where(np.abs(lat) > SIDESTEP_TIE, lat < 0, left)
    if moved is not None:
        mvx, mvy = moved
        drift = dx * mvy[bi, ui] - dy * mvx[bi, ui]  # > 0 means prior loop drifted left of d
        with np.errstate(invalid='ignore'):
            left = np.where(np.abs(drift) > SIDESTEP_DRIFT_TIE, drift > 0, left)
    nx = np.where(left, -dy, dy)
    ny = np.where(left, dx, -dx)
    if cfg.pin_sidestep_tangent:  # tangent to t: center vector rotated +/-90 deg on chosen side
        cx = rx[k, t] / dist[k, t]
        cy = ry[k, t] / dist[k, t]
        nx = np.where(left, -cy, cy)
        ny = np.where(left, cx, -cx)
    gain = f64(cfg.pin_sidestep_speed)
    out_x[bi, ui] = (nx * spd * gain).astype(np.float32)
    out_y[bi, ui] = (ny * spd * gain).astype(np.float32)


def _share_powers(sim: Sim) -> tuple[float, float]:
    """Return `(headon_power, cross_power)` for `avoid_share='speed'`."""
    cfg = sim.config
    p = float(cfg.avoid_share_power)
    ph, pc = cfg.avoid_share_power_headon, cfg.avoid_share_power_cross
    return (p if ph is None else float(ph)), (p if pc is None else float(pc))


def _share_gain(sim: Sim, vu, vt, p: float | None = None):
    """Turn-angle multiplier `min(1, 2*w_u / (w_u + w_t))` with `w = v^p` (`avoid_share='speed'`).
    Works on both scalars and arrays with bit-identical float64 evaluation.
    """
    p = float(sim.config.avoid_share_power) if p is None else float(p)
    wu, wt = np.float64(vu) ** p, np.float64(vt) ** p
    tot = wu + wt
    return np.minimum(np.where(tot > 0, 2.0 * wu / np.where(tot > 0, tot, 1.0), 1.0), 1.0)


def _share_turn(sx, sy, ax, ay, spd, g):
    """Scale the turn angle from seek `(sx, sy)` to avoid `(ax, ay)` by `g` at speed `spd`."""
    th0 = np.arctan2(np.float64(sy), np.float64(sx))
    d = np.remainder(np.arctan2(ay, ax) - th0 + np.pi, 2 * np.pi) - np.pi
    th = th0 + g * d
    return (np.cos(th) * spd).astype(np.float32), (np.sin(th) * spd).astype(np.float32)


# `pin_tie_grid`: outline tie tolerance and mirror symmetry tolerance (half of 1/4096 tile).
PIN_TIE_COST = 0.5 / 4096
PIN_MIRROR_TOL = 0.5 / 4096


def _pin_group(sim: Sim, b: int, t: int, cands: list[int], gap_max: float) -> list[int]:
    """Return sorted slots of pinned units connected to `t` via gaps `< gap_max`."""
    w = sim.world
    group, todo = {t}, [t]
    while todo:
        cur = todo.pop()
        for q in cands:
            if q in group:
                continue
            d = float(np.hypot(float(w.x[b, q] - w.x[b, cur]), float(w.y[b, q] - w.y[b, cur])))
            if d - float(w.radius[b, cur] + w.radius[b, q]) < gap_max:
                group.add(q)
                todo.append(q)
    return sorted(group)


def _pin_tie_side(sim: Sim, b: int, c: int, group: list[int], dxu: float, dyu: float) -> bool:
    """`pin_tie_grid` tie-breaker side (`True` = right): returns True if `group` is mirror-
    symmetric across the line through `c` along `(dxu, dyu)` or if `c` is on HOLD; otherwise
    delegates to `_grid_right`.
    """
    w = sim.world
    cx, cy = float(w.x[b, c]), float(w.y[b, c])
    rel = [
        (
            (float(w.x[b, q]) - cx) * dxu + (float(w.y[b, q]) - cy) * dyu,
            dxu * (float(w.y[b, q]) - cy) - dyu * (float(w.x[b, q]) - cx),
        )
        for q in group
    ]
    mirrored = all(
        any(abs(a2 - a) <= PIN_MIRROR_TOL and abs(l2 + lt) <= PIN_MIRROR_TOL for a2, l2 in rel)
        for a, lt in rel
    )
    if mirrored or int(w.order_kind[b, c]) == CommandKind.HOLD:
        return True
    return _grid_right(cx, cy, dxu, dyu)


def _grid_right(tx: float, ty: float, dxu: float, dyu: float) -> bool:
    """`pin_tie_grid` asymmetric tie side (`True` = right) from parity of `floor(8 * coord)`
    on the world axis most perpendicular to `(dxu, dyu)`.
    """
    if abs(dxu) >= abs(dyu):
        c, comp = ty, -dxu  # lateral axis y; y-component of right normal (dyu, -dxu)
    else:
        c, comp = tx, dyu  # lateral axis x; x-component of right normal
    up = math.floor(c * 8.0) % 2 == 0
    return (comp > 0) == up


def _pin_aim(
    sim: Sim, b: int, u: int, t: int, dxu: float, dyu: float, right: bool, pin, own
) -> tuple[float, float]:
    """Compute aim point around pinned blocker `t` (`pin_engaged`). Offsets `contact + avoid_margin`
    on side `right`; if the gap on that side is narrower than `pin_gap_closed * diameter`, chains
    across connected pinned units and picks the shorter outer edge (with `pin_tie_grid` handling
    aligned cluster ties).
    """
    w, cfg = sim.world, sim.config
    ux, uy = float(w.x[b, u]), float(w.y[b, u])
    ru = float(w.radius[b, u])
    gap_max = cfg.pin_gap_closed * 2.0 * ru

    def need(q: int) -> float:
        return float(w.radius[b, u] + w.radius[b, q]) - cfg.contact_slack + cfg.avoid_margin

    def lat(q: int) -> float:
        return dxu * (float(w.y[b, q]) - uy) - dyu * (float(w.x[b, q]) - ux)

    cands = [
        q
        for q in np.flatnonzero(pin).tolist()
        if q != u and q not in own
        and (float(w.x[b, q]) - ux) * dxu + (float(w.y[b, q]) - uy) * dyu > 0
    ]  # fmt: skip

    def chain_end(side_right: bool) -> int:
        cur, seen = t, {t}
        while True:
            nxt, nl = -1, 0.0
            lc = lat(cur)
            for q in cands:
                if q in seen:
                    continue
                d = float(np.hypot(float(w.x[b, q] - w.x[b, cur]), float(w.y[b, q] - w.y[b, cur])))
                if d - float(w.radius[b, cur] + w.radius[b, q]) >= gap_max:
                    continue
                lq = lat(q)
                if (
                    (lq < lc and (nxt < 0 or lq < nl))
                    if side_right
                    else (lq > lc and (nxt < 0 or lq > nl))
                ):
                    nxt, nl = q, lq
            if nxt < 0:
                return cur
            cur = nxt
            seen.add(cur)

    side = right
    end = chain_end(right) if cands else t
    if end != t:
        other = chain_end(not right)
        cost = abs(lat(end) - need(end)) if right else abs(lat(end) + need(end))
        cost_o = abs(lat(other) + need(other)) if right else abs(lat(other) - need(other))
        group = _pin_group(sim, b, t, cands, gap_max) if cfg.pin_tie_grid else []
        c = min(group, key=lambda q: abs(lat(q))) if group else -1
        if c >= 0 and abs(lat(c)) < TIE_EPS:
            # u is aligned with c in the cluster: compare outer extents relative to c.
            lc = lat(c)
            end_r, end_l = (end, other) if right else (other, end)
            ext_r = abs(lat(end_r) - lc - need(end_r))
            ext_l = abs(lat(end_l) - lc + need(end_l))
            if abs(ext_l - ext_r) <= PIN_TIE_COST:
                go_right = _pin_tie_side(sim, b, c, group, dxu, dyu)
            else:
                go_right = ext_r < ext_l
            if go_right != right:
                side, end = not right, other
        elif cost_o < cost:
            side, end = not right, other
    nx, ny = (dyu, -dxu) if side else (-dyu, dxu)
    rx, ry = float(w.x[b, end]) - ux, float(w.y[b, end]) - uy
    px, py = ux + rx + 0.0 * 0.0, uy + ry + 0.0 * 0.0
    return px + nx * need(end), py + ny * need(end)


def _avoid_unit(sim: Sim, b: int, u: int, vx, vy, ts: list[int]):
    """Scalar pairwise `sc2_avoid` for unit `u` over candidate neighbors `ts`; used as fallback
    from `_avoid_batch` when non-finite intermediates arise.
    """
    w, buf, m, cfg = sim.world, sim.buf, sim.motion, sim.config
    slack = cfg.contact_slack
    spd = float(np.hypot(vx, vy))
    if spd == 0:
        return vx, vy
    dxu, dyu = float(vx) / spd, float(vy) / spd
    sx, sy = vx, vy
    ux, uy = float(w.x[b, u]), float(w.y[b, u])
    # (key, aim_x, aim_y, moving_neighbor_or_-1, is_crossing_branch)
    best: tuple[float, float, float, int, bool] | None = None
    cap = np.inf
    pin = pinned_mask(sim)[b] if cfg.pin_engaged else None
    own = (int(w.engaged_target_slot[b, u]), int(w.order_target_slot[b, u]))
    for t in ts:
        if t == u or not w.alive[b, t] or w.is_flying[b, t]:
            continue
        rx, ry = float(w.x[b, t]) - ux, float(w.y[b, t]) - uy
        ahead = rx * dxu + ry * dyu
        lat = dxu * ry - dyu * rx
        dist = float(np.hypot(rx, ry))
        contact = float(w.radius[b, u] + w.radius[b, t]) - slack
        if ahead <= 0 or dist - contact >= cfg.avoid_range:
            continue
        if pin is not None and pin[t] and t not in own:
            # Pinned stationary blocker: closest-approach branch with zero velocity (no side check).
            wx, wy = 0.0 - dxu * spd, 0.0 - dyu * spd  # matches batched `tvx - d * spd` (tvx = 0)
            ww = wx * wx + wy * wy
            tau = 0.0 if ww == 0 else max(0.0, -(rx * wx + ry * wy) / ww)
            if tau > cfg.avoid_horizon:
                continue
            mx, my = rx + wx * tau, ry + wy * tau
            need = contact + cfg.avoid_margin
            if float(np.hypot(mx, my)) >= need:
                continue
            right = lat >= 0 if abs(lat) >= TIE_EPS else True
            if best is None or tau < best[0]:
                best = (tau, *_pin_aim(sim, b, u, t, dxu, dyu, right, pin, own), -1, False)
            continue
        if np.isnan(buf.waypoint_x[b, t]):
            if abs(lat) < TIE_EPS:
                key = dist - contact
                if best is None or key < best[0]:
                    # Nudge right along right normal `(dy, -dx)`.
                    best = (
                        key,
                        ux + rx + dyu * cfg.idle_nudge,
                        uy + ry - dxu * cfg.idle_nudge,
                        -1,
                        False,
                    )
            continue
        # Classify using neighbor's seek direction (actual heading drifts mid-avoidance).
        gx = float(buf.waypoint_x[b, t] - w.x[b, t])
        gy = float(buf.waypoint_y[b, t] - w.y[b, t])
        gn = float(np.hypot(gx, gy))
        if gn == 0:
            continue
        gx, gy = gx / gn, gy / gn
        cos = dxu * gx + dyu * gy
        # Neighbor velocity: previous-loop velocity, or full seek speed if starting from rest.
        tspd = float(m.speed[b, t])
        if tspd > 0:
            th = float(m.heading[b, t])
            tvx, tvy = np.cos(th) * tspd, np.sin(th) * tspd
        else:
            tspd = float(sim.params.speed[b, t])
            tvx, tvy = gx * tspd, gy * tspd
        if cos > PARALLEL_COS and abs(lat) < TIE_EPS:
            gap = dist - contact
            if gap < cfg.follow_gap:
                # Leader speed along d plus remaining gap: matches spacing at loop end.
                cap = min(cap, max(float(tvx * dxu + tvy * dyu) + max(gap, 0.0), 0.0))
            continue
        if -PARALLEL_COS <= cos <= cfg.avoid_cross_cos:
            # Crossing: ignore until within avoid_margin, then slide clockwise around t.
            if dist - contact < cfg.avoid_margin and dist > 0:
                tx_, ty_ = ry / dist, -rx / dist  # clockwise (right) tangent around t
                if best is None or best[0] > -1.0:
                    best = (-1.0, ux + tx_, uy + ty_, t, True)
            continue
        if (
            cfg.avoid_parallel_faster_only
            and cos > PARALLEL_COS
            and not sim.params.speed[b, u] > sim.params.speed[b, t]
        ):
            continue  # parallel and cannot overtake: skip avoidance
        # Closest-approach prediction:
        wx, wy = tvx - dxu * spd, tvy - dyu * spd
        ww = wx * wx + wy * wy
        tau = 0.0 if ww == 0 else max(0.0, -(rx * wx + ry * wy) / ww)
        if cfg.avoid_closing_only and not tau > 0.0:
            continue  # not closing: closest approach is now
        if tau > cfg.avoid_horizon:
            continue
        mx, my = rx + wx * tau, ry + wy * tau
        need = contact + cfg.avoid_margin
        if float(np.hypot(mx, my)) >= need:
            continue
        # Head-on / offset parallel: steer away from neighbor's side (right on dead-ahead tie).
        right = lat >= 0 if abs(lat) >= TIE_EPS else True
        if cfg.avoid_side_check and cos < -PARALLEL_COS:
            # Head-on with preferred side blocked by a parallel ally and opposite side open: flip.
            rb = _side_blocked(sim, b, u, dxu, dyu, right=True)
            lb = _side_blocked(sim, b, u, dxu, dyu, right=False)
            if right and rb and not lb:
                right = False
            elif not right and lb and not rb:
                right = True
        nx, ny = (dyu, -dxu) if right else (-dyu, dxu)
        px, py = ux + rx + tvx * tau, uy + ry + tvy * tau
        key = tau
        if best is None or key < best[0]:
            best = (key, px + nx * need, py + ny * need, t, False)
    if best is not None:
        ax, ay = best[1] - ux, best[2] - uy
        n = float(np.hypot(ax, ay))
        if n > 0:
            vx, vy = f32(ax / n * spd), f32(ay / n * spd)
            if cfg.avoid_share == 'speed' and best[3] >= 0:
                ph, pc = _share_powers(sim)
                g = _share_gain(
                    sim, sim.params.speed[b, u], sim.params.speed[b, best[3]], pc if best[4] else ph
                )
                if g < 1.0:
                    vx, vy = (f32(a) for a in _share_turn(sx, sy, ax, ay, spd, g))
    if cap < spd:
        mag = float(np.hypot(vx, vy))
        if mag > 0:
            vx, vy = f32(vx / mag * cap), f32(vy / mag * cap)
    return vx, vy


def _avoid_batch(sim: Sim, seek_x, seek_y, near: np.ndarray, out_x, out_y) -> list[tuple[int, int]]:
    """Batched `_avoid_unit` over all `(b, u, t)` in `near`, bit-identical to scalar calls.
    Returns any `(b, u)` rows with non-finite intermediates for scalar fallback.
    """
    w, buf, m, cfg = sim.world, sim.buf, sim.motion, sim.config
    f64 = np.float64
    pb, pu, pt = np.nonzero(near)
    n = w.x.shape[1]
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        vx32, vy32 = seek_x[pb, pu], seek_y[pb, pu]
        spd = np.hypot(vx32, vy32).astype(f64)
        dxu, dyu = vx32.astype(f64) / spd, vy32.astype(f64) / spd
        ux, uy = w.x[pb, pu].astype(f64), w.y[pb, pu].astype(f64)
        rx, ry = w.x[pb, pt].astype(f64) - ux, w.y[pb, pt].astype(f64) - uy
        ahead = rx * dxu + ry * dyu
        lat = dxu * ry - dyu * rx
        dist = np.hypot(rx, ry)
        contact = (w.radius[pb, pu] + w.radius[pb, pt]).astype(f64) - cfg.contact_slack
        gap = dist - contact
        valid = ~((ahead <= 0) | (gap >= cfg.avoid_range))
        bad = ~(np.isfinite(ahead) & np.isfinite(gap) & np.isfinite(lat))
        idle = np.isnan(buf.waypoint_x[pb, pt])
        pm = pinned_mask(sim) if cfg.pin_engaged else None
        if pm is not None:  # never treat own attack target as a pinned obstacle
            own = (pt == w.engaged_target_slot[pb, pu]) | (pt == w.order_target_slot[pb, pu])
            pinned = pm[pb, pt] & ~own
        else:
            pinned = np.zeros(pb.size, np.bool_)
        idle &= ~pinned
        tie = np.abs(lat) < TIE_EPS
        k_idle = valid & idle & tie
        gx = (buf.waypoint_x[pb, pt] - w.x[pb, pt]).astype(f64)
        gy = (buf.waypoint_y[pb, pt] - w.y[pb, pt]).astype(f64)
        gn = np.hypot(gx, gy)
        mov = valid & ~idle & ~pinned & ~(gn == 0)
        gx, gy = gx / gn, gy / gn
        cos = dxu * gx + dyu * gy
        tspd = m.speed[pb, pt].astype(f64)
        th = m.heading[pb, pt].astype(f64)
        pspd = sim.params.speed[pb, pt].astype(f64)
        pos = tspd > 0
        tvx = np.where(pos, np.cos(th) * tspd, gx * pspd)
        tvy = np.where(pos, np.sin(th) * tspd, gy * pspd)
        bad |= mov & ~(np.isfinite(cos) & np.isfinite(tvx) & np.isfinite(tvy))
        follow = mov & (cos > PARALLEL_COS) & tie
        rest = mov & ~follow
        cross = rest & (-PARALLEL_COS <= cos) & (cos <= cfg.avoid_cross_cos)
        pred = rest & ~cross
        pin_on = valid & pinned  # pinned blocker: zero-velocity closest-approach branch
        tvx = np.where(pin_on, 0.0, tvx)
        tvy = np.where(pin_on, 0.0, tvy)
        pred |= pin_on
        # Follow-speed cap: max(along + max(gap, 0), 0)
        fol_on = follow & (gap < cfg.follow_gap)
        g0 = np.where(0.0 > gap, 0.0, gap)
        sv = (tvx * dxu + tvy * dyu) + g0
        capv = np.where(0.0 > sv, 0.0, sv)
        bad |= fol_on & np.isnan(capv)
        # Crossing: clockwise rim slide on contact
        cross_on = cross & (gap < cfg.avoid_margin) & (dist > 0)
        # Closest approach:
        wx, wy = tvx - dxu * spd, tvy - dyu * spd
        ww = wx * wx + wy * wy
        q = -(rx * wx + ry * wy) / ww
        tau = np.where(ww == 0, 0.0, np.where(q > 0.0, q, 0.0))
        pr_on = pred & ~(tau > cfg.avoid_horizon)
        if cfg.avoid_closing_only:  # skip non-closing movers (tau == 0); keep pinned blockers
            pr_on &= (tau > 0.0) | pin_on
        mx, my = rx + wx * tau, ry + wy * tau
        need = contact + cfg.avoid_margin
        pr_on &= ~(np.hypot(mx, my) >= need)
        if cfg.avoid_parallel_faster_only:  # skip parallel movers we cannot overtake
            par = mov & (cos > PARALLEL_COS)
            pr_on &= ~(par & ~(sim.params.speed[pb, pu] > sim.params.speed[pb, pt]))
        bad |= pred & ~(np.isfinite(tau) & np.isfinite(mx) & np.isfinite(my))
        right = np.where(np.abs(lat) >= TIE_EPS, lat >= 0, True)
        if cfg.avoid_side_check:
            side = np.flatnonzero(pr_on & (cos < -PARALLEL_COS))
            if side.size:
                rb, lb = _side_blocked_many(sim, pb[side], pu[side], dxu[side], dyu[side])
                rs = right[side]
                right[side] = np.where(rs & rb & ~lb, False, np.where(~rs & lb & ~rb, True, rs))
        nx = np.where(right, dyu, -dyu)
        ny = np.where(right, -dxu, dxu)
        px, py = ux + rx + tvx * tau, uy + ry + tvy * tau
        key = np.where(k_idle, gap, np.where(cross_on, -1.0, tau))
        aim_x = np.where(
            k_idle,
            ux + rx + dyu * cfg.idle_nudge,
            np.where(cross_on, ux + ry / dist, px + nx * need),
        )
        aim_y = np.where(
            k_idle,
            uy + ry - dxu * cfg.idle_nudge,
            np.where(cross_on, uy + -rx / dist, py + ny * need),
        )
        sel = k_idle | cross_on | pr_on
        bad |= sel & ~(np.isfinite(key) & np.isfinite(aim_x) & np.isfinite(aim_y))
    # Reduce per unit (pb, pu are row-major ordered, so pairs per unit are contiguous).
    gid = pb * n + pu
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    grp = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, gid.size]))
    ub, uu = pb[starts], pu[starts]
    unit_bad = np.zeros(starts.size, dtype=bool)
    np.logical_or.at(unit_bad, grp, bad)

    def first_min(mask: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = np.where(mask, val, np.inf)
        vmin = np.minimum.reduceat(v, starts)
        hit = np.flatnonzero(mask & (v == vmin[grp]))
        first = np.full(starts.size, -1)
        g, fi = np.unique(grp[hit], return_index=True)
        first[g] = hit[fi]
        return first, vmin

    best, _ = first_min(sel, key)
    if pm is not None:  # pinned winners may need cluster chaining via `_pin_aim`
        for i in np.flatnonzero((best >= 0) & pin_on[np.where(best >= 0, best, 0)]).tolist():
            k = int(best[i])
            b_, u_, t_ = int(pb[k]), int(pu[k]), int(pt[k])
            own_ = (int(w.engaged_target_slot[b_, u_]), int(w.order_target_slot[b_, u_]))
            aim_x[k], aim_y[k] = _pin_aim(
                sim, b_, u_, t_, float(dxu[k]), float(dyu[k]), bool(right[k]), pm[b_], own_
            )
    capk, _ = first_min(fol_on, capv)
    vx = seek_x[ub, uu].copy()
    vy = seek_y[ub, uu].copy()
    spd_u = spd[starts]
    hb = best >= 0
    with np.errstate(invalid='ignore', divide='ignore'):
        bi = np.where(hb, best, 0)
        ax, ay = aim_x[bi] - ux[starts], aim_y[bi] - uy[starts]
        nrm = np.hypot(ax, ay)
        turn = hb & (nrm > 0)
        vx = np.where(turn, (ax / nrm * spd_u).astype(np.float32), vx)
        vy = np.where(turn, (ay / nrm * spd_u).astype(np.float32), vy)
        if cfg.avoid_share == 'speed':  # scale turn angle against moving neighbors
            sb = turn & (cross_on | (pr_on & ~pin_on))[bi]
            ph, pc = _share_powers(sim)
            vu_, vt_ = sim.params.speed[ub, uu], sim.params.speed[ub, pt[bi]]
            g = _share_gain(sim, vu_, vt_, ph)
            if pc != ph:  # separate cross-branch exponent (scalar p for bit-parity)
                g = np.where(cross_on[bi], _share_gain(sim, vu_, vt_, pc), g)
            sb &= g < 1.0
            if sb.any():
                sx, sy = seek_x[ub[sb], uu[sb]], seek_y[ub[sb], uu[sb]]
                vx[sb], vy[sb] = _share_turn(sx, sy, ax[sb], ay[sb], spd_u[sb], g[sb])
        cap = np.where(capk >= 0, capv[np.where(capk >= 0, capk, 0)], np.inf)
        mag = np.hypot(vx, vy)
        lim = (cap < spd_u) & (mag > 0)
        cap32 = cap.astype(np.float32)
        vx = np.where(lim, vx / mag * cap32, vx)
        vy = np.where(lim, vy / mag * cap32, vy)
    ok = ~unit_bad
    out_x[ub[ok], uu[ok]] = vx[ok]
    out_y[ub[ok], uu[ok]] = vy[ok]
    return [(int(b), int(u)) for b, u in zip(ub[unit_bad], uu[unit_bad], strict=True)]


def _avoid_candidates(sim: Sim, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Return `[b, u, t]` superset of neighbors ahead within `avoid_range` (with 1e-6 margin);
    pairs outside this mask are guaranteed skipped by pairwise avoidance.
    """
    w, buf, cfg = sim.world, sim.buf, sim.config
    mover = w.alive & ~w.is_flying & ~np.isnan(buf.waypoint_x)
    with np.errstate(invalid='ignore', divide='ignore'):
        spd = np.hypot(vx, vy).astype(np.float64)
        mover &= spd != 0
        dxu = vx.astype(np.float64) / spd
        dyu = vy.astype(np.float64) / spd
        x, y = w.x.astype(np.float64), w.y.astype(np.float64)
        rx = x[:, None, :] - x[:, :, None]
        ry = y[:, None, :] - y[:, :, None]
        ahead = rx * dxu[:, :, None] + ry * dyu[:, :, None]
        dist = np.hypot(rx, ry)
        contact = (w.radius[:, :, None] + w.radius[:, None, :]).astype(
            np.float64
        ) - cfg.contact_slack
        ok = (ahead > -1e-6) & (dist - contact < cfg.avoid_range + 1e-6)
    ok &= mover[:, :, None] & (w.alive & ~w.is_flying)[:, None, :]
    ok &= ~np.eye(w.x.shape[1], dtype=np.bool_)[None]
    if cfg.pin_engaged and cfg.pin_slide:
        # `pin_slide`: exclude own engaged/order target from avoidance (scoped by mode & orders).
        n = w.x.shape[1]
        pm = pinned_mask(sim) if cfg.pin_slide_mode == 'pinned' else None
        by = None
        if cfg.pin_slide_orders == 'attack_unit':
            by = w.order_kind == CommandKind.ATTACK_UNIT
        for slot in (w.engaged_target_slot, w.order_target_slot):
            s = slot.astype(np.int64)
            has = s != NO_SLOT
            if by is not None:
                has &= by
            if pm is not None:
                has &= np.take_along_axis(pm, np.where(has, s, 0) % n, axis=1)
            own = np.zeros(ok.shape, np.bool_)
            np.put_along_axis(own, np.where(has, s, 0)[:, :, None] % n, has[:, :, None], axis=2)
            ok &= ~own
    return ok


def _side_blocked_many(
    sim: Sim, b: np.ndarray, u: np.ndarray, dx: np.ndarray, dy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Batched `_side_blocked`: returns `(right_blocked, left_blocked)` bit-identically."""
    w, buf, cfg = sim.world, sim.buf, sim.config
    f64 = np.float64
    ok = w.alive[b] & ~w.is_flying[b] & ~np.isnan(buf.waypoint_x[b])  # [K, T]
    ok &= w.owner[b] == w.owner[b, u][:, None]
    ok[np.arange(len(b)), u] = False
    with np.errstate(invalid='ignore', divide='ignore'):
        rx = w.x[b].astype(f64) - w.x[b, u].astype(f64)[:, None]
        ry = w.y[b].astype(f64) - w.y[b, u].astype(f64)[:, None]
        dxc, dyc = dx[:, None], dy[:, None]
        lat = dxc * ry - dyc * rx
        along = rx * dxc + ry * dyc
        reach = (w.radius[b, u][:, None] + w.radius[b]).astype(f64)
        near = np.abs(along) < reach
        lim = reach + cfg.avoid_side_gap
        gx = (buf.waypoint_x[b] - w.x[b]).astype(f64)
        gy = (buf.waypoint_y[b] - w.y[b]).astype(f64)
        gn = np.hypot(gx, gy)
        par = (gn > 0) & ((dxc * gx + dyc * gy) / gn > PARALLEL_COS)
    base = ok & near & par
    rs, ls = -lat, lat
    rb = (base & (0 < rs) & (rs < lim)).any(axis=1)
    lb = (base & (0 < ls) & (ls < lim)).any(axis=1)
    return rb, lb


def _side_blocked(sim: Sim, b: int, u: int, dx: float, dy: float, *, right: bool) -> bool:
    """Return True if an adjacent parallel-moving ground ally occupies the right (or left) side
    of `u` within `(0, radii_sum + avoid_side_gap)` laterally and `+/-radii_sum` longitudinally.
    """
    w, buf, cfg = sim.world, sim.buf, sim.config
    ux, uy = float(w.x[b, u]), float(w.y[b, u])
    for t in range(sim.n_slots):
        if t == u or not w.alive[b, t] or w.is_flying[b, t] or np.isnan(buf.waypoint_x[b, t]):
            continue
        if w.owner[b, t] != w.owner[b, u]:
            continue
        rx, ry = float(w.x[b, t]) - ux, float(w.y[b, t]) - uy
        lat = dx * ry - dy * rx  # positive = left
        along = rx * dx + ry * dy
        reach = float(w.radius[b, u] + w.radius[b, t])
        side = -lat if right else lat
        if not (0 < side < reach + cfg.avoid_side_gap and abs(along) < reach):
            continue
        gx = float(buf.waypoint_x[b, t] - w.x[b, t])
        gy = float(buf.waypoint_y[b, t] - w.y[b, t])
        gn = float(np.hypot(gx, gy))
        if gn > 0 and (dx * gx + dy * gy) / gn > PARALLEL_COS:
            return True
    return False


# ---------------------------------------------------------------- contact_orbit

ORBIT_EPS = 1e-9  # constraint violation (tiles) and angular interval (rad) tolerance
ORBIT_TIE = 0.01  # cross-product tie threshold for dead-ahead blocker (defaults to right turn)
ORBIT_PREV_MIN = 0.02  # min previous step (tiles) for `orbit_from='prev'` before falling back
ORBIT_RIGHT_MAX = math.pi / 2  # max extra clockwise rotation for `orbit_movers='right'`


def _orbit(sim: Sim, seek_x: np.ndarray, seek_y: np.ndarray, out_x: np.ndarray, out_y: np.ndarray):
    """`contact_orbit`: rotate blocked moving units' desired velocity (`out_x`, `out_y`) out of
    pinned contact cones along a held turn side, updating `motion.orbit_*`.

    Batched row/column-compressed implementation bit-identical to `_orbit_ref` / `_orbit_unit`.
    """
    w, m, cfg = sim.world, sim.motion, sim.config
    f64 = np.float64
    disp = _orbit_disp(sim)
    vx, vy = out_x.astype(f64), out_y.astype(f64)
    spd = np.hypot(vx, vy)
    pin = pinned_mask(sim)
    mover = w.alive & ~w.is_flying & ~np.isnan(sim.buf.waypoint_x) & (spd > 0)
    bi, ui = np.nonzero(mover & pin.any(axis=1)[:, None])
    if bi.size == 0:
        return
    with np.errstate(invalid='ignore', divide='ignore'):
        rx = w.x[bi].astype(f64) - w.x[bi, ui].astype(f64)[:, None]
        ry = w.y[bi].astype(f64) - w.y[bi, ui].astype(f64)[:, None]
        dist = np.hypot(rx, ry)
        contact = (w.radius[bi, ui][:, None] + w.radius[bi]).astype(f64) - cfg.contact_slack
        gap = dist - contact
        su = spd[bi, ui]
        near = (dist > 0) & (gap < su[:, None])
    pob = near & pin[bi] & ~_orbit_own(sim, bi, ui)
    rows = np.flatnonzero(pob.any(axis=1))
    if rows.size == 0:
        return
    bi, ui, su = bi[rows], ui[rows], su[rows]
    rx, ry, dist, gap, near, pob = rx[rows], ry[rows], dist[rows], gap[rows], near[rows], pob[rows]
    vu, wu = vx[bi, ui], vy[bi, ui]
    with np.errstate(invalid='ignore', divide='ignore'):
        nx, ny = rx / dist, ry / dist
        allow = np.maximum(gap, 0.0)
        viol = vu[:, None] * nx + wu[:, None] * ny - allow
    blocked = (pob & (viol > ORBIT_EPS)).any(axis=1)
    eng = w.engaged_target_slot[bi, ui]
    tgt = np.where(eng != NO_SLOT, eng, w.order_target_slot[bi, ui])
    loop = w.game_loop[bi]
    side0 = m.orbit_side[bi, ui]
    held = (side0 != 0) & (loop - m.orbit_loop[bi, ui] <= cfg.orbit_hold)
    held &= m.orbit_target[bi, ui] == tgt
    sx, sy = seek_x[bi, ui].astype(f64), seek_y[bi, ui].astype(f64)
    wrong = np.zeros(bi.size, np.bool_)
    if cfg.orbit_lock:  # velocity fell onto the opposite side of seek while holding side
        wrong = held & ~blocked & ((sx * wu - sy * vu) * side0 < -ORBIT_EPS)
    rows = np.flatnonzero(blocked | wrong)
    if rows.size == 0:
        return
    bi, ui, su, vu, wu, sx, sy = (a[rows] for a in (bi, ui, su, vu, wu, sx, sy))
    nx, ny, allow, viol, near, pob = (a[rows] for a in (nx, ny, allow, viol, near, pob))
    eng, tgt, loop, side0, held, wrong = (a[rows] for a in (eng, tgt, loop, side0, held, wrong))
    # Compress columns per row to neighbors with `edge < speed` (padded columns have near=False).
    col = np.argsort(~near, axis=1, kind='stable')[:, : near.sum(axis=1).max()]
    nx, ny, allow, viol, near, pob = (
        np.take_along_axis(a, col, axis=1) for a in (nx, ny, allow, viol, near, pob)
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        # Reference heading `phi0` and unit vector `(dx, dy)` used to pick turn side.
        phi0 = np.arctan2(wu, vu)
        dx, dy = vu / su, wu / su
        if disp is not None:
            mx, my = disp[0][bi, ui], disp[1][bi, ui]
            mm = np.hypot(mx, my)
            hp = mm > ORBIT_PREV_MIN
            if cfg.orbit_from == 'prev_free':
                hp &= eng == NO_SLOT
            phi0 = np.where(hp, np.arctan2(my, mx), phi0)
            dx, dy = np.where(hp, mx / mm, dx), np.where(hp, my / mm, dy)
        phi0 = np.where(wrong, np.arctan2(sy, sx), phi0)
        # Turn side: keep held side, or turn away from the most-violated pinned blocker.
        k = np.argmax(np.where(pob, viol, -np.inf), axis=1)
        r = np.arange(bi.size)
        lat = dx * ny[r, k] - dy * nx[r, k]
        side = np.where(held, side0, np.where(lat < -ORBIT_TIE, 1, -1)).astype(np.int8)
        sgn = side.astype(f64)
        alpha = np.arctan2(ny, nx)
        h = np.arccos(np.clip(allow / su[:, None], -1.0, 1.0))
        phi, turned = _orbit_turn_many(phi0, np.zeros(bi.size), sgn, pob, alpha, h, cfg.orbit_iters)
        if cfg.orbit_movers != 'none':
            mob = near & np.take_along_axis(mover[bi], col, axis=1) & ~pob
            tvn = np.take_along_axis(vx[bi], col, axis=1) * nx
            tvn = tvn + np.take_along_axis(vy[bi], col, axis=1) * ny
            h2 = np.where(
                mob, np.arccos(np.clip((allow + np.maximum(tvn, 0.0)) / su[:, None], -1.0, 1.0)), h
            )
            obs2 = pob | mob
            sgn2 = np.full(bi.size, -1.0) if cfg.orbit_movers == 'right' else sgn
            phi2, turned2 = _orbit_turn_many(phi, turned, sgn2, obs2, alpha, h2, cfg.orbit_iters)
            d2 = np.remainder(phi2[:, None] - alpha + np.pi, 2 * np.pi) - np.pi
            good = ~(obs2 & (np.abs(d2) < h2 - ORBIT_EPS)).any(axis=1)
            good &= turned2 < cfg.orbit_max_turn
            if cfg.orbit_movers == 'right':
                good &= turned2 - turned < ORBIT_RIGHT_MAX
            phi, turned = np.where(good, phi2, phi), np.where(good, turned2, turned)
        go = turned < cfg.orbit_max_turn
        out_x[bi, ui] = np.where(go, np.cos(phi) * su, 0.0).astype(np.float32)
        out_y[bi, ui] = np.where(go, np.sin(phi) * su, 0.0).astype(np.float32)
    m.orbit_side[bi, ui] = side
    m.orbit_loop[bi, ui] = loop
    m.orbit_target[bi, ui] = tgt


def _orbit_disp(sim: Sim) -> tuple[np.ndarray, np.ndarray] | None:
    """Return previous-loop displacement `(dx, dy)` for `orbit_from='prev'/'prev_free'` (and
    record current positions), or None for `'steer'`.
    """
    w, m = sim.world, sim.motion
    if sim.config.orbit_from == 'steer':
        return None
    dx = w.x.astype(np.float64) - m.orbit_px.astype(np.float64)
    dy = w.y.astype(np.float64) - m.orbit_py.astype(np.float64)
    m.orbit_px[:] = w.x
    m.orbit_py[:] = w.y
    return dx, dy


def _orbit_own(sim: Sim, bi: np.ndarray, ui: np.ndarray) -> np.ndarray:
    """Return `[N, U]` mask marking each row's own engaged and order targets."""
    w = sim.world
    own = np.zeros((bi.size, w.x.shape[1]), np.bool_)
    r = np.arange(bi.size)
    for slot in (w.engaged_target_slot, w.order_target_slot):
        s = slot[bi, ui]
        has = s != NO_SLOT
        own[r[has], s[has]] = True
    return own


def _orbit_turn_many(phi, turned, sgn, obs, alpha, h, iters: int):
    """Batched `_orbit_turn`: rotate `phi` along `sgn` out of active forbidden cones `obs`."""
    cols = np.flatnonzero(obs.any(axis=0))
    if cols.size == 0:
        return phi, turned
    obs, alpha, h = obs[:, cols], alpha[:, cols], h[:, cols]
    hi = h - ORBIT_EPS
    phi, turned = phi.copy(), turned.copy()
    act = np.arange(phi.size)
    for _ in range(iters):
        p, s = phi[act], sgn[act]
        a, hh = alpha[act], h[act]
        d = np.remainder(p[:, None] - a + np.pi, 2 * np.pi) - np.pi
        inside = obs[act] & (np.abs(d) < hi[act])
        step = np.where(inside, np.where(s[:, None] > 0, hh - d, hh + d), 0.0).max(axis=1)
        go = step > 0
        if not go.any():
            break
        act, step = act[go], step[go]
        phi[act] = p[go] + s[go] * step
        turned[act] = turned[act] + step
    return phi, turned


def _orbit_inside(phi: float, alpha: float, h: float) -> tuple[bool, float]:
    """Return `(inside, wrapped_diff)` for whether `phi` lies inside `|phi - alpha| < h`."""
    d = float(np.remainder(phi - alpha + np.pi, 2 * np.pi)) - np.pi
    return abs(d) < h - ORBIT_EPS, d


def _orbit_turn(
    phi: float, turned: float, sgn: float, cons: list[tuple[float, float]], iters: int
) -> tuple[float, float]:
    """Rotate `phi` along `sgn` (+1 CCW, -1 CW) past all forbidden cones `(alpha, h)` up to
    `iters` iterations, returning `(new_phi, total_turned)`.
    """
    for _ in range(iters):
        step = 0.0
        for a, h in cons:
            inside, d = _orbit_inside(phi, a, h)
            if inside:
                step = max(step, h - d if sgn > 0 else h + d)
        if not step > 0:
            break
        phi, turned = phi + sgn * step, turned + step
    return phi, turned


def _orbit_unit(sim: Sim, b: int, u: int, vel, seek, pin: np.ndarray, disp):
    """Per-unit reference implementation of `contact_orbit` (batched by `_orbit`).

    Returns `(new_vx, new_vy, side, target)` or None if `u` needs no adjustment.
    """
    w, m, cfg = sim.world, sim.motion, sim.config
    vx, vy = vel

    def moving(t: int) -> bool:
        if not w.alive[b, t] or w.is_flying[b, t] or np.isnan(sim.buf.waypoint_x[b, t]):
            return False
        return float(np.hypot(float(vx[b, t]), float(vy[b, t]))) > 0

    if not moving(u):
        return None
    vu, wu = float(vx[b, u]), float(vy[b, u])
    su = float(np.hypot(vu, wu))
    xu, yu = float(w.x[b, u]), float(w.y[b, u])
    eng, otg = int(w.engaged_target_slot[b, u]), int(w.order_target_slot[b, u])
    own = {eng, otg} - {NO_SLOT}
    pins: list[tuple[float, float, float, float]] = []  # (nx, ny, allow, viol) in slot order
    movs: list[tuple[float, float, float]] = []  # (nx, ny, allow)
    for t in range(sim.n_slots):
        if t == u:
            continue
        rx, ry = float(w.x[b, t]) - xu, float(w.y[b, t]) - yu
        dist = float(np.hypot(rx, ry))
        gap = dist - (float(w.radius[b, u] + w.radius[b, t]) - cfg.contact_slack)
        if not (dist > 0 and gap < su):
            continue
        nx, ny = rx / dist, ry / dist
        allow = float(np.maximum(gap, 0.0))
        if pin[b, t] and t not in own:
            pins.append((nx, ny, allow, vu * nx + wu * ny - allow))
        elif moving(t):
            tvn = float(vx[b, t]) * nx + float(vy[b, t]) * ny
            movs.append((nx, ny, allow + float(np.maximum(tvn, 0.0))))
    if not pins:
        return None
    blocked = any(v > ORBIT_EPS for *_, v in pins)
    tgt = eng if eng != NO_SLOT else otg
    side0 = int(m.orbit_side[b, u])
    held = (
        side0 != 0
        and int(w.game_loop[b]) - int(m.orbit_loop[b, u]) <= cfg.orbit_hold
        and int(m.orbit_target[b, u]) == tgt
    )
    sx, sy = float(seek[0][b, u]), float(seek[1][b, u])
    wrong = cfg.orbit_lock and held and not blocked and (sx * wu - sy * vu) * side0 < -ORBIT_EPS
    if not (blocked or wrong):
        return None
    phi, dx, dy = float(np.arctan2(wu, vu)), vu / su, wu / su
    if disp is not None:
        mx, my = float(disp[0][b, u]), float(disp[1][b, u])
        mm = float(np.hypot(mx, my))
        if mm > ORBIT_PREV_MIN and (cfg.orbit_from == 'prev' or eng == NO_SLOT):
            phi, dx, dy = float(np.arctan2(my, mx)), mx / mm, my / mm
    if wrong:
        phi = float(np.arctan2(sy, sx))
    kx, ky, worst = 0.0, 0.0, -np.inf
    for nx, ny, _, v in pins:  # most-violated blocker (lowest slot on tie)
        if v > worst:
            kx, ky, worst = nx, ny, v
    side = side0 if held else (1 if dx * ky - dy * kx < -ORBIT_TIE else -1)
    sgn = float(side)

    def cone(nx: float, ny: float, allow: float) -> tuple[float, float]:
        return float(np.arctan2(ny, nx)), float(np.arccos(np.clip(allow / su, -1.0, 1.0)))

    cons = [cone(nx, ny, a) for nx, ny, a, _ in pins]
    phi, turned = _orbit_turn(phi, 0.0, sgn, cons, cfg.orbit_iters)
    if cfg.orbit_movers != 'none':
        cons2 = cons + [cone(nx, ny, a) for nx, ny, a in movs]
        sgn2 = -1.0 if cfg.orbit_movers == 'right' else sgn
        phi2, turned2 = _orbit_turn(phi, turned, sgn2, cons2, cfg.orbit_iters)
        good = not any(_orbit_inside(phi2, a, h)[0] for a, h in cons2)
        good = good and turned2 < cfg.orbit_max_turn
        if cfg.orbit_movers == 'right':
            good = good and turned2 - turned < ORBIT_RIGHT_MAX
        if good:
            phi, turned = phi2, turned2
    if not turned < cfg.orbit_max_turn:
        return f32(0), f32(0), side, tgt
    return f32(np.cos(phi) * su), f32(np.sin(phi) * su), side, tgt


def _orbit_ref(
    sim: Sim, seek_x: np.ndarray, seek_y: np.ndarray, out_x: np.ndarray, out_y: np.ndarray
) -> None:
    """Reference per-unit wrapper around `_orbit_unit` (bit-identical to `_orbit`, for tests)."""
    w, m = sim.world, sim.motion
    disp = _orbit_disp(sim)
    vel = (out_x.astype(np.float64), out_y.astype(np.float64))
    pin = pinned_mask(sim)
    res = [
        (b, u, _orbit_unit(sim, b, u, vel, (seek_x, seek_y), pin, disp))
        for b in range(sim.batch_size)
        for u in range(sim.n_slots)
    ]
    for b, u, r in res:
        if r is not None:
            out_x[b, u], out_y[b, u], m.orbit_side[b, u], m.orbit_target[b, u] = r
            m.orbit_loop[b, u] = w.game_loop[b]


IMPLS = {'none': none, 'separation': separation, 'sc2_avoid': sc2_avoid}

__all__ = [
    'IMPLS',
    'MARAUDER',
    'ORBIT_EPS',
    'ORBIT_PREV_MIN',
    'ORBIT_RIGHT_MAX',
    'ORBIT_TIE',
    'PARALLEL_COS',
    'PIN_MIRROR_TOL',
    'PIN_TIE_COST',
    'SIDESTEP_DRIFT_TIE',
    'SIDESTEP_RANGE',
    'SIDESTEP_TIE',
    'TIE_EPS',
    'none',
    'sc2_avoid',
    'separation',
]
