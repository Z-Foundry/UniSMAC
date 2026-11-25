"""Air-unit idle separation, moving flocking, and contact commit for `native`."""

from __future__ import annotations

import numpy as np

from unismac.contract import CommandKind

from ...sim import Sim, f32
from .avoid import lateral_clamp_q12
from .q12 import (
    AIR_STEP_FLOOR,
    BLEND_NEW,
    BLEND_OLD,
    OLD_DIR_MIN,
    ONE,
    QUANT_PAD,
    _trunc_div_int,
    _unit_accel_q12,
    air_height,
    d2q,
    dot_q12,
    facing_vec_q12,
    has_air_collide,
    is_air_only,
    lateral_accel_q12,
    norm,
    q12,
    quant_key,
    sep_radius_q12,
)


def air_idle_sep_step(
    px: int,
    py: int,
    vx: int,
    vy: int,
    s_max: int,
    sep_a: int,
    facing_a: float,
    slot_id_a: int,
    cd: int,
    air_only: bool,
    neighbors: list[tuple[int, int, int, float, int]],
    *,
    state: int = 0,
    goal: tuple[int, int] | None = None,
    r_allow: int = 0,
) -> tuple[int, int, int]:
    """Idle/arrived pure air-unit throttled separation.

    `neighbors` is pre-sorted by ascending quantized distance then descending slot ID, with
    entries `(x_b, y_b, sep_b, facing_b, slot_id_b)`. Returns `(out_vx, out_vy, new_cd)`.
    """
    max_sep = _trunc_div_int(s_max, 4)
    if not air_only:
        if state != 5 and (vx != 0 or vy != 0):
            vnx, vny, old_len = norm(vx, vy)
            new_len = max(min(old_len, s_max - max_sep) - max_sep, 0)
            return (vnx * new_len) >> 12, (vny * new_len) >> 12, cd
        return vx, vy, cd
    step = max(_trunc_div_int(s_max, 16), min(AIR_STEP_FLOOR, max_sep))
    cd += 1
    if cd >= 9:
        for xb, yb, sep_b, facing_b, slot_id_b in neighbors:
            nx, ny, dist = norm(px - xb, py - yb)
            if dist >= sep_a + sep_b:
                continue
            if dist == 0:
                if slot_id_a < slot_id_b:
                    nx, ny = facing_vec_q12(facing_a)
                else:
                    fx, fy = facing_vec_q12(facing_b)
                    nx, ny = -fx, -fy
            vcx = vx + ((nx * step) >> 12)
            vcy = vy + ((ny * step) >> 12)
            if vcx * vcx + vcy * vcy > max_sep * max_sep:
                vnx, vny, _ = norm(vcx, vcy)
                vcx = (vnx * max_sep) >> 12
                vcy = (vny * max_sep) >> 12
            if state >= 3 and goal is not None:
                gx, gy = goal
                if (px + vcx - gx) ** 2 + (py + vcy - gy) ** 2 > (px - gx) ** 2 + (py - gy) ** 2:
                    if (px + 4 * vcx - gx) ** 2 + (py + 4 * vcy - gy) ** 2 > r_allow * r_allow:
                        continue
            return vcx, vcy, cd
    if vx != 0 or vy != 0:
        vnx, vny, old_len = norm(vx, vy)
        new_len = max(min(old_len, s_max - step) - step, 0)
        return (vnx * new_len) >> 12, (vny * new_len) >> 12, cd
    return 0, 0, cd


def air_flock_step(
    px: int,
    py: int,
    gx: int,
    gy: int,
    vx_a: int,
    vy_a: int,
    mx: int,
    my: int,
    s: int,
    sep_a: int,
    height_a: float,
    accel_q12: int,
    lat_a_q12: int,
    slot_id_a: int,
    # Neighbor tuple: (xb, yb, gxb, gyb, vxb, vyb, smax_b, sep_b, height_b, slot_id_b)
    neighbors: list[tuple[int, int, int, int, int, int, int, int, float, int]],
) -> tuple[int, int]:
    """Moving air-unit flocking separation blending `N(8*m + N(sep_out))`."""
    if vx_a == 0 and vy_a == 0:
        return vx_a, vy_a
    va2 = dot_q12(vx_a, vy_a, vx_a, vy_a)
    count = 0
    sum_vbx = 0
    sum_vby = 0
    push_x = 0
    push_y = 0
    for xb, yb, gxb, gyb, vxb, vyb, smax_b, sep_b, height_b, slot_id_b in neighbors:
        if count >= 8:
            break
        if height_b != height_a:
            continue
        if dot_q12(vx_a, vy_a, xb - px, yb - py) < 0:
            continue
        if 50 * d2q(gx, gy, gxb, gyb) > d2q(gx, gy, px, py) + d2q(gxb, gyb, xb, yb):
            continue
        sep_sum = sep_a + sep_b
        ab_nx, ab_ny, dist = norm(xb - px, yb - py)
        if dist >= sep_sum:
            continue
        if (gx - gxb) ** 2 + (gy - gyb) ** 2 >= sep_sum * sep_sum:
            continue
        if 2 * dot_q12(vx_a, vy_a, vxb, vyb) < va2:
            continue
        count += 1
        sum_vbx += vxb
        sum_vby += vyb
        if dist > 0:
            mag = (sep_sum * smax_b) >> 12
            push_x -= (ab_nx * mag) >> 12
            push_y -= (ab_ny * mag) >> 12
        else:
            hx = _trunc_div_int(vx_a, 4)
            hy = _trunc_div_int(vy_a, 4)
            if slot_id_b >= slot_id_a:
                push_x -= hx
                push_y += hy
            else:
                push_x += hx
                push_y -= hy
    if count == 0:
        return vx_a, vy_a
    dx = _trunc_div_int(sum_vbx, count) + push_x - vx_a
    dy = _trunc_div_int(sum_vby, count) + push_y - vy_a
    if dot_q12(dx, dy, dx, dy) <= 0 and dx == 0 and dy == 0:
        return vx_a, vy_a
    qx, qy, _ = norm(vx_a, vy_a)
    long_v = dot_q12(qx, qy, dx, dy)
    lat_v = dot_q12(-qy, qx, dx, dy)
    max_long = _trunc_div_int(accel_q12, 2)
    max_lat = _trunc_div_int(lat_a_q12, 2)
    long_v = max(-max_long, min(max_long, long_v))
    lat_v = max(-max_lat, min(max_lat, lat_v))
    sep_out_x = ((qx * long_v) >> 12) + (((-qy) * lat_v) >> 12)
    sep_out_y = ((qy * long_v) >> 12) + ((qx * lat_v) >> 12)
    n_sepx, n_sepy, n_seplen = norm(sep_out_x, sep_out_y)
    if n_seplen == 0:
        return vx_a, vy_a
    fx, fy, _ = norm(8 * mx + n_sepx, 8 * my + n_sepy)
    return (fx * s) >> 12, (fy * s) >> 12


def _steer_air_env(
    sim: Sim,
    b: int,
    px_all: np.ndarray,
    py_all: np.ndarray,
    r_all: np.ndarray,
    smax: np.ndarray,
    has_wp: np.ndarray,
    is_lag: np.ndarray,
) -> None:
    """Moving pure air-unit flocking (`state > 5`); idle air units separate during contact."""
    w, buf, m, cfg = sim.world, sim.buf, sim.motion, sim.config
    info = sim.env_map[b]
    map_hi_x = ((info.width << 12) - 1) if info is not None else int(q12(sim.playable[b])[2]) - 1
    map_hi_y = ((info.height << 12) - 1) if info is not None else int(q12(sim.playable[b])[3]) - 1

    air_coll = [
        u
        for u in np.flatnonzero(w.alive[b]).tolist()
        if has_air_collide(int(w.unit_type[b, u]), bool(w.is_flying[b, u]))
    ]
    if not air_coll:
        return
    # Snapshot all units carrying the Air collision bit.
    cur_vx = {u: int(m.native_vx[b, u]) for u in air_coll}
    cur_vy = {u: int(m.native_vy[b, u]) for u in air_coll}
    for u in air_coll:
        ut = int(w.unit_type[b, u])
        if not is_air_only(ut, bool(w.is_flying[b, u])):
            continue
        if not has_wp[b, u]:
            # Drop to state 3 for _air_idle_contact once speed==0 so accel coasting continues.
            buf.vel_x[b, u] = buf.vel_y[b, u] = f32(0)
            continue
        lag_u = bool(is_lag[b, u])
        px, py = int(px_all[b, u]), int(py_all[b, u])
        gx = int(q12(buf.waypoint_x[b, u]))
        gy = int(q12(buf.waypoint_y[b, u]))
        r_a = int(r_all[b, u])
        s_max = int(smax[b, u])
        v_old_x, v_old_y = cur_vx[u], cur_vy[u]
        ux, uy, ulen = norm(v_old_x, v_old_y)
        if ulen <= OLD_DIR_MIN:
            ux, uy = 0, 0
        nx, ny, length = norm(gx - px, gy - py)
        st_in = int(m.native_state[b, u])
        if length > r_a + s_max and ulen > OLD_DIR_MIN and st_in != 8 and cfg.native_blend:
            mx, my, _ = norm(BLEND_NEW * nx + BLEND_OLD * ux, BLEND_NEW * ny + BLEND_OLD * uy)
        else:
            mx, my = nx, ny
        if length <= s_max:
            vax, vay = gx - px, gy - py
        else:
            vax, vay = (mx * s_max) >> 12, (my * s_max) >> 12
        # Sort candidate air neighbors by quantized distance.
        slot_id_a = int(m.native_slot[b, u]) if int(m.native_slot[b, u]) >= 0 else u
        h_a = air_height(ut, True)
        sep_a = sep_radius_q12(ut)
        acc_q = _unit_accel_q12(sim, b, u)
        lat_a = lateral_accel_q12(ut, acc_q)
        nbr_keyed: list[tuple[int, int, int]] = []
        for nb in air_coll:
            if nb == u:
                continue
            sid_b = int(m.native_slot[b, nb]) if int(m.native_slot[b, nb]) >= 0 else nb
            k_d = quant_key(int(px_all[b, nb]), int(py_all[b, nb]), px, py)
            lim = ((int(r_all[b, nb]) + r_a + QUANT_PAD) ** 2) >> 12
            if k_d <= lim:
                nbr_keyed.append((k_d, -sid_b, nb))
        nbr_keyed.sort()
        nbr_tuples: list[tuple[int, int, int, int, int, int, int, int, float, int]] = []
        for _, _, nb in nbr_keyed:
            ut_b = int(w.unit_type[b, nb])
            gxb = (
                int(q12(buf.waypoint_x[b, nb]))
                if has_wp[b, nb]
                else (int(m.native_gx[b, nb]) if m.native_has_g[b, nb] else int(px_all[b, nb]))
            )
            gyb = (
                int(q12(buf.waypoint_y[b, nb]))
                if has_wp[b, nb]
                else (int(m.native_gy[b, nb]) if m.native_has_g[b, nb] else int(py_all[b, nb]))
            )
            sid_b = int(m.native_slot[b, nb]) if int(m.native_slot[b, nb]) >= 0 else nb
            nbr_tuples.append(
                (
                    int(px_all[b, nb]),
                    int(py_all[b, nb]),
                    gxb,
                    gyb,
                    cur_vx[nb],
                    cur_vy[nb],
                    int(smax[b, nb]),
                    sep_radius_q12(ut_b),
                    air_height(ut_b, bool(w.is_flying[b, nb])),
                    sid_b,
                )
            )
        flocked = False
        if nbr_tuples and length > s_max:
            fx, fy = air_flock_step(
                px,
                py,
                gx,
                gy,
                vax,
                vay,
                mx,
                my,
                s_max,
                sep_a,
                h_a,
                acc_q,
                lat_a,
                slot_id_a,
                nbr_tuples,
            )
            if fx != vax or fy != vay:
                vax, vay = fx, fy
                flocked = True
        if flocked:
            if length > r_a + s_max and (ux != 0 or uy != 0):
                vax, vay = lateral_clamp_q12(vax, vay, v_old_x, v_old_y, ux, uy, s_max, lat_a)
            # Full-map rectangular clamp.
            clx = min(max(px + vax, 0), map_hi_x)
            cly = min(max(py + vay, 0), map_hi_y)
            vax, vay = clx - px, cly - py
            if not lag_u:
                cur_vx[u], cur_vy[u] = vax, vay
                m.native_state[b, u] = 9
                m.native_air_cd[b, u] = 0
            buf.vel_x[b, u] = f32(vax / ONE)
            buf.vel_y[b, u] = f32(vay / ONE)
        elif not lag_u:
            # Without flock repulsion, keep raw steering.seek velocity for accel compatibility.
            m.native_state[b, u] = 9
            m.native_air_cd[b, u] = 0


def _air_idle_contact(
    sim: Sim,
    x0: np.ndarray,
    y0: np.ndarray,
    smax: np.ndarray,
    moving_m: np.ndarray,
) -> None:
    """Idle pure air-unit separation and displacement commit for all flying units."""
    from .steer import _unit_target_slots

    w, m = sim.world, sim.motion
    fly_m = w.alive & w.is_flying
    if not fly_m.any():
        return
    r_all = q12(w.radius)
    tgt_all = _unit_target_slots(sim)
    for b in np.flatnonzero(fly_m.any(axis=1)).tolist():
        air_coll = [
            u
            for u in np.flatnonzero(w.alive[b]).tolist()
            if has_air_collide(int(w.unit_type[b, u]), bool(w.is_flying[b, u]))
        ]
        # Snapshot prior positions and states.
        px_map = {u: int(x0[b, u]) for u in air_coll}
        py_map = {u: int(y0[b, u]) for u in air_coll}
        tgt_row = tgt_all[b]
        info = sim.env_map[b]
        pl_q = q12(sim.playable[b])
        pl_x0, pl_y0, pl_x1, pl_y1 = int(pl_q[0]), int(pl_q[1]), int(pl_q[2]), int(pl_q[3])
        mw = ((info.width << 12) - 1) if info is not None else pl_x1 - 1
        mh = ((info.height << 12) - 1) if info is not None else pl_y1 - 1
        for u in air_coll:
            if not w.is_flying[b, u]:
                continue
            if moving_m[b, u] or float(m.speed[b, u]) > 0.0:
                # Moving or accel-coasting air units: keep kinematics integration and Q12 step.
                m.native_vx[b, u] = int(q12(w.x[b, u])) - px_map[u]
                m.native_vy[b, u] = int(q12(w.y[b, u])) - py_map[u]
                m.native_state[b, u] = 9
                m.native_air_cd[b, u] = 0
                continue
            ut = int(w.unit_type[b, u])
            air_only = is_air_only(ut, True)
            st = int(m.native_state[b, u])
            if st > 5:
                # Just stopped: zero leftover velocity so idle separation starts cleanly.
                st = 3
                m.native_state[b, u] = 3
                m.native_vx[b, u] = 0
                m.native_vy[b, u] = 0
            px, py = px_map[u], py_map[u]
            sid_a = int(m.native_slot[b, u]) if int(m.native_slot[b, u]) >= 0 else u
            r_a = int(r_all[b, u])
            nbr_keyed: list[tuple[int, int, int]] = []
            if air_only:
                owner_a = int(w.owner[b, u])
                for nb in air_coll:
                    if nb == u or int(w.owner[b, nb]) != owner_a:
                        continue
                    if moving_m[b, nb] or int(m.native_state[b, nb]) > 5:
                        continue
                    if int(tgt_row[nb]) == u:
                        continue
                    k_d = quant_key(px_map[nb], py_map[nb], px, py)
                    lim = ((int(r_all[b, nb]) + r_a + ONE + QUANT_PAD) ** 2) >> 12
                    if k_d <= lim:
                        sid_b = int(m.native_slot[b, nb]) if int(m.native_slot[b, nb]) >= 0 else nb
                        nbr_keyed.append((k_d, -sid_b, nb))
                nbr_keyed.sort()
            nbrs = [
                (
                    px_map[nb],
                    py_map[nb],
                    sep_radius_q12(int(w.unit_type[b, nb])),
                    float(w.facing[b, nb]),
                    int(m.native_slot[b, nb]) if int(m.native_slot[b, nb]) >= 0 else nb,
                )
                for _, _, nb in nbr_keyed
            ]
            goal_pt = (
                (int(m.native_gx[b, u]), int(m.native_gy[b, u]))
                if (m.native_has_g[b, u] and int(w.order_kind[b, u]) != int(CommandKind.NOOP))
                else None
            )
            ovx, ovy, new_cd = air_idle_sep_step(
                px,
                py,
                int(m.native_vx[b, u]),
                int(m.native_vy[b, u]),
                int(smax[b, u]),
                sep_radius_q12(ut),
                float(w.facing[b, u]),
                sid_a,
                int(m.native_air_cd[b, u]),
                air_only,
                nbrs,
                state=st,
                goal=goal_pt,
                r_allow=r_a,
            )
            m.native_air_cd[b, u] = new_cd
            if ovx != 0 or ovy != 0:
                in_pl = pl_x0 <= px < pl_x1 and pl_y0 <= py < pl_y1
                nx_ = min(max(px + ovx, pl_x0 if in_pl else 0), (pl_x1 - 1) if in_pl else mw)
                ny_ = min(max(py + ovy, pl_y0 if in_pl else 0), (pl_y1 - 1) if in_pl else mh)
                w.x[b, u] = f32(nx_ / ONE)
                w.y[b, u] = f32(ny_ / ONE)
                m.native_vx[b, u] = nx_ - px
                m.native_vy[b, u] = ny_ - py
            else:
                m.native_vx[b, u] = 0
                m.native_vy[b, u] = 0


__all__ = ['_air_idle_contact', '_steer_air_env', 'air_flock_step', 'air_idle_sep_step']
