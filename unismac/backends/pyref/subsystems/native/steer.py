"""State synchronization and fixed-point Q12 steering stage (`steering='native'`)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unismac.contract import NO_SLOT, CommandKind

from ...sim import Sim, f32
from .. import steering
from .air import _steer_air_env
from .avoid import _local_steer_avoid, lateral_clamp_q12
from .q12 import (
    BLEND_NEW,
    BLEND_OLD,
    OLD_DIR_MIN,
    ONE,
    QUANT_PAD,
    _trunc_div,
    _trunc_div2,
    _trunc_div_q12,
    _unit_accel_q12,
    _unit_decel_q12,
    inner_radius_q12,
    lateral_accel_q12,
    norm,
    norm_all,
    q12,
    quant_key,
    smax_all,
)
from .stop import early_stop_step, late_progress_step, near_goal_area_step
from .wall import WallMap, _get_wall_map, project_goal_single_edge


def _unit_target_slots(sim: Sim) -> np.ndarray:
    """Current target unit slot per unit, or NO_SLOT if none."""
    w = sim.world
    kind = w.order_kind
    tgt = np.full(kind.shape, NO_SLOT, np.int32)
    by_order = (kind == CommandKind.ATTACK_UNIT) | (kind == CommandKind.ABILITY)
    tgt = np.where(by_order, w.order_target_slot, tgt)
    auto = (kind == CommandKind.ATTACK_MOVE) | (kind == CommandKind.NOOP)
    tgt = np.where(auto & (w.engaged_target_slot != NO_SLOT), w.engaged_target_slot, tgt)
    return tgt


def _sync_stationary_states(sim: Sim, ground_m: np.ndarray, has_wp: np.ndarray) -> None:
    """Update `native_state` for stationary ground units (HOLD->0, in-range->8/3, idle->3)."""
    w, m, cfg = sim.world, sim.motion, sim.config
    still = ground_m & ~has_wp
    if not still.any():
        return
    is_hold = still & (w.order_kind == CommandKind.HOLD)
    m.native_state[is_hold] = 0
    atk_in = still & ~is_hold & (w.engaged_target_slot != NO_SLOT)
    if atk_in.any():
        if cfg.pin_lag:
            m.native_state[atk_in] = np.where(m.pinned_prev[atk_in], 3, 8).astype(np.int8)
        else:
            m.native_state[atk_in] = 3
    idle = still & ~is_hold & ~atk_in
    if idle.any():
        drop = idle & (m.native_state > 4)
        m.native_state[drop] = 3


def _steer_fast_vectorized(
    sim: Sim,
    seek_x: np.ndarray,
    seek_y: np.ndarray,
    has_wp: np.ndarray,
    ground_alive: np.ndarray,
    is_lag: np.ndarray,
) -> None:
    """Fast vectorized path when advanced native steering options are disabled."""
    w, buf, m, cfg = sim.world, sim.buf, sim.motion, sim.config
    _sync_stationary_states(sim, ground_alive, has_wp)
    ground = ground_alive & has_wp
    px, py = q12(w.x), q12(w.y)
    wx, wy = q12(buf.waypoint_x), q12(buf.waypoint_y)
    has_dest = np.isfinite(buf.dest_x)
    gx = np.where(has_dest, q12(buf.dest_x), wx)
    gy = np.where(has_dest, q12(buf.dest_y), wy)
    dx, dy = wx - px, wy - py
    nx, ny, length = norm_all(dx, dy)
    s = smax_all(sim)
    final = (wx == gx) & (wy == gy) & (length <= s)
    step = s
    if cfg.native_avoid:
        steering.sc2_avoid(sim)
        av_x, av_y = buf.vel_x.copy(), buf.vel_y.copy()
        avoided = ground & ((av_x != seek_x) | (av_y != seek_y))
        anx, any_, alen = norm_all(q12(av_x), q12(av_y))
        nx, ny = np.where(avoided, anx, nx), np.where(avoided, any_, ny)
        step = np.where(avoided, np.minimum(s, alen), s)
        final &= ~avoided
        seek_x, seek_y = av_x, av_y
    ux, uy, ulen = norm_all(m.native_vx, m.native_vy)
    no_old = ulen <= OLD_DIR_MIN
    if cfg.native_move_reissue:
        fresh = m.native_state == 8
    else:
        fresh = (m.cmd_loop >= 0) & (m.cmd_loop + cfg.move_lag == w.game_loop[:, None])
    replace = no_old | (length <= s) | fresh
    if not cfg.native_blend:
        replace = np.ones_like(replace)
    ux = np.where(replace, nx, ux)
    uy = np.where(replace, ny, uy)
    dir_x, dir_y, _ = norm_all(BLEND_NEW * nx + BLEND_OLD * ux, BLEND_NEW * ny + BLEND_OLD * uy)
    vx = _trunc_div(dir_x * step, np.full_like(s, ONE))
    vy = _trunc_div(dir_y * step, np.full_like(s, ONE))
    out_x = (vx / ONE).astype(np.float32)
    out_y = (vy / ONE).astype(np.float32)
    with np.errstate(invalid='ignore'):
        exact_x = (buf.waypoint_x - w.x).astype(np.float32)
        exact_y = (buf.waypoint_y - w.y).astype(np.float32)
    out_x = np.where(final, exact_x, out_x)
    out_y = np.where(final, exact_y, out_y)
    zero = ((length == 0) | (step == 0)) & ~final
    out_x = np.where(zero, np.float32(0), out_x)
    out_y = np.where(zero, np.float32(0), out_y)
    buf.vel_x[:] = np.where(ground, out_x, seek_x).astype(np.float32)
    buf.vel_y[:] = np.where(ground, out_y, seek_y).astype(np.float32)
    # Advance state 8 -> 9 (held on move_lag loops).
    adv = ground & ~is_lag
    m.native_state[adv & ((m.native_state == 8) | (m.native_state <= 7))] = 9


def steer(sim: Sim) -> None:
    """Fixed-point Q12 steering stage (`steering='native'`): writes `buf.vel_*`."""
    w, buf, m, cfg = sim.world, sim.buf, sim.motion, sim.config
    seek_x, seek_y = steering._seek_all(sim)  # noqa: SLF001
    has_wp = np.isfinite(buf.waypoint_x)
    ground_alive = w.alive & ~w.is_flying
    is_lag = (cfg.move_lag > 0) & (m.lag_loop == w.game_loop[:, None])

    if not (
        cfg.native_steer_avoid or cfg.native_stop or cfg.native_air_sep or cfg.native_wall_clearance
    ):
        _steer_fast_vectorized(sim, seek_x, seek_y, has_wp, ground_alive, is_lag)
        return

    # Full sequential native steering path.
    buf.vel_x[:] = seek_x
    buf.vel_y[:] = seek_y
    if cfg.native_avoid:
        steering.sc2_avoid(sim)
    smax = smax_all(sim)
    px_all, py_all = q12(w.x), q12(w.y)
    r_all = q12(w.radius)
    wx_all, wy_all = q12(buf.waypoint_x), q12(buf.waypoint_y)
    has_dest = np.isfinite(buf.dest_x)
    dest_x_all, dest_y_all = q12(buf.dest_x), q12(buf.dest_y)
    ord_x_all, ord_y_all = q12(w.order_x), q12(w.order_y)
    tgt_all = _unit_target_slots(sim)

    _sync_stationary_states(sim, ground_alive, has_wp)

    for b in range(sim.batch_size):
        wall = _get_wall_map(sim, b) if cfg.native_wall_clearance else None
        g_units = np.flatnonzero(ground_alive[b]).tolist()
        if g_units:
            _steer_ground_env(
                sim,
                b,
                g_units,
                px_all,
                py_all,
                r_all,
                smax,
                wx_all,
                wy_all,
                has_wp,
                has_dest,
                dest_x_all,
                dest_y_all,
                ord_x_all,
                ord_y_all,
                tgt_all,
                is_lag,
                seek_x,
                seek_y,
                wall,
            )
        if cfg.native_air_sep:
            _steer_air_env(sim, b, px_all, py_all, r_all, smax, has_wp, is_lag)


@dataclass(slots=True)
class _GroundSteerState:
    """Mutable per-env Q12 arrays shared across ground steering phases."""

    x_arr: list[int]
    y_arr: list[int]
    vx_arr: list[int]
    vy_arr: list[int]
    r_arr: list[int]
    s_arr: list[int]
    utype_arr: list[int]
    owner_arr: list[int]
    slot_id_arr: list[int]
    state_arr: list[int]
    flags_arr: list[int]
    order_arr: list[int]
    engaged_arr: list[int]
    struct_arr: list[bool]
    has_wp_arr: list[bool]
    wp_x_arr: list[int]
    wp_y_arr: list[int]
    target_arr: list[int]
    has_goal_arr: list[bool]
    goal_x_arr: list[int]
    goal_y_arr: list[int]
    proj_g_arr: list[bool]


def _build_ground_steer_state(
    sim: Sim,
    b: int,
    g_units: list[int],
    px_all: np.ndarray,
    py_all: np.ndarray,
    r_all: np.ndarray,
    smax: np.ndarray,
    wx_all: np.ndarray,
    wy_all: np.ndarray,
    has_wp: np.ndarray,
    has_dest: np.ndarray,
    dest_x_all: np.ndarray,
    dest_y_all: np.ndarray,
    ord_x_all: np.ndarray,
    ord_y_all: np.ndarray,
    tgt_all: np.ndarray,
) -> _GroundSteerState:
    """Extract per-env Q12 working arrays and effective goals for ground units."""
    w, m, cfg = sim.world, sim.motion, sim.config
    x_arr = [int(px_all[b, u]) for u in g_units]
    y_arr = [int(py_all[b, u]) for u in g_units]
    vx_arr = [int(m.native_vx[b, u]) for u in g_units]
    vy_arr = [int(m.native_vy[b, u]) for u in g_units]
    r_arr = [int(r_all[b, u]) for u in g_units]
    s_arr = [int(smax[b, u]) for u in g_units]
    utype_arr = [int(w.unit_type[b, u]) for u in g_units]
    owner_arr = [int(w.owner[b, u]) for u in g_units]
    slot_id_arr = [
        int(m.native_slot[b, u]) if int(m.native_slot[b, u]) >= 0 else u for u in g_units
    ]
    state_arr = [int(m.native_state[b, u]) for u in g_units]
    flags_arr = [int(m.native_flags[b, u]) for u in g_units]
    order_arr = [int(w.order_kind[b, u]) for u in g_units]
    engaged_arr = [int(w.engaged_target_slot[b, u]) for u in g_units]
    struct_arr = [bool(sim.params.structure[b, u]) for u in g_units]
    has_wp_arr = [bool(has_wp[b, u]) for u in g_units]
    wp_x_arr = [int(wx_all[b, u]) for u in g_units]
    wp_y_arr = [int(wy_all[b, u]) for u in g_units]
    target_arr = [int(tgt_all[b, u]) for u in g_units]
    has_goal_arr: list[bool] = []
    goal_x_arr: list[int] = []
    goal_y_arr: list[int] = []
    proj_g_arr: list[bool] = []
    for i, u in enumerate(g_units):
        # Override navmesh destination only if in-transit wall projection modified G.
        g_proj = (
            cfg.native_wall_clearance
            and bool(m.native_has_g[b, u])
            and (
                int(m.native_gx[b, u]) != int(ord_x_all[b, u])
                or int(m.native_gy[b, u]) != int(ord_y_all[b, u])
            )
        )
        proj_g_arr.append(g_proj)
        if target_arr[i] != NO_SLOT:
            has_goal_arr.append(True)
            if has_wp_arr[i]:
                goal_x_arr.append(wp_x_arr[i])
                goal_y_arr.append(wp_y_arr[i])
            else:
                ts = target_arr[i]
                goal_x_arr.append(int(px_all[b, ts]))
                goal_y_arr.append(int(py_all[b, ts]))
        elif has_wp_arr[i] and g_proj:
            has_goal_arr.append(True)
            goal_x_arr.append(int(m.native_gx[b, u]))
            goal_y_arr.append(int(m.native_gy[b, u]))
        elif has_dest[b, u]:
            has_goal_arr.append(True)
            goal_x_arr.append(int(dest_x_all[b, u]))
            goal_y_arr.append(int(dest_y_all[b, u]))
        elif m.native_has_g[b, u]:
            has_goal_arr.append(True)
            goal_x_arr.append(int(m.native_gx[b, u]))
            goal_y_arr.append(int(m.native_gy[b, u]))
        elif has_wp_arr[i]:
            has_goal_arr.append(True)
            goal_x_arr.append(wp_x_arr[i])
            goal_y_arr.append(wp_y_arr[i])
        else:
            has_goal_arr.append(False)
            goal_x_arr.append(0)
            goal_y_arr.append(0)
    return _GroundSteerState(
        x_arr=x_arr,
        y_arr=y_arr,
        vx_arr=vx_arr,
        vy_arr=vy_arr,
        r_arr=r_arr,
        s_arr=s_arr,
        utype_arr=utype_arr,
        owner_arr=owner_arr,
        slot_id_arr=slot_id_arr,
        state_arr=state_arr,
        flags_arr=flags_arr,
        order_arr=order_arr,
        engaged_arr=engaged_arr,
        struct_arr=struct_arr,
        has_wp_arr=has_wp_arr,
        wp_x_arr=wp_x_arr,
        wp_y_arr=wp_y_arr,
        target_arr=target_arr,
        has_goal_arr=has_goal_arr,
        goal_x_arr=goal_x_arr,
        goal_y_arr=goal_y_arr,
        proj_g_arr=proj_g_arr,
    )


def _apply_early_stops(
    sim: Sim,
    b: int,
    g_units: list[int],
    st: _GroundSteerState,
    ord_x_all: np.ndarray,
    ord_y_all: np.ndarray,
    is_lag: np.ndarray,
) -> None:
    """Phase 1: early-stage arrival/stop check for MOVE and untargeted ATTACK_MOVE."""
    w, buf, m = sim.world, sim.buf, sim.motion
    for i, u in enumerate(g_units):
        is_pt_ord = st.order_arr[i] == int(CommandKind.MOVE) or (
            st.order_arr[i] == int(CommandKind.ATTACK_MOVE) and st.engaged_arr[i] == NO_SLOT
        )
        if not is_pt_ord or is_lag[b, u]:
            continue
        px, py = st.x_arr[i], st.y_arr[i]
        ox, oy = int(ord_x_all[b, u]), int(ord_y_all[b, u])
        gx, gy = (st.goal_x_arr[i], st.goal_y_arr[i]) if st.has_goal_arr[i] else (ox, oy)
        wx, wy = (st.wp_x_arr[i], st.wp_y_arr[i]) if st.has_wp_arr[i] else (gx, gy)
        inner_r = inner_radius_q12(st.utype_arr[i], st.r_arr[i])
        acc_q = _unit_accel_q12(sim, b, u)
        new_st, new_tol, stopped = early_stop_step(
            px,
            py,
            gx,
            gy,
            ox,
            oy,
            wx,
            wy,
            st.vx_arr[i],
            st.vy_arr[i],
            st.state_arr[i],
            st.flags_arr[i],
            st.r_arr[i],
            inner_r,
            acc_q,
            int(m.native_k[b, u]),
            int(m.native_tol[b, u]),
        )
        m.native_tol[b, u] = new_tol
        if stopped:
            st.state_arr[i] = new_st
            m.native_state[b, u] = new_st
            st.order_arr[i] = int(CommandKind.NOOP)
            st.has_wp_arr[i] = False
            st.vx_arr[i] = st.vy_arr[i] = 0
            w.order_kind[b, u] = CommandKind.NOOP
            w.order_target_slot[b, u] = NO_SLOT
            w.order_x[b, u] = w.order_y[b, u] = np.nan
            buf.waypoint_x[b, u] = buf.waypoint_y[b, u] = np.nan
            buf.dest_x[b, u] = buf.dest_y[b, u] = np.nan
            buf.vel_x[b, u] = buf.vel_y[b, u] = f32(0)
            m.native_tol[b, u] = 0
            m.native_k[b, u] = 0
            # Keep native_gx/gy/has_g for subsequent near-goal crowd accumulation.


def _steer_ground_unit(
    sim: Sim,
    b: int,
    i: int,
    u: int,
    g_units: list[int],
    st: _GroundSteerState,
    wx_all: np.ndarray,
    wy_all: np.ndarray,
    has_dest: np.ndarray,
    dest_x_all: np.ndarray,
    dest_y_all: np.ndarray,
    is_lag: np.ndarray,
    seek_x: np.ndarray,
    seek_y: np.ndarray,
    wall: WallMap | None,
) -> tuple[int, int, int, int, int, int, int, bool] | None:
    """Phase 2: compute candidate displacement for ground unit `(i, u)`.

    Returns `(wx, wy, gx, gy, q_tol, bl, inner_r, is_attack_move)` when Phase 3 post-step
    progress update should run, or `None` if skipped (no waypoint, direct arrival decel, lag,
    or non-point order).
    """
    w, buf, m, cfg = sim.world, sim.buf, sim.motion, sim.config
    if not st.has_wp_arr[i]:
        st.vx_arr[i] = st.vy_arr[i] = 0
        buf.vel_x[b, u] = buf.vel_y[b, u] = f32(0)
        return None
    lag_u = bool(is_lag[b, u])
    px, py = st.x_arr[i], st.y_arr[i]
    wx, wy = st.wp_x_arr[i], st.wp_y_arr[i]
    gx, gy = (st.goal_x_arr[i], st.goal_y_arr[i]) if st.has_goal_arr[i] else (wx, wy)
    # Sync direct waypoint with projected G when previously modified.
    if st.proj_g_arr[i] and has_dest[b, u]:
        if wx == int(dest_x_all[b, u]) and wy == int(dest_y_all[b, u]):
            wx, wy = gx, gy
            st.wp_x_arr[i], st.wp_y_arr[i] = wx, wy
    s_max = st.s_arr[i]
    v_old_x, v_old_y = st.vx_arr[i], st.vy_arr[i]
    bl = st.flags_arr[i] & 0x0C
    ux, uy, vlen = norm(v_old_x, v_old_y)
    if vlen <= OLD_DIR_MIN:
        ux, uy = 0, 0
    c_spd = min(vlen, s_max)
    acc_q = _unit_accel_q12(sim, b, u)
    dec_q = _unit_decel_q12(sim, b, u, acc_q)
    lat_a = lateral_accel_q12(st.utype_arr[i], acc_q)
    inner_r = inner_radius_q12(st.utype_arr[i], st.r_arr[i])
    nx, ny, length = norm(wx - px, wy - py)

    is_attack_move = (
        st.order_arr[i] == int(CommandKind.ATTACK_MOVE) and st.engaged_arr[i] == NO_SLOT
    )
    is_point_move = st.order_arr[i] == int(CommandKind.MOVE) or is_attack_move
    q_tol = int(m.native_tol[b, u]) if (cfg.native_stop and is_point_move) else 0
    st_in = st.state_arr[i]
    st_cur = st_in
    was_st10 = st_in == 10
    if not lag_u:
        if is_point_move and st_in <= 7:
            # Initialize progress anchor when MOVE is written directly (e.g. idle_return).
            m.native_bx[b, u], m.native_by[b, u] = px, py
            m.native_k[b, u] = 0
            m.native_f[b, u] = True
            m.native_tol[b, u] = 0
        if st_cur == 8 or st_cur == 10 or st_cur <= 7:
            st_cur = 9

    # Direct arrival deceleration check: ||P - G||^2 <= q^2.
    if cfg.native_stop and is_point_move and (px - gx) ** 2 + (py - gy) ** 2 <= q_tol * q_tol:
        d_step = max(c_spd - dec_q, 0)
        vx_out = (ux * d_step) >> 12
        vy_out = (uy * d_step) >> 12
        if not lag_u:
            st.state_arr[i] = 10
            m.native_state[b, u] = 10
            st.vx_arr[i], st.vy_arr[i] = vx_out, vy_out
        buf.vel_x[b, u] = f32(vx_out / ONE)
        buf.vel_y[b, u] = f32(vy_out / ONE)
        return None

    # Predictive arrival deceleration check (W == G).
    step = min(c_spd + acc_q, s_max) if cfg.native_stop else s_max
    if cfg.native_stop and is_point_move and wx == gx and wy == gy:
        b_pred = (c_spd // dec_q) if dec_q != 0 else 1
        if was_st10:
            b_pred += 1
        hx = _trunc_div2(b_pred * v_old_x)
        hy = _trunc_div2(b_pred * v_old_y)
        if q_tol == 0:
            pred_hit = (hx * hx + hy * hy) > length * length
        else:
            rx, ry = (wx - px) - hx, (wy - py) - hy
            pred_hit = (rx * rx + ry * ry) <= q_tol * q_tol
        if pred_hit:
            st_cur = 10
            step = min(max(c_spd - acc_q - dec_q, 0) + acc_q, s_max)

    near_wp = (length - q_tol) <= step
    final_exact = (q_tol == 0) and near_wp and (wx == gx) and (wy == gy)
    # native_avoid (heuristic fallback comparison)
    if cfg.native_avoid:
        avx, avy = float(buf.vel_x[b, u]), float(buf.vel_y[b, u])
        if avx != float(seek_x[b, u]) or avy != float(seek_y[b, u]):
            anx, any_, alen = norm(int(round(avx * ONE)), int(round(avy * ONE)))
            nx, ny = anx, any_
            step = min(s_max, alen)
            final_exact = False

    if cfg.native_move_reissue or cfg.native_stop or cfg.native_steer_avoid:
        fresh = st_in == 8
    else:
        fresh = (int(m.cmd_loop[b, u]) >= 0) and (
            int(m.cmd_loop[b, u]) + cfg.move_lag == int(w.game_loop[b])
        )
    replace = (vlen <= OLD_DIR_MIN) or near_wp or fresh or (not cfg.native_blend)
    if replace:
        ux, uy = nx, ny

    if final_exact:
        vx_cand, vy_cand = wx - px, wy - py
        if not lag_u:
            st.vx_arr[i], st.vy_arr[i] = vx_cand, vy_cand
            st.state_arr[i] = st_cur
            m.native_state[b, u] = st_cur
        # Preserve exact float32 arrival when order point is off the Q12 grid.
        if st.proj_g_arr[i] and (wx != int(wx_all[b, u]) or wy != int(wy_all[b, u])):
            buf.vel_x[b, u] = f32(vx_cand / ONE)
            buf.vel_y[b, u] = f32(vy_cand / ONE)
        else:
            buf.vel_x[b, u] = f32(buf.waypoint_x[b, u] - w.x[b, u])
            buf.vel_y[b, u] = f32(buf.waypoint_y[b, u] - w.y[b, u])
    elif length == 0 or step == 0:
        vx_cand, vy_cand = 0, 0
        if not lag_u:
            st.vx_arr[i], st.vy_arr[i] = 0, 0
            st.state_arr[i] = st_cur
            m.native_state[b, u] = st_cur
        buf.vel_x[b, u] = buf.vel_y[b, u] = f32(0)
    else:
        mx, my, _ = norm(BLEND_NEW * nx + BLEND_OLD * ux, BLEND_NEW * ny + BLEND_OLD * uy)
        v0x = _trunc_div_q12(mx * step)
        v0y = _trunc_div_q12(my * step)
        vx_cand, vy_cand = v0x, v0y
        if cfg.native_steer_avoid:
            j_vec, tol_delta = _local_steer_avoid(
                i,
                px,
                py,
                v0x,
                v0y,
                s_max,
                st.r_arr[i],
                inner_r,
                wx,
                wy,
                gx,
                gy,
                st.target_arr[i],
                st.flags_arr[i],
                int(m.native_k[b, u]),
                st.owner_arr[i],
                u,
                st.x_arr,
                st.y_arr,
                st.vx_arr,
                st.vy_arr,
                st.r_arr,
                st.owner_arr,
                g_units,
                st.slot_id_arr,
                st.state_arr,
                st.order_arr,
                st.engaged_arr,
                st.has_wp_arr,
                st.wp_x_arr,
                st.wp_y_arr,
                st.goal_x_arr,
                st.goal_y_arr,
                st.has_goal_arr,
                st.target_arr,
                wall,
                st.struct_arr,
            )
            if tol_delta and not lag_u:
                m.native_tol[b, u] = int(m.native_tol[b, u]) + tol_delta
            if j_vec is not None:
                jx, jy = j_vec
                if bl != 0:
                    jx, jy, _ = norm(jx, jy)
                d1x, d1y, _ = norm(mx + 5 * ux + 4 * jx, my + 5 * uy + 4 * jy)
                # Floor-shift velocity step (>> 12).
                vx_cand = (d1x * step) >> 12
                vy_cand = (d1y * step) >> 12
            if not (near_wp and wx == gx and wy == gy):
                vx_cand, vy_cand = lateral_clamp_q12(
                    vx_cand, vy_cand, v_old_x, v_old_y, ux, uy, step, lat_a
                )
        if not lag_u:
            st.vx_arr[i], st.vy_arr[i] = vx_cand, vy_cand
            st.state_arr[i] = st_cur
            m.native_state[b, u] = st_cur
        buf.vel_x[b, u] = f32(vx_cand / ONE)
        buf.vel_y[b, u] = f32(vy_cand / ONE)

    if lag_u or not is_point_move:
        return None
    return wx, wy, gx, gy, q_tol, bl, inner_r, is_attack_move


def _accumulate_near_goal_crowd(
    i: int,
    gx: int,
    gy: int,
    q_tol: int,
    ntol: int,
    st: _GroundSteerState,
) -> int:
    """Accumulate near-goal crowded-neighbor radii and update tolerance `ntol`."""
    px, py = st.x_arr[i], st.y_arr[i]
    r_i = st.r_arr[i]
    rad_gate = q_tol + 10 * r_i
    if (px - gx) ** 2 + (py - gy) ** 2 > rad_gate * rad_gate:
        return ntol
    q_lim_base = q_tol + r_i + QUANT_PAD
    counted: list[int] = []
    for j in range(len(st.x_arr)):
        if j == i or st.struct_arr[j]:
            continue
        rj = st.r_arr[j]
        if quant_key(st.x_arr[j], st.y_arr[j], gx, gy) > ((rj + q_lim_base) ** 2) >> 12:
            continue
        if st.state_arr[j] <= 7:
            # Hostile or HOLD neighbors count unconditionally;
            # Attacking allies fall through to the G_j proximity check below.
            if st.owner_arr[j] != st.owner_arr[i] or st.order_arr[j] == int(CommandKind.HOLD):
                counted.append(rj)
                continue
        if st.has_goal_arr[j]:
            dgjx, dgjy = gx - st.goal_x_arr[j], gy - st.goal_y_arr[j]
            if dgjx * dgjx + dgjy * dgjy < (r_i + rj) ** 2:
                counted.append(rj)
    if counted:
        return near_goal_area_step(r_i, q_tol, ntol, counted)
    return ntol


def _update_ground_progress(
    sim: Sim,
    b: int,
    i: int,
    u: int,
    st: _GroundSteerState,
    wx: int,
    wy: int,
    gx: int,
    gy: int,
    q_tol: int,
    bl: int,
    inner_r: int,
    is_attack_move: bool,
    ord_x_all: np.ndarray,
    ord_y_all: np.ndarray,
    wall: WallMap | None,
) -> None:
    """Phase 3: post-step progress update, single-edge wall projection, and crowd tolerance."""
    w, buf, m, cfg = sim.world, sim.buf, sim.motion, sim.config
    px, py = st.x_arr[i], st.y_arr[i]
    ntol = int(m.native_tol[b, u])
    if cfg.native_stop:
        ox, oy = int(ord_x_all[b, u]), int(ord_y_all[b, u])
        tx, ty = (gx, gy) if (wx == gx and wy == gy) else (wx, wy)
        g_pass = True if wall is None else wall.passable_q12(gx, gy)
        pw2 = (px - wx) ** 2 + (py - wy) ** 2
        nbx, nby, nk, nf, ntol = late_progress_step(
            px,
            py,
            tx,
            ty,
            int(m.native_bx[b, u]),
            int(m.native_by[b, u]),
            ox,
            oy,
            gx,
            gy,
            int(m.native_k[b, u]),
            bool(m.native_f[b, u]),
            ntol,
            st.r_arr[i],
            bl,
            g_passable=g_pass,
            is_attack_move=is_attack_move,
            pw2_ge_16r2=pw2 >= (4 * st.r_arr[i]) ** 2,
        )
        m.native_bx[b, u], m.native_by[b, u] = nbx, nby
        m.native_k[b, u] = nk
        m.native_f[b, u] = nf
    # When blocked on a non-attack MOVE at W == G on an 8-loop beat, project wall edge and
    # accumulate crowd.
    if (not is_attack_move) and bl != 0 and wx == gx and wy == gy:
        if cfg.scan_phase in ('per_unit', 'tag_order', 'tag_rotate'):
            phase_due = (
                (int(w.game_loop[b]) - cfg.acquire_start_loop - int(sim.combat.scan_shift[b, u]))
                & 7
            ) == 0
        else:
            phase_due = ((int(w.game_loop[b]) + 1 - st.slot_id_arr[i]) & 7) == 0
        if phase_due:
            # In-transit single-edge projection when F == 0.
            if wall is not None and ntol == 0:
                proj = project_goal_single_edge(wall, px, py, gx, gy, inner_r)
                if proj is not None:
                    ng_x, ng_y = proj
                    m.native_gx[b, u], m.native_gy[b, u] = ng_x, ng_y
                    m.native_has_g[b, u] = True
                    gx, gy = ng_x, ng_y
                    st.goal_x_arr[i], st.goal_y_arr[i] = ng_x, ng_y
                    buf.dest_x[b, u] = buf.waypoint_x[b, u] = f32(ng_x / ONE)
                    buf.dest_y[b, u] = buf.waypoint_y[b, u] = f32(ng_y / ONE)
            # Near-goal crowded-neighbor area accumulation.
            if cfg.native_stop:
                ntol = _accumulate_near_goal_crowd(i, gx, gy, q_tol, ntol, st)
    if cfg.native_stop:
        m.native_tol[b, u] = ntol


def _steer_ground_env(
    sim: Sim,
    b: int,
    g_units: list[int],
    px_all: np.ndarray,
    py_all: np.ndarray,
    r_all: np.ndarray,
    smax: np.ndarray,
    wx_all: np.ndarray,
    wy_all: np.ndarray,
    has_wp: np.ndarray,
    has_dest: np.ndarray,
    dest_x_all: np.ndarray,
    dest_y_all: np.ndarray,
    ord_x_all: np.ndarray,
    ord_y_all: np.ndarray,
    tgt_all: np.ndarray,
    is_lag: np.ndarray,
    seek_x: np.ndarray,
    seek_y: np.ndarray,
    wall: WallMap | None,
) -> None:
    """Per-env sequential ground steering: early stop, arrival decel, local avoidance, and
    progress update.
    """
    st = _build_ground_steer_state(
        sim,
        b,
        g_units,
        px_all,
        py_all,
        r_all,
        smax,
        wx_all,
        wy_all,
        has_wp,
        has_dest,
        dest_x_all,
        dest_y_all,
        ord_x_all,
        ord_y_all,
        tgt_all,
    )
    if sim.config.native_stop:
        _apply_early_stops(sim, b, g_units, st, ord_x_all, ord_y_all, is_lag)
    for i, u in enumerate(g_units):
        post = _steer_ground_unit(
            sim,
            b,
            i,
            u,
            g_units,
            st,
            wx_all,
            wy_all,
            has_dest,
            dest_x_all,
            dest_y_all,
            is_lag,
            seek_x,
            seek_y,
            wall,
        )
        if post is not None:
            wx, wy, gx, gy, q_tol, bl, inner_r, is_attack_move = post
            _update_ground_progress(
                sim,
                b,
                i,
                u,
                st,
                wx,
                wy,
                gx,
                gy,
                q_tol,
                bl,
                inner_r,
                is_attack_move,
                ord_x_all,
                ord_y_all,
                wall,
            )


__all__ = ['_steer_ground_env', '_sync_stationary_states', '_unit_target_slots', 'steer']
