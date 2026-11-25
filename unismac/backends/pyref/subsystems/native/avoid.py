"""Local ground steering corridor avoidance and lateral velocity clamping for `native`."""

from __future__ import annotations

import math

from unismac.contract import NO_SLOT, CommandKind

from .q12 import (
    CONTACT_GATE,
    QUANT_PAD,
    _trunc_div_int,
    cross_q12,
    d2q,
    dot_q12,
    norm,
    quant_key,
)
from .wall import WallMap


def _merge_interval(intervals: list[tuple[int, int]], lo: int, hi: int) -> None:
    """Merge touching/overlapping closed intervals (`a <= hi_i and lo_i <= b`), up to 4 slots."""
    i = 0
    while i < len(intervals):
        ilo, ihi = intervals[i]
        if lo <= ihi and ilo <= hi:
            if ilo < lo:
                lo = ilo
            if ihi > hi:
                hi = ihi
            intervals.pop(i)
        else:
            i += 1
    if len(intervals) < 4:
        intervals.append((lo, hi))


def lateral_clamp_q12(
    vx: int, vy: int, v_old_x: int, v_old_y: int, ux: int, uy: int, s: int, lat_a: int
) -> tuple[int, int]:
    """Clamp lateral velocity change by `4 * LateralAcceleration` (floor-rounded)."""
    kx, ky = -uy, ux
    d_lat = dot_q12(vx - v_old_x, vy - v_old_y, kx, ky)
    lim = 4 * lat_a
    if d_lat > lim:
        diff = lim - d_lat
        wx_c = vx + ((kx * diff) >> 12)
        wy_c = vy + ((ky * diff) >> 12)
        wnx, wny, _ = norm(wx_c, wy_c)
        return (wnx * s) >> 12, (wny * s) >> 12
    if d_lat < -lim:
        diff = d_lat + lim
        wx_c = vx - ((kx * diff) >> 12)
        wy_c = vy - ((ky * diff) >> 12)
        wnx, wny, _ = norm(wx_c, wy_c)
        return (wnx * s) >> 12, (wny * s) >> 12
    return vx, vy


def _local_steer_avoid(
    a_idx: int,
    px: int,
    py: int,
    v0x: int,
    v0y: int,
    s_max: int,
    r_a: int,
    inner_r_a: int,
    wx_a: int,
    wy_a: int,
    gx_a: int,
    gy_a: int,
    target_slot_a: int,
    flags_a: int,
    k_a: int,
    owner_a: int,
    slot_a: int,
    # All ground unit arrays (indexed 0..n-1)
    x_arr: list[int],
    y_arr: list[int],
    vx_arr: list[int],
    vy_arr: list[int],
    r_arr: list[int],
    owner_arr: list[int],
    slot_arr: list[int],
    slot_id_arr: list[int],
    state_arr: list[int],
    order_arr: list[int],
    engaged_arr: list[int],
    has_wp_arr: list[bool],
    wp_x_arr: list[int],
    wp_y_arr: list[int],
    goal_x_arr: list[int],
    goal_y_arr: list[int],
    has_goal_arr: list[bool],
    target_arr: list[int],
    wall: WallMap | None,
    struct_arr: list[bool] | None = None,
) -> tuple[tuple[int, int] | None, int]:
    """Execute fixed-point bounded local ground steering avoidance.

    Returns `(J | None, tol_delta)`, where `tol_delta` accumulates stop-tolerance growth when
    goal is covered.
    """
    if (flags_a & 0x08) != 0 and k_a == 0:
        return None, 0
    qx, qy, qlen = norm(v0x, v0y)
    if qlen == 0:
        return None, 0
    cx = px + ((qx * (2 * r_a)) >> 12)
    cy = py + ((qy * (2 * r_a)) >> 12)
    qbase = s_max + 2 * r_a + QUANT_PAD
    cands: list[tuple[int, int, int]] = []
    for b in range(len(x_arr)):
        if b == a_idx:
            continue
        if struct_arr is not None and struct_arr[b]:
            # Footprinted static structures do not participate in dynamic unit avoidance or
            # goal-cover tolerance.
            continue
        if target_slot_a != NO_SLOT and slot_arr[b] == target_slot_a:
            continue
        k_dist = quant_key(x_arr[b], y_arr[b], cx, cy)
        lim = ((r_arr[b] + qbase) ** 2) >> 12
        if k_dist <= lim:
            cands.append((k_dist, -slot_id_arr[b], b))
    cands.sort()

    budget = 3
    intervals: list[tuple[int, int]] = []
    tol_delta = 0
    point_wg = (target_slot_a == NO_SLOT) and (wx_a == gx_a) and (wy_a == gy_a)
    w_len2 = (wx_a - px) ** 2 + (wy_a - py) ** 2
    ga_dist2 = (gx_a - px) ** 2 + (gy_a - py) ** 2

    for _, _, b in cands:
        if budget <= 0:
            break
        xb, yb = x_arr[b], y_arr[b]
        dab_x, dab_y = xb - px, yb - py
        # 1. Forward hemisphere (does not consume budget)
        if dot_q12(v0x, v0y, dab_x, dab_y) < 0:
            continue
        same_ally = owner_arr[b] == owner_a
        # 2. Allied neighbor targeting A (does not consume budget)
        if same_ally and target_arr[b] == slot_a:
            continue
        # 3. Allied neighbor gating (does not consume budget)
        if same_ally:
            ord_b = order_arr[b]
            eng_b = engaged_arr[b]
            if ord_b == int(CommandKind.NOOP) and eng_b == NO_SLOT:
                # Skip idle allies not in yielding state (state_B != 6).
                if state_arr[b] != 6:
                    continue
            elif ord_b == int(CommandKind.HOLD) or (eng_b != NO_SLOT and not has_wp_arr[b]):
                # Stationary HOLD or in-range attacking ally: treat as fixed obstacle.
                pass
            else:
                # Moving/chasing ally: apply co-directional rear-following skip gate.
                # Newly issued order (state == 8) keeps W_B == P_B before steering runs, so W_B
                # - X_B == 0.
                if state_arr[b] == 8:
                    tx, ty = 0, 0
                elif state_arr[b] > 8 and has_wp_arr[b]:
                    tx, ty = wp_x_arr[b] - xb, wp_y_arr[b] - yb
                else:
                    tx, ty = vx_arr[b], vy_arr[b]
                if dot_q12(tx, ty, dab_x, dab_y) >= 0:
                    continue
        # 4. Point-target W_A == G_A skip gate (does not consume budget)
        r_sum = r_a + r_arr[b]
        t_rad = r_sum - CONTACT_GATE
        if t_rad <= 0:
            continue
        t2 = t_rad * t_rad
        if point_wg:
            dgx, dgy = gx_a - xb, gy_a - yb
            if dgx * dgx + dgy * dgy < t2:
                p1x, p1y = px + v0x - xb, py + v0y - yb
                if p1x * p1x + p1y * p1y < t2:
                    tol_delta += 2 * t_rad
                continue
            if ga_dist2 < dab_x * dab_x + dab_y * dab_y:
                continue
        # Decrement neighbor avoidance budget (before state-9 same-goal skip).
        budget -= 1
        # State-9 same-destination skip.
        if state_arr[b] == 9 and has_goal_arr[b]:
            gbx, gby = goal_x_arr[b], goal_y_arr[b]
            if 50 * d2q(gx_a, gy_a, gbx, gby) <= d2q(gx_a, gy_a, px, py) + d2q(gbx, gby, xb, yb):
                continue

        vbx, vby = vx_arr[b], vy_arr[b]
        yx, yy = xb + vbx, yb + vby
        dx, dy = yx - px, yy - py
        a_proj = dot_q12(qx, qy, dx, dy)
        if not (0 < a_proj <= 16 * s_max + t_rad) or (dx == 0 and dy == 0):
            continue
        d2 = dx * dx + dy * dy
        wy2 = (wx_a - yx) ** 2 + (wy_a - yy) ** 2
        if d2 > t2 and d2 > w_len2 and wy2 > t2:
            continue

        # Moving neighbor left-block corridor [0, +4*(R_A + R_B)].
        if (vbx != 0 or vby != 0) and len(intervals) == 0:
            perp_x = ((qx * a_proj) >> 12) - dx
            perp_y = ((qy * a_proj) >> 12) - dy
            lat_dist = math.isqrt(perp_x * perp_x + perp_y * perp_y)
            if lat_dist < t_rad:
                if qx * dy >= qy * dx:
                    sdx, sdy = qy, -qx
                else:
                    sdx, sdy = -qy, qx
                denom = r_sum + CONTACT_GATE
                scale = _trunc_div_int((denom - lat_dist) << 12, denom)
                tsx = (((sdx * scale) >> 12) * s_max) >> 12
                tsy = (((sdy * scale) >> 12) * s_max) >> 12
                if tsx != 0 or tsy != 0:
                    s1 = 1 if ((dab_y * tsx) >> 12) >= ((dab_x * tsy) >> 12) else -1
                    s2 = 1 if ((vbx * dab_y) >> 12) >= ((vby * dab_x) >> 12) else -1
                    if s1 == s2:
                        _merge_interval(intervals, 0, 4 * r_sum)

        # Predicted center lateral interval [h - T, h + T].
        h = cross_q12(qx, qy, dx, dy)
        lo, hi = h - t_rad, h + t_rad
        if wall is not None and target_slot_a == NO_SLOT:
            sx = (-qy * t_rad) >> 12
            sy = (qx * t_rad) >> 12
            if hi > 0 and not wall.passable_q12(xb + sx, yb + sy):
                hi = 4 * (t_rad + inner_r_a)
            if lo < 0 and not wall.passable_q12(xb - sx, yb - sy):
                lo = -4 * (t_rad + inner_r_a)
        _merge_interval(intervals, lo, hi)

    # Choose corridor endpoint and steering vector J.
    l_env, r_env = 0, 0
    for lo, hi in intervals:
        if lo <= 0 <= hi:
            if lo < l_env:
                l_env = lo
            if hi > r_env:
                r_env = hi
    if l_env == r_env:
        return None, tol_delta
    e = r_env if (l_env + r_env < 0) else l_env
    g = (e - 1024) if e < 0 else (e + 1024)
    jx = (((-qy * g) >> 12) * s_max) >> 12
    jy = (((qx * g) >> 12) * s_max) >> 12
    return (jx, jy), tol_delta


__all__ = ['_local_steer_avoid', '_merge_interval', 'lateral_clamp_q12']
