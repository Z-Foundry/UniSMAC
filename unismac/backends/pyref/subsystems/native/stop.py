"""Arrival, blocked-progress, and crowded-goal stopping primitives for `native`."""

from __future__ import annotations

from .q12 import dot_q12


def early_stop_step(
    px: int,
    py: int,
    gx: int,
    gy: int,
    ox: int,
    oy: int,
    wx: int,
    wy: int,
    vx: int,
    vy: int,
    state: int,
    flags: int,
    r: int,
    inner_r: int,
    accel_q12: int,
    k: int,
    tol: int,
) -> tuple[int, int, bool]:
    """Early-stage arrival and blocked-stop check.

    Returns `(new_state, new_tol, stopped)`; when `stopped=True`, `new_state` is `3` (exact
    arrival) or `1` (tolerance stop).
    """
    if state == 11 and wx == gx and wy == gy and (flags & 0x09) == 0x08:
        if dot_q12(gx - px, gy - py, vx, vy) <= 0:
            tol += r
    if tol > 16 * r and k <= 63 and (ox - gx) ** 2 + (oy - gy) ** 2 < inner_r * inner_r:
        tol = 16 * r
    dpg2 = (px - gx) ** 2 + (py - gy) ** 2
    if dpg2 > tol * tol:
        return state, tol, False
    dpo2 = (px - ox) ** 2 + (py - oy) ** 2
    if dpo2 == 0:
        return 3, tol, True
    if vx * vx + vy * vy <= (2 * accel_q12) ** 2:
        return 1, tol, True
    return state, tol, False


def late_progress_step(
    px: int,
    py: int,
    tx: int,
    ty: int,
    bx: int,
    by: int,
    ox: int,
    oy: int,
    gx: int,
    gy: int,
    k: int,
    f_flag: bool,
    tol: int,
    r: int,
    m_snap: int,
    *,
    g_passable: bool = True,
    is_attack_move: bool = False,
    pw2_ge_16r2: bool = False,
) -> tuple[int, int, int, bool, int]:
    """Post-step progress counter and dynamic tolerance adjustment for move / attack-move orders.

    Returns `(new_bx, new_by, new_k, new_f, new_tol)`.
    """
    dpt2 = (px - tx) ** 2 + (py - ty) ** 2
    dbt2 = (bx - tx) ** 2 + (by - ty) ** 2
    if dpt2 < dbt2:
        if k == 0 and tol > 0 and m_snap == 0 and g_passable:
            tol = max(tol - r, 0)
        bx, by = px, py
        k = 0
    elif f_flag:
        f_flag = False
        k += 1
    elif px == bx and py == by:
        k += 8
    else:
        k += 1
    if m_snap == 0:
        thresh = 128 if (ox == gx and oy == gy) else 32
        if k > thresh:
            tol += r
    elif is_attack_move:
        # Attack-move orders use divisor 12 (threshold 128 or 10).
        thresh = 128 if (ox == gx and oy == gy and pw2_ge_16r2) else 10
        if k > thresh:
            tol += r
    else:
        if k >= 65:
            tol += r
    return bx, by, k, f_flag, tol


def near_goal_area_step(r: int, q_tol: int, tol: int, counted_radii: list[int]) -> int:
    """Near-goal crowded-neighbor radius accumulation (`H > M` increments `F += r`)."""
    h_acc = (r * r) >> 12
    m_val = max(h_acc, (q_tol * q_tol) >> 12)
    for rj in counted_radii:
        h_acc += ((rj + 512) ** 2) >> 12
    if m_val > 0 and h_acc > m_val:
        while h_acc > m_val:
            h_acc -= m_val
            tol += r
    return tol


__all__ = ['early_stop_step', 'late_progress_step', 'near_goal_area_step']
