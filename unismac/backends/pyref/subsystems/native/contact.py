"""Fixed-point Q12 contact resolution batches and position commit (`collision='native'`)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from unismac.contract import NO_SLOT, CommandKind

from ...sim import Sim
from .. import collision
from .air import _air_idle_contact
from .q12 import (
    CONTACT_GATE,
    CONTACT_TARGET,
    INNER_RADIUS_Q12,
    ONE,
    QUANT_PAD,
    W_HALF,
    W_HIGH,
    W_LOW,
    _isqrt_all,
    _trunc_div2,
    facing_vec_q12,
    norm,
    q12,
    quant_key,
    smax_all,
)
from .steer import _sync_stationary_states, _unit_target_slots
from .wall import WallMap, _get_wall_map, workpoint_correct


@dataclass(slots=True)
class _ContactContext:
    """Typed per-batch array and map inputs for `_Env` contact resolution."""

    x0: np.ndarray
    y0: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    r: np.ndarray
    s: np.ndarray
    top: np.ndarray
    moving: np.ndarray
    pin: np.ndarray | None = None
    hard_pin: np.ndarray | None = None
    state: np.ndarray | None = None
    flags: np.ndarray | None = None
    hold_still: np.ndarray | None = None
    order_kind: np.ndarray | None = None
    target: np.ndarray | None = None
    slot_id: np.ndarray | None = None
    inner_r: np.ndarray | None = None
    wall: WallMap | None = None
    rel: Callable[[int, int], bool] | None = None


class _Env:
    """Per-env integer working state for one contact batch (indexed over ground units)."""

    def __init__(
        self,
        sim: Sim,
        b: int,
        units: list[int],
        ctx: _ContactContext,
    ) -> None:
        cfg = sim.config
        self.units = units
        n = len(units)

        self.x = [int(ctx.x0[b, u]) for u in units]
        self.y = [int(ctx.y0[b, u]) for u in units]
        self.vx = [int(ctx.vx[b, u]) for u in units]
        self.vy = [int(ctx.vy[b, u]) for u in units]
        self.r = [int(ctx.r[b, u]) for u in units]
        self.s = [int(ctx.s[b, u]) for u in units]
        self.top = [int(ctx.top[b, u]) for u in units]
        self.owner = [int(sim.world.owner[b, u]) for u in units]
        self.facing = [float(sim.world.facing[b, u]) for u in units]
        self.moving = [bool(ctx.moving[b, u]) for u in units]
        self.pin = [bool(ctx.pin[b, u]) for u in units] if ctx.pin is not None else [False] * n
        self.hard_pin = (
            [bool(ctx.hard_pin[b, u]) for u in units] if ctx.hard_pin is not None else [False] * n
        )
        self.state = (
            [int(ctx.state[b, u]) for u in units]
            if ctx.state is not None
            else [9 if mv else 0 for mv in self.moving]
        )
        self.flags = [int(ctx.flags[b, u]) for u in units] if ctx.flags is not None else [0] * n
        self.hold_still = (
            [bool(ctx.hold_still[b, u]) for u in units]
            if ctx.hold_still is not None
            else [False] * n
        )
        self.order_kind = (
            [int(ctx.order_kind[b, u]) for u in units]
            if ctx.order_kind is not None
            else [int(CommandKind.MOVE) if mv else int(CommandKind.NOOP) for mv in self.moving]
        )
        self.target = (
            [int(ctx.target[b, u]) for u in units] if ctx.target is not None else [NO_SLOT] * n
        )
        self.slot_id = (
            [int(ctx.slot_id[b, u]) for u in units] if ctx.slot_id is not None else list(units)
        )
        self.inner_r = (
            [int(ctx.inner_r[b, u]) for u in units] if ctx.inner_r is not None else list(self.r)
        )
        self.wall: WallMap | None = ctx.wall
        self.rel_fn: Callable[[int, int], bool] | None = ctx.rel

        self.share_rule = cfg.native_share_rule
        self.batch_flags = cfg.native_batch_flags
        self.nbr_order = cfg.native_nbr_order
        self.wall_clearance = cfg.native_wall_clearance
        self.rel_same = cfg.native_relation
        self.rel_enemy = cfg.native_enemy_relation
        self.idle_w = cfg.native_idle_share
        self.speed_rule = cfg.native_speed_rule
        self.pushed: list[tuple[int, int]] = []  # (pushed_idle, mover) for idle_return

        # Conditional stationary HOLD flag update.
        if self.share_rule == 'priority_tree':
            for i in range(n):
                hb = self.order_kind[i] == int(CommandKind.HOLD)
                if hb != self.hold_still[i]:
                    self.hold_still[i] = (
                        hb
                        and self.vx[i] == 0
                        and self.vy[i] == 0
                        and self.state[i] <= 3
                        and (self.flags[i] & 0x0C) == 0
                    )

        # Pre-batch flag transition: promote prior contact/wall/motion to active bit 0x10 and
        # clear 0x0c.
        if self.batch_flags:
            for i in range(n):
                if (self.vx[i] != 0 or self.vy[i] != 0) or (self.flags[i] & 0x0C) != 0:
                    self.flags[i] = (self.flags[i] & 0xE3) | 0x10
        self.flag = [v != 0 or vy != 0 for v, vy in zip(self.vx, self.vy, strict=True)]
        self.contacted = [False] * n

        reach = [
            max(self.r[i], 0)
            + max(self.s[i], self.top[i], math.isqrt(self.vx[i] ** 2 + self.vy[i] ** 2), 0)
            + (QUANT_PAD if self.nbr_order == 'quantized' else 0)
            for i in range(n)
        ]
        self.nbr: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                ddx, ddy = self.x[j] - self.x[i], self.y[j] - self.y[i]
                lim = reach[i] + reach[j]
                if ddx * ddx + ddy * ddy < lim * lim:
                    self.nbr[i].append(j)
                    self.nbr[j].append(i)

    def _rel9(self, u: int, v: int) -> bool:
        if self.rel_fn is not None:
            return bool(self.rel_fn(u, v))
        return bool(self.rel_same if self.owner[u] == self.owner[v] else self.rel_enemy)

    def _p_order(self, i: int) -> bool:
        """Return whether unit is mobile (`s > 7` short-circuits True; else order mobility /
        idle state).
        """
        if self.state[i] > 7:
            return True
        if self.pin[i]:
            return False
        if self.order_kind[i] == int(CommandKind.HOLD):
            return False
        if self.order_kind[i] == int(CommandKind.NOOP):
            return self.state[i] != 6
        return True

    def _weight_tiebreak(self, a: int, b: int) -> int:
        """Hostile or mutual-target contact share tie-breaker (does not compare Smax)."""
        la = self.vx[a] ** 2 + self.vy[a] ** 2
        lb = self.vx[b] ** 2 + self.vy[b] ** 2
        qa = (la == 0) and self.hold_still[a]
        qb = (lb == 0) and self.hold_still[b]
        if qa != qb:
            return ONE if qa else 0
        if la == 0 and lb == 0:
            fa4 = (self.flags[a] & 0x04) != 0
            fb4 = (self.flags[b] & 0x04) != 0
            if fa4 != fb4:
                return 0 if fa4 else ONE
            fa10 = (self.flags[a] & 0x10) != 0
            fb10 = (self.flags[b] & 0x10) != 0
            if fa10 != fb10:
                return 0 if fa10 else ONE
            return W_HALF
        if la == 0:
            return ONE
        if lb == 0:
            return 0
        if la < lb:
            return W_HIGH
        if la > lb:
            return W_LOW
        return W_HALF

    def _weight_priority_tree(self, a: int, b: int) -> int:
        """Same-priority contact share decision tree (returns B's share out of 4096)."""
        if self.hard_pin[b]:
            return 0
        a_tgt_b = self.target[a] != NO_SLOT and self.target[a] == self.units[b]
        b_tgt_a = self.target[b] != NO_SLOT and self.target[b] == self.units[a]
        if a_tgt_b and b_tgt_a:
            return self._weight_tiebreak(a, b)
        if b_tgt_a:
            if not self._rel9(b, a):
                return self._weight_tiebreak(a, b)
            if self.state[b] > 7 or self._p_order(b):
                return ONE
            return W_HALF
        if a_tgt_b:
            if not self._rel9(a, b):
                return self._weight_tiebreak(a, b)
            if self.state[a] > 7 or self._p_order(a):
                return 0
            return W_HALF
        rb = self._rel9(b, a)
        ra = self._rel9(a, b)
        if not rb and not ra:
            return self._weight_tiebreak(a, b)
        if rb and not ra:
            return ONE
        if ra and not rb:
            return 0
        # Symmetric true relation: compare p(A) vs p(B), normalizing (0, 0) -> (1, 1).
        pa = self._p_order(a)
        pb = self._p_order(b)
        if not pa and not pb:
            pa = pb = True
        if pa != pb:
            return 0 if not pb else ONE
        sa, sb = self.state[a], self.state[b]
        if (sa | sb) > 7:
            if sb < 8:
                return W_HIGH
            if sa < 8:
                return W_LOW
            # Both s >= 8: compare Smax first (higher yields 7/8).
            if self.top[a] < self.top[b]:
                return W_HIGH
            if self.top[a] > self.top[b]:
                return W_LOW
        # Equal Smax or both s <= 7: compare squared step length (shorter yields 7/8, tie 2048).
        la = self.vx[a] ** 2 + self.vy[a] ** 2
        lb = self.vx[b] ** 2 + self.vy[b] ** 2
        if la == lb:
            return W_HALF
        return W_LOW if la < lb else W_HIGH

    def weight(self, a: int, b: int) -> int:
        """Return share assigned to B (out of 4096)."""
        if self.share_rule == 'priority_tree':
            return self._weight_priority_tree(a, b)
        # legacy mode (with dual-stationary fallback reset)
        if self.hard_pin[b]:
            return 0
        if self.pin[a] != self.pin[b]:
            return 0 if self.pin[b] else ONE
        za = self.vx[a] == 0 and self.vy[a] == 0
        zb = self.vx[b] == 0 and self.vy[b] == 0
        if za and zb:
            return W_HALF
        if zb:
            return self.idle_w
        if za:
            return ONE - self.idle_w
        if self.top[a] != self.top[b] and self.speed_rule != 'table':
            a_faster = self.top[a] > self.top[b]
            if self.speed_rule == 'faster_yields':
                return W_LOW if a_faster else W_HIGH
            return W_HIGH if a_faster else W_LOW
        la = self.vx[a] ** 2 + self.vy[a] ** 2
        lb = self.vx[b] ** 2 + self.vy[b] ** 2
        if la == lb:
            return W_HALF
        r = self.rel_same if self.owner[a] == self.owner[b] else self.rel_enemy
        a_longer = la > lb
        if r == 1:
            return W_HIGH if a_longer else W_LOW
        return W_LOW if a_longer else W_HIGH

    def clamp(self, i: int) -> None:
        s = self.s[i]
        vx, vy = self.vx[i], self.vy[i]
        if vx * vx + vy * vy <= s * s:
            return
        nx, ny, _ = norm(vx, vy)
        s = max(s, 0)
        self.vx[i], self.vy[i] = (nx * s) >> 12, (ny * s) >> 12

    def clamp_top(self, i: int) -> None:
        s = self.top[i]
        vx, vy = self.vx[i], self.vy[i]
        if vx * vx + vy * vy <= s * s:
            return
        nx, ny, _ = norm(vx, vy)
        s = max(s, 0)
        self.vx[i], self.vy[i] = (nx * s) >> 12, (ny * s) >> 12

    def _ordered_nbrs(self, a: int) -> list[int]:
        if self.nbr_order != 'quantized':
            return self.nbr[a]
        cx = _trunc_div2(2 * self.x[a] + self.vx[a])
        cy = _trunc_div2(2 * self.y[a] + self.vy[a])
        return sorted(
            self.nbr[a],
            key=lambda b: (quant_key(self.x[b], self.y[b], cx, cy), -self.slot_id[b]),
        )

    def contact(self, a: int) -> None:
        """Check work-point terrain clearance, then resolve contact against each neighbor B in
        order.
        """
        if self.wall_clearance and self.wall is not None:
            px, py, hit = workpoint_correct(
                self.wall,
                self.x[a],
                self.y[a],
                self.x[a] + self.vx[a],
                self.y[a] + self.vy[a],
                self.inner_r[a],
            )
            self.vx[a], self.vy[a] = px - self.x[a], py - self.y[a]
            if hit:
                self.flags[a] |= 0x08
        if self.r[a] <= 0:
            return
        pax, pay = self.x[a] + self.vx[a], self.y[a] + self.vy[a]
        for b in self._ordered_nbrs(a):
            if self.s[a] <= 0 and self.s[b] <= 0:
                self.vx[a] = self.vy[a] = self.vx[b] = self.vy[b] = 0
                pax, pay = self.x[a], self.y[a]
                continue
            dx = self.x[b] + self.vx[b] - pax
            dy = self.y[b] + self.vy[b] - pay
            gate = self.r[a] + self.r[b] - CONTACT_GATE
            if gate <= 0 or dx * dx + dy * dy >= gate * gate:
                continue
            # Set active flag 0x10 on B upon passing the strict contact gate.
            self.flags[b] |= 0x10
            self.flag[b] = True
            self.contacted[a] = self.contacted[b] = True
            ox, oy = self.x[b] - self.x[a], self.y[b] - self.y[a]
            if dx == 0 and dy == 0:
                nx, ny = facing_vec_q12(self.facing[a])
            else:
                nx, ny, _ = norm(dx, dy)
            k = self.r[a] + self.r[b] - CONTACT_TARGET
            tx, ty = (nx * k) >> 12, (ny * k) >> 12
            if ox * dx + oy * dy >= 0:
                cx, cy = dx - tx, dy - ty
            else:
                cx, cy = dx + tx, dy + ty
            w = self.weight(a, b)
            # Mark displaced unit(s) with contact flag 0x04 according to share weight.
            if w == 0:
                self.flags[a] |= 0x04
            elif w == ONE:
                self.flags[b] |= 0x04
            else:
                self.flags[a] |= 0x04
                self.flags[b] |= 0x04
            qx, qy = (w * cx) >> 12, (w * cy) >> 12
            if self.moving[a] and not self.moving[b] and (qx or qy):
                self.pushed.append((b, a))
            elif self.moving[b] and not self.moving[a] and (cx - qx or cy - qy):
                self.pushed.append((a, b))
            self.vx[b] -= qx
            self.vy[b] -= qy
            self.vx[a] += cx - qx
            self.vy[a] += cy - qy
            self.clamp(a)
            self.clamp(b)
            pax, pay = self.x[a] + self.vx[a], self.y[a] + self.vy[a]

    def run(self, second_pass: bool) -> None:
        n = len(self.units)
        if not self.batch_flags:
            for i in range(n):
                if self.flag[i]:
                    self.contact(i)
            if second_pass:
                for i in range(n):
                    if self.contacted[i]:
                        self.contact(i)
            for _ in range(2):
                for i in range(n):
                    if self.flag[i] and not self.moving[i] and self.vx[i] == 0 and self.vy[i] == 0:
                        self.contact(i)
            if self.wall_clearance and self.wall is not None:
                for i in range(n):
                    if self.vx[i] != 0 or self.vy[i] != 0 or (self.flags[i] & 0x0C) != 0:
                        px, py, hit = workpoint_correct(
                            self.wall,
                            self.x[i],
                            self.y[i],
                            self.x[i] + self.vx[i],
                            self.y[i] + self.vy[i],
                            self.inner_r[i],
                        )
                        self.vx[i], self.vy[i] = px - self.x[i], py - self.y[i]
                        if hit:
                            self.flags[i] |= 0x08
            return

        # Four-pass live batch flag resolution.
        for i in range(n):
            if self.flags[i] & 0x10:
                self.contact(i)
        if second_pass:
            for i in range(n):
                if self.flags[i] & 0x0C:
                    self.contact(i)
        for _ in range(2):
            for i in range(n):
                if (
                    (self.flags[i] & 0x10)
                    and self.state[i] <= 4
                    and self.vx[i] == 0
                    and self.vy[i] == 0
                ):
                    self.contact(i)
        # Post-batch: clear 0x10, clamp speed by Smax, re-check terrain, and preserve 0x0c
        # across loops.
        for i in range(n):
            self.flags[i] &= ~0x10
            if (self.vx[i] != 0 or self.vy[i] != 0) or (self.flags[i] & 0x0C) != 0:
                self.clamp_top(i)
                if self.wall_clearance and self.wall is not None:
                    px, py, hit = workpoint_correct(
                        self.wall,
                        self.x[i],
                        self.y[i],
                        self.x[i] + self.vx[i],
                        self.y[i] + self.vy[i],
                        self.inner_r[i],
                    )
                    self.vx[i], self.vy[i] = px - self.x[i], py - self.y[i]
                    if hit:
                        self.flags[i] |= 0x08


def _commit_clip_q12(
    sim: Sim,
    x0: np.ndarray,
    y0: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    sel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Commit positions clamped to playable or full half-open box `[L<<12, (H<<12)-1]` based on
    old X.
    """
    pl = q12(sim.playable)
    pl_x0, pl_y0 = pl[:, 0:1], pl[:, 1:2]
    pl_x1, pl_y1 = pl[:, 2:3], pl[:, 3:4]
    in_pl = (x0 >= pl_x0) & (x0 < pl_x1) & (y0 >= pl_y0) & (y0 < pl_y1)
    lo_x = np.where(in_pl, pl_x0, 0)
    lo_y = np.where(in_pl, pl_y0, 0)
    hi_x = np.where(in_pl, pl_x1 - 1, pl_x1 - 1)
    hi_y = np.where(in_pl, pl_y1 - 1, pl_y1 - 1)
    outside = sel & ~in_pl
    if outside.any():
        for b in np.flatnonzero(outside.any(axis=1)).tolist():
            info = sim.env_map[b]
            if info is not None:
                hi_x[b] = np.where(outside[b], (info.width << 12) - 1, hi_x[b])
                hi_y[b] = np.where(outside[b], (info.height << 12) - 1, hi_y[b])
    nx_ = np.clip(x0 + vx, lo_x, hi_x)
    ny_ = np.clip(y0 + vy, lo_y, hi_y)
    return nx_, ny_


def contact(sim: Sim) -> None:
    """`collision='native'`: native contact batch, wall clearance, air separation, and commit."""
    w, cfg, buf, m = sim.world, sim.config, sim.buf, sim.motion
    ground_m = w.alive & ~w.is_flying
    moving_m = np.isfinite(buf.waypoint_x)

    if cfg.steering != 'native':
        _sync_stationary_states(sim, ground_m, moving_m)
        m.native_state[ground_m & moving_m] = 9

    if cfg.native_share_rule == 'priority_tree':
        # HOLD and in-range attacking units stay stationary regardless of pin_after_fire.
        order_pin_now = (
            ground_m
            & ~moving_m
            & ((w.order_kind == CommandKind.HOLD) | (w.engaged_target_slot != NO_SLOT))
        )
        pin_now: np.ndarray | None = order_pin_now
        pin = order_pin_now & m.pinned_prev if cfg.pin_lag else order_pin_now
    else:
        pin_now = collision._pinned_now(sim) if cfg.pin_engaged else None  # noqa: SLF001
        pin = None
        if pin_now is not None:
            pin = pin_now & m.pinned_prev if cfg.pin_lag else pin_now

    x0, y0 = q12(buf.move_x0), q12(buf.move_y0)
    start_ok = np.isfinite(buf.move_x0)
    x0 = np.where(start_ok, x0, q12(w.x))
    y0 = np.where(start_ok, y0, q12(w.y))
    vx, vy = q12(w.x) - x0, q12(w.y) - y0
    top = smax_all(sim)
    s = np.maximum(top, _isqrt_all(vx * vx + vy * vy))
    r_q = q12(w.radius)
    inner_r_q = r_q.copy()
    if cfg.native_wall_clearance:
        for ut, ir in INNER_RADIUS_Q12.items():
            inner_r_q = np.where(w.unit_type == ut, ir, inner_r_q)
    slot_ids = np.where(m.native_slot >= 0, m.native_slot, np.arange(sim.n_slots)[None, :])
    tgt_slots = _unit_target_slots(sim)
    base_ctx = _ContactContext(
        x0=x0,
        y0=y0,
        vx=vx,
        vy=vy,
        r=r_q,
        s=s,
        top=top,
        moving=moving_m,
        pin=pin,
        state=m.native_state,
        flags=m.native_flags,
        hold_still=m.native_hold_still,
        order_kind=w.order_kind,
        target=tgt_slots,
        slot_id=slot_ids,
        inner_r=inner_r_q,
    )
    out_vx, out_vy = vx.copy(), vy.copy()
    loop = w.game_loop
    min_ground = 1 if (cfg.native_batch_flags or cfg.native_wall_clearance) else 2
    for b in np.flatnonzero(ground_m.sum(axis=1) >= min_ground).tolist():
        units = np.flatnonzero(ground_m[b]).tolist()
        ctx = (
            replace(base_ctx, wall=_get_wall_map(sim, b)) if cfg.native_wall_clearance else base_ctx
        )
        env = _Env(sim, b, units, ctx)
        env.run(cfg.native_second_pass == 'contacted')
        for i, u in enumerate(units):
            out_vx[b, u], out_vy[b, u] = env.vx[i], env.vy[i]
            if cfg.native_batch_flags or cfg.native_wall_clearance:
                m.native_flags[b, u] = env.flags[i] & 0xFF
            if cfg.native_share_rule == 'priority_tree':
                m.native_hold_still[b, u] = env.hold_still[i]
        if cfg.idle_return:
            for idle, mover in env.pushed:
                collision._mark_pushed(  # noqa: SLF001
                    sim, b, units[idle], units[mover], int(loop[b])
                )

    # Commit modified or out-of-bounds units using half-open box clamping.
    nx_, ny_ = _commit_clip_q12(sim, x0, y0, out_vx, out_vy, ground_m)
    clipped = (nx_ != x0 + out_vx) | (ny_ != y0 + out_vy)
    changed = ground_m & ((out_vx != vx) | (out_vy != vy) | clipped)
    if changed.any():
        w.x[changed] = (nx_ / ONE).astype(np.float32)[changed]
        w.y[changed] = (ny_ / ONE).astype(np.float32)[changed]
    collision._project(sim, ground_m)  # noqa: SLF001
    m.native_vx[:] = np.where(ground_m, q12(w.x) - x0, m.native_vx)
    m.native_vy[:] = np.where(ground_m, q12(w.y) - y0, m.native_vy)
    if not cfg.native_air_sep:
        m.native_vx[:] = np.where(ground_m, m.native_vx, 0)
        m.native_vy[:] = np.where(ground_m, m.native_vy, 0)
    else:
        _air_idle_contact(sim, x0, y0, top, moving_m)
    collision.after_contact(sim, ground_m, moving_m, pin_now)


__all__ = ['_ContactContext', '_Env', '_commit_clip_q12', 'contact']
