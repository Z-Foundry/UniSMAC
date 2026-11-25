"""Fixed-point Q12 ground/air movement and collision model (`steering='native'`,
`collision='native'`).

All unit positions and per-loop displacements use Q12 fixed-point integers
(1/4096 tile resolution) written back to float32 (`|coord| < 2^11` is exact).

Configurable rule groups:
- Contact kernel: `native_share_rule='priority_tree'` (same-priority contact
  share decision tree with stationary HOLD and in-range attack handling),
  `native_batch_flags` (four-pass live batch/wall flag propagation), and
  `native_nbr_order='quantized'` (0.25-tile box-quantized squared-distance
  neighbor ordering with slot-index tie-breaking).
- Local ground steering avoidance: `native_steer_avoid` (bounded 3-neighbor
  lateral corridor avoidance with `4 * LateralAcceleration` turn clamping).
- Arrival, stopping, reissue, air separation, and wall clearance:
  `native_move_reissue` (zero-tolerance same-point MOVE reissue skip),
  `native_stop` (end-of-path deceleration, two-stage blocked progress tolerance,
  and near-goal crowd accumulation), `native_air_sep` (throttled idle air
  separation and moving air flocking), and `native_wall_clearance` (axis-aligned
  blocking-edge clearance, wall sliding, and half-open playable clamping).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from unismac.contract import NO_SLOT, CommandKind

from .. import terrain
from ..sim import Sim, f32
from . import collision, steering

ONE = 4096  # Q12 representation of 1.0
CONTACT_GATE = 0x100  # Contact threshold: radius sum minus 1/16 tile
CONTACT_TARGET = 0xFC  # Separation target: radius sum minus 252/4096 tile
OLD_DIR_MIN = 0x3F  # Minimum previous-loop displacement to blend heading
BLEND_NEW, BLEND_OLD = 16, 15  # Heading blend weights (new vs previous)
W_HALF, W_LOW, W_HIGH = 2048, 512, 3584  # Contact displacement shares (out of 4096)
SMAX_MAX = 0xA000  # Maximum per-loop step speed cap (10.0 tiles/loop)
ACCEL_MAX = 0x3E80  # Maximum acceleration cap (Acceleration=1000)
GROUND_ACCEL_Q12 = 0x3E80
MEDIVAC_ACCEL_Q12 = 37  # 2.3125 * 16 (Medivac lateral/decel acceleration)
QUANT_PAD = 0x2D5  # 725: quantized quarter-tile half-diagonal padding
AIR_STEP_FLOOR = 0x20  # 32: minimum idle air-unit separation step

# Common SMAC unit type IDs
COLOSSUS = 4
BANELING = 9
SIEGE_TANK_SIEGED = 32
SIEGE_TANK = 33
MARINE = 48
REAPER = 49
MARAUDER = 51
THOR = 52
HELLION = 53
MEDIVAC = 54
BATTLECRUISER = 57
ZEALOT = 73
STALKER = 74
PHOENIX = 78
VOID_RAY = 80
IMMORTAL = 83
ZERGLING = 105
HYDRALISK = 107
ULTRALISK = 109
QUEEN = 126
HELLION_TANK = 484

# Unit inner collision radii in Q12 (1/4096 tile).
INNER_RADIUS_Q12: dict[int, int] = {
    COLOSSUS: 0x900,  # 0.5625
    BANELING: 0x600,  # 0.375
    SIEGE_TANK_SIEGED: 0xE00,  # 0.875
    SIEGE_TANK: 0xE00,  # 0.875
    MARINE: 0x600,  # 0.375
    REAPER: 0x600,  # 0.375
    MARAUDER: 0x600,  # 0.375
    THOR: 0xD00,  # 0.8125
    HELLION: 0x800,  # 0.5
    ZEALOT: 0x600,  # 0.375
    STALKER: 0x800,  # 0.5
    IMMORTAL: 0x800,  # 0.5
    ZERGLING: 0x600,  # 0.375
    HYDRALISK: 0x600,  # 0.375
    ULTRALISK: 0xC00,  # 0.75
    QUEEN: 0x800,  # 0.5
    HELLION_TANK: 0x800,  # 0.5
}

# Unit separation radii in Q12 (default 0.5 tile = 0x800).
SEPARATION_RADIUS_Q12: dict[int, int] = {
    COLOSSUS: 0xC00,  # 0.75
    BANELING: 0x600,  # 0.375
    SIEGE_TANK_SIEGED: 0x1000,  # 1.0
    SIEGE_TANK: 0x1000,  # 1.0
    MARINE: 0x600,  # 0.375
    REAPER: 0x600,  # 0.375
    THOR: 0x1000,  # 1.0
    HELLION: 0xA00,  # 0.625
    MEDIVAC: 0xC00,  # 0.75
    BATTLECRUISER: 0x1400,  # 1.25
    ZEALOT: 0x600,  # 0.375
    STALKER: 0xA00,  # 0.625
    PHOENIX: 0xC00,  # 0.75
    VOID_RAY: 0xC00,  # 0.75
    IMMORTAL: 0xA00,  # 0.625
    ZERGLING: 0x600,  # 0.375
    ULTRALISK: 0x1000,  # 1.0
    QUEEN: 0xE00,  # 0.875
    HELLION_TANK: 0xA00,  # 0.625
}
DEFAULT_SEPARATION_RADIUS_Q12 = 0x800

# Unit lateral acceleration caps in Q12 (trunc(accel * 16)).
LATERAL_ACCEL_Q12: dict[int, int] = {
    COLOSSUS: 737,  # 46.0625 * 16
    BANELING: 737,  # 46.0625 * 16
    SIEGE_TANK_SIEGED: 1024,  # 64.0 * 16
    SIEGE_TANK: 1024,  # 64.0 * 16
    MARINE: 737,  # 46.0625 * 16
    REAPER: 737,  # 46.0625 * 16
    MARAUDER: 1106,  # 69.125 * 16
    THOR: 737,  # 46.0625 * 16
    HELLION: 736,  # 46.0 * 16
    MEDIVAC: MEDIVAC_ACCEL_Q12,  # inherits Acceleration=2.3125 -> 37
    BATTLECRUISER: 737,  # 46.0625 * 16
    ZEALOT: 737,  # 46.0625 * 16
    STALKER: 736,  # 46.0 * 16
    ZERGLING: 737,  # 46.0625 * 16
    HYDRALISK: 736,  # 46.0 * 16
    ULTRALISK: 737,  # 46.0625 * 16
    QUEEN: 736,  # 46.0 * 16
    HELLION_TANK: 736,  # 46.0 * 16
}

# Air unit cruise heights (tiles).
AIR_HEIGHT: dict[int, float] = {
    MEDIVAC: 3.75,
    BATTLECRUISER: 3.75,
    PHOENIX: 3.75,
    VOID_RAY: 3.75,
}


def inner_radius_q12(unit_type: int, radius_q12: int) -> int:
    return INNER_RADIUS_Q12.get(unit_type, radius_q12)


def sep_radius_q12(unit_type: int) -> int:
    return SEPARATION_RADIUS_Q12.get(unit_type, DEFAULT_SEPARATION_RADIUS_Q12)


def lateral_accel_q12(unit_type: int, accel_q12: int) -> int:
    return LATERAL_ACCEL_Q12.get(unit_type, accel_q12)


def air_height(unit_type: int, is_flying: bool) -> float:
    if unit_type in AIR_HEIGHT:
        return AIR_HEIGHT[unit_type]
    return 3.75 if is_flying else 0.0


def is_air_only(unit_type: int, is_flying: bool) -> bool:
    """Return True for pure flying units (excludes tall ground walkers such as Colossus)."""
    return bool(is_flying) and unit_type != COLOSSUS


def has_air_collide(unit_type: int, is_flying: bool) -> bool:
    """Return True if Air collide bit is set (flying units and Colossus)."""
    return bool(is_flying) or unit_type == COLOSSUS


def _unit_accel_q12(sim: Sim, b: int, u: int) -> int:
    a = float(sim.params.accel[b, u])
    if math.isinf(a):
        return MEDIVAC_ACCEL_Q12 if int(sim.world.unit_type[b, u]) == MEDIVAC else GROUND_ACCEL_Q12
    return min(max(int(round(a * ONE)), 0), ACCEL_MAX)


def _unit_decel_q12(sim: Sim, b: int, u: int, accel_q12: int) -> int:
    d = float(sim.params.decel[b, u])
    if math.isinf(d):
        return accel_q12
    return min(max(int(round(d * ONE)), 0), ACCEL_MAX)


# ---------------------------------------------------------------- Q12 fixed-point primitives


def q12(a: np.ndarray) -> np.ndarray:
    """Convert float tiles to nearest Q12 integer (NaN maps to 0)."""
    with np.errstate(invalid='ignore'):
        return np.where(np.isfinite(a), np.rint(np.asarray(a, np.float64) * ONE), 0).astype(
            np.int64
        )


def _isqrt_all(n: np.ndarray) -> np.ndarray:
    """Elementwise `floor(sqrt(n))` for int64 `0 <= n < 2^53`."""
    r = np.floor(np.sqrt(n.astype(np.float64))).astype(np.int64)
    r = np.where(r * r > n, r - 1, r)
    return np.where((r + 1) * (r + 1) <= n, r + 1, r)


def _trunc_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Truncating division `a / b` toward zero (`b > 0`)."""
    q = np.abs(a) // b
    return np.where(a < 0, -q, q)


def _trunc_div_int(a: int, b: int) -> int:
    """Scalar integer division truncating toward zero (`b > 0`)."""
    return a // b if a >= 0 else -((-a) // b)


def _trunc_div2(a: int) -> int:
    return a // 2 if a >= 0 else -((-a) // 2)


def _trunc_div_q12(a: int) -> int:
    return a // ONE if a >= 0 else -((-a) // ONE)


def dot_q12(ax: int, ay: int, bx: int, by: int) -> int:
    return ((ax * bx) >> 12) + ((ay * by) >> 12)


def cross_q12(ax: int, ay: int, bx: int, by: int) -> int:
    return ((ax * by) >> 12) - ((ay * bx) >> 12)


def d2q(ax: int, ay: int, bx: int, by: int) -> int:
    return (((ax - bx) ** 2) >> 12) + (((ay - by) ** 2) >> 12)


def quant_key(bx: int, by: int, cx: int, cy: int) -> int:
    """Quarter-tile (512 Q12 center offset) quantized squared-distance neighbor sort key."""
    qx = (bx & ~0x3FF) + 0x200
    qy = (by & ~0x3FF) + 0x200
    return (((qx - cx) ** 2) >> 12) + (((qy - cy) ** 2) >> 12)


def facing_vec_q12(facing: float) -> tuple[int, int]:
    """Q12 unit facing vector `(cos f, sin f)` used for coincident-position tie-breaking."""
    return round(math.cos(facing) * ONE), round(math.sin(facing) * ONE)


def norm(vx: int, vy: int) -> tuple[int, int, int]:
    """Fixed-point Q12 vector normalization returning `(nx, ny, length)`.

    Pre-scales `s = vx^2 + vy^2` into `[2^28, 2^30]` and performs 8 paired
    (16-bit) inverse-square-root refinement steps with round-half-up output.
    Returns `(0, 0, 0)` for the zero vector.
    """
    sq = vx * vx + vy * vy
    if sq == 0:
        return 0, 0, 0
    if sq > 0x40000000:
        shift = -1
        while True:
            prev_shift = shift
            sq >>= 2
            shift = prev_shift + 1
            if sq <= 0x40000000:
                break
        k = prev_shift + 2
        x = vx >> k
        y = vy >> k
        ret_pre = 13 - shift
        ret_shift = 14 - shift
    else:
        shift = 0
        while sq <= 0x10000000:
            shift += 1
            sq <<= 2
        x = vx << shift
        y = vy << shift
        ret_pre = shift + 14
        ret_shift = shift + 15
    rad = sq
    rem = 0x40000000 - rad
    step_sq = rad
    acc_x = x
    acc_y = y
    inc_x = x
    inc_y = y
    for _ in range(8):
        trial = (step_sq >> 1) + rad
        acc_x <<= 1
        acc_y <<= 1
        thresh = trial + rad
        rem <<= 1
        if rem >= thresh:
            rem_mid = rem - thresh
            acc_x += inc_x
            acc_y += inc_y
        else:
            trial = rad
            rem_mid = rem
        step_sq >>= 2
        inc_x >>= 1
        inc_y >>= 1
        rad = trial + step_sq
        thresh = rad + trial
        rem_mid <<= 1
        if rem_mid >= thresh:
            rem = rem_mid - thresh
            acc_x += inc_x
            acc_y += inc_y
        else:
            rad = trial
            rem = rem_mid
    nx = (acc_x + 0x400) >> 11
    ny = (acc_y + 0x400) >> 11
    if ret_pre >= 0:
        length = (rad + (1 << ret_pre)) >> ret_shift
    elif ret_shift >= 0:
        length = rad >> ret_shift
    else:
        length = rad << (-ret_shift)
    return nx, ny, length


def norm_all(vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized batch version of `norm`, bitwise-identical to scalar `norm`."""
    vx_i = np.asarray(vx, dtype=np.int64)
    vy_i = np.asarray(vy, dtype=np.int64)
    sq = vx_i * vx_i + vy_i * vy_i
    nz = sq > 0
    if not nz.any():
        z = np.zeros_like(vx_i)
        return z, z.copy(), z.copy()
    sv = sq[nz].copy()
    xv = vx_i[nz].copy()
    yv = vy_i[nz].copy()
    big = sv > 0x40000000
    shift = np.where(big, -1, 0).astype(np.int64)
    if big.any():
        sb = sv[big]
        shift_b = shift[big]
        while True:
            step = sb > 0x40000000
            if not step.any():
                break
            sb[step] >>= 2
            shift_b[step] += 1
        kb = shift_b + 1
        sv[big] = sb
        shift[big] = shift_b
        xv[big] >>= kb
        yv[big] >>= kb
    small = ~big
    if small.any():
        ss = sv[small]
        shift_s = shift[small]
        while True:
            step = ss <= 0x10000000
            if not step.any():
                break
            ss[step] <<= 2
            shift_s[step] += 1
        sv[small] = ss
        shift[small] = shift_s
        xv[small] <<= shift_s
        yv[small] <<= shift_s
    ret_pre = np.where(big, 13 - shift, shift + 14)
    ret_shift = np.where(big, 14 - shift, shift + 15)
    rad = sv
    rem = 0x40000000 - rad
    step_sq = rad.copy()
    acc_x = xv.copy()
    acc_y = yv.copy()
    inc_x = xv.copy()
    inc_y = yv.copy()
    for _ in range(8):
        trial = (step_sq >> 1) + rad
        acc_x <<= 1
        acc_y <<= 1
        thresh = trial + rad
        rem <<= 1
        ge1 = rem >= thresh
        rem_mid = np.where(ge1, rem - thresh, rem)
        trial = np.where(ge1, trial, rad)
        acc_x = np.where(ge1, acc_x + inc_x, acc_x)
        acc_y = np.where(ge1, acc_y + inc_y, acc_y)
        step_sq >>= 2
        inc_x >>= 1
        inc_y >>= 1
        rad = trial + step_sq
        thresh = rad + trial
        rem_mid <<= 1
        ge2 = rem_mid >= thresh
        rem = np.where(ge2, rem_mid - thresh, rem_mid)
        rad = np.where(ge2, rad, trial)
        acc_x = np.where(ge2, acc_x + inc_x, acc_x)
        acc_y = np.where(ge2, acc_y + inc_y, acc_y)
    nx_v = (acc_x + 0x400) >> 11
    ny_v = (acc_y + 0x400) >> 11
    pos_pre = ret_pre >= 0
    len_v = np.empty_like(rad)
    if pos_pre.any():
        len_v[pos_pre] = (rad[pos_pre] + (1 << ret_pre[pos_pre])) >> ret_shift[pos_pre]
    if (~pos_pre).any():
        rs = ret_shift[~pos_pre]
        len_v[~pos_pre] = np.where(
            rs >= 0, rad[~pos_pre] >> np.maximum(rs, 0), rad[~pos_pre] << np.maximum(-rs, 0)
        )
    nx_out = np.zeros_like(vx_i)
    ny_out = np.zeros_like(vx_i)
    len_out = np.zeros_like(vx_i)
    nx_out[nz] = nx_v
    ny_out[nz] = ny_v
    len_out[nz] = len_v
    return nx_out, ny_out, len_out


def smax_all(sim: Sim) -> np.ndarray:
    """Per-unit maximum displacement speed Smax in Q12, clamped to `[0, SMAX_MAX]`."""
    p, c = sim.params, sim.combat
    speed = p.speed.astype(np.float64)
    loop = sim.world.game_loop[:, None]
    slowed = (c.slow_from <= loop) & (loop <= c.slow_to)
    if sim.config.charge:
        charging = c.charge_target != NO_SLOT
        speed = np.where(charging, p.charge_speed, speed)
        slowed &= ~charging
    speed = np.where(slowed, speed * c.slow_factor, speed)
    return np.clip(q12(speed), 0, SMAX_MAX)


# ---------------------------------------------------------------- Wall clearance geometry


WALL_SHRINK = 0x200  # 1/8 tile (512 Q12): corner chamfer shrink (matches navmesh.SHRINK)
WALL_CHAMFER = 0x800  # 1/2 tile (2048 Q12): flat-edge trim (matches navmesh.CHAMFER)


@dataclass(frozen=True)
class WallMap:
    """Axis-aligned blocking edges and 45-deg convex chamfers extracted from `MapInfo.pathing`.

    Flat edges stay on cell boundaries and use `q = InnerRadius - 0x100` clearance; convex
    corners trim flat edges per `navmesh.CHAMFER`/`SHRINK` and use 45-deg chamfers with
    `q + 1/8` clearance so both ends meet flat offsets seamlessly without pushing turning
    units back at obstacle corners.
    """

    grid: np.ndarray  # bool[H, W], True = passable
    width: int
    height: int
    # Vertical edges (y0, y1, ex, nx) in Q12, nx in {+1, -1} pointing to passable side
    vedges: tuple[tuple[int, int, int, int], ...]
    # Horizontal edges (x0, x1, ey, ny) in Q12, ny in {+1, -1} pointing to passable side
    hedges: tuple[tuple[int, int, int, int], ...]
    # Convex 45-deg chamfers (ax, ay, bx, by, sx, sy) in Q12, (sx, sy) -> passable quadrant
    chamfers: tuple[tuple[int, int, int, int, int, int], ...]

    def passable_cell(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.width and 0 <= iy < self.height and bool(self.grid[iy, ix])

    def passable_q12(self, x: int, y: int) -> bool:
        return self.passable_cell(x >> 12, y >> 12)


def build_wall_map(grid: np.ndarray) -> WallMap:
    """Build unshrunk flat edges and 45-deg convex chamfers from `MapInfo.pathing`."""
    g = np.asarray(grid, dtype=np.bool_)
    h, w = g.shape

    def blocked(ix: int, iy: int) -> bool:
        return not (0 <= ix < w and 0 <= iy < h and bool(g[iy, ix]))

    vedges: list[tuple[int, int, int, int]] = []
    for i in range(w + 1):
        for nx in (1, -1):
            pass_i, blk_i = (i, i - 1) if nx == 1 else (i - 1, i)
            j = 0
            while j < h:
                if blocked(blk_i, j) and not blocked(pass_i, j):
                    j0 = j
                    while j + 1 < h and blocked(blk_i, j + 1) and not blocked(pass_i, j + 1):
                        j += 1
                    ya = (j0 << 12) + (WALL_CHAMFER if not blocked(blk_i, j0 - 1) else 0)
                    yb = ((j + 1) << 12) - (WALL_CHAMFER if not blocked(blk_i, j + 1) else 0)
                    if ya < yb:
                        vedges.append((ya, yb, i << 12, nx))
                j += 1

    hedges: list[tuple[int, int, int, int]] = []
    chamfers: list[tuple[int, int, int, int, int, int]] = []
    for j in range(h + 1):
        for ny in (1, -1):
            pass_j, blk_j = (j, j - 1) if ny == 1 else (j - 1, j)
            i = 0
            while i < w:
                if blocked(i, blk_j) and not blocked(i, pass_j):
                    i0 = i
                    while i + 1 < w and blocked(i + 1, blk_j) and not blocked(i + 1, pass_j):
                        i += 1
                    lconvex = not blocked(i0 - 1, blk_j)
                    rconvex = not blocked(i + 1, blk_j)
                    xa = (i0 << 12) + (WALL_CHAMFER if lconvex else 0)
                    xb = ((i + 1) << 12) - (WALL_CHAMFER if rconvex else 0)
                    if xa < xb:
                        hedges.append((xa, xb, j << 12, ny))
                    cy = j << 12
                    if lconvex:
                        cx = i0 << 12
                        chamfers.append(
                            (
                                cx + WALL_CHAMFER,
                                cy - ny * WALL_SHRINK,
                                cx + WALL_SHRINK,
                                cy - ny * WALL_CHAMFER,
                                -1,
                                ny,
                            )
                        )
                    if rconvex:
                        cx = (i + 1) << 12
                        chamfers.append(
                            (
                                cx - WALL_CHAMFER,
                                cy - ny * WALL_SHRINK,
                                cx - WALL_SHRINK,
                                cy - ny * WALL_CHAMFER,
                                1,
                                ny,
                            )
                        )
                i += 1

    return WallMap(g, w, h, tuple(vedges), tuple(hedges), tuple(chamfers))


def _chamfer_closest(ox: int, oy: int, ax: int, ay: int, bx: int, by: int) -> tuple[int, int]:
    """Return closest point (Q12) on 45-deg chamfer segment `[A, B]` to `(ox, oy)`."""
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    dot = (ox - ax) * abx + (oy - ay) * aby
    if dot <= 0:
        return ax, ay
    if dot >= ab2:
        return bx, by
    return ax + _trunc_div_int(abx * dot, ab2), ay + _trunc_div_int(aby * dot, ab2)


def _get_wall_map(sim: Sim, b: int) -> WallMap | None:
    info = sim.env_map[b]
    if info is None:
        return None
    cached = sim.wall_cache.get(info.map_id)
    if cached is not None and cached[0] is info.pathing:
        wm = cached[1]
        assert isinstance(wm, WallMap)
        return wm
    wm = build_wall_map(info.pathing)
    sim.wall_cache[info.map_id] = (info.pathing, wm)
    return wm


def wall_point_correct(wall: WallMap, px: int, py: int, inner_r: int) -> tuple[int, int, bool]:
    """Correct point against flat edges (`q = inner_r - 0x100`) and convex corner chamfers.

    Only pushes out when perpendicular distance is strictly `< q` (`q + 1/8` for chamfers),
    leaving tangential coordinates unchanged. Returns `(ox, oy, hit)`.
    """
    if inner_r < CONTACT_GATE:
        return px, py, False
    if not wall.passable_q12(px, py):
        return px, py, True
    q = inner_r - CONTACT_GATE
    ox, oy = px, py
    hit = False
    for y0, y1, ex, nx in wall.vedges:
        if y0 < oy < y1:
            dn = (ox - ex) * nx
            if 0 <= dn < q:
                ox = ex + nx * q
                hit = True
    for x0, x1, ey, ny in wall.hedges:
        if x0 < ox < x1:
            dn = (oy - ey) * ny
            if 0 <= dn < q:
                oy = ey + ny * q
                hit = True
    rho = q + WALL_SHRINK
    rho2 = rho * rho
    for ax, ay, bx, by, sx, sy in wall.chamfers:
        # Coarse AABB pre-filter
        if ox < min(ax, bx) - rho or ox > max(ax, bx) + rho:
            continue
        if oy < min(ay, by) - rho or oy > max(ay, by) + rho:
            continue
        clx, cly = _chamfer_closest(ox, oy, ax, ay, bx, by)
        dx, dy = ox - clx, oy - cly
        d2 = dx * dx + dy * dy
        if sx * (ox - ax) + sy * (oy - ay) < 0:
            cnx, cny, _ = norm(sx, sy)
            cx = clx + ((cnx * rho) >> 12)
            cy = cly + ((cny * rho) >> 12)
            if wall.passable_q12(cx, cy):
                ox, oy = cx, cy
                hit = True
        elif 0 < d2 < rho2:
            cnx, cny, _ = norm(dx, dy)
            cx = clx + ((cnx * rho) >> 12)
            cy = cly + ((cny * rho) >> 12)
            if wall.passable_q12(cx, cy):
                ox, oy = cx, cy
                hit = True
    if hit and not wall.passable_q12(ox, oy):
        return px, py, True
    return ox, oy, hit


def _point_clear_seg(wall: WallMap, px: int, py: int, seg_r: int) -> bool:
    """Return True if `(px, py)` clears all flat edges (`seg_r`) and chamfers (`seg_r + 1/8`)."""
    if not wall.passable_q12(px, py):
        return False
    if seg_r <= 0:
        return True
    for y0, y1, ex, nx in wall.vedges:
        if y0 < py < y1 and 0 <= (px - ex) * nx < seg_r:
            return False
    for x0, x1, ey, ny in wall.hedges:
        if x0 < px < x1 and 0 <= (py - ey) * ny < seg_r:
            return False
    c_r = seg_r + WALL_SHRINK
    c_r2 = c_r * c_r
    for ax, ay, bx, by, sx, sy in wall.chamfers:
        if px < min(ax, bx) - c_r or px > max(ax, bx) + c_r:
            continue
        if py < min(ay, by) - c_r or py > max(ay, by) + c_r:
            continue
        if sx * (px - ax) + sy * (py - ay) < 0:
            return False
        clx, cly = _chamfer_closest(px, py, ax, ay, bx, by)
        dx, dy = px - clx, py - cly
        if dx * dx + dy * dy < c_r2:
            return False
    return True


def _segment_clear(wall: WallMap, x0: int, y0: int, x1: int, y1: int, seg_r: int) -> bool:
    """Segment passability check: entire segment clear and endpoint satisfies `seg_r = R -
    0x140`.
    """
    if not wall.passable_q12(x1, y1):
        return False
    if (x0 >> 12) != (x1 >> 12) or (y0 >> 12) != (y1 >> 12):
        if not terrain.line_of_sight(wall.grid, x0 / ONE, y0 / ONE, x1 / ONE, y1 / ONE):
            return False
    return _point_clear_seg(wall, x1, y1, seg_r)


def _wall_slide_bisect(
    wall: WallMap, x0: int, y0: int, px: int, py: int, inner_r: int
) -> tuple[int, int, bool]:
    """Check segment clearance (`R - 0x140`), bisect blocked segments down to `trunc(R/2)`, and
    correct endpoint.
    """
    seg_r = inner_r - 0x140
    if not wall.passable_q12(x0, y0):
        return wall_point_correct(wall, px, py, inner_r)
    if _segment_clear(wall, x0, y0, px, py, seg_r):
        ox, oy, hit = wall_point_correct(wall, px, py, inner_r)
        return (ox, oy, True) if hit else (px, py, False)
    _, _, x_hit = wall_point_correct(wall, x0, y0, inner_r)
    if not x_hit:
        ax, ay = x0, y0
        bx, by = px, py
        half_r = inner_r // 2
        half_r2 = half_r * half_r
        while (ax - bx) ** 2 + (ay - by) ** 2 > half_r2:
            mx = _trunc_div2(ax + bx)
            my = _trunc_div2(ay + by)
            if _segment_clear(wall, ax, ay, mx, my, seg_r):
                ax, ay = mx, my
            else:
                bx, by = mx, my
        for cx, cy in ((bx, by), (ax, ay)):
            ox, oy, _ = wall_point_correct(wall, cx, cy, inner_r)
            if _segment_clear(wall, x0, y0, ox, oy, 0):
                return ox, oy, True
        return x0, y0, True
    ox, oy, _ = wall_point_correct(wall, px, py, inner_r)
    if _segment_clear(wall, x0, y0, ox, oy, 0):
        return ox, oy, True
    return x0, y0, True


def workpoint_correct(
    wall: WallMap, x0: int, y0: int, px: int, py: int, inner_r: int
) -> tuple[int, int, bool]:
    """Work-point wall correction with axis-aligned wall-sliding fallback `(P.x, X.y)` then
    `(X.x, P.y)`.
    """
    if inner_r <= 0:
        ok = wall.passable_q12(px, py) and (
            not wall.passable_q12(x0, y0)
            or terrain.line_of_sight(wall.grid, x0 / ONE, y0 / ONE, px / ONE, py / ONE)
        )
        return (px, py, False) if ok else (x0, y0, True)
    ox, oy, hit = _wall_slide_bisect(wall, x0, y0, px, py, inner_r)
    if hit and ox == x0 and oy == y0:
        if px != x0:
            ox, oy, _ = _wall_slide_bisect(wall, x0, y0, px, y0, inner_r)
        if ox == x0 and oy == y0 and py != y0:
            ox, oy, _ = _wall_slide_bisect(wall, x0, y0, x0, py, inner_r)
    # Final passability re-check
    if not wall.passable_q12(ox, oy):
        return x0, y0, True
    if wall.passable_q12(x0, y0) and ((x0 >> 12) != (ox >> 12) or (y0 >> 12) != (oy >> 12)):
        if not terrain.line_of_sight(wall.grid, x0 / ONE, y0 / ONE, ox / ONE, oy / ONE):
            return x0, y0, True
    return ox, oy, hit


def ortho_project_edge_q12(
    gx: int, gy: int, e0x: int, e0y: int, e1x: int, e1y: int
) -> tuple[int, int]:
    """Orthogonally project point `G` onto finite segment `[E0, E1]` in 1/16-tile integer
    arithmetic.
    """
    ex16 = (e1x - e0x) >> 8
    ey16 = (e1y - e0y) >> 8
    e2 = ex16 * ex16 + ey16 * ey16
    if e2 == 0:
        return e0x, e0y
    dot = ex16 * (gx - e0x) + ey16 * (gy - e0y)
    if dot <= 0:
        return e0x, e0y
    if dot >= 256 * e2:
        return e1x, e1y
    t = _trunc_div_int(dot, e2)
    return e0x + ex16 * t, e0y + ey16 * t


def project_goal_single_edge(
    wall: WallMap, px: int, py: int, gx: int, gy: int, inner_r: int
) -> tuple[int, int] | None:
    """In-transit single-edge orthogonal projection and `InnerRadius` push-out when `W == G` and
    `F == 0`.
    """
    if not wall.passable_q12(px, py):
        return None
    dx, dy = gx - px, gy - py
    best_t = 2.0
    best_edge: tuple[int, int, int, int, int, int] | None = None
    cross_count = 0
    if dx != 0:
        for y0, y1, ex, nx in wall.vedges:
            # Only count crossings from passable toward blocked side (dx * nx < 0)
            if dx * nx >= 0:
                continue
            t = (ex - px) / dx
            if 0.0 <= t <= 1.0:
                yi = py + dy * t
                if y0 <= yi <= y1:
                    cross_count += 1
                    if t < best_t:
                        best_t = t
                        best_edge = (ex, y0, ex, y1, nx, 0)
    if dy != 0:
        for x0, x1, ey, ny in wall.hedges:
            if dy * ny >= 0:
                continue
            t = (ey - py) / dy
            if 0.0 <= t <= 1.0:
                xi = px + dx * t
                if x0 <= xi <= x1:
                    cross_count += 1
                    if t < best_t:
                        best_t = t
                        best_edge = (x0, ey, x1, ey, 0, ny)
    if best_edge is not None and cross_count == 1:
        e0x, e0y, e1x, e1y, nx, ny = best_edge
        prx, pry = ortho_project_edge_q12(gx, gy, e0x, e0y, e1x, e1y)
        q = max(inner_r - CONTACT_GATE, 0)
        ox, oy = prx + nx * q, pry + ny * q
        ox, oy, _ = wall_point_correct(wall, ox, oy, inner_r)
        if wall.passable_q12(ox, oy) and (ox != gx or oy != gy):
            return ox, oy
        return None
    if cross_count == 0 and wall.passable_q12(gx, gy):
        ox, oy, hit = wall_point_correct(wall, gx, gy, inner_r)
        if hit and (ox != gx or oy != gy):
            return ox, oy
    return None


# -------------------------------------------------------- Local ground steering avoidance


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


# -------------------------------------------------------- Arrival and blocked-stop primitives


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


# ---------------------------------------------------------------- Air-unit separation


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


# ---------------------------------------------------------------- State sync & steering


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


def steer(sim: Sim) -> None:
    """Fixed-point Q12 steering stage (`steering='native'`): writes `buf.vel_*`."""
    w, buf, m, cfg = sim.world, sim.buf, sim.motion, sim.config
    seek_x, seek_y = steering._seek_all(sim)  # noqa: SLF001
    has_wp = np.isfinite(buf.waypoint_x)
    ground_alive = w.alive & ~w.is_flying
    is_lag = (cfg.move_lag > 0) & (m.lag_loop == w.game_loop[:, None])

    # Fast vectorized path when advanced native steering options are disabled.
    if not (
        cfg.native_steer_avoid or cfg.native_stop or cfg.native_air_sep or cfg.native_wall_clearance
    ):
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
    w, buf, m, cfg = sim.world, sim.buf, sim.motion, sim.config
    n = len(g_units)
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

    # 1. Early-stage arrival/stop check for MOVE and untargeted ATTACK_MOVE.
    if cfg.native_stop:
        for i, u in enumerate(g_units):
            is_pt_ord = order_arr[i] == int(CommandKind.MOVE) or (
                order_arr[i] == int(CommandKind.ATTACK_MOVE) and engaged_arr[i] == NO_SLOT
            )
            if not is_pt_ord or is_lag[b, u]:
                continue
            px, py = x_arr[i], y_arr[i]
            ox, oy = int(ord_x_all[b, u]), int(ord_y_all[b, u])
            gx, gy = (goal_x_arr[i], goal_y_arr[i]) if has_goal_arr[i] else (ox, oy)
            wx, wy = (wp_x_arr[i], wp_y_arr[i]) if has_wp_arr[i] else (gx, gy)
            inner_r = inner_radius_q12(utype_arr[i], r_arr[i])
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
                vx_arr[i],
                vy_arr[i],
                state_arr[i],
                flags_arr[i],
                r_arr[i],
                inner_r,
                acc_q,
                int(m.native_k[b, u]),
                int(m.native_tol[b, u]),
            )
            m.native_tol[b, u] = new_tol
            if stopped:
                state_arr[i] = new_st
                m.native_state[b, u] = new_st
                order_arr[i] = int(CommandKind.NOOP)
                has_wp_arr[i] = False
                vx_arr[i] = vy_arr[i] = 0
                w.order_kind[b, u] = CommandKind.NOOP
                w.order_target_slot[b, u] = NO_SLOT
                w.order_x[b, u] = w.order_y[b, u] = np.nan
                buf.waypoint_x[b, u] = buf.waypoint_y[b, u] = np.nan
                buf.dest_x[b, u] = buf.dest_y[b, u] = np.nan
                buf.vel_x[b, u] = buf.vel_y[b, u] = f32(0)
                m.native_tol[b, u] = 0
                m.native_k[b, u] = 0
                # Keep native_gx/gy/has_g for subsequent near-goal crowd accumulation.

    # 2. Compute candidate displacement per unit in slot order.
    for i, u in enumerate(g_units):
        if not has_wp_arr[i]:
            vx_arr[i] = vy_arr[i] = 0
            buf.vel_x[b, u] = buf.vel_y[b, u] = f32(0)
            continue
        lag_u = bool(is_lag[b, u])
        px, py = x_arr[i], y_arr[i]
        wx, wy = wp_x_arr[i], wp_y_arr[i]
        gx, gy = (goal_x_arr[i], goal_y_arr[i]) if has_goal_arr[i] else (wx, wy)
        # Sync direct waypoint with projected G when previously modified.
        if proj_g_arr[i] and has_dest[b, u]:
            if wx == int(dest_x_all[b, u]) and wy == int(dest_y_all[b, u]):
                wx, wy = gx, gy
                wp_x_arr[i], wp_y_arr[i] = wx, wy
        s_max = s_arr[i]
        v_old_x, v_old_y = vx_arr[i], vy_arr[i]
        bl = flags_arr[i] & 0x0C
        ux, uy, vlen = norm(v_old_x, v_old_y)
        if vlen <= OLD_DIR_MIN:
            ux, uy = 0, 0
        c_spd = min(vlen, s_max)
        acc_q = _unit_accel_q12(sim, b, u)
        dec_q = _unit_decel_q12(sim, b, u, acc_q)
        lat_a = lateral_accel_q12(utype_arr[i], acc_q)
        inner_r = inner_radius_q12(utype_arr[i], r_arr[i])
        nx, ny, length = norm(wx - px, wy - py)

        is_attack_move = order_arr[i] == int(CommandKind.ATTACK_MOVE) and engaged_arr[i] == NO_SLOT
        is_point_move = order_arr[i] == int(CommandKind.MOVE) or is_attack_move
        q_tol = int(m.native_tol[b, u]) if (cfg.native_stop and is_point_move) else 0
        st_in = state_arr[i]
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
                state_arr[i] = 10
                m.native_state[b, u] = 10
                vx_arr[i], vy_arr[i] = vx_out, vy_out
            buf.vel_x[b, u] = f32(vx_out / ONE)
            buf.vel_y[b, u] = f32(vy_out / ONE)
            continue

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
                vx_arr[i], vy_arr[i] = vx_cand, vy_cand
                state_arr[i] = st_cur
                m.native_state[b, u] = st_cur
            # Preserve exact float32 arrival when order point is off the Q12 grid.
            if proj_g_arr[i] and (wx != int(wx_all[b, u]) or wy != int(wy_all[b, u])):
                buf.vel_x[b, u] = f32(vx_cand / ONE)
                buf.vel_y[b, u] = f32(vy_cand / ONE)
            else:
                buf.vel_x[b, u] = f32(buf.waypoint_x[b, u] - w.x[b, u])
                buf.vel_y[b, u] = f32(buf.waypoint_y[b, u] - w.y[b, u])
        elif length == 0 or step == 0:
            vx_cand, vy_cand = 0, 0
            if not lag_u:
                vx_arr[i], vy_arr[i] = 0, 0
                state_arr[i] = st_cur
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
                    r_arr[i],
                    inner_r,
                    wx,
                    wy,
                    gx,
                    gy,
                    target_arr[i],
                    flags_arr[i],
                    int(m.native_k[b, u]),
                    owner_arr[i],
                    u,
                    x_arr,
                    y_arr,
                    vx_arr,
                    vy_arr,
                    r_arr,
                    owner_arr,
                    g_units,
                    slot_id_arr,
                    state_arr,
                    order_arr,
                    engaged_arr,
                    has_wp_arr,
                    wp_x_arr,
                    wp_y_arr,
                    goal_x_arr,
                    goal_y_arr,
                    has_goal_arr,
                    target_arr,
                    wall,
                    struct_arr,
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
                vx_arr[i], vy_arr[i] = vx_cand, vy_cand
                state_arr[i] = st_cur
                m.native_state[b, u] = st_cur
            buf.vel_x[b, u] = f32(vx_cand / ONE)
            buf.vel_y[b, u] = f32(vy_cand / ONE)

        # 3. Post-step progress update, single-edge wall projection, and crowd tolerance.
        if lag_u or not is_point_move:
            continue
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
                r_arr[i],
                bl,
                g_passable=g_pass,
                is_attack_move=is_attack_move,
                pw2_ge_16r2=pw2 >= (4 * r_arr[i]) ** 2,
            )
            m.native_bx[b, u], m.native_by[b, u] = nbx, nby
            m.native_k[b, u] = nk
            m.native_f[b, u] = nf
        # When blocked on a non-attack MOVE at W == G on an 8-loop beat, project wall edge and
        # accumulate crowd.
        if (not is_attack_move) and bl != 0 and wx == gx and wy == gy:
            if cfg.scan_phase in ('per_unit', 'tag_order', 'tag_rotate'):
                phase_due = (
                    (
                        int(w.game_loop[b])
                        - cfg.acquire_start_loop
                        - int(sim.combat.scan_shift[b, u])
                    )
                    & 7
                ) == 0
            else:
                phase_due = ((int(w.game_loop[b]) + 1 - slot_id_arr[i]) & 7) == 0
            if phase_due:
                # In-transit single-edge projection when F == 0.
                if wall is not None and ntol == 0:
                    proj = project_goal_single_edge(wall, px, py, gx, gy, inner_r)
                    if proj is not None:
                        ng_x, ng_y = proj
                        m.native_gx[b, u], m.native_gy[b, u] = ng_x, ng_y
                        m.native_has_g[b, u] = True
                        gx, gy = ng_x, ng_y
                        goal_x_arr[i], goal_y_arr[i] = ng_x, ng_y
                        buf.dest_x[b, u] = buf.waypoint_x[b, u] = f32(ng_x / ONE)
                        buf.dest_y[b, u] = buf.waypoint_y[b, u] = f32(ng_y / ONE)
                # Near-goal crowded-neighbor area accumulation.
                if cfg.native_stop:
                    rad_gate = q_tol + 10 * r_arr[i]
                    if (px - gx) ** 2 + (py - gy) ** 2 <= rad_gate * rad_gate:
                        q_lim_base = q_tol + r_arr[i] + QUANT_PAD
                        counted: list[int] = []
                        for j in range(n):
                            if j == i or struct_arr[j]:
                                continue
                            rj = r_arr[j]
                            if (
                                quant_key(x_arr[j], y_arr[j], gx, gy)
                                > ((rj + q_lim_base) ** 2) >> 12
                            ):
                                continue
                            if state_arr[j] <= 7:
                                # Hostile or HOLD neighbors count unconditionally;
                                # Attacking allies fall through to the G_j proximity check below.
                                if owner_arr[j] != owner_arr[i] or order_arr[j] == int(
                                    CommandKind.HOLD
                                ):
                                    counted.append(rj)
                                    continue
                            if has_goal_arr[j]:
                                dgjx, dgjy = gx - goal_x_arr[j], gy - goal_y_arr[j]
                                if dgjx * dgjx + dgjy * dgjy < (r_arr[i] + rj) ** 2:
                                    counted.append(rj)
                        if counted:
                            ntol = near_goal_area_step(r_arr[i], q_tol, ntol, counted)
        if cfg.native_stop:
            m.native_tol[b, u] = ntol


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


# ---------------------------------------------------- Contact resolution batches


class _Env:
    """Per-env integer working state for one contact batch (indexed over ground units)."""

    def __init__(
        self,
        sim: Sim,
        b: int,
        units: list[int],
        ctx: dict[str, object],
    ) -> None:
        cfg = sim.config
        self.units = units
        n = len(units)

        def _arr(key: str) -> np.ndarray:
            v = ctx[key]
            assert isinstance(v, np.ndarray)
            return v

        self.x = [int(_arr('x0')[b, u]) for u in units]
        self.y = [int(_arr('y0')[b, u]) for u in units]
        self.vx = [int(_arr('vx')[b, u]) for u in units]
        self.vy = [int(_arr('vy')[b, u]) for u in units]
        self.r = [int(_arr('r')[b, u]) for u in units]
        self.s = [int(_arr('s')[b, u]) for u in units]
        self.top = [int(_arr('top')[b, u]) for u in units]
        self.owner = [int(sim.world.owner[b, u]) for u in units]
        self.facing = [float(sim.world.facing[b, u]) for u in units]
        self.moving = [bool(_arr('moving')[b, u]) for u in units]
        pin_v = ctx.get('pin')
        self.pin = (
            [bool(pin_v[b, u]) for u in units] if isinstance(pin_v, np.ndarray) else [False] * n
        )
        hard_v = ctx.get('hard_pin')
        self.hard_pin = (
            [bool(hard_v[b, u]) for u in units] if isinstance(hard_v, np.ndarray) else [False] * n
        )
        st_v = ctx.get('state')
        self.state = (
            [int(st_v[b, u]) for u in units]
            if isinstance(st_v, np.ndarray)
            else [9 if mv else 0 for mv in self.moving]
        )
        fl_v = ctx.get('flags')
        self.flags = [int(fl_v[b, u]) for u in units] if isinstance(fl_v, np.ndarray) else [0] * n
        hs_v = ctx.get('hold_still')
        self.hold_still = (
            [bool(hs_v[b, u]) for u in units] if isinstance(hs_v, np.ndarray) else [False] * n
        )
        ok_v = ctx.get('order_kind')
        self.order_kind = (
            [int(ok_v[b, u]) for u in units]
            if isinstance(ok_v, np.ndarray)
            else [int(CommandKind.MOVE) if mv else int(CommandKind.NOOP) for mv in self.moving]
        )
        tg_v = ctx.get('target')
        self.target = (
            [int(tg_v[b, u]) for u in units] if isinstance(tg_v, np.ndarray) else [NO_SLOT] * n
        )
        sid_v = ctx.get('slot_id')
        self.slot_id = (
            [int(sid_v[b, u]) for u in units] if isinstance(sid_v, np.ndarray) else list(units)
        )
        ir_v = ctx.get('inner_r')
        self.inner_r = (
            [int(ir_v[b, u]) for u in units] if isinstance(ir_v, np.ndarray) else list(self.r)
        )
        wall_v = ctx.get('wall')
        self.wall: WallMap | None = wall_v if isinstance(wall_v, WallMap) else None
        rel_v = ctx.get('rel')
        self.rel_fn: Callable[[int, int], bool] | None = rel_v if callable(rel_v) else None

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


def _air_idle_contact(
    sim: Sim,
    x0: np.ndarray,
    y0: np.ndarray,
    smax: np.ndarray,
    moving_m: np.ndarray,
) -> None:
    """Idle pure air-unit separation and displacement commit for all flying units."""
    w, m = sim.world, sim.motion
    fly_m = w.alive & w.is_flying
    if not fly_m.any():
        return
    r_all = q12(w.radius)
    for b in np.flatnonzero(fly_m.any(axis=1)).tolist():
        air_coll = [
            u
            for u in np.flatnonzero(w.alive[b]).tolist()
            if has_air_collide(int(w.unit_type[b, u]), bool(w.is_flying[b, u]))
        ]
        # Snapshot prior positions and states.
        px_map = {u: int(x0[b, u]) for u in air_coll}
        py_map = {u: int(y0[b, u]) for u in air_coll}
        tgt_row = _unit_target_slots(sim)[b]
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
    base_ctx: dict[str, object] = {
        'x0': x0,
        'y0': y0,
        'vx': vx,
        'vy': vy,
        'r': r_q,
        's': s,
        'top': top,
        'moving': moving_m,
        'pin': pin,
        'state': m.native_state,
        'flags': m.native_flags,
        'hold_still': m.native_hold_still,
        'order_kind': w.order_kind,
        'target': tgt_slots,
        'slot_id': slot_ids,
        'inner_r': inner_r_q,
    }
    out_vx, out_vy = vx.copy(), vy.copy()
    loop = w.game_loop
    min_ground = 1 if (cfg.native_batch_flags or cfg.native_wall_clearance) else 2
    for b in np.flatnonzero(ground_m.sum(axis=1) >= min_ground).tolist():
        units = np.flatnonzero(ground_m[b]).tolist()
        ctx = {**base_ctx, 'wall': _get_wall_map(sim, b)} if cfg.native_wall_clearance else base_ctx
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
