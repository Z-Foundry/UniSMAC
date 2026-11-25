"""Fixed-point Q12 unit tables, getters, and integer math primitives for `native`."""

from __future__ import annotations

import math

import numpy as np

from unismac.contract import NO_SLOT

from ...sim import Sim

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


__all__ = [
    'ACCEL_MAX',
    'AIR_HEIGHT',
    'AIR_STEP_FLOOR',
    'BANELING',
    'BATTLECRUISER',
    'BLEND_NEW',
    'BLEND_OLD',
    'COLOSSUS',
    'CONTACT_GATE',
    'CONTACT_TARGET',
    'DEFAULT_SEPARATION_RADIUS_Q12',
    'GROUND_ACCEL_Q12',
    'HELLION',
    'HELLION_TANK',
    'HYDRALISK',
    'IMMORTAL',
    'INNER_RADIUS_Q12',
    'LATERAL_ACCEL_Q12',
    'MARAUDER',
    'MARINE',
    'MEDIVAC',
    'MEDIVAC_ACCEL_Q12',
    'OLD_DIR_MIN',
    'ONE',
    'PHOENIX',
    'QUANT_PAD',
    'QUEEN',
    'REAPER',
    'SEPARATION_RADIUS_Q12',
    'SIEGE_TANK',
    'SIEGE_TANK_SIEGED',
    'SMAX_MAX',
    'STALKER',
    'THOR',
    'ULTRALISK',
    'VOID_RAY',
    'W_HALF',
    'W_HIGH',
    'W_LOW',
    'ZEALOT',
    'ZERGLING',
    '_isqrt_all',
    '_trunc_div',
    '_trunc_div2',
    '_trunc_div_int',
    '_trunc_div_q12',
    '_unit_accel_q12',
    '_unit_decel_q12',
    'air_height',
    'cross_q12',
    'd2q',
    'dot_q12',
    'facing_vec_q12',
    'has_air_collide',
    'inner_radius_q12',
    'is_air_only',
    'lateral_accel_q12',
    'norm',
    'norm_all',
    'q12',
    'quant_key',
    'sep_radius_q12',
    'smax_all',
]
