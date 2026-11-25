"""Wall clearance geometry, sliding, and single-edge goal projection for `native`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ... import terrain
from ...sim import Sim
from .q12 import CONTACT_GATE, ONE, _trunc_div2, _trunc_div_int, norm

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


__all__ = [
    'WALL_CHAMFER',
    'WALL_SHRINK',
    'WallMap',
    '_chamfer_closest',
    '_get_wall_map',
    '_point_clear_seg',
    '_segment_clear',
    '_wall_slide_bisect',
    'build_wall_map',
    'ortho_project_edge_q12',
    'project_goal_single_edge',
    'wall_point_correct',
    'workpoint_correct',
]
