"""SC2-style continuous navigation mesh (`pathing='navmesh'`).

Model:

* **Obstacle geometry**: union of impassable cells with each edge shrunk inward by `SHRINK`
  (1/8) and convex corners chamfered by cutting the triangle `x' + y' < 5/8` (where `x'`, `y'`
  are distances to the grid vertex, leaving chamfer endpoints at `(1/8, 1/2)` and `(1/2, 1/8)`
  from the vertex). Concave corners are shrunk without chamfering, matching SC2's
  `RequestQueryPathing` unwalkable footprint.
* **Pathing radius** `rho`: piecewise tier table (`path_radius`) by unit radius; hugging a
  flat obstacle places the unit center at distance `rho` from the shrunk edge, i.e. `rho - 1/8`
  from the footprint edge (e.g., Marine/Zealot 0.3125, Stalker 0.4375, Ultralisk 0.6875).
* **Shortest paths**: any-angle visibility graph over obstacles inflated by `rho`. Graph nodes
  sit outward along each convex chamfer vertex bisector by `rho / cos(theta / 2)` (45-deg turn
  at chamfers); edges require line-of-sight clearance `>= rho`. Ties within `1e-6` pick the
  path furthest to the right of `current -> goal`.
* **Unreachable goals**: when the goal is within `rho` of an obstacle or disconnected, the
  effective destination is the reachable ray endpoint closest to the goal, casting rays toward
  the goal from the current position and every reachable node until blocked at clearance `rho`
  (ties prefer the current position's ray). A 1-tile gap widens to 1.25 after shrink, so units
  with `rho = 0.8125` (radius 1.0) stop at the entrance.

All geometry uses pure Python `float` / NumPy `float64` functions depending only on
`(grid, rho, start, goal)`.

Known differences vs SC2:

* **Corner hugging differs by 0.02-0.12**: this graph targets miter vertices whereas SC2 aims
  at tangent points on the radius-`rho` corner arc, hugging corners slightly tighter.
* **Thor pathing radius vs Ultralisk**: both have radius 1.0, yet in SC2 a Thor stops further
  from a 1-tile gap than an Ultralisk (~0.83 offset); here pathing radius depends only on unit
  radius, so both use the same tier.
* **Corner jitter**: Ultralisk exhibits a 1-loop heading jitter from local collision at corners
  in SC2, which is not modeled here.

Untested regimes: cliff/map-border chamfer behavior and radii between 0.625 and 1.0 or above
1.0 (handled by linear interpolation and extrapolation).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

SHRINK = 0.125  # inward shrink distance per obstacle edge
CHAMFER = 0.5  # chamfer endpoint offset from grid vertex (other axis is SHRINK)
TIE_EPS = 1e-6  # path-length tie tolerance
LOS_EPS = 1e-6  # line-of-sight clearance tolerance (nodes sit exactly at rho)
CACHE_LIMIT = 4096  # max per-goal cache entries before clearing
# Radius -> pathing radius rho tiers (radii 0.375 and 0.5 share trajectories);
# linearly interpolated inside and extrapolated above.
PATH_RADIUS: tuple[tuple[float, float], ...] = ((0.5, 0.4375), (0.625, 0.5625), (1.0, 0.8125))


def path_radius(radius: float) -> float:
    """Return the pathing radius `rho` for a unit of given collision `radius`."""
    (r0, p0), (r1, p1), (r2, p2) = PATH_RADIUS
    if radius <= r0:
        return p0
    if radius <= r1:
        return p0 + (radius - r0) * (p1 - p0) / (r1 - r0)
    return p1 + (radius - r1) * (p2 - p1) / (r2 - r1)


@dataclass
class NavMesh:
    """Visibility graph for one grid at pathing radius `rho`. `edges` are obstacle segments
    `float64[E, 4]` `(x0, y0, x1, y1)`, `nodes` are vertices `float64[N, 2]`, and `adj` is
    pairwise visibility distance `float64[N, N]` (`inf` if blocked).
    """

    rho: float
    edges: np.ndarray
    nodes: np.ndarray
    adj: np.ndarray
    # goal -> shortest path length from each node (inf if unreachable); see `cost_to`
    cost_cache: dict[tuple[float, float], np.ndarray] = field(default_factory=dict)
    # goal -> ray endpoint from each node toward goal `float64[N, 2]`; see `destination`
    ray_cache: dict[tuple[float, float], np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------- Obstacle edges


def obstacle_edges(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """Return shrunk/chamfered obstacle boundary segments and convex corner tuples
    `(x, y, nx, ny)` (one per chamfer endpoint; outward bisector normal normalized in `build`).

    Cells outside the grid are treated as impassable (map borders match obstacles).
    """
    h, w = grid.shape
    blocked = np.ones((h + 2, w + 2), dtype=bool)
    blocked[1:-1, 1:-1] = ~grid.astype(bool)

    def b(ix: int, iy: int) -> bool:  # whether original cell (ix, iy) is impassable
        return bool(blocked[iy + 1, ix + 1])

    segs: list[tuple[float, float, float, float]] = []
    corners: list[tuple[float, float, float, float]] = []
    # Horizontal edges along y = j where (i, j - 1) and (i, j) differ in passability.
    for j in range(0, h + 1):
        for s in (1, -1):  # s=1: blocked below (normal +y); s=-1: blocked above (normal -y)
            bj, fj = (j - 1, j) if s == 1 else (j, j - 1)
            i = -1
            while i <= w:
                if not (b(i, bj) and not b(i, fj)):
                    i += 1
                    continue
                i0 = i
                while i + 1 <= w and b(i + 1, bj) and not b(i + 1, fj):
                    i += 1
                i1 = i  # cells i0..i1
                y = j - s * SHRINK
                # Left end: (i0 - 1, bj) passable -> convex; else ((i0 - 1, fj) blocked) -> concave
                lconvex = not b(i0 - 1, bj)
                rconvex = not b(i1 + 1, bj)
                xa = i0 + CHAMFER if lconvex else i0 - SHRINK
                xb = i1 + 1 - CHAMFER if rconvex else i1 + 1 + SHRINK
                segs.append((xa, y, xb, y))
                for convex, cx, sx in ((lconvex, float(i0), -1.0), (rconvex, float(i1 + 1), 1.0)):
                    if not convex:
                        continue
                    # Chamfer: (cx - sx*CHAMFER, y) to (cx - sx*SHRINK, j - s*CHAMFER)
                    p = (cx - sx * CHAMFER, y)
                    q = (cx - sx * SHRINK, j - s * CHAMFER)
                    segs.append((*p, *q))
                    # Outward normals: chamfer (sx, s)/sqrt(2), horiz (0, s), vert (sx, 0)
                    corners.append((*p, sx / math.sqrt(2.0), s * (1.0 + 1.0 / math.sqrt(2.0))))
                    corners.append((*q, sx * (1.0 + 1.0 / math.sqrt(2.0)), s / math.sqrt(2.0)))
                i += 1
    # Vertical edges along x = i where (i - 1, j) and (i, j) differ; chamfers added above.
    for i in range(0, w + 1):
        for s in (1, -1):  # s=1: blocked left (normal +x)
            bi, fi = (i - 1, i) if s == 1 else (i, i - 1)
            j = -1
            while j <= h:
                if not (b(bi, j) and not b(fi, j)):
                    j += 1
                    continue
                j0 = j
                while j + 1 <= h and b(bi, j + 1) and not b(fi, j + 1):
                    j += 1
                j1 = j
                x = i - s * SHRINK
                ya = j0 + CHAMFER if not b(bi, j0 - 1) else j0 - SHRINK
                yb = j1 + 1 - CHAMFER if not b(bi, j1 + 1) else j1 + 1 + SHRINK
                segs.append((x, ya, x, yb))
                j += 1
    return np.asarray(segs, dtype=np.float64).reshape(-1, 4), corners


# ---------------------------------------------------------------- Distance & visibility


def _point_seg_dist(px: np.ndarray, py: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Distance `[K, E]` from points `(px, py)[K]` to segments `e[E, 4]`."""
    x0, y0, x1, y1 = (e[:, k][None, :] for k in range(4))
    dx, dy = x1 - x0, y1 - y0
    ll = dx * dx + dy * dy
    t = np.where(
        ll > 0, ((px[:, None] - x0) * dx + (py[:, None] - y0) * dy) / np.where(ll > 0, ll, 1), 0
    )
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px[:, None] - (x0 + t * dx), py[:, None] - (y0 + t * dy))


def seg_clearance(a: np.ndarray, c: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Minimum distance `[K]` from `a[K, 2] -> c[K, 2]` to edges `e[E, 4]` (0 if crossing)."""
    if e.shape[0] == 0:
        return np.full(a.shape[0], np.inf)
    ax, ay, cx, cy = a[:, 0:1], a[:, 1:2], c[:, 0:1], c[:, 1:2]
    x0, y0, x1, y1 = (e[:, k][None, :] for k in range(4))

    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(ax, ay, cx, cy, x0, y0)
    o2 = orient(ax, ay, cx, cy, x1, y1)
    o3 = orient(x0, y0, x1, y1, ax, ay)
    o4 = orient(x0, y0, x1, y1, cx, cy)
    cross = (o1 * o2 < 0) & (o3 * o4 < 0)
    seg = np.concatenate([a, c], axis=1)
    d = np.minimum(
        np.minimum(_point_seg_dist(a[:, 0], a[:, 1], e), _point_seg_dist(c[:, 0], c[:, 1], e)),
        np.minimum(
            _point_seg_dist(e[:, 0], e[:, 1], seg).T, _point_seg_dist(e[:, 2], e[:, 3], seg).T
        ),
    )
    return np.where(cross, 0.0, d).min(axis=1)


def point_clearance(x: float, y: float, e: np.ndarray) -> float:
    if e.shape[0] == 0:
        return math.inf
    return float(_point_seg_dist(np.array([x]), np.array([y]), e).min())


def visible_from(mesh: NavMesh, x: float, y: float, pts: np.ndarray) -> np.ndarray:
    """Return bool mask of `pts[K, 2]` visible from `(x, y)` with clearance `>= rho` (clamped
    to the start's own clearance if pushed closer than `rho`, so units can move back out).
    """
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    need = max(min(mesh.rho, point_clearance(x, y, mesh.edges)) - LOS_EPS, LOS_EPS)
    a = np.broadcast_to(np.array([x, y], dtype=np.float64), pts.shape)
    return seg_clearance(np.ascontiguousarray(a), pts, mesh.edges) >= need


def visible_many(
    mesh: NavMesh, x: np.ndarray, y: np.ndarray, gx: np.ndarray, gy: np.ndarray
) -> np.ndarray:
    """Batched `visible_from(mesh, x[k], y[k], [[gx[k], gy[k]]])`, bit-identical per row."""
    k = x.shape[0]
    if k == 0:
        return np.zeros(0, dtype=bool)
    e = mesh.edges
    if e.shape[0] == 0:
        return np.ones(k, dtype=bool)
    return _visible_need(mesh, x, y, gx, gy, _need_many(mesh, x, y))


def _need_many(mesh: NavMesh, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row start clearance threshold `[K]` matching `visible_from`."""
    e = mesh.edges
    out = np.empty(x.shape[0])
    step = max(1, 1_000_000 // max(1, e.shape[0]))
    for k0 in range(0, x.shape[0], step):
        s = slice(k0, k0 + step)
        pc = _point_seg_dist(x[s], y[s], e).min(axis=1)
        # max(min(rho, pc) - eps, eps): matches Python min/max tie behavior
        m = np.where(pc < mesh.rho, pc, mesh.rho) - LOS_EPS
        out[s] = np.where(LOS_EPS > m, LOS_EPS, m)
    return out


def _visible_need(
    mesh: NavMesh, x: np.ndarray, y: np.ndarray, gx: np.ndarray, gy: np.ndarray, need: np.ndarray
) -> np.ndarray:
    """Return per-row `(x, y) -> (gx, gy)` clearance `>= need` (from `_need_many`)."""
    e = mesh.edges
    out = np.empty(x.shape[0], dtype=bool)
    step = max(1, 1_000_000 // max(1, e.shape[0]))
    for k0 in range(0, x.shape[0], step):
        s = slice(k0, k0 + step)
        a = np.stack([x[s], y[s]], axis=1)
        c = np.stack([gx[s], gy[s]], axis=1)
        out[s] = seg_clearance(a, c, e) >= need[s]
    return out


# ---------------------------------------------------------------- Graph


def build(grid: np.ndarray, rho: float) -> NavMesh:
    edges, corners = obstacle_edges(grid)
    nodes: list[tuple[float, float]] = []
    for px, py, nx, ny in corners:
        n = math.hypot(nx, ny)
        ux, uy = nx / n, ny / n
        # 45-deg flat-to-chamfer corner: offset rho / cos(22.5 deg) gives clearance rho to both
        k = rho / math.cos(math.pi / 8.0)
        nodes.append((px + k * ux, py + k * uy))
    pts = np.asarray(nodes, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0]:
        d = _point_seg_dist(pts[:, 0], pts[:, 1], edges).min(axis=1) if edges.shape[0] else None
        if d is not None:
            pts = pts[d >= rho - LOS_EPS]
        pts = np.unique(pts, axis=0)  # sorted unique order independent of construction
    n = pts.shape[0]
    adj = np.full((n, n), np.inf)
    if n:
        ii, jj = np.triu_indices(n, 1)
        chunk = max(1, 2_000_000 // max(1, edges.shape[0]))  # cap [K, E] scratch memory
        for k0 in range(0, ii.size, chunk):
            a, c = ii[k0 : k0 + chunk], jj[k0 : k0 + chunk]
            ok = seg_clearance(pts[a], pts[c], edges) >= rho - LOS_EPS
            dist = np.hypot(*(pts[c] - pts[a]).T)
            adj[a[ok], c[ok]] = dist[ok]
            adj[c[ok], a[ok]] = dist[ok]
    return NavMesh(rho=rho, edges=edges, nodes=pts, adj=adj)


def cost_to(mesh: NavMesh, gx: float, gy: float) -> np.ndarray:
    """Shortest path lengths from all nodes to `(gx, gy)` via Dijkstra (`inf` if blocked)."""
    key = (gx, gy)
    cached = mesh.cost_cache.get(key)
    if cached is not None:
        return cached
    if len(mesh.cost_cache) >= CACHE_LIMIT:  # chasing moves the goal every loop: bound cache
        mesh.cost_cache.clear()
    n = mesh.nodes.shape[0]
    cost = np.full(n, np.inf)
    if n and point_clearance(gx, gy, mesh.edges) >= mesh.rho - LOS_EPS:
        seen = visible_from(mesh, gx, gy, mesh.nodes)
        d0 = np.hypot(mesh.nodes[:, 0] - gx, mesh.nodes[:, 1] - gy)
        heap = [(float(d0[i]), int(i)) for i in np.flatnonzero(seen)]
        for c, i in heap:
            cost[i] = c
        heapq.heapify(heap)
        done = np.zeros(n, dtype=bool)
        while heap:
            c, i = heapq.heappop(heap)
            if done[i]:
                continue
            done[i] = True
            nc = c + mesh.adj[i]
            better = (nc < cost) & ~done
            for j in np.flatnonzero(better):
                cost[j] = nc[j]
                heapq.heappush(heap, (float(nc[j]), int(j)))
    mesh.cost_cache[key] = cost
    return cost


def ray_end(mesh: NavMesh, x: float, y: float, gx: float, gy: float) -> tuple[float, float]:
    """Advance from `(x, y)` toward `(gx, gy)` up to obstacle clearance `rho` (40 bisections)."""
    lo, hi = 0.0, 1.0
    if bool(visible_from(mesh, x, y, np.array([[gx, gy]]))[0]):
        return gx, gy
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        p = np.array([[x + mid * (gx - x), y + mid * (gy - y)]])
        if bool(visible_from(mesh, x, y, p)[0]):
            lo = mid
        else:
            hi = mid
    return x + lo * (gx - x), y + lo * (gy - y)


def ray_end_many(
    mesh: NavMesh, x: np.ndarray, y: np.ndarray, gx: np.ndarray, gy: np.ndarray
) -> np.ndarray:
    """Batched `ray_end` `[K, 2]`, bit-identical to scalar calls: bisects all rows in lockstep
    while computing each start's clearance threshold once.
    """
    k = x.shape[0]
    if k == 0:
        return np.zeros((0, 2))
    e = mesh.edges
    if e.shape[0] == 0:
        return np.stack([gx, gy], axis=1).astype(np.float64)
    step = max(1, 1_000_000 // e.shape[0])
    if k > step:  # rows are independent: chunk to cap [K, E] scratch memory
        return np.concatenate(
            [
                ray_end_many(mesh, *(v[k0 : k0 + step] for v in (x, y, gx, gy)))
                for k0 in range(0, k, step)
            ]
        )
    vis = _FixedStart(e, x, y, _need_many(mesh, x, y))
    seen = vis(gx, gy)
    lo, hi = np.zeros(k), np.ones(k)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ok = vis(x + mid * (gx - x), y + mid * (gy - y))
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    ex = np.where(seen, gx, x + lo * (gx - x))
    ey = np.where(seen, gy, y + lo * (gy - y))
    return np.stack([ex, ey], axis=1)


class _FixedStart:
    """Bit-identical `seg_clearance(a, c, e) >= need` for fixed start points `(x[k], y[k])`:
    precomputes all start-and-edge terms (`da`, relative endpoints, `o3`) once and evaluates only
    endpoint-dependent terms on each bisection step.
    """

    def __init__(self, e: np.ndarray, x: np.ndarray, y: np.ndarray, need: np.ndarray) -> None:
        self.need = need
        self.ax, self.ay = x[:, None], y[:, None]  # [K, 1]
        x0, y0, x1, y1 = (e[:, j][None, :] for j in range(4))  # [1, E]
        self.x0, self.y0 = x0, y0
        self.dx, self.dy = x1 - x0, y1 - y0
        ll = self.dx * self.dx + self.dy * self.dy
        self.llpos, self.llsafe = ll > 0, np.where(ll > 0, ll, 1)
        ax, ay = self.ax, self.ay
        # (y0 - ay), (x0 - ax) for orient(ax, ay, cx, cy, x0, y0); likewise for o2
        self.y0a, self.x0a, self.y1a, self.x1a = y0 - ay, x0 - ax, y1 - ay, x1 - ax
        self.o3 = self.dx * (ay - y0) - self.dy * (ax - x0)
        self.da = self._edge_dist(x, y)
        # Edge endpoints relative to query segment start `a`: `px[:, None] - x0` [E, K]
        self.e0x, self.e0y = e[:, 0][:, None], e[:, 1][:, None]
        self.e1x, self.e1y = e[:, 2][:, None], e[:, 3][:, None]
        sx, sy = x[None, :], y[None, :]
        self.sx, self.sy = sx, sy
        self.p0x, self.p0y = self.e0x - sx, self.e0y - sy
        self.p1x, self.p1y = self.e1x - sx, self.e1y - sy

    def _edge_dist(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Compute `_point_seg_dist(px, py, e)` `[K, E]`."""
        x0, y0, dx, dy = self.x0, self.y0, self.dx, self.dy
        t = np.where(
            self.llpos, ((px[:, None] - x0) * dx + (py[:, None] - y0) * dy) / self.llsafe, 0
        )
        t = np.clip(t, 0.0, 1.0)
        return np.hypot(px[:, None] - (x0 + t * dx), py[:, None] - (y0 + t * dy))

    def _to_query(self, qx, qy, wx, wy, dx, dy, llpos, llsafe) -> np.ndarray:
        """Compute `_point_seg_dist(edge_endpoints, seg).T` `[K, E]` with precomputed `wx, wy`."""
        t = np.where(llpos, (wx * dx + wy * dy) / llsafe, 0)
        t = np.clip(t, 0.0, 1.0)
        return np.hypot(qx - (self.sx + t * dx), qy - (self.sy + t * dy)).T

    def __call__(self, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        ax, ay = self.ax, self.ay
        ccx, ccy = cx[:, None], cy[:, None]
        o1 = (ccx - ax) * self.y0a - (ccy - ay) * self.x0a
        o2 = (ccx - ax) * self.y1a - (ccy - ay) * self.x1a
        o4 = self.dx * (ccy - self.y0) - self.dy * (ccx - self.x0)
        cross = (o1 * o2 < 0) & (self.o3 * o4 < 0)
        sdx, sdy = cx[None, :] - self.sx, cy[None, :] - self.sy
        sll = sdx * sdx + sdy * sdy
        spos, ssafe = sll > 0, np.where(sll > 0, sll, 1)
        d = np.minimum(
            np.minimum(self.da, self._edge_dist(cx, cy)),
            np.minimum(
                self._to_query(self.e0x, self.e0y, self.p0x, self.p0y, sdx, sdy, spos, ssafe),
                self._to_query(self.e1x, self.e1y, self.p1x, self.p1y, sdx, sdy, spos, ssafe),
            ),
        )
        return np.where(cross, 0.0, d).min(axis=1) >= self.need


def _next(mesh: NavMesh, x: float, y: float, gx: float, gy: float) -> tuple[float, float] | None:
    """Return the next waypoint from `(x, y)` toward `(gx, gy)`, or None if unreachable."""
    if bool(visible_from(mesh, x, y, np.array([[gx, gy]]))[0]):
        return gx, gy
    return _next_hidden(mesh, x, y, gx, gy)


def _next_hidden(
    mesh: NavMesh, x: float, y: float, gx: float, gy: float
) -> tuple[float, float] | None:
    """Graph-routed branch of `_next` when `(gx, gy)` is not directly visible from `(x, y)`."""
    cost = cost_to(mesh, gx, gy)
    if not np.isfinite(cost).any():
        return None
    seen = visible_from(mesh, x, y, mesh.nodes) & np.isfinite(cost)
    if not seen.any():
        return None
    idx = np.flatnonzero(seen)
    pts = mesh.nodes[idx]
    gap = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    idx, gap = idx[gap > TIE_EPS], gap[gap > TIE_EPS]  # already standing on node: advance past it
    if idx.size == 0:
        return None
    total = gap + cost[idx]
    best = float(total.min())
    tie = idx[total <= best + TIE_EPS]
    # Break equal-length ties in favor of the rightmost node: minimum cross (g - p) x (n - p)
    side = (gx - x) * (mesh.nodes[tie, 1] - y) - (gy - y) * (mesh.nodes[tie, 0] - x)
    k = int(tie[int(np.argmin(side))])
    return float(mesh.nodes[k, 0]), float(mesh.nodes[k, 1])


def destination(
    mesh: NavMesh,
    x: float,
    y: float,
    gx: float,
    gy: float,
    *,
    reachable: bool | None = None,
    ray: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return `(gx, gy)` if reachable, else the closest reachable ray endpoint (see module doc).
    Optional `reachable` and `ray` reuse precomputed `_next(...) is not None` and `ray_end(...)`.
    """
    if reachable is None:
        reachable = _next(mesh, x, y, gx, gy) is not None
    if reachable:
        return gx, gy
    best = ray if ray is not None else ray_end(mesh, x, y, gx, gy)
    d_best = math.hypot(best[0] - gx, best[1] - gy) - TIE_EPS  # ties prefer current position's ray
    # Cast rays toward goal from every node in the visible connected component (cached by goal).
    reach = _reachable(mesh, visible_from(mesh, x, y, mesh.nodes))
    if reach.any():
        ends = mesh.ray_cache.get((gx, gy))
        if ends is None:
            if len(mesh.ray_cache) >= CACHE_LIMIT:
                mesh.ray_cache.clear()
            n = mesh.nodes.shape[0]
            ends = ray_end_many(
                mesh, mesh.nodes[:, 0], mesh.nodes[:, 1], np.full(n, gx), np.full(n, gy)
            )
            mesh.ray_cache[(gx, gy)] = ends
        idx = np.flatnonzero(reach)
        d = np.hypot(ends[idx, 0] - gx, ends[idx, 1] - gy)
        k = int(np.argmin(d))
        if float(d[k]) < d_best:
            best = (float(ends[idx[k], 0]), float(ends[idx[k], 1]))
    return best


def _reachable(mesh: NavMesh, seed: np.ndarray) -> np.ndarray:
    reach = seed.copy()
    frontier: list[int] = [int(i) for i in np.flatnonzero(seed)]
    finite = np.isfinite(mesh.adj)
    while frontier:
        i = frontier.pop()
        for j in np.flatnonzero(finite[i] & ~reach):
            reach[j] = True
            frontier.append(int(j))
    return reach


def route(
    mesh: NavMesh, x: float, y: float, gx: float, gy: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return `(waypoint, destination)` for this loop, or `(x, y)` as waypoint if fully stuck.

    Equivalent to `end = destination(...)` followed by `_next(mesh, x, y, *end)`, avoiding the
    second `_next` call when the goal is reachable.
    """
    wp = _next(mesh, x, y, gx, gy)
    if wp is not None:
        return wp, (gx, gy)
    end = destination(mesh, x, y, gx, gy, reachable=False)
    wp = _next(mesh, x, y, *end)
    return (wp if wp is not None else (x, y)), end


def route_many(
    mesh: NavMesh, x: np.ndarray, y: np.ndarray, gx: np.ndarray, gy: np.ndarray
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Batched `route`, bit-identical per row: batches line-of-sight checks via `visible_many`
    and unreachable ray bisections via `ray_end_many`.
    """
    x, y, gx, gy = (np.asarray(v, dtype=np.float64) for v in (x, y, gx, gy))
    pts = [
        (float(a), float(b), float(c), float(d)) for a, b, c, d in zip(x, y, gx, gy, strict=True)
    ]
    seen = visible_many(mesh, x, y, gx, gy).tolist()
    wps = [(p[2], p[3]) if s else _next_hidden(mesh, *p) for p, s in zip(pts, seen, strict=True)]
    out: list[tuple[tuple[float, float], tuple[float, float]]] = [
        (wp, (p[2], p[3])) if wp is not None else ((p[0], p[1]), (p[2], p[3]))
        for wp, p in zip(wps, pts, strict=True)
    ]
    bad = [k for k, wp in enumerate(wps) if wp is None]
    if not bad:
        return out
    a = (x[bad], y[bad], gx[bad], gy[bad])
    rays = ray_end_many(mesh, *a)
    ends = [
        destination(
            mesh, *pts[k], reachable=False, ray=(float(r[0]), float(r[1]))
        )
        for k, r in zip(bad, rays, strict=True)
    ]  # fmt: skip
    ex = np.array([e[0] for e in ends], dtype=np.float64)
    ey = np.array([e[1] for e in ends], dtype=np.float64)
    seen_end = visible_many(mesh, a[0], a[1], ex, ey).tolist()
    for k, end, s in zip(bad, ends, seen_end, strict=True):
        px, py = pts[k][0], pts[k][1]
        wp = end if s else _next_hidden(mesh, px, py, *end)
        out[k] = ((wp if wp is not None else (px, py)), end)
    return out


__all__ = [
    'CHAMFER',
    'PATH_RADIUS',
    'SHRINK',
    'NavMesh',
    'build',
    'cost_to',
    'destination',
    'obstacle_edges',
    'path_radius',
    'point_clearance',
    'ray_end',
    'ray_end_many',
    'route',
    'route_many',
    'seg_clearance',
    'visible_from',
    'visible_many',
]
