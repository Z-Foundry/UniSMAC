"""Stage 4 global pathing: resolves each unit's target into `buf.waypoint_*` and `buf.dest_*`."""

from __future__ import annotations

import numpy as np

from unismac.contract import NO_SLOT, CommandKind

from .. import navmesh, terrain
from ..sim import Sim, f32


def goal(sim: Sim, b: int, u: int) -> tuple[np.float32, np.float32] | None:
    """Return unit's desired point: MOVE -> order point; ATTACK_MOVE without engaged target ->
    order point, or engaged target position when chasing out of range (`targeting='basic'` only
    engages in-range targets); ATTACK_UNIT / ABILITY (heal) out of range -> target position;
    idle unit under `auto_acquire` with out-of-range engaged target -> target position; else None.
    """
    w = sim.world
    kind = int(w.order_kind[b, u])
    engaged = int(w.engaged_target_slot[b, u]) != NO_SLOT
    if kind == CommandKind.MOVE or (kind == CommandKind.ATTACK_MOVE and not engaged):
        return w.order_x[b, u], w.order_y[b, u]
    if sim.buf.in_range[b, u]:
        return None
    t = NO_SLOT
    if kind in (CommandKind.ATTACK_UNIT, CommandKind.ABILITY):
        t = int(w.order_target_slot[b, u])
    elif kind == CommandKind.ATTACK_MOVE or (kind == CommandKind.NOOP and sim.config.auto_acquire):
        t = int(w.engaged_target_slot[b, u])
    if t != NO_SLOT:
        return w.x[b, t], w.y[b, t]
    return None


def goal_all(sim: Sim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched `goal`: returns `(has_goal, x, y)`, identical per alive unit to `goal`."""
    w, buf = sim.world, sim.buf
    kind = w.order_kind
    in_range = buf.in_range
    amove = kind == CommandKind.ATTACK_MOVE
    engaged = w.engaged_target_slot != NO_SLOT
    to_order = (kind == CommandKind.MOVE) | (amove & ~engaged)
    t = np.full(kind.shape, NO_SLOT, np.int64)
    by_order = (kind == CommandKind.ATTACK_UNIT) | (kind == CommandKind.ABILITY)
    t = np.where(by_order, w.order_target_slot, t)
    t = np.where(amove, w.engaged_target_slot, t)
    if sim.config.auto_acquire:
        t = np.where(kind == CommandKind.NOOP, w.engaged_target_slot, t)
    chase = ~to_order & ~in_range & (t != NO_SLOT)
    idx = t % w.x.shape[1]
    tx = np.take_along_axis(w.x, idx, axis=1)
    ty = np.take_along_axis(w.y, idx, axis=1)
    gx = np.where(to_order, w.order_x, np.where(chase, tx, np.float32(np.nan)))
    gy = np.where(to_order, w.order_y, np.where(chase, ty, np.float32(np.nan)))
    return to_order | chase, gx, gy


def _chase_lock(sim: Sim) -> None:
    """`chase_lock`: auto-engaging (`ATTACK_MOVE` or idle) units whose target is out of range
    hold position while `game_loop - combat.last_fire_loop <= loops` for their unit type
    (clearing waypoint and dest). Does not lock `ATTACK_UNIT`. With `chase_lock_target`, only
    locks when the engaged target matches `combat.last_fire_target`.
    """
    if not sim.config.chase_lock:
        return
    w, buf = sim.world, sim.buf
    lock = np.full(w.unit_type.shape, -1, np.int64)
    for unit_type, loops in sim.config.chase_lock:
        lock[w.unit_type == unit_type] = loops
    since = w.game_loop[:, None] - sim.combat.last_fire_loop
    auto = (w.order_kind == CommandKind.ATTACK_MOVE) | (w.order_kind == CommandKind.NOOP)
    wait = w.alive & auto & (w.engaged_target_slot != NO_SLOT) & ~buf.in_range
    wait &= (sim.combat.last_fire_loop >= 0) & (since <= lock)
    if sim.config.chase_lock_target:
        wait &= w.engaged_target_slot == sim.combat.last_fire_target
    for a in (buf.waypoint_x, buf.waypoint_y, buf.dest_x, buf.dest_y):
        a[wait] = np.nan


def _follow_hold(sim: Sim) -> None:
    """`follow_lock`: clear waypoint and dest for units held in place by `buf.follow_hold`
    (written by `targeting._follow_lock`); unions with `_chase_lock` when both are enabled.
    """
    if not (sim.config.follow_lock or sim.config.slop_on_arrival):
        return
    buf = sim.buf
    wait = buf.follow_hold
    for a in (buf.waypoint_x, buf.waypoint_y, buf.dest_x, buf.dest_y):
        a[wait] = np.nan


def _windup_hold(sim: Sim) -> None:
    """`attack_commit`: clear waypoint and dest for units finishing a committed windup while
    their target stepped out of range (`buf.windup_hold`, set by attack).
    """
    if not sim.config.attack_commit:
        return
    buf = sim.buf
    wait = buf.windup_hold
    for a in (buf.waypoint_x, buf.waypoint_y, buf.dest_x, buf.dest_y):
        a[wait] = np.nan


def _kill_hold(sim: Sim) -> None:
    """`kill_hold`: hold position for `ATTACK_MOVE` and idle units waiting for a post-kill
    rescan (`combat.scan_loop` not yet reached) with no engaged target.
    """
    if not sim.config.kill_hold:
        return
    w, buf, c = sim.world, sim.buf, sim.combat
    auto = (w.order_kind == CommandKind.ATTACK_MOVE) | (w.order_kind == CommandKind.NOOP)
    pending = (c.scan_loop >= 0) & (w.game_loop[:, None] < c.scan_loop)
    wait = w.alive & auto & (w.engaged_target_slot == NO_SLOT) & pending
    for a in (buf.waypoint_x, buf.waypoint_y, buf.dest_x, buf.dest_y):
        a[wait] = np.nan


def _has_dest_all(sim: Sim) -> np.ndarray:
    kind = sim.world.order_kind
    return sim.world.alive & ((kind == CommandKind.MOVE) | (kind == CommandKind.ATTACK_MOVE))


def _has_dest(sim: Sim, b: int, u: int) -> bool:
    return int(sim.world.order_kind[b, u]) in (CommandKind.MOVE, CommandKind.ATTACK_MOVE)


def straight(sim: Sim) -> None:
    """Direct straight-line pathing ignoring terrain passability (`pathing='none'`)."""
    w, buf = sim.world, sim.buf
    has, gx, gy = goal_all(sim)
    has &= w.alive
    buf.waypoint_x[:] = np.where(has, gx, np.nan)
    buf.waypoint_y[:] = np.where(has, gy, np.nan)
    dest = _has_dest_all(sim)
    buf.dest_x[:] = np.where(dest, w.order_x, np.nan)
    buf.dest_y[:] = np.where(dest, w.order_y, np.nan)
    _chase_lock(sim)
    _follow_hold(sim)
    _windup_hold(sim)
    _kill_hold(sim)


def route(
    sim: Sim, b: int, x: float, y: float, gx: float, gy: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute `(waypoint, destination)` for a ground unit from `(x, y)` to `(gx, gy)`.

    Runs cached cell-level A* (`terrain.find_path`, keyed by `(map_id, start_cell, goal_cell)`).
    Unreachable goals project to the closest point inside the nearest reachable cell. Smooths
    waypoints via line-of-sight string pulling from the unit's current position.
    """
    info = sim.env_map[b]
    assert info is not None
    grid = info.pathing
    start = terrain.cell_of(grid, x, y)
    goal_cell = terrain.cell_of(grid, gx, gy)
    key = (info.map_id, start, goal_cell)
    path = sim.path_cache.get(key)
    if path is None:
        path = sim.path_cache[key] = terrain.find_path(grid, start, goal_cell)
    end = (gx, gy) if path[-1] == goal_cell else terrain.clamp_into_cell(gx, gy, path[-1])
    if terrain.line_of_sight(grid, x, y, *end):
        return end, end
    waypoint = terrain.cell_center(path[min(1, len(path) - 1)])
    for cell in path[2:-1]:
        c = terrain.cell_center(cell)
        if not terrain.line_of_sight(grid, x, y, *c):
            break
        waypoint = c
    return waypoint, end


def route_navmesh(
    sim: Sim, b: int, u: int, x: float, y: float, gx: float, gy: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """`pathing='navmesh'`: route on the continuous navigation mesh for this unit's pathing
    radius (cached per `(map_id, rho)`).
    """
    info = sim.env_map[b]
    assert info is not None
    mesh = _mesh(sim, info, navmesh.path_radius(float(sim.world.radius[b, u])))
    return navmesh.route(mesh, x, y, gx, gy)


def _mesh(sim: Sim, info, rho: float) -> navmesh.NavMesh:
    key = (info.map_id, rho)
    mesh = sim.nav_cache.get(key)
    if mesh is None:
        mesh = sim.nav_cache[key] = navmesh.build(info.pathing, rho)
    return mesh


def grid(sim: Sim) -> None:
    """Grid A* pathing (`pathing='grid'`): ground units follow smoothed A* over
    `MapInfo.pathing`, while flying units move in straight lines.

    Uses A* plus per-loop line-of-sight string pulling rather than Theta* so A* paths depend
    only on `(start_cell, goal_cell)` and remain cacheable when units are pushed off-path.
    MOVE/ATTACK_MOVE arrival checks the effective destination (nearest reachable cell when
    blocked).
    """
    _ground(sim, lambda b, u, x, y, gx, gy: route(sim, b, x, y, gx, gy))
    _chase_lock(sim)
    _follow_hold(sim)
    _windup_hold(sim)
    _kill_hold(sim)


def navmesh_pathing(sim: Sim) -> None:
    """Continuous navmesh pathing (`pathing='grid'` capability): shrunk/chamfered obstacles,
    radius-tiered pathing radius, any-angle shortest paths with right-hand tie-breaking, and
    ray-endpoint fallback for unreachable goals; air units move straight.
    """
    w, buf = sim.world, sim.buf
    has_g, gx, gy = goal_all(sim)
    alive = w.alive
    dest_all = _has_dest_all(sim)
    # Chasing ATTACK_MOVE: waypoint tracks target `g`, destination remains order point.
    chase = dest_all & has_g & (w.order_kind == CommandKind.ATTACK_MOVE)
    chase &= w.engaged_target_slot != NO_SLOT
    dest = dest_all & ~chase
    # Ground units: batch direct line-of-sight checks first; visible goals write directly,
    # while obstructed units fall through to batched `navmesh.route_many`.
    ground = alive & ~w.is_flying & (dest | has_g)
    tx = np.where(dest, w.order_x, gx)
    ty = np.where(dest, w.order_y, gy)
    direct = np.zeros_like(ground)
    maps: dict[str, tuple[object, list[int]]] = {}
    for b in np.flatnonzero(ground.any(axis=1)).tolist():
        info = sim.env_map[b]
        assert info is not None
        maps.setdefault(info.map_id, (info, []))[1].append(b)
    for info, bs in maps.values():
        sub = np.zeros_like(ground)
        sub[bs] = ground[bs]
        for r in np.unique(w.radius[sub]).tolist():
            sel = sub & (w.radius == r)
            mesh = _mesh(sim, info, navmesh.path_radius(float(r)))
            vis = navmesh.visible_many(
                mesh,
                w.x[sel].astype(np.float64),
                w.y[sel].astype(np.float64),
                tx[sel].astype(np.float64),
                ty[sel].astype(np.float64),
            )
            direct[sel] = vis
    for a in (buf.waypoint_x, buf.waypoint_y, buf.dest_x, buf.dest_y):
        a.fill(np.nan)
    fly = alive & w.is_flying
    for out, src, m in (
        (buf.waypoint_x, gx, (fly & has_g) | (direct & has_g)),
        (buf.waypoint_y, gy, (fly & has_g) | (direct & has_g)),
        (buf.dest_x, w.order_x, (fly | direct) & dest),
        (buf.dest_y, w.order_y, (fly | direct) & dest),
    ):
        out[m] = src[m]
    buf.dest_x[chase] = w.order_x[chase]
    buf.dest_y[chase] = w.order_y[chase]
    # Remaining obstructed ground units: group by navmesh for batched `route_many`
    # (bit-identical to per-unit `route_navmesh`).
    slow = ground & ~direct
    groups: dict[tuple[str, float], tuple[navmesh.NavMesh, list[tuple[int, int]]]] = {}
    for b, u in zip(*np.nonzero(slow), strict=True):
        b, u = int(b), int(u)
        info = sim.env_map[b]
        assert info is not None
        rho = navmesh.path_radius(float(w.radius[b, u]))
        groups.setdefault((info.map_id, rho), (_mesh(sim, info, rho), []))[1].append((b, u))
    for mesh, units in groups.values():
        bi = np.array([b for b, _ in units])
        ui = np.array([u for _, u in units])
        res = navmesh.route_many(
            mesh,
            w.x[bi, ui].astype(np.float64),
            w.y[bi, ui].astype(np.float64),
            tx[bi, ui].astype(np.float64),
            ty[bi, ui].astype(np.float64),
        )
        for (b, u), (wp, end) in zip(units, res, strict=True):
            if dest[b, u]:
                buf.dest_x[b, u], buf.dest_y[b, u] = f32(end[0]), f32(end[1])
                if not has_g[b, u]:
                    continue
            buf.waypoint_x[b, u], buf.waypoint_y[b, u] = f32(wp[0]), f32(wp[1])
    _chase_lock(sim)
    _follow_hold(sim)
    _windup_hold(sim)
    _kill_hold(sim)


def _ground(sim: Sim, router) -> None:
    w, buf = sim.world, sim.buf
    for a in (buf.waypoint_x, buf.waypoint_y, buf.dest_x, buf.dest_y):
        a.fill(np.nan)
    for b, u in zip(*np.nonzero(w.alive), strict=True):
        b, u = int(b), int(u)
        g = goal(sim, b, u)
        has_dest = _has_dest(sim, b, u)
        if has_dest and g is not None and int(w.engaged_target_slot[b, u]) != NO_SLOT:
            # Chasing ATTACK_MOVE: waypoint tracks target, destination remains order point.
            buf.dest_x[b, u], buf.dest_y[b, u] = w.order_x[b, u], w.order_y[b, u]
            if w.is_flying[b, u]:
                buf.waypoint_x[b, u], buf.waypoint_y[b, u] = g
                continue
            wp, _ = router(b, u, float(w.x[b, u]), float(w.y[b, u]), float(g[0]), float(g[1]))
            buf.waypoint_x[b, u], buf.waypoint_y[b, u] = f32(wp[0]), f32(wp[1])
            continue
        if w.is_flying[b, u]:
            if g is not None:
                buf.waypoint_x[b, u], buf.waypoint_y[b, u] = g
            if has_dest:
                buf.dest_x[b, u], buf.dest_y[b, u] = w.order_x[b, u], w.order_y[b, u]
            continue
        x, y = float(w.x[b, u]), float(w.y[b, u])
        if has_dest:
            ox, oy = float(w.order_x[b, u]), float(w.order_y[b, u])
            wp, end = router(b, u, x, y, ox, oy)
            buf.dest_x[b, u], buf.dest_y[b, u] = f32(end[0]), f32(end[1])
            if g is not None:  # MOVE or non-engaged ATTACK_MOVE: g is the order point
                buf.waypoint_x[b, u], buf.waypoint_y[b, u] = f32(wp[0]), f32(wp[1])
        elif g is not None:
            wp, _ = router(b, u, x, y, float(g[0]), float(g[1]))
            buf.waypoint_x[b, u], buf.waypoint_y[b, u] = f32(wp[0]), f32(wp[1])


IMPLS = {'straight': straight, 'grid': grid, 'navmesh': navmesh_pathing}
