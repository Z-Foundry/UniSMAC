"""Tests for pyref `pin_sidestep_lat_scope`: scoping the `pin_sidestep_lat` lateral filter.

With `'marauder'`, the lateral filter applies only to pairs where unit u or blocker t is a
Marauder; default `'all'` applies it to every pair.
* Option validation: must be `'all'` or `'marauder'`, and `'marauder'` requires `pin_sidestep_lat`.
* On random crowded states with Marauders mixed in, batch `steering._sidestep` matches the per-unit
  reference (`test_pyref_pin_sidestep_lat._ref`) bitwise; all-Marine matches no `pin_sidestep_lat`
  bitwise, and all-Marauder matches `'all'` bitwise.
* Behavior: with an obliquely offset (lateral 0.5) HOLD blocker, any pair with a Marauder matches
  `'all'` (advances immediately), while Marine-vs-Marine matches `pin_sidestep` alone (sidesteps).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.sim import Sim
from unismac.backends.pyref.subsystems import steering
from unismac.contract import Command, CommandKind, Scenario, Spawn, WorldState

from .test_pyref_pin_sidestep import MAP_ID, MAPS, MARINE, STATIC, _path, _sim
from .test_pyref_pin_sidestep_lat import ON, _ref, _run

MARAUDER = steering.MARAUDER
LAT: dict[str, Any] = {**ON, 'pin_sidestep_lat': 0.42}
MRD: dict[str, Any] = {**LAT, 'pin_sidestep_lat_scope': 'marauder'}


def test_lat_scope_validated() -> None:
    assert PyrefConfig().pin_sidestep_lat_scope == 'all'
    assert PyrefConfig(**MRD).pin_sidestep_lat_scope == 'marauder'
    with pytest.raises(ValueError, match='pin_sidestep_lat_scope'):
        PyrefConfig(**LAT, pin_sidestep_lat_scope='marine')
    with pytest.raises(ValueError, match='pin_sidestep_lat_scope'):
        PyrefConfig(**ON, pin_sidestep_lat_scope='marauder')  # missing pin_sidestep_lat
    with pytest.raises(ValueError, match='pin_sidestep_lat_scope'):
        dataclasses.replace(PyrefConfig(**MRD), pin_sidestep_lat=None)
    assert PyrefConfig(**ON, pin_sidestep_lat_scope='all').pin_sidestep_lat is None


def _typed(sim: Sim, seed: int, frac: float) -> Sim:
    """Convert roughly frac of slots to Marauders (with radius; all when frac = 1)."""
    rng = np.random.default_rng(200 + seed)
    w = sim.world
    m = rng.random(w.unit_type.shape) < frac
    w.unit_type[m] = MARAUDER
    w.radius[m] = np.float32(0.5625)
    return sim


@pytest.mark.parametrize('drift', [False, True])
@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('lat', [0.2, 0.42])
def test_lat_scope_batch_matches_per_unit(seed: int, lat: float, drift: bool) -> None:
    kw: dict[str, Any] = {**ON, 'pin_sidestep_drift': drift, 'pin_sidestep_lat': lat}
    mk = lambda **c: _typed(_sim(6, seed, **c), seed, 0.35)  # noqa: E731
    bx, by = _run(mk(**kw, pin_sidestep_lat_scope='marauder'), seed, steering._sidestep)
    rx, ry = _run(mk(**kw, pin_sidestep_lat_scope='marauder'), seed, _ref)
    assert bx.tobytes() == rx.tobytes()
    assert by.tobytes() == ry.tobytes()
    # Marine pairs follow no-lat, Marauder pairs follow 'all': at 0.2 differs from both;
    # at 0.42 fewer pairs trigger the filter, so differs from at least one.
    ax, ay = _run(mk(**kw), seed, steering._sidestep)
    nx, ny = _run(mk(**ON, pin_sidestep_drift=drift), seed, steering._sidestep)
    d_all, d_none = ((ax != bx) | (ay != by)).sum(), ((nx != bx) | (ny != by)).sum()
    assert d_all + d_none >= 3
    if lat < 0.3:
        assert d_all >= 1 and d_none >= 1


@pytest.mark.parametrize('seed', range(4))
def test_lat_scope_no_marauder_equals_no_lat(seed: int) -> None:
    """27m_vs_30m has only Marines: 'marauder' matches no pin_sidestep_lat bitwise."""
    a = _run(_sim(6, seed, **MRD), seed, steering._sidestep)
    b = _run(_sim(6, seed, **ON), seed, steering._sidestep)
    c = _run(_sim(6, seed, **LAT), seed, steering._sidestep)
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
    assert (a[0] != c[0]).any() or (a[1] != c[1]).any()  # 'all' differs


@pytest.mark.parametrize('seed', range(4))
def test_lat_scope_all_marauder_equals_all(seed: int) -> None:
    a = _run(_typed(_sim(6, seed, **MRD), seed, 1.0), seed, steering._sidestep)
    b = _run(_typed(_sim(6, seed, **LAT), seed, 1.0), seed, steering._sidestep)
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()


# ---------------------------------------------------------------- Behavior


def _pair(dy: float, mover: int, holder: int, config: dict[str, Any]) -> list[WorldState]:
    """Player 1 holder (slot 0) HOLDs at (16, 16 + dy); mover (slot 1) starts behind it
    by (radius sum - contact_slack + 0.01) along x and MOVEs to (20, 16); a distant
    player 2 Marine idles. Returns 16 loops of states."""
    r = {MARINE: 0.375, MARAUDER: 0.5625}
    gap = r[mover] + r[holder] - float(config['contact_slack']) + 0.01
    spawns = (Spawn(1, holder, 16.0, 16.0 + dy), Spawn(1, mover, 16.0 - gap, 16.0),
              Spawn(2, MARINE, 28.0, 28.0))  # fmt: skip
    be = PyrefBackend(STATIC, MAPS, 1, PyrefConfig(**config))
    be.reset([Scenario(MAP_ID, spawns=spawns)], seeds=[0])
    assert be._sim is not None
    assert abs(float(be._sim.world.radius[0, 1]) - r[mover]) < 1e-6
    cmd = Command.noop(1, be._sim.n_slots)
    cmd.set(0, 0, CommandKind.HOLD)
    cmd.set(0, 1, CommandKind.MOVE, target_xy=(20.0, 16.0))
    out = [be.step(cmd, 1)]
    for _ in range(15):
        out.append(be.step(Command.noop(1, be._sim.n_slots), 1))
    be.close()
    return out


def _same(a: list[WorldState], b: list[WorldState]) -> bool:
    return all(
        getattr(x, n).tobytes() == getattr(y, n).tobytes()
        for x, y in zip(a, b, strict=True)
        for n in WorldState.UNIT_FIELDS
    )


@pytest.mark.parametrize(
    ('mover', 'holder'), [(MARAUDER, MARINE), (MARINE, MARAUDER), (MARAUDER, MARAUDER)]
)
def test_lat_scope_marauder_pair_like_all(mover: int, holder: int) -> None:
    mrd = _pair(0.5, mover, holder, MRD)
    assert _same(mrd, _pair(0.5, mover, holder, LAT))
    ps = _pair(0.5, mover, holder, ON)
    assert not _same(mrd, ps)
    a, b = _path(mrd, 1), _path(ps, 1)
    assert a[3, 0] - a[0, 0] > 0.2  # advances immediately
    assert abs(b[3, 0] - b[0, 0]) < 0.05  # pin_sidestep alone sidesteps without advancing
    assert abs(float(ps[-1].x[0, 0]) - 16.0) < 1e-6  # HOLD unit stays still


def test_lat_scope_marine_pair_like_no_lat() -> None:
    mrd = _pair(0.5, MARINE, MARINE, MRD)
    assert _same(mrd, _pair(0.5, MARINE, MARINE, ON))
    assert not _same(mrd, _pair(0.5, MARINE, MARINE, LAT))
