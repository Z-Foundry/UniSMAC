"""Visibility and 3D projectile tests for `pyref`: `visibility='sc2'` and `projectile_3d`.

Map `new_maps/mmmt`: central high ground x in [10, 23), y in [11, 21) (byte 207), surrounding
low ground (191), and east/west ramps.
- True sight follows height levels: ground units see same/lower levels; air units see over.
- Attacking reveals the attacker to the victim's owner for `reveal_loops` (47) from last hit.
- Targeting uses true sight: ATTACK_UNIT on an unseen target degrades to ATTACK_MOVE until
  true sight is gained, then restores ATTACK_UNIT.
- `projectile_3d`: projectiles travel along 3D distance.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from unismac.assets import load_map_info, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.backends.pyref.backend import capabilities_for
from unismac.backends.pyref.subsystems.visibility import levels
from unismac.contract import Command, CommandKind, Scenario, Spawn

MARINE, MEDIVAC, ZEALOT, STALKER = 48, 54, 73, 74
MMMT, FLAT = 'new_maps/mmmt', 'SMAC_Maps/32x32_flat'
STATIC = load_static_data()
MAPS = {m: load_map_info(m) for m in (MMMT, FLAT)}
P7 = PyrefConfig(attack='sc2', damage='sc2', targeting='sc2', visibility='sc2', pathing='navmesh')
LOW, HIGH = (16.5, 7.5), (16.5, 11.5)  # south low ground & high edge, dist 4 (< range 5)


def make(cfg: PyrefConfig = P7) -> PyrefBackend:
    return PyrefBackend(STATIC, MAPS, 1, cfg)


def scenario(*spawns: Spawn, map_id: str = MMMT) -> Scenario:
    return Scenario(map_id=map_id, spawns=spawns)


def run(backend: PyrefBackend, first: Command, loops: int, then: dict | None = None) -> list:
    """Issue `first` on loop 0 then noops (or `then[k]` on loop k); index k = loop k."""
    n = first.kind.shape[1]
    then = then or {}
    out = [backend.step(first, 1)]
    out += [backend.step(then.get(k, Command.noop(1, n)), 1) for k in range(1, loops)]
    return out


def hp(states: list, slot: int) -> list[float]:
    return [float(s.health[0, slot] + s.shield[0, slot]) for s in states]


def hit_loops(states: list, slot: int) -> list[int]:
    h = hp(states, slot)
    return [k for k in range(1, len(h)) if h[k] < h[k - 1]]


def test_levels() -> None:
    """West ramp y=12: bytes 191/195 are low ground (level 0), 199/203/207 high (level 1)."""
    info = MAPS[MMMT]
    row = info.terrain_height[12, 5:11].astype(int).tolist()
    assert row == [191, 191, 195, 199, 203, 207]
    assert levels(info)[12, 5:11].tolist() == [0, 0, 0, 1, 1, 1]
    assert not levels(MAPS[FLAT]).any()


@pytest.mark.parametrize('visibility', ['full', 'sc2'])
def test_low_ground_cannot_see_high_ground(visibility: str) -> None:
    cfg = (
        P7 if visibility == 'sc2' else dataclasses.replace(P7, visibility='full', targeting='basic')
    )
    backend = make(cfg)
    s = backend.reset([scenario(Spawn(1, MARINE, *LOW), Spawn(2, MARINE, *HIGH))])
    assert bool(s.visible[0, 0, 1]), 'high ground sees low ground'
    assert bool(s.visible[0, 1, 0]) == (visibility == 'full'), 'low ground cannot see high ground'
    assert bool(s.visible[0, 0, 0]) and bool(s.visible[0, 1, 1]), 'own units are always visible'


def test_air_unit_gives_high_ground_sight() -> None:
    backend = make()
    s = backend.reset(
        [
            scenario(
                Spawn(1, MARINE, *LOW), Spawn(2, MARINE, 16.5, 14.5), Spawn(1, MEDIVAC, 16.5, 9.5)
            )
        ]
    )
    assert bool(s.visible[0, 1, 0])
    assert bool(s.visible[0, 2, 1]), 'air unit is visible to lower-level owner'


def test_reveal_on_hit_lasts_47_loops() -> None:
    """High-ground Marine fires on HOLD low-ground Marine: attacker is revealed to player 1 from
    the first hit until 47 loops after the last hit; bystanders stay hidden; HOLD does not return
    fire without true sight."""
    backend = make()
    backend.reset(
        [scenario(Spawn(1, MARINE, *LOW), Spawn(2, MARINE, *HIGH), Spawn(2, ZEALOT, 19.5, 12.5))]
    )
    cmd = (
        Command.noop(1, 3)
        .set(0, 0, CommandKind.HOLD)
        .set(0, 1, CommandKind.ATTACK_UNIT, target_slot=0)
    )
    away = Command.noop(1, 3).set(0, 1, CommandKind.MOVE, target_xy=(16.5, 18.5))
    states = run(backend, cmd, 140, {40: away})
    hits = hit_loops(states, 0)
    assert hits and hits[-1] < 40
    seen = [bool(s.visible[0, 1, 0]) for s in states]
    assert not any(seen[: hits[0]])
    last = hits[-1]
    assert all(seen[hits[0] : last + 47]) and not any(seen[last + 47 :])
    assert not any(bool(s.visible[0, 2, 0]) for s in states), 'nearby Zealot is not revealed'
    assert hit_loops(states, 1) == [], 'HOLD does not fire on revealed-only enemy'


def test_attack_unit_degrades_until_true_sight() -> None:
    """Low-ground Marine issuing ATTACK_UNIT on unseen high-ground enemy degrades to ATTACK_MOVE
    toward target coordinates, climbs the ramp, then restores ATTACK_UNIT and fires upon sight."""
    backend = make()
    backend.reset([scenario(Spawn(1, MARINE, *LOW), Spawn(2, MARINE, *HIGH))])
    cmd = Command.noop(1, 2).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1)
    states = run(backend, cmd, 260)
    kinds = [int(s.order_kind[0, 0]) for s in states]
    assert kinds[0] == CommandKind.ATTACK_MOVE
    assert (float(states[0].order_x[0, 0]), float(states[0].order_y[0, 0])) == HIGH
    back = kinds.index(CommandKind.ATTACK_UNIT)
    assert int(states[back].order_target_slot[0, 0]) == 1
    assert bool(states[back - 1].visible[0, 1, 0]), 'true sight gained by end of previous loop'
    hits = hit_loops(states, 1)
    assert hits and hits[0] > back


def test_projectile_3d() -> None:
    """High-ground Stalker vs low-ground Zealot: 3D distance adds 1 loop; flat unchanged."""
    base = PyrefConfig(attack='sc2', damage='sc2')

    def first_hit(cfg: PyrefConfig, map_id: str) -> int:
        backend = make(cfg)
        backend.reset([scenario(Spawn(1, STALKER, *HIGH), Spawn(2, ZEALOT, *LOW), map_id=map_id)])
        cmd = Command.noop(1, 2).set(0, 0, CommandKind.ATTACK_UNIT, target_slot=1)
        return hit_loops(run(backend, cmd.set(0, 1, CommandKind.HOLD), 40), 1)[0]

    d3 = dataclasses.replace(base, projectile_3d=True)
    assert first_hit(d3, MMMT) == first_hit(base, MMMT) + 1
    assert first_hit(d3, FLAT) == first_hit(base, FLAT)


def test_capabilities_and_validation() -> None:
    caps = capabilities_for(P7)
    assert caps.fog and caps.terrain_height
    plain = capabilities_for(PyrefConfig())
    assert not plain.fog and not plain.terrain_height
    assert capabilities_for(PyrefConfig(projectile_3d=True)).terrain_height
    with pytest.raises(ValueError, match="targeting='sc2'"):
        PyrefConfig(visibility='sc2')
    with pytest.raises(ValueError, match='reveal_loops'):
        PyrefConfig(reveal_loops=-1)
    assert np.isfinite(P7.reveal_loops)
