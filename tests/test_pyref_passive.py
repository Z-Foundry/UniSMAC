"""Tests for pyref `passive_from_map`: derive `auto_acquire_owners` and `retreat_owners` on
reset from `assets/map_abilities.json` per-map passive behavior entries."""

from __future__ import annotations

import dataclasses

import pytest

from unismac.assets import load_map_info, load_passive_owners, load_static_data
from unismac.backends.pyref import PyrefBackend, PyrefConfig
from unismac.contract import Command, ContractError, Scenario, Spawn
from unismac.profiles import _V4

MARINE = 48
CORRIDOR, HARD_3M, SMAC_3M = 'new_maps/corridor', 'new_maps/3m', 'SMAC_Maps/3m'
STATIC = load_static_data()
MAPS = {m: load_map_info(m) for m in (CORRIDOR, HARD_3M, SMAC_3M)}
CFG = PyrefConfig(**{**_V4, 'commands': 'delayed', 'command_delay': 1, 'move_lag': 1})  # type: ignore[arg-type]
# Two Marines ~3.5 apart (within range 5) near the corridor spawn area
DUEL = (Spawn(1, MARINE, 19.0, 19.0), Spawn(2, MARINE, 21.5, 21.5))


def _reset(config: PyrefConfig, *map_ids: str) -> PyrefBackend:
    backend = PyrefBackend(STATIC, MAPS, len(map_ids), config)
    backend.reset([Scenario(map_id=m, spawns=DUEL) for m in map_ids])
    return backend


def _owners(backend: PyrefBackend) -> tuple[object, object]:
    sim = backend._sim  # noqa: SLF001 - inspect active config for test
    assert sim is not None
    return sim.config.auto_acquire_owners, sim.config.retreat_owners


def test_map_table() -> None:
    assert load_passive_owners(HARD_3M) == (1,)
    assert load_passive_owners(CORRIDOR) == ()
    assert load_passive_owners(SMAC_3M) is None


def test_default_off_keeps_profile() -> None:
    assert not PyrefConfig().passive_from_map
    assert _owners(_reset(CFG, CORRIDOR)) == ((2,), (1,))


@pytest.mark.parametrize(
    ('map_id', 'expected'),
    [(CORRIDOR, (None, ())), (HARD_3M, ((2,), (1,))), (SMAC_3M, ((2,), (1,)))],
)
def test_owners_from_map(map_id: str, expected: tuple[object, object]) -> None:
    on = dataclasses.replace(CFG, passive_from_map=True)
    assert _owners(_reset(on, map_id)) == expected
    # Missing entry (SMAC_3M) or identical owners (HARD_3M) keeps config equal to original
    if map_id != CORRIDOR:
        sim = _reset(on, map_id)._sim  # noqa: SLF001
        assert sim is not None and sim.config == on


def test_full_reset_switches_and_partial_must_match() -> None:
    on = dataclasses.replace(CFG, passive_from_map=True)
    backend = _reset(on, HARD_3M, HARD_3M)
    backend.reset([Scenario(map_id=CORRIDOR, spawns=DUEL)] * 2)
    assert _owners(backend) == (None, ())
    backend.reset([Scenario(map_id=CORRIDOR, spawns=DUEL)], env_ids=[1])
    with pytest.raises(ContractError, match='passive_from_map'):
        backend.reset([Scenario(map_id=HARD_3M, spawns=DUEL)], env_ids=[1])
    with pytest.raises(ContractError, match='passive_from_map'):
        _reset(on, CORRIDOR, HARD_3M)


def _p1_fires(config: PyrefConfig) -> bool:
    """Two idle Marines on corridor: whether player 1 fires in 40 loops (player 2 loses HP)."""
    backend = _reset(config, CORRIDOR)
    noop = Command.noop(1, len(DUEL))
    hp0 = float(backend.step(noop, 1).health[0, 1])
    for _ in range(40):
        s = backend.step(noop, 1)
    return float(s.health[0, 1]) < hp0


def test_corridor_player1_fights_back() -> None:
    assert not _p1_fires(CFG)
    assert _p1_fires(dataclasses.replace(CFG, passive_from_map=True))
