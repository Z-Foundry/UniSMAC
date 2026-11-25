"""Fixed per-game-loop subsystem pipeline and implementation registry for `pyref`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias

from .config import PyrefConfig
from .sim import Sim
from .subsystems import (
    abilities,
    attack,
    collision,
    commands,
    damage,
    kinematics,
    native,
    native_combat,
    pathing,
    steering,
    targeting,
    visibility,
)

Subsystem: TypeAlias = Callable[[Sim], None]

# Pipeline stage order; stage names match the corresponding fields on PyrefConfig.
# Register new implementations in each subsystem module's IMPLS mapping.
STAGES: tuple[str, ...] = (
    'commands',
    'targeting',
    'attack',
    'abilities',
    'pathing',
    'steering',
    'kinematics',
    'collision',
    'damage',
    'visibility',
)

REGISTRY: Mapping[str, Mapping[str, Subsystem]] = {
    'commands': commands.IMPLS,
    'targeting': targeting.IMPLS,
    'attack': attack.IMPLS,
    'abilities': abilities.IMPLS,
    'pathing': pathing.IMPLS,
    'steering': {**steering.IMPLS, 'native': native.steer},
    'kinematics': kinematics.IMPLS,
    'collision': {**collision.IMPLS, 'native': native.contact},
    'damage': damage.IMPLS,
    'visibility': visibility.IMPLS,
}


def resolve(config: PyrefConfig) -> tuple[Subsystem, ...]:
    """Resolve subsystem implementations from `config`; raise ValueError on unknown names."""
    chosen = []
    for stage in STAGES:
        name = getattr(config, stage)
        try:
            chosen.append(REGISTRY[stage][name])
        except KeyError:
            known = ', '.join(sorted(REGISTRY[stage]))
            raise ValueError(
                f'Unknown implementation {name!r} for stage {stage!r} (choices: {known})'
            ) from None
    return tuple(chosen)


def run_loop(sim: Sim, stages: tuple[Subsystem, ...]) -> None:
    """Advance the whole batch by one game loop.

    Runs `targeting.kill_scan` for same-loop dead targets when `kill_rescan_delay=0` (or
    `native_backswing=True`), runs `targeting.late_scan` when `scan_after_move` is enabled,
    and performs the 50% head-insert shuffle on `unit_order` when `native_loop_order` is set.
    """
    for stage in stages:
        stage(sim)
    if sim.config.kill_rescan_delay == 0 or sim.config.native_backswing:
        targeting.kill_scan(sim)
    if sim.config.scan_after_move:
        targeting.late_scan(sim)
    if sim.config.native_loop_order:
        native_combat.shuffle_unit_order(sim)
    sim.world.game_loop += 1
