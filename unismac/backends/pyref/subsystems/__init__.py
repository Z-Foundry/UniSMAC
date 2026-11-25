"""Pipeline subsystems. Each module exports `IMPLS: dict[str, Subsystem]` keyed by config name.

All subsystem functions share signature `(sim: Sim) -> None`, read/write `Sim` arrays only,
and process units in ascending `(b, u)` order.
"""

from __future__ import annotations

from . import (
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

__all__ = [
    'abilities',
    'attack',
    'collision',
    'commands',
    'damage',
    'kinematics',
    'native',
    'native_combat',
    'pathing',
    'steering',
    'targeting',
    'visibility',
]
