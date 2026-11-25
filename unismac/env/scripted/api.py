"""Helper utilities for SMAC-Hard scripted opponents, ported from `smac_hard/env/scripts/utils`.

Correspondence with upstream:

* Units: upstream passes `raw_pb2.Unit` protos; here `ScriptUnit` carries the fields accessed by
  scripts (`tag`, `unit_type`, `owner`, `pos.x` / `pos.y`, `health` / `health_max`,
  `shield` / `shield_max`, `energy`, `weapon_cooldown`), stored as Python floats widened from
  float32 to match proto values.
* Tags: uses `slot + 1` as a surrogate tag. Under `slot_order='tag'`, same-owner slots are sorted
  by ascending tag, so within-side tag comparisons and sorts match upstream.
* Actions: returns `RawAction(unit_tag, ability_id, target_tag, target_pos)` mirroring
  `ActionRawUnitCommand` (`queue_command` is always False); coordinates cast to float32 when
  converting to contract `Command` (matching proto `Point2D`).
* `agent_ability`: list of `AbilityEntry(unit_tag, abilities)` where each item exposes
  `ability_id`, matching `ResponseQueryAvailableAbilities`.
* `visible_matrix`: `{'blue': [...], 'red': [...]}` where `blue` lists alive Player 1 tags
  visible to Player 2 and `red` lists alive Player 2 tags visible to Player 1.
* `BASE_UNIT_TYPE` / `MAP_UNITS_TYPES` / `init_unit` from upstream `units_api` are only used by
  scripts outside `SCRIPT_DICT` (`2m_vs_1z`, `2s_vs_1sc`, `so_many_baneling`, `bane_vs_bane`);
  `new_maps` uses standard units on both sides, so those unused scripts and helpers are omitted.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

__all__ = [
    'ACTIONS',
    'AbilityEntry',
    'AbilityItem',
    'Point',
    'RawAction',
    'ScriptUnit',
    'UnitTypeId',
    'apply_ability',
    'attack',
    'center',
    'closer_than',
    'distance_to',
    'find_by_tag',
    'get_direction',
    'heal',
    'in_map_bounds',
    'move',
    'move_point',
    'nearest_n_units',
    'stop',
    'toward',
]


class UnitTypeId:
    """Subset of upstream `unit_typeid.UnitTypeId` used by scripts (plain int native IDs)."""

    COLOSSUS = 4
    BANELING = 9
    SIEGETANKSIEGED = 32
    SIEGETANK = 33
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
    VOIDRAY = 80
    IMMORTAL = 83
    SPINECRAWLER = 98
    SPORECRAWLER = 99
    ZERGLING = 105
    HYDRALISK = 107
    QUEEN = 126
    HELLIONTANK = 484


ACTIONS = {'move': 16, 'attack': 23, 'stop': 4, 'heal': 386}


class Point:
    __slots__ = ('x', 'y')

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f'Point({self.x}, {self.y})'


class ScriptUnit:
    """Unit snapshot exposed to scripts (subset of upstream `raw_pb2.Unit`)."""

    __slots__ = (
        'energy',
        'health',
        'health_max',
        'owner',
        'pos',
        'shield',
        'shield_max',
        'tag',
        'unit_type',
        'weapon_cooldown',
    )

    def __init__(
        self,
        tag: int,
        unit_type: int,
        owner: int,
        x: float,
        y: float,
        health: float,
        health_max: float,
        shield: float,
        shield_max: float,
        energy: float,
        weapon_cooldown: float,
    ) -> None:
        self.tag = tag
        self.unit_type = unit_type
        self.owner = owner
        self.pos = Point(x, y)
        self.health = health
        self.health_max = health_max
        self.shield = shield
        self.shield_max = shield_max
        self.energy = energy
        self.weapon_cooldown = weapon_cooldown

    def __repr__(self) -> str:
        return (
            f'ScriptUnit(tag={self.tag}, type={self.unit_type}, owner={self.owner}, '
            f'pos=({self.pos.x}, {self.pos.y}), health={self.health})'
        )


class AbilityItem(NamedTuple):
    ability_id: int


class AbilityEntry(NamedTuple):
    unit_tag: int
    abilities: tuple[AbilityItem, ...]


class RawAction(NamedTuple):
    """Single `ActionRawUnitCommand`; at most one of `target_tag` / `target_pos` is non-None."""

    unit_tag: int
    ability_id: int
    target_tag: int | None = None
    target_pos: tuple[float, float] | None = None


# ------------------------------------------------------------ actions_api


def heal(unit: Any, target: Any) -> RawAction:
    if type(target) is tuple:
        x, y = target
        return RawAction(unit.tag, ACTIONS['heal'], target_pos=(x, y))
    return RawAction(unit.tag, ACTIONS['heal'], target_tag=target.tag)


def attack(unit: Any, target: Any, visible_matrix: dict[str, list[int]]) -> RawAction | None:
    if unit is None or target is None:
        return None
    if type(target) is tuple:
        x, y = target
        return RawAction(unit.tag, ACTIONS['attack'], target_pos=(x, y))
    if target.tag in visible_matrix['blue']:
        return RawAction(unit.tag, ACTIONS['attack'], target_tag=target.tag)
    return RawAction(unit.tag, ACTIONS['move'], target_pos=(target.pos.x, target.pos.y))


def move(unit: Any, target: Any) -> RawAction:
    if type(target) is tuple:
        x, y = target
        return RawAction(unit.tag, ACTIONS['move'], target_pos=(x, y))
    return RawAction(unit.tag, ACTIONS['move'], target_pos=(target.pos.x, target.pos.y))


def move_point(unit: Any, x: float, y: float) -> RawAction:
    return RawAction(unit.tag, ACTIONS['move'], target_pos=(x, y))


def stop(unit: Any) -> RawAction:
    """Direct stop command constructed by upstream scripts (ability 4, no target)."""
    return RawAction(unit.tag, ACTIONS['stop'])


def apply_ability(unit: Any, ability: int, target: Any) -> RawAction:
    if type(target) is tuple:
        x, y = target
        return RawAction(unit.tag, ability, target_pos=(x, y))
    if target is None:
        return RawAction(unit.tag, ability)
    return RawAction(unit.tag, ability, target_tag=target.tag)


# ------------------------------------------------------------ distance_api


def distance_to(unit_from: Any, unit_to: Any) -> float:
    if type(unit_from) is tuple:
        x1, y1 = unit_from[0], unit_from[1]
    else:
        x1, y1 = unit_from.pos.x, unit_from.pos.y
    if type(unit_to) is tuple:
        x2, y2 = unit_to[0], unit_to[1]
    else:
        x2, y2 = unit_to.pos.x, unit_to.pos.y
    delta_x = x1 - x2
    delta_y = y1 - y2
    return math.sqrt(delta_x**2 + delta_y**2)


def nearest_n_units(target: Any, candidates: list[Any], num: int) -> list[Any]:
    if len(candidates) <= num:
        return candidates
    return sorted(candidates, key=lambda e: distance_to(e, target))[:num]


def toward(us: Any, ut: Any, d: float) -> tuple[float, float]:
    if isinstance(us, tuple):
        xs, ys = us
    else:
        xs, ys = us.pos.x, us.pos.y
    if isinstance(ut, tuple):
        xt, yt = ut
    else:
        xt, yt = ut.pos.x, ut.pos.y
    dis = distance_to((xs, ys), (xt, yt))
    return (xs + (xt - xs) / dis * d, ys + (yt - ys) / dis * d)


def in_map_bounds(pos: Any, map_state: Any) -> bool:
    return pos[0] < map_state.x and pos[1] < map_state.y


def get_direction(p1: Any, p2: Any) -> list[float]:
    diff = [p2[0] - p1[0], p2[1] - p1[1]]
    length = math.hypot(diff[0], diff[1])
    return [diff[0] / length, diff[1] / length]


def closer_than(target: Any, candidate: list[Any], dist: float) -> list[Any]:
    return [c for c in candidate if distance_to(c, target) <= dist]


# ------------------------------------------------------------ units_api


def find_by_tag(units: Any, tag: Any) -> Any:
    if type(units) is list or type(units) is tuple:
        for u in units:
            if u.tag == tag:
                return u
    elif type(units) is dict:
        for _, u in units.items():
            if u.tag == tag:
                return u
    return None


def center(units: list[Any]) -> tuple[float, float]:
    x = [u.pos.x for u in units]
    y = [u.pos.y for u in units]
    return (sum(x) / len(x), sum(y) / len(y))
