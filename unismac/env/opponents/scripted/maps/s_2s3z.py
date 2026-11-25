"""Scripts for `2s3z`: `2s3z` (attack nearest, retreat when low HP) and `2s3z_1` (Stalkers focus
enemy Stalkers, Zealots attack enemy nearest to ally center). Ported from upstream
`s_2s3z/script.py` and `s_2s3z/script_1.py`.

Differences from upstream: unused `enemy_zealots` / `enemy_stalkers` (in `2s3z`) and `MOVE_AMOUNT`
removed (side-effect-free). Preserved upstream behaviors:

* `2s3z` has no empty-list early return: when enemies are wiped out while allies survive,
  `nearest_n_units(...)[0]` raises `IndexError`;
* In `2s3z_1`, `enemy_zealots` filters by `STALKER` (same as `enemy_stalkers`), so Stalkers
  actually focus Stalkers first;
* The Zealot branch `distance_to > 0.1` is almost always True, making Zealot retreat unreachable;
* Trailing focus-center attacks in `2s3z` are appended after earlier commands (two commands/unit).
"""

from __future__ import annotations

from typing import Any

from ..api import UnitTypeId, attack, center, distance_to, move, nearest_n_units
from ..base import Script, register

SHOOT_RANGE = 6


@register('2s3z')
class ProtossRetreatNearest(Script):
    """Attack nearest enemy; retreat if close and low HP (Stalker < 50%, Zealot < 40%); focus
    enemy centroid once enemies are depleted."""

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        # Convert dicts to lists of alive units
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        self.actions_list: list[Any] = []

        stalkers = [unit for unit in agents if unit.unit_type == UnitTypeId.STALKER]
        zealots = [unit for unit in agents if unit.unit_type == UnitTypeId.ZEALOT]

        for stalker in stalkers:
            closest_enemy = nearest_n_units(stalker, enemies, 1)[0]

            if distance_to(stalker, closest_enemy) > SHOOT_RANGE:
                self.actions_list.append(attack(stalker, closest_enemy, visible_matrix))
            elif stalker.health / stalker.health_max < 0.5:
                self.actions_list.append(move(stalker, (23, 16)))
            else:
                self.actions_list.append(attack(stalker, closest_enemy, visible_matrix))

        for zealot in zealots:
            closest_enemy = nearest_n_units(zealot, enemies, 1)[0]

            if distance_to(zealot, closest_enemy) > 0.1:
                self.actions_list.append(attack(zealot, closest_enemy, visible_matrix))
            elif zealot.health / zealot.health_max < 0.4:
                self.actions_list.append(move(zealot, (23, 16)))
            else:
                self.actions_list.append(attack(zealot, closest_enemy, visible_matrix))

        if len(enemies) < 4 or len([eu for eu in enemies if eu.health / eu.health_max < 0.2]) > 2:
            for unit in agents:
                self.actions_list.append(attack(unit, center(enemies), visible_matrix))

        return self.actions_list


@register('2s3z_1')
class ProtossFocusStalker(Script):
    """Stalkers focus enemy Stalker nearest to ally centroid; Zealots attack enemy nearest to ally
    centroid; retreat when HP < 50%."""

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        # Convert dicts to lists of alive units
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        self.actions_list: list[Any] = []

        stalkers = [unit for unit in agents if unit.unit_type == UnitTypeId.STALKER]
        zealots = [unit for unit in agents if unit.unit_type == UnitTypeId.ZEALOT]

        # Upstream filters enemy_zealots by STALKER as well (preserved verbatim)
        enemy_zealots = sorted(
            [unit for unit in enemies if unit.unit_type == UnitTypeId.STALKER], key=lambda u: u.tag
        )
        enemy_stalkers = sorted(
            [unit for unit in enemies if unit.unit_type == UnitTypeId.STALKER], key=lambda u: u.tag
        )

        if not agents or not enemies:
            return []

        # Stalkers prioritize enemy_zealots (actually Stalkers due to upstream filter above)
        if enemy_zealots:
            target: Any = nearest_n_units(center(agents), enemy_zealots, 1)[0]
        elif enemy_stalkers:
            target = nearest_n_units(center(agents), enemy_stalkers, 1)[0]
        else:
            target = center(enemies)

        for stalker in stalkers:
            if stalker.health / stalker.health_max < 0.5:
                # Retreat toward spawn
                self.actions_list.append(move(stalker, (23, 16)))
            else:
                self.actions_list.append(attack(stalker, target, visible_matrix))

        target = nearest_n_units(center(agents), enemies, 1)[0]
        for zealot in zealots:
            if zealot.health / zealot.health_max < 0.5:
                self.actions_list.append(move(zealot, (23, 16)))
            else:
                self.actions_list.append(attack(zealot, target, visible_matrix))

        return self.actions_list
