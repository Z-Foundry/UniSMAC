"""Scripts for `3s5z` and `3s5z_vs_3s6z`: `3s5z` (attack nearest, retreat when low HP) and
`3s5z_1` (Stalkers switch between kiting and holding by count/step). Ported from upstream
`s_3s5z/script.py` and `s_3s5z/script_1.py`.

Differences from upstream: unused `enemy_zealots` / `enemy_stalkers` and `MOVE_AMOUNT` removed
(side-effect-free). Preserved upstream behaviors:

* Neither script has an empty-list early return: when enemies are wiped out while allies survive,
  `nearest_n_units(...)[0]` raises `IndexError`;
* In `3s5z`, the Zealot branch `distance_to > 0.1` is almost always True, making retreat
  unreachable; trailing focus-center commands append after earlier commands (two commands/unit);
* In `3s5z_1`'s third branch (`len(stalkers) < 3` and `iteration >= 100`), only in-range Stalkers
  retreat when `iteration % 50 == 0` and no Stalker commands are issued on other steps.
"""

from __future__ import annotations

from typing import Any

from ..api import UnitTypeId, attack, center, distance_to, move, nearest_n_units
from ..base import Script, register

SHOOT_RANGE = 6


@register('3s5z')
class ProtossKiteNearest(Script):
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


@register('3s5z_1')
class ProtossStalkerHold(Script):
    """At full Stalker count, retreat in-range Stalkers below 70% HP; when depleted, kite before
    step 100 then chase weakened enemies or retreat periodically; retreat in-range Zealots below
    50% HP, else attack nearest enemy."""

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        # Convert dicts to lists of alive units
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        self.actions_list: list[Any] = []

        stalkers = [unit for unit in agents if unit.unit_type == UnitTypeId.STALKER]
        zealots = [unit for unit in agents if unit.unit_type == UnitTypeId.ZEALOT]

        if iteration < 100 and len(stalkers) < 3:
            for stalker in stalkers:
                closest_enemy = nearest_n_units(stalker, enemies, 1)[0]
                if distance_to(stalker, closest_enemy) > SHOOT_RANGE:
                    self.actions_list.append(attack(stalker, closest_enemy, visible_matrix))
                else:
                    self.actions_list.append(move(stalker, (23, 16)))
        elif len(stalkers) >= 3:
            for stalker in stalkers:
                closest_enemy = nearest_n_units(stalker, enemies, 1)[0]
                if distance_to(stalker, closest_enemy) > SHOOT_RANGE:
                    self.actions_list.append(attack(stalker, closest_enemy, visible_matrix))
                elif stalker.health / stalker.health_max < 0.7:
                    self.actions_list.append(move(stalker, (23.0, 16.0)))
                else:
                    self.actions_list.append(attack(stalker, closest_enemy, visible_matrix))

        else:
            if (
                len(enemies) < 4
                or len([enemy for enemy in enemies if enemy.health / enemy.health_max < 0.2]) > 2
            ):
                for stalker in stalkers:
                    closest_enemy = nearest_n_units(stalker, enemies, 1)[0]
                    if distance_to(stalker, closest_enemy) < SHOOT_RANGE:
                        self.actions_list.append(attack(stalker, closest_enemy, visible_matrix))
                    else:
                        enemies_center = center(enemies)
                        self.actions_list.append(move(stalker, enemies_center))
            elif iteration % 50 == 0:
                for stalker in stalkers:
                    closest_enemy = nearest_n_units(stalker, enemies, 1)[0]
                    if distance_to(stalker, closest_enemy) < SHOOT_RANGE:
                        self.actions_list.append(move(stalker, (23.0, 16.0)))

        for zealot in zealots:
            closest_enemy = closest_enemy = nearest_n_units(zealot, enemies, 1)[0]
            if distance_to(zealot, closest_enemy) > SHOOT_RANGE:
                self.actions_list.append(attack(zealot, closest_enemy, visible_matrix))
            elif zealot.health / zealot.health_max < 0.5:
                self.actions_list.append(move(zealot, (23.0, 16.0)))
            else:
                self.actions_list.append(attack(zealot, closest_enemy, visible_matrix))

        return self.actions_list
