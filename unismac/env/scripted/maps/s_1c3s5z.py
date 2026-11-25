"""Scripts for `1c3s5z`: `1c3s5z_1` (Stalkers switch between kiting and holding by count/step,
Colossi attack nearest) and `1c3s5z_m` (regroup then attack nearest, Colossi focus densest group).
Ported from upstream `s_1c3s5z/script_d.py` and `s_1c3s5z/script_m.py`.

Differences from upstream: unused `MOVE_AMOUNT` removed; `best_target == None` written as
`is None` (same result). Preserved upstream behaviors:

* `script_m` issues regroup move commands when spread out, then still appends attack commands for
  every unit in the same step (two commands per unit);
* `script_m.find_best_attack_target` always returns the first highest-density enemy when non-empty,
  making the nearest-enemy fallback unreachable;
* `script_d`'s third branch (`len(stalkers) < 3` and `iteration >= 100`) only retreats in-range
  Stalkers when `iteration % 50 == 0` and issues no Stalker commands on other steps.
"""

from __future__ import annotations

from typing import Any

from ..api import UnitTypeId, attack, center, closer_than, distance_to, move, nearest_n_units
from ..base import Script, register

SHOOT_RANGE = 7


@register('1c3s5z_1')
class ColossusStalkerHold(Script):
    """At full Stalker count, retreat in-range Stalkers below 50% HP; when depleted, kite before
    step 100 then chase weakened enemies or retreat periodically; retreat low-HP in-range Zealots;
    Colossi attack nearest enemy."""

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.actions_list: list[Any] = []

        # Convert dicts to lists of alive units
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []

        stalkers = [unit for unit in agents if unit.unit_type == UnitTypeId.STALKER]
        zealots = [unit for unit in agents if unit.unit_type == UnitTypeId.ZEALOT]
        colossus = [unit for unit in agents if unit.unit_type == UnitTypeId.COLOSSUS]

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
                elif stalker.health / stalker.health_max < 0.5:
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
            closest_enemy = nearest_n_units(zealot, enemies, 1)[0]
            if distance_to(zealot, closest_enemy) > SHOOT_RANGE:
                self.actions_list.append(attack(zealot, closest_enemy, visible_matrix))
            elif zealot.health / zealot.health_max < 0.5:
                self.actions_list.append(move(zealot, (23.0, 16.0)))
            else:
                self.actions_list.append(attack(zealot, closest_enemy, visible_matrix))

        for c in colossus:
            closest_enemy = nearest_n_units(c, enemies, 1)[0]
            self.actions_list.append(attack(c, closest_enemy, visible_matrix))

        return self.actions_list


@register('1c3s5z_m')
class ColossusDensityFocus(Script):
    """Regroup if any unit is > 12 from army center; Zealots and Stalkers attack nearest enemy;
    Colossi focus the enemy with the most enemies within radius 3."""

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.actions_list: list[Any] = []

        self.units = [agent for _, agent in agents.items() if agent.health != 0]
        self.enemy_units = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        stalkers = [unit for unit in self.units if unit.unit_type == UnitTypeId.STALKER]
        zealots = [unit for unit in self.units if unit.unit_type == UnitTypeId.ZEALOT]
        colossi = [unit for unit in self.units if unit.unit_type == UnitTypeId.COLOSSUS]

        if not stalkers and not zealots and not colossi:
            return []

        if not self.enemy_units:
            return []

        army_center = center(stalkers + zealots + colossi)

        if self.units_too_spread(army_center):
            for unit in self.units:
                self.actions_list.append(move(unit, army_center))

        for zealot in zealots:
            closest_enemy = min(self.enemy_units, key=lambda e: distance_to(e, zealot))
            self.actions_list.append(attack(zealot, closest_enemy, visible_matrix))

        for stalker in stalkers:
            closest_enemy = min(self.enemy_units, key=lambda e: distance_to(e, stalker))
            self.actions_list.append(attack(stalker, closest_enemy, visible_matrix))

        for colossus in colossi:
            if self.enemy_units:
                best_target = self.find_best_attack_target(self.enemy_units)
                if best_target is None:
                    best_target = closest_enemy = min(
                        self.enemy_units, key=lambda e: distance_to(e, colossus)
                    )
                self.actions_list.append(attack(colossus, best_target, visible_matrix))

        return self.actions_list

    def units_too_spread(self, army_center: tuple[float, float]) -> bool:
        return any(distance_to(unit, army_center) > 12 for unit in self.units)

    def find_best_attack_target(self, enemies: list[Any]) -> Any:
        best_target = None
        highest_density = -1

        for enemy in enemies:
            nearby_enemies = closer_than(enemy, enemies, 3)
            if len(nearby_enemies) > highest_density:
                highest_density = len(nearby_enemies)
                best_target = enemy

        return best_target
