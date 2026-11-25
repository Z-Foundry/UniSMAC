"""Scripts for `mmmt`: `mmmt_fap` (form line, advance, focus priority targets) and `mmmt_fan`
(form line, advance, attack nearest). Ported from upstream `s_MMMT/script_form_atkp.py` and
`script_form_atkn.py`.

Deviations and preserved upstream issues:

* Upstream uses bare `return` (None) when no alive units exist; returning `[]` is equivalent;
* `cur_health` is never updated (stays 1), so Marine low-HP retreat never triggers (preserved);
* `mmmt_fan` indexes `marauders[0..2]` for the first 10 steps, raising `IndexError` with < 3
  Marauders (preserved);
* Medivacs call `min(marines + marauders, ...)`, raising `ValueError` when empty (preserved).
"""

from __future__ import annotations

from typing import Any

from ..api import (
    UnitTypeId,
    apply_ability,
    attack,
    center,
    closer_than,
    distance_to,
    move,
    nearest_n_units,
)
from ..base import Script, register


@register('mmmt_fap')
class FormAttackPriority(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []  # Upstream bare return (None)

        self.marines = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.MARINE], key=lambda a: a.tag
        )
        self.marauders = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.MARAUDER], key=lambda a: a.tag
        )
        self.medivacs = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.MEDIVAC], key=lambda a: a.tag
        )
        self.tanks = sorted(
            [
                a
                for a in agents
                if a.unit_type == UnitTypeId.SIEGETANK or a.unit_type == UnitTypeId.SIEGETANKSIEGED
            ],
            key=lambda a: a.tag,
        )

        self.enemy_marines = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MARINE], key=lambda a: a.tag
        )
        self.enemy_marauders = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MARAUDER], key=lambda a: a.tag
        )
        self.enemy_medivacs = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MEDIVAC], key=lambda a: a.tag
        )
        self.enemy_tanks = sorted(
            [
                a
                for a in enemies
                if a.unit_type == UnitTypeId.SIEGETANK or a.unit_type == UnitTypeId.SIEGETANKSIEGED
            ],
            key=lambda a: a.tag,
        )

        actions_list: list[Any] = []

        if self.init:
            self.pre_health: dict[int, float] = {}
            self.cur_health: dict[int, float] = {}
            for m in self.marines:
                self.pre_health[m.tag] = 1
                self.cur_health[m.tag] = 1
            self.init = False

        if self.enemy_marines:
            target_units = self.enemy_marines
        elif self.enemy_marauders:
            target_units = self.enemy_marauders
        elif self.enemy_tanks:
            target_units = self.enemy_tanks
        else:
            target_units = enemies

        # Tactic 1: opening formation (Marauders front, Marines middle, Medivacs/Tanks back)
        if iteration <= 10:
            center_x, center_y = center(self.marines + self.marauders)
            for marauder in self.marauders:
                actions_list.append(move(marauder, (center_x - 3, marauder.pos.y)))
            for marine in self.marines:
                actions_list.append(move(marine, (center_x, marine.pos.y)))
            for tank in self.tanks:
                actions_list.append(move(tank, (center_x + 2, tank.pos.y)))
            for medivac in self.medivacs:
                actions_list.append(move(medivac, (center_x + 1, medivac.pos.y)))

            return actions_list

        for tank in self.tanks:
            seige_target = min(enemies, key=lambda e: distance_to(tank, e))
            if distance_to(tank, seige_target) < 10 and tank.unit_type == UnitTypeId.SIEGETANK:
                # 388: Siege Mode
                actions_list.append(apply_ability(tank, 388, None))
            elif (
                distance_to(tank, seige_target) > 13
                and tank.unit_type == UnitTypeId.SIEGETANKSIEGED
            ):
                # 390: Unsiege
                actions_list.append(apply_ability(tank, 390, None))

        # Advance in formation toward enemies
        if (
            any([len(closer_than(marauder, enemies, 8)) == 0 for marauder in self.marauders])
            and self.engage
        ):
            for agent in agents:
                actions_list.append(move(agent, (agent.pos.x - 1, agent.pos.y)))
            return actions_list

        self.engage = False

        # Prioritize Marines, then Marauders, then Tanks, then remaining enemies
        if target_units:
            target = min(
                nearest_n_units(center(agents), target_units, 3), key=lambda e: e.health + e.shield
            )
        else:
            target = self.random.choice(enemies)
        for m in self.marines:
            if self.pre_health[m.tag] > self.cur_health[m.tag] and m.health / m.health_max < 0.3:
                actions_list.append(move(m, (m.pos.x + 1, m.pos.y)))
            else:
                if target:
                    actions_list.append(attack(m, target, visible_matrix))

        # Front line
        if self.marines:
            front_line = (center(self.marines)[0] - 2, center(self.marines)[1])
        else:
            front_line = (center(agents)[0] - 2, center(agents)[1])

        for m in self.marauders:
            if m.pos.x < front_line[0]:
                if target_units:
                    if target:
                        actions_list.append(attack(m, target, visible_matrix))
            else:
                actions_list.append(move(m, (front_line[0], m.pos.y)))

        for medivac in self.medivacs:
            all_units = self.marines + self.marauders
            most_injured_unit = min(all_units, key=lambda u: u.health / u.health_max)
            if most_injured_unit.health != most_injured_unit.health_max:
                actions_list.append(attack(medivac, most_injured_unit, visible_matrix))
            else:
                actions_list.append(move(medivac, (center(agents)[0] + 1, center(agents)[1])))
        for tank in self.tanks:
            actions_list.append(attack(tank, target, visible_matrix))

        for m in self.marines:
            self.pre_health[m.tag] = self.cur_health[m.tag]

        return [a for a in actions_list if a is not None]


@register('mmmt_fan')
class FormAttackNearest(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True
        self.center: tuple[float, float] | None = None

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []  # Upstream bare return (None)

        self.marines = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.MARINE], key=lambda a: a.tag
        )
        self.marauders = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.MARAUDER], key=lambda a: a.tag
        )
        self.medivacs = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.MEDIVAC], key=lambda a: a.tag
        )
        self.tanks = sorted(
            [
                a
                for a in agents
                if a.unit_type == UnitTypeId.SIEGETANK or a.unit_type == UnitTypeId.SIEGETANKSIEGED
            ],
            key=lambda a: a.tag,
        )

        self.enemy_marines = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MARINE], key=lambda a: a.tag
        )
        self.enemy_marauders = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MARAUDER], key=lambda a: a.tag
        )
        self.enemy_medivacs = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MEDIVAC], key=lambda a: a.tag
        )
        self.enemy_tanks = sorted(
            [
                a
                for a in enemies
                if a.unit_type == UnitTypeId.SIEGETANK or a.unit_type == UnitTypeId.SIEGETANKSIEGED
            ],
            key=lambda a: a.tag,
        )

        actions_list: list[Any] = []

        if self.init:
            self.pre_health: dict[int, float] = {}
            self.cur_health: dict[int, float] = {}
            for m in self.marines:
                self.pre_health[m.tag] = 1
                self.cur_health[m.tag] = 1
            self.init = False

        # Tactic 1: opening formation (Marauders front, Marines in column, Medivacs/Tanks back)
        if not self.center:
            self.center = center(self.marines + self.marauders)

        center_x, center_y = self.center
        if iteration <= 10:
            actions_list.append(move(self.marauders[0], (center_x - 2, center_y - 1)))
            actions_list.append(move(self.marauders[1], (center_x - 2, center_y + 1)))
            actions_list.append(move(self.marauders[2], (center_x - 2, center_y)))

            for m_idx, marine in enumerate(self.marines):
                actions_list.append(
                    move(marine, (center_x, center_y + (-len(self.marines) / 2 + m_idx) * 0.5))
                )
            for medivac in self.medivacs:
                actions_list.append(move(medivac, (center_x + 1, medivac.pos.y)))

            for tank in self.tanks:
                actions_list.append(move(tank, (center_x + 1, center_y)))
            return actions_list

        for tank in self.tanks:
            seige_target = min(enemies, key=lambda e: distance_to(tank, e))
            if distance_to(tank, seige_target) < 10 and tank.unit_type == UnitTypeId.SIEGETANK:
                # 388: Siege Mode
                actions_list.append(apply_ability(tank, 388, None))
            elif (
                distance_to(tank, seige_target) > 13
                and tank.unit_type == UnitTypeId.SIEGETANKSIEGED
            ):
                # 390: Unsiege
                actions_list.append(apply_ability(tank, 390, None))

        # Advance in formation (stricter: all Marauders have no enemy within 8)
        if (
            all([len(closer_than(marauder, enemies, 8)) == 0 for marauder in self.marauders])
            and self.engage
        ):
            for agent in agents:
                actions_list.append(move(agent, (agent.pos.x - 1, agent.pos.y)))
            return actions_list

        self.engage = False

        # Always attack nearest enemy
        for m in self.marines:
            if self.pre_health[m.tag] > self.cur_health[m.tag] and m.health / m.health_max < 0.3:
                actions_list.append(move(m, (m.pos.x + 1, m.pos.y)))
            else:
                if enemies:
                    target = nearest_n_units(m, enemies, 1)[0]
                    if target:
                        actions_list.append(attack(m, target, visible_matrix))

        # Front line
        if self.marines:
            front_line = (center(self.marines)[0] - 2, center(self.marines)[1])
        else:
            front_line = (center(agents)[0] - 2, center(agents)[1])

        for m in self.marauders:
            if m.pos.x < front_line[0]:
                if self.enemy_marines + self.enemy_marauders:
                    target = nearest_n_units(m, self.enemy_marines + self.enemy_marauders, 1)[0]
                    if target:
                        actions_list.append(attack(m, target, visible_matrix))
            else:
                actions_list.append(move(m, (front_line[0], m.pos.y)))

        for medivac in self.medivacs:
            all_units = self.marines + self.marauders
            most_injured_unit = min(all_units, key=lambda u: u.health / u.health_max)
            if most_injured_unit.health != most_injured_unit.health_max:
                actions_list.append(attack(medivac, most_injured_unit, visible_matrix))
            else:
                actions_list.append(move(medivac, (center(agents)[0] + 1, center(agents)[1])))

        for tank in self.tanks:
            units_left = self.enemy_marines + self.enemy_marauders + self.enemy_tanks
            if not units_left:
                break
            else:
                nearest_unit = min(units_left, key=lambda e: distance_to(tank, e))
                actions_list.append(attack(tank, nearest_unit, visible_matrix))

        for m in self.marines:
            self.pre_health[m.tag] = self.cur_health[m.tag]

        return [a for a in actions_list if a is not None]
