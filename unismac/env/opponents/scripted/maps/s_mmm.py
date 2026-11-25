"""Scripts for `MMM` and `MMM2`: `MMM` (form line, advance, focus lowest HP+shield among 3
nearest enemy Marines) and `MMM_2` (form line, advance, attack nearest). Ported from upstream
`s_MMM/script.py` and `s_MMM/script_2.py`.

Difference: `random.choice` uses `self.random.choice`. Preserved upstream behaviors:

* `cur_health` stays at 1 after init, so `pre_health > cur_health` is always False and Marine
  retreat never triggers; dicts only include Marines alive at init;
* `MMM`'s `random.choice(enemies)` targets the raw enemy dict (unreachable since `target_units`
  is always non-empty while enemies survive);
* `MMM_2` indexes `marauders[0]` / `marauders[1]` during formation, raising `IndexError` with
  fewer than 2 Marauders;
* If only Medivacs remain, `center(marines + marauders)` raises `ZeroDivisionError` and `min([])`
  raises `ValueError`;
* Advance condition is `any(...)` in `MMM` vs `all(...)` in `MMM_2` (False vs True when no
  Marauders remain).
"""

from __future__ import annotations

from typing import Any

from ..api import UnitTypeId, attack, center, closer_than, move, nearest_n_units
from ..base import Script, register


@register('MMM')
class MMMFormFocusMarines(Script):
    """Form line (Marauders front, Marines middle, Medivacs back) and advance; focus lowest
    HP+shield among 3 nearest enemy Marines (fallback: Marauders, any enemy); Medivacs support
    most injured ally."""

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.units = [agent for _, agent in agents.items() if agent.health != 0]
        self.enemy_units = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        self.marines = sorted(
            [a for a in self.units if a.unit_type == UnitTypeId.MARINE], key=lambda a: a.tag
        )
        self.marauders = sorted(
            [a for a in self.units if a.unit_type == UnitTypeId.MARAUDER], key=lambda a: a.tag
        )
        self.medivacs = sorted(
            [a for a in self.units if a.unit_type == UnitTypeId.MEDIVAC], key=lambda a: a.tag
        )

        self.enemy_marines = sorted(
            [a for a in self.enemy_units if a.unit_type == UnitTypeId.MARINE], key=lambda a: a.tag
        )
        self.enemy_marauders = sorted(
            [a for a in self.enemy_units if a.unit_type == UnitTypeId.MARAUDER],
            key=lambda a: a.tag,
        )
        self.enemy_medivacs = sorted(
            [a for a in self.enemy_units if a.unit_type == UnitTypeId.MEDIVAC],
            key=lambda a: a.tag,
        )

        if not self.units or not self.enemy_units:
            return []

        actions_list: list[Any] = []

        if self.init:
            self.pre_health: dict[int, Any] = {}
            self.cur_health: dict[int, Any] = {}
            for m in self.marines:
                self.pre_health[m.tag] = 1
                self.cur_health[m.tag] = 1
            self.init = False

        if self.enemy_marines:
            target_units = self.enemy_marines
        elif self.enemy_marauders:
            target_units = self.enemy_marauders
        else:
            target_units = self.enemy_units

        # Tactic 1: form line (Marauders front, Marines middle, Medivacs back)
        if iteration <= 5:
            center_x, center_y = center(self.marines + self.marauders)
            for marauder in self.marauders:
                actions_list.append(move(marauder, (center_x - 2, marauder.pos.y)))
            for marine in self.marines:
                actions_list.append(move(marine, (center_x, marine.pos.y)))
            for medivac in self.medivacs:
                actions_list.append(move(medivac, (center_x + 1, medivac.pos.y)))

            return actions_list
        # Advance in formation toward enemies
        if (
            any(
                [
                    len(closer_than(marauder, self.enemy_units, 8)) == 0
                    for marauder in self.marauders
                ]
            )
            and self.engage
        ):
            for agent in self.units:
                actions_list.append(move(agent, (agent.pos.x - 1, agent.pos.y)))
            return actions_list

        self.engage = False
        # Prioritize Marines, then Marauders, then Medivacs

        if target_units:
            target = min(
                nearest_n_units(center(self.units), target_units, 3),
                key=lambda e: e.health + e.shield,
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
            front_line = (center(self.units)[0] - 2, center(self.units)[1])

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
                actions_list.append(
                    move(medivac, (center(self.units)[0] + 1, center(self.units)[1]))
                )

        for m in self.marines:
            self.pre_health[m.tag] = self.cur_health[m.tag]

        return actions_list


@register('MMM_2')
class MMMFormNearest(Script):
    """Form line (Marines in column) and advance while all Marauders have no enemy within 8; then
    attack nearest (Marauders target only Marines/Marauders); Medivacs support most injured ally."""

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.units = [agent for _, agent in agents.items() if agent.health != 0]
        self.enemy_units = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        self.marines = sorted(
            [a for a in self.units if a.unit_type == UnitTypeId.MARINE], key=lambda a: a.tag
        )
        self.marauders = sorted(
            [a for a in self.units if a.unit_type == UnitTypeId.MARAUDER], key=lambda a: a.tag
        )
        self.medivacs = sorted(
            [a for a in self.units if a.unit_type == UnitTypeId.MEDIVAC], key=lambda a: a.tag
        )

        self.enemy_marines = sorted(
            [a for a in self.enemy_units if a.unit_type == UnitTypeId.MARINE], key=lambda a: a.tag
        )
        self.enemy_marauders = sorted(
            [a for a in self.enemy_units if a.unit_type == UnitTypeId.MARAUDER],
            key=lambda a: a.tag,
        )
        self.enemy_medivacs = sorted(
            [a for a in self.enemy_units if a.unit_type == UnitTypeId.MEDIVAC],
            key=lambda a: a.tag,
        )

        if not self.units or not self.enemy_units:
            return []

        actions_list: list[Any] = []

        if self.init:
            self.pre_health: dict[int, Any] = {}
            self.cur_health: dict[int, Any] = {}
            for m in self.marines:
                self.pre_health[m.tag] = 1
                self.cur_health[m.tag] = 1
            self.init = False

        # Tactic 1: form line (Marauders front, Marines in column, Medivacs back)
        center_x, center_y = center(self.marines + self.marauders)
        if iteration <= 5:
            actions_list.append(move(self.marauders[0], (center_x - 2, center_y - 1)))
            actions_list.append(move(self.marauders[1], (center_x - 2, center_y + 1)))
            for m_idx, marine in enumerate(self.marines):
                actions_list.append(
                    move(marine, (center_x, marine.pos.y - len(self.marines) / 2 + m_idx))
                )
            for medivac in self.medivacs:
                actions_list.append(move(medivac, (center_x + 1, medivac.pos.y)))

            return actions_list
        # Advance in formation (stricter condition)
        if (
            all(
                [
                    len(closer_than(marauder, self.enemy_units, 8)) == 0
                    for marauder in self.marauders
                ]
            )
            and self.engage
        ):
            for agent in self.units:
                actions_list.append(move(agent, (agent.pos.x - 1, agent.pos.y)))
            return actions_list

        self.engage = False

        # Always attack nearest enemy

        for m in self.marines:
            if self.pre_health[m.tag] > self.cur_health[m.tag] and m.health / m.health_max < 0.3:
                actions_list.append(move(m, (m.pos.x + 1, m.pos.y)))
            else:
                if self.enemy_units:
                    target = nearest_n_units(m, self.enemy_units, 1)[0]
                    if target:
                        actions_list.append(attack(m, target, visible_matrix))

        # Front line
        if self.marines:
            front_line = (center(self.marines)[0] - 2, center(self.marines)[1])
        else:
            front_line = (center(self.units)[0] - 2, center(self.units)[1])

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
                actions_list.append(
                    move(medivac, (center(self.units)[0] + 1, center(self.units)[1]))
                )

        for m in self.marines:
            self.pre_health[m.tag] = self.cur_health[m.tag]

        return actions_list
