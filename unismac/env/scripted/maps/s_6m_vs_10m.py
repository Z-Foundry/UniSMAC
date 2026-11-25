"""Scripts for `6m_vs_10m`: `6m_vs_10m_fatkw` (form line then focus weakest) and `6m_vs_10m_fatkn`
(form line then attack nearest). Ported verbatim from upstream `s_6m_vs_10m/form_atkw.py` and
`form_atkn.py`."""

from __future__ import annotations

from typing import Any

from ..api import attack, center, distance_to, move
from ..base import Script, register


@register('6m_vs_10m_fatkw')
class FormAttackWeakest(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.actions_list: list[Any] = []

        marines = sorted(
            [agent for _, agent in agents.items() if agent.health != 0], key=lambda a: a.tag
        )
        enemy_marines = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if self.init:
            self.pre_health: dict[int, float] = {}
            self.cur_health: dict[int, float] = {}
            for m in marines:
                self.pre_health[m.tag] = 1
                self.cur_health[m.tag] = 1
            self.init = False

        if not marines or not enemy_marines:
            return []

        for m in marines:
            self.cur_health[m.tag] = m.health / m.health_max

        # Form a vertical line at x=28 for the first 10 steps
        if iteration < 10:
            for i, m in enumerate(marines):
                self.actions_list.append(move(m, (28, 16 + (i - len(marines) / 2))))
        elif min([distance_to(e, center(marines)) for e in enemy_marines]) > 8 and self.engage:
            for m in marines:
                self.actions_list.append(move(m, (m.pos.x - 1, m.pos.y)))
        else:
            self.engage = False
            target = min(enemy_marines, key=lambda e: e.health)
            for _, m in enumerate(marines):
                if (
                    self.pre_health[m.tag] != self.cur_health[m.tag]
                    and m.health / m.health_max < 0.3
                ):
                    self.actions_list.append(move(m, (m.pos.x + 2, m.pos.y)))
                else:
                    self.actions_list.append(attack(m, target, visible_matrix))

        for m in marines:
            self.pre_health[m.tag] = self.cur_health[m.tag]

        return self.actions_list


@register('6m_vs_10m_fatkn')
class FormAttackNearest(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.actions_list: list[Any] = []

        marines = sorted(
            [agent for _, agent in agents.items() if agent.health != 0], key=lambda a: a.tag
        )
        enemy_marines = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if self.init:
            self.pre_health: dict[int, float] = {}
            self.cur_health: dict[int, float] = {}
            for m in marines:
                self.pre_health[m.tag] = 1
                self.cur_health[m.tag] = 1
            self.init = False

        if not marines or not enemy_marines:
            return []

        for m in marines:
            self.cur_health[m.tag] = m.health / m.health_max

        # Form a vertical line at x=28 for the first 10 steps
        if iteration < 10:
            for i, m in enumerate(marines):
                self.actions_list.append(move(m, (28, 16 + (i - len(marines) / 2))))
        elif min([distance_to(e, center(marines)) for e in enemy_marines]) > 8 and self.engage:
            for m in marines:
                self.actions_list.append(move(m, (m.pos.x - 1, m.pos.y)))
        else:
            self.engage = False
            for _, m in enumerate(marines):
                target = min(enemy_marines, key=lambda e: distance_to(e, m))

                if (
                    self.pre_health[m.tag] != self.cur_health[m.tag]
                    and m.health / m.health_max < 0.3
                ):
                    self.actions_list.append(move(m, (m.pos.x + 2, m.pos.y)))
                else:
                    self.actions_list.append(attack(m, target, visible_matrix))

        for m in marines:
            self.pre_health[m.tag] = self.cur_health[m.tag]

        return self.actions_list
