"""Scripts for the `8m` family (`3m`, `8m`, `5m_vs_6m`, `8m_vs_9m`, `10m_vs_11m`): `8m` (focus
enemy with x closest to ally centroid) and `8m_1` (focus lowest-HP enemy). Ported verbatim from
upstream `s_8m/script.py` and `s_8m/script_1.py`.

Preserved upstream behaviors:

* `cur_health` / `pre_health` are initialized once from the Marines alive on the first call;
* Both duplicate `if not marines or not enemy_marines` early returns are kept.
"""

from __future__ import annotations

import math
from typing import Any

from ..api import attack, center, distance_to, move
from ..base import Script, register


@register('8m')
class MarineLineFocusCentral(Script):
    """Form a vertical line, advance left to distance 8, then focus enemy closest in x to ally
    centroid; retreat damaged Marines below 30% HP."""

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.actions_list: list[Any] = []

        marines = [agent for _, agent in agents.items() if agent.health != 0]
        enemy_marines = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not marines or not enemy_marines:
            return []

        if self.init:
            self.pre_health: dict[int, Any] = {}
            self.cur_health: dict[int, Any] = {}
            for m in marines:
                self.pre_health[m.tag] = 1
                self.cur_health[m.tag] = 1
            self.init = False

        if not marines or not enemy_marines:
            return []

        for m in marines:
            self.cur_health[m.tag] = m.health / m.health_max

        # Form a vertical line at x=25 for the first 10 steps
        if iteration < 10:
            for i, m in enumerate(marines):
                self.actions_list.append(move(m, (25, 16 + (i - len(marines) / 2))))
        elif min([distance_to(e, center(marines)) for e in enemy_marines]) > 8 and self.engage:
            for m in marines:
                self.actions_list.append(move(m, (m.pos.x - 1, m.pos.y)))
        else:
            self.engage = False
            target = min(enemy_marines, key=lambda e: math.fabs(e.pos.x - center(marines)[0]))
            for _i, m in enumerate(marines):
                if (
                    self.pre_health[m.tag] != self.cur_health[m.tag]
                    and m.health / m.health_max < 0.3
                ):
                    self.actions_list.append(move(m, (m.pos.x + 1, m.pos.y)))
                else:
                    self.actions_list.append(attack(m, target, visible_matrix))

        for m in marines:
            self.pre_health[m.tag] = self.cur_health[m.tag]

        return self.actions_list


@register('8m_1')
class MarineLineFocusWeakest(Script):
    """Like `8m`, but focuses the lowest-HP enemy."""

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.actions_list: list[Any] = []

        marines = [agent for _, agent in agents.items() if agent.health != 0]
        enemy_marines = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not marines or not enemy_marines:
            return []

        if self.init:
            self.pre_health: dict[int, Any] = {}
            self.cur_health: dict[int, Any] = {}
            for m in marines:
                self.pre_health[m.tag] = 1
                self.cur_health[m.tag] = 1
            self.init = False

        if not marines or not enemy_marines:
            return []

        for m in marines:
            self.cur_health[m.tag] = m.health / m.health_max

        # Form a vertical line at x=25 for the first 10 steps
        if iteration < 10:
            for i, m in enumerate(marines):
                self.actions_list.append(move(m, (25, 16 + (i - len(marines) / 2))))
        elif min([distance_to(e, center(marines)) for e in enemy_marines]) > 8 and self.engage:
            for m in marines:
                self.actions_list.append(move(m, (m.pos.x - 1, m.pos.y)))
        else:
            self.engage = False
            target = min(enemy_marines, key=lambda e: e.health)
            for _i, m in enumerate(marines):
                if (
                    self.pre_health[m.tag] != self.cur_health[m.tag]
                    and m.health / m.health_max < 0.3
                ):
                    self.actions_list.append(move(m, (m.pos.x + 1, m.pos.y)))
                else:
                    self.actions_list.append(attack(m, target, visible_matrix))

        for m in marines:
            self.pre_health[m.tag] = self.cur_health[m.tag]

        return self.actions_list
