"""Scripts for `25m` and `27m_vs_30m`: `27m_a` (attack enemy centroid) and `27m_ap` (attack the
position of the lowest-HP unit among the 5 nearest enemies). Ported from upstream
`s_27m/script_a.py` and `s_27m/script_ap.py` (`script_l.py` is not in `SCRIPT_DICT`).

No behavioral differences from upstream; `self.formation` set in `update_formation` is kept even
though unread.
"""

from __future__ import annotations

from typing import Any

from ..api import attack, center, distance_to, move_point, nearest_n_units
from ..base import Script, register


@register('27m_a')
class MarineLineAttackCenter(Script):
    """Form a vertical line sorted by y for 10 steps, advance left to distance 9, then attack-move
    toward enemy centroid."""

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.actions_list: list[Any] = []

        marines = [agent for _, agent in agents.items() if agent.health != 0]
        enemy_marines = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not marines or not enemy_marines:
            return []

        if iteration < 10:
            self.update_formation(marines)
            return self.actions_list

        # Advance left
        if min([distance_to(center(marines), em) for em in enemy_marines]) > 9:
            for marine in marines:
                self.actions_list.append(move_point(marine, marine.pos.x - 2, marine.pos.y))
        else:
            for marine in marines:
                self.actions_list.append(attack(marine, center(enemy_marines), visible_matrix))

        return self.actions_list

    def update_formation(self, marines: list[Any]) -> None:
        marines_sorted = sorted(marines, key=lambda m: m.pos.y)

        for i, marine in enumerate(marines_sorted):
            position = (25, 16 + (i - len(marines) / 2) * 0.65)
            self.actions_list.append(move_point(marine, position[0], position[1]))
        self.formation = marines_sorted


@register('27m_ap')
class MarineLineAttackWeakPoint(Script):
    """Like `27m_a`, but once engaged each Marine attack-moves to the position of the weakest of
    its 5 nearest enemies."""

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.actions_list: list[Any] = []

        marines = [agent for _, agent in agents.items() if agent.health != 0]
        enemy_marines = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not marines or not enemy_marines:
            return []

        if iteration < 10:
            self.update_formation(marines)
            return self.actions_list

        # Advance left
        if min([distance_to(center(marines), em) for em in enemy_marines]) > 9:
            for marine in marines:
                self.actions_list.append(move_point(marine, marine.pos.x - 2, marine.pos.y))
        else:
            for marine in marines:
                target = nearest_n_units(marine, enemy_marines, 5)
                weakest_target = min(target, key=lambda e: e.health)

                self.actions_list.append(
                    attack(marine, (weakest_target.pos.x, weakest_target.pos.y), visible_matrix)
                )

        return self.actions_list

    def update_formation(self, marines: list[Any]) -> None:
        marines_sorted = sorted(marines, key=lambda m: m.pos.y)

        for i, marine in enumerate(marines_sorted):
            position = (25, 16 + (i - len(marines) / 2) * 0.65)
            self.actions_list.append(move_point(marine, position[0], position[1]))
        self.formation = marines_sorted
