"""Scripts for `3hl_vs_24zl` / `3rp_vs_24zl`: `3hl_vs_24zl_aw` (split into 3 lanes then assign
targets by index) and `3hl_vs_24zl_ag` (assign targets directly by index). Ported verbatim from
upstream `s_3hl_vs_24zl/attack_ways.py` and `attack_group.py`."""

from __future__ import annotations

from typing import Any

from ..api import attack, find_by_tag, move
from ..base import Script, register


@register('3hl_vs_24zl_aw')
class AttackWays(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init_assign = True
        self.target_tag_dict: dict[int, int] = {}  # agent tag -> enemy tag
        self.init_positions = [(24, 15), (21, 26), (21, 4)]

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        actions: list[Any] = []
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []

        if self.init_assign:
            for idx, agent in enumerate(agents):
                self.target_tag_dict[agent.tag] = enemies[idx % len(enemies)].tag
            self.init_assign = False

        if iteration < 25:
            for idx, agent in enumerate(agents):
                actions.append(move(agent, self.init_positions[idx % 3]))

            return actions

        for idx, agent in enumerate(agents):
            target = find_by_tag(enemies, self.target_tag_dict.get(agent.tag, None))
            if target is None or target.health == 0:
                # Reassign target
                target = enemies[idx % len(enemies)]
                self.target_tag_dict[agent.tag] = target.tag

            actions.append(attack(agent, target, visible_matrix))

        return actions


@register('3hl_vs_24zl_ag')
class AttackGroup(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init_assign = True
        self.target_tag_dict: dict[int, int] = {}  # agent tag -> enemy tag

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        actions: list[Any] = []
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []

        if self.init_assign:
            for idx, agent in enumerate(agents):
                self.target_tag_dict[agent.tag] = enemies[idx % len(enemies)].tag
            self.init_assign = False

        for idx, agent in enumerate(agents):
            target = find_by_tag(enemies, self.target_tag_dict.get(agent.tag, None))
            if target is None or target.health == 0:
                # Reassign target
                target = enemies[idx % len(enemies)]
                self.target_tag_dict[agent.tag] = target.tag

            actions.append(attack(agent, target, visible_matrix))

        return actions
