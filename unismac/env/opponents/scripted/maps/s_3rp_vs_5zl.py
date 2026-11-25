"""Scripts for `3rp_vs_5zl`: `3rp_vs_5zl_g` (assign targets by index, retreat when low HP and
shield) and `3rp_vs_5zl_c` (high/low cliff guards plus one roamer). Ported from upstream
`s_3rp_vs_5zl/script_group.py` and `script_cliff.py`.

Preserved upstream bugs: `script_cliff` raises `IndexError` at `agents[k]` if fewer than 5 units
are alive on first call; with fewer than 4 alive units, `self.targets[agent.tag]` is never
populated and raises `KeyError`.
"""

from __future__ import annotations

from typing import Any

from ..api import attack, distance_to, find_by_tag, move, toward
from ..base import Script, register


@register('3rp_vs_5zl_g')
class GroupRetreat(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.groups: dict[int, int] = {}  # agent tag -> enemy tag
        self.status: dict[int, str] = {}  # agent tag -> 'Attack' / 'Retreat'
        self.init = True

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []
        actions: list[Any] = []

        if self.init:
            for agent in agents:
                self.status[agent.tag] = 'Attack'
            self.init = False

        # Assign a target to each unit by index
        for a_id, agent in enumerate(agents):
            if agent.health / agent.health_max < 0.2 and agent.shield / agent.shield_max < 0.2:
                self.status[agent.tag] = 'Retreat'
            elif agent.shield / agent.shield_max > 0.8:
                self.status[agent.tag] = 'Attack'

            status = self.status[agent.tag]

            target = find_by_tag(enemies, self.groups.get(agent.tag, None))
            if target is None or target.health == 0:
                self.groups[agent.tag] = enemies[a_id % len(enemies)].tag
                target = enemies[a_id % len(enemies)]

            if status == 'Attack':
                actions.append(attack(agent, target, visible_matrix))
            elif status == 'Retreat':
                actions.append(move(agent, toward(agent, target, -2)))

        return actions


@register('3rp_vs_5zl_c')
class Cliff(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.up_cliff: list[int] = []
        self.down_cliff: list[int] = []
        self.free: int | None = None
        self.init = True
        self.cliff_y = 13
        self.targets: dict[int, int] = {}  # agent tag -> enemy tag (initialized on first call)

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []
        actions: list[Any] = []

        if self.init:
            self.up_cliff = [agents[0].tag, agents[1].tag]
            self.down_cliff = [agents[2].tag, agents[3].tag]
            self.free = agents[4].tag
            self.targets = {}
            self.init = False

        if len(agents) >= 4:
            # Group 1 guards upper cliff, group 2 guards lower cliff, roamer chases nearest enemy
            for agent in agents:
                nearest_target = min(enemies, key=lambda e: distance_to(e, agent))
                if agent.tag in self.up_cliff:
                    if nearest_target.pos.y > self.cliff_y:
                        actions.append(attack(agent, nearest_target, visible_matrix))
                    else:
                        actions.append(move(agent, (16, 16)))
                elif agent.tag in self.down_cliff:
                    if nearest_target.pos.y < self.cliff_y:
                        actions.append(attack(agent, nearest_target, visible_matrix))
                    else:
                        actions.append(move(agent, (16, 9)))
                elif agent.tag == self.free:
                    actions.append(attack(agent, nearest_target, visible_matrix))
        else:
            for _, agent in enumerate(agents):
                # Upstream indexes self.targets before populating it (KeyError), preserved verbatim.
                target = find_by_tag(enemies, self.targets[agent.tag])
                if target is None or target.health == 0:
                    target = min(enemies, key=lambda e: distance_to(e, agent))
                    self.targets[agent.tag] = target.tag
                actions.append(attack(agent, target, visible_matrix))

        return actions
