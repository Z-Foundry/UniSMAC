"""Scripts for `2vr_vs_3sc`: `2vr_vs_3sc_watkw` (Spore Crawlers reposition and focus weakest) and
`2vr_vs_3sc_watkn` (reposition and attack nearest). Ported from upstream
`s_2vr_vs_3sc/walk_aktw.py` and `walk_aktn.py`.

Preserved upstream behaviors: no empty-side guard (`min()` raises `ValueError` when enemies are
wiped out); missing ability entries raise `IndexError` at `[...][0]`. Checks 1727 before uproot
(3681) and 1731 before root (3680), matching upstream. Literal 140 (SporeCrawlerUprooted) is kept
as in upstream.
"""

from __future__ import annotations

from typing import Any

from ..api import apply_ability, attack, distance_to, move
from ..base import Script, register

# Uproot: 3681, Root: 3680


@register('2vr_vs_3sc_watkw')
class WalkAttackWeakest(Script):
    needs_abilities = True

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.center = (14.0, 21.0)
        self.attack_range = 7
        self.walk_range = 4

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        actions: list[Any] = []
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        for agent in agents:
            a_ability = [ab.abilities for ab in agent_ability if ab.unit_tag == agent.tag][0]
            avail_ability = [a.ability_id for a in a_ability]

            if agent.unit_type == 140:
                # Uprooted Spore Crawler
                target = min(
                    enemies, key=lambda e: (e.health + e.shield) / (e.health_max + e.shield_max)
                )

                if distance_to(agent, self.center) > 5:
                    actions.append(move(agent, self.center))
                else:
                    if distance_to(agent, target) >= 8 and distance_to(agent, self.center) < 4:
                        actions.append(move(agent, target))
                    else:
                        if 1731 in avail_ability:
                            actions.append(apply_ability(agent, 3680, (agent.pos.x, agent.pos.y)))

            elif agent.unit_type == 99:
                # Rooted Spore Crawler
                target = min(
                    enemies, key=lambda e: (e.health + e.shield) / (e.health_max + e.shield_max)
                )
                if distance_to(target, agent) >= 8 or distance_to(agent, self.center) > 5:
                    if 1727 in avail_ability:
                        actions.append(apply_ability(agent, 3681, None))
                else:
                    actions.append(attack(agent, target, visible_matrix))

        return actions


@register('2vr_vs_3sc_watkn')
class WalkAttackNearest(Script):
    needs_abilities = True

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.center = (14.0, 21.0)
        self.attack_range = 7
        self.walk_range = 4

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        actions: list[Any] = []
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]
        for agent in agents:
            a_ability = [ab.abilities for ab in agent_ability if ab.unit_tag == agent.tag][0]
            avail_ability = [a.ability_id for a in a_ability]

            if agent.unit_type == 140:
                # Uprooted Spore Crawler
                target = min(enemies, key=lambda e: distance_to(e, agent))

                if distance_to(agent, self.center) > 5:
                    actions.append(move(agent, self.center))
                else:
                    if distance_to(agent, target) >= 8 and distance_to(agent, self.center) < 4:
                        actions.append(move(agent, target))
                    else:
                        if 1731 in avail_ability:
                            actions.append(apply_ability(agent, 3680, (agent.pos.x, agent.pos.y)))

            elif agent.unit_type == 99:
                # Rooted Spore Crawler
                target = min(enemies, key=lambda e: distance_to(e, agent))
                if distance_to(target, agent) >= 8 or distance_to(agent, self.center) > 5:
                    if 1727 in avail_ability:
                        actions.append(apply_ability(agent, 3681, None))
                else:
                    actions.append(attack(agent, target, visible_matrix))

        return actions
