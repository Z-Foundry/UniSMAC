"""Scripts for `3st_vs_5zl`: `3st_vs_5zl_a` (staged formation advance then attack) and
`3st_vs_5zl_g` (assign targets by index, retreat when low HP and shield).

Ported from upstream `smac_hard/env/scripts/s_3st_vs_5zl/script_attack.py` and `script_group.py`.
Difference: `target == None` written as `target is None` (same result). Preserved behaviors:

* `attack` does not filter by unit type (all alive units are treated as "zealots" /
  "enemy_stalkers"); re-sorts by tag after `sort_zealots` (discarding spatial order, keeping
  duplicate tags so a unit may receive two commands); formation anchor uses `zealots[0]`;
* `group` initializes `status` only on step 0 (later-appearing units raise `KeyError`); units
  with `shield_max == 0` raise `ZeroDivisionError`; retreats via `toward(agent, target, -2)`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..api import attack, center, distance_to, find_by_tag, move, nearest_n_units, toward
from ..base import Script, register


@register('3st_vs_5zl_a')
class FormationAttack(Script):
    """Advance in staged formations toward enemies; once any unit is within 6.5, attack nearest."""

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.zealots: list[Any] = []
        self.enemy_stalkers: list[Any] = []

        self.pos_init = True
        self.stage = 0
        self.zealots_tags: list[int] = []
        self.center_distance = 13

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        actions_list: list[Any] = []

        self.zealots = sorted(
            [agent for _, agent in agents.items() if agent.health != 0], key=lambda a: a.tag
        )
        self.enemy_stalkers = sorted(
            [enemy for _, enemy in enemies.items() if enemy.health != 0], key=lambda e: e.tag
        )

        if not self.zealots or not self.enemy_stalkers:
            return []

        self.sort_zealots()
        self.zealots = sorted([unit for unit in self.zealots], key=lambda u: u.tag)

        if self.pos_init and self.close_check():
            target_center = self.get_target_center()
            agent_center = self.get_agent_center(target_center, self.zealots[0])

            pos_list = self.get_pos_list(agent_center)
            if self.pos_init:
                for p, u in zip(pos_list, self.zealots, strict=False):
                    actions_list.append(move(u, p))
                return actions_list

        for z in self.zealots:
            target = nearest_n_units(z, self.enemy_stalkers, 1)[0]
            actions_list.append(attack(z, target, visible_matrix))

        return actions_list

    def sort_zealots(self) -> None:
        # Record initial tag order by position (argmax/argmin may collide and duplicate a tag).
        if not self.zealots_tags:
            x = [u.pos.x for u in self.zealots]
            y = [u.pos.y for u in self.zealots]
            self.zealots_tags.append(self.zealots[np.argmax(x)].tag)
            self.zealots_tags.append(self.zealots[np.argmax(y)].tag)
            self.zealots_tags.append(self.zealots[np.argmin(y)].tag)
            self.zealots_tags.append(self.zealots[np.argmin(x)].tag)
            for z in self.zealots:
                if z.tag not in self.zealots_tags:
                    self.zealots_tags.append(z.tag)
                    break
        zealots = []
        for t in self.zealots_tags:
            for z in self.zealots:
                if z.tag == t:
                    zealots.append(z)
                    break
        self.zealots = zealots

    def close_check(self) -> bool:
        # Exit formation once any unit is within 6.5 of its nearest enemy.
        for z in self.zealots:
            if distance_to(z, nearest_n_units(z, self.enemy_stalkers, 1)[0]) < 6.5:
                self.pos_init = False
                return False
        return True

    def get_target_center(self) -> Any:
        target_center = center(self.enemy_stalkers)
        for s in self.enemy_stalkers:
            if distance_to(s, target_center) > 3:
                target_center = None
                break
        if target_center is not None:
            return target_center
        agent_center = center(self.zealots)
        closest_enemy = nearest_n_units(agent_center, self.enemy_stalkers, 1)[0]
        return (closest_enemy.pos.x, closest_enemy.pos.y)

    def get_agent_center(self, target_center, center_agnet) -> Any:
        # center_agnet is a unit (preserving upstream parameter spelling).
        if self.stage > 0:
            self.center_distance = 8
            return toward(target_center, center_agnet, self.center_distance)
        if distance_to(target_center, center_agnet) > self.center_distance:
            return (center_agnet.pos.x, center_agnet.pos.y)
        center_agnet_pos = toward(target_center, center_agnet, self.center_distance)
        return center_agnet_pos

    def _pos_list(self, x, y, stage):
        pos_list = [
            [
                (x, y),
                (x, y + 4),
                (x, y - 4),
                (x, y + 2),
                (x, y - 2),
            ],
            [
                (x, y),
                (x - 8, y + 8),
                (x - 8, y - 8),
                (x - 8 + 4 * math.sqrt(3), y + 4),
                (x - 8 + 4 * math.sqrt(3), y - 4),
            ],
            [
                (x, y),
                (x - 8 - 4 * math.sqrt(3), y + 4),
                (x - 8 - 4 * math.sqrt(3), y - 4),
                (x - 8 + 4 * math.sqrt(3), y + 4),
                (x - 8 + 4 * math.sqrt(3), y - 4),
            ],
        ]
        return pos_list[stage]

    def check_pos_list(self, pos_list) -> bool:
        for p, u in zip(pos_list, self.zealots, strict=False):
            if distance_to(p, u) > 1:
                return False
        return True

    def get_pos_list(self, agent_center) -> list[Any]:
        # Advance formation stage once positions are reached; finish after all 3 stages.
        x, y = agent_center
        while self.stage < 3:
            pos_list = self._pos_list(x, y, self.stage)
            if self.check_pos_list(pos_list):
                self.stage += 1
            else:
                return pos_list
        self.pos_init = False
        return []


@register('3st_vs_5zl_g')
class GroupAttackRetreat(Script):
    """Assign targets round-robin by index; retreat when HP and shield are both < 20%, re-engage
    once shield exceeds 80%."""

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.groups: dict[int, int] = {}  # agent tag -> target tag
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
