"""Scripts for `3s_vs_3z`: `3s_vs_3z_a` (staged formation advance then attack) and `3s_vs_3z_g`
(round-robin target split).

Ported from upstream `smac_hard/env/scripts/s_3s_vs_3z/script_attack.py` and `script_group.py`.
Difference: `UnitTypeId.X.value` replaced with int constants. Preserved upstream behaviors:
* Formation anchor is derived from `zealots[0]` (unit);
* `group` indexes `enemy_stalkers[i % len]` on step 0, raising `ZeroDivisionError` if both sides
  have units but no enemy Stalkers exist;
* `target_dict` stores unit objects rather than tags; Zealots appearing after step 0 raise
  `KeyError`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..api import UnitTypeId, attack, center, distance_to, move, nearest_n_units, toward
from ..base import Script, register


@register('3s_vs_3z_a')
class ZealotFormationAttack3z(Script):
    """Advance in staged formations toward Stalkers; once any Zealot is within 6.5, attack the
    nearest Stalker."""

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.pos_init = True
        self.stage = 0
        self.zealots_tags: list[int] = []
        self.center_distance = 13

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.units = [agent for _, agent in agents.items() if agent.health != 0]
        self.enemy_units = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        self.zealots = sorted(
            [a for a in self.units if a.unit_type == UnitTypeId.ZEALOT], key=lambda a: a.tag
        )
        self.enemy_stalkers = sorted(
            [a for a in self.enemy_units if a.unit_type == UnitTypeId.STALKER],
            key=lambda a: a.tag,
        )

        if not self.units or not self.enemy_units:
            return []

        actions_list: list[Any] = []

        self.sort_zealots()

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
            self.zealots_tags.append(self.zealots[np.argmin(y)].tag)
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
        # Exit formation once any Zealot is within 6.5 of its nearest Stalker.
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
                (x, y - 4),
                (x, y + 4),
            ],
            [
                (x, y),
                (x - 8, y - 8),
                (x - 8, y + 8),
            ],
            [
                (x, y),
                (x - 8 - 4 * math.sqrt(3), y - 4),
                (x - 8 - 4 * math.sqrt(3), y + 4),
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


@register('3s_vs_3z_g')
class ZealotSplitTarget3z(Script):
    """Round-robin target assignment by tag order on step 0; switch to nearest Stalker on kill."""

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.target_dict: dict[int, Any] = {}  # Zealot tag -> target unit object (upstream style)

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.units = [agent for _, agent in agents.items() if agent.health != 0]
        self.enemy_units = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        zealots = sorted(
            [a for a in self.units if a.unit_type == UnitTypeId.ZEALOT], key=lambda a: a.tag
        )
        enemy_stalkers = sorted(
            [a for a in self.enemy_units if a.unit_type == UnitTypeId.STALKER],
            key=lambda a: a.tag,
        )

        self.actions_list: list[Any] = []

        if not self.units or not self.enemy_units:
            return []

        if self.init:
            for i, zealot in enumerate(zealots):
                self.target_dict[zealot.tag] = enemy_stalkers[i % len(enemy_stalkers)]
            self.init = False

        if not zealots or not enemy_stalkers:
            return []

        for _i, z in enumerate(zealots):
            target = self.target_dict[z.tag]
            if target.health == 0:
                target = nearest_n_units(z, enemy_stalkers, 1)[0]
                self.target_dict[z.tag] = target

            self.actions_list.append(attack(z, target, visible_matrix))

        return self.actions_list
