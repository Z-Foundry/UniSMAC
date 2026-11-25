"""Script `2c_vs_64zg` for `2c_vs_64zg`: splits Zerglings into 4 groups guarding high/low ground
and attacks or retreats based on enemy Colossus elevation.

Ported from upstream `smac_hard/env/scripts/s_2c_vs_64zg/script.py` (listed twice in `POOLS` for
weighting). Differences: `UnitTypeId.X.value` replaced by int constants; unused scipy/numpy
removed; local `center` (shadowing helper) renamed to `group_center` with identical semantics.
Preserved upstream behaviors:

* Groups are partitioned once on first call; dead tags are pruned later, and imbalanced groups
  rebalance lowest-y units (checking left first; if left moves, right is skipped that step).
* Empty groups report center `(1000, 1000)`; `attack_above` / `attack_down` overlap on
  `y in (11.9, 12.1)`.
"""

from __future__ import annotations

from typing import Any

from ..api import UnitTypeId, attack, center, move, nearest_n_units
from ..base import Script, register


@register('2c_vs_64zg')
class ZerglingCliffGroups(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)

        self.group_tags: dict[str, list[int]] = {}

        self.positions = {
            'up_left': [(3.5, 14), (2, 16.5)],
            'up_right': [(26.5, 14), (28, 16.5)],
            'down_left': [(3.5, 6), (2.5, 10)],
            'down_right': [(26.5, 6), (27.5, 10)],
        }

        self.clif_y = 12
        self.up_group = ['up_left', 'up_right']
        self.down_group = ['down_left', 'down_right']

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.units = [agent for _, agent in agents.items() if agent.health != 0]
        self.enemy_units = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        self.zerglings = sorted(
            [a for a in self.units if a.unit_type == UnitTypeId.ZERGLING], key=lambda a: a.tag
        )
        self.enemy_colossuses = sorted(
            [a for a in self.enemy_units if a.unit_type == UnitTypeId.COLOSSUS],
            key=lambda a: a.tag,
        )

        if not self.units or not self.enemy_units:
            return []

        actions_list: list[Any] = []

        group_units, group_centers = self.group()
        if self.switch_group(
            group_units['up_left'], group_units['down_left'], 'left'
        ) or self.switch_group(group_units['up_right'], group_units['down_right'], 'right'):
            group_units, group_centers = self.group()

        # Move groups to rally points for the first 10 steps.
        if iteration < 10:
            for group_id, group in group_units.items():
                pos = self.positions[group_id][0]
                for unit in group:
                    actions_list.append(move(unit, pos))
            return actions_list
        self.check_state()
        if self.attack_above:
            for group_id in self.up_group:
                group = group_units[group_id]
                group_center = group_centers[group_id]
                target_enemy = nearest_n_units(group_center, self.attack_above, 1)[0]
                for unit in group:
                    actions_list.append(attack(unit, target_enemy, visible_matrix))
        else:
            for group_id in self.up_group:
                group = group_units[group_id]
                pos = self.positions[group_id][1]
                for unit in group:
                    actions_list.append(move(unit, pos))
        if self.attack_down:
            for group_id in self.down_group:
                group = group_units[group_id]
                group_center = group_centers[group_id]
                target_enemy = nearest_n_units(group_center, self.attack_down, 1)[0]
                for unit in group:
                    actions_list.append(attack(unit, target_enemy, visible_matrix))
        else:
            for group_id in self.down_group:
                group = group_units[group_id]
                pos = self.positions[group_id][1]
                for unit in group:
                    actions_list.append(move(unit, pos))

        return actions_list

    def group(self) -> tuple[dict[str, list[Any]], dict[str, Any]]:
        # Resolve alive units per group by stored tag and prune dead tags.
        if not self.group_tags:
            self._group()
        unit_tags = [unit.tag for unit in self.zerglings]
        group_units = {}
        for key, tags in self.group_tags.items():
            units, del_list = [], []
            for tag in tags:
                if tag in unit_tags:
                    tag_idx = unit_tags.index(tag)
                    units.append(self.zerglings[tag_idx])
                else:
                    del_list.append(tag)
            for tag in del_list:
                tags.remove(tag)
            group_units[key] = units
        group_centers = {}
        for key, group in group_units.items():
            group_centers[key] = center(group) if group else (1000, 1000)
        return group_units, group_centers

    def _group(self) -> None:
        self.group_tags = {'up_left': [], 'up_right': [], 'down_left': [], 'down_right': []}
        for unit in self.zerglings:
            if (unit.pos.x < 15.6 and unit.pos.y > 7.9) or (
                unit.pos.x < 16.3 and unit.pos.y > 10.1
            ):
                self.group_tags['up_left'].append(unit.tag)
            elif (unit.pos.x > 16.3 and unit.pos.y > 7.9) or unit.pos.y > 8.6:
                self.group_tags['up_right'].append(unit.tag)
            elif (unit.pos.x > 15.6 and unit.pos.y < 7.2) or unit.pos.x > 16.3:
                self.group_tags['down_right'].append(unit.tag)
            else:
                self.group_tags['down_left'].append(unit.tag)

    def check_state(self) -> None:
        self.attack_above = [u for u in self.enemy_colossuses if u.pos.y > 11.9]
        self.attack_down = [u for u in self.enemy_colossuses if u.pos.y < 12.1]

    def switch_group(self, g1, g2, key) -> bool:
        # High-ground group targets ~2x low-ground size; rebalance lowest-y units if off by > 2.
        if len(g1) > len(g2) * 2 + 2:
            n = (len(g1) - len(g2) * 2 - 1) // 2
            switch_list = sorted(g1, key=lambda e: e.pos.y)[:n]
            for unit in switch_list:
                self.group_tags['up_' + key].remove(unit.tag)
                self.group_tags['down_' + key].append(unit.tag)
            return True
        elif len(g1) < len(g2) * 2 - 2:
            n = (len(g2) * 2 - 1 - len(g1)) // 2
            switch_list = sorted(g2, key=lambda e: e.pos.y)[:n]
            for unit in switch_list:
                self.group_tags['down_' + key].remove(unit.tag)
                self.group_tags['up_' + key].append(unit.tag)
            return True

        return False
