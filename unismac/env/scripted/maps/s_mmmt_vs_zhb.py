"""Scripts for `mmmt_vs_zhb`: `mmmt_zhb_spe` (unit-counter target assignment) and `mmmt_zhb_sur`
(Zergling two-flank surround and Baneling backdoor). Ported from upstream
`s_MMMT_vs_ZHB/script_spe_target.py` and `script_surr.py`.

Deviation: upstream uses bare `return` (None) when no alive units exist; returning `[]` is
equivalent.
"""

from __future__ import annotations

import math
from typing import Any

from ..api import UnitTypeId, attack, center, distance_to, move, toward
from ..base import Script, register


@register('mmmt_zhb_spe')
class SpecialTarget(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True
        self.center = None

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []  # Upstream bare return (None)

        self.banelings = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.BANELING], key=lambda a: a.tag
        )
        self.hydralisks = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.HYDRALISK], key=lambda a: a.tag
        )
        self.zerglings = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.ZERGLING], key=lambda a: a.tag
        )

        self.enemy_marines = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MARINE], key=lambda a: a.tag
        )
        self.enemy_marauders = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MARAUDER], key=lambda a: a.tag
        )
        self.enemy_medivacs = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MEDIVAC], key=lambda a: a.tag
        )
        self.enemy_tanks = sorted(
            [
                a
                for a in enemies
                if a.unit_type == UnitTypeId.SIEGETANK or a.unit_type == UnitTypeId.SIEGETANKSIEGED
            ],
            key=lambda a: a.tag,
        )

        self.actions_list: list[Any] = []

        # Spread out at opening to mitigate tank splash damage
        if iteration < 10:
            self.spread_out_units(self.zerglings, 2)
            self.spread_out_units(self.hydralisks, 2)
            self.spread_out_units(self.banelings, 2)

        # Banelings -> Marines
        for baneling in self.banelings:
            if self.enemy_marines:
                target = min(self.enemy_marines, key=lambda e: distance_to(e, baneling))
                self.actions_list.append(attack(baneling, target, visible_matrix))

        # Zerglings -> Marauders (fallback: nearest enemy)
        for zergling in self.zerglings:
            if self.enemy_marauders:
                target = min(self.enemy_marauders, key=lambda e: distance_to(zergling, e))

            else:
                target = min(enemies, key=lambda e: distance_to(zergling, e))

            self.actions_list.append(attack(zergling, target, visible_matrix))

        # Hydralisks -> Tanks, then Medivacs, then remaining enemies
        for hydralisk in self.hydralisks:
            if self.enemy_tanks:
                target = min(self.enemy_tanks, key=lambda e: distance_to(hydralisk, e))
            elif self.enemy_medivacs:
                target = min(self.enemy_medivacs, key=lambda e: distance_to(hydralisk, e))
            elif enemies:
                target = min(enemies, key=lambda e: distance_to(hydralisk, e))

            # enemies is non-empty, so one branch above always matches (no else, matching upstream).
            self.actions_list.append(
                attack(hydralisk, target, visible_matrix)  # pyrefly: ignore[unbound-name]
            )

        return [a for a in self.actions_list if a is not None]

    def spread_out_units(self, units: list[Any], radius: float = 2.0) -> None:
        for i, unit in enumerate(units):
            angle = i * (2 * math.pi / len(units))
            offset = (math.cos(angle) * radius, math.sin(angle) * radius)
            self.actions_list.append(move(unit, (unit.pos.x + offset[0], unit.pos.y + offset[1])))


@register('mmmt_zhb_sur')
class Surround(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True
        self.center = None

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []  # Upstream bare return (None)

        self.banelings = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.BANELING], key=lambda a: a.tag
        )
        self.hydralisks = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.HYDRALISK], key=lambda a: a.tag
        )
        self.zerglings = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.ZERGLING], key=lambda a: a.tag
        )

        self.enemy_marines = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MARINE], key=lambda a: a.tag
        )
        self.enemy_marauders = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MARAUDER], key=lambda a: a.tag
        )
        self.enemy_medivacs = sorted(
            [a for a in enemies if a.unit_type == UnitTypeId.MEDIVAC], key=lambda a: a.tag
        )
        self.enemy_tanks = sorted(
            [
                a
                for a in enemies
                if a.unit_type == UnitTypeId.SIEGETANK or a.unit_type == UnitTypeId.SIEGETANKSIEGED
            ],
            key=lambda a: a.tag,
        )

        self.actions_list: list[Any] = []

        # Surround around the enemy centroid
        if enemies:
            enemy_center = center(enemies)

            # Zerglings: split into two flanking groups
            left_flank = self.zerglings[: len(self.zerglings) // 2]
            right_flank = self.zerglings[len(self.zerglings) // 2 :]

            for i, lz in enumerate(left_flank):
                offset = (-3, 2 + i % 3)
                self.actions_list.append(
                    attack(
                        lz,
                        (enemy_center[0] + offset[0], enemy_center[1] + offset[1]),
                        visible_matrix,
                    )
                )

            for i, rz in enumerate(right_flank):
                offset = (3, -2 - i % 3)
                self.actions_list.append(
                    attack(
                        rz,
                        (enemy_center[0] + offset[0], enemy_center[1] + offset[1]),
                        visible_matrix,
                    )
                )

            # Banelings: approach from behind the target
            for baneling in self.banelings:
                if self.enemy_marines:
                    target = min(self.enemy_marines, key=lambda e: distance_to(baneling, e))
                else:
                    target = min(enemies, key=lambda e: distance_to(baneling, e))
                # Target point 5 tiles behind target away from (29, 16)
                backdoor = toward(target, (29, 16), -5)
                self.actions_list.append(attack(baneling, backdoor, visible_matrix))

            # Hydralisks: Tanks, then Medivacs, then nearest enemy
            for hydralisk in self.hydralisks:
                if self.enemy_tanks:
                    target = min(self.enemy_tanks, key=lambda e: distance_to(e, hydralisk))
                elif self.enemy_medivacs:
                    target = min(self.enemy_medivacs, key=lambda e: distance_to(e, hydralisk))
                elif enemies:
                    target = min(enemies, key=lambda e: distance_to(e, hydralisk))

                self.actions_list.append(attack(hydralisk, target, visible_matrix))

        return [a for a in self.actions_list if a is not None]
