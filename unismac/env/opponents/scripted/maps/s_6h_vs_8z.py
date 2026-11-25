"""Scripts for `6h_vs_8z`: `6h_vs_8z_e` (attack-move to nearest Hydralisk position) and
`6h_vs_8z_h` (round-robin target split).

Ported from upstream `smac_hard/env/scripts/s_6h_vs_8z/script_easy.py` and `script_hard.py`.
Differences: `UnitTypeId.X.value` replaced by int constants; unused scipy/numpy removed.
Preserved upstream behaviors:

* `hard` indexes `enemy_hydralisk[i % len]` on step 0 before checking if Hydralisks exist,
  raising `ZeroDivisionError` if none are present;
* `hard` stores unit objects in `self.target` rather than tags; later-appearing Zealots raise
  `KeyError`.
"""

from __future__ import annotations

from typing import Any

from ..api import UnitTypeId, attack, distance_to, nearest_n_units
from ..base import Script, register


@register('6h_vs_8z_e')
class ZealotAttackNearestPoint(Script):
    """Each Zealot attack-moves to the position of its nearest Hydralisk."""

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.units = [agent for _, agent in agents.items() if agent.health != 0]
        self.enemy_units = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        self.zealots = sorted(
            [a for a in self.units if a.unit_type == UnitTypeId.ZEALOT], key=lambda a: a.tag
        )
        self.enemy_hydralisk = sorted(
            [a for a in self.enemy_units if a.unit_type == UnitTypeId.HYDRALISK],
            key=lambda a: a.tag,
        )

        if not self.units or not self.enemy_units:
            return []

        actions_list: list[Any] = []

        if not self.enemy_hydralisk:
            return []

        for zealot in self.zealots:
            target = min(self.enemy_hydralisk, key=lambda e: distance_to(e, zealot))
            actions_list.append(attack(zealot, (target.pos.x, target.pos.y), visible_matrix))

        return actions_list


@register('6h_vs_8z_h')
class ZealotSplitTarget(Script):
    """Assign Zealots to Hydralisks round-robin by tag on step 0; switch to nearest on kill."""

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)

        self.zealots: list[Any] = []
        self.ally_list = ['zealots']

        self.enemy_hydralisk: list[Any] = []
        self.enemy_list = ['enemy_hydralisk']

        self.target: dict[int, Any] = {}  # Zealot tag -> target unit object (upstream style)
        self.init = True

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        self.units = [agent for _, agent in agents.items() if agent.health != 0]
        self.enemy_units = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        self.zealots = sorted(
            [a for a in self.units if a.unit_type == UnitTypeId.ZEALOT], key=lambda a: a.tag
        )
        self.enemy_hydralisk = sorted(
            [a for a in self.enemy_units if a.unit_type == UnitTypeId.HYDRALISK],
            key=lambda a: a.tag,
        )

        if not self.units or not self.enemy_units:
            return []

        actions_list: list[Any] = []

        if self.init:
            for i, zealot in enumerate(self.zealots):
                self.target[zealot.tag] = self.enemy_hydralisk[i % len(self.enemy_hydralisk)]
            self.init = False

        if not self.enemy_hydralisk:
            return []

        for zealot in self.zealots:
            t = self.target[zealot.tag]
            if t.health == 0:
                # Target died: switch to nearest Hydralisk
                t = nearest_n_units(zealot, self.enemy_hydralisk, 1)[0]
                self.target[zealot.tag] = t
            actions_list.append(attack(zealot, t, visible_matrix))

        return actions_list
