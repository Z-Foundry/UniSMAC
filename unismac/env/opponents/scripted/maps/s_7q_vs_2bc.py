"""Scripts for `7q_vs_2bc`: `7q_vs_2bc_c` (Yamato Cannon) and `7q_vs_2bc_rc` (Yamato Cannon plus
low-HP retreat). Ported from upstream `s_7q_vs_2bc/script_cannon.py` and
`script_cannon_retreat.py`.

Deviations and preserved upstream issues:

* Upstream `script()` omitted the `visible_matrix` parameter; restored to the standard 5-argument
  signature here;
* When Yamato Cannon (401) is unavailable, upstream calls `attack(agent, weakest_enemy)` missing
  `visible_matrix`, always raising `TypeError`. Raised directly at the same site here.
"""

from __future__ import annotations

from typing import Any

from ..api import apply_ability, move
from ..base import Script, register

# Exact upstream TypeError message for missing visible_matrix in attack().
_ATTACK_MISSING_ARG = "attack() missing 1 required positional argument: 'visible_matrix'"


@register('7q_vs_2bc_c')
class YamatoCannon(Script):
    needs_abilities = True

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []
        actions: list[Any] = []

        # Upstream computes weakest_enemy only for the attack() call that raises TypeError.
        weakest_enemy = min(enemies, key=lambda e: e.health)  # noqa: F841
        for agent in agents:
            applied = False
            for abil in agent_ability:
                if abil.unit_tag == agent.tag:
                    for a in abil.abilities:
                        if a.ability_id == 401:  # Yamato Cannon
                            strongest_enemy = max(enemies, key=lambda e: e.health)
                            actions.append(apply_ability(agent, 401, strongest_enemy))
                            applied = True

                            break

            if not applied:
                # Upstream attack(agent, weakest_enemy) omits visible_matrix (preserve TypeError).
                raise TypeError(_ATTACK_MISSING_ARG)

        return actions


@register('7q_vs_2bc_rc')
class YamatoCannonRetreat(Script):
    needs_abilities = True

    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.status: dict[int, str] = {}  # agent tag -> 'Attack' / 'Retreat'
        self.start_position = (17, 3)

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []
        actions: list[Any] = []

        for agent in agents:
            if agent.health / agent.health_max >= 0.5:
                self.status[agent.tag] = 'Attack'
            else:
                self.status[agent.tag] = 'Retreat'

        # Upstream computes weakest_enemy only for the attack() call that raises TypeError.
        weakest_enemy = min(enemies, key=lambda e: e.health)  # noqa: F841
        for agent in agents:
            applied = False
            for abil in agent_ability:
                if abil.unit_tag == agent.tag:
                    for a in abil.abilities:
                        if a.ability_id == 401:  # Yamato Cannon
                            strongest_enemy = max(enemies, key=lambda e: e.health)
                            actions.append(apply_ability(agent, 401, strongest_enemy))
                            applied = True
                            break

            if not applied:
                if self.status[agent.tag] == 'Attack':
                    # Upstream attack(agent, weakest_enemy) omits visible_matrix (TypeError).
                    raise TypeError(_ATTACK_MISSING_ARG)
                else:
                    actions.append(move(agent, self.start_position))

        return actions
