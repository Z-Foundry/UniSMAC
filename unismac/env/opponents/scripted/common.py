"""Shared opponent scripts across scenarios: `base` (attack-move to fixed point), `atkw` (focus
weakest), `atkn` (attack nearest), and `noop`. Ported from upstream `base_script.py`,
`attack_weakest.py`, `attack_nearest.py`, and `noop.py`."""

from __future__ import annotations

from typing import Any

from .api import UnitTypeId, attack, center, distance_to, find_by_tag, move
from .base import Script, register


@register('base')
class BaseScript(Script):
    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        actions: list[Any] = []
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]
        if not agents or not enemies:
            return []
        if iteration < 5:
            return []
        m = self.map_name
        if m in ['3st_vs_5zl', '3rp_vs_5zl', '2c_vs_64zg']:
            for a in agents:
                actions.append(attack(a, (16, 16), visible_matrix))
        if m in ['6m_vs_10m', '3hl_vs_24zl', '3rp_vs_24zl']:
            for a in agents:
                actions.append(attack(a, (6, 16), visible_matrix))
        if m in ['7q_vs_2bc']:
            for a in agents:
                actions.append(attack(a, (16, 21), visible_matrix))
        if m in ['2vr_vs_3sc']:
            actions = []  # Spore Crawlers rely on engine auto-attack
        if m in ['mmmt', 'mmmt_vs_zhb', 'mmmt_vs_zspi']:
            for a in agents:
                if a.unit_type in [UnitTypeId.MEDIVAC]:
                    actions.append(move(a, center(agents)))
                else:
                    actions.append(attack(a, (3, 16), visible_matrix))
        if m in [
            '3m',
            '8m',
            '5m_vs_6m',
            '8m_vs_9m',
            '10m_vs_11m',
            '25m',
            '27m_vs_30m',
            '2s3z',
            '3s5z',
            '3s5z_vs_3s6z',
            '1c3s5z',
            'MMM',
            'MMM2',
            '3s_vs_3z',
            '3s_vs_4z',
            '3s_vs_5z',
        ]:
            for a in agents:
                if a.unit_type in [UnitTypeId.MEDIVAC]:
                    actions.append(move(a, center(agents)))
                else:
                    actions.append(attack(a, (9, 16), visible_matrix))
        if m in ['6h_vs_8z']:
            for a in agents:
                actions.append(attack(a, (14.5, 9.5), visible_matrix))
        if m in ['corridor']:
            for a in agents:
                actions.append(attack(a, (4.5, 4), visible_matrix))
        return actions


@register('atkw')
class AttackWeakest(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.target_dict: dict[int, int] = {}  # agent tag -> enemy tag

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        actions: list[Any] = []
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]
        if not agents or not enemies:
            return []
        if iteration < 10:
            return []
        weakest_enemy = min(
            enemies, key=lambda e: (e.health + e.shield) / (e.health_max + e.shield_max)
        )
        weakest_agent = min(agents, key=lambda e: e.health / e.health_max)
        for agent in agents:
            if agent.unit_type == UnitTypeId.MEDIVAC:
                actions.append(move(agent, (weakest_agent.pos.x, weakest_agent.pos.y)))
            else:
                target_tag = self.target_dict.get(agent.tag, None)
                target = find_by_tag(enemies, target_tag)
                if target is None or target.health == 0:
                    self.target_dict[agent.tag] = weakest_enemy.tag
                    target = weakest_enemy
                actions.append(attack(agent, target, visible_matrix))
        return actions


@register('atkn')
class AttackNearest(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.target_dict: dict[int, int] = {}  # agent tag -> enemy tag

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        actions: list[Any] = []
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]
        if not agents or not enemies:
            return []
        if iteration < 10:
            return []
        weakest_agent = min(agents, key=lambda e: e.health / e.health_max)
        for agent in agents:
            if agent.unit_type == UnitTypeId.MEDIVAC:
                actions.append(move(agent, (weakest_agent.pos.x + 2, weakest_agent.pos.y)))
            else:
                target_tag = self.target_dict.get(agent.tag, None)
                target = find_by_tag(enemies, target_tag)
                if target is None or target.health == 0:
                    nearest_enemy = min(enemies, key=lambda e: distance_to(agent, e))
                    self.target_dict[agent.tag] = nearest_enemy.tag
                    target_tag = nearest_enemy.tag
                actions.append(attack(agent, find_by_tag(enemies, target_tag), visible_matrix))
        return actions


@register('noop')
class Noop(Script):
    """Upstream `noop.py` (debugging only, not in any script pool): issues no commands."""

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        return []
