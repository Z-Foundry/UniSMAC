"""Scripts for `mmmt_vs_zspi`: `mmmt_zspi_spe` (Phoenix lifts Tanks, counter-based targeting,
low-HP Blink) and `mmmt_zspi_bio` (per-type priority targeting of bio / Tanks / Medivacs). Ported
from upstream `s_MMMT_vs_ZSPI/script_spe_target.py` and `script_atk_bio.py`.

Deviation: upstream uses bare `return` (None) when no alive units exist; returning `[]` is
equivalent.

Preserved upstream behavior: `targets.remove(...)` in `script_spe_target` mutates
`self.enemy_tanks` through aliasing after a Phoenix lifts a Tank.
"""

from __future__ import annotations

from typing import Any

from ..api import UnitTypeId, apply_ability, attack, center, distance_to, move, toward
from ..base import Script, register


@register('mmmt_zspi_spe')
class SpecialTarget(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True
        self.center = None
        self.start_position = (19, 16)
        self.enemy_position = (3, 16)

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []  # Upstream bare return (None)

        self.zealots = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.ZEALOT], key=lambda a: a.tag
        )
        self.stalkers = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.STALKER], key=lambda a: a.tag
        )
        self.immortals = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.IMMORTAL], key=lambda a: a.tag
        )
        self.phoenixes = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.PHOENIX], key=lambda a: a.tag
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

        # Phoenixes: prioritize lifting Siege Tanks
        if self.phoenixes:
            targets = self.enemy_tanks

            for phoenix in self.phoenixes:
                if targets:
                    closest_target = min(targets, key=lambda t: distance_to(t, phoenix))
                    if distance_to(phoenix, closest_target) < 10:
                        # 173: Graviton Beam
                        self.actions_list.append(apply_ability(phoenix, 173, closest_target))
                        targets.remove(closest_target)
                    else:
                        self.actions_list.append(attack(phoenix, closest_target, visible_matrix))
                else:
                    # Fallback to nearest enemy when no high-priority targets remain
                    if enemies:
                        ph_target = min(enemies, key=lambda e: distance_to(e, phoenix))
                        self.actions_list.append(attack(phoenix, ph_target, visible_matrix))

        # Zealots: charge into Marines to tank damage
        if self.zealots:
            if self.enemy_marines:
                for zealot in self.zealots:
                    self.actions_list.append(
                        attack(zealot, center(self.enemy_marines), visible_matrix)
                    )
            else:
                for zealot in self.zealots:
                    self.actions_list.append(attack(zealot, center(enemies), visible_matrix))

        # Immortals: Marauders, then Medivacs, then nearest
        if self.immortals:
            if self.enemy_marauders:
                for immortal in self.immortals:
                    im_target = min(self.enemy_marauders, key=lambda e: distance_to(e, immortal))
                    self.actions_list.append(attack(immortal, im_target, visible_matrix))
            elif self.enemy_medivacs:
                for immortal in self.immortals:
                    im_target = min(self.enemy_medivacs, key=lambda e: distance_to(e, immortal))
                    self.actions_list.append(attack(immortal, im_target, visible_matrix))
            else:
                for immortal in self.immortals:
                    im_target = min(enemies, key=lambda e: distance_to(e, immortal))
                    self.actions_list.append(attack(immortal, im_target, visible_matrix))

        # Stalkers: Marauders / Medivacs, Blink away at low HP
        if self.stalkers:
            if self.enemy_marauders:
                for stalker in self.stalkers:
                    im_target = min(self.enemy_marauders, key=lambda e: distance_to(e, stalker))
                    self.actions_list.append(attack(stalker, im_target, visible_matrix))
            elif self.enemy_medivacs:
                for stalker in self.stalkers:
                    im_target = min(self.enemy_medivacs, key=lambda e: distance_to(e, stalker))
                    self.actions_list.append(attack(stalker, im_target, visible_matrix))
            else:
                for stalker in self.stalkers:
                    im_target = min(enemies, key=lambda e: distance_to(e, stalker))
                    self.actions_list.append(attack(stalker, im_target, visible_matrix))

            # Blink away when below half HP
            for stalker in self.stalkers:
                if stalker.health / stalker.health_max < 0.5:
                    retreat_pos = toward(self.start_position, stalker, -5)
                    # 3687: Blink
                    self.actions_list.append(apply_ability(stalker, 3687, retreat_pos))

        # Attack-move when no enemies are visible (unreachable since enemies is non-empty)
        if not enemies:
            for unit in agents:
                self.actions_list.append(attack(unit, self.enemy_position, visible_matrix))

        return self.actions_list


@register('mmmt_zspi_bio')
class AttackBio(Script):
    def __init__(self, map_name: str) -> None:
        super().__init__(map_name)
        self.init = True
        self.engage = True
        self.center = None
        self.start_position = (19, 16)
        self.enemy_position = (3, 16)

    def script(self, agents, enemies, agent_ability, visible_matrix, iteration) -> list[Any]:
        agents = [agent for _, agent in agents.items() if agent.health != 0]
        enemies = [enemy for _, enemy in enemies.items() if enemy.health != 0]

        if not agents or not enemies:
            return []  # Upstream bare return (None)

        self.zealots = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.ZEALOT], key=lambda a: a.tag
        )
        self.stalkers = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.STALKER], key=lambda a: a.tag
        )
        self.immortals = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.IMMORTAL], key=lambda a: a.tag
        )
        self.phoenixes = sorted(
            [a for a in agents if a.unit_type == UnitTypeId.PHOENIX], key=lambda a: a.tag
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

        for phoenix in self.phoenixes:
            if phoenix.energy >= 50 and self.enemy_tanks:
                # 173: Graviton Beam
                target = min(self.enemy_tanks, key=lambda t: distance_to(t, phoenix))
                if distance_to(phoenix, target) < 10:
                    self.actions_list.append(apply_ability(phoenix, 173, target))
                else:
                    self.actions_list.append(move(phoenix, target))
            else:
                target = min(enemies, key=lambda m: distance_to(phoenix, m))
                self.actions_list.append(attack(phoenix, target, visible_matrix))

        bio = self.enemy_marauders + self.enemy_marines
        for zealot in self.zealots:
            if bio:
                target = min(bio, key=lambda b: distance_to(b, zealot))
                self.actions_list.append(attack(zealot, target, visible_matrix))

        targets = self.enemy_marauders + self.enemy_tanks
        for immortal in self.immortals:
            if targets:
                target = min(targets, key=lambda t: distance_to(t, immortal))
                self.actions_list.append(attack(immortal, target, visible_matrix))

        medivacs = self.enemy_medivacs
        for stalker in self.stalkers:
            if medivacs:
                target = min(medivacs, key=lambda m: distance_to(stalker, m))
            elif bio:
                target = min(bio, key=lambda b: distance_to(stalker, b))
            else:
                continue

            self.actions_list.append(attack(stalker, target, visible_matrix))

        return self.actions_list
