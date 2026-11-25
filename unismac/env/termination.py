"""Termination logic (upstream SMAC `update_units` `game_end_code`).

* Lost: all allies dead and at least one enemy alive; or on Medivac maps (`medivac_rules`:
  SMAC MMM and SMACv2 `terran_gen`), only Medivacs remain alive on the ally side.
* Won: all enemies dead and at least one ally alive; or on Medivac maps per `enemy_medivac_rule`:
  `last_unit_is_medivac` (SMAC) when exactly 1 enemy survives and it is a Medivac;
  `no_non_medivac_left` (SMACv2) when no non-Medivac enemy remains alive.
* Draw: both sides wiped out simultaneously (upstream checks loss before draw, so simultaneous
  wipeout on MMM counts as loss).

Precedence matches upstream: lost, then won, then draw.

In two-sided mode (`opponent.kind='external'`, `game_end_dual`), both sides share the same rule:
a side is eliminated when all its units die (or no non-Medivac unit survives on Medivac maps,
using each side's own `AllyTables.is_medivac`). Only Player 2 eliminated -> WON; only Player 1
eliminated -> LOST; both eliminated on the same step -> DRAW. `enemy_medivac_rule` is unused.
"""

from __future__ import annotations

import numpy as np

from .catalog import AllyTables, EnemyTables
from .config import TerminationConfig
from .units import Frame

__all__ = ['DRAW', 'LOST', 'NONE', 'WON', 'game_end', 'game_end_dual']

NONE, WON, LOST, DRAW = 0, 1, -1, 2


def game_end(
    frame: Frame,
    ally: AllyTables,
    enemy: EnemyTables,
    config: TerminationConfig,
    medivac_rules: bool,
) -> np.ndarray:
    """Return `int8[B]` termination codes: `NONE`, `WON`, `LOST`, or `DRAW`."""
    a_alive = frame.allies.alive
    e_alive = frame.enemies.alive
    n_a = a_alive.sum(axis=1)
    n_e = e_alive.sum(axis=1)
    lost = (n_a == 0) & (n_e > 0)
    won = (n_a > 0) & (n_e == 0)
    if medivac_rules:
        a_live = frame.allies.health > 0
        lost |= (a_live & ~ally.is_medivac).sum(axis=1) == 0
        if config.enemy_medivac_rule == 'last_unit_is_medivac':
            e_live = frame.enemies.health > 0
            won |= (e_live.sum(axis=1) == 1) & (e_live & enemy.is_medivac).any(axis=1)
        elif config.enemy_medivac_rule == 'no_non_medivac_left':
            e_live = frame.enemies.health > 0
            won |= (e_live & ~enemy.is_medivac).sum(axis=1) == 0
    draw = (n_a == 0) & (n_e == 0)
    out = np.full(a_alive.shape[0], NONE, np.int8)
    out[draw] = DRAW
    out[won] = WON
    out[lost] = LOST
    return out


def _out(frame: Frame, ally: AllyTables, medivac_rules: bool) -> np.ndarray:
    """Return `bool[B]`: side eliminated (all dead, or no surviving non-Medivac on Medivac maps)."""
    if medivac_rules:
        return ((frame.allies.health > 0) & ~ally.is_medivac).sum(axis=1) == 0
    return frame.allies.alive.sum(axis=1) == 0


def game_end_dual(
    frame: Frame, ally: AllyTables, frame2: Frame, ally2: AllyTables, medivac_rules: bool
) -> np.ndarray:
    """Return `int8[B]` termination codes from Player 1's perspective in two-sided mode, given
    Player 1's (`frame`, `ally`) and Player 2's (`frame2`, `ally2`) views and ally tables."""
    out1 = _out(frame, ally, medivac_rules)
    out2 = _out(frame2, ally2, medivac_rules)
    out = np.full(out1.shape[0], NONE, np.int8)
    out[out2] = WON
    out[out1] = LOST
    out[out1 & out2] = DRAW
    return out
