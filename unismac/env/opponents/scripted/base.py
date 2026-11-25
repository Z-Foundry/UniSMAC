"""Script base class and registry.

Upstream scripts implement `DecisionTreeScript(map_name)` with
`script(agents, enemies, agent_ability, visible_matrix, iteration)` returning a list of actions.
This port preserves that signature and semantics:

* `agents` / `enemies`: upstream `self.enemies` / `self.agents` dicts (index -> unit), i.e. the
  script's own (Player 2) units and opponent (Player 1) units sorted by `(unit_type, x, y)`.
  Dead units remain in the dict with `health == 0`.
* `iteration`: upstream `_episode_steps` (step count before executing this step, starting at 0).
* Upstream `random.choice` is replaced by `self.random.choice`, injected by `ScriptedOpponent`
  from the `opponent` RNG stream (see `unismac.env.opponents.scripted.opponent`).

The two upstream `7q_vs_2bc` scripts omitted `visible_matrix` from their signature (causing an
immediate `TypeError`); the standard 5-argument signature is restored here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

__all__ = ['POOLS', 'SCRIPTS', 'ChoiceSource', 'Script', 'register']


class ChoiceSource(Protocol):
    def choice(self, seq: Sequence[Any]) -> Any: ...


class _NoRandom:
    def choice(self, seq: Sequence[Any]) -> Any:
        raise RuntimeError(
            'Script random source is not injected (ScriptedOpponent sets script.random)'
        )


class Script:
    """Base class for upstream `DecisionTreeScript`; subclasses implement `script()`."""

    # Whether script() reads agent_ability (False skips backend ability queries).
    needs_abilities = False

    def __init__(self, map_name: str) -> None:
        self.map_name = map_name
        self.random: ChoiceSource = _NoRandom()

    def script(
        self,
        agents: dict[int, Any],
        enemies: dict[int, Any],
        agent_ability: list[Any],
        visible_matrix: dict[str, list[int]],
        iteration: int,
    ) -> list[Any]:
        raise NotImplementedError


# Script name -> class (matches upstream scripts/__init__.py aliases without the DTS_ prefix).
SCRIPTS: dict[str, type[Script]] = {}


def register(name: str) -> Callable[[type[Script]], type[Script]]:
    def deco(cls: type[Script]) -> type[Script]:
        if name in SCRIPTS:
            raise ValueError(f'duplicate registration for script {name!r}')
        SCRIPTS[name] = cls
        return cls

    return deco


# Upstream SCRIPT_DICT: scenario -> script pool (duplicates act as sampling weights, e.g.
# 2c_vs_64zg). Scenarios without an entry (2m_vs_1z, 2s_vs_1sc, so_many_baneling, bane_vs_bane,
# unit_test, pvt_large) cannot use scripted opponents.
_STD = ('base', 'atkw', 'atkn')
POOLS: dict[str, tuple[str, ...]] = {
    '3m': (*_STD, '8m', '8m_1'),
    '8m': (*_STD, '8m', '8m_1'),
    '5m_vs_6m': (*_STD, '8m', '8m_1'),
    '8m_vs_9m': (*_STD, '8m', '8m_1'),
    '10m_vs_11m': (*_STD, '8m', '8m_1'),
    '25m': (*_STD, '27m_a', '27m_ap'),
    '27m_vs_30m': (*_STD, '27m_a', '27m_ap'),
    '2s3z': (*_STD, '2s3z', '2s3z_1'),
    '3s5z': (*_STD, '3s5z', '3s5z_1'),
    '3s5z_vs_3s6z': (*_STD, '3s5z', '3s5z_1'),
    '1c3s5z': (*_STD, '1c3s5z_1', '1c3s5z_m'),
    'MMM': (*_STD, 'MMM', 'MMM_2'),
    'MMM2': (*_STD, 'MMM', 'MMM_2'),
    '3s_vs_3z': (*_STD, '3s_vs_3z_a', '3s_vs_3z_g'),
    '3s_vs_4z': (*_STD, '3s_vs_4z_a', '3s_vs_4z_g'),
    '3s_vs_5z': (*_STD, '3s_vs_5z_a', '3s_vs_5z_g'),
    '2c_vs_64zg': (*_STD, '2c_vs_64zg', '2c_vs_64zg'),
    '6h_vs_8z': (*_STD, '6h_vs_8z_e', '6h_vs_8z_h'),
    'corridor': (*_STD, 'corridor_d', 'corridor_y'),
    '3st_vs_5zl': (*_STD, '3st_vs_5zl_a', '3st_vs_5zl_g'),
    '7q_vs_2bc': (*_STD, '7q_vs_2bc_c', '7q_vs_2bc_rc'),
    '6m_vs_10m': (*_STD, '6m_vs_10m_fatkw', '6m_vs_10m_fatkn'),
    '2vr_vs_3sc': (*_STD, '2vr_vs_3sc_watkw', '2vr_vs_3sc_watkn'),
    '3hl_vs_24zl': (*_STD, '3hl_vs_24zl_aw', '3hl_vs_24zl_ag'),
    '3rp_vs_24zl': (*_STD, '3hl_vs_24zl_aw', '3hl_vs_24zl_ag'),
    '3rp_vs_5zl': (*_STD, '3rp_vs_5zl_g', '3rp_vs_5zl_c'),
    'mmmt': (*_STD, 'mmmt_fap', 'mmmt_fan'),
    'mmmt_vs_zhb': (*_STD, 'mmmt_zhb_spe', 'mmmt_zhb_sur'),
    'mmmt_vs_zspi': (*_STD, 'mmmt_zspi_spe', 'mmmt_zspi_bio'),
}
