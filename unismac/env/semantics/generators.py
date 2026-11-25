"""SMACv2 procedural scenario generation (ported from `distributions.py`): team compositions,
start positions, hit probabilities, and enemy masks.

Each episode draws from a single `generators` RNG stream in fixed order: `team_gen`,
`start_positions`, `attack`, `health`, `enemy_mask` (see `CapabilityConfig`), matching upstream
calls one-for-one:

* `weighted_teams`: `rng.choice(units, size=n, p=weights)` followed by shuffle (upstream uses
  stdlib `random.shuffle`; here `rng.permutation` via `unismac.env.semantics.rng.shuffle`) until
  the team is not composed solely of `exception_unit_types`; enemy team copies ally team, drawing
  `n_enemies - n_units` extra units or truncating as needed.
* `per_agent_uniform`: `rng.uniform(low, high, size=(n, len(low)))`.
* `reflect_position`: allies ~ `U((0, 0), (map_x / 2 - 1, map_y))`, enemies reflect x as
  `map_x - x`, with surplus enemies drawn from `U((map_x / 2, 0), (map_x, map_y))`.
* `surrounded`: allies spawn at map center; enemies partition via `n_groups = integers(1, 5)`,
  `multinomial`, `uniform(size=n_groups)`, and `choice(4, n_groups, replace=False)` diagonals.
* `surrounded_and_reflect`: `surrounded` when `random() > p`, else `reflect_position`.
* `mask`: `choice([0, 1], size=(n_units, n_enemies), p=[mp, 1 - mp])`.

Upstream creates an unseeded `Generator` per distribution; parity tests inject a shared
`Generator` into upstream so both draw identical values. Does not mutate the input config.

Rust counterpart (`accel='rust'`): `zgame-smac` `env::generators`, using bitwise-matched NumPy
distributions in `zgame-core` `npgen`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ...contract import ContractError
from ..config import CapabilityConfig, StartPositionsConfig, TeamGenConfig
from .catalog import (
    BANELING,
    COLOSSUS,
    HYDRALISK,
    MARAUDER,
    MARINE,
    MEDIVAC,
    STALKER,
    ZEALOT,
    ZERGLING,
)
from .rng import shuffle

__all__ = ['UNIT_NAMES', 'EpisodeDraw', 'ScenarioGenerator', 'draw_batch']

# SMACv2 unit name -> standard UnitTypeId.
UNIT_NAMES: dict[str, int] = {
    'baneling': BANELING,
    'colossus': COLOSSUS,
    'hydralisk': HYDRALISK,
    'marauder': MARAUDER,
    'marine': MARINE,
    'medivac': MEDIVAC,
    'stalker': STALKER,
    'zealot': ZEALOT,
    'zergling': ZERGLING,
}

# Fixed upstream spawn points when start_positions is disabled.
DEFAULT_ALLY_POS = (8.0, 16.0)
DEFAULT_ENEMY_POS = (24.0, 16.0)


@dataclass(frozen=True)
class EpisodeDraw:
    """Sampled attributes for one episode; disabled fields are None."""

    ally_team: tuple[str, ...] | None
    enemy_team: tuple[str, ...] | None
    ally_pos: np.ndarray | None  # float64 [A, 2] (spawn order)
    enemy_pos: np.ndarray | None  # float64 [E, 2]
    attack_prob: np.ndarray | None  # float64 [A] (indexed by agent ID)
    enemy_mask: np.ndarray | None  # int64 [A, E]; 1 hides enemy health, shield, and type


def _uniform(rng: np.random.Generator, low, high, n: int) -> np.ndarray:
    return rng.uniform(low=low, high=high, size=(n, len(low)))


class ScenarioGenerator:
    def __init__(self, config: CapabilityConfig, n_units: int, n_enemies: int) -> None:
        self.config = config
        self.n_units = n_units
        self.n_enemies = n_enemies
        tg = config.team_gen
        if tg is not None:
            unknown = [u for u in tg.unit_types if u not in UNIT_NAMES]
            if unknown or not tg.unit_types:
                raise ValueError(
                    f'capability.team_gen.unit_types contains unknown units: {unknown}'
                )
            if len(tg.weights) != len(tg.unit_types):
                raise ValueError(
                    'capability.team_gen.weights and unit_types must have equal length'
                )
            if set(tg.unit_types) <= set(tg.exception_unit_types):
                raise ValueError(
                    'All capability.team_gen unit_types are excluded; cannot sample a team'
                )
        for name in ('attack',):
            sub = getattr(config, name)
            if sub is not None and len(sub.lower_bound) != len(sub.upper_bound):
                raise ValueError(f'capability.{name} lower_bound and upper_bound lengths differ')
            if sub is not None and len(sub.lower_bound) != 1:
                raise ValueError(
                    f'capability.{name} only supports 1D bounds (one probability per agent)'
                )

    # ------------------------------------------------------------ Distributions
    def _gen_team(self, rng: np.random.Generator, tg: TeamGenConfig, n: int) -> list[str]:
        units = np.array(tg.unit_types)
        weights = np.array(tg.weights)
        team: list = []
        while not team or all(m in tg.exception_unit_types for m in team):
            team = list(rng.choice(units, size=(n,), p=weights))
            shuffle(rng, team)
        return [str(m) for m in team]

    def team(self, rng: np.random.Generator) -> tuple[list[str], list[str]]:
        tg = self.config.team_gen
        assert tg is not None
        ally = self._gen_team(rng, tg, self.n_units)
        enemy = list(ally)
        if self.n_enemies > self.n_units:
            enemy.extend(self._gen_team(rng, tg, self.n_enemies - self.n_units))
        elif self.n_enemies < self.n_units:
            enemy = enemy[: self.n_enemies]
        return ally, enemy

    def _reflect(self, rng, sp: StartPositionsConfig) -> tuple[np.ndarray, np.ndarray]:
        n, ne = self.n_units, self.n_enemies
        ally = _uniform(rng, (0, 0), (sp.map_x / 2 - 1, sp.map_y), n)
        enemy = np.zeros((ne, 2))
        if ne >= n:
            enemy[:n, 0] = sp.map_x - ally[:, 0]
            enemy[:n, 1] = ally[:, 1]
            if ne > n:
                enemy[n:, :] = _uniform(rng, (sp.map_x / 2, 0), (sp.map_x, sp.map_y), ne - n)
        else:
            enemy[:, 0] = sp.map_x - ally[:ne, 0]
            enemy[:, 1] = ally[:ne, 1]
        return ally, enemy

    def _surrounded(self, rng, sp: StartPositionsConfig) -> tuple[np.ndarray, np.ndarray]:
        off = 2
        mx, my = sp.map_x, sp.map_y
        centre = np.array([mx / 2, my / 2])
        diag_centre = {
            0: np.array([mx / 2 - off, my / 2 - off]),
            1: np.array([mx / 2 - off, my / 2 + off]),
            2: np.array([mx / 2 + off, my / 2 + off]),
            3: np.array([mx / 2 + off, my / 2 - off]),
        }
        corner = {
            0: np.array([0, 0]),
            1: np.array([0, my]),
            2: np.array([mx, my]),
            3: np.array([mx, 0]),
        }
        ally = np.tile(centre, (self.n_units, 1))
        enemy = np.zeros((self.n_enemies, 2))
        n_groups = rng.integers(1, 5)
        membership = rng.multinomial(self.n_enemies, np.ones(n_groups) / n_groups)
        group_pos = rng.uniform(size=(n_groups,))
        diagonals = rng.choice(np.array(range(4)), size=(n_groups,), replace=False)
        i = 0
        for g in range(n_groups):
            t = group_pos[g]
            d = int(diagonals[g])
            enemy[i : i + membership[g], :] = diag_centre[d] * t + corner[d] * (1 - t)
            i += int(membership[g])
        return ally, enemy

    def positions(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        sp = self.config.start_positions
        assert sp is not None
        if sp.dist_type == 'surrounded':
            return self._surrounded(rng, sp)
        if sp.dist_type == 'reflect_position':
            return self._reflect(rng, sp)
        if rng.random() > sp.p:
            return self._surrounded(rng, sp)
        return self._reflect(rng, sp)

    # ------------------------------------------------------------ Single episode
    def draw(self, rng: np.random.Generator) -> EpisodeDraw:
        c = self.config
        ally_team = enemy_team = None
        ally_pos = enemy_pos = attack = mask = None
        if c.team_gen is not None:
            a, e = self.team(rng)
            ally_team, enemy_team = tuple(a), tuple(e)
        if c.start_positions is not None:
            ally_pos, enemy_pos = self.positions(rng)
        if c.attack is not None:
            attack = _uniform(rng, c.attack.lower_bound, c.attack.upper_bound, self.n_units)[:, 0]
        if c.health is not None:
            raise ContractError(
                'capability.health is not implemented (contract lacks a kill primitive)'
            )
        if c.enemy_mask is not None:
            mp = c.enemy_mask.mask_probability
            mask = rng.choice([0, 1], size=(self.n_units, self.n_enemies), p=[mp, 1.0 - mp])
        return EpisodeDraw(ally_team, enemy_team, ally_pos, enemy_pos, attack, mask)


def draw_batch(
    generator: ScenarioGenerator,
    streams: Callable[[int], np.random.Generator],
    env_ids: Sequence[int],
) -> list[EpisodeDraw]:
    """Sample one `EpisodeDraw` per env in `env_ids`, using `streams(b)` (the per-env
    `generators` stream from `VecEnv.rng_streams`)."""
    return [generator.draw(streams(int(b))) for b in env_ids]
