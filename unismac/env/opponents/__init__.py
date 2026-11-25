"""Player-2 opponent policies.

* `EngineAI`: Built-in StarCraft II AI (`ComputerPlayer`), available on backends
  declaring `engine_ai`.
* `OpeningOrders`: Backend-agnostic approximation where player 2 is an
  `AgentPlayer` issuing a single opening `ATTACK_MOVE` toward the player-1
  spawn centroid on step 0.
* `NativeSMACOpponent` (`unismac.env.opponents.smac_bot`): Reproduces observed opening
  orders and Medivac escort healing across all 23 classic SMAC maps.
* `ScriptedOpponent` (`unismac.env.opponents.scripted`): Decision-tree scripted opponent
  pool ported from SMAC-Hard.
* `ExternalOpponent`: Two-sided / self-play mode (`opponent.kind='external'`)
  where player 2 is driven directly by `VecEnv.step(actions, enemy_actions)`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ...contract import AgentPlayer, Command, CommandKind, ComputerPlayer, Player, WorldState
from ..config import OpponentConfig

if TYPE_CHECKING:
    from ..core import VecEnv

__all__ = [
    'MULTI_LEVEL_SMAC_MAPS',
    'SMAC_TRIGGER_SPECS',
    'EngineAI',
    'ExternalOpponent',
    'NativeSMACOpponent',
    'OpeningOrders',
    'Opponent',
    'ScriptedOpponent',
    'make_opponent',
    'native_smac_overrides',
    'script_pool',
]


class Opponent:
    """Base interface for player-2 opponent controllers."""

    def player(self, race: str) -> Player:
        raise NotImplementedError

    def bind(self, env: VecEnv) -> None:
        """Called once after `VecEnv` construction."""

    def check_names(self, names: Sequence[str | None]) -> None:
        """Validate per-env opponent names passed to `VecEnv.reset(opponents=...)`."""
        given = sorted({n for n in names if n is not None})
        if given:
            raise ValueError(
                f'{type(self).__name__} is not a scripted opponent '
                f"(opponent.kind != 'scripted'); cannot select scripts {given}"
            )

    def on_reset(
        self,
        world: WorldState,
        env_ids: Sequence[int],
        names: Sequence[str | None] | None = None,
    ) -> None:
        """Called after `reset` for each reset environment index."""

    def issue(self, world: WorldState, command: Command, active: np.ndarray) -> None:
        """Write player-2 commands in-place before stepping the backend."""


class EngineAI(Opponent):
    """Built-in SC2 engine AI opponent."""

    def __init__(self, difficulty: str) -> None:
        self.difficulty = difficulty

    def player(self, race: str) -> Player:
        return ComputerPlayer(race, self.difficulty)  # type: ignore[arg-type]


class OpeningOrders(Opponent):
    """Issue one opening `ATTACK_MOVE` toward the player-1 centroid on step 0."""

    def __init__(self) -> None:
        self.target: np.ndarray | None = None  # float32[B, 2]
        self.pending: np.ndarray | None = None  # bool[B]

    def player(self, race: str) -> Player:
        return AgentPlayer(race)  # type: ignore[arg-type]

    def on_reset(
        self,
        world: WorldState,
        env_ids: Sequence[int],
        names: Sequence[str | None] | None = None,
    ) -> None:
        b_size = world.batch_size
        if self.target is None or self.target.shape[0] != b_size:
            self.target = np.zeros((b_size, 2), np.float32)
            self.pending = np.zeros(b_size, bool)
        assert self.pending is not None
        for b in env_ids:
            mine = world.alive[b] & (world.owner[b] == 1)
            self.target[b] = (world.x[b, mine].mean(), world.y[b, mine].mean())
            self.pending[b] = True

    def issue(self, world: WorldState, command: Command, active: np.ndarray) -> None:
        assert self.target is not None and self.pending is not None
        for b in np.flatnonzero(self.pending & active):
            for s in np.flatnonzero(world.alive[b] & (world.owner[b] == 2)):
                command.set(
                    int(b),
                    int(s),
                    CommandKind.ATTACK_MOVE,
                    target_xy=(float(self.target[b, 0]), float(self.target[b, 1])),
                )
            self.pending[b] = False


class ExternalOpponent(Opponent):
    """External player-2 controller for two-sided / self-play training."""

    def player(self, race: str) -> Player:
        return AgentPlayer(race)  # type: ignore[arg-type]

    def check_names(self, names: Sequence[str | None]) -> None:
        given = sorted({n for n in names if n is not None})
        if given:
            raise ValueError(
                'In two-sided mode (opponent.kind=external), player 2 is controlled by '
                f'external actions; cannot select per-env scripts {given}'
            )


def make_opponent(config: OpponentConfig) -> Opponent:
    """Instantiate an `Opponent` from `OpponentConfig`."""
    if config.kind == 'engine_ai':
        return EngineAI(config.difficulty)
    if config.kind == 'external':
        return ExternalOpponent()
    if config.kind == 'native_smac':
        from .smac_bot import NativeSMACOpponent

        return NativeSMACOpponent()
    if config.kind == 'scripted':
        from .scripted import ScriptedOpponent

        return ScriptedOpponent(config)
    return OpeningOrders()


from .scripted import ScriptedOpponent, script_pool  # noqa: E402
from .smac_bot import (  # noqa: E402
    MULTI_LEVEL_SMAC_MAPS,
    SMAC_TRIGGER_SPECS,
    NativeSMACOpponent,
    native_smac_overrides,
)
