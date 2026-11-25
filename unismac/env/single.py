"""Single-environment SMAC-compatible wrapper (`UnismacEnv`, `batch_size=1`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .core import VecEnv
    from .semantics.features import ObsLayout, StateLayout

__all__ = ['UnismacEnv']


class UnismacEnv:
    """Upstream `MultiAgentEnv` wrapper (`batch_size=1`), matching SMAC `StarCraft2Env`:

    * `reset() -> (obs: list[float32[D]], state: float32[S])`
    * `step(actions) -> (reward: float, terminated: bool, info: dict)`
    * `get_obs() -> list[float32[D]]`, `get_avail_actions() -> list[list[int]]`, etc.

    Supports context-manager usage (``with make_env(...) as env:``) and exposes
    read-only inspection properties (`accel`, `dual`, `mirror`, `obs_layout`,
    `state_layout`, `n_agents_for`, `n_actions_for`, `alive`) directly without
    reaching into `env.vec`.

    Call `reset()` before stepping again after termination (stepping while done returns
    `0, True, {}`).
    """

    def __init__(self, vec: VecEnv) -> None:
        if vec.batch_size != 1:
            raise ValueError('UnismacEnv only wraps a VecEnv with batch_size=1')
        self.vec = vec
        self.n_agents = vec.n_agents
        self.n_enemies = vec.n_enemies
        self.episode_limit = vec.episode_limit
        self.n_actions = vec.n_actions
        self.config = vec.config

    def __enter__(self) -> UnismacEnv:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        vec = self.vec
        extras = ''
        if vec.dual:
            extras += ' dual=True'
            if vec.mirror is not None:
                extras += f' mirror={vec.mirror!r}'
        return (
            f'<UnismacEnv {vec.spec.benchmark}/{vec.spec.name} '
            f'backend={vec.backend.capabilities.name!r} accel={vec.accel!r} '
            f'n_agents={self.n_agents} n_actions={self.n_actions}{extras}>'
        )

    @property
    def accel(self) -> str:
        """Resolved environment-layer acceleration tier (`'python'`, `'services'`, or `'rust'`)."""
        return self.vec.accel

    @property
    def dual(self) -> bool:
        """True when running in two-sided mode (`opponent.kind='external'`)."""
        return self.vec.dual

    @property
    def mirror(self) -> Literal['x', 'y'] | None:
        """Active mirror-spawn axis in two-sided mode (`'x'`, `'y'`, or `None`)."""
        return self.vec.mirror

    @property
    def obs_layout(self) -> ObsLayout:
        """Observation vector layout for Player 1."""
        return self.vec.obs_layout

    @property
    def state_layout(self) -> StateLayout:
        """Global state vector layout for Player 1."""
        return self.vec.state_layout

    def n_agents_for(self, side: int) -> int:
        """Return agent count for `side` (1 or 2 in two-sided mode)."""
        return self.vec.n_agents_for(side)

    def n_actions_for(self, side: int) -> int:
        """Return action count for `side` (1 or 2 in two-sided mode)."""
        return self.vec.n_actions_for(side)

    def alive(self, side: int = 1) -> np.ndarray:
        """Return `bool[A]` (fresh 1D array) of alive agents for `side`."""
        return self.vec.alive(side)[0]

    def reset(self) -> tuple[list[np.ndarray], np.ndarray]:
        self.vec.reset()
        return self.get_obs(), self.get_state()

    def step(self, actions: Any, enemy_actions: Any = None) -> tuple[float, bool, dict[str, Any]]:
        """Step the environment; `enemy_actions` provides Player 2 actions in two-sided mode."""
        acts = np.array([int(a) for a in actions], dtype=np.int64)[None]
        if enemy_actions is None:
            reward, terminated, infos = self.vec.step(acts)
        else:
            acts2 = np.array([int(a) for a in enemy_actions], dtype=np.int64)[None]
            reward, terminated, infos = self.vec.step(acts, acts2)
        return float(reward[0]), bool(terminated[0]), infos[0]

    def get_obs(self, side: int = 1) -> list[np.ndarray]:
        return list(self.vec.get_obs(side)[0])

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        return self.vec.get_obs()[0, agent_id]

    def get_state(self, side: int = 1) -> np.ndarray:
        return self.vec.get_state(side)[0]

    def get_avail_actions(self, side: int = 1) -> list[list[int]]:
        return self.vec.get_avail_actions(side)[0].tolist()

    def get_avail_agent_actions(self, agent_id: int) -> list[int]:
        return self.vec.get_avail_actions()[0, agent_id].tolist()

    def get_obs_size(self) -> int:
        return self.vec.get_obs_size()

    def get_state_size(self) -> int:
        return self.vec.get_state_size()

    def get_total_actions(self) -> int:
        return self.vec.get_total_actions()

    def get_env_info(self) -> dict[str, Any]:
        return self.vec.get_env_info()

    def get_cap_size(self) -> int:
        return self.vec.get_cap_size()

    def get_capabilities(self) -> np.ndarray:
        return self.vec.get_capabilities()[0]

    @property
    def max_reward(self) -> float | None:
        return self.vec.max_reward

    def get_stats(self) -> dict[str, Any]:
        return self.vec.get_stats()

    def seed(self) -> int | None:
        return self.vec.seed

    def close(self) -> None:
        self.vec.close()

    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError('Rendering is not supported in unismac')

    def save_replay(self) -> None:
        raise NotImplementedError('Saving replays is not supported in unismac')
