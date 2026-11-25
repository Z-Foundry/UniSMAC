"""PyMARL / upstream SMAC `MultiAgentEnv` adapter (`batch_size=1`, zero extra dependencies)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..env import VecEnv, make_vec_env
from ._common import require_single_env

if TYPE_CHECKING:
    from ..contract import Backend
    from ..env.semantics.features import ObsLayout, StateLayout

__all__ = ['PyMARLEnv', 'UnismacEnv', 'make_pymarl_env']


class PyMARLEnv:
    """Upstream SMAC / PyMARL `MultiAgentEnv` wrapper around a `batch_size=1` `VecEnv`.

    * `reset() -> (obs: list[float32[D]], state: float32[S])`
    * `step(actions, enemy_actions=None) -> (reward: float, terminated: bool, info: dict)`
    * `get_obs()`, `get_state()`, `get_avail_actions()`, `get_env_info()`, `get_stats()`, etc.

    Stepping after episode termination (before calling `reset()`) returns `(0.0, True, {})`,
    matching upstream SMAC `StarCraft2Env`.
    """

    def __init__(self, vec: VecEnv) -> None:
        require_single_env(vec, 'PyMARLEnv')
        self.vec = vec
        self.n_agents = vec.n_agents
        self.n_enemies = vec.n_enemies
        self.episode_limit = vec.episode_limit
        self.n_actions = vec.n_actions
        self.config = vec.config

    def __enter__(self) -> PyMARLEnv:
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
            f'<PyMARLEnv {vec.spec.benchmark}/{vec.spec.name} '
            f'backend={vec.runtime.backend.capabilities.name!r} accel={vec.accel!r} '
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
        """Reset the episode and return `(obs_list, state)`."""
        self.vec.reset()
        return list(self.vec.obs()[0]), self.vec.state()[0]

    def step(self, actions: Any, enemy_actions: Any = None) -> tuple[float, bool, dict[str, Any]]:
        """Step the environment; `enemy_actions` provides Player 2 actions in two-sided mode."""
        acts = np.array([int(a) for a in actions], dtype=np.int64)[None]
        if enemy_actions is None:
            batch = self.vec.step(acts)
        else:
            acts2 = np.array([int(a) for a in enemy_actions], dtype=np.int64)[None]
            batch = self.vec.step(acts, acts2)
        done = bool(batch.terminated[0] or batch.truncated[0] or self.vec.runtime.done()[0])
        return float(batch.reward[0]), done, batch.infos[0]

    def get_obs(self, side: int = 1) -> list[np.ndarray]:
        return list(self.vec.obs(side)[0])

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        return self.vec.obs()[0, agent_id]

    def get_state(self, side: int = 1) -> np.ndarray:
        return self.vec.state(side)[0]

    def get_avail_actions(self, side: int = 1) -> list[list[int]]:
        return self.vec.avail_actions(side)[0].tolist()

    def get_avail_agent_actions(self, agent_id: int) -> list[int]:
        return self.vec.avail_actions()[0, agent_id].tolist()

    def get_obs_size(self) -> int:
        """Upstream `get_obs_size()` (uses `reported_size` to preserve `quirks.timestep_pad`)."""
        return self.vec.obs_layout.reported_size

    def get_state_size(self) -> int:
        """Upstream `get_state_size()` (uses `reported_size` to preserve `quirks.timestep_pad`)."""
        return self.vec.state_layout.reported_size

    def get_total_actions(self) -> int:
        return self.vec.n_actions

    def get_cap_size(self) -> int:
        """SMACv2 `get_cap_size`: hit probability bit plus unit type bits."""
        return int(self.vec.attack_prob is not None) + self.vec.catalog.unit_type_bits

    def get_capabilities(self) -> np.ndarray:
        """Return `float32[A * cap_size]` (SMACv2 `get_capabilities`)."""
        return self.vec.capabilities()[0]

    def get_env_info(self) -> dict[str, Any]:
        """Return the PyMARL environment metadata dictionary."""
        vec = self.vec
        info: dict[str, Any] = {
            'state_shape': vec.state_layout.reported_size,
            'obs_shape': vec.obs_layout.reported_size,
        }
        if getattr(vec.spec, 'benchmark', None) == 'smacv2':
            info['cap_shape'] = int(vec.attack_prob is not None) + vec.catalog.unit_type_bits
        info.update(
            {
                'n_actions': vec.n_actions,
                'n_agents': vec.n_agents,
                'episode_limit': vec.episode_limit,
                'agent_features': list(vec.state_layout.ally),
                'enemy_features': list(vec.state_layout.enemy),
            }
        )
        return info

    @property
    def max_reward(self) -> float | None:
        """`max_reward` of env 0 (`None` before `reset()`)."""
        v = self.vec.runtime.max_rewards()[0]
        return None if np.isnan(v) else float(v)

    def get_stats(self) -> dict[str, Any]:
        """Return PyMARL battle statistics derived from kernel runtime counters."""
        counters = self.vec.runtime.counters()
        won = int(counters['battles_won'])
        games = int(counters['battles_game'])
        timeouts = int(counters['timeouts'])
        return {
            'battles_won': won,
            'battles_game': games,
            'battles_draw': timeouts,
            'win_rate': won / games if games else 0.0,
            'timeouts': timeouts,
            'restarts': 0,
        }

    def seed(self) -> int | None:
        """Return the active seed for env 0 (`env_seeds[0]` if set, else batch seed)."""
        per_env = self.vec.runtime.env_seeds[0]
        return self.vec._seed if per_env is None else per_env

    def close(self) -> None:
        self.vec.close()

    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError('Rendering is not supported in unismac')

    def save_replay(self) -> None:
        raise NotImplementedError('Saving replays is not supported in unismac')


UnismacEnv = PyMARLEnv


def make_pymarl_env(
    preset: str,
    scenario: str,
    *,
    backend: Literal['auto', 'sc2', 'pyref', 'zgame'] | Backend | None = None,
    profile: str | None = None,
    dual: bool = False,
    seed: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    zgame_threads: int | None = None,
    accel: Literal['auto', 'python', 'services', 'rust'] = 'auto',
) -> PyMARLEnv:
    """Create a single-environment PyMARL / SMAC `MultiAgentEnv` adapter (`batch_size=1`)."""
    vec = make_vec_env(
        preset,
        scenario,
        backend=backend,
        profile=profile,
        dual=dual,
        batch_size=1,
        seed=seed,
        overrides=overrides,
        zgame_threads=zgame_threads,
        accel=accel,
    )
    return PyMARLEnv(vec)
