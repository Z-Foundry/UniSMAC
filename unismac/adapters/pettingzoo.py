"""PettingZoo `ParallelEnv` adapter with first-class two-sided self-play (`dual=True`).

Requires the optional `pettingzoo` extra: ``pip install "unismac[pettingzoo]"``.

* Exposes `ally_0..ally_{N-1}` in single-sided mode and appends `enemy_0..enemy_{M-1}` when
  `dual=True`, supporting heterogeneous asymmetric maps natively via per-agent spaces.
* Embeds `int8` `'action_mask'` inside each agent's observation dict by default
  (`obs_with_mask=True`), or places it inside `infos[agent]['action_mask']` when
  `obs_with_mask=False`.
* Exposes global CTDE state via `state()` / `state_space` (Player 1 canonical view) and
  side-specific helpers `state_for(side)` / `state_space_for(side)` for dual self-play critics.
* For turn-based `AECEnv` consumers, wrap on demand via
  ``pettingzoo.utils.conversions.parallel_to_aec(env)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..env import VecEnv, make_vec_env
from ._common import (
    InvalidActionMode,
    missing_extra_error,
    require_single_env,
    sanitize_actions,
    validate_invalid_action_mode,
    validate_render_mode,
)

try:
    from gymnasium import spaces
    from pettingzoo import ParallelEnv
except ImportError as exc:
    raise missing_extra_error(
        'unismac.adapters.pettingzoo',
        'pettingzoo',
        'pettingzoo>=1.24.3 and gymnasium>=1.0',
        exc,
    ) from None

if TYPE_CHECKING:
    from ..contract import Backend

__all__ = ['PettingZooParallelEnv', 'make_pettingzoo_env']

DeadAgentMode = Literal['keep', 'remove']


def _validate_dead_agent_mode(mode: str) -> DeadAgentMode:
    if mode == 'keep':
        return 'keep'
    if mode == 'remove':
        return 'remove'
    raise ValueError(
        f"PettingZooParallelEnv got unknown dead_agent_mode={mode!r}; expected 'keep' or 'remove'"
    )


def _agent_obs_space(obs_dim: int, n_actions: int, obs_with_mask: bool) -> Any:
    obs_box = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    if not obs_with_mask:
        return obs_box
    return spaces.Dict(
        {
            'observation': obs_box,
            'action_mask': spaces.Box(low=0, high=1, shape=(n_actions,), dtype=np.int8),
        }
    )


class PettingZooParallelEnv(ParallelEnv):
    """PettingZoo `ParallelEnv` wrapping a `batch_size=1` `VecEnv` (single-sided or `dual=True`).

    * `dead_agent_mode='keep'` (default) keeps all agents in `env.agents` until episode end so
      team-shared SMAC rewards (including terminal win bonuses) remain intact; dead units receive
      an `action_mask` permitting only `0` (NOOP).
    * `dead_agent_mode='remove'` terminates and removes agents on the step they die, automatically
      filling `0` (NOOP) into the underlying kernel slot on subsequent steps.
    * `obs_with_mask=True` (default) returns `{'observation': float32[D], 'action_mask': int8[K]}`;
      `obs_with_mask=False` returns `float32[D]` and places `'action_mask'` in `infos[agent]`.
    """

    def __init__(
        self,
        vec: VecEnv,
        *,
        dead_agent_mode: DeadAgentMode = 'keep',
        obs_with_mask: bool = True,
        invalid_action_mode: InvalidActionMode = 'raise',
        render_mode: str | None = None,
    ) -> None:
        require_single_env(vec, 'PettingZooParallelEnv')
        validate_render_mode(render_mode, 'PettingZooParallelEnv')
        self.dead_agent_mode: DeadAgentMode = _validate_dead_agent_mode(dead_agent_mode)
        self.obs_with_mask = bool(obs_with_mask)
        self.invalid_action_mode: InvalidActionMode = validate_invalid_action_mode(
            invalid_action_mode, 'PettingZooParallelEnv'
        )
        self.render_mode = render_mode
        self.vec = vec
        self.dual = vec.dual
        self.metadata: dict[str, Any] = {
            'name': f'unismac_{vec.spec.benchmark}_{vec.spec.name}_v0',
            'is_parallelizable': True,
            'render_modes': [],
        }

        self._ally_agents: tuple[str, ...] = tuple(f'ally_{i}' for i in range(vec.n_agents_for(1)))
        self._enemy_agents: tuple[str, ...] = (
            tuple(f'enemy_{i}' for i in range(vec.n_agents_for(2))) if vec.dual else ()
        )
        self.possible_agents: list[str] = list(self._ally_agents) + list(self._enemy_agents)
        self.agents: list[str] = []
        self._ally_index: dict[str, int] = {a: i for i, a in enumerate(self._ally_agents)}
        self._enemy_index: dict[str, int] = {a: i for i, a in enumerate(self._enemy_agents)}

        # Pre-build and cache space instances so repeated calls satisfy `is` identity checks.
        obs1_space = _agent_obs_space(
            vec.obs_layout_for(1).size, vec.n_actions_for(1), self.obs_with_mask
        )
        act1_space = spaces.Discrete(vec.n_actions_for(1))
        self._observation_spaces: dict[str, Any] = dict.fromkeys(self._ally_agents, obs1_space)
        self._action_spaces: dict[str, Any] = dict.fromkeys(self._ally_agents, act1_space)

        self._state_spaces: dict[int, Any] = {
            1: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(vec.state_layout_for(1).size,),
                dtype=np.float32,
            )
        }
        if vec.dual:
            obs2_space = _agent_obs_space(
                vec.obs_layout_for(2).size, vec.n_actions_for(2), self.obs_with_mask
            )
            act2_space = spaces.Discrete(vec.n_actions_for(2))
            for agent in self._enemy_agents:
                self._observation_spaces[agent] = obs2_space
                self._action_spaces[agent] = act2_space
            self._state_spaces[2] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(vec.state_layout_for(2).size,),
                dtype=np.float32,
            )
        self.state_space: Any = self._state_spaces[1]

    def __enter__(self) -> PettingZooParallelEnv:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        vec = self.vec
        extras = ' dual=True' if self.dual else ''
        return (
            f'<PettingZooParallelEnv {vec.spec.benchmark}/{vec.spec.name} '
            f'backend={vec.runtime.backend.capabilities.name!r} accel={vec.accel!r} '
            f'agents={len(self.possible_agents)}{extras}>'
        )

    def observation_space(self, agent: str) -> Any:
        """Return the cached observation space for `agent` (valid for all `possible_agents`)."""
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> Any:
        """Return the cached action space for `agent` (valid for all `possible_agents`)."""
        return self._action_spaces[agent]

    def state(self) -> np.ndarray:
        """Return Player 1's canonical global state vector `float32[S]`."""
        return self.vec.state(side=1)[0].copy()

    def state_for(self, side: Literal[1, 2]) -> np.ndarray:
        """Return the global state vector `float32[S_side]` from `side`'s perspective."""
        if side not in self._state_spaces:
            raise ValueError(f'Invalid side {side!r} for dual={self.dual}')
        return self.vec.state(side=side)[0].copy()

    def state_space_for(self, side: Literal[1, 2]) -> Any:
        """Return the cached global state space for `side` (1, or 2 in `dual=True` mode)."""
        if side not in self._state_spaces:
            raise ValueError(f'Invalid side {side!r} for dual={self.dual}')
        return self._state_spaces[side]

    def _pack_agent_obs(self, obs: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> Any:
        if self.obs_with_mask:
            return {'observation': obs.copy(), 'action_mask': mask.copy()}
        info['action_mask'] = mask.copy()
        return obs.copy()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        seeds = None if seed is None else [int(seed)]
        opponents: list[str | None] | None = None
        if isinstance(options, Mapping) and options.get('opponent') is not None:
            opponents = [str(options['opponent'])]
        self.vec.reset(seeds=seeds, opponents=opponents)

        obs1 = self.vec.obs(1)[0]
        mask1 = self.vec.avail_actions(1)[0].astype(np.int8, copy=False)
        alive1 = self.vec.alive(1)[0]

        obs2: np.ndarray | None = None
        mask2: np.ndarray | None = None
        alive2: np.ndarray | None = None
        if self.dual:
            obs2 = self.vec.obs(2)[0]
            mask2 = self.vec.avail_actions(2)[0].astype(np.int8, copy=False)
            alive2 = self.vec.alive(2)[0]

        if self.dead_agent_mode == 'remove':
            active_agents = [a for i, a in enumerate(self._ally_agents) if alive1[i]]
            if self.dual and alive2 is not None:
                active_agents.extend(a for i, a in enumerate(self._enemy_agents) if alive2[i])
            self.agents = active_agents
        else:
            self.agents = list(self.possible_agents)

        observations: dict[str, Any] = {}
        infos: dict[str, dict[str, Any]] = {}
        for agent in self.agents:
            if agent in self._ally_index:
                idx = self._ally_index[agent]
                info: dict[str, Any] = {'side': 1}
                observations[agent] = self._pack_agent_obs(obs1[idx], mask1[idx], info)
                infos[agent] = info
            else:
                assert obs2 is not None and mask2 is not None
                idx = self._enemy_index[agent]
                info = {'side': 2}
                observations[agent] = self._pack_agent_obs(obs2[idx], mask2[idx], info)
                infos[agent] = info
        return observations, infos

    def step(
        self, actions: Mapping[str, Any]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        prev_agents = list(self.agents)
        expected = set(prev_agents)
        got = set(actions.keys())
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise ValueError(
                f'actions keys do not match active agents (missing={missing}, unexpected={extra})'
            )

        ally_acts = np.zeros((1, self.vec.n_agents_for(1)), dtype=np.int64)
        enemy_acts = np.zeros((1, self.vec.n_agents_for(2)), dtype=np.int64) if self.dual else None
        for agent in prev_agents:
            act_val = int(actions[agent])
            if agent in self._ally_index:
                ally_acts[0, self._ally_index[agent]] = act_val
            else:
                assert enemy_acts is not None
                enemy_acts[0, self._enemy_index[agent]] = act_val

        active = ~self.vec.runtime.done()
        if self.invalid_action_mode == 'fallback' and active[0]:
            ally_acts, remapped1 = sanitize_actions(
                ally_acts, self.vec.avail_actions(1), active, mode='fallback'
            )
            if self.dual and enemy_acts is not None:
                enemy_acts, remapped2 = sanitize_actions(
                    enemy_acts, self.vec.avail_actions(2), active, mode='fallback'
                )
            else:
                remapped2 = np.zeros(1, dtype=np.int64)
        else:
            remapped1 = np.zeros(1, dtype=np.int64)
            remapped2 = np.zeros(1, dtype=np.int64)

        if self.dual:
            batch = self.vec.step(ally_acts, enemy_acts)
        else:
            batch = self.vec.step(ally_acts)

        obs1 = self.vec.obs(1)[0]
        mask1 = self.vec.avail_actions(1)[0].astype(np.int8, copy=False)
        alive1 = self.vec.alive(1)[0]
        obs2: np.ndarray | None = None
        mask2: np.ndarray | None = None
        alive2: np.ndarray | None = None
        if self.dual:
            obs2 = self.vec.obs(2)[0]
            mask2 = self.vec.avail_actions(2)[0].astype(np.int8, copy=False)
            alive2 = self.vec.alive(2)[0]

        ep_term = bool(batch.terminated[0])
        ep_trunc = bool(batch.truncated[0])
        ep_done = ep_term or ep_trunc
        raw_info = batch.infos[0]
        rew1 = float(batch.reward[0])
        rew2 = float(batch.reward_enemy[0]) if batch.reward_enemy is not None else 0.0

        observations: dict[str, Any] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}
        next_agents: list[str] = []

        for agent in prev_agents:
            if agent in self._ally_index:
                idx = self._ally_index[agent]
                is_alive = bool(alive1[idx])
                info: dict[str, Any] = {
                    'side': 1,
                    'battle_won': bool(raw_info.get('battle_won', False)),
                    'dead_allies': int(raw_info.get('dead_allies', 0)),
                    'dead_enemies': int(raw_info.get('dead_enemies', 0)),
                }
                if self.invalid_action_mode == 'fallback':
                    info['invalid_actions_remapped'] = int(remapped1[0])
                observations[agent] = self._pack_agent_obs(obs1[idx], mask1[idx], info)
                rewards[agent] = rew1
            else:
                assert obs2 is not None and mask2 is not None and alive2 is not None
                idx = self._enemy_index[agent]
                is_alive = bool(alive2[idx])
                info = {
                    'side': 2,
                    'battle_won': bool(raw_info.get('battle_lost', False)),
                    'dead_allies': int(raw_info.get('dead_enemies', 0)),
                    'dead_enemies': int(raw_info.get('dead_allies', 0)),
                }
                if self.invalid_action_mode == 'fallback':
                    info['invalid_actions_remapped'] = int(remapped2[0])
                observations[agent] = self._pack_agent_obs(obs2[idx], mask2[idx], info)
                rewards[agent] = rew2

            if self.dead_agent_mode == 'remove':
                term = ep_term or (not is_alive)
                trunc = (not term) and ep_trunc
            else:
                term = ep_term
                trunc = ep_trunc

            terminations[agent] = term
            truncations[agent] = trunc
            infos[agent] = info
            if not (ep_done or term or trunc):
                next_agents.append(agent)

        self.agents = next_agents
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.vec.close()


def make_pettingzoo_env(
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
    dead_agent_mode: DeadAgentMode = 'keep',
    obs_with_mask: bool = True,
    invalid_action_mode: InvalidActionMode = 'raise',
    render_mode: str | None = None,
) -> PettingZooParallelEnv:
    """Create a PettingZoo `ParallelEnv` adapter (`batch_size=1`, single-sided or `dual=True`)."""
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
    return PettingZooParallelEnv(
        vec,
        dead_agent_mode=dead_agent_mode,
        obs_with_mask=obs_with_mask,
        invalid_action_mode=invalid_action_mode,
        render_mode=render_mode,
    )
