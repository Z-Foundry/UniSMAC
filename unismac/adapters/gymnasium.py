"""Gymnasium 1.0+ `Env` and `VectorEnv` adapters for single-sided UniSMAC environments.

Requires the optional `gymnasium` extra: ``pip install "unismac[gymnasium]"``.

* `GymnasiumEnv` wraps a `batch_size=1`, single-sided (`dual=False`) `VecEnv`.
* `GymnasiumVecEnv` wraps a batched (`batch_size >= 1`), single-sided `VecEnv` and natively
  bridges Gymnasium 1.0+ `AutoresetMode.NEXT_STEP`, `SAME_STEP`, and `DISABLED` onto the
  Rust-accelerated kernel without multiprocessing overhead.

For two-sided self-play (`dual=True`), use `unismac.adapters.pettingzoo.PettingZooParallelEnv`
or `unismac.VecEnv` directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..env import VecEnv, make_vec_env
from ._common import (
    InvalidActionMode,
    missing_extra_error,
    require_single_env,
    require_single_sided,
    sanitize_actions,
    validate_invalid_action_mode,
    validate_render_mode,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.vector import AutoresetMode, VectorEnv
    from gymnasium.vector.utils import batch_space
except ImportError as exc:
    raise missing_extra_error(
        'unismac.adapters.gymnasium', 'gymnasium', 'gymnasium>=1.0', exc
    ) from None

if TYPE_CHECKING:
    from ..contract import Backend

__all__ = ['GymnasiumEnv', 'GymnasiumVecEnv', 'make_gym_env', 'make_gym_vec_env']

ObsMode = Literal['dict', 'obs_only']


def _validate_obs_mode(obs_mode: str, adapter_name: str) -> ObsMode:
    if obs_mode == 'dict':
        return 'dict'
    if obs_mode == 'obs_only':
        return 'obs_only'
    raise ValueError(
        f"{adapter_name} got unknown obs_mode={obs_mode!r}; expected 'dict' or 'obs_only'"
    )


def _resolve_autoreset_mode(mode: AutoresetMode | str) -> AutoresetMode:
    if isinstance(mode, AutoresetMode):
        return mode
    key = str(mode).strip().lower().replace('-', '_')
    if key in ('next_step', 'nextstep'):
        return AutoresetMode.NEXT_STEP
    if key in ('same_step', 'samestep'):
        return AutoresetMode.SAME_STEP
    if key == 'disabled':
        return AutoresetMode.DISABLED
    raise ValueError(
        f'Unknown autoreset_mode {mode!r}; expected AutoresetMode or '
        "'next_step', 'same_step', or 'disabled'"
    )


def _build_single_spaces(vec: VecEnv, *, obs_mode: ObsMode, include_state: bool) -> tuple[Any, Any]:
    """Construct `(observation_space, action_space)` for one single-sided environment."""
    n_agents = vec.n_agents
    n_actions = vec.n_actions
    obs_dim = vec.obs_layout.size
    state_dim = vec.state_layout.size

    obs_box = spaces.Box(low=-np.inf, high=np.inf, shape=(n_agents, obs_dim), dtype=np.float32)
    if obs_mode == 'dict':
        obs_dict: dict[str, Any] = {'obs': obs_box}
        if include_state:
            obs_dict['state'] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
            )
        obs_dict['action_mask'] = spaces.Box(
            low=0, high=1, shape=(n_agents, n_actions), dtype=np.int8
        )
        observation_space: Any = spaces.Dict(obs_dict)
    else:
        observation_space = obs_box

    action_space = spaces.MultiDiscrete(np.full(n_agents, n_actions, dtype=np.int64))
    return observation_space, action_space


def _slice_batch_obs(batch_obs: Any, env_id: int) -> Any:
    """Extract a single environment's observation copy from a batched observation."""
    if isinstance(batch_obs, dict):
        return {k: v[env_id].copy() for k, v in batch_obs.items()}
    return batch_obs[env_id].copy()


def _pack_vector_infos(
    raw_infos: Sequence[Mapping[str, Any]], valid_mask: np.ndarray
) -> dict[str, Any]:
    """Pack per-env info dicts into Gymnasium 1.0+ dict-of-arrays with `_key` validity masks."""
    b_size = len(raw_infos)
    keys: list[str] = []
    seen: set[str] = set()
    for b in range(b_size):
        if not valid_mask[b]:
            continue
        for k in raw_infos[b]:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    packed: dict[str, Any] = {}
    for k in keys:
        mask = np.zeros(b_size, dtype=bool)
        sample: Any = None
        for b in range(b_size):
            if valid_mask[b] and k in raw_infos[b]:
                sample = raw_infos[b][k]
                break
        if isinstance(sample, (bool, np.bool_)):
            arr: np.ndarray = np.zeros(b_size, dtype=bool)
        elif isinstance(sample, (int, np.integer)):
            arr = np.zeros(b_size, dtype=np.int64)
        elif isinstance(sample, (float, np.floating)):
            arr = np.zeros(b_size, dtype=np.float64)
        elif isinstance(sample, np.ndarray):
            arr = np.zeros((b_size, *sample.shape), dtype=sample.dtype)
        else:
            arr = np.empty(b_size, dtype=object)

        for b in range(b_size):
            if valid_mask[b] and k in raw_infos[b]:
                arr[b] = raw_infos[b][k]
                mask[b] = True
        packed[k] = arr
        packed[f'_{k}'] = mask
    return packed


class GymnasiumEnv(gym.Env):
    """Single-environment Gymnasium 1.0+ adapter wrapping a `batch_size=1`, `dual=False` `VecEnv`.

    * Default `obs_mode='dict'` exposes `{'obs': float32[A, D], 'state': float32[S],
      'action_mask': int8[A, K]}` (omit `'state'` with `include_state=False`).
    * `obs_mode='obs_only'` returns `float32[A, D]` directly and places `'action_mask'` (and
      `'state'` when `include_state=True`) inside `info`.
    * `invalid_action_mode='raise'` (default) enforces strict legal-action validation; set
      `'fallback'` to remap illegal actions deterministically (`1` STOP for alive units, `0` NOOP
      for dead units) and report counts in `info['invalid_actions_remapped']` (required for
      unmasked random sampling such as `gymnasium.utils.env_checker.check_env`).
    """

    metadata: dict[str, Any] = {'render_modes': []}

    def __init__(
        self,
        vec: VecEnv,
        *,
        obs_mode: ObsMode = 'dict',
        include_state: bool = True,
        invalid_action_mode: InvalidActionMode = 'raise',
        render_mode: str | None = None,
    ) -> None:
        require_single_env(vec, 'GymnasiumEnv')
        require_single_sided(vec, 'GymnasiumEnv')
        validate_render_mode(render_mode, 'GymnasiumEnv')
        self.obs_mode: ObsMode = _validate_obs_mode(obs_mode, 'GymnasiumEnv')
        self.include_state = bool(include_state)
        self.invalid_action_mode: InvalidActionMode = validate_invalid_action_mode(
            invalid_action_mode, 'GymnasiumEnv'
        )
        self.render_mode = render_mode
        self.vec = vec
        self.observation_space, self.action_space = _build_single_spaces(
            vec, obs_mode=self.obs_mode, include_state=self.include_state
        )

    def __repr__(self) -> str:
        vec = self.vec
        return (
            f'<GymnasiumEnv {vec.spec.benchmark}/{vec.spec.name} '
            f'backend={vec.runtime.backend.capabilities.name!r} accel={vec.accel!r} '
            f'obs_mode={self.obs_mode!r} invalid_action_mode={self.invalid_action_mode!r}>'
        )

    def _pack_obs(self) -> Any:
        obs = self.vec.obs()[0].copy()
        if self.obs_mode == 'obs_only':
            return obs
        out: dict[str, np.ndarray] = {'obs': obs}
        if self.include_state:
            out['state'] = self.vec.state()[0].copy()
        out['action_mask'] = self.vec.avail_actions()[0].astype(np.int8, copy=True)
        return out

    def _obs_only_info(self) -> dict[str, Any]:
        extra: dict[str, Any] = {
            'action_mask': self.vec.avail_actions()[0].astype(np.int8, copy=True),
        }
        if self.include_state:
            extra['state'] = self.vec.state()[0].copy()
        return extra

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        seeds = None if seed is None else [int(seed)]
        opponents: list[str | None] | None = None
        if isinstance(options, Mapping) and options.get('opponent') is not None:
            opponents = [str(options['opponent'])]
        self.vec.reset(seeds=seeds, opponents=opponents)
        obs = self._pack_obs()
        info = self._obs_only_info() if self.obs_mode == 'obs_only' else {}
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        arr = np.asarray(action, dtype=np.int64)
        if arr.ndim == 1:
            acts = arr[None, :]
        else:
            acts = arr
        if acts.shape != (1, self.vec.n_agents):
            raise ValueError(
                f'Expected action shape ({self.vec.n_agents},), got {tuple(arr.shape)}'
            )
        active = ~self.vec.runtime.done()
        if self.invalid_action_mode == 'fallback' and active[0]:
            avail = self.vec.avail_actions()
            acts, remapped = sanitize_actions(acts, avail, active, mode='fallback')
        else:
            remapped = np.zeros(1, dtype=np.int64)
            if not active[0]:
                acts = np.zeros((1, self.vec.n_agents), dtype=np.int64)

        batch = self.vec.step(acts)
        obs = self._pack_obs()
        info = dict(batch.infos[0])
        if self.invalid_action_mode == 'fallback':
            info['invalid_actions_remapped'] = int(remapped[0])
        if self.obs_mode == 'obs_only':
            info.update(self._obs_only_info())
        return (
            obs,
            float(batch.reward[0]),
            bool(batch.terminated[0]),
            bool(batch.truncated[0]),
            info,
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.vec.close()


class GymnasiumVecEnv(VectorEnv):
    """Batched Gymnasium 1.0+ `VectorEnv` adapter wrapping a single-sided (`dual=False`) `VecEnv`.

    Supports all three Gymnasium 1.0+ `AutoresetMode` policies:
    * `AutoresetMode.NEXT_STEP` (default): steps ongoing envs and resets previously finished envs
      in one pass with a single batched observation read.
    * `AutoresetMode.SAME_STEP`: snapshots terminal observations/infos into `infos['final_obs']` /
      `infos['final_info']` (with `_final_obs` / `_final_info` masks) and resets finished slots
      within the same step.
    * `AutoresetMode.DISABLED`: leaves finished slots frozen until the caller triggers selective
      resets via `reset(options={'reset_mask': bool_mask})`.
    """

    def __init__(
        self,
        vec: VecEnv,
        *,
        obs_mode: ObsMode = 'dict',
        include_state: bool = True,
        invalid_action_mode: InvalidActionMode = 'raise',
        autoreset_mode: AutoresetMode | str = 'next_step',
        render_mode: str | None = None,
    ) -> None:
        require_single_sided(vec, 'GymnasiumVecEnv')
        validate_render_mode(render_mode, 'GymnasiumVecEnv')
        self.obs_mode: ObsMode = _validate_obs_mode(obs_mode, 'GymnasiumVecEnv')
        self.include_state = bool(include_state)
        self.invalid_action_mode: InvalidActionMode = validate_invalid_action_mode(
            invalid_action_mode, 'GymnasiumVecEnv'
        )
        self.autoreset_mode: AutoresetMode = _resolve_autoreset_mode(autoreset_mode)
        self.vec = vec
        self.num_envs = vec.batch_size
        self.single_observation_space, self.single_action_space = _build_single_spaces(
            vec, obs_mode=self.obs_mode, include_state=self.include_state
        )
        self.observation_space = batch_space(self.single_observation_space, n=self.num_envs)
        self.action_space = batch_space(self.single_action_space, n=self.num_envs)
        self.metadata: dict[str, Any] = {
            'render_modes': [],
            'autoreset_mode': self.autoreset_mode,
        }
        self.render_mode = render_mode
        self.spec = None
        self.closed = False
        self._np_random: Any = None
        self._np_random_seed: int | None = None
        self._autoreset_envs = np.zeros(self.num_envs, dtype=bool)

    def __enter__(self) -> GymnasiumVecEnv:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        vec = self.vec
        return (
            f'<GymnasiumVecEnv {vec.spec.benchmark}/{vec.spec.name} '
            f'num_envs={self.num_envs} backend={vec.runtime.backend.capabilities.name!r} '
            f'accel={vec.accel!r} autoreset_mode={self.autoreset_mode.name}>'
        )

    def _pack_batch_obs(self) -> Any:
        obs = self.vec.obs().copy()
        if self.obs_mode == 'obs_only':
            return obs
        out: dict[str, np.ndarray] = {'obs': obs}
        if self.include_state:
            out['state'] = self.vec.state().copy()
        out['action_mask'] = self.vec.avail_actions().astype(np.int8, copy=True)
        return out

    def _obs_only_batch_infos(self) -> dict[str, Any]:
        all_valid = np.ones(self.num_envs, dtype=bool)
        extra: dict[str, Any] = {
            'action_mask': self.vec.avail_actions().astype(np.int8, copy=True),
            '_action_mask': all_valid,
        }
        if self.include_state:
            extra['state'] = self.vec.state().copy()
            extra['_state'] = all_valid.copy()
        return extra

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        b_size = self.num_envs
        env_ids: list[int] | None = None
        if isinstance(options, Mapping) and options.get('reset_mask') is not None:
            mask = np.asarray(options['reset_mask'], dtype=bool)
            if mask.shape != (b_size,):
                raise ValueError(f'reset_mask must have shape ({b_size},), got {tuple(mask.shape)}')
            env_ids = np.flatnonzero(mask).tolist()

        seeds: list[int | None] | None = None
        if seed is not None:
            if isinstance(seed, (int, np.integer)):
                base = int(seed)
                self._np_random, self._np_random_seed = gym.utils.seeding.np_random(base)
                if env_ids is None:
                    seeds = [base + i for i in range(b_size)]
                else:
                    seeds = [base + b for b in env_ids]
            elif isinstance(seed, Sequence):
                if len(seed) == b_size:
                    first = next((int(s) for s in seed if s is not None), None)
                    if first is not None:
                        self._np_random, self._np_random_seed = gym.utils.seeding.np_random(first)
                    if env_ids is None:
                        seeds = [None if s is None else int(s) for s in seed]
                    else:
                        seeds = [None if seed[b] is None else int(seed[b]) for b in env_ids]
                elif env_ids is not None and len(seed) == len(env_ids):
                    seeds = [None if s is None else int(s) for s in seed]
                else:
                    raise ValueError(
                        f'seed sequence length {len(seed)} does not match num_envs={b_size}'
                    )
            else:
                raise TypeError(f'seed must be int, sequence of ints, or None; got {type(seed)!r}')

        opponents: list[str | None] | None = None
        if isinstance(options, Mapping) and options.get('opponents') is not None:
            raw_opp = list(options['opponents'])
            if env_ids is not None and len(raw_opp) == b_size:
                opponents = [raw_opp[b] for b in env_ids]
            else:
                opponents = raw_opp

        if env_ids is None:
            self.vec.reset(seeds=seeds, opponents=opponents)
            self._autoreset_envs[:] = False
        elif env_ids:
            self.vec.reset(env_ids, seeds=seeds, opponents=opponents)
            self._autoreset_envs[env_ids] = False

        obs = self._pack_batch_obs()
        infos = self._obs_only_batch_infos() if self.obs_mode == 'obs_only' else {}
        return obs, infos

    def step(self, actions: Any) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        b_size = self.num_envs
        acts = np.asarray(actions, dtype=np.int64)
        if acts.shape != (b_size, self.vec.n_agents):
            raise ValueError(
                f'Expected actions shape ({b_size}, {self.vec.n_agents}), got {tuple(acts.shape)}'
            )

        if self.autoreset_mode == AutoresetMode.NEXT_STEP:
            return self._step_next(acts)
        if self.autoreset_mode == AutoresetMode.SAME_STEP:
            return self._step_same(acts)
        return self._step_disabled(acts)

    def _prepare_actions(
        self, acts: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.invalid_action_mode == 'fallback' and active.any():
            avail = self.vec.avail_actions()
            return sanitize_actions(acts, avail, active, mode='fallback')
        safe = acts.copy() if not active.all() else acts
        if not active.all():
            safe[~active] = 0
        return safe, np.zeros(self.num_envs, dtype=np.int64)

    def _enrich_raw_infos(
        self, raw_infos: Sequence[Mapping[str, Any]], active: np.ndarray, remapped: np.ndarray
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for b, raw in enumerate(raw_infos):
            item = dict(raw)
            if active[b] and self.invalid_action_mode == 'fallback':
                item['invalid_actions_remapped'] = int(remapped[b])
            out.append(item)
        return out

    def _step_next(
        self, acts: np.ndarray
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        b_size = self.num_envs
        if self._autoreset_envs.all():
            self.vec.reset()
            self._autoreset_envs[:] = False
            obs = self._pack_batch_obs()
            infos = self._obs_only_batch_infos() if self.obs_mode == 'obs_only' else {}
            return (
                obs,
                np.zeros(b_size, dtype=np.float64),
                np.zeros(b_size, dtype=bool),
                np.zeros(b_size, dtype=bool),
                infos,
            )

        reset_mask = self._autoreset_envs.copy()
        active = (~reset_mask) & (~self.vec.runtime.done())
        safe_acts, remapped = self._prepare_actions(acts, active)
        batch = self.vec.step(safe_acts)
        if reset_mask.any():
            self.vec.reset(np.flatnonzero(reset_mask).tolist())

        obs = self._pack_batch_obs()
        rewards = np.where(reset_mask, 0.0, batch.reward).astype(np.float64, copy=False)
        terminations = np.where(reset_mask, False, batch.terminated)
        truncations = np.where(reset_mask, False, batch.truncated)
        raw_infos = self._enrich_raw_infos(batch.infos, active, remapped)
        infos = _pack_vector_infos(raw_infos, active)
        if self.obs_mode == 'obs_only':
            infos.update(self._obs_only_batch_infos())
        self._autoreset_envs = terminations | truncations
        return obs, rewards, terminations, truncations, infos

    def _step_same(
        self, acts: np.ndarray
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        b_size = self.num_envs
        active = ~self.vec.runtime.done()
        safe_acts, remapped = self._prepare_actions(acts, active)
        batch = self.vec.step(safe_acts)
        rewards = batch.reward.astype(np.float64, copy=True)
        terminations = batch.terminated.copy()
        truncations = batch.truncated.copy()
        done_mask = terminations | truncations
        raw_infos = self._enrich_raw_infos(batch.infos, active, remapped)

        if done_mask.any():
            term_obs = self._pack_batch_obs()
            term_extra = self._obs_only_batch_infos() if self.obs_mode == 'obs_only' else None
            done_ids = np.flatnonzero(done_mask).tolist()
            final_obs = np.empty(b_size, dtype=object)
            final_info = np.empty(b_size, dtype=object)
            for b in done_ids:
                final_obs[b] = _slice_batch_obs(term_obs, b)
                fi = dict(raw_infos[b])
                if term_extra is not None:
                    fi['action_mask'] = term_extra['action_mask'][b].copy()
                    if self.include_state:
                        fi['state'] = term_extra['state'][b].copy()
                final_info[b] = fi
            self.vec.reset(done_ids)
            obs = self._pack_batch_obs()
            infos = _pack_vector_infos(raw_infos, active & ~done_mask)
            infos['final_obs'] = final_obs
            infos['_final_obs'] = done_mask.copy()
            infos['final_info'] = final_info
            infos['_final_info'] = done_mask.copy()
        else:
            obs = self._pack_batch_obs()
            infos = _pack_vector_infos(raw_infos, active)

        if self.obs_mode == 'obs_only':
            infos.update(self._obs_only_batch_infos())
        return obs, rewards, terminations, truncations, infos

    def _step_disabled(
        self, acts: np.ndarray
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        active = ~self.vec.runtime.done()
        safe_acts, remapped = self._prepare_actions(acts, active)
        batch = self.vec.step(safe_acts)
        obs = self._pack_batch_obs()
        raw_infos = self._enrich_raw_infos(batch.infos, active, remapped)
        infos = _pack_vector_infos(raw_infos, active)
        if self.obs_mode == 'obs_only':
            infos.update(self._obs_only_batch_infos())
        return (
            obs,
            batch.reward.astype(np.float64, copy=True),
            batch.terminated.copy(),
            batch.truncated.copy(),
            infos,
        )

    def render(self) -> None:
        return None

    def close_extras(self, **kwargs: Any) -> None:
        del kwargs
        self.vec.close()


def make_gym_env(
    preset: str,
    scenario: str,
    *,
    backend: Literal['auto', 'sc2', 'pyref', 'zgame'] | Backend | None = None,
    profile: str | None = None,
    seed: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    zgame_threads: int | None = None,
    accel: Literal['auto', 'python', 'services', 'rust'] = 'auto',
    obs_mode: ObsMode = 'dict',
    include_state: bool = True,
    invalid_action_mode: InvalidActionMode = 'raise',
    render_mode: str | None = None,
) -> GymnasiumEnv:
    """Create a single-environment Gymnasium 1.0+ adapter (`batch_size=1`, `dual=False`)."""
    vec = make_vec_env(
        preset,
        scenario,
        backend=backend,
        profile=profile,
        dual=False,
        batch_size=1,
        seed=seed,
        overrides=overrides,
        zgame_threads=zgame_threads,
        accel=accel,
    )
    return GymnasiumEnv(
        vec,
        obs_mode=obs_mode,
        include_state=include_state,
        invalid_action_mode=invalid_action_mode,
        render_mode=render_mode,
    )


def make_gym_vec_env(
    preset: str,
    scenario: str,
    *,
    backend: Literal['auto', 'sc2', 'pyref', 'zgame'] | Backend | None = None,
    profile: str | None = None,
    batch_size: int = 1,
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    overrides: Mapping[str, Any] | None = None,
    zgame_threads: int | None = None,
    accel: Literal['auto', 'python', 'services', 'rust'] = 'auto',
    obs_mode: ObsMode = 'dict',
    include_state: bool = True,
    invalid_action_mode: InvalidActionMode = 'raise',
    autoreset_mode: AutoresetMode | str = 'next_step',
    render_mode: str | None = None,
) -> GymnasiumVecEnv:
    """Create a batched Gymnasium 1.0+ `VectorEnv` adapter (`dual=False`)."""
    vec = make_vec_env(
        preset,
        scenario,
        backend=backend,
        profile=profile,
        dual=False,
        batch_size=batch_size,
        seed=seed,
        seeds=seeds,
        overrides=overrides,
        zgame_threads=zgame_threads,
        accel=accel,
    )
    return GymnasiumVecEnv(
        vec,
        obs_mode=obs_mode,
        include_state=include_state,
        invalid_action_mode=invalid_action_mode,
        autoreset_mode=autoreset_mode,
        render_mode=render_mode,
    )
