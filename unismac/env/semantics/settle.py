"""Per-env step settlement (the per-env loop in `VecEnv.step`): converts termination codes into
`terminated`, `infos`, win/loss/timeout counters, terminal rewards, and `max_reward` scaling.

This module provides the Python reference implementation; the Rust counterpart lives in
`zgame-smac` (`_zgame.env.settle`) and is used when `accel='services'` (see `unismac.env.accel`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import RewardConfig
from .termination import LOST, NONE, WON
from .units import Frame

if TYPE_CHECKING:
    from ..core import VecEnv

__all__ = ['settle', 'settle_enemy']


def settle(
    env: VecEnv, active: np.ndarray, end: np.ndarray, cur: Frame, reward: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return `(reward float64[B], raw_done bool[B], infos)` and update `env._battles_game`,
    `_battles_won`, `_timeouts`, `_episode_count`, and `_done` in place.

    Inactive envs get `raw_done=True`, empty info `{}`, and reward 0. Active envs record death
    counts (health == 0); non-NONE `end` finishes the battle (adding `reward.win` / `reward.defeat`,
    or +/-1 if sparse); otherwise hitting `episode_limit` records a timeout (adding
    `episode_limit=True` to info when `continuing_episode` is enabled). Rewards are then scaled by
    `max_reward / scale_rate`. Input `reward` is not mutated.
    """
    reward = reward.copy()
    terminated = np.zeros(env.batch_size, bool)
    infos: list[dict[str, Any]] = []
    rc = env.config.reward
    for b in range(env.batch_size):
        if not active[b]:
            terminated[b] = True
            infos.append({})
            continue
        info: dict[str, Any] = {
            'battle_won': False,
            'dead_allies': int((cur.allies.health[b] == 0).sum()),
            'dead_enemies': int((cur.enemies.health[b] == 0).sum()),
        }
        if end[b] != NONE:
            terminated[b] = True
            env._battles_game[b] += 1
            if end[b] == WON:
                env._battles_won[b] += 1
                info['battle_won'] = True
                reward[b] = reward[b] + rc.win if not rc.sparse else 1
            elif end[b] == LOST:
                reward[b] = reward[b] + rc.defeat if not rc.sparse else -1
        elif env._episode_steps[b] >= env.episode_limit:
            terminated[b] = True
            if env.config.termination.continuing_episode:
                info['episode_limit'] = True
            env._battles_game[b] += 1
            env._timeouts[b] += 1
        if terminated[b]:
            env._episode_count[b] += 1
        infos.append(info)
    if rc.scale:
        reward = reward / (env._max_rewards / rc.scale_rate)
    reward = np.where(active, reward, 0.0)
    env._done |= terminated
    return reward, terminated, infos


def settle_enemy(
    active: np.ndarray,
    end: np.ndarray,
    reward: np.ndarray,
    max_rewards: np.ndarray,
    config: RewardConfig,
    infos: list[dict[str, Any]],
) -> np.ndarray:
    """Two-sided mode settlement for Player 2: returns `float64[B]` Player 2 rewards and adds
    `battle_lost` (Player 1 lost, i.e. `end == LOST`) and `reward_enemy` to active env infos.

    Mirrors `settle`'s reward logic from Player 2's perspective: `LOST` (Player 2 wins) adds `win`,
    `WON` adds `defeat` (+/-1 if sparse), scaled by Player 2's `max_rewards / scale_rate`; inactive
    envs receive 0. Input `reward` is not mutated; battle counters remain tracked for Player 1.
    """
    reward = reward.copy()
    for b in range(reward.shape[0]):
        if not active[b]:
            continue
        if end[b] == LOST:
            reward[b] = reward[b] + config.win if not config.sparse else 1
        elif end[b] == WON:
            reward[b] = reward[b] + config.defeat if not config.sparse else -1
    if config.scale:
        reward = reward / (max_rewards / config.scale_rate)
    reward = np.where(active, reward, 0.0)
    for b in np.flatnonzero(active):
        infos[b]['battle_lost'] = bool(end[b] == LOST)
        infos[b]['reward_enemy'] = float(reward[b])
    return reward
