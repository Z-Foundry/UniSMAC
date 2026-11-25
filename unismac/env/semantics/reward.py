"""Rewards (upstream SMAC `reward_battle` and `max_reward` normalization).

Reproduces upstream floating-point accumulation order element-wise: sums sequentially in float64
by agent / enemy index (matching upstream Python float loops), using `services.ordered_sum`
instead of `np.sum`'s pairwise summation (see `unismac.env.semantics.services` for the Rust
counterpart).

`fill_max_rewards` belongs to the reset system layer; Rust counterpart: zgame-smac
`env::tables::fill_max_rewards`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import RewardConfig
from . import services
from .units import Frame, Side

__all__ = ['BattleReward', 'fill_max_rewards', 'max_reward']


def max_reward(n_enemies: int, config: RewardConfig, health_max, shield_max) -> float:
    """Upstream `max_reward`: `n_enemies * death_value + win` plus enemy health_max and shield_max
    accumulated in enemy index order.

    Upstream computes this once on the first episode (`_episode_count == 0`) and reuses it.
    """
    total = n_enemies * config.death_value + config.win
    for h, s in zip(health_max, shield_max, strict=True):
        total += float(h) + float(s)
    return float(total)


def fill_max_rewards(
    out: np.ndarray,
    env_ids: Sequence[int],
    n_enemies: int,
    config: RewardConfig,
    health_max: np.ndarray,
    shield_max: np.ndarray,
) -> None:
    """Populate `out[b]` with `max_reward` for any env in `env_ids` where it is still NaN
    (first reset), given enemy `[B, E]` max health and shield arrays."""
    for b in env_ids:
        if np.isnan(out[b]):
            out[b] = max_reward(n_enemies, config, health_max[b], shield_max[b])


class BattleReward:
    """Death tracking and per-step reward computation, shape `[B]`."""

    def __init__(
        self,
        batch_size: int,
        n_agents: int,
        n_enemies: int,
        config: RewardConfig,
        ops: services.Services = services.PYTHON,
    ):
        self.config = config
        self.ops = ops
        self.dead_ally = np.zeros((batch_size, n_agents), bool)
        self.dead_enemy = np.zeros((batch_size, n_enemies), bool)

    def reset(self, env_ids: Sequence[int]) -> None:
        self.dead_ally[list(env_ids)] = False
        self.dead_enemy[list(env_ids)] = False

    def step(self, prev: Frame, cur: Frame, active: np.ndarray) -> np.ndarray:
        """Return `float64[B]` step rewards; updates death tracking for `active` envs only,
        returning 0 for inactive envs.

        Accumulates in index order: `deaths` sums allies first (penalized only when
        `only_positive` is False) then enemies; `d_ally` and `d_enemy` sum per unit. Units that
        died this step contribute their previous health+shield; surviving units contribute the
        step delta.
        """
        cfg = self.config
        b_size = active.shape[0]
        if cfg.sparse:
            return np.zeros(b_size)
        neg = cfg.negative_scale
        a_died, a_track, a_prev, a_loss = self._side(
            prev.allies, cur.allies, active, self.dead_ally
        )
        e_died, e_track, e_prev, e_loss = self._side(
            prev.enemies, cur.enemies, active, self.dead_enemy
        )
        # Death rewards: -death_value * neg per dead ally, +death_value per dead enemy
        a_death = np.full(a_died.shape, -(cfg.death_value * neg), np.float64)
        e_death = np.full(e_died.shape, cfg.death_value, np.float64)
        a_mask = a_died & (not cfg.only_positive)
        deaths = self.ops.ordered_sum(
            np.concatenate([a_death, e_death], axis=1), np.concatenate([a_mask, e_died], axis=1)
        )
        d_ally = self.ops.ordered_sum(np.where(a_died, a_prev * neg, neg * a_loss), a_track)
        d_enemy = self.ops.ordered_sum(np.where(e_died, e_prev, e_loss), e_track)
        if cfg.only_positive:
            total = d_enemy + deaths
            reward = np.abs(total) if cfg.only_positive_mode == 'abs' else np.maximum(total, 0.0)
        else:
            reward = d_enemy + deaths - d_ally
        return np.where(active, reward, 0.0)

    @staticmethod
    def _side(
        prev: Side, cur: Side, active: np.ndarray, dead: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute `(died_this_step, tracked, prev_hp_plus_shield, delta)` `[B, N]` for one side
        and update `dead` in place."""
        track = active[:, None] & ~dead
        prev_hp = prev.health + prev.shield
        died = track & (cur.health == 0)
        dead |= died
        return died, track, prev_hp, prev_hp - cur.health - cur.shield
