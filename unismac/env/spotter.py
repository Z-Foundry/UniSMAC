"""SMACv2 spotter gating (`obs_enemies` / `enemy_tags`), shapes `[B, E, A]` / `[B, E]`.

Upstream `get_obs_agent` iterates sequentially over agents and enemies with side effects
(iteration order is part of the semantics):

1. When enemy `e` is within agent `a`'s sight range (or FOV cone) and `enemy_tags[e]` is unset:
   set `obs_enemies[e, a] = 1`, `enemy_tags[e] = a`, and for each other agent `a'` (ascending
   index) draw `rand() < prob_obs_enemy`, setting `obs_enemies[e, a'] = 1` on success. Does not
   check whether the enemy is alive.
2. Agent `a` reads `obs_enemies[e, a]` as its visibility gate (`gate[a, e]`): 0 hides the enemy.
3. With `quirks.enemy_block_reset`, a 0 gate clears the entire enemy feature block written so far.

Agent processing order (when `quirks.obs_shuffle` is enabled) and spotter draws both come from
the `obs_shuffle` RNG stream; computed once per frame and cached (unlike upstream, which resamples
on repeated `get_obs` calls). After each step, dead ally `a` clears any enemy tagged by it
(`tag=None`, `obs_enemies[e, :] = 0`, `obs_enemies[:, a] = 0`); dead enemies clear their own tag
and `obs_enemies[e, :] = 0`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = ['Spotter']

NO_TAG = -1


class Spotter:
    def __init__(self, batch_size: int, n_agents: int, n_enemies: int, prob: float) -> None:
        self.n_agents = n_agents
        self.n_enemies = n_enemies
        self.prob = float(prob)
        self.seen = np.zeros((batch_size, n_enemies, n_agents), bool)  # obs_enemies[e, a]
        self.tag = np.full((batch_size, n_enemies), NO_TAG, np.int64)  # enemy_tags[e]
        # Last gate values read by agent a from obs_enemies[e, a] ([B, A, E]).
        self.gate = np.ones((batch_size, n_agents, n_enemies), bool)

    def reset(self, env_ids: Sequence[int]) -> None:
        ids = list(env_ids)
        self.seen[ids] = False
        self.tag[ids] = NO_TAG
        self.gate[ids] = True

    def after_step(
        self, ally_dead: np.ndarray, enemy_dead: np.ndarray, rows: Iterable[int]
    ) -> None:
        """Update spotter state for `rows` given `ally_dead` (`bool[B, A]`) and `enemy_dead`
        (`bool[B, E]`, health == 0)."""
        for b in rows:
            for a in np.flatnonzero(ally_dead[b]):
                for e in range(self.n_enemies):
                    if self.tag[b, e] == a:
                        self.tag[b, e] = NO_TAG
                        self.seen[b, e, :] = False
                        self.seen[b, :, a] = False
            dead_e = np.flatnonzero(enemy_dead[b])
            self.tag[b, dead_e] = NO_TAG
            self.seen[b, dead_e, :] = False

    def observe(
        self,
        b: int,
        order: Sequence[int],
        raw: np.ndarray,
        alive: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Update env `b` across agents in `order` and write `self.gate[b]`.

        `raw` (`bool[A, E]`): enemy inside agent sight range / cone (ignoring enemy alive status);
        `alive` (`bool[A]`): ally alive flags.
        """
        seen, tag = self.seen[b], self.tag[b]
        gate = np.ones((self.n_agents, self.n_enemies), bool)
        for a in order:
            if not alive[a]:
                continue  # Upstream: dead agents skip the per-enemy loop
            for e in range(self.n_enemies):
                if raw[a, e] and tag[e] == NO_TAG:
                    seen[e, a] = True
                    tag[e] = a
                    for other in range(self.n_agents):
                        if other != a and rng.random() < self.prob:
                            seen[e, other] = True
                gate[a, e] = seen[e, a]
        self.gate[b] = gate
