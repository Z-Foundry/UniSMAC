"""Counter-based RNG streams: Philox streams keyed by `(seed, stream, env_id, episode, frame)`.

* `(seed, stream, env_id)` determines the Philox key, and `(episode, frame)` occupies the high
  counter words, so environments in a batch are independent and any `(env, episode, frame)` can
  be replayed regardless of call order.
* Stream names: `generators` (procedural episode generation, 1 per episode), `attack`
  (`stochastic_attack`, 1 per step), `obs_shuffle` (observation ordering and spotter draws, 1 per
  frame), `opponent` (scripted opponent), `obs_shuffle_p2` (Player 2 observation ordering and
  spotter draws in two-sided mode), and `smac_trigger` (native SMAC trigger random points, e.g.
  `2m_vs_1z`, 1 per episode).
* Draw order within a stream is fixed by each caller (see `generators` and `spotter`).

The Rust side reproduces Philox4x64-10 raw bits bitwise (`zgame-core` `rng::PhiloxStream`,
providing `random()` / `uniform()` doubles); `Streams` packs a batch of keys and counters into
arrays for Rust consumption.
"""

from __future__ import annotations

from collections.abc import MutableSequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ['STREAMS', 'Streams', 'entropy_seed', 'shuffle', 'stream', 'stream_key']

STREAMS = ('generators', 'attack', 'obs_shuffle', 'opponent', 'obs_shuffle_p2', 'smac_trigger')
_MASK64 = (1 << 64) - 1


def entropy_seed() -> int:
    """Draw a system-entropy seed when `seed=None` (subsequent draws remain reproducible)."""
    return int(np.random.SeedSequence().generate_state(1, np.uint64)[0])


def stream_key(seed: int, name: str, env_id: int) -> tuple[int, int]:
    """Return the Philox key (two 64-bit words) for `(seed, name, env_id)`."""
    if name not in STREAMS:
        raise ValueError(f'Unknown RNG stream {name!r}; expected one of {STREAMS}')
    if env_id < 0:
        raise ValueError('env_id / episode / frame must be non-negative')
    return int(seed) & _MASK64, STREAMS.index(name) << 32 | int(env_id)


def stream(seed: int, name: str, env_id: int, episode: int, frame: int = 0) -> np.random.Generator:
    """Return a fresh `Generator` positioned at the start of the stream for this key."""
    key = stream_key(seed, name, env_id)
    if min(episode, frame) < 0:
        raise ValueError('env_id / episode / frame must be non-negative')
    return _generator(key, int(frame), int(episode))


def _generator(key: tuple[int, int], frame: int, episode: int) -> np.random.Generator:
    k = np.array(key, np.uint64)
    counter = np.array([0, 0, frame, episode], np.uint64)
    return np.random.Generator(np.random.Philox(key=k, counter=counter))


@dataclass(frozen=True)
class Streams:
    """Batch of identically named RNG streams.

    Env `b` has key `(key0[b], key1[b])` (`stream_key`) and counter
    `(0, 0, frame[b], episode[b])`, each `uint64[B]`. `streams(b)` returns the `Generator` for
    env `b` (matching `stream(...)`); `table()` returns `uint64[4, B]` for Rust.
    """

    key0: np.ndarray
    key1: np.ndarray
    frame: np.ndarray
    episode: np.ndarray

    def __call__(self, b: int) -> np.random.Generator:
        key = (int(self.key0[b]), int(self.key1[b]))
        return _generator(key, int(self.frame[b]), int(self.episode[b]))

    def table(self) -> np.ndarray:
        return np.stack([self.key0, self.key1, self.frame, self.episode]).astype(np.uint64)


def shuffle(rng: np.random.Generator, items: MutableSequence[Any]) -> None:
    """Shuffle in place via `items[:] = [items[i] for i in rng.permutation(len(items))]`.

    Upstream uses stdlib `random.shuffle`; parity tests patch it with this function.
    """
    perm = rng.permutation(len(items))
    items[:] = [items[int(i)] for i in perm]
