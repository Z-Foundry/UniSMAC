"""Counter-based deterministic RNG for `pyref`.

Each sample is uniquely determined by `(seed, stream, game_loop, slot)`, independent of call
order: Philox4x64-10 uses key `(seed, stream_index)` and counter `(slot, game_loop, 0, 0)`,
taking the upper 53 bits of the first 64-bit output divided by 2^53 to yield a uniform float
in [0, 1). Native backends reproduce the same values bit-for-bit from raw Philox bits.

Kept separate from `unismac.env.semantics.rng`: the backend does not depend on the env layer,
and both use distinct stream names.
"""

from __future__ import annotations

import numpy as np

# Stream name -> index. Only append new streams; never renumber existing ones.
STREAMS = ('spawn_facing', 'cooldown', 'scan_phase', 'update_order')
_MASK64 = (1 << 64) - 1


def uniform(seed: int, stream: str, game_loop: int, slot: int) -> float:
    """Return a deterministic uniform float in [0, 1) for the given key."""
    key = np.array([int(seed) & _MASK64, STREAMS.index(stream)], np.uint64)
    counter = np.array([int(slot), int(game_loop), 0, 0], np.uint64)
    raw = np.random.Philox(key=key, counter=counter).random_raw(1)
    return float(int(raw[0]) >> 11) / float(1 << 53)
