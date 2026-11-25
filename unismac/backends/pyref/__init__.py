"""Pure-Python reference backend (`pyref`) for UniSMAC.

Each game loop executes a fixed subsystem pipeline (`pipeline.STAGES`) whose
implementations are selected via `PyrefConfig`. State arrays use `[B, U]`
struct-of-arrays NumPy buffers with deterministic seeded RNG streams.
"""

from .backend import PyrefBackend
from .config import PyrefConfig

__all__ = ['PyrefBackend', 'PyrefConfig']
