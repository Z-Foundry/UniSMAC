"""StarCraft II backend (requires optional ``unismac[sc2]`` dependency and local SC2 client).

Importing this package does not import PySC2; runtime setup and SC2 launch occur on first
``reset``.
"""

from .backend import CAPABILITIES, Sc2Backend

__all__ = ['CAPABILITIES', 'Sc2Backend']
