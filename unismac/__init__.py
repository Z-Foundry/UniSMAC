"""UniSMAC: backend-agnostic StarCraft II multi-agent benchmark environments."""

from .contract import CONTRACT_VERSION
from .env import UnismacEnv, VecEnv, make_env, make_vec_env
from .profiles import STANDARD_PROFILE, list_profiles

__all__ = [
    'CONTRACT_VERSION',
    'STANDARD_PROFILE',
    'UnismacEnv',
    'VecEnv',
    'list_profiles',
    'make_env',
    'make_vec_env',
]
__version__ = '0.1.1'
