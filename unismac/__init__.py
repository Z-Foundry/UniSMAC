"""UniSMAC: backend-agnostic StarCraft II multi-agent benchmark environments."""

from .contract import CONTRACT_VERSION
from .env import StepBatch, VecEnv, make_vec_env
from .profiles import STANDARD_PROFILE, list_profiles

__all__ = [
    'CONTRACT_VERSION',
    'STANDARD_PROFILE',
    'StepBatch',
    'VecEnv',
    'list_profiles',
    'make_vec_env',
]
__version__ = '0.2.0'
