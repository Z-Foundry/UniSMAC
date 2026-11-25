"""UniSMAC: backend-agnostic StarCraft II multi-agent benchmark environments."""

from .contract import CONTRACT_VERSION
from .env import UnismacEnv, VecEnv, make_env, make_vec_env

__all__ = ['CONTRACT_VERSION', 'UnismacEnv', 'VecEnv', 'make_env', 'make_vec_env']
__version__ = '0.1.0'
