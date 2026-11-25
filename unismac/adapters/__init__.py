"""Multi-ecosystem RL adapters built strictly on top of the purified `unismac.env` kernel.

Submodules:
* `unismac.adapters.pymarl` — zero-dependency PyMARL / SMAC `MultiAgentEnv` adapter
  (`PyMARLEnv`, `UnismacEnv`, `make_pymarl_env`).
* `unismac.adapters.gymnasium` — Gymnasium 1.0+ `Env` and `VectorEnv` adapters
  (`pip install "unismac[gymnasium]"`).
* `unismac.adapters.pettingzoo` — PettingZoo `ParallelEnv` adapter with first-class
  two-sided self-play (`pip install "unismac[pettingzoo]"`).

This top-level package never imports optional third-party dependencies (`gymnasium` or
`pettingzoo`); import the corresponding submodule explicitly to use those adapters.
"""

from __future__ import annotations

from .pymarl import PyMARLEnv, UnismacEnv, make_pymarl_env

__all__ = ['PyMARLEnv', 'UnismacEnv', 'make_pymarl_env']
