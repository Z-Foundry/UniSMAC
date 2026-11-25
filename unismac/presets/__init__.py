"""Named presets pairing an `EnvConfig` with its corresponding upstream benchmark.

* `smac`: reproduces upstream SMAC (SC2 backend, `SMAC_Maps`, `map_trigger` spawning, built-in
  `very_hard` AI, `timestep_pad` and `terrain_height_flip` enabled).
* `smac-portable`: uses only portable contract capabilities for pyref and SC2 (`explicit`
  spawning, standard units, dual-agent setup, `OpeningOrders` opponent; uses `new_maps` on SC2).
* `smacv2` / `smacv2-portable`: corresponding presets for SMACv2.
* `smac-hard` / `smac-hard-portable`: SMAC-Hard (`new_maps`, scripted opponents); see `smac_hard`.

Presets are defined in Python without YAML dependencies.
"""

from __future__ import annotations

from .registry import PRESETS, Preset, get, names

__all__ = ['PRESETS', 'Preset', 'get', 'names']
