"""Rust-accelerated backend (`ZgameBackend`) backed by the companion `zfoundry-zgame` package.

`ZgameBackend` implements the UniSMAC `Backend` contract using the compiled
`zgame._zgame` extension module. Call `build_info()` to inspect the loaded
extension metadata (version, SHA-256 digest, commit, dirty flag, and build time).
"""

from .backend import CAPABILITIES, ZgameBackend, build_info, capabilities_for, config_options

__all__ = ['CAPABILITIES', 'ZgameBackend', 'build_info', 'capabilities_for', 'config_options']
