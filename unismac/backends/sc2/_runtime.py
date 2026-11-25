"""StarCraft II runtime helpers: locate `SC2PATH`, Python 3.11+ shim, absl flags, and maps."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

__all__ = [
    'map_file',
    'map_sha256',
    'patch_random_shuffle',
    'prepare',
    'sc2path',
]


def sc2path() -> Path:
    """Locate the StarCraft II installation directory and export `SC2PATH` for PySC2."""
    candidates: list[Path] = []
    if os.environ.get('SC2PATH'):
        candidates.append(Path(os.environ['SC2PATH']))
    else:
        candidates.append(Path('~/StarCraftII'))
        if os.name == 'nt':
            candidates.append(Path('C:/Program Files (x86)/StarCraft II'))
        elif os.uname().sysname == 'Darwin':
            candidates.append(Path('/Applications/StarCraft II'))
    for cand in candidates:
        path = cand.expanduser().resolve()
        if (path / 'Versions').is_dir():
            os.environ['SC2PATH'] = str(path)
            return path
    tried = ', '.join(str(c) for c in candidates)
    raise RuntimeError(
        f'StarCraft II installation not found (missing Versions/). Tried: {tried}. '
        'Please set the SC2PATH environment variable.'
    )


def patch_random_shuffle() -> bool:
    """Restore the two-argument `random.shuffle(seq, rand)` signature required by PySC2 4.0.0."""
    if getattr(random.shuffle, '__pysc2_compat__', False):
        return False
    _shuffle = random.shuffle

    def shuffle(x, rand=None):
        del rand
        _shuffle(x)

    shuffle.__pysc2_compat__ = True  # type: ignore[attr-defined]
    shuffle.__doc__ = _shuffle.__doc__
    random.shuffle = shuffle
    return True


def ensure_flags_parsed() -> None:
    """Mark absl flags parsed with defaults when running outside `absl.app.run`."""
    import pysc2.lib.sc_process  # noqa: F401
    import pysc2.run_configs  # noqa: F401
    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()


def prepare() -> Path:
    """Prepare the PySC2 runtime: apply shims, locate `SC2PATH`, and initialize absl flags."""
    patch_random_shuffle()
    path = sc2path()
    ensure_flags_parsed()
    return path


def map_file(map_id: str) -> Path:
    """Resolve the `.SC2Map` path for `map_id` (relative to `Maps/`, without extension)."""
    path = sc2path() / 'Maps' / f'{map_id}.SC2Map'
    if not path.is_file():
        raise FileNotFoundError(f'Map {map_id!r} not found: {path}')
    return path


def map_sha256(map_id: str) -> str:
    return hashlib.sha256(map_file(map_id).read_bytes()).hexdigest()
