"""SC2 process and match session management.

``Session`` launches 1 or 2 SC2 processes based on player configuration:

* 1 ``AgentPlayer`` + 1 ``ComputerPlayer``: single process, ``Participant + Computer``
  (for SMAC / SMACv2).
* 2 ``AgentPlayer``: two processes, dual ``Participant``, with ports allocated via
  ``pysc2.lib.portspicker`` (for SMAC-Hard). With ``realtime=False``, both sides wait on each
  other's ``step``, so ``step`` / ``observe`` / ``actions`` always run in parallel via
  ``RunParallel``.

Note: ``SMAC_Maps/*`` only exposes 1 user slot; joining with two Participants fails with
``ResponseJoinGame.Error.NoSpaceForUser``. Use ``new_maps/*`` or ``Melee/*`` for dual-agent play.

Maps are loaded directly via ``LocalMap(map_path, map_data)``, bypassing ``pysc2.maps`` registry.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from typing import Any

from ...contract import AgentPlayer, ComputerPlayer, ContractError, Player
from . import _runtime

# Values are protobuf enum integers, typed as Any for generated message constructors.
_RACES: dict[str, Any] = {'random': 4, 'terran': 1, 'zerg': 2, 'protoss': 3}  # common_pb2.Race
_DIFFICULTIES: dict[str, Any] = {
    'very_easy': 1,
    'easy': 2,
    'medium': 3,
    'medium_hard': 4,
    'hard': 5,
    'harder': 6,
    'very_hard': 7,
    'cheat_vision': 8,
    'cheat_money': 9,
    'cheat_insane': 10,
}  # sc2api_pb2.Difficulty


def _shutdown(procs: list[Any], controllers: list[Any], ports: list[int]) -> None:
    """Shut down processes and return ports; idempotent. Shared by close() and weakref.finalize."""
    from pysc2.lib import portspicker

    for c in controllers:
        try:
            c.quit()
        except Exception:  # noqa: BLE001 - process may have already exited
            pass
    for p in procs:
        try:
            p.close()
        except Exception:  # noqa: BLE001
            pass
    if ports:
        portspicker.return_ports(ports)
    procs.clear()
    controllers.clear()
    ports.clear()


class Session:
    """An SC2 match. ``controllers[i]`` corresponds to 1-based player ``agent_ids[i]``."""

    def __init__(
        self,
        map_id: str,
        players: Sequence[Player],
        *,
        seed: int | None = None,
    ) -> None:
        _runtime.prepare()
        from pysc2 import run_configs
        from pysc2.lib import portspicker, run_parallel
        from s2clientprotocol import sc2api_pb2 as sc_pb

        players = tuple(players)
        n_agents = sum(isinstance(p, AgentPlayer) for p in players)
        if len(players) != 2 or n_agents == 0:
            raise ContractError(
                f'SC2 backend requires 2 players with at least 1 AgentPlayer, got {players!r}'
            )
        self.map_id = map_id
        self.players = players
        self.agent_ids = [i + 1 for i, p in enumerate(players) if isinstance(p, AgentPlayer)]

        self._procs: list[Any] = []
        self.controllers: list[Any] = []
        self._ports: list[int] = []
        self._finalizer = weakref.finalize(
            self, _shutdown, self._procs, self.controllers, self._ports
        )
        self._parallel = run_parallel.RunParallel()

        run_config = run_configs.get()
        map_path = str(_runtime.map_file(map_id))
        try:
            if n_agents == 2:
                self._ports.extend(portspicker.pick_unused_ports(4))
            for _ in range(n_agents):
                proc = run_config.start(want_rgb=False)
                self._procs.append(proc)
                self.controllers.append(proc.controller)

            create = sc_pb.RequestCreateGame(
                local_map=sc_pb.LocalMap(map_path=map_path, map_data=run_config.map_data(map_path)),
                realtime=False,
            )
            if seed is not None:
                create.random_seed = int(seed) & 0x7FFFFFFF
            for p in players:
                if isinstance(p, ComputerPlayer):
                    create.player_setup.add(
                        type=sc_pb.Computer,
                        race=_RACES[p.race],
                        difficulty=_DIFFICULTIES[p.difficulty],
                    )
                else:
                    create.player_setup.add(type=sc_pb.Participant)
            self.controllers[0].create_game(create)

            options = sc_pb.InterfaceOptions(raw=True, score=False)
            joins = []
            for pid in self.agent_ids:
                player = players[pid - 1]
                join = sc_pb.RequestJoinGame(race=_RACES[player.race], options=options)
                if n_agents == 2:
                    join.shared_port = 0  # unused
                    join.server_ports.game_port = self._ports[0]
                    join.server_ports.base_port = self._ports[1]
                    join.client_ports.add(game_port=self._ports[2], base_port=self._ports[3])
                joins.append(join)
            self._parallel.run(
                [(c.join_game, j) for c, j in zip(self.controllers, joins, strict=True)]
            )

            # Verify each controller's assigned player ID (assigned by join order).
            obs = self.observe()
            got = [o.observation.player_common.player_id for o in obs]
            if sorted(got) != self.agent_ids:
                raise RuntimeError(f'Player ID mismatch: expected {self.agent_ids}, got {got}')
            order = sorted(range(len(got)), key=lambda i: got[i])
            self.controllers[:] = [self.controllers[i] for i in order]
            self._procs[:] = [self._procs[i] for i in order]

            self.game_info = self.controllers[0].game_info()
            self.data_raw = self.controllers[0].data_raw()
            self.ping = self.controllers[0].ping()
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------ Core operations

    @property
    def n_agents(self) -> int:
        return len(self.controllers)

    @property
    def has_computer(self) -> bool:
        return self.n_agents < len(self.players)

    @property
    def version(self) -> str:
        """Game version string reported by SC2 ping (e.g., ``'4.10.0'``)."""
        return str(self.ping.game_version)

    def _run(self, calls: Sequence[Callable[..., Any]]) -> list[Any]:
        if len(calls) == 1:
            return [calls[0]()]
        return self._parallel.run(calls)

    def observe(self, disable_fog: bool = False) -> list[Any]:
        """Return ``ResponseObservation`` per controller in ``agent_ids`` order."""
        return self._run([lambda c=c: c.observe(disable_fog=disable_fog) for c in self.controllers])

    def step(self, n_loops: int) -> None:
        """Advance all controllers together by ``n_loops`` game loops."""
        self._run([lambda c=c: c.step(n_loops) for c in self.controllers])

    def actions(self, requests: Sequence[Any | None]) -> list[Any]:
        """Send ``RequestAction`` per controller (``None`` skips sending)."""
        calls = [
            (lambda c=c, r=r: c.actions(r))
            for c, r in zip(self.controllers, requests, strict=True)
            if r is not None and len(r.actions) > 0
        ]
        return self._run(calls) if calls else []

    def debug(self, commands: Sequence[Any]) -> None:
        """Send global debug commands via Player 1's controller."""
        if commands:
            self.controllers[0].debug(list(commands))

    def close(self) -> None:
        self._finalizer()
        try:
            self._parallel.shutdown()
        except Exception:  # noqa: BLE001
            pass

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive
