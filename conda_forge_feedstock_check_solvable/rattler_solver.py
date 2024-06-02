import asyncio
import copy
import os
import pprint
from typing import List

import cachetools.func
from rattler import Channel, MatchSpec, Platform, RepoDataRecord, solve

from conda_forge_feedstock_check_solvable.utils import (
    DEFAULT_RUN_EXPORTS,
    get_run_exports,
    print_debug,
    print_warning,
)


class RattlerSolver:
    def __init__(self, channels, platform_arch) -> None:
        _channels = []
        for c in channels:
            if c == "defaults":
                _channels.append("https://repo.anaconda.com/pkgs/main")
                _channels.append("https://repo.anaconda.com/pkgs/r")
                _channels.append("https://repo.anaconda.com/pkgs/msys2")
            else:
                _channels.append(c)
        self.channels = [Channel(c) for c in _channels]
        self.platform_arch = platform_arch
        self.platforms = [Platform(self.platform_arch), Platform("noarch")]

    def solve(
        self,
        specs: List[str],
        get_run_exports: bool = False,
        ignore_run_exports_from: List[str] = None,
        ignore_run_exports: List[str] = None,
        constraints=None,
    ):
        ignore_run_exports_from = ignore_run_exports_from or []
        ignore_run_exports = ignore_run_exports or []
        success = False
        err = None
        run_exports = copy.deepcopy(DEFAULT_RUN_EXPORTS)

        try:
            _specs = [MatchSpec(s) for s in specs]

            print_debug(
                "RATTLER running solver for specs \n\n%s\n", pprint.pformat(_specs)
            )

            solution = asyncio.run(
                solve(
                    channels=self.channels,
                    specs=_specs,
                    platforms=self.platforms,
                    # virtual_packages=self.virtual_packages,
                )
            )
            success = True
            str_solution = [
                f"{record.name.normalized} {record.version} {record.build}"
                for record in solution
            ]

            if get_run_exports:
                run_exports = self._get_run_exports(
                    solution,
                    _specs,
                    [MatchSpec(igrf) for igrf in ignore_run_exports_from],
                    [MatchSpec(igr) for igr in ignore_run_exports],
                )

        except Exception as e:
            err = str(e)
            print_warning(
                "RATTLER failed to solve specs \n\n%s\n\nfor channels "
                "\n\n%s\n\nThe reported errors are:\n\n%s\n",
                pprint.pformat(_specs),
                pprint.pformat(self.channels),
                err,
            )
            success = False
            run_exports = copy.deepcopy(DEFAULT_RUN_EXPORTS)
            str_solution = None

        if get_run_exports:
            return success, err, str_solution, run_exports
        else:
            return success, err, str_solution

    def _get_run_exports(
        self,
        repodata_records: List[RepoDataRecord],
        _specs: List[MatchSpec],
        ignore_run_exports_from: List[MatchSpec],
        ignore_run_exports: List[MatchSpec],
    ):
        """Given a set of repodata records, produce a
        dict with the weak and strong run exports for the packages.

        We only look up export data for things explicitly listed in the original
        specs.
        """
        names = {s.name for s in _specs}
        ign_rex_from = {s.name for s in ignore_run_exports_from}
        ign_rex = {s.name for s in ignore_run_exports}
        run_exports = copy.deepcopy(DEFAULT_RUN_EXPORTS)
        for record in repodata_records:
            lt_name = record.name
            if lt_name in names and lt_name not in ign_rex_from:
                channel_url = record.channel
                subdir = record.subdir
                file_name = record.file_name
                rx = get_run_exports(os.path.join(channel_url, subdir), file_name)
                for key in rx:
                    rx[key] = {v for v in rx[key] if v not in ign_rex}
                for key in DEFAULT_RUN_EXPORTS:
                    run_exports[key] |= rx[key]

        return run_exports


@cachetools.func.ttl_cache(maxsize=8, ttl=60)
def rattler_solver_factory(channels, platform):
    return RattlerSolver(list(channels), platform)
