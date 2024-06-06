import asyncio
import copy
import datetime
import os
import pprint
import textwrap
from functools import lru_cache
from typing import List

from rattler import Channel, MatchSpec, Platform, RepoDataRecord, solve

from conda_forge_feedstock_check_solvable.utils import (
    DEFAULT_RUN_EXPORTS,
    get_run_exports,
    print_debug,
    print_warning,
)


class RattlerSolver:
    """Run the rattler solver (resolvo).

    Parameters
    ----------
    channels : list of str
        A list of the channels (e.g., `[conda-forge]`, etc.)
    platform : str
        The platform to be used (e.g., `linux-64`).

    Example
    -------
    >>> solver = RattlerSolver(['conda-forge', 'conda-forge'], "linux-64")
    >>> solver.solve(["xtensor 0.18"])
    """

    def __init__(self, channels, platform_arch) -> None:
        self.channels = channels
        _channels = []
        for c in channels:
            if c == "defaults":
                _channels.append("https://repo.anaconda.com/pkgs/main")
                _channels.append("https://repo.anaconda.com/pkgs/r")
                _channels.append("https://repo.anaconda.com/pkgs/msys2")
            else:
                _channels.append(c)
        self._channels = [Channel(c) for c in _channels]
        self.platform_arch = platform_arch
        self._platforms = [Platform(self.platform_arch), Platform("noarch")]

    def solve(
        self,
        specs: List[str],
        get_run_exports: bool = False,
        ignore_run_exports_from: List[str] = None,
        ignore_run_exports: List[str] = None,
        constraints=None,
        timeout: int | None = None,
    ):
        """Solve given a set of specs.

        Parameters
        ----------
        specs : list of str
            A list of package specs. You can use `conda.models.match_spec.MatchSpec`
            to get them to the right form by calling
            `MatchSpec(mypec).conda_build_form()`
        get_run_exports : bool, optional
            If True, return run exports else do not.
        ignore_run_exports_from : list, optional
            A list of packages from which to ignore the run exports.
        ignore_run_exports : list, optional
            A list of things that should be ignore in the run exports.
        constraints : list, optional
            A list of package specs to apply as constraints to the solve.
            These packages are not included in the solution.
        timeout : int, optional
            The time in seconds to wait for the solver to finish before giving up.

        Returns
        -------
        solvable : bool
            True if the set of specs has a solution, False otherwise.
        err : str
            The errors as a string. If no errors, is None.
        solution : list of str
            A list of concrete package specs for the env.
        run_exports : dict of list of str
            A dictionary with the weak and strong run exports for the packages.
            Only returned if get_run_exports is True.
        """
        ignore_run_exports_from = ignore_run_exports_from or []
        ignore_run_exports = ignore_run_exports or []
        success = False
        err = None
        run_exports = copy.deepcopy(DEFAULT_RUN_EXPORTS)

        try:
            _specs = [MatchSpec(s) for s in specs]
            _constraints = [MatchSpec(c) for c in constraints] if constraints else None

            print_debug(
                "RATTLER running solver for specs \n\n%s\n", pprint.pformat(_specs)
            )

            if timeout is not None:
                timeout = datetime.timedelta(seconds=timeout)

            solution = asyncio.run(
                solve(
                    channels=self._channels,
                    specs=_specs,
                    platforms=self._platforms,
                    timeout=timeout,
                    constraints=_constraints,
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
                "RATTLER failed to solve specs \n\n%s\n\nwith "
                "constraints \n\n%s\n\nfor channels "
                "\n\n%s\n\non platform "
                "\n\n%s\n\nThe reported errors are:\n\n%s\n",
                textwrap.indent(pprint.pformat(specs), "    "),
                textwrap.indent(pprint.pformat(constraints), "    "),
                textwrap.indent(pprint.pformat(self.channels), "    "),
                textwrap.indent(pprint.pformat(self.platform_arch), "    "),
                textwrap.indent(err, "    "),
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


@lru_cache(maxsize=128)
def rattler_solver_factory(channels, platform):
    return RattlerSolver(list(channels), platform)
