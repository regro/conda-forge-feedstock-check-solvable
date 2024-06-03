"""This module has code to use mamba to test if a given package can be solved.

The basic workflow is for yaml file in .ci_support

1. run the conda_build api to render the recipe
2. pull out the host/build and run requirements, possibly for more than one output.
3. send them to mamba to check if they can be solved.

Most of the code here is due to @wolfv in this gist,
https://gist.github.com/wolfv/cd12bd4a448c77ff02368e97ffdf495a.
"""

import copy
import pprint
from typing import List, Tuple

import cachetools.func
import libmambapy as api
import rapidjson as json
from conda.base.context import context
from conda.models.match_spec import MatchSpec

from conda_forge_feedstock_check_solvable.mamba_utils import load_channels
from conda_forge_feedstock_check_solvable.utils import (
    DEFAULT_RUN_EXPORTS,
    convert_spec_to_conda_build,
    get_run_exports,
    print_debug,
    print_warning,
)

pkgs_dirs = context.pkgs_dirs

PACKAGE_CACHE = api.MultiPackageCache(pkgs_dirs)

# turn off pip for python
api.Context().add_pip_as_python_dependency = False

# set strict channel priority
api.Context().channel_priority = api.ChannelPriority.kStrict


class MambaSolver:
    """Run the mamba solver.

    Parameters
    ----------
    channels : list of str
        A list of the channels (e.g., `[conda-forge]`, etc.)
    platform : str
        The platform to be used (e.g., `linux-64`).

    Example
    -------
    >>> solver = MambaSolver(['conda-forge', 'conda-forge'], "linux-64")
    >>> solver.solve(["xtensor 0.18"])
    """

    def __init__(self, channels, platform):
        self.channels = channels
        self.platform = platform
        self.pool = api.Pool()

        self.repos = []
        self.index = load_channels(
            self.pool,
            self.channels,
            self.repos,
            platform=platform,
            has_priority=True,
        )
        for repo in self.repos:
            # need set_installed for add_pin, not sure why
            repo.set_installed()

    def solve(
        self,
        specs,
        get_run_exports=False,
        ignore_run_exports_from=None,
        ignore_run_exports=None,
        constraints=None,
        timeout=None,
    ) -> Tuple[bool, List[str]]:
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
            Ignored by mamba.

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
        if timeout is not None:
            raise RuntimeError("The `timeout` keyword is not supported by mamba!")

        ignore_run_exports_from = ignore_run_exports_from or []
        ignore_run_exports = ignore_run_exports or []

        solver_options = [(api.SOLVER_FLAG_ALLOW_DOWNGRADE, 1)]
        solver = api.Solver(self.pool, solver_options)

        _specs = [convert_spec_to_conda_build(s) for s in specs]
        _constraints = [convert_spec_to_conda_build(s) for s in constraints or []]

        print_debug(
            "MAMBA running solver for specs \n\n%s\nconstraints: %s\n",
            pprint.pformat(_specs),
            pprint.pformat(_constraints),
        )
        for constraint in _constraints:
            solver.add_pin(constraint)

        solver.add_jobs(_specs, api.SOLVER_INSTALL)
        success = solver.solve()

        err = None
        if not success:
            print_warning(
                "MAMBA failed to solve specs \n\n%s\n\nfor channels "
                "\n\n%s\n\nThe reported errors are:\n\n%s\n",
                pprint.pformat(_specs),
                pprint.pformat(self.channels),
                solver.explain_problems(),
            )
            err = solver.explain_problems()
            solution = None
            run_exports = copy.deepcopy(DEFAULT_RUN_EXPORTS)
        else:
            t = api.Transaction(
                self.pool,
                solver,
                PACKAGE_CACHE,
            )

            solution = []
            _, to_link, _ = t.to_conda()
            for _, _, jdata in to_link:
                data = json.loads(jdata)
                solution.append(
                    " ".join([data["name"], data["version"], data["build"]]),
                )

            if get_run_exports:
                print_debug(
                    "MAMBA getting run exports for \n\n%s\n",
                    pprint.pformat(solution),
                )
                run_exports = self._get_run_exports(
                    to_link,
                    _specs,
                    ignore_run_exports_from,
                    ignore_run_exports,
                )

        if get_run_exports:
            return success, err, solution, run_exports
        else:
            return success, err, solution

    def _get_run_exports(
        self,
        link_tuples,
        _specs,
        ignore_run_exports_from,
        ignore_run_exports,
    ):
        """Given tuples of (channel, file, json repodata shard) produce a
        dict with the weak and strong run exports for the packages.

        We only look up export data for things explicitly listed in the original
        specs.
        """
        names = {MatchSpec(s).get_exact_value("name") for s in _specs}
        ign_rex_from = {
            MatchSpec(s).get_exact_value("name") for s in ignore_run_exports_from
        }
        ign_rex = {MatchSpec(s).get_exact_value("name") for s in ignore_run_exports}
        run_exports = copy.deepcopy(DEFAULT_RUN_EXPORTS)
        for link_tuple in link_tuples:
            lt_name = json.loads(link_tuple[-1])["name"]
            if lt_name in names and lt_name not in ign_rex_from:
                rx = get_run_exports(link_tuple[0], link_tuple[1])
                for key in rx:
                    rx[key] = {v for v in rx[key] if v not in ign_rex}
                for key in DEFAULT_RUN_EXPORTS:
                    run_exports[key] |= rx[key]

        return run_exports


@cachetools.func.ttl_cache(maxsize=8, ttl=600)
def mamba_solver_factory(channels, platform):
    return MambaSolver(list(channels), platform)
