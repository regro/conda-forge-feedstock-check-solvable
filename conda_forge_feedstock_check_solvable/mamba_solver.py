"""This module has code to use mamba to test if a given package can be solved.

The basic workflow is for yaml file in .ci_support

1. run the conda_build api to render the recipe
2. pull out the host/build and run requirements, possibly for more than one output.
3. send them to mamba to check if they can be solved.

Most of the code here is due to @wolfv in this gist,
https://gist.github.com/wolfv/cd12bd4a448c77ff02368e97ffdf495a.
"""

import atexit
import copy
import functools
import os
import pathlib
import pprint
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

import cachetools.func
import libmambapy as api
import rapidjson as json
from conda.base.context import context
from conda.models.match_spec import MatchSpec

from conda_forge_feedstock_check_solvable.mamba_utils import load_channels
from conda_forge_feedstock_check_solvable.utils import (
    ALL_PLATFORMS,
    DEFAULT_RUN_EXPORTS,
    MAX_GLIBC_MINOR,
    MINIMUM_CUDA_VERS,
    MINIMUM_OSX_64_VERS,
    MINIMUM_OSX_ARM64_VERS,
    convert_spec_to_conda_build,
    get_run_export,
    print_debug,
    print_warning,
)

pkgs_dirs = context.pkgs_dirs

PACKAGE_CACHE = api.MultiPackageCache(pkgs_dirs)

# turn off pip for python
api.Context().add_pip_as_python_dependency = False

# set strict channel priority
api.Context().channel_priority = api.ChannelPriority.kStrict


@dataclass(frozen=True)
class FakePackage:
    name: str
    version: str = "1.0"
    build_string: str = ""
    build_number: int = 0
    noarch: str = ""
    depends: FrozenSet[str] = field(default_factory=frozenset)
    timestamp: int = field(
        default_factory=lambda: int(time.mktime(time.gmtime()) * 1000),
    )

    def to_repodata_entry(self):
        out = self.__dict__.copy()
        if self.build_string:
            build = f"{self.build_string}_{self.build_number}"
        else:
            build = f"{self.build_number}"
        out["depends"] = list(out["depends"])
        out["build"] = build
        fname = f"{self.name}-{self.version}-{build}.tar.bz2"
        return fname, out


class FakeRepoData:
    def __init__(self, base_dir: pathlib.Path):
        self.base_path = base_dir
        self.packages_by_subdir: Dict[FakePackage, Set[str]] = defaultdict(set)

    @property
    def channel_url(self):
        return f"file://{str(self.base_path.absolute())}"

    def add_package(self, package: FakePackage, subdirs: Iterable[str] = ()):
        subdirs = frozenset(subdirs)
        if not subdirs:
            subdirs = frozenset(["noarch"])
        self.packages_by_subdir[package].update(subdirs)

    def _write_subdir(self, subdir):
        packages = {}
        out = {"info": {"subdir": subdir}, "packages": packages}
        for pkg, subdirs in self.packages_by_subdir.items():
            if subdir not in subdirs:
                continue
            fname, info_dict = pkg.to_repodata_entry()
            info_dict["subdir"] = subdir
            packages[fname] = info_dict

        (self.base_path / subdir).mkdir(exist_ok=True)
        (self.base_path / subdir / "repodata.json").write_text(json.dumps(out))

    def write(self):
        all_subdirs = ALL_PLATFORMS.copy()
        all_subdirs.add("noarch")
        for subdirs in self.packages_by_subdir.values():
            all_subdirs.update(subdirs)

        for subdir in all_subdirs:
            self._write_subdir(subdir)

        print_debug("Wrote fake repodata to %s", self.base_path)
        import glob

        for filename in glob.iglob(str(self.base_path / "**"), recursive=True):
            print_debug(filename)
        print_debug("repo: %s", self.channel_url)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write()


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
                rx = get_run_export(link_tuple[0], link_tuple[1])
                for key in rx:
                    rx[key] = {v for v in rx[key] if v not in ign_rex}
                for key in DEFAULT_RUN_EXPORTS:
                    run_exports[key] |= rx[key]

        return run_exports


@cachetools.func.ttl_cache(maxsize=8, ttl=60)
def _mamba_factory(channels, platform):
    return MambaSolver(list(channels), platform)


@functools.lru_cache(maxsize=1)
def virtual_package_repodata():
    # TODO: we might not want to use TemporaryDirectory
    import shutil

    # tmp directory in github actions
    runner_tmp = os.environ.get("RUNNER_TEMP")
    tmp_dir = tempfile.mkdtemp(dir=runner_tmp)

    if not runner_tmp:
        # no need to bother cleaning up on CI
        def clean():
            shutil.rmtree(tmp_dir, ignore_errors=True)

        atexit.register(clean)

    tmp_path = pathlib.Path(tmp_dir)
    repodata = FakeRepoData(tmp_path)

    # glibc
    for glibc_minor in range(12, MAX_GLIBC_MINOR + 1):
        repodata.add_package(FakePackage("__glibc", "2.%d" % glibc_minor))

    # cuda - get from cuda-version on conda-forge
    try:
        cuda_pkgs = json.loads(
            subprocess.check_output(
                "CONDA_SUBDIR=linux-64 conda search cuda-version -c conda-forge --json",
                shell=True,
                text=True,
                stderr=subprocess.PIPE,
            )
        )
        cuda_vers = [pkg["version"] for pkg in cuda_pkgs["cuda-version"]]
    except Exception:
        cuda_vers = []
    # extra hard coded list to make sure we don't miss anything
    cuda_vers += MINIMUM_CUDA_VERS
    cuda_vers = set(cuda_vers)
    for cuda_ver in cuda_vers:
        repodata.add_package(FakePackage("__cuda", cuda_ver))

    for osx_ver in MINIMUM_OSX_64_VERS:
        repodata.add_package(FakePackage("__osx", osx_ver), subdirs=["osx-64"])
    for osx_ver in MINIMUM_OSX_ARM64_VERS:
        repodata.add_package(
            FakePackage("__osx", osx_ver), subdirs=["osx-arm64", "osx-64"]
        )

    repodata.add_package(
        FakePackage("__win", "0"),
        subdirs=list(subdir for subdir in ALL_PLATFORMS if subdir.startswith("win")),
    )
    repodata.add_package(
        FakePackage("__linux", "0"),
        subdirs=list(subdir for subdir in ALL_PLATFORMS if subdir.startswith("linux")),
    )
    repodata.add_package(
        FakePackage("__unix", "0"),
        subdirs=list(
            subdir for subdir in ALL_PLATFORMS if not subdir.startswith("win")
        ),
    )
    repodata.write()

    return repodata.channel_url
