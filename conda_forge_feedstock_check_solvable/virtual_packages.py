import atexit
import functools
import os
import pathlib
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Set

import rapidjson as json

from conda_forge_feedstock_check_solvable.utils import (
    ALL_PLATFORMS,
    MAX_GLIBC_MINOR,
    MINIMUM_CUDA_VERS,
    MINIMUM_OSX_64_VERS,
    MINIMUM_OSX_ARM64_VERS,
    print_debug,
)


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
