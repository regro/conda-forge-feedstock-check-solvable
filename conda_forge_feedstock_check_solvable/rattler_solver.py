import contextlib
import functools
import glob
import io
import json
import os
import subprocess
import traceback
import wurlitzer
import conda_build.config
import conda_build.api
import conda_build.variants
import asyncio

from ruamel.yaml import YAML
from rattler import (
    solve,
    fetch_repo_data,
    Channel,
    Platform,
    MatchSpec,
    GenericVirtualPackage,
    PackageName,
    Version,
)

VERBOSITY = 1
VERBOSITY_PREFIX = {
    0: "CRITICAL",
    1: "WARNING",
    2: "INFO",
    3: "DEBUG",
}
MAX_GLIBC_MINOR = 50


def _clean_reqs(reqs, names):
    reqs = [r for r in reqs if not any(r.split(" ")[0] == nm for nm in names)]
    return reqs


def _filter_problematic_reqs(reqs):
    """There are some reqs that have issues when used in certain contexts"""
    problem_reqs = {
        # This causes a strange self-ref for arrow-cpp
        "parquet-cpp",
    }
    reqs = [r for r in reqs if r.split(" ")[0] not in problem_reqs]
    return reqs


def apply_pins(reqs, host_req, build_req, outnames, m):
    from conda_build.render import get_pin_from_build

    pin_deps = host_req if m.is_cross else build_req

    full_build_dep_versions = {
        dep.split()[0]: " ".join(dep.split()[1:])
        for dep in _clean_reqs(pin_deps, outnames)
    }

    pinned_req = []
    for dep in reqs:
        try:
            pinned_req.append(
                get_pin_from_build(m, dep, full_build_dep_versions),
            )
        except Exception:
            # in case we couldn't apply pins for whatever
            # reason, fall back to the req
            pinned_req.append(dep)

    pinned_req = _filter_problematic_reqs(pinned_req)
    return pinned_req


def print_verb(fmt, *args, verbosity=0):
    from inspect import currentframe, getframeinfo

    frameinfo = getframeinfo(currentframe())

    if verbosity <= VERBOSITY:
        if args:
            msg = fmt % args
        else:
            msg = fmt
        print(
            VERBOSITY_PREFIX[verbosity]
            + ":"
            + __name__
            + ":"
            + "%d" % frameinfo.lineno
            + ":"
            + msg,
            flush=True,
        )


def print_debug(fmt, *args):
    print_verb(fmt, *args, verbosity=3)


def print_critical(fmt, *args):
    print_verb(fmt, *args, verbosity=0)


def print_warning(fmt, *args):
    print_verb(fmt, *args, verbosity=1)


def print_info(fmt, *args):
    print_verb(fmt, *args, verbosity=2)


@contextlib.contextmanager
def suppress_conda_build_logging():
    debug_env_var = "CONDA_FORGE_FEEDSTOCK_CHECK_SOLVABLE_DEBUG"
    suppress_logging = debug_env_var not in os.environ

    outerr = io.StringIO()

    if not suppress_logging:
        yield None
        return

    try:
        with io.StringIO() as fout, io.StringIO() as ferr, wurlitzer.pipes(
            stdout=outerr, stderr=wurlitzer.STDOUT
        ):
            yield None
    except Exception as e:
        print("EXCEPTION: captured C-level I/O: %r" % outerr.getvalue(), flush=True)
        traceback.print_exc()
        raise e


@functools.lru_cache(maxsize=1)
def virtual_packages_repodata():
    virtual_packages = []

    for glibc_minor in range(12, MAX_GLIBC_MINOR + 1):
        virtual_packages.append(
            GenericVirtualPackage(
                PackageName("__glibc"), Version(f"2.{glibc_minor}"), "0"
            )
        )

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
    cuda_vers += [
        "9.2",
        "10.0",
        "10.1",
        "10.2",
        "11.0",
        "11.1",
        "11.2",
        "11.3",
        "11.4",
        "11.5",
        "11.6",
        "11.7",
        "11.8",
        "12.0",
        "12.1",
        "12.2",
        "12.3",
        "12.4",
        "12.5",
    ]
    cuda_vers = set(cuda_vers)
    for cuda_ver in cuda_vers:
        virtual_packages.append(
            GenericVirtualPackage(PackageName("__cuda"), Version(cuda_ver), "0")
        )

    for osx_ver in [
        "10.9",
        "10.10",
        "10.11",
        "10.12",
        "10.13",
        "10.14",
        "10.15",
        "10.16",
    ]:
        virtual_packages.append(
            GenericVirtualPackage(
                PackageName("__osx"),
                Version(osx_ver),
                "0",
            ),
        )

    for osx_major in range(11, 17):
        for osx_minor in range(0, 17):
            virtual_packages.append(
                GenericVirtualPackage(
                    PackageName("__osx"),
                    Version(f"{osx_major}.{osx_minor}"),
                    "0",
                ),
            )

    for arch in [
        "x86",
        "x86_64",
        "aarch64",
        "armv6l",
        "armv7l",
        "ppc64le",
        "ppc64",
        "s390x",
        "riscv32",
        "riscv64",
    ]:
        virtual_packages.append(
            GenericVirtualPackage(
                PackageName("__archspec"),
                Version("1"),
                arch,
            ),
        )

    for pkg in ["__win", "__unix", "__linux"]:
        virtual_packages.append(
            GenericVirtualPackage(
                PackageName(pkg),
                Version("0"),
                "0",
            ),
        )

    return virtual_packages


def _check_solvable_in_subprocess(
    feedstock_dir, cache_dir, additional_channels, build_platform, verbosity, conn
):
    try:
        res = _is_recipe_solvable(
            feedstock_dir,
            cache_dir,
            additional_channels=additional_channels,
            build_platform=build_platform,
            verbosity=verbosity,
        )
        conn.send(res)
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()


def is_recipe_solvable(
    feedstock_dir,
    cache_dir,
    additional_channels=None,
    timeout=600,
    build_platform=None,
    verbosity=1,
):
    """Compute if a recipe is solvable.

    We look through each of the conda build configs in the feedstock
    .ci_support dir and test each ones host and run requirements.
    The final result is a logical AND of all of the results for each CI
    support config.

    Parameters
    ----------
    feedstock_dir : str
        The directory of the feedstock.
    additional_channels : list of str, optional
        If given, these channels will be used in addition to the main ones.
    timeout : int, optional
        If not None, then the work will be run in a separate process and
        this function will return True if the work doesn't complete before `timeout`
        seconds.
    verbosity : int
        An int indicating the level of verbosity from 0 (no output) to 3
        (gobbs of output).

    Returns
    -------
    solvable : bool
        The logical AND of the solvability of the recipe on all platforms
        in the CI scripts.
    errors : list of str
        A list of errors from mamba. Empty if recipe is solvable.
    solvable_by_variant : dict
        A lookup by variant config that shows if a particular config is solvable
    """
    if timeout:
        from multiprocessing import Process, Pipe

        parent_conn, child_conn = Pipe()
        p = Process(
            target=_check_solvable_in_subprocess,
            args=(
                feedstock_dir,
                additional_channels,
                build_platform,
                verbosity,
                child_conn,
            ),
        )
        p.start()
        if parent_conn.poll(timeout):
            res = parent_conn.recv()
            if isinstance(res, Exception):
                res = (
                    False,
                    [repr(res)],
                    {},
                )
        else:
            print_warning("RATTLER SOLVER TIMEOUT for %s", feedstock_dir)
            res = (
                True,
                [],
                {},
            )

        parent_conn.close()

        p.join(0)
        p.terminate()
        p.kill()
        try:
            p.close()
        except ValueError:
            pass
    else:
        res = _is_recipe_solvable(
            feedstock_dir,
            cache_dir,
            additional_channels=additional_channels,
            build_platform=build_platform,
            verbosity=verbosity,
        )

    return res


def _is_recipe_solvable(
    feedstock_dir,
    cache_dir,
    additional_channels=None,
    build_platform=None,
    verbosity=1,
):
    VERBOSITY = verbosity

    build_platform = build_platform or {}

    additional_channels = additional_channels or []
    virtual_packages = virtual_packages_repodata()
    os.environ["CONDA_OVERRIDE_GLIBC"] = "2.%d" % MAX_GLIBC_MINOR

    errors = []
    cbcs = sorted(glob.glob(os.path.join(feedstock_dir, ".ci_support", "*.yaml")))
    if len(cbcs) == 0:
        errors.append(
            "No `.ci_support/*.yaml` files found! This can happen when a rerender "
            "results in no builds for a recipe (e.g., a recipe is python 2.7 only). "
            "This attempted migration is being reported as not solvable.",
        )
        print_warning(errors[-1])
        return False, errors, {}

    if not os.path.exists(os.path.join(feedstock_dir, "recipe", "meta.yaml")):
        errors.append(
            "No `recipe/meta.yaml` file found! This issue is quite weird and "
            "someone should investigate!",
        )
        print_warning(errors[-1])
        return False, errors, {}

    print_info("CHECKING FEEDSTOCK: %s", os.path.basename(feedstock_dir))
    solvable = True
    solvable_by_cbc = {}

    for cbc_fname in cbcs:
        # we need to extract the platform (e.g., osx, linux) and arch (e.g., 64, aarm64)
        # conda smithy forms a string that is
        #
        #  {{ platform }} if arch == 64
        #  {{ platform }}_{{ arch }} if arch != 64
        #
        # Thus we undo that munging here.
        _parts = os.path.basename(cbc_fname).split("_")
        platform = _parts[0]
        arch = _parts[1]
        if arch not in ["32", "aarch64", "ppc64le", "armv7l", "arm64"]:
            arch = "64"

        print_info("CHECKING RECIPE SOLVABLE: %s", os.path.basename(cbc_fname))
        _solvable, _errors = _is_recipe_solvable_on_platform(
            os.path.join(feedstock_dir, "recipe"),
            cbc_fname,
            cache_dir,
            platform,
            arch,
            build_platform_arch=(
                build_platform.get(f"{platform}_{arch}", f"{platform}_{arch}")
            ),
            virtual_packages=virtual_packages,
            additional_channels=additional_channels,
        )
        solvable = solvable and _solvable
        cbc_name = os.path.basename(cbc_fname).rsplit(".", maxsplit=1)[0]
        errors.extend([f"{cbc_name}: {e}" for e in _errors])
        solvable_by_cbc[cbc_name] = _solvable

    del os.environ["CONDA_OVERRIDE_GLIBC"]

    return solvable, errors, solvable_by_cbc


def _is_recipe_solvable_on_platform(
    recipe_dir,
    cbc_path,
    cache_dir,
    platform,
    arch,
    build_platform_arch=None,
    virtual_packages=None,
    additional_channels=None,
):
    # parse the channel sources from the CBC
    parser = YAML(typ="jinja2")
    parser.indent(mapping=2, sequence=4, offset=2)
    parser.width = 320

    with open(cbc_path) as fp:
        cbc_cfg = parser.load(fp.read())

    if "channel_sources" in cbc_cfg:
        channel_sources = []
        for source in cbc_cfg["channel_sources"]:
            # channel_sources might be part of some zip_key
            channel_sources.extend([c.strip() for c in source.split(",")])
    else:
        channel_sources = ["conda-forge", "defaults", "msys2"]

    if "msys2" not in channel_sources:
        channel_sources.append("msys2")

    if additional_channels:
        channel_sources = list(additional_channels) + channel_sources

    with suppress_conda_build_logging():
        for att in range(2):
            try:
                if att == 1:
                    os.system("rm -f %s/conda_build_config.yaml" % recipe_dir)

                config = conda_build.config.get_or_merge_config(
                    None,
                    platform=platform,
                    arch=arch,
                    variant_config_files=[cbc_path],
                )

                cbc, _ = conda_build.variants.get_package_combined_spec(
                    recipe_dir,
                    config=config,
                )
            except Exception as e:
                if att == 1:
                    raise e

        metas = conda_build.api.render(
            recipe_dir,
            platform=platform,
            arch=arch,
            ignore_system_variants=True,
            variants=cbc,
            permit_undefined_jinja=True,
            finalize=False,
            bypass_env_check=True,
            channel_urls=channel_sources,
        )

    if build_platform_arch is not None:
        build_platform, build_arch = build_platform_arch.split("_")
    else:
        build_platform, build_arch = platform, arch

    repo_data = asyncio.run(
        fetch_repo_data(
            channels=[Channel(c) for c in channel_sources],
            platforms=[
                Platform(f"{platform}-{arch}"),
                Platform("noarch"),
            ],
            cache_path=cache_dir,
            callback=None,
        )
    )

    build_repo_data = asyncio.run(
        fetch_repo_data(
            channels=[Channel(c) for c in channel_sources],
            platforms=[
                Platform(f"{build_platform}-{build_arch}"),
                Platform("noarch"),
            ],
            cache_path=cache_dir,
            callback=None,
        )
    )

    solvable = True
    errors = []
    outnames = [m.name() for m, _, _ in metas]

    for m, _, _ in metas:
        print_debug("checking recipe %s", m.name())

        build_req = m.get_value("requirements/build", [])
        host_req = m.get_value("requirements/host", [])
        run_req = m.get_value("requirements/run", [])

        if build_req:
            build_req = _clean_reqs(build_req, outnames)

            try:
                _solved = solve(
                    specs=[MatchSpec(r) for r in build_req],
                    available_packages=build_repo_data,
                    virtual_packages=virtual_packages,
                )
            except Exception as e:
                solvable = False
                errors.append(e)

            if m.is_cross:
                host_req = list(set(host_req) | set(build_req))
                if not (m.noarch or m.noarch_python):
                    run_req = list(set(run_req) | set(build_req))
            else:
                run_req = list(set(run_req) | set(build_req))
                if not (m.noarch or m.noarch_python):
                    if not m.build_is_host:
                        host_req = list(set(host_req) | set(build_req))

        if host_req:
            host_req = _clean_reqs(host_req, outnames)
            try:
                _solved = solve(
                    specs=[MatchSpec(r) for r in host_req],
                    available_packages=repo_data,
                    virtual_packages=virtual_packages,
                )
            except Exception as e:
                solvable = False
                errors.append(e)

            if m.is_cross:
                if m.noarch or m.noarch_python:
                    run_req = list(set(run_req) | set(host_req))
                else:
                    run_req = list(set(run_req) | set(host_req))

        if run_req:
            run_req = apply_pins(run_req, host_req or [], build_req or [], outnames, m)
            run_req = _clean_reqs(run_req, outnames)
            try:
                _solved = solve(
                    specs=[MatchSpec(r) for r in run_req],
                    available_packages=repo_data,
                    virtual_packages=virtual_packages,
                )
            except Exception as e:
                solvable = False
                errors.append(e)

        tst_req = (
            m.get_value("test/requires", [])
            + m.get_value("test/requirements", [])
            + run_req
        )
        if tst_req:
            tst_req = _clean_reqs(tst_req, outnames)
            try:
                _solved = solve(
                    specs=[MatchSpec(r) for r in tst_req],
                    available_packages=repo_data,
                    virtual_packages=virtual_packages,
                )
            except Exception as e:
                solvable = False
                errors.append(e)

    return solvable, errors
