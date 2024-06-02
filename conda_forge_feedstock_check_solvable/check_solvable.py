import glob
import os
import time
from typing import Dict, List, Tuple

import conda_build.api
import psutil
from ruamel.yaml import YAML

import conda_forge_feedstock_check_solvable.utils
from conda_forge_feedstock_check_solvable.mamba_solver import mamba_solver_factory
from conda_forge_feedstock_check_solvable.rattler_solver import rattler_solver_factory
from conda_forge_feedstock_check_solvable.utils import (
    MAX_GLIBC_MINOR,
    apply_pins,
    get_run_exports,
    print_debug,
    print_info,
    print_warning,
    remove_reqs_by_name,
    suppress_output,
)
from conda_forge_feedstock_check_solvable.virtual_packages import (
    virtual_package_repodata,
)


def _func(feedstock_dir, additional_channels, build_platform, verbosity, solver, conn):
    try:
        res = _is_recipe_solvable(
            feedstock_dir,
            additional_channels=additional_channels,
            build_platform=build_platform,
            verbosity=verbosity,
            solver=solver,
        )
        conn.send(res)
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()


def is_recipe_solvable(
    feedstock_dir,
    additional_channels=None,
    timeout=600,
    build_platform=None,
    verbosity=1,
    solver="mamba",
) -> Tuple[bool, List[str], Dict[str, bool]]:
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
    solver : str
        The solver to use. One of `mamba` or `rattler`.

    Returns
    -------
    solvable : bool
        The logical AND of the solvability of the recipe on all platforms
        in the CI scripts.
    errors : list of str
        A list of errors from the solver. Empty if recipe is solvable.
    solvable_by_variant : dict
        A lookup by variant config that shows if a particular config is solvable
    """
    if timeout and solver in ["mamba", "rattler"]:
        from multiprocessing import Pipe, Process

        parent_conn, child_conn = Pipe()
        p = Process(
            target=_func,
            args=(
                feedstock_dir,
                additional_channels,
                build_platform,
                verbosity,
                solver,
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
            print_warning("SOLVER TIMEOUT for %s", feedstock_dir)
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
            additional_channels=additional_channels,
            build_platform=build_platform,
            verbosity=verbosity,
            solver=solver,
            timeout=timeout,
        )

    return res


def _is_recipe_solvable(
    feedstock_dir,
    additional_channels=(),
    build_platform=None,
    verbosity=1,
    solver="mamba",
    timeout=None,
) -> Tuple[bool, List[str], Dict[str, bool]]:
    conda_forge_feedstock_check_solvable.utils.VERBOSITY = verbosity
    start_time = time.time()

    build_platform = build_platform or {}

    additional_channels = additional_channels or []
    additional_channels += [virtual_package_repodata()]
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
        if timeout and time.time() - start_time > timeout:
            print_warning("SOLVER TIMEOUT for %s", feedstock_dir)
            return True, [], {}

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
            platform,
            arch,
            build_platform_arch=(
                build_platform.get(f"{platform}_{arch}", f"{platform}_{arch}")
            ),
            additional_channels=additional_channels,
            solver_backend=solver,
            timeout=timeout,
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
    platform,
    arch,
    build_platform_arch=None,
    additional_channels=(),
    solver_backend="mamba",
    timeout=None,
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

    print_debug(
        "SOLVER using channels %s on platform-arch %s-%s",
        channel_sources,
        platform,
        arch,
    )

    # here we extract the conda build config in roughly the same way that
    # it would be used in a real build
    print_debug("rendering recipe with conda build")

    with suppress_output():
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
                if att == 0:
                    pass
                else:
                    raise e

        # now we render the meta.yaml into an actual recipe
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

    # get build info
    if build_platform_arch is not None:
        build_platform, build_arch = build_platform_arch.split("_")
    else:
        build_platform, build_arch = platform, arch

    # now we loop through each one and check if we can solve it
    # we check run and host and ignore the rest
    print_debug("getting solver")
    with suppress_output():
        if solver_backend == "rattler":
            solver_factory = rattler_solver_factory
        elif solver_backend == "mamba":
            solver_factory = mamba_solver_factory
        else:
            raise ValueError(f"Unknown solver backend {solver_backend}")

        solver = solver_factory(tuple(channel_sources), f"{platform}-{arch}")
        build_solver = solver_factory(
            tuple(channel_sources),
            f"{build_platform}-{build_arch}",
        )

    solvable = True
    errors = []
    outnames = [m.name() for m, _, _ in metas]
    for m, _, _ in metas:
        print_debug("checking recipe %s", m.name())

        build_req = m.get_value("requirements/build", [])
        host_req = m.get_value("requirements/host", [])
        run_req = m.get_value("requirements/run", [])
        run_constrained = m.get_value("requirements/run_constrained", [])

        ign_runex = m.get_value("build/ignore_run_exports", [])
        ign_runex_from = m.get_value("build/ignore_run_exports_from", [])

        if build_req:
            build_req = remove_reqs_by_name(build_req, outnames)
            _solvable, _err, build_req, build_rx = build_solver.solve(
                build_req,
                get_run_exports=True,
                ignore_run_exports_from=ign_runex_from,
                ignore_run_exports=ign_runex,
                timeout=timeout,
            )
            solvable = solvable and _solvable
            if _err is not None:
                errors.append(_err)

            run_constrained = list(set(run_constrained) | build_rx["strong_constrains"])

            if m.is_cross:
                host_req = list(set(host_req) | build_rx["strong"])
                if not (m.noarch or m.noarch_python):
                    run_req = list(set(run_req) | build_rx["strong"])
            else:
                if m.noarch or m.noarch_python:
                    if m.build_is_host:
                        run_req = list(set(run_req) | build_rx["noarch"])
                else:
                    run_req = list(set(run_req) | build_rx["strong"])
                    if m.build_is_host:
                        run_req = list(set(run_req) | build_rx["weak"])
                        run_constrained = list(
                            set(run_constrained) | build_rx["weak_constrains"]
                        )
                    else:
                        host_req = list(set(host_req) | build_rx["strong"])

        if host_req:
            host_req = remove_reqs_by_name(host_req, outnames)
            _solvable, _err, host_req, host_rx = solver.solve(
                host_req,
                get_run_exports=True,
                ignore_run_exports_from=ign_runex_from,
                ignore_run_exports=ign_runex,
                timeout=timeout,
            )
            solvable = solvable and _solvable
            if _err is not None:
                errors.append(_err)

            if m.is_cross:
                if m.noarch or m.noarch_python:
                    run_req = list(set(run_req) | host_rx["noarch"])
                else:
                    run_req = list(set(run_req) | host_rx["weak"] | host_rx["strong"])

                run_constrained = list(
                    set(run_constrained)
                    | host_rx["weak_constrains"]
                    | host_rx["strong_constrains"]
                )

        run_constrained = apply_pins(
            run_constrained, host_req or [], build_req or [], outnames, m
        )
        if run_req:
            run_req = apply_pins(run_req, host_req or [], build_req or [], outnames, m)
            run_req = remove_reqs_by_name(run_req, outnames)
            _solvable, _err, _ = solver.solve(
                run_req, constraints=run_constrained, timeout=timeout
            )
            solvable = solvable and _solvable
            if _err is not None:
                errors.append(_err)

        tst_req = (
            m.get_value("test/requires", [])
            + m.get_value("test/requirements", [])
            + run_req
        )
        if tst_req:
            tst_req = remove_reqs_by_name(tst_req, outnames)
            _solvable, _err, _ = solver.solve(
                tst_req, constraints=run_constrained, timeout=timeout
            )
            solvable = solvable and _solvable
            if _err is not None:
                errors.append(_err)

    print_info("RUN EXPORT CACHE STATUS: %s", get_run_exports.cache_info())
    print_info(
        "SOLVER MEM USAGE: %d MB",
        psutil.Process().memory_info().rss // 1024**2,
    )

    return solvable, errors
