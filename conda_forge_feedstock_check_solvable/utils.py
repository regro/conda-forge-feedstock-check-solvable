
import contextlib
import copy
import functools
import io
import json
import os
import subprocess
import tempfile
import traceback
import wurlitzer
import conda_package_handling.api

from conda_build.utils import download_channeldata
from conda_forge_metadata.artifact_info import (
    get_artifact_info_as_json,
)

ALL_PLATFORMS = {
    "linux-aarch64",
    "linux-ppc64le",
    "linux-64",
    "osx-64",
    "osx-arm64",
    "win-64",
}

DEFAULT_RUN_EXPORTS = {
    "weak": set(),
    "strong": set(),
    "noarch": set(),
}

# I cannot get python logging to work correctly with all of the hacks to
# make conda-build be quiet.
# so theis is a thing
VERBOSITY = 1
VERBOSITY_PREFIX = {
    0: "CRITICAL",
    1: "WARNING",
    2: "INFO",
    3: "DEBUG",
}

MAX_GLIBC_MINOR = 50

# these characters are start requirements that do not need to be munged from
# 1.1 to 1.1.*
REQ_START = ["!=", "==", ">", "<", ">=", "<=", "~="]

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


def print_critical(fmt, *args):
    print_verb(fmt, *args, verbosity=0)


def print_warning(fmt, *args):
    print_verb(fmt, *args, verbosity=1)


def print_info(fmt, *args):
    print_verb(fmt, *args, verbosity=2)


def print_debug(fmt, *args):
    print_verb(fmt, *args, verbosity=3)


def _munge_req_star(req):
    reqs = []

    # now we split on ',' and '|'
    # once we have all of the parts, we then munge the star
    csplit = req.split(",")
    ncs = len(csplit)
    for ic, p in enumerate(csplit):

        psplit = p.split("|")
        nps = len(psplit)
        for ip, pp in enumerate(psplit):
            # clear white space
            pp = pp.strip()

            # finally add the star if we need it
            if any(pp.startswith(__v) for __v in REQ_START) or "*" in pp:
                reqs.append(pp)
            else:
                if pp.startswith("="):
                    pp = pp[1:]
                reqs.append(pp + ".*")

            # add | back on the way out
            if ip != nps - 1:
                reqs.append("|")

        # add , back on the way out
        if ic != ncs - 1:
            reqs.append(",")

    # put it all together
    return "".join(reqs)


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


def _strip_anaconda_tokens(url):
    if "/t/" in url:
        parts = url.split("/")
        tindex = parts.index("t")
        new_parts = [p for i, p in enumerate(parts) if i != tindex and i != tindex + 1]
        return "/".join(new_parts)
    else:
        return url


def _get_run_export_download(link_tuple):
    c, pkg, jdata = link_tuple

    with tempfile.TemporaryDirectory(dir=os.environ.get("RUNNER_TEMP")) as tmpdir:
        try:
            # download
            subprocess.run(
                f"cd {tmpdir} && curl -s -L {c}/{pkg} --output {pkg}",
                shell=True,
            )

            # unpack and read if it exists
            if os.path.exists(f"{tmpdir}/{pkg}"):
                conda_package_handling.api.extract(f"{tmpdir}/{pkg}")

            if pkg.endswith(".tar.bz2"):
                pkg_nm = pkg[: -len(".tar.bz2")]
            else:
                pkg_nm = pkg[: -len(".conda")]

            rxpth = f"{tmpdir}/{pkg_nm}/info/run_exports.json"

            if os.path.exists(rxpth):
                with open(rxpth) as fp:
                    run_exports = json.load(fp)
            else:
                run_exports = {}

            for key in DEFAULT_RUN_EXPORTS:
                if key in run_exports:
                    print_debug(
                        "RUN EXPORT: %s %s %s",
                        pkg,
                        key,
                        run_exports.get(key, []),
                    )
                run_exports[key] = set(run_exports.get(key, []))

        except Exception as e:
            print("Could not get run exports for %s: %s", pkg, repr(e))
            run_exports = None
            pass

    return link_tuple, run_exports


@functools.lru_cache(maxsize=10240)
def _get_run_export(link_tuple):

    run_exports = None

    if "https://" in link_tuple[0]:
        https = _strip_anaconda_tokens(link_tuple[0])
        channel_url = https.rsplit("/", maxsplit=1)[0]
        if "conda.anaconda.org" in channel_url:
            channel_url = channel_url.replace(
                "conda.anaconda.org",
                "conda-static.anaconda.org",
            )
    else:
        channel_url = link_tuple[0].rsplit("/", maxsplit=1)[0]

    cd = download_channeldata(channel_url)
    data = json.loads(link_tuple[2])
    name = data["name"]

    if cd.get("packages", {}).get(name, {}).get("run_exports", {}):
        data = get_artifact_info_as_json(
            link_tuple[0].split("/")[-2:][0],  # channel
            link_tuple[0].split("/")[-2:][1],  # subdir
            link_tuple[1],  # package
        )

        if data is not None:
            rx = data.get("rendered_recipe", {}).get("build", {}).get("run_exports", {})
            if rx:
                run_exports = copy.deepcopy(
                    DEFAULT_RUN_EXPORTS,
                )
                if isinstance(rx, str):
                    # some packages have a single string
                    # eg pyqt
                    rx = [rx]

                for k in rx:
                    if k in DEFAULT_RUN_EXPORTS:
                        print_debug(
                            "RUN EXPORT: %s %s %s",
                            name,
                            k,
                            rx[k],
                        )
                        run_exports[k].update(rx[k])
                    else:
                        print_debug(
                            "RUN EXPORT: %s %s %s",
                            name,
                            "weak",
                            [k],
                        )
                        run_exports["weak"].add(k)

        # fall back to getting repodata shard if needed
        if run_exports is None:
            print_info(
                "RUN EXPORTS: downloading package %s/%s/%s"
                % (channel_url, link_tuple[0].split("/")[-1], link_tuple[1]),
            )
            run_exports = _get_run_export_download(link_tuple)[1]
    else:
        run_exports = copy.deepcopy(DEFAULT_RUN_EXPORTS)

    return run_exports
