import contextlib
import copy
import functools
import io
import os
import subprocess
import tempfile
import traceback
from collections.abc import Mapping

import conda_package_handling.api
import rapidjson as json
import requests
import wurlitzer
import zstandard
from conda.models.match_spec import MatchSpec
from conda_build.utils import download_channeldata
from conda_forge_metadata.artifact_info import get_artifact_info_as_json


DEFAULT_RUN_EXPORTS = {
    "weak": set(),
    "strong": set(),
    "noarch": set(),
    "strong_constrains": set(),
    "weak_constrains": set(),
}

MAX_GLIBC_MINOR = 50

# these characters are start requirements that do not need to be munged from
# 1.1 to 1.1.*
REQ_START = ["!=", "==", ">", "<", ">=", "<=", "~="]

ALL_PLATFORMS = {
    "linux-aarch64",
    "linux-ppc64le",
    "linux-64",
    "osx-64",
    "osx-arm64",
    "win-64",
}

MINIMUM_CUDA_VERS = [
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

MINIMUM_OSX_64_VERS = [
    "10.9",
    "10.10",
    "10.11",
    "10.12",
    "10.13",
    "10.14",
    "10.15",
    "10.16",
]
MINIMUM_OSX_ARM64_VERS = MINIMUM_OSX_64_VERS + [
    f"{osx_major}.{osx_minor}"
    for osx_minor in range(0, 17)
    for osx_major in range(11, 17)
]

# I cannot get python logging to work correctly with all of the hacks to
# make conda-build be quiet.
# so this is a thing
VERBOSITY = 1
VERBOSITY_PREFIX = {
    0: "CRITICAL",
    1: "WARNING",
    2: "INFO",
    3: "DEBUG",
}


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


@contextlib.contextmanager
def suppress_conda_build_logging():
    if "CONDA_FORGE_FEEDSTOCK_CHECK_SOLVABLE_DEBUG" in os.environ:
        suppress = False
    else:
        suppress = True

    outerr = io.StringIO()

    if not suppress:
        try:
            yield None
        finally:
            pass
        return

    try:
        fout = io.StringIO()
        ferr = io.StringIO()
        with contextlib.redirect_stdout(fout), contextlib.redirect_stderr(ferr):
            with wurlitzer.pipes(stdout=outerr, stderr=wurlitzer.STDOUT):
                yield None

    except Exception as e:
        print("EXCEPTION: captured C-level I/O: %r" % outerr.getvalue(), flush=True)
        traceback.print_exc()
        raise e
    finally:
        pass


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


def _norm_spec(myspec):
    m = MatchSpec(myspec)

    # this code looks like MatchSpec.conda_build_form() but munges stars in the
    # middle
    parts = [m.get_exact_value("name")]

    version = m.get_raw_value("version")
    build = m.get_raw_value("build")
    if build and not version:
        raise RuntimeError("spec '%s' has build but not version!" % myspec)

    if version:
        parts.append(_munge_req_star(m.version.spec_str))
    if build:
        parts.append(build)

    return " ".join(parts)

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


def _strip_anaconda_tokens(url):
    if "/t/" in url:
        parts = url.split("/")
        tindex = parts.index("t")
        new_parts = [p for i, p in enumerate(parts) if i != tindex and i != tindex + 1]
        return "/".join(new_parts)
    else:
        return url


@functools.cache
def _fetch_json_zst(url):
    try:
        res = requests.get(url)
    except requests.RequestException:
        # If the URL is invalid return None
        return None
    compressed_binary = res.content
    binary = zstandard.decompress(compressed_binary)
    return json.loads(binary.decode("utf-8"))


@functools.lru_cache(maxsize=10240)
def _get_run_export(link_tuple):
    """
    Given a tuple of (channel, file, json repodata) as returned by libmamba solver,
    fetch the run exports for the artifact. There are several possible sources:

    1. CEP-12 run_exports.json file served in the channel/subdir (next to repodata.json)
    2. conda-forge-metadata fetchers (libcgraph, oci, etc)
    3. The full artifact (conda or tar.bz2) as a last resort
    """
    full_channel_url, filename, json_payload = link_tuple
    if "https://" in full_channel_url:
        https = _strip_anaconda_tokens(full_channel_url)
        channel_url = https.rsplit("/", maxsplit=1)[0]
        if "conda.anaconda.org" in channel_url:
            channel_url = channel_url.replace(
                "conda.anaconda.org",
                "conda-static.anaconda.org",
            )
    else:
        channel_url = full_channel_url.rsplit("/", maxsplit=1)[0]

    channel = full_channel_url.split("/")[-2:][0]
    subdir = full_channel_url.split("/")[-2:][1]
    data = json.loads(json_payload)
    name = data["name"]
    rx = {}

    # First source: CEP-12 run_exports.json
    run_exports_json = _fetch_json_zst(f"{channel_url}/{subdir}/run_exports.json.zst")
    if run_exports_json:
        if filename.endswith(".conda"):
            pkgs = run_exports_json.get("packages.conda", {})
        else:
            pkgs = run_exports_json.get("packages", {})
        rx = pkgs.get(filename, {}).get("run_exports", {})

    # Second source: conda-forge-metadata fetchers
    if not rx:
        cd = download_channeldata(channel_url)
        if cd.get("packages", {}).get(name, {}).get("run_exports", {}):
            artifact_data = get_artifact_info_as_json(
                channel,
                subdir,
                filename,
            )
            if artifact_data is not None:
                rx = (
                    artifact_data.get("rendered_recipe", {})
                    .get("build", {})
                    .get("run_exports", {})
                )

            # Third source: download from the full artifact
            if not rx:
                print_info(
                    "RUN EXPORTS: downloading package %s/%s/%s"
                    % (channel_url, subdir, link_tuple[1]),
                )
                rx = _get_run_export_download(link_tuple)[1]

    # Sanitize run_exports data
    run_exports = copy.deepcopy(DEFAULT_RUN_EXPORTS)
    if rx:
        if isinstance(rx, str):
            # some packages have a single string
            # eg pyqt
            rx = {"weak": [rx]}

        if not isinstance(rx, Mapping):
            # list is equivalent to weak
            rx = {"weak": rx}

        for k, spec_list in rx.items():
            if k in DEFAULT_RUN_EXPORTS:
                print_debug(
                    "RUN EXPORT: %s %s %s",
                    name,
                    k,
                    spec_list,
                )
                run_exports[k].update(spec_list)
            else:
                print_warning(
                    "RUN EXPORT: unrecognized run_export key in %s: %s=%s",
                    name,
                    k,
                    spec_list,
                )

    return run_exports

