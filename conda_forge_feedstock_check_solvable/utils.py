import contextlib
import copy
import functools
import io
import os
import subprocess
import tempfile
import time
import traceback
import unittest.mock
from collections.abc import Mapping

import conda_build.api
import conda_package_handling.api
import rapidjson as json
import requests
import wurlitzer
import zstandard
from conda.models.match_spec import MatchSpec
from conda_build.jinja_context import context_processor as conda_build_context_processor
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
MAX_FUTURE_VERSION = 20

ARCHSPEC_X86_64_VERSIONS = ["x86_64_v1", "x86_64_v2", "x86_64_v3", "x86_64_v4"]

# these characters are start requirements that do not need to be munged from
# 1.1 to 1.1.*
REQ_START_NOSTAR = ["!=", "==", ">", "<", ">=", "<=", "~="]

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
] + [
    f"{cuda_major}.{cuda_minor}"
    for cuda_minor in range(MAX_FUTURE_VERSION)
    for cuda_major in range(12, MAX_FUTURE_VERSION)
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
    for osx_minor in range(0, MAX_FUTURE_VERSION)
    for osx_major in range(11, MAX_FUTURE_VERSION)
]

PROBLEMATIC_REQS = {
    # This causes a strange self-ref for arrow-cpp
    "parquet-cpp",
}


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


def print_verb(fmt, *args, verbosity=0, stack_bump=1):
    from inspect import stack

    frameinfo = stack()[stack_bump]

    if verbosity <= VERBOSITY:
        if args:
            msg = fmt % args
        else:
            msg = fmt
        print(
            VERBOSITY_PREFIX[verbosity]
            + ":"
            + frameinfo.frame.f_globals["__name__"]
            + ":"
            + "%d" % frameinfo.lineno
            + ":"
            + msg,
            flush=True,
        )


def print_critical(fmt, *args):
    print_verb(fmt, *args, verbosity=0, stack_bump=2)


def print_warning(fmt, *args):
    print_verb(fmt, *args, verbosity=1, stack_bump=2)


def print_info(fmt, *args):
    print_verb(fmt, *args, verbosity=2, stack_bump=2)


def print_debug(fmt, *args):
    print_verb(fmt, *args, verbosity=3, stack_bump=2)


@contextlib.contextmanager
def override_env_var(name, value):
    old_value = os.environ.get(name, None)
    try:
        os.environ[name] = value
        yield
    finally:
        if old_value is None:
            del os.environ[name]
        else:
            os.environ[name] = old_value


@contextlib.contextmanager
def suppress_output():
    if "CONDA_FORGE_FEEDSTOCK_CHECK_SOLVABLE_DEBUG" in os.environ or VERBOSITY > 2:
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


class TimeoutTimerException(Exception):
    pass


class TimeoutTimer:
    def __init__(self, timeout, name=None):
        self.timeout = timeout
        self.name = name
        self._start = time.monotonic()

    @property
    def elapsed(self):
        return time.monotonic() - self._start

    @property
    def remaining(self):
        return self.timeout - self.elapsed

    def raise_for_timeout(self):
        if self.elapsed > self.timeout:
            raise TimeoutTimerException("timeout out for %s" % self.name)


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
            if any(pp.startswith(__v) for __v in REQ_START_NOSTAR) or "*" in pp:
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


def convert_spec_to_conda_build(myspec):
    """Normalize a spec string to a conda-build form, turning requirements like
    numpy =1.0 to numpy 1.0.*."""
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


def _get_run_exports_from_download(channel_url, subdir, pkg):
    print_debug(
        "RUN EXPORTS: trying download for %s/%s/%s",
        channel_url.split("/")[-2:][0],
        subdir,
        pkg,
    )

    with tempfile.TemporaryDirectory(dir=os.environ.get("RUNNER_TEMP")) as tmpdir:
        try:
            # download
            subprocess.run(
                f"cd {tmpdir} && curl -s -L "
                f"{channel_url}/{subdir}/{pkg} --output {pkg}",
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

        except Exception as e:
            print_debug(
                "RUN EXPORTS: could not get run_exports from download: %s",
                repr(e),
            )
            run_exports = None

    return run_exports


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
    except requests.RequestException as e:
        # If the URL is invalid return None
        print_debug(
            "RUN EXPORTS: could not get run_exports from run_export.json: %s",
            repr(e),
        )
        return None
    compressed_binary = res.content
    binary = zstandard.decompress(compressed_binary)
    return json.loads(binary.decode("utf-8"))


@functools.cache
def _download_channeldata(channel_url):
    return download_channeldata(channel_url)


def _get_run_exports_from_run_exports_json(channel_url, subdir, filename):
    print_debug(
        "RUN EXPORTS: trying run_exports.json for %s/%s/%s",
        channel_url.split("/")[-2:][0],
        subdir,
        filename,
    )

    run_exports_json = _fetch_json_zst(f"{channel_url}/{subdir}/run_exports.json.zst")

    if run_exports_json is None:
        return None

    if filename.endswith(".conda"):
        if "packages.conda" not in run_exports_json:
            return None
        pkgs = run_exports_json.get("packages.conda")
    else:
        if "packages" not in run_exports_json:
            return None
        pkgs = run_exports_json.get("packages")

    if filename not in pkgs:
        return None

    if "run_exports" not in pkgs.get(filename):
        return None

    return pkgs.get(filename).get("run_exports", {})


def _has_run_exports_in_channel_data(channel_url, filename):
    cd = _download_channeldata(channel_url)
    name_ver, _ = filename.rsplit("-", 1)
    name, ver = name_ver.rsplit("-", 1)

    if "packages" not in cd:
        return True

    if name not in cd["packages"]:
        return True

    if "run_exports" not in cd["packages"][name]:
        return True

    if ver not in cd["packages"][name]["run_exports"]:
        return False
    else:
        return True


def _get_run_exports_from_artifact_info(channel, subdir, filename):
    print_debug(
        "RUN EXPORTS: trying conda-forge-metadata for %s/%s/%s",
        channel,
        subdir,
        filename,
    )

    try:
        with suppress_output():
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
        else:
            rx = None
    except Exception as e:
        print_debug(
            "RUN EXPORTS: could not get run_exports from conda-forge-metadata: %s",
            repr(e),
        )
        rx = None

    return rx


def _convert_run_exports_to_canonical_form(rx):
    run_exports = copy.deepcopy(DEFAULT_RUN_EXPORTS)
    if rx is not None:
        if isinstance(rx, str):
            # some packages have a single string
            # eg pyqt
            rx = {"weak": [rx]}

        if not isinstance(rx, Mapping):
            # list is equivalent to weak
            rx = {"weak": rx}

        for k, spec_list in rx.items():
            if k in DEFAULT_RUN_EXPORTS:
                run_exports[k].update(spec_list)
            else:
                print_warning(
                    "RUN EXPORTS: unrecognized run_export key in %s=%s",
                    k,
                    spec_list,
                )
    return run_exports


@functools.lru_cache(maxsize=10240)
def get_run_exports(full_channel_url, filename):
    """Given (channel, file), fetch the run exports for the artifact.

    There are several possible sources:

    1. CEP-12 run_exports.json file served in the channel/subdir (next to repodata.json)
    2. conda-forge-metadata fetchers (oci, etc)
    3. The full artifact (conda or tar.bz2) as a last resort

    Each is tried in turn and the first that works is used.
    """
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

    # First source: CEP-12 run_exports.json
    rx = _get_run_exports_from_run_exports_json(channel_url, subdir, filename)

    # Second source: conda-forge-metadata fetchers
    if rx is None and _has_run_exports_in_channel_data(channel_url, filename):
        rx = _get_run_exports_from_artifact_info(channel, subdir, filename)

        # Third source: download from the full artifact
        if rx is None:
            rx = _get_run_exports_from_download(channel_url, subdir, filename)

    # Sanitize run_exports data
    run_exports = _convert_run_exports_to_canonical_form(rx)
    print_debug(
        "RUN EXPORTS: found run exports for %s/%s/%s: %s",
        channel,
        subdir,
        filename,
        run_exports,
    )

    return run_exports


def remove_reqs_by_name(reqs, names):
    """Remove requirements by name given a list of names."""
    _names = set(names)
    return [r for r in reqs if r.split(" ")[0] not in _names]


def _filter_problematic_reqs(reqs):
    """There are some reqs that have issues when used in certain contexts"""
    reqs = [r for r in reqs if r.split(" ")[0] not in PROBLEMATIC_REQS]
    return reqs


def apply_pins(reqs, host_req, build_req, outnames, m):
    """Apply pins to requirements given host, build requirements,
    the output names, and the metadata from conda-build."""
    from conda_build.render import get_pin_from_build

    pin_deps = host_req if m.is_cross else build_req

    full_build_dep_versions = {
        dep.split()[0]: " ".join(dep.split()[1:])
        for dep in remove_reqs_by_name(pin_deps, outnames)
    }

    pinned_req = []
    for dep in reqs:
        try:
            pinned_req.append(
                get_pin_from_build(m, dep, full_build_dep_versions),
            )
        except Exception as e:
            print_critical(
                "Failed to apply pin for {}, falling back to req: {}".format(
                    dep, repr(e)
                ),
            )
            # in case we couldn't apply pins for whatever
            # reason, fall back to the req
            pinned_req.append(dep)

    pinned_req = _filter_problematic_reqs(pinned_req)
    return pinned_req


def _render_with_name(name, *args, **kwargs):
    out = name + "("
    for arg in args:
        out += repr(arg) + ","
    for k, v in kwargs.items():
        out += k + "=" + repr(v) + ","
    out = out[:-1]
    out += ")"
    return out


def _custom_context_processor(*args, **kwargs):
    """Custom context processor for conda_build that changes pin_compatible."""
    ctx = conda_build_context_processor(*args, **kwargs)
    ctx["pin_compatible"] = lambda *args, **kwargs: _render_with_name(
        "pin_compatible", *args, **kwargs
    )
    return ctx


def conda_build_api_render(*args, **kwargs):
    """Run conda_build.api.render with a patched jinja2 context for pin_compatible."""
    with unittest.mock.patch(
        "conda_build.jinja_context.context_processor", new=_custom_context_processor
    ):
        return conda_build.api.render(*args, **kwargs)


def _apply_pin_compatible(
    version,
    build,
    lower_bound=None,
    upper_bound=None,
    min_pin="x.x.x.x.x.x",
    max_pin="x",
    exact=False,
):
    from conda_build.utils import apply_pin_expressions

    if exact:
        return (version + " " + build).strip()
    else:
        _version = lower_bound or version
        if _version:
            if upper_bound:
                if min_pin or lower_bound:
                    compatibility = ">=" + str(_version) + ","
                compatibility += f"<{upper_bound}"
            else:
                compatibility = apply_pin_expressions(_version, min_pin, max_pin)
            return compatibility.strip()
        else:
            raise ValueError("No version or lower bound found for pin_compatible!")


def _strip_quotes(s):
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    elif s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    else:
        return s


def replace_pin_compatible(reqs, host_reqs, strict=False):
    """Replace pin_compatible with the appropriate version constraint.

    Parameters
    ----------
    reqs : list of str
        The requirements to replace pin_compatible in.
    host_reqs : list of str
        The requirements from the host section which is used to determine the
        compatibility ranges.
    strict : bool, optional
        If True, raise an error if the package is not found in the host section.
        If False, just add the package without a version constraint. Default is False
        to match conda-build behavior.

    Returns
    -------
    list of str
        The requirements with pin_compatible replaced with the appropriate version
        constraint.
    """
    host_lookup = {req.split(" ")[0]: req.split(" ")[1:] for req in host_reqs}

    new_reqs = []
    for req in reqs:
        if "pin_compatible(" in req:
            if not req.startswith("pin_compatible("):
                raise ValueError("Very odd pinning: %s!" % req)
            parts = req.rsplit(")", 1)
            if len(parts) == 2:
                build = parts[1].strip()
            else:
                build = ""

            parts = parts[0].split("pin_compatible(")[1]
            parts = parts.split(",")
            name = _strip_quotes(parts[0].strip())
            parts = parts[1:]

            if name not in host_lookup or not host_lookup[name]:
                if strict:
                    if name not in host_lookup:
                        raise ValueError(
                            "Very odd pinning: %s! Package %s not found in host %r!"
                            % (req, name, host_lookup)
                        )
                    else:
                        raise ValueError(
                            (
                                "Very odd pinning: %s! Package found in host "
                                "but no version %r!"
                            )
                            % (req, host_lookup[name])
                        )
                else:
                    if build:
                        new_reqs.append(name + " * " + build)
                    else:
                        new_reqs.append(name)
                    continue

            host_version = host_lookup[name][0]
            if len(host_lookup[name]) > 1:
                host_build = host_lookup[name][1]
                if build and "exact=true" in req.lower():
                    raise ValueError(
                        "Build string cannot be given for "
                        "pin_compatible with exact=True! %r" % req
                    )
            else:
                host_build = ""

            args = []
            kwargs = {}
            for part in parts:
                if "=" in part:
                    k, v = part.split("=")
                    kwargs[k.strip()] = _strip_quotes(v.strip())
                else:
                    args.append(part.strip())

            new_reqs.append(
                (
                    name
                    + " "
                    + _apply_pin_compatible(host_version, host_build, *args, **kwargs)
                    + " "
                    + build
                ).strip()
            )
        else:
            new_reqs.append(req)

    return new_reqs
