import inspect
import pprint

import pytest
from flaky import flaky

try:
    from conda_forge_feedstock_check_solvable.mamba_solver import (
        MambaSolver,
        mamba_solver_factory,
    )
except Exception:
    MambaSolver = mamba_solver_factory = None

from conda_forge_feedstock_check_solvable.rattler_solver import (
    RattlerSolver,
    rattler_solver_factory,
)
from conda_forge_feedstock_check_solvable.utils import apply_pins, suppress_output
from conda_forge_feedstock_check_solvable.virtual_packages import (
    virtual_package_repodata,
)


@flaky
def test_solvers_apply_pins(tmp_path, solver_factory):
    with open(tmp_path / "meta.yaml", "w") as fp:
        fp.write(
            """\
{% set name = "cf-autotick-bot-test-package" %}
{% set version = "0.9" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  path: .

build:
  number: 8

requirements:
  host:
    - python
    - pip
    - jpeg
  run:
    - python

test:
  commands:
    - echo "works!"

about:
  home: https://github.com/regro/cf-scripts
  license: BSD-3-Clause
  license_family: BSD
  license_file: LICENSE
  summary: testing feedstock for the regro-cf-autotick-bot

extra:
  recipe-maintainers:
    - beckermr
    - conda-forge/bot
""",
        )

    with open(tmp_path / "conda_build_config.yaml", "w") as fp:
        fp.write(
            """\
pin_run_as_build:
  python:
    min_pin: x.x
    max_pin: x.x
python:
- 3.8.* *_cpython
""",
        )
    import conda_build.api

    with suppress_output():
        config = conda_build.config.get_or_merge_config(
            None,
            platform="linux",
            arch="64",
            variant_config_files=[],
        )
        cbc, _ = conda_build.variants.get_package_combined_spec(
            str(tmp_path),
            config=config,
        )

        solver = solver_factory(("conda-forge",), "linux-64")

        metas = conda_build.api.render(
            str(tmp_path),
            platform="linux",
            arch="64",
            ignore_system_variants=True,
            variants=cbc,
            permit_undefined_jinja=True,
            finalize=False,
            bypass_env_check=True,
            channel_urls=("conda-forge",),
        )

    m = metas[0][0]
    outnames = [m.name() for m, _, _ in metas]
    build_req = m.get_value("requirements/build", [])
    host_req = m.get_value("requirements/host", [])
    run_req = m.get_value("requirements/run", [])
    _, _, build_req, rx = solver.solve(build_req, get_run_exports=True)
    print("build req: %s" % pprint.pformat(build_req))
    print("build rex: %s" % pprint.pformat(rx))
    host_req = list(set(host_req) | rx["strong"])
    run_req = list(set(run_req) | rx["strong"])
    _, _, host_req, rx = solver.solve(host_req, get_run_exports=True)
    print("host req: %s" % pprint.pformat(host_req))
    print("host rex: %s" % pprint.pformat(rx))
    run_req = list(set(run_req) | rx["weak"])
    run_req = apply_pins(run_req, host_req, build_req, outnames, m)
    print("run req: %s" % pprint.pformat(run_req))
    assert any(r.startswith("python >=3.8") for r in run_req)
    assert any(r.startswith("jpeg >=") for r in run_req)


@flaky
def test_solvers_constraints(solver_factory):
    with suppress_output():
        solver = solver_factory(("conda-forge",), "osx-64")
        solvable, err, solution = solver.solve(
            ["simplejson"], constraints=["python=3.10", "zeromq=4.2"]
        )
    assert solvable, err
    python = [pkg for pkg in solution if pkg.split()[0] == "python"][0]
    name, version, build = python.split(None, 2)
    assert version.startswith("3.10.")
    assert not any(pkg.startswith("zeromq") for pkg in solution), pprint.pformat(
        solution
    )


@flaky
def test_solvers_constraints_unsolvable(solver_factory):
    with suppress_output():
        solver = solver_factory(("conda-forge",), "osx-64")
        solvable, err, solution = solver.solve(
            ["simplejson"], constraints=["python=3.10", "python=3.11"]
        )
    assert not solvable, pprint.pformat(solution)


@flaky
def test_solvers_nvcc_with_virtual_package(solver_factory):
    with suppress_output():
        virtual_packages = virtual_package_repodata()
        solver = solver_factory(
            (virtual_packages, "conda-forge"), "linux-64"
        )
        out = solver.solve(
            ["gcc_linux-64 7.*", "gxx_linux-64 7.*", "nvcc_linux-64 11.0.*"]
        )
    print(out)
    assert out[0], out[1]


@flaky
def test_solvers_archspec_with_virtual_package(solver_factory):
    with suppress_output():
        virtual_packages = virtual_package_repodata()

        # Not having the fake virtual packages should fail
        solver = solver_factory(("conda-forge",), "linux-64")
        out = solver.solve(["pythia8 8.312"])
        assert not out[0], out[1]

        # Including the fake virtual packages should succeed
        solver = solver_factory(
            (virtual_packages, "conda-forge"), "linux-64"
        )
        out = solver.solve(["pythia8 8.312"])
    assert out[0], out[1]


@flaky
def test_solvers_hang(solver_factory):
    with suppress_output():
        solver = solver_factory(("conda-forge",), "osx-64")
        res = solver.solve(
            [
                "pytest",
                "selenium",
                "requests-mock",
                "ncurses >=6.2,<7.0a0",
                "libffi >=3.2.1,<4.0a0",
                "xz >=5.2.5,<6.0a0",
                "nbconvert >=5.6",
                "sqlalchemy",
                "jsonschema",
                "six >=1.11",
                "python_abi 3.9.* *_cp39",
                "tornado",
                "jupyter",
                "requests",
                "jupyter_client",
                "notebook >=4.2",
                "tk >=8.6.10,<8.7.0a0",
                "openssl >=1.1.1h,<1.1.2a",
                "readline >=8.0,<9.0a0",
                "fuzzywuzzy",
                "python >=3.9,<3.10.0a0",
                "traitlets",
                "sqlite >=3.33.0,<4.0a0",
                "alembic",
                "zlib >=1.2.11,<1.3.0a0",
                "python-dateutil",
                "nbformat",
                "jupyter_core",
            ],
        )
    assert res[0]

    with suppress_output():
        solver = solver_factory(("conda-forge",), "linux-64")
        res = solver.solve(
            [
                "gdal >=2.1.0",
                "ncurses >=6.2,<7.0a0",
                "geopandas",
                "scikit-image >=0.16.0",
                "pandas",
                "pyproj >=2.2.0",
                "libffi >=3.2.1,<4.0a0",
                "six",
                "tk >=8.6.10,<8.7.0a0",
                "spectral",
                "zlib >=1.2.11,<1.3.0a0",
                "shapely",
                "readline >=8.0,<9.0a0",
                "python >=3.8,<3.9.0a0",
                "numpy",
                "python_abi 3.8.* *_cp38",
                "xz >=5.2.5,<6.0a0",
                "openssl >=1.1.1h,<1.1.2a",
                "sqlite >=3.33.0,<4.0a0",
            ],
        )
    assert res[0]


@pytest.mark.skipif(
    MambaSolver is None or mamba_solver_factory is None,
    reason="mamba not available",
)
@pytest.mark.parametrize("mamba_factory", [MambaSolver, mamba_solver_factory])
@pytest.mark.parametrize("rattler_factory", [RattlerSolver, rattler_solver_factory])
def test_solvers_compare_output(mamba_factory, rattler_factory):
    if inspect.isfunction(mamba_factory) and inspect.isfunction(rattler_factory):
        mamba_factory.cache_clear()
        rattler_factory.cache_clear()

    specs_linux = (
        "libutf8proc >=2.8.0,<3.0a0",
        "orc >=2.0.1,<2.0.2.0a0",
        "glog >=0.7.0,<0.8.0a0",
        "libabseil * cxx17*",
        "libgcc-ng >=12",
        "libbrotlidec >=1.1.0,<1.2.0a0",
        "bzip2 >=1.0.8,<2.0a0",
        "libbrotlienc >=1.1.0,<1.2.0a0",
        "libgoogle-cloud-storage >=2.24.0,<2.25.0a0",
        "libstdcxx-ng >=12",
        "re2",
        "gflags >=2.2.2,<2.3.0a0",
        "libabseil >=20240116.2,<20240117.0a0",
        "libre2-11 >=2023.9.1,<2024.0a0",
        "libgoogle-cloud >=2.24.0,<2.25.0a0",
        "lz4-c >=1.9.3,<1.10.0a0",
        "libbrotlicommon >=1.1.0,<1.2.0a0",
        "aws-sdk-cpp >=1.11.329,<1.11.330.0a0",
        "snappy >=1.2.0,<1.3.0a0",
        "zstd >=1.5.6,<1.6.0a0",
        "aws-crt-cpp >=0.26.9,<0.26.10.0a0",
        "libzlib >=1.2.13,<2.0a0",
    )
    constraints_linux = ("apache-arrow-proc * cpu", "arrow-cpp <0.0a0")

    specs_linux_again = (
        "glog >=0.7.0,<0.8.0a0",
        "bzip2 >=1.0.8,<2.0a0",
        "lz4-c >=1.9.3,<1.10.0a0",
        "libbrotlidec >=1.1.0,<1.2.0a0",
        "zstd >=1.5.6,<1.6.0a0",
        "gflags >=2.2.2,<2.3.0a0",
        "libzlib >=1.2.13,<2.0a0",
        "libbrotlienc >=1.1.0,<1.2.0a0",
        "re2",
        "aws-sdk-cpp >=1.11.329,<1.11.330.0a0",
        "libgoogle-cloud-storage >=2.24.0,<2.25.0a0",
        "libgoogle-cloud >=2.24.0,<2.25.0a0",
        "libstdcxx-ng >=12",
        "libutf8proc >=2.8.0,<3.0a0",
        "libabseil * cxx17*",
        "snappy >=1.2.0,<1.3.0a0",
        "__glibc >=2.17,<3.0.a0",
        "orc >=2.0.1,<2.0.2.0a0",
        "libgcc-ng >=12",
        "libabseil >=20240116.2,<20240117.0a0",
        "libbrotlicommon >=1.1.0,<1.2.0a0",
        "libre2-11 >=2023.9.1,<2024.0a0",
        "aws-crt-cpp >=0.26.9,<0.26.10.0a0",
    )
    constraints_linux_again = ("arrow-cpp <0.0a0", "apache-arrow-proc * cuda")

    specs_win = (
        "re2",
        "libabseil * cxx17*",
        "vc >=14.2,<15",
        "libbrotlidec >=1.1.0,<1.2.0a0",
        "lz4-c >=1.9.3,<1.10.0a0",
        "aws-sdk-cpp >=1.11.329,<1.11.330.0a0",
        "libbrotlicommon >=1.1.0,<1.2.0a0",
        "snappy >=1.2.0,<1.3.0a0",
        "ucrt >=10.0.20348.0",
        "orc >=2.0.1,<2.0.2.0a0",
        "zstd >=1.5.6,<1.6.0a0",
        "libcrc32c >=1.1.2,<1.2.0a0",
        "libre2-11 >=2023.9.1,<2024.0a0",
        "libbrotlienc >=1.1.0,<1.2.0a0",
        "libcurl >=8.8.0,<9.0a0",
        "libabseil >=20240116.2,<20240117.0a0",
        "bzip2 >=1.0.8,<2.0a0",
        "libgoogle-cloud >=2.24.0,<2.25.0a0",
        "vc14_runtime >=14.29.30139",
        "libzlib >=1.2.13,<2.0a0",
        "libgoogle-cloud-storage >=2.24.0,<2.25.0a0",
        "libutf8proc >=2.8.0,<3.0a0",
        "aws-crt-cpp >=0.26.9,<0.26.10.0a0",
    )
    constraints_win = ("arrow-cpp <0.0a0", "apache-arrow-proc * cuda")

    channels = (virtual_package_repodata(), "conda-forge", "msys2")

    platform = "linux-64"
    mamba_solver = mamba_factory(channels, platform)
    rattler_solver = rattler_factory(channels, platform)
    mamba_solvable, mamba_err, mamba_solution = mamba_solver.solve(
        specs_linux, constraints=constraints_linux
    )
    rattler_solvable, rattler_err, rattler_solution = rattler_solver.solve(
        specs_linux, constraints=constraints_linux
    )
    assert set(mamba_solution or []) == set(rattler_solution or [])
    assert mamba_solvable == rattler_solvable

    platform = "linux-64"
    mamba_solver = mamba_factory(channels, platform)
    rattler_solver = rattler_factory(channels, platform)
    mamba_solvable, mamba_err, mamba_solution = mamba_solver.solve(
        specs_linux_again, constraints=constraints_linux_again
    )
    rattler_solvable, rattler_err, rattler_solution = rattler_solver.solve(
        specs_linux_again, constraints=constraints_linux_again
    )
    assert set(mamba_solution or []) == set(rattler_solution or [])
    assert mamba_solvable == rattler_solvable

    platform = "linux-64"
    mamba_solver = mamba_factory(channels, platform)
    rattler_solver = rattler_factory(channels, platform)
    mamba_solvable, mamba_err, mamba_solution = mamba_solver.solve(
        specs_linux, constraints=constraints_linux
    )
    rattler_solvable, rattler_err, rattler_solution = rattler_solver.solve(
        specs_linux, constraints=constraints_linux
    )
    assert set(mamba_solution or []) == set(rattler_solution or [])
    assert mamba_solvable == rattler_solvable

    platform = "win-64"
    mamba_solver = mamba_factory(channels, platform)
    rattler_solver = rattler_factory(channels, platform)
    mamba_solvable, mamba_err, mamba_solution = mamba_solver.solve(
        specs_win, constraints=constraints_win
    )
    rattler_solvable, rattler_err, rattler_solution = rattler_solver.solve(
        specs_win, constraints=constraints_win
    )
    assert set(mamba_solution or []) == set(rattler_solution or [])
    assert mamba_solvable == rattler_solvable

    if inspect.isfunction(mamba_factory) and inspect.isfunction(rattler_factory):
        assert (
            mamba_factory.cache_info().misses > rattler_factory.cache_info().misses
        ), {
            "mamba cache info": mamba_factory.cache_info(),
            "rattler cache info": rattler_factory.cache_info(),
        }


@pytest.mark.skipif(
    MambaSolver is None or mamba_solver_factory is None,
    reason="mamba not available",
)
@pytest.mark.parametrize("mamba_factory", [MambaSolver, mamba_solver_factory])
@pytest.mark.parametrize("rattler_factory", [RattlerSolver, rattler_solver_factory])
def test_solvers_python(mamba_factory, rattler_factory):
    channels = (virtual_package_repodata(), "conda-forge", "msys2")
    platform = "linux-64"
    for _ in range(4):
        mamba_solver = mamba_factory(channels, platform)
        rattler_solver = rattler_factory(channels, platform)
        mamba_solvable, mamba_err, mamba_solution = mamba_solver.solve(
            ["python"],
        )
        rattler_solvable, rattler_err, rattler_solution = rattler_solver.solve(
            ["python"],
        )
        assert set(mamba_solution or []) == set(rattler_solution or [])
        assert mamba_solvable == rattler_solvable


def test_solvers_python3_pin(solver_factory):
    specs = [
        "tetgen",
        "hdf5",
        "libgfortran5 >=12.3.0",
        "triangle",
        "scipy",
        "numpy",
        "pytest-xdist",
        "ncurses",
        "pychrono >=7",
        "cython",
        "h5py * mpi_mpich_*",
        "petsc4py",
        "future",
        "mpi4py",
        "pytest",
        "libgcc-ng >=12",
        "h5py",
        "openblas",
        "python 3",
        "hdf5 * mpi_mpich_*",
        "libgfortran-ng",
        "matplotlib-base",
        "libstdcxx-ng >=12",
    ]
    channels = (virtual_package_repodata(), "conda-forge")
    platform = "linux-64"
    solver = solver_factory(channels, platform)
    solvable, err, solution = solver.solve(specs)
    assert solvable, (err, solution)
