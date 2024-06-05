import pprint

from flaky import flaky

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

        solver = solver_factory(("conda-forge", "defaults"), "linux-64")

        metas = conda_build.api.render(
            str(tmp_path),
            platform="linux",
            arch="64",
            ignore_system_variants=True,
            variants=cbc,
            permit_undefined_jinja=True,
            finalize=False,
            bypass_env_check=True,
            channel_urls=("conda-forge", "defaults"),
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
            (virtual_packages, "conda-forge", "defaults"), "linux-64"
        )
        out = solver.solve(
            ["gcc_linux-64 7.*", "gxx_linux-64 7.*", "nvcc_linux-64 11.0.*"]
        )
    print(out)
    assert out[0], out[1]


@flaky
def test_solvers_hang(solver_factory):
    with suppress_output():
        solver = solver_factory(("conda-forge", "defaults"), "osx-64")
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
        solver = solver_factory(("conda-forge", "defaults"), "linux-64")
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
