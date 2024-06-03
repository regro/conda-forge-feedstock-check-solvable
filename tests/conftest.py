import os
import pathlib
import shutil

import pytest

from conda_forge_feedstock_check_solvable.mamba_solver import mamba_solver_factory
from conda_forge_feedstock_check_solvable.rattler_solver import rattler_solver_factory

FEEDSTOCK_DIR = os.path.join(os.path.dirname(__file__), "test_feedstock")

ALL_SOLVERS = ["rattler", "mamba"]


def pytest_addoption(parser):
    parser.addoption(
        "--solver",
        action="append",
        default=[],
        help="conda solver to use",
    )


@pytest.fixture()
def feedstock_dir(tmp_path):
    ci_support = tmp_path / ".ci_support"
    ci_support.mkdir(exist_ok=True)
    src_ci_support = pathlib.Path(FEEDSTOCK_DIR) / ".ci_support"
    for fn in os.listdir(src_ci_support):
        shutil.copy(src_ci_support / fn, ci_support / fn)
    return str(tmp_path)


@pytest.fixture(scope="session", params=ALL_SOLVERS)
def solver(request):
    solvers = request.config.getoption("solver") or ALL_SOLVERS
    if request.param in solvers:
        yield request.param


@pytest.fixture(scope="session", params=ALL_SOLVERS)
def solver_factory(request):
    solvers = request.config.getoption("solver") or ALL_SOLVERS
    if request.param == "mamba" and "mamba" in solvers:
        yield mamba_solver_factory
    elif request.param == "rattler" and "rattler" in solvers:
        yield rattler_solver_factory
    else:
        raise ValueError(f"Unknown solver {request.param}")
