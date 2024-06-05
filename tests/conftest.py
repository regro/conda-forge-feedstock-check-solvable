import os
import pathlib
import shutil

import pytest

from conda_forge_feedstock_check_solvable.mamba_solver import mamba_solver_factory

FEEDSTOCK_DIR = os.path.join(os.path.dirname(__file__), "test_feedstock")
ALL_SOLVERS = ["mamba"]


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


def pytest_generate_tests(metafunc):
    if "solver" in metafunc.fixturenames:
        metafunc.parametrize(
            "solver", metafunc.config.getoption("solver") or ALL_SOLVERS
        )
    if "solver_factory" in metafunc.fixturenames:
        solvers = metafunc.config.getoption("solver") or ALL_SOLVERS
        factories = []
        for solver in solvers:
            if solver == "mamba":
                factories.append(mamba_solver_factory)
            else:
                raise ValueError(f"Unknown solver {solver}")
        metafunc.parametrize("solver_factory", factories)
