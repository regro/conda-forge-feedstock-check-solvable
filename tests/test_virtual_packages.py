import os
from textwrap import dedent

from flaky import flaky

from conda_forge_feedstock_check_solvable.check_solvable import is_recipe_solvable
from conda_forge_feedstock_check_solvable.virtual_packages import (
    FakePackage,
    FakeRepoData,
)


@flaky
def test_virtual_package(feedstock_dir, tmp_path, solver):
    recipe_file = os.path.join(feedstock_dir, "recipe", "meta.yaml")
    os.makedirs(os.path.dirname(recipe_file), exist_ok=True)

    with FakeRepoData(tmp_path) as repodata:
        for pkg in [
            FakePackage("fakehostvirtualpkgdep", depends=frozenset(["__virtual >=10"])),
            FakePackage("__virtual", version="10"),
        ]:
            repodata.add_package(pkg)

    with open(recipe_file, "w") as fp:
        fp.write(
            dedent(
                """
    package:
      name: "cf-autotick-bot-test-package"
      version: "0.9"

    source:
      path: .

    build:
      number: 8

    requirements:
      host:
        - python
        - fakehostvirtualpkgdep
        - pip
      run:
        - python
    """,
            ),
        )

    solvable, err, solve_by_variant = is_recipe_solvable(
        feedstock_dir,
        additional_channels=[repodata.channel_url],
        solver=solver,
    )
    assert solvable
