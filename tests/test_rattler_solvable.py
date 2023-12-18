
import os
import pathlib
import shutil
import subprocess
import pytest
import pprint
from flaky import flaky

from conda_forge_feedstock_check_solvable.rattler_solver import is_recipe_solvable


FEEDSTOCK_DIR = os.path.join(os.path.dirname(__file__), "test_feedstock")

@pytest.fixture()
def feedstock_dir(tmp_path):
    ci_support = tmp_path / ".ci_support"
    ci_support.mkdir(exist_ok=True)
    src_ci_support = pathlib.Path(FEEDSTOCK_DIR) / ".ci_support"
    for fn in os.listdir(src_ci_support):
        shutil.copy(src_ci_support / fn, ci_support / fn)
    return str(tmp_path)

@pytest.fixture()
def rattler_cache(tmp_path):
    return tmp_path / "rattler_cache"

@flaky
def test_is_recipe_solvable_ok(feedstock_dir, rattler_cache):
    recipe_file = os.path.join(feedstock_dir, "recipe", "meta.yaml")
    os.makedirs(os.path.dirname(recipe_file), exist_ok=True)
    with open(recipe_file, "w") as fp:
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
    assert is_recipe_solvable(feedstock_dir, rattler_cache)[0]

@flaky
def test_unsolvable_for_particular_python(feedstock_dir, rattler_cache):
    recipe_file = os.path.join(feedstock_dir, "recipe", "meta.yaml")
    os.makedirs(os.path.dirname(recipe_file), exist_ok=True)
    with open(recipe_file, "w") as fp:
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
    run:
    - python
    - galsim

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
    solvable, errors, solvable_by_variant = is_recipe_solvable(feedstock_dir, rattler_cache)
    print(solvable_by_variant)
    assert not solvable
    # we don't have galsim for this variant so this is an expected failure
    assert not solvable_by_variant["linux_aarch64_python3.6.____cpython"]
    assert not solvable_by_variant["linux_ppc64le_python3.6.____cpython"]
    # But we do have this one
    assert solvable_by_variant["linux_python3.7.____cpython"]

@flaky
def test_r_base_cross_solvable(rattler_cache):
    feedstock_dir = os.path.join(os.path.dirname(__file__), "r-base-feedstock")
    solvable, errors, _ = is_recipe_solvable(feedstock_dir, rattler_cache)
    assert solvable, pprint.pformat(errors)

    solvable, errors, _ = is_recipe_solvable(
        feedstock_dir,
        rattler_cache,
        build_platform={"osx_arm64": "osx_64"},
    )
    assert solvable, pprint.pformat(errors)

@flaky
def test_xgboost_solvable(rattler_cache):
    feedstock_dir = os.path.join(os.path.dirname(__file__), "xgboost-feedstock")
    solvable, errors, _ = is_recipe_solvable(feedstock_dir, rattler_cache)
    assert solvable, pprint.pformat(errors)

def clone_and_checkout_repo(base_path: pathlib.Path, origin_url: str, ref: str):
    subprocess.run(
        f"cd {base_path} && git clone {origin_url} repo",
        shell=True,
    )
    return str(base_path / "repo")


@flaky
def test_arrow_solvable(tmp_path, rattler_cache):
    feedstock_dir = clone_and_checkout_repo(
        tmp_path,
        "https://github.com/conda-forge/arrow-cpp-feedstock",
        ref="main",
    )
    solvable, errors, solvable_by_variant = is_recipe_solvable(feedstock_dir, rattler_cache)
    pprint.pprint(solvable_by_variant)
    assert solvable, pprint.pformat(errors)

@flaky
def test_guiqwt_solvable(tmp_path, rattler_cache):
    """test for run exports as a single string in pyqt"""
    feedstock_dir = clone_and_checkout_repo(
        tmp_path,
        "https://github.com/conda-forge/guiqwt-feedstock",
        ref="main",
    )
    solvable, errors, solvable_by_variant = is_recipe_solvable(feedstock_dir, rattler_cache)
    pprint.pprint(solvable_by_variant)
    assert solvable, pprint.pformat(errors)

@flaky
def test_datalad_solvable(tmp_path, rattler_cache):
    """has an odd thing where it hangs"""
    feedstock_dir = clone_and_checkout_repo(
        tmp_path,
        "https://github.com/conda-forge/datalad-feedstock",
        ref="main",
    )
    solvable, errors, solvable_by_variant = is_recipe_solvable(feedstock_dir, rattler_cache)
    pprint.pprint(solvable_by_variant)
    assert solvable, pprint.pformat(errors)

@flaky
def test_grpcio_solvable(tmp_path, rattler_cache):
    """grpcio has a runtime dep on openssl which has strange pinning things in it"""
    feedstock_dir = clone_and_checkout_repo(
        tmp_path,
        "https://github.com/conda-forge/grpcio-feedstock",
        ref="main",
    )
    solvable, errors, solvable_by_variant = is_recipe_solvable(feedstock_dir, rattler_cache)
    pprint.pprint(solvable_by_variant)
    assert solvable, pprint.pformat(errors)

@flaky
def test_cupy_solvable(tmp_path, rattler_cache):
    """grpcio has a runtime dep on openssl which has strange pinning things in it"""
    feedstock_dir = clone_and_checkout_repo(
        tmp_path,
        "https://github.com/conda-forge/cupy-feedstock",
        ref="main",
    )
    subprocess.run(
        f"cd {feedstock_dir} && git checkout 72d6c5808ca79c9cd9a3eb4064a72586c73c3430",
        shell=True,
        check=True,
    )
    solvable, errors, solvable_by_variant = is_recipe_solvable(feedstock_dir, rattler_cache)
    pprint.pprint(solvable_by_variant)
    assert solvable, pprint.pformat(errors)

@flaky
def test_is_recipe_solvable_notok(feedstock_dir, rattler_cache):
    recipe_file = os.path.join(feedstock_dir, "recipe", "meta.yaml")
    os.makedirs(os.path.dirname(recipe_file), exist_ok=True)
    with open(recipe_file, "w") as fp:
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
    - python >=4.0  # [osx]
    - python  # [not osx]
    - pip
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
    assert not is_recipe_solvable(feedstock_dir, rattler_cache)[0]

@flaky
def test_arrow_solvable_timeout(tmp_path, rattler_cache):
    feedstock_dir = clone_and_checkout_repo(
        tmp_path,
        "https://github.com/conda-forge/arrow-cpp-feedstock",
        ref="main",
    )
    # let's run this over and over again to make sure nothing weird is happening
    # with the killed processes
    for _ in range(6):
        solvable, errors, solvable_by_variant = is_recipe_solvable(
            feedstock_dir,
            rattler_cache,
            timeout=10,
        )
        assert solvable
        assert errors == []
        assert solvable_by_variant == {}

@flaky
def test_pillow_solvable(tmp_path, rattler_cache):
    """pillow acted up for python310"""
    feedstock_dir = clone_and_checkout_repo(
        tmp_path,
        "https://github.com/conda-forge/pillow-feedstock",
        ref="main",
    )

    subprocess.run(
        f"cd {feedstock_dir} && git checkout 0cae9b1b3450fd8862ac0f48f3389fc349702810",
        shell=True,
        check=True,
    )

    with open(
        os.path.join(feedstock_dir, ".ci_support", "migrations", "python310.yaml"),
        "w",
    ) as fp:
        fp.write(
            """\
migrator_ts: 1634137107
__migrator:
    migration_number: 1
    operation: key_add
    primary_key: python
    ordering:
        python:
            - 3.6.* *_cpython
            - 3.7.* *_cpython
            - 3.8.* *_cpython
            - 3.9.* *_cpython
            - 3.10.* *_cpython  # new entry
            - 3.6.* *_73_pypy
            - 3.7.* *_73_pypy
    paused: false
    longterm: True
    pr_limit: 40
    max_solver_attempts: 10  # this will make the bot retry "not solvable" stuff 10 times
    exclude:
      # this shouldn't attempt to modify the python feedstocks
      - python
      - pypy3.6
      - pypy-meta
      - cross-python
      - python_abi
    exclude_pinned_pkgs: false

python:
  - 3.10.* *_cpython
# additional entries to add for zip_keys
numpy:
  - 1.21
python_impl:
  - cpython
""",  # noqa
        )  # noqa

    subprocess.run(
        f"cd {feedstock_dir} && conda smithy rerender --no-check-uptodate",
        shell=True,
        check=True,
    )

    solvable, errors, solvable_by_variant = is_recipe_solvable(feedstock_dir, rattler_cache)
    pprint.pprint(solvable_by_variant)
    assert solvable, pprint.pformat(errors)
    assert any("python3.10" in k for k in solvable_by_variant)
