from setuptools import setup, find_packages

__version__ = None
with open("conda_forge_feedstock_check_solvable/_version.py") as fp:
    exec(fp.read().strip())

setup(
    name="conda-forge-feedstock-check-solvable",
    version=__version__,
    description=(
        "A mamba-based package to check if a "
        "conda-forge feedstock is solvable."
    ),
    author="Conda-forge-tick Development Team",
    author_email="",
    url="https://github.com/regro/conda-forge-feedstock-check-solvable",
    packages=find_packages(),
)
