import pytest

from conda_forge_feedstock_check_solvable.utils import (
    convert_spec_to_conda_build,
    get_run_export,
)


@pytest.mark.parametrize(
    "inreq,outreq",
    [
        ("blah 1.1*", "blah 1.1.*"),
        ("blah * *_osx", "blah * *_osx"),
        ("blah 1.1", "blah 1.1.*"),
        ("blah =1.1", "blah 1.1.*"),
        ("blah * *_osx", "blah * *_osx"),
        ("blah 1.2 *_osx", "blah 1.2.* *_osx"),
        ("blah >=1.1", "blah >=1.1"),
        ("blah >=1.1|5|>=5,<10|19.0", "blah >=1.1|5.*|>=5,<10|19.0.*"),
        ("blah >=1.1|5| >=5 , <10 |19.0", "blah >=1.1|5.*|>=5,<10|19.0.*"),
    ],
)
def test_convert_spec_to_conda_build(inreq, outreq):
    assert convert_spec_to_conda_build(inreq) == outreq


@pytest.mark.parametrize(
    "full_channel_url,filename,expected",
    [
        (
            "https://conda.anaconda.org/conda-forge/osx-arm64",
            "openjdk-22.0.1-hbeb2e11_0.conda",
            None,
        ),
    ],
)
def test_utils_get_run_export(full_channel_url, filename, expected):
    get_run_export(full_channel_url, filename) == expected
