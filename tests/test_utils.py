import pytest

from conda_forge_feedstock_check_solvable.utils import (
    DEFAULT_RUN_EXPORTS,
    _convert_run_exports_to_canonical_form,
    _get_run_exports_from_artifact_info,
    _get_run_exports_from_download,
    _get_run_exports_from_run_exports_json,
    _has_run_exports_in_channel_data,
    convert_spec_to_conda_build,
    get_run_exports,
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
            DEFAULT_RUN_EXPORTS,
        ),
        (
            "https://conda.anaconda.org/conda-forge/osx-64",
            "mpich-4.2.1-hd33e60e_100.conda",
            {
                "noarch": set(),
                "strong": set(),
                "strong_constrains": set(),
                "weak_constrains": set(),
                "weak": {"mpich >=4.2.1,<5.0a0"},
            },
        ),
    ],
)
def test_utils_get_run_exports(full_channel_url, filename, expected):
    if expected == DEFAULT_RUN_EXPORTS:
        assert not _has_run_exports_in_channel_data(
            full_channel_url.rsplit("/", 1)[0], filename
        )
    else:
        assert _has_run_exports_in_channel_data(
            full_channel_url.rsplit("/", 1)[0], filename
        )

    assert get_run_exports(full_channel_url, filename) == expected

    assert (
        _convert_run_exports_to_canonical_form(
            _get_run_exports_from_artifact_info(
                full_channel_url.split("/")[-2],
                full_channel_url.split("/")[-1],
                filename,
            )
        )
        == expected
    )

    assert (
        _convert_run_exports_to_canonical_form(
            _get_run_exports_from_run_exports_json(
                full_channel_url.rsplit("/", 1)[0],
                full_channel_url.split("/")[-1],
                filename,
            )
        )
        == expected
    )

    assert (
        _convert_run_exports_to_canonical_form(
            _get_run_exports_from_download(
                full_channel_url.rsplit("/", 1)[0],
                full_channel_url.split("/")[-1],
                filename,
            )
        )
        == expected
    )
