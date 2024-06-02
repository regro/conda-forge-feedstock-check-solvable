import os
import shutil
import pathlib

import pytest

FEEDSTOCK_DIR = os.path.join(os.path.dirname(__file__), "test_feedstock")


@pytest.fixture()
def feedstock_dir(tmp_path):
    ci_support = tmp_path / ".ci_support"
    ci_support.mkdir(exist_ok=True)
    src_ci_support = pathlib.Path(FEEDSTOCK_DIR) / ".ci_support"
    for fn in os.listdir(src_ci_support):
        shutil.copy(src_ci_support / fn, ci_support / fn)
    return str(tmp_path)
