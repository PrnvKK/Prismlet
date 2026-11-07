import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--weights", default=None,
                     help="directory with the official Prism weights")


@pytest.fixture(scope="session")
def weights_dir(request):
    value = request.config.getoption("--weights")
    if not value:
        pytest.skip("weight-dependent test: pass --weights <weights dir>")
    path = pathlib.Path(value)
    if not (path / "tokenizer.json").exists():
        pytest.skip(f"weight-dependent test: {path} has no tokenizer.json")
    return path
