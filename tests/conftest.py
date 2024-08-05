import os

import pytest

from hari_plotter import ColorScheme, Simulation
from hari_plotter.color_scheme import initialize_colormap


def pytest_addoption(parser):
    parser.addoption(
        "--setup",
        action="store_true",
        default=False,
        help="Save generated files to the baseline directory.",
    )


@pytest.fixture
def save_dir(request, tmpdir):
    if request.config.getoption("--setup"):
        return "tests/baseline"
    else:
        return tmpdir


@pytest.fixture(scope="module")
def setup_simulation():
    ColorScheme.default_colormap_name, ColorScheme.default_colormap = initialize_colormap(
        {
            "Name": "custom1",
            "Colors": [
                (255 / 255, 79 / 255, 20 / 255),
                (1 / 255, 180 / 255, 155 / 255),
                (71 / 255, 71 / 255, 240 / 255),
            ],
        }
    )
    S = Simulation.from_dir("tests/big_test")
    return S
