import pytest
from hari_plotter.multirun import Multirun
from hari_plotter.simulation import Simulation
from hari_plotter.model import Model


class TestSimulation:

    @classmethod
    def setup_class(cls):
        cls.degroot = Simulation.from_toml('tests/degroot.toml')
        cls.activity = Simulation.from_toml('tests/activity.toml')

    def test_from_toml(self):
        assert isinstance(self.degroot, Simulation)
        assert isinstance(self.activity, Simulation)
        assert isinstance(self.degroot.model, Model)
        assert isinstance(self.activity.model, Model)

    def test_to_toml(self, tmp_path):
        filename = tmp_path / "test.toml"
        self.degroot.to_toml(filename)
        assert filename.exists(), 'File not created'


@pytest.fixture
def simulation_fixture():
    return TestSimulation()


@pytest.fixture
def mock_multirun(simulation_fixture):
    return Multirun([simulation_fixture.degroot, simulation_fixture.activity])


def test_multirun_initialization(simulation_fixture):
    mr = Multirun([simulation_fixture.degroot, simulation_fixture.activity])
    assert len(mr.simulations) == 2
    assert mr.simulations[0] == simulation_fixture.degroot
    assert mr.simulations[1] == simulation_fixture.activity


def test_multirun_join(mock_multirun, simulation_fixture):
    another_multirun = Multirun([simulation_fixture.degroot])
    mock_multirun.join(another_multirun)
    assert len(mock_multirun.simulations) == 3


def test_multirun_append(mock_multirun, simulation_fixture):
    new_simulation = Simulation.from_toml('tests/activity.toml')
    mock_multirun.append(new_simulation)
    assert len(mock_multirun.simulations) == 3
