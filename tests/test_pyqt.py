import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import pytest

# Define a main window class


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Test Window")
        self.setGeometry(100, 100, 280, 80)


@pytest.fixture(scope="module")
def app():
    """Create a QApplication object for the test module."""
    app = QApplication(sys.argv)
    yield app
    # Cleanup code after tests in the module have run
    app.quit()


@pytest.fixture
def window():
    """Create a MainWindow object for each test."""
    return MainWindow()


def test_window_title(window):
    """Test if the window title is set correctly."""
    assert window.windowTitle() == "PyQt5 Test Window"


def test_window_size(window):
    """Test if the window size is set correctly."""
    assert window.geometry().width() == 280
    assert window.geometry().height() == 80
