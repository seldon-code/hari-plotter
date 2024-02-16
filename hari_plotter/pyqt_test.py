import pytest
from PyQt5.QtCore import Qt
from hari_plotter.gui.main_window import MainWindow


def test_button_click(qtbot):
    # Initialize the application window
    window = MainWindow()

    # Use qtbot to manage the setup and teardown of the widget
    qtbot.addWidget(window)

    # Ensure the window is shown
    window.show()
