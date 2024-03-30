import argparse
import sys

from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QHBoxLayout,
                             QLabel, QMainWindow, QPushButton, QTextEdit,
                             QVBoxLayout, QWidget)

from hari_plotter.gui.add_plot_window import AddPlotWindow
from hari_plotter.gui.qt_plotter import QtPlotter
from hari_plotter.interface import Interface
from hari_plotter.simulation import Simulation


class InfoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interface Information")
        self.setGeometry(200, 200, 600, 400)
        self.layout = QVBoxLayout(self)
        self.info_label = QTextEdit()
        self.info_label.setReadOnly(True)
        self.layout.addWidget(self.info_label)

    def set_info_text(self, text):
        self.info_label.setText(text)


class MainWindow(QMainWindow):
    def __init__(self, initial_dir_path=None, settings_path=None):
        super().__init__()
        self.setWindowTitle("Hari Plotter")
        self.setGeometry(100, 100, 800, 600)

        self.plotter = None  # To store the loaded interface
        self._group_index = 0  # Initialize group index
        self.figure = Figure()
        self.plotter = QtPlotter(None, figure=self.figure)

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Menu Bar
        self.menu_bar = self.menuBar()

        # File Menu
        self.file_menu = self.menu_bar.addMenu("&File")

        # Open Action
        self.open_action = QAction("&Open", self)
        self.open_action.triggered.connect(self.open_directory_dialog)
        self.file_menu.addAction(self.open_action)

        # Import Settings Action
        self.import_settings_action = QAction("&Import Settings", self)
        self.import_settings_action.triggered.connect(
            self.import_plotter_settings)
        self.import_settings_action.setEnabled(False)  # Disabled by default
        self.file_menu.addAction(self.import_settings_action)

        # Export Settings Action
        self.export_settings_action = QAction("&Export Settings", self)
        self.export_settings_action.triggered.connect(
            self.export_plotter_settings)
        self.export_settings_action.setEnabled(False)  # Disabled by default
        self.file_menu.addAction(self.export_settings_action)

        # Edit Menu
        self.edit_menu = self.menu_bar.addMenu("&Edit")

        # Information Action
        self.info_action = QAction("&Information", self)
        self.info_action.triggered.connect(self.show_info_window)
        self.info_action.setEnabled(False)  # Disabled by default
        self.edit_menu.addAction(self.info_action)

        # Add Plot Action
        self.add_plot_action = QAction("&Add Plot", self)
        self.add_plot_action.triggered.connect(self.show_add_plot_window)
        self.add_plot_action.setEnabled(False)  # Disabled by default
        self.edit_menu.addAction(self.add_plot_action)

        # Group navigation bar
        self.nav_bar_layout = QHBoxLayout()
        self.left_button = QPushButton("<")
        self.left_button.clicked.connect(self.decrement_group_index)
        self.nav_bar_layout.addWidget(self.left_button)

        self.group_index_label = QLabel("0/0")
        self.nav_bar_layout.addWidget(self.group_index_label)

        self.right_button = QPushButton(">")
        self.right_button.clicked.connect(self.increment_group_index)
        self.nav_bar_layout.addWidget(self.right_button)

        # Initially disable navigation buttons
        self.left_button.setEnabled(False)
        self.right_button.setEnabled(False)

        # Add navigation bar to the main layout at the top
        self.layout.addLayout(self.nav_bar_layout)

        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        if initial_dir_path:
            self.open_directory_dialog(initial_dir_path)

        if self.plotter.is_initialized and settings_path:
            self.import_plotter_settings(settings_path)

        self.update_group_index_label()

    @property
    def group_index(self):
        return self._group_index

    @group_index.setter
    def group_index(self, i: int):
        self._group_index = i
        self.update_group_index_label()

    def open_directory_dialog(self, dir_path=None):
        if not dir_path:
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select Directory")
        # Check that directory was selected
        if dir_path:
            # Use the selected directory to create an Interface object
            S = Simulation.from_dir(dir_path)
            S.group(num_intervals=10, interval_size=1)
            self.plotter.update_interface(Interface.create_interface(S))

            # Update the interface label with the string representation of the interface
            self.plot_current_group()

            # Enable the Information menu option
            self.info_action.setEnabled(True)
            self.add_plot_action.setEnabled(True)
            self.left_button.setEnabled(True)
            self.right_button.setEnabled(True)
            self.import_settings_action.setEnabled(True)
            self.export_settings_action.setEnabled(True)

            self.update_group_index_label()

    def plot_current_group(self):
        if self.plotter.is_initialized:
            print('Group: ' + str(self.group_index) +
                  '/'+str(len(self.plotter._interfaces)-1))
            self.plotter.plot(self.group_index)
            self.canvas.draw_idle()  # Refresh the canvas to display the new plot

    def show_info_window(self):
        if self.plotter.is_initialized:
            # Pass self to set MainWindow as parent
            self.info_window = InfoWindow()
            self.info_window.set_info_text(str(self.plotter._interfaces))
            self.info_window.show()

    def increment_group_index(self):
        if self.plotter.is_initialized:
            group_index = self.group_index + 1
            self.group_index = group_index % len(
                self.plotter._interfaces.groups)
            self.plot_current_group()

    def decrement_group_index(self):
        if self.plotter.is_initialized:
            group_index = self.group_index - 1
            if group_index == -1:
                group_index = len(self.plotter._interfaces.groups)-1
            self.group_index = group_index
            self.plot_current_group()

    def update_group_index_label(self):
        if self.plotter.is_initialized:
            self.group_index_label.setText(
                str(self.group_index)+'/'+str(len(self.plotter._interfaces)-1))
        else:
            self.group_index_label.setText('0/0')

    def show_add_plot_window(self):
        if self.plotter.is_initialized:
            self.add_plot_window = AddPlotWindow(self)
            self.add_plot_window.show()

    def export_plotter_settings(self):
        # Use QFileDialog to get the filename from the user
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "plotter_settings", "Python Files (*.py)", options=options)
        if fileName:
            if not fileName.endswith('.py'):
                fileName += '.py'  # Ensure the file has a .py extension
            # Write the plotter settings to the file
            with open(fileName, 'w') as file:
                file.write(self.plotter.to_code())

    def import_plotter_settings(self, settings_path=None):
        if not settings_path:  # If no settings path is provided, open dialog
            options = QFileDialog.Options()
            settings_path, _ = QFileDialog.getOpenFileName(
                self, "Load Settings", "", "Python Files (*.py)", options=options)

        if settings_path:
            with open(settings_path, 'r') as file:
                code = file.read()
                modified_code = code.replace('plotter.', 'self.plotter.')
                try:
                    exec(modified_code, {'self': self})
                    self.plot_current_group()
                except Exception as e:
                    print(f"Error executing loaded settings: {e}")

    def closeEvent(self, event):
        QApplication.closeAllWindows()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Hari Plotter Application")
    parser.add_argument("path", nargs='?',
                        help="Path to the directory to load", default=None)
    parser.add_argument("-s", "--settings",
                        help="Path to the settings file", default=None)

    args = parser.parse_args()

    app = QApplication(sys.argv)
    mainWin = MainWindow(args.path, args.settings)
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
