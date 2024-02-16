import sys

from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QHBoxLayout,
                             QLabel, QMainWindow, QPushButton, QTextEdit,
                             QVBoxLayout, QWidget)

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interface Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.interface = None  # To store the loaded interface
        self.group_index = 0  # Initialize group index

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Menu Bar
        self.menu_bar = self.menuBar()
        self.file_menu = self.menu_bar.addMenu("&File")

        # Open Action
        self.open_action = QAction("&Open", self)
        self.open_action.triggered.connect(self.open_directory_dialog)
        self.file_menu.addAction(self.open_action)

        # Information Action
        self.info_action = QAction("&Information", self)
        self.info_action.triggered.connect(self.show_info_window)
        self.info_action.setEnabled(False)  # Disabled by default
        self.file_menu.addAction(self.info_action)

        # Group navigation bar
        self.nav_bar_layout = QHBoxLayout()
        self.left_button = QPushButton("<")
        self.left_button.clicked.connect(self.decrement_group_index)
        self.nav_bar_layout.addWidget(self.left_button)

        self.group_index_label = QLabel("0")
        self.nav_bar_layout.addWidget(self.group_index_label)

        self.right_button = QPushButton(">")
        self.right_button.clicked.connect(self.increment_group_index)
        self.nav_bar_layout.addWidget(self.right_button)

        # Initially disable navigation buttons
        self.left_button.setEnabled(False)
        self.right_button.setEnabled(False)

        # Add navigation bar to the main layout at the top
        self.layout.addLayout(self.nav_bar_layout)

        # Label to display the interface string
        self.interface_label = QLabel("No interface loaded", self)
        self.layout.addWidget(self.interface_label)

    def open_directory_dialog(self):
        if False:  # test mode
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select Directory")
        dir_path = '/home/ivan/Science/sociophysics/ActivityDriven/0/'
        if dir_path:
            # Use the selected directory to create an Interface object
            S = Simulation.from_dir(dir_path)
            S.group(num_intervals=10, interval_size=1)
            self.interface = Interface.create_interface(S)

            # Update the interface label with the string representation of the interface
            self.update_interface_label()

            # Enable the Information menu option
            self.info_action.setEnabled(True)
            self.left_button.setEnabled(True)
            self.right_button.setEnabled(True)

    def update_interface_label(self):
        self.interface_label.setText(
            str(self.interface.groups[self.group_index]))

    def show_info_window(self):
        if self.interface:
            # Pass self to set MainWindow as parent
            self.info_window = InfoWindow()
            self.info_window.set_info_text(str(self.interface))
            self.info_window.show()

    def increment_group_index(self):
        if self.interface and self.group_index < len(self.interface.groups) - 1:
            self.group_index += 1
            self.update_group_index_label()
            self.update_interface_label()

    def decrement_group_index(self):
        if self.interface and self.group_index > 0:
            self.group_index -= 1
            self.update_group_index_label()
            self.update_interface_label()

    def update_group_index_label(self):
        self.group_index_label.setText(str(self.group_index))
        # Additional functionality can be added here to update other parts of the UI based on the new group index

    def closeEvent(self, event):
        QApplication.closeAllWindows()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
