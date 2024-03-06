from __future__ import annotations

import sys
from abc import abstractmethod

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QSpinBox, QTextEdit,
                             QVBoxLayout, QWidget)

from hari_plotter.gui.parameters_widgets import ParameterWidgetFactory
from hari_plotter.interface import Interface


class PlotSettingsWidget(QWidget):
    def __init__(self, parent: AddPlotWindow = None):
        super().__init__(parent)
        self.add_plot_window = parent
        self.layout = QVBoxLayout(self)
        self.parameter_widgets = []  # Store parameter widgets

    def update_settings(self, plot_type):
        # Clear existing settings UI
        for widget in self.parameter_widgets:
            widget.deleteLater()
        self.parameter_widgets.clear()

        # Get the settings for the selected plot type
        self.plot_class = self.add_plot_window.main_window.plotter.get_plot_class(
            plot_type)
        settings = self.plot_class.settings(
            self.add_plot_window.main_window.plotter.interface)

        # Create UI for each setting using the factory
        for setting in settings:
            widget = ParameterWidgetFactory.create_widget(setting)
            self.layout.addWidget(widget)
            self.parameter_widgets.append(widget)

    def get_settings(self):
        collected_settings = {}
        for widget in self.parameter_widgets:
            collected_settings[widget.parameter.parameter_name] = widget.get_value(
            )
        return self.plot_class.qt_to_settings(collected_settings)


class AddPlotWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # Store reference to MainWindow
        self.setWindowTitle("Add Plot")
        self.setGeometry(300, 300, 400, 200)  # Adjusted for more content
        layout = QVBoxLayout(self)

        # Plot type selection
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(
            self.main_window.plotter.available_plot_types)
        layout.addWidget(QLabel("Select plot type:"))
        layout.addWidget(self.plot_type_combo)

        # Initialize PlotSettingsWidget
        self.settings_widget = PlotSettingsWidget(self)
        layout.addWidget(self.settings_widget)

        # Update settings display when the plot type changes
        self.plot_type_combo.currentIndexChanged.connect(
            self.update_settings_display)
        # Initialize settings display
        self.update_settings_display()

        # Position selection
        self.position_layout = QHBoxLayout()
        self.row_spin_box = QSpinBox()
        # Assuming a max of 10 rows for simplicity
        self.row_spin_box.setRange(0, 9)
        self.col_spin_box = QSpinBox()
        # Assuming a max of 10 columns for simplicity
        self.col_spin_box.setRange(0, 9)
        self.position_layout.addWidget(QLabel("Row:"))
        self.position_layout.addWidget(self.row_spin_box)
        self.position_layout.addWidget(QLabel("Column:"))
        self.position_layout.addWidget(self.col_spin_box)
        layout.addLayout(self.position_layout)

        # OK button
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.add_plot)
        layout.addWidget(self.ok_button)

    def update_settings_display(self):
        selected_plot_type = self.plot_type_combo.currentText()
        # For simplicity, using an empty string for plotter info
        self.settings_widget.update_settings(selected_plot_type)

    def add_plot(self):
        plot_type = self.plot_type_combo.currentText()
        row = self.row_spin_box.value()
        col = self.col_spin_box.value()
        settings = self.settings_widget.get_settings()

        self.main_window.plotter.add_plot(
            plot_type,
            settings,
            row=row,
            col=col
        )
        # Assuming this method exists in MainWindow to refresh the plot
        self.main_window.plot_current_group()
        self.close()
