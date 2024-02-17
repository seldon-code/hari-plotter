from __future__ import annotations

import sys
from abc import abstractmethod

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QSpinBox, QTextEdit,
                             QVBoxLayout, QWidget)

from hari_plotter.plot import (BoolParameter, FloatParameter, ListParameter,
                               NoneOrFloatParameter, NoneRangeParameter)


class ParameterWidgetFactory:
    @staticmethod
    def create_widget(parameter):
        if isinstance(parameter, ListParameter):
            return ListParameterWidget(parameter)
        elif isinstance(parameter, BoolParameter):
            return BoolParameterWidget(parameter)
        elif isinstance(parameter, FloatParameter):
            return FloatParameterWidget(parameter)
        elif isinstance(parameter, NoneOrFloatParameter):
            return NoneOrFloatParameterWidget(parameter)
        elif isinstance(parameter, NoneRangeParameter):
            return NoneRangeParameterWidget(parameter)
        else:
            raise ValueError("Unsupported parameter type")


class BaseParameterWidget(QWidget):
    def __init__(self, parameter, parent=None):
        super().__init__(parent)
        self.parameter = parameter

    @abstractmethod
    def get_value(self):
        raise NotImplementedError("Subclasses must implement get_value method")

    @abstractmethod
    def is_valid(self):
        raise NotImplementedError("Subclasses must implement is_valid method")

    def create_parameter_label(self, text, comment):
        label = QLabel(text)
        if comment:
            label.setText(f"{text} (?)")
            label.setToolTip(comment)
            # Change cursor to indicate clickable or hoverable
            label.setCursor(Qt.WhatsThisCursor)
        return label


class ListParameterWidget(BaseParameterWidget):
    def __init__(self, parameter: ListParameter, parent=None):
        super().__init__(parameter, parent)
        layout = QVBoxLayout(self)
        self.combo_box = QComboBox()
        self.combo_box.addItems(parameter.arguments)
        label = self.create_parameter_label(parameter.name, parameter.comment)
        layout.addWidget(label)
        layout.addWidget(self.combo_box)

    def get_value(self):
        if self.is_valid():
            return self.combo_box.currentText()
        else:
            raise ValueError("Invalid value")

    def is_valid(self):
        return self.parameter.validate(self.combo_box.currentText())


class BoolParameterWidget(BaseParameterWidget):
    def __init__(self, parameter: BoolParameter, parent=None):
        super().__init__(parameter, parent)
        layout = QVBoxLayout(self)
        self.checkbox = QCheckBox()
        layout.addWidget(self.checkbox)
        label = self.create_parameter_label(parameter.name, parameter.comment)
        layout.insertWidget(0, label)  # Insert label above the checkbox

    def get_value(self):
        if self.is_valid():
            return self.checkbox.isChecked()
        else:
            raise ValueError("Invalid value")

    def is_valid(self):
        return self.parameter.validate(self.checkbox.isChecked())


class FloatParameterWidget(BaseParameterWidget):
    def __init__(self, parameter: FloatParameter, parent=None):
        super().__init__(parameter, parent)
        layout = QVBoxLayout(self)

        self.spin_box = QDoubleSpinBox()
        self.spin_box.setValue(self.parameter.default_value)
        if self.parameter.limits[0] is not None:
            self.spin_box.setMinimum(self.parameter.limits[0])
        if self.parameter.limits[1] is not None:
            self.spin_box.setMaximum(self.parameter.limits[1])

        label = self.create_parameter_label(parameter.name, parameter.comment)
        layout.addWidget(label)
        layout.addWidget(self.spin_box)

    def get_value(self):
        if self.is_valid():
            return self.spin_box.value()
        else:
            raise ValueError(f"Invalid value for {self.parameter.name}")

    def is_valid(self):
        return self.parameter.validate(self.spin_box.value())


class NoneOrFloatParameterWidget(BaseParameterWidget):
    def __init__(self, parameter: NoneOrFloatParameter, parent=None):
        super().__init__(parameter, parent)
        layout = QVBoxLayout(self)

        # Checkbox to enable/disable float input
        self.enable_checkbox = QCheckBox("Enable")
        self.enable_checkbox.setChecked(parameter.default_value is not None)
        self.enable_checkbox.toggled.connect(self.toggle_spin_box_state)
        layout.addWidget(self.enable_checkbox)

        # Float input field
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setEnabled(self.enable_checkbox.isChecked())
        if parameter.default_value is not None:
            self.spin_box.setValue(parameter.default_value)
        if parameter.limits[0] is not None:
            self.spin_box.setMinimum(parameter.limits[0])
        if parameter.limits[1] is not None:
            self.spin_box.setMaximum(parameter.limits[1])

        label = self.create_parameter_label(parameter.name, parameter.comment)
        layout.addWidget(label)
        layout.addWidget(self.spin_box)

    def toggle_spin_box_state(self, checked):
        self.spin_box.setEnabled(checked)

    def get_value(self):
        if self.enable_checkbox.isChecked():
            if self.is_valid():
                return self.spin_box.value()
            else:
                raise ValueError(f"Invalid value for {self.parameter.name}")
        else:
            return None

    def is_valid(self):
        # When disabled, consider it always valid
        if not self.enable_checkbox.isChecked():
            return True
        # When enabled, validate the float value
        return self.parameter.validate(self.spin_box.value())


class NoneRangeParameterWidget(BaseParameterWidget):
    def __init__(self, parameter: NoneRangeParameter, parent=None):
        super().__init__(parameter, parent)
        layout = QVBoxLayout(self)

        # Checkbox and SpinBox for minimum limit
        self.enable_min_checkbox = QCheckBox("Enable Min")
        self.enable_min_checkbox.setChecked(
            parameter.default_min_value is not None)
        self.enable_min_checkbox.toggled.connect(
            self.toggle_min_spin_box_state)
        layout.addWidget(self.enable_min_checkbox)

        self.min_spin_box = QDoubleSpinBox()
        self.min_spin_box.setEnabled(self.enable_min_checkbox.isChecked())
        if parameter.default_min_value is not None:
            self.min_spin_box.setValue(parameter.default_min_value)
        if parameter.limits[0] is not None:
            self.min_spin_box.setMinimum(parameter.limits[0])
        if parameter.limits[1] is not None:
            self.min_spin_box.setMaximum(parameter.limits[1])

        layout.addWidget(self.min_spin_box)

        # Checkbox and SpinBox for maximum limit
        self.enable_max_checkbox = QCheckBox("Enable Max")
        self.enable_max_checkbox.setChecked(
            parameter.default_max_value is not None)
        self.enable_max_checkbox.toggled.connect(
            self.toggle_max_spin_box_state)
        layout.addWidget(self.enable_max_checkbox)

        self.max_spin_box = QDoubleSpinBox()
        self.max_spin_box.setEnabled(self.enable_max_checkbox.isChecked())
        if parameter.default_max_value is not None:
            self.max_spin_box.setValue(parameter.default_max_value)
        if parameter.limits[0] is not None:
            self.max_spin_box.setMinimum(parameter.limits[0])
        if parameter.limits[1] is not None:
            self.max_spin_box.setMaximum(parameter.limits[1])

        layout.addWidget(self.max_spin_box)

        # Ensure max is always >= min
        self.min_spin_box.valueChanged.connect(self.ensure_min_less_than_max)
        self.max_spin_box.valueChanged.connect(
            self.ensure_max_greater_than_min)

        label = self.create_parameter_label(parameter.name, parameter.comment)
        layout.insertWidget(0, label)  # Insert at the top

    def toggle_min_spin_box_state(self, checked):
        self.min_spin_box.setEnabled(checked)
        if checked and self.min_spin_box.value() > self.max_spin_box.value():
            self.min_spin_box.setValue(self.max_spin_box.value())

    def toggle_max_spin_box_state(self, checked):
        self.max_spin_box.setEnabled(checked)
        if checked and self.max_spin_box.value() < self.min_spin_box.value():
            self.max_spin_box.setValue(self.min_spin_box.value())

    def ensure_min_less_than_max(self):
        if self.min_spin_box.value() > self.max_spin_box.value():
            self.max_spin_box.setValue(self.min_spin_box.value())

    def ensure_max_greater_than_min(self):
        if self.max_spin_box.value() < self.min_spin_box.value():
            self.min_spin_box.setValue(self.max_spin_box.value())

    def get_value(self):
        min_value = self.min_spin_box.value() if self.enable_min_checkbox.isChecked() else None
        max_value = self.max_spin_box.value() if self.enable_max_checkbox.isChecked() else None
        if self.is_valid(min_value, max_value):
            return (min_value, max_value)
        else:
            raise ValueError(f"Invalid range for {self.parameter.name}")

    def is_valid(self, min_value, max_value):
        # Validate based on the parameter's rules
        return self.parameter.validate((min_value, max_value))
