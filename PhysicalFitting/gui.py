# gui.py
#
# Gui for PhysicalFitting
#
# David Lister
# October 2023
#

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QGridLayout, QFileDialog, QSlider, QLabel, QSizePolicy, QLineEdit, QHBoxLayout, QVBoxLayout
import pyqtgraph as pg
import logging
import sys

logger = logging.getLogger("PhysicalFitting")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.debug("Logger Started")


class CustomSlider(QWidget):
    def __init__(self, param_name, initial_value, min_value, max_value, model_update_func, parent=None):
        super(CustomSlider, self).__init__(parent)

        self.param_name = param_name
        self.min_value = min_value
        self.max_value = max_value
        self.present_value = initial_value
        self.update_function = model_update_func

        self.resolution_target = 1000
        self.resolution_factor = self.resolution_target/(self.max_value - self.min_value)

        # Create label
        self.label = QLabel(self.param_name)

        # Create lower bound QLineEdit
        self.lower_bound_edit = QLineEdit(str(self.min_value))
        self.lower_bound_edit.setFixedWidth(60)  # set width to keep it consistent

        # Create upper bound QLineEdit
        self.upper_bound_edit = QLineEdit(str(self.max_value))
        self.upper_bound_edit.setFixedWidth(60)  # set width to keep it consistent

        # Create QSlider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.min_value * self.resolution_factor)
        self.slider.setMaximum(self.max_value * self.resolution_factor)
        self.slider.setValue(self.present_value * self.resolution_factor)

        # Layouts
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.lower_bound_edit)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.upper_bound_edit)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addLayout(slider_layout)

        self.setLayout(main_layout)

        # Connect signals and slots
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.lower_bound_edit.textChanged.connect(self.on_bound_changed)
        self.upper_bound_edit.textChanged.connect(self.on_bound_changed)

    def get_actual_value(self):
        return self.slider.value() / self.resolution_factor

    def get_fit_guess_and_bounds(self):
        return (self.get_actual_value(), self.min_value, self.max_value)

    def on_slider_changed(self, value):
        """Handle slider value change."""
        self.update_function()

    def on_bound_changed(self):
        """Handle changes in the bounds."""
        try:
            new_min = float(self.lower_bound_edit.text())
            new_max = float(self.upper_bound_edit.text())
            if new_min < new_max:
                actual_value = self.get_actual_value()
                self.min_value = new_min
                self.max_value = new_max
                self.resolution_factor = self.resolution_target/(new_max - new_min)
                self.slider.setMinimum(new_min * self.resolution_factor)
                self.slider.setMaximum(new_max * self.resolution_factor)
                self.slider.setValue(actual_value * self.resolution_factor)
            else:
                self.lower_bound_edit.setText(str(self.min_value))
                self.upper_bound_edit.setText(str(self.max_value))

        except ValueError:
            self.lower_bound_edit.setText(str(self.min_value))
            self.upper_bound_edit.setText(str(self.max_value))


class InternalState:
    """Holds state separate from GUI"""
    def __init__(self):
        self.model = None


class MainWindow(QMainWindow):
    """
    Main GUI Class
    """
    def __init__(self, model=None, state=None):
        super().__init__()
        if state is None:
            self.state = InternalState()
            if model is not None:
                self.state.model = model
        else:
            self.state = state

        self.logger = logging.getLogger("PhysicalFitting.GUI")
        self.logger.debug("Main window started")

        self.setWindowTitle("Physical Fitting")
        self.colour = self.palette().color(QtGui.QPalette.Window)
        self.main_pen = pg.mkPen(color=(20, 20, 20))
        self.fit_pen = pg.mkPen(color=(153, 0, 0))
        self.reference_pen = pg.mkPen(color=(12, 105, 201))
        self.header_font = QtGui.QFont("Sans Serif", 12)
        self.body_font = QtGui.QFont("Sans Serif", 12)

        self.layout = QGridLayout()

        # Create widgets
        # Link them to functions
        # Add widgets to layout

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
