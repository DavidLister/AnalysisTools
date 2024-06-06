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
import PhysicalFitting

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
        self.label = QLabel(self.param_name + '  ')

        # Create lower bound QLineEdit
        self.lower_bound_edit = QLineEdit(f"{self.min_value:.2e}")
        self.lower_bound_edit.setFixedWidth(60)  # set width to keep it consistent

        # Create upper bound QLineEdit
        self.upper_bound_edit = QLineEdit(f"{self.max_value:.2e}")
        self.upper_bound_edit.setFixedWidth(60)  # set width to keep it consistent

        # Create QSlider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.min_value * self.resolution_factor)
        self.slider.setMaximum(self.max_value * self.resolution_factor)
        self.slider.setValue(self.present_value * self.resolution_factor)

        # Layouts
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.label)
        slider_layout.addWidget(self.lower_bound_edit)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.upper_bound_edit)

        main_layout = QVBoxLayout()
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
        self.update_function(self.get_actual_value())

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
                self.lower_bound_edit.setText(f"{self.min_value:.2e}")
                self.upper_bound_edit.setText(f"{self.max_value:.2e}")

        except ValueError:
            self.lower_bound_edit.setText(f"{self.min_value:.2e}")
            self.upper_bound_edit.setText(f"{self.max_value:.2e}")


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
        self.color = self.palette().color(QtGui.QPalette.Window)
        self.main_pen = pg.mkPen(color=(20, 20, 20))
        self.fit_pen = pg.mkPen(color=(153, 0, 0))
        self.reference_pen = pg.mkPen(color=(12, 105, 201))
        self.header_font = QtGui.QFont("Sans Serif", 12)
        self.body_font = QtGui.QFont("Sans Serif", 12)

        self.layout = QGridLayout()

        # Create widgets
        # Link them to functions
        # Add widgets to layout

        self.build_slider_gui()

        # self.test_slider = CustomSlider("Test", 5, 0, 10, test)
        # self.layout.addWidget(self.test_slider)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

    def build_slider_gui(self):
        self.fit_sliders = []

        if self.state.model[PhysicalFitting.common.FIT_PARAMETERS] is not None:
            for parameter in self.state.model[PhysicalFitting.common.FIT_PARAMETERS].keys():
                bounds = self.state.model[PhysicalFitting.common.FIT_PARAMETERS][parameter]
                self.fit_sliders.append(CustomSlider(parameter, bounds[0], bounds[1], bounds[2], test))
                self.layout.addWidget(self.fit_sliders[-1])

        self.spacing_line = QWidget()
        self.spacing_line.setFixedHeight(2)
        self.spacing_line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.spacing_line.setStyleSheet("background-color: #c0c0c0;")
        self.layout.addWidget(self.spacing_line)

        self.fix_sliders = []
        if self.state.model[PhysicalFitting.common.FIXED_PARAMETERS] is not None:
            for parameter in self.state.model[PhysicalFitting.common.FIXED_PARAMETERS].keys():
                value = self.state.model[PhysicalFitting.common.FIXED_PARAMETERS][parameter]
                if value == 0:
                    upper = 1
                    lower = -1
                else:
                    upper = value * 10
                    lower = value / 10
                self.fix_sliders.append(CustomSlider(parameter, value, lower, upper, test))
                self.layout.addWidget(self.fix_sliders[-1])

def test(val):
    print(val)

if __name__ == "__main__":
    fit_model_def = {"Ga_urbach_left": {PhysicalFitting.common.MODEL: PhysicalFitting.models.spectra.s_urbach_tail_left,
                                        PhysicalFitting.common.PARAMETERS: {'E_peak': "Ga_E_peak",
                                                                            'E_u': 'E_u',
                                                                            'E_0': 'Ga_E_peak',
                                                                            'A': 'Ga_urbach_scale_left'}},
                     "Ga_urbach_right": {
                         PhysicalFitting.common.MODEL: PhysicalFitting.models.spectra.s_urbach_tail_right,
                         PhysicalFitting.common.PARAMETERS: {'E_peak': "Ga_E_peak",
                                                             'E_u': 'E_u',
                                                             'E_0': 'Ga_E_peak',
                                                             'A': 'Ga_urbach_scale_right'}},
                     "In_urbach_left": {PhysicalFitting.common.MODEL: PhysicalFitting.models.spectra.s_urbach_tail_left,
                                        PhysicalFitting.common.PARAMETERS: {'E_peak': "In_E_peak",
                                                                            'E_u': 'E_u',
                                                                            'E_0': 'In_E_peak',
                                                                            'A': 'In_urbach_scale_left'}},
                     "In_urbach_right": {
                         PhysicalFitting.common.MODEL: PhysicalFitting.models.spectra.s_urbach_tail_right,
                         PhysicalFitting.common.PARAMETERS: {'E_peak': "In_E_peak",
                                                             'E_u': 'E_u',
                                                             'E_0': 'In_E_peak',
                                                             'A': 'In_urbach_scale_right'}},
                     "Ga_lorentzian": {PhysicalFitting.common.MODEL: PhysicalFitting.models.generic.s_lorentz,
                                       PhysicalFitting.common.PARAMETERS: {'x0': "Ga_E_peak",
                                                                           'fwhm': "Ga_lorentz_fwhm",
                                                                           'scale': "Ga_lorentz_scale"}},
                     "In_lorentzian": {PhysicalFitting.common.MODEL: PhysicalFitting.models.generic.s_lorentz,
                                       PhysicalFitting.common.PARAMETERS: {'x0': "In_E_peak",
                                                                           'fwhm': "In_lorentz_fwhm",
                                                                           'scale': "In_lorentz_scale"}},
                     "Background": {PhysicalFitting.common.MODEL: PhysicalFitting.models.generic.s_linear,
                                    PhysicalFitting.common.PARAMETERS: {'m': "background_m",
                                                                        'b': "background_b"}},
                     PhysicalFitting.common.FIXED_PARAMETERS: {"In_E_peak": 3.35692,  # eV
                                                               "In_lorentz_fwhm": 0.00015,  # eV
                                                               "background_m": 0,  # counts
                                                               },
                     PhysicalFitting.common.FIT_PARAMETERS: {"In_lorentz_scale": (10, 1, 200),  # (guess, min, max)
                                                             "background_b": (5000, 100, 40000),
                                                             "Ga_lorentz_fwhm": (0.0002, 0.00005, 0.001),
                                                             "Ga_lorentz_scale": (200, 100, 10000),
                                                             "Ga_E_peak": (3.360, 3.355, 3.365),  # eV
                                                             "E_u": (0.002, 0.0005, 0.05),  # eV
                                                             "Ga_urbach_scale_left": (1e5, 1e4, 1e7),
                                                             "Ga_urbach_scale_right": (1e4, 2e5, 5e6),
                                                             "In_urbach_scale_left": (1e5, 1e3, 1e6),
                                                             "In_urbach_scale_right": (1e5, 1e3, 1e6)
                                                             }}



    app = QApplication([])

    window = MainWindow(model=fit_model_def)
    window.show()

    sys.exit(app.exec())
