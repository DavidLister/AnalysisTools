# use_cases.py
#
# File to explore use cases that I want the final package to satisfy
# This is not supposed to be working python at the moment.
#
# David Lister
# September 2023
#

import analysis_tools as AT
import numpy as np
import common

ureg = AT.ureg  # A unit registry from the pint package

fname = "some_file"
data = AT.load_PL_data(fname)  # Data will be loaded from file and the units will be converted into the energy vs counts

# data will be passed around as a dictionary of numpy arrays.
# For the photoluminescence (PL) data, this will look like:

data = {"Energy": np.array((1, 2, 3)) * ureg.parse_units("meV"),
        "Counts": np.array((123, 234, 623)) * ureg.parse_units("Counts"),
        "Name": "S1234"}  # Either from the fname or the file


# Model will be a dict of sub-models
# All variable and parameters included in the file definition
# The values FIXED_PARAMETERS and FIT_PARAMETERS are special.
# Parameters are defined in a sub-dict for each type
# Otherwise strings are assumed to be sub-models

FIXED_PARAMETERS = common.FIXED_PARAMETERS
FIT_PARAMETERS = common.FIT_PARAMETERS
PARAMETERS = common.PARAMETERS
MODEL = common.MODEL
AUTO = common.AUTO
DOMAIN = common.DOMAIN

initial_model = {"Ga_Pair_Naurita_Thermal": {MODEL: AT.Model_Pair_Naurita_Thermal,
                                             PARAMETERS: {"scale": "A_Ga_Pair", # Dictionary indices refer to those defined in the model
                                                          "Nd": "Ga_Nd",
                                                          "Temp": "Temperature",
                                                          "E0":"E0_Ga",
                                                          "Ebind":"Ebind_Ga_ZnO"}
                                             },

                 "Ga_Lorentzian": {MODEL: AT.Model_Lorentzian,
                                   PARAMETERS: {"scale": "A_Ga_Lorentzian",
                                                "fwhm": "FWHM_Ga_Lorentzian",
                                                "x_offset": "E0_Ga"},
                                   DOMAIN: (3 * ureg.parse_units("meV"), 4 * ureg.parse_units("meV"))
                                   },

                 "In_Pair_Naurita_Thermal": {MODEL: AT.Model_Pair_Naurita_Thermal,
                                             PARAMETERS: {"scale": "A_In_Pair",
                                                          "Nd": "In_Nd",
                                                          "Temp": "Temperature",
                                                          "E0":"E0_In",
                                                          "Ebind":"Ebind_In_ZnO"}
                                             },

                 "In_Lorentzian": {MODEL: AT.Model_Lorentzian,
                                   PARAMETERS: {"scale": "A_In_Lorentzian",
                                                "fwhm": "FWHM_In_Lorentzian",
                                                "x_offset": "E0_In"}
                                   },

                 FIXED_PARAMETERS: {"E0_Ga": AT.values.E0_Ga_ZnO,
                                    "Ebind_Ga_ZnO": AT.values.Ebind_Ga_ZnO,
                                    "E0_In": AT.values.E0_In_ZnO,
                                    "Ebind_In_ZnO": AT.values.Ebind_In_ZnO
                                    },

                 FIT_PARAMETERS: {"A_Ga_Pair":AUTO, # Auto will trigger subroutine to automatically determine guess. Only available for some parameters, typically scaling
                                  "Ga_Nd": 1e18 * ureg.parse_units(),
                                  "A_In_Pair": AUTO,
                                  "In_Nd": 1e16,
                                  "Temperature": 15 * ureg.kelvin}
                 }


AT.plot_model(data["Energy"], data["Counts"], initial_model)  # Would plot model and residuals, format is (x, y, model)

fitted_model, fit_quality = AT.fit_model(data["Energy"], data["Counts"], initial_model)  # Would return a dict of the same structure, but with updated fit parameters, plus error on the fit params

print(fit_quality)

AT.plot_model(data["Energy"], data["Counts"], fitted_model)
