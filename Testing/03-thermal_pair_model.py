# 02-simple_pair_model.py
#
# A test program that tries to fit a pair model to spectroscopic data.
# David Lister
# October 2023
#



import numpy as np
import PhysicalFitting
import matplotlib.pyplot as plt
import matplotlib
import time

import common

if __name__ == "__main__":  # Needed for multi-worked global optimization to be stable.

    matplotlib.use('TkAgg')

    ureg = PhysicalFitting.ureg


    def convert_loaded_data(data):
        z = 1.000289  # Correction of energy term for the refractive index of air
        wavelength = data.transpose()[0][::-1] * ureg.nm
        energy = wavelength.to("eV", 'sp') / z

        intensity = data.transpose()[1][::-1]
        return energy.magnitude, intensity


    H835_energy, H835_intensity = convert_loaded_data(np.genfromtxt('H835_2.txt',skip_header=4))
    H836_energy, H836_intensity = convert_loaded_data(np.genfromtxt('H836_1.txt',skip_header=4))
    H841_energy, H841_intensity = convert_loaded_data(np.genfromtxt('H841_2.txt',skip_header=4))
    data_H839 = np.genfromtxt('H839_2.txt',skip_header=4)
    data_H839[:,0] = data_H839[:,0] + 368.89-368.675 # Calibration was off
    H839_energy, H839_intensity = convert_loaded_data(data_H839)

    test_energy = H836_energy
    test_intensity = H836_intensity
    title = "H836"

    plt.plot(test_energy, test_intensity)
    plt.show()



    fit_model_def = {"Ga_kittel_pair": {PhysicalFitting.common.MODEL: PhysicalFitting.models.spectra.s_pair_kittel_model_thermal,
                                        PhysicalFitting.common.PARAMETERS: {'E_peak': "Ga_E_peak",
                                                                            'E_bind': "Ga_E_bind",
                                                                            'r_bohr': "Ga_r_bohr",
                                                                            'scale': "Ga_pair_scale",
                                                                            'nd': "Ga_nd",
                                                                            'T': 'temperature'}},
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
                     PhysicalFitting.common.FIXED_PARAMETERS: {"Ga_E_peak": 3.3600,  # eV
                                                               "Ga_E_bind": 0.0159,  # eV
                                                               "Ga_r_bohr": 0.8,  # nm
                                                               "In_E_peak": 3.35692,  # eV
                                                               "In_lorentz_fwhm": 0.00015,  # eV
                                                               "background_m": 0
                                                               },
                     PhysicalFitting.common.FIT_PARAMETERS: {"Ga_nd": (1e18, 1e15, 1e19),
                                                             "Ga_pair_scale": (3000, 100, 10000),
                                                             "In_lorentz_scale": (60, 1, 200),
                                                             "background_b": (5000, 100, 40000),
                                                             "temperature": (10, 1, 1000),
                                                             "Ga_lorentz_fwhm": (0.002, 0.0001, 0.010),
                                                             "Ga_lorentz_scale": (500, 10, 10000)
                                                             }}


    fit_model = PhysicalFitting.model_classes.CompositeModel(fit_model_def)
    error_model = PhysicalFitting.solver.error_l2norm
    start = time.time()
    parameters, fit = PhysicalFitting.solver.fit_model_global(test_energy, test_intensity, fit_model, error_model,
                                                          max_iteration=5000, method="Nelder-Mead",
                                                          antialiasing=common.GAUSSIAN_AA10X)
    end = time.time()
    print(f"Model fitting took {end - start:.2f} seconds")

    for parameter in parameters:
        print(f"{parameter}: {parameters[parameter]}")

    print(fit)

    new_x = np.linspace(min(test_energy), max(test_energy), 10*len(test_energy))
    plt.plot(test_energy, test_intensity, 'x', label="Input")
    y_model = fit_model.run(new_x, parameters, antialiasing=common.GAUSSIAN_AA10X, width=test_energy[1] - test_energy[0])
    plt.plot(new_x, y_model, label="Model")
    plt.legend()
    plt.title(title)
    plt.show()
