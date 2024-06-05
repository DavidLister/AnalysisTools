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



    fit_model_def = {"Ga_urbach_left": {PhysicalFitting.common.MODEL: PhysicalFitting.models.spectra.s_urbach_tail_left,
                                        PhysicalFitting.common.PARAMETERS: {'E_peak': "Ga_E_peak",
                                                                            'E_u': 'Ga_E_u_left',
                                                                            'E_0': 'Ga_E_peak',
                                                                            'A': 'Ga_urbach_scale_left'}},
                     "Ga_urbach_right": {PhysicalFitting.common.MODEL: PhysicalFitting.models.spectra.s_urbach_tail_right,
                                        PhysicalFitting.common.PARAMETERS: {'E_peak': "Ga_E_peak",
                                                                            'E_u': 'Ga_E_u_right',
                                                                            'E_0': 'Ga_E_peak',
                                                                            'A': 'Ga_urbach_scale_right'}},
                     "In_urbach_left": {PhysicalFitting.common.MODEL: PhysicalFitting.models.spectra.s_urbach_tail_left,
                                        PhysicalFitting.common.PARAMETERS: {'E_peak': "In_E_peak",
                                                                            'E_u': 'In_E_u_left',
                                                                            'E_0': 'In_E_peak',
                                                                            'A': 'In_urbach_scale_left'}},
                     "In_urbach_right": {PhysicalFitting.common.MODEL: PhysicalFitting.models.spectra.s_urbach_tail_right,
                                         PhysicalFitting.common.PARAMETERS: {'E_peak': "In_E_peak",
                                                                             'E_u': 'In_E_u_right',
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
                     PhysicalFitting.common.FIT_PARAMETERS: {"In_lorentz_scale": (110, 1, 200), # (guess, min, max)
                                                             "background_b": (5000, 100, 40000),
                                                             "Ga_lorentz_fwhm": (0.0002, 0.00005, 0.001),
                                                             "Ga_lorentz_scale": (1000, 100, 10000),
                                                             "Ga_E_peak": (3.360, 3.355, 3.365),  # eV
                                                             "Ga_E_u_left": (0.002, 0.0005, 0.05),  # eV
                                                             "Ga_urbach_scale_left": (1e5, 1e4, 1e7),
                                                             "Ga_E_u_right": (-0.001, -0.005, -0.0001),  # eV
                                                             "Ga_urbach_scale_right": (1e4, 2e5, 5e6),
                                                             "In_E_u_left": (0.002, 0.0001, 0.01),  # eV
                                                             "In_urbach_scale_left": (1e5, 1e3, 1e6),
                                                             "In_E_u_right": (-0.002, -0.01, -0.0001),  # eV
                                                             "In_urbach_scale_right": (1e5, 1e3, 1e6)
                                                             }}


    fit_model = PhysicalFitting.model_classes.CompositeModel(fit_model_def)
    error_model = PhysicalFitting.solver.error_l2norm
    start = time.time()
    parameters, fit = PhysicalFitting.solver.fit_model_global(test_energy, test_intensity, fit_model, error_model,
                                                          method="differential_evolution",
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
